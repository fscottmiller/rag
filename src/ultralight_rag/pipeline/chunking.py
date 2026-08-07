"""Chonkie-backed chunking with a small stable interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into non-empty chunks."""


def _chunk_text(result: Any) -> str:
    return getattr(result, "text", str(result)).strip()


@dataclass
class ChonkieChunker(BaseChunker):
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64

    def __post_init__(self) -> None:
        if self.chunk_size < 1 or self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be smaller than size")
        try:
            from chonkie import RecursiveChunker, SentenceChunker, TokenChunker
        except ImportError as exc:
            raise RuntimeError("chonkie is required for document chunking") from exc

        chunkers = {
            "recursive": RecursiveChunker,
            "sentence": SentenceChunker,
            "token": TokenChunker,
        }
        try:
            chunker_type = chunkers[self.strategy.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}") from exc

        kwargs = {"chunk_size": self.chunk_size}
        if self.strategy.lower() != "recursive":
            kwargs["chunk_overlap"] = self.chunk_overlap
        self._chunker = chunker_type(**kwargs)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        results = self._chunker.chunk(text)
        chunks = []
        for item in results:
            start = getattr(item, "start_index", None)
            end = getattr(item, "end_index", None)
            if self.strategy.lower() == "recursive" and self.chunk_overlap and start is not None:
                piece = text[max(0, int(start) - self.chunk_overlap) : int(end)]
            else:
                piece = _chunk_text(item)
            piece = piece.strip()
            if piece:
                chunks.append(piece)
        return chunks
