import pytest

from src.pipeline.chunking import BaseChunker
from src.pipeline.embeddings import BaseEmbedder
from src.service import RAGService
from src.storage.sqlite import SQLiteStore


class KeywordEmbedder(BaseEmbedder):
    def embed(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [float("python" in lower), float("sqlite" in lower), float("fastapi" in lower)]
            )
        return vectors


class FixedChunker(BaseChunker):
    def chunk(self, text):
        return [part.strip() for part in text.split("|") if part.strip()]


@pytest.fixture
def service():
    return RAGService(SQLiteStore(), KeywordEmbedder(), FixedChunker())
