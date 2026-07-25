"""Shared ingestion and retrieval service used by REST and MCP adapters."""

from __future__ import annotations

from typing import Any

from .config import Settings
from .pipeline.chunking import BaseChunker, ChonkieChunker
from .pipeline.embeddings import BaseEmbedder, create_embedder
from .storage.sqlite import SQLiteStore


class RAGService:
    def __init__(
        self,
        store: SQLiteStore | None = None,
        embedder: BaseEmbedder | None = None,
        chunker: BaseChunker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.store = store or SQLiteStore(self.settings.database_path)
        self.embedder = embedder or create_embedder(
            self.settings.embedding_provider,
            self.settings.embedding_model,
            self.settings.ollama_url,
        )
        self.chunker = chunker or ChonkieChunker(
            self.settings.chunker, self.settings.chunk_size, self.settings.chunk_overlap
        )

    def _prepare(self, content: str, chunker: BaseChunker | None, embedder: BaseEmbedder | None) -> tuple[list[str], list[list[float]]]:
        chunks = (chunker or self.chunker).chunk(content)
        if not chunks:
            raise ValueError("content must contain at least one non-whitespace character")
        return chunks, (embedder or self.embedder).embed(chunks)

    def _selected_embedder(self, embedding_choice: str | None) -> BaseEmbedder:
        if not embedding_choice:
            return self.embedder
        provider = embedding_choice.lower().replace("_", "-")
        model = self.settings.ollama_model if provider == "ollama" else self.settings.embedding_model
        return create_embedder(embedding_choice, model, self.settings.ollama_url)

    def ingest(
        self,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        chunking_strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_choice: str | None = None,
    ) -> dict[str, Any]:
        chunker = self.chunker
        if any(value is not None for value in (chunking_strategy, chunk_size, chunk_overlap)):
            chunker = ChonkieChunker(
                chunking_strategy or self.settings.chunker,
                chunk_size or self.settings.chunk_size,
                chunk_overlap if chunk_overlap is not None else self.settings.chunk_overlap,
            )
        selected = self._selected_embedder(embedding_choice)
        chunks, embeddings = self._prepare(content, chunker, selected)
        return self.store.create_document(title, content, metadata or {}, chunks, embeddings)

    def update(self, document_id: str, title: str, content: str, metadata: dict[str, Any] | None = None, **options: Any) -> dict[str, Any]:
        chunker = self.chunker
        if any(options.get(value) is not None for value in ("chunking_strategy", "chunk_size", "chunk_overlap")):
            chunker = ChonkieChunker(
                options.get("chunking_strategy") or self.settings.chunker,
                options.get("chunk_size") or self.settings.chunk_size,
                options.get("chunk_overlap") if options.get("chunk_overlap") is not None else self.settings.chunk_overlap,
            )
        selected = self._selected_embedder(options.get("embedding_choice"))
        chunks, embeddings = self._prepare(content, chunker, selected)
        return self.store.replace_document(document_id, title, content, metadata or {}, chunks, embeddings)

    def search(self, query: str, top_k: int = 5, filter_metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        return self.store.search(self.embedder.embed_one(query), top_k, filter_metadata)

    def list_documents(self) -> list[dict[str, Any]]:
        return self.store.list_documents()

    def get_document(self, document_id: str) -> dict[str, Any]:
        return self.store.get_document(document_id)

    def delete_document(self, document_id: str) -> None:
        self.store.delete_document(document_id)
