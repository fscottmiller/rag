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
            self.settings.embedding_url,
            self.settings.embedding_api_key,
            self.settings.embedding_timeout,
            self.settings.embedding_dimensions,
            self.settings.embedding_batch_size,
        )
        self.chunker = chunker or ChonkieChunker(
            self.settings.chunker, self.settings.chunk_size, self.settings.chunk_overlap
        )

    def _prepare(self, content: str) -> tuple[list[str], list[list[float]]]:
        if len(content.encode("utf-8")) > self.settings.max_document_bytes:
            raise ValueError(f"document content exceeds {self.settings.max_document_bytes} bytes")
        chunks = self.chunker.chunk(content)
        if not chunks:
            raise ValueError("content must contain at least one non-whitespace character")
        embeddings = []
        for start in range(0, len(chunks), self.settings.embedding_batch_size):
            embeddings.extend(
                self.embedder.embed(chunks[start : start + self.settings.embedding_batch_size])
            )
        return chunks, embeddings

    def ingest(
        self,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not title.strip():
            raise ValueError("title must contain at least one non-whitespace character")
        chunks, embeddings = self._prepare(content)
        return self.store.create_document(title, content, metadata or {}, chunks, embeddings)

    def update(
        self,
        document_id: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not title.strip():
            raise ValueError("title must contain at least one non-whitespace character")
        self.store.get_document(document_id)
        chunks, embeddings = self._prepare(content)
        return self.store.replace_document(
            document_id, title, content, metadata or {}, chunks, embeddings
        )

    def search(
        self, query: str, top_k: int = 5, filter_metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must contain at least one non-whitespace character")
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        return self.store.search(self.embedder.embed_one(query), top_k, filter_metadata)

    def list_documents(self) -> list[dict[str, Any]]:
        return self.store.list_documents()

    def get_document(self, document_id: str) -> dict[str, Any]:
        return self.store.get_document(document_id)

    def delete_document(self, document_id: str) -> None:
        self.store.delete_document(document_id)
