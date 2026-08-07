"""Shared ingestion and retrieval service used by REST and MCP adapters."""

from __future__ import annotations

import logging
from typing import Any

from .config import Settings
from .pipeline.chunking import BaseChunker, ChonkieChunker
from .pipeline.embeddings import (
    BaseEmbedder,
    EmbeddingProviderError,
    create_embedder,
    embedding_configuration,
)
from .storage.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


class DocumentTooLargeError(ValueError):
    """The request contains more document content than configured."""


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
        self.embedder = (
            embedder
            if embedder is not None
            else create_embedder(
                self.settings.embedding_provider,
                self.settings.embedding_model,
                self.settings.embedding_url,
                self.settings.embedding_api_key,
                self.settings.embedding_timeout,
                self.settings.embedding_dimensions,
                self.settings.embedding_batch_size,
            )
        )
        self.chunker = (
            chunker
            if chunker is not None
            else ChonkieChunker(
                self.settings.chunker, self.settings.chunk_size, self.settings.chunk_overlap
            )
        )
        configuration = embedding_configuration(self.embedder)
        self._embedding_identity = (
            (configuration.provider, configuration.model, configuration.fingerprint)
            if configuration is not None
            else None
        )
        if configuration is None:
            if self.store.is_persistent():
                raise ValueError(
                    "Persistent indexes require a built-in embedder with a known identity"
                )
        else:
            self.store.ensure_embedding_configuration(
                configuration.provider, configuration.model, configuration.fingerprint
            )

    def _prepare(self, content: str) -> tuple[list[str], list[list[float]]]:
        if len(content.encode("utf-8")) > self.settings.max_document_bytes:
            raise DocumentTooLargeError(
                f"document content exceeds {self.settings.max_document_bytes} bytes"
            )
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
        self.store.preflight_embedding_configuration(self._embedding_identity)
        try:
            chunks, embeddings = self._prepare(content)
        except EmbeddingProviderError:
            logger.warning("Embedding provider failure while ingesting document %r", title)
            raise
        document = self.store.create_document(
            title,
            content,
            metadata or {},
            chunks,
            embeddings,
            expected_embedding_identity=self._embedding_identity,
        )
        logger.info(
            "Document ingested: id=%s title=%r chunk_count=%d", document["id"], title, len(chunks)
        )
        return document

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
        self.store.preflight_embedding_configuration(self._embedding_identity)
        try:
            chunks, embeddings = self._prepare(content)
        except EmbeddingProviderError:
            logger.warning(
                "Embedding provider failure while updating document: id=%s title=%r",
                document_id,
                title,
            )
            raise
        document = self.store.replace_document(
            document_id,
            title,
            content,
            metadata or {},
            chunks,
            embeddings,
            expected_embedding_identity=self._embedding_identity,
        )
        logger.info(
            "Document updated: id=%s title=%r chunk_count=%d", document_id, title, len(chunks)
        )
        return document

    def search(
        self, query: str, top_k: int = 5, filter_metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must contain at least one non-whitespace character")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        self.store.preflight_embedding_configuration(self._embedding_identity)
        try:
            query_embedding = self.embedder.embed_one(query)
        except EmbeddingProviderError:
            logger.warning("Embedding provider failure while searching")
            raise
        results = self.store.search(
            query_embedding,
            top_k,
            filter_metadata,
            expected_embedding_identity=self._embedding_identity,
        )
        logger.debug("Search executed: top_k=%d result_count=%d", top_k, len(results))
        return results

    def list_documents(self) -> list[dict[str, Any]]:
        return self.store.list_documents()

    def get_document(self, document_id: str) -> dict[str, Any]:
        return self.store.get_document(document_id)

    def delete_document(self, document_id: str) -> None:
        self.store.preflight_embedding_configuration(self._embedding_identity)
        self.store.delete_document(
            document_id, expected_embedding_identity=self._embedding_identity
        )
        logger.info("Document deleted: id=%s", document_id)
