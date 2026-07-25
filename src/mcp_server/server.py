"""FastMCP adapter over the same RAGService used by REST."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..service import RAGService


def create_mcp(service: RAGService | None = None) -> FastMCP:
    rag = service or RAGService()
    server = FastMCP("transient-rag")

    @server.tool()
    def rag_search(query: str, top_k: int = 5, filter_metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Search indexed chunks by semantic similarity."""
        return rag.search(query, top_k, filter_metadata)

    @server.tool()
    def list_documents() -> list[dict[str, Any]]:
        """List documents currently held in the transient index."""
        return rag.list_documents()

    @server.tool()
    def get_document(document_id: str) -> dict[str, Any]:
        """Retrieve one document, metadata, and its chunks."""
        return rag.get_document(document_id)

    @server.tool()
    def upload_document(title: str, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Ingest a document directly into the temporary index."""
        return rag.ingest(title, content, metadata)

    @server.tool()
    def delete_document(document_id: str) -> dict[str, str]:
        """Delete a document and all of its indexed chunks."""
        rag.delete_document(document_id)
        return {"status": "deleted", "document_id": document_id}

    return server


mcp = create_mcp()

if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))
