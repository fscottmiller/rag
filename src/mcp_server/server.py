"""FastMCP adapter over the same RAGService used by REST."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from ..auth import Authorizer
from ..service import RAGService


_ALLOWED_TRANSPORTS = {"stdio", "streamable-http"}


def get_transport() -> str:
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in _ALLOWED_TRANSPORTS:
        raise ValueError("MCP_TRANSPORT must be 'stdio' or 'streamable-http'")
    return transport


def create_mcp(service: RAGService | None = None) -> FastMCP:
    rag = service or RAGService()
    authorizer = Authorizer(rag.settings)
    server = FastMCP(
        "transient-rag",
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8000")),
        streamable_http_path=os.getenv("MCP_PATH", "/mcp"),
    )

    def authorize(ctx: Context, action: str) -> None:
        request_context = ctx._request_context
        request = request_context.request if request_context is not None else None
        headers = getattr(request, "headers", {}) if request is not None else {}
        authorizer.authorize(headers, action)

    @server.tool()
    def rag_search(
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Search indexed chunks by semantic similarity."""
        if ctx is not None:
            authorize(ctx, "read")
        return rag.search(query, top_k, filter_metadata)

    @server.tool()
    def list_documents(ctx: Context | None = None) -> list[dict[str, Any]]:
        """List documents currently held in the transient index."""
        if ctx is not None:
            authorize(ctx, "read")
        return rag.list_documents()

    @server.tool()
    def get_document(document_id: str, ctx: Context | None = None) -> dict[str, Any]:
        """Retrieve one document, metadata, and its chunks."""
        if ctx is not None:
            authorize(ctx, "read")
        return rag.get_document(document_id)

    @server.tool()
    def upload_document(
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Ingest a document directly into the temporary index."""
        if ctx is not None:
            authorize(ctx, "write")
        return rag.ingest(title, content, metadata)

    @server.tool()
    def delete_document(document_id: str, ctx: Context | None = None) -> dict[str, str]:
        """Delete a document and all of its indexed chunks."""
        if ctx is not None:
            authorize(ctx, "delete")
        rag.delete_document(document_id)
        return {"status": "deleted", "document_id": document_id}

    return server


mcp = create_mcp()

if __name__ == "__main__":
    mcp.run(transport=get_transport())
