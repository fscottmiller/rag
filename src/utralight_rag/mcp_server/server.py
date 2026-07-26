"""FastMCP adapter over the same RAGService used by REST."""

from __future__ import annotations

import os
from typing import Any

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from ..auth import Authorizer
from ..service import RAGService
from ..storage.sqlite import DocumentNotFoundError

_ALLOWED_TRANSPORTS = {"stdio", "streamable-http"}


def get_transport() -> str:
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in _ALLOWED_TRANSPORTS:
        raise ValueError("MCP_TRANSPORT must be 'stdio' or 'streamable-http'")
    return transport


def create_mcp(
    service: RAGService | None = None,
    *,
    streamable_http_path: str | None = None,
    transport_security: TransportSecuritySettings | None = None,
) -> FastMCP:
    rag = service or RAGService()
    authorizer = Authorizer(rag.settings)
    server = FastMCP(
        "utralight-rag",
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8000")),
        streamable_http_path=streamable_http_path or os.getenv("MCP_PATH", "/mcp"),
        transport_security=transport_security,
    )

    def authorize(ctx: Context, action: str) -> None:
        try:
            request = ctx.request_context.request
        except ValueError:
            request = None
        headers = getattr(request, "headers", {}) if request is not None else {}
        authorizer.authorize(headers, action)

    @server.tool()
    async def rag_search(
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        *,
        ctx: Context,
    ) -> list[dict[str, Any]]:
        """Search indexed chunks by semantic similarity."""
        authorize(ctx, "read")
        return await anyio.to_thread.run_sync(rag.search, query, top_k, filter_metadata)

    @server.tool()
    async def list_documents(*, ctx: Context) -> list[dict[str, Any]]:
        """List documents currently held in the index."""
        authorize(ctx, "read")
        return await anyio.to_thread.run_sync(rag.list_documents)

    @server.tool()
    async def get_document(document_id: str, *, ctx: Context) -> dict[str, Any]:
        """Retrieve one document, metadata, and its chunks."""
        authorize(ctx, "read")
        try:
            return await anyio.to_thread.run_sync(rag.get_document, document_id)
        except DocumentNotFoundError:
            raise ValueError(f"Document not found: {document_id}") from None

    @server.tool()
    async def upload_document(
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        ctx: Context,
    ) -> dict[str, Any]:
        """Ingest a document directly into the index."""
        authorize(ctx, "write")
        return await anyio.to_thread.run_sync(rag.ingest, title, content, metadata)

    @server.tool()
    async def delete_document(document_id: str, *, ctx: Context) -> dict[str, str]:
        """Delete a document and all of its indexed chunks."""
        authorize(ctx, "write")
        try:
            await anyio.to_thread.run_sync(rag.delete_document, document_id)
        except DocumentNotFoundError:
            raise ValueError(f"Document not found: {document_id}") from None
        return {"status": "deleted", "document_id": document_id}

    return server


_default_mcp: FastMCP | None = None


def get_mcp() -> FastMCP:
    global _default_mcp
    if _default_mcp is None:
        _default_mcp = create_mcp()
    return _default_mcp


def __getattr__(name: str) -> Any:
    if name == "mcp":
        return get_mcp()
    raise AttributeError(name)


if __name__ == "__main__":
    get_mcp().run(transport=get_transport())
