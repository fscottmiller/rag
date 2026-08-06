"""FastMCP adapter over the same RAGService used by REST."""

from __future__ import annotations

import os
from typing import Annotated, Any

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent
from pydantic import Field

from ..auth import AuthenticationError, AuthorizationError, Authorizer
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

    def authorize(
        ctx: Context, action: str, *, denied_content: dict[str, Any]
    ) -> CallToolResult | None:
        """Authorize a tool call, returning a structured denial result on failure.

        REST distinguishes AuthenticationError (401) from AuthorizationError (403).
        MCP tools have no status-code channel: any exception raised here would be
        flattened by the SDK into a generic `isError` text result, indistinguishable
        by anything but prose (see mcp.server.fastmcp.tools.base.Tool.run and
        mcp.server.lowlevel.server.Server.call_tool, which catch bare `Exception`
        and stringify it). Returning a `CallToolResult` directly bypasses that
        flattening, so the denial reason survives in machine-readable form as
        `_meta.error_type` alongside the human-readable text in `content`.
        `denied_content` fills `structuredContent` with a value that already
        satisfies the calling tool's declared output schema (an empty result list
        for list-shaped tools, an empty object for dict-shaped tools) since the SDK
        validates `structuredContent` against that schema even for error results.
        """
        try:
            request = ctx.request_context.request
        except ValueError:
            request = None
        headers = getattr(request, "headers", {}) if request is not None else {}
        try:
            authorizer.authorize(headers, action)
        except (AuthenticationError, AuthorizationError) as exc:
            error_type = (
                "authentication_required"
                if isinstance(exc, AuthenticationError)
                else "not_authorized"
            )
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))],
                isError=True,
                structuredContent=denied_content,
                _meta={"error_type": error_type},
            )
        return None

    @server.tool()
    async def rag_search(
        query: str,
        top_k: Annotated[int, Field(ge=1, le=100)] = 5,
        filter_metadata: dict[str, Any] | None = None,
        *,
        ctx: Context,
    ) -> list[dict[str, Any]]:
        """Search indexed chunks by semantic similarity."""
        if denial := authorize(ctx, "read", denied_content={"result": []}):
            return denial
        return await anyio.to_thread.run_sync(rag.search, query, top_k, filter_metadata)

    @server.tool()
    async def list_documents(*, ctx: Context) -> list[dict[str, Any]]:
        """List documents currently held in the index."""
        if denial := authorize(ctx, "read", denied_content={"result": []}):
            return denial
        return await anyio.to_thread.run_sync(rag.list_documents)

    @server.tool()
    async def get_document(document_id: str, *, ctx: Context) -> dict[str, Any]:
        """Retrieve one document, metadata, and its chunks."""
        if denial := authorize(ctx, "read", denied_content={}):
            return denial
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
        if denial := authorize(ctx, "write", denied_content={}):
            return denial
        return await anyio.to_thread.run_sync(rag.ingest, title, content, metadata)

    @server.tool()
    async def delete_document(document_id: str, *, ctx: Context) -> dict[str, str]:
        """Delete a document and all of its indexed chunks."""
        if denial := authorize(ctx, "write", denied_content={}):
            return denial
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
