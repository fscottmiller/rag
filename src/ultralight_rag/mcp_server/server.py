"""FastMCP adapter over the same RAGService used by REST."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, Any

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent
from pydantic import Field

from ..auth import AuthenticationError, AuthorizationError, Authorizer
from ..pipeline.embeddings import EmbeddingProviderError, EmbeddingProviderUnavailableError
from ..service import RAGService
from ..storage.sqlite import DocumentNotFoundError

_ALLOWED_TRANSPORTS = {"stdio", "streamable-http"}

# `CallToolResult` cannot appear inside a `Union`/`|` return annotation -- the SDK
# raises `InvalidSignature` at tool-registration time if it does (see
# `mcp.server.fastmcp.utilities.func_metadata.func_metadata`, which explicitly
# rejects a `T | CallToolResult` return annotation). Instead, the SDK supports
# `Annotated[CallToolResult, RealReturnType]` for tools that sometimes bypass
# normal result conversion by returning a `CallToolResult` directly: the
# advertised `outputSchema` is still derived from `RealReturnType` -- byte-for-byte
# what it would be if the annotation had just been `RealReturnType` -- because
# `func_metadata` special-cases `Annotated[CallToolResult, ...]` and rebuilds the
# schema from the second element (see its `issubclass(return_type_expr,
# CallToolResult)` branch). At runtime, `FuncMetadata.convert_result` checks
# `isinstance(result, CallToolResult)` unconditionally -- before consulting the
# annotation at all -- so returning a `CallToolResult` already worked prior to
# this change; only the *declared* type was a lie. This alias makes it honest:
# mypy resolves `Annotated[X, ...]` to `X` (PEP 593) and ignores the second
# argument entirely for type-checking purposes -- it does NOT replicate the
# SDK's special-casing. So a plain `Annotated[CallToolResult, T]` alias makes
# mypy believe every one of these tools always returns bare `CallToolResult`,
# which then misflags the ordinary success-path returns (the actual `T`
# values) as errors, even though those are exactly as correct as the
# denial/failure paths that return `CallToolResult` directly.
#
# To give mypy the honest union type (`CallToolResult | T`) it needs to check
# *both* paths, without changing what the SDK sees at runtime, the alias is
# defined differently for static analysis than for execution:
#   - `TYPE_CHECKING` is only True while a type checker is running, so mypy
#     type-checks against `CallToolResult | T` and correctly accepts returns
#     of either shape while still rejecting anything else.
#   - At runtime `TYPE_CHECKING` is always False, so the SDK (which resolves
#     this module's annotations via `get_type_hints`, evaluated against these
#     same module globals, after `from __future__ import annotations` has
#     deferred them to strings) sees exactly `Annotated[CallToolResult, T]` --
#     unchanged from before, so `func_metadata`'s
#     `issubclass(return_type_expr, CallToolResult)` special case still fires
#     and the emitted `outputSchema` is still derived from `T`, byte-for-byte.
if TYPE_CHECKING:
    SearchResult = CallToolResult | list[dict[str, Any]]
    DocumentResult = CallToolResult | dict[str, Any]
    DeleteResult = CallToolResult | dict[str, str]
else:
    SearchResult = Annotated[CallToolResult, list[dict[str, Any]]]
    DocumentResult = Annotated[CallToolResult, dict[str, Any]]
    DeleteResult = Annotated[CallToolResult, dict[str, str]]


def get_transport() -> str:
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in _ALLOWED_TRANSPORTS:
        raise ValueError("MCP_TRANSPORT must be 'stdio' or 'streamable-http'")
    return transport


def _provider_failure(
    exc: EmbeddingProviderError, *, denied_content: dict[str, Any]
) -> CallToolResult:
    """Map an embedding-provider failure to a structured result, matching the
    `authorize()` pattern above: `_meta.error_type` carries the machine-readable
    reason and `content` carries a generic, client-safe message. The real
    detail (upstream response bodies, hostnames, quota text) is logged
    server-side in pipeline/embeddings.py, not forwarded here -- see F12/section C."""
    if isinstance(exc, EmbeddingProviderUnavailableError):
        error_type = "embedding_provider_unavailable"
        message = "The embedding provider is currently unavailable."
    else:
        error_type = "embedding_provider_error"
        message = "The embedding provider returned an invalid response."
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
        structuredContent=denied_content,
        _meta={"error_type": error_type},
    )


def create_mcp(
    service: RAGService | None = None,
    *,
    streamable_http_path: str | None = None,
    transport_security: TransportSecuritySettings | None = None,
) -> FastMCP:
    rag = service or RAGService()
    authorizer = Authorizer(rag.settings)
    server = FastMCP(
        "ultralight-rag",
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8000")),
        streamable_http_path=streamable_http_path
        if streamable_http_path is not None
        else os.getenv("MCP_PATH", "/mcp"),
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
    ) -> SearchResult:
        """Search indexed chunks by semantic similarity."""
        if denial := authorize(ctx, "read", denied_content={"result": []}):
            return denial
        try:
            return await anyio.to_thread.run_sync(rag.search, query, top_k, filter_metadata)
        except EmbeddingProviderError as exc:
            return _provider_failure(exc, denied_content={"result": []})

    @server.tool()
    async def list_documents(*, ctx: Context) -> SearchResult:
        """List documents currently held in the index."""
        if denial := authorize(ctx, "read", denied_content={"result": []}):
            return denial
        return await anyio.to_thread.run_sync(rag.list_documents)

    @server.tool()
    async def get_document(document_id: str, *, ctx: Context) -> DocumentResult:
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
    ) -> DocumentResult:
        """Ingest a document directly into the index."""
        if denial := authorize(ctx, "write", denied_content={}):
            return denial
        try:
            return await anyio.to_thread.run_sync(rag.ingest, title, content, metadata)
        except EmbeddingProviderError as exc:
            return _provider_failure(exc, denied_content={})

    @server.tool()
    async def delete_document(document_id: str, *, ctx: Context) -> DeleteResult:
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
    # get_transport() validates its result against _ALLOWED_TRANSPORTS at
    # runtime (raising ValueError otherwise), but its declared return type is
    # plain `str`, not the `Literal["stdio", "sse", "streamable-http"]`
    # `FastMCP.run` expects, so mypy cannot statically prove membership.
    # Narrowing the return type to a Literal would require duplicating
    # _ALLOWED_TRANSPORTS as a Literal alias for no behavioral benefit;
    # pre-existing/unrelated to this PR.
    get_mcp().run(transport=get_transport())  # type: ignore[arg-type]
