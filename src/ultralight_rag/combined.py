"""Combined REST and streamable HTTP MCP application."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .api.main import _same_origin, create_app
from .mcp_server.server import create_mcp
from .service import RAGService


class MCPOriginMiddleware:
    """Reject cross-origin browser MCP requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            origin = Headers(scope=scope).get("origin")
            if origin and not _same_origin(origin, Request(scope)):
                response = PlainTextResponse("Invalid Origin header", status_code=403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_combined_app(service: RAGService | None = None) -> FastAPI:
    """Create one REST app and streamable HTTP MCP adapter over one service."""
    # Only a service we construct ourselves is ours to close. A caller who
    # injects `service` still holds that reference and may reuse it (across
    # tests, across other app instances, ...) after this app shuts down, so
    # closing its store out from under them would be a surprise, not a favor.
    owns_store = service is None
    rag = service or RAGService()
    mcp_path = os.getenv("MCP_PATH", "/mcp")
    mcp = create_mcp(
        rag,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        ),
    )
    mcp_http = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            # The `finally` (rather than code after the `async with`) makes
            # this run even if the session manager's startup or shutdown
            # raises, and only after its own `__aexit__` has completed, so
            # the store outlives every session the manager was still
            # servicing. `SQLiteStore.close()` is idempotent, which keeps
            # this safe alongside the autouse store-closing fixture in
            # tests/conftest.py.
            if owns_store:
                rag.store.close()

    app = create_app(rag, lifespan=lifespan)
    app.mount(mcp_path, MCPOriginMiddleware(mcp_http))
    app.state.mcp = mcp
    return app


# Lazy singleton, mirroring api/main.py's get_app()/__getattr__ and
# mcp_server/server.py's get_mcp()/__getattr__: a bare `import
# ultralight_rag.combined` must not open a SQLite connection or construct an
# embedder. `app` is only realized on first attribute access, which is what
# `uvicorn ultralight_rag.combined:app` (and anything else naming `.app`)
# triggers.
_default_app: FastAPI | None = None


def get_app() -> FastAPI:
    global _default_app
    if _default_app is None:
        _default_app = create_combined_app()
    return _default_app


def __getattr__(name: str) -> Any:
    if name == "app":
        return get_app()
    raise AttributeError(name)
