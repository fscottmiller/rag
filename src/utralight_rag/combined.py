"""Combined REST and streamable HTTP MCP application."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

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
        async with mcp.session_manager.run():
            yield

    app = create_app(rag, lifespan=lifespan)
    app.mount(mcp_path, MCPOriginMiddleware(mcp_http))
    app.state.mcp = mcp
    return app


app = create_combined_app()
