"""Combined REST and streamable HTTP MCP application."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator
from urllib.parse import urlsplit

from fastapi import FastAPI
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .api.main import create_app
from .mcp_server.server import create_mcp
from .service import RAGService


def _matches_trusted_host(host: str, trusted_hosts: tuple[str, ...]) -> bool:
    return any(
        pattern == "*" or host == pattern or (pattern.startswith("*.") and host.endswith(pattern[1:]))
        for pattern in trusted_hosts
    )


class MCPOriginMiddleware:
    """Apply Starlette's trusted-host patterns to browser MCP requests."""

    def __init__(self, app: ASGIApp, trusted_hosts: tuple[str, ...]) -> None:
        self.app = app
        self.trusted_hosts = trusted_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            origin = Headers(scope=scope).get("origin")
            if origin and not self._is_trusted_origin(origin):
                await PlainTextResponse("Invalid Origin header", status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)

    def _is_trusted_origin(self, origin: str) -> bool:
        parsed = urlsplit(origin)
        try:
            parsed.port
        except ValueError:
            return False
        return (
            parsed.scheme in {"http", "https"}
            and not (parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password)
            and parsed.hostname is not None
            and _matches_trusted_host(parsed.hostname, self.trusted_hosts)
        )


def create_combined_app(service: RAGService | None = None) -> FastAPI:
    """Create one REST app and streamable HTTP MCP adapter over one service."""
    rag = service or RAGService()
    mcp_path = os.getenv("MCP_PATH", "/mcp")
    # Starlette's outer TrustedHostMiddleware enforces these host patterns. MCP only
    # supports literal Host/Origin values, so mirror Starlette's pattern semantics for Origin.
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
    app.mount(mcp_path, MCPOriginMiddleware(mcp_http, rag.settings.trusted_hosts))
    app.state.mcp = mcp
    return app


app = create_combined_app()
