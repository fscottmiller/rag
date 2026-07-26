"""Combined REST and streamable HTTP MCP application."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from mcp.server.transport_security import TransportSecuritySettings

from .api.main import app as rest_app
from .api.main import create_app
from .mcp_server.server import create_mcp
from .service import RAGService


def create_combined_app(service: RAGService | None = None) -> FastAPI:
    """Create one REST app and streamable HTTP MCP adapter over one service."""
    rag = service or rest_app.state.rag
    mcp_path = os.getenv("MCP_PATH", "/mcp")
    mcp = create_mcp(
        rag,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    mcp_http = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with mcp.session_manager.run():
            yield

    app = create_app(rag, lifespan=lifespan)
    app.mount(mcp_path, mcp_http)
    app.state.mcp = mcp
    return app


app = create_combined_app()
