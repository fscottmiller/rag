"""Regression tests for MCP tool authorization (finding F2).

Before this file existed, `tests/test_mcp.py` never constructed a service in
`trusted-proxy` auth mode, so the `authorize(...)` calls in
`src/utralight_rag/mcp_server/server.py` were completely unexercised: deleting
every `authorize(...)` call from every tool still left the full suite green.
These tests lock in the denial behavior (and non-mutation on denied writes)
for both ways a tool can be invoked, and also cover the distinguishable
`authentication_required` / `not_authorized` error mapping added for finding
F11(a).
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from utralight_rag.combined import create_combined_app
from utralight_rag.mcp_server.server import create_mcp
from utralight_rag.service import RAGService


@pytest.fixture
def trusted_service(service: RAGService) -> RAGService:
    """The shared `service` fixture (from conftest.py), reconfigured to run in
    `trusted-proxy` auth mode against the same store/embedder/chunker."""
    return RAGService(
        service.store,
        service.embedder,
        service.chunker,
        replace(service.settings, auth_mode="trusted-proxy", trusted_hosts=("testserver",)),
    )


def _mcp_session(client: TestClient) -> str:
    """Initialize an MCP session over the mounted streamable-HTTP transport and
    return its `mcp-session-id`, mirroring the working pattern already used in
    tests/test_mcp.py (see test_combined_app_mounts_streamable_http_over_the_same_service)."""
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
        headers={"accept": "application/json, text/event-stream"},
    )
    assert response.status_code == 200
    session_id = response.headers["mcp-session-id"]
    notified = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers={"accept": "application/json, text/event-stream", "mcp-session-id": session_id},
    )
    assert notified.status_code == 202
    return session_id


def _call_tool(
    client: TestClient,
    session_id: str,
    name: str,
    arguments: dict[str, Any],
    *,
    role: str | None = None,
    user: str | None = None,
) -> dict[str, Any]:
    """POST a `tools/call` JSON-RPC request with optional trusted-proxy headers
    and return the parsed `result` object from the (SSE-framed) response."""
    headers = {"accept": "application/json, text/event-stream", "mcp-session-id": session_id}
    if user is not None:
        headers["Cf-Access-Authenticated-User-Email"] = user
    if role is not None:
        headers["X-Auth-Request-Role"] = role
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        headers=headers,
    )
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        return response.json()["result"]
    for line in response.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:") :])["result"]
    raise AssertionError(f"no JSON-RPC result found in response body: {response.text!r}")


# ---------------------------------------------------------------------------
# Path 1: direct `await server.call_tool(...)`, bypassing HTTP entirely.
#
# `ctx.request_context.request` raises ValueError with no live HTTP request
# behind the call, so `authorize()` always resolves to empty headers here --
# this is exactly what happens under the stdio transport too, since stdio has
# no HTTP request at all. In `trusted-proxy` mode that means every tool call
# is denied, regardless of what role a caller might otherwise have held.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_call_with_no_headers_denies_a_read_tool(trusted_service):
    server = create_mcp(trusted_service)
    result = await server.call_tool("list_documents", {})
    assert result.isError is True
    assert result.meta == {"error_type": "authentication_required"}
    assert "Trusted proxy identity is required" in result.content[0].text


@pytest.mark.asyncio
async def test_direct_call_with_no_headers_denies_a_write_tool_without_mutating(trusted_service):
    server = create_mcp(trusted_service)
    result = await server.call_tool("upload_document", {"title": "MCP guide", "content": "python"})
    assert result.isError is True
    assert result.meta == {"error_type": "authentication_required"}
    assert "Trusted proxy identity is required" in result.content[0].text
    assert trusted_service.list_documents() == []


@pytest.mark.asyncio
async def test_direct_call_denies_every_tool_documenting_the_stdio_edge_case(trusted_service):
    """Under stdio there is no HTTP request, so `authorize` always sees empty
    headers and `trusted-proxy` mode denies every tool call -- this is real,
    current behavior (a sharp operational edge), not a bug."""
    server = create_mcp(trusted_service)
    for name, arguments in [
        ("rag_search", {"query": "python"}),
        ("list_documents", {}),
        ("get_document", {"document_id": "missing"}),
        ("upload_document", {"title": "t", "content": "python"}),
        ("delete_document", {"document_id": "missing"}),
    ]:
        result = await server.call_tool(name, arguments)
        assert result.isError is True, f"{name} should have been denied"
        assert result.meta == {"error_type": "authentication_required"}
    assert trusted_service.list_documents() == []


# ---------------------------------------------------------------------------
# Path 2: through the mounted streamable-HTTP app, where real headers flow.
# ---------------------------------------------------------------------------


def test_http_no_headers_denies_read_and_write_tools(trusted_service):
    app = create_combined_app(trusted_service)
    with TestClient(app, base_url="http://testserver") as client:
        session_id = _mcp_session(client)

        read_result = _call_tool(client, session_id, "list_documents", {})
        assert read_result["isError"] is True
        assert read_result["_meta"]["error_type"] == "authentication_required"

        write_result = _call_tool(
            client, session_id, "upload_document", {"title": "t", "content": "python"}
        )
        assert write_result["isError"] is True
        assert write_result["_meta"]["error_type"] == "authentication_required"

    assert trusted_service.list_documents() == []


def test_http_reader_role_can_read_but_not_write(trusted_service):
    admin_document = trusted_service.ingest("Seed", "python guide")

    app = create_combined_app(trusted_service)
    with TestClient(app, base_url="http://testserver") as client:
        session_id = _mcp_session(client)
        reader = {"role": "reader", "user": "alice@example.com"}

        search_result = _call_tool(client, session_id, "rag_search", {"query": "python"}, **reader)
        assert search_result["isError"] is False

        list_result = _call_tool(client, session_id, "list_documents", {}, **reader)
        assert list_result["isError"] is False
        assert [d["id"] for d in list_result["structuredContent"]["result"]] == [
            admin_document["id"]
        ]

        get_result = _call_tool(
            client, session_id, "get_document", {"document_id": admin_document["id"]}, **reader
        )
        assert get_result["isError"] is False

        upload_result = _call_tool(
            client, session_id, "upload_document", {"title": "t", "content": "python"}, **reader
        )
        assert upload_result["isError"] is True
        assert upload_result["_meta"]["error_type"] == "not_authorized"
        assert (
            "This role is not allowed to perform this action" in upload_result["content"][0]["text"]
        )

        delete_result = _call_tool(
            client,
            session_id,
            "delete_document",
            {"document_id": admin_document["id"]},
            **reader,
        )
        assert delete_result["isError"] is True
        assert delete_result["_meta"]["error_type"] == "not_authorized"

    # The reader's denied write and delete must not have mutated the index.
    assert [d["id"] for d in trusted_service.list_documents()] == [admin_document["id"]]


def test_http_admin_role_can_perform_every_action(trusted_service):
    app = create_combined_app(trusted_service)
    with TestClient(app, base_url="http://testserver") as client:
        session_id = _mcp_session(client)
        admin = {"role": "admin", "user": "root@example.com"}

        upload_result = _call_tool(
            client, session_id, "upload_document", {"title": "t", "content": "python"}, **admin
        )
        assert upload_result["isError"] is False
        document_id = upload_result["structuredContent"]["id"]

        assert _call_tool(client, session_id, "list_documents", {}, **admin)["isError"] is False
        assert (
            _call_tool(client, session_id, "get_document", {"document_id": document_id}, **admin)[
                "isError"
            ]
            is False
        )
        assert (
            _call_tool(client, session_id, "rag_search", {"query": "python"}, **admin)["isError"]
            is False
        )

        delete_result = _call_tool(
            client, session_id, "delete_document", {"document_id": document_id}, **admin
        )
        assert delete_result["isError"] is False

    assert trusted_service.list_documents() == []


def test_http_unknown_role_is_denied_for_read_and_write(trusted_service):
    app = create_combined_app(trusted_service)
    with TestClient(app, base_url="http://testserver") as client:
        session_id = _mcp_session(client)
        unknown = {"role": "superuser", "user": "mallory@example.com"}

        read_result = _call_tool(client, session_id, "list_documents", {}, **unknown)
        assert read_result["isError"] is True
        assert read_result["_meta"]["error_type"] == "not_authorized"

        write_result = _call_tool(
            client, session_id, "upload_document", {"title": "t", "content": "python"}, **unknown
        )
        assert write_result["isError"] is True
        assert write_result["_meta"]["error_type"] == "not_authorized"

    assert trusted_service.list_documents() == []
