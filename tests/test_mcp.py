from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from utralight_rag.combined import create_combined_app
from utralight_rag.mcp_server.server import create_mcp, get_transport
from utralight_rag.service import RAGService


def test_mcp_transport_defaults_to_stdio_and_only_allows_streamable_http(monkeypatch):
    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    assert get_transport() == "stdio"
    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    assert get_transport() == "streamable-http"
    monkeypatch.setenv("MCP_TRANSPORT", "sse")
    with pytest.raises(ValueError, match="stdio.*streamable-http"):
        get_transport()


@pytest.mark.asyncio
async def test_mcp_tools_share_service_lifecycle(service):
    server = create_mcp(service)
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == {
        "rag_search",
        "list_documents",
        "get_document",
        "upload_document",
        "delete_document",
    }
    assert all("ctx" not in tool.inputSchema.get("properties", {}) for tool in tools)

    _, created = await server.call_tool(
        "upload_document",
        {"title": "MCP guide", "content": "Python context", "metadata": {"source": "mcp"}},
    )
    document_id = created["id"]
    assert created["metadata"] == {"source": "mcp"}

    _, listed = await server.call_tool("list_documents", {})
    assert listed["result"][0]["id"] == document_id
    _, document = await server.call_tool("get_document", {"document_id": document_id})
    assert document["content"] == "Python context"
    _, results = await server.call_tool("rag_search", {"query": "python"})
    assert results["result"][0]["document_id"] == document_id

    _, deleted = await server.call_tool("delete_document", {"document_id": document_id})
    assert deleted == {"status": "deleted", "document_id": document_id}
    _, listed = await server.call_tool("list_documents", {})
    assert listed["result"] == []


def test_combined_app_mounts_streamable_http_over_the_same_service(service):
    public_service = RAGService(
        service.store,
        service.embedder,
        service.chunker,
        replace(service.settings, trusted_hosts=("rag.example.com",)),
    )
    app = create_combined_app(public_service)
    assert app.state.rag is public_service
    assert app.state.mcp.session_manager is not None
    assert any(getattr(route, "path", None) == "/mcp" for route in app.routes)

    with TestClient(app, base_url="https://rag.example.com") as client:
        assert client.get("/documents").status_code == 200
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
        assert response.headers["mcp-session-id"]
        assert '"serverInfo"' in response.text


def test_mcp_accepts_trusted_host_with_nondefault_port(service):
    public_service = RAGService(
        service.store,
        service.embedder,
        service.chunker,
        replace(service.settings, trusted_hosts=("localhost",)),
    )
    app = create_combined_app(public_service)

    with TestClient(app, base_url="http://localhost:8000") as client:
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
