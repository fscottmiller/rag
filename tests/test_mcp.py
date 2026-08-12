import sqlite3
import subprocess
import sys
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from mcp.server.fastmcp.exceptions import ToolError

from ultralight_rag.combined import create_combined_app
from ultralight_rag.mcp_server.server import create_mcp, get_transport
from ultralight_rag.service import RAGService


def test_mcp_transport_defaults_to_stdio_and_only_allows_streamable_http(monkeypatch):
    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    assert get_transport() == "stdio"
    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    assert get_transport() == "streamable-http"
    monkeypatch.setenv("MCP_TRANSPORT", "sse")
    with pytest.raises(ValueError, match="stdio.*streamable-http"):
        get_transport()


def test_combined_import_does_not_create_an_unused_rest_app():
    import ultralight_rag.api as api
    import ultralight_rag.api.main as api_main

    assert api_main._default_app is None
    assert api_main.app is api_main.get_app()
    assert api.app is api_main.get_app()


def test_combined_bare_import_does_not_construct_a_service():
    """A bare `import ultralight_rag.combined` must stay side-effect-free.

    `create_combined_app()` opens a SQLite connection and constructs an
    embedder, so running it eagerly at module scope -- the historical bug
    this guards against -- would mean import alone (a test collecting this
    module, a docs build, `--help`) pays that cost. Checked from a fresh
    subprocess rather than in-process: by the time this test runs, other
    tests in this module have already called `create_combined_app` many
    times over, and pytest-randomly may reorder collection, so nothing about
    in-process module state before or after those calls would reliably
    isolate "did the bare import itself construct anything."
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            # Assert the side effect, not just the sentinel. `_default_app is
            # None` alone passes even with the historical bug restored, because
            # a module-scope `app = create_combined_app()` binds `app` directly
            # and never touches the lazy singleton -- so the connection and the
            # embedder are built while the sentinel still reads None. The
            # sqlite3.connect spy catches the cost itself, and the vars() check
            # catches the eager binding: __getattr__ only fires for names absent
            # from the module dict, so `'app' not in vars(...)` is exactly the
            # laziness invariant.
            "import sqlite3\n"
            "_connect, opened = sqlite3.connect, []\n"
            "sqlite3.connect = lambda *a, **k: (opened.append(a), _connect(*a, **k))[1]\n"
            "import ultralight_rag.combined as combined\n"
            "assert not opened, f'bare import opened a database: {opened}'\n"
            "assert 'app' not in vars(combined), 'bare import bound app eagerly'\n"
            "assert combined._default_app is None, 'bare import realized the lazy app'\n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_combined_app_attribute_lazily_realizes_the_singleton():
    import ultralight_rag.combined as combined

    combined._default_app = None
    try:
        assert combined.app is combined.get_app()
        assert combined.app is combined._default_app
    finally:
        combined._default_app = None


def test_combined_getattr_rejects_unknown_names():
    import ultralight_rag.combined as combined

    with pytest.raises(AttributeError):
        _ = combined.not_a_real_attribute


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
    with pytest.raises(ToolError, match="Document not found"):
        await server.call_tool("get_document", {"document_id": document_id})
    with pytest.raises(ToolError, match="Document not found"):
        await server.call_tool("delete_document", {"document_id": document_id})
    _, listed = await server.call_tool("list_documents", {})
    assert listed["result"] == []


@pytest.mark.asyncio
async def test_mcp_search_top_k_advertises_the_same_bounds_as_rest(service):
    """REST's SearchPayload.top_k is `ge=1, le=100` (src/ultralight_rag/api/models.py).
    rag_search's inputSchema previously advertised no bounds at all, so a client
    had no way to discover the limit short of trying an out-of-range value."""
    server = create_mcp(service)
    tools = await server.list_tools()
    top_k_schema = next(tool for tool in tools if tool.name == "rag_search").inputSchema[
        "properties"
    ]["top_k"]
    assert top_k_schema["minimum"] == 1
    assert top_k_schema["maximum"] == 100
    assert top_k_schema["default"] == 5


@pytest.mark.asyncio
async def test_mcp_output_schemas_are_unaffected_by_the_call_tool_result_annotation(service):
    """rag_search/list_documents/get_document/upload_document/delete_document are
    all annotated `Annotated[CallToolResult, RealReturnType]` (see server.py) so
    the tools can return a `CallToolResult` directly on the authorization-denial
    and provider-failure paths while the *advertised* structured-output schema
    is still derived from `RealReturnType`, byte-for-byte identical to what it
    would be for a plain `-> RealReturnType` annotation. This locks that contract
    in: the schemas below are exactly what each tool advertised before N3's
    return-type annotations were widened, captured with `list_tools()` against
    both the old and new code and diffed to confirm zero drift."""
    server = create_mcp(service)
    schemas = {tool.name: tool.outputSchema for tool in await server.list_tools()}

    for name in ("rag_search", "list_documents"):
        schema = schemas[name]
        assert schema["type"] == "object"
        assert schema["required"] == ["result"]
        assert schema["properties"]["result"]["type"] == "array"
        assert schema["properties"]["result"]["items"] == {
            "type": "object",
            "additionalProperties": True,
        }

    for name in ("get_document", "upload_document"):
        schema = schemas[name]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is True
        assert "required" not in schema

    delete_schema = schemas["delete_document"]
    assert delete_schema["type"] == "object"
    assert delete_schema["additionalProperties"] == {"type": "string"}


@pytest.mark.asyncio
async def test_mcp_document_tools_offload_storage_calls(service, monkeypatch):
    document = service.ingest("MCP guide", "Python context")
    calls = []

    async def run_sync(function, *args):
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr("ultralight_rag.mcp_server.server.anyio.to_thread.run_sync", run_sync)
    server = create_mcp(service)
    await server.call_tool("list_documents", {})
    await server.call_tool("get_document", {"document_id": document["id"]})
    await server.call_tool("delete_document", {"document_id": document["id"]})

    assert calls == [
        (service.list_documents, ()),
        (service.get_document, (document["id"],)),
        (service.delete_document, (document["id"],)),
    ]


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


def test_combined_lifespan_closes_the_store_it_constructed():
    """A `create_combined_app()` with no injected service owns the `RAGService`
    it builds internally, so its store should not outlive the app.
    """
    app = create_combined_app()
    store = app.state.rag.store
    with TestClient(app):
        store.connection.execute("SELECT 1")  # still open while the app is up
    with pytest.raises(sqlite3.ProgrammingError):
        store.connection.execute("SELECT 1")


def test_combined_lifespan_leaves_an_injected_service_store_open(service):
    """A caller who injects `service` still holds that reference and may
    reuse it after this app shuts down, so the lifespan must not close a
    store it does not own.
    """
    app = create_combined_app(service)
    with TestClient(app):
        pass
    service.store.connection.execute("SELECT 1")  # still open after shutdown


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


@pytest.mark.parametrize(
    ("trusted_hosts", "base_url"),
    [("*", "https://any.example"), ("*.example.com", "https://mcp.example.com")],
)
def test_mcp_supports_starlette_trusted_host_patterns(service, trusted_hosts, base_url):
    public_service = RAGService(
        service.store,
        service.embedder,
        service.chunker,
        replace(service.settings, trusted_hosts=(trusted_hosts,)),
    )
    with TestClient(create_combined_app(public_service), base_url=base_url) as client:
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
            headers={"accept": "application/json, text/event-stream", "origin": base_url},
        )
    assert response.status_code == 200


def test_mcp_rejects_untrusted_origin_for_wildcard_host(service):
    public_service = RAGService(
        service.store,
        service.embedder,
        service.chunker,
        replace(service.settings, trusted_hosts=("*.example.com",)),
    )
    with TestClient(
        create_combined_app(public_service), base_url="https://mcp.example.com"
    ) as client:
        response = client.post(
            "/mcp",
            json={},
            headers={"accept": "application/json", "origin": "https://evil.example"},
        )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "origin",
    [
        "http://mcp.example.com",
        "https://mcp.example.com:444",
        "https://sibling.example.com",
        "https://[",
    ],
)
def test_mcp_rejects_cross_origin_for_wildcard_host(service, origin):
    public_service = RAGService(
        service.store,
        service.embedder,
        service.chunker,
        replace(service.settings, trusted_hosts=("*.example.com",)),
    )
    with TestClient(
        create_combined_app(public_service), base_url="https://mcp.example.com"
    ) as client:
        response = client.post(
            "/mcp", json={}, headers={"accept": "application/json", "origin": origin}
        )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_mcp_origin_middleware_rejects_cross_origin():
    from ultralight_rag.combined import MCPOriginMiddleware

    async def inner_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"OK"})

    middleware = MCPOriginMiddleware(inner_app)

    scope = {
        "type": "http",
        "headers": [
            (b"origin", b"https://evil.example"),
            (b"host", b"good.example"),
        ],
        "server": ("good.example", 443),
        "path": "/",
        "query_string": b"",
    }

    responses = []

    async def send(message):
        responses.append(message)

    async def receive():
        return {"type": "http.request", "body": b""}

    await middleware(scope, receive, send)

    assert len(responses) == 2
    assert responses[0] == {
        "type": "http.response.start",
        "status": 403,
        "headers": [
            (b"content-length", b"21"),
            (b"content-type", b"text/plain; charset=utf-8"),
        ],
    }
    assert responses[1] == {"type": "http.response.body", "body": b"Invalid Origin header"}


@pytest.mark.asyncio
async def test_mcp_origin_middleware_allows_same_origin():
    from ultralight_rag.combined import MCPOriginMiddleware

    async def inner_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"OK"})

    middleware = MCPOriginMiddleware(inner_app)

    scope = {
        "type": "http",
        "headers": [
            (b"origin", b"https://good.example"),
            (b"host", b"good.example"),
        ],
        "server": ("good.example", 443),
        "path": "/",
        "query_string": b"",
        "scheme": "https",
    }

    responses = []

    async def send(message):
        responses.append(message)

    async def receive():
        return {"type": "http.request", "body": b""}

    await middleware(scope, receive, send)

    assert len(responses) == 2
    assert responses[0] == {"type": "http.response.start", "status": 200}
    assert responses[1] == {"type": "http.response.body", "body": b"OK"}
