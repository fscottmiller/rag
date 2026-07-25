import pytest

from src.mcp_server.server import create_mcp


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
