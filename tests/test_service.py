import pytest

from utralight_rag.config import Settings
from utralight_rag.service import RAGService
from utralight_rag.storage.sqlite import DocumentNotFoundError, SQLiteStore


def test_document_crud_and_cascade(service):
    created = service.ingest("Guide", "Python intro | SQLite storage", {"team": "search"})
    assert created["chunk_count"] == 2
    assert service.list_documents()[0]["id"] == created["id"]
    assert service.get_document(created["id"])["content"] == "Python intro | SQLite storage"

    updated = service.update(created["id"], "Updated", "FastAPI endpoint", {"team": "api"})
    assert updated["title"] == "Updated"
    assert updated["chunk_count"] == 1
    assert service.search("fastapi")[0]["document_id"] == created["id"]

    service.delete_document(created["id"])
    assert service.list_documents() == []
    assert service.search("fastapi") == []
    with pytest.raises(DocumentNotFoundError):
        service.get_document(created["id"])


def test_vector_search_and_metadata_filter(service):
    python_doc = service.ingest("Python", "Python language", {"topic": "code"})
    service.ingest("SQLite", "SQLite vector storage", {"topic": "data"})

    result = service.search("python", top_k=2)
    assert result[0]["document_id"] == python_doc["id"]
    assert result[0]["score"] == pytest.approx(1.0)
    filtered = service.search("sqlite", filter_metadata={"topic": "code"})
    assert filtered and all(item["metadata"]["topic"] == "code" for item in filtered)
    assert service.search("sqlite", filter_metadata={"topic": "missing"}) == []


def test_service_uses_one_configured_chunking_profile(service):
    configured = RAGService(
        SQLiteStore(),
        service.embedder,
        settings=Settings(chunker="token", chunk_size=2, chunk_overlap=0),
    )
    first = configured.ingest("First", "one two three four")
    second = configured.ingest("Second", "one two three four")
    assert first["chunk_count"] == second["chunk_count"]
    assert first["chunk_count"] >= 2


def test_service_rejects_bad_titles(service):
    created = service.ingest("Guide", "content")
    with pytest.raises(ValueError, match="title"):
        service.ingest("   ", "content")
    with pytest.raises(ValueError, match="title"):
        service.update(created["id"], "   ", "content")


def test_invalid_content_query_and_top_k(service):
    with pytest.raises(ValueError):
        service.ingest("Empty", "   ")
    with pytest.raises(ValueError, match="query"):
        service.search("   ")
    with pytest.raises(ValueError, match="between 1 and 100"):
        service.search("query", top_k=0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        service.search("query", top_k=101)
