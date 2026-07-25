import pytest

from src.storage.sqlite import DocumentNotFoundError


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


def test_invalid_content_and_top_k(service):
    with pytest.raises(ValueError):
        service.ingest("Empty", "   ")
    with pytest.raises(ValueError):
        service.search("query", top_k=0)
