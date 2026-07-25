from fastapi.testclient import TestClient

from src.api.main import create_app


def test_rest_lifecycle(service):
    client = TestClient(create_app(service))
    response = client.post("/documents", json={"title": "REST", "content": "one|two", "metadata": {"source": "test"}})
    assert response.status_code == 201
    document = response.json()
    document_id = document["id"]
    assert document["chunk_count"] == 2

    assert client.get("/documents").json()[0]["id"] == document_id
    assert client.get(f"/documents/{document_id}").json()["content"] == "one|two"
    response = client.put(f"/documents/{document_id}", json={"title": "REST 2", "content": "replacement"})
    assert response.status_code == 200
    assert response.json()["title"] == "REST 2"
    assert client.delete(f"/documents/{document_id}").status_code == 204
    assert client.get(f"/documents/{document_id}").status_code == 404


def test_rest_search(service):
    client = TestClient(create_app(service))
    client.post("/documents", json={"title": "Search", "content": "Python content"})
    response = client.post("/search", json={"query": "python"})
    assert response.status_code == 200
    assert response.json()[0]["title"] == "Search"


def test_rest_file_upload(service):
    client = TestClient(create_app(service))
    response = client.post(
        "/documents",
        files={"file": ("notes.md", b"uploaded|markdown")},
        data={"metadata": '{"source": "file"}'},
    )
    assert response.status_code == 201
    assert response.json()["title"] == "notes.md"
    assert response.json()["metadata"] == {"source": "file"}

