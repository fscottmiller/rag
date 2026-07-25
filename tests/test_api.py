from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import Settings
from src.service import RAGService
from src.storage.sqlite import SQLiteStore


def test_rest_lifecycle(service):
    client = TestClient(create_app(service))
    response = client.post(
        "/documents", json={"title": "REST", "content": "one|two", "metadata": {"source": "test"}}
    )
    assert response.status_code == 201
    document = response.json()
    document_id = document["id"]
    assert document["chunk_count"] == 2

    assert client.get("/documents").json()[0]["id"] == document_id
    assert client.get(f"/documents/{document_id}").json()["content"] == "one|two"
    response = client.put(
        f"/documents/{document_id}", json={"title": "REST 2", "content": "replacement"}
    )
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


def test_rest_not_found_and_validation_errors(service):
    client = TestClient(create_app(service))
    missing = "missing-document"
    assert client.get(f"/documents/{missing}").status_code == 404
    assert (
        client.put(f"/documents/{missing}", json={"title": "x", "content": "x"}).status_code == 404
    )
    assert client.delete(f"/documents/{missing}").status_code == 404
    assert client.post("/documents", json={"title": "", "content": "x"}).status_code == 422
    assert client.post("/documents", json={"title": "x", "content": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 101}).status_code == 422
    assert client.post("/search", json={"query": " "}).status_code == 400


def test_rest_rejects_per_document_index_options(service):
    client = TestClient(create_app(service))
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"one two three four")},
        data={"chunk_size": "2", "title": "Token notes"},
    )
    assert response.status_code == 422
    assert (
        client.post(
            "/documents",
            json={"title": "Token notes", "content": "content", "embedding_choice": "other"},
        ).status_code
        == 422
    )


def test_rest_rejects_bad_form_metadata_and_bad_json(service):
    client = TestClient(create_app(service))
    bad_metadata = client.post(
        "/documents",
        files={"file": ("notes.txt", b"content")},
        data={"metadata": "not-json"},
    )
    assert bad_metadata.status_code == 422
    bad_json = client.post(
        "/documents",
        content="{not-json",
        headers={"content-type": "application/json"},
    )
    assert bad_json.status_code == 422


def test_rest_form_content_and_service_validation_errors(service):
    client = TestClient(create_app(service))
    form = client.post("/documents", data={"title": "Form", "content": "plain text"})
    assert form.status_code == 201
    assert form.json()["title"] == "Form"

    empty_create = client.post("/documents", json={"title": "Whitespace", "content": " "})
    assert empty_create.status_code == 400
    document_id = form.json()["id"]
    empty_update = client.put(
        f"/documents/{document_id}",
        json={"title": " ", "content": "replacement"},
    )
    assert empty_update.status_code == 400


def test_trusted_proxy_roles_control_document_mutations(service):
    protected_service = RAGService(
        SQLiteStore(),
        service.embedder,
        service.chunker,
        Settings(auth_mode="trusted-proxy"),
    )
    client = TestClient(create_app(protected_service))
    reader_headers = {
        "Cf-Access-Authenticated-User-Email": "reader@example.test",
        "X-Auth-Request-Role": "reader",
    }
    admin_headers = {
        "Cf-Access-Authenticated-User-Email": "admin@example.test",
        "X-Auth-Request-Role": "admin",
    }

    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers=reader_headers).status_code == 200
    assert client.post(
        "/documents",
        json={"title": "Reader", "content": "content"},
        headers=reader_headers,
    ).status_code == 403

    created = client.post(
        "/documents",
        json={"title": "Admin", "content": "content"},
        headers=admin_headers,
    )
    assert created.status_code == 201
    document_id = created.json()["id"]
    assert client.get(f"/documents/{document_id}", headers=reader_headers).status_code == 200
    assert client.post("/search", json={"query": "content"}, headers=reader_headers).status_code == 200
    assert client.put(
        f"/documents/{document_id}",
        json={"title": "Changed", "content": "content"},
        headers=reader_headers,
    ).status_code == 403
    assert client.delete(f"/documents/{document_id}", headers=reader_headers).status_code == 403
    assert client.delete(f"/documents/{document_id}", headers=admin_headers).status_code == 204
