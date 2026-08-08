from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from ultralight_rag.api.main import BodySizeLimitMiddleware, create_app
from ultralight_rag.config import Settings
from ultralight_rag.service import RAGService
from ultralight_rag.storage.sqlite import SQLiteStore


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


def test_rest_rejects_untrusted_host_even_when_origin_matches(service):
    client = TestClient(create_app(service))
    response = client.post(
        "/documents",
        json={"title": "Injected", "content": "dns rebinding"},
        headers={"Origin": "http://evil.example", "Host": "evil.example"},
    )
    assert response.status_code == 400
    assert service.list_documents() == []


def test_rest_rejects_cross_origin_mutations_in_open_mode(service):
    client = TestClient(create_app(service))
    response = client.post(
        "/documents",
        data={"title": "Injected", "content": "cross-origin"},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403
    assert service.list_documents() == []


def test_rest_rejects_same_host_with_different_scheme(service):
    with TestClient(create_app(service), base_url="https://testserver") as client:
        response = client.post(
            "/documents",
            json={"title": "Injected", "content": "cross-scheme"},
            headers={"Origin": "http://testserver"},
        )
    assert response.status_code == 403
    assert service.list_documents() == []


def test_rest_rejects_explicit_port_zero_origin(service):
    with TestClient(create_app(service), base_url="https://testserver") as client:
        response = client.post(
            "/documents",
            json={"title": "Injected", "content": "cross-port"},
            headers={"Origin": "https://testserver:0"},
        )
    assert response.status_code == 403
    assert service.list_documents() == []


@pytest.mark.parametrize(
    "origin", ["https://user@testserver", "https://testserver:bad", "https://["]
)
def test_rest_rejects_malformed_origin(service, origin):
    with TestClient(create_app(service), base_url="https://testserver") as client:
        response = client.post(
            "/documents",
            json={"title": "Injected", "content": "malformed-origin"},
            headers={"Origin": origin},
        )
    assert response.status_code == 403
    assert service.list_documents() == []


def test_rest_rejects_documents_over_configured_limit(service):
    limited = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=8)
    )
    client = TestClient(create_app(limited))

    valid = client.post("/documents", json={"title": "Small", "content": "1"})
    assert valid.status_code == 201
    response = client.post("/documents", json={"title": "Large", "content": "123456789"})
    assert response.status_code == 413

    created = valid.json()["id"]
    response = client.put(
        f"/documents/{created}", json={"title": "Still small", "content": "123456789"}
    )
    assert response.status_code == 413


def _post_document_at_boundary(client, path, content):
    if path == "json":
        return client.post("/documents", json={"title": "Boundary", "content": content})
    if path == "multipart":
        return client.post(
            "/documents",
            files={"file": ("notes.txt", content.encode("utf-8"), "text/plain")},
        )
    return client.post("/documents", data={"title": "Boundary", "content": content})


@pytest.mark.parametrize("path", ["json", "multipart", "form"])
def test_rest_accepts_document_content_at_exact_byte_limit(service, path):
    limit = 8
    limited = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=limit)
    )
    client = TestClient(create_app(limited))
    content = "x" * limit
    assert len(content.encode("utf-8")) == limit

    response = _post_document_at_boundary(client, path, content)
    assert response.status_code == 201


@pytest.mark.parametrize("path", ["json", "multipart", "form"])
def test_rest_rejects_document_content_one_byte_over_limit(service, path):
    limit = 8
    limited = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=limit)
    )
    client = TestClient(create_app(limited))
    content = "x" * (limit + 1)
    assert len(content.encode("utf-8")) == limit + 1

    response = _post_document_at_boundary(client, path, content)
    assert response.status_code == 413


def test_rest_rejects_multibyte_content_by_utf8_byte_length_not_char_length(service):
    # "e" with an acute accent encodes to 2 UTF-8 bytes each: 2 characters, 4 bytes.
    # A char-length check (len(content)) would wrongly accept this against a limit of
    # 3; only a correct byte-length check rejects it with 413.
    limit = 3
    limited = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=limit)
    )
    client = TestClient(create_app(limited))
    content = "éé"
    assert len(content) <= limit
    assert len(content.encode("utf-8")) > limit

    response = client.post("/documents", json={"title": "Multibyte", "content": content})
    assert response.status_code == 413


def test_rest_put_accepts_and_rejects_document_content_at_exact_byte_limit(service):
    limit = 8
    limited = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=limit)
    )
    client = TestClient(create_app(limited))
    created = client.post("/documents", json={"title": "Seed", "content": "seed"})
    document_id = created.json()["id"]

    at_limit = client.put(
        f"/documents/{document_id}", json={"title": "Exact", "content": "x" * limit}
    )
    assert at_limit.status_code == 200

    over_limit = client.put(
        f"/documents/{document_id}", json={"title": "Over", "content": "x" * (limit + 1)}
    )
    assert over_limit.status_code == 413


def test_rest_put_accepts_multipart_file_upload(service):
    client = TestClient(create_app(service))
    created = client.post("/documents", json={"title": "Seed", "content": "seed content"})
    document_id = created.json()["id"]

    response = client.put(
        f"/documents/{document_id}",
        files={"file": ("updated.md", b"replaced|via|multipart")},
        data={"metadata": '{"source": "file-update"}'},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "updated.md"
    assert body["metadata"] == {"source": "file-update"}

    fetched = client.get(f"/documents/{document_id}")
    assert fetched.json()["content"] == "replaced|via|multipart"


def test_rest_put_accepts_plain_form_fields(service):
    client = TestClient(create_app(service))
    created = client.post("/documents", json={"title": "Seed", "content": "seed content"})
    document_id = created.json()["id"]

    response = client.put(
        f"/documents/{document_id}",
        data={"title": "Form Update", "content": "replaced via form"},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Form Update"

    fetched = client.get(f"/documents/{document_id}")
    assert fetched.json()["content"] == "replaced via form"


def _put_document_at_boundary(client, document_id, path, content):
    if path == "json":
        return client.put(
            f"/documents/{document_id}", json={"title": "Boundary", "content": content}
        )
    if path == "multipart":
        return client.put(
            f"/documents/{document_id}",
            files={"file": ("notes.txt", content.encode("utf-8"), "text/plain")},
        )
    return client.put(f"/documents/{document_id}", data={"title": "Boundary", "content": content})


@pytest.mark.parametrize("path", ["json", "multipart", "form"])
def test_rest_put_accepts_document_content_at_exact_byte_limit_all_content_types(service, path):
    limit = 8
    limited = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=limit)
    )
    client = TestClient(create_app(limited))
    created = client.post("/documents", json={"title": "Seed", "content": "seedling"})
    document_id = created.json()["id"]
    content = "x" * limit
    assert len(content.encode("utf-8")) == limit

    response = _put_document_at_boundary(client, document_id, path, content)
    assert response.status_code == 200


@pytest.mark.parametrize("path", ["json", "multipart", "form"])
def test_rest_put_rejects_document_content_one_byte_over_limit_all_content_types(service, path):
    limit = 8
    limited = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=limit)
    )
    client = TestClient(create_app(limited))
    created = client.post("/documents", json={"title": "Seed", "content": "seedling"})
    document_id = created.json()["id"]
    content = "x" * (limit + 1)
    assert len(content.encode("utf-8")) == limit + 1

    response = _put_document_at_boundary(client, document_id, path, content)
    assert response.status_code == 413


def test_rest_put_rejects_per_document_index_options(service):
    client = TestClient(create_app(service))
    created = client.post("/documents", json={"title": "Seed", "content": "seed content"})
    document_id = created.json()["id"]

    response = client.put(
        f"/documents/{document_id}",
        files={"file": ("notes.txt", b"one two three four")},
        data={"chunk_size": "2", "title": "Token notes"},
    )
    assert response.status_code == 422
    assert (
        client.put(
            f"/documents/{document_id}",
            json={"title": "Token notes", "content": "content", "embedding_choice": "other"},
        ).status_code
        == 422
    )


def test_rest_rejects_large_request_before_parsing(service):
    limited = RAGService(
        SQLiteStore(),
        service.embedder,
        service.chunker,
        Settings(max_document_bytes=8, max_request_bytes=32),
    )
    client = TestClient(create_app(limited))
    response = client.post(
        "/documents",
        content=b"not parsed because this body is too large",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413


async def test_body_size_limit_rejects_chunked_body_before_parsing():
    called = False

    async def inner(_scope, _receive, _send):
        nonlocal called
        called = True

    messages = [
        {"type": "http.request", "body": b"123", "more_body": True},
        {"type": "http.request", "body": b"45", "more_body": False},
    ]

    async def receive():
        return messages.pop(0)

    sent = []

    async def send(message):
        sent.append(message)

    await BodySizeLimitMiddleware(inner, max_bytes=4)(
        {"type": "http", "headers": []}, receive, send
    )

    assert not called
    assert sent[0]["status"] == 413


async def test_body_size_limit_rejects_invalid_content_length():
    sent = []

    async def send(message):
        sent.append(message)

    await BodySizeLimitMiddleware(None, max_bytes=4)(
        {"type": "http", "headers": [(b"content-length", b"invalid")]},
        None,
        send,
    )

    assert sent[0]["status"] == 413


def test_rest_rejects_large_file_upload(service):
    limited = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=8)
    )
    client = TestClient(create_app(limited))

    response = client.post("/documents", files={"file": ("large.txt", b"123456789", "text/plain")})
    assert response.status_code == 413


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
    assert client.post("/search", json={"query": "x", "top_k": True}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": "1"}).status_code == 422
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
    assert (
        client.post(
            "/documents",
            json={"title": "Reader", "content": "content"},
            headers=reader_headers,
        ).status_code
        == 403
    )

    created = client.post(
        "/documents",
        json={"title": "Admin", "content": "content"},
        headers=admin_headers,
    )
    assert created.status_code == 201
    document_id = created.json()["id"]
    assert client.get(f"/documents/{document_id}", headers=reader_headers).status_code == 200
    assert (
        client.post("/search", json={"query": "content"}, headers=reader_headers).status_code == 200
    )
    assert (
        client.put(
            f"/documents/{document_id}",
            json={"title": "Changed", "content": "content"},
            headers=reader_headers,
        ).status_code
        == 403
    )
    assert client.delete(f"/documents/{document_id}", headers=reader_headers).status_code == 403
    assert client.delete(f"/documents/{document_id}", headers=admin_headers).status_code == 204


def test_rest_runtime_error_returns_500():
    mock_service = Mock(spec=RAGService)
    mock_service.settings = Mock()
    mock_service.settings.max_request_bytes = 1000000
    mock_service.settings.trusted_hosts = ["*"]
    mock_service.settings.auth_mode = "none"

    mock_service.list_documents.side_effect = RuntimeError("Something went wrong")

    app = create_app(mock_service)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/documents")
    assert response.status_code == 500
