"""Regression tests for finding F12: logging + accurate embedding-provider error codes.

Covers:
  - the new EmbeddingProviderError hierarchy and its classification of every
    provider failure path (unavailable/timeout vs. invalid response),
  - REST mapping of that hierarchy to 503 / 502 for every route that can
    trigger embedding work (POST /documents, PUT /documents/{id}, POST /search),
  - that a genuine bug (a bare RuntimeError, not one of our exception types)
    is NOT reclassified and still surfaces as an opaque 500,
  - that upstream response detail is logged server-side but never appears in
    the HTTP response body or the MCP tool's text content,
  - MCP tool parity: rag_search / upload_document report embedding-provider
    failures as a structured CallToolResult with `_meta.error_type`, the same
    pattern already used for authorization denials,
  - the service-layer audit/observability logging (document ingest/update/
    delete/search identified by id; authorization denials identified by
    principal and action) and that document content/chunk text is never
    logged.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from utralight_rag.api.main import create_app
from utralight_rag.mcp_server.server import create_mcp
from utralight_rag.pipeline.embeddings import (
    BaseEmbedder,
    EmbeddingProviderError,
    EmbeddingProviderResponseError,
    EmbeddingProviderUnavailableError,
    OpenAICompatibleEmbedder,
)
from utralight_rag.service import RAGService


class FailingEmbedder(BaseEmbedder):
    """Deterministically raises a configured exception on every embed call,
    the same shape of failure a genuinely down/misbehaving provider produces."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def embed(self, texts):
        raise self._exc


def _failing_service(service: RAGService, exc: Exception) -> RAGService:
    return RAGService(service.store, FailingEmbedder(exc), service.chunker, service.settings)


# ---------------------------------------------------------------------------
# REST: status code mapping (section B) for every route that embeds.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (EmbeddingProviderUnavailableError("boom"), 503),
        (EmbeddingProviderResponseError("boom"), 502),
    ],
)
def test_rest_maps_embedding_provider_errors_to_accurate_status_codes(
    service, exc, expected_status
):
    failing = _failing_service(service, exc)
    client = TestClient(create_app(failing))

    create_response = client.post("/documents", json={"title": "Doc", "content": "python"})
    assert create_response.status_code == expected_status

    search_response = client.post("/search", json={"query": "python"})
    assert search_response.status_code == expected_status


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (EmbeddingProviderUnavailableError("boom"), 503),
        (EmbeddingProviderResponseError("boom"), 502),
    ],
)
def test_rest_maps_embedding_provider_errors_on_put(service, exc, expected_status):
    healthy_client = TestClient(create_app(service))
    created = healthy_client.post("/documents", json={"title": "Seed", "content": "python"})
    document_id = created.json()["id"]

    failing = _failing_service(service, exc)
    failing_client = TestClient(create_app(failing))
    update_response = failing_client.put(
        f"/documents/{document_id}", json={"title": "Seed", "content": "replacement"}
    )
    assert update_response.status_code == expected_status


def test_rest_bare_runtime_error_still_surfaces_as_500_not_reclassified(service):
    """A genuine bug elsewhere in the service (a plain RuntimeError, NOT an
    EmbeddingProviderError) must not be caught and relabeled as an upstream
    embedding-provider failure -- otherwise section A's whole point (don't
    catch bare RuntimeError, which would mask real bugs as 502/503) is
    defeated."""
    failing = _failing_service(service, RuntimeError("a real bug, not a provider failure"))
    # raise_server_exceptions=False mirrors how a real deployment behaves: an
    # unhandled exception becomes an opaque 500 response instead of propagating
    # out of the test call, which is exactly what section A requires -- a bare
    # RuntimeError must NOT be caught and relabeled as 502/503.
    client = TestClient(create_app(failing), raise_server_exceptions=False)

    create_response = client.post("/documents", json={"title": "Doc", "content": "python"})
    assert create_response.status_code == 500

    search_response = client.post("/search", json={"query": "python"})
    assert search_response.status_code == 500


def test_rest_does_not_leak_provider_detail_but_logs_it(service, caplog):
    """Reproduces the review's probe end-to-end through the real
    OpenAICompatibleEmbedder: a provider whose response contains a
    recognizable secret-like string must have that string in the log record,
    never in the HTTP response body (section C)."""
    leaked_detail = "acct_super_secret_9f31 quota exceeded at internal-upstream.example"
    body = json.dumps({"error": leaked_detail}).encode()
    with _local_error_server(500, body) as url:
        embedder = OpenAICompatibleEmbedder("model", url, provider="ollama")
        failing = RAGService(service.store, embedder, service.chunker, service.settings)
        client = TestClient(create_app(failing))

        with caplog.at_level(logging.WARNING, logger="utralight_rag"):
            response = client.post("/documents", json={"title": "Doc", "content": "python"})

    assert response.status_code == 502
    body_text = response.text
    assert leaked_detail not in body_text
    assert "acct_super_secret_9f31" not in body_text
    assert response.json()["detail"] == "Embedding provider returned an invalid response"

    logged_text = "\n".join(record.getMessage() for record in caplog.records)
    assert leaked_detail in logged_text


# ---------------------------------------------------------------------------
# Live-probe reproduction: actual HTTP failures classified through the real
# OpenAICompatibleEmbedder, not just the REST-layer mapping above.
# ---------------------------------------------------------------------------


@contextmanager
def _local_error_server(status: int, body: bytes):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/embeddings"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_openai_compatible_embedder_classifies_http_error_as_response_error(caplog):
    leaked_detail = "sk-live-should-never-reach-a-client 9f31"
    body = json.dumps({"error": leaked_detail}).encode()
    with _local_error_server(500, body) as url:
        embedder = OpenAICompatibleEmbedder("model", url, provider="ollama")
        with caplog.at_level(logging.WARNING, logger="utralight_rag"):
            with pytest.raises(EmbeddingProviderResponseError) as excinfo:
                embedder.embed(["one"])
    assert not isinstance(excinfo.value, EmbeddingProviderUnavailableError)
    # The raised exception (used for pytest.raises(RuntimeError, match=...) call
    # sites and for server-side logging) still carries the detail...
    assert leaked_detail in str(excinfo.value)
    # ...and the detail is logged server-side too, independent of whether an
    # adapter later chooses to forward the exception's str() to a client.
    assert leaked_detail in "\n".join(record.getMessage() for record in caplog.records)


def test_openai_compatible_embedder_classifies_unreachable_host_as_unavailable():
    embedder = OpenAICompatibleEmbedder(
        "model", "http://127.0.0.1:1/v1/embeddings", timeout=1, provider="ollama"
    )
    with pytest.raises(EmbeddingProviderUnavailableError) as excinfo:
        embedder.embed(["one"])
    assert not isinstance(excinfo.value, EmbeddingProviderResponseError)


def test_openai_compatible_embedder_classifies_malformed_json_as_response_error():
    with _local_error_server(200, b"{not-json") as url:
        embedder = OpenAICompatibleEmbedder("model", url, provider="ollama")
        with pytest.raises(EmbeddingProviderResponseError):
            embedder.embed(["one"])


def test_openai_compatible_embedder_classifies_shape_errors_as_response_error():
    cases = [
        json.dumps({"data": []}).encode(),
        json.dumps({"data": [None]}).encode(),
        json.dumps({"data": [{"index": "0", "embedding": [0, 1]}]}).encode(),
        json.dumps({"data": [{"index": 0}]}).encode(),
    ]
    for body in cases:
        with _local_error_server(200, body) as url:
            embedder = OpenAICompatibleEmbedder("model", url, provider="ollama")
            with pytest.raises(EmbeddingProviderResponseError):
                embedder.embed(["one"])


def test_embedding_provider_errors_still_match_as_plain_runtime_error():
    """Existing call sites (and tests) use pytest.raises(RuntimeError, match=...);
    the new hierarchy must not break that."""
    assert issubclass(EmbeddingProviderUnavailableError, RuntimeError)
    assert issubclass(EmbeddingProviderResponseError, RuntimeError)
    assert issubclass(EmbeddingProviderError, RuntimeError)
    with pytest.raises(RuntimeError, match="boom"):
        raise EmbeddingProviderUnavailableError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        raise EmbeddingProviderResponseError("boom")


# ---------------------------------------------------------------------------
# MCP parity (section E): provider failures must be as distinguishable as
# authorization denials already are, via the same CallToolResult /
# _meta.error_type pattern.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_rag_search_reports_unavailable_provider_without_leaking_detail(service):
    leaked_detail = "internal-upstream.example quota-key-9f31"
    failing = _failing_service(
        service,
        EmbeddingProviderUnavailableError(f"Embedding endpoint request failed: {leaked_detail}"),
    )
    server = create_mcp(failing)

    result = await server.call_tool("rag_search", {"query": "python"})

    assert result.isError is True
    assert result.meta == {"error_type": "embedding_provider_unavailable"}
    assert result.structuredContent == {"result": []}
    assert leaked_detail not in result.content[0].text


@pytest.mark.asyncio
async def test_mcp_upload_document_reports_invalid_provider_response_without_mutating(service):
    leaked_detail = "acct_9f31 rejected"
    failing = _failing_service(
        service,
        EmbeddingProviderResponseError(f"Embedding endpoint returned HTTP 500: {leaked_detail}"),
    )
    server = create_mcp(failing)

    result = await server.call_tool("upload_document", {"title": "t", "content": "python"})

    assert result.isError is True
    assert result.meta == {"error_type": "embedding_provider_error"}
    assert leaked_detail not in result.content[0].text
    assert failing.list_documents() == []


# ---------------------------------------------------------------------------
# Service-layer audit/observability logging (section D).
# ---------------------------------------------------------------------------


def test_service_logs_document_lifecycle_by_id_never_content(service, caplog):
    leaked_content = "extremely-secret-document-body-9f31"
    with caplog.at_level(logging.DEBUG, logger="utralight_rag"):
        created = service.ingest("My Title", leaked_content, metadata={"source": "test"})
        document_id = created["id"]
        service.update(document_id, "My Title 2", "replacement-body-content")
        service.search("python")
        service.delete_document(document_id)

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert document_id in log_text
    assert "My Title" in log_text
    # Document content and chunk text must never be logged.
    assert leaked_content not in log_text
    assert "replacement-body-content" not in log_text


def test_service_logs_embedding_provider_failure_with_document_context(service, caplog):
    failing = _failing_service(service, EmbeddingProviderResponseError("boom"))
    with caplog.at_level(logging.WARNING, logger="utralight_rag"):
        with pytest.raises(EmbeddingProviderResponseError):
            failing.ingest("Failing Doc", "python")
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Failing Doc" in log_text


def test_service_logs_authorization_denials_with_principal_and_action(service, caplog):
    from dataclasses import replace

    from utralight_rag.auth import AuthorizationError, Authorizer

    protected_settings = replace(service.settings, auth_mode="trusted-proxy")
    authorizer = Authorizer(protected_settings)
    with caplog.at_level(logging.WARNING, logger="utralight_rag"):
        with pytest.raises(AuthorizationError):
            authorizer.authorize(
                {
                    protected_settings.proxy_user_header: "mallory@example.test",
                    protected_settings.proxy_role_header: "reader",
                },
                "write",
            )
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "mallory@example.test" in log_text
    assert "write" in log_text
