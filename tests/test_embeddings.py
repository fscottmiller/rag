import json
import sys
import threading
import time
import urllib.error
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from utralight_rag.pipeline import embeddings as embeddings_module
from utralight_rag.pipeline.embeddings import (
    BaseEmbedder,
    FastEmbedEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)


class OneVectorEmbedder(BaseEmbedder):
    def embed(self, texts):
        return [[float(len(text))] for text in texts]


def test_base_embedder_embeds_one_text():
    assert OneVectorEmbedder().embed_one("hello") == [5.0]


def test_embedder_factory_supports_provider_aliases():
    assert isinstance(create_embedder("fastembed", "model"), FastEmbedEmbedder)
    assert isinstance(
        create_embedder("sentence_transformers", "model", "url"), SentenceTransformerEmbedder
    )
    ollama_embedder = create_embedder("ollama", "model", "http://ollama/v1/embeddings")
    assert isinstance(ollama_embedder, OpenAICompatibleEmbedder)
    assert ollama_embedder.url == "http://ollama/v1/embeddings"
    openai_embedder = create_embedder(
        "openai_compatible", "model", "https://embedding.test/v1/embeddings", "secret", 5, 768
    )
    assert isinstance(openai_embedder, OpenAICompatibleEmbedder)
    assert openai_embedder.url == "https://embedding.test/v1/embeddings"
    assert openai_embedder.dimensions == 768
    with pytest.raises(ValueError, match="API key"):
        create_embedder("openai-compatible", "model")
    with pytest.raises(ValueError, match="https"):
        create_embedder("openai-compatible", "model", "http://embedding.test", "secret")
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedder("unknown", "model", "url")


def test_fastembed_is_lazy_and_converts_vectors(monkeypatch):
    constructed = []

    class Vector:
        def tolist(self):
            return [1.0, 2.0]

    class TextEmbedding:
        def __init__(self, model_name):
            constructed.append(model_name)

        def embed(self, texts):
            return (Vector() for _ in texts)

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=TextEmbedding))
    embedder = create_embedder("fastembed", "test-model")
    assert constructed == []
    assert embedder.embed([]) == []
    assert constructed == []
    assert embedder.embed(["one", "two"]) == [[1.0, 2.0], [1.0, 2.0]]
    assert constructed == ["test-model"]


@pytest.mark.parametrize(
    "vectors",
    [[], [[]], [[1.0], [1.0, 2.0]], [["1.0"]], [[float("nan")]], [[True]]],
)
def test_fastembed_rejects_malformed_vectors(monkeypatch, vectors):
    class Vector:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return self.value

    class TextEmbedding:
        def __init__(self, model_name):
            pass

        def embed(self, texts):
            return (Vector(vector) for vector in vectors)

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=TextEmbedding))
    with pytest.raises(RuntimeError, match="FastEmbed returned invalid embeddings"):
        FastEmbedEmbedder("test-model").embed(["one"])


def test_fastembed_rejects_vectors_without_tolist(monkeypatch):
    class TextEmbedding:
        def __init__(self, model_name):
            pass

        def embed(self, texts):
            return [object()]

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=TextEmbedding))
    with pytest.raises(RuntimeError, match="FastEmbed returned invalid embeddings"):
        FastEmbedEmbedder("test-model").embed(["one"])


def test_fastembed_constructs_its_model_once_under_concurrent_first_use(monkeypatch):
    constructed = []
    start = threading.Barrier(3)

    class Vector:
        def tolist(self):
            return [1.0]

    class TextEmbedding:
        def __init__(self, model_name):
            constructed.append(model_name)
            time.sleep(0.01)

        def embed(self, texts):
            return (Vector() for _ in texts)

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=TextEmbedding))
    embedder = FastEmbedEmbedder("test-model")

    def embed():
        start.wait()
        assert embedder.embed(["text"]) == [[1.0]]

    threads = [threading.Thread(target=embed) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert constructed == ["test-model"]


def test_ollama_uses_openai_compatible_embeddings_endpoint():
    embedder = create_embedder(
        "ollama",
        "nomic-test",
        "http://ollama:11434/v1/embeddings",
        "",
        12,
        None,
    )
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.url == "http://ollama:11434/v1/embeddings"
    assert embedder.api_key == ""
    assert embedder.timeout == 12


def test_openai_compatible_embedder_sends_batch_and_preserves_indexes():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            requests.append((self.path, self.headers.get("Authorization"), payload))
            if self.path == "/http-error":
                self.send_response(503)
                self.end_headers()
                return
            elif self.path == "/too-large":
                body = b"xxxx"
            elif self.path == "/invalid-json":
                body = b"{not-json"
            elif self.path == "/bad":
                body = json.dumps({"data": []}).encode()
            elif self.path == "/invalid-embedding":
                body = json.dumps({"data": [{"index": 0, "embedding": "invalid"}]}).encode()
            elif self.path == "/duplicate-index":
                body = json.dumps(
                    {
                        "data": [
                            {"index": 0, "embedding": [0, 1]},
                            {"index": 0, "embedding": [2, 3]},
                        ]
                    }
                ).encode()
            elif self.path == "/missing-index":
                body = json.dumps(
                    {
                        "data": [
                            {"index": 0, "embedding": [0, 1]},
                            {"index": 2, "embedding": [2, 3]},
                        ]
                    }
                ).encode()
            elif self.path == "/nonfinite":
                body = json.dumps({"data": [{"index": 0, "embedding": ["NaN"]}]}).encode()
            elif self.path == "/non-dict":
                body = json.dumps({"data": [None]}).encode()
            elif self.path == "/bad-index":
                body = json.dumps({"data": [{"index": "0", "embedding": [0, 1]}]}).encode()
            elif self.path == "/batch":
                body = json.dumps(
                    {
                        "data": [
                            {"index": index, "embedding": [float(index), 1.0]}
                            for index in range(len(payload["input"]))
                        ]
                    }
                ).encode()
            else:
                body = json.dumps(
                    {
                        "data": [
                            {"index": 1, "embedding": [2, 3]},
                            {"index": 0, "embedding": [0, 1]},
                        ]
                    }
                ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        embedder = OpenAICompatibleEmbedder(
            "custom-embedding-model",
            f"{base_url}/v1/embeddings",
            "test-key",
            timeout=5,
            dimensions=2,
            provider="ollama",
        )
        assert embedder.embed(["first", "second"]) == [[0.0, 1.0], [2.0, 3.0]]
        first_request = requests[0]
        assert embedder.embed([]) == []
        requests.clear()
        batched = OpenAICompatibleEmbedder(
            "model", f"{base_url}/batch", batch_size=1, provider="ollama"
        )
        assert batched.embed(["first", "second"]) == [[0.0, 1.0], [0.0, 1.0]]
        assert len(requests) == 2
        with pytest.raises(RuntimeError, match="unexpected data count"):
            OpenAICompatibleEmbedder("model", f"{base_url}/bad", provider="ollama").embed(["one"])
        with pytest.raises(RuntimeError, match="HTTP 503"):
            OpenAICompatibleEmbedder("model", f"{base_url}/http-error", provider="ollama").embed(
                ["one"]
            )
        too_large = OpenAICompatibleEmbedder("model", f"{base_url}/too-large", provider="ollama")
        too_large.max_response_bytes = 3
        with pytest.raises(RuntimeError, match="exceeds size limit"):
            too_large.embed(["one"])
        with pytest.raises(RuntimeError, match="invalid JSON"):
            OpenAICompatibleEmbedder("model", f"{base_url}/invalid-json", provider="ollama").embed(
                ["one"]
            )
        with pytest.raises(RuntimeError, match="invalid embeddings"):
            OpenAICompatibleEmbedder(
                "model", f"{base_url}/invalid-embedding", provider="ollama"
            ).embed(["one"])
        for path in ("duplicate-index", "missing-index", "nonfinite", "non-dict", "bad-index"):
            with pytest.raises(RuntimeError, match="invalid embeddings"):
                OpenAICompatibleEmbedder("model", f"{base_url}/{path}", provider="ollama").embed(
                    ["one", "two"] if path in ("duplicate-index", "missing-index") else ["one"]
                )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert first_request == (
        "/v1/embeddings",
        "Bearer test-key",
        {
            "model": "custom-embedding-model",
            "input": ["first", "second"],
            "encoding_format": "float",
            "dimensions": 2,
        },
    )


def test_openai_compatible_embedder_validates_configuration():
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleEmbedder("")
    with pytest.raises(ValueError, match="URL"):
        OpenAICompatibleEmbedder("model", "")
    with pytest.raises(ValueError, match="batch size"):
        OpenAICompatibleEmbedder("model", batch_size=0)

    with pytest.raises(ValueError, match="timeout"):
        OpenAICompatibleEmbedder("model", timeout=0)
    with pytest.raises(ValueError, match="dimensions"):
        OpenAICompatibleEmbedder("model", dimensions=0)


def test_openai_compatible_embedder_rejects_non_http_urls():
    with pytest.raises(ValueError, match="http or https"):
        OpenAICompatibleEmbedder("model", "file:///tmp/embeddings")
    with pytest.raises(ValueError, match="http or https"):
        OpenAICompatibleEmbedder("model", "ftp://embedding.example/v1/embeddings")


def test_openai_compatible_embedder_rejects_http_endpoint():
    with pytest.raises(ValueError, match="https"):
        OpenAICompatibleEmbedder("model", "http://embedding.example/v1/embeddings", "secret")


def test_openai_compatible_embedder_limits_http_error_body(monkeypatch):
    class ErrorBody:
        def read(self, size):
            assert size == 4
            return b"error"

        def close(self):
            pass

    calls = []

    def fake_open(request, timeout=None):
        calls.append((request, timeout))
        raise urllib.error.HTTPError("http://embedding.test", 503, "Unavailable", {}, ErrorBody())

    # The embedder issues requests through the module's no-redirect opener (see F1
    # fix), not urllib.request.urlopen directly, so that is what must be patched for
    # this test to actually exercise the code path it claims to.
    monkeypatch.setattr(embeddings_module._opener, "open", fake_open)
    embedder = OpenAICompatibleEmbedder("model", "http://embedding.test", provider="ollama")
    embedder.max_response_bytes = 3
    with pytest.raises(RuntimeError, match="HTTP 503: error"):
        embedder.embed(["one"])
    assert len(calls) == 1


@contextmanager
def _local_json_server(body: bytes, status: int = 200):
    """Serve a fixed raw response body to any POST, on 127.0.0.1 with an OS-assigned port."""

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


def test_openai_compatible_embedder_does_not_follow_redirects_with_credentials():
    """Regression test for F1: a redirect must never carry the Authorization header
    to another host, and a fabricated response from the redirect target must never
    be accepted as a real embedding."""
    secondary_requests = []

    class SecondaryHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            secondary_requests.append((self.path, self.headers.get("Authorization")))
            body = json.dumps({"data": [{"index": 0, "embedding": [9.0, 9.0]}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    secondary = ThreadingHTTPServer(("127.0.0.1", 0), SecondaryHandler)
    secondary_thread = threading.Thread(target=secondary.serve_forever, daemon=True)
    secondary_thread.start()
    secondary_url = f"http://127.0.0.1:{secondary.server_port}/steal"

    primary_requests = []

    class PrimaryHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            primary_requests.append((self.path, self.headers.get("Authorization")))
            self.send_response(302)
            self.send_header("Location", secondary_url)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            return

    primary = ThreadingHTTPServer(("127.0.0.1", 0), PrimaryHandler)
    primary_thread = threading.Thread(target=primary.serve_forever, daemon=True)
    primary_thread.start()

    try:
        embedder = OpenAICompatibleEmbedder(
            "model",
            f"http://127.0.0.1:{primary.server_port}/v1/embeddings",
            "SUPER-SECRET-KEY",
            provider="ollama",
        )
        with pytest.raises(RuntimeError, match="HTTP 302"):
            embedder.embed(["one"])
    finally:
        primary.shutdown()
        primary_thread.join()
        primary.server_close()
        secondary.shutdown()
        secondary_thread.join()
        secondary.server_close()

    assert primary_requests == [("/v1/embeddings", "Bearer SUPER-SECRET-KEY")]
    # The redirect target must never be contacted at all, let alone receive the key.
    assert secondary_requests == []


def test_openai_compatible_embedder_wraps_timeout_and_connection_errors(monkeypatch):
    def raise_timeout(*_args, **_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(embeddings_module._opener, "open", raise_timeout)
    embedder = OpenAICompatibleEmbedder("model", "http://embedding.test", provider="ollama")
    with pytest.raises(RuntimeError, match=r"Embedding endpoint request failed: timed out"):
        embedder.embed(["one"])

    def raise_url_error(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(embeddings_module._opener, "open", raise_url_error)
    with pytest.raises(RuntimeError, match=r"Embedding endpoint request failed"):
        embedder.embed(["one"])


def test_openai_compatible_embedder_rejects_response_item_missing_embedding_key():
    body = json.dumps({"data": [{"index": 0}]}).encode()
    with _local_json_server(body) as url:
        embedder = OpenAICompatibleEmbedder("model", url, provider="ollama")
        with pytest.raises(RuntimeError, match="invalid embeddings"):
            embedder.embed(["one"])


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_openai_compatible_embedder_rejects_bare_json_non_finite_tokens(token):
    # json.loads accepts bare (unquoted) NaN/Infinity/-Infinity tokens by default and
    # turns them into real non-finite floats, unlike the "NaN"-as-string case above.
    body = ('{"data": [{"index": 0, "embedding": [%s, 1.0]}]}' % token).encode()
    with _local_json_server(body) as url:
        embedder = OpenAICompatibleEmbedder("model", url, provider="ollama")
        with pytest.raises(RuntimeError, match="invalid embeddings"):
            embedder.embed(["one"])
