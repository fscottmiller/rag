import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lightweight_rag.pipeline.embeddings import (
    BaseEmbedder,
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
    assert isinstance(
        create_embedder("sentence_transformers", "model", "url"), SentenceTransformerEmbedder
    )
    ollama_embedder = create_embedder("ollama", "model", "http://ollama/v1/embeddings")
    assert isinstance(ollama_embedder, OpenAICompatibleEmbedder)
    assert ollama_embedder.url == "http://ollama/v1/embeddings"
    openai_embedder = create_embedder(
        "openai_compatible", "model", "http://embedding.test/v1/embeddings", "secret", 5, 768
    )
    assert isinstance(openai_embedder, OpenAICompatibleEmbedder)
    assert openai_embedder.url == "http://embedding.test/v1/embeddings"
    assert openai_embedder.dimensions == 768
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedder("unknown", "model", "url")


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
            if self.path == "/invalid-json":
                body = b"{not-json"
            elif self.path == "/bad":
                body = json.dumps({"data": []}).encode()
            elif self.path == "/invalid-embedding":
                body = json.dumps({"data": [{"index": 0, "embedding": "invalid"}]}).encode()
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
        )
        assert embedder.embed(["first", "second"]) == [[0.0, 1.0], [2.0, 3.0]]
        assert embedder.embed([]) == []
        with pytest.raises(RuntimeError, match="unexpected data count"):
            OpenAICompatibleEmbedder("model", f"{base_url}/bad").embed(["one"])
        with pytest.raises(RuntimeError, match="HTTP 503"):
            OpenAICompatibleEmbedder("model", f"{base_url}/http-error").embed(["one"])
        with pytest.raises(RuntimeError, match="invalid JSON"):
            OpenAICompatibleEmbedder("model", f"{base_url}/invalid-json").embed(["one"])
        with pytest.raises(RuntimeError, match="invalid embeddings"):
            OpenAICompatibleEmbedder("model", f"{base_url}/invalid-embedding").embed(["one"])
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert requests[0] == (
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
    with pytest.raises(ValueError, match="timeout"):
        OpenAICompatibleEmbedder("model", timeout=0)
    with pytest.raises(ValueError, match="dimensions"):
        OpenAICompatibleEmbedder("model", dimensions=0)
