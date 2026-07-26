"""Pluggable embedding providers used by the ingestion and search pipeline."""

import json
import math
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from typing import Any, Sequence
from urllib.parse import parse_qsl, urlparse, urlsplit, urlunsplit

_PROVIDER_ALIASES = {
    "openai": "openai-compatible",
    "openai-compatible-api": "openai-compatible",
    "sentence-transformer": "sentence-transformers",
    "transformers": "sentence-transformers",
}


def canonical_provider(provider: str) -> str:
    normalized = provider.lower().replace("_", "-")
    return _PROVIDER_ALIASES.get(normalized, normalized)


@dataclass(frozen=True)
class EmbeddingConfiguration:
    provider: str
    model: str
    fingerprint: str


def _endpoint_digest(url: str) -> str:
    parsed = urlsplit(url)
    # Userinfo is credentials, not index identity. Fragments are not sent in HTTP requests.
    endpoint = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.rsplit("@", 1)[-1].lower(),
            parsed.path,
            parsed.query,
            "",
        )
    )
    return sha256(endpoint.encode()).hexdigest()


def _has_query_credentials(query: str) -> bool:
    return any(
        "".join(character for character in name.lower() if character.isalnum()).endswith(
            ("key", "token", "secret", "password", "credential", "signature")
        )
        or "".join(character for character in name.lower() if character.isalnum())
        in {"auth", "authorization", "sig"}
        for name, _ in parse_qsl(query, keep_blank_values=True)
    )


def _configuration(provider: str, model: str, **settings: object) -> EmbeddingConfiguration:
    provider = canonical_provider(provider)
    values = {"provider": provider, "model": model, **settings}
    fingerprint = sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return EmbeddingConfiguration(provider, model, fingerprint)


def _validated_embeddings(vectors: list[object], expected_count: int) -> list[list[float]]:
    if len(vectors) != expected_count or not all(isinstance(vector, list) for vector in vectors):
        raise RuntimeError("Embedding provider returned invalid embeddings")
    try:
        embeddings = [[float(value) for value in vector] for vector in vectors]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Embedding provider returned invalid embeddings") from exc
    if embeddings and (
        not embeddings[0]
        or any(len(vector) != len(embeddings[0]) for vector in embeddings)
        or any(not math.isfinite(value) for vector in embeddings for value in vector)
    ):
        raise RuntimeError("Embedding provider returned invalid embeddings")
    return embeddings


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a sequence of texts."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if not model_name.strip():
            raise ValueError("embedding model must not be empty")
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(list(texts), convert_to_numpy=True).tolist()


class FastEmbedEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        if not model_name.strip():
            raise ValueError("embedding model must not be empty")
        self.model_name = model_name
        self._model = None
        self._model_lock = Lock()

    @property
    def model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = list(texts)
        if not inputs:
            return []
        try:
            vectors = [embedding.tolist() for embedding in self.model.embed(inputs)]
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("FastEmbed returned invalid embeddings") from exc
        try:
            return _validated_embeddings(vectors, len(inputs))
        except RuntimeError as exc:
            raise RuntimeError("FastEmbed returned invalid embeddings") from exc


class OpenAICompatibleEmbedder(BaseEmbedder):
    """Embed text through an OpenAI-compatible /v1/embeddings endpoint."""

    max_response_bytes = 64 * 1024 * 1024

    def __init__(
        self,
        model: str,
        url: str = "https://api.openai.com/v1/embeddings",
        api_key: str = "",
        timeout: float = 60.0,
        dimensions: int | None = None,
        batch_size: int = 64,
        provider: str = "openai-compatible",
    ) -> None:
        if not model.strip():
            raise ValueError("embedding model must not be empty")
        if not url.strip():
            raise ValueError("embedding URL must not be empty")
        url = url.strip()
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("embedding URL must use an http or https scheme")
        if "@" in parsed_url.netloc or _has_query_credentials(parsed_url.query):
            raise ValueError("embedding URL must not contain credentials; use embedding API key")
        if timeout <= 0:
            raise ValueError("embedding timeout must be positive")
        if dimensions is not None and dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        if batch_size < 1:
            raise ValueError("embedding batch size must be positive")
        self.model = model
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.provider = canonical_provider(provider)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = list(texts)
        return [
            vector
            for start in range(0, len(inputs), self.batch_size)
            for vector in self._embed_batch(inputs[start : start + self.batch_size])
        ]

    def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        payload: dict[str, Any] = {
            "model": self.model,
            "input": inputs,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
                if len(raw) > self.max_response_bytes:
                    raise RuntimeError("Embedding endpoint response exceeds size limit")
                body = json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read(self.max_response_bytes + 1).decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Embedding endpoint returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Embedding endpoint request failed: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("Embedding endpoint returned invalid JSON") from exc

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != len(inputs):
            raise RuntimeError("Embedding endpoint returned an unexpected data count")
        if not all(isinstance(item, dict) for item in data):
            raise RuntimeError("Embedding endpoint returned invalid embeddings")
        indexes = [item.get("index") for item in data]
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indexes):
            raise RuntimeError("Embedding endpoint returned invalid embeddings")
        if set(indexes) != set(range(len(inputs))):
            raise RuntimeError("Embedding endpoint returned invalid embeddings")
        data = sorted(data, key=lambda item: item["index"])
        try:
            vectors = [item["embedding"] for item in data]
        except KeyError as exc:
            raise RuntimeError("Embedding endpoint returned invalid embeddings") from exc
        try:
            return _validated_embeddings(vectors, len(inputs))
        except RuntimeError as exc:
            raise RuntimeError("Embedding endpoint returned invalid embeddings") from exc


def create_embedder(
    provider: str,
    model: str,
    embedding_url: str = "https://api.openai.com/v1/embeddings",
    embedding_api_key: str = "",
    embedding_timeout: float = 60.0,
    embedding_dimensions: int | None = None,
    embedding_batch_size: int = 64,
) -> BaseEmbedder:
    normalized = canonical_provider(provider)
    if normalized == "fastembed":
        return FastEmbedEmbedder(model)
    if normalized == "sentence-transformers":
        return SentenceTransformerEmbedder(model)
    if normalized == "ollama":
        return OpenAICompatibleEmbedder(
            model,
            embedding_url,
            embedding_api_key,
            embedding_timeout,
            embedding_dimensions,
            embedding_batch_size,
            normalized,
        )
    if normalized == "openai-compatible":
        if not embedding_api_key.strip():
            raise ValueError("embedding API key must not be empty for external providers")
        return OpenAICompatibleEmbedder(
            model,
            embedding_url,
            embedding_api_key,
            embedding_timeout,
            embedding_dimensions,
            embedding_batch_size,
            normalized,
        )
    raise ValueError(f"Unknown embedding provider: {provider}")


def embedding_configuration(embedder: BaseEmbedder) -> EmbeddingConfiguration | None:
    """Return a persistent index identity for built-in embedders only."""
    if type(embedder) is FastEmbedEmbedder:
        return _configuration("fastembed", embedder.model_name)
    if type(embedder) is SentenceTransformerEmbedder:
        return _configuration("sentence-transformers", embedder.model_name)
    if type(embedder) is OpenAICompatibleEmbedder:
        return _configuration(
            embedder.provider,
            embedder.model,
            dimensions=embedder.dimensions,
            endpoint=_endpoint_digest(embedder.url),
        )
    return None
