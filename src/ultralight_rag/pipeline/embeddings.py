"""Pluggable embedding providers used by the ingestion and search pipeline."""

import json
import logging
import math
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Real
from threading import Lock
from typing import Any
from urllib.parse import parse_qsl, urlparse, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

_PROVIDER_ALIASES = {
    "openai": "openai-compatible",
    "openai-compatible-api": "openai-compatible",
    "sentence-transformer": "sentence-transformers",
    "transformers": "sentence-transformers",
}


class EmbeddingProviderError(RuntimeError):
    """An embedding provider failed to produce usable embeddings.

    Subclasses ``RuntimeError`` (not a new, unrelated base) so call sites that
    already do ``pytest.raises(RuntimeError, match=...)`` keep passing
    unchanged, while giving the API/MCP adapters a specific type to catch
    instead of bare ``RuntimeError`` -- which would risk mislabeling genuine
    bugs elsewhere in the service as upstream failures. Callers should not
    raise this class directly; raise one of the two subclasses below so the
    adapters can map the failure to an accurate status code / error type.
    """


class EmbeddingProviderUnavailableError(EmbeddingProviderError):
    """The provider could not be reached at all: connection refused, DNS
    failure, or a timeout. The request never got a response, so retrying
    later (once the provider is back) is likely to succeed. Maps to HTTP 503
    in the REST adapter."""


class EmbeddingProviderResponseError(EmbeddingProviderError):
    """The provider was reached but returned something unusable: a non-2xx
    HTTP status, malformed JSON, the wrong shape, non-finite values, or a
    local model producing garbage output. The provider itself is misbehaving,
    not the network path to it. Maps to HTTP 502 in the REST adapter."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow HTTP redirects.

    ``urllib.request.urlopen`` follows 3xx redirects by default, resending the
    ``Authorization`` header (and any other request headers) to whatever host the
    redirect points to, without re-validating the URL scheme. That would defeat the
    https-only enforcement on external embedding endpoints and let a compromised or
    malicious provider exfiltrate the API key, or feed back fabricated vectors.
    Returning ``None`` here tells urllib no handler will perform the redirect, which
    surfaces it as a normal ``urllib.error.HTTPError`` for the original 3xx status.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# Built once at import time via build_opener() (never urllib.request.install_opener(),
# which would mutate global process-wide state) so every embedding request goes through
# an opener that never follows redirects.
_opener = urllib.request.build_opener(_NoRedirectHandler)


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
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for vector in vectors
        for value in vector
    ):
        raise RuntimeError("Embedding provider returned invalid embeddings")
    embeddings = [[float(value) for value in vector] for vector in vectors]
    if embeddings and (
        not embeddings[0] or any(len(vector) != len(embeddings[0]) for vector in embeddings)
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
        inputs = list(texts)
        try:
            return _validated_embeddings(
                self.model.encode(inputs, convert_to_numpy=True).tolist(), len(inputs)
            )
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            # exc is raised by the local model library against `texts` -- document
            # content or the search query, both user-supplied -- so %r escapes it the
            # same way provider response detail is escaped below, instead of letting
            # an embedded newline forge a log line.
            logger.error(
                "SentenceTransformers embedder (model=%s) returned invalid embeddings: %r",
                self.model_name,
                exc,
            )
            raise EmbeddingProviderResponseError(
                "SentenceTransformers returned invalid embeddings"
            ) from exc


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
            # exc is raised by the local model library against `texts` -- document
            # content or the search query, both user-supplied -- so %r escapes it the
            # same way provider response detail is escaped below, instead of letting
            # an embedded newline forge a log line.
            logger.error(
                "FastEmbed embedder (model=%s) returned invalid embeddings: %r",
                self.model_name,
                exc,
            )
            raise EmbeddingProviderResponseError("FastEmbed returned invalid embeddings") from exc
        try:
            return _validated_embeddings(vectors, len(inputs))
        except RuntimeError as exc:
            logger.error(
                "FastEmbed embedder (model=%s) returned invalid embeddings: %r",
                self.model_name,
                exc,
            )
            raise EmbeddingProviderResponseError("FastEmbed returned invalid embeddings") from exc


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
        self.provider = canonical_provider(provider)
        if self.provider == "openai-compatible" and parsed_url.scheme != "https":
            raise ValueError("external embedding URL must use https")
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
        # S310 flags this as a possible arbitrary-scheme URL open, but self.url is
        # already constrained to http/https (and to https with no embedded
        # credentials for external providers) by the scheme/credential checks in
        # __init__, and the request is dispatched through _opener, which never
        # follows redirects - so a malicious 3xx response can't retarget it to an
        # unvalidated scheme or host after the fact.
        request = urllib.request.Request(  # noqa: S310
            self.url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        body = self._execute_request(request)
        return self._validate_response(body, len(inputs))

    def _execute_request(self, request: urllib.request.Request) -> Any:
        try:
            with _opener.open(request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
                if len(raw) > self.max_response_bytes:
                    logger.error("Embedding endpoint %s response exceeds size limit", self.url)
                    raise EmbeddingProviderResponseError(
                        "Embedding endpoint response exceeds size limit"
                    )
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            # Logged here, in full, for operators. The exception message below also
            # carries this detail (existing tests assert on it via
            # pytest.raises(RuntimeError, match=...)), but callers in api/main.py and
            # mcp_server/server.py must never forward str(exc) to a client: the
            # provider's raw response body can contain upstream hostnames, quota
            # details, or account identifiers.
            detail = exc.read(self.max_response_bytes + 1).decode("utf-8", errors="replace")[:500]
            logger.error("Embedding endpoint %s returned HTTP %s: %r", self.url, exc.code, detail)
            raise EmbeddingProviderResponseError(
                f"Embedding endpoint returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.warning("Embedding endpoint %s request failed: %r", self.url, exc)
            raise EmbeddingProviderUnavailableError(
                f"Embedding endpoint request failed: {exc}"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("Embedding endpoint %s returned invalid JSON: %r", self.url, exc)
            raise EmbeddingProviderResponseError(
                "Embedding endpoint returned invalid JSON"
            ) from exc

    def _validate_response(self, body: Any, inputs_count: int) -> list[list[float]]:
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != inputs_count:
            logger.error(
                "Embedding endpoint %s returned an unexpected data count (expected %d)",
                self.url,
                inputs_count,
            )
            raise EmbeddingProviderResponseError(
                "Embedding endpoint returned an unexpected data count"
            )
        if not all(isinstance(item, dict) for item in data):
            logger.error("Embedding endpoint %s returned non-dict embedding items", self.url)
            raise EmbeddingProviderResponseError("Embedding endpoint returned invalid embeddings")
        indexes = [item.get("index") for item in data]
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indexes):
            logger.error("Embedding endpoint %s returned non-integer embedding indexes", self.url)
            raise EmbeddingProviderResponseError("Embedding endpoint returned invalid embeddings")
        if set(indexes) != set(range(inputs_count)):
            logger.error("Embedding endpoint %s returned mismatched embedding indexes", self.url)
            raise EmbeddingProviderResponseError("Embedding endpoint returned invalid embeddings")
        data = sorted(data, key=lambda item: item["index"])
        try:
            vectors = [item["embedding"] for item in data]
        except KeyError as exc:
            logger.error("Embedding endpoint %s response item missing 'embedding' key", self.url)
            raise EmbeddingProviderResponseError(
                "Embedding endpoint returned invalid embeddings"
            ) from exc
        try:
            return _validated_embeddings(vectors, inputs_count)
        except RuntimeError as exc:
            logger.error("Embedding endpoint %s returned invalid embeddings: %r", self.url, exc)
            raise EmbeddingProviderResponseError(
                "Embedding endpoint returned invalid embeddings"
            ) from exc


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
