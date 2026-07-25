"""Pluggable embedding providers used by the ingestion and search pipeline."""

import json
import math
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Sequence


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a sequence of texts."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
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


class OpenAICompatibleEmbedder(BaseEmbedder):
    """Embed text through an OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        model: str,
        url: str = "https://api.openai.com/v1/embeddings",
        api_key: str = "",
        timeout: float = 60.0,
        dimensions: int | None = None,
        batch_size: int = 64,
    ) -> None:
        if not model.strip():
            raise ValueError("embedding model must not be empty")
        if not url.strip():
            raise ValueError("embedding URL must not be empty")
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
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
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
            embeddings = [[float(value) for value in item["embedding"]] for item in data]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Embedding endpoint returned invalid embeddings") from exc
        if any(not all(math.isfinite(value) for value in vector) for vector in embeddings):
            raise RuntimeError("Embedding endpoint returned invalid embeddings")
        return embeddings


def create_embedder(
    provider: str,
    model: str,
    embedding_url: str = "https://api.openai.com/v1/embeddings",
    embedding_api_key: str = "",
    embedding_timeout: float = 60.0,
    embedding_dimensions: int | None = None,
    embedding_batch_size: int = 64,
) -> BaseEmbedder:
    normalized = provider.lower().replace("_", "-")
    if normalized in {"sentence-transformers", "sentence-transformer", "transformers"}:
        return SentenceTransformerEmbedder(model)
    if normalized in {"ollama", "openai", "openai-compatible", "openai-compatible-api"}:
        return OpenAICompatibleEmbedder(
            model,
            embedding_url,
            embedding_api_key,
            embedding_timeout,
            embedding_dimensions,
            embedding_batch_size,
        )
    raise ValueError(f"Unknown embedding provider: {provider}")
