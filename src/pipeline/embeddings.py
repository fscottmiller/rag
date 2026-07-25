"""Pluggable embedding providers used by the ingestion and search pipeline."""

import json
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
    ) -> None:
        if not model.strip():
            raise ValueError("embedding model must not be empty")
        if not url.strip():
            raise ValueError("embedding URL must not be empty")
        if timeout <= 0:
            raise ValueError("embedding timeout must be positive")
        if dimensions is not None and dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        self.model = model
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = list(texts)
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
        if data and all(isinstance(item, dict) and "index" in item for item in data):
            data = sorted(data, key=lambda item: item["index"])
        try:
            return [[float(value) for value in item["embedding"]] for item in data]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Embedding endpoint returned invalid embeddings") from exc


class OllamaEmbedder(BaseEmbedder):
    def __init__(
        self, model: str = "nomic-embed-text", url: str = "http://localhost:11434"
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        result = []
        for text in texts:
            request = urllib.request.Request(
                f"{self.url}/api/embeddings",
                data=json.dumps({"model": self.model, "prompt": text}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                result.append(json.loads(response.read())["embedding"])
        return result


def create_embedder(
    provider: str,
    model: str,
    ollama_url: str,
    embedding_url: str = "https://api.openai.com/v1/embeddings",
    embedding_api_key: str = "",
    embedding_timeout: float = 60.0,
    embedding_dimensions: int | None = None,
) -> BaseEmbedder:
    normalized = provider.lower().replace("_", "-")
    if normalized in {"sentence-transformers", "sentence-transformer", "transformers"}:
        return SentenceTransformerEmbedder(model)
    if normalized == "ollama":
        return OllamaEmbedder(model, ollama_url)
    if normalized in {"openai", "openai-compatible", "openai-compatible-api"}:
        return OpenAICompatibleEmbedder(
            model,
            embedding_url,
            embedding_api_key,
            embedding_timeout,
            embedding_dimensions,
        )
    raise ValueError(f"Unknown embedding provider: {provider}")
