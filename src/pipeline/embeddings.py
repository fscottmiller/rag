"""Pluggable embedding providers used by the ingestion and search pipeline."""

from abc import ABC, abstractmethod
from typing import Sequence


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


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model: str = "nomic-embed-text", url: str = "http://localhost:11434") -> None:
        self.model = model
        self.url = url.rstrip("/")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import json
        import urllib.request

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


def create_embedder(provider: str, model: str, ollama_url: str) -> BaseEmbedder:
    normalized = provider.lower().replace("_", "-")
    if normalized in {"sentence-transformers", "sentence-transformer", "transformers"}:
        return SentenceTransformerEmbedder(model)
    if normalized == "ollama":
        return OllamaEmbedder(model, ollama_url)
    raise ValueError(f"Unknown embedding provider: {provider}")
