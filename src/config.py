"""Environment-backed configuration for the transient RAG service."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    database_path: str = ":memory:"
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_url: str = "https://api.openai.com/v1/embeddings"
    embedding_api_key: str = field(default="", repr=False)
    embedding_timeout: float = 60.0
    embedding_dimensions: int | None = None
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"
    chunker: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("RAG_EMBEDDING_PROVIDER", "sentence-transformers")
        normalized_provider = provider.lower().replace("_", "-")
        default_model = (
            "text-embedding-3-small"
            if normalized_provider in {"openai", "openai-compatible", "openai-compatible-api"}
            else "all-MiniLM-L6-v2"
        )
        return cls(
            database_path=os.getenv("RAG_DATABASE_PATH", ":memory:"),
            embedding_provider=provider,
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", default_model),
            embedding_url=os.getenv("RAG_EMBEDDING_URL", "https://api.openai.com/v1/embeddings"),
            embedding_api_key=os.getenv("RAG_EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            embedding_timeout=float(os.getenv("RAG_EMBEDDING_TIMEOUT", "60")),
            embedding_dimensions=(
                int(os.environ["RAG_EMBEDDING_DIMENSIONS"])
                if os.getenv("RAG_EMBEDDING_DIMENSIONS")
                else None
            ),
            ollama_url=os.getenv("RAG_OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("RAG_OLLAMA_MODEL", "nomic-embed-text"),
            chunker=os.getenv("RAG_CHUNKER", "recursive"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "64")),
        )
