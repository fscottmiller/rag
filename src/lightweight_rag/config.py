"""Environment-backed configuration for the lightweight RAG service."""

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
    auth_mode: str = "none"
    proxy_user_header: str = "Cf-Access-Authenticated-User-Email"
    proxy_role_header: str = "X-Auth-Request-Role"
    proxy_admin_role: str = "admin"
    proxy_reader_role: str = "reader"
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
            else "nomic-embed-text"
            if normalized_provider == "ollama"
            else "all-MiniLM-L6-v2"
        )
        default_url = (
            "http://localhost:11434/v1/embeddings"
            if normalized_provider == "ollama"
            else "https://api.openai.com/v1/embeddings"
        )
        return cls(
            database_path=os.getenv("RAG_DATABASE_PATH", ":memory:"),
            embedding_provider=provider,
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", default_model),
            embedding_url=os.getenv("RAG_EMBEDDING_URL", default_url),
            embedding_api_key=os.getenv(
                "RAG_EMBEDDING_API_KEY",
                "" if normalized_provider == "ollama" else os.getenv("OPENAI_API_KEY", ""),
            ),
            embedding_timeout=float(os.getenv("RAG_EMBEDDING_TIMEOUT", "60")),
            embedding_dimensions=(
                int(os.environ["RAG_EMBEDDING_DIMENSIONS"])
                if os.getenv("RAG_EMBEDDING_DIMENSIONS")
                else None
            ),
            auth_mode=os.getenv("RAG_AUTH_MODE", "none"),
            proxy_user_header=os.getenv(
                "RAG_PROXY_USER_HEADER", "Cf-Access-Authenticated-User-Email"
            ),
            proxy_role_header=os.getenv("RAG_PROXY_ROLE_HEADER", "X-Auth-Request-Role"),
            proxy_admin_role=os.getenv("RAG_PROXY_ADMIN_ROLE", "admin"),
            proxy_reader_role=os.getenv("RAG_PROXY_READER_ROLE", "reader"),
            chunker=os.getenv("RAG_CHUNKER", "recursive"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "64")),
        )
