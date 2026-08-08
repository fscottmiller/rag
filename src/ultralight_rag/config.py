"""Environment-backed configuration for the Ultralight RAG service."""

import os
from dataclasses import dataclass, field
from typing import TypedDict

class ProviderConfig(TypedDict):
    model: str
    url: str
    uses_openai_key: bool

PROVIDER_DEFAULTS: dict[str, ProviderConfig] = {
    "openai": {
        "model": "text-embedding-3-small",
        "url": "https://api.openai.com/v1/embeddings",
        "uses_openai_key": True,
    },
    "openai-compatible": {
        "model": "text-embedding-3-small",
        "url": "https://api.openai.com/v1/embeddings",
        "uses_openai_key": True,
    },
    "openai-compatible-api": {
        "model": "text-embedding-3-small",
        "url": "https://api.openai.com/v1/embeddings",
        "uses_openai_key": True,
    },
    "ollama": {
        "model": "nomic-embed-text",
        "url": "http://localhost:11434/v1/embeddings",
        "uses_openai_key": False,
    },
    "fastembed": {
        "model": "BAAI/bge-small-en-v1.5",
        "url": "https://api.openai.com/v1/embeddings",
        "uses_openai_key": False,
    },
    "default": {
        "model": "all-MiniLM-L6-v2",
        "url": "https://api.openai.com/v1/embeddings",
        "uses_openai_key": False,
    },
}

DEFAULT_MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_REQUEST_BYTES = DEFAULT_MAX_DOCUMENT_BYTES + 64 * 1024


@dataclass(frozen=True)
class Settings:
    database_path: str = ":memory:"
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
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
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    embedding_batch_size: int = 64
    trusted_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "testserver")

    def __post_init__(self) -> None:
        if self.max_document_bytes < 1:
            raise ValueError("max_document_bytes must be positive")
        if self.max_request_bytes < 1:
            raise ValueError("max_request_bytes must be positive")
        if self.max_request_bytes < self.max_document_bytes:
            raise ValueError("max_request_bytes must not be smaller than max_document_bytes")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        if not self.trusted_hosts or any(not host.strip() for host in self.trusted_hosts):
            raise ValueError("trusted_hosts must contain at least one non-empty host")

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("RAG_EMBEDDING_PROVIDER")
        configured_api_key = os.getenv("RAG_EMBEDDING_API_KEY", "").strip()

        if provider is None:
            api_key = configured_api_key or os.getenv("OPENAI_API_KEY", "").strip()
            provider = "openai-compatible" if api_key else "fastembed"
        else:
            normalized_explicit_provider = provider.lower().replace("_", "-")
            provider_config = PROVIDER_DEFAULTS.get(normalized_explicit_provider, PROVIDER_DEFAULTS["default"])
            api_key = configured_api_key or (
                os.getenv("OPENAI_API_KEY", "").strip() if provider_config["uses_openai_key"] else ""
            )

        normalized_provider = provider.lower().replace("_", "-")
        provider_config = PROVIDER_DEFAULTS.get(normalized_provider, PROVIDER_DEFAULTS["default"])
        default_model = provider_config["model"]
        default_url = provider_config["url"]

        return cls(
            database_path=os.getenv("RAG_DATABASE_PATH", ":memory:"),
            embedding_provider=provider,
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", default_model),
            embedding_url=os.getenv("RAG_EMBEDDING_URL", default_url),
            embedding_api_key=api_key,
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
            max_document_bytes=int(
                os.getenv("RAG_MAX_DOCUMENT_BYTES", str(DEFAULT_MAX_DOCUMENT_BYTES))
            ),
            max_request_bytes=int(
                os.getenv("RAG_MAX_REQUEST_BYTES", str(DEFAULT_MAX_REQUEST_BYTES))
            ),
            embedding_batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64")),
            trusted_hosts=tuple(
                host.strip()
                for host in os.getenv("RAG_TRUSTED_HOSTS", "localhost,127.0.0.1,testserver").split(
                    ","
                )
                if host.strip()
            ),
        )
