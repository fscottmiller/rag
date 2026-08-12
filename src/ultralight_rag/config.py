"""Environment-backed configuration for the Ultralight RAG service."""

import os
from dataclasses import dataclass, field
from typing import TypedDict

from .pipeline.embeddings import canonical_provider


class ProviderConfig(TypedDict):
    model: str
    url: str
    uses_openai_key: bool


# Keyed by canonical provider name. `canonical_provider` already folds the
# aliases (`openai`, `openai_compatible`, `openai-compatible-api`, ...) onto
# these keys, so listing them here too would be three copies of one dict that
# could drift apart. Importing from `pipeline.embeddings` is safe in this
# direction: that module imports nothing from this package.
PROVIDER_DEFAULTS: dict[str, ProviderConfig] = {
    "openai-compatible": {
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
    # Listed explicitly rather than relying on "default" below, which used to
    # supply this model by coincidence.
    "sentence-transformers": {
        "model": "all-MiniLM-L6-v2",
        "url": "https://api.openai.com/v1/embeddings",
        "uses_openai_key": False,
    },
    # Fallback for an unrecognized provider. `create_embedder` rejects those
    # outright, so this only has to keep `Settings` constructible long enough
    # to reach that error.
    "default": {
        "model": "all-MiniLM-L6-v2",
        "url": "https://api.openai.com/v1/embeddings",
        "uses_openai_key": False,
    },
}


def provider_defaults(provider: str) -> ProviderConfig:
    """Resolve a provider name -- in any alias or casing -- to its defaults."""
    return PROVIDER_DEFAULTS.get(canonical_provider(provider), PROVIDER_DEFAULTS["default"])


DEFAULT_MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_REQUEST_BYTES = DEFAULT_MAX_DOCUMENT_BYTES + 64 * 1024


@dataclass(frozen=True)
class Settings:
    database_path: str = ":memory:"
    embedding_provider: str = "fastembed"
    # Empty means "use this provider's default", resolved in __post_init__.
    # An empty model or URL was never valid -- every embedder rejects one --
    # so it can carry that meaning without a None sentinel widening the
    # declared type of a field that is always a str by the time anyone reads it.
    embedding_model: str = ""
    embedding_url: str = ""
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
    trusted_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")

    def __post_init__(self) -> None:
        # Resolve provider defaults here rather than in from_env() alone, so a
        # programmatically constructed Settings agrees with an env-derived one.
        # Previously the field defaults were hardcoded to FastEmbed's model and
        # OpenAI's URL, so Settings(embedding_provider="ollama") silently built
        # an OpenAI-compatible embedder pointed at OpenAI with a FastEmbed model
        # name. An explicitly passed model or URL still wins; only an unset one
        # is filled in.
        defaults = provider_defaults(self.embedding_provider)
        if not self.embedding_model:
            object.__setattr__(self, "embedding_model", defaults["model"])
        if not self.embedding_url:
            object.__setattr__(self, "embedding_url", defaults["url"])
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
            # No provider configured: an API key from either variable selects
            # the OpenAI-compatible provider, otherwise fall back to local.
            api_key = configured_api_key or os.getenv("OPENAI_API_KEY", "").strip()
            provider = "openai-compatible" if api_key else "fastembed"
        else:
            # OPENAI_API_KEY is only a fallback for providers that speak
            # OpenAI's auth; Ollama in particular must never pick it up.
            api_key = configured_api_key or (
                os.getenv("OPENAI_API_KEY", "").strip()
                if provider_defaults(provider)["uses_openai_key"]
                else ""
            )

        # Model and URL are left empty when unset so __post_init__ resolves
        # them from the provider, keeping one source of truth for both
        # construction paths.
        return cls(
            database_path=os.getenv("RAG_DATABASE_PATH", ":memory:"),
            embedding_provider=provider,
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", ""),
            embedding_url=os.getenv("RAG_EMBEDDING_URL", ""),
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
                for host in os.getenv("RAG_TRUSTED_HOSTS", "localhost,127.0.0.1").split(",")
                if host.strip()
            ),
        )
