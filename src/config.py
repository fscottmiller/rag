"""Environment-backed configuration for the transient RAG service."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    database_path: str = ":memory:"
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"
    chunker: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_path=os.getenv("RAG_DATABASE_PATH", ":memory:"),
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "sentence-transformers"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            ollama_url=os.getenv("RAG_OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("RAG_OLLAMA_MODEL", "nomic-embed-text"),
            chunker=os.getenv("RAG_CHUNKER", "recursive"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "64")),
        )
