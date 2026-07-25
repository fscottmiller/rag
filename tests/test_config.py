import pytest

from src.config import Settings


def test_settings_defaults_are_transient_and_local():
    settings = Settings()
    assert settings.database_path == ":memory:"
    assert settings.embedding_provider == "sentence-transformers"
    assert settings.chunker == "recursive"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64


def test_settings_read_all_environment_values(monkeypatch):
    values = {
        "RAG_DATABASE_PATH": "index.db",
        "RAG_EMBEDDING_PROVIDER": "openai-compatible",
        "RAG_EMBEDDING_MODEL": "custom-model",
        "RAG_EMBEDDING_URL": "https://provider.example/v1/embeddings",
        "RAG_EMBEDDING_API_KEY": "test-key",
        "RAG_EMBEDDING_TIMEOUT": "12.5",
        "RAG_EMBEDDING_DIMENSIONS": "768",
        "RAG_OLLAMA_URL": "http://ollama:11434",
        "RAG_OLLAMA_MODEL": "custom-embed",
        "RAG_CHUNKER": "sentence",
        "RAG_CHUNK_SIZE": "128",
        "RAG_CHUNK_OVERLAP": "16",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    assert Settings.from_env() == Settings(
        database_path="index.db",
        embedding_provider="openai-compatible",
        embedding_model="custom-model",
        embedding_url="https://provider.example/v1/embeddings",
        embedding_api_key="test-key",
        embedding_timeout=12.5,
        embedding_dimensions=768,
        ollama_url="http://ollama:11434",
        ollama_model="custom-embed",
        chunker="sentence",
        chunk_size=128,
        chunk_overlap=16,
    )


def test_openai_compatible_settings_have_safe_defaults_and_key_fallback(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    settings = Settings.from_env()
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_url == "https://api.openai.com/v1/embeddings"
    assert settings.embedding_api_key == "fallback-key"
    assert "fallback-key" not in repr(settings)


def test_settings_reject_non_integer_chunk_configuration(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "not-an-integer")
    with pytest.raises(ValueError):
        Settings.from_env()
