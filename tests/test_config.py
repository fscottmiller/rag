import pytest

from utralight_rag.config import Settings


def test_settings_defaults_are_in_memory_and_fastembed():
    settings = Settings()
    assert settings.database_path == ":memory:"
    assert settings.embedding_provider == "fastembed"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.chunker == "recursive"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.max_document_bytes == 10 * 1024 * 1024
    assert settings.max_request_bytes == 10 * 1024 * 1024 + 64 * 1024
    assert settings.embedding_batch_size == 64
    assert settings.trusted_hosts == ("localhost", "127.0.0.1", "testserver")


def test_settings_read_all_environment_values(monkeypatch):
    values = {
        "RAG_DATABASE_PATH": "index.db",
        "RAG_EMBEDDING_PROVIDER": "openai-compatible",
        "RAG_EMBEDDING_MODEL": "custom-model",
        "RAG_EMBEDDING_URL": "https://provider.example/v1/embeddings",
        "RAG_EMBEDDING_API_KEY": "test-key",
        "RAG_EMBEDDING_TIMEOUT": "12.5",
        "RAG_EMBEDDING_DIMENSIONS": "768",
        "RAG_CHUNKER": "sentence",
        "RAG_CHUNK_SIZE": "128",
        "RAG_CHUNK_OVERLAP": "16",
        "RAG_MAX_DOCUMENT_BYTES": "2048",
        "RAG_MAX_REQUEST_BYTES": "4096",
        "RAG_EMBEDDING_BATCH_SIZE": "8",
        "RAG_TRUSTED_HOSTS": "app.example, localhost",
        "RAG_AUTH_MODE": "trusted-proxy",
        "RAG_PROXY_USER_HEADER": "X-User",
        "RAG_PROXY_ROLE_HEADER": "X-Role",
        "RAG_PROXY_ADMIN_ROLE": "owner",
        "RAG_PROXY_READER_ROLE": "viewer",
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
        chunker="sentence",
        chunk_size=128,
        chunk_overlap=16,
        max_document_bytes=2048,
        max_request_bytes=4096,
        embedding_batch_size=8,
        trusted_hosts=("app.example", "localhost"),
        auth_mode="trusted-proxy",
        proxy_user_header="X-User",
        proxy_role_header="X-Role",
        proxy_admin_role="owner",
        proxy_reader_role="viewer",
    )


def test_ollama_uses_common_openai_compatible_settings(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings.from_env()
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.embedding_url == "http://localhost:11434/v1/embeddings"
    assert settings.embedding_api_key == ""


def test_openai_compatible_settings_have_safe_defaults_and_key_fallback(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_API_KEY", " fallback-key ")
    settings = Settings.from_env()
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_url == "https://api.openai.com/v1/embeddings"
    assert settings.embedding_api_key == "fallback-key"
    assert "fallback-key" not in repr(settings)


def test_unset_embedding_provider_uses_available_api_key(monkeypatch):
    monkeypatch.delenv("RAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    settings = Settings.from_env()
    assert settings.embedding_provider == "openai-compatible"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_api_key == "fallback-key"


def test_whitespace_embedding_key_falls_back_to_openai_key(monkeypatch):
    monkeypatch.delenv("RAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_API_KEY", "  \t")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    settings = Settings.from_env()
    assert settings.embedding_provider == "openai-compatible"
    assert settings.embedding_api_key == "fallback-key"

    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "openai-compatible")
    assert Settings.from_env().embedding_api_key == "fallback-key"


def test_unset_embedding_provider_without_api_key_uses_fastembed(monkeypatch):
    monkeypatch.delenv("RAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("RAG_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings.from_env()
    assert settings.embedding_provider == "fastembed"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"


def test_settings_reject_invalid_limits_and_hosts():
    with pytest.raises(ValueError, match="max_document_bytes"):
        Settings(max_document_bytes=0)
    with pytest.raises(ValueError, match="max_request_bytes"):
        Settings(max_request_bytes=0)
    with pytest.raises(ValueError, match="smaller"):
        Settings(max_document_bytes=10, max_request_bytes=9)
    with pytest.raises(ValueError, match="embedding_batch_size"):
        Settings(embedding_batch_size=0)
    with pytest.raises(ValueError, match="trusted_hosts"):
        Settings(trusted_hosts=())


def test_settings_reject_non_integer_chunk_configuration(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "not-an-integer")
    with pytest.raises(ValueError):
        Settings.from_env()
