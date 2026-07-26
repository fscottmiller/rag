import pytest

from utralight_rag.config import Settings
from utralight_rag.pipeline.embeddings import FastEmbedEmbedder, OpenAICompatibleEmbedder
from utralight_rag.service import RAGService
from utralight_rag.storage.sqlite import DocumentNotFoundError, SQLiteStore


def test_document_crud_and_cascade(service):
    created = service.ingest("Guide", "Python intro | SQLite storage", {"team": "search"})
    assert created["chunk_count"] == 2
    assert service.list_documents()[0]["id"] == created["id"]
    assert service.get_document(created["id"])["content"] == "Python intro | SQLite storage"

    updated = service.update(created["id"], "Updated", "FastAPI endpoint", {"team": "api"})
    assert updated["title"] == "Updated"
    assert updated["chunk_count"] == 1
    assert service.search("fastapi")[0]["document_id"] == created["id"]

    service.delete_document(created["id"])
    assert service.list_documents() == []
    assert service.search("fastapi") == []
    with pytest.raises(DocumentNotFoundError):
        service.get_document(created["id"])


def test_vector_search_and_metadata_filter(service):
    python_doc = service.ingest("Python", "Python language", {"topic": "code"})
    service.ingest("SQLite", "SQLite vector storage", {"topic": "data"})

    result = service.search("python", top_k=2)
    assert result[0]["document_id"] == python_doc["id"]
    assert result[0]["score"] == pytest.approx(1.0)
    filtered = service.search("sqlite", filter_metadata={"topic": "code"})
    assert filtered and all(item["metadata"]["topic"] == "code" for item in filtered)
    assert service.search("sqlite", filter_metadata={"topic": "missing"}) == []


def test_service_uses_one_configured_chunking_profile(service):
    configured = RAGService(
        SQLiteStore(),
        service.embedder,
        settings=Settings(chunker="token", chunk_size=2, chunk_overlap=0),
    )
    first = configured.ingest("First", "one two three four")
    second = configured.ingest("Second", "one two three four")
    assert first["chunk_count"] == second["chunk_count"]
    assert first["chunk_count"] >= 2


def test_service_rejects_bad_titles(service):
    created = service.ingest("Guide", "content")
    with pytest.raises(ValueError, match="title"):
        service.ingest("   ", "content")
    with pytest.raises(ValueError, match="title"):
        service.update(created["id"], "   ", "content")


def test_invalid_content_query_and_top_k(service):
    with pytest.raises(ValueError):
        service.ingest("Empty", "   ")
    with pytest.raises(ValueError, match="query"):
        service.search("   ")
    with pytest.raises(ValueError, match="between 1 and 100"):
        service.search("query", top_k=0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        service.search("query", top_k=101)


def test_update_checks_document_existence_before_embedding(service):
    class CountingEmbedder:
        calls = 0

        def embed(self, texts):
            self.calls += 1
            return [[1.0, 0.0, 0.0] for _ in texts]

    embedder = CountingEmbedder()
    service.embedder = embedder
    with pytest.raises(DocumentNotFoundError):
        service.update("missing", "Title", "content")
    assert embedder.calls == 0


def test_ingestion_batches_embedding_requests(service):
    class BatchCountingEmbedder:
        def __init__(self):
            self.batch_sizes = []

        def embed(self, texts):
            self.batch_sizes.append(len(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

    embedder = BatchCountingEmbedder()
    configured = RAGService(
        SQLiteStore(),
        embedder,
        service.chunker,
        Settings(embedding_batch_size=2),
    )
    configured.ingest("Batched", "one|two|three")
    assert embedder.batch_sizes == [2, 1]


def test_service_enforces_document_limit(service):
    configured = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=4)
    )
    with pytest.raises(ValueError, match="document content"):
        configured.ingest("Too large", "12345")


def test_service_requires_reindex_for_legacy_nonempty_index(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    legacy = SQLiteStore(str(database))
    legacy.create_document("Old", "text", {}, ["text"], [[1.0] * 384])
    legacy.close()

    with pytest.raises(ValueError, match="reindex"):
        RAGService(
            SQLiteStore(str(database)),
            FastEmbedEmbedder(),
            object(),
            Settings(database_path=str(database)),
        )


def test_service_requires_reindex_for_nonempty_two_column_metadata(tmp_path):
    database = tmp_path / "legacy-metadata.sqlite3"
    legacy = SQLiteStore(str(database))
    legacy.create_document("Old", "text", {}, ["text"], [[1.0]])
    legacy.connection.executescript(
        """
        DROP TABLE index_metadata;
        CREATE TABLE index_metadata (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            embedding_provider TEXT NOT NULL,
            embedding_model TEXT NOT NULL
        );
        INSERT INTO index_metadata VALUES (1, 'fastembed', 'BAAI/bge-small-en-v1.5');
        """
    )
    legacy.close()

    with pytest.raises(ValueError, match="configuration"):
        RAGService(SQLiteStore(str(database)), FastEmbedEmbedder(), object())


def test_service_rejects_different_embedding_model_for_existing_index(tmp_path):
    database = tmp_path / "index.sqlite3"
    settings = Settings(database_path=str(database), embedding_model="first-model")
    first = RAGService(
        SQLiteStore(str(database)), FastEmbedEmbedder("first-model"), object(), settings
    )
    first.store.create_document("First", "text", {}, ["text"], [[1.0]])
    first.store.close()

    with pytest.raises(ValueError, match="configuration"):
        RAGService(
            SQLiteStore(str(database)),
            FastEmbedEmbedder("second-model"),
            object(),
            Settings(database_path=str(database), embedding_model="second-model"),
        )


def test_service_uses_injected_builtin_identity_not_settings(tmp_path):
    database = tmp_path / "index.sqlite3"
    service = RAGService(
        SQLiteStore(str(database)),
        FastEmbedEmbedder("injected-model"),
        object(),
        Settings(database_path=str(database), embedding_model="settings-model"),
    )
    metadata = service.store.connection.execute(
        "SELECT embedding_provider, embedding_model FROM index_metadata"
    ).fetchone()
    assert tuple(metadata) == ("fastembed", "injected-model")


def test_service_rejects_unknown_embedder_for_persistent_index(tmp_path):
    with pytest.raises(ValueError, match="known identity"):
        RAGService(SQLiteStore(str(tmp_path / "index.sqlite3")), object(), object())


def test_failed_embedder_startup_does_not_write_index_metadata(tmp_path):
    store = SQLiteStore(str(tmp_path / "index.sqlite3"))
    with pytest.raises(ValueError, match="API key"):
        RAGService(store, settings=Settings(database_path="ignored", embedding_provider="openai"))
    assert store.connection.execute("SELECT * FROM index_metadata").fetchone() is None


def test_empty_index_can_replace_embedding_identity(tmp_path):
    database = tmp_path / "index.sqlite3"
    first = RAGService(SQLiteStore(str(database)), FastEmbedEmbedder("first"), object())
    first.store.close()
    second = RAGService(SQLiteStore(str(database)), FastEmbedEmbedder("second"), object())
    model = second.store.connection.execute(
        "SELECT embedding_model FROM index_metadata"
    ).fetchone()[0]
    assert model == "second"


def test_equivalent_provider_aliases_and_endpoint_settings_share_identity(tmp_path):
    database = tmp_path / "index.sqlite3"
    url = "https://user:password@embedding.example/v1/embeddings"
    first = RAGService(
        SQLiteStore(str(database)),
        OpenAICompatibleEmbedder("model", url, "api-secret", dimensions=384, provider="openai"),
        object(),
    )
    fingerprint = first.store.connection.execute(
        "SELECT embedding_provider, embedding_fingerprint FROM index_metadata"
    ).fetchone()
    assert fingerprint[0] == "openai-compatible"
    assert "user" not in fingerprint[1] and "password" not in fingerprint[1]
    first.store.close()
    second = RAGService(
        SQLiteStore(str(database)),
        OpenAICompatibleEmbedder(
            "model", url, "another-api-secret", dimensions=384, provider="openai_compatible"
        ),
        object(),
    )
    assert second.store.connection.execute("SELECT COUNT(*) FROM index_metadata").fetchone()[0] == 1


def test_external_endpoint_or_dimensions_change_index_identity(tmp_path):
    database = tmp_path / "index.sqlite3"
    first = RAGService(
        SQLiteStore(str(database)),
        OpenAICompatibleEmbedder(
            "model", "https://one.example/v1/embeddings", "key", dimensions=384
        ),
        object(),
    )
    first.store.create_document("First", "text", {}, ["text"], [[1.0]])
    first.store.close()
    with pytest.raises(ValueError, match="configuration"):
        RAGService(
            SQLiteStore(str(database)),
            OpenAICompatibleEmbedder(
                "model", "https://two.example/v1/embeddings", "key", dimensions=768
            ),
            object(),
        )
