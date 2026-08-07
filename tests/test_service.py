import pytest

from utralight_rag.config import Settings
from utralight_rag.pipeline.embeddings import FastEmbedEmbedder, OpenAICompatibleEmbedder
from utralight_rag.service import DocumentTooLargeError, RAGService
from utralight_rag.storage.sqlite import DocumentNotFoundError, SQLiteStore


class _OneChunk:
    def chunk(self, content):
        return [content]


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


def test_service_accepts_content_at_exact_document_byte_limit(service):
    limit = 8
    configured = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=limit)
    )
    content = "x" * limit
    assert len(content.encode("utf-8")) == limit
    document = configured.ingest("Exact", content)
    assert document["chunk_count"] == 1


def test_service_rejects_content_one_byte_over_document_limit(service):
    limit = 8
    configured = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=limit)
    )
    content = "x" * (limit + 1)
    assert len(content.encode("utf-8")) == limit + 1
    with pytest.raises(DocumentTooLargeError):
        configured.ingest("Over", content)


def test_service_measures_document_size_in_utf8_bytes_not_characters(service):
    # "e" with an acute accent encodes to 2 UTF-8 bytes each: 2 characters, 4 bytes.
    # A char-length check (len(content)) would wrongly accept this; only a byte-length
    # check (len(content.encode("utf-8"))) correctly rejects it against a limit of 3.
    limit = 3
    configured = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=limit)
    )
    content = "éé"
    assert len(content) <= limit
    assert len(content.encode("utf-8")) > limit
    with pytest.raises(DocumentTooLargeError):
        configured.ingest("Multibyte", content)


def test_service_update_accepts_and_rejects_at_exact_document_byte_limit(service):
    limit = 8
    configured = RAGService(
        SQLiteStore(), service.embedder, service.chunker, Settings(max_document_bytes=limit)
    )
    document = configured.ingest("Seed", "seed")

    updated = configured.update(document["id"], "Exact", "x" * limit)
    assert updated["chunk_count"] == 1

    with pytest.raises(DocumentTooLargeError):
        configured.update(document["id"], "Over", "x" * (limit + 1))


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


def test_two_live_services_reject_stale_identity_before_vector_table_creation(tmp_path):
    database = tmp_path / "index.sqlite3"
    first = RAGService(SQLiteStore(str(database)), FastEmbedEmbedder("first"), _OneChunk())
    second = RAGService(SQLiteStore(str(database)), FastEmbedEmbedder("second"), _OneChunk())
    first.embedder.embed = lambda texts: [[1.0] for _ in texts]
    second.embedder.embed = lambda texts: [[1.0, 0.0] for _ in texts]

    with pytest.raises(ValueError, match="configuration"):
        first.ingest("First", "text")
    second.ingest("Second", "text")
    with pytest.raises(ValueError, match="configuration"):
        first.search("text")


def test_stale_identity_rejects_before_any_embedding_call(tmp_path):
    service = RAGService(
        SQLiteStore(str(tmp_path / "index.sqlite3")), FastEmbedEmbedder("first"), _OneChunk()
    )
    service.embedder.embed = lambda texts: [[1.0] for _ in texts]
    document = service.ingest("First", "text")
    service.store.connection.execute("UPDATE index_metadata SET embedding_model = 'stale'")
    service.store.connection.commit()
    calls = 0

    def embed(texts):
        nonlocal calls
        calls += 1
        return [[1.0] for _ in texts]

    service.embedder.embed = embed
    for operation in (
        lambda: service.ingest("Second", "text"),
        lambda: service.update(document["id"], "Updated", "text"),
        lambda: service.search("text"),
        lambda: service.delete_document(document["id"]),
    ):
        with pytest.raises(ValueError, match="configuration"):
            operation()
    assert calls == 0
    assert service.get_document(document["id"])["id"] == document["id"]


def test_delete_rechecks_embedding_identity_within_transaction(tmp_path):
    store = SQLiteStore(str(tmp_path / "index.sqlite3"))
    service = RAGService(store, FastEmbedEmbedder("first"), _OneChunk())
    service.embedder.embed = lambda texts: [[1.0] for _ in texts]
    document = service.ingest("First", "text")
    store.connection.execute("UPDATE index_metadata SET embedding_model = 'stale'")
    store.connection.commit()

    with pytest.raises(ValueError, match="configuration"):
        store.delete_document(
            document["id"], expected_embedding_identity=service._embedding_identity
        )
    assert store.get_document(document["id"])["id"] == document["id"]


def test_persistent_index_rejects_builtin_embedder_subclass(tmp_path):
    class AlteredFastEmbedder(FastEmbedEmbedder):
        def embed(self, texts):
            return [[1.0] for _ in texts]

    with pytest.raises(ValueError, match="known identity"):
        RAGService(SQLiteStore(str(tmp_path / "index.sqlite3")), AlteredFastEmbedder(), _OneChunk())


def test_equivalent_provider_aliases_and_endpoint_settings_share_identity(tmp_path):
    database = tmp_path / "index.sqlite3"
    url = "https://embedding.example/v1/embeddings?api-version=2026-01-01"
    first = RAGService(
        SQLiteStore(str(database)),
        OpenAICompatibleEmbedder("model", url, "api-secret", dimensions=384, provider="openai"),
        object(),
    )
    fingerprint = first.store.connection.execute(
        "SELECT embedding_provider, embedding_fingerprint FROM index_metadata"
    ).fetchone()
    assert fingerprint[0] == "openai-compatible"
    first.store.close()
    second = RAGService(
        SQLiteStore(str(database)),
        OpenAICompatibleEmbedder(
            "model", url, "another-api-secret", dimensions=384, provider="openai_compatible"
        ),
        object(),
    )
    assert second.store.connection.execute("SELECT COUNT(*) FROM index_metadata").fetchone()[0] == 1


def test_embedding_endpoint_rejects_embedded_credentials():
    # The query-string cases ("api_key", "accessToken") are strict subsets of
    # the per-arm parametrized suffix tests below; only the userinfo
    # (user:password@) case is unique to this test, so it is kept on its own.
    with pytest.raises(ValueError, match="credentials"):
        OpenAICompatibleEmbedder(
            "model", "https://user:password@embedding.example/v1/embeddings", "api-secret"
        )


@pytest.mark.parametrize("name", ["auth", "authorization", "sig"])
def test_embedding_endpoint_rejects_set_membership_parameter_names(name):
    # These names match the predicate's set-membership arm exactly but do not
    # end with any of the suffix-arm strings ("key", "token", "secret",
    # "password", "credential", "signature"). If the predicate's `or` were
    # mutated to `and`, these would no longer be rejected.
    url = f"https://embedding.example/v1/embeddings?{name}=secret"
    with pytest.raises(ValueError, match="credentials"):
        OpenAICompatibleEmbedder("model", url, "api-secret")


@pytest.mark.parametrize(
    "name",
    [
        "apikey",
        "clientToken",
        "apiSecret",
        "userPassword",
        "clientCredential",
        "requestSignature",
    ],
)
def test_embedding_endpoint_rejects_every_credential_suffix(name):
    # Each of these names ends with one of the six suffix-arm strings but is
    # not itself in the membership set {"auth", "authorization", "sig"}. This
    # isolates the suffix arm and, individually, pins each tuple entry so a
    # mutant that drops any single suffix is caught.
    url = f"https://embedding.example/v1/embeddings?{name}=secret"
    with pytest.raises(ValueError, match="credentials"):
        OpenAICompatibleEmbedder("model", url, "api-secret")


@pytest.mark.parametrize(
    "name",
    [
        "API-KEY",
        "api_key",
        "Api.Key",
        "api_key_",
        "api.key!",
    ],
)
def test_embedding_endpoint_normalizes_parameter_names_before_matching(name):
    # The predicate strips non-alphanumeric characters and lowercases before
    # comparing, so differently-punctuated/cased spellings of the same
    # logical parameter name must all be treated identically.
    #
    # "api_key_" and "api.key!" specifically pin the isalnum() stripping
    # half of that claim (not just the lowercasing half): their punctuation
    # lands *after* the "key" suffix, so `name.lower()` alone would end with
    # "_" / "!" rather than "key" and the suffix arm would stop matching. The
    # other three cases only place punctuation *before* the suffix, where
    # `.lower().endswith(...)` already matches without any stripping -- so
    # they cannot, by themselves, prove the isalnum() filter is load-bearing.
    url = f"https://embedding.example/v1/embeddings?{name}=secret"
    with pytest.raises(ValueError, match="credentials"):
        OpenAICompatibleEmbedder("model", url, "api-secret")


@pytest.mark.parametrize("name", ["si-g", "a.u.t.h"])
def test_embedding_endpoint_normalizes_membership_parameter_names_before_matching(name):
    # The set-membership arm (`in {"auth", "authorization", "sig"}`) is
    # compared against the same stripped/lowercased name. "si-g" and
    # "a.u.t.h" have punctuation embedded *within* the matched substring, so
    # they only normalize into the membership set once isalnum() stripping
    # is applied: `name.lower()` alone leaves "si-g" and "a.u.t.h", neither
    # of which is a member. This pins the isalnum() half of the
    # normalization claim for the membership arm, mirroring the suffix-arm
    # cases above.
    url = f"https://embedding.example/v1/embeddings?{name}=secret"
    with pytest.raises(ValueError, match="credentials"):
        OpenAICompatibleEmbedder("model", url, "api-secret")


@pytest.mark.parametrize("name", ["api-version", "model", "timeout"])
def test_embedding_endpoint_allows_benign_query_parameters(name):
    # Getting the predicate wrong in the other direction would break
    # legitimate endpoints; these benign names must not be flagged.
    url = f"https://embedding.example/v1/embeddings?{name}=value"
    embedder = OpenAICompatibleEmbedder("model", url, "api-secret")
    assert embedder.url == url


def test_embedding_endpoint_keeps_non_secret_query_parameters_in_identity(tmp_path):
    database = tmp_path / "index.sqlite3"
    first = RAGService(
        SQLiteStore(str(database)),
        OpenAICompatibleEmbedder(
            "model", "https://embedding.example/v1/embeddings?api-version=2026-01-01", "key"
        ),
        object(),
    )
    first.store.create_document("First", "text", {}, ["text"], [[1.0]])
    first.store.close()

    with pytest.raises(ValueError, match="configuration"):
        RAGService(
            SQLiteStore(str(database)),
            OpenAICompatibleEmbedder(
                "model", "https://embedding.example/v1/embeddings?api-version=2026-02-01", "key"
            ),
            object(),
        )


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
