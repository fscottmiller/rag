import pytest

from utralight_rag.storage.sqlite import DocumentNotFoundError, SQLiteStore


def vector_rows(store: SQLiteStore) -> int:
    return store.connection.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]


def test_storage_round_trip_and_chunk_metadata():
    store = SQLiteStore()
    created = store.create_document(
        "Guide",
        "full text",
        {"source": "test", "nested": {"section": 2}},
        ["first", "second"],
        [[1.0, 0.0], [0.0, 1.0]],
        document_id="doc-1",
    )

    assert created["id"] == "doc-1"
    assert created["chunk_count"] == 2
    assert created["metadata"] == {"source": "test", "nested": {"section": 2}}
    document = store.get_document("doc-1")
    assert document["content"] == "full text"
    assert [chunk["ordinal"] for chunk in document["chunks"]] == [0, 1]
    assert [chunk["text"] for chunk in document["chunks"]] == ["first", "second"]
    assert vector_rows(store) == 2
    store.close()


def test_file_database_reloads_vector_dimension(tmp_path):
    database = tmp_path / "index.sqlite3"
    first = SQLiteStore(str(database))
    first.create_document("Guide", "text", {}, ["text"], [[1.0, 0.0]])
    assert first.connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert first.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

    first.close()

    reopened = SQLiteStore(str(database))
    assert reopened.search([1.0, 0.0], 1)[0]["score"] == pytest.approx(1.0)
    reopened.close()


def test_replace_and_delete_remove_old_vectors():
    store = SQLiteStore()
    created = store.create_document("Old", "old", {}, ["old", "also old"], [[1.0, 0.0], [0.9, 0.1]])
    assert vector_rows(store) == 2
    with pytest.raises(ValueError, match="one embedding"):
        store.replace_document(created["id"], "Bad", "bad", {}, ["one", "two"], [[1.0, 0.0]])
    assert store.get_document(created["id"])["title"] == "Old"

    replaced = store.replace_document(
        created["id"], "New", "new", {"version": 2}, ["new"], [[0.0, 1.0]]
    )
    assert replaced["title"] == "New"
    assert replaced["chunk_count"] == 1
    assert replaced["chunks"][0]["text"] == "new"
    assert vector_rows(store) == 1

    store.delete_document(created["id"])
    assert vector_rows(store) == 0
    assert store.list_documents() == []
    with pytest.raises(DocumentNotFoundError):
        store.delete_document(created["id"])
    store.close()


def test_replace_and_delete_preserve_not_found_contract_for_empty_documents():
    store = SQLiteStore()
    store.create_document("Empty", "", {}, [], [], document_id="empty")
    store.replace_document("empty", "Still empty", "", {}, [], [])
    store.delete_document("empty")
    with pytest.raises(DocumentNotFoundError):
        store.replace_document("missing", "Missing", "", {}, [], [])
    store.close()


def test_search_refreshes_vector_dimension_for_late_writer(tmp_path):
    database = tmp_path / "shared.sqlite3"
    reader = SQLiteStore(str(database))
    writer = SQLiteStore(str(database))
    assert reader.search([1.0, 0.0], 1) == []
    writer.create_document("Guide", "text", {}, ["text"], [[1.0, 0.0]])

    assert reader.search([1.0, 0.0], 1)[0]["title"] == "Guide"
    reader.close()
    writer.close()


def test_storage_rejects_mismatched_or_empty_embeddings():
    store = SQLiteStore()
    with pytest.raises(ValueError, match="one embedding"):
        store.create_document("Bad", "text", {}, ["one", "two"], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="same non-zero dimension"):
        store.create_document("Bad", "text", {}, ["one", "two"], [[1.0, 0.0], [1.0]])
    with pytest.raises(ValueError, match="same non-zero dimension"):
        store.create_document("Bad", "text", {}, ["one"], [[]])
    with pytest.raises(ValueError, match="finite"):
        store.create_document("Bad", "text", {}, ["one"], [[float("nan")]])
    with pytest.raises(ValueError, match="finite"):
        store.create_document("Bad", "text", {}, ["one"], [["not-a-number"]])
    for value in (True, "1.0"):
        with pytest.raises(ValueError, match="finite"):
            store.create_document("Bad", "text", {}, ["one"], [[value]])
    store.close()


def test_metadata_filter_searches_beyond_initial_candidate_window():
    store = SQLiteStore()
    for index in range(60):
        store.create_document(
            f"Document {index}",
            "SQLite content",
            {"group": "other"},
            [f"chunk {index}"],
            [[1.0, 0.0]],
        )
    target = store.create_document(
        "Target",
        "SQLite content",
        {"group": "target"},
        ["target chunk"],
        [[1.0, 0.0]],
    )

    results = store.search([1.0, 0.0], 1, {"group": "target"})
    assert [item["document_id"] for item in results] == [target["id"]]
    store.close()


def test_metadata_filter_distinguishes_missing_and_null_values():
    store = SQLiteStore()
    store.create_document("Missing", "text", {}, ["text"], [[1.0, 0.0]])
    explicit_null = store.create_document("Null", "text", {"topic": None}, ["text"], [[1.0, 0.0]])

    results = store.search([1.0, 0.0], 2, {"topic": None})
    assert [item["document_id"] for item in results] == [explicit_null["id"]]
    store.close()


def test_search_rejects_wrong_embedding_dimension_and_empty_index_filter():
    store = SQLiteStore()
    created = store.create_document("Guide", "text", {}, ["text"], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dimension"):
        store.search([1.0], 1)
    store.delete_document(created["id"])
    assert store.search([1.0, 0.0], 1, {"group": "none"}) == []
    store.close()


@pytest.mark.parametrize("vector", [[float("nan")], [float("inf")], [True], ["1.0"]])
def test_search_rejects_invalid_embedding_coordinates(vector):
    store = SQLiteStore()
    with pytest.raises(ValueError, match="finite"):
        store.search(vector, 1)
    store.close()


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_search_rejects_non_positive_or_non_integer_top_k(top_k):
    store = SQLiteStore()
    with pytest.raises(ValueError, match="positive integer"):
        store.search([1.0], top_k)
    store.close()
