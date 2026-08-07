import pytest

from utralight_rag.pipeline.chunking import BaseChunker
from utralight_rag.pipeline.embeddings import BaseEmbedder
from utralight_rag.service import RAGService
from utralight_rag.storage.sqlite import SQLiteStore


class KeywordEmbedder(BaseEmbedder):
    def embed(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [float("python" in lower), float("sqlite" in lower), float("fastapi" in lower)]
            )
        return vectors


class FixedChunker(BaseChunker):
    def chunk(self, text):
        return [part.strip() for part in text.split("|") if part.strip()]


@pytest.fixture(autouse=True)
def _close_sqlite_stores_opened_during_test(monkeypatch):
    """Close every `SQLiteStore` a test opens, however it opened it.

    `SQLiteStore` owns a live `sqlite3.Connection` with no lifetime bound
    except a manual `.close()`. Across the suite, tests construct stores in
    many different ways -- via the `service` fixture below, via ad hoc
    `SQLiteStore(...)` calls inside individual tests, sometimes several per
    test -- and historically none of them were closed, so the connection was
    only released at GC. That raises `ResourceWarning: unclosed database`
    under this project's zero-warning enforcement.

    Rather than requiring every call site to remember a `.close()` or a
    `with SQLiteStore(...) as store:` block, this autouse fixture wraps
    `SQLiteStore.__init__` for the duration of each test to record every
    instance the test creates, then closes all of them at teardown.
    `SQLiteStore.close()` is idempotent, so this stays safe even for stores a
    test already closed itself.

    Limitation -- do not create a `SQLiteStore` in a module- or session-scoped
    fixture. This fixture is function-scoped, so pytest instantiates any
    higher-scoped fixture *before* the patch is installed; such a store is
    built by the original `__init__`, is never tracked, and leaks. It does not
    fail quietly, but it fails badly: the `ResourceWarning` surfaces as an
    unraisable exception at session teardown, attributed to no test at all,
    which is far harder to diagnose than the per-test failures this fixture
    was written to eliminate. Use a function-scoped fixture, or close the
    store explicitly with `with SQLiteStore(...) as store:`.
    """
    opened: list[SQLiteStore] = []
    original_init = SQLiteStore.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        opened.append(self)

    monkeypatch.setattr(SQLiteStore, "__init__", _tracking_init)
    yield
    for store in opened:
        store.close()


@pytest.fixture
def service():
    return RAGService(SQLiteStore(), KeywordEmbedder(), FixedChunker())
