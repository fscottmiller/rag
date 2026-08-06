"""Concurrency tests for SQLiteStore: many threads sharing one connection.

ARCHITECTURE.md claims the single sqlite3 connection (check_same_thread=False,
serialized by a threading.RLock via the _synchronized decorator) "keeps the
single-connection design safe for concurrent threadpool requests." Nothing in
the rest of the suite drove concurrent calls against a real store before this
file existed -- the only threaded test elsewhere covers FastEmbed's lazy model
init, not SQLiteStore. Every Starlette read handler is a synchronous def, so
the real server really does dispatch reads (and writes) across threadpool
threads, making this claim load-bearing.

Each test uses threading.Barrier to force maximum, deterministic contention
at a single instant instead of sleeping and hoping for an interleaving.
"""

import queue
import sqlite3
import threading

from utralight_rag.storage.sqlite import SQLiteStore

THREAD_COUNT = 12


def _run_threads(targets):
    """Run each callable in its own thread and re-raise any thread exception.

    Exceptions raised inside a thread do not propagate to the main thread by
    default, so a crashed worker would otherwise pass the test silently.
    """
    errors: queue.Queue = queue.Queue()

    def wrap(target):
        try:
            target()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread below
            errors.put(exc)

    threads = [threading.Thread(target=wrap, args=(target,)) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "thread did not finish - suspected deadlock"
    if not errors.empty():
        raise errors.get()


def test_concurrent_create_document_from_many_threads_produces_exact_counts():
    store = SQLiteStore()
    barrier = threading.Barrier(THREAD_COUNT)

    def make_creator(index):
        def create():
            barrier.wait()
            store.create_document(
                f"Doc {index}",
                "text",
                {},
                [f"chunk {index}"],
                [[float(index), 1.0]],
                document_id=f"doc-{index}",
            )

        return create

    _run_threads([make_creator(i) for i in range(THREAD_COUNT)])

    documents = store.list_documents()
    assert len(documents) == THREAD_COUNT
    assert {doc["id"] for doc in documents} == {f"doc-{i}" for i in range(THREAD_COUNT)}
    assert store.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == THREAD_COUNT
    assert store.connection.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == THREAD_COUNT
    store.close()


def test_concurrent_search_calls_return_coherent_results_without_corruption():
    store = SQLiteStore()
    for index in range(THREAD_COUNT):
        store.create_document(
            f"Doc {index}",
            "text",
            {},
            [f"chunk {index}"],
            [[1.0, 0.0]],
            document_id=f"doc-{index}",
        )

    barrier = threading.Barrier(THREAD_COUNT)
    outcomes: queue.Queue = queue.Queue()

    def searcher():
        barrier.wait()
        outcomes.put(store.search([1.0, 0.0], THREAD_COUNT))

    _run_threads([searcher for _ in range(THREAD_COUNT)])

    all_document_ids = {f"doc-{i}" for i in range(THREAD_COUNT)}
    result_batches = list(outcomes.queue)
    assert len(result_batches) == THREAD_COUNT
    for results in result_batches:
        assert len(results) == THREAD_COUNT
        assert {item["document_id"] for item in results} == all_document_ids
        scores = [item["score"] for item in results]
        assert scores == sorted(scores, reverse=True)
        # every indexed vector equals the query vector here, so every score
        # must be exactly the identical-vector score -- a torn/interleaved
        # read would show up as a stray low or malformed score.
        assert all(score == 1.0 for score in scores)
    store.close()


def test_concurrent_creates_and_searches_interleave_without_error():
    # Mixes writers and readers on one Barrier so create_document and search
    # race for the same RLock at the same instant -- the scenario
    # ARCHITECTURE.md's safety claim is actually about. A shared connection
    # used from multiple threads without serialization would surface as
    # sqlite3.ProgrammingError ("SQLite objects created in a thread can only
    # be used in that same thread") or as malformed/partial rows.
    store = SQLiteStore()
    writer_count = THREAD_COUNT // 2
    reader_count = THREAD_COUNT // 2
    barrier = threading.Barrier(writer_count + reader_count)
    search_outcomes: queue.Queue = queue.Queue()

    def make_writer(index):
        def write():
            barrier.wait()
            store.create_document(
                f"Doc {index}",
                "text",
                {},
                [f"chunk {index}"],
                [[1.0, 0.0]],
                document_id=f"doc-{index}",
            )

        return write

    def reader():
        barrier.wait()
        search_outcomes.put(store.search([1.0, 0.0], 100))

    _run_threads(
        [make_writer(i) for i in range(writer_count)] + [reader for _ in range(reader_count)]
    )

    assert store.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == writer_count
    assert store.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == writer_count
    assert store.connection.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == writer_count

    valid_document_ids = {f"doc-{i}" for i in range(writer_count)}
    for results in search_outcomes.queue:
        # A search racing a write must see either the pre-write or the
        # post-write state cleanly -- every row it does return must be a
        # fully-written, well-formed result belonging to a real document.
        for item in results:
            assert item["document_id"] in valid_document_ids
            assert isinstance(item["score"], float)
            assert item["title"]
    store.close()


def test_concurrent_access_never_raises_sqlite_programming_error():
    # Direct, explicit check for the specific failure mode an unserialized
    # shared connection produces: sqlite3.ProgrammingError. _run_threads
    # already re-raises any thread exception, so this test fails loudly
    # (rather than silently) if that ever happens.
    store = SQLiteStore()
    barrier = threading.Barrier(THREAD_COUNT)

    def make_worker(index):
        def work():
            barrier.wait()
            if index % 2 == 0:
                store.create_document(
                    f"Doc {index}",
                    "text",
                    {},
                    [f"chunk {index}"],
                    [[1.0, 0.0]],
                    document_id=f"doc-{index}",
                )
            else:
                store.search([1.0, 0.0], 10)

        return work

    try:
        _run_threads([make_worker(i) for i in range(THREAD_COUNT)])
    except sqlite3.ProgrammingError as exc:  # pragma: no cover - would fail the test below
        raise AssertionError(f"shared connection was not safely serialized: {exc}") from exc
    finally:
        store.close()
