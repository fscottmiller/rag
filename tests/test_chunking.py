from types import SimpleNamespace

import pytest

from ultralight_rag.pipeline.chunking import ChonkieChunker


def test_chonkie_strategies_return_text_chunks():
    text = "First sentence. Second sentence. Third sentence."
    for strategy in ("recursive", "sentence", "token"):
        chunks = ChonkieChunker(strategy=strategy, chunk_size=20, chunk_overlap=2).chunk(text)
        assert chunks
        assert all(isinstance(chunk, str) and chunk.strip() for chunk in chunks)


def test_chunker_returns_empty_for_whitespace():
    assert ChonkieChunker(chunk_size=20, chunk_overlap=2).chunk("  \n\t") == []


def test_recursive_chunker_applies_configured_overlap():
    text = "0123456789" * 12
    without_overlap = ChonkieChunker(chunk_size=20, chunk_overlap=0).chunk(text)
    with_overlap = ChonkieChunker(chunk_size=20, chunk_overlap=5).chunk(text)

    assert len(without_overlap) == len(with_overlap)
    assert with_overlap[1].startswith(without_overlap[0][-5:])
    assert with_overlap != without_overlap


@pytest.mark.parametrize(
    ("stub_result", "expected"),
    [
        pytest.param(
            SimpleNamespace(start_index=0, text="stubbed chunk text"),
            ["stubbed chunk text"],
            id="start_index-only",
        ),
        pytest.param(
            SimpleNamespace(end_index=5, text="stubbed chunk text"),
            ["stubbed chunk text"],
            id="end_index-only",
        ),
        pytest.param(
            SimpleNamespace(
                start_index=0,
                end_index=len("irrelevant input"),
                text="stubbed chunk text",
            ),
            ["irrelevant input"],
            id="both-present-takes-overlap-branch",
        ),
    ],
)
def test_recursive_chunker_index_fallback(monkeypatch, stub_result, expected):
    # Stub chunk results mirror Chonkie result shapes the code must not
    # assume are always fully populated: start-only and end-only fall back
    # to the plain chunk text, while a fully-populated stub must still take
    # the overlap-extension branch (a negative control against a future
    # change that makes that branch permanently unreachable).
    chunker = ChonkieChunker(strategy="recursive", chunk_size=20, chunk_overlap=5)
    monkeypatch.setattr(chunker, "_chunker", SimpleNamespace(chunk=lambda text: [stub_result]))

    text = "irrelevant input"
    assert chunker.chunk(text) == expected


def test_recursive_chunker_warns_once_per_call_on_missing_index(monkeypatch, caplog):
    # A malformed result set (many items missing an index) must log exactly
    # one warning per chunk() call, not one per item, or it floods the log.
    chunker = ChonkieChunker(strategy="recursive", chunk_size=20, chunk_overlap=5)
    stub_results = [SimpleNamespace(start_index=0, text=f"chunk {i}") for i in range(5)]
    monkeypatch.setattr(chunker, "_chunker", SimpleNamespace(chunk=lambda text: stub_results))

    with caplog.at_level("WARNING", logger="ultralight_rag.pipeline.chunking"):
        chunks = chunker.chunk("irrelevant input")

    assert chunks == [f"chunk {i}" for i in range(5)]
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    # The warning must never carry document content.
    for record in warnings:
        assert "chunk 0" not in record.getMessage()
        assert "irrelevant input" not in record.getMessage()


@pytest.mark.parametrize("text_value", [b"bee", 123, ["a"], None])
def test_chunk_text_treats_non_str_text_attribute_as_missing(text_value):
    # A duck-typed result whose .text is present but not a str must fall back
    # to str(result), same as a fully absent .text attribute.
    #
    # bytes is the case that motivated the isinstance guard: it was previously
    # returned verbatim and appended to chunks, so a non-str slipped into
    # document text silently. int/list were returned verbatim too but crashed
    # at the caller's .strip(); None already fell through. Only the non-None
    # values distinguish this guard from a plain `is not None` check, so they
    # are what make this a regression test rather than a tautology.
    from ultralight_rag.pipeline.chunking import _chunk_text

    class Weird:
        text = text_value

        def __str__(self) -> str:
            return "weird repr"

    assert _chunk_text(Weird()) == "weird repr"
    assert _chunk_text(SimpleNamespace(text="plain")) == "plain"


def test_chunker_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="chunk_size"):
        ChonkieChunker(chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        ChonkieChunker(chunk_size=4, chunk_overlap=4)
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        ChonkieChunker(strategy="unknown")
