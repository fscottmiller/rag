import pytest

from utralight_rag.pipeline.chunking import ChonkieChunker


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


def test_chunker_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="chunk_size"):
        ChonkieChunker(chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        ChonkieChunker(chunk_size=4, chunk_overlap=4)
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        ChonkieChunker(strategy="unknown")
