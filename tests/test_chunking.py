from src.pipeline.chunking import ChonkieChunker


def test_chonkie_strategies_return_text_chunks():
    text = "First sentence. Second sentence. Third sentence."
    for strategy in ("recursive", "sentence", "token"):
        chunks = ChonkieChunker(strategy=strategy, chunk_size=20, chunk_overlap=2).chunk(text)
        assert chunks
        assert all(isinstance(chunk, str) and chunk.strip() for chunk in chunks)
