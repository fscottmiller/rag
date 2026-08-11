import time
from src.ultralight_rag.pipeline.embeddings import OpenAICompatibleEmbedder

embedder = OpenAICompatibleEmbedder(
    model="dummy",
    url="http://127.0.0.1:8080/v1/embeddings",
    dimensions=10,
    provider="ollama"
)

start = time.time()
for _ in range(50):
    embedder.embed(["hello world"] * 10)
end = time.time()

print(f"Time taken: {end - start:.4f} seconds")
