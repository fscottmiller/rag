import time
import httpx
from src.ultralight_rag.pipeline.embeddings import OpenAICompatibleEmbedder

embedder = OpenAICompatibleEmbedder(
    model="dummy",
    url="http://127.0.0.1:8080/v1/embeddings",
    dimensions=10,
    provider="ollama"
)
client = httpx.Client()

start = time.time()
for _ in range(500):
    client.post(
        embedder.url,
        json={"model": embedder.model, "input": ["hello world"], "encoding_format": "float", "dimensions": 10},
        headers={"Content-Type": "application/json"}
    )
end = time.time()

print(f"Time taken with httpx (reusing client): {end - start:.4f} seconds")
