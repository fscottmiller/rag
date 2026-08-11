import asyncio
import time
from fastapi import FastAPI
import uvicorn
import httpx
from src.ultralight_rag.pipeline.embeddings import OpenAICompatibleEmbedder

app = FastAPI()

@app.post("/v1/embeddings")
async def embeddings(payload: dict):
    # Simulate processing time
    await asyncio.sleep(0.01)
    inputs = payload.get("input", [])
    data = [{"index": i, "embedding": [0.1] * (payload.get("dimensions") or 10)} for i, _ in enumerate(inputs)]
    return {"data": data}

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8084, log_level="critical")

if __name__ == "__main__":
    import threading
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(1)

    embedder_urllib = OpenAICompatibleEmbedder(
        model="dummy",
        url="http://127.0.0.1:8084/v1/embeddings",
        dimensions=10,
        provider="ollama"
    )

    class EmbedderHttpxPersist(OpenAICompatibleEmbedder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Use httpx with connection pooling/keep-alive!
            self.client = httpx.Client(timeout=self.timeout)

        def _embed_batch(self, inputs):
            if not inputs: return []
            payload = {"model": self.model, "input": inputs, "encoding_format": "float"}
            if self.dimensions is not None: payload["dimensions"] = self.dimensions
            headers = {"Content-Type": "application/json"}
            if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"

            resp = self.client.post(self.url, json=payload, headers=headers)
            body = resp.json()
            data = body.get("data")
            return [item["embedding"] for item in sorted(data, key=lambda i: i["index"])]

    embedder_httpx = EmbedderHttpxPersist(
        model="dummy",
        url="http://127.0.0.1:8084/v1/embeddings",
        dimensions=10,
        provider="ollama"
    )

    # warmup
    for _ in range(5):
        embedder_urllib.embed(["hello world"])
        embedder_httpx.embed(["hello world"])

    start = time.time()
    for _ in range(200):
        embedder_urllib.embed(["hello world"] * 10)
    print(f"urllib time: {time.time() - start:.4f}s")

    start = time.time()
    for _ in range(200):
        embedder_httpx.embed(["hello world"] * 10)
    print(f"httpx (with connection pooling) time: {time.time() - start:.4f}s")
