import asyncio
import time
from fastapi import FastAPI
import uvicorn
import httpx
import urllib3
import json
import urllib.request
import urllib.error
from urllib.parse import urlparse

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
    uvicorn.run(app, host="127.0.0.1", port=8085, log_level="critical")

if __name__ == "__main__":
    import threading
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(1)

    embedder_urllib = OpenAICompatibleEmbedder(
        model="dummy",
        url="http://127.0.0.1:8085/v1/embeddings",
        dimensions=10,
        provider="ollama"
    )

    class EmbedderUrllib3(OpenAICompatibleEmbedder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Use urllib3 with connection pooling!
            parsed_url = urlparse(self.url)
            self.pool = urllib3.PoolManager()

        def _embed_batch(self, inputs):
            if not inputs: return []
            payload = {"model": self.model, "input": inputs, "encoding_format": "float"}
            if self.dimensions is not None: payload["dimensions"] = self.dimensions
            headers = {"Content-Type": "application/json"}
            if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"

            resp = self.pool.request('POST', self.url, body=json.dumps(payload).encode('utf-8'), headers=headers, timeout=self.timeout)
            body = json.loads(resp.data.decode('utf-8'))
            data = body.get("data")
            return [item["embedding"] for item in sorted(data, key=lambda i: i["index"])]

    embedder_urllib3 = EmbedderUrllib3(
        model="dummy",
        url="http://127.0.0.1:8085/v1/embeddings",
        dimensions=10,
        provider="ollama"
    )

    # warmup
    for _ in range(5):
        embedder_urllib.embed(["hello world"])
        embedder_urllib3.embed(["hello world"])

    start = time.time()
    for _ in range(200):
        embedder_urllib.embed(["hello world"] * 10)
    print(f"urllib time: {time.time() - start:.4f}s")

    start = time.time()
    for _ in range(200):
        embedder_urllib3.embed(["hello world"] * 10)
    print(f"urllib3 (with connection pooling) time: {time.time() - start:.4f}s")
