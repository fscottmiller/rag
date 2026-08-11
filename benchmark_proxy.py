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
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="critical")

if __name__ == "__main__":
    import threading
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(1) # wait for server

    embedder_urllib = OpenAICompatibleEmbedder(
        model="dummy",
        url="http://127.0.0.1:8081/v1/embeddings",
        dimensions=10,
        provider="ollama"
    )

    # benchmark urllib
    start = time.time()
    for _ in range(20):
        embedder_urllib.embed(["hello world"] * 10)
    print(f"urllib time: {time.time() - start:.4f}s")
