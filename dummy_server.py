from fastapi import FastAPI
import uvicorn
import asyncio

app = FastAPI()

@app.post("/v1/embeddings")
async def embeddings(payload: dict):
    inputs = payload.get("input", [])
    data = []
    for i, _ in enumerate(inputs):
        data.append({"index": i, "embedding": [0.1] * (payload.get("dimensions") or 10)})
    return {"data": data}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
