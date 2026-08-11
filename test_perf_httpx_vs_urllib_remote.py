import asyncio
import time
import threading
import urllib.request
import urllib.error
import urllib.parse
from fastapi import FastAPI
import uvicorn
import httpx
from src.ultralight_rag.pipeline.embeddings import OpenAICompatibleEmbedder
import socket
import json

class SlowAcceptServer:
    def __init__(self, host='127.0.0.1', port=8092):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        self.running = True

    def run(self):
        while self.running:
            try:
                self.sock.settimeout(1.0)
                conn, addr = self.sock.accept()

                # Simulate TCP handshake + SSL latency
                time.sleep(0.05)

                threading.Thread(target=self.handle_client, args=(conn,)).start()
            except socket.timeout:
                pass
            except Exception:
                break

    def handle_client(self, conn):
        with conn:
            try:
                conn.settimeout(1.0)
                while True:
                    data = b""
                    while b"\r\n\r\n" not in data:
                        chunk = conn.recv(1024)
                        if not chunk: break
                        data += chunk
                    if not data: break

                    # Content-Length check (simplified)
                    import re
                    match = re.search(b"Content-Length: (\\d+)", data)
                    if match:
                        length = int(match.group(1))
                        header_end = data.find(b"\r\n\r\n") + 4
                        body_len = len(data) - header_end
                        while body_len < length:
                            chunk = conn.recv(1024)
                            if not chunk: break
                            data += chunk
                            body_len += len(chunk)

                    # Simulating processing
                    time.sleep(0.01)

                    # Send response
                    body = b'{"data": [{"index": 0, "embedding": [0.1, 0.2]}]}'
                    resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
                    conn.sendall(resp)
            except Exception:
                pass

if __name__ == "__main__":
    server = SlowAcceptServer()
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(1)

    class EmbedderUrllib:
        def __init__(self, url):
            self.url = url
            self.model = "dummy"

        def embed(self, texts):
            payload = {"model": self.model, "input": texts, "encoding_format": "float"}
            request = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=5) as response:
                body = json.loads(response.read())
            return [item["embedding"] for item in body["data"]]

    embedder_urllib = EmbedderUrllib("http://127.0.0.1:8092/v1/embeddings")

    embedder_httpx = OpenAICompatibleEmbedder(
        model="dummy",
        url="http://127.0.0.1:8092/v1/embeddings",
        dimensions=2,
        provider="ollama"
    )

    print("Sequential (20 requests with simulated 50ms connection latency):")

    start = time.time()
    for _ in range(20):
        embedder_urllib.embed(["hello"])
    print(f"  urllib (no keep-alive): {time.time() - start:.4f}s")

    start = time.time()
    for _ in range(20):
        embedder_httpx.embed(["hello"])
    print(f"  httpx (with connection pooling): {time.time() - start:.4f}s")
