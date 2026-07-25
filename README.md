# Transient RAG MCP

A small, local Retrieval-Augmented Generation (RAG) service. It stores documents and embeddings in SQLite, then exposes the same index through REST and MCP.

The default database is in memory. Data is intentionally temporary.

## Quick start

```bash
uv sync
uv sync --extra local-embeddings
uv run uvicorn src.api.main:app --reload
```

The second command installs the default Sentence Transformers provider. For an OpenAI-compatible embedding API, configure the endpoint, model, and credentials in the environment:

```bash
RAG_EMBEDDING_PROVIDER=openai-compatible \
RAG_EMBEDDING_URL=https://api.openai.com/v1/embeddings \
RAG_EMBEDDING_MODEL=text-embedding-3-small \
RAG_EMBEDDING_API_KEY="$OPENAI_API_KEY" \
uv run uvicorn src.api.main:app --reload
```

Ollama remains available as an optional provider for deployments that already run it locally.

Open the interactive API documentation at <http://127.0.0.1:8000/docs>.

## Index isolation

Each running service instance owns one index, one SQLite database, and one embedding/chunking configuration. Run separate instances for independent indexes rather than configuring collections inside one process:

```bash
RAG_DATABASE_PATH=/var/lib/rag/index-a.sqlite uv run uvicorn src.api.main:app --port 8001
RAG_DATABASE_PATH=/var/lib/rag/index-b.sqlite uv run uvicorn src.api.main:app --port 8002
```

Use a separate database path and port for every instance. This keeps vector spaces, chunking behavior, and lifecycle management isolated.

## REST API

Create a document:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H 'content-type: application/json' \
  -d '{"title":"Notes","content":"Useful context","metadata":{"source":"manual"}}'
```

Search the index:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"useful context","top_k":5}'
```

Available endpoints:

- `POST /documents` — add JSON content or upload a text/Markdown file.
- `GET /documents` — list indexed documents.
- `GET /documents/{id}` — get a document and its chunks.
- `PUT /documents/{id}` — replace a document.
- `DELETE /documents/{id}` — remove a document and its vectors.
- `POST /search` — search chunks by embedding similarity.

## MCP

Run the MCP server over stdio:

```bash
uv run python -m src.mcp_server.server
```

Use SSE instead:

```bash
MCP_TRANSPORT=sse uv run python -m src.mcp_server.server
```

Available tools: `rag_search`, `list_documents`, `get_document`, `upload_document`, and `delete_document`.

When REST and MCP run as separate processes, set the same file-backed `RAG_DATABASE_PATH` for both if they should share an index. With the default `:memory:` database, each process has its own transient index.

## Configuration

Set these environment variables as needed:

These variables define the configuration of the current index, not per-document defaults. All documents ingested by one instance use the same chunking settings and embedding configuration. To use another configuration, start another instance with its own database path.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATABASE_PATH` | `:memory:` | Use a SQLite file instead of memory. |
| `RAG_EMBEDDING_PROVIDER` | `sentence-transformers` | Select `sentence-transformers`, `openai-compatible`, or `ollama`. |
| `RAG_EMBEDDING_MODEL` | provider-dependent | Embedding model name sent to the configured provider. |
| `RAG_EMBEDDING_URL` | `https://api.openai.com/v1/embeddings` | OpenAI-compatible embeddings endpoint. |
| `RAG_EMBEDDING_API_KEY` | `OPENAI_API_KEY` fallback | Optional Bearer token for the embeddings endpoint. |
| `RAG_EMBEDDING_TIMEOUT` | `60` | Embedding request timeout in seconds. |
| `RAG_EMBEDDING_DIMENSIONS` | unset | Optional output dimension sent to compatible providers. |
| `RAG_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL. |
| `RAG_OLLAMA_MODEL` | `nomic-embed-text` | Ollama embedding model. |
| `RAG_CHUNKER` | `recursive` | Select `recursive`, `sentence`, or `token`. |
| `RAG_CHUNK_SIZE` | `512` | Target chunk size. |
| `RAG_CHUNK_OVERLAP` | `64` | Overlap between chunks. |

Run the automated tests with coverage enforcement:

```bash
uv run coverage run -m pytest
uv run coverage report
```

The suite fails if total branch-aware coverage drops below 95%. GitHub Actions runs it automatically on every push and pull request against Python 3.11, 3.12, and 3.13.

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and trade-offs.
