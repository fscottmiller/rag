# Utralight RAG MCP

A small, local Retrieval-Augmented Generation (RAG) service. It stores documents and embeddings in SQLite, then exposes the same index through REST and MCP.

The default database is in memory for a zero-setup local experience. Set `RAG_DATABASE_PATH` to use file-backed SQLite when data should survive process restarts.

## Quick start

```bash
uv sync
uv sync --extra local-embeddings
uv run uvicorn utralight_rag.api.main:app --reload
```

The second command installs the default Sentence Transformers provider. For any OpenAI-compatible embedding API, configure the endpoint, model, and optional credentials in the environment:

```bash
RAG_EMBEDDING_PROVIDER=openai-compatible \
RAG_EMBEDDING_URL=https://api.openai.com/v1/embeddings \
RAG_EMBEDDING_MODEL=text-embedding-3-small \
RAG_EMBEDDING_API_KEY="$OPENAI_API_KEY" \
uv run uvicorn utralight_rag.api.main:app --reload
```

Ollama uses the same OpenAI-compatible protocol and settings; there are no separate Ollama variables:

```bash
RAG_EMBEDDING_PROVIDER=ollama \
RAG_EMBEDDING_URL=http://localhost:11434/v1/embeddings \
RAG_EMBEDDING_MODEL=nomic-embed-text \
uv run uvicorn utralight_rag.api.main:app --reload
```

Open the interactive API documentation at <http://127.0.0.1:8000/docs>.

## Index isolation

Each running service instance owns one index, one SQLite database, and one embedding/chunking configuration. Run separate instances for independent indexes rather than configuring collections inside one process:

```bash
RAG_DATABASE_PATH=/var/lib/rag/index-a.sqlite uv run uvicorn utralight_rag.api.main:app --port 8001
RAG_DATABASE_PATH=/var/lib/rag/index-b.sqlite uv run uvicorn utralight_rag.api.main:app --port 8002
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

Run the MCP server over stdio (the default):

```bash
uv run python -m utralight_rag.mcp_server.server
```

Use streamable HTTP when an MCP client or proxy needs an HTTP endpoint:

```bash
MCP_TRANSPORT=streamable-http \
MCP_HOST=127.0.0.1 \
MCP_PORT=8000 \
MCP_PATH=/mcp \
uv run python -m utralight_rag.mcp_server.server
```

Available tools: `rag_search`, `list_documents`, `get_document`, `upload_document`, and `delete_document`.

When REST and MCP run as separate processes, set the same file-backed `RAG_DATABASE_PATH` for both if they should share an index. With the default `:memory:` database, each process has its own in-memory index.

## Authentication and authorization

The default runtime mode is `RAG_AUTH_MODE=none`: REST and MCP requests are unauthenticated, and non-browser callers can read, search, upload, update, and delete documents. REST rejects cross-origin browser mutations even in open mode to prevent malicious webpages from posting to a local instance. Use this mode only on a trusted local network.

For a deployment behind an authenticating reverse proxy or Cloudflare Access tunnel, set `RAG_AUTH_MODE=trusted-proxy`. The application trusts the proxy to authenticate the request and to overwrite the configured identity and role headers:

```bash
RAG_AUTH_MODE=trusted-proxy \
RAG_PROXY_USER_HEADER=Cf-Access-Authenticated-User-Email \
RAG_PROXY_ROLE_HEADER=X-Auth-Request-Role \
RAG_PROXY_ADMIN_ROLE=admin \
RAG_PROXY_READER_ROLE=reader \
uv run uvicorn utralight_rag.api.main:app --host 127.0.0.1 --port 8000
```

Configure the proxy to strip client-supplied versions of these headers and set them only after successful authentication. Do not expose the application directly in trusted-proxy mode: it does not validate proxy credentials itself. The `admin` role can perform every operation. The `reader` role can list and retrieve documents and run searches, but cannot upload, update, or delete documents. The same policy applies to MCP tools over streamable HTTP; stdio is intended for local use.

## Configuration

Set these environment variables as needed:

These variables define the configuration of the current index, not per-document defaults. All documents ingested by one instance use the same chunking settings and embedding configuration. To use another configuration, start another instance with its own database path.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATABASE_PATH` | `:memory:` | Use a SQLite file instead of memory. |
| `RAG_EMBEDDING_PROVIDER` | `sentence-transformers` | Select `sentence-transformers`, `openai-compatible`, or `ollama`. |
| `RAG_EMBEDDING_MODEL` | provider-dependent | Embedding model name sent to the configured provider. Ollama defaults to `nomic-embed-text`. |
| `RAG_EMBEDDING_URL` | provider-dependent | OpenAI-compatible embeddings endpoint. Ollama defaults to `http://localhost:11434/v1/embeddings`. |
| `RAG_EMBEDDING_API_KEY` | `OPENAI_API_KEY` fallback | Optional Bearer token for the embeddings endpoint. |
| `RAG_EMBEDDING_TIMEOUT` | `60` | Embedding request timeout in seconds. |
| `RAG_EMBEDDING_DIMENSIONS` | unset | Optional output dimension sent to compatible providers. |
| `RAG_CHUNKER` | `recursive` | Select `recursive`, `sentence`, or `token`. |
| `RAG_CHUNK_SIZE` | `512` | Target chunk size. |
| `RAG_CHUNK_OVERLAP` | `64` | Overlap between chunks; recursive chunking prepends the prior chunk's trailing characters. |
| `RAG_MAX_DOCUMENT_BYTES` | `10485760` | Maximum document content size accepted by REST. |
| `RAG_EMBEDDING_BATCH_SIZE` | `64` | Maximum number of chunks sent to an embedding provider per request. |
| `MCP_TRANSPORT` | `stdio` | Select `stdio` or `streamable-http`. |
| `MCP_HOST` | `127.0.0.1` | Bind host for streamable HTTP. |
| `MCP_PORT` | `8000` | Bind port for streamable HTTP. |
| `MCP_PATH` | `/mcp` | Streamable HTTP path. |
| `RAG_AUTH_MODE` | `none` | Select `none` or `trusted-proxy`. |
| `RAG_PROXY_USER_HEADER` | `Cf-Access-Authenticated-User-Email` | Trusted proxy header containing the authenticated identity. |
| `RAG_PROXY_ROLE_HEADER` | `X-Auth-Request-Role` | Trusted proxy header containing `admin` or `reader`. |
| `RAG_PROXY_ADMIN_ROLE` | `admin` | Value in the role header that grants full access. |
| `RAG_PROXY_READER_ROLE` | `reader` | Value in the role header that grants read/search access. |

Run the automated tests with coverage enforcement:

```bash
uv run coverage run -m pytest
uv run coverage report
```

The suite fails if total branch-aware coverage drops below 95%. GitHub Actions runs it automatically on every push and pull request against Python 3.11, 3.12, and 3.13.

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and trade-offs.
