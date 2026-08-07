# Utralight RAG MCP

A small, local Retrieval-Augmented Generation (RAG) service. It stores documents and embeddings in SQLite, then exposes the same index through REST and MCP.

The default database is in memory for a zero-setup local experience. Set `RAG_DATABASE_PATH` to use file-backed SQLite when data should survive process restarts.

## Quick start

```bash
uv sync
uv run uvicorn utralight_rag.combined:app --host 127.0.0.1 --port 8001
```

Python 3.11 through 3.13 are supported.

FastEmbed and `BAAI/bge-small-en-v1.5` are the default local provider and model; the model downloads on first embedding request. If `RAG_EMBEDDING_PROVIDER` is unset and `RAG_EMBEDDING_API_KEY` or `OPENAI_API_KEY` is set, the service instead uses OpenAI-compatible embeddings. Configure an external provider explicitly with credentials:

```bash
RAG_EMBEDDING_PROVIDER=openai-compatible \
RAG_EMBEDDING_URL=https://api.openai.com/v1/embeddings \
RAG_EMBEDDING_MODEL=text-embedding-3-small \
RAG_EMBEDDING_API_KEY="$OPENAI_API_KEY" \
uv run uvicorn utralight_rag.combined:app --host 127.0.0.1 --port 8001
```

Sentence Transformers remains an optional legacy local provider; install its extra before selecting it, since its PyTorch runtime is not part of the base install:

```bash
uv sync --extra local-embeddings
RAG_EMBEDDING_PROVIDER=sentence-transformers uv run uvicorn utralight_rag.combined:app --host 127.0.0.1 --port 8001
```

Ollama uses the same OpenAI-compatible protocol and settings; there are no separate Ollama variables:

```bash
RAG_EMBEDDING_PROVIDER=ollama \
RAG_EMBEDDING_URL=http://localhost:11434/v1/embeddings \
RAG_EMBEDDING_MODEL=nomic-embed-text \
uv run uvicorn utralight_rag.combined:app --host 127.0.0.1 --port 8001
```

Open the interactive API documentation at <http://127.0.0.1:8001/docs>.

## Index isolation

Each running service instance owns one index, one SQLite database, and one embedding/chunking configuration. Run separate instances for independent indexes rather than configuring collections inside one process:

```bash
RAG_DATABASE_PATH=/var/lib/rag/index-a.sqlite uv run uvicorn utralight_rag.combined:app --port 8001
RAG_DATABASE_PATH=/var/lib/rag/index-b.sqlite uv run uvicorn utralight_rag.combined:app --port 8002
```

Use a separate database path and port for every instance. This keeps vector spaces, chunking behavior, and lifecycle management isolated.

### Upgrade existing indexes

Indexes created before the FastEmbed default did not record their embedding identity. Move or delete a nonempty existing database and re-ingest its documents before starting this version; this prevents 384-dimensional legacy `all-MiniLM-L6-v2` vectors from being mixed with `BAAI/bge-small-en-v1.5` vectors. New indexes record the canonical provider, model, dimensions, and a non-secret endpoint fingerprint; empty indexes can adopt a new identity.

## REST API

Create a document:

```bash
curl -X POST http://127.0.0.1:8001/documents \
  -H 'content-type: application/json' \
  -d '{"title":"Notes","content":"Useful context","metadata":{"source":"manual"}}'
```

Search the index:

```bash
curl -X POST http://127.0.0.1:8001/search \
  -H 'content-type: application/json' \
  -d '{"query":"useful context","top_k":5}'
```

Each result's `score` is `1.0 - cosine_distance`, so it ranges from **-1.0 to 1.0**, not 0 to 1: 1.0 for an identical vector, 0.0 for orthogonal vectors, and -1.0 for opposed vectors.

Available endpoints:

- `POST /documents` — add JSON content or upload a text/Markdown file.
- `GET /documents` — list indexed documents.
- `GET /documents/{id}` — get a document and its chunks.
- `PUT /documents/{id}` — replace a document; accepts JSON, a multipart file upload, or plain
  form fields, the same as `POST /documents`.
- `DELETE /documents/{id}` — remove a document and its vectors.
- `POST /search` — search chunks by embedding similarity.

`POST /documents`, `PUT /documents/{id}`, and `POST /search` all trigger embedding-provider work; if the provider is unreachable or times out they return `503`, and if it is reached but returns something unusable (an error status, malformed JSON, or the wrong shape) they return `502`. Neither response includes the provider's raw response body, endpoint URL, or anything derived from the API key -- that detail is logged server-side only. A genuine bug elsewhere in the service still surfaces as an opaque `500`, exactly as before.

## Logging

The library uses the standard `logging` module (`logging.getLogger(__name__)` per module) and never configures handlers, levels, or `logging.basicConfig` itself -- that is the embedding application's responsibility. At minimum it logs, server-side only: embedding-provider failures (with full upstream detail, at `WARNING`/`ERROR`); authorization denials in `trusted-proxy` mode, identified by principal and attempted action; and document ingest/update/delete/search, identified by document id. It never logs the embedding API key, document content, or chunk text -- `Settings.embedding_api_key` is `repr=False` for the same reason.

## MCP

The combined entry point serves REST and streamable HTTP MCP from one process and one `RAGService`:

```bash
RAG_DATABASE_PATH=/var/lib/rag/index.sqlite \
uv run uvicorn utralight_rag.combined:app --host 127.0.0.1 --port 8001
```

The MCP endpoint is `/mcp` by default. Register it in Claude Code with:

```bash
claude mcp add --transport http rag http://127.0.0.1:8001/mcp
```

Because REST accepts multipart file uploads, large documents can be ingested directly from disk without putting their contents into an MCP tool call. The same is true for updates — `PUT` takes the identical JSON/multipart/form request shapes as `POST`, so an edited file can be re-uploaded in place without reading it back into memory as a JSON string first:

```bash
curl -F 'file=@/path/to/google-sre-book.md' http://127.0.0.1:8001/documents
curl -F 'file=@/path/to/clean-code.txt' http://127.0.0.1:8001/documents
curl -X PUT -F 'file=@/path/to/google-sre-book.md' http://127.0.0.1:8001/documents/{id}
```

When a multipart or form update omits `title`, the title is (re)derived from the uploaded filename, the same as on create — pass `title` explicitly in the form fields if the update should keep the document's existing title.

The indexed documents are then available through `rag_search`, `list_documents`, and `get_document`. Available tools are `rag_search`, `list_documents`, `get_document`, `upload_document`, and `delete_document`. `rag_search` and `upload_document` report embedding-provider failures the same way authorization denials are already reported: a `CallToolResult` with `isError: true` and a machine-readable `_meta.error_type` (`embedding_provider_unavailable` or `embedding_provider_error`), never the provider's raw response detail.

Run the MCP server over stdio (the default) when a client should spawn it directly; this mode remains a separate process:

```bash
uv run python -m utralight_rag.mcp_server.server
```

## Authentication and authorization

The default runtime mode is `RAG_AUTH_MODE=none`: REST and MCP requests are unauthenticated, and non-browser callers can read, search, upload, update, and delete documents. REST rejects cross-origin browser mutations even in open mode to prevent malicious webpages from posting to a local instance. The REST host must also be in `RAG_TRUSTED_HOSTS`, which prevents DNS-rebinding requests from bypassing that protection. Use this mode only on a trusted local network.

For a deployment behind an authenticating reverse proxy or Cloudflare Access tunnel, set `RAG_AUTH_MODE=trusted-proxy`. The application trusts the proxy to authenticate the request and to overwrite the configured identity and role headers:

```bash
RAG_AUTH_MODE=trusted-proxy \
RAG_PROXY_USER_HEADER=Cf-Access-Authenticated-User-Email \
RAG_PROXY_ROLE_HEADER=X-Auth-Request-Role \
RAG_PROXY_ADMIN_ROLE=admin \
RAG_PROXY_READER_ROLE=reader \
uv run uvicorn utralight_rag.combined:app --host 127.0.0.1 --port 8001
```

Configure the proxy to strip client-supplied versions of these headers and set them only after successful authentication. Do not expose the application directly in trusted-proxy mode: it does not validate proxy credentials itself. The `admin` role can perform every operation. The `reader` role can list and retrieve documents and run searches, but cannot upload, update, or delete documents. The same policy applies to MCP tools over streamable HTTP; stdio is intended for local use.

## Configuration

Set these environment variables as needed:

These variables define the configuration of the current index, not per-document defaults. All documents ingested by one instance use the same chunking settings and embedding configuration. To use another configuration, start another instance with its own database path.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATABASE_PATH` | `:memory:` | Use a SQLite file instead of memory. |
| `RAG_EMBEDDING_PROVIDER` | `fastembed`, or `openai-compatible` when an API key is set | Select `fastembed`, `sentence-transformers`, `openai-compatible`, or `ollama`. |
| `RAG_EMBEDDING_MODEL` | provider-dependent | FastEmbed defaults to `BAAI/bge-small-en-v1.5`; Ollama defaults to `nomic-embed-text`. |
| `RAG_EMBEDDING_URL` | provider-dependent | External OpenAI-compatible endpoint; it must use HTTPS. Ollama defaults to `http://localhost:11434/v1/embeddings` and may use HTTP; non-HTTP schemes are rejected. |
| `RAG_EMBEDDING_API_KEY` | `OPENAI_API_KEY` fallback for OpenAI-compatible providers | Required for external OpenAI-compatible providers; optional for Ollama. Ollama never reads `OPENAI_API_KEY`. |
| `RAG_EMBEDDING_TIMEOUT` | `60` | Embedding request timeout in seconds. |
| `RAG_EMBEDDING_DIMENSIONS` | unset | Optional output dimension sent to compatible providers. |
| `RAG_CHUNKER` | `recursive` | Select `recursive`, `sentence`, or `token`. |
| `RAG_CHUNK_SIZE` | `512` | Target chunk size. |
| `RAG_CHUNK_OVERLAP` | `64` | Overlap between chunks; recursive chunking prepends the prior chunk's trailing characters. |
| `RAG_MAX_DOCUMENT_BYTES` | `10485760` | Maximum document content size accepted by REST after parsing. |
| `RAG_MAX_REQUEST_BYTES` | `10551296` | Maximum raw HTTP request body accepted before parsing; keep this at or above the document limit to allow request overhead. |
| `RAG_TRUSTED_HOSTS` | `localhost,127.0.0.1,testserver` | Comma-separated host allowlist for REST requests; configure the public host when deploying behind a proxy. |
| `RAG_EMBEDDING_BATCH_SIZE` | `64` | Maximum number of chunks sent to an embedding provider per request. |
| `MCP_TRANSPORT` | `stdio` | Select `stdio` or `streamable-http`. Only used by the standalone `utralight_rag.mcp_server.server` process, not the combined app. |
| `MCP_HOST` | `127.0.0.1` | Bind host for the standalone MCP server. Ignored by the combined app, which binds to uvicorn's `--host` instead. |
| `MCP_PORT` | `8000` | Bind port for the standalone MCP server. Ignored by the combined app, which binds to uvicorn's `--port` instead. |
| `MCP_PATH` | `/mcp` | Streamable HTTP path, used by both the standalone MCP server and the combined app's MCP mount. |
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

The suite fails if total branch-aware coverage drops below 95%, measured to two decimal places (`fail_under = 95`, `precision = 2` in `pyproject.toml`). It also fails on any warning pytest doesn't already know about (`filterwarnings = ["error", ...]` in `pyproject.toml`, with a small number of narrowly-scoped, individually-commented ignores for known pre-existing issues) and runs tests in a random order each time (`pytest-randomly`) to surface hidden ordering dependencies between tests.

GitHub Actions runs the following on every push and pull request, in parallel jobs so a slow check never blocks the others (see `.github/workflows/tests.yml`):

| Job | What it checks |
| --- | --- |
| `test` | Tests + coverage gate, matrixed across Python 3.11, 3.12, 3.13. |
| `lint` | `ruff check` and `ruff format --check`. |
| `lockfile` | `uv lock --check` -- fails if `uv.lock` is out of sync with `pyproject.toml`. |
| `dependency-audit` | `pip-audit` against the resolved dependency set -- fails on known CVEs in any dependency, direct or transitive. |
| `build-and-import` | Builds the wheel, installs *that artifact* into a clean virtualenv, and imports it -- catches packaging mistakes that `uv sync`'s editable install would never see. |

A separate weekly/manual-only workflow (`.github/workflows/mutation.yml`) runs mutation testing (`mutmut`) against the highest-risk modules (`auth.py`, `storage/sqlite.py`, `pipeline/embeddings.py`) to check whether the test suite would actually notice if their logic were broken, not just executed. It's report-only and scoped to these three files because a full-repo run is too slow for routine use.

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and trade-offs.
