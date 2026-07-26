# Utralight RAG MCP server architecture

## Overview

Utralight provides a local context index with two adapters: a resource-oriented FastAPI REST API and a FastMCP server. Both adapters call `RAGService`; they do not contain separate ingestion or retrieval implementations. The combined streamable-HTTP entry point mounts both adapters in one FastAPI process and injects the same service into each.

Each service process owns exactly one index and one SQLite database. The default database is in memory; set `RAG_DATABASE_PATH` to use one local SQLite file when a process restart should preserve that index. Independent indexes run as independent service instances rather than as collections inside one process. The design intentionally does not include background jobs, evaluation metrics, synthetic QA generation, or benchmark code.
REST handlers that call the synchronous service run as regular FastAPI handlers, so Starlette dispatches them to its threadpool. Multipart request parsing remains asynchronous, but embedding work is explicitly offloaded before it reaches the service. `SQLiteStore` serializes connection operations with a re-entrant lock; this keeps the single-connection design safe for concurrent threadpool requests. SQLite WAL mode and a busy timeout provide additional safety when a stdio MCP process happens to access the same file-backed index.


## Components

- `src/utralight_rag/api`: JSON and multipart REST routes for document CRUD and search.
- `src/utralight_rag/mcp_server`: FastMCP tools with direct mappings to the service operations.
- `src/utralight_rag/combined.py`: Combined REST and streamable HTTP MCP entry point with one injected service.
- `src/utralight_rag/storage`: SQLite schema and sqlite-vec virtual table. Documents and chunks are ordinary relational rows; vectors are stored in `vec_chunks`.
- `src/utralight_rag/pipeline`: `BaseChunker` and `BaseEmbedder` interfaces plus Chonkie, FastEmbed, legacy local, and OpenAI-compatible embedding implementations.
- `src/utralight_rag/service.py`: shared orchestration for chunking, embedding, CRUD, and vector search.

The vector table uses cosine distance so the returned `score` remains an interpretable similarity approximation. Metadata filters are applied after KNN retrieval in Python; the service deliberately scans enough candidates for a Utralight deployment, but large filtered indexes would need database-side filtering or metadata indexes.

## Architectural decisions

### ADR-001: Python 3.11+

Python 3.11 is the minimum supported version because it provides modern typing and good support across FastAPI, FastMCP, Chonkie, and sentence-transformers. The project is managed with `uv` and uses `pyproject.toml` plus `uv.lock` for reproducible environments.

Alternative: older Python versions would increase compatibility burden and are not needed for this service.

### ADR-002: SQLite and sqlite-vec

SQLite provides zero-service local storage and a single-file deployment option. sqlite-vec adds KNN vector search without an external database or operational infrastructure. Foreign keys and explicit vector-row deletion keep document deletion consistent with the vector index.

Alternative: Milvus, Qdrant, Pinecone, and similar services provide more scale but violate Utralight's minimal infrastructure goal.

### ADR-003: Chonkie for chunking

Chonkie is wrapped by `BaseChunker`, keeping the service independent of Chonkie's concrete API while providing recursive, sentence, and token strategies. Chunking strategy, size, and overlap are fixed by the service instance so all documents in one index have consistent retrieval behavior. Because Chonkie's recursive chunker has no overlap option, Utralight applies the configured overlap by extending each recursive chunk with trailing characters from its predecessor.

Alternative: custom splitting would be smaller initially but would duplicate boundary and tokenization behavior that Chonkie already provides.

### ADR-004: Pluggable embeddings

`BaseEmbedder` is the stable boundary for embedding providers. FastEmbed using `BAAI/bge-small-en-v1.5` is the default local provider and lazy-loads its model on first use. When no provider is configured, a non-empty `RAG_EMBEDDING_API_KEY` or `OPENAI_API_KEY` selects the OpenAI-compatible provider; otherwise FastEmbed is selected. External OpenAI-compatible providers require credentials and HTTPS, while explicit Ollama remains credential-optional, may use HTTP, and uses its `/v1/embeddings` endpoint and `nomic-embed-text` model. Sentence Transformers remains an optional legacy local provider. Ingestion sends chunks in bounded `RAG_EMBEDDING_BATCH_SIZE` batches, and provider responses must contain exactly one finite vector for each input index. The selected embedding configuration is fixed for the service instance and is used for both ingestion and query execution.

Alternative: separate Ollama settings would duplicate the same protocol configuration and make provider switching harder.

### ADR-005: Shared service layer

REST and MCP are transport adapters around exactly one `RAGService`. This avoids behavior drift and makes the core lifecycle easy to test without running either server.

Alternative: implementing logic separately in route and tool handlers would be simpler for a prototype but would make fixes inconsistent.

### ADR-006: Utralight storage lifecycle

The default `:memory:` database keeps local setup fast and dependency-free. File-backed SQLite is also supported when an index should survive process restarts or be shared with a separately spawned stdio MCP process. The combined streamable-HTTP entry point uses one in-process store for both adapters, avoiding duplicate embedding models. Migrations, replication, retention policies, and production-scale storage operations are intentionally out of scope.

### ADR-007: One index per service instance

A service process owns one SQLite index, one embedding configuration, and one chunking configuration. To run independent indexes, deploy independent service instances with separate database paths and ports. This keeps collection routing, mixed embedding spaces, and per-document configuration out of the core service.

This also makes index isolation explicit: one process owns one database file, and an index can be removed by stopping the instance and deleting its database file. The additional operational cost is limited to process supervision and port/configuration management, which is appropriate for a small number of Utralight indexes.

Alternative: collections inside one process would reduce the number of processes but would require collection-aware API and MCP contracts, routing, authorization, and configuration validation. That complexity is deferred unless the number of indexes makes process-per-index management impractical.

### ADR-008: Explicit runtime authorization modes

The default `none` mode keeps local development frictionless and grants non-browser callers full CRUD and search access. REST rejects cross-origin browser mutations even in this mode, preventing a malicious webpage from posting to a local instance. A trusted-host allowlist also rejects DNS-rebinding requests before routing, so the client-controlled Host header is never used as the security decision. `trusted-proxy` mode delegates authentication to a reverse proxy and maps its identity and role headers to exactly two roles: `admin` for all operations and `reader` for list/get/search only. Header names and role values must be non-empty, and the two configured roles must be distinct. The application rejects missing identities and unknown roles, but intentionally does not verify proxy credentials. Deployments must prevent direct access and strip client-supplied identity headers at the proxy boundary.

Alternative: embedding authentication in this service would duplicate the proxy or Cloudflare Access identity provider and add credential lifecycle concerns outside the service's scope.

### ADR-009: Stdio-first MCP transports

MCP uses stdio by default for local clients. Streamable HTTP is an explicit opt-in for networked clients and proxy deployments. Legacy network transport support is not exposed by this adapter.

Alternative: a network transport as the default would make local use less secure and less compatible with desktop MCP clients.

### ADR-010: Separate transport and document limits

The REST adapter enforces `RAG_MAX_REQUEST_BYTES` in ASGI middleware before request parsing, including chunked bodies, and returns 413 when the raw body is too large. `RAG_MAX_DOCUMENT_BYTES` remains an exact UTF-8 content limit checked after JSON or multipart parsing and by the service for both POST and PUT. The request limit defaults to the document limit plus overhead so valid documents are not rejected because of JSON or multipart framing.

Alternative: comparing Content-Length directly with the document limit is incorrect because request framing, titles, metadata, and multipart fields are not document content; parsing first also permits avoidable resource exhaustion.

## Running

```bash
uv sync

# One process serves REST and streamable HTTP MCP on one index.
RAG_DATABASE_PATH=/var/lib/rag/index.sqlite \
  uv run uvicorn utralight_rag.combined:app --host 127.0.0.1 --port 8001

# Independent indexes still use separate combined instances.
RAG_DATABASE_PATH=/var/lib/rag/index-a.sqlite uv run uvicorn utralight_rag.combined:app --port 8002
RAG_DATABASE_PATH=/var/lib/rag/index-b.sqlite uv run uvicorn utralight_rag.combined:app --port 8003

# Stdio remains client-spawned and separate from the combined entry point.
uv run python -m utralight_rag.mcp_server.server
uv run pytest
```

The MCP process uses stdio by default and supports streamable HTTP as the only network transport. FastEmbed is included for the default local provider; Sentence Transformers remains optional because its PyTorch runtime is large. Ollama uses the same OpenAI-compatible embedding settings as other compatible providers. Set `RAG_AUTH_MODE=trusted-proxy` only when a reverse proxy authenticates requests and overwrites the configured identity and role headers; otherwise the default `none` mode grants full access to every caller.
