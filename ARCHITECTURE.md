# Transient RAG MCP server architecture

## Overview

This project provides a short-lived local context index with two adapters: a resource-oriented FastAPI REST API and a FastMCP server. Both adapters call `RAGService`; they do not contain separate ingestion or retrieval implementations.

Each service process owns exactly one index and one SQLite database. The default database is in memory; set `RAG_DATABASE_PATH` to use one local SQLite file when a process restart should preserve that index. Independent indexes run as independent service instances rather than as collections inside one process. The design intentionally does not include background jobs, evaluation metrics, synthetic QA generation, or benchmark code.
REST handlers that call the synchronous service run as regular FastAPI handlers, so Starlette dispatches them to its threadpool. Multipart request parsing remains asynchronous, but embedding work is explicitly offloaded before it reaches the service. `SQLiteStore` serializes connection operations with a re-entrant lock; this keeps the single-connection design safe for concurrent threadpool requests. This is a deliberate transient-scale ceiling rather than a substitute for a multi-process database layer.


## Components

- `src/transient_rag/api`: JSON and multipart REST routes for document CRUD and search.
- `src/transient_rag/mcp_server`: FastMCP tools with direct mappings to the service operations.
- `src/transient_rag/storage`: SQLite schema and sqlite-vec virtual table. Documents and chunks are ordinary relational rows; vectors are stored in `vec_chunks`.
- `src/transient_rag/pipeline`: `BaseChunker` and `BaseEmbedder` interfaces plus Chonkie, local, and OpenAI-compatible embedding implementations.
- `src/transient_rag/service.py`: shared orchestration for chunking, embedding, CRUD, and vector search.

The vector table uses cosine distance so the returned `score` remains an interpretable similarity approximation. Metadata filters are applied after KNN retrieval in Python; the service deliberately scans enough candidates for its transient scale, but large filtered indexes would need database-side filtering or metadata indexes.

## Architectural decisions

### ADR-001: Python 3.11+

Python 3.11 is the minimum supported version because it provides modern typing and good support across FastAPI, FastMCP, Chonkie, and sentence-transformers. The project is managed with `uv` and uses `pyproject.toml` plus `uv.lock` for reproducible environments.

Alternative: older Python versions would increase compatibility burden and are not needed for this service.

### ADR-002: SQLite and sqlite-vec

SQLite provides zero-service local storage and a single-file deployment option. sqlite-vec adds KNN vector search without an external database or operational infrastructure. Foreign keys and explicit vector-row deletion keep document deletion consistent with the vector index.

Alternative: Milvus, Qdrant, Pinecone, and similar services provide more scale but violate the minimal, transient infrastructure goal.

### ADR-003: Chonkie for chunking

Chonkie is wrapped by `BaseChunker`, keeping the service independent of Chonkie's concrete API while providing recursive, sentence, and token strategies. Chunking strategy, size, and overlap are fixed by the service instance so all documents in one index have consistent retrieval behavior.

Alternative: custom splitting would be smaller initially but would duplicate boundary and tokenization behavior that Chonkie already provides.

### ADR-004: Pluggable embeddings

`BaseEmbedder` is the stable boundary for embedding providers. Sentence Transformers using `all-MiniLM-L6-v2` is the default local provider. OpenAI-compatible providers, including Ollama, use one endpoint/model/API key/timeout/dimensions configuration through `RAG_EMBEDDING_*`; Ollama defaults to its `/v1/embeddings` endpoint and `nomic-embed-text` model. The selected embedding configuration is fixed for the service instance and is used for both ingestion and query execution.

Alternative: separate Ollama settings would duplicate the same protocol configuration and make provider switching harder.

### ADR-005: Shared service layer

REST and MCP are transport adapters around exactly one `RAGService`. This avoids behavior drift and makes the core lifecycle easy to test without running either server.

Alternative: implementing logic separately in route and tool handlers would be simpler for a prototype but would make fixes inconsistent.

### ADR-006: Transient lifecycle

The default `:memory:` database reflects the primary use case: fast contextual recall for one process. File-backed SQLite is supported as an opt-in convenience, but migrations, replication, retention policies, and long-term storage concerns are intentionally out of scope.

### ADR-007: One index per service instance

A service process owns one SQLite index, one embedding configuration, and one chunking configuration. To run independent indexes, deploy independent service instances with separate database paths and ports. This keeps collection routing, mixed embedding spaces, and per-document configuration out of the core service.

This also makes index isolation explicit: one process owns one database file, and an index can be discarded by stopping the instance and removing its transient database. The additional operational cost is limited to process supervision and port/configuration management, which is appropriate for the expected small number of lightweight indexes.

Alternative: collections inside one process would reduce the number of processes but would require collection-aware API and MCP contracts, routing, authorization, and configuration validation. That complexity is deferred unless the number of indexes makes process-per-index management impractical.

### ADR-008: Explicit runtime authorization modes

The default `none` mode keeps local development frictionless and grants every caller full CRUD and search access. `trusted-proxy` mode delegates authentication to a reverse proxy and maps its identity and role headers to exactly two roles: `admin` for all operations and `reader` for list/get/search only. The application rejects missing identities and unknown roles, but intentionally does not verify proxy credentials. Deployments must prevent direct access and strip client-supplied identity headers at the proxy boundary.

Alternative: embedding authentication in this service would duplicate the proxy or Cloudflare Access identity provider and add credential lifecycle concerns outside the service's scope.

### ADR-009: Stdio-first MCP transports

MCP uses stdio by default for local clients. Streamable HTTP is an explicit opt-in for networked clients and proxy deployments. Legacy network transport support is not exposed by this adapter.

Alternative: a network transport as the default would make local use less secure and less compatible with desktop MCP clients.

## Running

```bash
uv sync
# Install the default local model provider when using sentence-transformers:
uv sync --extra local-embeddings

# Terminal 1: one instance owns one index.
RAG_DATABASE_PATH=/var/lib/rag/index-a.sqlite uv run uvicorn transient_rag.api.main:app --port 8001
# Terminal 2:
RAG_DATABASE_PATH=/var/lib/rag/index-b.sqlite uv run uvicorn transient_rag.api.main:app --port 8002

uv run python -m transient_rag.mcp_server.server
MCP_TRANSPORT=streamable-http MCP_HOST=127.0.0.1 MCP_PORT=8000 MCP_PATH=/mcp \
  uv run python -m transient_rag.mcp_server.server
uv run pytest
```

The MCP process uses stdio by default and supports streamable HTTP as the only network transport. Sentence Transformers is an optional dependency because its PyTorch runtime is large. Ollama uses the same OpenAI-compatible embedding settings as other compatible providers. Set `RAG_AUTH_MODE=trusted-proxy` only when a reverse proxy authenticates requests and overwrites the configured identity and role headers; otherwise the default `none` mode grants full access to every caller.
