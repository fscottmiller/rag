# Transient RAG MCP server architecture

## Overview

This project provides a short-lived local context index with two adapters: a resource-oriented FastAPI REST API and a FastMCP server. Both adapters call `RAGService`; they do not contain separate ingestion or retrieval implementations.

Each service process owns exactly one index and one SQLite database. The default database is in memory; set `RAG_DATABASE_PATH` to use one local SQLite file when a process restart should preserve that index. Independent indexes run as independent service instances rather than as collections inside one process. The design intentionally does not include background jobs, evaluation metrics, synthetic QA generation, or benchmark code.

## Components

- `src/api`: JSON and multipart REST routes for document CRUD and search.
- `src/mcp_server`: FastMCP tools with direct mappings to the service operations.
- `src/storage`: SQLite schema and sqlite-vec virtual table. Documents and chunks are ordinary relational rows; vectors are stored in `vec_chunks`.
- `src/pipeline`: `BaseChunker` and `BaseEmbedder` interfaces plus Chonkie, local, and OpenAI-compatible embedding implementations.
- `src/service.py`: shared orchestration for chunking, embedding, CRUD, and vector search.

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

`BaseEmbedder` is the stable boundary for embedding providers. Sentence Transformers using `all-MiniLM-L6-v2` is the default local provider. An OpenAI-compatible provider can be configured with an endpoint, model, optional dimensions, and API key through environment variables; Ollama remains an optional local provider. The selected embedding configuration is fixed for the service instance and is used for both ingestion and query execution.

Alternative: a hosted embedding API adds latency, credentials, and external availability requirements, but the OpenAI-compatible interface keeps the integration lightweight and works with multiple hosted or self-hosted compatible providers.

### ADR-005: Shared service layer

REST and MCP are transport adapters around exactly one `RAGService`. This avoids behavior drift and makes the core lifecycle easy to test without running either server.

Alternative: implementing logic separately in route and tool handlers would be simpler for a prototype but would make fixes inconsistent.

### ADR-006: Transient lifecycle

The default `:memory:` database reflects the primary use case: fast contextual recall for one process. File-backed SQLite is supported as an opt-in convenience, but migrations, replication, retention policies, and long-term storage concerns are intentionally out of scope.

### ADR-007: One index per service instance

A service process owns one SQLite index, one embedding configuration, and one chunking configuration. To run independent indexes, deploy independent service instances with separate database paths and ports. This keeps collection routing, mixed embedding spaces, and per-document configuration out of the core service.

This also makes index isolation explicit: one process owns one database file, and an index can be discarded by stopping the instance and removing its transient database. The additional operational cost is limited to process supervision and port/configuration management, which is appropriate for the expected small number of lightweight indexes.

Alternative: collections inside one process would reduce the number of processes but would require collection-aware API and MCP contracts, routing, authorization, and configuration validation. That complexity is deferred unless the number of indexes makes process-per-index management impractical.

## Running

```bash
uv sync
# Install the default local model provider when using sentence-transformers:
uv sync --extra local-embeddings

# Terminal 1: one instance owns one index.
RAG_DATABASE_PATH=/var/lib/rag/index-a.sqlite uv run uvicorn src.api.main:app --port 8001
# Terminal 2:
RAG_DATABASE_PATH=/var/lib/rag/index-b.sqlite uv run uvicorn src.api.main:app --port 8002

uv run python -m src.mcp_server.server
MCP_TRANSPORT=sse uv run python -m src.mcp_server.server
uv run pytest
```

The MCP process uses FastMCP's stdio transport by default. Set `MCP_TRANSPORT=sse` for SSE transport; the transport choice does not change service logic. Sentence Transformers is an optional dependency because its PyTorch runtime is large; Ollama remains available as an optional provider for deployments that already run a local Ollama model.
