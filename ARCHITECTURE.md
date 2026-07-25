# Transient RAG MCP server architecture

## Overview

This project provides a short-lived local context index with two adapters: a resource-oriented FastAPI REST API and a FastMCP server. Both adapters call `RAGService`; they do not contain separate ingestion or retrieval implementations.

The default database is an in-memory SQLite database. Set `RAG_DATABASE_PATH` to use one local SQLite file when a process restart should preserve the index. The design intentionally does not include background jobs, evaluation metrics, synthetic QA generation, or benchmark code.

## Components

- `src/api`: JSON and multipart REST routes for document CRUD and search.
- `src/mcp_server`: FastMCP tools with direct mappings to the service operations.
- `src/storage`: SQLite schema and sqlite-vec virtual table. Documents and chunks are ordinary relational rows; vectors are stored in `vec_chunks`.
- `src/pipeline`: `BaseChunker` and `BaseEmbedder` interfaces plus Chonkie and local embedding implementations.
- `src/service.py`: shared orchestration for chunking, embedding, CRUD, and vector search.

## Architectural decisions

### ADR-001: Python 3.11+

Python 3.11 is the minimum supported version because it provides modern typing and good support across FastAPI, FastMCP, Chonkie, and sentence-transformers. The project is managed with `uv` and uses `pyproject.toml` plus `uv.lock` for reproducible environments.

Alternative: older Python versions would increase compatibility burden and are not needed for this service.

### ADR-002: SQLite and sqlite-vec

SQLite provides zero-service local storage and a single-file deployment option. sqlite-vec adds KNN vector search without an external database or operational infrastructure. Foreign keys and explicit vector-row deletion keep document deletion consistent with the vector index.

Alternative: Milvus, Qdrant, Pinecone, and similar services provide more scale but violate the minimal, transient infrastructure goal.

### ADR-003: Chonkie for chunking

Chonkie is wrapped by `BaseChunker`, keeping the service independent of Chonkie's concrete API while providing recursive, sentence, and token strategies. Chunk size and overlap are configurable per request.

Alternative: custom splitting would be smaller initially but would duplicate boundary and tokenization behavior that Chonkie already provides.

### ADR-004: Pluggable embeddings

`BaseEmbedder` is the stable boundary for embedding providers. Sentence Transformers using `all-MiniLM-L6-v2` is the default local provider; Ollama is available for deployments that already run a local Ollama model. Embeddings are computed only during ingestion and query execution, with no remote model service required by the default.

Alternative: a hosted embedding API would add latency, credentials, and external availability requirements.

### ADR-005: Shared service layer

REST and MCP are transport adapters around exactly one `RAGService`. This avoids behavior drift and makes the core lifecycle easy to test without running either server.

Alternative: implementing logic separately in route and tool handlers would be simpler for a prototype but would make fixes inconsistent.

### ADR-006: Transient lifecycle

The default `:memory:` database reflects the primary use case: fast contextual recall for one process. File-backed SQLite is supported as an opt-in convenience, but migrations, replication, retention policies, and long-term storage concerns are intentionally out of scope.

## Running

```bash
uv sync
# Install the default local model provider when using sentence-transformers:
uv sync --extra local-embeddings
uv run uvicorn src.api.main:app --reload
uv run python -m src.mcp_server.server
MCP_TRANSPORT=sse uv run python -m src.mcp_server.server
uv run pytest
```

The MCP process uses FastMCP's stdio transport by default. Set `MCP_TRANSPORT=sse` for SSE transport; the transport choice does not change service logic. Sentence Transformers is an optional dependency because its PyTorch runtime is large; Ollama can be selected with `RAG_EMBEDDING_PROVIDER=ollama` when a local Ollama model is available.
