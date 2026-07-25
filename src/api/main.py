"""FastAPI application exposing resource-oriented document endpoints."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from ..service import RAGService
from ..storage.sqlite import DocumentNotFoundError
from .models import DocumentPayload, SearchPayload


def create_app(service: RAGService | None = None) -> FastAPI:
    rag = service or RAGService()
    app = FastAPI(title="Transient RAG MCP", version="0.1.0")
    app.state.rag = rag

    @app.post("/documents", status_code=201)
    async def create_document(request: Request) -> dict[str, Any]:
        try:
            payload = await _read_document_request(request)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return rag.ingest(**payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/documents")
    async def list_documents() -> list[dict[str, Any]]:
        return rag.list_documents()

    @app.get("/documents/{document_id}")
    async def get_document(document_id: str) -> dict[str, Any]:
        try:
            return rag.get_document(document_id)
        except DocumentNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")

    @app.put("/documents/{document_id}")
    async def update_document(document_id: str, payload: DocumentPayload) -> dict[str, Any]:
        try:
            return rag.update(document_id, **payload.model_dump())
        except DocumentNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/documents/{document_id}", status_code=204)
    async def delete_document(document_id: str) -> None:
        try:
            rag.delete_document(document_id)
        except DocumentNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")

    @app.post("/search")
    async def search(payload: SearchPayload) -> list[dict[str, Any]]:
        try:
            return rag.search(payload.query, payload.top_k, payload.filter_metadata)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


async def _read_document_request(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        return DocumentPayload.model_validate(await request.json()).model_dump()
    form = await request.form()
    upload = form.get("file")
    if isinstance(upload, UploadFile):
        content = (await upload.read()).decode("utf-8")
        title = str(form.get("title") or upload.filename or "untitled")
    else:
        content = str(form.get("content") or "")
        title = str(form.get("title") or "untitled")
    metadata: Any = form.get("metadata", {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata) if metadata else {}

    payload = {"title": title, "content": content, "metadata": metadata}
    forbidden_options = {
        name: form.get(name)
        for name in ("chunking_strategy", "chunk_size", "chunk_overlap", "embedding_choice")
        if form.get(name) not in (None, "")
    }
    if forbidden_options:
        raise ValueError("Per-document embedding and chunking overrides are not supported")
    return DocumentPayload.model_validate(payload).model_dump()


app = create_app()
