"""FastAPI application exposing resource-oriented document endpoints."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import Lifespan

from ..auth import AuthenticationError, AuthorizationError, Authorizer
from ..service import DocumentTooLargeError, RAGService
from ..storage.sqlite import DocumentNotFoundError
from .models import DocumentPayload, SearchPayload


def _same_origin(origin: str, request: Request) -> bool:
    parsed = urlsplit(origin)
    if parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
        return False
    try:
        origin_port = parsed.port or {"http": 80, "https": 443}.get(parsed.scheme)
    except ValueError:
        return False
    request_port = request.url.port or {"http": 80, "https": 443}.get(request.url.scheme)
    return (parsed.scheme, parsed.hostname, origin_port) == (
        request.url.scheme,
        request.url.hostname,
        request_port,
    )


class BodySizeLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (value for key, value in scope["headers"] if key.lower() == b"content-length"), None
        )
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send)
                return

        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await self._reject(send)
                return
            more_body = message.get("more_body", False)

        sent = False

        async def replay() -> dict[str, Any]:
            nonlocal sent
            if sent:
                return await receive()
            sent = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)

    async def _reject(self, send: Any) -> None:
        body = b"request body exceeds configured limit"
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": body})


def create_app(
    service: RAGService | None = None, lifespan: Lifespan[FastAPI] | None = None
) -> FastAPI:
    rag = service or RAGService()
    authorizer = Authorizer(rag.settings)
    app = FastAPI(title="Utralight RAG MCP", version="0.1.0", lifespan=lifespan)
    app.state.rag = rag
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=rag.settings.max_request_bytes)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(rag.settings.trusted_hosts))

    def require(request: Request, action: str) -> None:
        if action == "write" and authorizer.mode == "none":
            origin = request.headers.get("origin")
            if origin:
                if not _same_origin(origin, request):
                    raise HTTPException(
                        status_code=403,
                        detail="Cross-origin document mutations are not allowed",
                    )
        try:
            authorizer.authorize(request.headers, action)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except AuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/documents", status_code=201)
    async def create_document(request: Request) -> dict[str, Any]:
        require(request, "write")
        try:
            payload = await _read_document_request(request)
        except DocumentTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return await run_in_threadpool(rag.ingest, **payload)
        except DocumentTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/documents")
    def list_documents(request: Request) -> list[dict[str, Any]]:
        require(request, "read")
        return rag.list_documents()

    @app.get("/documents/{document_id}")
    def get_document(request: Request, document_id: str) -> dict[str, Any]:
        require(request, "read")
        try:
            return rag.get_document(document_id)
        except DocumentNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")

    @app.put("/documents/{document_id}")
    def update_document(
        request: Request, document_id: str, payload: DocumentPayload
    ) -> dict[str, Any]:
        require(request, "write")
        try:
            return rag.update(document_id, **payload.model_dump())
        except DocumentNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
        except DocumentTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/documents/{document_id}", status_code=204)
    def delete_document(request: Request, document_id: str) -> None:
        require(request, "write")
        try:
            rag.delete_document(document_id)
        except DocumentNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")

    @app.post("/search")
    def search(request: Request, payload: SearchPayload) -> list[dict[str, Any]]:
        require(request, "read")
        try:
            return rag.search(payload.query, payload.top_k, payload.filter_metadata)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


async def _read_document_request(request: Request) -> dict[str, Any]:
    max_bytes = request.app.state.rag.settings.max_document_bytes
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        payload = DocumentPayload.model_validate(await request.json()).model_dump()
        if len(payload["content"].encode("utf-8")) > max_bytes:
            raise DocumentTooLargeError(f"document content exceeds {max_bytes} bytes")
        return payload
    form = await request.form()
    upload = form.get("file")
    if isinstance(upload, UploadFile):
        content_bytes = await upload.read(max_bytes + 1)
        if len(content_bytes) > max_bytes:
            raise DocumentTooLargeError(f"document content exceeds {max_bytes} bytes")
        content = content_bytes.decode("utf-8")
        title = str(form.get("title") or upload.filename or "untitled")
    else:
        content = str(form.get("content") or "")
        if len(content.encode("utf-8")) > max_bytes:
            raise DocumentTooLargeError(f"document content exceeds {max_bytes} bytes")
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
