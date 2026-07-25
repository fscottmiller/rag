from typing import Any

from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunking_strategy: str | None = None
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)
    embedding_choice: str | None = None


class SearchPayload(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    filter_metadata: dict[str, Any] | None = None
