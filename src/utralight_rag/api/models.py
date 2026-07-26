from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100, strict=True)
    filter_metadata: dict[str, Any] | None = None
