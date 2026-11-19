"""Public response models used to generate the OpenAPI contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ResponseMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    as_of: datetime
    timezone: str
    cache: Literal["hit", "miss", "bypass"]


class CollectionResponse(BaseModel):
    data: list[dict[str, Any]]
    meta: ResponseMeta


class ObjectResponse(BaseModel):
    data: dict[str, Any]
    meta: ResponseMeta


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    service: str
    version: str
    details: dict[str, Any] | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
