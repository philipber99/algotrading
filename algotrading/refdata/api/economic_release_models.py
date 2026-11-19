"""Typed HTTP contract for economic releases."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EconomicReleaseItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    internal_name: str | None = Field(default=None, max_length=40)
    report: str = Field(max_length=70)
    period: str = Field(max_length=20)
    previous: float | None = None
    estimate: float | None = None
    actual: float | None = None
    impact: str | None = Field(default=None, max_length=10)
    status: Literal["released", "due_pending", "scheduled"]


class EconomicReleaseMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=1000)
    offset: int = Field(ge=0)
    as_of: datetime
    timezone: str
    view: Literal["upcoming", "today", "prior", "range"]
    range_start: datetime
    range_end: datetime
    source: Literal["sql"] = "sql"
    warnings: list[str]


class EconomicReleaseResponse(BaseModel):
    items: list[EconomicReleaseItem]
    meta: EconomicReleaseMeta
