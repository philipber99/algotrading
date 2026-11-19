"""SQL-only economic-release route."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query, Request, Response

from ..economic_release_models import EconomicReleaseMeta, EconomicReleaseResponse
from ..models import ErrorResponse


router = APIRouter(tags=["economic releases"])


@router.get(
    "/api/v1/economic-releases",
    response_model=EconomicReleaseResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def economic_releases(
    request: Request,
    response: Response,
    view: Literal["upcoming", "today", "prior", "range"] = "upcoming",
    start: str | None = None,
    end: str | None = None,
    internal_name: list[str] | None = Query(default=None),
    impact: list[str] | None = Query(default=None),
    actual_status: Literal["all", "available", "pending"] = "all",
    limit: int = Query(default=250, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    timezone: str | None = None,
) -> EconomicReleaseResponse:
    # Query the durable release store without triggering provider refreshes.
    page = request.app.state.economic_release_service.query(
        view=view,
        start=start,
        end=end,
        internal_names=tuple(internal_name or ()),
        impacts=tuple(impact or ()),
        actual_status=actual_status,
        limit=limit,
        offset=offset,
        timezone_name=timezone,
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Cache"] = "BYPASS"
    response.headers["X-Data-As-Of"] = page.as_of.isoformat()
    return EconomicReleaseResponse(
        items=list(page.items),
        meta=EconomicReleaseMeta(
            count=len(page.items),
            total=page.total,
            limit=page.limit,
            offset=page.offset,
            as_of=page.as_of,
            timezone=page.timezone,
            view=page.view,
            range_start=page.range_start,
            range_end=page.range_end,
            warnings=list(page.warnings),
        ),
    )
