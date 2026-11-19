from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Query, Request, Response

from ..analytics_models import (
    AnalyticsResponse,
    CompanyAnalyticsData,
    FINRAShortInterestData,
    FINRAShortVolumeData,
)
from .analytics_common import analytics_response, analytics_service


router = APIRouter(prefix="/api/v1/symbols", tags=["Company Analytics"])
finra_router = APIRouter(prefix="/api/v1/finra", tags=["FINRA Reference Data"])


@finra_router.get(
    "/short-volume",
    response_model=AnalyticsResponse[FINRAShortVolumeData],
)
def finra_short_volume(
    request: Request,
    response: Response,
    symbols: list[str] = Query(...),
    as_of: date | None = None,
    session_limit: int = Query(21, ge=1, le=252),
):
    """Return daily consolidated FINRA history for up to 500 symbols."""
    requested = tuple(
        symbol.strip()
        for value in symbols
        for symbol in value.split(",")
        if symbol.strip()
    )
    active = analytics_service(request)
    return analytics_response(
        active,
        active.finra_short_volume(
            requested, as_of=as_of, session_limit=session_limit
        ),
        response,
    )


@router.get(
    "/{symbol}/short-interest",
    response_model=AnalyticsResponse[FINRAShortInterestData],
)
def company_short_interest(
    symbol: str,
    request: Request,
    response: Response,
    as_of: date | None = None,
    report_limit: int = Query(130, ge=1, le=130),
):
    active = analytics_service(request)
    return analytics_response(
        active,
        active.company_short_interest(
            symbol, as_of=as_of, report_limit=report_limit
        ),
        response,
    )

@router.get(
    "/{symbol}/analytics",
    response_model=AnalyticsResponse[CompanyAnalyticsData],
)
def company_analytics(
    symbol: str,
    request: Request,
    response: Response,
    benchmark: Literal["SPY", "QQQ"] = "SPY",
    range_name: Literal["1d", "1m", "3m", "6m", "1y", "5y"] = Query("1y", alias="range"),
    as_of: date | None = None,
):
    active = analytics_service(request)
    return analytics_response(active, active.company(symbol, benchmark=benchmark, range_name=range_name, as_of=as_of), response)
