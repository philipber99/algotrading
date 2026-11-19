"""REST routes for reusable futures positioning and participation analytics."""

from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Query, Request, Response

from ..futures_analytics_models import (
    FuturesAnalyticsResponse,
    FuturesCrowdingData,
    FuturesMarketHealthData,
    FuturesParticipationData,
    FuturesPositioningData,
    FuturesTermStructureData,
)


PositionWindow = Literal["1w", "4w", "13w"]
HistoryWindow = Literal["1y", "3y", "5y"]
ParticipationWindow = Literal["1d", "5d", "20d"]

router = APIRouter(
    prefix="/api/v1/analytics/futures",
    tags=["Futures Analytics"],
)


def futures_service(request: Request):
    return request.app.state.futures_analytics_service


def futures_response(active, result, response: Response):
    response.headers["X-Cache"] = result.cache.upper()
    response.headers["X-Data-As-Of"] = result.as_of.isoformat()
    response.headers["Cache-Control"] = (
        "private, max-age=5" if result.cache != "bypass" else "no-store"
    )
    return {
        "data": result.data,
        "meta": {
            "as_of": result.as_of,
            "timezone": active.timezone,
            "cache": result.cache,
            "warnings": list(result.warnings),
            "sql_query_count": result.query_count,
        },
    }


def _csv_values(values: list[str] | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return defaults
    # Accept repeated or comma-separated filters and preserve the first occurrence.
    return tuple(dict.fromkeys(
        item.strip().upper()
        for value in values
        for item in str(value).split(",")
        if item.strip()
    ))


@router.get("/positioning", response_model=FuturesAnalyticsResponse[FuturesPositioningData])
def positioning(
    request: Request,
    response: Response,
    products: list[str] | None = Query(None),
    participants: list[str] | None = Query(None),
    change_window: PositionWindow = "4w",
    history_window: HistoryWindow = "5y",
    as_of: date | None = None,
    include_series: bool = False,
):
    active = futures_service(request)
    result = active.positioning(
        products=_csv_values(products, ("ES", "NQ")),
        participants=_csv_values(participants, ("ASSET_MANAGER", "LEVERAGED_MONEY")),
        change_window=change_window,
        history_window=history_window,
        as_of=as_of,
        include_series=include_series,
    )
    return futures_response(active, result, response)


@router.get("/participation", response_model=FuturesAnalyticsResponse[FuturesParticipationData])
def participation(
    request: Request,
    response: Response,
    products: list[str] | None = Query(None),
    windows: list[str] | None = Query(None),
    as_of: date | None = None,
    include_series: bool = False,
):
    active = futures_service(request)
    result = active.participation(
        products=_csv_values(products, ("ES", "NQ")),
        windows=tuple(value.lower() for value in _csv_values(windows, ("1D", "5D", "20D"))),
        as_of=as_of,
        include_series=include_series,
    )
    return futures_response(active, result, response)


@router.get("/term-structure", response_model=FuturesAnalyticsResponse[FuturesTermStructureData])
def term_structure(
    request: Request,
    response: Response,
    product: str = Query("VX", min_length=1, max_length=16),
    as_of: date | None = None,
    months: int = Query(6, ge=1, le=12),
    compare_sessions: int = Query(5, ge=0, le=60),
):
    active = futures_service(request)
    result = active.term_structure(
        product=product, as_of=as_of, months=months,
        compare_sessions=compare_sessions,
    )
    return futures_response(active, result, response)


@router.get("/crowding", response_model=FuturesAnalyticsResponse[FuturesCrowdingData])
def crowding(
    request: Request,
    response: Response,
    products: list[str] | None = Query(None),
    participants: list[str] | None = Query(None),
    history_window: HistoryWindow = "5y",
    change_window: PositionWindow = "4w",
    as_of: date | None = None,
    limit: int = Query(10, ge=1, le=100),
):
    active = futures_service(request)
    result = active.crowding(
        products=_csv_values(products, ()) or None,
        participants=_csv_values(participants, ()) or None,
        history_window=history_window, change_window=change_window,
        as_of=as_of, limit=limit,
    )
    return futures_response(active, result, response)


@router.get(
    "/market-health/overview",
    response_model=FuturesAnalyticsResponse[FuturesMarketHealthData],
)
def market_health_overview(
    request: Request,
    response: Response,
    position_window: PositionWindow = "4w",
    history_window: HistoryWindow = "5y",
    as_of: date | None = None,
):
    active = futures_service(request)
    result = active.market_health_overview(
        position_window=position_window,
        history_window=history_window,
        as_of=as_of,
    )
    return futures_response(active, result, response)


__all__ = ["router"]
