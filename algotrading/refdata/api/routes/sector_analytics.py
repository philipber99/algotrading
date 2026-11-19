from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Query, Request, Response

from ..analytics_models import (
    AnalyticsResponse,
    SectorConstituentsData,
    SectorPerformanceData,
    SectorRelationshipsData,
    SectorSummaryData,
)
from .analytics_common import analytics_response, analytics_service


router = APIRouter(prefix="/api/v1/analytics", tags=["Sector Analytics"])


@router.get("/sectors", response_model=AnalyticsResponse[SectorSummaryData])
def sectors(
    request: Request,
    response: Response,
    universe: Literal["SP100", "SP500", "NASDAQ100"] = "SP100",
    grouping: Literal["sector", "industry"] = "sector",
    period: Literal["today", "1d", "1w", "1m", "3m", "6m", "1y"] = "1m",
    weighting: Literal["cap_weighted", "equal_weighted"] = "cap_weighted",
    include_position_scoreboard: bool = False,
):
    active = analytics_service(request)
    return analytics_response(active, active.sectors(universe=universe, grouping=grouping, period=period, weighting=weighting, include_position_scoreboard=include_position_scoreboard), response)


@router.get("/sectors/{classification_id}/constituents", response_model=AnalyticsResponse[SectorConstituentsData])
def constituents(
    classification_id: str,
    request: Request,
    response: Response,
    universe: Literal["SP100", "SP500", "NASDAQ100"] = "SP100",
    as_of: date | None = None,
    benchmark: Literal["SPY", "QQQ"] = "SPY",
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    sort: Literal["symbol", "company_name", "current_price", "return_1d", "return_3m", "return_1y"] = "symbol",
    direction: Literal["asc", "desc"] = "asc",
):
    active = analytics_service(request)
    result = active.constituents(classification_id, universe=universe, as_of=as_of, benchmark=benchmark, page=page, page_size=page_size, sort=sort, direction=direction)
    return analytics_response(active, result, response)


@router.get("/sector-performance", response_model=AnalyticsResponse[SectorPerformanceData])
def sector_performance(
    request: Request,
    response: Response,
    classification_id: list[str] = Query(...),
    universe: Literal["SP100", "SP500", "NASDAQ100"] = "SP100",
    grouping: Literal["sector", "industry"] = "sector",
    range_name: Literal["intraday", "1m", "3m", "6m", "1y", "3y"] = Query("3m", alias="range"),
    weighting: Literal["cap_weighted", "equal_weighted"] = "cap_weighted",
):
    identifiers = tuple(item for value in classification_id for item in value.split(",") if item)
    active = analytics_service(request)
    result = active.sector_performance(identifiers, universe=universe, grouping=grouping, range_name=range_name, weighting=weighting)
    return analytics_response(active, result, response)


@router.get("/sector-relationships", response_model=AnalyticsResponse[SectorRelationshipsData])
def sector_relationships(
    request: Request,
    response: Response,
    universe: Literal["SP100", "SP500", "NASDAQ100"] = "SP100",
    grouping: Literal["sector", "industry"] = "sector",
    period: Literal["1m", "3m", "6m", "1y"] = "3m",
    weighting: Literal["cap_weighted", "equal_weighted"] = "cap_weighted",
    method: Literal["pearson", "spearman"] = "pearson",
    left_id: str | None = None,
    right_id: str | None = None,
):
    active = analytics_service(request)
    result = active.sector_relationships(universe=universe, grouping=grouping, period=period, weighting=weighting, method=method, left_id=left_id, right_id=right_id)
    return analytics_response(active, result, response)
