from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query, Request, Response

from ..analytics_models import AnalyticsResponse, MarketOverviewData
from .analytics_common import analytics_response, analytics_service


router = APIRouter(prefix="/api/v1/analytics", tags=["Market Analytics"])


@router.get("/market-overview", response_model=AnalyticsResponse[MarketOverviewData])
def market_overview(
    request: Request,
    response: Response,
    universe: Literal["SP100", "SP500", "NASDAQ100"] = "SP100",
    benchmark: Literal["SPY", "QQQ"] = "SPY",
    anchor: Literal["open", "prior_close", "monday", "last_week"] = "prior_close",
    mover_period: Literal["today", "1w", "1m", "3m"] = "today",
    mover_limit: int = Query(10, ge=1, le=50),
    weighting: Literal["cap_weighted", "equal_weighted"] = "cap_weighted",
):
    active = analytics_service(request)
    result = active.market_overview(
        universe=universe, benchmark=benchmark, anchor=anchor,
        mover_period=mover_period, mover_limit=mover_limit, weighting=weighting,
    )
    return analytics_response(active, result, response)
