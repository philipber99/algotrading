"""Thin REST router for reusable tracked-ETF Market Flow analytics."""

from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Query, Request, Response

from ..market_flow_models import (
    MarketFlowCategoriesData,
    MarketFlowETFDetailData,
    MarketFlowETFListData,
    MarketFlowOverviewData,
    MarketFlowResponse,
    MarketFlowRotationsData,
)


Window = Literal["1d", "1w", "1m", "3m", "6m"]
ExposureType = Literal[
    "BROAD_MARKET", "GICS_SECTOR", "INDUSTRY", "TREASURY_DURATION",
    "INFLATION_LINKED", "INVESTMENT_GRADE_CREDIT", "HIGH_YIELD_CREDIT",
]

router = APIRouter(
    prefix="/api/v1/analytics/market-flows",
    tags=["Market Flow Analytics"],
)


def market_flow_service(request: Request):
    return request.app.state.market_flow_service


def market_flow_response(active, result, response: Response):
    # Carry the shared analytics cache and observation time into the HTTP contract.
    response.headers["X-Cache"] = result.cache.upper()
    response.headers["X-Data-As-Of"] = result.as_of.isoformat()
    response.headers["Cache-Control"] = "private, max-age=5" if result.cache != "bypass" else "no-store"
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


@router.get("/overview", response_model=MarketFlowResponse[MarketFlowOverviewData])
def overview(
    request: Request,
    response: Response,
    window: Window = "1d",
    as_of: date | None = None,
):
    active = market_flow_service(request)
    return market_flow_response(active, active.overview(window=window, as_of=as_of), response)


@router.get("/categories", response_model=MarketFlowResponse[MarketFlowCategoriesData])
def categories(
    request: Request,
    response: Response,
    window: Window = "1d",
    as_of: date | None = None,
    view: Literal["all", "mandate", "economic"] = "all",
    level: str | None = Query(None, min_length=1, max_length=64),
    parent_id: str | None = Query(None, min_length=1, max_length=256),
    node_id: str | None = Query(None, min_length=1, max_length=256),
    include_constituents: bool = False,
    include_flow_series: bool = False,
    limit: int = Query(200, ge=1, le=500),
):
    active = market_flow_service(request)
    result = active.categories(
        window=window, as_of=as_of, view=view, level=level, parent_id=parent_id,
        node_id=node_id, include_constituents=include_constituents,
        include_flow_series=include_flow_series, limit=limit,
    )
    return market_flow_response(active, result, response)


@router.get("/rotations", response_model=MarketFlowResponse[MarketFlowRotationsData])
def rotations(
    request: Request,
    response: Response,
    window: Window = "1d",
    as_of: date | None = None,
    dimension: Literal["sector", "industry", "capitalization", "fixed_income"] = "sector",
    limit: int = Query(25, ge=1, le=100),
):
    active = market_flow_service(request)
    result = active.rotations(
        window=window, as_of=as_of, dimension=dimension, limit=limit,
    )
    return market_flow_response(active, result, response)


@router.get("/etfs", response_model=MarketFlowResponse[MarketFlowETFListData])
def etfs(
    request: Request,
    response: Response,
    asset_class: Literal["EQUITY", "FIXED_INCOME"] | None = None,
    exposure_type: ExposureType | None = None,
    as_of: date | None = None,
):
    active = market_flow_service(request)
    result = active.etfs(
        asset_class=asset_class, exposure_type=exposure_type, as_of=as_of,
    )
    return market_flow_response(active, result, response)


@router.get("/etfs/{symbol}", response_model=MarketFlowResponse[MarketFlowETFDetailData])
def etf_detail(
    symbol: str,
    request: Request,
    response: Response,
    window: Window = "1d",
    as_of: date | None = None,
):
    active = market_flow_service(request)
    result = active.etf_detail(symbol, window=window, as_of=as_of)
    return market_flow_response(active, result, response)


__all__ = ["router"]
