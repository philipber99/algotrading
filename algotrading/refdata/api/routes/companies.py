from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Query, Request, Response

from ..stock_models import (
    CompanyInformationItem,
    EstimateItem,
    FinancialItem,
    GuidanceItem,
    OverviewItem,
    ProfileItem,
    StockCollectionResponse,
    StockObjectResponse,
)
from .common import collection_response, object_response, service

router = APIRouter(prefix="/api/v1/symbols", tags=["Stock Analysis"])


@router.get("/{symbol}/profile", response_model=StockObjectResponse[ProfileItem])
def profile(symbol: str, request: Request, response: Response):
    active = service(request)
    return object_response(active, active.profile(symbol), response)


@router.get("/{symbol}/company-information", response_model=StockObjectResponse[CompanyInformationItem])
def company_information(symbol: str, request: Request, response: Response):
    active = service(request)
    return object_response(active, active.company_information(symbol), response)


@router.get("/{symbol}/fundamentals", response_model=StockCollectionResponse[FinancialItem])
def fundamentals(
    symbol: str,
    request: Request,
    response: Response,
    period_type: Literal["annual", "quarterly"] = "annual",
    limit: int | None = Query(None, ge=1, le=40),
    as_of: datetime | None = None,
    revision_mode: Literal["latest", "all"] = "latest",
    refresh: bool = True,
):
    active = service(request)
    # Use comparable default history spans for annual and quarterly fundamentals.
    resolved_limit = limit if limit is not None else (5 if period_type == "annual" else 8)
    result = active.fundamentals(
        symbol,
        period_type=period_type,
        limit=resolved_limit,
        as_of=as_of,
        revision_mode=revision_mode,
        refresh=refresh,
    )
    return collection_response(active, result, response)


@router.get("/{symbol}/estimates", response_model=StockCollectionResponse[EstimateItem])
def estimates(symbol: str, request: Request, response: Response):
    active = service(request)
    return collection_response(active, active.estimates(symbol), response)


@router.get("/{symbol}/guidance", response_model=StockCollectionResponse[GuidanceItem])
def guidance(
    symbol: str,
    request: Request,
    response: Response,
    include_inactive: bool = False,
):
    active = service(request)
    return collection_response(
        active, active.guidance(symbol, include_inactive=include_inactive), response
    )


@router.get("/{symbol}/overview", response_model=StockObjectResponse[OverviewItem])
def overview(
    symbol: str,
    request: Request,
    response: Response,
    refresh: bool = True,
    include_current_price: bool = True,
):
    active = service(request)
    return object_response(
        active,
        active.overview(
            symbol,
            refresh=refresh,
            include_current_price=include_current_price,
        ),
        response,
    )
