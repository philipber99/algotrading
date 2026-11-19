from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response

from ..stock_models import ClassificationItem, ConstituentItem, StockCollectionResponse
from .common import collection_response, service

router = APIRouter(prefix="/api/v1/classifications", tags=["Stock Analysis"])


@router.get("", response_model=StockCollectionResponse[ClassificationItem])
def classifications(
    request: Request,
    response: Response,
    taxonomy: str | None = None,
    sector: str | None = None,
    industry_group: str | None = None,
    industry: str | None = None,
    sub_industry: str | None = None,
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    active = service(request)
    result = active.classifications(
        taxonomy=taxonomy,
        sector=sector,
        industry_group=industry_group,
        industry=industry,
        sub_industry=sub_industry,
        limit=limit,
        offset=offset,
    )
    return collection_response(active, result, response)


@router.get("/{classification_id}/constituents", response_model=StockCollectionResponse[ConstituentItem])
def constituents(
    classification_id: int,
    request: Request,
    response: Response,
    limit: int = Query(1000, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    active = service(request)
    result = active.classification_constituents(
        classification_id, limit=limit, offset=offset
    )
    return collection_response(active, result, response)
