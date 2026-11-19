from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query, Request, Response

from ..stock_models import ConstituentItem, IndexItem, StockCollectionResponse
from .common import collection_response, service

router = APIRouter(prefix="/api/v1/indexes", tags=["Stock Analysis"])


@router.get("", response_model=StockCollectionResponse[IndexItem])
def indexes(request: Request, response: Response):
    active = service(request)
    return collection_response(active, active.indexes(), response)


@router.get("/{index_id}/constituents", response_model=StockCollectionResponse[ConstituentItem])
def constituents(
    index_id: str,
    request: Request,
    response: Response,
    as_of: date | None = None,
    limit: int = Query(1000, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    active = service(request)
    result = active.index_constituents(
        index_id, as_of=as_of or active.now().date(), limit=limit, offset=offset
    )
    return collection_response(active, result, response)
