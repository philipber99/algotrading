from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response

from ..stock_models import StockCollectionResponse, UniverseItem
from .common import collection_response, service

router = APIRouter(prefix="/api/v1/universe", tags=["Stock Analysis"])


@router.get("/symbols", response_model=StockCollectionResponse[UniverseItem])
def universe_symbols(
    request: Request,
    response: Response,
    active_only: bool = True,
    symbols: list[str] = Query(default=[]),
    security_type: str | None = None,
    index_id: str | None = None,
    classification_id: int | None = Query(None, ge=1),
    include_stale_daily_bars: bool = True,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    active = service(request)
    result = active.universe_symbols(
        active_only=active_only,
        symbols=symbols,
        security_type=security_type,
        index_id=index_id,
        classification_id=classification_id,
        include_stale_daily_bars=include_stale_daily_bars,
        limit=limit,
        offset=offset,
    )
    return collection_response(active, result, response)
