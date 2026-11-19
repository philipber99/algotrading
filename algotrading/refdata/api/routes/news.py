from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query, Request, Response

from ..stock_models import NewsItem, StockCollectionResponse
from .common import collection_response, service

router = APIRouter(prefix="/api/v1/symbols", tags=["Stock Analysis"])


@router.get("/{symbol}/news", response_model=StockCollectionResponse[NewsItem])
def news(
    symbol: str,
    request: Request,
    response: Response,
    since: datetime | None = None,
    limit: int = Query(25, ge=1, le=100),
):
    active = service(request)
    result = active.news(symbol, since=since, limit=limit)
    payload = collection_response(active, result, response)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Cache"] = "BYPASS"
    return payload
