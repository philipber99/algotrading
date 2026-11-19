from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ..stock_models import DataStatusItem, StockCollectionResponse
from .common import collection_response, service

router = APIRouter(tags=["Stock Analysis"])


@router.get("/api/v1/data-status", response_model=StockCollectionResponse[DataStatusItem])
def data_status(request: Request, response: Response):
    active = service(request)
    return collection_response(active, active.data_status(), response)
