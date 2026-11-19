"""Provider-backed ETF fund exposure, distinct from issuer classification."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ..stock_models import ETFExposureItem, StockObjectResponse
from .common import object_response, service

router = APIRouter(prefix="/api/v1/etfs", tags=["Stock Analysis"])


@router.get("/{symbol}/exposure", response_model=StockObjectResponse[ETFExposureItem])
def etf_exposure(symbol: str, request: Request, response: Response):
    active = service(request)
    return object_response(active, active.etf_exposure(symbol), response)
