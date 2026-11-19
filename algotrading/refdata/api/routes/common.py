from __future__ import annotations

from fastapi import Request, Response

from algotrading.refdata.stock_analysis.service import StockAnalysisService, StockServiceResult


def service(request: Request) -> StockAnalysisService:
    return request.app.state.stock_analysis_service


def collection_response(
    active: StockAnalysisService, result: StockServiceResult, response: Response
):
    _headers(result, response)
    return {"data": result.data, "meta": _meta(active, result)}


def object_response(
    active: StockAnalysisService, result: StockServiceResult, response: Response
):
    _headers(result, response)
    return {"data": result.data, "meta": _meta(active, result)}


def _meta(active: StockAnalysisService, result: StockServiceResult):
    return {
        "count": result.count,
        "as_of": result.as_of,
        "timezone": active.timezone,
        "cache": result.cache,
        "stale": result.stale,
        "warnings": list(result.warnings),
        "provider": result.provider,
    }


def _headers(result: StockServiceResult, response: Response) -> None:
    response.headers["X-Cache"] = result.cache.upper()
    response.headers["X-Data-As-Of"] = result.as_of.isoformat()
    # Allow brief private caching only when the service did not bypass its cache.
    response.headers["Cache-Control"] = (
        "no-store" if result.cache == "bypass" else "private, max-age=30"
    )
