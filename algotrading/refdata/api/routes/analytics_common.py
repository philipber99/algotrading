from __future__ import annotations

from fastapi import Request, Response

from algotrading.refdata.stock_analysis.analytics.service import (
    AnalyticsService,
    AnalyticsServiceResult,
)


def analytics_service(request: Request) -> AnalyticsService:
    return request.app.state.analytics_service


def analytics_response(
    active: AnalyticsService,
    result: AnalyticsServiceResult,
    response: Response,
):
    # Expose data time and cache state consistently across analytics routes.
    response.headers["X-Cache"] = result.cache.upper()
    response.headers["X-Data-As-Of"] = result.as_of.isoformat()
    response.headers["Cache-Control"] = "private, max-age=3" if result.cache != "bypass" else "no-store"
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
