"""Unchanged cache metadata and time-of-day route contracts."""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, Query, Request, Response
from starlette.concurrency import run_in_threadpool

from ..cache.model import NormalizeRequest, ProfileRequest, canonical_json
from ..cache.query_engine import TimeOfDayQueryEngine


router = APIRouter()


@router.get("/v1/cache/metadata")
def metadata(request: Request, response: Response) -> Any:
    return _etag_response(request, response, _engine(request).metadata())


@router.get("/v1/cache/windows")
def windows(request: Request, response: Response) -> Any:
    return _etag_response(request, response, _engine(request).windows())


@router.get("/v1/cache/symbols")
def symbols(request: Request, response: Response) -> Any:
    return _etag_response(request, response, _engine(request).symbols())


@router.get("/v1/cache/coverage")
def coverage(
    request: Request,
    response: Response,
    symbol: str | None = Query(default=None),
    window: str | None = Query(default=None),
) -> Any:
    return _etag_response(
        request,
        response,
        _engine(request).coverage(symbol, window),
    )


@router.post("/v1/time-of-day/normalize")
async def normalize(
    body: NormalizeRequest,
    request: Request,
    response: Response,
) -> Any:
    payload = await run_in_threadpool(_engine(request).normalize, body)
    return _etag_response(request, response, payload)


@router.post("/v1/time-of-day/profile")
async def profile(
    body: ProfileRequest,
    request: Request,
    response: Response,
) -> Any:
    payload, hit = await run_in_threadpool(_engine(request).profile, body)
    response.headers["X-Cache"] = "HIT" if hit else "MISS"
    return _etag_response(request, response, payload)


def _engine(request: Request) -> TimeOfDayQueryEngine:
    return request.app.state.cache_engine


def _etag_response(
    request: Request,
    response: Response,
    payload: dict[str, Any],
) -> Any:
    etag = '"' + hashlib.sha256(canonical_json(payload)).hexdigest() + '"'
    response.headers["ETag"] = etag
    if request.headers.get("If-None-Match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return payload


__all__ = ["router"]
