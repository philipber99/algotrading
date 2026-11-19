"""Bounded multi-leg spread-screening REST routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response
from starlette.concurrency import run_in_threadpool

from .. import SCREENER_MODEL_VERSION
from ..errors import ScreenerNotReadyError
from ..screener.config import ScreenerConfig
from ..screener.models import (
    CapabilitiesResponse,
    ScreeningRequest,
    ScreeningResponse,
)


router = APIRouter(prefix="/v1/screener", tags=["screener"])


@router.get("/health")
def health(request: Request, response: Response) -> dict[str, Any]:
    service = getattr(request.app.state, "screener_service", None)
    if service is None or not getattr(service, "ready", True):
        response.status_code = 503
        return {
            "status": "not_ready",
            "model_version": SCREENER_MODEL_VERSION,
            "error": "SCREENER_NOT_READY",
        }
    return service.health()


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities(request: Request) -> CapabilitiesResponse:
    service = getattr(request.app.state, "screener_service", None)
    if service is not None:
        return service.capabilities()
    config = ScreenerConfig()
    return CapabilitiesResponse(
        strategies=[
            "vertical",
            "calendar",
            "diagonal",
            "butterfly",
            "iron_condor",
            "double_calendar",
        ],
        static_delta_hedging=True,
        dynamic_delta_hedging=False,
        price_bases=["mid", "bid", "ask"],
        directional_assumption_modes=["zero", "risk_neutral", "explicit"],
        minimum_dte=0,
        maximum_dte=config.maximum_dte,
        maximum_results_per_strategy=config.max_results_per_strategy,
        maximum_simulation_paths=config.maximum_simulation_paths,
        maximum_holding_sessions=config.maximum_holding_sessions,
        model_version=SCREENER_MODEL_VERSION,
    )


@router.post("/spreads", response_model=ScreeningResponse)
async def spreads(body: ScreeningRequest, request: Request) -> ScreeningResponse:
    service = getattr(request.app.state, "screener_service", None)
    if service is None or not getattr(service, "ready", True):
        raise ScreenerNotReadyError("The spread screener is not initialized")
    return await run_in_threadpool(
        service.screen,
        body,
        request_id=getattr(request.state, "request_id", "unknown"),
    )


__all__ = ["router"]
