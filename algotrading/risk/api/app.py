"""FastAPI lifecycle, middleware, routes, and sanitized error handling."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any
from threading import RLock

from fastapi import FastAPI, Path, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from .bootstrap import build_context
from .broker_source import BrokerAccountSource
from .config import RiskApiConfig
from .errors import RiskApiError
from .portfolio_service import PortfolioApplicationService
from .hedge_service import HedgeLabService
from .routes import router as portfolio_router
from algotrading.risk.transactions.repository import (
    TransactionRepository,
    TransactionRepositoryUnavailable,
)
from algotrading.risk.option_prices.repository import (
    OptionPriceRepository,
    OptionPriceRepositoryUnavailable,
)
from algotrading.risk.option_prices.service import RiskOptionHistoryService
from algotrading.risk.alerts.engine import AlertEngine
from algotrading.risk.alerts.persistence import AlertStore, default_alert_state_path
from algotrading.risk.positions.strategies import StrategyGroupStore
from .models import (
    AccountResponse,
    AccountsResponse,
    ErrorResponse,
    LiveHealthResponse,
    PositionsResponse,
    ReadyHealthResponse,
    ResponseMeta,
)
from .service import SERVICE_NAME, SERVICE_VERSION, AccountSource, RiskService

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_ACCOUNT_PATH = Annotated[
    str,
    Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
]
_ACCOUNT_QUERY = Annotated[
    str | None,
    Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
]


def create_app(
    context: dict[str, Any] | None = None,
    *,
    source: AccountSource | None = None,
    service: RiskService | None = None,
    environment: str = "refdata",
    config: RiskApiConfig | None = None,
    source_factory: Callable[[dict[str, Any]], AccountSource] = BrokerAccountSource,
) -> FastAPI:
    """Create an app; tests can inject a source or complete service."""

    if context is None and source is None and service is None:
        context = build_context(environment, quiet_configuration=True)
    context = context or {}
    settings = config or RiskApiConfig.from_context(context)
    logger = _logger(context)
    configuration_loaded = True

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal source, service
        if service is None:
            if source is None:
                source = source_factory(context)
            service = RiskService(
                source,
                cache_ttl_seconds=settings.account_cache_ttl_seconds,
            )
        # Create lifespan-scoped services once and share them across request handlers.
        app.state.risk_service = service
        portfolio_service = PortfolioApplicationService(service, context)
        app.state.portfolio_service = portfolio_service
        app.state.tian_service = portfolio_service.tian_service
        state_path = default_alert_state_path(context)
        app.state.alert_store = AlertStore(state_path)
        context.update(app.state.alert_store.load_position_configuration())
        app.state.alert_engine = AlertEngine(context, portfolio_service.reference_client)
        app.state.strategy_group_store = StrategyGroupStore(state_path)
        try:
            app.state.transaction_repository = TransactionRepository.from_context(context)
            app.state.transaction_repository_error = None
        except TransactionRepositoryUnavailable as exc:
            app.state.transaction_repository = None
            app.state.transaction_repository_error = str(exc)
        try:
            app.state.option_price_repository = OptionPriceRepository.from_context(
                context
            )
            app.state.option_price_repository_error = None
        except OptionPriceRepositoryUnavailable as exc:
            app.state.option_price_repository = None
            app.state.option_price_repository_error = str(exc)
        # Share exact-option history between portfolio routes and Hedge Lab.
        app.state.option_history_service = RiskOptionHistoryService(
            app.state.option_price_repository,
            portfolio_service.reference_client,
        )
        app.state.historical_revaluation_lock = RLock()
        app.state.historical_revaluation_cache = {}
        app.state.hedge_lab_service = HedgeLabService(
            portfolio_service,
            risk_history_service=app.state.option_history_service,
            historical_revaluation_cache=app.state.historical_revaluation_cache,
            historical_revaluation_lock=app.state.historical_revaluation_lock,
        )
        readiness = service.readiness()
        logger.info(
            "Risk API startup ready=%s broker_constructed=%s methods_callable=%s",
            bool(
                readiness.get("broker_constructed")
                and readiness.get("required_methods_callable")
            ),
            bool(readiness.get("broker_constructed")),
            bool(readiness.get("required_methods_callable")),
        )
        yield
        portfolio_service.close()
        # This application owns no streaming or broker shutdown operation.

    app = FastAPI(
        title="Algotrading Risk API",
        description=(
            "Read-only normalized access to broker accounts and open positions."
        ),
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    app.include_router(portfolio_router)

    app.add_middleware(
        GZipMiddleware,
        minimum_size=settings.gzip_minimum_size,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
            allow_headers=["Accept", "Content-Type", "X-Request-ID"],
        )

    @app.middleware("http")
    async def request_metadata(request: Request, call_next):
        request_id = _request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        # Prevent account payloads from being retained by HTTP caches.
        if _is_account_response(request.url.path):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"

        # Log route templates so account IDs never become access-log path fields.
        route = request.scope.get("route")
        route_template = getattr(route, "path", "<unmatched>")
        logger.info(
            "request_id=%s route=%s status=%d elapsed_ms=%.2f cache=%s count=%s exception=%s",
            request_id,
            route_template,
            response.status_code,
            elapsed_ms,
            getattr(request.state, "cache", "none"),
            getattr(request.state, "result_count", "none"),
            getattr(request.state, "exception_class", "none"),
        )
        return response

    @app.exception_handler(RiskApiError)
    async def service_error_handler(
        request: Request, exc: RiskApiError
    ) -> JSONResponse:
        request.state.exception_class = type(exc).__name__
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_content(
                code=exc.code,
                message=exc.message,
                request_id=request.state.request_id,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Expose a fixed validation error without echoing submitted values.
        request.state.exception_class = type(exc).__name__
        return JSONResponse(
            status_code=422,
            content=_error_content(
                code="VALIDATION_ERROR",
                message="Request validation failed",
                request_id=request.state.request_id,
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request.state.exception_class = type(exc).__name__
        # Record exception types while keeping internal exception text out of responses.
        logger.error(
            "request_id=%s unhandled_exception=%s",
            request.state.request_id,
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content=_error_content(
                code="INTERNAL_ERROR",
                message="An unexpected server error occurred",
                request_id=request.state.request_id,
            ),
        )

    error_responses = {
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    }

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "documentation": "/docs",
        }

    @app.get("/health/live", response_model=LiveHealthResponse)
    def live() -> LiveHealthResponse:
        return LiveHealthResponse(service=SERVICE_NAME, version=SERVICE_VERSION)

    @app.get(
        "/health/ready",
        response_model=ReadyHealthResponse,
        responses={503: {"model": ReadyHealthResponse}},
    )
    def ready(request: Request, response: Response) -> ReadyHealthResponse:
        status = _service(request).readiness()
        is_ready = bool(
            configuration_loaded
            and status.get("broker_constructed")
            and status.get("required_methods_callable")
        )
        if not is_ready:
            response.status_code = 503
        return ReadyHealthResponse(
            status="ready" if is_ready else "not_ready",
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            configuration_loaded=configuration_loaded,
            broker_constructed=bool(status.get("broker_constructed")),
            required_methods_callable=bool(status.get("required_methods_callable")),
            diagnostic=status.get("diagnostic"),
            last_account_request_at=status.get("last_account_request_at"),
            last_account_request_outcome=status.get("last_account_request_outcome"),
        )

    @app.get(
        "/api/v1/accounts",
        response_model=AccountsResponse,
        responses=error_responses,
    )
    def accounts(
        request: Request,
        include_positions: bool = True,
        refresh: bool = False,
    ) -> AccountsResponse:
        result = _service(request).accounts(
            include_positions=include_positions,
            refresh=refresh,
        )
        _record_result(request, result.cache, len(result.data))
        return AccountsResponse(
            data=result.data,
            meta=_meta(request, len(result.data), result.as_of, result.cache),
        )

    @app.get(
        "/api/v1/accounts/{account_id}",
        response_model=AccountResponse,
        responses=error_responses,
    )
    def account(
        request: Request,
        account_id: _ACCOUNT_PATH,
        include_positions: bool = True,
        refresh: bool = False,
    ) -> AccountResponse:
        result = _service(request).account(
            account_id,
            include_positions=include_positions,
            refresh=refresh,
        )
        _record_result(request, result.cache, 1)
        return AccountResponse(
            data=result.data,
            meta=_meta(request, 1, result.as_of, result.cache),
        )

    @app.get(
        "/api/v1/accounts/{account_id}/positions",
        response_model=PositionsResponse,
        responses=error_responses,
    )
    def account_positions(
        request: Request,
        account_id: _ACCOUNT_PATH,
        refresh: bool = False,
    ) -> PositionsResponse:
        result = _service(request).positions(
            account_id=account_id,
            refresh=refresh,
        )
        _record_result(request, result.cache, len(result.data))
        return PositionsResponse(
            data=result.data,
            meta=_meta(request, len(result.data), result.as_of, result.cache),
        )

    @app.get(
        "/api/v1/positions",
        response_model=PositionsResponse,
        responses=error_responses,
    )
    def positions(
        request: Request,
        account_id: _ACCOUNT_QUERY = None,
        refresh: bool = False,
    ) -> PositionsResponse:
        result = _service(request).positions(
            account_id=account_id,
            refresh=refresh,
        )
        _record_result(request, result.cache, len(result.data))
        return PositionsResponse(
            data=result.data,
            meta=_meta(request, len(result.data), result.as_of, result.cache),
        )

    return app


def _service(request: Request) -> RiskService:
    return request.app.state.risk_service


def _meta(
    request: Request,
    count: int,
    as_of,
    cache,
) -> ResponseMeta:
    return ResponseMeta(
        count=count,
        as_of=as_of,
        cache=cache,
        request_id=request.state.request_id,
    )


def _record_result(request: Request, cache: str, count: int) -> None:
    request.state.cache = cache
    request.state.result_count = count


def _request_id(candidate: str | None) -> str:
    if candidate and _SAFE_REQUEST_ID.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


def _is_account_response(path: str) -> bool:
    return path == "/api/v1/positions" or path.startswith("/api/v1/accounts")


def _error_content(*, code: str, message: str, request_id: str) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": None,
        }
    }


def _logger(context: dict[str, Any]) -> logging.Logger:
    candidate = context.get("logger")
    if isinstance(candidate, logging.Logger):
        return candidate
    return logging.getLogger("algotrading.risk.api")
