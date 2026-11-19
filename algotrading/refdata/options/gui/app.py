"""FastAPI application lifecycle shared by cache and screener routes."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware

from algotrading.refdata.options.Common import create_clickhouse_client

from . import SERVICE_NAME, SERVICE_TITLE, SERVICE_VERSION
from .bootstrap import load_context
from .cache.query_engine import TimeOfDayQueryEngine
from .cache.errors import QueryError
from .cache.startup_update import update_cache_if_needed
from .config import ApiConfig
from .errors import ApiError
from .routes.cache import router as cache_router
from .routes.screener import router as screener_router


def create_app(
    context: dict[str, Any] | None = None,
    engine: TimeOfDayQueryEngine | None = None,
    screener_service: Any | None = None,
    *,
    environment: str = "refdata",
    update_cache: bool = False,
    clickhouse_client_factory: Callable[..., Any] = create_clickhouse_client,
) -> FastAPI:
    """Create independently degradable cache and screener components."""

    logger = _logger(context)
    api_config = ApiConfig.from_context(context or {})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal context, engine, screener_service, logger
        if context is None and engine is None:
            context = load_context(
                environment,
                logger_name="algotrading.refdata.options.gui.api",
            )
            logger = _logger(context)

        # Cache/query startup is isolated from ClickHouse-backed screening.
        if update_cache:
            try:
                await run_in_threadpool(
                    update_cache_if_needed,
                    context or {},
                    logger=logger,
                )
            except Exception as exc:
                logger.warning(
                    "event=gui_api_cache_update_failed action=use_previous_generation "
                    "error_class=%s",
                    type(exc).__name__,
                )

        if engine is None:
            engine = TimeOfDayQueryEngine(
                context or {},
                logger=logger,
            )
        app.state.cache_engine = engine
        try:
            await run_in_threadpool(engine.startup)
        except Exception as exc:
            logger.warning(
                "No usable active cache was available at API startup error_class=%s",
                type(exc).__name__,
            )

        app.state.screener_service = screener_service
        app.state.screener_error = None
        if screener_service is None and context is None:
            # An injected cache engine is a valid cache-only application. Do
            # not guess ClickHouse configuration or delay cache routes.
            app.state.screener_error = "ConfigurationUnavailable"
        elif screener_service is None:
            screener_pricing = None
            repository = None
            try:
                from .screener.config import ScreenerConfig
                from .screener.repository import ScreenerRepository
                from .screener.scenarios import PricingResources
                from .screener.service import ScreenerService

                screener_config = ScreenerConfig.from_context(context or {})
                screener_pricing = PricingResources(
                    context or {}, config=screener_config
                )
                database = str(
                    (context or {}).get(
                        "options_chain.clickhouse.database", "algotrading"
                    )
                )
                client = clickhouse_client_factory(
                    context or {},
                    database=database,
                )
                repository = ScreenerRepository(client, database)
                screener_service = ScreenerService(
                    context or {},
                    repository,
                    config=screener_config,
                    pricing=screener_pricing,
                    logger=logger,
                )
                app.state.screener_service = screener_service
            except Exception as exc:
                if repository is not None:
                    try:
                        await run_in_threadpool(repository.close)
                    except Exception as cleanup_error:
                        logger.warning(
                            "Screener repository cleanup failed error_class=%s",
                            type(cleanup_error).__name__,
                        )
                if screener_pricing is not None:
                    try:
                        await run_in_threadpool(screener_pricing.close)
                    except Exception as cleanup_error:
                        logger.warning(
                            "Screener pricing cleanup failed error_class=%s",
                            type(cleanup_error).__name__,
                        )
                # Do not log configuration values, SQL, or connection strings.
                app.state.screener_error = type(exc).__name__
                logger.warning(
                    "event=screener_initialization_failed error_class=%s "
                    "action=cache_routes_remain_available",
                    type(exc).__name__,
                )
        try:
            yield
        finally:
            active_screener = getattr(app.state, "screener_service", None)
            if active_screener is not None:
                try:
                    await run_in_threadpool(active_screener.shutdown)
                except Exception as exc:
                    logger.warning(
                        "Screener shutdown failed error_class=%s",
                        type(exc).__name__,
                    )
            try:
                await run_in_threadpool(engine.shutdown)
            except Exception as exc:
                logger.warning(
                    "Cache query-engine shutdown failed error_class=%s",
                    type(exc).__name__,
                )

    app = FastAPI(
        title=SERVICE_TITLE,
        description=(
            "Read-only option time-of-day history and deterministic multi-leg "
            "option-spread screening."
        ),
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=api_config.gzip_minimum_size)
    if api_config.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(api_config.allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "If-None-Match", "X-Request-ID"],
        )

    @app.middleware("http")
    async def request_metadata(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        request.state.started_at = started
        content_length = request.headers.get("content-length")
        if (
            content_length is not None
            and content_length.isdigit()
            and int(content_length) > api_config.max_request_bytes
        ):
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "REQUEST_TOO_LARGE",
                        "message": "Request body exceeds the configured limit",
                        "request_id": request_id,
                    }
                },
                headers={"X-Request-ID": request_id},
            )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s status=%s request_id=%s elapsed_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            request_id,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        active = getattr(request.app.state, "cache_engine", None)
        request_id = getattr(request.state, "request_id", None)
        if exc.status_code == 409 and exc.details.get("diagnostics_version"):
            diagnostic = exc.details
            summary = diagnostic.get("request_summary", {})
            coverage = diagnostic.get("cache_coverage", {})
            funnel = diagnostic.get("funnel", {})
            rejections = diagnostic.get("rejections", {})
            elapsed_ms = (
                time.perf_counter()
                - getattr(request.state, "started_at", time.perf_counter())
            ) * 1000
            logger.info(
                "event=cache_profile_coverage_unavailable request_id=%s "
                "generation=%s symbol=%s window=%s mode=%s dominant_reason=%s "
                "selected_sessions=%s complete_observations=%s "
                "entry_session_minute=%s latest_common_session_minute=%s elapsed_ms=%.2f",
                request_id,
                active.generation_id if active is not None else None,
                summary.get("underlying_symbol"),
                summary.get("historical_window"),
                summary.get("mode"),
                rejections.get("dominant_reason"),
                funnel.get("sessions_selected_by_window", 0),
                funnel.get("complete_observations", 0),
                summary.get("entry_session_minute"),
                coverage.get("latest_common_session_minute"),
                elapsed_ms,
            )
        content: dict[str, Any] = {
            "code": exc.code,
            "message": exc.message,
            "details": exc.details or None,
            "request_id": request_id,
        }
        if isinstance(exc, QueryError):
            content["cache_generation_id"] = (
                active.generation_id if active is not None else None
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": content},
            headers={"Retry-After": "1"} if exc.status_code == 503 else None,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled Options GUI API exception error_class=%s", type(exc).__name__
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected server error occurred",
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "documentation": "/docs",
        }

    @app.get("/health/live")
    def live() -> dict[str, Any]:
        return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}

    @app.get("/health/ready")
    def ready(request: Request, response: Response) -> dict[str, Any]:
        active = request.app.state.cache_engine
        try:
            active.ensure_current()
        except Exception as exc:
            logger.debug(
                "Readiness cache reload failed error_class=%s", type(exc).__name__
            )
        response.status_code = 200 if active.ready else 503
        return {
            "status": "ok" if active.ready else "not_ready",
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "details": active.health(),
        }

    app.include_router(cache_router)
    app.include_router(screener_router)
    return app


def _logger(context: dict[str, Any] | None) -> logging.Logger:
    if context:
        for value in context.values():
            if isinstance(value, logging.Logger):
                return value
    return logging.getLogger("algotrading.refdata.options.gui.api")


__all__ = ["create_app"]
