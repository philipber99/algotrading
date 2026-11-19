"""FastAPI application and REST routes."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any, Literal

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware

from .bootstrap import build_context
from .app_identity import SERVICE_NAME
from .models import (
    CollectionResponse,
    ErrorResponse,
    HealthResponse,
    ObjectResponse,
    ResponseMeta,
)
from .live_models import (
    FuturesSpecificationRequest,
    FuturesSpecificationResponse,
    OptionChainRequest,
    OptionChainResponse,
    OptionExpirationRequest,
    OptionExpirationResponse,
    OptionContractTermsRequest,
    OptionContractTermsResponse,
    OptionMarketHistoryRequest,
    OptionMarketHistoryResponse,
    OptionsQuoteRequest,
    OptionsQuoteResponse,
    RiskFreeCurveHistoryRequest,
    RiskFreeCurveHistoryResponse,
    RiskFreeCurveResponse,
)
from .reference_ingress import (
    CmeReferenceDataService,
    FredRiskFreeCurveService,
    LiveOptionService,
)
from .service import (
    SERVICE_VERSION,
    InvalidRequestError,
    RefDataService,
    ServiceError,
    ServiceResult,
)
from .routes import stock_analysis_router
from .economic_calendar_updater import (
    EconomicCalendarUpdater,
    RefDataTradingCalendar,
    build_economic_calendar_updater,
)
from .earnings_calendar_updater import EarningsCalendarUpdater
from algotrading.refdata.stock_analysis.service import StockAnalysisService
from algotrading.refdata.stock_analysis.analytics import AnalyticsService
from algotrading.refdata.stock_analysis.market_flow import MarketFlowService
from algotrading.refdata.stock_analysis.futures_analytics import FuturesAnalyticsService
from algotrading.refdata.stock_analysis.economic_calendar import (
    EconomicReleaseReadService,
)
from algotrading.refdata.stock_analysis.earnings_calendar import EarningsCalendarReconciler
from algotrading.refdata.repositories import (
    StockAnalysisConnectionFactory,
    StockAnalysisRepositories,
)

def create_app(
    context: dict[str, Any] | None = None,
    service: RefDataService | None = None,
    stock_service: StockAnalysisService | None = None,
    economic_release_service: EconomicReleaseReadService | None = None,
    economic_updater: EconomicCalendarUpdater | None = None,
    earnings_updater: EarningsCalendarUpdater | None = None,
    analytics_service: AnalyticsService | None = None,
    market_flow_service: MarketFlowService | None = None,
    futures_analytics_service: FuturesAnalyticsService | None = None,
    environment: str = "refdata",
) -> FastAPI:
    logger = _logger_from_context(context)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal context, service, stock_service, economic_release_service, economic_updater, earnings_updater, analytics_service, market_flow_service, futures_analytics_service, logger
        if context is None and service is None:
            context = build_context(environment, quiet_configuration=True)
            logger = _logger_from_context(context)
        if service is None:
            service = RefDataService(context or {})
        effective_context = context or service.context
        app.state.refdata_context = effective_context
        # Share provider-backed ingress services for the application lifespan.
        app.state.live_option_service = LiveOptionService(
            effective_context,
            service._provider,
            now_fn=service.now,
            market_open_fn=service._market_is_open,
            latest_completed_session_close_fn=service.latest_completed_session_close,
        )
        app.state.fred_curve_service = FredRiskFreeCurveService(
            effective_context, now_fn=service.now
        )
        app.state.cme_reference_service = CmeReferenceDataService(
            effective_context, now_fn=service.now
        )
        if stock_service is None:
            stock_service = StockAnalysisService(
                context or {}, price_service=service
            )

        repositories = getattr(stock_service, "repositories", None)
        if repositories is None:
            repositories = StockAnalysisRepositories(
                StockAnalysisConnectionFactory(context or {})
            )
        if economic_release_service is None:
            economic_release_service = EconomicReleaseReadService(
                context or {}, repositories.economic_calendar
            )
        if economic_updater is None:
            economic_updater = build_economic_calendar_updater(
                context or {},
                repositories,
                RefDataTradingCalendar(service),
            )
        if earnings_updater is None:
            earnings_updater = EarningsCalendarUpdater(
                context or {},
                repositories.earnings_calendar,
                repositories.fetch_state,
                EarningsCalendarReconciler(repositories.earnings_calendar),
            )
        if analytics_service is None:
            analytics_service = AnalyticsService(
                context or {},
                repositories.analytics,
                economic_repository=repositories.economic_calendar,
                calendar_service=service,
                live_intraday_source=(context or {}).get("StreamingPrices"),
                quote_source=service,
            )
        if market_flow_service is None:
            market_flow_service = MarketFlowService(
                context or {}, repositories.market_flow,
                calendar_service=service,
            )
        if futures_analytics_service is None:
            futures_analytics_service = FuturesAnalyticsService(
                context or {}, repositories.futures, now_fn=service.now,
            )

        app.state.refdata_service = service
        app.state.stock_analysis_service = stock_service
        app.state.economic_release_service = economic_release_service
        app.state.economic_calendar_updater = economic_updater
        app.state.earnings_calendar_updater = earnings_updater
        app.state.analytics_service = analytics_service
        app.state.market_flow_service = market_flow_service
        app.state.futures_analytics_service = futures_analytics_service
        # Build the quote polling universe from durable constituents and configured symbols.
        configure_quote_precache = getattr(
            service, "configure_quote_precache", None
        )
        if callable(configure_quote_precache):

            def quote_universe_symbols() -> list[str]:
                configured_indexes = _list_setting(
                    (context or {}).get(
                        "refdata.api.quote_precache.indexes",
                        "SP100,SP500,NASDAQ100",
                    )
                )
                symbols = set(_list_setting(
                    (context or {}).get(
                        "refdata.api.quote_precache.additional_symbols",
                        "SPY,QQQ,VIX",
                    )
                ))
                as_of = service.now().date()
                for index_id in configured_indexes:
                    result = stock_service.index_constituents(
                        index_id,
                        as_of=as_of,
                        limit=5000,
                        offset=0,
                    )
                    symbols.update(
                        str(row.get("symbol") or "")
                        for row in result.data
                        if row.get("symbol")
                    )
                return sorted(symbols)

            configure_quote_precache(quote_universe_symbols)
        configure_daily_bars_repository = getattr(
            service, "configure_daily_bars_repository", None
        )
        if callable(configure_daily_bars_repository):
            configure_daily_bars_repository(repositories.analytics)
        configure_event_calendar_repositories = getattr(
            service, "configure_event_calendar_repositories", None
        )
        if callable(configure_event_calendar_repositories):
            configure_event_calendar_repositories(
                repositories.earnings_calendar,
                repositories.dividends,
            )
        if cors_origins:
            logger.info("Browser CORS enabled for %d configured origin(s)", len(cors_origins))
        else:
            logger.info("Browser CORS disabled; no allowed origins are configured")
        # Run blocking startup work outside the async server event loop.
        await run_in_threadpool(service.startup)
        await run_in_threadpool(stock_service.startup)
        start_quote_precache = getattr(service, "start_quote_precache", None)
        if callable(start_quote_precache):
            start_quote_precache()
        analytics_warmup = getattr(analytics_service, "start_background_warmup", None)
        if callable(analytics_warmup):
            analytics_warmup()
        economic_updater.start()
        earnings_updater.start()
        try:
            # Keep background workers alive while the application serves requests.
            yield
        finally:
            # Stop managed workers before shutting down their underlying services.
            try:
                await earnings_updater.stop()
            finally:
                try:
                    await economic_updater.stop()
                finally:
                    stop_quote_precache = getattr(
                        service, "stop_quote_precache", None
                    )
                    if callable(stop_quote_precache):
                        await run_in_threadpool(stop_quote_precache)
                    stop_analytics_warmup = getattr(
                        analytics_service, "stop_background_warmup", None
                    )
                    if callable(stop_analytics_warmup):
                        await run_in_threadpool(stop_analytics_warmup)
                    await run_in_threadpool(stock_service.shutdown)
                    await run_in_threadpool(service.shutdown)

    app = FastAPI(
        title="Algotrading Reference Data API",
        description=(
            "Read-oriented REST access to trading calendars, market status, "
            "OHLCV bars, current prices, and market events."
        ),
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    app.include_router(stock_analysis_router)

    gzip_minimum = int((context or {}).get("refdata.api.gzip_minimum_size", 1000))
    app.add_middleware(GZipMiddleware, minimum_size=gzip_minimum)

    cors_origins = _list_setting((context or {}).get("refdata.api.cors_origins", []))
    if "*" in cors_origins:
        raise ValueError("refdata.api.cors_origins must contain explicit origins")
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Accept", "Content-Type", "X-Request-ID"],
        )

    @app.middleware("http")
    async def request_metadata(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "%s %s completed in %.2f ms",
                request.method,
                request.url.path,
                elapsed_ms,
            )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(
        request: Request, exc: ServiceError
    ) -> JSONResponse:
        detail = {
            "code": exc.code,
            "message": exc.message,
            "request_id": getattr(request.state, "request_id", None),
        }
        if exc.details is not None:
            detail["details"] = exc.details
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": detail},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    # Validation exceptions can embed submitted values and credentials.
                    "message": "Request validation failed",
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled REST API exception exception=%s", type(exc).__name__)
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

    error_responses = {
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    }

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "documentation": "/docs",
        }

    @app.get("/health/live", response_model=HealthResponse)
    def liveness() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
        )

    @app.get("/health/ready", response_model=HealthResponse)
    def readiness(request: Request, response: Response) -> HealthResponse:
        active_service = _service(request)
        active_stock_service = request.app.state.stock_analysis_service
        # Readiness requires both core pricing and durable reference-data services.
        core_ready = bool(active_service.ready)
        durable_ready = bool(getattr(active_stock_service, "ready", True))
        stock_details_fn = getattr(active_stock_service, "health_details", None)
        details = {
            "core_price": active_service.health_details(),
            "durable_stock_analysis_sql": (
                stock_details_fn()
                if callable(stock_details_fn)
                else {"ready": durable_ready}
            ),
        }
        if not (core_ready and durable_ready):
            response.status_code = 503
            return HealthResponse(
                status="not_ready",
                service=SERVICE_NAME,
                version=SERVICE_VERSION,
                details=details,
            )
        return HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            details=details,
        )

    @app.get(
        "/api/v1/market/status",
        response_model=ObjectResponse,
        responses=error_responses,
    )
    def market_status(request: Request, response: Response) -> ObjectResponse:
        return _object_response(
            _service(request),
            _service(request).market_status(),
            response,
        )

    @app.get(
        "/api/v1/calendars/trading",
        response_model=CollectionResponse,
        responses=error_responses,
    )
    def trading_calendar(
        request: Request,
        response: Response,
        start_date: date,
        end_date: date,
    ) -> CollectionResponse:
        active = _service(request)
        return _collection_response(
            active,
            active.trading_calendar(start_date, end_date),
            response,
        )

    @app.get(
        "/api/v1/quotes",
        response_model=CollectionResponse,
        responses=error_responses,
        description=(
            "Returns batched day open/bid/ask/mark quotes on trading days. "
            "On non-trading days, open and mark come from the last trading "
            "day's SQL bar and bid/ask are omitted."
        ),
    )
    def day_quotes(
        request: Request,
        response: Response,
        symbols: list[str] = Query(...),
    ) -> CollectionResponse:
        requested = [
            symbol.strip()
            for value in symbols
            for symbol in value.split(",")
            if symbol.strip()
        ]
        active = _service(request)
        return _collection_response(active, active.day_quotes(requested), response)

    @app.get(
        "/api/v1/bars",
        response_model=CollectionResponse,
        responses=error_responses,
        description=(
            "Returns daily price history for multiple symbols using one bounded "
            "SQL read. Historical rows are cached across API restarts for the "
            "current calendar day. Current quotes are opt-in and batch loaded."
        ),
    )
    def bulk_bars(
        request: Request,
        response: Response,
        symbols: list[str] = Query(...),
        start: str = Query(
            ...,
            description="Inclusive ISO date or RFC 3339 timestamp",
        ),
        end: str = Query(
            ...,
            description="Inclusive ISO date or RFC 3339 timestamp",
        ),
        interval: Literal["1d"] = "1d",
        include_current_quote: bool = False,
        remote_query: bool = Query(
            False,
            alias="remote-query",
            description=(
                "When true, query the configured equity provider only after the "
                "local database has been checked for missing daily bars."
            ),
        ),
    ) -> CollectionResponse:
        del interval
        requested = [
            symbol.strip()
            for value in symbols
            for symbol in value.split(",")
            if symbol.strip()
        ]
        active = _service(request)
        result = active.bulk_bars(
            requested,
            _parse_datetime(start, "start"),
            _parse_datetime(end, "end", end_of_day=True),
            include_current_quote=include_current_quote,
            remote_query=remote_query,
        )
        return _collection_response(active, result, response)

    @app.get(
        "/api/v1/symbols/{symbol}/bars",
        response_model=CollectionResponse,
        responses=error_responses,
    )
    def bars(
        symbol: str,
        request: Request,
        response: Response,
        start: str = Query(
            ...,
            description="Inclusive ISO date or RFC 3339 timestamp",
        ),
        end: str = Query(
            ...,
            description="Inclusive ISO date or RFC 3339 timestamp",
        ),
        interval: Literal["1d", "1m"] = "1d",
        market_hours_only: bool = False,
        remote_query: bool = Query(
            False,
            alias="remote-query",
            description=(
                "Query the configured provider after checking the local database "
                "for missing bars. One-minute requests must stay within one local date."
            ),
        ),
    ) -> CollectionResponse:
        active = _service(request)
        result = active.bars(
            symbol,
            _parse_datetime(start, "start"),
            _parse_datetime(end, "end", end_of_day=True),
            interval,
            market_hours_only,
            remote_query,
        )
        return _collection_response(active, result, response)

    @app.get(
        "/api/v1/symbols/{symbol}/intraday-bars",
        response_model=CollectionResponse,
        responses=error_responses,
        description=(
            "Returns detailed one-minute bars for a single symbol and the "
            "current or latest trading session. This endpoint does not use "
            "the streaming-prices cache and is not intended for bulk retrieval."
        ),
    )
    def intraday_bars(
        symbol: str,
        request: Request,
        response: Response,
        market_hours_only: bool = True,
    ) -> CollectionResponse:
        active = _service(request)
        return _collection_response(
            active,
            active.intraday_bars(symbol, market_hours_only),
            response,
        )

    @app.get(
        "/api/v1/futures/{contract}/current-price",
        response_model=ObjectResponse,
        responses=error_responses,
    )
    def current_futures_price(
        contract: str,
        request: Request,
        response: Response,
        view: Literal["price", "bar"] = "price",
    ) -> ObjectResponse:
        active = _service(request)
        return _object_response(
            active,
            active.current_price(contract, view, instrument_type="FUTURE"),
            response,
        )

    @app.get(
        "/api/v1/symbols/{symbol}/current-price",
        response_model=ObjectResponse,
        responses=error_responses,
    )
    def current_price(
        symbol: str,
        request: Request,
        response: Response,
        view: Literal["price", "bar"] = "price",
    ) -> ObjectResponse:
        active = _service(request)
        return _object_response(
            active,
            active.current_price(symbol, view),
            response,
        )

    @app.get(
        "/api/v1/symbols/{symbol}/splits",
        response_model=CollectionResponse,
        responses=error_responses,
    )
    def splits(
        symbol: str,
        request: Request,
        response: Response,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> CollectionResponse:
        active = _service(request)
        return _collection_response(
            active,
            active.stock_splits(symbol, start_date, end_date),
            response,
        )

    @app.get(
        "/api/v1/symbols/{symbol}/earnings",
        response_model=CollectionResponse,
        responses=error_responses,
    )
    def earnings(
        symbol: str,
        request: Request,
        response: Response,
        start_date: date,
        end_date: date,
    ) -> CollectionResponse:
        active = _service(request)
        return _collection_response(
            active,
            active.earnings(symbol, start_date, end_date),
            response,
        )

    @app.get(
        "/api/v1/symbols/{symbol}/dividends",
        response_model=CollectionResponse,
        responses=error_responses,
    )
    def dividends(
        symbol: str,
        request: Request,
        response: Response,
        start_date: date,
        end_date: date,
        date_type: Literal[
            "ex_dividend_date",
            "declaration_date",
            "record_date",
            "pay_date",
        ] = "ex_dividend_date",
    ) -> CollectionResponse:
        active = _service(request)
        return _collection_response(
            active,
            active.dividends(symbol, start_date, end_date, date_type),
            response,
        )

    @app.get(
        "/api/v1/calendars/earnings",
        response_model=CollectionResponse,
        responses=error_responses,
    )
    def earnings_calendar(
        request: Request,
        response: Response,
        start_date: date,
        end_date: date,
    ) -> CollectionResponse:
        active = _service(request)
        return _collection_response(
            active,
            active.earnings_calendar(start_date, end_date),
            response,
        )

    @app.get(
        "/api/v1/calendars/dividends",
        response_model=CollectionResponse,
        responses=error_responses,
    )
    def dividend_calendar(
        request: Request,
        response: Response,
        start_date: date,
        end_date: date,
        date_type: Literal[
            "ex_dividend_date",
            "declaration_date",
            "record_date",
            "pay_date",
        ] = "ex_dividend_date",
    ) -> CollectionResponse:
        active = _service(request)
        return _collection_response(
            active,
            active.dividend_calendar(start_date, end_date, date_type),
            response,
        )

    @app.get(
        "/api/v1/calendars/fomc",
        response_model=CollectionResponse,
        responses=error_responses,
    )
    def fomc(
        request: Request,
        response: Response,
        start_date: date,
        end_date: date,
        last_day_only: bool = True,
    ) -> CollectionResponse:
        active = _service(request)
        return _collection_response(
            active,
            active.fomc(start_date, end_date, last_day_only),
            response,
        )

    @app.get(
        "/api/v1/calendars/economic",
        response_model=CollectionResponse,
        responses=error_responses,
        description=(
            "Returns the economic-calendar data currently stored by the "
            "reference-data package."
        ),
    )
    def economic_calendar(
        request: Request,
        response: Response,
        start_date: date,
        end_date: date,
    ) -> CollectionResponse:
        active = _service(request)
        return _collection_response(
            active,
            active.economic_calendar(start_date, end_date),
            response,
        )

    @app.post(
        "/api/v1/options/quotes",
        response_model=OptionsQuoteResponse,
        responses=error_responses,
        description=(
            "Returns a bounded, ordered, best-effort batch of current exact OCC "
            "option quotes. The route never falls back to a chain or database."
        ),
    )
    def option_quotes(
        payload: OptionsQuoteRequest,
        request: Request,
        response: Response,
    ) -> OptionsQuoteResponse:
        result = request.app.state.live_option_service.option_quotes(payload.symbols)
        response.headers["X-Cache"] = result.meta.cache.upper()
        response.headers["X-Data-As-Of"] = result.meta.as_of.isoformat()
        return result

    @app.post(
        "/api/v1/options/market-history/query",
        response_model=OptionMarketHistoryResponse,
        responses=error_responses,
        description=(
            "Returns exact-OCC daily OHLC history from a restart-persistent, "
            "sliding-TTL cache and requests only missing date intervals."
        ),
    )
    def option_market_history(
        payload: OptionMarketHistoryRequest,
        request: Request,
        response: Response,
    ) -> OptionMarketHistoryResponse:
        maximum_days = max(
            1,
            int(
                request.app.state.refdata_context.get(
                    "refdata.api.option_market_history_max_days", 376
                )
            ),
        )
        # Reject oversized history windows before acquiring provider resources.
        if (payload.end_at - payload.start_at).total_seconds() > maximum_days * 86400:
            raise InvalidRequestError(
                f"option market-history interval may not exceed {maximum_days} days",
                details={
                    "endpoint": "/api/v1/options/market-history/query",
                    "requested_start": payload.start_at.isoformat(),
                    "requested_end": payload.end_at.isoformat(),
                    "maximum_days": maximum_days,
                    "exclusive_end": True,
                },
            )
        result = _service(request).option_market_history(payload)
        response.headers["X-Cache"] = result.meta.cache.upper()
        response.headers["X-Data-As-Of"] = result.meta.as_of.isoformat()
        return result

    @app.post(
        "/api/v1/options/contract-terms/query",
        response_model=OptionContractTermsResponse,
        responses=error_responses,
        description=(
            "Resolves a bounded set of exact OCC contracts without requiring a full chain."
        ),
    )
    def option_contract_terms(
        payload: OptionContractTermsRequest,
        request: Request,
        response: Response,
    ) -> OptionContractTermsResponse:
        result = request.app.state.live_option_service.option_contract_terms(
            payload.symbols
        )
        response.headers["X-Cache"] = result.meta.cache.upper()
        response.headers["X-Data-As-Of"] = result.meta.as_of.isoformat()
        return result

    @app.post(
        "/api/v1/options/expirations/query",
        response_model=OptionExpirationResponse,
        responses=error_responses,
        description=(
            "Loads the provider's lightweight option-expiration calendar without a chain."
        ),
    )
    def option_expirations(
        payload: OptionExpirationRequest,
        request: Request,
        response: Response,
    ) -> OptionExpirationResponse:
        result = request.app.state.live_option_service.option_expirations(payload)
        response.headers["X-Cache"] = result.meta.cache.upper()
        response.headers["X-Data-As-Of"] = result.meta.as_of.isoformat()
        return result

    @app.post(
        "/api/v1/options/chains/query",
        response_model=OptionChainResponse,
        responses=error_responses,
        description=(
            "Explicitly loads one tightly bounded current option chain for Hedge Lab."
        ),
    )
    def option_chain(
        payload: OptionChainRequest,
        request: Request,
        response: Response,
    ) -> OptionChainResponse:
        result = request.app.state.live_option_service.option_chain(payload)
        response.headers["X-Cache"] = result.meta.cache.upper()
        response.headers["X-Data-As-Of"] = result.meta.as_of.isoformat()
        return result

    @app.get(
        "/api/v1/rates/risk-free-curve",
        response_model=RiskFreeCurveResponse,
        responses=error_responses,
    )
    def risk_free_curve(
        request: Request,
        response: Response,
        as_of: date | None = None,
    ) -> RiskFreeCurveResponse:
        result = request.app.state.fred_curve_service.curve(as_of)
        response.headers["X-Cache"] = result.meta.cache.upper()
        response.headers["X-Data-As-Of"] = result.meta.as_of.isoformat()
        return result

    @app.post(
        "/api/v1/rates/risk-free-curves/query",
        response_model=RiskFreeCurveHistoryResponse,
        responses=error_responses,
        description="Loads a bounded point-in-time curve history in one request.",
    )
    def risk_free_curve_history(
        payload: RiskFreeCurveHistoryRequest,
        request: Request,
        response: Response,
    ) -> RiskFreeCurveHistoryResponse:
        result = request.app.state.fred_curve_service.curves(payload.as_of_dates)
        response.headers["X-Cache"] = result.meta.cache.upper()
        response.headers["X-Data-As-Of"] = result.meta.as_of.isoformat()
        return result

    @app.post(
        "/api/v1/futures/contract-specifications",
        response_model=FuturesSpecificationResponse,
        responses=error_responses,
    )
    def futures_contract_specifications(
        payload: FuturesSpecificationRequest,
        request: Request,
        response: Response,
    ) -> FuturesSpecificationResponse:
        result = request.app.state.cme_reference_service.specifications(payload.contracts)
        response.headers["X-Cache"] = result.meta.cache.upper()
        response.headers["X-Data-As-Of"] = result.meta.as_of.isoformat()
        return result

    return app


def _service(request: Request) -> RefDataService:
    return request.app.state.refdata_service


def _parse_datetime(value: str, field_name: str, end_of_day: bool = False) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidRequestError(
            f"{field_name} must be an ISO date or RFC 3339 timestamp"
        ) from exc
    if end_of_day and len(value) == 10:
        parsed = parsed.replace(
            hour=23,
            minute=59,
            second=59,
            microsecond=999999,
        )
    return parsed


def _collection_response(
    service: RefDataService,
    result: ServiceResult,
    response: Response,
) -> CollectionResponse:
    _set_result_headers(result, response)
    return CollectionResponse(
        data=list(result.data),
        meta=_meta(service, result),
    )


def _object_response(
    service: RefDataService,
    result: ServiceResult,
    response: Response,
) -> ObjectResponse:
    _set_result_headers(result, response)
    return ObjectResponse(
        data=dict(result.data),
        meta=_meta(service, result),
    )


def _meta(service: RefDataService, result: ServiceResult) -> ResponseMeta:
    return ResponseMeta(
        count=result.count,
        as_of=result.loaded_at,
        timezone=service.timezone,
        cache=result.cache_state,
    )


def _set_result_headers(result: ServiceResult, response: Response) -> None:
    response.headers["X-Cache"] = result.cache_state.upper()
    response.headers["X-Data-As-Of"] = result.loaded_at.isoformat()
    if result.ttl_seconds > 0:
        response.headers["Cache-Control"] = (
            f"private, max-age={int(result.ttl_seconds)}"
        )
    else:
        response.headers["Cache-Control"] = "no-store"


def _logger_from_context(context: dict[str, Any] | None) -> logging.Logger:
    if context:
        try:
            from algotrading.ContextFields import PARAM_LOGGER

            configured = context.get(PARAM_LOGGER)
            if isinstance(configured, logging.Logger):
                return configured
        except ImportError:
            pass
    return logging.getLogger("algotrading.refdata.api")


def _list_setting(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item) for item in (value or [])]
