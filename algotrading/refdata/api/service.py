"""Thread-safe service adapter around algotrading.refdata.MarketData."""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from importlib import import_module
from pathlib import Path
from time import monotonic
from typing import Any, Literal, Sequence

import pandas as pd

from algotrading.refdata.stock_analysis.symbols import normalize_security_symbol
from algotrading.refdata.ReferenceData import is_fx, is_future, is_index, is_option

from .cache import (
    PersistentDailyBarsCache,
    PersistentOptionBarsCache,
    SingleFlightTTLCache,
    default_option_bars_cache_path,
)
from .live_models import (
    LiveResponseMeta,
    OptionMarketHistoryItem,
    OptionMarketHistoryRequest,
    OptionMarketHistoryResponse,
)
from .option_symbols import parse_occ_symbol
from .serialization import (
    aware_datetime,
    dataframe_records,
    json_value,
    marketdata_datetime,
    resolve_timezone,
    scalar_price,
    timezone_name,
)

SERVICE_VERSION = "1.7.0"
_CANONICAL_FUTURE_CONTRACT = re.compile(
    r"^[A-Z0-9]{1,6}[FGHJKMNQUVXZ][0-9]{1,2}$"
)


def _normalized_quote_timestamp(value: Any) -> str | None:
    """Normalize provider epoch/ISO quote times at the RefData boundary."""

    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        observed = value
    elif isinstance(value, (int, float)):
        seconds = float(value) / 1000.0 if abs(float(value)) > 10_000_000_000 else float(value)
        try:
            observed = datetime.fromtimestamp(seconds, timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    else:
        try:
            observed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return observed.astimezone(timezone.utc).isoformat()

class ServiceError(Exception):
    status_code = 500
    code = "SERVICE_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidRequestError(ServiceError):
    status_code = 422
    code = "INVALID_REQUEST"


class NotFoundError(ServiceError):
    status_code = 404
    code = "NOT_FOUND"


class UpstreamUnavailableError(ServiceError):
    status_code = 503
    code = "UPSTREAM_UNAVAILABLE"


class BusyError(ServiceError):
    status_code = 503
    code = "SERVICE_BUSY"


class ProviderCategoryUnsupportedError(ServiceError):
    status_code = 503
    code = "PROVIDER_CATEGORY_UNSUPPORTED"


class ProviderEntitlementRequiredError(ServiceError):
    status_code = 503
    code = "PROVIDER_ENTITLEMENT_REQUIRED"


class ProviderRateLimitedError(ServiceError):
    status_code = 503
    code = "PROVIDER_RATE_LIMITED"


class ProviderUnavailableError(ServiceError):
    status_code = 503
    code = "PROVIDER_UNAVAILABLE"


class ProviderPartialResponseError(ServiceError):
    status_code = 503
    code = "PROVIDER_PARTIAL_RESPONSE"


class DataUnavailableError(ServiceError):
    status_code = 503
    code = "DATA_UNAVAILABLE"


class DataIntegrityServiceError(ServiceError):
    status_code = 503
    code = "DATA_INTEGRITY_ERROR"


class RefreshInProgressError(ServiceError):
    status_code = 503
    code = "REFRESH_IN_PROGRESS"


class IntradayDataUnavailableError(ServiceError):
    status_code = 404
    code = "INTRADAY_DATA_UNAVAILABLE"


class IntradayDataIncompleteError(ServiceError):
    status_code = 503
    code = "INTRADAY_DATA_INCOMPLETE"


@dataclass(frozen=True)
class CachedPayload:
    data: Any
    count: int
    loaded_at: datetime


@dataclass(frozen=True)
class ServiceResult:
    data: Any
    count: int
    loaded_at: datetime
    cache_state: Literal["hit", "miss", "bypass"]
    ttl_seconds: float


@dataclass(frozen=True)
class QuoteCacheEntry:
    record: dict[str, Any]
    loaded_at: datetime
    session_date: date
    during_market_hours: bool


class RefDataService:
    def __init__(
        self,
        context: dict[str, Any],
        market_data_module: Any | None = None,
        now_fn: Callable[[], datetime] | None = None,
        cache_clock: Callable[[], float] = monotonic,
    ) -> None:
        self.context = context
        self.market_data = market_data_module or import_module(
            "algotrading.refdata.MarketData"
        )
        self.logger = self._resolve_logger()
        self.local_timezone = resolve_timezone(
            context.get(
                "refdata.api.timezone",
                context.get(
                    self._context_field("PARAM_LOCAL_TIMEZONE", "local_timezone")
                ),
            )
        )
        self._now_fn = now_fn or self._default_now

        self._cache = SingleFlightTTLCache[CachedPayload](
            max_entries=self._int("refdata.api.cache.max_entries", 1024),
            enabled=self._bool("refdata.api.cache.enabled", True),
            clock=cache_clock,
        )
        # Bound provider concurrency independently of the number of HTTP requests.
        self._upstream_slots = threading.BoundedSemaphore(
            self._int("refdata.api.max_concurrent_upstream_calls", 8)
        )
        self._upstream_wait_seconds = self._float(
            "refdata.api.upstream_slot_wait_seconds", 5.0
        )
        self._calendar_refresh_lock = threading.Lock()
        self._split_load_lock = threading.Lock()
        self._provider_lock = threading.Lock()
        self._option_history_provider_lock = threading.Lock()
        self._historical_bars_provider_lock = threading.Lock()
        self._calendar_loaded_at = 0.0
        self._refdata_source: Any | None = None
        self._option_history_source: Any | None = None
        self._historical_bars_source: Any | None = None
        self._quote_cache_lock = threading.RLock()
        self._quote_cache: dict[str, QuoteCacheEntry] = {}
        self._quote_cache_generation = 0
        self._quote_universe_loader: Callable[[], list[str] | tuple[str, ...]] | None = None
        self._quote_universe_date: date | None = None
        self._quote_universe: tuple[str, ...] = ()
        self._quote_precache_thread: threading.Thread | None = None
        self._quote_precache_stop = threading.Event()
        self._quote_precache_last_poll: datetime | None = None
        self._quote_precache_last_error: str | None = None
        self._daily_bars_repository: Any | None = None
        self._earnings_calendar_repository: Any | None = None
        self._dividends_repository: Any | None = None
        self._persistent_daily_bars: PersistentDailyBarsCache | None = None
        self._persistent_option_bars: PersistentOptionBarsCache | None = None
        self._option_history_lock = threading.RLock()
        self._ready = False
        self._startup_error: str | None = None

    @property
    def timezone(self) -> str:
        return timezone_name(self.local_timezone)

    @property
    def ready(self) -> bool:
        return self._ready

    def startup(self) -> None:
        try:
            self._ensure_calendar_fresh(force=True)
            self._provider()
            self._ready = True
            self._startup_error = None
        except Exception as exc:
            self._ready = False
            # Health diagnostics must not publish provider connection or credential text.
            self._startup_error = type(exc).__name__
            raise

    def shutdown(self) -> None:
        # Stop quote polling before releasing providers and persistent cache connections.
        self.stop_quote_precache()
        with self._historical_bars_provider_lock:
            historical = self._historical_bars_source
            self._historical_bars_source = None
        close = getattr(historical, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                self.logger.error(
                    "Error closing historical bar provider exception=%s", type(exc).__name__
                )
        streaming_key = self._context_field(
            "PARAM_STREAMING_PRICES", "streaming_prices"
        )
        streaming_prices = self.context.get(streaming_key)
        if streaming_prices is not None:
            for method_name in ("close", "shutdown", "stop"):
                method = getattr(streaming_prices, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as exc:
                        self.logger.error(
                            "Error shutting down streaming prices client exception=%s",
                            type(exc).__name__,
                        )
                    break
        self._cache.clear()
        with self._quote_cache_lock:
            self._quote_cache.clear()
            self._quote_universe = ()
            self._quote_universe_date = None
        if self._persistent_daily_bars is not None:
            self._persistent_daily_bars.close()
            self._persistent_daily_bars = None
        if self._persistent_option_bars is not None:
            self._persistent_option_bars.close()
            self._persistent_option_bars = None
        self._ready = False

    def health_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "ready": self._ready,
            "timezone": self.timezone,
            "cache": self._cache.stats(),
            "quote_cache": self.quote_cache_details(),
            "streaming_prices_available": bool(
                self.context.get("refdata.api.streaming_prices_available", False)
            ),
        }
        if self._startup_error:
            details["startup_error"] = self._startup_error
        return details

    def configure_quote_precache(
        self, symbol_loader: Callable[[], list[str] | tuple[str, ...]]
    ) -> None:
        """Supply the durable-universe loader used by the quote polling thread."""
        self._quote_universe_loader = symbol_loader

    def configure_daily_bars_repository(self, repository: Any) -> None:
        """Attach the bounded SQL repository used by the multi-symbol route."""
        self._daily_bars_repository = repository
        if not self._bool("refdata.api.daily_bars_cache.enabled", True):
            return
        configured = str(
            self.context.get("refdata.api.daily_bars_cache.path", "") or ""
        ).strip()
        if not configured:
            return
        path = Path(configured).expanduser()
        if not path.is_absolute():
            base = self.context.get("AlgoHome") or os.getenv("ALGO_HOME")
            if not base:
                self.logger.warning(
                    "Persistent daily-bars cache disabled: relative path has no ALGO_HOME"
                )
                return
            path = Path(str(base)).expanduser() / path
        try:
            self._persistent_daily_bars = PersistentDailyBarsCache(path)
        except Exception as exc:
            self.logger.error(
                "Persistent daily-bars cache initialization failed exception=%s",
                type(exc).__name__,
            )

    def configure_event_calendar_repositories(
        self,
        earnings_repository: Any,
        dividends_repository: Any,
    ) -> None:
        """Attach repositories used by the all-symbol event calendar routes."""
        self._earnings_calendar_repository = earnings_repository
        self._dividends_repository = dividends_repository

    def start_quote_precache(self) -> None:
        enabled = self._bool("refdata.api.quote_precache.enabled", False)
        if not enabled or self._quote_precache_thread is not None:
            return
        self._quote_precache_stop.clear()
        self._quote_precache_thread = threading.Thread(
            target=self._quote_precache_loop,
            name="refdata-quote-precache",
            daemon=True,
        )
        self._quote_precache_thread.start()

    def stop_quote_precache(self) -> None:
        self._quote_precache_stop.set()
        thread = self._quote_precache_thread
        if thread is not None and thread.is_alive():
            thread.join(
                timeout=max(
                    2.0,
                    self._float(
                        "refdata.api.quote_precache.shutdown_wait_seconds", 10
                    ),
                )
            )
        self._quote_precache_thread = None

    def refresh_quote_precache(self, *, force: bool = False) -> int:
        """Refresh the configured quote universe once.

        ``force`` is used for the one startup load outside market hours. Closed
        sessions are populated from the latest completed-session SQL marks;
        normal loop iterations return without doing any closed-session work.
        """
        # Use live provider quotes during the session and stored closes outside it.
        during_market_hours = self._market_is_open()
        if not force and not during_market_hours:
            return 0
        symbols = self._quote_precache_symbols()
        if not symbols:
            return 0
        _, current_session = self._current_or_last_session()
        session = (
            current_session
            if during_market_hours
            else self._latest_completed_session()
        )
        session_date = self._session_date(session)
        if during_market_hours:
            self._discard_prior_quote_sessions(session_date)
        records = (
            self._load_provider_quote_records(symbols, session_date)
            if during_market_hours
            else self._sql_day_quote_records(symbols, session)
        )
        loaded_at = self.now()
        self._store_quote_records(
            records,
            loaded_at=loaded_at,
            session_date=session_date,
            during_market_hours=during_market_hours,
        )
        with self._quote_cache_lock:
            self._quote_precache_last_poll = loaded_at
            self._quote_precache_last_error = None
        return len(records)

    def quote_cache_details(self) -> dict[str, Any]:
        with self._quote_cache_lock:
            return {
                "enabled": self._bool("refdata.api.quote_precache.enabled", False),
                "entries": len(self._quote_cache),
                "universe_size": len(self._quote_universe),
                "poll_interval_seconds": self._float(
                    "refdata.api.quote_precache.poll_interval_seconds", 60
                ),
                "thread_running": bool(
                    self._quote_precache_thread
                    and self._quote_precache_thread.is_alive()
                ),
                "last_poll": (
                    self._quote_precache_last_poll.isoformat()
                    if self._quote_precache_last_poll
                    else None
                ),
                "last_error": self._quote_precache_last_error,
            }

    def market_status(self) -> ServiceResult:
        self._ensure_calendar_fresh()
        now = self.now()

        def load() -> CachedPayload:
            trading_day = bool(
                self._invoke(
                    "trading-day status",
                    self.market_data.is_trading_day_today,
                    self.context,
                )
            )
            trading_hours = bool(
                self._invoke(
                    "trading-hours status",
                    self.market_data.is_trading_hours,
                    self.context,
                )
            )
            calendar = self._invoke(
                "current trading session",
                self.market_data.get_trading_calendar,
                self.context,
                self._naive(now),
                self._naive(now),
            )

            sessions = dataframe_records(
                calendar, self.local_timezone, index_field=None
            )
            session = sessions[0] if sessions else None
            payload = {
                "as_of": now.isoformat(),
                "timezone": self.timezone,
                "is_trading_day": trading_day,
                "is_trading_hours": trading_hours,
                "session": session,
            }
            return CachedPayload(payload, 1, now)

        return self._result(("market_status",), 0, load)

    def trading_calendar(self, start_date: date, end_date: date) -> ServiceResult:
        self._validate_range(start_date, end_date, "trading calendar")
        self._validate_max_days(
            start_date,
            end_date,
            self._int("refdata.api.max_calendar_days", 3660),
            "trading calendar",
        )
        self._ensure_calendar_fresh()

        key = ("trading_calendar", start_date.isoformat(), end_date.isoformat())
        ttl = self._float("refdata.api.cache.calendar_ttl_seconds", 21600)

        def load() -> CachedPayload:
            dataframe = self._invoke(
                "trading calendar",
                self.market_data.get_trading_calendar,
                self.context,
                datetime.combine(start_date, time.min),
                datetime.combine(end_date, time.max),
            )
            records = dataframe_records(dataframe, self.local_timezone)
            return CachedPayload(records, len(records), self.now())

        return self._result(key, ttl, load)

    def day_quotes(self, symbols: list[str] | tuple[str, ...]) -> ServiceResult:
        normalized = tuple(dict.fromkeys(self.normalize_symbol(item) for item in symbols))
        if not normalized:
            raise InvalidRequestError("at least one symbol is required")
        maximum = self._int("refdata.api.max_quote_symbols", 1000)
        if len(normalized) > maximum:
            raise InvalidRequestError(
                f"quote request contains {len(normalized)} symbols; maximum is {maximum}"
            )

        trading_today, current_session = self._current_or_last_session()
        during_market_hours = trading_today and self._market_is_open()
        session = (
            current_session
            if during_market_hours
            else self._latest_completed_session()
        )
        session_date = self._session_date(session)
        # Immediately after the closing bell the durable daily bar may not have
        # landed yet.  The batched exchange quote is already complete and gives
        # the GUI today's open/mark without falling through to the small scalar
        # safety net used for genuinely missing historical rows.
        after_market_close = (
            trading_today
            and not during_market_hours
            and aware_datetime(current_session["close"], self.local_timezone)
            <= self.now()
        )
        if during_market_hours:
            self._discard_prior_quote_sessions(session_date)
        # This short response TTL only coalesces concurrent HTTP requests. The
        # per-symbol quote cache below uses the configurable polling interval.
        ttl = self._float("refdata.api.cache.day_quotes_ttl_seconds", 2)
        with self._quote_cache_lock:
            generation = self._quote_cache_generation
        # Include quote generation so a polling refresh invalidates response-cache entries.
        key = (
            "day_quotes",
            normalized,
            session_date.isoformat(),
            trading_today,
            during_market_hours,
            generation,
        )

        def load() -> CachedPayload:
            loaded_at = self.now()
            cached = self._cached_quote_records(
                normalized,
                session_date=session_date,
                during_market_hours=during_market_hours,
            )
            cached_symbols = {str(row["symbol"]) for row in cached}
            # Request only symbols absent from the current session's quote cache.
            missing = tuple(symbol for symbol in normalized if symbol not in cached_symbols)
            loaded: list[dict[str, Any]] = []
            if missing:
                loaded = (
                    self._load_provider_quote_records(missing, session_date)
                    if during_market_hours or after_market_close
                    else self._sql_day_quote_records(missing, session)
                )
                self._store_quote_records(
                    loaded,
                    loaded_at=loaded_at,
                    session_date=session_date,
                    during_market_hours=during_market_hours,
                )
            by_symbol = {
                str(row["symbol"]): row
                for row in (*cached, *loaded)
            }
            records = [by_symbol[symbol] for symbol in normalized if symbol in by_symbol]

            return CachedPayload(records, len(records), loaded_at)

        return self._result(key, ttl, load)

    def intraday_bars(
        self,
        symbol: str,
        market_hours_only: bool = True,
    ) -> ServiceResult:
        """Return one-minute bars for today's or the latest completed session.

        Current trading-day data comes directly from the configured reference-data
        provider. On non-trading days the last session is read from SQL. Neither
        path consults the streaming-prices object.
        """
        symbol = self.normalize_symbol(symbol)
        trading_today, session = self._current_or_last_session()
        session_date = self._session_date(session)
        start, end = self._intraday_session_range(
            session, trading_today, market_hours_only
        )
        ttl = (
            self._float("refdata.api.cache.detailed_intraday_bars_ttl_seconds", 2)
            if trading_today
            else self._float("refdata.api.cache.intraday_bars_ttl_seconds", 300)
        )
        key = (
            "detailed_intraday_bars",
            symbol,
            session_date.isoformat(),
            market_hours_only,
        )

        def load() -> CachedPayload:
            if trading_today:
                provider = self._provider()
                get_bars = getattr(provider, "get_bars", None)
                if not callable(get_bars):
                    raise ProviderCategoryUnsupportedError(
                        "The configured reference-data provider does not support one-minute bars"
                    )
                dataframe = self._invoke(
                    f"provider one-minute bars for {symbol}",
                    get_bars,
                    symbol,
                    start,
                    end,
                    "1min",
                )
            else:
                dataframe = self._invoke(
                    f"SQL one-minute bars for {symbol}",
                    self.market_data.prices_sql,
                    self.context,
                    symbol,
                    start,
                    end,
                    frequency="1min",
                    raise_missing=False,
                    mkt_hours_only=market_hours_only,
                )

            dataframe = self._filter_intraday_frame(dataframe, start, end)
            records = dataframe_records(
                dataframe, self.local_timezone, index_field="timestamp"
            )
            if not records:
                raise IntradayDataUnavailableError(
                    f"One-minute data is unavailable for {symbol} on {session_date.isoformat()}"
                )
            max_rows = self._int("refdata.api.max_detailed_intraday_rows", 2000)
            if len(records) > max_rows:
                raise InvalidRequestError(
                    f"Query returned {len(records)} bars; maximum is {max_rows}"
                )
            return CachedPayload(records, len(records), self.now())

        return self._result(key, ttl, load)

    def bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: Literal["1d", "1m"],
        market_hours_only: bool = False,
        remote_query: bool = False,
    ) -> ServiceResult:
        symbol = self.normalize_symbol(symbol)
        start_local = marketdata_datetime(start, self.local_timezone)
        end_local = marketdata_datetime(end, self.local_timezone)
        self._validate_range(start_local, end_local, "bars")

        if interval == "1m":
            max_days = self._int("refdata.api.max_1m_days", 31)
            frequency = "1min"
            historical_ttl = self._float(
                "refdata.api.cache.intraday_bars_ttl_seconds", 300
            )
        elif interval == "1d":
            max_days = self._int("refdata.api.max_1d_days", 7305)
            frequency = "1day"
            historical_ttl = self._float(
                "refdata.api.cache.daily_bars_ttl_seconds", 900
            )
        else:
            raise InvalidRequestError("interval must be either '1d' or '1m'")
        if remote_query and interval == "1m" and start_local.date() != end_local.date():
            raise InvalidRequestError(
                "One-minute remote-query requests must stay within one local date"
            )

        self._validate_max_days(start_local, end_local, max_days, f"{interval} bars")
        self._ensure_calendar_fresh()

        # Use a short TTL for mutable current-session bars and a longer one for history.
        touches_today = end_local.date() >= self.now().date()
        ttl = (
            self._float("refdata.api.cache.current_bars_ttl_seconds", 2)
            if touches_today
            else historical_ttl
        )
        key = (
            "bars",
            symbol,
            start_local.isoformat(),
            end_local.isoformat(),
            interval,
            market_hours_only,
            remote_query,
        )

        def load() -> CachedPayload:
            price_context = self.context
            dataframe = self._invoke(
                f"{interval} bars for {symbol}",
                self.market_data.prices_sql,
                price_context,
                symbol,
                start_local,
                end_local,
                frequency=frequency,
                raise_missing=False,
                mkt_hours_only=market_hours_only,
                include_current_price=False,
            )
            if interval == "1m" and remote_query and (dataframe is None or dataframe.empty):
                # SQL is authoritative when present. A missing session can be
                # fetched on demand without requiring a bulk ingestion run.
                # Keep this opt-in path bounded to one date and use the same
                # split-adjusted provider as the live intraday endpoint.
                remote_start, remote_end = start_local, end_local
                if market_hours_only:
                    sessions = self.trading_calendar(start_local.date(), end_local.date()).data
                    if sessions:
                        remote_start = max(remote_start, marketdata_datetime(pd.Timestamp(sessions[0]["open"]), self.local_timezone))
                        remote_end = min(remote_end, marketdata_datetime(pd.Timestamp(sessions[0]["close"]), self.local_timezone) - timedelta(microseconds=1))
                    else:
                        remote_end = remote_start - timedelta(microseconds=1)
                if remote_end >= remote_start:
                    get_bars = getattr(self._historical_bars_provider(), "get_bars", None)
                    if not callable(get_bars):
                        raise ProviderCategoryUnsupportedError(
                            "The configured reference-data provider does not support one-minute bars"
                        )
                    dataframe = self._invoke(
                        f"remote one-minute bars for {symbol}",
                        get_bars, symbol, remote_start, remote_end, "1min",
                    )
                    dataframe = self._filter_intraday_frame(dataframe, remote_start, remote_end)
            if interval == "1m":
                coverage = getattr(dataframe, "attrs", {}).get("coverage", {})
                if isinstance(coverage, dict) and (
                    coverage.get("complete") is False or coverage.get("incomplete") is True
                ):
                    safe_coverage = {
                        key: coverage[key]
                        for key in ("available_start", "available_end")
                        if key in coverage
                    }
                    missing_sessions = coverage.get("missing_sessions")
                    if isinstance(missing_sessions, (list, tuple)):
                        safe_coverage["missing_sessions"] = list(missing_sessions[:20])
                        safe_coverage["missing_session_count"] = len(missing_sessions)
                    raise IntradayDataIncompleteError(
                        f"One-minute data coverage is incomplete for {symbol}",
                        details=safe_coverage,
                    )
            records = dataframe_records(
                dataframe, self.local_timezone, index_field="timestamp"
            )
            if interval == "1d" and remote_query and self._daily_bars_missing(
                {symbol: records}, (symbol,), start_local.date(), end_local.date()
            ):
                remote_rows = self._remote_daily_bars(
                    (symbol,), start_local, end_local
                )
                records = self._merge_daily_bar_rows(records, remote_rows, symbol)
                self._cache_remote_daily_bars(
                    (symbol,), remote_rows, start_local.date(), end_local.date()
                )
            if interval == "1m" and not records:
                raise IntradayDataUnavailableError(
                    f"One-minute data is unavailable for {symbol} in the requested range"
                )
            max_rows = self._int("refdata.api.max_bar_rows", 100000)
            if len(records) > max_rows:
                raise InvalidRequestError(
                    f"Query returned {len(records)} bars; maximum is {max_rows}. "
                    "Use a smaller time range."
                )
            return CachedPayload(records, len(records), self.now())

        return self._result(key, ttl, load)

    def bulk_bars(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        *,
        include_current_quote: bool = False,
        remote_query: bool = False,
    ) -> ServiceResult:
        """Load daily bars for many symbols with one bounded SQL operation."""
        normalized = tuple(
            dict.fromkeys(self.normalize_symbol(str(item)) for item in symbols)
        )
        if not normalized:
            raise InvalidRequestError("at least one symbol is required")
        maximum = self._int("refdata.api.max_bulk_bar_symbols", 1000)
        if len(normalized) > maximum:
            raise InvalidRequestError(
                f"bar request contains {len(normalized)} symbols; maximum is {maximum}"
            )
        start_local = marketdata_datetime(start, self.local_timezone)
        end_local = marketdata_datetime(end, self.local_timezone)
        self._validate_range(start_local, end_local, "bars")
        self._validate_max_days(
            start_local,
            end_local,
            self._int("refdata.api.max_1d_days", 7305),
            "1d bars",
        )
        if self._daily_bars_repository is None:
            raise UpstreamUnavailableError(
                "The bulk daily-bars SQL repository is not configured"
            )

        # Scope persistent daily bars to the current cache day.
        cache_date = self.now().date()
        start_date, end_date = start_local.date(), end_local.date()
        cached_rows: list[dict[str, Any]] = []
        covered: set[str] = set()
        if self._persistent_daily_bars is not None:
            try:
                cached_rows, covered = self._persistent_daily_bars.load(
                    normalized,
                    start=start_date,
                    end=end_date,
                    cache_date=cache_date,
                )
            except Exception as exc:
                self.logger.error(
                    "Persistent daily-bars cache read failed exception=%s",
                    type(exc).__name__,
                )
        # Load uncovered symbols together through the bounded daily-bars SQL repository.
        missing = tuple(symbol for symbol in normalized if symbol not in covered)
        loaded_rows: list[dict[str, Any]] = []
        if missing:
            raw_rows = self._invoke(
                f"daily bars for {len(missing)} symbols",
                self._daily_bars_repository.daily_bars,
                missing,
                start=start_date,
                end=end_date,
            )
            loaded_rows = [
                json_value(dict(row), self.local_timezone)
                for row in (raw_rows or [])
            ]
            if self._persistent_daily_bars is not None:
                try:
                    self._persistent_daily_bars.store(
                        missing,
                        loaded_rows,
                        start=start_date,
                        end=end_date,
                        cache_date=cache_date,
                    )
                except Exception as exc:
                    self.logger.error(
                        "Persistent daily-bars cache write failed exception=%s",
                        type(exc).__name__,
                    )

        # Enforce the response row limit after combining cached and newly loaded bars.
        total_rows = len(cached_rows) + len(loaded_rows)
        max_rows = self._int("refdata.api.max_bulk_bar_rows", 1000000)
        if total_rows > max_rows:
            raise InvalidRequestError(
                f"Query returned {total_rows} bars; maximum is {max_rows}. "
                "Use fewer symbols or a smaller time range."
            )

        by_symbol: dict[str, list[dict[str, Any]]] = {
            symbol: [] for symbol in normalized
        }
        for row in (*cached_rows, *loaded_rows):
            symbol = str(row.get("symbol") or "").upper()
            if symbol not in by_symbol:
                continue
            bar = dict(row)
            bar.pop("symbol", None)
            by_symbol[symbol].append(bar)
        for rows in by_symbol.values():
            rows.sort(key=lambda row: str(row.get("timestamp") or ""))

        if remote_query:
            missing_symbols = self._daily_bars_missing(
                by_symbol, normalized, start_date, end_date
            )
            if missing_symbols:
                remote_rows = self._remote_daily_bars(
                    missing_symbols, start_local, end_local
                )
                for symbol in missing_symbols:
                    by_symbol[symbol] = self._merge_daily_bar_rows(
                        by_symbol[symbol], remote_rows, symbol
                    )
                self._cache_remote_daily_bars(
                    missing_symbols, remote_rows, start_date, end_date
                )

        # Append an optional current quote without replacing historical daily bars.
        if include_current_quote:
            quote_result = self.day_quotes(normalized)
            for quote in quote_result.data:
                symbol = str(quote.get("symbol") or "").upper()
                session_date = str(quote.get("session_date") or "")
                if (
                    symbol not in by_symbol
                    or not session_date
                    or session_date < start_date.isoformat()
                    or session_date > end_date.isoformat()
                    or any(
                        str(row.get("timestamp") or "").startswith(session_date)
                        for row in by_symbol[symbol]
                    )
                ):
                    continue
                open_price = self._finite_number(quote.get("open"))
                mark = self._finite_number(quote.get("mark"))
                if open_price is None or mark is None:
                    continue
                by_symbol[symbol].append(
                    {
                        "timestamp": session_date,
                        "open": open_price,
                        "high": max(open_price, mark),
                        "low": min(open_price, mark),
                        "close": mark,
                        "volume": None,
                        "provisional": True,
                    }
                )

        data = [
            {"symbol": symbol, "bars": by_symbol[symbol]}
            for symbol in normalized
        ]
        return ServiceResult(
            data=data,
            count=len(data),
            loaded_at=self.now(),
            cache_state="hit" if not missing else "miss",
            ttl_seconds=self._float(
                "refdata.api.cache.daily_bars_ttl_seconds", 900
            ),
        )

    def _daily_bars_missing(
        self,
        by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> tuple[str, ...]:
        """Return equity symbols whose database history has a requested gap.

        The trading calendar makes this an actual coverage check rather than a
        "did any row exist" check.  Provider work remains an explicit opt-in.
        """
        expected = {
            str(row.get("date") or row.get("timestamp") or "")[:10]
            for row in self.trading_calendar(start, end).data
        }
        missing: list[str] = []
        for symbol in symbols:
            if not self._is_remote_equity(symbol):
                continue
            observed = {
                str(row.get("timestamp") or "")[:10]
                for row in by_symbol.get(symbol, ())
            }
            if not observed or (expected and not expected.issubset(observed)):
                missing.append(symbol)
        return tuple(missing)

    @staticmethod
    def _is_remote_equity(symbol: str) -> bool:
        return not (is_fx(symbol) or is_future(symbol) or is_option(symbol) or is_index(symbol))

    def _remote_daily_bars(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        provider = self._historical_bars_provider()
        get_bars = getattr(provider, "get_bars", None)
        if not callable(get_bars):
            raise UpstreamUnavailableError(
                "The configured reference-data provider does not support daily bars"
            )

        def load_symbol(symbol: str) -> list[dict[str, Any]]:
            dataframe = self._invoke(
                f"remote daily bars for {symbol}",
                get_bars,
                symbol,
                self._naive(start),
                self._naive(end),
                "1day",
            )
            return [
                {"symbol": symbol, **row}
                for row in dataframe_records(
                    dataframe, self.local_timezone, index_field="timestamp"
                )
                if str(row.get("timestamp") or "")
            ]

        normalized = tuple(dict.fromkeys(symbols))
        # Keep remote history workers within the shared provider concurrency limit.
        worker_limit = min(
            self._int("refdata.api.remote_daily_bars_workers", 6),
            self._int("refdata.api.max_concurrent_upstream_calls", 8),
        )
        workers = max(1, min(len(normalized), worker_limit))
        if workers == 1:
            return [row for symbol in normalized for row in load_symbol(symbol)]
        loaded: dict[str, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="refdata-daily"
        ) as executor:
            futures = {
                symbol: executor.submit(load_symbol, symbol)
                for symbol in normalized
            }
            for symbol in normalized:
                loaded[symbol] = futures[symbol].result()
        rows: list[dict[str, Any]] = []
        for symbol in normalized:
            rows.extend(loaded[symbol])
        return rows

    @staticmethod
    def _merge_daily_bar_rows(
        local_rows: Sequence[Mapping[str, Any]],
        remote_rows: Sequence[Mapping[str, Any]],
        symbol: str,
    ) -> list[dict[str, Any]]:
        # Prefer the durable database value when both sources have a bar for a
        # session; remote data is solely a gap repair path.
        merged = {
            str(row.get("timestamp") or ""): {
                key: value for key, value in row.items() if key != "symbol"
            }
            for row in remote_rows
            if str(row.get("symbol") or "").upper() == symbol
        }
        merged.update({str(row.get("timestamp") or ""): dict(row) for row in local_rows})
        return [merged[key] for key in sorted(merged) if key]

    def _cache_remote_daily_bars(
        self,
        symbols: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        start: date,
        end: date,
    ) -> None:
        if not rows or self._persistent_daily_bars is None:
            return
        try:
            self._persistent_daily_bars.store(
                symbols, [dict(row) for row in rows], start=start, end=end,
                cache_date=self.now().date(),
            )
        except Exception as exc:
            self.logger.error(
                "Remote daily-bars cache write failed exception=%s", type(exc).__name__
            )

    def option_market_history(
        self,
        payload: OptionMarketHistoryRequest,
    ) -> OptionMarketHistoryResponse:
        """Load exact-contract daily OHLC, filling only cache coverage gaps."""
        start_date = payload.start_at.date()
        end_date = (payload.end_at - timedelta(microseconds=1)).date()
        now = self.now().astimezone(timezone.utc)
        ttl = timedelta(
            seconds=max(
                1.0,
                self._float(
                    "refdata.api.option_history_cache.ttl_seconds",
                    7 * 24 * 60 * 60,
                ),
            )
        )
        cache = self._option_history_cache()
        provider = self._option_history_provider()
        get_bars = getattr(provider, "get_bars", None)
        if not callable(get_bars):
            raise ProviderCategoryUnsupportedError(
                "The configured reference-data provider does not support exact option bars"
            )

        rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
        cache_hit = False
        provider_loaded = False
        # Serialize exact-contract gap fills to prevent duplicate provider downloads.
        with self._option_history_lock:
            for requested_symbol in payload.symbols:
                occ = parse_occ_symbol(requested_symbol)
                lookup = cache.load(
                    occ.canonical,
                    start=start_date,
                    end=end_date,
                    now=now,
                    ttl=ttl,
                )
                cache_hit = cache_hit or lookup.hit
                for missing_start, missing_end in lookup.missing_ranges:
                    dataframe = self._invoke(
                        f"1d option bars for {occ.canonical}",
                        get_bars,
                        occ.provider,
                        datetime.combine(missing_start, time.min),
                        datetime.combine(missing_end, time.max),
                        "1day",
                    )
                    loaded_rows = self._option_bar_records(
                        dataframe,
                        start=missing_start,
                        end=missing_end,
                    )
                    cache.store(
                        occ.canonical,
                        loaded_rows,
                        start=missing_start,
                        end=missing_end,
                        now=now,
                        ttl=ttl,
                    )
                    provider_loaded = True
                final = cache.load(
                    occ.canonical,
                    start=start_date,
                    end=end_date,
                    now=now,
                    ttl=ttl,
                )
                rows_by_symbol[occ.canonical] = final.rows

        # Daily option bars provide marks without fabricating historical Greeks.
        warnings = (
            ["historical_option_greeks_unavailable"]
            if payload.include_greeks
            else []
        )
        source = type(provider).__name__
        items: list[OptionMarketHistoryItem] = []
        missing_symbols: list[str] = []
        for requested_symbol in payload.symbols:
            occ = parse_occ_symbol(requested_symbol)
            rows = rows_by_symbol.get(occ.canonical, [])
            if not rows:
                missing_symbols.append(occ.canonical)
                continue
            for row in rows:
                at = aware_datetime(row["timestamp"], timezone.utc).astimezone(
                    timezone.utc
                )
                close = self._finite_number(row.get("close"))
                items.append(
                    OptionMarketHistoryItem(
                        requested_symbol=requested_symbol,
                        canonical_occ_symbol=occ.canonical,
                        at=at,
                        session_date=date.fromisoformat(str(row["session_date"])),
                        status="ok",
                        mark=close,
                        last=close,
                        open=self._finite_number(row.get("open")),
                        high=self._finite_number(row.get("high")),
                        low=self._finite_number(row.get("low")),
                        close=close,
                        volume=self._finite_number(row.get("volume")),
                        source=source,
                        warnings=list(warnings),
                    )
                )
        items.sort(key=lambda item: (item.canonical_occ_symbol or "", item.at))
        maximum_rows = self._int(
            "refdata.api.option_market_history_max_rows", 250_000
        )
        if len(items) > maximum_rows:
            raise InvalidRequestError(
                f"Query returned {len(items)} option bars; maximum is {maximum_rows}. "
                "Use fewer symbols or a smaller time range."
            )
        cache_state: Literal["hit", "miss", "bypass"] = (
            "miss" if provider_loaded or not cache_hit else "hit"
        )
        return OptionMarketHistoryResponse(
            data=items,
            missing_symbols=missing_symbols,
            meta=LiveResponseMeta(
                as_of=now,
                source=source,
                cache=cache_state,
                warnings=list(warnings),
            ),
        )

    def _option_history_cache(self) -> PersistentOptionBarsCache:
        cache = self._persistent_option_bars
        if cache is not None:
            return cache
        with self._option_history_lock:
            cache = self._persistent_option_bars
            if cache is None:
                try:
                    cache = PersistentOptionBarsCache(
                        default_option_bars_cache_path(self.context)
                    )
                except Exception as exc:
                    self.logger.error(
                        "Persistent option-history cache initialization failed exception=%s",
                        type(exc).__name__,
                    )
                    raise UpstreamUnavailableError(
                        "The persistent option-history cache is unavailable"
                    ) from exc
                self._persistent_option_bars = cache
        return cache

    def _option_bar_records(
        self,
        dataframe: Any,
        *,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        if dataframe is None or not isinstance(dataframe, pd.DataFrame):
            raise UpstreamUnavailableError(
                "The exact-option bar provider returned an invalid response"
            )
        records = dataframe_records(
            dataframe,
            self.local_timezone,
            index_field="timestamp",
        )
        normalized: list[dict[str, Any]] = []
        for row in records:
            timestamp = row.get("timestamp")
            if timestamp is None:
                continue
            try:
                session_date = pd.Timestamp(timestamp).date()
            except (TypeError, ValueError):
                continue
            if session_date < start or session_date > end:
                continue
            close = self._finite_number(row.get("close"))
            if close is None:
                continue
            # Daily prices become observable after the session has completed.
            available_at = datetime.combine(
                session_date + timedelta(days=1),
                time.min,
                tzinfo=timezone.utc,
            )
            normalized.append(
                {
                    "session_date": session_date,
                    "timestamp": available_at.isoformat(),
                    "open": self._finite_number(row.get("open")),
                    "high": self._finite_number(row.get("high")),
                    "low": self._finite_number(row.get("low")),
                    "close": close,
                    "volume": self._finite_number(row.get("volume")),
                }
            )
        return normalized

    def current_price(
        self,
        symbol: str,
        view: Literal["price", "bar"] = "price",
        *,
        instrument_type: Literal["AUTO", "FUTURE"] = "AUTO",
    ) -> ServiceResult:
        symbol = self.normalize_symbol(symbol)
        future_contract = bool(_CANONICAL_FUTURE_CONTRACT.fullmatch(symbol))
        if instrument_type == "FUTURE" and not future_contract:
            raise InvalidRequestError("contract is not a canonical futures symbol")
        is_future_contract = instrument_type == "FUTURE" or future_contract
        # Translate canonical futures contracts into the broker's slash-prefixed notation.
        provider_symbol = f"/{symbol}" if is_future_contract else symbol
        if view not in {"price", "bar"}:
            raise InvalidRequestError("view must be either 'price' or 'bar'")

        if view == "price":
            cached = self._current_cached_quote(symbol)
            if cached is not None:
                price = scalar_price(cached.record.get("mark"))
                if price is not None and price > 0:
                    return ServiceResult(
                        data={"symbol": symbol, "price": price},
                        count=1,
                        loaded_at=cached.loaded_at,
                        cache_state="hit",
                        ttl_seconds=self._float(
                            "refdata.api.quote_precache.poll_interval_seconds", 60
                        ),
                    )

            # Prefer the normalized day-quote path before the legacy scalar
            # loader. Some providers publish a usable mark for ETFs (for
            # example XLK and TLT) while omitting ``lastPrice``; the legacy
            # loader treats those otherwise valid responses as missing.
            if provider_symbol == symbol:
                quotes = self.day_quotes([symbol])
                row = next(
                    (
                        item
                        for item in quotes.data
                        if str(item.get("symbol") or "").upper() == symbol
                    ),
                    None,
                )
                price = scalar_price(row.get("mark")) if row is not None else None
                if price is not None and price > 0:
                    return ServiceResult(
                        data={"symbol": symbol, "price": price},
                        count=1,
                        loaded_at=quotes.loaded_at,
                        cache_state=quotes.cache_state,
                        ttl_seconds=quotes.ttl_seconds,
                    )

        ttl = self._float("refdata.api.cache.current_price_ttl_seconds", 1)
        key = ("current_price", symbol, view, instrument_type)

        def load() -> CachedPayload:
            if view == "price" and is_future_contract:
                provider = self._provider()
                get_quotes = getattr(provider, "get_quotes", None)
                if callable(get_quotes):
                    raw_quotes = self._invoke(
                        f"current futures quote for {symbol}",
                        get_quotes,
                        [provider_symbol],
                    )
                    rows = self._provider_quote_records(
                        (provider_symbol,), raw_quotes, provider, self.now().date()
                    )
                    mark = scalar_price(rows[0].get("mark")) if rows else None
                    if mark is not None and mark > 0:
                        return CachedPayload(
                            {"symbol": symbol, "price": mark},
                            1,
                            self.now(),
                        )

            value = self._invoke(
                f"current {view} for {symbol}",
                self.market_data.retrieve_current_price,
                self.context,
                provider_symbol,
                bar=view == "bar",
            )
            loaded_at = self.now()

            if view == "price":
                price = scalar_price(value)
                if price is None:
                    raise NotFoundError(f"No current price is available for {symbol}")
                data = {"symbol": symbol, "price": price}
            else:
                records = dataframe_records(
                    value, self.local_timezone, index_field="timestamp"
                )
                if not records:
                    raise NotFoundError(f"No current bar is available for {symbol}")
                data = {"symbol": symbol, "bar": records[-1]}

            return CachedPayload(data, 1, loaded_at)

        return self._result(key, ttl, load)

    def stock_splits(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ServiceResult:
        symbol = self.normalize_symbol(symbol)
        if start_date is not None and end_date is not None:
            self._validate_range(start_date, end_date, "stock splits")

        ttl = self._float("refdata.api.cache.splits_ttl_seconds", 900)
        key = (
            "stock_splits",
            symbol,
            start_date.isoformat() if start_date else None,
            end_date.isoformat() if end_date else None,
        )

        def load() -> CachedPayload:
            # MarketData keeps a process-lifetime split cache. Clear only this
            # symbol when the API TTL expires so newly loaded DB rows are seen.
            with self._split_load_lock:
                internal_cache = getattr(self.market_data, "cached_stock_splits", None)
                if isinstance(internal_cache, dict):
                    internal_cache.pop(symbol, None)
                dataframe = self._invoke(
                    f"stock splits for {symbol}",
                    self.market_data.get_stock_splits,
                    self.context,
                    symbol,
                )

            if dataframe is not None and len(dataframe) > 0:
                working = dataframe
                if start_date is not None:
                    working = working.loc[pd.Timestamp(start_date) :]
                if end_date is not None:
                    working = working.loc[: pd.Timestamp(end_date)]
            else:
                working = dataframe

            records = dataframe_records(
                working, self.local_timezone, index_field="date"
            )
            return CachedPayload(records, len(records), self.now())

        return self._result(key, ttl, load)

    def earnings(self, symbol: str, start_date: date, end_date: date) -> ServiceResult:
        symbol = self.normalize_symbol(symbol)
        self._validate_event_range(start_date, end_date, "earnings")
        ttl = self._float("refdata.api.cache.earnings_ttl_seconds", 900)
        key = ("earnings", symbol, start_date.isoformat(), end_date.isoformat())

        def load() -> CachedPayload:
            dataframe = self._invoke(
                f"earnings calendar for {symbol}",
                self.market_data.get_earnings_calendar,
                self.context,
                symbol,
                datetime.combine(start_date, time.min),
                datetime.combine(end_date, time.max),
            )
            records = dataframe_records(dataframe, self.local_timezone)
            return CachedPayload(records, len(records), self.now())

        return self._result(key, ttl, load)

    def earnings_calendar(self, start_date: date, end_date: date) -> ServiceResult:
        """Return all stored earnings releases in the requested date range."""
        self._validate_event_range(start_date, end_date, "earnings")
        ttl = self._float("refdata.api.cache.earnings_ttl_seconds", 900)
        key = ("earnings_calendar", start_date.isoformat(), end_date.isoformat())

        def load() -> CachedPayload:
            if self._earnings_calendar_repository is None:
                raise DataUnavailableError(
                    "The all-symbol earnings calendar repository is unavailable"
                )
            rows = self._invoke(
                "all-symbol earnings calendar",
                self._earnings_calendar_repository.range_rows,
                start_date,
                end_date,
            )
            records = json_value(rows, self.local_timezone)
            return CachedPayload(records, len(records), self.now())

        return self._result(key, ttl, load)

    def dividends(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        date_type: str = "ex_dividend_date",
    ) -> ServiceResult:
        symbol = self.normalize_symbol(symbol)
        self._validate_event_range(start_date, end_date, "dividends")
        allowed_types = {
            "ex_dividend_date",
            "declaration_date",
            "record_date",
            "pay_date",
        }
        if date_type not in allowed_types:
            raise InvalidRequestError(
                f"date_type must be one of {sorted(allowed_types)}"
            )

        ttl = self._float("refdata.api.cache.dividends_ttl_seconds", 900)
        key = (
            "dividends",
            symbol,
            start_date.isoformat(),
            end_date.isoformat(),
            date_type,
        )

        def load() -> CachedPayload:
            dataframe = self._invoke(
                f"dividends for {symbol}",
                self.market_data.get_dividend_dates,
                self.context,
                symbol,
                datetime.combine(start_date, time.min),
                datetime.combine(end_date, time.max),
                search_criteria=date_type,
            )
            records = dataframe_records(dataframe, self.local_timezone)
            return CachedPayload(records, len(records), self.now())

        return self._result(key, ttl, load)

    def dividend_calendar(
        self,
        start_date: date,
        end_date: date,
        date_type: str = "ex_dividend_date",
    ) -> ServiceResult:
        """Return all stored dividends in the requested date range."""
        self._validate_event_range(start_date, end_date, "dividends")
        allowed_types = {
            "ex_dividend_date",
            "declaration_date",
            "record_date",
            "pay_date",
        }
        if date_type not in allowed_types:
            raise InvalidRequestError(
                f"date_type must be one of {sorted(allowed_types)}"
            )

        ttl = self._float("refdata.api.cache.dividends_ttl_seconds", 900)
        key = (
            "dividend_calendar",
            start_date.isoformat(),
            end_date.isoformat(),
            date_type,
        )

        def load() -> CachedPayload:
            if self._dividends_repository is None:
                raise DataUnavailableError(
                    "The all-symbol dividend calendar repository is unavailable"
                )
            rows = self._invoke(
                "all-symbol dividend calendar",
                self._dividends_repository.range_rows,
                start_date,
                end_date,
                date_type=date_type,
            )
            records = json_value(rows, self.local_timezone)
            return CachedPayload(records, len(records), self.now())

        return self._result(key, ttl, load)

    def fomc(
        self,
        start_date: date,
        end_date: date,
        last_day_only: bool = True,
    ) -> ServiceResult:
        self._validate_event_range(start_date, end_date, "FOMC calendar")
        ttl = self._float("refdata.api.cache.fomc_ttl_seconds", 3600)
        key = (
            "fomc",
            start_date.isoformat(),
            end_date.isoformat(),
            last_day_only,
        )

        def load() -> CachedPayload:
            dates = self._invoke(
                "FOMC calendar",
                self.market_data.get_fomc_dates,
                self.context,
                datetime.combine(start_date, time.min),
                datetime.combine(end_date, time.max),
                last_day_only=last_day_only,
            )
            records = [
                {"date": json_value(value, self.local_timezone)}
                for value in (dates or [])
            ]
            return CachedPayload(records, len(records), self.now())

        return self._result(key, ttl, load)

    def economic_calendar(self, start_date: date, end_date: date) -> ServiceResult:
        self._validate_event_range(start_date, end_date, "economic calendar")
        ttl = self._float("refdata.api.cache.economic_ttl_seconds", 60)
        key = (
            "economic_calendar",
            start_date.isoformat(),
            end_date.isoformat(),
        )

        def load() -> CachedPayload:
            dataframe = self._invoke(
                "economic calendar",
                self.market_data.get_economic_calendar,
                self.context,
                datetime.combine(start_date, time.min),
                datetime.combine(end_date, time.max),
            )
            records = dataframe_records(dataframe, self.local_timezone)
            return CachedPayload(records, len(records), self.now())

        return self._result(key, ttl, load)

    def now(self) -> datetime:
        return aware_datetime(self._now_fn(), self.local_timezone)

    def _provider(self) -> Any:
        if self._refdata_source is not None:
            return self._refdata_source
        with self._provider_lock:
            if self._refdata_source is None:
                configured = str(
                    self.context.get("refdata.providers.prices", "") or ""
                ).strip()
                if configured:
                    self._refdata_source = self._invoke(
                        "configured reference-data provider initialization",
                        self._construct_provider,
                        configured,
                    )
                else:
                    self._refdata_source = self._invoke(
                        "reference-data provider initialization",
                        self.market_data.get_refdata_source,
                        self.context,
                    )
        return self._refdata_source

    def _option_history_provider(self) -> Any:
        """Return the optional exact-option-history provider.

        An empty setting deliberately shares the primary provider so existing
        deployments retain their current behavior and credentials.
        """
        configured = str(
            self.context.get("refdata.providers.option_history", "") or ""
        ).strip()
        if not configured:
            return self._provider()
        if self._option_history_source is not None:
            return self._option_history_source
        with self._option_history_provider_lock:
            if self._option_history_source is None:
                self._option_history_source = self._invoke(
                    "option-history provider initialization",
                    self._construct_provider,
                    configured,
                )
        return self._option_history_source

    def _historical_bars_provider(self) -> Any:
        configured = str(self.context.get("refdata.providers.historical_bars", "") or "").strip()
        if not configured:
            return self._provider()
        with self._historical_bars_provider_lock:
            if self._historical_bars_source is None:
                self._historical_bars_source = self._invoke(
                    "historical bar provider initialization", self._construct_provider, configured,
                )
        return self._historical_bars_source

    def _construct_provider(self, class_path: str) -> Any:
        module_name, separator, class_name = class_path.rpartition(".")
        if not separator or not module_name or not class_name:
            raise ValueError(
                "reference-data provider class must be a fully qualified class path"
            )
        provider_class = getattr(import_module(module_name), class_name)
        return provider_class(self.context)

    def _quote_precache_loop(self) -> None:
        interval = max(
            1.0,
            self._float("refdata.api.quote_precache.poll_interval_seconds", 60),
        )
        first = True
        while not self._quote_precache_stop.is_set():
            try:
                self.refresh_quote_precache(force=first)
            except Exception as exc:
                with self._quote_cache_lock:
                    self._quote_precache_last_error = type(exc).__name__
                self.logger.error(
                    "Quote precache refresh failed exception=%s", type(exc).__name__
                )
            first = False
            if self._quote_precache_stop.wait(interval):
                return

    def _quote_precache_symbols(self) -> tuple[str, ...]:
        today = self.now().date()
        with self._quote_cache_lock:
            if self._quote_universe_date == today and self._quote_universe:
                return self._quote_universe
        loader = self._quote_universe_loader
        if loader is None:
            return ()
        raw_symbols = loader()
        normalized: list[str] = []
        for raw in raw_symbols:
            try:
                symbol = self.normalize_symbol(str(raw))
            except InvalidRequestError:
                self.logger.warning(
                    "Ignoring invalid symbol from quote-precache universe: %r", raw
                )
                continue
            if symbol not in normalized:
                normalized.append(symbol)
        # Bound the daily polling universe before starting provider requests.
        maximum = self._int("refdata.api.quote_precache.max_symbols", 1000)
        result = tuple(normalized[:maximum])
        with self._quote_cache_lock:
            self._quote_universe_date = today
            self._quote_universe = result
        return result

    def _load_provider_quote_records(
        self, symbols: tuple[str, ...], session_date: date
    ) -> list[dict[str, Any]]:
        provider = self._provider()
        get_quotes = getattr(provider, "get_quotes", None)
        if not callable(get_quotes):
            raise ProviderCategoryUnsupportedError(
                "The configured reference-data provider does not support batched quotes"
            )
        # Respect both quote request limits when batching the precache universe.
        batch_size = max(
            1,
            min(
                self._int("refdata.api.quote_precache.batch_size", 500),
                self._int("refdata.api.max_quote_symbols", 1000),
            ),
        )
        records: list[dict[str, Any]] = []
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start : start + batch_size]
            raw_quotes = self._invoke(
                f"day quotes for {len(batch)} symbols",
                get_quotes,
                list(batch),
            )
            records.extend(
                self._provider_quote_records(
                    batch, raw_quotes, provider, session_date
                )
            )
        return records

    def _store_quote_records(
        self,
        records: list[dict[str, Any]],
        *,
        loaded_at: datetime,
        session_date: date,
        during_market_hours: bool,
    ) -> None:
        if not records:
            return
        maximum = max(1, self._int("refdata.api.quote_precache.max_entries", 2000))
        with self._quote_cache_lock:
            for source in records:
                symbol = str(source.get("symbol") or "").upper()
                if not symbol:
                    continue
                self._quote_cache.pop(symbol, None)
                self._quote_cache[symbol] = QuoteCacheEntry(
                    record=dict(source),
                    loaded_at=loaded_at,
                    session_date=session_date,
                    during_market_hours=during_market_hours,
                )
            while len(self._quote_cache) > maximum:
                self._quote_cache.pop(next(iter(self._quote_cache)))
            self._quote_cache_generation += 1

    def _cached_quote_records(
        self,
        symbols: tuple[str, ...],
        *,
        session_date: date,
        during_market_hours: bool,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        stale_symbols: list[str] = []
        now = self.now()
        max_age = max(
            1.0,
            self._float("refdata.api.quote_precache.poll_interval_seconds", 60),
        )
        with self._quote_cache_lock:
            for symbol in symbols:
                entry = self._quote_cache.get(symbol)
                if entry is None:
                    continue
                invalid_for_session = (
                    (
                        not entry.during_market_hours
                        or entry.session_date != session_date
                        or (now - entry.loaded_at).total_seconds() >= max_age
                    )
                    if during_market_hours
                    else (
                        entry.during_market_hours
                        or entry.session_date != session_date
                    )
                )
                if invalid_for_session:
                    stale_symbols.append(symbol)
                    continue
                records.append(dict(entry.record))
            for symbol in stale_symbols:
                self._quote_cache.pop(symbol, None)
            if stale_symbols:
                self._quote_cache_generation += 1
        return records

    def _discard_prior_quote_sessions(self, session_date: date) -> None:
        """Prevent an old close from becoming visible again after a new session opens."""
        with self._quote_cache_lock:
            stale = [
                symbol
                for symbol, entry in self._quote_cache.items()
                if entry.session_date != session_date
                or not entry.during_market_hours
            ]
            for symbol in stale:
                self._quote_cache.pop(symbol, None)
            if stale:
                self._quote_cache_generation += 1

    def _current_cached_quote(self, symbol: str) -> QuoteCacheEntry | None:
        with self._quote_cache_lock:
            entry = self._quote_cache.get(symbol)
        if entry is None:
            return None
        try:
            trading_today, session = self._current_or_last_session()
            during_market_hours = trading_today and self._market_is_open()
            if during_market_hours:
                self._discard_prior_quote_sessions(self._session_date(session))
                with self._quote_cache_lock:
                    entry = self._quote_cache.get(symbol)
                if entry is None:
                    return None
                max_age = max(
                    1.0,
                    self._float(
                        "refdata.api.quote_precache.poll_interval_seconds", 60
                    ),
                )
                if (self.now() - entry.loaded_at).total_seconds() >= max_age:
                    with self._quote_cache_lock:
                        self._quote_cache.pop(symbol, None)
                        self._quote_cache_generation += 1
                    return None
            else:
                completed_session = self._latest_completed_session()
                if (
                    entry.during_market_hours
                    or entry.session_date != self._session_date(completed_session)
                ):
                    return None
        except ServiceError:
            return None
        return entry

    def _market_is_open(self) -> bool:
        return bool(
            self._invoke(
                "trading-hours status",
                self.market_data.is_trading_hours,
                self.context,
            )
        )

    def _current_or_last_session(self) -> tuple[bool, Any]:
        self._ensure_calendar_fresh()
        now = self.now()
        trading_today = bool(
            self._invoke(
                "trading-day status",
                self.market_data.is_trading_day_today,
                self.context,
            )
        )
        calendar = self._invoke(
            "current or last trading session",
            self.market_data.get_trading_calendar,
            self.context,
            self._naive(now - timedelta(days=20)),
            self._naive(now),
        )
        if calendar is None or len(calendar) == 0:
            raise UpstreamUnavailableError("No recent trading session is available")

        candidates = calendar.copy()
        session_dates = pd.to_datetime(candidates["date"]).dt.date
        if trading_today:
            candidates = candidates.loc[session_dates == now.date()]
        else:
            candidates = candidates.loc[session_dates < now.date()]
        if len(candidates) == 0:
            raise UpstreamUnavailableError("No applicable trading session is available")
        candidates = candidates.assign(
            _session_date=pd.to_datetime(candidates["date"]).dt.date
        ).sort_values("_session_date")
        return trading_today, candidates.iloc[-1]

    def _latest_completed_session(self) -> Any:
        """Return the most recent exchange session whose close is observable."""

        self._ensure_calendar_fresh()
        now = self.now()
        calendar = self._invoke(
            "latest completed trading session",
            self.market_data.get_trading_calendar,
            self.context,
            self._naive(now - timedelta(days=20)),
            self._naive(now),
        )
        if calendar is None or len(calendar) == 0:
            raise UpstreamUnavailableError("No recent trading session is available")
        candidates = calendar.copy()
        # Publish only sessions whose exchange close has already occurred.
        completed = candidates.loc[
            candidates["close"].map(
                lambda value: aware_datetime(value, self.local_timezone) <= now
            )
        ]
        if len(completed) == 0:
            raise UpstreamUnavailableError(
                "No completed trading session is available"
            )
        completed = completed.assign(
            _session_close=completed["close"].map(
                lambda value: aware_datetime(value, self.local_timezone)
            )
        ).sort_values("_session_close")
        return completed.iloc[-1]

    def latest_completed_session_close(self) -> datetime:
        """Publish the latest completed close for other RefData services."""

        session = self._latest_completed_session()
        return aware_datetime(
            session["close"], self.local_timezone
        ).astimezone(timezone.utc)

    @staticmethod
    def _session_date(session: Any) -> date:
        value = session["date"]
        return pd.Timestamp(value).date()

    def _intraday_session_range(
        self,
        session: Any,
        trading_today: bool,
        market_hours_only: bool,
    ) -> tuple[datetime, datetime]:
        session_date = self._session_date(session)
        day_start = datetime.combine(session_date, time.min)
        day_end = datetime.combine(session_date, time.max)
        if market_hours_only:
            start = marketdata_datetime(session["open"], self.local_timezone)
            session_end = marketdata_datetime(session["close"], self.local_timezone)
        else:
            start = day_start
            session_end = day_end
        if trading_today:
            end = (
                min(self._naive(self.now()), session_end)
                if market_hours_only
                else self._naive(self.now())
            )
        else:
            end = session_end
        return start, max(start, end)

    def _provider_quote_records(
        self,
        symbols: tuple[str, ...],
        raw_quotes: Any,
        provider: Any,
        session_date: date,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_quotes, Mapping):
            raise UpstreamUnavailableError(
                "The configured reference-data provider returned an invalid quote response"
            )
        lookup = {str(key).upper(): value for key, value in raw_quotes.items()}
        sanitize = getattr(provider, "sanitize_symbol", None)
        records: list[dict[str, Any]] = []
        for symbol in symbols:
            candidates = [symbol]
            if callable(sanitize):
                try:
                    candidates.append(str(sanitize(symbol)).upper())
                except Exception:
                    self.logger.debug("Provider symbol normalization failed for %s", symbol)
            raw = next((lookup.get(item) for item in candidates if item in lookup), None)
            if not isinstance(raw, Mapping):
                continue
            records.append(
                {
                    "symbol": symbol,
                    "session_date": session_date.isoformat(),
                    "open": self._finite_number(raw.get("open")),
                    "bid": self._finite_number(raw.get("bid")),
                    "ask": self._finite_number(raw.get("ask")),
                    "mark": self._finite_number(raw.get("mark")),
                    "quote_time": _normalized_quote_timestamp(
                        raw.get("quote_time")
                    ),
                    "trade_time": _normalized_quote_timestamp(
                        raw.get("trade_time")
                    ),
                    "source_kind": "exchange_quote",
                }
            )
        return records

    def _sql_day_quote_records(
        self, symbols: tuple[str, ...], session: Any
    ) -> list[dict[str, Any]]:
        session_date = self._session_date(session)
        start = datetime.combine(session_date, time.min)
        end = datetime.combine(session_date, time.max)
        session_close = aware_datetime(session["close"], self.local_timezone)
        # Label SQL close quotes with the completed session's actual close time.
        quote_time = session_close.astimezone(timezone.utc).isoformat()
        records: list[dict[str, Any]] = []
        missing: list[str] = []
        for symbol in symbols:
            dataframe = self._invoke(
                f"last trading-day bar for {symbol}",
                self.market_data.prices_sql,
                self.context,
                symbol,
                start,
                end,
                frequency="1day",
                raise_missing=False,
            )
            if dataframe is None or len(dataframe) == 0:
                missing.append(symbol)
                continue
            working = dataframe
            if not isinstance(working.index, pd.RangeIndex):
                matching = pd.to_datetime(working.index).date == session_date
                # Some legacy daily-bar repositories label a completed US
                # session at midnight UTC.  In west-coast local time that can
                # serialize as the following calendar date.  The SQL request
                # is already bounded to the authoritative session, so retain
                # its last returned bar when the label alone does not match.
                if matching.any():
                    working = working.loc[matching]
            if len(working) == 0:
                missing.append(symbol)
                continue
            row = working.iloc[-1]
            mark = self._finite_number(row.get("close"))
            if mark is None or mark <= 0:
                missing.append(symbol)
                continue
            records.append(
                {
                    "symbol": symbol,
                    "session_date": session_date.isoformat(),
                    "open": self._finite_number(row.get("open")),
                    "mark": mark,
                    "quote_time": quote_time,
                    "trade_time": None,
                    "source_kind": "completed_session_mark",
                }
            )
        # Mutual-fund NAVs and a small number of thin securities may not have
        # a local daily-bar row even though the configured provider publishes
        # a latest completed-session price.  Keep this fallback bounded; it is
        # only used after both the shared quote cache and SQL have missed.
        maximum = max(
            0, self._int("refdata.api.max_scalar_quote_fallbacks", 25)
        )
        for symbol in missing[:maximum]:
            try:
                value = self._invoke(
                    f"latest completed-session price for {symbol}",
                    self.market_data.retrieve_current_price,
                    self.context,
                    symbol,
                    bar=False,
                )
                mark = scalar_price(value)
            except Exception as exc:
                self.logger.debug(
                    "Completed-session scalar fallback failed for %s exception=%s",
                    symbol,
                    type(exc).__name__,
                )
                continue
            if mark is None or mark <= 0:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "session_date": session_date.isoformat(),
                    "open": mark,
                    "mark": mark,
                    "quote_time": quote_time,
                    "trade_time": None,
                    "source_kind": "completed_session_mark",
                }
            )
        return records

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        return scalar_price(value)

    def _filter_intraday_frame(
        self, dataframe: Any, start: datetime, end: datetime
    ) -> Any:
        if (
            dataframe is None
            or not isinstance(dataframe, pd.DataFrame)
            or len(dataframe) == 0
        ):
            return dataframe
        if isinstance(dataframe.index, pd.RangeIndex):
            return dataframe
        working = dataframe.copy()
        index = pd.DatetimeIndex(working.index)
        if index.tz is not None:
            start_bound = pd.Timestamp(aware_datetime(start, self.local_timezone)).tz_convert(index.tz)
            end_bound = pd.Timestamp(aware_datetime(end, self.local_timezone)).tz_convert(index.tz)
        else:
            start_bound = pd.Timestamp(start)
            end_bound = pd.Timestamp(end)
        return working.loc[(index >= start_bound) & (index <= end_bound)]

    def normalize_symbol(self, symbol: str) -> str:
        try:
            return normalize_security_symbol(symbol)
        except ValueError:
            raise InvalidRequestError(
                "symbol contains unsupported characters or is too long"
            ) from None

    def _validate_event_range(
        self, start_date: date, end_date: date, label: str
    ) -> None:
        self._validate_range(start_date, end_date, label)
        self._validate_max_days(
            start_date,
            end_date,
            self._int("refdata.api.max_event_days", 3660),
            label,
        )

    @staticmethod
    def _validate_range(start: Any, end: Any, label: str) -> None:
        if end < start:
            raise InvalidRequestError(f"{label} end must be on or after its start")

    @staticmethod
    def _validate_max_days(
        start: date | datetime,
        end: date | datetime,
        maximum_days: int,
        label: str,
    ) -> None:
        if (end - start) > timedelta(days=maximum_days):
            raise InvalidRequestError(
                f"{label} range exceeds the configured maximum of {maximum_days} days"
            )

    def _result(
        self,
        key: tuple[Any, ...],
        ttl_seconds: float,
        loader: Callable[[], CachedPayload],
    ) -> ServiceResult:
        lookup = self._cache.get_or_load(key, ttl_seconds, loader)
        if ttl_seconds <= 0 or not self._bool("refdata.api.cache.enabled", True):
            state: Literal["hit", "miss", "bypass"] = "bypass"
        else:
            state = "hit" if lookup.hit else "miss"
        return ServiceResult(
            data=lookup.value.data,
            count=lookup.value.count,
            loaded_at=lookup.value.loaded_at,
            cache_state=state,
            ttl_seconds=ttl_seconds,
        )

    def _invoke(
        self, label: str, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        acquired = self._upstream_slots.acquire(timeout=self._upstream_wait_seconds)
        if not acquired:
            raise BusyError(
                "The service is at its configured upstream concurrency limit"
            )
        try:
            return function(*args, **kwargs)
        except ServiceError:
            raise
        except ValueError as exc:
            # Provider error text may contain authenticated URLs or connection details.
            self.logger.warning(
                "%s rejected the request exception=%s", label, type(exc).__name__
            )
            raise InvalidRequestError(f"Upstream rejected the request for {label}") from exc
        except Exception as exc:
            self.logger.error(
                "Failed to retrieve %s exception=%s", label, type(exc).__name__
            )
            raise UpstreamUnavailableError(f"Unable to retrieve {label}") from exc
        finally:
            self._upstream_slots.release()

    def _ensure_calendar_fresh(self, force: bool = False) -> None:
        refresh_seconds = self._float("refdata.api.calendar_refresh_seconds", 21600)
        current = monotonic()
        if (
            not force
            and self._calendar_loaded_at > 0
            and current - self._calendar_loaded_at < refresh_seconds
        ):
            return

        # Allow only one request to refresh shared trading-calendar state at a time.
        with self._calendar_refresh_lock:
            current = monotonic()
            if (
                not force
                and self._calendar_loaded_at > 0
                and current - self._calendar_loaded_at < refresh_seconds
            ):
                return

            if hasattr(self.market_data, "trading_calendar_df"):
                self.market_data.trading_calendar_df = None

            now = self.now()
            calendar = self._invoke(
                "trading calendar initialization",
                self.market_data.get_trading_calendar,
                self.context,
                self._naive(now - timedelta(days=7)),
                self._naive(now + timedelta(days=20)),
            )
            if calendar is None or len(calendar) == 0:
                raise UpstreamUnavailableError(
                    "Trading calendar is unavailable for the current date"
                )
            self._calendar_loaded_at = current
            self._cache.delete_where(
                lambda key: (
                    isinstance(key, tuple)
                    and bool(key)
                    and key[0] in {"trading_calendar", "market_status"}
                )
            )

    def _default_now(self) -> datetime:
        common = import_module("algotrading.Common")
        return common.get_datetime(self.context)

    def _resolve_logger(self) -> logging.Logger:
        logger_key = self._context_field("PARAM_LOGGER", "logger")
        logger = self.context.get(logger_key)
        return (
            logger
            if isinstance(logger, logging.Logger)
            else logging.getLogger("algotrading.refdata.api")
        )

    @staticmethod
    def _context_field(name: str, fallback: str) -> str:
        try:
            module = import_module("algotrading.ContextFields")
            return getattr(module, name, fallback)
        except (ImportError, AttributeError):
            return fallback

    def _naive(self, value: datetime) -> datetime:
        return value.astimezone(self.local_timezone).replace(tzinfo=None)

    def _bool(self, key: str, default: bool) -> bool:
        value = self.context.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _int(self, key: str, default: int) -> int:
        return int(self.context.get(key, default))

    def _float(self, key: str, default: float) -> float:
        return float(self.context.get(key, default))
