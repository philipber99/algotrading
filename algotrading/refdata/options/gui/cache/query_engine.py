"""Low-latency Parquet query engine for Time-of-Day strategy profiles."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable, Mapping, NoReturn, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import CacheConfig, CacheQueryConfig
from .diagnostics import LegDiagnostics, ProfileDiagnostics
from .errors import (
    BusyError,
    CacheWarning,
    CoverageUnavailableError,
    InvalidRequestError,
    NotReadyError,
    WARNING_LOW_SAMPLE_COUNT,
    WARNING_MONEYNESS_INTERPOLATED,
    WARNING_PARTIAL_SESSION_COVERAGE,
)
from .model import NormalizeRequest, ProfileRequest, canonical_json
from .pricing import (
    create_solver_registry,
    evaluate_surface_iv_targets,
    price_iv_only_jobs,
)
from .store import CacheStore
from ..pricing_pool import (
    PricingPool,
    PricingPoolBusy,
    PricingPoolFailure,
    PricingPoolTimeout,
)


UTC = timezone.utc
def _model_dict(value: Any) -> dict[str, Any]:
    method = getattr(value, "model_dump", None)
    if callable(method):
        return method(mode="json")
    return value.dict()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class ByteBoundedLRU:
    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self._entries: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._loading: set[str] = set()
        self.hits = 0
        self.misses = 0

    def get_or_load(self, key: str, loader: Callable[[], Any]) -> tuple[Any, bool]:
        with self._condition:
            while True:
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return entry[0], True
                if key not in self._loading:
                    self._loading.add(key)
                    self.misses += 1
                    break
                self._condition.wait()
        try:
            value = loader()
        except BaseException:
            with self._condition:
                self._loading.discard(key)
                self._condition.notify_all()
            raise
        size = len(canonical_json(value))
        with self._condition:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            if size <= self.maximum_bytes:
                self._entries[key] = (value, size)
                self._bytes += size
                while self._bytes > self.maximum_bytes and self._entries:
                    _, (_, removed_size) = self._entries.popitem(last=False)
                    self._bytes -= removed_size
            self._loading.discard(key)
            self._condition.notify_all()
        return value, False

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "maximum_bytes": self.maximum_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "loads_in_progress": len(self._loading),
            }


@dataclass(frozen=True)
class CanonicalLeg:
    index: int
    side: str
    put_call: str
    quantity: float
    strike: float
    expiration_at: datetime
    target_dte_seconds: float
    moneyness: float
    expiration_group: str
    strike_group: str | None
    shared_strike_group: str | None

    @property
    def sign(self) -> float:
        return 1.0 if self.side == "long" else -1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "side": self.side,
            "put_call": self.put_call,
            "quantity": self.quantity,
            "strike": self.strike,
            "expiration_at": self.expiration_at.isoformat(),
            "target_dte_days": self.target_dte_seconds / 86400.0,
            "m": self.moneyness,
            "expiration_group": self.expiration_group,
            "strike_group": self.strike_group,
            "shared_strike_group": self.shared_strike_group,
        }


@dataclass
class PathObservation:
    session_date: date
    session_minute: int
    value: float = 0.0
    entry_value: float | None = None
    normalization: float = 1.0


@dataclass(frozen=True)
class PendingValuation:
    observation_index: int
    leg_index: int
    multiplier: float
    surface_row_index: int
    target_moneyness: float
    job: Mapping[str, Any]


@dataclass(frozen=True)
class ExpirationMatchResult:
    matches: Mapping[str, dict[str, Any]] | None
    reason: str | None = None
    expiration_group: str | None = None
    target_dte_days: float | None = None
    nearest_dte_days: float | None = None
    difference_days: float | None = None


class TimeOfDayQueryEngine:
    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        cache_config: CacheConfig | None = None,
        api_config: CacheQueryConfig | None = None,
        store: CacheStore | None = None,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.monotonic,
        solver_registry: Any | None = None,
        pricing_executor: Any | None = None,
    ) -> None:
        self.context = context
        self.cache_config = cache_config or CacheConfig.from_context(context)
        self.api_config = api_config or CacheQueryConfig.from_context(context)
        self.store = store or CacheStore(self.cache_config)
        self.logger = logger or logging.getLogger(__name__)
        self.clock = clock
        self.solver_registry = solver_registry
        self._pricing_executor = pricing_executor
        self._pricing_pool = (
            None
            if solver_registry is not None
            else PricingPool(
                context,
                workers=self.api_config.resolved_pricing_workers,
                batch_size=self.api_config.pricing_batch_size,
                max_pending_batches=(
                    self.api_config.resolved_max_pending_pricing_batches
                ),
                executor=pricing_executor,
                clock=clock,
            )
        )
        self._manifest: dict[str, Any] | None = None
        self._active_mtime_ns: int | None = None
        self._last_poll = -math.inf
        self._session_rows: list[dict[str, Any]] = []
        self._ready = False
        self._startup_error: str | None = None
        self._lock = threading.RLock()
        self._query_slots = threading.BoundedSemaphore(
            self.api_config.max_concurrent_queries
        )
        self._results = ByteBoundedLRU(self.api_config.result_cache_mb * 1024 * 1024)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def generation_id(self) -> str | None:
        return (
            str(self._manifest["generation_id"]) if self._manifest is not None else None
        )

    def startup(self) -> None:
        try:
            self._start_pricing_pool()
            self._reload(force=True)
            self._ready = True
            self._startup_error = None
        except Exception as exc:
            self._ready = False
            self._startup_error = type(exc).__name__
            raise

    def shutdown(self) -> None:
        if self._pricing_pool is not None:
            self._pricing_pool.shutdown()
        self._pricing_executor = None
        with self._lock:
            self._results.clear()
            self._session_rows = []
            self._manifest = None
            self._ready = False

    def ensure_current(self) -> None:
        now = self.clock()
        if now - self._last_poll < self.api_config.manifest_poll_seconds:
            return
        self._last_poll = now
        try:
            stat = self.store.active_path.stat()
        except FileNotFoundError as exc:
            if self._manifest is None:
                raise NotReadyError("No active cache generation exists") from exc
            return
        if self._active_mtime_ns != stat.st_mtime_ns:
            self._reload(force=False)

    def _reload(self, *, force: bool) -> None:
        with self._lock:
            stat = self.store.active_path.stat()
            if not force and self._active_mtime_ns == stat.st_mtime_ns:
                return
            candidate = self.store.load_active_manifest(validate_files=True)
            if candidate is None:
                raise NotReadyError("No active cache generation exists")
            session_rows = self._load_session_rows(candidate)
            if not session_rows:
                raise NotReadyError("Active cache has no session index rows")
            self._manifest = candidate
            self._session_rows = session_rows
            self._active_mtime_ns = stat.st_mtime_ns
            self._results.clear()
            self._ready = True
            self.logger.info(
                "event=cache_api_reload generation=%s sessions=%s files=%s",
                candidate["generation_id"],
                len({row["session_date"] for row in session_rows}),
                len(candidate.get("files", ())),
            )

    def health(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "ready": self._ready,
            "cache_generation_id": self.generation_id,
            "result_cache": self._results.stats(),
            "pricing_pool": {
                "workers": self.api_config.resolved_pricing_workers,
                "batch_size": self.api_config.pricing_batch_size,
                "maximum_pending_batches": (
                    self.api_config.resolved_max_pending_pricing_batches
                ),
                "pending_batches": (
                    self._pricing_pool.pending_batches
                    if self._pricing_pool is not None
                    else 0
                ),
            },
        }
        if self._manifest is not None:
            details["data_through_session"] = self._manifest.get("data_through_session")
        if self._startup_error:
            details["startup_error"] = self._startup_error
        return details

    def metadata(self) -> dict[str, Any]:
        manifest = self._require_manifest()
        return {
            key: manifest.get(key)
            for key in (
                "cache_schema_version",
                "generation_id",
                "build_id",
                "build_mode",
                "built_at_utc",
                "source_cutoff_utc",
                "data_through_session",
                "data_through_session_by_symbol",
                "source_watermarks_by_symbol",
                "coverage_gaps_by_symbol",
                "base_id",
                "increments",
                "symbols",
                "sample_interval_minutes",
                "max_snapshot_age_seconds",
                "price_bases",
                "quality_status",
                "surface_definition_ids",
                "calculation_definition_ids",
                "validation",
                "extraction_metrics",
            )
        } | {
            "windows": self.windows()["windows"],
            "file_count": len(manifest.get("files", ())),
        }

    def windows(self) -> dict[str, Any]:
        manifest = self._require_manifest()
        configured = list(manifest.get("windows", ()))
        symbols = list(manifest.get("symbols", ()))
        result = []
        for window in configured:
            requested = int(window["sessions"])
            per_symbol: dict[str, int] = {}
            first_dates: list[date] = []
            last_dates: list[date] = []
            for symbol in symbols:
                dates = self._symbol_dates(symbol)
                selected = dates[-requested:]
                per_symbol[symbol] = len(selected)
                if selected:
                    first_dates.append(selected[0])
                    last_dates.append(selected[-1])
            result.append(
                {
                    **dict(window),
                    "available_sessions": min(per_symbol.values(), default=0),
                    "available_sessions_by_symbol": per_symbol,
                    "available_from": max(first_dates).isoformat()
                    if first_dates
                    else None,
                    "available_through": min(last_dates).isoformat()
                    if last_dates
                    else None,
                    "complete": bool(per_symbol)
                    and all(value >= requested for value in per_symbol.values()),
                }
            )
        return {
            "cache_generation_id": manifest["generation_id"],
            "data_through_session": manifest["data_through_session"],
            "default_window": manifest.get("default_window"),
            "windows": result,
        }

    def symbols(self) -> dict[str, Any]:
        manifest = self._require_manifest()
        values = []
        for symbol in manifest.get("symbols", ()):
            rows = [
                row for row in self._session_rows if row["underlying_symbol"] == symbol
            ]
            dates = sorted({row["session_date"] for row in rows})
            values.append(
                {
                    "symbol": symbol,
                    "session_count": len(dates),
                    "first_session": dates[0].isoformat() if dates else None,
                    "last_session": dates[-1].isoformat() if dates else None,
                    "minimum_coverage_fraction": min(
                        (float(row["coverage_fraction"]) for row in rows),
                        default=0.0,
                    ),
                    "surface_coverage": dict(
                        manifest.get("coverage_by_symbol", {}).get(symbol, {})
                    ),
                }
            )
        return {
            "cache_generation_id": manifest["generation_id"],
            "symbols": values,
        }

    def coverage(
        self,
        symbol: str | None = None,
        window: str | None = None,
    ) -> dict[str, Any]:
        manifest = self._require_manifest()
        symbol = symbol.upper() if symbol else None
        rows = [
            row
            for row in self._session_rows
            if symbol is None or row["underlying_symbol"] == symbol
        ]
        if symbol and symbol not in set(manifest.get("symbols", ())):
            raise CoverageUnavailableError(f"Symbol {symbol} is not cached")
        all_dates = sorted({row["session_date"] for row in rows})
        configured_windows = {
            str(item["name"]): dict(item) for item in manifest.get("windows", ())
        }
        if window is not None and window not in configured_windows:
            raise InvalidRequestError(
                f"Unknown window {window!r}; available={sorted(configured_windows)}"
            )
        if window is not None:
            session_limit = int(configured_windows[window]["sessions"])
            selected_dates = sorted({row["session_date"] for row in rows})[
                -session_limit:
            ]
            selected = set(selected_dates)
            rows = [row for row in rows if row["session_date"] in selected]
        dates = sorted({row["session_date"] for row in rows})
        configured_coverage = manifest.get("coverage_by_symbol", {})
        if symbol:
            surface_coverage = dict(configured_coverage.get(symbol, {}))
        else:
            members = list(configured_coverage.values())
            surface_coverage = {
                "minimum_dte_days": min(
                    (float(item["minimum_dte_days"]) for item in members),
                    default=None,
                ),
                "maximum_dte_days": max(
                    (float(item["maximum_dte_days"]) for item in members),
                    default=None,
                ),
                "minimum_moneyness": min(
                    (float(item["minimum_moneyness"]) for item in members),
                    default=None,
                ),
                "maximum_moneyness": max(
                    (float(item["maximum_moneyness"]) for item in members),
                    default=None,
                ),
            }
        actual_surface_coverage = self._surface_bounds_for_dates(
            manifest,
            symbol=symbol,
            session_dates=dates,
        )
        actual_minute_availability = None
        if actual_surface_coverage is not None:
            actual_minute_availability = actual_surface_coverage.pop(
                "session_minute_availability"
            )
            surface_coverage = actual_surface_coverage
        minute_coverage = _session_minute_coverage(
            rows,
            int(manifest["sample_interval_minutes"]),
        )
        per_window_session_counts = {
            name: min(len(all_dates), int(item["sessions"]))
            for name, item in sorted(configured_windows.items())
        }
        return {
            "cache_generation_id": manifest["generation_id"],
            "symbol": symbol,
            "window": window,
            "session_count": len(dates),
            "first_session": dates[0].isoformat() if dates else None,
            "last_session": dates[-1].isoformat() if dates else None,
            "session_minute_min": min(
                (int(row["first_sample_minute"]) for row in rows), default=None
            ),
            "session_minute_max": max(
                (int(row["last_sample_minute"]) for row in rows), default=None
            ),
            **minute_coverage,
            "session_minute_availability": _session_minute_availability(
                rows, int(manifest["sample_interval_minutes"])
            )
            if actual_minute_availability is None
            else actual_minute_availability,
            "moneyness_min": surface_coverage.get("minimum_moneyness"),
            "moneyness_max": surface_coverage.get("maximum_moneyness"),
            "minimum_log_forward_moneyness": surface_coverage.get("minimum_moneyness"),
            "maximum_log_forward_moneyness": surface_coverage.get("maximum_moneyness"),
            "minimum_dte_days": surface_coverage.get("minimum_dte_days"),
            "maximum_dte_days": surface_coverage.get("maximum_dte_days"),
            "price_bases": surface_coverage.get("price_bases", manifest["price_bases"]),
            "surface_coverage": surface_coverage,
            "per_window_session_counts": per_window_session_counts,
            "windows": self.windows()["windows"],
        }

    def normalize(self, request: NormalizeRequest) -> dict[str, Any]:
        legs = self._canonicalize(
            request.underlying_symbol,
            request.as_of_utc,
            request.legs,
        )
        return {
            "underlying_symbol": request.underlying_symbol.upper(),
            "as_of_utc": _utc(request.as_of_utc).isoformat(),
            "strategy_signature": {
                "legs": [item.as_dict() for item in legs],
                "expiration_groups": sorted({item.expiration_group for item in legs}),
            },
            "warnings": [],
        }

    def profile(self, request: ProfileRequest) -> tuple[dict[str, Any], bool]:
        manifest = self._require_manifest()
        try:
            self._validate_profile_request(request, manifest)
        except CoverageUnavailableError as exc:
            diagnostics = self._new_diagnostics(request, ())
            reason = (
                "PRICE_BASIS_UNAVAILABLE"
                if request.price_basis not in set(manifest.get("price_bases", ()))
                else "SYMBOL_NOT_CACHED"
            )
            diagnostics.reject(reason, stage="request_validation")
            self._raise_with_diagnostics(exc, diagnostics)
        request_payload = _model_dict(request)
        key = hashlib.sha256(
            canonical_json(
                {
                    "generation": manifest["generation_id"],
                    "request": request_payload,
                }
            )
        ).hexdigest()
        if not self._query_slots.acquire(timeout=0.05):
            raise BusyError("All profile query slots are in use")
        try:
            deadline = self.clock() + self.api_config.query_timeout_seconds
            result, hit = self._results.get_or_load(
                key,
                lambda: self._profile_uncached(request, deadline=deadline),
            )
            return result, hit
        finally:
            self._query_slots.release()

    def _profile_uncached(
        self,
        request: ProfileRequest,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        manifest = self._require_manifest()
        symbol = request.underlying_symbol.upper()
        legs = self._canonicalize(symbol, request.as_of_utc, request.legs)
        diagnostics = self._new_diagnostics(request, legs)
        try:
            selected_session_rows, window = self._select_session_rows(
                request,
                diagnostics=diagnostics,
            )
        except CoverageUnavailableError as exc:
            self._raise_with_diagnostics(exc, diagnostics)
        selected_dates = sorted({row["session_date"] for row in selected_session_rows})
        self._populate_request_coverage(
            diagnostics,
            selected_session_rows,
            window,
        )
        try:
            table = self._load_surface_table(
                manifest,
                symbol=symbol,
                session_dates=selected_dates,
                price_basis=request.price_basis,
            )
        except CoverageUnavailableError as exc:
            diagnostics.reject(
                "NO_SURFACE_FILE_FOR_WINDOW",
                stage="surface_load",
                example={"symbol": symbol, "window": window["name"]},
            )
            self._raise_with_diagnostics(exc, diagnostics)
        self._check_deadline(deadline)
        diagnostics.set_stage("surface_rows_loaded", table.num_rows)
        if table.num_rows == 0:
            diagnostics.reject(
                "NO_SURFACE_FILE_FOR_WINDOW",
                stage="surface_load",
                example={"symbol": symbol, "window": window["name"]},
            )
            self._raise_with_diagnostics(
                CoverageUnavailableError(
                    f"No surface rows cover {symbol} and the selected window"
                ),
                diagnostics,
            )
        scalar_rows = _surface_scalar_rows(table)
        rows = _deduplicate_surface_rows(scalar_rows)
        diagnostics.set_stage(
            "surface_rows_after_overlay_deduplication",
            len(rows),
        )
        rows = self._filter_surface_rows(
            rows,
            request.cohort.quality_flags_excluded,
            diagnostics=diagnostics,
        )
        diagnostics.set_stage("surface_rows_after_quality_filter", len(rows))
        base_interval = int(manifest["sample_interval_minutes"])
        requested_interval = request.aggregation.interval_minutes
        if requested_interval % base_interval:
            raise InvalidRequestError(
                "aggregation.interval_minutes must be a multiple of the cached "
                f"{base_interval}-minute interval"
            )
        interval_rows = []
        for row in rows:
            if int(row["session_minute"]) % requested_interval == 0:
                interval_rows.append(row)
            else:
                diagnostics.reject(
                    "INTERVAL_EXCLUDED",
                    stage="interval_filter",
                )
        rows = interval_rows
        diagnostics.set_stage("surface_rows_after_interval_filter", len(rows))
        grouped = _group_surface_rows(rows)
        diagnostics.set_stage("candidate_session_minutes", len(grouped))
        warnings: list[CacheWarning] = []
        if request.mode == "comparable_entry":
            observations = self._comparable_observations(
                grouped,
                legs,
                request,
                warnings,
                table,
                deadline,
                diagnostics,
            )
        else:
            observations = self._enter_hold_observations(
                grouped,
                legs,
                request,
                warnings,
                table,
                deadline,
                diagnostics,
            )
        self._check_deadline(deadline)
        diagnostics.set_stage("complete_observations", len(observations))
        diagnostics.set_stage(
            "sessions_with_complete_observations",
            len({item.session_date for item in observations}),
        )
        if not observations:
            if not diagnostics.rejection_counts:
                diagnostics.reject(
                    "NO_COMPLETE_OBSERVATIONS",
                    stage="valuation",
                )
            self._raise_with_diagnostics(
                CoverageUnavailableError(self._coverage_failure_message(diagnostics)),
                diagnostics,
            )
        profile = self._aggregate(
            observations,
            selected_session_count=len(selected_dates),
            request=request,
        )
        diagnostics.set_stage("profile_bins", len(profile))
        if not profile:
            diagnostics.reject(
                "ZERO_ENTRY_VALUE_FOR_NORMALIZATION",
                stage="aggregation",
            )
            self._raise_with_diagnostics(
                CoverageUnavailableError(
                    "No profile bins could be produced from complete observations"
                ),
                diagnostics,
            )
        self._check_deadline(deadline)
        minimum_count = request.cohort.minimum_sample_count
        if any(item["session_count"] < minimum_count for item in profile):
            warnings.append(
                CacheWarning(
                    WARNING_LOW_SAMPLE_COUNT,
                    "One or more profile bins contain fewer sessions than requested",
                    {"minimum_sample_count": minimum_count},
                )
            )
            diagnostics.reject(
                "INSUFFICIENT_SAMPLE_COUNT",
                stage="aggregation",
                amount=sum(item["session_count"] < minimum_count for item in profile),
            )
            diagnostics.add_suggestion(
                "LOWER_MINIMUM_SAMPLE_COUNT",
                "One or more profile bins contain fewer sessions than requested.",
                {
                    "requested_minimum_sample_count": minimum_count,
                    "minimum_observed_sample_count": min(
                        (int(item["session_count"]) for item in profile),
                        default=0,
                    ),
                },
            )
        if any(
            float(row["coverage_fraction"]) < request.cohort.minimum_session_coverage
            for row in selected_session_rows
        ):
            warnings.append(
                CacheWarning(
                    WARNING_PARTIAL_SESSION_COVERAGE,
                    "Some selected sessions have partial cached bin coverage",
                )
            )
        warning_dicts = _deduplicate_warnings(warnings)
        self._populate_suggestions(diagnostics)
        return {
            "cache_generation_id": manifest["generation_id"],
            "data_through_session": manifest["data_through_session"],
            "historical_window": {
                **window,
                "selected_sessions": len(selected_dates),
                "first_selected_session": selected_dates[0].isoformat(),
                "last_selected_session": selected_dates[-1].isoformat(),
            },
            "strategy_signature": {
                "underlying_symbol": symbol,
                "mode": request.mode,
                "value_component": request.value_component,
                "price_basis": request.price_basis,
                "entry_session_minute": request.entry_session_minute,
                "legs": [item.as_dict() for item in legs],
            },
            "warnings": warning_dicts,
            "profile": profile,
            "diagnostics": diagnostics.as_dict(),
        }

    def _select_session_rows(
        self,
        request: ProfileRequest,
        *,
        diagnostics: ProfileDiagnostics | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        manifest = self._require_manifest()
        requested_name = request.cohort.historical_window or str(
            manifest.get("default_window")
        )
        windows = {
            str(item["name"]): dict(item) for item in manifest.get("windows", ())
        }
        if requested_name not in windows:
            raise InvalidRequestError(
                f"Unknown historical_window {requested_name!r}; "
                f"available={sorted(windows)}"
            )
        window = windows[requested_name]
        sessions = int(window["sessions"])
        if sessions > self.api_config.max_lookback_sessions:
            raise InvalidRequestError(
                f"Historical window {requested_name!r} exceeds the API limit"
            )
        symbol = request.underlying_symbol.upper()
        symbol_rows = [
            row for row in self._session_rows if row["underlying_symbol"] == symbol
        ]
        if diagnostics is not None:
            diagnostics.set_stage("symbol_sessions", len(symbol_rows))
        rows = []
        for row in symbol_rows:
            if (
                float(row["coverage_fraction"])
                >= request.cohort.minimum_session_coverage
            ):
                rows.append(row)
            elif diagnostics is not None:
                diagnostics.reject(
                    "SESSION_COVERAGE_BELOW_MINIMUM",
                    stage="session_coverage",
                    example={
                        "session_date": row["session_date"],
                        "coverage_fraction": float(row["coverage_fraction"]),
                    },
                )
        if diagnostics is not None:
            diagnostics.set_stage("sessions_after_coverage_filter", len(rows))
        if request.cohort.days_of_week is not None:
            invalid = set(request.cohort.days_of_week).difference(range(7))
            if invalid:
                raise InvalidRequestError(
                    f"days_of_week must use 0 through 6; invalid={sorted(invalid)}"
                )
            selected_days = set(request.cohort.days_of_week)
            kept = []
            for row in rows:
                if int(row["day_of_week"]) in selected_days:
                    kept.append(row)
                elif diagnostics is not None:
                    diagnostics.reject(
                        "DAY_OF_WEEK_EXCLUDED",
                        stage="day_of_week",
                        example={"session_date": row["session_date"]},
                    )
            rows = kept
        if diagnostics is not None:
            diagnostics.set_stage("sessions_after_day_of_week_filter", len(rows))
        rows = sorted(rows, key=lambda row: row["session_date"])[-sessions:]
        if diagnostics is not None:
            diagnostics.set_stage("sessions_selected_by_window", len(rows))
        if not rows:
            raise CoverageUnavailableError(
                f"No cached sessions are available for {symbol}"
            )
        return rows, window

    def _canonicalize(
        self,
        symbol: str,
        as_of: datetime,
        raw_legs: Sequence[Any],
    ) -> tuple[CanonicalLeg, ...]:
        manifest = self._require_manifest()
        normalized_symbol = symbol.upper()
        if normalized_symbol not in set(manifest.get("symbols", ())):
            raise CoverageUnavailableError(
                f"Symbol {normalized_symbol} is not present in the active cache"
            )
        as_of = _utc(as_of)
        strike_expirations: dict[float, set[str]] = defaultdict(set)
        for leg in raw_legs:
            key = round(float(leg.strike), 8)
            strike_expirations[key].add(_utc(leg.expiration_at).isoformat())
        legs: list[CanonicalLeg] = []
        for index, leg in enumerate(raw_legs):
            expiration = _utc(leg.expiration_at)
            dte = (expiration - as_of).total_seconds()
            if dte <= 0:
                raise InvalidRequestError(
                    f"Leg {index} expiration must be after as_of_utc"
                )
            moneyness = math.log(float(leg.strike) / float(leg.anchor_forward))
            if not math.isfinite(moneyness):
                raise InvalidRequestError(f"Leg {index} moneyness is invalid")
            strike_key = round(float(leg.strike), 8)
            shared_strike_group = leg.shared_strike_group
            if shared_strike_group is None and len(strike_expirations[strike_key]) > 1:
                shared_strike_group = f"implicit:{strike_key:.8f}"
            legs.append(
                CanonicalLeg(
                    index=index,
                    side=leg.side,
                    put_call=leg.put_call,
                    quantity=float(leg.quantity),
                    strike=float(leg.strike),
                    expiration_at=expiration,
                    target_dte_seconds=dte,
                    moneyness=moneyness,
                    expiration_group=expiration.isoformat(),
                    strike_group=leg.strike_group,
                    shared_strike_group=shared_strike_group,
                )
            )
        return tuple(legs)

    def _validate_profile_request(
        self,
        request: ProfileRequest,
        manifest: Mapping[str, Any],
    ) -> None:
        if request.underlying_symbol.upper() not in set(manifest.get("symbols", ())):
            raise CoverageUnavailableError(
                f"Symbol {request.underlying_symbol.upper()} is not present in the active cache"
            )
        if request.mode == "enter_and_hold" and request.entry_session_minute is None:
            raise InvalidRequestError(
                "entry_session_minute is required for enter_and_hold"
            )
        if (
            request.mode == "comparable_entry"
            and request.output_metric != "strategy_value"
        ):
            raise InvalidRequestError(
                "pnl/return output metrics require enter_and_hold mode"
            )
        quantiles = request.aggregation.quantiles
        if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
            raise InvalidRequestError(
                "aggregation.quantiles must contain two increasing values in [0, 1]"
            )
        if request.price_basis not in set(manifest.get("price_bases", ())):
            raise CoverageUnavailableError(
                f"Price basis {request.price_basis!r} is not cached"
            )
        if (
            request.aggregation.normalization == "capital_at_risk"
            and request.mode != "enter_and_hold"
            and request.aggregation.normalization_value is None
        ):
            raise InvalidRequestError(
                "capital_at_risk normalization requires normalization_value "
                "outside enter_and_hold mode"
            )

    def _comparable_observations(
        self,
        grouped: Mapping[tuple[date, int], list[dict[str, Any]]],
        legs: Sequence[CanonicalLeg],
        request: ProfileRequest,
        warnings: list[CacheWarning],
        surface_table: pa.Table,
        deadline: float,
        diagnostics: ProfileDiagnostics | None = None,
    ) -> list[PathObservation]:
        by_day: dict[date, list[int]] = defaultdict(list)
        for session_date, minute in grouped:
            by_day[session_date].append(minute)
        observations: list[PathObservation] = []
        pending: list[PendingValuation] = []
        for session_date, minutes in sorted(by_day.items()):
            anchor_matches: dict[str, dict[str, Any]] | None = None
            for minute in sorted(set(minutes)):
                match_result = _match_expirations(
                    grouped[(session_date, minute)],
                    legs,
                    request.cohort.dte_tolerance_days * 86400.0,
                )
                if match_result.matches is None:
                    self._record_expiration_failure(
                        diagnostics,
                        match_result,
                        legs,
                        session_date=session_date,
                        session_minute=minute,
                    )
                    continue
                matches = match_result.matches
                if diagnostics is not None:
                    diagnostics.funnel["dte_matched_session_minutes"] += 1
                if anchor_matches is None:
                    anchor_matches = matches
                if any(
                    not math.isfinite(float(row["forward_price"]))
                    or float(row["forward_price"]) <= 0.0
                    for row in matches.values()
                ):
                    if diagnostics is not None:
                        diagnostics.reject(
                            "INVALID_FORWARD_PRICE",
                            stage="comparable_matching",
                            example={
                                "session_date": session_date,
                                "session_minute": minute,
                            },
                        )
                    continue
                current_shared_strikes = _shared_strikes(legs, matches)
                anchor_shared_strikes = _shared_strikes(legs, anchor_matches)
                observation = PathObservation(session_date, minute)
                observation.normalization = self._normalization_value(
                    request,
                    legs,
                    entry_value=None,
                )
                observation_index = len(observations)
                observation_pending: list[PendingValuation] = []
                failed_reason: str | None = None
                failed_leg: int | None = None
                failed_example: dict[str, Any] | None = None
                for leg in legs:
                    row = matches[leg.expiration_group]
                    target_moneyness = leg.moneyness
                    if leg.shared_strike_group:
                        target_moneyness = math.log(
                            current_shared_strikes[leg.shared_strike_group]
                            / float(row["forward_price"])
                        )
                    reason = _moneyness_rejection(row, target_moneyness)
                    if diagnostics is not None:
                        leg_diagnostic = diagnostics.leg(leg.index)
                        if leg.shared_strike_group:
                            leg_diagnostic.observe_moneyness(target_moneyness)
                        if reason == "MONEYNESS_BELOW_SURFACE_COVERAGE":
                            leg_diagnostic.moneyness_below_failures += 1
                        elif reason == "MONEYNESS_ABOVE_SURFACE_COVERAGE":
                            leg_diagnostic.moneyness_above_failures += 1
                    if reason is not None:
                        if failed_reason is None:
                            failed_reason = reason
                            failed_leg = leg.index
                            failed_example = {
                                "session_date": session_date,
                                "session_minute": minute,
                                "requested_moneyness": target_moneyness,
                                "available_moneyness_min": row[
                                    "minimum_log_forward_moneyness"
                                ],
                                "available_moneyness_max": row[
                                    "maximum_log_forward_moneyness"
                                ],
                            }
                        continue
                    if request.value_component == "total_value":
                        pricing_anchor = row
                        strike = (
                            current_shared_strikes[leg.shared_strike_group]
                            if leg.shared_strike_group
                            else float(row["forward_price"]) * math.exp(leg.moneyness)
                        )
                    else:
                        pricing_anchor = anchor_matches[leg.expiration_group]
                        strike = (
                            anchor_shared_strikes[leg.shared_strike_group]
                            if leg.shared_strike_group
                            else float(pricing_anchor["forward_price"])
                            * math.exp(leg.moneyness)
                        )
                    observation_pending.append(
                        PendingValuation(
                            observation_index=observation_index,
                            leg_index=leg.index,
                            multiplier=leg.sign * leg.quantity,
                            surface_row_index=int(row["__row_index"]),
                            target_moneyness=target_moneyness,
                            job=_iv_job(pricing_anchor, row, leg, strike),
                        )
                    )
                if failed_reason is not None:
                    if diagnostics is not None:
                        diagnostics.reject(
                            failed_reason,
                            stage="comparable_surface_coverage",
                            leg_index=failed_leg,
                            example=failed_example,
                        )
                    continue
                pending.extend(observation_pending)
                observations.append(observation)
        observations = self._finalize_valuations(
            observations,
            pending,
            surface_table,
            deadline=deadline,
            diagnostics=diagnostics,
        )
        if observations:
            warnings.append(
                CacheWarning(
                    WARNING_MONEYNESS_INTERPOLATED,
                    "Strategy legs were evaluated at exact fitted-surface coordinates",
                )
            )
        return observations

    def _enter_hold_observations(
        self,
        grouped: Mapping[tuple[date, int], list[dict[str, Any]]],
        legs: Sequence[CanonicalLeg],
        request: ProfileRequest,
        warnings: list[CacheWarning],
        surface_table: pa.Table,
        deadline: float,
        diagnostics: ProfileDiagnostics | None = None,
    ) -> list[PathObservation]:
        requested_entry = int(request.entry_session_minute or 0)
        by_day: dict[date, list[int]] = defaultdict(list)
        for session_date, minute in grouped:
            by_day[session_date].append(minute)
        observations: list[PathObservation] = []
        pending: list[PendingValuation] = []
        selected_dates = (
            diagnostics.selected_dates if diagnostics is not None else sorted(by_day)
        )
        for session_date in selected_dates:
            raw_minutes = by_day.get(session_date, [])
            minutes = sorted(set(raw_minutes))
            candidates = [minute for minute in minutes if minute >= requested_entry]
            if not candidates:
                if diagnostics is not None:
                    diagnostics.reject(
                        "NO_SESSION_MINUTE_AT_OR_AFTER_ENTRY",
                        stage="entry_selection",
                        example={
                            "session_date": session_date,
                            "requested_entry_session_minute": requested_entry,
                            "last_available_session_minute": max(minutes, default=None),
                        },
                    )
                continue
            entry_minute = candidates[0]
            if (
                request.path_end_session_minute is not None
                and entry_minute > request.path_end_session_minute
            ):
                if diagnostics is not None:
                    diagnostics.reject(
                        "ENTRY_AFTER_PATH_END",
                        stage="entry_selection",
                        example={
                            "session_date": session_date,
                            "resolved_entry_session_minute": entry_minute,
                            "path_end_session_minute": request.path_end_session_minute,
                        },
                    )
                continue
            if diagnostics is not None:
                diagnostics.record_entry_resolution(requested_entry, entry_minute)
                diagnostics.increment_stage("entry_eligible_sessions")
            entry_match_result = _match_expirations(
                grouped[(session_date, entry_minute)],
                legs,
                request.cohort.dte_tolerance_days * 86400.0,
            )
            if entry_match_result.matches is None:
                self._record_expiration_failure(
                    diagnostics,
                    entry_match_result,
                    legs,
                    session_date=session_date,
                    session_minute=entry_minute,
                )
                continue
            entry_matches = entry_match_result.matches
            if diagnostics is not None:
                diagnostics.funnel["dte_matched_session_minutes"] += 1
            invalid_entry_forward = next(
                (
                    leg
                    for leg in legs
                    if not math.isfinite(
                        float(entry_matches[leg.expiration_group]["forward_price"])
                    )
                    or float(entry_matches[leg.expiration_group]["forward_price"])
                    <= 0.0
                ),
                None,
            )
            if invalid_entry_forward is not None:
                if diagnostics is not None:
                    diagnostics.reject(
                        "INVALID_FORWARD_PRICE",
                        stage="entry_matching",
                        leg_index=invalid_entry_forward.index,
                        example={
                            "session_date": session_date,
                            "session_minute": entry_minute,
                        },
                    )
                continue
            frozen: dict[int, tuple[datetime, float, dict[str, Any]]] = {}
            shared_strikes = _shared_strikes(legs, entry_matches)
            for leg in legs:
                entry_row = entry_matches[leg.expiration_group]
                strike = float(entry_row["forward_price"]) * math.exp(leg.moneyness)
                if leg.shared_strike_group:
                    strike = shared_strikes[leg.shared_strike_group]
                frozen[leg.index] = (
                    entry_row["expiration_at"],
                    strike,
                    entry_row,
                )
            for minute in candidates:
                if (
                    request.path_end_session_minute is not None
                    and minute > request.path_end_session_minute
                ):
                    continue
                if diagnostics is not None:
                    diagnostics.entry["later_path_candidate_points"] += 1
                current_rows = {
                    row["expiration_at"]: row for row in grouped[(session_date, minute)]
                }
                observation = PathObservation(session_date, minute)
                observation_index = len(observations)
                complete = True
                failure_reason: str | None = None
                failure_leg: int | None = None
                failure_example: dict[str, Any] | None = None
                observation_pending: list[PendingValuation] = []
                for leg in legs:
                    expiration, strike, anchor = frozen[leg.index]
                    row = current_rows.get(expiration)
                    if row is None:
                        complete = False
                        failure_reason = "MISSING_LATER_EXPIRATION"
                        failure_leg = leg.index
                        failure_example = {
                            "session_date": session_date,
                            "session_minute": minute,
                            "expiration_at": expiration,
                        }
                        break
                    if (
                        not math.isfinite(float(row["forward_price"]))
                        or float(row["forward_price"]) <= 0
                    ):
                        complete = False
                        failure_reason = "INVALID_FORWARD_PRICE"
                        failure_leg = leg.index
                        failure_example = {
                            "session_date": session_date,
                            "session_minute": minute,
                        }
                        break
                    current_m = math.log(strike / float(row["forward_price"]))
                    reason = _moneyness_rejection(row, current_m)
                    if diagnostics is not None:
                        leg_diagnostic = diagnostics.leg(leg.index)
                        leg_diagnostic.observe_moneyness(current_m)
                        if reason == "MONEYNESS_BELOW_SURFACE_COVERAGE":
                            leg_diagnostic.moneyness_below_failures += 1
                        elif reason == "MONEYNESS_ABOVE_SURFACE_COVERAGE":
                            leg_diagnostic.moneyness_above_failures += 1
                    if reason is not None:
                        complete = False
                        failure_reason = reason
                        failure_leg = leg.index
                        failure_example = {
                            "session_date": session_date,
                            "session_minute": minute,
                            "requested_moneyness": current_m,
                            "available_moneyness_min": row[
                                "minimum_log_forward_moneyness"
                            ],
                            "available_moneyness_max": row[
                                "maximum_log_forward_moneyness"
                            ],
                        }
                        break
                    if request.value_component == "total_value":
                        pricing_anchor = row
                    else:
                        pricing_anchor = anchor
                    observation_pending.append(
                        PendingValuation(
                            observation_index=observation_index,
                            leg_index=leg.index,
                            multiplier=leg.sign * leg.quantity,
                            surface_row_index=int(row["__row_index"]),
                            target_moneyness=current_m,
                            job=_iv_job(pricing_anchor, row, leg, strike),
                        )
                    )
                if complete:
                    pending.extend(observation_pending)
                    observations.append(observation)
                elif diagnostics is not None and failure_reason is not None:
                    diagnostics.reject(
                        failure_reason,
                        stage="later_path",
                        leg_index=failure_leg,
                        example=failure_example,
                    )
        observations = self._finalize_valuations(
            observations,
            pending,
            surface_table,
            deadline=deadline,
            diagnostics=diagnostics,
        )
        if diagnostics is not None:
            diagnostics.entry["later_path_complete_points"] = len(observations)
        final: list[PathObservation] = []
        by_session: dict[date, list[PathObservation]] = defaultdict(list)
        for observation in observations:
            by_session[observation.session_date].append(observation)
        for session_date, values in by_session.items():
            values.sort(key=lambda item: item.session_minute)
            entry_value = values[0].value
            normalizer = self._normalization_value(
                request,
                legs,
                entry_value=entry_value,
            )
            for observation in values:
                observation.entry_value = entry_value
                observation.normalization = normalizer
                final.append(observation)
        if final:
            warnings.append(
                CacheWarning(
                    WARNING_MONEYNESS_INTERPOLATED,
                    "Frozen historical strikes were evaluated at their exact later moneyness",
                )
            )
        return final

    def _finalize_valuations(
        self,
        observations: list[PathObservation],
        pending: Sequence[PendingValuation],
        surface_table: pa.Table,
        *,
        deadline: float,
        diagnostics: ProfileDiagnostics | None = None,
    ) -> list[PathObservation]:
        if not observations or not pending:
            return []
        indices = pa.array(
            [item.surface_row_index for item in pending],
            type=pa.int64(),
        )
        selected_surfaces = surface_table.take(indices)
        targets = np.asarray(
            [item.target_moneyness for item in pending],
            dtype=np.float64,
        )
        volatility = evaluate_surface_iv_targets(selected_surfaces, targets)
        invalid_surface_members = [
            item
            for item, value in zip(pending, volatility)
            if not math.isfinite(float(value)) or float(value) <= 0.0
        ]
        invalid_observations = {
            item.observation_index for item in invalid_surface_members
        }
        if diagnostics is not None:
            for item in invalid_surface_members:
                diagnostics.leg(item.leg_index).surface_value_failures += 1
            for observation_index in sorted(invalid_observations):
                member = next(
                    item
                    for item in invalid_surface_members
                    if item.observation_index == observation_index
                )
                observation = observations[observation_index]
                diagnostics.reject(
                    "SURFACE_VALUE_UNAVAILABLE",
                    stage="surface_evaluation",
                    leg_index=member.leg_index,
                    example={
                        "session_date": observation.session_date,
                        "session_minute": observation.session_minute,
                    },
                )
        valid_members = [
            (item, float(value))
            for item, value in zip(pending, volatility)
            if item.observation_index not in invalid_observations
        ]
        if not valid_members:
            return []
        jobs = [
            {**dict(item.job), "volatility": value} for item, value in valid_members
        ]
        prices = self._price_iv_jobs(jobs, deadline=deadline)
        invalid_price_members = []
        for (item, _), price in zip(valid_members, prices):
            if not math.isfinite(float(price)) or float(price) < 0.0:
                invalid_observations.add(item.observation_index)
                invalid_price_members.append(item)
        if diagnostics is not None:
            already_reported = {
                item.observation_index for item in invalid_surface_members
            }
            for item in invalid_price_members:
                diagnostics.leg(item.leg_index).surface_value_failures += 1
                if item.observation_index in already_reported:
                    continue
                already_reported.add(item.observation_index)
                observation = observations[item.observation_index]
                diagnostics.reject(
                    "OPTION_PRICING_UNAVAILABLE",
                    stage="option_pricing",
                    leg_index=item.leg_index,
                    example={
                        "session_date": observation.session_date,
                        "session_minute": observation.session_minute,
                    },
                )
        for (item, _), price in zip(valid_members, prices):
            if item.observation_index not in invalid_observations:
                observations[item.observation_index].value += item.multiplier * float(
                    price
                )
        return [
            observation
            for index, observation in enumerate(observations)
            if index not in invalid_observations
        ]

    def _normalization_value(
        self,
        request: ProfileRequest,
        legs: Sequence[CanonicalLeg],
        *,
        entry_value: float | None,
    ) -> float:
        explicit = request.aggregation.normalization_value
        if explicit is not None:
            return float(explicit)
        method = request.aggregation.normalization
        if method == "none":
            return 1.0
        if method == "underlying":
            return float(legs[0].strike / math.exp(legs[0].moneyness))
        if method == "spread_width":
            widths = []
            by_expiration: dict[str, list[float]] = defaultdict(list)
            for leg in legs:
                by_expiration[leg.expiration_group].append(leg.strike)
            for strikes in by_expiration.values():
                if len(strikes) >= 2:
                    widths.append(max(strikes) - min(strikes))
            if not widths or max(widths) <= 0:
                raise InvalidRequestError(
                    "spread_width normalization requires two distinct strikes "
                    "in at least one expiration"
                )
            return max(widths)
        if method == "capital_at_risk":
            if entry_value is None or abs(entry_value) <= 1.0e-12:
                raise InvalidRequestError(
                    "capital_at_risk normalization could not derive a nonzero entry value"
                )
            return abs(entry_value)
        raise InvalidRequestError(f"Unsupported normalization method {method!r}")

    def _aggregate(
        self,
        observations: Sequence[PathObservation],
        *,
        selected_session_count: int,
        request: ProfileRequest,
    ) -> list[dict[str, Any]]:
        values_by_minute: dict[int, list[float]] = defaultdict(list)
        anchored_by_minute: dict[int, list[float]] = defaultdict(list)
        for observation in observations:
            value = observation.value
            raw_entry_change: float | None = None
            if request.output_metric != "strategy_value":
                if observation.entry_value is None:
                    continue
                raw_entry_change = observation.value - observation.entry_value
                value = raw_entry_change
                if request.output_metric == "return_from_entry":
                    if abs(observation.entry_value) <= 1.0e-12:
                        continue
                    value /= abs(observation.entry_value)
            value /= observation.normalization
            values_by_minute[observation.session_minute].append(value)
            if (
                request.current_strategy_value is not None
                and request.output_metric == "pnl_from_entry"
                and raw_entry_change is not None
            ):
                anchored_by_minute[observation.session_minute].append(
                    float(request.current_strategy_value) + raw_entry_change
                )
        lower_q, upper_q = request.aggregation.quantiles
        result = []
        for minute, raw_values in sorted(values_by_minute.items()):
            values = np.asarray(raw_values, dtype=np.float64)
            if not len(values):
                continue
            item = {
                "session_minute": minute,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "lower": float(np.quantile(values, lower_q)),
                "upper": float(np.quantile(values, upper_q)),
                # Compatibility aliases for the recommended 10/90 default.
                "p10": float(np.quantile(values, lower_q)),
                "p90": float(np.quantile(values, upper_q)),
                "lower_quantile": lower_q,
                "upper_quantile": upper_q,
                "stddev": float(np.std(values)),
                "session_count": int(len(values)),
                "coverage_fraction": len(values) / selected_session_count,
            }
            if minute in anchored_by_minute:
                anchored = np.asarray(
                    anchored_by_minute[minute],
                    dtype=np.float64,
                )
                item.update(
                    anchored_mean=float(np.mean(anchored)),
                    anchored_median=float(np.median(anchored)),
                    anchored_p10=float(np.quantile(anchored, lower_q)),
                    anchored_p90=float(np.quantile(anchored, upper_q)),
                )
            result.append(item)
        return result

    def _filter_surface_rows(
        self,
        rows: Sequence[dict[str, Any]],
        excluded: Sequence[int | str],
        *,
        diagnostics: ProfileDiagnostics | None = None,
    ) -> list[dict[str, Any]]:
        numeric_mask = 0
        names: set[str] = set()
        for value in excluded:
            if isinstance(value, int):
                numeric_mask |= value
            elif str(value).isdigit():
                numeric_mask |= int(value)
            else:
                names.add(str(value).strip().upper())
        supported = {"FIT_ARBITRAGE", "STALE_SAMPLE"}
        unknown = names.difference(supported)
        if unknown:
            raise InvalidRequestError(
                f"Unknown quality_flags_excluded values: {sorted(unknown)}"
            )
        result = []
        for row in rows:
            if numeric_mask and int(row["quality_flags"]) & numeric_mask:
                if diagnostics is not None:
                    diagnostics.reject(
                        "QUALITY_FLAG_EXCLUDED",
                        stage="quality_filter",
                    )
                continue
            if "FIT_ARBITRAGE" in names and (
                int(row["butterfly_arbitrage_flags"])
                or int(row["calendar_arbitrage_flags"])
            ):
                if diagnostics is not None:
                    diagnostics.reject(
                        "FIT_ARBITRAGE_EXCLUDED",
                        stage="quality_filter",
                    )
                continue
            if "STALE_SAMPLE" in names and float(row["sample_age_seconds"]) >= float(
                self._manifest["max_snapshot_age_seconds"]
            ):
                if diagnostics is not None:
                    diagnostics.reject(
                        "STALE_SAMPLE_EXCLUDED",
                        stage="quality_filter",
                    )
                continue
            result.append(row)
        return result

    def _new_diagnostics(
        self,
        request: ProfileRequest,
        legs: Sequence[CanonicalLeg],
    ) -> ProfileDiagnostics:
        manifest = self._require_manifest()
        symbol = request.underlying_symbol.upper()
        window_name = request.cohort.historical_window or str(
            manifest.get("default_window")
        )
        summary = {
            "underlying_symbol": symbol,
            "historical_window": window_name,
            "mode": request.mode,
            "value_component": request.value_component,
            "price_basis": request.price_basis,
            "as_of_utc": _utc(request.as_of_utc),
            "entry_session_minute": request.entry_session_minute,
            "path_end_session_minute": request.path_end_session_minute,
            "requested_interval_minutes": request.aggregation.interval_minutes,
            "dte_tolerance_days": float(request.cohort.dte_tolerance_days),
            "minimum_session_coverage": float(request.cohort.minimum_session_coverage),
            "minimum_sample_count": int(request.cohort.minimum_sample_count),
            "days_of_week": request.cohort.days_of_week,
            "quality_flags_excluded": list(request.cohort.quality_flags_excluded),
            "leg_count": len(request.legs),
        }
        surface = dict(manifest.get("coverage_by_symbol", {}).get(symbol, {}))
        coverage = {
            "cache_generation_id": manifest.get("generation_id"),
            "data_through_session": manifest.get("data_through_session"),
            "symbol": symbol,
            "window": window_name,
            "available_sessions_before_filters": 0,
            "selected_sessions": 0,
            "first_selected_session": None,
            "last_selected_session": None,
            "sample_interval_minutes": int(
                manifest.get(
                    "sample_interval_minutes",
                    self.cache_config.sample_interval_minutes,
                )
            ),
            "earliest_any_session_minute": None,
            "latest_any_session_minute": None,
            "earliest_common_session_minute": None,
            "latest_common_session_minute": None,
            "minimum_dte_days": surface.get("minimum_dte_days"),
            "maximum_dte_days": surface.get("maximum_dte_days"),
            "minimum_log_forward_moneyness": surface.get("minimum_moneyness"),
            "maximum_log_forward_moneyness": surface.get("maximum_moneyness"),
            "price_bases": list(manifest.get("price_bases", ())),
        }
        tracker = ProfileDiagnostics(summary, coverage)
        tracker.entry["requested_entry_session_minute"] = (
            int(request.entry_session_minute)
            if request.entry_session_minute is not None
            else None
        )
        tracker.legs = [
            LegDiagnostics(
                leg_index=leg.index,
                put_call=leg.put_call,
                side=leg.side,
                target_dte_days=leg.target_dte_seconds / 86400.0,
                dte_tolerance_days=float(request.cohort.dte_tolerance_days),
                target_log_forward_moneyness=leg.moneyness,
                expiration_group=leg.expiration_group,
            )
            for leg in legs
        ]
        for leg in tracker.legs:
            leg.observed_moneyness_min = leg.target_log_forward_moneyness
            leg.observed_moneyness_max = leg.target_log_forward_moneyness
        return tracker

    def _populate_request_coverage(
        self,
        diagnostics: ProfileDiagnostics,
        selected_rows: Sequence[Mapping[str, Any]],
        window: Mapping[str, Any],
    ) -> None:
        dates = sorted({row["session_date"] for row in selected_rows})
        minute_coverage = _session_minute_coverage(
            selected_rows,
            int(diagnostics.cache_coverage["sample_interval_minutes"]),
        )
        diagnostics.selected_dates = list(dates)
        diagnostics.cache_coverage.update(
            {
                "window": str(window["name"]),
                "available_sessions_before_filters": diagnostics.funnel[
                    "symbol_sessions"
                ],
                "selected_sessions": len(dates),
                "first_selected_session": dates[0] if dates else None,
                "last_selected_session": dates[-1] if dates else None,
                **minute_coverage,
            }
        )

    def _record_expiration_failure(
        self,
        diagnostics: ProfileDiagnostics | None,
        result: ExpirationMatchResult,
        legs: Sequence[CanonicalLeg],
        *,
        session_date: date,
        session_minute: int,
    ) -> None:
        if diagnostics is None or result.reason is None:
            return
        failed_legs = [
            leg for leg in legs if leg.expiration_group == result.expiration_group
        ]
        for leg in failed_legs:
            diagnostics.leg(leg.index).observe_dte(
                result.nearest_dte_days,
                failed=True,
            )
        first_leg = failed_legs[0].index if failed_legs else None
        diagnostics.reject(
            result.reason,
            stage="expiration_matching",
            leg_index=first_leg,
            example={
                "session_date": session_date,
                "session_minute": session_minute,
                "requested_dte_days": result.target_dte_days,
                "closest_available_dte_days": result.nearest_dte_days,
                "dte_difference_days": result.difference_days,
            },
        )

    def _raise_with_diagnostics(
        self,
        error: CoverageUnavailableError,
        diagnostics: ProfileDiagnostics,
    ) -> NoReturn:
        diagnostics.cache_coverage["available_sessions_before_filters"] = (
            diagnostics.funnel["symbol_sessions"]
        )
        self._populate_suggestions(diagnostics)
        message = (
            self._coverage_failure_message(diagnostics)
            if diagnostics.dominant_reason is not None
            else error.message
        )
        raise CoverageUnavailableError(message, details=diagnostics.as_dict())

    def _populate_suggestions(self, diagnostics: ProfileDiagnostics) -> None:
        reason = diagnostics.dominant_reason
        coverage = diagnostics.cache_coverage
        summary = diagnostics.request_summary
        if reason == "NO_SESSION_MINUTE_AT_OR_AFTER_ENTRY":
            common = coverage.get("latest_common_session_minute")
            requested = summary.get("entry_session_minute")
            if common is not None and requested is not None and requested > common:
                diagnostics.add_suggestion(
                    "USE_EARLIER_ENTRY_MINUTE",
                    "The requested entry minute is later than the common cached session range.",
                    {
                        "requested_entry_session_minute": requested,
                        "recommended_maximum_entry_session_minute": common,
                    },
                )
        if diagnostics.rejection_counts["DTE_OUTSIDE_TOLERANCE"]:
            nearest = [
                value
                for leg in diagnostics.legs
                for value in (leg.nearest_dte_days_min, leg.nearest_dte_days_max)
                if value is not None
            ]
            required = max(
                (
                    abs(value - leg.target_dte_days)
                    for leg in diagnostics.legs
                    for value in (leg.nearest_dte_days_min, leg.nearest_dte_days_max)
                    if value is not None
                ),
                default=None,
            )
            current = float(summary["dte_tolerance_days"])
            if required is not None and current < required <= 31.0:
                diagnostics.add_suggestion(
                    "INCREASE_DTE_TOLERANCE",
                    "Cached expirations exist outside the requested DTE tolerance.",
                    {"minimum_required_tolerance_days": required},
                )
            if nearest:
                diagnostics.add_suggestion(
                    "USE_SUPPORTED_DTE",
                    "Use a DTE represented by the selected cache window.",
                    {
                        "nearest_dte_days_min": min(nearest),
                        "nearest_dte_days_max": max(nearest),
                    },
                )
        if (
            diagnostics.rejection_counts["MONEYNESS_BELOW_SURFACE_COVERAGE"]
            or diagnostics.rejection_counts["MONEYNESS_ABOVE_SURFACE_COVERAGE"]
        ):
            diagnostics.add_suggestion(
                "USE_STRIKE_WITHIN_SURFACE_COVERAGE",
                "Use strikes whose log-forward moneyness is inside fitted surface coverage.",
                {
                    "minimum_log_forward_moneyness": coverage.get(
                        "minimum_log_forward_moneyness"
                    ),
                    "maximum_log_forward_moneyness": coverage.get(
                        "maximum_log_forward_moneyness"
                    ),
                },
            )
        if diagnostics.rejection_counts["SESSION_COVERAGE_BELOW_MINIMUM"]:
            diagnostics.add_suggestion(
                "LOWER_MINIMUM_SESSION_COVERAGE",
                "Some cached sessions were excluded by the requested coverage threshold.",
                {
                    "requested_minimum_session_coverage": summary[
                        "minimum_session_coverage"
                    ]
                },
            )
        if (
            any(
                diagnostics.rejection_counts[item]
                for item in (
                    "QUALITY_FLAG_EXCLUDED",
                    "FIT_ARBITRAGE_EXCLUDED",
                    "STALE_SAMPLE_EXCLUDED",
                )
            )
            and summary["quality_flags_excluded"]
        ):
            diagnostics.add_suggestion(
                "REMOVE_OR_RELAX_QUALITY_EXCLUSIONS",
                "Requested quality exclusions removed cached surface rows.",
                {"quality_flags_excluded": summary["quality_flags_excluded"]},
            )
        if reason == "PRICE_BASIS_UNAVAILABLE":
            diagnostics.add_suggestion(
                "USE_AVAILABLE_PRICE_BASIS",
                "Use a price basis present in the active cache.",
                {"available_price_bases": coverage["price_bases"]},
            )

    @staticmethod
    def _coverage_failure_message(diagnostics: ProfileDiagnostics) -> str:
        reason = diagnostics.dominant_reason
        if reason == "NO_SESSION_MINUTE_AT_OR_AFTER_ENTRY":
            minute = diagnostics.request_summary.get("entry_session_minute")
            return (
                "No historical sessions contain a cached sample at or after "
                f"entry_session_minute={minute}"
            )
        if reason in {"NO_EXPIRATION_CANDIDATE", "DTE_OUTSIDE_TOLERANCE"}:
            tolerance = diagnostics.request_summary["dte_tolerance_days"]
            return (
                "No complete historical paths matched all strategy-leg DTEs "
                f"within ±{tolerance} days"
            )
        messages = {
            "MONEYNESS_BELOW_SURFACE_COVERAGE": "Requested strategy moneyness is below fitted surface coverage",
            "MONEYNESS_ABOVE_SURFACE_COVERAGE": "Requested strategy moneyness is above fitted surface coverage",
            "MISSING_LATER_EXPIRATION": "A required held expiration is missing from later historical path points",
            "FIT_ARBITRAGE_EXCLUDED": "Quality exclusions removed all usable fitted surfaces",
            "QUALITY_FLAG_EXCLUDED": "Quality exclusions removed all usable fitted surfaces",
            "STALE_SAMPLE_EXCLUDED": "Stale-sample exclusions removed all usable fitted surfaces",
            "NO_SURFACE_FILE_FOR_WINDOW": "No surface data covers the selected symbol and historical window",
            "PRICE_BASIS_UNAVAILABLE": "The requested price basis is unavailable in the active cache",
            "SYMBOL_NOT_CACHED": "The requested symbol is unavailable in the active cache",
        }
        return messages.get(
            reason,
            "No complete historical strategy observations satisfy the requested cache coverage",
        )

    def _load_session_rows(
        self,
        manifest: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        paths = [
            self.store.root / str(record["path"])
            for record in manifest.get("files", ())
            if record.get("dataset") == "session_index"
        ]
        if not paths:
            return []
        allowed_dates = {
            date.fromisoformat(str(value))
            for value in manifest.get("session_dates", ())
        }
        rows: list[dict[str, Any]] = []
        for file_rank, path in enumerate(paths):
            # These are individual immutable files, not dataset roots. Disable
            # Hive partition discovery so parent directories such as
            # ``session_date=2026-07-31`` do not collide with the typed
            # ``session_date`` column stored inside the Parquet file.
            table = pq.read_table(path, partitioning=None)
            for row in table.to_pylist():
                if allowed_dates and row["session_date"] not in allowed_dates:
                    continue
                row["__file_rank"] = file_rank
                rows.append(row)
        # A force-rebuilt increment may supersede the same logical session.
        deduplicated: dict[tuple[str, date], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["underlying_symbol"]), row["session_date"])
            existing = deduplicated.get(key)
            if existing is None or int(row["__file_rank"]) >= int(
                existing["__file_rank"]
            ):
                deduplicated[key] = row
        for row in deduplicated.values():
            row.pop("__file_rank", None)
        return sorted(
            deduplicated.values(),
            key=lambda row: (row["underlying_symbol"], row["session_date"]),
        )

    def _load_surface_table(
        self,
        manifest: Mapping[str, Any],
        *,
        symbol: str,
        session_dates: Sequence[date],
        price_basis: str,
    ) -> pa.Table:
        selected = set(session_dates)
        records = []
        for file_rank, record in enumerate(manifest.get("files", ())):
            if record.get("dataset") != "surface_observations":
                continue
            if record.get("symbol") not in {None, symbol}:
                continue
            minimum = record.get("minimum_session")
            maximum = record.get("maximum_session")
            if (
                minimum
                and maximum
                and not any(
                    str(minimum) <= item.isoformat() <= str(maximum)
                    for item in selected
                )
            ):
                continue
            records.append((file_rank, self.store.root / str(record["path"])))
        if not records:
            raise CoverageUnavailableError(
                f"No surface Parquet files cover {symbol} and the selected window"
            )
        columns = [
            "underlying_symbol",
            "session_date",
            "session_minute",
            "snapshot_id",
            "source_collected_at",
            "sample_age_seconds",
            "expiration_at",
            "price_basis",
            "pricing_method",
            "fit_method",
            "svi_a",
            "svi_b",
            "svi_rho",
            "svi_m",
            "svi_sigma",
            "pchip_k",
            "pchip_total_variance",
            "spot_price",
            "forward_price",
            "time_to_expiry_seconds",
            "risk_free_rate",
            "dividend_yield",
            "quality_flags",
            "butterfly_arbitrage_flags",
            "calendar_arbitrage_flags",
            "minimum_log_forward_moneyness",
            "maximum_log_forward_moneyness",
        ]
        tables: list[pa.Table] = []
        filters = [
            ("underlying_symbol", "=", symbol),
            ("price_basis", "=", price_basis),
            ("session_date", "in", list(session_dates)),
        ]
        for file_rank, path in records:
            table = pq.read_table(
                path,
                columns=columns,
                filters=filters,
                partitioning=None,
            )
            if table.num_rows:
                table = table.append_column(
                    "__file_rank",
                    pa.array([file_rank] * table.num_rows, type=pa.int32()),
                )
                tables.append(table)
        if not tables:
            return pa.Table.from_arrays(
                [pa.array([], type=pa.int32())],
                names=["__file_rank"],
            )
        return pa.concat_tables(tables)

    def _surface_bounds_for_dates(
        self,
        manifest: Mapping[str, Any],
        *,
        symbol: str | None,
        session_dates: Sequence[date],
    ) -> dict[str, Any] | None:
        if not session_dates:
            return None
        selected = set(session_dates)
        tables: list[pa.Table] = []
        columns = [
            "underlying_symbol",
            "session_date",
            "session_minute",
            "expiration_at",
            "time_to_expiry_seconds",
            "minimum_log_forward_moneyness",
            "maximum_log_forward_moneyness",
            "price_basis",
        ]
        filters: list[tuple[str, str, Any]] = [
            ("session_date", "in", list(session_dates))
        ]
        if symbol is not None:
            filters.append(("underlying_symbol", "=", symbol))
        for file_rank, record in enumerate(manifest.get("files", ())):
            if record.get("dataset") != "surface_observations":
                continue
            if symbol is not None and record.get("symbol") not in {None, symbol}:
                continue
            minimum = record.get("minimum_session")
            maximum = record.get("maximum_session")
            if (
                minimum
                and maximum
                and not any(
                    str(minimum) <= item.isoformat() <= str(maximum)
                    for item in selected
                )
            ):
                continue
            table = pq.read_table(
                self.store.root / str(record["path"]),
                columns=columns,
                filters=filters,
                partitioning=None,
            )
            if table.num_rows:
                table = table.append_column(
                    "__file_rank",
                    pa.array([file_rank] * table.num_rows, type=pa.int32()),
                )
                tables.append(table)
        if not tables:
            return None
        rows = _deduplicate_surface_rows(pa.concat_tables(tables).to_pylist())
        dte = (
            np.asarray(
                [float(row["time_to_expiry_seconds"]) for row in rows],
                dtype=np.float64,
            )
            / 86400.0
        )
        minimum_moneyness = np.asarray(
            [float(row["minimum_log_forward_moneyness"]) for row in rows],
            dtype=np.float64,
        )
        maximum_moneyness = np.asarray(
            [float(row["maximum_log_forward_moneyness"]) for row in rows],
            dtype=np.float64,
        )
        availability: dict[int, set[date]] = defaultdict(set)
        for row in rows:
            availability[int(row["session_minute"])].add(row["session_date"])
        return {
            "minimum_dte_days": float(np.nanmin(dte)),
            "maximum_dte_days": float(np.nanmax(dte)),
            "minimum_moneyness": float(np.nanmin(minimum_moneyness)),
            "maximum_moneyness": float(np.nanmax(maximum_moneyness)),
            "price_bases": sorted({str(row["price_basis"]) for row in rows}),
            "session_minute_availability": [
                {
                    "session_minute": minute,
                    "session_count": len(availability[minute]),
                }
                for minute in sorted(availability)
            ],
        }

    def _symbol_dates(self, symbol: str) -> list[date]:
        return sorted(
            {
                row["session_date"]
                for row in self._session_rows
                if row["underlying_symbol"] == symbol
            }
        )

    def _get_solver_registry(self) -> Any:
        with self._lock:
            if self.solver_registry is None:
                self.solver_registry = create_solver_registry(self.context)
            return self.solver_registry

    def _start_pricing_pool(self) -> None:
        if self.solver_registry is not None:
            return
        if self._pricing_pool is None:
            raise BusyError("Pricing process pool is unavailable")
        self._pricing_pool.start()

    def _check_deadline(self, deadline: float) -> None:
        if self.clock() > deadline:
            raise BusyError(
                "Profile query exceeded the configured execution-time limit"
            )

    def _price_iv_jobs(
        self,
        jobs: Sequence[Mapping[str, Any]],
        *,
        deadline: float,
    ) -> np.ndarray:
        if self.solver_registry is not None:
            return price_iv_only_jobs(
                jobs,
                context=self.context,
                registry=self._get_solver_registry(),
            )
        self._start_pricing_pool()
        if self._pricing_pool is None:
            raise BusyError("Pricing process pool is unavailable")
        try:
            return self._pricing_pool.price_jobs(jobs, deadline=deadline)
        except PricingPoolTimeout as exc:
            raise BusyError(
                "Profile query exceeded the configured execution-time limit"
            ) from exc
        except PricingPoolBusy as exc:
            raise BusyError(
                "Profile pricing exceeded the configured pending-work limit"
            ) from exc
        except PricingPoolFailure as exc:
            raise BusyError("Profile pricing is temporarily unavailable") from exc

    def _require_manifest(self) -> dict[str, Any]:
        self.ensure_current()
        if not self._ready or self._manifest is None:
            raise NotReadyError("No validated active cache generation is loaded")
        return self._manifest


def _group_surface_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[date, int], list[dict[str, Any]]]:
    grouped: dict[tuple[date, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["session_date"], int(row["session_minute"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["expiration_at"])
    return grouped


def _surface_scalar_rows(table: pa.Table) -> list[dict[str, Any]]:
    """Materialize only scalar query columns; fitted list arrays stay in Arrow."""

    scalar_names = [
        name
        for name in table.column_names
        if name not in {"pchip_k", "pchip_total_variance"}
    ]
    columns = {name: table[name].to_pylist() for name in scalar_names}
    return [
        {
            **{name: values[index] for name, values in columns.items()},
            "__row_index": index,
        }
        for index in range(table.num_rows)
    ]


def _deduplicate_surface_rows(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply immutable-file overlay order for force-rebuilt increments."""

    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["underlying_symbol"],
            row["session_date"],
            int(row["session_minute"]),
            row["expiration_at"],
            row["price_basis"],
        )
        existing = selected.get(key)
        if existing is None or int(row.get("__file_rank", 0)) >= int(
            existing.get("__file_rank", 0)
        ):
            selected[key] = row
    result = list(selected.values())
    for row in result:
        row.pop("__file_rank", None)
    return result


def _match_expirations(
    rows: Sequence[dict[str, Any]],
    legs: Sequence[CanonicalLeg],
    tolerance_seconds: float,
) -> ExpirationMatchResult:
    targets: dict[str, float] = {}
    for leg in legs:
        targets.setdefault(leg.expiration_group, leg.target_dte_seconds)
    available = list(rows)
    selected: dict[str, dict[str, Any]] = {}
    used: set[datetime] = set()
    for group, target in sorted(targets.items(), key=lambda item: item[1]):
        candidates = [row for row in available if row["expiration_at"] not in used]
        if not candidates:
            return ExpirationMatchResult(
                None,
                "NO_EXPIRATION_CANDIDATE",
                group,
                target / 86400.0,
            )
        choice = min(
            candidates,
            key=lambda row: abs(float(row["time_to_expiry_seconds"]) - target),
        )
        nearest_seconds = float(choice["time_to_expiry_seconds"])
        difference = abs(nearest_seconds - target)
        if difference > tolerance_seconds:
            return ExpirationMatchResult(
                None,
                "DTE_OUTSIDE_TOLERANCE",
                group,
                target / 86400.0,
                nearest_seconds / 86400.0,
                difference / 86400.0,
            )
        selected[group] = choice
        used.add(choice["expiration_at"])
    return ExpirationMatchResult(selected)


def _moneyness_rejection(
    row: Mapping[str, Any],
    target: float,
) -> str | None:
    minimum = float(row["minimum_log_forward_moneyness"])
    maximum = float(row["maximum_log_forward_moneyness"])
    if target < minimum:
        return "MONEYNESS_BELOW_SURFACE_COVERAGE"
    if target > maximum:
        return "MONEYNESS_ABOVE_SURFACE_COVERAGE"
    return None


def _session_minute_coverage(
    rows: Sequence[Mapping[str, Any]],
    interval: int,
) -> dict[str, int | None]:
    del interval  # Bounds come from the exact per-session index values.
    if not rows:
        return {
            "earliest_any_session_minute": None,
            "latest_any_session_minute": None,
            "earliest_common_session_minute": None,
            "latest_common_session_minute": None,
        }
    first = [int(row["first_sample_minute"]) for row in rows]
    last = [int(row["last_sample_minute"]) for row in rows]
    return {
        "earliest_any_session_minute": min(first),
        "latest_any_session_minute": max(last),
        "earliest_common_session_minute": max(first),
        "latest_common_session_minute": min(last),
    }


def _session_minute_availability(
    rows: Sequence[Mapping[str, Any]],
    interval: int,
) -> list[dict[str, int]]:
    if not rows:
        return []
    counts: Counter[int] = Counter()
    for row in rows:
        first = int(row["first_sample_minute"])
        last = int(row["last_sample_minute"])
        start = first + ((-first) % interval)
        for minute in range(start, last + 1, interval):
            counts[minute] += 1
    return [
        {"session_minute": minute, "session_count": int(counts[minute])}
        for minute in sorted(counts)
    ]


def _shared_strikes(
    legs: Sequence[CanonicalLeg],
    matches: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    grouped: dict[str, list[CanonicalLeg]] = defaultdict(list)
    for leg in legs:
        if leg.shared_strike_group:
            grouped[leg.shared_strike_group].append(leg)
    result: dict[str, float] = {}
    for group, members in grouped.items():
        reference = min(members, key=lambda item: item.expiration_at)
        row = matches[reference.expiration_group]
        result[group] = float(row["forward_price"]) * math.exp(reference.moneyness)
    return result


def _iv_job(
    anchor: Mapping[str, Any],
    surface_row: Mapping[str, Any],
    leg: CanonicalLeg,
    strike: float,
) -> dict[str, Any]:
    return {
        "pricing_method": anchor["pricing_method"],
        "snapshot_id": surface_row["snapshot_id"],
        "put_call": leg.put_call,
        "spot": float(anchor["spot_price"]),
        "strike": strike,
        "tenor_seconds": float(anchor["time_to_expiry_seconds"]),
        "rate": float(anchor["risk_free_rate"]),
        "dividend_yield": float(anchor["dividend_yield"]),
        "source_collected_at": anchor["source_collected_at"],
        "expiration_at": anchor["expiration_at"],
    }


def _deduplicate_warnings(
    warnings: Sequence[CacheWarning],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for warning in warnings:
        key = (warning.code, json.dumps(warning.details, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        result.append(warning.as_dict())
    return result
