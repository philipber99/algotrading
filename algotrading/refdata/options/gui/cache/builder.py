"""Daily incremental and weekly full cache-generation orchestration."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .config import CACHE_SCHEMA_VERSION, CacheConfig, chunks
from .errors import ConfigurationError, SourceBacklogError, ValidationError
from .model import SESSION_INDEX_SCHEMA
from .pricing import evaluate_surface_iv_targets
from .queries import ClickHouseExtractor, SymbolSourceWatermark
from .sampling import TradingSession, sample_surface_states, sessions_from_dataframe
from .store import CacheStore, WrittenFile, new_id, utc_now
from .validation import (
    ValidationReport,
    table_from_rows,
    validate_surface_table,
)


UTC = timezone.utc


@dataclass(frozen=True)
class BuildOptions:
    mode: str
    symbols: tuple[str, ...] | None = None
    session_dates: tuple[date, ...] = ()
    history_start: date | None = None
    history_end: date | None = None
    dry_run: bool = False
    force: bool = False
    publish: bool = True


@dataclass
class BuildResult:
    status: str
    mode: str
    build_id: str
    generation_id: str | None = None
    data_through_session: str | None = None
    symbols: tuple[str, ...] = ()
    session_dates: tuple[str, ...] = ()
    row_count: int = 0
    file_count: int = 0
    duration_seconds: float = 0.0
    no_op: bool = False
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "build_id": self.build_id,
            "generation_id": self.generation_id,
            "data_through_session": self.data_through_session,
            "symbols": list(self.symbols),
            "session_dates": list(self.session_dates),
            "row_count": self.row_count,
            "file_count": self.file_count,
            "duration_seconds": self.duration_seconds,
            "no_op": self.no_op,
            "message": self.message,
            "metrics": dict(self.metrics),
            "validation": dict(self.validation),
        }


class CalendarProvider:
    def __init__(
        self,
        context: Mapping[str, Any],
        config: CacheConfig,
        *,
        market_data: Any | None = None,
    ) -> None:
        self.context = context
        self.config = config
        if market_data is None:
            from algotrading.refdata import MarketData as market_data
        self.market_data = market_data

    def sessions(self, start: date, end: date) -> tuple[TradingSession, ...]:
        dataframe = self.market_data.get_trading_calendar(
            self.context,
            datetime.combine(start, datetime.min.time()),
            datetime.combine(end, datetime.max.time()),
        )
        result = sessions_from_dataframe(
            dataframe,
            local_timezone=self.config.local_timezone,
        )
        if not result:
            raise ConfigurationError(
                f"Trading calendar returned no sessions from {start} through {end}"
            )
        return result


class CacheBuilder:
    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        client: Any,
        config: CacheConfig | None = None,
        store: CacheStore | None = None,
        calendar: CalendarProvider | None = None,
        logger: logging.Logger | None = None,
        now_fn: Any = utc_now,
    ) -> None:
        self.context = context
        self.config = config or CacheConfig.from_context(context)
        self.store = store or CacheStore(self.config)
        self.calendar = calendar or CalendarProvider(context, self.config)
        self.logger = logger or logging.getLogger(__name__)
        self.client = client
        self.now_fn = now_fn

    def build(self, options: BuildOptions) -> BuildResult:
        started = time.monotonic()
        mode = options.mode.lower()
        if mode not in {"daily", "weekly"}:
            raise ConfigurationError("Build mode must be daily or weekly")
        build_id = new_id("build")
        symbols = tuple(options.symbols or self.config.symbols)
        unknown = set(symbols).difference(self.config.symbols)
        if unknown:
            raise ConfigurationError(
                f"Symbol override contains symbols outside the publication unit: "
                f"{sorted(unknown)}"
            )
        if (
            symbols != self.config.symbols
            and options.publish
            and not options.dry_run
        ):
            raise ConfigurationError(
                "A symbol subset is a test/benchmark scope and cannot be "
                "published; use --no-publish or --dry-run"
            )
        extractor = ClickHouseExtractor(
            self.client,
            self.context,
            self.config,
            logger=self.logger,
            build_id=build_id,
        )

        with self.store.lock():
            active = self.store.load_active_manifest()
            explicit_end = (
                datetime.combine(
                    options.history_end
                    or (
                        max(options.session_dates)
                        if options.session_dates
                        else None
                    ),
                    datetime.max.time(),
                    tzinfo=UTC,
                )
                if options.history_end is not None or options.session_dates
                else None
            )
            source_watermarks = extractor.source_watermarks(
                symbols,
                end_utc=explicit_end,
            )
            surface_ends = [
                item.latest_surface_at
                for item in source_watermarks.values()
                if item.latest_surface_at is not None
            ]
            if not surface_ends:
                raise SourceBacklogError(
                    "No completed surface state is available for the configured symbols",
                    details={"symbols": list(symbols)},
                )
            source_end = max(surface_ends)
            all_sessions = self._candidate_sessions(
                source_end,
                mode=mode,
                active=active,
                source_watermarks=source_watermarks,
                session_dates=options.session_dates,
                history_start=options.history_start,
                history_end=options.history_end,
            )
            if not all_sessions:
                return self._no_op_result(
                    started=started,
                    mode=mode,
                    build_id=build_id,
                    symbols=symbols,
                    active=active,
                    extractor=extractor,
                )
            if options.session_dates:
                candidate_dates = {item.session_date for item in all_sessions}
                missing_dates = [
                    item for item in options.session_dates if item not in candidate_dates
                ]
                if missing_dates:
                    raise ConfigurationError(
                        "Requested session dates are not completed/available: "
                        f"{missing_dates}"
                    )
            states = extractor.load_surface_states(
                symbols,
                min(item.open_utc for item in all_sessions),
                max(item.close_utc for item in all_sessions),
            )
            sampled = sample_surface_states(
                states,
                all_sessions,
                interval_minutes=self.config.sample_interval_minutes,
                max_age_seconds=self.config.max_snapshot_age_seconds,
            )
            sessions_by_symbol, coverage_gaps = (
                self._available_sessions_by_symbol(
                    all_sessions,
                    sampled,
                    symbols=symbols,
                    source_watermarks=source_watermarks,
                    strict=bool(options.session_dates),
                    report_leading_gaps=(
                        mode == "daily" and active is not None
                    ),
                )
            )
            targets_by_symbol, bootstrap = self._target_sessions_by_symbol(
                mode,
                sessions_by_symbol,
                active,
                options,
            )
            targets = self._session_union(targets_by_symbol)
            if not targets:
                if active is None:
                    raise SourceBacklogError(
                        "No symbols have publishable completed sessions",
                        details={"symbols": coverage_gaps},
                    )
                return self._no_op_result(
                    started=started,
                    mode=mode,
                    build_id=build_id,
                    symbols=symbols,
                    active=active,
                    extractor=extractor,
                )

            plan = {
                "mode": mode,
                "bootstrap": bootstrap,
                "symbols": list(symbols),
                "sessions": [item.session_date.isoformat() for item in targets],
                "sessions_by_symbol": {
                    symbol: [item.session_date.isoformat() for item in sessions]
                    for symbol, sessions in targets_by_symbol.items()
                },
                "source_watermarks_by_symbol": {
                    symbol: watermark.as_dict(
                        tolerance_seconds=self.config.max_snapshot_age_seconds
                    )
                    for symbol, watermark in source_watermarks.items()
                },
                "coverage_gaps_by_symbol": coverage_gaps,
                "source_start_utc": min(
                    item.open_utc for item in targets
                ).isoformat(),
                "source_end_utc": source_end.isoformat(),
                "maximum_window_sessions": self.config.maximum_window_sessions,
            }
            self.logger.info("event=cache_build_plan plan=%s", plan)
            if options.dry_run:
                result = BuildResult(
                    status="ok",
                    mode=mode,
                    build_id=build_id,
                    generation_id=(
                        str(active["generation_id"]) if active is not None else None
                    ),
                    data_through_session=(
                        str(active["data_through_session"])
                        if active is not None
                        else None
                    ),
                    symbols=symbols,
                    session_dates=tuple(
                        item.session_date.isoformat() for item in targets
                    ),
                    no_op=False,
                    message="Dry-run completed; no files were written",
                    metrics={**extractor.metrics.as_dict(), "plan": plan},
                )
                result.duration_seconds = time.monotonic() - started
                return result

            staging = self.store.create_staging(build_id)
            try:
                result = self._execute_build(
                    mode=mode,
                    build_id=build_id,
                    symbols=symbols,
                    sessions=targets,
                    sessions_by_symbol=targets_by_symbol,
                    active=active,
                    bootstrap=bootstrap,
                    publish=options.publish,
                    staging=staging,
                    extractor=extractor,
                    sampled=sampled,
                    source_watermarks=source_watermarks,
                    coverage_gaps=coverage_gaps,
                )
            except Exception:
                self.logger.exception(
                    "event=cache_build status=failed mode=%s build_id=%s staging=%s",
                    mode,
                    build_id,
                    staging,
                )
                # Retain staging by default for diagnosis and resumability
                # inspection. Cleanup removes it only after the grace period.
                raise
        result.duration_seconds = time.monotonic() - started
        return result

    def _no_op_result(
        self,
        *,
        started: float,
        mode: str,
        build_id: str,
        symbols: tuple[str, ...],
        active: Mapping[str, Any] | None,
        extractor: ClickHouseExtractor,
    ) -> BuildResult:
        result = BuildResult(
            status="ok",
            mode=mode,
            build_id=build_id,
            generation_id=str(active["generation_id"]) if active else None,
            data_through_session=(
                str(active["data_through_session"]) if active else None
            ),
            symbols=symbols,
            no_op=True,
            message="Cache is already current; no completed sessions require generation",
            metrics=extractor.metrics.as_dict(),
        )
        result.duration_seconds = time.monotonic() - started
        return result

    def _candidate_sessions(
        self,
        source_end: datetime,
        *,
        mode: str,
        active: Mapping[str, Any] | None,
        source_watermarks: Mapping[str, SymbolSourceWatermark] | None = None,
        session_dates: Sequence[date],
        history_start: date | None,
        history_end: date | None,
    ) -> tuple[TradingSession, ...]:
        if session_dates:
            start = min(session_dates)
            end = max(session_dates)
        else:
            end = min(history_end or source_end.date(), source_end.date())
            if history_start is not None:
                start = history_start
            elif mode == "daily" and active is not None:
                global_data_through = date.fromisoformat(
                    str(active["data_through_session"])
                )
                per_symbol = self._manifest_symbol_data_through(active)
                pending_starts = []
                for symbol, watermark in (source_watermarks or {}).items():
                    cached = date.fromisoformat(
                        per_symbol.get(symbol, global_data_through.isoformat())
                    )
                    if (
                        watermark.latest_chain_at is not None
                        and watermark.latest_chain_at.date() > cached
                    ):
                        pending_starts.append(cached + timedelta(days=1))
                start = min(
                    pending_starts,
                    default=global_data_through + timedelta(days=1),
                )
            else:
                # Calendar lookup may extend farther back, but the source-state
                # query below is sliced to the last configured number of
                # sessions and therefore never scans surplus history.
                start = end - timedelta(
                    days=self.config.maximum_window_sessions * 4 + 31
                )
        if start > end:
            return ()
        now = self.now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        sessions = self.calendar.sessions(start, end)
        completed = tuple(
            session
            for session in sessions
            if session.close_utc <= now
            and session.open_utc <= source_end
            and (not session_dates or session.session_date in set(session_dates))
        )
        if (
            not session_dates
            and history_start is None
            and not (mode == "daily" and active is not None)
        ):
            completed = completed[-self.config.maximum_window_sessions :]
        return completed

    def _available_sessions(
        self,
        sessions: Sequence[TradingSession],
        sampled: Sequence[Any],
        *,
        symbols: Sequence[str],
        strict: bool,
    ) -> tuple[TradingSession, ...]:
        counts: dict[tuple[str, date], int] = defaultdict(int)
        for item in sampled:
            counts[(item.state.underlying_symbol, item.session.session_date)] += 1
        good: set[date] = set()
        for session in sessions:
            expected = len(
                session.sample_targets(self.config.sample_interval_minutes)
            )
            if expected and all(
                counts[(symbol, session.session_date)]
                >= expected * self.config.minimum_session_coverage
                for symbol in symbols
            ):
                good.add(session.session_date)
        if not good:
            raise SourceBacklogError(
                "No completed sessions meet the sampling coverage threshold",
                details={"symbols": list(symbols)},
            )
        first_good = min(good)
        incomplete = [
            item.session_date.isoformat()
            for item in sessions
            if item.session_date >= first_good and item.session_date not in good
        ]
        if incomplete or (strict and len(good) != len(sessions)):
            raise SourceBacklogError(
                "Completed surface snapshots do not meet the sampling coverage threshold",
                details={"incomplete_sessions": incomplete},
            )
        return tuple(item for item in sessions if item.session_date in good)

    def _available_sessions_by_symbol(
        self,
        sessions: Sequence[TradingSession],
        sampled: Sequence[Any],
        *,
        symbols: Sequence[str],
        source_watermarks: Mapping[str, SymbolSourceWatermark],
        strict: bool,
        report_leading_gaps: bool = False,
    ) -> tuple[
        dict[str, tuple[TradingSession, ...]],
        dict[str, dict[str, Any]],
    ]:
        counts: dict[tuple[str, date], int] = defaultdict(int)
        for item in sampled:
            counts[(item.state.underlying_symbol, item.session.session_date)] += 1

        result: dict[str, tuple[TradingSession, ...]] = {}
        gaps: dict[str, dict[str, Any]] = {}
        tolerance = self.config.max_snapshot_age_seconds
        for symbol in symbols:
            watermark = source_watermarks.get(
                symbol,
                SymbolSourceWatermark(symbol, None, None),
            )
            surface_is_current = watermark.surface_is_current(
                tolerance_seconds=tolerance
            )
            eligible = tuple(sessions)
            if surface_is_current and watermark.latest_chain_at is not None:
                cutoff = watermark.latest_chain_at + timedelta(seconds=tolerance)
                eligible = tuple(
                    session for session in sessions if session.close_utc <= cutoff
                )

            good: set[date] = set()
            for session in eligible:
                expected = len(
                    session.sample_targets(self.config.sample_interval_minutes)
                )
                if (
                    expected
                    and counts[(symbol, session.session_date)]
                    >= expected * self.config.minimum_session_coverage
                ):
                    good.add(session.session_date)

            if not good:
                if surface_is_current and not eligible:
                    result[symbol] = ()
                    continue
                gaps[symbol] = {
                    "reason": "no_complete_sessions",
                    "missing_sessions": [
                        session.session_date.isoformat() for session in eligible
                    ],
                    "latest_chain_at": (
                        watermark.latest_chain_at.isoformat()
                        if watermark.latest_chain_at is not None
                        else None
                    ),
                    "latest_surface_at": (
                        watermark.latest_surface_at.isoformat()
                        if watermark.latest_surface_at is not None
                        else None
                    ),
                }
                result[symbol] = ()
                continue

            first_good = min(good)
            missing = [
                session.session_date.isoformat()
                for session in eligible
                if session.session_date not in good
            ]
            trailing_missing = [
                item for item in missing if date.fromisoformat(item) >= first_good
            ]
            leading_missing = [
                item for item in missing if date.fromisoformat(item) < first_good
            ]
            reported_gaps = (
                missing
                if strict
                else [
                    *(leading_missing if report_leading_gaps else []),
                    *trailing_missing,
                ]
            )
            if reported_gaps:
                gaps[symbol] = {
                    "reason": "missing_sampling_coverage",
                    "missing_sessions": reported_gaps,
                    "latest_chain_at": (
                        watermark.latest_chain_at.isoformat()
                        if watermark.latest_chain_at is not None
                        else None
                    ),
                    "latest_surface_at": (
                        watermark.latest_surface_at.isoformat()
                        if watermark.latest_surface_at is not None
                        else None
                    ),
                    "surface_is_current": surface_is_current,
                }
            if strict and missing:
                continue
            result[symbol] = tuple(
                session
                for session in eligible
                if session.session_date in good
            )

        strict_failures = {
            symbol: details
            for symbol, details in gaps.items()
        }
        if strict and strict_failures:
            raise SourceBacklogError(
                "Completed surface snapshots do not meet per-symbol sampling coverage",
                details={"symbols": strict_failures},
            )
        return result, gaps

    @staticmethod
    def _session_union(
        sessions_by_symbol: Mapping[str, Sequence[TradingSession]],
    ) -> tuple[TradingSession, ...]:
        by_date = {
            session.session_date: session
            for sessions in sessions_by_symbol.values()
            for session in sessions
        }
        return tuple(by_date[item] for item in sorted(by_date))

    def _target_sessions(
        self,
        mode: str,
        sessions: Sequence[TradingSession],
        active: Mapping[str, Any] | None,
        options: BuildOptions,
    ) -> tuple[tuple[TradingSession, ...], bool]:
        by_date = {item.session_date: item for item in sessions}
        if options.session_dates:
            missing = [item for item in options.session_dates if item not in by_date]
            if missing:
                raise ConfigurationError(
                    f"Requested session dates are not completed/available: {missing}"
                )
            requested = tuple(by_date[item] for item in options.session_dates)
        else:
            requested = tuple(sessions)

        if mode == "weekly":
            return (
                tuple(requested[-self.config.maximum_window_sessions :]),
                False,
            )

        if active is None:
            return (
                tuple(requested[-self.config.maximum_window_sessions :]),
                True,
            )
        data_through = date.fromisoformat(str(active["data_through_session"]))
        missing = tuple(item for item in requested if item.session_date > data_through)
        if options.force and options.session_dates:
            return requested, False
        return missing, False

    def _target_sessions_by_symbol(
        self,
        mode: str,
        sessions_by_symbol: Mapping[str, Sequence[TradingSession]],
        active: Mapping[str, Any] | None,
        options: BuildOptions,
    ) -> tuple[dict[str, tuple[TradingSession, ...]], bool]:
        bootstrap = mode == "daily" and active is None
        global_data_through = (
            date.fromisoformat(str(active["data_through_session"]))
            if active is not None
            else None
        )
        data_through_by_symbol = self._manifest_symbol_data_through(active)
        requested_dates = set(options.session_dates)
        result: dict[str, tuple[TradingSession, ...]] = {}
        for symbol, sessions in sessions_by_symbol.items():
            symbol_data_through = (
                date.fromisoformat(
                    data_through_by_symbol.get(
                        symbol,
                        global_data_through.isoformat(),
                    )
                )
                if global_data_through is not None
                else None
            )
            requested = tuple(
                session
                for session in sessions
                if not requested_dates or session.session_date in requested_dates
            )
            if mode == "weekly" or bootstrap:
                result[symbol] = requested[-self.config.maximum_window_sessions :]
            elif options.force and options.session_dates:
                result[symbol] = requested
            else:
                result[symbol] = tuple(
                    session
                    for session in requested
                    if symbol_data_through is None
                    or session.session_date > symbol_data_through
                )
        return result, bootstrap

    def _execute_build(
        self,
        *,
        mode: str,
        build_id: str,
        symbols: tuple[str, ...],
        sessions: Sequence[TradingSession],
        sessions_by_symbol: Mapping[str, Sequence[TradingSession]],
        active: Mapping[str, Any] | None,
        bootstrap: bool,
        publish: bool,
        staging: Any,
        extractor: ClickHouseExtractor,
        sampled: Sequence[Any],
        source_watermarks: Mapping[str, SymbolSourceWatermark],
        coverage_gaps: Mapping[str, Mapping[str, Any]],
    ) -> BuildResult:
        generation_id = new_id("generation")
        effective_mode = "weekly" if mode == "weekly" or bootstrap else "daily"
        base_id = (
            new_id("base")
            if effective_mode == "weekly"
            else str(active["base_id"])
        )
        previous_dates = (
            []
            if effective_mode == "weekly" or active is None
            else [str(item) for item in active.get("session_dates", ())]
        )
        target_dates = [item.session_date.isoformat() for item in sessions]
        combined_dates = sorted(set((*previous_dates, *target_dates)))
        # Never expose more history than the largest configured window.
        combined_dates = combined_dates[-self.config.maximum_window_sessions :]
        inherited_files = (
            []
            if effective_mode == "weekly" or active is None
            else [
                dict(item)
                for item in active.get("files", ())
                if str(item.get("maximum_session")) in set(combined_dates)
                or item.get("maximum_session") is None
            ]
        )
        increment_ids = (
            []
            if effective_mode == "weekly" or active is None
            else list(active.get("increments", ()))
        )
        written: list[WrittenFile] = []
        validation_totals = ValidationReport()
        definition_ids: set[int] = set(
            int(item)
            for item in (
                active.get("surface_definition_ids", ())
                if effective_mode == "daily" and active is not None
                else ()
            )
        )
        calculation_ids: set[int] = set(
            int(item)
            for item in (
                active.get("calculation_definition_ids", ())
                if effective_mode == "daily" and active is not None
                else ()
            )
        )
        coverage_by_symbol: dict[str, dict[str, float]] = {
            str(symbol): {
                str(key): float(value)
                for key, value in dict(coverage).items()
            }
            for symbol, coverage in (
                active.get("coverage_by_symbol", {}).items()
                if effective_mode == "daily" and active is not None
                else ()
            )
        }
        month_groups: dict[tuple[int, int], list[TradingSession]] = defaultdict(list)
        for session in sessions:
            month_groups[(session.session_date.year, session.session_date.month)].append(
                session
            )

        for (year, month), month_sessions in sorted(month_groups.items()):
            for symbol_group in chunks(symbols, self.config.chunk_symbol_count):
                for symbol in symbol_group:
                    symbol_month_sessions = tuple(
                        item
                        for item in sessions_by_symbol.get(symbol, ())
                        if item.session_date.year == year
                        and item.session_date.month == month
                    )
                    if not symbol_month_sessions:
                        continue
                    selected_dates = {
                        item.session_date for item in symbol_month_sessions
                    }
                    symbol_sampled = tuple(
                        item
                        for item in sampled
                        if item.state.underlying_symbol == symbol
                        and item.session.session_date in selected_dates
                    )
                    written.extend(
                        self._process_and_write_symbol_chunk(
                            staging=staging,
                            effective_mode=effective_mode,
                            base_id=base_id,
                            build_id=build_id,
                            symbol=symbol,
                            year=year,
                            month=month,
                            source_tables=extractor.iter_surface_tables(
                                symbol_sampled
                            ),
                            sessions=symbol_month_sessions,
                            validation_totals=validation_totals,
                            definition_ids=definition_ids,
                            calculation_ids=calculation_ids,
                            coverage_by_symbol=coverage_by_symbol,
                        )
                    )

        if effective_mode == "daily":
            increment_ids.extend(target_dates)
            increment_ids = sorted(set(increment_ids))
        manifest = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "generation_id": generation_id,
            "build_id": build_id,
            "build_mode": (
                "daily_bootstrap" if bootstrap else mode
            ),
            "built_at_utc": utc_now().isoformat(),
            "source_cutoff_utc": max(
                item.close_utc for item in sessions
            ).isoformat(),
            "data_through_session": combined_dates[-1],
            "base_id": base_id,
            "increments": increment_ids,
            "symbols": list(symbols),
            "session_dates": combined_dates,
            "windows": [item.as_dict() for item in self.config.windows],
            "default_window": self.config.default_window,
            "sample_interval_minutes": self.config.sample_interval_minutes,
            "max_snapshot_age_seconds": self.config.max_snapshot_age_seconds,
            "price_bases": list(self.config.price_bases),
            "surface_definition_ids": sorted(definition_ids),
            "calculation_definition_ids": sorted(calculation_ids),
            "coverage_by_symbol": coverage_by_symbol,
            "data_through_session_by_symbol": self._symbol_data_through(
                symbols,
                sessions_by_symbol,
                active if effective_mode == "daily" else None,
            ),
            "source_watermarks_by_symbol": {
                symbol: watermark.as_dict(
                    tolerance_seconds=self.config.max_snapshot_age_seconds
                )
                for symbol, watermark in source_watermarks.items()
            },
            "coverage_gaps_by_symbol": self._merge_coverage_gaps(
                active if effective_mode == "daily" else None,
                coverage_gaps,
                sessions_by_symbol,
            ),
            "quality_status": "ok",
            "validation": validation_totals.as_dict(),
            "extraction_metrics": extractor.metrics.as_dict(),
            "files": inherited_files,
        }

        if publish:
            published = self.store.publish(staging, manifest, written)
            status_message = "Cache generation published atomically"
        else:
            # --no-publish intentionally leaves staging for benchmarking.
            published = {**manifest, "files": inherited_files + [
                item.manifest_record() for item in written
            ]}
            status_message = "Cache generation validated but not published"

        return BuildResult(
            status="ok",
            mode=mode,
            build_id=build_id,
            generation_id=(
                generation_id if publish else (
                    str(active["generation_id"]) if active else None
                )
            ),
            data_through_session=(
                published["data_through_session"]
                if publish
                else (
                    str(active["data_through_session"]) if active else None
                )
            ),
            symbols=symbols,
            session_dates=tuple(target_dates),
            row_count=validation_totals.surface_rows
            - validation_totals.rejected_rows,
            file_count=len(written),
            message=status_message,
            metrics=extractor.metrics.as_dict(),
            validation=validation_totals.as_dict(),
        )

    @staticmethod
    def _symbol_data_through(
        symbols: Sequence[str],
        sessions_by_symbol: Mapping[str, Sequence[TradingSession]],
        active: Mapping[str, Any] | None,
    ) -> dict[str, str]:
        previous = CacheBuilder._manifest_symbol_data_through(active)
        for symbol in symbols:
            dates = [
                session.session_date.isoformat()
                for session in sessions_by_symbol.get(symbol, ())
            ]
            if dates:
                previous[symbol] = max(dates)
        return {
            symbol: str(previous[symbol])
            for symbol in symbols
            if symbol in previous
        }

    @staticmethod
    def _manifest_symbol_data_through(
        active: Mapping[str, Any] | None,
    ) -> dict[str, str]:
        if active is None:
            return {}
        result = {
            str(symbol): str(value)
            for symbol, value in active.get(
                "data_through_session_by_symbol", {}
            ).items()
        }
        for record in active.get("files", ()):
            if record.get("dataset") != "session_index":
                continue
            symbol = record.get("symbol")
            maximum = record.get("maximum_session")
            if symbol and maximum and str(maximum) > result.get(str(symbol), ""):
                result[str(symbol)] = str(maximum)
        return result

    @staticmethod
    def _merge_coverage_gaps(
        active: Mapping[str, Any] | None,
        current: Mapping[str, Mapping[str, Any]],
        sessions_by_symbol: Mapping[str, Sequence[TradingSession]],
    ) -> dict[str, dict[str, Any]]:
        result = {
            str(symbol): dict(details)
            for symbol, details in (
                active.get("coverage_gaps_by_symbol", {}).items()
                if active is not None
                else ()
            )
        }
        built_dates = {
            symbol: {
                session.session_date.isoformat() for session in sessions
            }
            for symbol, sessions in sessions_by_symbol.items()
        }
        for symbol in tuple(result):
            remaining = sorted(
                set(result[symbol].get("missing_sessions", ()))
                - built_dates.get(symbol, set())
            )
            if remaining:
                result[symbol]["missing_sessions"] = remaining
            else:
                result.pop(symbol)

        for symbol, details in current.items():
            merged = dict(details)
            merged["missing_sessions"] = sorted(
                set(result.get(symbol, {}).get("missing_sessions", ()))
                | set(details.get("missing_sessions", ()))
            )
            result[symbol] = merged
        return result

    def _validate_session_coverage(
        self,
        symbol: str,
        sessions: Sequence[TradingSession],
        session_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        by_date = {row["session_date"]: row for row in session_rows}
        missing = [
            item.session_date.isoformat()
            for item in sessions
            if item.session_date not in by_date
        ]
        low = [
            {
                "session_date": str(row["session_date"]),
                "coverage_fraction": float(row["coverage_fraction"]),
            }
            for row in session_rows
            if float(row["coverage_fraction"])
            < self.config.minimum_session_coverage
        ]
        if missing or low:
            raise SourceBacklogError(
                f"Session coverage is incomplete for {symbol}",
                details={"symbol": symbol, "missing_sessions": missing, "low_coverage": low},
            )

    def _process_and_write_symbol_chunk(
        self,
        *,
        staging: Any,
        effective_mode: str,
        base_id: str,
        build_id: str,
        symbol: str,
        year: int,
        month: int,
        source_tables: Any,
        sessions: Sequence[TradingSession],
        validation_totals: ValidationReport,
        definition_ids: set[int],
        calculation_ids: set[int],
        coverage_by_symbol: dict[str, dict[str, float]],
    ) -> list[WrittenFile]:
        self.logger.info(
            "event=cache_symbol_partition status=started symbol=%s year=%s "
            "month=%s",
            symbol,
            year,
            month,
        )
        session_dates = [item.session_date.isoformat() for item in sessions]
        surface_relative, session_relative = self._chunk_paths(
            effective_mode=effective_mode,
            base_id=base_id,
            build_id=build_id,
            symbol=symbol,
            year=year,
            month=month,
            session_dates=session_dates,
        )
        session_stats: dict[date, dict[str, Any]] = {}
        local_source_rows = 0
        local_rejected_rows = 0
        symbol_coverage = coverage_by_symbol.setdefault(
            symbol,
            {
                "minimum_dte_days": math.inf,
                "maximum_dte_days": -math.inf,
                "minimum_moneyness": math.inf,
                "maximum_moneyness": -math.inf,
            },
        )

        def validated_tables():
            nonlocal local_source_rows, local_rejected_rows
            for batch in surface_tables_source:
                valid, report = validate_surface_table(
                    batch,
                    # Enforce the configured fraction over the complete symbol
                    # partition below; batches remain bounded for memory.
                    max_rejected_fraction=0.999999,
                )
                validation_totals.surface_rows += report.surface_rows
                validation_totals.rejected_rows += report.rejected_rows
                validation_totals.duplicate_keys += report.duplicate_keys
                validation_totals.rejection_reasons.update(
                    report.rejection_reasons
                )
                local_source_rows += report.surface_rows
                local_rejected_rows += report.rejected_rows
                if not valid.num_rows:
                    continue
                definition_ids.update(
                    int(value.as_py())
                    for value in pc.unique(valid["surface_definition_id"])
                )
                calculation_ids.update(
                    int(value.as_py())
                    for value in pc.unique(valid["calculation_definition_id"])
                )
                dte = self._numeric(valid, "time_to_expiry_seconds") / 86400.0
                minimum = self._numeric(
                    valid, "minimum_log_forward_moneyness"
                )
                maximum = self._numeric(
                    valid, "maximum_log_forward_moneyness"
                )
                symbol_coverage["minimum_dte_days"] = min(
                    symbol_coverage["minimum_dte_days"], float(np.min(dte))
                )
                symbol_coverage["maximum_dte_days"] = max(
                    symbol_coverage["maximum_dte_days"], float(np.max(dte))
                )
                symbol_coverage["minimum_moneyness"] = min(
                    symbol_coverage["minimum_moneyness"], float(np.min(minimum))
                )
                symbol_coverage["maximum_moneyness"] = max(
                    symbol_coverage["maximum_moneyness"], float(np.max(maximum))
                )
                self._update_session_stats(session_stats, valid)
                yield valid

        surface_tables_source = iter(source_tables)
        surface_file = self.store.write_parquet_batches(
            staging,
            dataset="surface_observations",
            relative_path=surface_relative,
            tables=validated_tables(),
            symbol=symbol,
            session_values=[item.session_date for item in sessions],
        )
        if local_source_rows:
            rejected_fraction = local_rejected_rows / local_source_rows
            if rejected_fraction > self.config.max_rejected_row_fraction:
                raise ValidationError(
                    f"Rejected surface-row fraction {rejected_fraction:.4f} "
                    f"exceeds {self.config.max_rejected_row_fraction:.4f}"
                )
        session_rows = self._session_index_rows(
            symbol,
            sessions,
            session_stats,
        )
        self._validate_session_coverage(symbol, sessions, session_rows)
        validation_totals.session_rows += len(session_rows)
        session_table = table_from_rows(session_rows, SESSION_INDEX_SCHEMA)
        session_file = self.store.write_parquet(
            staging,
            dataset="session_index",
            relative_path=session_relative,
            table=session_table,
            symbol=symbol,
            session_values=[row["session_date"] for row in session_rows],
        )
        self.logger.info(
            "event=cache_symbol_partition status=completed symbol=%s year=%s "
            "month=%s surface_rows=%s session_rows=%s surface_bytes=%s "
            "session_bytes=%s",
            symbol,
            year,
            month,
            surface_file.row_count,
            session_file.row_count,
            surface_file.byte_size,
            session_file.byte_size,
        )
        return [surface_file, session_file]

    @staticmethod
    def _numeric(table: pa.Table, name: str) -> np.ndarray:
        return np.asarray(
            table[name].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )

    def _update_session_stats(
        self,
        stats: dict[date, dict[str, Any]],
        table: pa.Table,
    ) -> None:
        dates = np.asarray(
            table["session_date"].combine_chunks().to_numpy(zero_copy_only=False)
        )
        minutes = self._numeric(table, "session_minute").astype(np.int64)
        spots = self._numeric(table, "spot_price")
        quality = self._numeric(table, "quality_flags").astype(np.uint64)
        atm = evaluate_surface_iv_targets(
            table,
            np.zeros(table.num_rows, dtype=np.float64),
        )
        for raw_date in np.unique(dates):
            session_date = date.fromisoformat(str(raw_date)[:10])
            selected = dates == raw_date
            selected_minutes = minutes[selected]
            current_minute = int(np.min(selected_minutes))
            entry = stats.setdefault(
                session_date,
                {
                    "minutes": set(),
                    "opening_minute": current_minute,
                    "opening_spot_sum": 0.0,
                    "opening_spot_count": 0,
                    "opening_iv_sum": 0.0,
                    "opening_iv_count": 0,
                    "quality_flags": 0,
                },
            )
            entry["minutes"].update(
                int(value) for value in np.unique(selected_minutes)
            )
            entry["quality_flags"] |= int(np.bitwise_or.reduce(quality[selected]))
            if current_minute < entry["opening_minute"]:
                entry.update(
                    opening_minute=current_minute,
                    opening_spot_sum=0.0,
                    opening_spot_count=0,
                    opening_iv_sum=0.0,
                    opening_iv_count=0,
                )
            if current_minute == entry["opening_minute"]:
                opening = selected & (minutes == current_minute)
                valid_spot = spots[opening & np.isfinite(spots)]
                valid_iv = atm[opening & np.isfinite(atm)]
                entry["opening_spot_sum"] += float(np.sum(valid_spot))
                entry["opening_spot_count"] += int(len(valid_spot))
                entry["opening_iv_sum"] += float(np.sum(valid_iv))
                entry["opening_iv_count"] += int(len(valid_iv))

    def _session_index_rows(
        self,
        symbol: str,
        sessions: Sequence[TradingSession],
        stats: Mapping[date, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for session in sessions:
            entry = stats.get(session.session_date)
            if not entry:
                continue
            minutes = sorted(entry["minutes"])
            expected = len(
                session.sample_targets(self.config.sample_interval_minutes)
            )
            result.append(
                {
                    "underlying_symbol": symbol,
                    "session_date": session.session_date,
                    "session_open_utc": session.open_utc,
                    "session_close_utc": session.close_utc,
                    "regular_session_minutes": session.duration_minutes,
                    "is_early_close": session.is_early_close,
                    "day_of_week": session.session_date.weekday(),
                    "first_sample_minute": minutes[0],
                    "last_sample_minute": minutes[-1],
                    "expected_bin_count": expected,
                    "available_bin_count": len(minutes),
                    "coverage_fraction": len(minutes) / expected if expected else 0.0,
                    "opening_spot": (
                        entry["opening_spot_sum"] / entry["opening_spot_count"]
                        if entry["opening_spot_count"]
                        else None
                    ),
                    "opening_atm_iv": (
                        entry["opening_iv_sum"] / entry["opening_iv_count"]
                        if entry["opening_iv_count"]
                        else None
                    ),
                    "quality_flags": entry["quality_flags"],
                    "rejected_snapshot_count": 0,
                    "stale_sample_count": 0,
                }
            )
        return result

    def _chunk_paths(
        self,
        *,
        effective_mode: str,
        base_id: str,
        build_id: str,
        symbol: str,
        year: int,
        month: int,
        session_dates: Sequence[str],
    ) -> tuple[str, str]:
        if effective_mode == "weekly":
            root = f"data/base/{base_id}"
            suffix = f"part-{year}{month:02d}-{build_id[-8:]}.parquet"
            surface_relative = (
                f"{root}/surface_observations/symbol={symbol}/"
                f"session_year={year}/{suffix}"
            )
            session_relative = (
                f"{root}/session_index/symbol={symbol}/"
                f"session_year={year}/{suffix}"
            )
        else:
            if len(session_dates) == 1:
                partition = f"session_date={session_dates[0]}"
            else:
                partition = f"build={build_id}"
            root = f"data/increments/{partition}"
            suffix = f"part-{year}{month:02d}-{build_id[-8:]}.parquet"
            surface_relative = (
                f"{root}/surface_observations/symbol={symbol}/{suffix}"
            )
            session_relative = f"{root}/session_index/symbol={symbol}/{suffix}"
        return surface_relative, session_relative
