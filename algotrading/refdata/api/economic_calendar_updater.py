"""Lifespan-managed economic-calendar coverage and actual updater."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from time import monotonic
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from algotrading.refdata.providers.contracts import ECONOMIC_CALENDAR
from algotrading.refdata.providers.registry import ProviderRegistry
from algotrading.refdata.repositories.errors import AdvisoryLockUnavailable
from algotrading.refdata.stock_analysis.economic_calendar import (
    EconomicCalendarReconciler,
    ReconciliationSummary,
)


@dataclass(frozen=True)
class EconomicCalendarUpdaterConfig:
    enabled: bool = True
    horizon_days: int = 60
    poll_interval_seconds: int = 1800
    coverage_tail_slack_days: int = 3
    daily_overlap_days: int = 1
    lock_wait_seconds: int = 0

    @classmethod
    def from_context(cls, context: dict[str, Any]) -> "EconomicCalendarUpdaterConfig":
        prefix = "refdata.economic_calendar.auto_update."
        enabled = context.get(prefix + "enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(prefix + "enabled must be boolean")
        values = {
            "horizon_days": _integer(context.get(prefix + "horizon_days", 60), "horizon_days", 1, 366),
            "poll_interval_seconds": _integer(context.get(prefix + "poll_interval_seconds", 1800), "poll_interval_seconds", 60, 86400),
            "coverage_tail_slack_days": _integer(context.get(prefix + "coverage_tail_slack_days", 3), "coverage_tail_slack_days", 0, 30),
            "daily_overlap_days": _integer(context.get(prefix + "daily_overlap_days", 1), "daily_overlap_days", 0, 7),
            "lock_wait_seconds": _integer(context.get(prefix + "lock_wait_seconds", 0), "lock_wait_seconds", 0, 60),
        }
        if values["coverage_tail_slack_days"] > values["horizon_days"]:
            raise ValueError("coverage_tail_slack_days may not exceed horizon_days")
        return cls(enabled=enabled, **values)


@dataclass(frozen=True)
class UpdateIteration:
    provider_called: bool = False
    lock_acquired: bool = False
    due_before: int = 0
    due_after: int = 0
    next_delay_seconds: float = 1800
    reconciliation: ReconciliationSummary | None = None
    error_code: str | None = None


class RefDataTradingCalendar:
    """Adapter that delegates market-day decisions to the existing calendar service."""

    def __init__(self, service: Any) -> None:
        self.service = service

    def is_market_day(self, day: date) -> bool:
        return bool(self.service.trading_calendar(day, day).data)

    def next_market_day(self, after: date) -> date | None:
        end = after + timedelta(days=370)
        rows = self.service.trading_calendar(after + timedelta(days=1), end).data
        if not rows:
            return None
        value = rows[0].get("date") if isinstance(rows[0], dict) else None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10]) if value else None


class EconomicCalendarUpdater:
    def __init__(
        self,
        context: dict[str, Any],
        repository: Any,
        lock_repository: Any,
        reconciler: EconomicCalendarReconciler,
        trading_calendar: Any,
        *,
        registry: ProviderRegistry | None = None,
        provider: Any | None = None,
        config: EconomicCalendarUpdaterConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.context = context
        self.repository = repository
        self.lock_repository = lock_repository
        self.reconciler = reconciler
        self.trading_calendar = trading_calendar
        self.registry = registry or ProviderRegistry(context)
        self.injected_provider = provider
        self.config = config or EconomicCalendarUpdaterConfig.from_context(context)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleeper = sleeper
        self.logger = logger or context.get("logger") or logging.getLogger(__name__)
        self.market_timezone = ZoneInfo(
            str(context.get("ExchangeTZ", context.get("exchange_timezone", "America/New_York")))
        )
        self.storage_timezone = ZoneInfo(
            str(context.get("LocalTZ", context.get("local_timezone", "America/Los_Angeles")))
        )
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._worker_stop = threading.Event()
        self._executor: ThreadPoolExecutor | None = None
        self._inflight: asyncio.Future[Any] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> bool:
        if not self.config.enabled or self.running:
            return False
        self._stop_event = asyncio.Event()
        self._worker_stop.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="economic-calendar-updater"
        )
        self._task = asyncio.create_task(self._run(), name="economic-calendar-updater")
        return True

    async def stop(self, timeout_seconds: float = 5.0) -> None:
        task = self._task
        if task is None:
            return
        # Signal the worker before draining managed background tasks.
        self._worker_stop.set()
        if self._stop_event is not None:
            self._stop_event.set()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            self.logger.error(
                "event=economic_calendar_shutdown_failure error_code=WORKER_JOIN_TIMEOUT inflight=%s",
                bool(self._inflight and not self._inflight.done()),
            )
            raise
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        provider = self.injected_provider
        close = getattr(provider, "close", None)
        if callable(close):
            close()
        registry_close = getattr(self.registry, "close", None)
        if callable(registry_close):
            registry_close()
        self._executor = None
        self._inflight = None
        self._task = None
        self._stop_event = None

    async def _run(self) -> None:
        try:
            try:
                startup = await self._run_worker(self.startup_check_once)
                delay = (
                    startup.next_delay_seconds if startup.error_code is not None else 0.0
                )
            except Exception as exc:
                self.logger.error(
                    "event=economic_calendar_startup_task_failure error_code=%s",
                    type(exc).__name__,
                )
                delay = float(self.config.poll_interval_seconds)
            while self._stop_event is not None and not self._stop_event.is_set():
                if delay > 0:
                    await self._sleep(delay)
                if self._stop_event.is_set():
                    break
                try:
                    result = await self._run_worker(self.market_day_check_once)
                    delay = max(1.0, result.next_delay_seconds)
                except Exception as exc:
                    self.logger.error(
                        "event=economic_calendar_iteration_failure error_code=%s",
                        type(exc).__name__,
                    )
                    delay = float(self.config.poll_interval_seconds)
        except Exception as exc:
            self.logger.error(
                "event=economic_calendar_task_failure error_code=%s",
                type(exc).__name__,
            )

    async def _sleep(self, seconds: float) -> None:
        if self.sleeper is not None:
            assert self._stop_event is not None
            sleeper_task = asyncio.create_task(self.sleeper(seconds))
            stop_task = asyncio.create_task(self._stop_event.wait())
            done, pending = await asyncio.wait(
                {sleeper_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task is sleeper_task:
                    task.result()
            return
        assert self._stop_event is not None
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _run_worker(self, function: Callable[[], Any]) -> Any:
        if self._worker_stop.is_set():
            return UpdateIteration(
                next_delay_seconds=self.config.poll_interval_seconds,
                error_code="STOP_REQUESTED",
            )
        if self._inflight is not None and not self._inflight.done():
            raise RuntimeError("Economic calendar updater iteration overlapped")
        if self._executor is None:
            raise RuntimeError("Economic calendar updater worker is not initialized")
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, function)
        self._inflight = future
        try:
            return await asyncio.shield(future)
        finally:
            if future.done():
                self._inflight = None

    def startup_check_once(self) -> UpdateIteration:
        return self.refresh_horizon_once(reason="startup_coverage")

    def weekly_refresh_once(self) -> UpdateIteration:
        """Force the shared calendar reconciliation across the weekly window."""
        lookback_days = _integer(
            self.context.get(
                "refdata.economic_calendar.weekly_update.lookback_days", 10
            ),
            "weekly_update.lookback_days",
            0,
            366,
        )
        return self.refresh_horizon_once(
            force_refresh=True,
            lookback_days=lookback_days,
            reason="weekly_batch",
        )

    def refresh_horizon_once(
        self,
        *,
        force_refresh: bool = False,
        lookback_days: int = 0,
        reason: str = "coverage",
    ) -> UpdateIteration:
        """Reconcile one bounded provider window using the API update path."""
        lookback = _integer(lookback_days, "lookback_days", 0, 366)
        started = monotonic()
        now = self._market_now()
        today = now.date()
        range_start = today - timedelta(days=lookback)
        horizon_end = today + timedelta(days=self.config.horizon_days)
        # Convert exchange-date windows into the database storage timezone.
        start = self._storage_midnight(range_start)
        end_exclusive = self._storage_midnight(horizon_end + timedelta(days=1))
        db_now = self._storage_naive(now)
        threshold_market = datetime.combine(
            horizon_end - timedelta(days=self.config.coverage_tail_slack_days),
            time.min,
            self.market_timezone,
        )
        threshold = self._storage_naive(threshold_market)
        count = 0
        maximum = None
        refreshed = False
        locked = False
        log_event = (
            "economic_calendar_startup_check"
            if reason == "startup_coverage"
            else "economic_calendar_horizon_refresh"
        )
        try:
            if self._worker_stop.is_set():
                return UpdateIteration(
                    next_delay_seconds=self.config.poll_interval_seconds,
                    error_code="STOP_REQUESTED",
                )
            count, maximum = self.repository.future_coverage(db_now, end_exclusive)
            if not force_refresh and _coverage_sufficient(count, maximum, threshold):
                return UpdateIteration(next_delay_seconds=self.config.poll_interval_seconds)
            provider = self._provider()
            # Use one cross-process lock for each startup coverage window.
            lock_key = f"startup:{today.isoformat()}:{horizon_end.isoformat()}"
            with self.lock_repository.advisory_lock(
                ECONOMIC_CALENDAR, lock_key, self.config.lock_wait_seconds
            ):
                locked = True
                count, maximum = self.repository.future_coverage(db_now, end_exclusive)
                if not force_refresh and _coverage_sufficient(count, maximum, threshold):
                    return UpdateIteration(
                        lock_acquired=True,
                        next_delay_seconds=self.config.poll_interval_seconds,
                    )
                if self._worker_stop.is_set():
                    return UpdateIteration(
                        lock_acquired=True,
                        next_delay_seconds=self.config.poll_interval_seconds,
                        error_code="STOP_REQUESTED",
                    )
                refreshed = True
                # Refresh only after rechecking durable coverage under the lock.
                result = provider.fetch_economic_releases(range_start, horizon_end)
                if self._worker_stop.is_set():
                    return UpdateIteration(
                        provider_called=True,
                        lock_acquired=True,
                        next_delay_seconds=self.config.poll_interval_seconds,
                        error_code="STOP_REQUESTED",
                    )
                summary = self.reconciler.reconcile(
                    result, start, end_exclusive, max_distance_days=7
                )
                self._log_reconciliation(reason, summary)
                return UpdateIteration(
                    provider_called=True,
                    lock_acquired=True,
                    next_delay_seconds=self.config.poll_interval_seconds,
                    reconciliation=summary,
                )
        except AdvisoryLockUnavailable:
            self.logger.info(
                "event=%s result=refresh_in_progress reason=%s range_start=%s range_end=%s",
                log_event, reason, range_start, horizon_end,
            )
            return UpdateIteration(
                next_delay_seconds=self.config.poll_interval_seconds,
                error_code="REFRESH_IN_PROGRESS",
            )
        except Exception as exc:
            self.logger.warning(
                "event=%s reason=%s error_code=%s",
                log_event, reason, type(exc).__name__,
            )
            return UpdateIteration(
                provider_called=refreshed,
                lock_acquired=locked,
                next_delay_seconds=self.config.poll_interval_seconds,
                error_code=type(exc).__name__,
            )
        finally:
            self.logger.info(
                "event=%s enabled=%s reason=%s force_refresh=%s range_start=%s range_end=%s db_count=%s db_max_timestamp=%s coverage_sufficient=%s refresh_attempted=%s lock_acquired=%s duration_seconds=%.3f",
                log_event, self.config.enabled, reason, force_refresh, range_start,
                horizon_end, count, maximum,
                _coverage_sufficient(count, maximum, threshold), refreshed, locked,
                monotonic() - started,
            )

    def market_day_check_once(self) -> UpdateIteration:
        started = monotonic()
        now = self._market_now()
        today = now.date()
        interval = float(self.config.poll_interval_seconds)
        next_delay = interval
        provider_start: date | None = None
        provider_end: date | None = None
        if self.trading_calendar is None:
            raise RuntimeError("Trading calendar is required for market-day polling")
        if not self.trading_calendar.is_market_day(today):
            next_delay = self._non_market_delay(now, interval)
            self.logger.debug(
                "event=economic_calendar_actual_poll market_date=%s due_missing_before=0 provider_range_start=None provider_range_end=None lock_acquired=False due_missing_after=0 next_check_at=%s duration_seconds=%.3f",
                today,
                (now + timedelta(seconds=next_delay)).isoformat(),
                monotonic() - started,
            )
            return UpdateIteration(next_delay_seconds=next_delay)
        start = self._storage_midnight(today)
        tomorrow = self._storage_midnight(today + timedelta(days=1))
        db_now = self._storage_naive(now)
        provider_called = False
        locked = False
        due_before = 0
        due_after = 0
        if self._worker_stop.is_set():
            return UpdateIteration(
                next_delay_seconds=interval, error_code="STOP_REQUESTED"
            )
        try:
            due = self.repository.due_rows(start, tomorrow, db_now)
            due_before = len(due)
            if not due:
                next_delay = self._scheduled_delay(db_now, tomorrow, interval)
                return UpdateIteration(next_delay_seconds=next_delay)
            provider = self._provider()
            with self.lock_repository.advisory_lock(
                ECONOMIC_CALENDAR,
                f"actuals:{today.isoformat()}",
                self.config.lock_wait_seconds,
            ):
                locked = True
                due = self.repository.due_rows(start, tomorrow, db_now)
                if not due:
                    next_delay = self._scheduled_delay(db_now, tomorrow, interval)
                    return UpdateIteration(
                        lock_acquired=True,
                        due_before=due_before,
                        next_delay_seconds=next_delay,
                    )
                if self._worker_stop.is_set():
                    return UpdateIteration(
                        lock_acquired=True,
                        due_before=due_before,
                        next_delay_seconds=interval,
                        error_code="STOP_REQUESTED",
                    )
                # Include neighboring dates so revised release times can reconcile to stored events.
                overlap = self.config.daily_overlap_days
                provider_start = today - timedelta(days=overlap)
                provider_end = today + timedelta(days=overlap)
                provider_called = True
                result = provider.fetch_economic_releases(provider_start, provider_end)
                if self._worker_stop.is_set():
                    return UpdateIteration(
                        provider_called=True,
                        lock_acquired=True,
                        due_before=due_before,
                        next_delay_seconds=interval,
                        error_code="STOP_REQUESTED",
                    )
                window_start = self._storage_midnight(provider_start)
                window_end = self._storage_midnight(provider_end + timedelta(days=1))
                summary = self.reconciler.reconcile(
                    result, window_start, window_end, max_distance_days=2
                )
                self._log_reconciliation("missing_actuals", summary)
                due_after = len(self.repository.due_rows(start, tomorrow, db_now))
                next_delay = interval if due_after else self._scheduled_delay(db_now, tomorrow, interval)
                return UpdateIteration(
                    provider_called=True,
                    lock_acquired=True,
                    due_before=due_before,
                    due_after=due_after,
                    next_delay_seconds=next_delay,
                    reconciliation=summary,
                )
        except AdvisoryLockUnavailable:
            self.logger.info(
                "event=economic_calendar_actual_poll result=refresh_in_progress market_date=%s",
                today,
            )
            return UpdateIteration(
                due_before=due_before,
                next_delay_seconds=interval,
                error_code="REFRESH_IN_PROGRESS",
            )
        except Exception as exc:
            self.logger.warning(
                "event=economic_calendar_actual_poll market_date=%s error_code=%s",
                today, type(exc).__name__,
            )
            return UpdateIteration(
                provider_called=provider_called,
                lock_acquired=locked,
                due_before=due_before,
                due_after=due_after,
                next_delay_seconds=interval,
                error_code=type(exc).__name__,
            )
        finally:
            self.logger.debug(
                "event=economic_calendar_actual_poll market_date=%s due_missing_before=%s provider_range_start=%s provider_range_end=%s lock_acquired=%s due_missing_after=%s next_check_at=%s duration_seconds=%.3f",
                today, due_before, provider_start, provider_end, locked, due_after,
                (now + timedelta(seconds=next_delay)).isoformat(),
                monotonic() - started,
            )

    def _provider(self) -> Any:
        return self.injected_provider or self.registry.provider_for(ECONOMIC_CALENDAR)

    def _market_now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(self.market_timezone)

    def _storage_midnight(self, market_date: date) -> datetime:
        market_value = datetime.combine(market_date, time.min, self.market_timezone)
        return self._storage_naive(market_value)

    def _storage_naive(self, value: datetime) -> datetime:
        return value.astimezone(self.storage_timezone).replace(tzinfo=None)

    def _scheduled_delay(
        self, db_now: datetime, tomorrow: datetime, interval: float
    ) -> float:
        next_release = self.repository.next_unreleased(db_now, tomorrow)
        if next_release is None:
            return interval
        return max(1.0, min(interval, (next_release - db_now).total_seconds()))

    def _non_market_delay(self, now: datetime, interval: float) -> float:
        next_day = getattr(self.trading_calendar, "next_market_day", lambda _: None)(now.date())
        if next_day is None:
            return interval
        target = datetime.combine(next_day, time.min, self.market_timezone)
        return max(1.0, (target - now).total_seconds())

    def _log_reconciliation(self, reason: str, summary: ReconciliationSummary) -> None:
        fields = " ".join(f"{key}={value}" for key, value in summary.log_fields().items())
        self.logger.info(
            "event=economic_calendar_reconciliation reason=%s %s", reason, fields
        )


def build_economic_calendar_updater(
    context: dict[str, Any],
    repositories: Any,
    trading_calendar: Any | None = None,
    *,
    registry: ProviderRegistry | None = None,
    logger: logging.Logger | None = None,
) -> EconomicCalendarUpdater:
    """Build the updater shared by the API lifecycle and batch entry points."""
    storage_timezone = str(
        context.get(
            "LocalTZ",
            context.get("local_timezone", "America/Los_Angeles"),
        )
    )
    return EconomicCalendarUpdater(
        context,
        repositories.economic_calendar,
        repositories.fetch_state,
        EconomicCalendarReconciler(
            repositories.economic_calendar,
            storage_timezone=storage_timezone,
        ),
        trading_calendar,
        registry=registry,
        logger=logger,
    )


def _coverage_sufficient(
    count: int, maximum: datetime | None, threshold: datetime
) -> bool:
    return count > 0 and maximum is not None and maximum >= threshold


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result
