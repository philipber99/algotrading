"""Lifespan-managed polling for today's pending earnings results."""

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

from algotrading.refdata.providers.contracts import EARNINGS_CALENDAR
from algotrading.refdata.providers.registry import ProviderRegistry
from algotrading.refdata.repositories.errors import AdvisoryLockUnavailable
from algotrading.refdata.stock_analysis.earnings_calendar import (
    EarningsCalendarReconciler,
    EarningsReconciliationSummary,
)


@dataclass(frozen=True)
class EarningsCalendarUpdaterConfig:
    enabled: bool = True
    poll_interval_seconds: int = 1800
    lock_wait_seconds: int = 0

    @classmethod
    def from_context(cls, context: dict[str, Any]) -> "EarningsCalendarUpdaterConfig":
        prefix = "refdata.earnings_calendar.auto_update."
        enabled = context.get(prefix + "enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(prefix + "enabled must be boolean")
        return cls(
            enabled=enabled,
            poll_interval_seconds=_integer(
                context.get(prefix + "poll_interval_seconds", 1800),
                "poll_interval_seconds",
                60,
                86400,
            ),
            lock_wait_seconds=_integer(
                context.get(prefix + "lock_wait_seconds", 0),
                "lock_wait_seconds",
                0,
                60,
            ),
        )


@dataclass(frozen=True)
class EarningsUpdateIteration:
    provider_called: bool = False
    lock_acquired: bool = False
    due_before: int = 0
    due_after: int = 0
    next_delay_seconds: float = 1800
    reconciliation: EarningsReconciliationSummary | None = None
    error_code: str | None = None


class EarningsCalendarUpdater:
    def __init__(
        self,
        context: dict[str, Any],
        repository: Any,
        lock_repository: Any,
        reconciler: EarningsCalendarReconciler,
        *,
        registry: ProviderRegistry | None = None,
        provider: Any | None = None,
        config: EarningsCalendarUpdaterConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.context = context
        self.repository = repository
        self.lock_repository = lock_repository
        self.reconciler = reconciler
        self.registry = registry or ProviderRegistry(context)
        self.injected_provider = provider
        self.config = config or EarningsCalendarUpdaterConfig.from_context(context)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleeper = sleeper
        self.logger = logger or context.get("logger") or logging.getLogger(__name__)
        self.market_timezone = ZoneInfo(
            str(
                context.get(
                    "ExchangeTZ",
                    context.get("exchange_timezone", "America/New_York"),
                )
            )
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
            max_workers=1, thread_name_prefix="earnings-calendar-updater"
        )
        self._task = asyncio.create_task(self._run(), name="earnings-calendar-updater")
        return True

    async def stop(self, timeout_seconds: float = 5.0) -> None:
        task = self._task
        if task is None:
            return
        # Signal the blocking worker before waiting for the async polling task to stop.
        self._worker_stop.set()
        if self._stop_event is not None:
            self._stop_event.set()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            self.logger.error(
                "event=earnings_calendar_shutdown_failure "
                "error_code=WORKER_JOIN_TIMEOUT inflight=%s",
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
        delay = 0.0
        try:
            while self._stop_event is not None and not self._stop_event.is_set():
                if delay > 0:
                    await self._sleep(delay)
                if self._stop_event.is_set():
                    break
                try:
                    result = await self._run_worker(self.check_once)
                    delay = max(1.0, result.next_delay_seconds)
                except Exception as exc:
                    self.logger.error(
                        "event=earnings_calendar_iteration_failure error_code=%s",
                        type(exc).__name__,
                    )
                    delay = float(self.config.poll_interval_seconds)
        except Exception as exc:
            self.logger.error(
                "event=earnings_calendar_task_failure error_code=%s",
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
            return EarningsUpdateIteration(
                next_delay_seconds=self.config.poll_interval_seconds,
                error_code="STOP_REQUESTED",
            )
        if self._inflight is not None and not self._inflight.done():
            raise RuntimeError("Earnings calendar updater iteration overlapped")
        if self._executor is None:
            raise RuntimeError("Earnings calendar updater worker is not initialized")
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, function)
        self._inflight = future
        try:
            return await asyncio.shield(future)
        finally:
            if future.done():
                self._inflight = None

    def check_once(self) -> EarningsUpdateIteration:
        started = monotonic()
        now = self._market_now()
        # Define the polling day in exchange time rather than the server timezone.
        today = now.date()
        tomorrow = today + timedelta(days=1)
        interval = float(self.config.poll_interval_seconds)
        next_delay = interval
        provider_called = False
        locked = False
        due_before = 0
        due_after = 0
        if self._worker_stop.is_set():
            return EarningsUpdateIteration(
                next_delay_seconds=interval, error_code="STOP_REQUESTED"
            )
        try:
            due = self.repository.due_rows(today, tomorrow)
            due_before = len(due)
            if not due:
                next_delay = self._next_day_delay(now)
                return EarningsUpdateIteration(next_delay_seconds=next_delay)
            provider = self._provider()
            with self.lock_repository.advisory_lock(
                EARNINGS_CALENDAR,
                f"actuals:{today.isoformat()}",
                self.config.lock_wait_seconds,
            ):
                locked = True
                due = self.repository.due_rows(today, tomorrow)
                if not due:
                    next_delay = self._next_day_delay(now)
                    return EarningsUpdateIteration(
                        lock_acquired=True,
                        due_before=due_before,
                        next_delay_seconds=next_delay,
                    )
                if self._worker_stop.is_set():
                    return EarningsUpdateIteration(
                        lock_acquired=True,
                        due_before=due_before,
                        next_delay_seconds=interval,
                        error_code="STOP_REQUESTED",
                    )
                # Fetch actuals only after rechecking pending rows under the shared refresh lock.
                provider_called = True
                result = provider.fetch_earnings_releases(today, today)
                if self._worker_stop.is_set():
                    return EarningsUpdateIteration(
                        provider_called=True,
                        lock_acquired=True,
                        due_before=due_before,
                        next_delay_seconds=interval,
                        error_code="STOP_REQUESTED",
                    )
                summary = self.reconciler.reconcile(result, today, tomorrow)
                self._log_reconciliation(summary)
                due_after = len(self.repository.due_rows(today, tomorrow))
                # Poll again only while today still has pending earnings results.
                next_delay = interval if due_after else self._next_day_delay(now)
                return EarningsUpdateIteration(
                    provider_called=True,
                    lock_acquired=True,
                    due_before=due_before,
                    due_after=due_after,
                    next_delay_seconds=next_delay,
                    reconciliation=summary,
                )
        except AdvisoryLockUnavailable:
            self.logger.info(
                "event=earnings_calendar_actual_poll "
                "result=refresh_in_progress market_date=%s",
                today,
            )
            return EarningsUpdateIteration(
                due_before=due_before,
                next_delay_seconds=interval,
                error_code="REFRESH_IN_PROGRESS",
            )
        except Exception as exc:
            self.logger.warning(
                "event=earnings_calendar_actual_poll market_date=%s error_code=%s",
                today,
                type(exc).__name__,
            )
            return EarningsUpdateIteration(
                provider_called=provider_called,
                lock_acquired=locked,
                due_before=due_before,
                due_after=due_after,
                next_delay_seconds=interval,
                error_code=type(exc).__name__,
            )
        finally:
            self.logger.debug(
                "event=earnings_calendar_actual_poll market_date=%s "
                "due_missing_before=%s lock_acquired=%s due_missing_after=%s "
                "next_check_at=%s duration_seconds=%.3f",
                today,
                due_before,
                locked,
                due_after,
                (now + timedelta(seconds=next_delay)).isoformat(),
                monotonic() - started,
            )

    def _provider(self) -> Any:
        return self.injected_provider or self.registry.provider_for(EARNINGS_CALENDAR)

    def _market_now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(self.market_timezone)

    def _next_day_delay(self, now: datetime) -> float:
        tomorrow = now.date() + timedelta(days=1)
        target = datetime.combine(tomorrow, time.min, self.market_timezone)
        return max(1.0, (target - now).total_seconds())

    def _log_reconciliation(self, summary: EarningsReconciliationSummary) -> None:
        fields = " ".join(f"{key}={value}" for key, value in summary.log_fields().items())
        self.logger.info("event=earnings_calendar_reconciliation %s", fields)


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
