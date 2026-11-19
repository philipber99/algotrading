from __future__ import annotations

import asyncio
import unittest
from contextlib import contextmanager
from datetime import date, datetime, timezone

from algotrading.refdata.api.earnings_calendar_updater import (
    EarningsCalendarUpdater,
    EarningsCalendarUpdaterConfig,
)
from algotrading.refdata.repositories.errors import AdvisoryLockUnavailable
from algotrading.refdata.stock_analysis.earnings_calendar import (
    EarningsReconciliationSummary,
)
from algotrading.refdata.stock_analysis.models import ProviderResult


NOW = datetime(2026, 8, 9, 16, tzinfo=timezone.utc)


class Repository:
    def __init__(self):
        self.due = []
        self.calls = []

    def due_rows(self, start, end):
        self.calls.append((start, end))
        value = self.due.pop(0) if len(self.due) > 1 else (
            self.due[0] if self.due else []
        )
        return list(value)


class Locks:
    def __init__(self, available=True):
        self.available = available
        self.entered = 0
        self.exited = 0
        self.keys = []

    @contextmanager
    def advisory_lock(self, category, key, wait):
        self.keys.append((category, key, wait))
        if not self.available:
            raise AdvisoryLockUnavailable("busy")
        self.entered += 1
        try:
            yield
        finally:
            self.exited += 1


class Provider:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def fetch_earnings_releases(self, start, end):
        self.calls.append((start, end))
        if self.error:
            raise self.error
        return ProviderResult(provider="TEST")


class Reconciler:
    def __init__(self):
        self.calls = []

    def reconcile(self, result, start, end):
        self.calls.append((result, start, end))
        return EarningsReconciliationSummary(fetched=result.count)


def updater(
    repository=None, locks=None, provider=None, reconciler=None,
    config=None, sleeper=None,
):
    return EarningsCalendarUpdater(
        {"ExchangeTZ": "America/New_York"},
        repository or Repository(),
        locks or Locks(),
        reconciler or Reconciler(),
        provider=provider or Provider(),
        config=config or EarningsCalendarUpdaterConfig(),
        clock=lambda: NOW,
        sleeper=sleeper,
    )


class ConfigurationTests(unittest.TestCase):
    def test_defaults_disable_and_validation(self):
        config = EarningsCalendarUpdaterConfig.from_context({})
        self.assertTrue(config.enabled)
        self.assertEqual(config.poll_interval_seconds, 1800)
        disabled = EarningsCalendarUpdaterConfig.from_context({
            "refdata.earnings_calendar.auto_update.enabled": False,
        })
        self.assertFalse(disabled.enabled)
        for context in (
            {"refdata.earnings_calendar.auto_update.enabled": "false"},
            {"refdata.earnings_calendar.auto_update.poll_interval_seconds": 59},
            {"refdata.earnings_calendar.auto_update.poll_interval_seconds": 60.5},
            {"refdata.earnings_calendar.auto_update.lock_wait_seconds": -1},
        ):
            with self.subTest(context=context), self.assertRaises(ValueError):
                EarningsCalendarUpdaterConfig.from_context(context)


class PollingTests(unittest.TestCase):
    def test_no_due_rows_make_no_provider_call_until_next_day(self):
        repository = Repository()
        repository.due = [[]]
        provider = Provider()
        result = updater(repository=repository, provider=provider).check_once()
        self.assertEqual(provider.calls, [])
        self.assertEqual(result.next_delay_seconds, 12 * 60 * 60)

    def test_due_rows_poll_recheck_reconcile_and_stop_when_complete(self):
        due = [{"symbol": "AAPL", "eps": None, "revenue": None}]
        repository = Repository()
        repository.due = [due, due, []]
        provider = Provider()
        reconciler = Reconciler()
        locks = Locks()
        result = updater(
            repository=repository,
            provider=provider,
            reconciler=reconciler,
            locks=locks,
        ).check_once()
        self.assertEqual(provider.calls, [(date(2026, 8, 9), date(2026, 8, 9))])
        self.assertEqual(reconciler.calls[0][1:], (
            date(2026, 8, 9), date(2026, 8, 10)
        ))
        self.assertEqual(result.due_before, 1)
        self.assertEqual(result.due_after, 0)
        self.assertEqual(result.next_delay_seconds, 12 * 60 * 60)
        self.assertEqual(locks.entered, locks.exited)

    def test_remaining_due_failure_and_lock_contention_retry_in_30_minutes(self):
        due = [{"symbol": "AAPL", "eps": None, "revenue": None}]
        repository = Repository()
        repository.due = [due, due, due]
        result = updater(repository=repository).check_once()
        self.assertEqual(result.due_after, 1)
        self.assertEqual(result.next_delay_seconds, 1800)

        repository = Repository()
        repository.due = [due, due]
        result = updater(
            repository=repository,
            provider=Provider(RuntimeError("failed")),
        ).check_once()
        self.assertEqual(result.next_delay_seconds, 1800)
        self.assertEqual(result.error_code, "RuntimeError")

        repository = Repository()
        repository.due = [due]
        provider = Provider()
        result = updater(
            repository=repository,
            provider=provider,
            locks=Locks(available=False),
        ).check_once()
        self.assertEqual(result.error_code, "REFRESH_IN_PROGRESS")
        self.assertEqual(provider.calls, [])


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_starts_no_task(self):
        instance = updater(config=EarningsCalendarUpdaterConfig(enabled=False))
        self.assertFalse(instance.start())
        self.assertFalse(instance.running)
        await instance.stop()

    async def test_shutdown_wakes_idle_next_day_sleep(self):
        entered = asyncio.Event()

        async def sleeper(_):
            entered.set()
            await asyncio.Event().wait()

        repository = Repository()
        repository.due = [[]]
        instance = updater(repository=repository, sleeper=sleeper)
        self.assertTrue(instance.start())
        await asyncio.wait_for(entered.wait(), timeout=1)
        await instance.stop(timeout_seconds=1)
        self.assertFalse(instance.running)


if __name__ == "__main__":
    unittest.main()
