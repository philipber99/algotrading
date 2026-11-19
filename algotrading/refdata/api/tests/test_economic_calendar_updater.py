from __future__ import annotations

import asyncio
import threading
import unittest
from contextlib import contextmanager
from datetime import date, datetime, timezone

from algotrading.refdata.api.economic_calendar_updater import (
    EconomicCalendarUpdater,
    EconomicCalendarUpdaterConfig,
)
from algotrading.refdata.repositories.errors import AdvisoryLockUnavailable
from algotrading.refdata.stock_analysis.economic_calendar import ReconciliationSummary
from algotrading.refdata.stock_analysis.models import ProviderResult


NOW = datetime(2026, 8, 8, 16, tzinfo=timezone.utc)


class Repository:
    def __init__(self):
        self.coverages = []
        self.due = []
        self.next_timestamp = None
        self.coverage_calls = []
        self.due_calls = []

    def future_coverage(self, start, end):
        self.coverage_calls.append((start, end))
        return self.coverages.pop(0) if len(self.coverages) > 1 else self.coverages[0]

    def due_rows(self, start, end, now):
        self.due_calls.append((start, end, now))
        value = self.due.pop(0) if len(self.due) > 1 else (self.due[0] if self.due else [])
        return list(value)

    def next_unreleased(self, now, tomorrow):
        return self.next_timestamp


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
        self.calls = []
        self.error = error

    def fetch_economic_releases(self, start, end):
        self.calls.append((start, end))
        if self.error:
            raise self.error
        return ProviderResult(provider="TEST")


class Reconciler:
    def __init__(self):
        self.calls = []

    def reconcile(self, result, start, end, *, max_distance_days):
        self.calls.append((result, start, end, max_distance_days))
        return ReconciliationSummary(fetched=result.count)


class Calendar:
    def __init__(self, market=True, next_day=None):
        self.market = market
        self.next_day = next_day
        self.calls = []

    def is_market_day(self, day):
        self.calls.append(day)
        return self.market

    def next_market_day(self, after):
        return self.next_day


def updater(
    repository=None, locks=None, provider=None, reconciler=None,
    calendar=None, config=None, sleeper=None,
):
    return EconomicCalendarUpdater(
        {"ExchangeTZ": "America/New_York", "LocalTZ": "America/New_York"},
        repository or Repository(),
        locks or Locks(),
        reconciler or Reconciler(),
        calendar or Calendar(),
        provider=provider or Provider(),
        config=config or EconomicCalendarUpdaterConfig(),
        clock=lambda: NOW,
        sleeper=sleeper,
    )


class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_explicit_disable(self):
        config = EconomicCalendarUpdaterConfig.from_context({})
        self.assertTrue(config.enabled)
        self.assertEqual(config.horizon_days, 60)
        self.assertEqual(config.poll_interval_seconds, 1800)
        disabled = EconomicCalendarUpdaterConfig.from_context({
            "refdata.economic_calendar.auto_update.enabled": False
        })
        self.assertFalse(disabled.enabled)

    def test_invalid_config_is_rejected(self):
        key = "refdata.economic_calendar.auto_update."
        invalid = (
            {key + "enabled": "false"},
            {key + "horizon_days": 0},
            {key + "poll_interval_seconds": 59},
            {key + "poll_interval_seconds": 60.5},
            {key + "coverage_tail_slack_days": -1},
            {key + "daily_overlap_days": 8},
            {key + "lock_wait_seconds": -1},
        )
        for context in invalid:
            with self.subTest(context=context), self.assertRaises(ValueError):
                EconomicCalendarUpdaterConfig.from_context(context)


class StartupUpdaterTests(unittest.TestCase):
    def test_adequate_tail_coverage_makes_no_provider_call(self):
        repo = Repository()
        repo.coverages = [(3, datetime(2026, 10, 5, 8))]
        provider = Provider()
        result = updater(repository=repo, provider=provider).startup_check_once()
        self.assertFalse(result.provider_called)
        self.assertEqual(provider.calls, [])

    def test_market_window_is_converted_to_storage_timezone(self):
        repo = Repository()
        repo.coverages = [(1, datetime(2026, 10, 5, 8))]
        instance = EconomicCalendarUpdater(
            {"ExchangeTZ": "America/New_York", "LocalTZ": "America/Los_Angeles"},
            repo, Locks(), Reconciler(), Calendar(),
            provider=Provider(), config=EconomicCalendarUpdaterConfig(),
            clock=lambda: NOW,
        )
        instance.startup_check_once()
        start, end = repo.coverage_calls[0]
        self.assertEqual(start, datetime(2026, 8, 8, 9))
        self.assertEqual(end, datetime(2026, 10, 7, 21))

    def test_zero_or_short_coverage_fetches_full_horizon_even_on_weekend(self):
        for coverage in ((0, None), (2, datetime(2026, 8, 20))):
            with self.subTest(coverage=coverage):
                repo = Repository()
                repo.coverages = [coverage, coverage]
                provider = Provider()
                reconciler = Reconciler()
                result = updater(
                    repository=repo, provider=provider, reconciler=reconciler,
                    calendar=Calendar(market=False),
                ).startup_check_once()
                self.assertTrue(result.provider_called)
                self.assertEqual(provider.calls, [(date(2026, 8, 8), date(2026, 10, 7))])
                self.assertEqual(reconciler.calls[0][3], 7)

    def test_weekly_refresh_forces_shared_reconciliation_with_lookback(self):
        repo = Repository()
        repo.coverages = [(1, datetime(2026, 8, 29, 8))]
        provider = Provider()
        reconciler = Reconciler()
        instance = updater(
            repository=repo,
            provider=provider,
            reconciler=reconciler,
        )

        result = instance.weekly_refresh_once()

        self.assertEqual(
            provider.calls,
            [(date(2026, 7, 29), date(2026, 10, 7))],
        )
        self.assertTrue(result.provider_called)
        self.assertTrue(result.lock_acquired)
        self.assertIsNotNone(result.reconciliation)
        self.assertEqual(reconciler.calls[0][3], 7)

    def test_coverage_is_rechecked_after_lock(self):
        repo = Repository()
        repo.coverages = [
            (0, None),
            (1, datetime(2026, 10, 5, 8)),
        ]
        provider = Provider()
        locks = Locks()
        result = updater(repository=repo, provider=provider, locks=locks).startup_check_once()
        self.assertFalse(result.provider_called)
        self.assertTrue(result.lock_acquired)
        self.assertEqual(len(repo.coverage_calls), 2)
        self.assertEqual(locks.entered, locks.exited)

    def test_lock_contention_and_provider_failure_make_no_writes(self):
        repo = Repository()
        repo.coverages = [(0, None)]
        provider = Provider()
        reconciler = Reconciler()
        result = updater(
            repository=repo, locks=Locks(available=False),
            provider=provider, reconciler=reconciler,
        ).startup_check_once()
        self.assertEqual(result.error_code, "REFRESH_IN_PROGRESS")
        self.assertEqual(provider.calls, [])
        self.assertEqual(reconciler.calls, [])

        repo.coverages = [(0, None)]
        marker = "SYNTHETIC_CALENDAR_CREDENTIAL"
        failing = Provider(RuntimeError(f"api_key={marker}"))
        active = updater(
            repository=repo, provider=failing, reconciler=reconciler
        )
        with self.assertLogs(active.logger, level="WARNING") as logs:
            result = active.startup_check_once()
        self.assertTrue(result.provider_called)
        self.assertEqual(reconciler.calls, [])
        self.assertEqual(result.next_delay_seconds, 1800)
        self.assertNotIn(marker, "\n".join(logs.output))
        self.assertIn("RuntimeError", "\n".join(logs.output))


class MarketDayUpdaterTests(unittest.TestCase):
    def test_holiday_and_no_due_make_zero_provider_calls(self):
        repo = Repository()
        repo.due = [[]]
        provider = Provider()
        result = updater(
            repository=repo,
            provider=provider,
            calendar=Calendar(market=False, next_day=date(2026, 8, 10)),
        ).market_day_check_once()
        self.assertEqual(provider.calls, [])
        self.assertGreater(result.next_delay_seconds, 1800)
        result = updater(
            repository=repo, provider=provider, calendar=Calendar(market=True)
        ).market_day_check_once()
        self.assertEqual(provider.calls, [])

    def test_future_release_schedules_earlier_check_without_polling(self):
        repo = Repository()
        repo.due = [[]]
        repo.next_timestamp = datetime(2026, 8, 8, 12, 10)
        provider = Provider()
        result = updater(repository=repo, provider=provider).market_day_check_once()
        self.assertEqual(result.next_delay_seconds, 600)
        self.assertEqual(provider.calls, [])

    def test_due_release_polls_overlap_reconciles_and_stops_when_complete(self):
        repo = Repository()
        due = [{"actual": None}]
        repo.due = [due, due, []]
        provider = Provider()
        reconciler = Reconciler()
        locks = Locks()
        result = updater(
            repository=repo, provider=provider, reconciler=reconciler, locks=locks
        ).market_day_check_once()
        self.assertEqual(provider.calls, [(date(2026, 8, 7), date(2026, 8, 9))])
        self.assertEqual(reconciler.calls[0][3], 2)
        self.assertEqual(result.due_before, 1)
        self.assertEqual(result.due_after, 0)
        self.assertEqual(locks.entered, locks.exited)

    def test_remaining_due_or_failure_waits_full_interval(self):
        due = [{"actual": None}]
        repo = Repository()
        repo.due = [due, due, due]
        result = updater(repository=repo).market_day_check_once()
        self.assertEqual(result.next_delay_seconds, 1800)
        self.assertEqual(result.due_after, 1)

        repo = Repository()
        repo.due = [due, due]
        result = updater(
            repository=repo, provider=Provider(RuntimeError("failed"))
        ).market_day_check_once()
        self.assertEqual(result.next_delay_seconds, 1800)
        self.assertIsNotNone(result.error_code)

    def test_daily_lock_contention_suppresses_provider_call(self):
        repo = Repository()
        repo.due = [[{"actual": None}]]
        provider = Provider()
        result = updater(
            repository=repo, provider=provider, locks=Locks(available=False)
        ).market_day_check_once()
        self.assertEqual(result.error_code, "REFRESH_IN_PROGRESS")
        self.assertEqual(provider.calls, [])


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_starts_no_task(self):
        instance = updater(config=EconomicCalendarUpdaterConfig(enabled=False))
        self.assertFalse(instance.start())
        self.assertFalse(instance.running)
        await instance.stop()

    async def test_shutdown_cancels_sleeping_task(self):
        entered = asyncio.Event()

        async def sleeper(_):
            entered.set()
            await asyncio.Event().wait()

        repo = Repository()
        repo.coverages = [(1, datetime(2026, 10, 5, 8))]
        instance = updater(repository=repo, sleeper=sleeper)
        self.assertTrue(instance.start())
        await asyncio.wait_for(entered.wait(), timeout=1)
        await instance.stop(timeout_seconds=1)
        self.assertFalse(instance.running)

    async def test_shutdown_joins_slow_provider_and_releases_lock(self):
        entered = threading.Event()
        release = threading.Event()

        class SlowProvider(Provider):
            def fetch_economic_releases(self, start, end):
                entered.set()
                release.wait(2)
                return super().fetch_economic_releases(start, end)

            def close(self):
                self.closed = True

        repo = Repository()
        repo.coverages = [(0, None), (0, None)]
        locks = Locks()
        provider = SlowProvider()
        instance = updater(repository=repo, locks=locks, provider=provider)
        self.assertTrue(instance.start())
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        stop_task = asyncio.create_task(instance.stop(timeout_seconds=1))
        release.set()
        await stop_task
        self.assertEqual(locks.entered, locks.exited)
        self.assertFalse(instance.running)
        self.assertTrue(provider.closed)

    async def test_join_timeout_is_explicit_and_handle_is_retained(self):
        entered = threading.Event()
        release = threading.Event()

        class SlowProvider(Provider):
            def fetch_economic_releases(self, start, end):
                entered.set()
                release.wait(2)
                return ProviderResult(provider="TEST")

        repo = Repository()
        repo.coverages = [(0, None), (0, None)]
        instance = updater(repository=repo, provider=SlowProvider())
        instance.start()
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        with self.assertRaises(asyncio.TimeoutError):
            await instance.stop(timeout_seconds=0.01)
        self.assertIsNotNone(instance._task)
        release.set()
        await instance.stop(timeout_seconds=1)
        self.assertFalse(instance.running)

    async def test_repeated_start_stop_owns_no_worker(self):
        repo = Repository()
        repo.coverages = [(1, datetime(2026, 10, 5, 8))]
        instance = updater(repository=repo)
        for _ in range(2):
            self.assertTrue(instance.start())
            await asyncio.sleep(0)
            await instance.stop(timeout_seconds=1)
            self.assertIsNone(instance._executor)
            self.assertIsNone(instance._inflight)


if __name__ == "__main__":
    unittest.main()
