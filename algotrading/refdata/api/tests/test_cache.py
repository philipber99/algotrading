from __future__ import annotations

import sqlite3
import sys
import threading
import time
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.cache import PersistentDailyBarsCache, SingleFlightTTLCache


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class CacheTests(unittest.TestCase):
    def test_hit_and_expiry(self):
        clock = ManualClock()
        cache = SingleFlightTTLCache(max_entries=2, clock=clock)
        loads = 0

        def loader():
            nonlocal loads
            loads += 1
            return loads

        self.assertFalse(cache.get_or_load("a", 5, loader).hit)
        self.assertTrue(cache.get_or_load("a", 5, loader).hit)
        self.assertEqual(loads, 1)

        clock.value = 6
        result = cache.get_or_load("a", 5, loader)
        self.assertFalse(result.hit)
        self.assertEqual(result.value, 2)

    def test_lru_bound(self):
        cache = SingleFlightTTLCache(max_entries=2)
        cache.get_or_load("a", 60, lambda: 1)
        cache.get_or_load("b", 60, lambda: 2)
        cache.get_or_load("c", 60, lambda: 3)
        self.assertEqual(cache.stats()["entries"], 2)

    def test_duplicate_load_is_coalesced(self):
        cache = SingleFlightTTLCache(max_entries=2)
        loads = 0
        barrier = threading.Barrier(3)
        results = []

        def loader():
            nonlocal loads
            loads += 1
            time.sleep(0.05)
            return 42

        def worker():
            barrier.wait()
            results.append(cache.get_or_load("same", 60, loader).value)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(loads, 1)
        self.assertEqual(results, [42, 42])

    def test_daily_bars_survive_restart_and_invalidate_next_calendar_day(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily-bars.sqlite3"
            row = {
                "symbol": "MSFT",
                "timestamp": "2026-07-23T00:00:00+00:00",
                "open": 100,
                "high": 102,
                "low": 99,
                "close": 101,
                "volume": 1000,
            }
            cache = PersistentDailyBarsCache(path)
            cache.store(
                ["MSFT"],
                [row],
                start=date(2026, 7, 1),
                end=date(2026, 7, 24),
                cache_date=date(2026, 7, 24),
            )
            cache.close()

            restarted = PersistentDailyBarsCache(path)
            rows, covered = restarted.load(
                ["MSFT"],
                start=date(2026, 7, 1),
                end=date(2026, 7, 24),
                cache_date=date(2026, 7, 24),
            )
            self.assertEqual(covered, {"MSFT"})
            self.assertEqual(rows[0]["close"], 101)

            rows, covered = restarted.load(
                ["MSFT"],
                start=date(2026, 7, 1),
                end=date(2026, 7, 24),
                cache_date=date(2026, 7, 25),
            )
            self.assertEqual(rows, [])
            self.assertEqual(covered, set())
            restarted.close()

    def test_legacy_empty_daily_bar_coverage_is_treated_as_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily-bars.sqlite3"
            cache = PersistentDailyBarsCache(path)
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    INSERT INTO daily_bar_coverage
                        (cache_date, symbol, range_start, range_end)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("2026-08-19", "GEV", "2026-07-22", "2026-08-20"),
                )

            rows, covered = cache.load(
                ["GEV"],
                start=date(2026, 7, 22),
                end=date(2026, 8, 20),
                cache_date=date(2026, 8, 19),
            )

            self.assertEqual(rows, [])
            self.assertEqual(covered, set())
            cache.close()


if __name__ == "__main__":
    unittest.main()
