from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone

from ..config import CacheQueryConfig, ConfigurationError, parse_windows
from ..sampling import (
    SurfaceStateSnapshot,
    TradingSession,
    sample_surface_states,
)


UTC = timezone.utc


class WindowConfigTests(unittest.TestCase):
    def test_multiple_concise_windows(self):
        windows = parse_windows(
            {
                "options_chain.time_of_day_cache.windows": "10d:10,3m:63",
            }
        )
        self.assertEqual(
            [(item.name, item.sessions) for item in windows],
            [("10d", 10), ("3m", 63)],
        )

    def test_json_and_per_window_override(self):
        windows = parse_windows(
            {
                "options_chain.time_of_day_cache.windows": '{"short": 5}',
                "options_chain.time_of_day_cache.window.long.sessions": 100,
                "options_chain.time_of_day_cache.window.long.label": "Long",
            }
        )
        self.assertEqual(
            [(item.name, item.sessions, item.label) for item in windows],
            [("short", 5, "short"), ("long", 100, "Long")],
        )

    def test_api_pricing_pool_defaults_and_explicit_values(self):
        automatic = CacheQueryConfig.from_context({})
        self.assertGreaterEqual(automatic.resolved_pricing_workers, 1)
        self.assertEqual(
            automatic.resolved_max_pending_pricing_batches,
            automatic.resolved_pricing_workers * 2,
        )
        explicit = CacheQueryConfig.from_context(
            {
                "options_chain.time_of_day_api.pricing_workers": 3,
                "options_chain.time_of_day_api.pricing_batch_size": 128,
                "options_chain.time_of_day_api.max_pending_pricing_batches": 5,
            }
        )
        self.assertEqual(explicit.resolved_pricing_workers, 3)
        self.assertEqual(explicit.pricing_batch_size, 128)
        self.assertEqual(explicit.resolved_max_pending_pricing_batches, 5)

    def test_negative_pricing_pool_values_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            CacheQueryConfig.from_context(
                {"options_chain.time_of_day_api.pricing_workers": -1}
            )


class SamplingTests(unittest.TestCase):
    def test_latest_at_or_before_target_and_no_future_lookahead(self):
        session = TradingSession(
            date(2026, 7, 27),
            datetime(2026, 7, 27, 13, 30, tzinfo=UTC),
            datetime(2026, 7, 27, 13, 45, tzinfo=UTC),
            False,
        )
        states = [
            SurfaceStateSnapshot(
                "a",
                "MSFT",
                session.open_utc + timedelta(minutes=4, seconds=30),
                1,
                1,
                None,
                "complete",
            ),
            SurfaceStateSnapshot(
                "future",
                "MSFT",
                session.open_utc + timedelta(minutes=5, seconds=1),
                1,
                1,
                None,
                "complete",
            ),
        ]
        sampled = sample_surface_states(
            states,
            [session],
            interval_minutes=5,
            max_age_seconds=120,
        )
        first = next(item for item in sampled if item.session_minute == 5)
        self.assertEqual(first.state.snapshot_id, "a")
        self.assertGreaterEqual(first.age_seconds, 0)

    def test_stale_snapshot_does_not_fill_bin(self):
        session = TradingSession(
            date(2026, 7, 27),
            datetime(2026, 7, 27, 13, 30, tzinfo=UTC),
            datetime(2026, 7, 27, 13, 40, tzinfo=UTC),
            False,
        )
        state = SurfaceStateSnapshot(
            "old",
            "MSFT",
            session.open_utc,
            1,
            1,
            None,
            "complete",
        )
        sampled = sample_surface_states(
            [state],
            [session],
            interval_minutes=5,
            max_age_seconds=60,
        )
        self.assertEqual(sampled, ())


if __name__ == "__main__":
    unittest.main()
