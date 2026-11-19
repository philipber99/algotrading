from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ..builder import BuildOptions, CacheBuilder
from ..config import CacheConfig
from ..errors import SourceBacklogError
from ..queries import SymbolSourceWatermark
from ..sampling import SampledSnapshot, SurfaceStateSnapshot, TradingSession


UTC = timezone.utc


class _Calendar:
    def __init__(self, sessions):
        self.values = tuple(sessions)

    def sessions(self, start, end):
        return tuple(
            item for item in self.values if start <= item.session_date <= end
        )


def _sessions(count: int = 5):
    first = date(2026, 7, 20)
    result = []
    for offset in range(count):
        session_date = first + timedelta(days=offset)
        opened = datetime.combine(
            session_date,
            datetime.min.time(),
            tzinfo=UTC,
        ) + timedelta(hours=13, minutes=30)
        result.append(
            TradingSession(
                session_date,
                opened,
                opened + timedelta(minutes=5),
                False,
            )
        )
    return tuple(result)


class BuildPlanningTests(unittest.TestCase):
    def _builder(self, root: Path, sessions, *, symbols="MSFT"):
        context = {
            "options_chain.time_of_day_cache.output_directory": str(root),
            "options_chain.time_of_day_cache.symbols": symbols,
            "options_chain.time_of_day_cache.windows": "short:3",
            "options_chain.time_of_day_cache.sample_interval_minutes": 5,
        }
        config = CacheConfig.from_context(context)
        return CacheBuilder(
            context,
            client=object(),
            config=config,
            calendar=_Calendar(sessions),
            now_fn=lambda: datetime(2026, 8, 1, tzinfo=UTC),
        )

    def test_weekly_candidate_is_bounded_to_largest_window(self):
        sessions = _sessions()
        with tempfile.TemporaryDirectory() as raw:
            builder = self._builder(Path(raw), sessions)
            result = builder._candidate_sessions(
                sessions[-1].close_utc,
                mode="weekly",
                active=None,
                session_dates=(),
                history_start=None,
                history_end=None,
            )
        self.assertEqual(
            [item.session_date for item in result],
            [item.session_date for item in sessions[-3:]],
        )

    def test_initial_build_skips_only_nonexistent_leading_history(self):
        sessions = _sessions(3)
        sampled = []
        for index, session in enumerate(sessions[1:], start=1):
            state = SurfaceStateSnapshot(
                f"00000000-0000-0000-0000-{index:012d}",
                "MSFT",
                session.close_utc,
                1,
                1,
                None,
                "complete",
            )
            sampled.append(
                SampledSnapshot(state, session, 5, session.close_utc, 0.0)
            )
        with tempfile.TemporaryDirectory() as raw:
            builder = self._builder(Path(raw), sessions)
            available = builder._available_sessions(
                sessions,
                sampled,
                symbols=("MSFT",),
                strict=False,
            )
            with self.assertRaises(SourceBacklogError):
                builder._available_sessions(
                    sessions,
                    sampled,
                    symbols=("MSFT",),
                    strict=True,
                )
        self.assertEqual(
            [item.session_date for item in available],
            [item.session_date for item in sessions[1:]],
        )

    def test_retired_symbol_stops_at_aligned_last_chain_and_surface(self):
        sessions = _sessions(3)
        sampled = []
        for symbol, symbol_sessions in (
            ("ACTIVE", sessions),
            ("RETIRED", sessions[:2]),
        ):
            for index, session in enumerate(symbol_sessions, start=1):
                state = SurfaceStateSnapshot(
                    f"{symbol}-{index}",
                    symbol,
                    session.close_utc,
                    1,
                    1,
                    None,
                    "complete",
                )
                sampled.append(
                    SampledSnapshot(state, session, 5, session.close_utc, 0.0)
                )
        watermarks = {
            "ACTIVE": SymbolSourceWatermark(
                "ACTIVE", sessions[-1].close_utc, sessions[-1].close_utc
            ),
            "RETIRED": SymbolSourceWatermark(
                "RETIRED", sessions[1].close_utc, sessions[1].close_utc
            ),
        }

        with tempfile.TemporaryDirectory() as raw:
            builder = self._builder(
                Path(raw), sessions, symbols="ACTIVE,RETIRED"
            )
            available, gaps = builder._available_sessions_by_symbol(
                sessions,
                sampled,
                symbols=("ACTIVE", "RETIRED"),
                source_watermarks=watermarks,
                strict=False,
            )

        self.assertEqual(
            [item.session_date for item in available["ACTIVE"]],
            [item.session_date for item in sessions],
        )
        self.assertEqual(
            [item.session_date for item in available["RETIRED"]],
            [item.session_date for item in sessions[:2]],
        )
        self.assertEqual(gaps, {})

    def test_newer_chain_without_surface_isolated_as_a_backlog(self):
        sessions = _sessions(3)
        sampled = []
        for index, session in enumerate(sessions[:2], start=1):
            state = SurfaceStateSnapshot(
                f"retired-{index}",
                "RETIRED",
                session.close_utc,
                1,
                1,
                None,
                "complete",
            )
            sampled.append(
                SampledSnapshot(state, session, 5, session.close_utc, 0.0)
            )
        watermarks = {
            "RETIRED": SymbolSourceWatermark(
                "RETIRED", sessions[-1].close_utc, sessions[1].close_utc
            )
        }

        with tempfile.TemporaryDirectory() as raw:
            builder = self._builder(Path(raw), sessions, symbols="RETIRED")
            available, gaps = builder._available_sessions_by_symbol(
                sessions,
                sampled,
                symbols=("RETIRED",),
                source_watermarks=watermarks,
                strict=False,
            )

        self.assertEqual(
            [item.session_date for item in available["RETIRED"]],
            [item.session_date for item in sessions[:2]],
        )
        details = gaps["RETIRED"]
        self.assertEqual(
            details["missing_sessions"],
            [sessions[-1].session_date.isoformat()],
        )
        self.assertFalse(details["surface_is_current"])

    def test_symbol_resumes_after_gap_and_later_sessions_are_selected(self):
        sessions = _sessions(4)
        sampled = []
        for index in (0, 1, 3):
            session = sessions[index]
            state = SurfaceStateSnapshot(
                f"resumed-{index}",
                "RESUMED",
                session.close_utc,
                1,
                1,
                None,
                "complete",
            )
            sampled.append(
                SampledSnapshot(state, session, 5, session.close_utc, 0.0)
            )
        watermarks = {
            "RESUMED": SymbolSourceWatermark(
                "RESUMED", sessions[-1].close_utc, sessions[-1].close_utc
            )
        }

        with tempfile.TemporaryDirectory() as raw:
            builder = self._builder(Path(raw), sessions, symbols="RESUMED")
            available, gaps = builder._available_sessions_by_symbol(
                sessions,
                sampled,
                symbols=("RESUMED",),
                source_watermarks=watermarks,
                strict=False,
                report_leading_gaps=True,
            )

        self.assertEqual(
            [item.session_date for item in available["RESUMED"]],
            [sessions[0].session_date, sessions[1].session_date, sessions[3].session_date],
        )
        self.assertEqual(
            gaps["RESUMED"]["missing_sessions"],
            [sessions[2].session_date.isoformat()],
        )

    def test_explicit_incomplete_symbol_request_still_fails(self):
        sessions = _sessions(2)
        state = SurfaceStateSnapshot(
            "retired-1",
            "RETIRED",
            sessions[0].close_utc,
            1,
            1,
            None,
            "complete",
        )
        sampled = [
            SampledSnapshot(state, sessions[0], 5, sessions[0].close_utc, 0.0)
        ]
        watermarks = {
            "RETIRED": SymbolSourceWatermark(
                "RETIRED", sessions[-1].close_utc, sessions[0].close_utc
            )
        }

        with tempfile.TemporaryDirectory() as raw:
            builder = self._builder(Path(raw), sessions, symbols="RETIRED")
            with self.assertRaises(SourceBacklogError):
                builder._available_sessions_by_symbol(
                    sessions,
                    sampled,
                    symbols=("RETIRED",),
                    source_watermarks=watermarks,
                    strict=True,
                )

    def test_daily_targets_use_each_symbols_own_cached_watermark(self):
        sessions = _sessions(3)
        active = {
            "data_through_session": sessions[-1].session_date.isoformat(),
            "data_through_session_by_symbol": {
                "ACTIVE": sessions[-1].session_date.isoformat(),
                "BACKLOG": sessions[0].session_date.isoformat(),
            },
        }
        sessions_by_symbol = {
            "ACTIVE": sessions,
            "BACKLOG": sessions,
        }

        with tempfile.TemporaryDirectory() as raw:
            builder = self._builder(
                Path(raw), sessions, symbols="ACTIVE,BACKLOG"
            )
            targets, bootstrap = builder._target_sessions_by_symbol(
                "daily",
                sessions_by_symbol,
                active,
                BuildOptions(mode="daily"),
            )

        self.assertFalse(bootstrap)
        self.assertEqual(targets["ACTIVE"], ())
        self.assertEqual(
            [item.session_date for item in targets["BACKLOG"]],
            [item.session_date for item in sessions[1:]],
        )

    def test_daily_manifest_preserves_old_gaps_but_does_not_retry_them(self):
        sessions = _sessions(4)
        missing_date = sessions[1].session_date.isoformat()
        active = {
            "coverage_gaps_by_symbol": {
                "RESUMED": {
                    "reason": "missing_sampling_coverage",
                    "missing_sessions": [missing_date],
                }
            }
        }

        merged = CacheBuilder._merge_coverage_gaps(
            active,
            {},
            {"RESUMED": (sessions[-1],)},
        )

        self.assertEqual(
            merged["RESUMED"]["missing_sessions"],
            [missing_date],
        )


if __name__ == "__main__":
    unittest.main()
