from __future__ import annotations

import logging
import sys
import tempfile
import threading
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.service import IntradayDataUnavailableError, InvalidRequestError, RefDataService
from api.tests.fakes import FIXED_NOW, FakeMarketData


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.market_data = FakeMarketData()
        self.context = {
            "logger": logging.getLogger("test-refdata-api"),
            "local_timezone": "UTC",
            "refdata.api.timezone": "UTC",
            "refdata.api.cache.enabled": True,
            "refdata.api.calendar_refresh_seconds": 999999,
        }
        self.now_value = FIXED_NOW
        self.service = RefDataService(
            self.context,
            market_data_module=self.market_data,
            now_fn=lambda: self.now_value,
        )
        self.service.startup()

    def tearDown(self):
        self.service.shutdown()

    def test_market_status(self):
        result = self.service.market_status()
        self.assertTrue(result.data["is_trading_day"])
        self.assertTrue(result.data["is_trading_hours"])
        self.assertEqual(result.data["session"]["date"], "2026-07-24")

    def test_startup_health_does_not_expose_provider_exception_text(self):
        marker = "SYNTHETIC_STARTUP_CREDENTIAL"
        self.service._provider = Mock(side_effect=RuntimeError(marker))

        with self.assertRaises(RuntimeError):
            self.service.startup()

        details = self.service.health_details()
        self.assertFalse(details["ready"])
        self.assertEqual(details["startup_error"], "RuntimeError")
        self.assertNotIn(marker, str(details))

    def test_bars_are_cached_and_do_not_request_gap_repairs(self):
        start = datetime(2026, 7, 23, tzinfo=timezone.utc)
        end = datetime(2026, 7, 23, 23, 59, tzinfo=timezone.utc)

        first = self.service.bars("msft", start, end, "1d")
        second = self.service.bars("MSFT", start, end, "1d")

        self.assertEqual(first.cache_state, "miss")
        self.assertEqual(second.cache_state, "hit")
        self.assertEqual(self.market_data.calls["prices_sql"], 1)
        self.assertEqual(self.market_data.last_prices_kwargs["frequency"], "1day")
        self.assertFalse(self.market_data.last_prices_kwargs["raise_missing"])
        self.assertFalse(hasattr(self.market_data, "update_symbol_prices"))

    def test_remote_minute_history_fills_an_empty_sql_session_and_caches_it(self):
        self.market_data.prices_sql = Mock(return_value=pd.DataFrame())
        start = datetime(2026, 7, 24)
        end = datetime(2026, 7, 24, 23, 59, 59)
        first = self.service.bars("RSP", start, end, "1m", remote_query=True)
        second = self.service.bars("RSP", start, end, "1m", remote_query=True)
        self.assertEqual(len(first.data), 2)
        self.assertEqual(first.data[0]["timestamp"], "2026-07-24T09:30:00+00:00")
        self.assertEqual(second.cache_state, "hit")
        self.assertEqual(self.market_data.calls["provider_get_bars"], 1)
        self.assertEqual(self.market_data.prices_sql.call_count, 1)
        self.assertEqual(self.market_data.refdata_source.last_bars_args, ("RSP", start, end, "1min"))

    def test_remote_minute_history_keeps_sql_when_available(self):
        result = self.service.bars("RSP", datetime(2026, 7, 23), datetime(2026, 7, 23, 23, 59), "1m", remote_query=True)
        self.assertEqual(len(result.data), 2)
        self.assertEqual(self.market_data.calls.get("provider_get_bars", 0), 0)

    def test_remote_minute_history_is_opt_in_and_bounded_to_one_local_date(self):
        self.market_data.prices_sql = Mock(return_value=pd.DataFrame())
        with self.assertRaises(IntradayDataUnavailableError):
            self.service.bars("RSP", datetime(2026, 7, 23), datetime(2026, 7, 23, 23, 59), "1m")
        with self.assertRaisesRegex(InvalidRequestError, "one local date"):
            self.service.bars("RSP", datetime(2026, 7, 23), datetime(2026, 7, 24), "1m", remote_query=True)
        self.assertEqual(self.market_data.calls.get("provider_get_bars", 0), 0)

    def test_remote_minute_history_does_not_substitute_today_or_daily_bars(self):
        self.market_data.prices_sql = Mock(return_value=pd.DataFrame())
        # The fake provider returns July 24 even when July 23 is requested.
        with self.assertRaises(IntradayDataUnavailableError):
            self.service.bars("RSP", datetime(2026, 7, 23), datetime(2026, 7, 23, 23, 59), "1m", remote_query=True)
        self.assertEqual(self.market_data.calls["provider_get_bars"], 1)
        self.assertEqual(self.market_data.refdata_source.last_bars_args[-1], "1min")

    def test_remote_minute_history_preserves_regular_session_filter(self):
        self.market_data.prices_sql = Mock(return_value=pd.DataFrame())
        self.service.bars("RSP", datetime(2026, 7, 24), datetime(2026, 7, 24, 23, 59), "1m", market_hours_only=True, remote_query=True)
        _, start, end, _ = self.market_data.refdata_source.last_bars_args
        self.assertEqual(start, datetime(2026, 7, 24, 9, 30))
        self.assertEqual(end, datetime(2026, 7, 24, 15, 59, 59, 999999))

    def test_remote_minute_history_filters_utc_provider_rows_in_local_timezone(self):
        self.service.local_timezone = "America/Los_Angeles"
        self.market_data.prices_sql = Mock(return_value=pd.DataFrame())
        frame = pd.DataFrame({"open": [1, 2, 3], "close": [1, 2, 3]}, index=pd.to_datetime([
            "2026-11-05T07:59:00Z", "2026-11-05T08:00:00Z", "2026-11-06T07:59:00Z",
        ]))
        self.market_data.refdata_source.get_bars = Mock(return_value=frame)
        result = self.service.bars("RSP", datetime(2026, 11, 5), datetime(2026, 11, 5, 23, 59, 59), "1m", remote_query=True)
        self.assertEqual([row["close"] for row in result.data], [2, 3])

    def test_bulk_daily_bars_use_one_sql_read_and_never_load_per_symbol_quotes(self):
        class Repository:
            def __init__(self):
                self.calls = []

            def daily_bars(self, symbols, *, start, end):
                self.calls.append((tuple(symbols), start, end))
                return [
                    {
                        "symbol": symbol,
                        "timestamp": datetime(2026, 7, 23),
                        "open": 100.0,
                        "high": 102.0,
                        "low": 99.0,
                        "close": 101.0,
                        "volume": 1000,
                    }
                    for symbol in symbols
                ]

        repository = Repository()
        self.service.configure_daily_bars_repository(repository)
        result = self.service.bulk_bars(
            ["MSFT", "AAPL"],
            datetime(2026, 7, 1, tzinfo=timezone.utc),
            datetime(2026, 7, 24, tzinfo=timezone.utc),
        )

        self.assertEqual(len(repository.calls), 1)
        self.assertEqual(repository.calls[0][0], ("MSFT", "AAPL"))
        self.assertEqual([item["symbol"] for item in result.data], ["MSFT", "AAPL"])
        self.assertEqual(self.market_data.calls.get("retrieve_current_price", 0), 0)
        self.assertEqual(self.market_data.calls.get("provider_get_quotes", 0), 0)

    def test_bulk_daily_bars_rechecks_sql_after_an_empty_result(self):
        class Repository:
            def __init__(self):
                self.calls = 0

            def daily_bars(self, symbols, *, start, end):
                self.calls += 1
                if self.calls == 1:
                    return []
                return [
                    {
                        "symbol": symbol,
                        "timestamp": datetime(2026, 8, 18),
                        "open": 630.0,
                        "high": 645.0,
                        "low": 625.0,
                        "close": 640.0,
                        "volume": 1000,
                    }
                    for symbol in symbols
                ]

        with tempfile.TemporaryDirectory() as directory:
            context = {
                **self.context,
                "refdata.api.daily_bars_cache.path": str(
                    Path(directory) / "daily-bars.sqlite3"
                ),
            }
            service = RefDataService(
                context,
                market_data_module=self.market_data,
                now_fn=lambda: datetime(2026, 8, 19, tzinfo=timezone.utc),
            )
            repository = Repository()
            service.configure_daily_bars_repository(repository)
            try:
                first = service.bulk_bars(
                    ["GEV"],
                    datetime(2026, 7, 22, tzinfo=timezone.utc),
                    datetime(2026, 8, 20, tzinfo=timezone.utc),
                )
                second = service.bulk_bars(
                    ["GEV"],
                    datetime(2026, 7, 22, tzinfo=timezone.utc),
                    datetime(2026, 8, 20, tzinfo=timezone.utc),
                )
            finally:
                service.shutdown()

        self.assertEqual(first.data[0]["bars"], [])
        self.assertEqual(repository.calls, 2)
        self.assertEqual(second.data[0]["bars"][0]["close"], 640.0)

    def test_bulk_daily_bars_can_append_quotes_with_one_batch_provider_call(self):
        class Repository:
            def daily_bars(self, symbols, *, start, end):
                return []

        self.service.configure_daily_bars_repository(Repository())
        result = self.service.bulk_bars(
            ["MSFT", "AAPL"],
            datetime(2026, 7, 1, tzinfo=timezone.utc),
            datetime(2026, 7, 24, tzinfo=timezone.utc),
            include_current_quote=True,
        )

        self.assertEqual(self.market_data.calls["provider_get_quotes"], 1)
        self.assertTrue(all(item["bars"][0]["provisional"] for item in result.data))

    def test_bulk_daily_bars_remote_query_repairs_missing_database_history(self):
        class Repository:
            def daily_bars(self, symbols, *, start, end):
                return []

        self.service.configure_daily_bars_repository(Repository())
        result = self.service.bulk_bars(
            ["MSFT"],
            datetime(2026, 7, 23, tzinfo=timezone.utc),
            datetime(2026, 7, 24, tzinfo=timezone.utc),
            remote_query=True,
        )

        self.assertEqual(self.market_data.calls.get("prices_sql", 0), 0)
        self.assertEqual(self.market_data.calls["provider_get_bars"], 1)
        self.assertEqual(result.data[0]["symbol"], "MSFT")
        self.assertEqual(result.data[0]["bars"][0]["close"], 100.5)

    def test_minute_bars_preserve_today_price_loading(self):
        start = datetime(2026, 7, 23, tzinfo=timezone.utc)
        end = datetime(2026, 7, 23, 23, 59, tzinfo=timezone.utc)

        self.service.bars("SPY", start, end, "1m", market_hours_only=True)

        self.assertNotIn(
            "refdata.load_intraday_for_today",
            self.market_data.last_prices_context,
        )
        self.assertTrue(self.market_data.last_prices_kwargs["mkt_hours_only"])

    def test_current_price_prefers_normalized_mark_when_provider_omits_last_price(self):
        result = self.service.current_price("MSFT", "price")
        self.assertEqual(result.data["price"], 101.1)
        self.assertEqual(self.market_data.calls["provider_get_quotes"], 1)
        self.assertEqual(self.market_data.calls.get("retrieve_current_price", 0), 0)

    def test_current_futures_prices_use_explicit_provider_symbols_and_marks(self):
        contracts = ("ESU26", "NQU26", "ZNU26", "CLV26", "6EU26")

        for contract in contracts:
            with self.subTest(contract=contract):
                result = self.service.current_price(
                    contract.lower(), "price", instrument_type="FUTURE"
                )
                self.assertEqual(
                    result.data, {"symbol": contract, "price": 101.1}
                )
                self.assertEqual(
                    self.market_data.refdata_source.last_quote_symbols,
                    [f"/{contract}"],
                )

        self.assertEqual(
            self.market_data.calls.get("provider_get_quotes", 0), len(contracts)
        )
        self.assertEqual(self.market_data.calls.get("retrieve_current_price", 0), 0)

    def test_day_quotes_use_configured_provider_on_trading_days(self):
        result = self.service.day_quotes(["msft", "AAPL", "MSFT"])

        self.assertEqual([row["symbol"] for row in result.data], ["MSFT", "AAPL"])
        self.assertEqual(result.data[0]["open"], 100.0)
        self.assertEqual(result.data[0]["mark"], 101.1)
        self.assertEqual(result.data[0]["quote_time"], FIXED_NOW.isoformat())
        self.assertEqual(result.data[0]["trade_time"], FIXED_NOW.isoformat())
        self.assertEqual(result.data[0]["source_kind"], "exchange_quote")
        self.assertEqual(self.market_data.calls["provider_get_quotes"], 1)
        self.assertEqual(self.market_data.calls.get("prices_sql", 0), 0)

    def test_precached_quotes_serve_gui_and_internal_current_price_without_remote_calls(self):
        self.service.configure_quote_precache(lambda: ["MSFT", "AAPL", "MSFT"])

        count = self.service.refresh_quote_precache(force=True)
        quote_calls = self.market_data.calls["provider_get_quotes"]
        quotes = self.service.day_quotes(["MSFT", "AAPL"])
        current = self.service.current_price("MSFT", "price")

        self.assertEqual(count, 2)
        self.assertEqual([row["symbol"] for row in quotes.data], ["MSFT", "AAPL"])
        self.assertEqual(current.data["price"], 101.1)
        self.assertEqual(current.cache_state, "hit")
        self.assertEqual(self.market_data.calls["provider_get_quotes"], quote_calls)
        self.assertEqual(self.market_data.calls.get("retrieve_current_price", 0), 0)

    def test_stale_market_hours_quote_falls_back_to_one_provider_batch(self):
        self.context["refdata.api.quote_precache.poll_interval_seconds"] = 60
        self.service.configure_quote_precache(lambda: ["MSFT", "AAPL"])
        self.service.refresh_quote_precache(force=True)
        self.now_value += timedelta(seconds=60)

        quotes = self.service.day_quotes(["MSFT", "AAPL"])

        self.assertEqual(len(quotes.data), 2)
        self.assertEqual(self.market_data.calls["provider_get_quotes"], 2)

    def test_precache_polls_once_while_closed_and_refreshes_at_market_open(self):
        self.market_data.trading_hours = False
        self.service.configure_quote_precache(lambda: ["MSFT"])

        self.assertEqual(self.service.refresh_quote_precache(force=True), 1)
        self.assertEqual(self.service.refresh_quote_precache(force=False), 0)
        closed = self.service.day_quotes(["MSFT"])
        self.assertEqual(
            closed.data[0]["source_kind"], "completed_session_mark"
        )
        self.assertEqual(self.market_data.calls.get("provider_get_quotes", 0), 0)

        self.market_data.trading_hours = True
        self.service.day_quotes(["MSFT"])
        self.assertEqual(self.market_data.calls["provider_get_quotes"], 1)

    def test_market_open_expires_off_hours_snapshot_and_closed_mode_recovers_close(self):
        self.market_data.trading_hours = False
        self.service.configure_quote_precache(lambda: ["MSFT"])
        self.service.refresh_quote_precache(force=True)

        def unavailable_quotes(symbols):
            self.market_data._called("provider_get_quotes")
            return {}

        self.market_data.refdata_source.get_quotes = unavailable_quotes
        self.market_data.trading_hours = True
        self.assertEqual(self.service.day_quotes(["MSFT"]).data, [])

        self.market_data.trading_hours = False
        recovered = self.service.day_quotes(["MSFT"])
        self.assertEqual(recovered.data[0]["mark"], 101.0)
        self.assertEqual(
            recovered.data[0]["source_kind"], "completed_session_mark"
        )
        self.assertEqual(self.market_data.calls["provider_get_quotes"], 1)

    def test_premarket_day_quotes_use_previous_completed_session_close(self):
        self.market_data.trading_day = True
        self.market_data.trading_hours = False

        result = self.service.day_quotes(["MSFT"])

        self.assertEqual(result.data, [{
            "symbol": "MSFT",
            "session_date": "2026-07-23",
            "open": 100.0,
            "mark": 101.0,
            "quote_time": "2026-07-23T16:00:00+00:00",
            "trade_time": None,
            "source_kind": "completed_session_mark",
        }])
        self.assertEqual(self.market_data.calls.get("provider_get_quotes", 0), 0)
        self.assertEqual(self.service.latest_completed_session_close(), datetime(
            2026, 7, 23, 16, 0, tzinfo=timezone.utc
        ))

    def test_after_close_day_quotes_use_batched_provider_before_daily_bar_lands(self):
        self.market_data.trading_day = True
        self.market_data.trading_hours = False
        self.now_value = datetime(2026, 7, 24, 17, 0, tzinfo=timezone.utc)
        self.market_data.prices_sql = lambda *args, **kwargs: pd.DataFrame()

        result = self.service.day_quotes(["MSFT", "AAPL"])

        self.assertEqual([row["symbol"] for row in result.data], ["MSFT", "AAPL"])
        self.assertEqual(result.data[0]["open"], 100.0)
        self.assertEqual(result.data[0]["mark"], 101.1)
        self.assertEqual(result.data[0]["source_kind"], "exchange_quote")
        self.assertEqual(self.market_data.calls["provider_get_quotes"], 1)
        self.assertEqual(self.market_data.calls.get("prices_sql", 0), 0)

    def test_premarket_discards_cached_intraday_quote_for_completed_close(self):
        self.service.configure_quote_precache(lambda: ["MSFT"])
        self.service.refresh_quote_precache(force=True)
        self.assertEqual(self.market_data.calls["provider_get_quotes"], 1)
        self.market_data.trading_hours = False

        quote = self.service.day_quotes(["MSFT"]).data[0]

        self.assertEqual(quote["mark"], 101.0)
        self.assertEqual(quote["session_date"], "2026-07-23")
        self.assertEqual(quote["source_kind"], "completed_session_mark")
        self.assertEqual(self.market_data.calls["provider_get_quotes"], 1)

    def test_precache_runs_on_an_independent_stoppable_thread(self):
        refreshed = threading.Event()
        original = self.market_data.refdata_source.get_quotes

        def observed(symbols):
            try:
                return original(symbols)
            finally:
                refreshed.set()

        self.market_data.refdata_source.get_quotes = observed
        self.context["refdata.api.quote_precache.enabled"] = True
        self.context["refdata.api.quote_precache.poll_interval_seconds"] = 60
        self.service.configure_quote_precache(lambda: ["MSFT"])

        self.service.start_quote_precache()
        self.assertTrue(refreshed.wait(1))
        self.assertTrue(self.service.quote_cache_details()["thread_running"])
        self.service.stop_quote_precache()
        self.assertFalse(self.service.quote_cache_details()["thread_running"])

    def test_day_quotes_use_last_trading_day_sql_when_market_is_closed(self):
        self.market_data.trading_day = False

        result = self.service.day_quotes(["MSFT"])

        self.assertEqual(result.data, [{
            "symbol": "MSFT",
            "session_date": "2026-07-23",
            "open": 100.0,
            "mark": 101.0,
            "quote_time": "2026-07-23T16:00:00+00:00",
            "trade_time": None,
            "source_kind": "completed_session_mark",
        }])
        self.assertEqual(self.market_data.calls.get("provider_get_quotes", 0), 0)
        self.assertEqual(self.market_data.last_prices_kwargs["frequency"], "1day")

    def test_closed_day_quote_keeps_timezone_shifted_sql_bar(self):
        self.market_data.trading_day = False
        original = self.market_data.prices_sql

        def shifted(*args, **kwargs):
            frame = original(*args, **kwargs).iloc[:1].copy()
            frame.index = pd.DatetimeIndex([pd.Timestamp("2026-07-24")])
            return frame

        self.market_data.prices_sql = shifted

        result = self.service.day_quotes(["MSFT"])

        self.assertEqual(result.data[0]["mark"], 101.0)
        self.assertEqual(result.data[0]["session_date"], "2026-07-23")
        self.assertEqual(
            result.data[0]["source_kind"], "completed_session_mark"
        )

    def test_closed_day_quote_uses_bounded_scalar_fallback_when_sql_is_empty(self):
        self.market_data.trading_day = False
        self.context["refdata.api.max_scalar_quote_fallbacks"] = 1
        self.market_data.prices_sql = lambda *args, **kwargs: pd.DataFrame()

        result = self.service.day_quotes(["SWVXX", "MISSING"])

        self.assertEqual([row["symbol"] for row in result.data], ["SWVXX"])
        self.assertEqual(result.data[0]["mark"], 102.0)
        self.assertEqual(
            result.data[0]["quote_time"], "2026-07-23T16:00:00+00:00"
        )
        self.assertEqual(self.market_data.calls["retrieve_current_price"], 1)

    def test_detailed_intraday_bars_use_provider_without_streaming_prices(self):
        result = self.service.intraday_bars("MSFT", market_hours_only=True)

        self.assertEqual(result.count, 2)
        self.assertEqual(self.market_data.calls["provider_get_bars"], 1)
        self.assertEqual(self.market_data.calls.get("prices_sql", 0), 0)
        self.assertEqual(
            self.market_data.refdata_source.last_bars_args[-1], "1min"
        )

    def test_detailed_intraday_bars_use_sql_on_non_trading_days(self):
        self.market_data.trading_day = False

        result = self.service.intraday_bars("MSFT", market_hours_only=False)

        self.assertEqual(result.count, 1)
        self.assertEqual(self.market_data.calls.get("provider_get_bars", 0), 0)
        self.assertEqual(self.market_data.last_prices_kwargs["frequency"], "1min")

    def test_split_cache_is_refreshed_at_api_load_boundary(self):
        self.market_data.cached_stock_splits["MSFT"] = "stale"
        result = self.service.stock_splits("MSFT", date(2020, 1, 1), date(2021, 1, 1))
        self.assertEqual(result.count, 1)
        self.assertEqual(result.data[0]["ratio"], 0.25)

    def test_event_endpoints(self):
        start = date(2026, 1, 1)
        end = date(2026, 12, 31)
        self.assertEqual(self.service.earnings("MSFT", start, end).count, 1)
        self.assertEqual(self.service.dividends("MSFT", start, end).count, 1)
        self.assertEqual(self.service.fomc(start, end).count, 1)
        self.assertEqual(self.service.economic_calendar(start, end).count, 1)

    def test_all_symbol_event_calendars_use_one_repository_query_each(self):
        class EarningsRepository:
            def __init__(self):
                self.calls = []

            def range_rows(self, start_date, end_date):
                self.calls.append((start_date, end_date))
                return [{"symbol": "MSFT", "date": date(2026, 8, 26), "eps_estimate": 3.67}]

        class DividendsRepository:
            def __init__(self):
                self.calls = []

            def range_rows(self, start_date, end_date, *, date_type):
                self.calls.append((start_date, end_date, date_type))
                return [{"symbol": "MSFT", "ex_dividend_date": date(2026, 8, 27), "amount": 0.83}]

        earnings_repository = EarningsRepository()
        dividends_repository = DividendsRepository()
        self.service.configure_event_calendar_repositories(
            earnings_repository, dividends_repository
        )
        start, end = date(2026, 8, 17), date(2026, 9, 7)

        earnings = self.service.earnings_calendar(start, end)
        dividends = self.service.dividend_calendar(start, end)

        self.assertEqual(earnings.count, 1)
        self.assertEqual(earnings.data[0]["date"], "2026-08-26")
        self.assertEqual(dividends.count, 1)
        self.assertEqual(dividends.data[0]["ex_dividend_date"], "2026-08-27")
        self.assertEqual(earnings_repository.calls, [(start, end)])
        self.assertEqual(
            dividends_repository.calls, [(start, end, "ex_dividend_date")]
        )

    def test_rejects_sql_metacharacters_in_symbol(self):
        with self.assertRaises(InvalidRequestError):
            self.service.current_price("MSFT'; DROP TABLE x;--")


if __name__ == "__main__":
    unittest.main()
