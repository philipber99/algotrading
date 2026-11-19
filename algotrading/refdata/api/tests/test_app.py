from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from fastapi.testclient import TestClient

    from api.app import create_app
    from api.service import RefDataService
    from api.tests.fakes import FIXED_NOW, FakeMarketData

    HTTP_TESTS_AVAILABLE = True
except ImportError:
    HTTP_TESTS_AVAILABLE = False


@unittest.skipUnless(
    HTTP_TESTS_AVAILABLE,
    "FastAPI/httpx test dependencies are not installed",
)
class AppTests(unittest.TestCase):
    def setUp(self):
        context = {
            "logger": logging.getLogger("test-refdata-api"),
            "local_timezone": "UTC",
            "refdata.api.timezone": "UTC",
            "refdata.api.cache.enabled": True,
            "refdata.api.calendar_refresh_seconds": 999999,
            "refdata.economic_calendar.auto_update.enabled": False,
            "refdata.earnings_calendar.auto_update.enabled": False,
        }
        self.market_data = FakeMarketData()
        service = RefDataService(
            context,
            market_data_module=self.market_data,
            now_fn=lambda: FIXED_NOW,
        )
        self.client_context = TestClient(
            create_app(context=context, service=service), raise_server_exceptions=False
        )
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_status_and_headers(self):
        response = self.client.get("/api/v1/market/status")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["data"]["is_trading_hours"])
        self.assertIn("X-Request-ID", response.headers)
        self.assertEqual(response.headers["X-Cache"], "BYPASS")

    def test_bars_contract(self):
        response = self.client.get(
            "/api/v1/symbols/MSFT/bars",
            params={
                "start": "2026-07-23",
                "end": "2026-07-23",
                "interval": "1d",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["meta"]["count"], 2)
        self.assertIn("timestamp", payload["data"][0])
        self.assertEqual(self.market_data.last_prices_range[1].hour, 23)
        self.assertEqual(self.market_data.last_prices_range[1].minute, 59)

    def test_explicit_futures_current_price_contract_uses_provider_symbol(self):
        response = self.client.get(
            "/api/v1/futures/ESU26/current-price",
            params={"view": "price"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["data"], {"symbol": "ESU26", "price": 101.1}
        )
        self.assertEqual(
            self.market_data.refdata_source.last_quote_symbols, ["/ESU26"]
        )

    def test_bulk_bars_contract_uses_one_repository_call(self):
        class Repository:
            def __init__(self):
                self.calls = 0

            def daily_bars(self, symbols, *, start, end):
                self.calls += 1
                return [
                    {
                        "symbol": symbol,
                        "timestamp": "2026-07-23",
                        "open": 100,
                        "high": 102,
                        "low": 99,
                        "close": 101,
                        "volume": 1000,
                    }
                    for symbol in symbols
                ]

        repository = Repository()
        self.client.app.state.refdata_service.configure_daily_bars_repository(
            repository
        )
        response = self.client.get(
            "/api/v1/bars",
            params={
                "symbols": "MSFT,AAPL",
                "start": "2026-07-01",
                "end": "2026-07-24",
                "remote-query": "true",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(repository.calls, 1)
        payload = response.json()
        self.assertEqual([row["symbol"] for row in payload["data"]], ["MSFT", "AAPL"])
        self.assertEqual(payload["data"][0]["bars"][0]["close"], 101)

    def test_all_symbol_event_calendar_contracts(self):
        class EarningsRepository:
            def range_rows(self, start_date, end_date):
                return [{"symbol": "MSFT", "date": start_date, "eps_estimate": 3.67}]

        class DividendsRepository:
            def range_rows(self, start_date, end_date, *, date_type):
                return [{"symbol": "MSFT", date_type: end_date, "amount": 0.83}]

        self.client.app.state.refdata_service.configure_event_calendar_repositories(
            EarningsRepository(), DividendsRepository()
        )
        params = {"start_date": "2026-08-17", "end_date": "2026-09-07"}

        earnings = self.client.get("/api/v1/calendars/earnings", params=params)
        dividends = self.client.get("/api/v1/calendars/dividends", params=params)

        self.assertEqual(earnings.status_code, 200, earnings.text)
        self.assertEqual(dividends.status_code, 200, dividends.text)
        self.assertEqual(earnings.json()["data"][0]["symbol"], "MSFT")
        self.assertEqual(dividends.json()["data"][0]["amount"], 0.83)

    def test_invalid_symbol(self):
        response = self.client.get("/api/v1/symbols/BAD%27SYMBOL/current-price")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")

    def test_validation_error_does_not_echo_submitted_values(self):
        marker = "SYNTHETIC_PRIVATE_INPUT"
        response = self.client.get(
            "/api/v1/symbols/MSFT/bars",
            params={"start": "2026-07-23", "end": "2026-07-23", "interval": marker},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["message"], "Request validation failed")
        self.assertNotIn(marker, response.text)

    def test_provider_errors_do_not_expose_credentials_in_responses_or_logs(self):
        marker = "SYNTHETIC_PROVIDER_CREDENTIAL"
        for error, status in ((ValueError, 422), (RuntimeError, 503)):
            with self.subTest(error=error.__name__):
                self.market_data.prices_sql = Mock(
                    side_effect=error(f"https://example.invalid/?api_key={marker}")
                )
                with self.assertLogs("test-refdata-api", level="WARNING") as logs:
                    response = self.client.get(
                        "/api/v1/symbols/MSFT/bars",
                        params={"start": "2026-07-23", "end": "2026-07-23"},
                    )
                self.assertEqual(response.status_code, status)
                self.assertNotIn(marker, response.text)
                self.assertNotIn(marker, "\n".join(logs.output))
                self.assertIn(error.__name__, "\n".join(logs.output))

    def test_unhandled_exception_logs_only_its_type(self):
        marker = "SYNTHETIC_INTERNAL_CREDENTIAL"

        def fail():
            raise RuntimeError(marker)

        self.client.app.add_api_route("/_test/security-failure", fail)
        with self.assertLogs("test-refdata-api", level="ERROR") as logs:
            response = self.client.get("/_test/security-failure")

        self.assertEqual(response.status_code, 500)
        self.assertNotIn(marker, response.text)
        self.assertNotIn(marker, "\n".join(logs.output))
        self.assertIn("RuntimeError", "\n".join(logs.output))

    def test_day_quotes_batch_contract(self):
        response = self.client.get(
            "/api/v1/quotes",
            params=[("symbols", "MSFT,AAPL"), ("symbols", "NVDA")],
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["meta"]["count"], 3)
        self.assertEqual(payload["data"][0]["symbol"], "MSFT")
        self.assertEqual(
            set(payload["data"][0]),
            {
                "symbol", "session_date", "open", "bid", "ask", "mark",
                "quote_time", "trade_time", "source_kind",
            },
        )

    def test_detailed_intraday_bars_contract(self):
        response = self.client.get(
            "/api/v1/symbols/MSFT/intraday-bars",
            params={"market_hours_only": True},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["meta"]["count"], 2)
        self.assertIn("timestamp", payload["data"][0])


if __name__ == "__main__":
    unittest.main()
