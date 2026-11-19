from __future__ import annotations

import logging
import threading
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal

try:
    import pandas as pd
    from fastapi.testclient import TestClient

    from algotrading.refdata.api.app import create_app
    from algotrading.refdata.api.service import RefDataService
    from algotrading.refdata.api.tests.fakes import FIXED_NOW, FakeMarketData
    from algotrading.refdata.stock_analysis.service import StockServiceResult

    HTTP_TESTS_AVAILABLE = True
except ImportError:
    HTTP_TESTS_AVAILABLE = False
    FIXED_NOW = datetime(2026, 7, 24, 10, 0, tzinfo=timezone.utc)

    class FakeMarketData:
        pass


def result(data, *, cache="db", provider=None):
    return StockServiceResult(
        data,
        len(data) if isinstance(data, list) else 1,
        FIXED_NOW,
        cache=cache,
        provider=provider,
    )


class FakeStockService:
    timezone = "UTC"

    def __init__(self):
        self.universe_requests = []

    def startup(self):
        pass

    def shutdown(self):
        pass

    def now(self):
        return FIXED_NOW

    def universe_symbols(self, **kwargs):
        self.universe_requests.append(kwargs)
        return result([{
            "security_id": 1, "symbol": "AAPL", "exchange": "NASDAQ",
            "security_type": "common_stock", "active": True,
            "company_id": 2, "company_name": "Apple", "classification_id": 3,
            "taxonomy": "FMP", "sector": "Technology", "industry": "Hardware",
            "sub_industry": None, "latest_daily_bar_date": date(2026, 7, 24),
            "daily_bar_fresh": True,
        }])

    def indexes(self):
        return result([
            {"index_id": "SP100", "name": "S&P 100", "benchmark_security_id": 8, "benchmark_symbol": "OEF", "source_provider": "iShares OEF"},
            {"index_id": "NASDAQ100", "name": "NASDAQ-100", "benchmark_security_id": 7, "benchmark_symbol": "QQQ", "source_provider": "FMP"},
            {"index_id": "SP500", "name": "S&P 500", "benchmark_security_id": 9, "benchmark_symbol": "SPY", "source_provider": "FMP"},
        ])

    def index_constituents(self, *args, **kwargs):
        index_id = args[0]
        provider = "iShares OEF" if index_id == "SP100" else "FMP"
        return result([{"index_id": index_id, "security_id": 1, "symbol": "AAPL", "effective_from": date(2020, 1, 1), "source_provider": provider}])

    def classifications(self, **kwargs):
        return result([{"classification_id": 3, "taxonomy": "FMP", "sector": "Technology", "source_provider": "FMP"}])

    def classification_constituents(self, *args, **kwargs):
        return result([{"security_id": 1, "symbol": "AAPL", "company_name": "Apple"}])

    def etf_exposure(self, symbol):
        return result({
            "symbol": symbol, "status": "available", "asset_class": "Equity",
            "sector_weights": [{"sector": "Technology", "weight_percent": 100}],
        }, cache="provider", provider="FMP")

    def profile(self, symbol):
        return result({"security_id": 1, "symbol": symbol, "company_id": 2, "legal_name": "Apple Inc.", "display_name": "Apple", "index_memberships": [], "market_cap": None})

    def fundamentals(self, *args, **kwargs):
        return result([{
            "financial_id": 1, "filing_id": 2, "period_type": "annual",
            "fiscal_year": 2025, "period_end": date(2025, 9, 30), "currency": "USD",
            "revenue": Decimal("100"), "net_income": Decimal("25"),
            "profit_margin": Decimal("0.25"), "accession": "x", "form_type": "10-K",
            "filed_at": date(2025, 10, 31), "filing_url": "https://sec.test/x",
            "source_provider": "SEC",
        }])

    def estimates(self, symbol):
        return result([{
            "estimate_id": 1, "period_type": "annual", "target_fiscal_year": 2027,
            "target_period_end": date(2027, 9, 30), "currency": "USD",
            "eps_mean": Decimal("8"), "eps_analyst_count": 20,
            "source_provider": "FMP", "provider_as_of": FIXED_NOW,
        }])

    def guidance(self, *args, **kwargs):
        return result([{
            "guidance_id": 1, "issued_at": FIXED_NOW,
            "target_period_end": date(2026, 9, 30), "period_type": "annual",
            "metric": "revenue", "value_low": Decimal("100"),
            "value_high": Decimal("110"), "unit": "USD", "basis": "GAAP",
            "status": "active", "source_url": "https://sec.test/g",
            "source_provider": "SEC",
        }])

    def overview(self, symbol, **kwargs):
        profile = self.profile(symbol).data
        return result({
            "profile": profile, "index_memberships": [], "market_cap": None,
            "annual_financials": self.fundamentals().data,
            "estimates": self.estimates(symbol).data,
            "guidance": self.guidance().data, "events": {},
            "current_forward_pe": {"status": "available", "value": Decimal("20")},
        })

    def news(self, *args, **kwargs):
        return result([{
            "provider_article_id": "n1", "title": "Headline",
            "url": "https://news.test/1", "publisher": "Wire",
            "published_at": FIXED_NOW, "symbols": ["AAPL"],
            "source_provider": "FMP",
        }], cache="bypass", provider="FMP")

    def data_status(self):
        return result([{"category": "company_profile", "provider": "FMP", "result_count": 1, "due_count": 0, "error_count": 0}])


class QuoteUniverseStockService(FakeStockService):
    def __init__(self):
        self.index_requests = []

    def index_constituents(self, *args, **kwargs):
        index_id = args[0]
        self.index_requests.append(index_id)
        symbol = "AAPL" if index_id == "SP100" else "MSFT"
        return result([{
            "index_id": index_id,
            "security_id": len(self.index_requests),
            "symbol": symbol,
            "effective_from": date(2020, 1, 1),
            "source_provider": "test",
        }])


class EmptyIntradayMarketData(FakeMarketData):
    coverage = None

    def prices_sql(self, context, symbol, start, end, **kwargs):
        frame = pd.DataFrame()
        if self.coverage is not None:
            frame.attrs["coverage"] = self.coverage
        return frame


@unittest.skipUnless(HTTP_TESTS_AVAILABLE, "FastAPI/httpx test dependencies are not installed")
class StockAnalysisRouteTests(unittest.TestCase):
    def setUp(self):
        context = {
            "logger": logging.getLogger("stock-route-tests"),
            "local_timezone": "UTC",
            "refdata.api.timezone": "UTC",
            "refdata.api.cache.enabled": False,
            "refdata.api.calendar_refresh_seconds": 999999,
            "refdata.economic_calendar.auto_update.enabled": False,
            "refdata.earnings_calendar.auto_update.enabled": False,
        }
        legacy = RefDataService(
            context, market_data_module=FakeMarketData(), now_fn=lambda: FIXED_NOW
        )
        self.stock_service = FakeStockService()
        self.context = TestClient(
            create_app(context=context, service=legacy, stock_service=self.stock_service)
        )
        self.client = self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_every_stock_analysis_route_and_headers(self):
        paths = (
            "/api/v1/universe/symbols",
            "/api/v1/indexes",
            "/api/v1/indexes/SP500/constituents",
            "/api/v1/classifications",
            "/api/v1/classifications/3/constituents",
            "/api/v1/symbols/AAPL/profile",
            "/api/v1/etfs/CIBR/exposure",
            "/api/v1/symbols/AAPL/fundamentals",
            "/api/v1/symbols/AAPL/estimates",
            "/api/v1/symbols/AAPL/guidance",
            "/api/v1/symbols/AAPL/overview",
            "/api/v1/data-status",
        )
        for path in paths:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn("meta", response.json())
                self.assertIn("X-Data-As-Of", response.headers)

    def test_news_is_always_bypass_and_no_store(self):
        response = self.client.get("/api/v1/symbols/AAPL/news")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["data"][0]["title"], "Headline")
        self.assertEqual(response.json()["data"][0]["symbols"], ["AAPL"])
        self.assertEqual(response.json()["data"][0]["source_provider"], "FMP")
        self.assertEqual(response.headers["X-Cache"], "BYPASS")
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_all_three_supported_index_contracts(self):
        listed = self.client.get("/api/v1/indexes")
        self.assertEqual(
            {item["index_id"] for item in listed.json()["data"]},
            {"SP100", "NASDAQ100", "SP500"},
        )
        for index_id in ("SP100", "NASDAQ100", "SP500"):
            response = self.client.get(f"/api/v1/indexes/{index_id}/constituents")
            self.assertEqual(response.status_code, 200, response.text)
            if index_id == "SP100":
                self.assertEqual(
                    response.json()["data"][0]["source_provider"], "iShares OEF"
                )
        universe = self.client.get("/api/v1/universe/symbols?index_id=SP100")
        self.assertEqual(universe.status_code, 200, universe.text)

    def test_pagination_validation(self):
        response = self.client.get("/api/v1/universe/symbols?limit=5001")
        self.assertEqual(response.status_code, 422)

    def test_universe_symbol_filter_is_accepted(self):
        response = self.client.get(
            "/api/v1/universe/symbols",
            params=[("symbols", "TSM"), ("symbols", "WMT")],
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self.stock_service.universe_requests[-1]["symbols"], ["TSM", "WMT"]
        )


@unittest.skipUnless(HTTP_TESTS_AVAILABLE, "FastAPI/httpx test dependencies are not installed")
class QuotePrecacheLifecycleTests(unittest.TestCase):
    def test_startup_builds_the_configured_index_union_and_supplements(self):
        context = {
            "logger": logging.getLogger("quote-precache-lifecycle-tests"),
            "local_timezone": "UTC",
            "refdata.api.timezone": "UTC",
            "refdata.api.cache.enabled": False,
            "refdata.api.calendar_refresh_seconds": 999999,
            "refdata.api.quote_precache.enabled": True,
            "refdata.api.quote_precache.poll_interval_seconds": 60,
            "refdata.economic_calendar.auto_update.enabled": False,
            "refdata.earnings_calendar.auto_update.enabled": False,
        }
        market_data = FakeMarketData()
        stock_service = QuoteUniverseStockService()
        refreshed = threading.Event()
        observed_symbols = []
        original = market_data.refdata_source.get_quotes

        def observed_get_quotes(symbols):
            observed_symbols.extend(symbols)
            try:
                return original(symbols)
            finally:
                refreshed.set()

        market_data.refdata_source.get_quotes = observed_get_quotes
        service = RefDataService(
            context,
            market_data_module=market_data,
            now_fn=lambda: FIXED_NOW,
        )

        with TestClient(create_app(
            context=context,
            service=service,
            stock_service=stock_service,
        )):
            self.assertTrue(refreshed.wait(1))
            self.assertEqual(
                stock_service.index_requests,
                ["SP100", "SP500", "NASDAQ100"],
            )
            self.assertEqual(
                set(observed_symbols),
                {"AAPL", "MSFT", "SPY", "QQQ", "VIX"},
            )


@unittest.skipUnless(HTTP_TESTS_AVAILABLE, "FastAPI/httpx test dependencies are not installed")
class IntradayAvailabilityTests(unittest.TestCase):
    def client_for(self, market_data):
        context = {
            "logger": logging.getLogger("intraday-tests"),
            "local_timezone": "UTC",
            "refdata.api.timezone": "UTC",
            "refdata.api.cache.enabled": False,
            "refdata.api.calendar_refresh_seconds": 999999,
            "refdata.economic_calendar.auto_update.enabled": False,
            "refdata.earnings_calendar.auto_update.enabled": False,
        }
        service = RefDataService(
            context, market_data_module=market_data, now_fn=lambda: FIXED_NOW
        )
        return TestClient(
            create_app(context=context, service=service, stock_service=FakeStockService())
        )

    def test_empty_minute_bars_are_typed_but_daily_empty_is_unchanged(self):
        with self.client_for(EmptyIntradayMarketData()) as client:
            params = {"start": "2026-07-23", "end": "2026-07-23", "interval": "1m"}
            response = client.get("/api/v1/symbols/EMPTY/bars", params=params)
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["error"]["code"], "INTRADAY_DATA_UNAVAILABLE")
            params["interval"] = "1d"
            response = client.get("/api/v1/symbols/EMPTY/bars", params=params)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["meta"]["count"], 0)

    def test_explicit_incomplete_coverage_is_typed(self):
        market = EmptyIntradayMarketData()
        market.coverage = {"complete": False, "available_start": "2026-07-23T10:00:00"}
        with self.client_for(market) as client:
            response = client.get(
                "/api/v1/symbols/PARTIAL/bars",
                params={"start": "2026-07-23", "end": "2026-07-23", "interval": "1m"},
            )
            self.assertEqual(response.status_code, 503)
            payload = response.json()["error"]
            self.assertEqual(payload["code"], "INTRADAY_DATA_INCOMPLETE")
            self.assertIn("available_start", payload["details"])



if __name__ == "__main__":
    unittest.main()
