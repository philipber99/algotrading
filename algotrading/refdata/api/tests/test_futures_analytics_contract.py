from __future__ import annotations

from datetime import datetime, timezone
import logging
import unittest

from fastapi.testclient import TestClient

from algotrading.refdata.api.app import create_app
from algotrading.refdata.api.service import RefDataService
from algotrading.refdata.api.tests.fakes import FakeMarketData
from algotrading.refdata.api.tests.test_market_flow_contract import FakeMarketFlowService
from algotrading.refdata.api.tests.test_stock_analysis import FakeStockService
from algotrading.refdata.stock_analysis.futures_analytics import FuturesAnalyticsService
from algotrading.refdata.stock_analysis.tests.test_futures_analytics_service import (
    DEFAULT_PAIRS,
    FuturesFixtureRepository,
)


class FuturesAnalyticsContractTests(unittest.TestCase):
    def build_client(self, *, partial=False):
        context = {
            "logger": logging.getLogger("futures-analytics-contract"),
            "local_timezone": "UTC", "refdata.api.timezone": "UTC",
            "refdata.economic_calendar.auto_update.enabled": False,
            "refdata.earnings_calendar.auto_update.enabled": False,
        }
        refdata = RefDataService(
            context, market_data_module=FakeMarketData(),
            now_fn=lambda: datetime(2026, 9, 15, 12, tzinfo=timezone.utc),
        )
        counts = None
        if partial:
            counts = {product: 0 for product, _ in DEFAULT_PAIRS}
            counts.update({"ES": 3, "NQ": 4, "VX": 30})
        futures = FuturesAnalyticsService(
            context, FuturesFixtureRepository(daily_counts=counts),
            now_fn=lambda: datetime(2026, 9, 15, 12, tzinfo=timezone.utc),
        )
        return TestClient(create_app(
            context=context, service=refdata, stock_service=FakeStockService(),
            analytics_service=object(), market_flow_service=FakeMarketFlowService(),
            futures_analytics_service=futures,
        ))

    def test_all_routes_return_typed_envelopes_headers_and_finite_json(self):
        with self.build_client() as client:
            responses = [
                client.get("/api/v1/analytics/futures/positioning", params={"products": "ES,NQ", "participants": ["ASSET_MANAGER", "LEVERAGED_MONEY"], "change_window": "13w", "history_window": "5y", "as_of": "2026-09-12", "include_series": "true"}),
                client.get("/api/v1/analytics/futures/participation", params={"products": "ES,NQ", "windows": "1d,5d,20d", "as_of": "2026-09-12"}),
                client.get("/api/v1/analytics/futures/term-structure", params={"product": "VX", "months": 6, "compare_sessions": 5, "as_of": "2026-09-12"}),
                client.get("/api/v1/analytics/futures/crowding", params={"history_window": "3y", "change_window": "4w", "limit": 8, "as_of": "2026-09-12"}),
                client.get("/api/v1/analytics/futures/market-health/overview", params={"position_window": "1w", "history_window": "1y", "as_of": "2026-09-12"}),
            ]
            for response in responses:
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn(response.headers["X-Cache"], {"MISS", "HIT"})
                self.assertIn("data", response.json())
                encoded = response.text
                self.assertNotIn("NaN", encoded)
                self.assertNotIn("Infinity", encoded)

    def test_invalid_windows_and_unknown_product_use_stable_errors(self):
        with self.build_client() as client:
            self.assertEqual(client.get("/api/v1/analytics/futures/positioning", params={"change_window": "2w"}).status_code, 422)
            self.assertEqual(client.get("/api/v1/analytics/futures/positioning", params={"history_window": "2y"}).status_code, 422)
            unknown = client.get("/api/v1/analytics/futures/participation", params={"products": "NOPE"})
            self.assertEqual(unknown.status_code, 404)
            self.assertEqual(unknown.json()["error"]["code"], "NOT_FOUND")

    def test_default_requests_and_all_supported_context_windows(self):
        with self.build_client() as client:
            for path in (
                "/api/v1/analytics/futures/positioning",
                "/api/v1/analytics/futures/participation",
                "/api/v1/analytics/futures/term-structure",
                "/api/v1/analytics/futures/crowding",
                "/api/v1/analytics/futures/market-health/overview",
            ):
                self.assertEqual(client.get(path).status_code, 200, path)
            for change_window in ("1w", "4w", "13w"):
                response = client.get(
                    "/api/v1/analytics/futures/positioning",
                    params={"change_window": change_window, "history_window": "1y"},
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    response.json()["data"]["requested_change_window"], change_window
                )
            for history_window in ("1y", "3y", "5y"):
                response = client.get(
                    "/api/v1/analytics/futures/crowding",
                    params={"history_window": history_window},
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    response.json()["data"]["requested_history_window"], history_window
                )

    def test_partial_history_is_http_200_with_explicit_state(self):
        with self.build_client(partial=True) as client:
            response = client.get("/api/v1/analytics/futures/market-health/overview")
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()["data"]
            self.assertTrue(data["state"]["partial"])
            self.assertTrue(data["state"]["code"].startswith("PARTIAL_POSITIONING_"))
            self.assertEqual(data["dimensions"]["participation"]["status"], "insufficient_history")


if __name__ == "__main__":
    unittest.main()
