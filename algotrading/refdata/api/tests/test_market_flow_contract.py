from __future__ import annotations

from datetime import date, datetime, timezone
import logging
import unittest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from algotrading.refdata.api.app import create_app
from algotrading.refdata.api.market_flow_models import MarketFlowOverviewData
from algotrading.refdata.api.service import NotFoundError, RefDataService
from algotrading.refdata.api.tests.fakes import FakeMarketData
from algotrading.refdata.api.tests.test_stock_analysis import FakeStockService
from algotrading.refdata.stock_analysis.analytics.service import AnalyticsServiceResult


AS_OF = date(2026, 9, 1)
STAMP = datetime(2026, 9, 1, tzinfo=timezone.utc)


def metric(value=1.0, unit="ratio", status="available", reason=None):
    return {
        "status": status, "value": value if status == "available" else None,
        "unit": unit, "reason": None if status == "available" else reason or "unavailable",
        "observation_count": 1 if status == "available" else 0,
        "data_as_of": AS_OF, "anchor_as_of": date(2026, 8, 31), "warnings": [],
    }


def windows():
    return [{
        "window": window, "sessions": count, "available": window != "6m",
        "reason": None if window != "6m" else "insufficient_sessions",
        "actual_start_date": AS_OF if window != "6m" else None,
        "actual_end_date": AS_OF, "observation_count": count if window != "6m" else 80,
    } for window, count in (("1d", 1), ("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126))]


def coverage():
    return {name: metric(.9) for name in (
        "etf_asset_coverage", "etf_count_coverage", "nav_share_coverage",
        "consecutive_session_coverage", "exposure_classification_coverage",
        "price_coverage", "finra_overlay_coverage", "data_freshness",
        "mandate_lookthrough_agreement", "flow_price_agreement",
    )}


class FakeMarketFlowService:
    timezone = "UTC"

    def __init__(self):
        self.calls = []

    def result(self, data):
        return AnalyticsServiceResult(data=data, as_of=STAMP, cache="miss", warnings=("fixture_warning",), query_count=7)

    def overview(self, **kwargs):
        self.calls.append(("overview", kwargs))
        return self.result({
            "requested_window": kwargs["window"], "actual_start_date": AS_OF,
            "actual_end_date": AS_OF, "window_availability": windows(),
            "source_as_of": {"etf": AS_OF, "exposure": AS_OF, "price": AS_OF, "finra": None},
            "scope": {"tracked_etf_count": 44},
            "state": {"code": "neutral", "label": "Neutral tracked-ETF flow"},
            "headline": {"tracked_equity_etf_net_flow": metric(10, "usd")},
            "instruments": [], "sector_comparison": [], "series": [],
            "methodology": {"neutral_pressure_bps": 5}, "coverage": coverage(),
            "warnings": ["fixture_warning"],
        })

    def categories(self, **kwargs):
        self.calls.append(("categories", kwargs))
        node = {
            "node_id": "root:tracked", "parent_id": None, "level": "root",
            "name": "Tracked Universe", "symbol": None, "has_children": True,
            "sort_key": "00", "view": "all", "capital_allocation": {},
            "market_behavior": {}, "rotation": {},
            "composition_and_contributors": {}, "short_volume_context": {},
            "data_quality": {},
        }
        return self.result({
            "requested_window": kwargs["window"], "actual_start_date": AS_OF,
            "actual_end_date": AS_OF, "view": kwargs["view"],
            "parent_id": kwargs["parent_id"],
            "include_constituents": kwargs["include_constituents"], "nodes": [node],
            "coverage": {}, "source_as_of": {"etf": AS_OF}, "warnings": [],
        })

    def rotations(self, **kwargs):
        self.calls.append(("rotations", kwargs))
        point = {
            "node_id": "economic:sector:technology", "name": "Technology",
            "dimension": kwargs["dimension"], "relative_flow_pressure": metric(2, "bps"),
            "relative_performance": metric(.01, "decimal_return"),
            "acceleration": metric(None, "bps", "unavailable", "requires_20_sessions"),
            "persistence": metric(.8), "breadth": metric(.4),
            "abnormal_volume": metric(1.2, "multiple"),
            "tracked_assets": metric(1e9, "usd"), "confidence": {"level": "moderate"},
            "evidence_score": metric(3, "score"),
        }
        return self.result({
            "requested_window": kwargs["window"], "actual_start_date": AS_OF,
            "actual_end_date": AS_OF, "dimension": kwargs["dimension"],
            "points": [point], "destinations": [], "sources": [],
            "evidence_pairs": [], "coverage": {},
            "source_as_of": {"etf": AS_OF}, "warnings": [],
        })

    def etfs(self, **kwargs):
        self.calls.append(("etfs", kwargs))
        item = {
            "symbol": "SPY", "fund_name": "SPDR S&P 500 ETF Trust",
            "issuer": "State Street", "asset_class": "EQUITY",
            "exposure_type": "BROAD_MARKET", "exposure_name": "S&P 500",
            "latest_reference_date": AS_OF, "latest_exposure_date": AS_OF,
            "earliest_reference_date": date(2026, 1, 2), "available_windows": windows(),
            "has_nav": True, "has_shares": True, "has_net_assets": True,
            "has_market_price": True, "has_exposure": True,
            "has_fixed_income_characteristics": False,
        }
        return self.result({
            "as_of": AS_OF, "actual_start_date": AS_OF, "actual_end_date": AS_OF,
            "asset_class": kwargs["asset_class"],
            "exposure_type": kwargs["exposure_type"], "etfs": [item],
            "source_as_of": {"etf": AS_OF}, "warnings": [],
        })

    def etf_detail(self, symbol, **kwargs):
        self.calls.append(("detail", {"symbol": symbol, **kwargs}))
        if symbol.upper() != "SPY":
            raise NotFoundError("Unknown tracked ETF symbol")
        return self.result({
            "symbol": "SPY", "identity": {"asset_class": "EQUITY"},
            "requested_window": kwargs["window"], "actual_start_date": AS_OF,
            "actual_end_date": AS_OF, "window_availability": windows(),
            "flow_series": [], "primary_market": {}, "secondary_market": {},
            "metrics": {}, "exposures": {"type": "equity"},
            "fixed_income_characteristics": None, "short_volume_context": {},
            "data_quality": {}, "source_as_of": {"etf": AS_OF}, "warnings": [],
        })


class MarketFlowContractTests(unittest.TestCase):
    def setUp(self):
        context = {
            "logger": logging.getLogger("market-flow-contract"),
            "local_timezone": "UTC", "refdata.api.timezone": "UTC",
            "refdata.economic_calendar.auto_update.enabled": False,
            "refdata.earnings_calendar.auto_update.enabled": False,
        }
        service = RefDataService(context, market_data_module=FakeMarketData(), now_fn=lambda: STAMP)
        self.market_flow = FakeMarketFlowService()
        self.client_context = TestClient(create_app(
            context=context, service=service, stock_service=FakeStockService(),
            analytics_service=object(), market_flow_service=self.market_flow,
        ))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_all_five_routes_headers_and_as_of_propagation(self):
        query = {"as_of": "2026-08-31"}
        responses = [
            self.client.get("/api/v1/analytics/market-flows/overview", params={**query, "window": "1w"}),
            self.client.get("/api/v1/analytics/market-flows/categories", params={**query, "window": "1w", "node_id": "root:tracked", "include_flow_series": True, "limit": 20}),
            self.client.get("/api/v1/analytics/market-flows/rotations", params={**query, "window": "1w", "dimension": "sector", "limit": 10}),
            self.client.get("/api/v1/analytics/market-flows/etfs", params={**query, "asset_class": "EQUITY"}),
            self.client.get("/api/v1/analytics/market-flows/etfs/SPY", params={**query, "window": "1w"}),
        ]
        for response in responses:
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["X-Cache"], "MISS")
            self.assertEqual(response.headers["X-Data-As-Of"], STAMP.isoformat())
            self.assertEqual(response.json()["meta"]["sql_query_count"], 7)
        self.assertTrue(all(call[1]["as_of"] == date(2026, 8, 31) for call in self.market_flow.calls))
        category_call = next(call for call in self.market_flow.calls if call[0] == "categories")[1]
        self.assertEqual(category_call["node_id"], "root:tracked")
        self.assertTrue(category_call["include_flow_series"])
        unavailable = responses[2].json()["data"]["points"][0]["acceleration"]
        self.assertEqual((unavailable["status"], unavailable["value"], unavailable["reason"]), ("unavailable", None, "requires_20_sessions"))
        invalid = responses[0].json()["data"]
        invalid["headline"]["tracked_equity_etf_net_flow"]["status"] = "invented"
        with self.assertRaises(ValidationError):
            MarketFlowOverviewData.model_validate(invalid)

    def test_typed_query_validation_and_unknown_etf(self):
        self.assertEqual(self.client.get("/api/v1/analytics/market-flows/overview", params={"window": "1y"}).status_code, 422)
        self.assertEqual(self.client.get("/api/v1/analytics/market-flows/rotations", params={"dimension": "theme"}).status_code, 422)
        self.assertEqual(self.client.get("/api/v1/analytics/market-flows/rotations", params={"limit": 101}).status_code, 422)
        self.assertEqual(self.client.get("/api/v1/analytics/market-flows/categories", params={"limit": 0}).status_code, 422)
        self.assertEqual(self.client.get("/api/v1/analytics/market-flows/etfs", params={"asset_class": "COMMODITY"}).status_code, 422)
        unknown = self.client.get("/api/v1/analytics/market-flows/etfs/NOPE")
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json()["error"]["code"], "NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
