from __future__ import annotations

import logging
import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from algotrading.refdata.api.app import create_app
from algotrading.refdata.api.service import RefDataService
from algotrading.refdata.api.tests.fakes import FakeMarketData
from algotrading.refdata.api.tests.test_stock_analysis import FakeStockService
from algotrading.refdata.stock_analysis.economic_calendar import EconomicReleaseReadService


NOW = datetime(2026, 8, 8, 16, tzinfo=timezone.utc)


class Updater:
    def __init__(self):
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    async def stop(self):
        self.stops += 1


class Repository:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls = []
        self.error = None

    def _matching(self, start, end, kwargs):
        result = [row for row in self.rows if start <= row["timestamp"] < end]
        names = kwargs.get("internal_names", ())
        impacts = kwargs.get("impacts", ())
        status = kwargs.get("actual_status", "all")
        if names:
            result = [row for row in result if row["internal_name"] in names]
        if impacts:
            result = [row for row in result if row["impact"] in impacts]
        if status == "available":
            result = [row for row in result if row["actual"] is not None]
        if status == "pending":
            result = [row for row in result if row["actual"] is None]
        reverse = kwargs.get("order") == "desc"
        return sorted(result, key=lambda row: row["timestamp"], reverse=reverse)

    def count_range(self, start, end, **kwargs):
        if self.error:
            raise self.error
        self.calls.append("count")
        return len(self._matching(start, end, kwargs))

    def query_range(self, start, end, **kwargs):
        if self.error:
            raise self.error
        self.calls.append("query")
        result = self._matching(start, end, kwargs)
        offset = kwargs["offset"]
        return result[offset:offset + kwargs["limit"]]


def release(hour, report, actual=None, impact="High", internal="manufacturing_pmi"):
    return {
        "timestamp": datetime(2026, 8, 8, hour),
        "internal_name": internal,
        "report": report,
        "period": "Jul",
        "previous": 1.0,
        "estimate": 2.0,
        "actual": actual,
        "impact": impact,
    }


class EconomicReleaseRouteTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "logger": logging.getLogger("economic-route-tests"),
            "local_timezone": "UTC",
            "refdata.api.timezone": "America/New_York",
            "ExchangeTZ": "America/New_York",
            "LocalTZ": "America/New_York",
            "refdata.api.cache.enabled": False,
            "refdata.api.calendar_refresh_seconds": 999999,
            "refdata.earnings_calendar.auto_update.enabled": False,
        }
        self.repository = Repository([
            release(9, "Released Zero", 0.0),
            release(11, "Due"),
            release(13, "Scheduled", impact="Low", internal=None),
        ])
        read_service = EconomicReleaseReadService(
            self.context, self.repository, now_fn=lambda: NOW
        )
        legacy = RefDataService(
            self.context, market_data_module=FakeMarketData(), now_fn=lambda: NOW
        )
        self.updater = Updater()
        self.earnings_updater = Updater()
        self.client_context = TestClient(create_app(
            context=self.context,
            service=legacy,
            stock_service=FakeStockService(),
            economic_release_service=read_service,
            economic_updater=self.updater,
            earnings_updater=self.earnings_updater,
        ))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_today_typed_status_offset_and_no_store(self):
        response = self.client.get("/api/v1/economic-releases?view=today")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual([item["status"] for item in payload["items"]], [
            "released", "due_pending", "scheduled"
        ])
        self.assertTrue(payload["items"][0]["timestamp"].endswith("-04:00"))
        self.assertEqual(payload["meta"]["source"], "sql")
        self.assertEqual(payload["meta"]["total"], 3)
        self.assertIsNone(payload["items"][2]["internal_name"])
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Cache"], "BYPASS")

    def test_filters_pagination_and_empty_are_200(self):
        response = self.client.get(
            "/api/v1/economic-releases",
            params=[
                ("view", "range"), ("start", "2026-08-08"),
                ("end", "2026-08-08"), ("impact", "High"),
                ("actual_status", "pending"), ("limit", "1"),
                ("offset", "0"),
            ],
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["items"][0]["report"], "Due")
        response = self.client.get(
            "/api/v1/economic-releases",
            params=[
                ("view", "range"), ("start", "2026-08-08"),
                ("end", "2026-08-08"),
                ("internal_name", "manufacturing_pmi"),
                ("actual_status", "available"),
            ],
        )
        self.assertEqual(response.json()["items"][0]["actual"], 0.0)
        response = self.client.get(
            "/api/v1/economic-releases?view=range&start=2027-01-01&end=2027-01-02"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"], [])

    def test_invalid_inputs_use_structured_422(self):
        for query in (
            "view=range&start=2026-08-01",
            "view=range&start=bad&end=2026-08-01",
            "timezone=Not/A_Zone",
            "limit=1001",
            "offset=-1",
            "view=wrong",
        ):
            with self.subTest(query=query):
                response = self.client.get("/api/v1/economic-releases?" + query)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIn(response.json()["error"]["code"], {"INVALID_REQUEST", "VALIDATION_ERROR"})

    def test_sql_error_is_sanitized_503(self):
        self.repository.error = RuntimeError("SELECT password FROM secrets")
        response = self.client.get("/api/v1/economic-releases")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "DATA_UNAVAILABLE")
        self.assertNotIn("password", response.text)

    def test_route_has_no_provider_side_effect_and_old_route_still_works(self):
        response = self.client.get("/api/v1/economic-releases?view=today")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.repository.calls, ["count", "query"])
        old = self.client.get(
            "/api/v1/calendars/economic?start_date=2026-08-08&end_date=2026-08-08"
        )
        self.assertEqual(old.status_code, 200)
        self.assertIn("data", old.json())
        self.assertEqual(self.updater.starts, 1)
        self.assertEqual(self.earnings_updater.starts, 1)


if __name__ == "__main__":
    unittest.main()
