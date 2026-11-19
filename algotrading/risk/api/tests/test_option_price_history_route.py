from __future__ import annotations

from datetime import datetime, timezone
import logging

from fastapi.testclient import TestClient

from algotrading.risk.option_prices.service import RiskOptionHistoryService

from ..app import create_app
from .fakes import FakeSource


CALL = "XYZ260918C00100000"


class Repository:
    def query_history(self, symbols, *, start_date, end_date):
        return [
            {
                "requested_symbol": CALL,
                "canonical_occ_symbol": CALL,
                "at": "2026-08-19T00:00:00Z",
                "session_date": "2026-08-18",
                "status": "ok",
                "mark": 5.25,
                "last": 5.25,
                "source": "risk_option_price_capture",
                "warnings": [],
            }
        ]


class Reference:
    def __init__(self):
        self.calls = []

    def option_market_history(self, payload):
        self.calls.append(payload)
        return {"data": [], "meta": {"warnings": []}}


def test_risk_gui_option_history_route_is_documented_sql_first_and_broker_free():
    source = FakeSource()
    reference = Reference()
    app = create_app(
        context={"logger": logging.getLogger("option-route")}, source=source
    )
    with TestClient(app) as client:
        app.state.portfolio_service.reference_client = reference
        app.state.option_history_service = RiskOptionHistoryService(
            Repository(),
            reference,
            now_fn=lambda: datetime(2026, 8, 20, tzinfo=timezone.utc),
        )
        schema = client.get("/openapi.json").json()
        response = client.post(
            "/api/v1/risk-gui/options/market-history/query",
            json={
                "symbols": [CALL],
                "start_at": "2026-08-18T00:00:00Z",
                "end_at": "2026-08-19T00:00:00Z",
                "interval": "1d",
                "include_greeks": False,
            },
        )

    assert "/api/v1/risk-gui/options/market-history/query" in schema["paths"]
    assert response.status_code == 200, response.text
    assert response.headers["X-Cache"] == "HIT"
    assert response.json()["data"][0]["mark"] == 5.25
    assert reference.calls == []
    assert source.all_calls == 0
