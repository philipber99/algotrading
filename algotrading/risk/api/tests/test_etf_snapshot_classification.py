from __future__ import annotations

from datetime import timedelta
import logging

from fastapi.testclient import TestClient

from ..app import create_app
from ..config import RiskApiConfig
from .fakes import FIXED_NOW, FakeSource, account, position


def test_snapshot_and_sector_exposure_use_fund_not_issuer_classification(tmp_path):
    raw_account = account(include_positions=False)
    rows = []
    for symbol in ("CIBR", "SOXL", "HEWJ", "TLH", "TLT", "XLP", "FUNDX", "FUNDY"):
        raw = position(symbol)
        raw["instrument"]["type"] = "ETF"
        rows.append(raw)
    raw_account["securitiesAccount"]["positions"] = rows
    context = {
        "logger": logging.getLogger("etf-snapshot-test"),
        "risk.cache.path": str(tmp_path / "cache.sqlite"),
        "risk.state.path": str(tmp_path / "state.sqlite"),
    }
    app = create_app(
        context=context, source=FakeSource(all_payload=[raw_account]),
        config=RiskApiConfig(),
    )
    with TestClient(app) as client:
        app.state.portfolio_service._now = lambda: FIXED_NOW
        reference = app.state.portfolio_service.reference_client
        quote_at = (FIXED_NOW - timedelta(days=1, hours=2, minutes=30)).isoformat()
        reference.capabilities = lambda: {"available": {}, "supported": {}}
        reference.quotes = lambda symbols: [
            {"symbol": symbol, "mark": 100, "quote_time": quote_at,
             "source": "test", "source_kind": "completed_session_mark"}
            for symbol in symbols
        ]
        reference.security_classifications = lambda symbols: [
            {"symbol": symbol, "security_type": "ETF",
             "sector": "Financial Services", "industry": "Asset Management"}
            for symbol in symbols
        ]
        calls = []

        def lookup(symbol):
            calls.append(symbol)
            if symbol == "FUNDX":
                return {"status": "available", "asset_class": "Equity",
                        "sector_weights": [{"sector": "Technology",
                                            "weight_percent": 100}]}
            return {"status": "unavailable", "reason": "provider_unavailable"}

        reference.etf_exposure = lookup
        reference.option_quotes = lambda symbols: {"data": [], "meta": {}}
        reference.futures_specifications = lambda symbols: {"data": [], "meta": {}}
        snapshot = client.get("/api/v1/portfolio/snapshot", params={"refresh": "true"})
        assert snapshot.status_code == 200, snapshot.text
        positions = {p["exact_symbol"]: p for p in snapshot.json()["data"]["positions"]}
        assert {symbol: positions[symbol]["sector"] for symbol in positions} == {
            "CIBR": "Technology", "SOXL": "Technology",
            "HEWJ": "International Equity", "TLH": "U.S. Treasuries",
            "TLT": "U.S. Treasuries", "XLP": "Consumer Defensive",
            "FUNDX": "Technology",
            "FUNDY": "Unclassified",
        }
        assert "etf_exposure_unclassified:provider_unavailable" in (
            positions["FUNDY"]["data_quality"]["warnings"]
        )
        assert sorted(calls) == ["FUNDX", "FUNDY"]
        sectors = client.get("/api/v1/portfolio/exposures")
        assert sectors.status_code == 200, sectors.text
        assert {row["sector"] for row in sectors.json()["data"]["sectors"]} == {
            "Technology", "International Equity", "U.S. Treasuries",
            "Consumer Defensive", "Unclassified",
        }
