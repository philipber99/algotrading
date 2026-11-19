from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import logging
from pathlib import Path
import re

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from algotrading.risk.market_inputs.client import ReferenceDataClientError
from algotrading.risk.pnl.models import (
    PnlActiveLeg,
    PnlAnalyzerRequest,
    PnlAnalyzerResponse,
    PnlAttributionRow,
    PnlChronologyEvent,
    PnlDailyPoint,
    PnlFactorAttribution,
    PnlMarketHistoryResponse,
    PnlReplayPoint,
    PnlScenarioOutcome,
    PnlSummary,
    PnlTrade,
    PnlTradeQueryRequest,
    PnlTradeQueryResponse,
    PnlTransaction,
)
from algotrading.risk.pnl.reconstruction import normalize_execution_legs, reconstruct_lifecycles
from algotrading.risk.transactions.models import (
    TransactionCoverage,
    TransactionFeeModel,
    TransactionItemModel,
    TransactionModel,
)

from ..app import create_app
from ..config import RiskApiConfig
from .fakes import FakeSource


UTC = timezone.utc
START = datetime(2026, 8, 1, tzinfo=UTC)
END = datetime(2026, 8, 4, tzinfo=UTC)
ACCOUNT = "FAKE-0001"
CALL_100 = "XYZ260918C00100000"
CALL_105 = "XYZ260918C00105000"


def transaction(row_id, at, legs, *, fee=0, fee_components=()):
    return TransactionModel(
        id=row_id,
        broker_transaction_id=f"broker-{row_id}",
        transaction_at=at,
        transaction_type="TRADE",
        fees=Decimal(str(fee)),
        ingested_at=at,
        fee_components=[
            TransactionFeeModel(
                fee_index=index,
                fee_type=fee_type,
                amount=Decimal(str(amount)),
            )
            for index, (fee_type, amount) in enumerate(fee_components)
        ],
        items=[
            TransactionItemModel(
                item_index=index,
                symbol=symbol,
                asset_type=asset_type,
                instruction=instruction,
                position_effect=effect,
                quantity=Decimal(str(quantity)),
                price=Decimal(str(price)),
            )
            for index, (symbol, asset_type, instruction, effect, quantity, price) in enumerate(legs)
        ],
    )


class Repository:
    def __init__(self, rows):
        self.rows = rows
        self.reads = 0

    def query_lifecycle_history(self, account_id, **kwargs):
        assert account_id == ACCOUNT
        self.reads += 1
        coverage = TransactionCoverage(
            requested_start_at=kwargs["requested_start_at"],
            requested_end_at=kwargs["requested_end_at"],
            sync_through_by_type={"TRADE": kwargs["requested_end_at"]},
            complete=True,
            missing_intervals=[],
        )
        return list(self.rows), coverage, END


class Reference:
    def bars(self, symbols, *, start, end):
        return [
            {
                "symbol": symbol,
                "bars": [
                    {"timestamp": "2026-07-31", "open": 98, "high": 100, "low": 97, "close": 99, "volume": 900},
                    {"timestamp": "2026-08-01", "open": 99, "high": 102, "low": 98, "close": 101, "volume": 1000},
                    {"timestamp": "2026-08-02", "open": 101, "high": 106, "low": 100, "close": 105, "volume": 1100},
                    {"timestamp": "2026-08-03", "open": 105, "high": 111, "low": 104, "close": 110, "volume": 1200},
                ],
            }
            for symbol in symbols
        ]

    def security_classifications(self, symbols):
        return [{"symbol": item, "sector": "Technology"} for item in symbols]

    def option_market_history(self, payload):
        raise ReferenceDataClientError(
            "Reference Data request failed status=501 code=OPTION_MARK_HISTORY_NOT_IMPLEMENTED"
        )


def app_client(tmp_path, rows):
    source = FakeSource()
    context = {
        "logger": logging.getLogger("pnl-routes"),
        "risk.cache.path": str(tmp_path / "cache.sqlite"),
        "risk.state.path": str(tmp_path / "state.sqlite"),
        "risk.refdata.base_url": "http://127.0.0.1:1",
    }
    app = create_app(
        context=context,
        source=source,
        config=RiskApiConfig(account_cache_ttl_seconds=60),
    )
    return app, source, Repository(rows)


def query_payload(snapshot_id):
    return {
        "account_ids": [ACCOUNT],
        "portfolio_snapshot_id": snapshot_id,
        "start_at": START.isoformat(),
        "end_at": END.isoformat(),
        "include_closing_after_end": True,
        "lookahead_end_at": (END + timedelta(days=2)).isoformat(),
        "sectors": [],
        "instrument_types": [],
    }


def test_pnl_routes_are_in_openapi_snapshot_bound_strict_and_do_not_call_broker(tmp_path):
    rows = [
        transaction(1, START + timedelta(hours=1), [("XYZ", "EQUITY", "BUY", "OPENING", 10, 100)], fee=-99, fee_components=(("COMMISSION", 1), ("SEC_FEE", 0.12))),
        transaction(2, START + timedelta(days=1), [("XYZ", "EQUITY", "SELL", "CLOSING", 10, 110)], fee=-99, fee_components=(("COMMISSION", 1), ("SEC_FEE", 0.12))),
    ]
    app, source, repository = app_client(tmp_path, rows)
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
        loaded = client.get("/api/v1/portfolio/snapshot", params={"refresh": "true"}).json()
        app.state.transaction_repository = repository
        app.state.portfolio_service.reference_client = Reference()
        before = source.all_calls
        response = client.post(
            "/api/v1/portfolio/pnl/trades/query",
            json=query_payload(loaded["portfolio_snapshot_id"]),
        )
        stale = client.post(
            "/api/v1/portfolio/pnl/trades/query",
            json=query_payload("stale-snapshot"),
        )
        invalid = query_payload(loaded["portfolio_snapshot_id"])
        invalid["end_at"] = invalid["start_at"]
        bad_interval = client.post(
            "/api/v1/portfolio/pnl/trades/query", json=invalid
        )
    assert {
        "/api/v1/portfolio/pnl/market-history/query",
        "/api/v1/portfolio/pnl/trades/query",
        "/api/v1/portfolio/pnl/trades/analyze",
    } <= set(schema["paths"])
    assert response.status_code == 200, response.text
    validated = PnlTradeQueryResponse.model_validate(response.json())
    assert validated.data.summary.period_pnl_usd == pytest.approx(97.76)
    assert validated.data.summary.fee_components_usd == {
        "COMMISSION": -2.0,
        "SEC_FEE": -0.24,
    }
    with pytest.raises(ValidationError):
        PnlTradeQueryResponse.model_validate({**response.json(), "unexpected": True})
    assert repository.reads == 1
    assert source.all_calls == before
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "SNAPSHOT_MISMATCH"
    assert bad_interval.status_code == 422
    assert bad_interval.json()["error"]["code"] == "VALIDATION_ERROR"


def test_pnl_market_history_is_daily_sql_first_and_bounded(tmp_path):
    app, _, _ = app_client(tmp_path, [])

    class RoutingReference(Reference):
        def __init__(self):
            self.requests = []

        def bars(self, symbols, *, start, end, remote_query=False):
            self.requests.append((tuple(symbols), start, end, remote_query))
            return super().bars(symbols, start=start, end=end)

    reference = RoutingReference()
    with TestClient(app) as client:
        app.state.portfolio_service.reference_client = reference
        response = client.post(
            "/api/v1/portfolio/pnl/market-history/query",
            json={
                "symbols": ["XYZ"],
                "start_at": "2026-08-01T13:30:00Z",
                "end_at": "2026-08-04T00:00:00Z",
            },
        )
        invalid = client.post(
            "/api/v1/portfolio/pnl/market-history/query",
            json={
                "symbols": ["XYZ"],
                "start_at": "2026-08-01T00:00:00Z",
                "end_at": "2027-09-01T00:00:00Z",
            },
        )

    assert response.status_code == 200, response.text
    validated = PnlMarketHistoryResponse.model_validate(response.json())
    assert [point.close for point in validated.data[0].points] == [101, 105, 110]
    assert [point.open for point in validated.data[0].points] == [99, 101, 105]
    assert [point.high for point in validated.data[0].points] == [102, 106, 111]
    assert [point.low for point in validated.data[0].points] == [98, 100, 104]
    assert [point.volume for point in validated.data[0].points] == [1000, 1100, 1200]
    assert reference.requests == [
        (("XYZ",), START.date(), (END - timedelta(microseconds=1)).date(), False)
    ]
    assert invalid.status_code == 422


def test_analyzer_returns_option_chronology_when_refdata_history_is_501(tmp_path):
    rows = [
        transaction(10, START, [(CALL_100, "OPTION", "BUY", "OPENING", 1, 5), (CALL_105, "OPTION", "SELL", "OPENING", 1, 3)], fee=-1),
        transaction(11, START + timedelta(days=2), [(CALL_100, "OPTION", "SELL", "CLOSING", 1, 7), (CALL_105, "OPTION", "BUY", "CLOSING", 1, 4)], fee=-1),
    ]
    executions, _ = normalize_execution_legs(ACCOUNT, rows)
    trade_id = reconstruct_lifecycles(executions)[0].trade_id
    app, source, repository = app_client(tmp_path, rows)
    with TestClient(app) as client:
        loaded = client.get("/api/v1/portfolio/snapshot", params={"refresh": "true"}).json()
        app.state.transaction_repository = repository
        app.state.portfolio_service.reference_client = Reference()
        before = source.all_calls
        response = client.post(
            "/api/v1/portfolio/pnl/trades/analyze",
            json={
                "account_ids": [ACCOUNT],
                "portfolio_snapshot_id": loaded["portfolio_snapshot_id"],
                "trade_ids": [trade_id],
                "transaction_ids": [],
                "decision_at": None,
                "fill_model": "historical_touch_plus_tick",
                "modifications": [],
                "exit_rule": {"kind": "actual", "value": None},
            },
        )
    assert response.status_code == 200, response.text
    validated = PnlAnalyzerResponse.model_validate(response.json())
    assert validated.data_quality.historical_option_marks_available is False
    assert validated.data_quality.attribution_method == "price_history_fallback"
    assert validated.data_quality.surface_coverage_percent == 0.0
    assert validated.data.actual_path == []
    assert len(validated.data.chronology) == 4
    assert sum(item.pnl_usd for item in validated.data.attribution) == pytest.approx(
        validated.data.lifecycle_pnl_usd, abs=0.01
    )
    assert source.all_calls == before


def test_named_server_contracts_match_gui_pnl_api_source():
    gui_contract = Path(__file__).resolve().parents[5] / "risk_gui" / "app" / "pnl-api.ts"
    if not gui_contract.exists():
        pytest.skip("the companion Risk GUI source is not present in this checkout")
    source = gui_contract.read_text(encoding="utf-8")
    contracts = {
        "PnlTradeQueryRequest": PnlTradeQueryRequest,
        "PnlTradeQueryResponse": PnlTradeQueryResponse,
        "PnlTransaction": PnlTransaction,
        "PnlTrade": PnlTrade,
        "PnlSummary": PnlSummary,
        "PnlPoint": PnlDailyPoint,
        "PnlCompositionRow": PnlAttributionRow,
        "PnlAnalyzerRequest": PnlAnalyzerRequest,
        "PnlAnalyzerResponse": PnlAnalyzerResponse,
        "PnlReplayPoint": PnlReplayPoint,
        "PnlChronologyEvent": PnlChronologyEvent,
        "PnlAttribution": PnlFactorAttribution,
        "PnlActiveLeg": PnlActiveLeg,
        "PnlScenarioOutcome": PnlScenarioOutcome,
        "PnlMarketHistoryResponse": PnlMarketHistoryResponse,
    }
    for type_name, model in contracts.items():
        assert _typescript_direct_fields(source, type_name) == set(model.model_fields), type_name


def _typescript_direct_fields(source: str, type_name: str) -> set[str]:
    marker = re.search(rf"export type {re.escape(type_name)}\s*=\s*{{", source)
    assert marker is not None, f"missing GUI type {type_name}"
    depth = 1
    fields: set[str] = set()
    for line in source[marker.end() :].splitlines():
        if depth == 1:
            direct_segment = line.split("{", 1)[0]
            for field in re.finditer(
                r"(?:^|;)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:",
                direct_segment,
            ):
                fields.add(field.group(1))
        depth += line.count("{") - line.count("}")
        if depth == 0:
            break
    return fields
