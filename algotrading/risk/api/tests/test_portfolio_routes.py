from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from ..app import create_app
from ..config import RiskApiConfig
from ..portfolio_service import PortfolioApplicationService
from ..service import RiskService
from .fakes import FakeSource, account, position


def test_portfolio_snapshot_is_explicit_coherent_and_analysis_is_snapshot_bound(tmp_path):
    source = FakeSource()
    context = {"logger": logging.getLogger("portfolio-test"), "risk.cache.path": str(tmp_path/"cache.sqlite"),
               "risk.state.path": str(tmp_path/"state.sqlite"), "risk.refdata.base_url": "http://127.0.0.1:1"}
    app = create_app(context=context, source=source, config=RiskApiConfig(account_cache_ttl_seconds=60))
    with TestClient(app) as client:
        missing = client.get("/api/v1/portfolio/snapshot")
        loaded = client.get("/api/v1/portfolio/snapshot", params={"refresh": "true"})
        again = client.get("/api/v1/portfolio/snapshot", params={"refresh": "true"})
        tail = client.get("/api/v1/portfolio/tail-risk")
        health = client.get("/api/v1/data-health")
    assert missing.status_code == 409
    assert loaded.status_code == 200
    assert source.all_calls == 1
    assert loaded.json()["data"]["kpis"]["margin_loan_balance_usd"] == 11
    assert loaded.json()["data"]["balances"][0]["margin_balance_usd"] == 11
    assert again.json()["portfolio_snapshot_id"] == loaded.json()["portfolio_snapshot_id"]
    assert again.json()["cache"]["state"] == "suppressed"
    assert tail.json()["portfolio_snapshot_id"] == loaded.json()["portfolio_snapshot_id"]
    assert tail.json()["data"]["status"] == "insufficient_data"
    assert health.json()["data"]["forbidden_dependencies"] == {"clickhouse": False, "options_gui_api": False}


def test_openapi_contains_new_contracts_and_transaction_reads_never_call_broker(tmp_path):
    source = FakeSource()
    context = {"logger": logging.getLogger("portfolio-openapi"), "risk.cache.path": str(tmp_path/"cache.sqlite"),
               "risk.state.path": str(tmp_path/"state.sqlite")}
    with TestClient(create_app(context=context, source=source, config=RiskApiConfig())) as client:
        schema = client.get("/openapi.json").json()
        transaction = client.get("/api/v1/accounts/TEST/transactions/status")
    expected = {"/api/v1/portfolio/snapshot", "/api/v1/options/iv-greeks", "/api/v1/hedge-lab/options/chain", "/api/v1/accounts/{account_id}/transactions", "/api/v1/reference/companies/{symbol}"}
    assert expected <= set(schema["paths"])
    assert transaction.status_code == 503
    assert source.all_calls == 0 and source.one_calls == []


def test_position_alert_configuration_round_trips_and_validates_references(tmp_path):
    context = {
        "logger": logging.getLogger("position-alert-configuration"),
        "risk.cache.path": str(tmp_path / "cache.sqlite"),
        "risk.state.path": str(tmp_path / "state.sqlite"),
    }
    payload = {
        "position_alert_conditions": [{
            "name": "Bearish momentum",
            "predicates": [{"duration": "3M", "signal": "RSI", "comparator": "<", "target": 35, "join": "AND"}],
        }],
        "position_alert_rules": [{
            "rule_id": "long-equity-rsi",
            "side": "Long",
            "position_type": "Equity",
            "condition_name": "Bearish momentum",
            "once": True,
        }],
    }
    with TestClient(create_app(context=context, source=FakeSource(), config=RiskApiConfig())) as client:
        saved = client.patch("/api/v1/risk-configuration", json=payload)
        loaded = client.get("/api/v1/risk-configuration")
        invalid = client.patch("/api/v1/risk-configuration", json={
            "position_alert_rules": [{**payload["position_alert_rules"][0], "condition_name": "Missing"}],
        })
    assert saved.status_code == 200, saved.text
    assert loaded.json()["data"]["risk.alerts.position.conditions"] == payload["position_alert_conditions"]
    assert loaded.json()["data"]["risk.alerts.position.rules"] == payload["position_alert_rules"]
    assert invalid.status_code == 422


def test_iv_greeks_route_supplies_missing_rate_from_reference_curve(tmp_path):
    source = FakeSource()
    context = {
        "logger": logging.getLogger("portfolio-tian-rate"),
        "risk.cache.path": str(tmp_path / "cache.sqlite"),
        "risk.state.path": str(tmp_path / "state.sqlite"),
        "risk.rates.configured_flat_continuous_rate": None,
    }
    app = create_app(context=context, source=source, config=RiskApiConfig())
    curve = {
        "curve_id": "test",
        "curve_version": "v1",
        "anchors": [
            {
                "series_id": "DGS1MO",
                "tenor_days": 30,
                "observation_date": "2026-08-14",
                "simple_annual_rate_decimal": 0.04,
                "status": "ok",
            },
            {
                "series_id": "DGS3MO",
                "tenor_days": 90,
                "observation_date": "2026-08-14",
                "simple_annual_rate_decimal": 0.05,
                "status": "ok",
            },
        ],
    }
    with TestClient(app) as client:
        app.state.portfolio_service.reference_client.risk_free_curve = (
            lambda _as_of: {"data": curve}
        )
        response = client.post(
            "/api/v1/options/iv-greeks",
            json={
                "valuation_at": "2026-08-15T20:00:00Z",
                "contracts": [
                    {
                        "request_id": "route-rate",
                        "canonical_occ_symbol": "SPY260918C00500000",
                        "underlying_symbol": "SPY",
                        "put_call": "CALL",
                        "strike": 500,
                        "expiration_at": "2026-09-18T20:00:00Z",
                        "spot": 510,
                        "mark": 15.5,
                    }
                ],
            },
        )

    assert response.status_code == 200, response.text
    item = response.json()["data"][0]
    assert item["status"] in {"ok", "usable_with_warnings"}
    assert item["reason"] != "risk_free_rate_unavailable"
    assert item["risk_free_rate_source"] == "request"


def test_portfolio_snapshot_ignores_invalid_provider_iv(tmp_path):
    symbol = "SPY270917C00500000"
    malformed_symbol = "QQQ270917C00500000"
    source = FakeSource(all_payload=[account(include_positions=False)])
    source.all_payload[0]["securitiesAccount"]["positions"] = [
        position(symbol), position(malformed_symbol)
    ]
    context = {
        "logger": logging.getLogger("portfolio-invalid-iv"),
        "risk.cache.path": str(tmp_path / "cache.sqlite"),
        "risk.state.path": str(tmp_path / "state.sqlite"),
        "risk.rates.configured_flat_continuous_rate": 0.04,
    }

    class CapturingTianService:
        def __init__(self):
            self.source_ivs = []

        def solve(self, request):
            self.source_ivs = [contract.source_iv for contract in request.contracts]
            return SimpleNamespace(
                as_of=request.valuation_at,
                method_version="test_tian",
                data=[
                    SimpleNamespace(
                        canonical_occ_symbol=contract.canonical_occ_symbol,
                        model_dump=lambda **_kwargs: {"status": "unable_to_evaluate"},
                    )
                    for contract in request.contracts
                ],
            )

    tian = CapturingTianService()
    reference = SimpleNamespace(
        capabilities=lambda: {"available": {"risk_free_curve": False}, "supported": {}},
        quotes=lambda _symbols: [
            {"symbol": "SPY", "mark": 510.0},
            {"symbol": "QQQ", "mark": 510.0},
        ],
        security_classifications=lambda _symbols: [],
        option_quotes=lambda _symbols: {
            "data": [
                {
                    "canonical_occ_symbol": symbol,
                    "underlying_price": 510.0,
                    "bid": 15.0,
                    "ask": 16.0,
                    "mark": 15.5,
                    "provider_iv": -9.99,
                    "provider_delta": 0.5,
                },
                {
                    "canonical_occ_symbol": malformed_symbol,
                    "underlying_price": 510.0,
                    "bid": 15.0,
                    "ask": 16.0,
                    "mark": 15.5,
                    "quote_time": "not-a-timestamp",
                    "provider_delta": 0.5,
                },
            ],
            "meta": {},
        },
    )
    portfolio = PortfolioApplicationService(
        RiskService(source), context, reference_client=reference, tian_service=tian
    )
    snapshot = portfolio.snapshot(["all"], refresh=True)

    assert f"{symbol}:provider_iv_unusable" in snapshot.data_quality.warnings
    assert (
        f"{malformed_symbol}:exercise_style_unknown_current_greeks_unavailable"
        in snapshot.data_quality.warnings
    )
    assert tian.source_ivs == []
    assert snapshot.data.positions[0].greek_source is None
    assert snapshot.data.positions[0].delta is None
    assert snapshot.data.positions[0].implied_volatility is None


def test_portfolio_snapshot_requests_mutual_fund_quote_and_uses_mark_as_nav(
    tmp_path,
):
    payload = account(include_positions=False)
    fund_position = position("SWVXX")
    fund_position["instrument"]["type"] = "NO_LOAD_TAXABLE"
    fund_position["longQuantity"] = 1000
    fund_position["averagePrice"] = 1.0
    fund_position["marketValue"] = 1000
    payload["securitiesAccount"]["positions"] = [fund_position]
    requested = []

    def quotes(symbols):
        requested.append(tuple(symbols))
        return [{
            "symbol": "SWVXX",
            "mark": 1.0,
            "quote_time": "2026-08-26T22:45:00Z",
            "source_kind": "exchange_quote",
        }]

    reference = SimpleNamespace(
        capabilities=lambda: {"available": {}, "supported": {}},
        quotes=quotes,
        security_classifications=lambda _symbols: [],
    )
    now = datetime(2026, 8, 27, 23, 0, tzinfo=timezone.utc)
    portfolio = PortfolioApplicationService(
        RiskService(FakeSource(all_payload=[payload])),
        {
            "logger": logging.getLogger("portfolio-mutual-fund-nav"),
            "risk.cache.path": str(tmp_path / "cache.sqlite"),
            "risk.state.path": str(tmp_path / "state.sqlite"),
        },
        reference_client=reference,
        now_fn=lambda: now,
    )

    snapshot = portfolio.snapshot(["all"], refresh=True)
    fund = snapshot.data.positions[0]

    assert requested == [("SWVXX",)]
    assert fund.mark == 1.0
    assert fund.signed_market_value_usd == 1000
    assert fund.field_validity["mark"].valid_for_risk is True
    assert snapshot.data_quality.coverage.positions_modeled == 1


def test_company_profile_proxy_and_exposure_asset_scope(tmp_path, monkeypatch):
    source = FakeSource()
    context = {"logger": logging.getLogger("portfolio-profile"), "risk.cache.path": str(tmp_path/"cache.sqlite"),
               "risk.state.path": str(tmp_path/"state.sqlite"), "risk.refdata.base_url": "http://127.0.0.1:1"}
    app = create_app(context=context, source=source, config=RiskApiConfig())
    with TestClient(app) as client:
        client.get("/api/v1/portfolio/snapshot", params={"refresh": "true"})
        app.state.portfolio_service.reference_client.company_profile = lambda symbol: {
            "symbol": symbol.upper(), "display_name": "Taiwan Semiconductor", "sector": "Technology",
        }
        profile = client.get("/api/v1/reference/companies/tsm")
        options = client.get("/api/v1/portfolio/exposures", params={"asset_scope": "options"})
        monkeypatch.setattr(
            "algotrading.risk.api.routes._factor_analysis",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("factor history should be skipped")
            ),
        )
        lightweight = client.get(
            "/api/v1/portfolio/exposures",
            params={"asset_scope": "all", "include_factors": "false"},
        )
    assert profile.status_code == 200
    assert profile.json()["data"]["sector"] == "Technology"
    assert options.status_code == 200
    assert options.json()["data"]["sectors"] == []
    assert lightweight.status_code == 200
    assert lightweight.json()["data"]["factors"]["status"] == "not_requested"


def test_risk_calendar_only_includes_high_impact_economic_events(tmp_path):
    payload = account(include_positions=False)
    held = position("SPY")
    held["instrument"]["type"] = "EQUITY"
    payload["securitiesAccount"]["positions"] = [held]
    source = FakeSource(all_payload=[payload])
    context = {"logger": logging.getLogger("portfolio-calendar"), "risk.cache.path": str(tmp_path/"cache.sqlite"),
               "risk.state.path": str(tmp_path/"state.sqlite"), "risk.refdata.base_url": "http://127.0.0.1:1"}
    app = create_app(context=context, source=source, config=RiskApiConfig())
    earnings_calls = []
    dividend_calls = []
    with TestClient(app) as client:
        client.get("/api/v1/portfolio/snapshot", params={"refresh": "true"})
        reference = app.state.portfolio_service.reference_client
        reference.earnings_calendar = lambda *args, **kwargs: earnings_calls.append(kwargs) or []
        reference.dividend_calendar = lambda *args, **kwargs: dividend_calls.append(kwargs) or []
        reference.fomc_calendar = lambda *args, **kwargs: [{"date": "2026-08-17"}]
        reference.economic_calendar = lambda *args, **kwargs: [
            {"timestamp": "2026-08-17T08:30:00-04:00", "report": "High report", "impact": "High"},
            {"timestamp": "2026-08-17T10:00:00-04:00", "report": "Low report", "impact": "Low"},
            {"timestamp": "2026-08-17T14:00:00-04:00", "report": "Fed Interest Rate Decision", "internal_name": "fed_interest_rates", "impact": "High"},
        ]
        calendar = client.get("/api/v1/portfolio/risk-calendar")
    titles = [item["title"] for item in calendar.json()["data"]["events"]]
    assert "High report" in titles
    assert "Low report" not in titles
    assert titles.count("FOMC meeting") == 1
    assert "Fed Interest Rate Decision" not in titles
    assert len(earnings_calls) == 1
    assert len(dividend_calls) == 1


def test_pnl_explain_uses_broker_day_fallback_then_previous_refresh_anchor(tmp_path):
    source = FakeSource()
    context = {"logger": logging.getLogger("portfolio-pnl"), "risk.cache.path": str(tmp_path/"cache.sqlite"),
               "risk.state.path": str(tmp_path/"state.sqlite"), "risk.refdata.base_url": "http://127.0.0.1:1"}
    app = create_app(context=context, source=source, config=RiskApiConfig())
    with TestClient(app) as client:
        first = client.get("/api/v1/portfolio/snapshot", params={"refresh": "true"}).json()
        fallback = client.get("/api/v1/portfolio/pnl-explain").json()["data"]
        first_at = app.state.portfolio_service._now()
        app.state.portfolio_service._now = lambda: first_at + timedelta(seconds=61)
        second = client.get("/api/v1/portfolio/snapshot", params={"refresh": "true"}).json()
        anchored = client.get("/api/v1/portfolio/pnl-explain").json()["data"]
    assert fallback["anchor_mode"] == "broker_current_day_fallback"
    assert fallback["actual_pnl"] == -1.25
    assert fallback["reason"] is None
    assert anchored["anchor_mode"] == "refresh_to_refresh"
    assert anchored["prior_anchor_at"].replace("+00:00", "Z") == first["as_of"]
    assert anchored["current_anchor_at"].replace("+00:00", "Z") == second["as_of"]
    assert anchored["modeled_value_change"] == 0
