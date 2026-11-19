from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import math

from fastapi.testclient import TestClient
import pytest

from algotrading.risk.market_inputs.client import ReferenceDataClientError
from algotrading.risk.models import Coverage, DataQuality, NormalizedPosition

from ..app import create_app
from ..config import RiskApiConfig
from ..hedge_models import HedgeAssessmentRequest, HedgeCandidateQueryRequest, HedgeProposalRequest
from ..hedge_service import (
    _candidate_sort_key,
    _current_metrics,
    _future_contract_ladders,
    _select_active_future_contracts,
    _terminal_payoff,
)
from ..portfolio_service import _standard_share_option_deliverable
from .fakes import FIXED_NOW, FakeSource, account, position


def _equity_source() -> FakeSource:
    payload = account(include_positions=False)
    raw = position("AAPL")
    raw["instrument"]["type"] = "EQUITY"
    raw["marketValue"] = 200.0
    raw["longQuantity"] = 2.0
    payload["securitiesAccount"]["positions"] = [raw]
    return FakeSource(all_payload=[payload])


def _bars(symbols, *, start, end, **_kwargs):
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        price = 80.0 + symbol_index
        bars = []
        for day_index in range(190):
            at = FIXED_NOW.date() - timedelta(days=190 - day_index)
            wave = math.sin((day_index + 1) * (symbol_index + 2) * 0.071)
            price *= 1.0 + 0.0005 * (symbol_index % 3 - 1) + 0.003 * wave
            bars.append({"timestamp": at.isoformat(), "close": price})
        rows.append({"symbol": symbol, "bars": bars})
    return rows


@pytest.fixture
def hedge_client(tmp_path):
    context = {
        "logger": logging.getLogger("hedge-lab-routes"),
        "risk.cache.path": str(tmp_path / "cache.sqlite"),
        "risk.state.path": str(tmp_path / "state.sqlite"),
        "risk.history.minimum_observations": 126,
        "risk.hedge_lab.maximum_equity_symbols": 3,
    }
    app = create_app(context=context, source=_equity_source(), config=RiskApiConfig())
    with TestClient(app) as client:
        app.state.portfolio_service._now = lambda: FIXED_NOW
        reference = app.state.portfolio_service.reference_client
        quote_as_of = (FIXED_NOW - timedelta(days=1, hours=2, minutes=30)).isoformat()
        reference.capabilities = lambda: {
            "available": {"options_chain": False, "risk_free_curve": False},
            "supported": {},
        }
        reference.quotes = lambda symbols: [
            {
                "symbol": symbol,
                "mark": 100.0,
                "quote_time": quote_as_of,
                "source": "test",
                "source_kind": "completed_session_mark",
            }
            for symbol in symbols
        ]
        reference.security_classifications = lambda symbols: [
            {"symbol": symbol, "security_type": "EQUITY", "sector": "Technology"}
            for symbol in symbols
        ]
        reference.universe_symbols = reference.security_classifications
        reference.option_quotes = lambda symbols: {"data": [], "missing_symbols": list(symbols), "meta": {}}
        reference.futures_specifications = lambda symbols: {"data": [], "meta": {}}
        reference.bars = _bars
        reference.current_price = lambda symbol: {
            "data": {"symbol": symbol, "price": 100.0},
            "meta": {"as_of": quote_as_of},
        }
        snapshot_response = client.get("/api/v1/portfolio/snapshot", params={"refresh": "true"})
        assert snapshot_response.status_code == 200, snapshot_response.text
        yield client, app, snapshot_response.json()


def _request(snapshot, **updates):
    held = snapshot["data"]["positions"][0]
    value = {
        "account_ids": snapshot["data"]["account_ids"],
        "portfolio_snapshot_id": snapshot["portfolio_snapshot_id"],
        "target": {
            "kind": "Security",
            "label": "AAPL",
            "position_ids": [held["position_id"], held["position_id"]],
            "economic_position_ids": [held["economic_position_id"]],
        },
        "objective": "VaR / Tail Risk",
        "objective_parameters": {
            "near_confidence": 0.95,
            "far_confidence": 0.975,
            "hedge_region": "marginal",
        },
    }
    value.update(updates)
    return value


def test_openapi_and_target_assessment_are_strict_and_snapshot_coherent(hedge_client):
    client, _app, snapshot = hedge_client
    response = client.post("/api/v1/hedge-lab/targets/assess", json=_request(snapshot))
    schema = client.get("/openapi.json").json()

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["portfolio_snapshot_id"] == snapshot["portfolio_snapshot_id"]
    assert result["configuration_version"] == snapshot["configuration_version"]
    assert result["as_of"] == snapshot["as_of"]
    assert result["methodology_version"] == "hedge_target_assessment_v1"
    assert result["data"]["analysis_only"] is True
    assert len(result["data"]["target"]["position_ids"]) == 1
    assert len(result["data"]["distribution"]) >= 126
    contribution = result["data"]["tail_risk"]["contributions"][0]
    assert contribution["position_ids"] == result["data"]["target"]["position_ids"]
    assert contribution["economic_position_id"] == result["data"]["target"]["economic_position_ids"][0]
    assert result["data"]["factors"]["methodology_version"] == "dollar_factor_regression_v3"
    assert result["data"]["factors"]["items"][0]["response_curve"][0] == {
        "factor_move_percent": -4.0,
        "pnl_usd": pytest.approx(
            -4.0
            * result["data"]["factors"]["items"][0][
                "estimated_pnl_for_plus_1pct_factor_move_usd"
            ]
        ),
    }
    assert {
        "/api/v1/hedge-lab/targets/assess",
        "/api/v1/hedge-lab/candidates/query",
        "/api/v1/hedge-lab/proposals/evaluate",
        "/api/v1/hedge-lab/options/expirations",
        "/api/v1/options/reprice",
    } <= set(schema["paths"])
    assert schema["components"]["schemas"]["HedgeAssessmentRequest"]["additionalProperties"] is False


def test_unknown_and_stale_target_ids_are_rejected_deterministically(hedge_client):
    client, _app, snapshot = hedge_client
    unknown = _request(snapshot)
    unknown["target"]["position_ids"] = ["missing"]
    stale = _request(snapshot, portfolio_snapshot_id="sha256:stale")

    assert client.post("/api/v1/hedge-lab/targets/assess", json=unknown).status_code == 422
    assert client.post("/api/v1/hedge-lab/targets/assess", json=stale).status_code == 409


def test_option_chain_upstream_failure_is_a_sanitized_503(hedge_client):
    client, app, snapshot = hedge_client
    reference = app.state.portfolio_service.reference_client
    reference.capabilities = lambda: {
        "available": {"options_chain": True, "risk_free_curve": False},
        "supported": {},
    }
    snapshot_response, _ = app.state.portfolio_service.require_snapshot(
        snapshot["data"]["account_ids"], snapshot["portfolio_snapshot_id"]
    )
    snapshot_response.data.capabilities["available"]["options_chain"] = True

    def unavailable(_payload):
        raise ReferenceDataClientError(
            "Reference Data request failed status=503 code=UPSTREAM_UNAVAILABLE"
        )

    reference.option_chain = unavailable
    response = client.post(
        "/api/v1/hedge-lab/options/chain",
        json={
            "account_ids": snapshot["data"]["account_ids"],
            "portfolio_snapshot_id": snapshot["portfolio_snapshot_id"],
            "symbol": "QQQ",
            "contract_type": "PUT",
            "strike_count": 100,
            "from_date": FIXED_NOW.date().isoformat(),
            "to_date": FIXED_NOW.date().isoformat(),
            "include_underlying_quote": True,
        },
    )

    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "REFERENCE_DATA_UNAVAILABLE"
    assert response.json()["error"]["message"] == (
        "Reference market data is temporarily unavailable"
    )


def test_expiration_discovery_uses_calendar_only_refdata_route(
    hedge_client,
):
    client, app, snapshot = hedge_client
    calls = []
    snapshot_response, _ = app.state.portfolio_service.require_snapshot(
        snapshot["data"]["account_ids"], snapshot["portfolio_snapshot_id"]
    )
    snapshot_response.data.capabilities["available"]["options_chain"] = True

    def option_expirations(payload):
        calls.append(payload)
        return {
            "data": {
                "symbol": "AAPL",
                "expirations": [{
                    "expiration": "2026-09-18",
                    "days_to_expiration": 34,
                    "expiration_type": "S",
                    "settlement_type": "P",
                    "option_roots": ["AAPL"],
                    "standard": True,
                }],
            },
            "meta": {"as_of": FIXED_NOW.isoformat(), "cache": "miss"},
        }

    app.state.portfolio_service.reference_client.option_expirations = option_expirations
    response = client.post(
        "/api/v1/hedge-lab/options/expirations",
        json={
            "account_ids": snapshot["data"]["account_ids"],
            "portfolio_snapshot_id": snapshot["portfolio_snapshot_id"],
            "symbol": "AAPL",
            "from_date": "2026-08-15",
            "to_date": "2026-11-15",
            "include_underlying_quote": True,
        },
    )

    assert response.status_code == 200, response.text
    assert calls == [{
        "symbol": "AAPL",
        "from_date": "2026-08-15",
        "to_date": "2026-11-15",
    }]
    assert "contracts" not in response.json()["data"]
    assert response.json()["data"]["expirations"] == [{
        "expiration": "2026-09-18",
        "days_to_expiration": 34,
        "expiration_type": "S",
        "settlement_type": "P",
        "option_roots": ["AAPL"],
        "standard": True,
    }]


def test_hedge_chain_refreshes_after_cached_chain_hard_expiry(hedge_client):
    _client, app, snapshot = hedge_client
    service = app.state.portfolio_service
    response, _ = service.require_snapshot(
        snapshot["data"]["account_ids"], snapshot["portfolio_snapshot_id"]
    )
    response.data.capabilities["available"]["options_chain"] = True
    current = [FIXED_NOW]
    service._now = lambda: current[0]
    calls: list[datetime] = []

    def option_chain(bounds):
        calls.append(current[0])
        return {
            "data": {
                "symbol": bounds["symbol"],
                "underlying_quote": {"mark": 100.0},
                "contracts": [],
            },
            "meta": {"as_of": current[0].isoformat()},
        }

    service.reference_client.option_chain = option_chain
    kwargs = {
        "portfolio_snapshot_id": snapshot["portfolio_snapshot_id"],
        "symbol": "SPY",
        "contract_type": "ALL",
        "strike_count": 40,
        "from_date": FIXED_NOW.date().isoformat(),
        "to_date": (FIXED_NOW.date() + timedelta(days=30)).isoformat(),
        "include_underlying_quote": True,
    }

    first = service.hedge_chain(snapshot["data"]["account_ids"], **kwargs)
    current[0] = FIXED_NOW + timedelta(minutes=4)
    cached = service.hedge_chain(snapshot["data"]["account_ids"], **kwargs)
    current[0] = FIXED_NOW + timedelta(minutes=6)
    refreshed = service.hedge_chain(snapshot["data"]["account_ids"], **kwargs)

    assert len(calls) == 2
    assert cached["source_as_of"] == first["source_as_of"]
    assert refreshed["source_as_of"] != first["source_as_of"]


def test_dated_option_reprice_uses_the_source_date_curve(hedge_client):
    client, app, _snapshot = hedge_client
    requested_dates = []

    def curve(valuation_date):
        requested_dates.append(valuation_date)
        return {
            "data": {
                "curve_id": "test-usd",
                "curve_version": "test-v1",
                "anchors": [{
                    "series_id": "DGS3MO",
                    "tenor_days": 90,
                    "observation_date": FIXED_NOW.date().isoformat(),
                    "simple_annual_rate_decimal": 0.04,
                    "status": "ok",
                }],
            },
        }

    app.state.portfolio_service.reference_client.risk_free_curve = curve
    response = client.post(
        "/api/v1/options/reprice",
        json={
            "valuation_at": (FIXED_NOW + timedelta(days=7)).isoformat(),
            "contracts": [{
                "request_id": "dated-curve",
                "canonical_occ_symbol": "QQQ260918P00450000",
                "underlying_symbol": "QQQ",
                "put_call": "PUT",
                "strike": 450,
                "expiration_at": (FIXED_NOW + timedelta(days=34)).isoformat(),
                "source_valuation_at": FIXED_NOW.isoformat(),
                "source_spot": 460,
                "source_mark": 7.5,
                "implied_volatility": 0.22,
                "scenario_spot": 440,
            }],
        },
    )

    assert response.status_code == 200, response.text
    assert requested_dates == [FIXED_NOW.date()]
    assert response.json()["data"][0]["status"] == "ok"
    assert response.json()["data"][0]["reason"] == "complete"


def test_candidates_are_bounded_deterministic_priced_and_directly_usable(hedge_client):
    client, _app, snapshot = hedge_client
    payload = {**_request(snapshot), "limit": 2}
    first = client.post("/api/v1/hedge-lab/candidates/query", json=payload)
    second = client.post("/api/v1/hedge-lab/candidates/query", json=payload)

    assert first.status_code == 200, first.text
    assert first.json()["data"]["items"] == second.json()["data"]["items"]
    assert 0 < len(first.json()["data"]["items"]) <= 2, first.json()["data"]["warnings"]
    leg = first.json()["data"]["items"][0]["legs"][0]
    assert leg["mark"] > 0 and leg["multiplier"] > 0
    assert leg["source"]

    request_leg_keys = {
        "client_leg_id", "instrument_type", "side", "quantity", "symbol",
        "canonical_occ_symbol", "expiration_at", "put_call", "strike", "mark",
        "multiplier", "hidden",
    }
    proposal_leg = {
        key: value for key, value in leg.items() if key in request_leg_keys
    }
    proposal_leg["hidden"] = False
    evaluated = client.post(
        "/api/v1/hedge-lab/proposals/evaluate",
        json={**_request(snapshot), "legs": [proposal_leg]},
    )
    assert evaluated.status_code == 200, evaluated.text
    evaluated_data = evaluated.json()["data"]
    expected_effect = (
        (1 if leg["side"] == "BUY" else -1)
        * leg["quantity"] * leg["mark"]
    )
    assert evaluated_data["portfolio"]["change"]["directional_delta_exposure_usd"] == expected_effect
    assert evaluated_data["var_after_usd"] == evaluated_data["tail_risk"]["var_after_usd"]
    assert evaluated_data["pricing_audit"][0]["client_supplied"] is True


def test_option_proposal_prefers_fresh_exact_quote_over_stale_client_mark(
    hedge_client,
):
    client, app, snapshot = hedge_client
    app.state.hedge_lab_service._risk_free_rate = lambda *_args: 0.04
    reference = app.state.portfolio_service.reference_client
    expiration = FIXED_NOW + timedelta(days=30)
    canonical = f"SPY{expiration:%y%m%d}P00100000"
    quote_time = FIXED_NOW - timedelta(days=1, hours=2, minutes=30)
    reference.option_quotes = lambda _symbols: {
        "data": [{
            "canonical_occ_symbol": canonical,
            "status": "ok",
            "bid": 4.9,
            "ask": 5.1,
            "mark": 5.0,
            "underlying_price": 100.0,
            "quote_time": quote_time.isoformat(),
            "source_kind": "completed_session_mark",
        }],
        "missing_symbols": [],
        "meta": {"as_of": quote_time.isoformat()},
    }
    reference.dividends = lambda *_args, **_kwargs: []
    payload = {
        **_request(
            snapshot,
            objective="Greeks",
            objective_parameters={
                "selected_greeks": ["delta"],
                "targets": {"delta": 0},
            },
        ),
        "legs": [{
            "client_leg_id": "stale-picker-mark",
            "instrument_type": "OPTION",
            "side": "BUY",
            "quantity": 1,
            "symbol": "SPY",
            "canonical_occ_symbol": canonical,
            "expiration_at": expiration.isoformat(),
            "put_call": "PUT",
            "strike": 100,
            "mark": 0.01,
            "multiplier": 100,
            "hidden": False,
        }],
    }

    result = client.post("/api/v1/hedge-lab/proposals/evaluate", json=payload)

    assert result.status_code == 200, result.text
    data = result.json()["data"]
    assert data["active_legs"][0]["mark"] == pytest.approx(5.0)
    assert data["active_legs"][0]["multiplier"] == pytest.approx(100.0)
    delta_change = (
        data["portfolio"]["after"]["directional_delta_exposure_usd"]
        - data["portfolio"]["before"]["directional_delta_exposure_usd"]
    )
    resolved = data["active_legs"][0]
    assert delta_change == pytest.approx(
        resolved["quantity"]
        * resolved["multiplier"]
        * resolved["delta"]
        * resolved["spot"]
    )
    assert data["pricing_audit"][0]["client_supplied"] is False
    assert "stale-picker-mark:client_mark_replaced_by_exact_quote" in result.json()[
        "data_quality"
    ]["warnings"]


def test_factor_proposal_reuses_revaluation_and_reports_beta_and_clean_curve(
    hedge_client,
):
    client, app, snapshot = hedge_client
    service = app.state.hedge_lab_service
    service._history_cache.clear()
    warmed = client.get("/api/v1/portfolio/tail-risk")
    assert warmed.status_code == 200, warmed.text
    assert service._history_cache is app.state.historical_revaluation_cache
    original_factory = service._historical_revaluation_engine
    revaluation_calls: list[tuple[str, ...]] = []

    class CountingEngine:
        def __init__(self, delegate):
            self.delegate = delegate

        def revalue(self, positions, **kwargs):
            revaluation_calls.append(tuple(item.position_id for item in positions))
            return self.delegate.revalue(positions, **kwargs)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    service._historical_revaluation_engine = lambda: CountingEngine(original_factory())
    payload = {
        **_request(
            snapshot,
            objective="Factors",
            objective_parameters={"factor": "Market", "target_exposure": 0},
        ),
        "legs": [{
            "client_leg_id": "factor-qqq",
            "instrument_type": "EQUITY",
            "side": "BUY",
            "quantity": 10,
            "symbol": "QQQ",
            "mark": 100,
            "multiplier": 1,
            "hidden": False,
        }],
        "evaluation_mode": "build",
    }

    response = client.post("/api/v1/hedge-lab/proposals/evaluate", json=payload)

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert response.json()["methodology_version"] == "hedge_proposal_build_evaluation_v1"
    assert data["evaluation_mode"] == "build"
    assert data["scenarios"] == []
    assert data["scenario_response"] == []
    assert data["tail_risk"]["reason"] == "deferred_until_full_validation"
    # The held portfolio was already valued by the app-wide tail endpoint, so
    # Hedge Analysis needs only the additive-leg solve.  The Factor objective
    # must consume those results instead of launching two more.
    assert len(revaluation_calls) == 1
    objective = data["target"]["objective"]
    market_before = next(
        item for item in objective["before"]["items"] if item["name"] == "Market"
    )
    market_after = next(
        item for item in objective["after"]["items"] if item["name"] == "Market"
    )
    assert market_before["beta"] is not None
    assert market_after["beta"] is not None
    assert market_before["estimated_pnl_for_plus_1pct_factor_move_usd"] == pytest.approx(
        market_before["coefficient_usd_per_unit_factor_return"] * 0.01
    )
    assert market_after["estimated_pnl_for_plus_1pct_factor_move_usd"] == pytest.approx(
        market_after["coefficient_usd_per_unit_factor_return"] * 0.01
    )
    assert market_before["beta"] == pytest.approx(
        market_before["coefficient_usd_per_unit_factor_return"]
        / data["target"]["before"]["directional_delta_exposure_usd"]
    )
    assert market_after["beta"] == pytest.approx(
        market_after["coefficient_usd_per_unit_factor_return"]
        / data["target"]["after"]["directional_delta_exposure_usd"]
    )
    assert "unit_contributions" not in objective["before"]
    assert "scenario_provenance" not in objective["before"]
    full_payload = {key: value for key, value in payload.items() if key != "evaluation_mode"}
    full_response = client.post(
        "/api/v1/hedge-lab/proposals/evaluate", json=full_payload
    )
    assert full_response.status_code == 200, full_response.text
    full_data = full_response.json()["data"]
    assert len(revaluation_calls) == 1
    assert [row["definition"]["broad_equity_shock"] for row in full_data["scenario_response"]] == [
        -0.1, -0.05, 0.0, 0.05, 0.1,
    ]
    assert all(row["definition"]["iv_points"] == 0 for row in full_data["scenario_response"])
    assert all(row["definition"]["rate_basis_points"] == 0 for row in full_data["scenario_response"])


def test_tail_build_reuses_additive_revaluation_and_defers_validation_work(
    hedge_client,
):
    client, app, snapshot = hedge_client
    service = app.state.hedge_lab_service
    service._history_cache.clear()
    warmed = client.get("/api/v1/portfolio/tail-risk")
    assert warmed.status_code == 200, warmed.text
    original_factory = service._historical_revaluation_engine
    revaluation_calls: list[tuple[str, ...]] = []

    class CountingEngine:
        def __init__(self, delegate):
            self.delegate = delegate

        def revalue(self, positions, **kwargs):
            revaluation_calls.append(tuple(item.position_id for item in positions))
            return self.delegate.revalue(positions, **kwargs)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    service._historical_revaluation_engine = lambda: CountingEngine(original_factory())
    payload = {
        **_request(snapshot),
        "legs": [{
            "client_leg_id": "tail-qqq",
            "instrument_type": "EQUITY",
            "side": "BUY",
            "quantity": 10,
            "symbol": "QQQ",
            "mark": 100,
            "multiplier": 1,
            "hidden": False,
        }],
        "evaluation_mode": "build",
    }

    response = client.post("/api/v1/hedge-lab/proposals/evaluate", json=payload)

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert response.json()["methodology_version"] == "hedge_proposal_build_evaluation_v1"
    assert data["evaluation_mode"] == "build"
    assert data["scenarios"] == []
    assert data["scenario_response"] == []
    assert data["maximum_loss"]["reason"] == "deferred_until_full_validation"
    assert data["target"]["objective"]["before"]["near"]["var_usd"] is not None
    assert data["target"]["objective"]["after"]["near"]["var_usd"] is not None
    # The portfolio result is warm, and the proposed result is assembled from
    # that cached history plus exactly one additive-leg revaluation.
    assert len(revaluation_calls) == 1

    full_payload = {key: value for key, value in payload.items() if key != "evaluation_mode"}
    full_response = client.post(
        "/api/v1/hedge-lab/proposals/evaluate", json=full_payload
    )
    assert full_response.status_code == 200, full_response.text
    assert full_response.json()["data"]["scenario_response"]
    # Validate reuses the same warmed base/addition histories too; the primary
    # tail objective must not launch another before/after repricing pass.
    assert len(revaluation_calls) == 1


def test_effect_preview_does_not_run_full_proposal_revaluation(hedge_client):
    client, app, snapshot = hedge_client

    def unexpected_evaluation(_payload):
        raise AssertionError("effects preview must not run proposal evaluation")

    app.state.hedge_lab_service.evaluate = unexpected_evaluation
    response = client.post(
        "/api/v1/hedge-lab/candidates/query",
        json={**_request(snapshot), "limit": 4, "evaluation_mode": "effects"},
    )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["items"]
    assert data["ranking"]["full_revaluation_applied"] is False
    assert response.json()["methodology_version"] == "hedge_candidate_effect_preview_v1"
    greek_effect_keys = {
        "option_delta_exposure_effect_usd",
        "gamma_pnl_for_squared_1pct_move_effect_usd",
        "vega_per_1pct_effect_usd",
        "theta_per_day_effect_usd",
        "rho_per_1pct_effect_usd",
    }
    for candidate in data["items"]:
        assert greek_effect_keys <= candidate["residuals"].keys()
        assert all(
            isinstance(candidate["residuals"][key], (int, float))
            for key in greek_effect_keys
        )


def test_greek_effect_preview_is_twelve_fixed_spy_option_templates(hedge_client):
    client, app, snapshot = hedge_client
    response, _ = app.state.portfolio_service.require_snapshot(
        snapshot["data"]["account_ids"], snapshot["portfolio_snapshot_id"]
    )
    response.data.capabilities["available"]["options_chain"] = True
    requested_chains = []
    app.state.portfolio_service.hedge_expirations = lambda *_args, **_kwargs: {
        "data": {"expirations": [
            {"expiration": (FIXED_NOW + timedelta(days=dte)).date().isoformat()}
            for dte in (7, 30, 90)
        ]}
    }
    app.state.portfolio_service.hedge_chain = lambda *_args, **kwargs: requested_chains.append(kwargs) or {"data": {}}

    rows = []
    for dte in (7, 30, 90):
        expiration = FIXED_NOW + timedelta(days=dte)
        stamp = expiration.strftime("%y%m%d")
        for put_call, strike, delta in (
            ("PUT", 95.0, -0.25),
            ("PUT", 100.0, -0.50),
            ("CALL", 100.0, 0.50),
            ("CALL", 105.0, 0.25),
        ):
            flag = "P" if put_call == "PUT" else "C"
            rows.append({
                "canonical_occ_symbol": f"SPY{stamp}{flag}{int(strike * 1000):08d}",
                "expiration_at": expiration,
                "put_call": put_call,
                "strike": strike,
                "mark": 2.0,
                "multiplier": 100.0,
                "spot": 100.0,
                "delta": delta,
                "gamma": 0.02,
                "vega_per_1pct": 0.10,
                "theta_per_day": -0.05,
                "rho_per_1pct": 0.03,
                "quote_time": FIXED_NOW,
                "bid": 1.9,
                "ask": 2.1,
                "open_interest": 5000,
            })
    app.state.hedge_lab_service._normalized_chain_rows = (
        lambda *_args, **_kwargs: rows
    )
    result = client.post(
        "/api/v1/hedge-lab/candidates/query",
        json={
            **_request(snapshot),
            "objective": "Greeks",
            "objective_parameters": {
                "selected_greeks": ["delta", "gamma", "vega", "theta"],
                "targets": {"delta": 0, "gamma": 0, "vega": 0, "theta": 0},
            },
            "limit": 12,
            "evaluation_mode": "effects",
        },
    )

    assert result.status_code == 200, result.text
    items = result.json()["data"]["items"]
    assert len(items) == 12
    assert {item["preview_dte"] for item in items} == {7, 30, 90}
    assert {item["preview_contract"] for item in items} == {
        "CALL ATM", "PUT ATM", "PUT OTM 25 delta", "CALL OTM 25 delta",
    }
    assert all(item["legs"][0]["instrument_type"] == "OPTION" for item in items)
    assert len(requested_chains) == 3
    assert all(call["from_date"] == call["to_date"] for call in requested_chains)
    assert all(call["strike_count"] == 12 for call in requested_chains)
    assert result.json()["methodology_version"] == "spy_cross_greek_effect_preview_v1"


def test_chain_quote_newer_than_snapshot_is_priced_at_chain_request_time(
    hedge_client,
):
    _client, app, _snapshot = hedge_client
    app.state.hedge_lab_service._risk_free_rate = lambda *_args: 0.04
    current = FIXED_NOW + timedelta(days=2, minutes=5)
    app.state.portfolio_service._now = lambda: current
    expiration = FIXED_NOW + timedelta(days=30)
    canonical = f"SPY{expiration:%y%m%d}C00100000"
    chain_data = app.state.portfolio_service._enrich_chain_contract_terms({
        "underlying_quote": {"mark": 100.0},
        "contracts": [{
            "canonical_occ_symbol": canonical,
            "put_call": "CALL",
            "strike": 100.0,
            "expiration": expiration.date().isoformat(),
            "bid": 4.9,
            "ask": 5.1,
            "mark": 5.0,
            "quote_time": current.isoformat(),
            "multiplier": 100.0,
            "deliverable": (
                "[{'symbol': 'SPY', 'assetType': 'STOCK', "
                "'deliverableUnits': 100.0, 'currencyType': None}]"
            ),
        }],
    })

    rows = app.state.hedge_lab_service._normalized_chain_rows(
        {"source_as_of": current.isoformat(), "data": chain_data},
        "SPY",
        FIXED_NOW,
    )

    assert len(rows) == 1
    assert rows[0]["canonical_occ_symbol"] == canonical
    assert rows[0]["quote_time"] == current


def test_standard_spy_chain_terms_do_not_call_remote_contract_metadata(hedge_client):
    _client, app, _snapshot = hedge_client

    def unexpected_terms(_symbols):
        raise AssertionError("standard SPY chain rows must resolve locally")

    app.state.portfolio_service.reference_client.option_contract_terms = unexpected_terms
    enriched = app.state.portfolio_service._enrich_chain_contract_terms({
        "contracts": [{
            "canonical_occ_symbol": "SPY260918C00100000",
            "put_call": "CALL",
            "strike": 100.0,
            "multiplier": 100.0,
            "deliverable": (
                "[{'symbol': 'SPY', 'assetType': 'STOCK', "
                "'deliverableUnits': 100.0, 'currencyType': None}]"
            ),
        }],
    })

    contract = enriched["contracts"][0]
    assert datetime.fromisoformat(contract["expiration_at"]) == datetime(
        2026, 9, 18, 20, tzinfo=timezone.utc
    )
    assert contract["multiplier"] == 100.0
    assert contract["instrument_terms"]["exercise_style"] == "american"


def test_standard_share_deliverable_detection_is_conservative():
    assert _standard_share_option_deliverable(
        "[{'symbol': 'SPY', 'assetType': 'STOCK', "
        "'deliverableUnits': 100.0, 'currencyType': None}]",
        "SPY",
    )
    assert not _standard_share_option_deliverable(
        [{"symbol": "SPY", "deliverableUnits": 50.0}],
        "SPY",
    )
    assert not _standard_share_option_deliverable(
        [{"component_type": "cash", "cashAmount": 100.0}],
        "SPY",
    )
    assert not _standard_share_option_deliverable("unparsed deliverable", "SPY")


def test_candidate_universe_reserves_capacity_for_liquid_market_proxies(hedge_client):
    _client, app, snapshot = hedge_client
    _response, selected_snapshot = app.state.portfolio_service.require_snapshot(
        snapshot["data"]["account_ids"], snapshot["portfolio_snapshot_id"]
    )

    symbols = app.state.hedge_lab_service._relevant_symbols(selected_snapshot)

    assert symbols == ["SPY", "QQQ", "IWM"]


def test_portfolio_candidates_follow_measured_fit_not_instrument_class_priority(hedge_client):
    _client, app, snapshot = hedge_client
    _response, selected_snapshot = app.state.portfolio_service.require_snapshot(
        snapshot["data"]["account_ids"], snapshot["portfolio_snapshot_id"]
    )
    app.state.hedge_lab_service.context["risk.hedge_lab.maximum_equity_symbols"] = 12
    held = snapshot["data"]["positions"][0]
    payload = {
        **_request(snapshot),
        "target": {
            "kind": "Portfolio",
            "label": "Portfolio",
            "position_ids": [held["position_id"]],
            "economic_position_ids": [held["economic_position_id"]],
        },
        "objective": "Scenario Loss",
        "objective_parameters": {
            "default_equity_shock": -0.03,
            "iv_points": 3,
            "horizon_calendar_days": 1,
        },
        "limit": 12,
    }
    symbols = app.state.hedge_lab_service._relevant_symbols(
        selected_snapshot,
        target_kind="Portfolio",
        objective="Scenario Loss",
        objective_parameters=payload["objective_parameters"],
    )
    request = HedgeCandidateQueryRequest.model_validate(payload)
    candidates = [
        {"candidate_id": "company", "universe_category": "company_equity", "universe_priority": 8, "fit_score": 99},
        {"candidate_id": "sector", "universe_category": "sector_etf", "universe_priority": 7, "fit_score": 80},
        {"candidate_id": "index-option", "universe_category": "index_etf_option", "universe_priority": 2, "fit_score": 60},
        {"candidate_id": "index", "universe_category": "index_etf", "universe_priority": 0, "fit_score": 50},
    ]

    assert symbols[:4] == ["SPY", "QQQ", "IWM", "DIA"]
    assert symbols.index("XLK") < symbols.index("AAPL")
    assert [item["candidate_id"] for item in sorted(candidates, key=lambda item: _candidate_sort_key(request, item))] == [
        "company", "sector", "index-option", "index",
    ]


def test_portfolio_scenario_scan_returns_priced_index_candidates(hedge_client):
    client, _app, snapshot = hedge_client
    held = snapshot["data"]["positions"][0]
    response = client.post(
        "/api/v1/hedge-lab/candidates/query",
        json={
            **_request(snapshot),
            "target": {
                "kind": "Portfolio",
                "label": "Portfolio",
                "position_ids": [held["position_id"]],
                "economic_position_ids": [held["economic_position_id"]],
            },
            "objective": "Scenario Loss",
            "objective_parameters": {
                "default_equity_shock": -0.03,
                "iv_points": 3,
                "horizon_calendar_days": 1,
            },
            "limit": 3,
        },
    )

    assert response.status_code == 200, response.text
    items = response.json()["data"]["items"]
    assert items, response.json()["data"]["warnings"]
    assert all(item["universe_category"] == "index_etf" for item in items)
    assert {item["legs"][0]["symbol"] for item in items} <= {"SPY", "QQQ", "IWM", "DIA"}


def test_cme_expirations_select_product_specific_active_contracts():
    as_of = datetime(2026, 8, 26, 13, 0, tzinfo=timezone.utc)
    ladders = _future_contract_ladders(as_of.date())
    expirations = {
        "ESU26": "2026-09-18T21:00:00Z",
        "NQU26": "2026-09-18T21:00:00Z",
        "ZNU26": "2026-09-21T21:00:00Z",
        "CLQ26": "2026-07-21T18:30:00Z",
        "CLU26": "2026-08-20T18:30:00Z",
        "CLV26": "2026-09-22T18:30:00Z",
        "6EU26": "2026-09-14T14:16:00Z",
    }
    rows = [{
        "requested_contract": contract,
        "canonical_contract": contract,
        "status": "ok",
        "price_multiplier": 50,
        "last_trade_at": expiration,
    } for contract, expiration in expirations.items()]

    selected, warnings = _select_active_future_contracts(
        ladders,
        rows,
        as_of=as_of,
        roll_buffer_days=2,
    )

    assert selected == ["ESU26", "NQU26", "ZNU26", "CLV26", "6EU26"]
    assert all(warning.endswith(":active_futures_contract_unavailable") for warning in warnings)


def test_hidden_legs_are_audited_but_invariant_and_snapshot_is_not_mutated(hedge_client):
    client, app, snapshot = hedge_client
    before = app.state.portfolio_service.require_snapshot(
        snapshot["data"]["account_ids"], snapshot["portfolio_snapshot_id"]
    )[0].data.model_dump(mode="json")
    payload = {
        **_request(snapshot),
        "legs": [{
            "client_leg_id": "hidden-leg",
            "instrument_type": "EQUITY",
            "side": "BUY",
            "quantity": 50,
            "symbol": "SPY",
            "mark": 100,
            "hidden": True,
        }],
    }
    result = client.post("/api/v1/hedge-lab/proposals/evaluate", json=payload)
    after = app.state.portfolio_service.require_snapshot(
        snapshot["data"]["account_ids"], snapshot["portfolio_snapshot_id"]
    )[0].data.model_dump(mode="json")

    assert result.status_code == 200, result.text
    data = result.json()["data"]
    assert data["active_legs"] == []
    assert data["hidden_legs"][0]["client_leg_id"] == "hidden-leg"
    assert data["portfolio"]["change"]["directional_delta_exposure_usd"] == 0
    assert data["tail_risk"]["distribution_before"] == data["tail_risk"]["distribution_after"]
    assert len(result.content) < 64 * 1024 * 1024
    assert {
        "target", "portfolio", "greeks", "factors", "tail_risk", "scenarios",
        "maximum_loss", "active_legs", "pricing_audit",
    } <= set(data)
    assert before == after


def test_objective_validation_and_option_greek_units():
    with pytest.raises(ValueError):
        HedgeAssessmentRequest.model_validate({
            "portfolio_snapshot_id": "snapshot",
            "target": {"kind": "Security", "label": "x", "position_ids": ["p"]},
            "objective": "VaR / Tail Risk",
            "objective_parameters": {"near_confidence": 0.951, "far_confidence": 0.975},
        })
    with pytest.raises(ValueError):
        HedgeProposalRequest.model_validate({
            "portfolio_snapshot_id": "snapshot",
            "target": {"kind": "Security", "label": "x", "position_ids": ["p"]},
            "objective": "Factors",
            "objective_parameters": {},
            "legs": [
                {"client_leg_id": "same", "instrument_type": "EQUITY", "side": "BUY", "quantity": 1, "symbol": "SPY"},
                {"client_leg_id": "same", "instrument_type": "EQUITY", "side": "BUY", "quantity": 1, "symbol": "QQQ"},
            ],
        })

    quality = DataQuality(status="usable", coverage=Coverage(positions_total=1, positions_modeled=1))
    option = NormalizedPosition(
        position_id="option", economic_position_id="economic", account_id="a",
        instrument_id="SPY260918C00100000", asset_type="OPTION",
        exact_symbol="SPY260918C00100000", underlying_symbol="SPY", put_call="CALL",
        strike=100, signed_quantity=2, multiplier=100, mark=5,
        signed_market_value_usd=1000, delta=0.5, gamma=0.01,
        vega_per_1pct=0.2, theta_per_day=-0.1, rho_per_1pct=0.03,
        data_quality=quality, raw_broker_fields={"underlying_price": 100},
    )
    equity = NormalizedPosition(
        position_id="equity", economic_position_id="economic", account_id="a",
        instrument_id="SPY", asset_type="EQUITY", exact_symbol="SPY",
        underlying_symbol="SPY", signed_quantity=10, multiplier=1, mark=100,
        signed_market_value_usd=1000, data_quality=quality,
    )
    metrics = _current_metrics([option, equity])
    assert metrics["directional_delta_exposure_usd"] == 11_000
    assert metrics["option_delta_exposure_usd"] == 10_000
    assert metrics["gamma_pnl_for_squared_1pct_move_usd"] == 1
    assert metrics["vega_per_1pct_usd"] == 40
    assert metrics["theta_per_day_usd"] == -20
    assert metrics["rho_per_1pct_usd"] == 6

    inferred_spot = option.model_copy(update={
        "raw_broker_fields": {},
        "delta_exposure_usd": 10_000,
    })
    inferred_metrics = _current_metrics([inferred_spot])
    assert inferred_metrics["option_delta_exposure_usd"] == 10_000
    assert inferred_metrics["gamma_pnl_for_squared_1pct_move_usd"] == 1
    assert inferred_metrics["vega_per_1pct_usd"] == 40
    assert inferred_metrics["theta_per_day_usd"] == -20

    missing_greeks = option.model_copy(update={
        "position_id": "missing-greeks",
        "delta": None,
        "gamma": None,
        "vega_per_1pct": None,
        "theta_per_day": None,
        "rho_per_1pct": None,
    })
    assert _current_metrics([missing_greeks])["option_delta_exposure_usd"] is None

    later_expiration = option.model_copy(update={
        "position_id": "later-expiration",
        "expiration_at": FIXED_NOW + timedelta(days=60),
    })
    earlier_expiration = option.model_copy(update={
        "position_id": "earlier-expiration",
        "expiration_at": FIXED_NOW + timedelta(days=30),
    })
    payoff = _terminal_payoff([earlier_expiration, later_expiration])
    assert payoff["status"] == "insufficient_data"
    assert payoff["reason"] == "incomplete_option_terminal_payoff_inputs"
