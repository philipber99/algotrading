from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ..pricing.tian_service import (
    IvGreeksContract,
    IvGreeksRequest,
    TianIvGreeksService,
    TianRepriceContract,
    TianRepriceRequest,
)


VALUATION = datetime(2026, 8, 17, 16, tzinfo=timezone.utc)


def contract(**updates):
    values = dict(request_id="one", canonical_occ_symbol="SPY260918C00500000", underlying_symbol="SPY",
                  put_call="CALL", strike=500, expiration_at=VALUATION + timedelta(days=32), spot=510,
                  bid=15, ask=16, risk_free_rate=0.04, dividend_yield=0.01)
    values.update(updates)
    return IvGreeksContract(**values)


def test_real_vectorized_tian_returns_explicit_units_and_method_for_multiple_tenors():
    service = TianIvGreeksService({"risk.options.requested_dtype": "float64"}, now_fn=lambda: VALUATION)
    request = IvGreeksRequest(valuation_at=VALUATION, contracts=[
        contract(), contract(request_id="two", canonical_occ_symbol="SPY270115P00500000", put_call="PUT", expiration_at=VALUATION+timedelta(days=151), bid=20, ask=21),
    ])
    result = service.solve(request)
    assert len(result.data) == 2
    for item in result.data:
        assert item.method == "tian"
        assert item.algorithm_version.startswith("tian_")
        assert item.requested_dtype == item.actual_dtype == "float64"
        assert item.implied_volatility > 0
        assert item.vega_per_1pct is not None and item.theta_per_day is not None


def test_target_selection_missing_rate_expiry_and_cache_fingerprint():
    clock = [0.0]
    service = TianIvGreeksService({"risk.options.cache_ttl_seconds": 60}, now_fn=lambda: VALUATION, clock=lambda: clock[0])
    missing = service.solve(IvGreeksRequest(valuation_at=VALUATION, contracts=[contract(bid=None, ask=None, mark=None, last=None)]))
    assert missing.data[0].reason == "no_usable_market_price"
    expired = service.solve(IvGreeksRequest(valuation_at=VALUATION, contracts=[contract(expiration_at=VALUATION-timedelta(seconds=1))]))
    assert expired.data[0].reason == "expired_or_invalid_tenor"
    no_rate = service.solve(IvGreeksRequest(valuation_at=VALUATION, contracts=[contract(risk_free_rate=None)]))
    assert no_rate.data[0].reason == "risk_free_rate_unavailable"
    first = service.solve(IvGreeksRequest(valuation_at=VALUATION, contracts=[contract()])).data[0]
    cached = service.solve(IvGreeksRequest(valuation_at=VALUATION, contracts=[contract(request_id="cached-caller")])).data[0]
    changed = service.solve(IvGreeksRequest(valuation_at=VALUATION, contracts=[contract(spot=511)])).data[0]
    assert first.model_price == cached.model_price
    assert cached.request_id == "cached-caller"
    assert changed.implied_volatility != pytest.approx(first.implied_volatility)


def test_fixed_iv_tian_repricing_preserves_put_direction_and_advances_time():
    service = TianIvGreeksService({"risk.options.requested_dtype": "float64"}, now_fn=lambda: VALUATION)
    put = contract(
        canonical_occ_symbol="SPY260918P00500000",
        put_call="PUT",
        bid=7,
        ask=8,
    )
    iv = service.solve(IvGreeksRequest(valuation_at=VALUATION, contracts=[put])).data[0].implied_volatility
    assert iv is not None
    common = dict(
        canonical_occ_symbol=put.canonical_occ_symbol,
        underlying_symbol=put.underlying_symbol,
        put_call="PUT",
        strike=put.strike,
        expiration_at=put.expiration_at,
        source_valuation_at=VALUATION,
        source_spot=put.spot,
        source_mark=7.5,
        implied_volatility=iv,
        risk_free_rate=0.04,
        dividend_yield=0.01,
    )
    result = service.reprice(TianRepriceRequest(
        valuation_at=VALUATION + timedelta(days=7),
        contracts=[
            TianRepriceContract(request_id="down", scenario_spot=450, **common),
            TianRepriceContract(request_id="up", scenario_spot=550, **common),
        ],
    ))
    assert [item.status for item in result.data] == ["ok", "ok"]
    assert result.data[0].price > 7.5
    assert result.data[1].price < 7.5
    assert result.data[0].price > result.data[1].price


def test_fixed_iv_tian_repricing_rejects_dates_at_or_after_expiry():
    service = TianIvGreeksService({}, now_fn=lambda: VALUATION)
    result = service.reprice(TianRepriceRequest(
        valuation_at=VALUATION + timedelta(days=32),
        contracts=[TianRepriceContract(
            request_id="expired",
            canonical_occ_symbol="SPY260918P00500000",
            underlying_symbol="SPY",
            put_call="PUT",
            strike=500,
            expiration_at=VALUATION + timedelta(days=32),
            source_valuation_at=VALUATION,
            source_spot=510,
            source_mark=7.5,
            implied_volatility=.2,
            scenario_spot=500,
            risk_free_rate=.04,
        )],
    ))
    assert result.data[0].status == "unable_to_evaluate"
    assert result.data[0].reason == "valuation_at_or_after_expiration"
