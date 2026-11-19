from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest

from algotrading.risk.api.bootstrap import DEFAULT_CONTEXT, _validate_semantic_context
from algotrading.risk.api.hedge_models import HedgeCandidateQueryRequest
from algotrading.risk.api.portfolio_models import PortfolioSnapshotResponse
from algotrading.risk.api.portfolio_service import (
    PortfolioApplicationService,
    _cache_key,
)
from algotrading.risk.api.hedge_service import (
    _bounded_hedge_quantity,
    _future_contract_ladders,
    _objective_future_roots,
)
from algotrading.risk.clusters.hierarchical import cluster_risk_units
from algotrading.risk.exposures.portfolio import portfolio_kpis, portfolio_risk_coverage
from algotrading.risk.instruments.terms import build_instrument_terms
from algotrading.risk.market_inputs.freshness import evaluate_quote_freshness
from algotrading.risk.models import DataQuality, PortfolioSnapshot
from algotrading.risk.payoff.engine import (
    certify_maximum_loss,
    option_intrinsic_per_contract,
)
from algotrading.risk.positions.normalization import normalize_broker_position
from algotrading.risk.pricing.tian_service import (
    IvGreeksContract,
    IvGreeksRequest,
    TianIvGreeksService,
)
from algotrading.risk.scenarios.engine import ScenarioDefinition, ScenarioEngine
from algotrading.risk.scenarios.etf_response import build_etf_history_shocks


UTC = timezone.utc
NOW = datetime(2026, 8, 26, 18, tzinfo=UTC)
CALL = "XYZ260918C00100000"


def _terms(
    *,
    symbol: str = CALL,
    subtype: str = "equity_option",
    style: str = "american",
    settlement: str = "physical",
    components=None,
    aggregate: float = 10_000,
    expiration: datetime | None = None,
    settlement_style: str = "PM",
):
    return build_instrument_terms(
        symbol,
        broker_type="OPTION",
        option_terms={
            "terms_status": "adjusted" if components else "standard",
            "asset_subtype": subtype,
            "price_multiplier": 100,
            "shares_per_contract": 100 if components is None else None,
            "deliverable_components": components or [
                {
                    "component_type": "security",
                    "security_identifier": "XYZ",
                    "quantity": 100,
                }
            ],
            "aggregate_exercise_amount": aggregate,
            "expiration_at": (expiration or datetime(2026, 9, 18, 20, tzinfo=UTC)).isoformat(),
            "exercise_style": style,
            "settlement_type": settlement,
            "settlement_style": settlement_style,
            "exchange_calendar": "XNYS",
            "calendar_version": "fixture-calendar-v1",
            "source": "fixture",
            "source_version": "fixture-v1",
        },
    )


def _option(
    *,
    symbol=CALL,
    quantity=1.0,
    mark=5.0,
    delta=0.5,
    gamma=0.01,
    vega=0.2,
    theta=-0.1,
    rho=0.03,
    terms=None,
    quote_time=None,
    as_of=None,
    dividends=(),
):
    return normalize_broker_position(
        {
            "instrument": {"symbol": symbol, "type": "OPTION"},
            "long_quantity": max(quantity, 0),
            "short_quantity": max(-quantity, 0),
        },
        account_id="account",
        option_quote={
            "mark": mark,
            "underlying_price": 100,
            **({"quote_time": quote_time.isoformat()} if quote_time else {}),
        },
        greek={
            "delta": delta,
            "gamma": gamma,
            "vega_per_1pct": vega,
            "theta_per_day": theta,
            "rho_per_1pct": rho,
            "implied_volatility": 0.25,
            "risk_free_rate_continuous": 0.04,
            "dividend_yield_continuous": 0.0,
            "dividend_inputs": list(dividends),
        },
        instrument_terms=terms or _terms(symbol=symbol),
        as_of=as_of,
    )


def _stock(*, symbol="XYZ", quantity=100, mark=100, quote_time=None, as_of=None):
    return normalize_broker_position(
        {
            "instrument": {"symbol": symbol, "type": "EQUITY"},
            "long_quantity": max(quantity, 0),
            "short_quantity": max(-quantity, 0),
        },
        account_id="account",
        quote={
            "mark": mark,
            **({"quote_time": quote_time.isoformat()} if quote_time else {}),
        },
        as_of=as_of,
    )


def _etf(*, symbol="SCC", quantity=10, mark=30):
    return normalize_broker_position(
        {
            "instrument": {"symbol": symbol, "type": "ETF"},
            "long_quantity": quantity,
        },
        account_id="account",
        quote={"mark": mark},
    )


def _money_market_fund(*, symbol="SWVXX", quantity=10_000, mark=1.0):
    return normalize_broker_position(
        {
            "instrument": {"symbol": symbol, "type": "NO_LOAD_TAXABLE"},
            "long_quantity": quantity,
        },
        account_id="account",
        quote={"mark": mark},
    )


def _snapshot(*positions):
    return PortfolioSnapshot(
        portfolio_snapshot_id="fixture",
        configuration_version="fixture",
        as_of=NOW,
        account_ids=("account",),
        portfolio_value_usd=100_000,
        positions=positions,
        data_quality=DataQuality(status="usable"),
    )


def test_stale_small_option_is_excluded_under_marked_value_limit():
    stale = _option(
        quote_time=NOW - timedelta(days=1),
        as_of=NOW,
    )
    coverage = portfolio_risk_coverage([stale])
    kpis = portfolio_kpis([stale], portfolio_value_usd=100_000)

    assert stale.mark == 5
    assert stale.delta is not None
    assert stale.data_quality.status == "insufficient_data"
    assert coverage.market_value.modeled_positions == 0
    assert coverage.delta.modeled_positions == 0
    assert coverage.gamma.modeled_positions == 0
    assert kpis.net_delta_exposure_usd == 0
    assert kpis.options_vega_per_1pct_usd == 0
    assert any(
        warning.startswith("positions_excluded_below_marked_value_materiality_limit")
        for warning in kpis.warnings
    )


def test_stale_option_above_marked_value_limit_still_fails_closed():
    stale = _option(
        quantity=10,
        quote_time=NOW - timedelta(days=1),
        as_of=NOW,
    )
    kpis = portfolio_kpis([stale], portfolio_value_usd=100_000)

    assert abs(stale.signed_quantity * stale.multiplier * stale.mark) == 5_000
    assert kpis.net_delta_exposure_usd is None
    assert kpis.options_vega_per_1pct_usd is None


def test_tiny_premium_unmodeled_short_call_makes_delta_denominator_unknown():
    stock = _stock(quantity=100, mark=100)
    call = _option(quantity=-1, mark=0.01, delta=None, gamma=None)
    coverage = portfolio_risk_coverage([stock, call])

    assert stock.signed_market_value_usd == 10_000
    assert abs(call.signed_market_value_usd) == 1
    assert coverage.delta.count_percent == 50
    assert coverage.delta.percent is None
    assert call.position_id in coverage.delta.missing_position_ids
    assert call.position_id in coverage.delta.unavailable_denominator_position_ids


def test_missing_gamma_does_not_change_complete_vega_coverage():
    complete = _option(symbol="XYZ260918C00090000")
    missing_gamma = _option(symbol="XYZ260918P00110000", gamma=None)
    coverage = portfolio_risk_coverage([complete, missing_gamma])
    assert coverage.gamma.count_percent == 50
    assert coverage.gamma.percent is None
    assert coverage.vega.count_percent == 100


def test_scenario_lists_stale_required_position_as_unmodeled():
    stale = _stock(quote_time=NOW - timedelta(days=1), as_of=NOW)
    result = ScenarioEngine({}).evaluate(
        _snapshot(stale),
        ScenarioDefinition(scenario_id="down", name="Down", broad_equity_shock=-0.05),
    )
    assert result["quality_state"] == "insufficient_data"
    assert result["contributions"] == []
    assert result["unmodeled_position_ids"] == [stale.position_id]


def test_hard_stale_snapshot_preserves_nested_risk_coverage():
    snapshot_at = NOW - timedelta(seconds=301)
    response = PortfolioSnapshotResponse.model_validate(
        {
            "portfolio_snapshot_id": "snapshot",
            "configuration_version": "configuration",
            "as_of": snapshot_at,
            "display_timezone": "America/Los_Angeles",
            "data_quality": {
                "status": "usable_with_warnings",
                "risk_coverage": {
                    "delta": {
                        "eligible_positions": 2,
                        "modeled_positions": 1,
                        "count_percent": 50,
                        "weight_basis": "absolute_delta_exposure_usd",
                        "missing_position_ids": ["missing-option"],
                    }
                },
            },
            "data": {
                "account_ids": ["account"],
                "positions": [],
                "kpis": {
                    "portfolio_value_usd": None,
                    "gross_delta_exposure_usd": None,
                    "net_delta_exposure_usd": None,
                    "gross_market_value_usd": None,
                    "leverage": None,
                    "options_vega_per_1pct_usd": None,
                    "options_theta_per_day_usd": None,
                    "margin_utilization": None,
                    "modeled_market_value_percent": None,
                    "modeled_gross_delta_percent": None,
                },
                "balances": [],
                "capabilities": {},
            },
            "cache": {
                "state": "miss",
                "age_seconds": 0,
                "source_as_of": snapshot_at,
            },
        }
    )

    class EmptyCache:
        @staticmethod
        def get(_key):
            return None

    service = object.__new__(PortfolioApplicationService)
    service.cache = EmptyCache()
    service._latest_by_scope = {"account": response}
    cached = service._load_cached(
        _cache_key(("account",), "configuration"), NOW
    )

    assert cached is not None
    assert cached.cache.state == "hard_stale"
    assert cached.data_quality.status == "insufficient_data"
    assert cached.data_quality.risk_coverage == response.data_quality.risk_coverage
    assert "portfolio_snapshot_hard_stale" in cached.data_quality.warnings


def test_mixed_adjusted_deliverable_separates_premium_and_exercise_economics():
    terms = _terms(
        components=[
            {"component_type": "security", "security_identifier": "XYZ", "quantity": 50},
            {"component_type": "cash", "cash_amount": 500, "currency": "USD"},
        ],
        aggregate=5_000,
    )
    option = _option(terms=terms, mark=5)
    assert option.signed_market_value_usd == 500
    assert option_intrinsic_per_contract(option, 120) == 1_500
    assert option_intrinsic_per_contract(option, 120) != 100 * (120 - 100)


def test_cash_settled_option_uses_intrinsic_value_times_contract_multiplier():
    terms = _terms(
        settlement="cash",
        components=[],
        aggregate=10_000,
    )
    option = _option(terms=terms, mark=5)

    assert terms.deliverable is not None
    assert terms.deliverable.fully_supported
    assert terms.deliverable.components == ()
    assert option_intrinsic_per_contract(option, 120) == 2_000
    assert certify_maximum_loss([option])["status"] == "exact_bounded"


def test_unsupported_deliverable_fails_payoff_and_expiry_scenario_closed():
    terms = _terms(
        components=[{"component_type": "other", "description": "warrant rights"}],
        aggregate=5_000,
        expiration=NOW + timedelta(days=1),
    )
    option = _option(terms=terms)
    payoff = certify_maximum_loss([option])
    scenario = ScenarioEngine({}).evaluate(
        _snapshot(option),
        ScenarioDefinition(
            scenario_id="expiry", name="Expiry", broad_equity_shock=0.2,
            horizon_calendar_days=2,
        ),
    )
    assert payoff["status"] == "insufficient_data"
    assert scenario["quality_state"] == "usable_with_warnings"
    assert scenario["excluded_position_ids"] == [option.position_id]


def test_european_option_uses_quantlib_analytic_fallback():
    service = TianIvGreeksService(
        {"risk.rates.configured_flat_continuous_rate": 0.04}, now_fn=lambda: NOW
    )
    response = service.solve(
        IvGreeksRequest(
            valuation_at=NOW,
            contracts=[
                IvGreeksContract(
                    request_id="european",
                    canonical_occ_symbol=CALL,
                    underlying_symbol="XYZ",
                    put_call="CALL",
                    exercise_type="EUROPEAN",
                    strike=100,
                    expiration_at=NOW + timedelta(days=30),
                    spot=100,
                    mark=5,
                    risk_free_rate=0.04,
                    dividend_yield=0,
                )
            ],
        )
    )
    item = response.data[0]
    assert item.status == "ok"
    assert item.method == "quantlib_analytic_european"
    assert item.reason == "complete"
    assert item.implied_volatility is not None
    assert item.delta is not None


def test_unknown_economic_vix_european_option_still_models_iv_scenario():
    symbol = "VIX261021C00015500"
    terms = _terms(symbol=symbol, style="european", subtype="index_option").model_copy(
        update={"economic_exposure_type": "unknown", "underlying_identifier": "VIX"}
    )
    option = _option(symbol=symbol, terms=terms)

    result = ScenarioEngine({}).evaluate(
        _snapshot(option),
        ScenarioDefinition(
            scenario_id="market-iv",
            name="Market and IV",
            default_equity_shock=-0.03,
            iv_points=3,
        ),
    )

    assert result["unmodeled_position_ids"] == []
    assert result["contributions"][0]["calculation_method"] == (
        "quantlib_analytic_european_change"
    )

    sector_only = ScenarioEngine({}).evaluate(
        _snapshot(option),
        ScenarioDefinition(
            scenario_id="sector-only",
            name="Unrelated sector shock",
            sector_shocks={"Technology": -0.08},
        ),
    )
    assert sector_only["unmodeled_position_ids"] == []
    assert sector_only["contributions"][0]["pnl_usd"] == pytest.approx(0.0)


def test_am_and_pm_authoritative_utc_timestamps_survive_dst_offsets():
    winter_am = _terms(
        symbol="XYZ260116C00100000",
        expiration=datetime(2026, 1, 16, 14, 30, tzinfo=UTC),
        settlement_style="AM",
    )
    summer_pm = _terms(
        expiration=datetime(2026, 9, 18, 20, 0, tzinfo=UTC),
        settlement_style="PM",
    )
    assert winter_am.expiration_at.isoformat() == "2026-01-16T14:30:00+00:00"
    assert summer_pm.expiration_at.isoformat() == "2026-09-18T20:00:00+00:00"
    assert winter_am.settlement_style == "AM"
    assert summer_pm.settlement_style == "PM"


def test_dividend_american_call_uses_optional_escrowed_dividend_mode():
    dividend = {"ex_dividend_date": "2026-09-01", "amount": 1.0}
    option = _option(dividends=[dividend])
    result = ScenarioEngine({}).evaluate(
        _snapshot(option),
        ScenarioDefinition(scenario_id="move", name="Move", broad_equity_shock=0.02),
    )
    assert result["quality_state"] == "usable"
    assert result["unmodeled_position_ids"] == []
    contribution = result["contributions"][0]
    assert "escrowed_dividend" in contribution["calculation_method"]
    assert contribution["solver_diagnostics"]["escrowed_dividend"]["event_count"] == 1


def test_scenario_engine_reuses_identical_tian_reprice_states():
    option = _option()
    snapshot = _snapshot(option)
    scenario = ScenarioDefinition(
        scenario_id="cached-move",
        name="Cached move",
        broad_equity_shock=-0.03,
        iv_points=3,
    )
    engine = ScenarioEngine({})

    first = engine.evaluate(snapshot, scenario)
    cached_states = len(engine._reprice_cache)
    second = engine.evaluate(snapshot, scenario)

    assert cached_states == 2
    assert len(engine._reprice_cache) == cached_states
    assert second["scenario_pnl_usd"] == first["scenario_pnl_usd"]


def test_dividend_american_call_fails_closed_when_escrowed_mode_disabled():
    dividend = {"ex_dividend_date": "2026-09-01", "amount": 1.0}
    option = _option(dividends=[dividend])
    result = ScenarioEngine(
        {"risk.options.tian.escrowed_dividend_mode_enabled": False}
    ).evaluate(
        _snapshot(option),
        ScenarioDefinition(scenario_id="move", name="Move", broad_equity_shock=0.02),
    )
    assert result["quality_state"] == "usable_with_warnings"
    assert result["excluded_position_ids"] == [option.position_id]


def test_inverse_etf_history_response_uses_market_and_sector_proxies():
    days = [str(index) for index in range(180)]
    returns = {
        "SPY": {day: math.sin(index * 0.31) * 0.01 for index, day in enumerate(days)},
        "QQQ": {day: math.cos(index * 0.17) * 0.012 for index, day in enumerate(days)},
        "XLY": {
            day: math.sin(index * 0.07 + 0.4) * 0.009
            for index, day in enumerate(days)
        },
    }
    returns["SCC"] = {day: -2 * returns["XLY"][day] for day in days}

    shocks, diagnostics = build_etf_history_shocks(
        [_etf()],
        returns,
        broad_equity_shock=0,
        sector_shocks={"Consumer Cyclical": -0.03},
    )

    assert shocks["SCC"] == pytest.approx(0.06, abs=0.001)
    assert diagnostics["positions"][0]["r_squared"] > 0.99
    assert diagnostics["unmodeled_symbols"] == []


def test_expiry_path_assigns_shares_and_carries_post_expiry_pnl():
    expiry = NOW + timedelta(days=1)
    option = _option(terms=_terms(expiration=expiry))
    result = ScenarioEngine({}).evaluate(
        _snapshot(option),
        ScenarioDefinition(
            scenario_id="expiry-path",
            name="Expiry path",
            exact_underlying_shocks={"XYZ": 0.20},
            horizon_calendar_days=2,
            path_rule="explicit_dated_points",
            dated_path_points={expiry: 110},
        ),
    )
    contribution = result["contributions"][0]
    lifecycle = contribution["lifecycle_event"]
    assert contribution["pnl_usd"] == pytest.approx(1_500)
    assert lifecycle["resulting_underlying_quantity"] == 100
    assert lifecycle["expiry_to_horizon_pnl_usd"] == pytest.approx(1_000)


def test_expiry_crossing_without_path_is_unmodeled():
    expiry = NOW + timedelta(days=1)
    option = _option(terms=_terms(expiration=expiry))
    result = ScenarioEngine({}).evaluate(
        _snapshot(option),
        ScenarioDefinition(
            scenario_id="no-path", name="No path", broad_equity_shock=0.2,
            horizon_calendar_days=2, path_rule=None,
        ),
    )
    assert result["unmodeled_position_ids"] == [option.position_id]


@pytest.mark.parametrize(
    ("symbol", "mapping", "scenario", "expected"),
    [
        ("TLT", {"economic_exposure_type": "rates", "effective_duration": 10}, {"rate_basis_points": 100}, -1_000),
        ("GLD", {"economic_exposure_type": "commodity", "factor": "GC"}, {"commodity_root_shocks": {"GC": -0.05}}, -500),
        ("FXE", {"economic_exposure_type": "currency", "factor": "EURUSD"}, {"fx_pair_shocks": {"EURUSD": 0.02}}, 200),
    ],
)
def test_etf_economic_mappings_use_only_their_typed_channel(symbol, mapping, scenario, expected):
    position = normalize_broker_position(
        {"instrument": {"symbol": symbol, "type": "ETF"}, "long_quantity": 100},
        account_id="account",
        quote={"mark": 100},
    )
    context = {"risk.scenarios.economic_instrument_mappings": {symbol: mapping}}
    result = ScenarioEngine(context).evaluate(
        _snapshot(position),
        ScenarioDefinition(scenario_id="mapped", name="Mapped", **scenario),
    )
    assert result["scenario_pnl_usd"] == pytest.approx(expected)
    assert not result["unmodeled_position_ids"]


def test_bond_etf_does_not_receive_broad_equity_shock():
    bond = normalize_broker_position(
        {"instrument": {"symbol": "TLT", "type": "ETF"}, "long_quantity": 100},
        account_id="account", quote={"mark": 100},
    )
    result = ScenarioEngine(
        {"risk.scenarios.economic_instrument_mappings": {"TLT": {"economic_exposure_type": "rates", "effective_duration": 10}}}
    ).evaluate(
        _snapshot(bond),
        ScenarioDefinition(scenario_id="equity", name="Equity", broad_equity_shock=-0.1),
    )
    assert result["scenario_pnl_usd"] is None
    assert result["unmodeled_position_ids"] == [bond.position_id]


def test_scenario_contributions_reconcile_and_partition_every_position():
    fresh = _stock()
    stale = _stock(symbol="ABC", quote_time=NOW - timedelta(days=1), as_of=NOW)
    result = ScenarioEngine({}).evaluate(
        _snapshot(fresh, stale),
        ScenarioDefinition(scenario_id="down", name="Down", broad_equity_shock=-0.1),
    )
    assert result["scenario_pnl_usd"] == pytest.approx(
        sum(item["pnl_usd"] for item in result["contributions"]), abs=0.01
    )
    covered = {item["position_id"] for item in result["contributions"]} | set(result["unmodeled_position_ids"])
    assert covered == {fresh.position_id, stale.position_id}


def test_scenario_treats_money_market_funds_as_cash_equivalents():
    fund = _money_market_fund()
    result = ScenarioEngine({}).evaluate(
        _snapshot(fund),
        ScenarioDefinition(scenario_id="down", name="Down", broad_equity_shock=-0.1),
    )

    assert result["scenario_pnl_usd"] == 0.0
    assert result["coverage"]["eligible_positions"] == 0
    assert result["unmodeled_position_ids"] == []
    assert result["contributions"][0]["calculation_method"] == (
        "cash_equivalent_no_direct_channel"
    )


@pytest.mark.parametrize(
    ("target", "unit", "maximum", "overhedge", "expected"),
    [(1_000, 260, 10, 0.25, 4.0), (100, 130, 10, 0.25, None), (10_000, 100, 3, 0.25, 3.0)],
)
def test_candidate_quantity_respects_integer_limit_and_overhedge(target, unit, maximum, overhedge, expected):
    assert _bounded_hedge_quantity(
        target, unit, maximum_quantity=maximum,
        maximum_overhedge_fraction=overhedge,
    ) == expected


def _factor_request(factor):
    return HedgeCandidateQueryRequest.model_validate(
        {
            "portfolio_snapshot_id": "snapshot",
            "target": {"kind": "Portfolio", "label": "Portfolio", "position_ids": ["p"]},
            "objective": "Factors",
            "objective_parameters": {"factor": factor, "target_exposure": 0},
        }
    )


def test_future_objective_mapping_excludes_unrelated_contracts_and_includes_micros():
    equity_roots = _objective_future_roots(
        HedgeCandidateQueryRequest.model_validate(
            {
                "portfolio_snapshot_id": "snapshot",
                "target": {"kind": "Portfolio", "label": "Portfolio", "position_ids": ["p"]},
                "objective": "Scenario Loss",
                "objective_parameters": {},
            }
        )
    )
    assert {"MES", "MNQ", "M2K"} <= equity_roots
    assert not ({"CL", "ZN", "6E"} & equity_roots)
    assert _objective_future_roots(_factor_request("Gold")) == {"GC", "MGC"}
    assert {"ES", "MES", "NQ", "MNQ", "RTY", "M2K"} <= set(_future_contract_ladders(NOW.date()))


def test_cluster_zero_history_weight_is_scenario_only_and_low_overlap_is_not_used():
    result = cluster_risk_units(
        {"a": [-100, -50, 10], "b": [-80, -40, 20]},
        exposures_usd={"a": 100_000, "b": 100_000},
        minimum_correlation=0.5,
        minimum_cluster_loss_usd=1,
        historical_pnl_by_unit={"a": {"d1": -1}, "b": {"d1": -1}},
        source_weights={"scenario": 1, "historical": 0, "factor": 0},
        minimum_pairwise_overlap=2,
    )
    assert result
    assert result[0]["correlation_source_mix"] == {
        "scenario": 1.0, "historical": 0.0, "factor": 0.0
    }
    assert result[0]["pairwise_source_coverage"][0]["historical_overlap"] == 1
    assert "historical" not in result[0]["pairwise_source_coverage"][0]["source_weights_used"]


def test_holiday_early_close_and_futures_sessions_use_versioned_calendars():
    holiday = evaluate_quote_freshness(
        datetime(2026, 11, 25, 21, tzinfo=UTC),
        as_of=datetime(2026, 11, 26, 18, tzinfo=UTC),
        context={}, source_kind="daily_close", calendar_id="XNYS",
    )
    early = evaluate_quote_freshness(
        datetime(2026, 11, 27, 18, tzinfo=UTC),
        as_of=datetime(2026, 11, 27, 20, tzinfo=UTC),
        context={}, source_kind="daily_close", calendar_id="XNYS",
    )
    future = evaluate_quote_freshness(
        datetime(2026, 8, 25, 23, 5, tzinfo=UTC),
        as_of=datetime(2026, 8, 25, 23, 10, tzinfo=UTC),
        context={}, calendar_id="CME_GLOBEX", exchange_timezone="America/Chicago",
    )
    assert holiday.valid_for_risk
    assert holiday.session_date.isoformat() == "2026-11-25"
    assert early.valid_for_risk
    assert early.session_close_at.hour == 13
    assert future.valid_for_risk
    assert future.market_session_state == "open"
    assert future.calendar_version.startswith("refdata_exchange_calendars")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("risk.hedge_lab.maximum_candidate_quantity", 0),
        ("risk.hedge_lab.maximum_overhedge_fraction", 1.1),
        ("risk.option_history.minimum_contract_iv_observations", 1),
        ("risk.hedge_lab.allowed_futures_roots", ["NOT_A_ROOT"]),
        ("risk.clusters.source_weights", {"scenario": 0, "history": 0, "factor": 0}),
    ],
)
def test_semantic_configuration_rejects_invalid_consumed_controls(key, value):
    context = dict(DEFAULT_CONTEXT)
    context[key] = value
    with pytest.raises(ValueError):
        _validate_semantic_context(context)
