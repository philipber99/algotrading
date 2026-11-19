from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from algotrading.risk.history.planner import HistoryWindowPlanner
from algotrading.risk.history.revaluation import (
    CurrentPortfolioHistoricalRevaluationEngine,
)
from algotrading.risk.api.hedge_service import HedgeLabService
from algotrading.risk.api.routes import _tail_analysis
from algotrading.risk.instruments.terms import build_instrument_terms
from algotrading.risk.option_prices.service import RiskOptionHistoryService
from algotrading.risk.market_inputs.client import ReferenceDataClientError
from algotrading.risk.positions.normalization import normalize_broker_position
from algotrading.risk.pricing.tian_service import IvGreeksContract, IvGreeksRequest
from algotrading.risk.tail.historical import historical_var_es
from algotrading.refdata.api.option_symbols import parse_occ_symbol


UTC = timezone.utc
AS_OF = datetime(2026, 8, 27, 16, tzinfo=UTC)
FIXTURE_PATH = Path(__file__).with_name("fixtures") / "var_tail_87_position_fixture.json"


class _SolveItem:
    status = "ok"
    reason = "complete"
    implied_volatility = 0.25
    delta = 0.5
    gamma = 0.01
    theta_per_day = -0.02
    vega_per_1pct = 0.08
    rho_per_1pct = 0.02

    def model_dump(self, **_kwargs):
        return {
            "status": self.status,
            "reason": self.reason,
            "implied_volatility": self.implied_volatility,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta_per_day": self.theta_per_day,
            "vega_per_1pct": self.vega_per_1pct,
            "rho_per_1pct": self.rho_per_1pct,
        }


class _RepriceItem:
    def __init__(self, price: float):
        self.status = "ok"
        self.price = price
        self.solver_diagnostics = {"fixture": "deterministic_full_reprice"}


class _TianFixture:
    def solve(self, request):
        return SimpleNamespace(data=[_SolveItem() for _ in request.contracts])

    def reprice(self, request):
        rows = []
        for contract in request.contracts:
            spot_change = contract.scenario_spot - contract.source_spot
            quote_change = 0.50 * spot_change + 0.05 * contract.iv_shift_points
            rows.append(_RepriceItem(max(0.01, float(contract.source_mark) + quote_change)))
        return SimpleNamespace(data=rows)


class _OptionRepository:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []

    def query_history(self, symbols, *, start_date, end_date):
        wanted = set(symbols)
        self.calls.append((tuple(symbols), start_date, end_date))
        return [
            dict(row)
            for row in self.rows
            if row["canonical_occ_symbol"] in wanted
            and start_date <= date.fromisoformat(row["session_date"]) <= end_date
        ]


class _ReferenceFixture:
    def __init__(self, sessions, option_rows, *, price_fn=None):
        self.sessions = tuple(sessions)
        self.option_rows = list(option_rows)
        self.price_fn = price_fn or _daily_price
        self.context = {"risk.option_prices.max_history_days": 376}
        self.option_calls = []
        self.bar_calls = []

    def option_history_maximum_days(self):
        return 376

    def option_market_history(self, payload):
        self.option_calls.append(dict(payload))
        # The fixture deliberately reproduces delisted/unavailable contracts:
        # remote history cannot supply the SQL gaps.
        return {"data": [], "meta": {"warnings": ["fixture_remote_marks_unavailable"]}}

    def bars(self, symbols, *, start, end, remote_query=False):
        self.bar_calls.append(
            {
                "symbols": tuple(symbols),
                "start": start,
                "end": end,
                "remote_query": remote_query,
            }
        )
        output = []
        for symbol in symbols:
            if symbol == "SCC" and not remote_query:
                rows = []
            else:
                rows = [
                    {
                        "session_date": session.isoformat(),
                        "timestamp": f"{session.isoformat()}T20:00:00Z",
                        "close": self.price_fn(index),
                    }
                    for index, session in enumerate(self.sessions)
                    if start <= session <= end
                ]
            output.append({"symbol": symbol, "bars": rows})
        return output

    def dividend_calendar(self, **_kwargs):
        return []

    def risk_free_curve(self, _session):
        raise ValueError("fixture uses configured flat rate")


def _option_terms(symbol: str):
    contract = parse_occ_symbol(symbol)
    root = contract.root
    return build_instrument_terms(
        symbol,
        broker_type="OPTION",
        option_terms={
            "terms_status": "standard",
            "asset_subtype": "equity_option",
            "price_multiplier": 100,
            "shares_per_contract": 100,
            "deliverable_components": [
                {
                    "component_type": "security",
                    "security_identifier": root,
                    "quantity": 100,
                }
            ],
            "aggregate_exercise_amount": float(contract.strike) * 100,
            "expiration_at": "2026-12-18T21:00:00Z",
            "exercise_style": "american",
            "settlement_type": "physical",
            "source": "deterministic_fixture",
            "source_version": "reported-87-v1",
        },
    )


def _option_symbol(root: str, put_call: str = "C", strike: int = 100) -> str:
    return f"{root}261218{put_call}{strike * 1000:08d}"


def _option_history_row(symbol: str, session: date, session_index: int):
    available = datetime.combine(
        session + timedelta(days=1), datetime.min.time(), UTC
    )
    return {
        "requested_symbol": symbol,
        "canonical_occ_symbol": symbol,
        "session_date": session.isoformat(),
        "at": available.isoformat(),
        "observed_at": available.isoformat(),
        "status": "ok",
        "mark": 4.0 + 0.01 * session_index,
        "mark_basis": "daily_close",
        "underlying_price": _daily_price(session_index),
        "source": "risk_mysql_option_prices",
        "warnings": [],
    }


def _daily_price(session_index: int) -> float:
    return 100.0 + 0.8 * ((7 * session_index) % 19 - 9) + 0.01 * session_index


@pytest.fixture(scope="module")
def reported_portfolio_fixture():
    manifest = json.loads(FIXTURE_PATH.read_text())
    plan = HistoryWindowPlanner().plan(
        as_of=AS_OF,
        requested_return_count=manifest["requested_returns"],
        calendar_id="XNYS",
        history_kind="exact_option_mark",
        advertised_maximum_days=376,
    )
    sessions = plan.required_sessions
    equities = [
        normalize_broker_position(
            {
                "instrument": {"symbol": f"EQ{index:02d}A", "type": "EQUITY"},
                "long_quantity": 10 + index,
            },
            account_id="reported-portfolio",
            quote={"mark": 100.0 + index},
        )
        for index in range(31)
    ]
    exact_option_symbols = [
        _option_symbol(f"O{index:02d}") for index in range(45)
    ] + [
        _option_symbol("UNH", "C", 500),
        _option_symbol("UNH", "P", 500),
    ]
    missing_option_symbols = [_option_symbol(f"M{index:02d}") for index in range(8)]
    options = [
        normalize_broker_position(
            {
                "instrument": {"symbol": symbol, "type": "OPTION"},
                "long_quantity": 1,
            },
            account_id="reported-portfolio",
            option_quote={"mark": 5.0, "underlying_price": 100.0},
            instrument_terms=_option_terms(symbol),
        )
        for symbol in [*exact_option_symbols, *missing_option_symbols]
    ]
    scc = normalize_broker_position(
        {
            "instrument": {"symbol": "SCC", "type": "ETF"},
            "long_quantity": 50,
        },
        account_id="reported-portfolio",
        quote={"mark": 100.0},
    )
    cash_equivalent = normalize_broker_position(
        {
            "instrument": {"symbol": "SWVXX", "type": "NO_LOAD_TAXABLE"},
            "long_quantity": 10_000,
        },
        account_id="reported-portfolio",
        quote={"mark": 1.0},
    )
    rows = []
    for option_index, symbol in enumerate(exact_option_symbols[:-2]):
        observation_count = 3 + option_index % 37
        for session in sessions[-observation_count:]:
            rows.append(_option_history_row(symbol, session, sessions.index(session)))
    for symbol in exact_option_symbols[-2:]:
        for session in sessions[-2:]:
            rows.append(_option_history_row(symbol, session, sessions.index(session)))

    reference = _ReferenceFixture(sessions, rows)
    repository = _OptionRepository(rows)
    option_history = RiskOptionHistoryService(
        repository, reference, now_fn=lambda: AS_OF
    )
    shocks = {
        session.isoformat(): 0.04 * ((index % 9) - 4)
        for index, session in enumerate(sessions[1:])
    }
    configured_factor = {
        "bucket": "all",
        "source_kind": "daily_option_price_derived",
        "calibration_observations": 252,
        "calibration_error": 0.04,
        "economic_relationship": "same exact underlying",
        "shocks": shocks,
    }
    roots = {symbol[:-15] for symbol in [*exact_option_symbols, *missing_option_symbols]}
    context = {
        "risk.history.lookback_sessions": 252,
        "risk.history.minimum_scenarios": 126,
        "risk.tail_risk.minimum_gross_delta_coverage": 0.90,
        "risk.option_prices.max_history_days": 376,
        "risk.history.daily_bar_maximum_days": 376,
        "risk.pnl.remote_daily_bar_symbols": ("SCC",),
        "risk.rates.configured_flat_continuous_rate": 0.04,
        "risk.dividends.default_continuous_yield": 0.0,
        "risk.option_history.minimum_contract_iv_observations": 126,
        "risk.history.minimum_option_volatility_factor_overlap": 126,
        "risk.history.maximum_option_volatility_factor_error": 0.10,
        "risk.history.option_volatility_factors": {
            root: configured_factor for root in roots
        },
    }
    engine = CurrentPortfolioHistoricalRevaluationEngine(
        reference,
        context,
        risk_history_service=option_history,
    )
    engine.option_resolver.tian_service = _TianFixture()
    positions = [*equities, *options, scc]
    broker_positions = [*positions, cash_equivalent]
    return {
        "manifest": manifest,
        "plan": plan,
        "positions": positions,
        "equities": equities,
        "exact_options": options[:47],
        "missing_options": options[47:],
        "scc": scc,
        "cash_equivalent": cash_equivalent,
        "broker_positions": broker_positions,
        "reference": reference,
        "repository": repository,
        "option_rows": rows,
        "result": engine.revalue(
            broker_positions, as_of=AS_OF, lookback_sessions=252
        ),
    }


def _intersection(series_by_id, ids):
    values = [set(series_by_id[position_id]) for position_id in ids]
    return sorted(set.intersection(*values)) if values else []


def _matrix(series_by_id, ids, dates):
    return [[series_by_id[position_id][day] for position_id in ids] for day in dates]


def test_reported_87_position_fixture_reproduces_old_failure_and_full_remediation(
    reported_portfolio_fixture,
):
    fixture = reported_portfolio_fixture
    manifest = fixture["manifest"]
    result = fixture["result"]
    positions = fixture["positions"]
    exact_ids = [
        position.position_id
        for position in [*fixture["equities"], *fixture["exact_options"]]
    ]
    old_dates = _intersection(result.unit_pnl, exact_ids)
    all_ids = [position.position_id for position in positions]
    new_dates = _intersection(result.scenario_pnl, all_ids)

    assert len(positions) == manifest["eligible_position_count"] == 87
    assert len(fixture["broker_positions"]) == 88
    assert fixture["cash_equivalent"].exact_symbol == manifest["excluded_cash_symbol"]
    assert fixture["cash_equivalent"].position_id not in {
        row.position_id for row in result.position_coverage
    }
    assert len(exact_ids) == manifest["reported_pre_remediation"]["modeled_positions"] == 78
    assert 100 * len(exact_ids) / len(positions) == pytest.approx(
        manifest["reported_pre_remediation"]["coverage_percent"]
    )
    assert old_dates == [fixture["plan"].last_required_session.isoformat()]
    assert len(old_dates) == manifest["reported_pre_remediation"]["complete_intersection_dates"]
    assert len(new_dates) == 252
    assert result.missing_position_ids == ()
    assert set(result.scenario_pnl) == set(all_ids)
    assert all(len(result.scenario_pnl[position_id]) == 252 for position_id in all_ids)
    assert all(
        set(result.scenario_pnl[position_id]) == set(new_dates)
        for position_id in all_ids
    )
    assert len(result.position_coverage) == 87
    assert all(row.current_state_scenario_count == 252 for row in result.position_coverage)

    after = historical_var_es(
        _matrix(result.scenario_pnl, all_ids, new_dates),
        confidence=0.95,
        unit_ids=all_ids,
        dates=new_dates,
    )
    assert after["observations"] == 252
    assert after["var_usd"] >= 0
    assert after["expected_shortfall_usd"] >= after["var_usd"]


def test_fixture_remote_requests_are_bounded_and_only_cover_sql_missing_sessions(
    reported_portfolio_fixture,
):
    fixture = reported_portfolio_fixture
    reference = fixture["reference"]
    sql_keys = {
        (row["canonical_occ_symbol"], row["session_date"])
        for row in fixture["option_rows"]
    }
    assert reference.option_calls
    for call in reference.option_calls:
        start = datetime.fromisoformat(call["start_at"])
        end = datetime.fromisoformat(call["end_at"])
        assert end - start <= timedelta(days=376)
        assert end > start
        for symbol in call["symbols"]:
            requested_sessions = {
                session.isoformat()
                for session in fixture["plan"].required_sessions
                if start.date() <= session < end.date()
            }
            assert requested_sessions
            assert not any((symbol, session) in sql_keys for session in requested_sessions)

    scc_remote = [
        call
        for call in reference.bar_calls
        if call["symbols"] == ("SCC",) and call["remote_query"] is True
    ]
    assert len(scc_remote) == 1
    assert scc_remote[0]["start"] == fixture["plan"].first_required_session
    assert scc_remote[0]["end"] == fixture["plan"].last_required_session


def test_fixture_diagnostics_name_two_unh_one_interval_contracts_and_eight_missing_mark_contracts(
    reported_portfolio_fixture,
):
    fixture = reported_portfolio_fixture
    coverage = {row.exact_symbol: row for row in fixture["result"].position_coverage}
    unh = fixture["manifest"]["reported_pre_remediation"]["unh_one_interval_contracts"]
    missing_symbols = [position.exact_symbol for position in fixture["missing_options"]]

    assert [coverage[symbol].exact_interval_count for symbol in unh] == [1, 1]
    assert all(coverage[symbol].sql_mark_count == 0 for symbol in missing_symbols)
    assert all(coverage[symbol].remote_mark_count == 0 for symbol in missing_symbols)
    assert all(coverage[symbol].current_state_scenario_count == 252 for symbol in missing_symbols)
    assert all(coverage[symbol].proxy_usage_count == 252 for symbol in missing_symbols)
    mappings = {
        row["source_symbol"]: row for row in fixture["result"].proxy_mappings
    }
    assert all(mappings[symbol]["proxy_kind"] == "same_underlying_option_volatility_factor" for symbol in missing_symbols)
    assert all(mappings[symbol]["calibration_observations"] == 252 for symbol in missing_symbols)
    assert all(mappings[symbol]["calibration_error"] == pytest.approx(0.04) for symbol in missing_symbols)


def test_missing_contract_uses_calibrated_same_underlying_iv_factor_not_copied_marks():
    plan = HistoryWindowPlanner().plan(
        as_of=AS_OF,
        requested_return_count=10,
        calendar_id="XNYS",
        history_kind="exact_option_mark",
    )
    source_symbols = [
        _option_symbol("ALT", "C", 100),
        _option_symbol("ALT", "P", 100),
    ]
    target_symbol = _option_symbol("ALT", "C", 102)
    rows = [
        _option_history_row(symbol, session, index)
        for symbol in source_symbols
        for index, session in enumerate(plan.required_sessions)
    ]
    for row in rows:
        row["underlying_price"] = 100.0 + 0.1 * plan.required_sessions.index(
            date.fromisoformat(row["session_date"])
        )
    reference = _ReferenceFixture(
        plan.required_sessions, rows, price_fn=lambda index: 100.0 + 0.1 * index
    )
    history_service = RiskOptionHistoryService(
        _OptionRepository(rows), reference, now_fn=lambda: AS_OF
    )
    positions = [
        normalize_broker_position(
            {
                "instrument": {"symbol": symbol, "type": "OPTION"},
                "long_quantity": 1,
            },
            account_id="same-underlying-factor",
            option_quote={"mark": 5.0, "underlying_price": 100.0},
            instrument_terms=_option_terms(symbol),
        )
        for symbol in [*source_symbols, target_symbol]
    ]
    engine = CurrentPortfolioHistoricalRevaluationEngine(
        reference,
        {
            "risk.rates.configured_flat_continuous_rate": 0.04,
            "risk.dividends.default_continuous_yield": 0.0,
            "risk.history.minimum_same_underlying_volatility_contracts": 2,
        },
        risk_history_service=history_service,
    )
    engine.option_resolver.tian_service = _TianFixture()

    result = engine.revalue(positions, as_of=AS_OF, lookback_sessions=10)
    target = positions[-1]
    methods = {
        row["volatility_shock"]["method"]
        for row in result.scenario_provenance[target.position_id].values()
    }

    assert len(result.scenario_pnl[target.position_id]) == 10
    assert methods == {"same_underlying_bucket_iv_factor:91_180d:atm"}
    assert target.position_id not in result.unit_pnl
    assert all(
        row["volatility_shock"].get("source_kind") == "daily_option_price_derived"
        for row in result.scenario_provenance[target.position_id].values()
    )
    assert all(
        row["volatility_shock"].get("calibration_quality")
        == "passed_cross_contract_dispersion_gate"
        for row in result.scenario_provenance[target.position_id].values()
    )


def test_portfolio_tail_and_hedge_tail_share_failures_coverage_and_structured_422():
    plan = HistoryWindowPlanner().plan(
        as_of=AS_OF,
        requested_return_count=10,
        calendar_id="XNYS",
        history_kind="exact_option_mark",
    )

    class RejectingReference(_ReferenceFixture):
        def option_market_history(self, payload):
            self.option_calls.append(dict(payload))
            raise ReferenceDataClientError(
                "fixture rejected",
                http_status=422,
                server_error_code="INVALID_REQUEST",
                server_message="option history interval rejected",
                server_details={"maximum_days": 376, "exclusive_end": True},
                request_id="tail-structured-422",
                endpoint="/api/v1/options/market-history/query",
                requested_start=payload["start_at"],
                requested_end=payload["end_at"],
            )

    symbol = _option_symbol("ERR", "P", 100)
    position = normalize_broker_position(
        {
            "instrument": {"symbol": symbol, "type": "OPTION"},
            "long_quantity": 1,
        },
        account_id="structured-tail-failure",
        option_quote={
            "mark": 5.0,
            "underlying_price": 100.0,
            "surface_state": {"implied_volatility": 0.25},
        },
        instrument_terms=_option_terms(symbol),
    )
    reference = RejectingReference(
        plan.required_sessions,
        [],
        price_fn=lambda index: 100.0 + 0.1 * index,
    )
    history_service = RiskOptionHistoryService(
        _OptionRepository([]), reference, now_fn=lambda: AS_OF
    )
    context = {
        "risk.history.lookback_sessions": 10,
        "risk.history.minimum_observations": 126,
        "risk.tail_risk.minimum_gross_delta_coverage": 0.90,
        "risk.rates.configured_flat_continuous_rate": 0.04,
        "risk.dividends.default_continuous_yield": 0.0,
    }
    portfolio_service = SimpleNamespace(
        context=context,
        reference_client=reference,
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                portfolio_service=portfolio_service,
                option_history_service=history_service,
            )
        )
    )
    response = SimpleNamespace(
        as_of=AS_OF,
        data=SimpleNamespace(positions=(position,)),
    )

    portfolio_tail = _tail_analysis(request, response, 0.95)
    hedge_tail = HedgeLabService(
        portfolio_service, risk_history_service=history_service
    )._tail_analysis((position,), AS_OF, 0.95)

    assert portfolio_tail["status"] == hedge_tail["status"] == "insufficient_data"
    assert portfolio_tail["failure_causes"] == hedge_tail["failure_causes"]
    assert portfolio_tail["coverage"] == hedge_tail["coverage"]
    assert portfolio_tail["missing_position_ids"] == hedge_tail["missing_position_ids"]
    assert [row["code"] for row in portfolio_tail["failure_causes"]] == [
        "complete_scenarios_below_minimum",
        "upstream_history_range_failed",
    ]
    upstream = portfolio_tail["failure_causes"][-1]["failed_ranges"][0]
    assert upstream["error"]["http_status"] == 422
    assert upstream["error"]["server_error_code"] == "INVALID_REQUEST"
    assert upstream["error"]["server_details"] == {
        "maximum_days": 376,
        "exclusive_end": True,
    }
    assert upstream["error"]["request_id"] == "tail-structured-422"


def test_bounded_current_greek_bridge_matches_full_fixture_and_rejects_outside_domain():
    symbol = _option_symbol("BRG", "C", 100)
    position = normalize_broker_position(
        {
            "instrument": {"symbol": symbol, "type": "OPTION"},
            "long_quantity": 1,
        },
        account_id="bridge-calibration",
        option_quote={"mark": 5.0, "underlying_price": 100.0},
        instrument_terms=_option_terms(symbol),
    )
    context = {
        "risk.option_history.greek_bridge_calibration": {
            "median_absolute_error_fraction": 0.03,
            "p95_absolute_error_fraction": 0.08,
            "maximum_absolute_error_fraction": 0.12,
            "loss_sign_preserved": True,
            "tail_ordering_preserved": True,
        },
        "risk.option_history.greek_bridge_maximum_spot_return": 0.03,
        "risk.option_history.greek_bridge_maximum_iv_points": 3.0,
        "risk.option_history.greek_bridge_minimum_dte": 7,
        "risk.option_history.greek_bridge_maximum_dte": 730,
        "risk.option_history.greek_bridge_maximum_log_moneyness": 0.25,
    }
    engine = CurrentPortfolioHistoricalRevaluationEngine(
        _ReferenceFixture((), ()), context
    )
    solved = engine.option_resolver.tian_service.solve(
        IvGreeksRequest(
            valuation_at=AS_OF,
            contracts=[
                IvGreeksContract(
                    request_id="bridge-calibration-anchor",
                    canonical_occ_symbol=symbol,
                    underlying_symbol="BRG",
                    put_call="CALL",
                    exercise_type="AMERICAN",
                    strike=100.0,
                    expiration_at=position.expiration_at,
                    spot=100.0,
                    mark=5.0,
                    quote_time=AS_OF,
                    risk_free_rate=0.04,
                    dividend_yield=0.0,
                )
            ],
        )
    ).data[0]
    assert solved.status == "ok"
    assert solved.implied_volatility is not None
    position = position.model_copy(
        update={
            "delta": solved.delta,
            "gamma": solved.gamma,
            "theta_per_day": solved.theta_per_day,
            "vega_per_1pct": solved.vega_per_1pct,
            "rho_per_1pct": solved.rho_per_1pct,
        }
    )
    current_bridge = {"spot": 100.0, "rate": 0.04, "dividend": 0.0}
    current_full = {
        **current_bridge,
        "iv": solved.implied_volatility,
    }
    errors = []
    full_losses = []
    bridge_losses = []
    for underlying_return, iv_points in [
        (-0.025, 2.5),
        (-0.02, 2.0),
        (-0.01, 1.0),
        (0.01, -1.0),
        (0.02, -2.0),
        (0.025, -2.5),
    ]:
        full, full_method, _ = engine._reprice_option(
            position,
            as_of=AS_OF,
            current=current_full,
            underlying_return=underlying_return,
            iv_points=iv_points,
            rate_points=0.0,
        )
        bridge, method, _ = engine._reprice_option(
            position,
            as_of=AS_OF,
            current=current_bridge,
            underlying_return=underlying_return,
            iv_points=iv_points,
            rate_points=0.0,
        )
        assert full_method == "current_exact_mark_anchored_full_tian_repricing"
        assert method == "bounded_current_greek_bridge"
        assert full is not None
        assert bridge is not None
        errors.append(abs(bridge - full) / max(1.0, abs(full)))
        full_losses.append(-full)
        bridge_losses.append(-bridge)

    ordered_errors = sorted(errors)
    assert ordered_errors[len(ordered_errors) // 2] <= 0.05
    assert ordered_errors[-1] <= 0.10  # empirical 95th percentile for this grid
    assert ordered_errors[-1] <= 0.15
    assert all((left >= 0) == (right >= 0) for left, right in zip(full_losses, bridge_losses))
    assert sorted(range(len(full_losses)), key=full_losses.__getitem__) == sorted(
        range(len(bridge_losses)), key=bridge_losses.__getitem__
    )
    assert len(engine._option_base_reprice_cache) == 1

    fixed_iv, fixed_iv_method, fixed_iv_diagnostics = engine._reprice_option(
        position,
        as_of=AS_OF,
        current=current_full,
        underlying_return=0.02,
        iv_points=0.0,
        rate_points=0.0,
        prefer_fixed_iv_greek_bridge=True,
    )
    move = 2.0
    expected_fixed_iv = position.signed_quantity * float(position.multiplier) * (
        float(position.delta) * move
        + 0.5 * float(position.gamma) * move * move
        + float(position.theta_per_day)
    )
    assert fixed_iv == pytest.approx(expected_fixed_iv)
    assert fixed_iv_method == "current_mark_anchored_fixed_iv_delta_gamma_bridge"
    assert fixed_iv_diagnostics["authoritative_current_mark"] == position.mark

    rejected, rejected_method, _ = engine._reprice_option(
        position,
        as_of=AS_OF,
        current=current_bridge,
        underlying_return=0.031,
        iv_points=1.0,
        rate_points=0.0,
    )
    assert rejected is None
    assert rejected_method == "full_repricing_failed_and_bridge_outside_approved_domain"


def test_unsupported_option_styles_subtypes_and_current_mark_only_fail_closed_explicitly():
    symbol = _option_symbol("CLS", "P", 100)
    position = normalize_broker_position(
        {
            "instrument": {"symbol": symbol, "type": "OPTION"},
            "long_quantity": 1,
        },
        account_id="explicit-fail-closed",
        option_quote={"mark": 5.0, "underlying_price": 100.0},
        instrument_terms=_option_terms(symbol),
    )
    engine = CurrentPortfolioHistoricalRevaluationEngine(
        _ReferenceFixture((), ()),
        {
            "risk.rates.configured_flat_continuous_rate": 0.04,
            "risk.dividends.default_continuous_yield": 0.0,
        },
    )
    base_terms = position.instrument_terms
    assert base_terms is not None
    european = position.model_copy(
        update={
            "instrument_terms": base_terms.model_copy(
                update={"exercise_style": "european"}
            )
        }
    )
    european_state, european_diagnostics, european_failure = (
        engine._current_option_state(european, AS_OF)
    )
    assert european_failure is None
    assert european_state["iv"] > 0
    assert european_diagnostics["method"] == "current_exact_mark_inverted_iv"

    variants = {
        "unsupported_exercise_style:unknown": base_terms.model_copy(
            update={"exercise_style": "unknown"}
        ),
        "unsupported_option_subtype:adjusted_deliverable": base_terms.model_copy(
            update={"adjusted_contract": True}
        ),
        "unsupported_option_subtype:futures_option": base_terms.model_copy(
            update={"contract_subtype": "futures_option"}
        ),
    }
    for expected, terms in variants.items():
        variant = position.model_copy(update={"instrument_terms": terms})
        _, _, failure = engine._current_option_state(variant, AS_OF)
        assert failure == expected

    class NoIvSolver:
        def solve(self, request):
            return SimpleNamespace(
                data=[SimpleNamespace(implied_volatility=None) for _ in request.contracts]
            )

    engine.option_resolver.tian_service = NoIvSolver()
    _, _, failure = engine._current_option_state(position, AS_OF)
    assert failure == "current_mark_without_nonlinear_sensitivity"


def test_current_surface_anchor_has_priority_over_mark_iv_inversion():
    symbol = _option_symbol("SRF", "C", 100)
    position = normalize_broker_position(
        {
            "instrument": {"symbol": symbol, "type": "OPTION"},
            "long_quantity": 1,
        },
        account_id="current-surface",
        option_quote={
            "mark": 5.0,
            "underlying_price": 100.0,
            "surface_state": {
                "surface_id": "current-fixture-surface",
                "implied_volatility": 0.31,
            },
        },
        instrument_terms=_option_terms(symbol),
    )

    class SolverMustNotRun:
        def solve(self, _request):
            raise AssertionError("surface anchor must precede current IV inversion")

    engine = CurrentPortfolioHistoricalRevaluationEngine(
        _ReferenceFixture((), ()),
        {
            "risk.rates.configured_flat_continuous_rate": 0.04,
            "risk.dividends.default_continuous_yield": 0.0,
        },
    )
    engine.option_resolver.tian_service = SolverMustNotRun()
    current, provenance, failure = engine._current_option_state(position, AS_OF)

    assert failure is None
    assert current["iv"] == pytest.approx(0.31)
    assert provenance["method"] == "current_surface"


def test_engine_clips_one_contract_to_authoritative_listing_date_before_remote_query():
    plan = HistoryWindowPlanner().plan(
        as_of=AS_OF,
        requested_return_count=10,
        calendar_id="XNYS",
        history_kind="exact_option_mark",
    )
    symbol = _option_symbol("NEW", "C", 100)
    terms = _option_terms(symbol).model_copy(
        update={"listing_date": date(2026, 8, 20)}
    )
    position = normalize_broker_position(
        {
            "instrument": {"symbol": symbol, "type": "OPTION"},
            "long_quantity": 1,
        },
        account_id="authoritative-listing-date",
        option_quote={"mark": 5.0, "underlying_price": 100.0},
        instrument_terms=terms,
    )
    reference = _ReferenceFixture(plan.required_sessions, [])
    history_service = RiskOptionHistoryService(
        _OptionRepository([]), reference, now_fn=lambda: AS_OF
    )
    engine = CurrentPortfolioHistoricalRevaluationEngine(
        reference,
        {
            "risk.rates.configured_flat_continuous_rate": 0.04,
            "risk.dividends.default_continuous_yield": 0.0,
        },
        risk_history_service=history_service,
    )
    engine.option_resolver.tian_service = _TianFixture()

    engine.revalue((position,), as_of=AS_OF, lookback_sessions=10)

    assert len(reference.option_calls) == 1
    assert reference.option_calls[0]["start_at"] == "2026-08-20T00:00:00+00:00"
    assert reference.option_calls[0]["end_at"] == "2026-08-27T00:00:00+00:00"
