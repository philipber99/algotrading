from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from algotrading.refdata.api.live_models import OptionMarketHistoryRequest
from algotrading.refdata.api.market_calendars import (
    CALENDAR_VERSION,
    calendar_for_futures_root,
    exchange_sessions,
)
from algotrading.risk.api.hedge_service import (
    HedgeLabService,
    _bounded_hedge_quantities,
    _bounded_hedge_quantity,
    _objective_sizing_exposure,
    _primary_objective_usable,
)
from algotrading.risk.api.hedge_models import HedgeCandidateQueryRequest
from algotrading.risk.api.portfolio_models import ScenarioEvaluateRequest
from algotrading.risk.history.revaluation import (
    CurrentPortfolioHistoricalRevaluationEngine,
)
from algotrading.risk.instruments.terms import build_instrument_terms
from algotrading.risk.market_inputs.freshness import evaluate_quote_freshness
from algotrading.risk.option_history.resolver import HistoricalOptionStateResolver
from algotrading.risk.option_prices.service import RiskOptionHistoryService
from algotrading.risk.payoff.engine import certify_maximum_loss
from algotrading.risk.positions.normalization import normalize_broker_position
from algotrading.risk.scenarios.engine import ScenarioDefinition, ScenarioEngine
from algotrading.risk.models import PortfolioSnapshot


UTC = timezone.utc
NOW = datetime(2026, 8, 27, 22, tzinfo=UTC)
CALL = "XYZ260918C00100000"


class ReferenceHistory:
    def __init__(self, rows):
        self.rows = list(rows)

    def option_market_history(self, _payload):
        return {"data": self.rows, "meta": {"warnings": []}}

    def bars(self, symbols, **_kwargs):
        return [{"symbol": symbol, "bars": []} for symbol in symbols]

    def dividend_calendar(self, **_kwargs):
        return []

    def risk_free_curve(self, _session):
        return {"data": None}


class SolverMustNotRun:
    def solve(self, _request):
        raise AssertionError("unknown exercise style must not invoke model pricing")


def exact_row(day: int, mark: float, **fields):
    at = datetime(2026, 8, day, 20, tzinfo=UTC)
    return {
        "requested_symbol": CALL,
        "canonical_occ_symbol": CALL,
        "at": at.isoformat(),
        "observed_at": at.isoformat(),
        "session_date": at.date().isoformat(),
        "status": "ok",
        "mark": mark,
        "mark_basis": "official_close",
        "source": "risk_mysql_option_prices",
        **fields,
    }


def partial_option():
    terms = build_instrument_terms(
        CALL,
        broker_type="OPTION",
        option_terms={
            "terms_status": "partial",
            "price_multiplier": 100,
            "expiration_at": "2026-09-18T20:00:00Z",
            "exercise_style": "unknown",
            "settlement_type": "unknown",
            "asset_subtype": "unknown",
            "source": "archived_exact_multiplier",
            "source_version": "fixture-v1",
        },
    )
    return normalize_broker_position(
        {
            "instrument": {"symbol": CALL, "type": "OPTION"},
            "long_quantity": 1,
        },
        account_id="account",
        option_quote={"mark": 5.0, "underlying_price": 100.0},
        instrument_terms=terms,
    )


def snapshot(*positions):
    return PortfolioSnapshot(
        portfolio_snapshot_id="snapshot",
        configuration_version="configuration",
        as_of=NOW,
        account_ids=("account",),
        portfolio_value_usd=10_000,
        positions=tuple(positions),
        data_quality={"status": "usable"},
    )


def stock(symbol: str, *, sector="Technology"):
    return normalize_broker_position(
        {
            "instrument": {"symbol": symbol, "type": "EQUITY"},
            "long_quantity": 1,
        },
        account_id="account",
        quote={"mark": 10.0},
        classification={"sector": sector},
    )


def test_partial_terms_keep_exact_mark_history_but_fail_payoff_fields_closed():
    option = partial_option()
    assert not option.field_validity["instrument_terms"].valid_for_risk
    assert option.field_validity["premium_multiplier"].valid_for_risk
    assert option.field_validity["historical_mark_revaluation_terms"].valid_for_risk
    assert not option.field_validity["exercise_style"].valid_for_risk
    assert not option.field_validity["settlement"].valid_for_risk
    assert not option.field_validity["payoff_deliverable"].valid_for_risk

    reference = ReferenceHistory([exact_row(24, 5.0), exact_row(25, 6.0)])
    result = CurrentPortfolioHistoricalRevaluationEngine(
        reference, {"risk.option_history.minimum_contract_iv_observations": 126}
    ).revalue([option], as_of=NOW, lookback_sessions=5)
    assert result.missing_position_ids == (option.position_id,)
    assert result.unit_pnl[option.position_id]["2026-08-25"] == 100.0
    assert result.unit_methods[option.position_id] == "observed_price_only"
    assert option.position_id not in result.scenario_pnl
    assert any(
        warning.endswith(":authoritative_option_deliverable_unavailable")
        for warning in result.warnings
    )


def test_continuous_future_history_requires_explicit_roll_adjustment_contract():
    future = normalize_broker_position(
        {
            "instrument": {"symbol": "/ESU26", "type": "FUTURE"},
            "long_quantity": 1,
        },
        account_id="account",
        quote={"mark": 6_500.0},
        instrument_terms=build_instrument_terms("/ESU26", broker_type="FUTURE"),
    )

    class FutureHistory:
        def __init__(self):
            self.calls = []

        def bars(self, symbols, **_kwargs):
            self.calls.append(list(symbols))
            return [
                {
                    "symbol": "ES_CONT",
                    "bars": [
                        {"timestamp": "2026-08-24", "close": 6_400.0},
                        {"timestamp": "2026-08-25", "close": 6_500.0},
                    ],
                }
            ]

    rejected_reference = FutureHistory()
    rejected = CurrentPortfolioHistoricalRevaluationEngine(
        rejected_reference,
        {"risk.history.futures_series_map": {"ES": {"symbol": "ES_CONT"}}},
    ).revalue([future], as_of=NOW, lookback_sessions=5)
    assert rejected.missing_position_ids == (future.position_id,)
    assert rejected_reference.calls == []

    accepted_reference = FutureHistory()
    accepted = CurrentPortfolioHistoricalRevaluationEngine(
        accepted_reference,
        {
            "risk.history.futures_series_map": {
                "ES": {
                    "symbol": "ES_CONT",
                    "roll_adjustment_method": "back_adjusted_additive_v1",
                }
            }
        },
    ).revalue([future], as_of=NOW, lookback_sessions=5)
    assert accepted.missing_position_ids == ()
    assert accepted_reference.calls == [["ES_CONT"]]
    assert accepted.unit_methods[future.position_id] == (
        "configured_roll_adjusted_futures_history:back_adjusted_additive_v1"
    )


def test_unknown_style_skips_local_iv_and_preserves_exact_observed_repricing():
    option = partial_option()
    reference = ReferenceHistory(
        [
            exact_row(24, 5.0, underlying_price=100.0),
            exact_row(25, 6.0, underlying_price=101.0),
        ]
    )
    resolved = HistoricalOptionStateResolver(
        reference,
        {"risk.rates.configured_flat_continuous_rate": 0.04, "risk.dividends.default_continuous_yield": 0.0},
        tian_service=SolverMustNotRun(),
    ).resolve(
        [CALL],
        start_at=datetime(2026, 8, 24, tzinfo=UTC),
        end_at=datetime(2026, 8, 26, tzinfo=UTC),
        terms_by_symbol={CALL: option.instrument_terms},
    )[CALL]
    assert resolved.intervals[0].method == "observed_price_only"
    assert resolved.intervals[0].exact_pnl_per_contract == 100.0


def test_bounded_quantity_search_enumerates_every_admissible_integer():
    assert _bounded_hedge_quantities(
        1_000, 260, maximum_quantity=10, maximum_overhedge_fraction=0.25
    ) == (1.0, 2.0, 3.0, 4.0)
    assert _bounded_hedge_quantity(
        10_000, 100, maximum_quantity=100, maximum_overhedge_fraction=0.25
    ) == 100.0


def test_candidate_endpoint_fully_evaluates_integer_set_and_selects_objective_optimum():
    held = stock("XYZ")
    selected = snapshot(held)

    class Service(HedgeLabService):
        def __init__(self):
            super().__init__(
                SimpleNamespace(
                    context={
                        "risk.hedge_lab.maximum_candidate_quantity": 10,
                        "risk.hedge_lab.maximum_overhedge_fraction": 0.0,
                    },
                    reference_client=SimpleNamespace(),
                ),
                risk_history_service=object(),
            )
            self.evaluated = []
            self.worsen = False

        def _resolve_target(self, payload):
            response = SimpleNamespace(
                as_of=NOW,
                data=SimpleNamespace(capabilities={"available": {}}),
            )
            return response, selected, payload.target.model_dump(mode="json")

        def _assessment_data(self, *_args):
            return {
                "current": {
                    "directional_delta_exposure_usd": 1_000.0,
                    "gross_market_value_usd": 1_000.0,
                },
                "scenario": {},
            }, []

        def _relevant_symbols(self, *_args, **_kwargs):
            return ["XYZ"]

        def _universe_rows(self, _symbols):
            return [{"symbol": "XYZ"}], []

        def _current_prices(self, _symbols, **_kwargs):
            return {"XYZ": {"mark": 100.0, "as_of": NOW, "source": "fixture"}}, []

        def _option_candidates(self, *_args, **_kwargs):
            return [], []

        def _future_candidates(self, *_args, **_kwargs):
            return [], []

        def evaluate(self, payload):
            quantity = float(payload.legs[0].quantity)
            self.evaluated.append(quantity)
            before = {
                "option_delta_exposure_usd": 1_000.0,
                "gamma_pnl_for_squared_1pct_move_usd": 0.0,
                "vega_per_1pct_usd": 0.0,
                "theta_per_day_usd": 0.0,
                "rho_per_1pct_usd": 0.0,
            }
            after = {
                **before,
                "option_delta_exposure_usd": (
                    1_000.0 + 100.0 * quantity
                    if self.worsen
                    else 1_000.0 - 100.0 * quantity
                ),
            }
            return SimpleNamespace(
                methodology_version="fixture-full-evaluation-v1",
                data_quality=SimpleNamespace(status="usable", warnings=[]),
                data={
                    "target": {"objective": {"before": before, "after": after}},
                    "portfolio": {"before": {"gross_market_value_usd": 1_000.0}, "after": {}},
                    "tail_risk": {}, "factors": {}, "maximum_loss": {},
                    "scenarios": [], "margin": {},
                    "liquidity": {"status": "insufficient_data"}, "residuals": {},
                },
            )

        def _analysis_response(self, _response, methodology_version, data, **_kwargs):
            return SimpleNamespace(methodology_version=methodology_version, data=data)

    service = Service()
    request = HedgeCandidateQueryRequest.model_validate(
        {
            "portfolio_snapshot_id": "snapshot",
            "target": {"kind": "Portfolio", "label": "Portfolio", "position_ids": [held.position_id]},
            "objective": "Greeks",
            "objective_parameters": {"selected_greeks": ["delta"], "targets": {"delta": 0}},
            "limit": 10,
        }
    )
    response = service.candidates(request)
    assert sorted(service.evaluated) == list(range(1, 11))
    assert response.data["items"][0]["legs"][0]["quantity"] == 10.0
    assert response.data["items"][0]["primary_objective_improvement"] > 0
    service.worsen = True
    service.evaluated.clear()
    worsening = service.candidates(request)
    assert sorted(service.evaluated) == list(range(1, 11))
    assert worsening.data["items"] == []


def test_delta_neutral_tail_objective_discovers_convex_and_collar_sizes():
    held = normalize_broker_position(
        {"instrument": {"symbol": "XYZ", "type": "EQUITY"}, "long_quantity": 100},
        account_id="account",
        quote={"mark": 100.0},
        classification={"sector": "Technology"},
    )
    request = HedgeCandidateQueryRequest.model_validate(
        {
            "portfolio_snapshot_id": "snapshot",
            "target": {"kind": "Portfolio", "label": "Portfolio", "position_ids": [held.position_id]},
            "objective": "VaR / Tail Risk",
            "objective_parameters": {
                "near_confidence": 0.95,
                "far_confidence": 0.975,
                "hedge_region": "near_curve",
            },
        }
    )
    assessment = {
        "current": {"directional_delta_exposure_usd": 0.0, "gross_market_value_usd": 10_000.0},
        "tail_risk": {"near": {"var_usd": 1_000.0}, "far": {"expected_shortfall_usd": 1_500.0}},
        "scenario": {},
    }
    assert _objective_sizing_exposure(request, assessment, 0.0, 500.0) == 1_000.0
    service = HedgeLabService(
        SimpleNamespace(
            context={
                "risk.hedge_lab.maximum_candidate_quantity": 10,
                "risk.hedge_lab.maximum_overhedge_fraction": 0.25,
            },
            reference_client=SimpleNamespace(),
        ),
        risk_history_service=object(),
    )
    expiry = NOW + timedelta(days=30)
    option_terms = {
        "deliverable": {"fully_supported": True, "shares_per_contract": 100.0}
    }
    common = {
        "expiration_at": expiry, "spot": 100.0, "multiplier": 100.0,
        "mark": 2.0, "gamma": 0.01, "vega_per_1pct": 0.1,
        "theta_per_day": -0.02, "rho_per_1pct": 0.03,
        "quote_time": NOW, "bid": 1.9, "ask": 2.1,
        "open_interest": 1000.0, "volume": 100.0,
        "instrument_terms": option_terms,
    }
    rows = [
        {**common, "canonical_occ_symbol": "XYZ260926P00095000", "put_call": "PUT", "strike": 95.0, "delta": -0.05},
        {**common, "canonical_occ_symbol": "XYZ260926P00090000", "put_call": "PUT", "strike": 90.0, "delta": -0.03},
        {**common, "canonical_occ_symbol": "XYZ260926C00105000", "put_call": "CALL", "strike": 105.0, "delta": 0.05},
    ]
    candidates = service._structures_from_chain(
        request, assessment, snapshot(held), "XYZ", rows, 0.0
    )
    assert any("put" in item["name"].lower() for item in candidates)
    assert any("protective collar" in item["name"].lower() for item in candidates)

    maximum_loss_request = HedgeCandidateQueryRequest.model_validate(
        {
            "portfolio_snapshot_id": "snapshot",
            "target": {
                "kind": "Portfolio",
                "label": "Portfolio",
                "position_ids": [held.position_id],
            },
            "objective": "Maximum Loss",
            "objective_parameters": {"desired_loss_floor_usd": 0},
        }
    )
    maximum_loss_assessment = {
        **assessment,
        "maximum_loss": {"maximum_loss_usd": 2_000.0},
    }
    assert _objective_sizing_exposure(
        maximum_loss_request, maximum_loss_assessment, 0.0, 500.0
    ) == 2_000.0
    assert any(
        "put" in item["name"].lower()
        for item in service._structures_from_chain(
            maximum_loss_request,
            maximum_loss_assessment,
            snapshot(held),
            "XYZ",
            rows,
            0.0,
        )
    )

    rows[-1] = {
        **rows[-1],
        "instrument_terms": {
            "deliverable": {"fully_supported": False, "shares_per_contract": 100.0}
        },
    }
    unsupported_call = service._structures_from_chain(
        request, assessment, snapshot(held), "XYZ", rows, 0.0
    )
    assert not any("protective collar" in item["name"].lower() for item in unsupported_call)


def test_scenario_objective_allows_harmless_warnings_but_requires_full_coverage():
    request = HedgeCandidateQueryRequest.model_validate(
        {
            "portfolio_snapshot_id": "snapshot",
            "target": {"kind": "Portfolio", "label": "Portfolio", "position_ids": ["p"]},
            "objective": "Scenario Loss",
            "objective_parameters": {"default_equity_shock": -0.05},
        }
    )
    row = {
        "scenario_pnl_usd": -100.0,
        "quality_state": "usable_with_warnings",
        "unmodeled_position_ids": [],
        "material_unmodeled_position_ids": [],
    }
    evaluation = {"target": {"objective": {"before": row, "after": row}}}
    assert _primary_objective_usable(request, evaluation)
    incomplete = {
        "target": {
            "objective": {
                "before": {**row, "unmodeled_position_ids": ["missing"]},
                "after": row,
            }
        }
    }
    assert not _primary_objective_usable(request, incomplete)


def test_many_small_positions_cannot_hide_one_unmodeled_nonlinear_position():
    positions = [stock(f"S{index}") for index in range(100)]
    positions.append(partial_option())
    result = ScenarioEngine({}).evaluate(
        snapshot(*positions),
        ScenarioDefinition(scenario_id="down", name="Down", broad_equity_shock=-0.1),
    )
    assert result["coverage"]["count_percent"] > 99
    assert result["scenario_pnl_usd"] is None
    assert result["quality_state"] == "insufficient_data"
    assert result["material_unmodeled_position_ids"] == [positions[-1].position_id]


def test_unmodeled_short_and_unknown_fund_exposures_also_fail_closed():
    short = normalize_broker_position(
        {
            "instrument": {"symbol": "SHORT", "type": "EQUITY"},
            "short_quantity": 1,
        },
        account_id="account",
        quote={},
        classification={"sector": "Technology"},
    )
    fund = normalize_broker_position(
        {
            "instrument": {"symbol": "FUNDX", "type": "MUTUAL_FUND"},
            "long_quantity": 1,
        },
        account_id="account",
        quote={"mark": 10.0},
        classification={"sector": "Technology"},
    )
    result = ScenarioEngine({}).evaluate(
        snapshot(stock("MODELED"), short, fund),
        ScenarioDefinition(
            scenario_id="exact_only",
            name="Exact only",
            exact_underlying_shocks={"MODELED": -0.1},
        ),
    )
    assert result["scenario_pnl_usd"] is None
    assert set(result["material_unmodeled_position_ids"]) == {
        short.position_id,
        fund.position_id,
    }


def test_payoff_certifies_independent_underlyings_separately_and_not_mixed_expiries():
    independent = certify_maximum_loss([stock("AAA"), stock("BBB")])
    assert independent["status"] == "exact_bounded"
    assert len(independent["underlying_results"]) == 2
    assert independent["terminal_payoff"] == []

    def option(symbol: str):
        strike = float(symbol[-8:]) / 1000.0
        terms = build_instrument_terms(
            symbol,
            broker_type="OPTION",
            option_terms={
                "terms_status": "standard",
                "price_multiplier": 100,
                "shares_per_contract": 100,
                "aggregate_exercise_amount": strike * 100,
                "expiration_at": (
                    "2026-09-18T20:00:00Z"
                    if "260918" in symbol
                    else "2026-10-16T20:00:00Z"
                ),
                "exercise_style": "american",
                "settlement_type": "physical",
                "asset_subtype": "equity_option",
            },
        )
        return normalize_broker_position(
            {"instrument": {"symbol": symbol, "type": "OPTION"}, "long_quantity": 1},
            account_id="account",
            option_quote={"mark": 5.0, "underlying_price": 100.0},
            instrument_terms=terms,
        )

    mixed_expiries = certify_maximum_loss(
        [option("XYZ260918C00100000"), option("XYZ261016C00110000")]
    )
    assert mixed_expiries["status"] == "bounded_stress_only"
    assert mixed_expiries["maximum_loss_usd"] is None


def test_public_scenario_path_contract_accepts_typed_points_and_rejects_invalid_shapes():
    point = NOW + timedelta(days=1)
    request = ScenarioEvaluateRequest.model_validate(
        {
            "portfolio_snapshot_id": "snapshot",
            "horizon_calendar_days": 2,
            "path_rule": "explicit_dated_points",
            "dated_path_points": {point.isoformat(): 95.0},
        }
    )
    assert list(request.dated_path_points) == [point]
    with pytest.raises(ValueError):
        ScenarioEvaluateRequest.model_validate(
            {
                "portfolio_snapshot_id": "snapshot",
                "path_rule": "explicit_dated_points",
                "dated_path_points": {},
            }
        )
    with pytest.raises(ValueError):
        ScenarioEvaluateRequest.model_validate(
            {
                "portfolio_snapshot_id": "snapshot",
                "path_rule": "instantaneous_and_held",
                "dated_path_points": {point.isoformat(): 95.0},
            }
        )


def test_non_equity_mapping_precedes_sector_shock():
    fund = normalize_broker_position(
        {
            "instrument": {"symbol": "TLT", "type": "ETF"},
            "long_quantity": 100,
        },
        account_id="account",
        quote={"mark": 100.0},
        classification={"sector": "Technology"},
    )
    result = ScenarioEngine(
        {
            "risk.scenarios.economic_instrument_mappings": {
                "TLT": {"economic_exposure_type": "rates", "effective_duration": 10}
            }
        }
    ).evaluate(
        snapshot(fund),
        ScenarioDefinition(
            scenario_id="rates",
            name="Rates",
            sector_shocks={"Technology": -0.20},
            rate_basis_points=100,
        ),
    )
    assert result["scenario_pnl_usd"] == -1_000.0


class Repository:
    def query_history(self, *_args, **_kwargs):
        return [exact_row(24, 5.0)]


class RemoteFill(ReferenceHistory):
    def __init__(self):
        super().__init__(
            [
                exact_row(
                    24, 999.0, underlying_price=101.0, provider_iv=0.2,
                    provider_delta=0.4, provider_gamma=0.01,
                    provider_theta_per_day=-0.02, provider_vega_per_1pct=0.1,
                    provider_rho_per_1pct=0.03,
                )
            ]
        )
        self.calls = 0

    def option_market_history(self, payload):
        self.calls += 1
        return super().option_market_history(payload)


def test_complete_sql_marks_do_not_request_remote_greek_only_fields():
    remote = RemoteFill()
    service = RiskOptionHistoryService(Repository(), remote, now_fn=lambda: NOW)
    response = service.query(
        OptionMarketHistoryRequest(
            symbols=[CALL],
            start_at=datetime(2026, 8, 24, tzinfo=UTC),
            end_at=datetime(2026, 8, 25, tzinfo=UTC),
            include_greeks=True,
        )
    )
    assert remote.calls == 0
    assert response.data[0].mark == 5.0
    assert response.data[0].provider_delta is None
    assert response.diagnostics["greek_gaps_trigger_remote_requests"] is False


def test_cross_year_and_product_calendars_are_bounded_and_distinct():
    sessions_2025 = exchange_sessions("XNYS", date(2025, 12, 24), date(2025, 12, 26))
    sessions_2027 = exchange_sessions("XNYS", date(2027, 11, 24), date(2027, 11, 26))
    assert date(2025, 12, 25) not in sessions_2025
    assert date(2027, 11, 25) not in sessions_2027
    assert calendar_for_futures_root("ES") == "CME_EQUITY_INDEX"
    assert calendar_for_futures_root("CL") == "CME_ENERGY"
    assert CALENDAR_VERSION.endswith("_v2")
    with pytest.raises(ValueError):
        exchange_sessions("XNYS", date(2036, 1, 1), date(2036, 1, 2))

    energy = evaluate_quote_freshness(
        datetime(2027, 11, 26, 18, 20, tzinfo=UTC),
        as_of=datetime(2027, 11, 26, 18, 25, tzinfo=UTC),
        context={}, calendar_id="CME_ENERGY", exchange_timezone="America/Chicago",
    )
    assert energy.calendar_id == "CME_ENERGY"
    assert energy.calendar_version == CALENDAR_VERSION


def test_acceptance_matrix_and_runtime_dependencies_are_release_gated():
    matrix = Path(__file__).with_name("SEMANTIC_ACCEPTANCE_MATRIX.md").read_text()
    assert all(f"| O{index:02d} |" in matrix for index in range(1, 46))
    assert all(f"| N{index:02d} |" in matrix for index in range(1, 52))
    assert "test_second_implementation_audit.py" in matrix

    requirements = Path(__file__).parents[1] / "api" / "requirements.txt"
    declared = requirements.read_text()
    assert "torch>=" in declared
    assert "pyarrow>=" in declared

    gui_test = Path(__file__).resolve().parents[4] / "risk_gui" / "tests" / "rendered-html.test.mjs"
    gui_contract = gui_test.read_text()
    assert "runtime contracts" in gui_contract
    assert "validateRiskResponse" in gui_contract
