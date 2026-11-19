from __future__ import annotations

from datetime import date, datetime, timezone
from threading import Barrier
from types import SimpleNamespace

import pytest

from algotrading.refdata.api.live_models import OptionMarketHistoryRequest
from algotrading.refdata.api.option_symbols import parse_occ_symbol
from algotrading.risk.history.revaluation import CurrentPortfolioHistoricalRevaluationEngine
from algotrading.risk.api.hedge_service import HedgeLabService
from algotrading.risk.instruments.terms import build_instrument_terms
from algotrading.risk.option_history.resolver import HistoricalOptionStateResolver
from algotrading.risk.option_prices.service import RiskOptionHistoryService
from algotrading.risk.pnl.attribution import _price_history_fallback
from algotrading.risk.pnl.valuation import ExactOptionObservation
from algotrading.risk.positions.normalization import normalize_broker_position


UTC = timezone.utc
CALL = "XYZ260918C00100000"
PUT = "XYZ260918P00100000"
START = datetime(2026, 8, 19, tzinfo=UTC)
END = datetime(2026, 8, 22, tzinfo=UTC)


class _Dump:
    def __init__(self, value):
        self.value = value

    def model_dump(self, **_kwargs):
        return dict(self.value)


class _IvSolver:
    def __init__(self, *, usable: bool):
        self.usable = usable

    def solve(self, request):
        rows = []
        for index, _contract in enumerate(request.contracts):
            rows.append(
                _Dump(
                    {
                        "reason": "complete" if self.usable else "iv_inversion_failed",
                        "implied_volatility": 0.20 + 0.01 * index if self.usable else None,
                        "delta": 0.50 if self.usable else None,
                        "gamma": 0.01 if self.usable else None,
                        "theta_per_day": -0.02 if self.usable else None,
                        "vega_per_1pct": 0.10 if self.usable else None,
                        "rho_per_1pct": 0.03 if self.usable else None,
                    }
                )
            )
        return SimpleNamespace(data=rows)


class _Reference:
    def __init__(self, rows):
        self.rows = rows

    def option_market_history(self, _payload):
        return {"data": list(self.rows), "meta": {"warnings": []}}

    def bars(self, symbols, **_kwargs):
        return [{"symbol": symbol, "bars": []} for symbol in symbols]

    def dividend_calendar(self, **_kwargs):
        return []

    def risk_free_curve(self, _session):
        raise ValueError("fixture uses configured flat rate")


def _terms(symbol=CALL):
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
                    "security_identifier": "XYZ",
                    "quantity": 100,
                }
            ],
            "aggregate_exercise_amount": 10_000,
            "expiration_at": "2026-09-18T20:00:00Z",
            "exercise_style": "american",
            "settlement_type": "physical",
            "source": "fixture",
            "source_version": "fixture-v1",
        },
    )


def _row(symbol, session, mark, spot, **updates):
    value = {
        "requested_symbol": symbol,
        "canonical_occ_symbol": symbol,
        "session_date": session,
        "at": f"{session}T20:00:00Z",
        "observed_at": f"{session}T20:01:00Z",
        "status": "ok",
        "mark": mark,
        "mark_basis": "daily_close",
        "underlying_price": spot,
        "source": "refdata_remote",
        "warnings": [],
    }
    value.update(updates)
    return value


def _resolver(rows, *, iv=True, minimum=2):
    context = {
        "risk.rates.configured_flat_continuous_rate": 0.04,
        "risk.dividends.default_continuous_yield": 0.0,
        "risk.option_history.minimum_contract_iv_observations": minimum,
    }
    return HistoricalOptionStateResolver(
        _Reference(rows), context, tian_service=_IvSolver(usable=iv)
    )


def test_historical_curve_dates_are_prefetched_concurrently():
    entered = Barrier(2, timeout=2)

    class ParallelReference(_Reference):
        def risk_free_curve(self, session):
            entered.wait()
            return {"data": {"requested_as_of": session.isoformat()}}

    resolver = HistoricalOptionStateResolver(
        ParallelReference([]),
        {"risk.history.risk_free_curve_prefetch_workers": 2},
        tian_service=_IvSolver(usable=False),
    )
    sessions = {date(2026, 8, 20), date(2026, 8, 21)}

    curves = resolver._risk_free_curves(sessions)

    assert set(curves) == sessions


def test_historical_curve_dates_use_batch_contract_when_available():
    class BatchReference(_Reference):
        def __init__(self):
            super().__init__([])
            self.calls = []

        def risk_free_curves(self, sessions):
            self.calls.append(tuple(sessions))
            return {
                session: {"data": {"requested_as_of": session.isoformat()}}
                for session in sessions
            }

        def risk_free_curve(self, _session):
            raise AssertionError("per-date curve endpoint should not be used")

    reference = BatchReference()
    resolver = HistoricalOptionStateResolver(
        reference, {}, tian_service=_IvSolver(usable=False)
    )
    sessions = {date(2026, 8, 20), date(2026, 8, 21)}

    curves = resolver._risk_free_curves(sessions)

    assert set(curves) == sessions
    assert reference.calls == [(date(2026, 8, 20), date(2026, 8, 21))]


def test_two_exact_marks_use_contract_iv_and_reconcile_signed_short_position():
    rows = [_row(CALL, "2026-08-20", 4.0, 100), _row(CALL, "2026-08-21", 5.25, 103)]
    series = _resolver(rows).resolve(
        [CALL], start_at=START, end_at=END, terms_by_symbol={CALL: _terms()},
        quantity_by_symbol={CALL: -2},
    )[CALL]

    interval = series.intervals[0]
    assert interval.method == "contract_iv"
    assert interval.exact_pnl_per_contract == pytest.approx(125)
    assert interval.exact_position_pnl_usd == pytest.approx(-250)
    assert abs(interval.reconciliation_difference_usd) <= 0.01


def test_iv_failure_uses_captured_greek_bridge_and_preserves_exact_total():
    rows = [
        _row(CALL, "2026-08-20", 4.0, 100, provider_delta=0.5, provider_gamma=0.01),
        _row(CALL, "2026-08-21", 5.0, 104),
    ]
    interval = _resolver(rows, iv=False).resolve(
        [CALL], start_at=START, end_at=END, terms_by_symbol={CALL: _terms()}
    )[CALL].intervals[0]

    assert interval.method == "bounded_greek_bridge"
    assert interval.exact_position_pnl_usd == pytest.approx(100)
    assert interval.observed_repricing_pnl_usd is not None
    assert interval.residual_pnl_usd == 0


def test_marks_only_are_observed_repricing_not_generic_model_residual():
    rows = [_row(CALL, "2026-08-20", 4.0, 100), _row(CALL, "2026-08-21", 2.5, 130)]
    interval = _resolver(rows, iv=False).resolve(
        [CALL], start_at=START, end_at=END, terms_by_symbol={CALL: _terms()}
    )[CALL].intervals[0]

    assert interval.method == "observed_price_only"
    assert interval.observed_repricing_pnl_usd == pytest.approx(-150)
    assert interval.residual_pnl_usd == 0
    assert "observed_option_repricing_driver_detail_unavailable" in interval.warnings


def test_surface_endpoint_state_has_priority_over_contract_iv():
    state = {"surface_id": "fixture-surface"}
    rows = [
        _row(CALL, "2026-08-20", 4.0, 100, surface_state=state),
        _row(CALL, "2026-08-21", 5.0, 102, surface_state=state),
    ]
    interval = _resolver(rows).resolve(
        [CALL], start_at=START, end_at=END, terms_by_symbol={CALL: _terms()}
    )[CALL].intervals[0]
    assert interval.method == "surface"
    assert interval.exact_position_pnl_usd == pytest.approx(100)


@pytest.mark.parametrize(
    ("rows", "input_mode", "intervals"),
    [([], "insufficient", 0), ([_row(CALL, "2026-08-20", 4.0, 100)], "single_mark_only", 0)],
)
def test_missing_or_single_exact_mark_never_fabricates_an_interval(rows, input_mode, intervals):
    series = _resolver(rows, iv=False).resolve(
        [CALL], start_at=START, end_at=END, terms_by_symbol={CALL: _terms()}
    )[CALL]
    assert series.input_mode == input_mode
    assert len(series.intervals) == intervals


def test_observation_captured_after_analysis_boundary_is_not_used():
    rows = [
        _row(CALL, "2026-08-20", 4.0, 100),
        _row(CALL, "2026-08-21", 5.0, 101, observed_at="2026-08-22T01:00:00Z"),
    ]
    series = _resolver(rows, iv=False).resolve(
        [CALL], start_at=START, end_at=END, terms_by_symbol={CALL: _terms()}
    )[CALL]
    assert len(series.observations) == 1
    assert series.input_mode == "single_mark_only"


def test_contract_iv_series_requires_configured_history_minimum():
    rows = [_row(CALL, "2026-08-20", 4.0, 100), _row(CALL, "2026-08-21", 5.0, 102)]
    series = _resolver(rows, minimum=126).resolve(
        [CALL], start_at=START, end_at=END, terms_by_symbol={CALL: _terms()}
    )[CALL]
    assert series.intervals[0].method == "contract_iv"
    assert not series.time_series_usable
    assert "contract_iv_history_minimum_observations_not_met" in series.warnings


def test_sql_mark_completeness_does_not_trigger_a_remote_greek_only_request():
    class Repository:
        def query_history(self, _symbols, **_kwargs):
            return [_row(CALL, "2026-08-20", 4.25, 100, source="risk_mysql_option_prices")]

    remote = _Reference([_row(CALL, "2026-08-20", 9.0, 101, provider_delta=0.45)])
    service = RiskOptionHistoryService(
        Repository(), remote, now_fn=lambda: END
    )
    result = service.query(
        OptionMarketHistoryRequest(symbols=[CALL], start_at=START, end_at=END),
        required_sessions_by_symbol={
            CALL: (datetime(2026, 8, 20, tzinfo=UTC).date(),)
        },
    )
    selected = next(item for item in result.data if item.session_date.isoformat() == "2026-08-20")
    assert selected.mark == pytest.approx(4.25)
    assert selected.provider_delta is None
    assert selected.source == "risk_mysql_option_prices"
    assert result.diagnostics["requested_remote_ranges"] == []
    assert result.diagnostics["greek_gaps_trigger_remote_requests"] is False


def test_delta_neutral_straddle_keeps_nonlinear_exact_mark_history():
    rows = [
        _row(CALL, "2026-08-20", 4.0, 100), _row(CALL, "2026-08-21", 7.0, 100),
        _row(PUT, "2026-08-20", 4.0, 100), _row(PUT, "2026-08-21", 6.0, 100),
    ]
    reference = _Reference(rows)
    context = {
        "risk.rates.configured_flat_continuous_rate": 0.04,
        "risk.dividends.default_continuous_yield": 0.0,
        "risk.option_history.minimum_contract_iv_observations": 126,
    }
    engine = CurrentPortfolioHistoricalRevaluationEngine(reference, context)
    positions = []
    for symbol, delta in ((CALL, 0.5), (PUT, -0.5)):
        terms = _terms(symbol)
        positions.append(
            normalize_broker_position(
                {"instrument": {"symbol": symbol, "type": "OPTION"}, "long_quantity": 1},
                account_id="account",
                option_quote={"mark": 4.0, "underlying_price": 100},
                greek={"delta": delta},
                instrument_terms=terms,
            )
        )
    result = engine.revalue(positions, as_of=END, lookback_sessions=10)
    combined = sum(series["2026-08-21"] for series in result.unit_pnl.values())
    assert sum(position.delta_exposure_usd or 0 for position in positions) == 0
    assert combined == pytest.approx(500)
    assert set(result.unit_methods.values()) == {"contract_iv"}
    assert not result.missing_position_ids


def test_pnl_tail_factor_cluster_and_hedge_share_the_same_marks_only_interval():
    rows = [_row(CALL, "2026-08-20", 4.0, 100), _row(CALL, "2026-08-21", 5.0, 130)]
    resolver = _resolver(rows, iv=False)
    partial_terms = build_instrument_terms(
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
    series = resolver.resolve(
        [CALL], start_at=START, end_at=END, terms_by_symbol={CALL: partial_terms}
    )[CALL]
    position = normalize_broker_position(
        {"instrument": {"symbol": CALL, "type": "OPTION"}, "long_quantity": 1},
        account_id="account",
        option_quote={"mark": 5.0, "underlying_price": 130},
        instrument_terms=partial_terms,
    )
    engine = CurrentPortfolioHistoricalRevaluationEngine(
        _Reference(rows),
        {
            "risk.rates.configured_flat_continuous_rate": 0.04,
            "risk.dividends.default_continuous_yield": 0.0,
            "risk.option_history.minimum_contract_iv_observations": 126,
        },
    )
    engine.option_resolver.tian_service = _IvSolver(usable=False)
    historical = engine.revalue([position], as_of=END, lookback_sessions=10)
    warnings: list[str] = []
    pnl_components = _price_history_fallback(
        parse_occ_symbol(CALL),
        ExactOptionObservation(
            canonical_occ_symbol=CALL, at=datetime(2026, 8, 20, 20, tzinfo=UTC),
            observed_at=datetime(2026, 8, 20, 20, 1, tzinfo=UTC), mark=4.0,
            session_date=datetime(2026, 8, 20, tzinfo=UTC).date(),
        ),
        ExactOptionObservation(
            canonical_occ_symbol=CALL, at=datetime(2026, 8, 21, 20, tzinfo=UTC),
            observed_at=datetime(2026, 8, 21, 20, 1, tzinfo=UTC), mark=5.0,
            session_date=datetime(2026, 8, 21, tzinfo=UTC).date(),
        ),
        start_spot=100,
        end_spot=130,
        start_rate=0.04,
        end_rate=0.04,
        start_dividend=0.0,
        end_dividend=0.0,
        pricer=None,
        warnings=warnings,
        expiration_at=partial_terms.expiration_at,
    )

    assert position.field_validity["historical_mark_revaluation_terms"].valid_for_risk
    assert not position.field_validity["exercise_style"].valid_for_risk
    assert series.intervals[0].method == "observed_price_only"
    assert historical.unit_interval_methods[position.position_id]["2026-08-21"] == "observed_price_only"
    assert sum(pnl_components.values()) * 100 == pytest.approx(
        historical.unit_pnl[position.position_id]["2026-08-21"]
    )
    assert any(item.endswith(":observed_price_only") for item in warnings)


def test_hedge_lab_requires_and_forwards_risk_sql_option_history_dependency():
    history_service = object()
    portfolio_service = SimpleNamespace(context={}, reference_client=object())
    with pytest.raises(ValueError, match="Risk SQL option-history"):
        HedgeLabService(portfolio_service, risk_history_service=None)

    service = HedgeLabService(
        portfolio_service, risk_history_service=history_service
    )
    engine = service._historical_revaluation_engine()
    assert engine.option_resolver.risk_history_service is history_service


def test_stale_current_option_is_excluded_even_when_exact_history_exists():
    rows = [_row(CALL, "2026-08-20", 4.0, 100), _row(CALL, "2026-08-21", 5.0, 102)]
    position = normalize_broker_position(
        {"instrument": {"symbol": CALL, "type": "OPTION"}, "long_quantity": 1},
        account_id="account",
        option_quote={
            "mark": 5.0,
            "underlying_price": 102,
            "quote_time": "2026-08-20T15:00:00Z",
        },
        instrument_terms=_terms(),
        as_of=END,
    )
    engine = CurrentPortfolioHistoricalRevaluationEngine(_Reference(rows), {})
    result = engine.revalue([position], as_of=END, lookback_sessions=10)

    assert position.field_validity["mark"].valid_for_risk is False
    assert result.unit_pnl == {}
    assert result.missing_position_ids == (position.position_id,)
