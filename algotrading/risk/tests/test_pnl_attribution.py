from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from algotrading.refdata.api.option_symbols import parse_occ_symbol
from algotrading.risk.market_inputs.client import ReferenceDataClientError
from algotrading.risk.option_prices.models import (
    ChainSnapshotFlag,
    ChainSnapshotStatus,
    OptionSurfaceObservation,
    OptionSurfaceSnapshot,
    OptionSurfaceSnapshotHeader,
    SurfaceObservationSource,
)
from algotrading.risk.option_prices.repository import (
    OptionPriceRepository,
    OptionPriceRepositoryUnavailable,
    OptionSurfaceHistoryLimitExceeded,
)
from algotrading.risk.option_prices.service import (
    HybridOptionHistoryClient,
    RiskOptionHistoryService,
)
from algotrading.risk.pnl.attribution import (
    FACTOR_ORDER,
    OptionMarketState,
    TianBatchRepricingAdapter,
    _price_history_fallback,
    _greek_approximation,
    _symmetric_waterfall,
    attribute_selection,
)
from algotrading.risk.pnl.reconstruction import ExecutionLeg, Lifecycle
from algotrading.risk.pnl.valuation import (
    ExactOptionObservation,
    MarketMarkBook,
    _required_cash,
)


UTC = timezone.utc
EXPIRY = date(2026, 9, 18)
EXPIRATION_AT = datetime(2026, 9, 18, 20, tzinfo=UTC)
CALL_95 = "XYZ260918C00095000"
CALL_100 = "XYZ260918C00100000"
CALL_105 = "XYZ260918C00105000"
OPEN = datetime(2026, 8, 1, 12, tzinfo=UTC)
BOUNDARIES = tuple(
    datetime(2026, 8, day, tzinfo=UTC) for day in (2, 3, 4)
)
CLOSE = datetime(2026, 8, 4, 12, tzinfo=UTC)


def leg(index, at, symbol, quantity, price, asset_type="OPTION", fee=0.0):
    multiplier = 100.0 if asset_type == "OPTION" else 1.0
    return ExecutionLeg(
        account_id="A",
        repository_transaction_id=index,
        broker_transaction_id=f"broker-{index}",
        item_index=0,
        timestamp=at,
        order_id=None,
        description=None,
        symbol=symbol,
        asset_type=asset_type,
        underlying_symbol="XYZ",
        instruction="BUY" if quantity > 0 else "SELL",
        position_effect="OPENING" if at == OPEN else "CLOSING",
        signed_quantity=quantity,
        fill_price=price,
        multiplier=multiplier,
        expiration_at=EXPIRATION_AT if asset_type == "OPTION" else None,
        fees_usd=fee,
        cash_effect_usd=-quantity * price * multiplier,
        parent_item_count=1,
    )


def vertical(*, observation_count=3):
    executions = [
        leg(1, OPEN, CALL_100, 1, 5),
        leg(2, OPEN, CALL_105, -1, 2),
        leg(3, CLOSE, CALL_100, -1, 7),
        leg(4, CLOSE, CALL_105, 1, 3),
    ]
    return Lifecycle(
        account_id="A",
        underlying_symbol="XYZ",
        instrument_type="Options",
        executions=executions,
        inventory={CALL_100: 0.0, CALL_105: 0.0},
        closed_at=CLOSE,
    )


def equity():
    executions = [
        leg(10, OPEN, "XYZ", 2, 100, "EQUITY"),
        leg(11, CLOSE, "XYZ", -2, 110, "EQUITY"),
    ]
    return Lifecycle(
        account_id="A",
        underlying_symbol="XYZ",
        instrument_type="Equity",
        executions=executions,
        inventory={"XYZ": 0.0},
        closed_at=CLOSE,
    )


class Reference:
    def __init__(self, *, rich=True, market_inputs=True, observations=3):
        self.rich = rich
        self.market_inputs = market_inputs
        self.observations = observations

    def option_market_history(self, payload):
        marks = {
            CALL_100: (5.2, 6.1, 6.8),
            CALL_105: (2.1, 2.7, 3.0),
        }
        rows = []
        for symbol in payload["symbols"]:
            for at, mark, spot in zip(
                BOUNDARIES[: self.observations],
                marks[symbol][: self.observations],
                (100.0, 102.0, 101.0),
            ):
                row = {
                    "canonical_occ_symbol": symbol,
                    "at": at.isoformat(),
                    "status": "ok",
                    "mark": mark,
                    "source": "fixture",
                }
                if self.rich:
                    row.update(
                        underlying_price=spot,
                        provider_iv=0.20 + (mark - 2.0) * 0.01,
                        provider_delta=0.5,
                        provider_gamma=0.01,
                        provider_theta_per_day=-0.02,
                        provider_vega_per_1pct=0.08,
                        provider_rho_per_1pct=0.03,
                    )
                rows.append(row)
        return {"data": rows, "missing_symbols": [], "meta": {}}

    def risk_free_curve(self, as_of):
        if not self.market_inputs:
            raise ReferenceDataClientError("curve unavailable")
        return {
            "data": {
                "curve_id": "fixture",
                "curve_version": "v1",
                "anchors": [
                    {
                        "series_id": "FIXTURE",
                        "tenor_days": 90,
                        "observation_date": as_of.isoformat(),
                        "simple_annual_rate_decimal": 0.04,
                        "status": "ok",
                    }
                ],
            }
        }

    def dividends(self, symbol, *, start, end):
        if not self.market_inputs:
            raise ReferenceDataClientError("dividends unavailable")
        return []

    def bars(self, symbols, *, start, end, remote_query=False):
        return []


class FakeRegistry:
    def solve_iv_only_many(self, jobs):
        return [
            SimpleNamespace(iv=0.10 + 0.02 * float(job.target_price))
            for job in jobs
        ]

    def price_many(self, requests, volatilities, methods, requested_dtypes=None):
        return [
            0.35 * request.spot
            - 0.30 * request.strike
            + 0.002 * (request.spot - request.strike) ** 2
            + 8.0 * volatility * (1.0 + 0.01 * (request.strike - 100.0))
            + 5.0 * request.rate
            - 3.0 * request.dividend_yield
            - 0.005 * request.tenor_seconds / 86400.0
            for request, volatility in zip(requests, volatilities)
        ]


class SurfaceRepository:
    def __init__(self, snapshots=(), *, fail=False):
        self.snapshots = list(snapshots)
        self.fail = fail
        self.calls = []

    def query_surface_history(self, symbols, *, start_at, end_at):
        self.calls.append((tuple(symbols), start_at, end_at))
        if self.fail:
            raise OptionPriceRepositoryUnavailable("database unavailable")
        return list(self.snapshots)


def surface(snapshot_id, captured_at, spot, prices, *, status=1):
    observations = tuple(
        OptionSurfaceObservation(
            snapshot_id=snapshot_id,
            canonical_occ_symbol=symbol,
            bid=price - 0.1,
            ask=price + 0.1,
            mark=price,
            last=price,
            quote_time=captured_at,
            trade_time=None,
            observation_source=int(SurfaceObservationSource.OPTION_CHAIN),
            security_status=1,
        )
        for symbol, price in zip((CALL_95, CALL_100, CALL_105), prices)
    )
    return OptionSurfaceSnapshot(
        header=OptionSurfaceSnapshotHeader(
            snapshot_id=snapshot_id,
            run_id=f"run-{snapshot_id}",
            captured_at=captured_at,
            underlying_symbol="XYZ",
            expiration_date=EXPIRY,
            spot=spot,
            spot_source=1,
            underlying_quote_time=captured_at,
            snapshot_status=status,
            snapshot_flags=0,
            error_code=None,
        ),
        observations=observations,
    )


def complete_surfaces():
    return [
        surface(1, BOUNDARIES[0] - timedelta(hours=1), 100, (8, 5, 2)),
        surface(2, BOUNDARIES[1] - timedelta(hours=1), 102, (9, 6, 3)),
        surface(3, BOUNDARIES[2] - timedelta(hours=1), 101, (8.5, 5.7, 2.6)),
    ]


def mark_book(lifecycles, reference):
    snapshot = SimpleNamespace(
        as_of=datetime(2026, 8, 5, tzinfo=UTC),
        data=SimpleNamespace(positions=[]),
    )
    marks = MarketMarkBook(snapshot, reference)
    marks.load_history(
        lifecycles,
        start_at=OPEN,
        end_at=CLOSE + timedelta(microseconds=1),
        maximum_option_symbols=20,
    )
    marks.load_execution_fills(lifecycles)
    return marks


def test_current_option_observation_uses_normalized_position_underlying_price():
    as_of = datetime(2026, 8, 5, tzinfo=UTC)
    position = SimpleNamespace(
        account_id="A",
        exact_symbol=CALL_100,
        canonical_occ_symbol=CALL_100,
        mark=5.5,
        implied_volatility=0.22,
        delta=0.51,
        gamma=0.02,
        theta_per_day=-0.03,
        vega_per_1pct=0.09,
        rho_per_1pct=0.04,
        quote_source="fixture",
        raw_broker_fields={"underlying_price": 103.25},
    )
    snapshot = SimpleNamespace(
        as_of=as_of,
        data=SimpleNamespace(positions=[position]),
    )

    observation = MarketMarkBook(snapshot, Reference()).option_observation(
        "A", CALL_100, as_of
    )

    assert observation is not None
    assert observation.underlying_price == pytest.approx(103.25)


def run_attribution(
    lifecycles,
    *,
    reference=None,
    repository=None,
    registry=None,
    context=None,
):
    reference = reference or Reference()
    marks = mark_book(lifecycles, reference)
    fees = sum(item.fees_usd for lifecycle in lifecycles for item in lifecycle.executions)
    lifecycle_pnl = sum(
        _required_cash(item)
        for lifecycle in lifecycles
        for item in lifecycle.executions
    ) + fees
    return attribute_selection(
        lifecycles=lifecycles,
        marks=marks,
        option_price_repository=repository,
        reference_client=reference,
        context=context or {},
        analysis_start=OPEN,
        analysis_end=CLOSE,
        lifecycle_pnl=lifecycle_pnl,
        fees=fees,
        solver_registry=registry or FakeRegistry(),
    ), lifecycle_pnl


def assert_reconciles(result, lifecycle_pnl):
    assert tuple(item.factor for item in result.factors) == FACTOR_ORDER
    assert sum(item.pnl_usd for item in result.factors) == pytest.approx(
        lifecycle_pnl, abs=0.01
    )


def test_vertical_with_complete_surfaces_uses_full_revaluation_and_reconciles():
    result, pnl = run_attribution(
        [vertical()], repository=SurfaceRepository(complete_surfaces())
    )

    assert result.attribution_method == "surface_revaluation"
    assert result.surface_coverage_percent == 100.0
    assert_reconciles(result, pnl)
    factors = {item.factor: item.pnl_usd for item in result.factors}
    assert factors["Underlying movement"] != 0
    assert factors["Volatility"] != 0
    assert factors["Term structure"] == 0


def test_vertical_and_underlying_shares_aggregate_in_one_ordered_factor_set():
    result, pnl = run_attribution(
        [vertical(), equity()], repository=SurfaceRepository(complete_surfaces())
    )

    assert result.attribution_method == "surface_revaluation"
    assert {item.factor: item.pnl_usd for item in result.factors}[
        "Underlying movement"
    ] != 0
    assert_reconciles(result, pnl)


def test_no_surfaces_uses_contract_price_fallback_and_repository_failure_is_graceful():
    empty, pnl = run_attribution([vertical()], repository=SurfaceRepository())
    unavailable, unavailable_pnl = run_attribution(
        [vertical()], repository=SurfaceRepository(fail=True)
    )

    assert empty.attribution_method == unavailable.attribution_method == "price_history_fallback"
    assert empty.surface_coverage_percent == unavailable.surface_coverage_percent == 0.0
    assert "fallback_contract_iv_revaluation_used" in empty.warnings
    assert "surface_repository_unavailable" in unavailable.warnings
    assert_reconciles(empty, pnl)
    assert_reconciles(unavailable, unavailable_pnl)


def test_no_surface_sql_observations_use_configured_inputs_for_contract_iv_fallback():
    """Regression: rich SQL rows must not collapse to an all-residual result."""

    class SqlHistoryRepository:
        def query_history(self, symbols, *, start_date, end_date):
            rows = []
            marks_by_symbol = {
                CALL_100: (5.2, 6.1),
                CALL_105: (2.1, 2.7),
            }
            for symbol in symbols:
                for session, mark, spot in zip(
                    (date(2026, 8, 1), date(2026, 8, 2)),
                    marks_by_symbol[symbol],
                    (100.0, 102.0),
                ):
                    # Use the exact column names returned by the MySQL repository,
                    # then exercise its public history mapping before Pydantic and
                    # hybrid-history validation.
                    raw = {
                        "bid": mark - 0.1,
                        "ask": mark + 0.1,
                        "mark": mark,
                        "last": mark,
                        "open_price": mark - 0.2,
                        "high_price": mark + 0.2,
                        "low_price": mark - 0.3,
                        "close_price": mark,
                        "underlying_price": spot,
                        "provider_iv": 0.25,
                        "provider_delta": None,
                        "provider_gamma": None,
                        "provider_theta_per_day": None,
                        "provider_vega_per_1pct": None,
                        "provider_rho_per_1pct": None,
                        "volume": 10,
                        "mark_source": 1,
                        "captured_at": datetime(
                            2026, 8, 2, 1, tzinfo=UTC
                        ),
                    }
                    rows.append(
                        OptionPriceRepository._history_item(raw, symbol, session)
                    )
            return rows

    class UnavailableMarketInputs:
        def option_market_history(self, payload):
            raise ReferenceDataClientError("remote history unavailable")

        def risk_free_curve(self, as_of):
            raise ReferenceDataClientError("curve unavailable")

        def dividends(self, symbol, *, start, end):
            raise ReferenceDataClientError("dividends unavailable")

        def bars(self, symbols, *, start, end, remote_query=False):
            return []

    upstream = UnavailableMarketInputs()
    history_service = RiskOptionHistoryService(
        SqlHistoryRepository(),
        upstream,
        now_fn=lambda: datetime(2026, 8, 5, tzinfo=UTC),
    )
    reference = HybridOptionHistoryClient(upstream, history_service)
    lifecycles = [vertical()]
    marks = mark_book(lifecycles, reference)

    for symbol in (CALL_100, CALL_105):
        observations = marks.exact_option_history(symbol)
        assert len(observations) == 2
        assert [item.underlying_price for item in observations] == [100.0, 102.0]
        assert [item.provider_iv for item in observations] == [0.25, 0.25]
        assert all(item.mark_basis == "provider_mark" for item in observations)
        assert all(item.observed_at is not None for item in observations)

    lifecycle_pnl = sum(
        _required_cash(item)
        for lifecycle in lifecycles
        for item in lifecycle.executions
    )
    result = attribute_selection(
        lifecycles=lifecycles,
        marks=marks,
        option_price_repository=SurfaceRepository(),
        reference_client=reference,
        context={
            "risk.rates.configured_flat_continuous_rate": 0.04,
            "risk.dividends.default_continuous_yield": 0.0,
        },
        analysis_start=OPEN,
        analysis_end=CLOSE,
        lifecycle_pnl=lifecycle_pnl,
        fees=0.0,
        solver_registry=FakeRegistry(),
    )

    factors = {item.factor: item.pnl_usd for item in result.factors}
    assert result.attribution_method == "price_history_fallback"
    assert result.surface_coverage_percent == 0.0
    assert any(
        abs(factors[name]) > 0.01
        for name in ("Underlying movement", "Volatility", "Time decay")
    )
    assert factors["Skew"] == factors["Term structure"] == 0.0
    assert factors["Model residual"] != pytest.approx(lifecycle_pnl)
    assert_reconciles(result, lifecycle_pnl)


def test_surface_limit_breach_downgrades_to_price_history():
    class LimitedRepository(SurfaceRepository):
        def query_surface_history(self, symbols, *, start_at, end_at):
            raise OptionSurfaceHistoryLimitExceeded("too many points")

    result, pnl = run_attribution([vertical()], repository=LimitedRepository())

    assert result.attribution_method == "price_history_fallback"
    assert result.surface_coverage_percent == 0.0
    assert "surface_history_limit_exceeded" in result.warnings
    assert_reconciles(result, pnl)


def test_partial_surface_coverage_is_between_zero_and_one_hundred():
    snapshots = complete_surfaces()[:2]
    snapshots.append(
        surface(
            4,
            BOUNDARIES[2] - timedelta(hours=1),
            101,
            (8.5, 5.7, 2.6),
            status=int(ChainSnapshotStatus.FAILED),
        )
    )
    result, pnl = run_attribution(
        [vertical()], repository=SurfaceRepository(snapshots)
    )

    assert result.attribution_method == "price_history_fallback"
    assert result.surface_coverage_percent == pytest.approx(50.0)
    assert_reconciles(result, pnl)


def test_usable_partial_delayed_truncated_surface_is_warned_not_rejected():
    snapshots = complete_surfaces()
    snapshots[1] = replace(
        snapshots[1],
        header=replace(
            snapshots[1].header,
            snapshot_status=int(ChainSnapshotStatus.PARTIAL),
            snapshot_flags=int(ChainSnapshotFlag.DELAYED)
            | int(ChainSnapshotFlag.TRUNCATED),
        ),
    )
    result, pnl = run_attribution(
        [vertical()], repository=SurfaceRepository(snapshots)
    )

    assert result.attribution_method == "surface_revaluation"
    assert any(item.startswith("surface_partial_used:") for item in result.warnings)
    assert any(item.startswith("surface_delayed_used:") for item in result.warnings)
    assert any(item.startswith("surface_truncated_used:") for item in result.warnings)
    assert_reconciles(result, pnl)


def test_old_exact_marks_without_iv_greeks_or_spot_flow_to_observed_repricing():
    reference = Reference(rich=False)
    result, pnl = run_attribution(
        [vertical()], reference=reference, repository=SurfaceRepository()
    )

    factors = {item.factor: item.pnl_usd for item in result.factors}
    assert result.attribution_method == "price_history_fallback"
    assert factors["Volatility"] == 0
    assert factors["Option repricing residual"] == pytest.approx(pnl)
    assert factors["Model residual"] == 0.0
    assert any(
        item.startswith("fallback_exact_price_driver_detail_unavailable:")
        for item in result.warnings
    )
    assert_reconciles(result, pnl)


def test_expiry_session_endpoint_uses_terminal_intrinsic_revaluation():
    option = parse_occ_symbol("XYZ260804C00100000")
    start = ExactOptionObservation(
        canonical_occ_symbol=option.canonical,
        at=datetime(2026, 8, 3, 20, tzinfo=UTC),
        mark=2.0,
        session_date=date(2026, 8, 3),
    )
    end = ExactOptionObservation(
        canonical_occ_symbol=option.canonical,
        at=datetime(2026, 8, 4, 20, tzinfo=UTC),
        mark=1.0,
        session_date=date(2026, 8, 4),
    )
    warnings: list[str] = []

    contributions = _price_history_fallback(
        option,
        start,
        end,
        start_spot=100.0,
        end_spot=101.0,
        start_rate=0.04,
        end_rate=0.04,
        start_dividend=0.0,
        end_dividend=0.0,
            pricer=TianBatchRepricingAdapter({}, registry=FakeRegistry()),
            warnings=warnings,
            expiration_at=datetime(2026, 8, 4, 20, tzinfo=UTC),
    )

    assert contributions
    assert contributions["time"] != 0.0
    assert any(
        item.startswith("fallback_terminal_revaluation_used:") for item in warnings
    )
    assert not any(item.endswith(":expired_hybrid_state") for item in warnings)


def test_exact_mark_below_intrinsic_is_left_in_residual_with_specific_reason():
    option = parse_occ_symbol(CALL_100)
    start = ExactOptionObservation(
        canonical_occ_symbol=option.canonical,
        at=BOUNDARIES[0],
        mark=1.0,
        session_date=BOUNDARIES[0].date(),
    )
    end = replace(start, at=BOUNDARIES[1], mark=1.5)
    warnings: list[str] = []

    contributions = _price_history_fallback(
        option,
        start,
        end,
        start_spot=102.0,
        end_spot=103.0,
        start_rate=0.04,
        end_rate=0.04,
        start_dividend=0.0,
        end_dividend=0.0,
            pricer=TianBatchRepricingAdapter({}, registry=FakeRegistry()),
            warnings=warnings,
            expiration_at=EXPIRATION_AT,
    )

    assert contributions == {"observed_repricing": 0.5}
    assert any(
        item.endswith(":exact_mark_outside_no_arbitrage_bounds")
        for item in warnings
    )


def test_contract_iv_fallback_puts_contract_iv_in_volatility_only():
    result, pnl = run_attribution([vertical()], repository=SurfaceRepository())
    factors = {item.factor: item.pnl_usd for item in result.factors}

    assert factors["Volatility"] != 0
    assert factors["Skew"] == 0
    assert factors["Term structure"] == 0
    assert_reconciles(result, pnl)


def test_missing_rates_and_dividends_warn_zero_factors_and_reconcile():
    reference = Reference(market_inputs=False)
    result, pnl = run_attribution(
        [vertical()], reference=reference, repository=SurfaceRepository()
    )
    factors = {item.factor: item.pnl_usd for item in result.factors}

    assert factors["Rates"] == factors["Dividends"] == 0
    assert any(
        item.startswith("fallback_rate_held_constant:")
        for item in result.warnings
    )
    assert any(
        item.startswith("fallback_dividend_held_constant:")
        for item in result.warnings
    )
    assert_reconciles(result, pnl)


def test_equity_only_uses_mark_revaluation_and_never_queries_surfaces():
    repository = SurfaceRepository(fail=True)
    result, pnl = run_attribution([equity()], repository=repository)

    assert result.attribution_method == "mark_revaluation"
    assert result.surface_coverage_percent is None
    assert repository.calls == []
    assert {item.factor: item.pnl_usd for item in result.factors}[
        "Underlying movement"
    ] == pytest.approx(20.0)
    assert_reconciles(result, pnl)


def test_missing_execution_spot_keeps_only_edges_in_residual():
    result, pnl = run_attribution(
        [vertical()], repository=SurfaceRepository(complete_surfaces())
    )
    factors = {item.factor: item.pnl_usd for item in result.factors}

    assert factors["Model residual"] != 0
    assert_reconciles(result, pnl)


def test_commission_and_fees_is_the_exact_signed_persisted_total():
    lifecycle = vertical()
    lifecycle.executions[0] = replace(lifecycle.executions[0], fees_usd=-1.23)
    lifecycle.executions[1] = replace(lifecycle.executions[1], fees_usd=-0.27)
    result, pnl = run_attribution(
        [lifecycle], repository=SurfaceRepository(complete_surfaces())
    )

    assert {item.factor: item.pnl_usd for item in result.factors}[
        "Commission & fees"
    ] == -1.5
    assert_reconciles(result, pnl)


def test_fallback_waterfall_isolates_spot_contract_iv_and_elapsed_time():
    occ = parse_occ_symbol(CALL_100)
    adapter = SimpleNamespace(
        prices=lambda option, states, **_kwargs: FakeRegistry().price_many(
            [
                SimpleNamespace(
                    spot=state.spot,
                    strike=float(option.strike),
                    rate=state.rate,
                    dividend_yield=state.dividend_yield,
                    tenor_seconds=(
                        datetime(2026, 9, 18, 20, tzinfo=UTC) - state.at
                    ).total_seconds(),
                )
                for state in states
            ],
            [state.total_iv for state in states],
            [None] * len(states),
        )
    )
    base = OptionMarketState(
        at=BOUNDARIES[0],
        spot=100.0,
        rate=0.04,
        dividend_yield=0.0,
        volatility=0.25,
    )
    cases = (
        (replace(base, spot=102.0), "underlying"),
        (replace(base, volatility=0.30), "volatility"),
        (replace(base, at=BOUNDARIES[1]), "time"),
    )
    for end, expected in cases:
        contributions = _symmetric_waterfall(
            occ,
            base,
            end,
            adapter,
            expiration_at=EXPIRATION_AT,
            factors=("time", "underlying", "rates", "dividends", "volatility"),
        )
        assert abs(contributions[expected]) > 1e-6
        assert all(
            abs(value) < 1e-9
            for factor, value in contributions.items()
            if factor != expected
        )


def test_greek_bridge_uses_percentage_point_units_and_does_not_double_count_gamma():
    start = ExactOptionObservation(
        canonical_occ_symbol=CALL_100,
        at=BOUNDARIES[0],
        mark=5.0,
        provider_iv=0.20,
        provider_delta=0.40,
        provider_gamma=0.10,
        provider_theta_per_day=-0.02,
        provider_vega_per_1pct=0.08,
        provider_rho_per_1pct=0.03,
    )
    end = replace(
        start,
        at=BOUNDARIES[1],
        mark=6.0,
        provider_iv=0.22,
        provider_delta=0.60,
        provider_gamma=0.20,
        provider_theta_per_day=-0.04,
        provider_vega_per_1pct=0.12,
        provider_rho_per_1pct=0.05,
    )

    values = _greek_approximation(
        start,
        end,
        start_spot=100.0,
        end_spot=102.0,
        start_rate=0.03,
        end_rate=0.04,
    )

    assert values == pytest.approx(
        {
            "underlying": 1.0,
            "time": -0.03,
            "volatility": 0.20,
            "rates": 0.04,
        }
    )
    start_only = _greek_approximation(
        start,
        replace(end, provider_delta=None),
        start_spot=100.0,
        end_spot=102.0,
        start_rate=0.03,
        end_rate=0.04,
    )
    assert start_only["underlying"] == pytest.approx(1.0)


def test_iv_inversion_failure_uses_captured_greek_bridge_and_warns():
    class IvFailureRegistry(FakeRegistry):
        def solve_iv_only_many(self, jobs):
            return [SimpleNamespace(iv=None) for _ in jobs]

    result, pnl = run_attribution(
        [vertical()],
        repository=SurfaceRepository(),
        registry=IvFailureRegistry(),
    )
    factors = {item.factor: item.pnl_usd for item in result.factors}

    assert any(
        item.startswith("fallback_greek_bridge_used:")
        for item in result.warnings
    )
    assert any(
        abs(factors[name]) > 0.01
        for name in ("Underlying movement", "Volatility", "Time decay")
    )
    assert factors["Skew"] == factors["Term structure"] == 0.0
    assert_reconciles(result, pnl)


def test_long_and_short_option_intervals_apply_multiplier_exactly_once():
    def single(quantity):
        executions = [
            leg(30, OPEN, CALL_100, quantity, 5.0),
            leg(31, CLOSE, CALL_100, -quantity, 7.0),
        ]
        return Lifecycle(
            account_id="A",
            underlying_symbol="XYZ",
            instrument_type="Options",
            executions=executions,
            inventory={CALL_100: 0.0},
            closed_at=CLOSE,
        )

    long_result, long_pnl = run_attribution(
        [single(1.0)], repository=SurfaceRepository()
    )
    short_result, short_pnl = run_attribution(
        [single(-1.0)], repository=SurfaceRepository()
    )
    long_factors = {item.factor: item.pnl_usd for item in long_result.factors}
    short_factors = {item.factor: item.pnl_usd for item in short_result.factors}

    for factor in (
        "Underlying movement",
        "Volatility",
        "Time decay",
        "Rates",
        "Dividends",
    ):
        assert short_factors[factor] == pytest.approx(-long_factors[factor])
    assert long_pnl == pytest.approx(-short_pnl)
    assert_reconciles(long_result, long_pnl)
    assert_reconciles(short_result, short_pnl)


def test_future_underlying_bar_is_not_used_for_an_earlier_option_session():
    class MissingSpotReference(Reference):
        def option_market_history(self, payload):
            response = super().option_market_history(payload)
            for row in response["data"]:
                row["underlying_price"] = None
            return response

        def bars(self, symbols, *, start, end, remote_query=False):
            return [
                {
                    "symbol": "XYZ",
                    "bars": [
                        {
                            "session_date": "2026-08-05",
                            "timestamp": "2026-08-06T00:00:00+00:00",
                            "close": 999.0,
                        }
                    ],
                }
            ]

    result, pnl = run_attribution(
        [vertical()],
        reference=MissingSpotReference(),
        repository=SurfaceRepository(),
    )
    factors = {item.factor: item.pnl_usd for item in result.factors}

    assert factors["Underlying movement"] == 0.0
    assert factors["Option repricing residual"] == pytest.approx(pnl)
    assert factors["Model residual"] == 0.0
    assert any(
        item.startswith("fallback_underlying_history_unavailable:XYZ:")
        for item in result.warnings
    )
    assert_reconciles(result, pnl)


def test_unpriced_execution_edges_do_not_suppress_daily_fallback_intervals():
    result, pnl = run_attribution([vertical()], repository=SurfaceRepository())
    factors = {item.factor: item.pnl_usd for item in result.factors}

    assert "fallback_contract_iv_revaluation_used" in result.warnings
    assert any(
        item.endswith(":underlying_history_unavailable")
        for item in result.warnings
    )
    assert any(
        abs(factors[name]) > 0.01
        for name in ("Underlying movement", "Volatility", "Time decay")
    )
    assert_reconciles(result, pnl)


def test_same_day_option_trade_uses_execution_interval_and_reports_missing_spot():
    close = OPEN + timedelta(hours=6)
    lifecycle = Lifecycle(
        account_id="A",
        underlying_symbol="XYZ",
        instrument_type="Options",
        executions=[
            leg(40, OPEN, CALL_100, 1.0, 5.0),
            leg(41, close, CALL_100, -1.0, 6.0),
        ],
        inventory={CALL_100: 0.0},
        closed_at=close,
    )
    reference = Reference()
    marks = mark_book([lifecycle], reference)
    pnl = sum(_required_cash(item) for item in lifecycle.executions)
    result = attribute_selection(
        lifecycles=[lifecycle],
        marks=marks,
        option_price_repository=SurfaceRepository(),
        reference_client=reference,
        context={},
        analysis_start=OPEN,
        analysis_end=close,
        lifecycle_pnl=pnl,
        fees=0.0,
        solver_registry=FakeRegistry(),
    )

    assert result.attribution_method == "price_history_fallback"
    assert result.surface_coverage_percent == 0.0
    assert any(item.endswith(":underlying_history_unavailable") for item in result.warnings)
    assert_reconciles(result, pnl)


def test_execution_fill_edges_are_revalued_for_both_calendar_sides():
    class ExecutionPriceReference(Reference):
        def historical_minute_bars(self, symbol, *, start_at, end_at):
            spot = 99.0 if end_at.date() == OPEN.date() else 101.0
            return [
                {
                    "timestamp": (end_at - timedelta(minutes=2)).isoformat(),
                    "close": spot,
                }
            ]

    class ExactLinearRegistry(FakeRegistry):
        @staticmethod
        def _base(request):
            days = request.tenor_seconds / 86400.0
            return (
                0.02 * request.spot
                + 0.1 * request.rate
                - 0.05 * request.dividend_yield
                - 0.001 * days
            )

        def solve_iv_only_many(self, jobs):
            return [
                SimpleNamespace(
                    iv=(float(job.target_price) - self._base(job.request)) / 10.0
                )
                for job in jobs
            ]

        def price_many(self, requests, volatilities, methods, requested_dtypes=None):
            return [
                self._base(request) + 10.0 * volatility
                for request, volatility in zip(requests, volatilities)
            ]

    result, pnl = run_attribution(
        [vertical()],
        reference=ExecutionPriceReference(),
        repository=SurfaceRepository(),
        registry=ExactLinearRegistry(),
    )
    factors = {item.factor: item.pnl_usd for item in result.factors}

    assert factors["Model residual"] == pytest.approx(0.0, abs=0.01)
    assert any(
        abs(factors[name]) > 0.01
        for name in ("Underlying movement", "Volatility", "Time decay")
    )
    assert not any(
        item.endswith(":opening_execution_edge")
        or item.endswith(":closing_execution_edge")
        for item in result.warnings
    )
    assert_reconciles(result, pnl)
