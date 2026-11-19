from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from ..execution import execution_quality, spread_prices
from ..models import CandidateDraft, CandidateLeg, ContractRecord, ScreeningRequest
from ..strategies.butterfly import ButterflyStrategy
from ..strategies.calendar import CalendarStrategy
from ..strategies.diagonal import DiagonalStrategy
from ..strategies.double_calendar import DoubleCalendarStrategy
from ..strategies.iron_condor import IronCondorStrategy
from ..strategies.vertical import VerticalStrategy
from ..config import ScreenerConfig


UTC = timezone.utc
NOW = datetime(2026, 7, 31, 18, 0, tzinfo=UTC)


def contract(
    contract_id: int,
    strike: float,
    dte: int,
    put_call: str = "CALL",
    *,
    bid: float = 2.0,
    ask: float = 2.2,
    delta: float = 0.40,
) -> ContractRecord:
    return ContractRecord(
        contract_id=contract_id,
        option_symbol=f"OPT{contract_id}",
        put_call=put_call,
        strike=strike,
        expiration_at=NOW + timedelta(days=dte),
        dte=float(dte),
        exercise_type="AMERICAN",
        multiplier=100.0,
        is_mini=False,
        is_nonstandard=False,
        symbol_map_mismatch=False,
        bid=bid,
        ask=ask,
        bid_size=20,
        ask_size=25,
        volume=100,
        open_interest=500,
        quote_time=NOW,
        effective_iv=0.25,
        delta=delta,
        gamma=0.01,
        theta=-0.03,
        vega=0.08,
        rho=0.01,
        iv_value_source="result",
        greeks_value_source="result",
        result_origin="calculated",
        quality_flags=0,
        fitted_iv=0.24,
        iv_residual=0.01,
        pricing_method="tian",
    )


def test_hand_calculated_synthetic_prices_and_signs():
    long = contract(1, 100, 30, bid=4.0, ask=4.4)
    short = contract(2, 105, 30, bid=1.8, ask=2.0)
    candidate = CandidateDraft(
        strategy="vertical",
        variant="call_debit_vertical",
        outlook="bullish",
        legs=(CandidateLeg("long", long), CandidateLeg("short", short)),
    )
    prices = spread_prices(candidate)
    assert prices.synthetic_bid == pytest.approx(2.0)
    assert prices.synthetic_midpoint == pytest.approx(2.3)
    assert prices.synthetic_ask == pytest.approx(2.6)
    assert prices.entry("natural") > 0


def test_vertical_generation_sides_and_exact_payoff():
    strategy = VerticalStrategy()
    request = ScreeningRequest(
        underlying_symbol="QQQ",
        enabled_strategies=["vertical"],
        surface_relative_value=False,
    )
    generated = strategy.generate(
        [contract(1, 100, 30), contract(2, 105, 30)],
        102.0,
        request,
        20,
    )
    variants = {item.variant: item for item in generated}
    assert set(variants) == {"call_debit_vertical", "call_credit_vertical"}
    debit = variants["call_debit_vertical"]
    assert [(leg.side, leg.contract.strike) for leg in debit.legs] == [
        ("long", 100),
        ("short", 105),
    ]
    payoff = strategy.payoff_metrics(debit, 2.0)
    assert payoff.exact_maximum_profit == pytest.approx(300.0)
    assert payoff.exact_maximum_loss == pytest.approx(200.0)
    assert payoff.breakeven == pytest.approx(102.0)


def test_put_vertical_sides_are_correct():
    strategy = VerticalStrategy()
    request = ScreeningRequest(
        underlying_symbol="QQQ",
        enabled_strategies=["vertical"],
        option_types=["put"],
        surface_relative_value=False,
    )
    generated = strategy.generate(
        [contract(1, 95, 30, "PUT", delta=-0.3), contract(2, 100, 30, "PUT", delta=-0.5)],
        98.0,
        request,
        20,
    )
    debit = next(item for item in generated if item.variant == "put_debit_vertical")
    assert [(leg.side, leg.contract.strike) for leg in debit.legs] == [
        ("short", 95),
        ("long", 100),
    ]


def test_calendar_and_diagonal_structure_rules():
    near = contract(1, 100, 21)
    far_same = contract(2, 100, 49)
    far_different = contract(3, 105, 49)
    request = ScreeningRequest(underlying_symbol="QQQ", surface_relative_value=False)
    calendars = CalendarStrategy().generate([near, far_same, far_different], 102, request, 20)
    diagonals = DiagonalStrategy().generate([near, far_same, far_different], 102, request, 20)
    assert calendars
    assert all(CalendarStrategy().structurally_valid(item) for item in calendars)
    assert diagonals
    assert all(DiagonalStrategy().structurally_valid(item) for item in diagonals)
    assert all(item.legs[0].contract.strike != item.legs[1].contract.strike for item in diagonals)


def test_balanced_butterfly_generation_quantity_and_exact_payoff():
    request = ScreeningRequest(
        underlying_symbol="QQQ",
        enabled_strategies=["butterfly"],
        option_types=["call"],
        surface_relative_value=False,
    )
    strategy = ButterflyStrategy()
    generated = strategy.generate(
        [contract(1, 95, 30), contract(2, 100, 30), contract(3, 105, 30)],
        100.0,
        request,
        20,
    )
    assert len(generated) == 1
    candidate = generated[0]
    assert strategy.structurally_valid(candidate)
    assert [(leg.side, leg.quantity, leg.contract.strike) for leg in candidate.legs] == [
        ("long", 1, 95),
        ("short", 2, 100),
        ("long", 1, 105),
    ]
    payoff = strategy.payoff_metrics(candidate, 1.0)
    assert payoff.exact_maximum_profit == pytest.approx(400.0)
    assert payoff.exact_maximum_loss == pytest.approx(100.0)
    assert payoff.breakevens == pytest.approx((96.0, 104.0))


def test_short_iron_condor_generation_and_exact_payoff():
    request = ScreeningRequest(
        underlying_symbol="QQQ",
        enabled_strategies=["iron_condor"],
        surface_relative_value=False,
    )
    strategy = IronCondorStrategy()
    generated = strategy.generate(
        [
            contract(1, 90, 30, "PUT", delta=-0.10),
            contract(2, 95, 30, "PUT", delta=-0.25),
            contract(3, 105, 30, "CALL", delta=0.25),
            contract(4, 110, 30, "CALL", delta=0.10),
        ],
        100.0,
        request,
        20,
    )
    assert len(generated) == 1
    candidate = generated[0]
    assert strategy.structurally_valid(candidate)
    payoff = strategy.payoff_metrics(candidate, -2.0)
    assert payoff.exact_maximum_profit == pytest.approx(200.0)
    assert payoff.exact_maximum_loss == pytest.approx(300.0)
    assert payoff.breakevens == pytest.approx((93.0, 107.0))


def test_double_calendar_pairs_matching_contracts_across_two_terms():
    request = ScreeningRequest(
        underlying_symbol="QQQ",
        enabled_strategies=["double_calendar"],
        surface_relative_value=False,
    )
    strategy = DoubleCalendarStrategy()
    generated = strategy.generate(
        [
            contract(1, 95, 21, "PUT", delta=-0.30),
            contract(2, 105, 21, "CALL", delta=0.30),
            contract(3, 95, 49, "PUT", delta=-0.35),
            contract(4, 105, 49, "CALL", delta=0.35),
        ],
        100.0,
        request,
        20,
    )
    assert len(generated) == 1
    candidate = generated[0]
    assert candidate.variant == "mixed_double_calendar"
    assert strategy.structurally_valid(candidate)
    assert [(leg.side, leg.contract.dte) for leg in candidate.legs] == [
        ("short", 21.0),
        ("short", 21.0),
        ("long", 49.0),
        ("long", 49.0),
    ]


def test_diagonal_admission_is_order_independent_and_later_groups_compete():
    near = contract(10, 100, 14)
    early_far = contract(11, 105, 30)
    later_far = contract(12, 105, 60)
    request = ScreeningRequest(
        underlying_symbol="QQQ",
        enabled_strategies=["diagonal"],
        surface_relative_value=False,
    )
    strategy = DiagonalStrategy()
    forward = strategy.generate(
        [near, early_far, later_far], 102, request, 1
    )
    reverse = strategy.generate(
        [later_far, early_far, near], 102, request, 1
    )
    assert [
        tuple(leg.contract.contract_id for leg in item.legs) for item in forward
    ] == [
        tuple(leg.contract.contract_id for leg in item.legs) for item in reverse
    ]
    assert forward[0].legs[1].contract.contract_id == later_far.contract_id


def test_execution_quality_distinguishes_tight_and_wide_markets():
    tight = CandidateDraft(
        strategy="vertical",
        variant="call_debit_vertical",
        outlook="bullish",
        legs=(
            CandidateLeg("long", contract(1, 100, 30, bid=4.0, ask=4.1)),
            CandidateLeg("short", contract(2, 105, 30, bid=2.0, ask=2.1)),
        ),
    )
    wide = CandidateDraft(
        strategy="vertical",
        variant="call_debit_vertical",
        outlook="bullish",
        legs=(
            CandidateLeg("long", contract(3, 100, 30, bid=1.0, ask=3.0)),
            CandidateLeg("short", contract(4, 105, 30, bid=0.5, ask=2.5)),
        ),
    )
    config = ScreenerConfig()
    tight_quality = execution_quality(tight, spread_prices(tight), config)
    wide_quality = execution_quality(wide, spread_prices(wide), config)
    assert tight_quality.score > wide_quality.score
    assert tight_quality.rating == "good"
    assert wide_quality.rating == "poor"


def test_execution_quality_marks_stale_quotes_poor():
    stale_contract = replace(
        contract(8, 100, 30), quote_time=NOW - timedelta(minutes=10)
    )
    candidate = CandidateDraft(
        strategy="vertical",
        variant="call_debit_vertical",
        outlook="bullish",
        legs=(
            CandidateLeg("long", stale_contract),
            CandidateLeg("short", contract(9, 105, 30)),
        ),
    )
    quality = execution_quality(
        candidate,
        spread_prices(candidate),
        ScreenerConfig(),
        as_of=NOW,
    )
    assert quality.rating == "poor"
    assert quality.per_leg_reasons[stale_contract.option_symbol] == ["STALE_QUOTE"]
