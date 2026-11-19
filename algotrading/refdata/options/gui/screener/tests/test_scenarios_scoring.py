from __future__ import annotations

import numpy as np
import pytest

from ..models import CandidateDraft, CandidateLeg
from ..scenarios import (
    deterministic_seed,
    distribution_metrics,
    geometric_paths,
    interpolate_price_grid,
    PricingResources,
    residual_decay,
    static_hedge_shares,
)
from ..scoring import SCORE_WEIGHTS, weighted_score
from .test_strategies_execution import contract


def test_paths_and_seed_are_deterministic():
    request = {"underlying_symbol": "QQQ", "mode": "zero"}
    seed = deterministic_seed("snapshot-1", request)
    first = geometric_paths(
        spot=100,
        sessions=5,
        path_count=10,
        annualized_drift=0,
        annualized_volatility=0.2,
        seed=seed,
    )
    second = geometric_paths(
        spot=100,
        sessions=5,
        path_count=10,
        annualized_drift=0,
        annualized_volatility=0.2,
        seed=seed,
    )
    assert np.array_equal(first, second)
    assert deterministic_seed("snapshot-2", request) != seed


def test_residual_half_life_is_exact():
    assert residual_decay(0.04, 5, 5) == pytest.approx(0.02)
    assert residual_decay(0.04, 10, 5) == pytest.approx(0.01)


def test_distribution_metrics_are_hand_checkable():
    pop, target, ev, percentiles, holding, basis = distribution_metrics(
        np.asarray([-10.0, 0.0, 10.0, 20.0]),
        np.asarray([False, False, True, True]),
        np.asarray([5, 5, 2, 4]),
        minimum_hits=2,
    )
    assert pop == pytest.approx(0.5)
    assert target == pytest.approx(0.5)
    assert ev == pytest.approx(5.0)
    assert holding == pytest.approx(3.0)
    assert basis == "target_hit_median"
    assert percentiles["p50"] == pytest.approx(5.0)


def test_static_hedge_neutralizes_once_with_rounding():
    long = contract(1, 100, 30, delta=0.43)
    short = contract(2, 105, 30, delta=0.18)
    candidate = CandidateDraft(
        strategy="vertical",
        variant="call_debit_vertical",
        outlook="bullish",
        legs=(CandidateLeg("long", long), CandidateLeg("short", short)),
    )
    shares = static_hedge_shares(candidate)
    assert shares == -25
    initial_option_delta = (0.43 - 0.18) * 100
    assert abs(initial_option_delta + shares) <= 0.5


def test_score_weights_apply_exactly_and_require_every_component():
    components = {name: 1.0 for name in SCORE_WEIGHTS}
    assert weighted_score(components) == pytest.approx(1.0)
    components["expected_value_return"] = 0.0
    assert weighted_score(components) == pytest.approx(0.65)
    with pytest.raises(ValueError):
        weighted_score({"probability": 1.0})


def test_price_grid_interpolation_is_bounded_and_exact_for_linear_values():
    values = interpolate_price_grid(
        np.asarray([90.0, 100.0, 110.0]),
        np.asarray([1.0, 3.0, 5.0]),
        np.asarray([95.0, 105.0]),
    )
    assert values.tolist() == pytest.approx([2.0, 4.0], abs=1.0e-12)
    with pytest.raises(ValueError, match="extrapolate"):
        interpolate_price_grid(
            np.asarray([90.0, 100.0]),
            np.asarray([1.0, 2.0]),
            np.asarray([101.0]),
        )


def test_pricing_resources_default_all_positive_tenors_to_tian():
    resources = PricingResources({}, registry=object())
    assert resources.pricing_method("AMERICAN", 30 * 86400) == "tian"
    assert resources.pricing_method("AMERICAN", 2 * 86400) == "tian"
    assert resources.pricing_method("EUROPEAN", 30 * 86400) == "tian"


def test_pricing_resources_configured_mode_retains_exercise_aware_routes():
    resources = PricingResources(
        {"options_chain.gui.screener.pricing_method": "configured"},
        registry=object(),
    )
    assert resources.pricing_method("AMERICAN", 30 * 86400) == "crank_nicolson"
    assert resources.pricing_method("AMERICAN", 2 * 86400) == "tian"
    assert resources.pricing_method("EUROPEAN", 30 * 86400) == "none"
