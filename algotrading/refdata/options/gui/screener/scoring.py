"""Transparent component scoring and deterministic ranking helpers."""

from __future__ import annotations

import math
from typing import Mapping

from .models import CandidateDraft, ScoreResult


SCORE_WEIGHTS = {
    "expected_value_return": 0.35,
    "probability": 0.20,
    "execution_quality": 0.20,
    "robustness": 0.15,
    "capital_efficiency": 0.10,
}


def weighted_score(components: Mapping[str, float]) -> float:
    missing = set(SCORE_WEIGHTS).difference(components)
    if missing:
        raise ValueError(f"Missing required score components: {sorted(missing)}")
    values: dict[str, float] = {}
    for name in SCORE_WEIGHTS:
        value = float(components[name])
        if not math.isfinite(value):
            raise ValueError(f"Score component {name} is not finite")
        values[name] = max(0.0, min(1.0, value))
    return sum(SCORE_WEIGHTS[name] * values[name] for name in SCORE_WEIGHTS)


def score_candidate(
    *,
    expected_pnl: float,
    capital_at_risk: float | None,
    probability_of_profit: float,
    target_probability: float,
    execution_score: float,
    robustness: float,
    target_profit: float,
) -> ScoreResult:
    risk = capital_at_risk if capital_at_risk and capital_at_risk > 0 else None
    expected_return = expected_pnl / risk if risk else 0.0
    expected_value_return = 0.5 + 0.5 * math.tanh(expected_return * 3.0)
    probability = (probability_of_profit + target_probability) / 2.0
    capital_efficiency = (
        max(0.0, min(1.0, target_profit / risk)) if risk else 0.0
    )
    components = {
        "expected_value_return": expected_value_return,
        "probability": probability,
        "execution_quality": execution_score,
        "robustness": robustness,
        "capital_efficiency": capital_efficiency,
    }
    return ScoreResult(overall=weighted_score(components), **components)


def surface_relative_factors(candidate: CandidateDraft) -> dict[str, float]:
    long_below = 0.0
    short_above = 0.0
    residual_edge = 0.0
    for leg in candidate.legs:
        residual = float(leg.contract.iv_residual or 0.0)
        if leg.side == "long":
            long_below += leg.quantity * max(0.0, -residual)
            residual_edge -= residual * leg.quantity
        else:
            short_above += leg.quantity * max(0.0, residual)
            residual_edge += residual * leg.quantity
    net_theta = sum(
        leg.sign * leg.quantity * float(leg.contract.theta or 0)
        for leg in candidate.legs
    )
    net_vega = sum(
        leg.sign * leg.quantity * float(leg.contract.vega or 0)
        for leg in candidate.legs
    )
    short_legs = [leg for leg in candidate.legs if leg.side == "short"]
    long_legs = [leg for leg in candidate.legs if leg.side == "long"]
    front_back_residual = 0.0
    front_back_variance = 0.0
    skew_relationship = 0.0
    if short_legs and long_legs:
        short = min(short_legs, key=lambda leg: leg.contract.expiration_at)
        long = max(long_legs, key=lambda leg: leg.contract.expiration_at)
        front_back_residual = float(short.contract.iv_residual or 0.0) - float(
            long.contract.iv_residual or 0.0
        )
        short_variance = float(short.contract.effective_iv or 0.0) ** 2 * max(
            short.contract.dte, 0.0
        ) / 365.25
        long_variance = float(long.contract.effective_iv or 0.0) ** 2 * max(
            long.contract.dte, 0.0
        ) / 365.25
        front_back_variance = long_variance - short_variance
        strike_distance = long.contract.strike - short.contract.strike
        if strike_distance:
            skew_relationship = -math.copysign(
                abs(front_back_residual), strike_distance
            )
    return {
        "long_leg_below_surface": long_below,
        "short_leg_above_surface": short_above,
        "net_iv_residual_edge": residual_edge,
        "front_back_residual_edge": front_back_residual,
        "front_back_total_variance": front_back_variance,
        "skew_relationship": skew_relationship,
        "theta_vega_alignment": math.tanh((net_theta + net_vega) / 10.0),
    }


def fast_prescore(
    candidate: CandidateDraft,
    execution_score: float,
    reward_risk_hint: float,
) -> float:
    factors = surface_relative_factors(candidate)
    edge = 0.5 + 0.5 * math.tanh(factors["net_iv_residual_edge"] * 10.0)
    return max(
        0.0,
        min(1.0, 0.45 * execution_score + 0.35 * edge + 0.20 * reward_risk_hint),
    )


__all__ = [
    "SCORE_WEIGHTS",
    "fast_prescore",
    "score_candidate",
    "surface_relative_factors",
    "weighted_score",
]
