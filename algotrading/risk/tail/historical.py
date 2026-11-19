"""Deterministic weighted empirical VaR, ES, and reconciling contributions."""

from __future__ import annotations

import math
from typing import Any, Sequence


def exponential_weights(count: int, half_life: float) -> list[float]:
    if count < 1 or half_life <= 0:
        raise ValueError("count and half_life must be positive")
    decay = math.log(2) / half_life
    # Input observations run oldest to newest, so recent sessions carry more weight.
    raw = [math.exp(-decay * (count - 1 - index)) for index in range(count)]
    total = sum(raw)
    return [value / total for value in raw]


def weighted_quantile(values: Sequence[float], weights: Sequence[float], alpha: float) -> float:
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be non-empty and aligned")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be greater than zero and at most one")
    pairs = [(float(value), float(weight)) for value, weight in zip(values, weights)]
    if any(not math.isfinite(value) for value, _ in pairs):
        raise ValueError("values must be finite")
    if any(not math.isfinite(weight) or weight < 0 for _, weight in pairs):
        raise ValueError("weights must be finite and non-negative")
    pairs.sort()
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        raise ValueError("weights must have positive mass")
    threshold = alpha * total
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def historical_var_es(
    unit_pnl_rows: Sequence[Sequence[float]],
    *,
    confidence: float = 0.95,
    half_life_sessions: float = 63,
    unit_ids: Sequence[str] | None = None,
    dates: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not unit_pnl_rows:
        raise ValueError("at least one historical observation is required")
    width = len(unit_pnl_rows[0])
    if width < 1 or any(len(row) != width for row in unit_pnl_rows):
        raise ValueError("historical scenario matrix must be rectangular")
    if any(not math.isfinite(float(value)) for row in unit_pnl_rows for value in row):
        raise ValueError("historical scenario matrix must not contain missing/nonfinite values")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be greater than zero and less than one")
    weights = exponential_weights(len(unit_pnl_rows), half_life_sessions)
    portfolio_pnl = [sum(float(value) for value in row) for row in unit_pnl_rows]
    # Gains contribute zero loss without removing their probability mass.
    losses = [max(0.0, -value) for value in portfolio_pnl]
    var = weighted_quantile(losses, weights, confidence)
    tail_weights, tail_weight = _fractional_tail_weights(
        losses, weights, confidence
    )
    if tail_weight <= 0:
        raise ValueError("tail has no weight")
    es = sum(loss * weight for loss, weight in zip(losses, tail_weights)) / tail_weight
    identifiers = list(unit_ids or [f"unit-{index}" for index in range(width)])
    if len(identifiers) != width:
        raise ValueError("unit_ids length must match matrix width")
    # Attribute the same portfolio tail to units so contributions reconcile to ES.
    contributions = []
    for column, unit_id in enumerate(identifiers):
        contribution = sum(
            (
                -float(unit_pnl_rows[index][column])
                if portfolio_pnl[index] < 0
                else 0.0
            ) * tail_weights[index]
            for index in range(len(unit_pnl_rows))
        ) / tail_weight
        standalone_losses = [max(0.0, -float(row[column])) for row in unit_pnl_rows]
        standalone_var = weighted_quantile(standalone_losses, weights, confidence)
        standalone_tail_weights, standalone_weight = _fractional_tail_weights(
            standalone_losses, weights, confidence
        )
        standalone_es = sum(
            loss * weight
            for loss, weight in zip(standalone_losses, standalone_tail_weights)
        ) / standalone_weight
        contributions.append(
            {"unit_id": unit_id, "es_contribution_usd": contribution, "standalone_es_usd": standalone_es}
        )
    standalone_total = sum(item["standalone_es_usd"] for item in contributions)
    benefit = standalone_total - es
    worst = sorted(range(len(losses)), key=lambda index: losses[index], reverse=True)[:10]
    return {
        "method": "current_portfolio_filtered_historical_simulation_v4",
        "methodology_version": "current_portfolio_filtered_historical_simulation_v4",
        "confidence": confidence,
        "observations": len(unit_pnl_rows),
        "half_life_sessions": half_life_sessions,
        "var_usd": var,
        "expected_shortfall_usd": es,
        "contributions": contributions,
        "contribution_sum_usd": sum(item["es_contribution_usd"] for item in contributions),
        "diversification_benefit_usd": benefit,
        "diversification_benefit_percent": benefit / standalone_total if standalone_total else None,
        "diversification_label": "Diversification benefit" if benefit >= 0 else "Concentration penalty",
        "grouping_basis": "economic_risk_unit",
        "tail_probability_mass": tail_weight,
        "configured_tail_probability_mass": 1.0 - confidence,
        "observation_weights": [
            {
                "date": dates[index] if dates and index < len(dates) else None,
                "weight": weights[index],
                "tail_weight": tail_weights[index],
                "pnl_usd": portfolio_pnl[index],
                "loss_usd": losses[index],
            }
            for index in range(len(weights))
        ],
        "worst_observations": [
            {
                "date": dates[index] if dates and index < len(dates) else None,
                "loss_usd": losses[index],
                "weight": weights[index],
                "tail_weight": tail_weights[index],
            }
            for index in worst
        ],
    }


def _fractional_tail_weights(
    losses: Sequence[float], weights: Sequence[float], confidence: float
) -> tuple[list[float], float]:
    """Consume exactly ``1-confidence`` mass from the worst observations."""

    if len(losses) != len(weights) or not losses:
        raise ValueError("losses and weights must be non-empty and aligned")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be greater than zero and less than one")
    pairs = [(float(loss), float(weight), index) for index, (loss, weight) in enumerate(zip(losses, weights))]
    if any(not math.isfinite(loss) for loss, _, _ in pairs):
        raise ValueError("losses must be finite")
    if any(not math.isfinite(weight) or weight < 0 for _, weight, _ in pairs):
        raise ValueError("weights must be finite and non-negative")
    total = sum(weight for _, weight, _ in pairs)
    if total <= 0:
        raise ValueError("weights must have positive mass")
    target = (1.0 - confidence) * total
    remaining = target
    included = [0.0] * len(pairs)
    for _loss, weight, index in sorted(pairs, key=lambda row: (-row[0], row[2])):
        if remaining <= max(1e-15, target * 1e-15):
            break
        # Split the boundary observation to consume exactly the configured tail mass.
        consumed = min(weight, remaining)
        included[index] = consumed
        remaining -= consumed
    if remaining > max(1e-12, target * 1e-10):
        raise ValueError("tail probability mass could not be consumed")
    return included, target


__all__ = [
    "exponential_weights",
    "historical_var_es",
    "weighted_quantile",
]
