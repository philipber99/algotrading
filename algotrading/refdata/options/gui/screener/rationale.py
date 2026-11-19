"""Deterministic measured-factor rationales; no language model is used."""

from __future__ import annotations

from .models import CandidateDraft, RationaleResult


def build_rationale(
    candidate: CandidateDraft,
    *,
    surface_factors: dict[str, float],
    execution_rating: str,
    expected_return: float,
) -> RationaleResult:
    reasons: list[tuple[str, float, str]] = []
    short_edge = surface_factors.get("short_leg_above_surface", 0.0)
    long_edge = surface_factors.get("long_leg_below_surface", 0.0)
    if short_edge > 0.0025:
        reasons.append(
            (
                "SHORT_LEG_ABOVE_SURFACE",
                short_edge,
                f"the short leg is {short_edge * 100:.1f} IV points above fitted value",
            )
        )
    if long_edge > 0.0025:
        reasons.append(
            (
                "LONG_LEG_BELOW_SURFACE",
                long_edge,
                f"the long leg is {long_edge * 100:.1f} IV points below fitted value",
            )
        )
    if expected_return > 0:
        reasons.append(
            (
                "POSITIVE_MODELED_RETURN_ON_RISK",
                expected_return,
                f"modeled expected return on risk is {expected_return * 100:.1f}%",
            )
        )
    execution_value = {"poor": 0.0, "questionable": 0.5, "good": 1.0}[execution_rating]
    reasons.append(
        (
            f"{execution_rating.upper()}_EXECUTION_QUALITY",
            execution_value,
            f"the combined market has {execution_rating} execution quality",
        )
    )
    reasons.sort(key=lambda item: (-abs(item[1]), item[0]))
    selected = reasons[:3]
    if len(selected) == 1:
        body = selected[0][2]
    else:
        body = ", ".join(item[2] for item in selected[:-1]) + ", and " + selected[-1][2]
    sentence = f"Ranks on the selected model because {body}."
    return RationaleResult(
        sentence=sentence,
        reason_codes=[item[0] for item in selected],
        values={item[0]: item[1] for item in selected},
    )


__all__ = ["build_rationale"]
