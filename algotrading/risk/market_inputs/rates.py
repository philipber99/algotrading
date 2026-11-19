"""Interpolate FRED simple annual anchors and convert for Tian."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class InterpolatedRate:
    tenor_days: float
    lower_series_id: str
    upper_series_id: str
    lower_observation_date: date
    upper_observation_date: date
    interpolation_weight: float
    simple_annual_rate_decimal: float
    continuous_annual_rate_decimal: float
    curve_id: str
    curve_version: str
    maximum_observation_age_days: int
    warnings: tuple[str, ...] = ()


def interpolate_risk_free_rate(
    curve: Mapping[str, Any],
    *,
    valuation_date: date,
    tenor_days: float,
    max_age_days: int = 5,
    allow_flat_extrapolation: bool = True,
) -> InterpolatedRate:
    if not math.isfinite(tenor_days) or tenor_days < 0:
        raise ValueError("tenor_days must be finite and non-negative")
    raw_anchors = curve.get("anchors")
    if not isinstance(raw_anchors, Sequence):
        raise ValueError("curve anchors are unavailable")
    anchors = []
    for raw in raw_anchors:
        if not isinstance(raw, Mapping) or raw.get("status") != "ok":
            continue
        try:
            observation = date.fromisoformat(str(raw["observation_date"]))
            days = int(raw["tenor_days"])
            rate = float(raw["simple_annual_rate_decimal"])
        except (KeyError, TypeError, ValueError):
            continue
        # Discard stale and future observations before interpolating the curve.
        age = (valuation_date - observation).days
        if age < 0 or age > max_age_days or not math.isfinite(rate) or rate <= -1:
            continue
        anchors.append((days, str(raw["series_id"]), observation, rate, age))
    anchors.sort(key=lambda item: (item[0], item[1]))
    if not anchors:
        raise ValueError("no current usable curve anchors")
    lower = anchors[0]
    upper = anchors[-1]
    warnings: list[str] = []
    # Use endpoint rates outside the observed tenor range.
    if tenor_days <= anchors[0][0]:
        lower = upper = anchors[0]
    elif tenor_days >= anchors[-1][0]:
        if tenor_days > anchors[-1][0]:
            if not allow_flat_extrapolation:
                raise ValueError("tenor exceeds longest curve anchor")
            warnings.append("curve_flat_extrapolation")
        lower = upper = anchors[-1]
    else:
        for candidate_lower, candidate_upper in zip(anchors, anchors[1:]):
            if candidate_lower[0] <= tenor_days <= candidate_upper[0]:
                lower, upper = candidate_lower, candidate_upper
                break
    # Interpolate simple annual rates before converting to continuous compounding.
    if lower[0] == upper[0]:
        weight = 0.0
        simple = lower[3]
    else:
        weight = (tenor_days - lower[0]) / (upper[0] - lower[0])
        simple = lower[3] + weight * (upper[3] - lower[3])
    if simple <= -1:
        raise ValueError("simple annual rate must be greater than -1")
    return InterpolatedRate(
        tenor_days=tenor_days,
        lower_series_id=lower[1],
        upper_series_id=upper[1],
        lower_observation_date=lower[2],
        upper_observation_date=upper[2],
        interpolation_weight=weight,
        simple_annual_rate_decimal=simple,
        continuous_annual_rate_decimal=math.log1p(simple),
        curve_id=str(curve.get("curve_id") or "unknown"),
        curve_version=str(curve.get("curve_version") or "unknown"),
        maximum_observation_age_days=max(lower[4], upper[4]),
        warnings=tuple(warnings),
    )


__all__ = ["InterpolatedRate", "interpolate_risk_free_rate"]
