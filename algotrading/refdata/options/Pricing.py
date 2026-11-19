"""Shared repricing configuration and deterministic method routing."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping

from algotrading.refdata.options.Common import OPTIONS_CALCS_GREEKS_PREFIX
from algotrading.refdata.options.solvers.Types import RepricingMethod


@dataclass(frozen=True)
class MethodSelection:
    method: RepricingMethod
    reason: str


@dataclass(frozen=True)
class PricingMethodRouting:
    default_method: RepricingMethod
    short_tenor_method: RepricingMethod
    short_tenor_threshold_days: float

    @property
    def short_tenor_threshold_seconds(self) -> float:
        """Convert the configured days value once at the routing boundary."""
        return self.short_tenor_threshold_days * 86400.0


_REMOVED_ROUTING_KEYS = (
    "options_calcs.greeks.zero_dte_method",
    "options_calcs.greeks.zero_dte_threshold_seconds",
)


def configured_methods(
    context: Mapping[str, Any],
) -> PricingMethodRouting:
    prefix = OPTIONS_CALCS_GREEKS_PREFIX
    present_removed = [key for key in _REMOVED_ROUTING_KEYS if key in context]
    if present_removed:
        raise ValueError(
            "removed options routing configuration key(s): "
            + ", ".join(present_removed)
            + "; use options_calcs.greeks.short_tenor_method and "
            "options_calcs.greeks.short_tenor_threshold_days"
        )
    method = RepricingMethod.parse(context.get(prefix + "method", "crank_nicolson"))
    short_tenor = RepricingMethod.parse(
        context.get(prefix + "short_tenor_method", "tian")
    )
    raw_threshold = context.get(prefix + "short_tenor_threshold_days", 3.0)
    if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, numbers.Real):
        raise ValueError(
            "options_calcs.greeks.short_tenor_threshold_days must be numeric"
        )
    threshold_days = float(raw_threshold)
    if method is RepricingMethod.NONE:
        raise ValueError("options_calcs.greeks.method cannot be none")
    if short_tenor not in (RepricingMethod.CRR, RepricingMethod.TIAN):
        raise ValueError(
            "options_calcs.greeks.short_tenor_method must be crr or tian"
        )
    if not math.isfinite(threshold_days) or threshold_days <= 0.0:
        raise ValueError(
            "options_calcs.greeks.short_tenor_threshold_days must be positive and finite"
        )
    return PricingMethodRouting(method, short_tenor, threshold_days)


def select_method(
    exercise_type: str,
    tenor_seconds: float,
    *,
    routing: PricingMethodRouting,
) -> MethodSelection:
    if str(exercise_type).strip().upper() in {"E", "EUROPEAN"}:
        return MethodSelection(RepricingMethod.NONE, "source_only_european")
    if not math.isfinite(float(tenor_seconds)) or tenor_seconds <= 0:
        return MethodSelection(RepricingMethod.NONE, "expired_or_invalid_tenor")
    if tenor_seconds < routing.short_tenor_threshold_seconds:
        return MethodSelection(routing.short_tenor_method, "short_tenor_override")
    return MethodSelection(routing.default_method, "configured_default")
