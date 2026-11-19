"""Extensible strategy protocol and common payoff contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Iterable

from ..models import CandidateDraft, ContractRecord, ScreeningRequest, StrategyName


@dataclass(frozen=True)
class PayoffMetrics:
    exact_maximum_profit: float | None
    exact_maximum_loss: float | None
    breakeven: float | None
    basis: str
    capital_at_risk: float | None
    breakevens: tuple[float, ...] = ()


class StrategyDefinition(ABC):
    name: StrategyName

    @abstractmethod
    def generate(
        self,
        contracts: Iterable[ContractRecord],
        spot: float,
        request: ScreeningRequest,
        limit: int,
    ) -> list[CandidateDraft]:
        """Generate deterministic bounded candidates."""

    @abstractmethod
    def structurally_valid(self, candidate: CandidateDraft) -> bool:
        """Validate the strategy's invariant independently of generation."""

    @abstractmethod
    def payoff_metrics(
        self,
        candidate: CandidateDraft,
        entry_value: float,
    ) -> PayoffMetrics:
        """Return exact expiration metrics where mathematically available."""

    def planned_holding_sessions(
        self,
        candidate: CandidateDraft,
        request: ScreeningRequest,
        configured_maximum: int,
    ) -> int:
        override = request.exit_policy.maximum_holding_sessions
        maximum = min(configured_maximum, override or configured_maximum)
        return max(1, maximum)


class StrategyRegistry:
    def __init__(self, strategies: Iterable[StrategyDefinition]) -> None:
        self._strategies = {strategy.name: strategy for strategy in strategies}
        required = {
            "vertical",
            "calendar",
            "diagonal",
            "butterfly",
            "iron_condor",
            "double_calendar",
        }
        missing = required.difference(self._strategies)
        if missing:
            raise ValueError(f"Strategy registry is missing {sorted(missing)}")

    def get(self, name: StrategyName) -> StrategyDefinition:
        return self._strategies[name]

    def enabled(self, names: Iterable[StrategyName]) -> tuple[StrategyDefinition, ...]:
        return tuple(self._strategies[name] for name in names)


def net_delta(candidate: CandidateDraft) -> float:
    return sum(
        leg.sign * leg.quantity * float(leg.contract.delta or 0)
        for leg in candidate.legs
    )


def outlook_from_delta(delta: float) -> str:
    if delta > 0.10:
        return "bullish"
    if delta < -0.10:
        return "bearish"
    return "neutral"


def exact_defined_risk_payoff(
    candidate: CandidateDraft,
    entry_value: float,
) -> PayoffMetrics:
    """Evaluate a same-expiration, zero-tail-slope strategy exactly.

    The signed strategy value uses the same convention as ``spread_prices``:
    positive values are debits and negative values are credits.  A piecewise
    linear option payoff can attain an extremum only at zero or at a strike
    when its slope is flat beyond the highest strike.
    """

    expirations = {leg.contract.expiration_at for leg in candidate.legs}
    if len(expirations) != 1:
        raise ValueError("exact payoff requires one shared expiration")
    tail_slope = sum(
        leg.sign * leg.quantity
        for leg in candidate.legs
        if leg.contract.put_call == "CALL"
    )
    if tail_slope != 0:
        raise ValueError("exact defined-risk payoff requires zero upper-tail slope")
    multipliers = {float(leg.contract.multiplier) for leg in candidate.legs}
    if len(multipliers) != 1:
        raise ValueError("exact payoff requires one shared contract multiplier")
    multiplier = multipliers.pop()
    strikes = sorted({float(leg.contract.strike) for leg in candidate.legs})
    nodes = [0.0, *strikes]

    def pnl_per_share(spot: float) -> float:
        value = -entry_value
        for leg in candidate.legs:
            strike = float(leg.contract.strike)
            intrinsic = (
                max(spot - strike, 0.0)
                if leg.contract.put_call == "CALL"
                else max(strike - spot, 0.0)
            )
            value += leg.sign * leg.quantity * intrinsic
        return value

    values = [pnl_per_share(spot) for spot in nodes]
    maximum_profit = max(0.0, max(values)) * multiplier
    maximum_loss = max(0.0, -min(values)) * multiplier
    roots: list[float] = []
    tolerance = 1.0e-10
    for index in range(len(nodes) - 1):
        left_x, right_x = nodes[index], nodes[index + 1]
        left_y, right_y = values[index], values[index + 1]
        if math.isclose(left_y, 0.0, abs_tol=tolerance):
            roots.append(left_x)
        if left_y * right_y < 0:
            roots.append(
                left_x + (right_x - left_x) * (-left_y) / (right_y - left_y)
            )
    if math.isclose(values[-1], 0.0, abs_tol=tolerance):
        roots.append(nodes[-1])
    unique_roots: list[float] = []
    for value in roots:
        if value < 0:
            continue
        if not unique_roots or not math.isclose(
            value, unique_roots[-1], rel_tol=1.0e-10, abs_tol=1.0e-8
        ):
            unique_roots.append(value)
    return PayoffMetrics(
        exact_maximum_profit=maximum_profit,
        exact_maximum_loss=maximum_loss,
        breakeven=unique_roots[0] if unique_roots else None,
        breakevens=tuple(unique_roots),
        basis="exact_expiration_payoff",
        capital_at_risk=maximum_loss,
    )


__all__ = [
    "PayoffMetrics",
    "StrategyDefinition",
    "StrategyRegistry",
    "exact_defined_risk_payoff",
    "net_delta",
    "outlook_from_delta",
]
