"""Bounded balanced long call and put butterfly generation."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable

from ..models import CandidateDraft, CandidateLeg, ContractRecord, ScreeningRequest
from .base import (
    PayoffMetrics,
    StrategyDefinition,
    exact_defined_risk_payoff,
    net_delta,
    outlook_from_delta,
)


class ButterflyStrategy(StrategyDefinition):
    name = "butterfly"

    def generate(
        self,
        contracts: Iterable[ContractRecord],
        spot: float,
        request: ScreeningRequest,
        limit: int,
    ) -> list[CandidateDraft]:
        groups: dict[tuple[object, str], list[ContractRecord]] = defaultdict(list)
        for contract in contracts:
            if abs(contract.strike / spot - 1.0) <= (
                request.strikes.maximum_moneyness_fraction + 1.0e-12
            ):
                groups[(contract.expiration_at, contract.put_call)].append(contract)
        minimum_wing = request.strikes.minimum_vertical_width_fraction * spot
        maximum_wing = request.strikes.maximum_vertical_width_fraction * spot
        admitted: list[tuple[tuple[object, ...], CandidateDraft]] = []
        seen: set[tuple[int, int, int]] = set()
        for key in sorted(groups, key=lambda item: (item[0], item[1])):
            chain = sorted(groups[key], key=lambda item: (item.strike, item.contract_id))
            by_strike: dict[float, list[ContractRecord]] = defaultdict(list)
            for contract in chain:
                by_strike[round(contract.strike, 8)].append(contract)
            for middle_index, middle in enumerate(chain):
                for low in chain[:middle_index]:
                    wing = middle.strike - low.strike
                    if wing < minimum_wing:
                        continue
                    if wing > maximum_wing:
                        continue
                    high_contracts = by_strike.get(
                        round(middle.strike + wing, 8), ()
                    )
                    for high in high_contracts:
                        identity = (
                            low.contract_id,
                            middle.contract_id,
                            high.contract_id,
                        )
                        if identity in seen:
                            continue
                        seen.add(identity)
                        provisional = CandidateDraft(
                            strategy="butterfly",
                            variant=f"{middle.put_call.lower()}_long_butterfly",
                            outlook="neutral",
                            legs=(
                                CandidateLeg("long", low),
                                CandidateLeg("short", middle, quantity=2),
                                CandidateLeg("long", high),
                            ),
                        )
                        delta = net_delta(provisional)
                        outlook = outlook_from_delta(delta)
                        if request.outlook != "any" and request.outlook != outlook:
                            continue
                        candidate = CandidateDraft(
                            strategy=provisional.strategy,
                            variant=provisional.variant,
                            outlook=outlook,
                            legs=provisional.legs,
                        )
                        priority = (
                            abs(middle.strike / spot - 1.0),
                            abs(delta),
                            wing / spot,
                            middle.expiration_at,
                            *identity,
                        )
                        admitted.append((priority, candidate))
        admitted.sort(key=lambda item: item[0])
        return [candidate for _, candidate in admitted[:limit]]

    def structurally_valid(self, candidate: CandidateDraft) -> bool:
        if len(candidate.legs) != 3:
            return False
        low, middle, high = sorted(
            candidate.legs, key=lambda leg: leg.contract.strike
        )
        return bool(
            low.side == high.side == "long"
            and middle.side == "short"
            and low.quantity == high.quantity == 1
            and middle.quantity == 2
            and len({leg.contract.put_call for leg in candidate.legs}) == 1
            and len({leg.contract.expiration_at for leg in candidate.legs}) == 1
            and low.contract.strike < middle.contract.strike < high.contract.strike
            and math.isclose(
                middle.contract.strike - low.contract.strike,
                high.contract.strike - middle.contract.strike,
                rel_tol=1.0e-9,
                abs_tol=1.0e-8,
            )
        )

    def payoff_metrics(
        self,
        candidate: CandidateDraft,
        entry_value: float,
    ) -> PayoffMetrics:
        return exact_defined_risk_payoff(candidate, entry_value)

    def planned_holding_sessions(
        self,
        candidate: CandidateDraft,
        request: ScreeningRequest,
        configured_maximum: int,
    ) -> int:
        before_expiration = max(
            1,
            int(min(leg.contract.dte for leg in candidate.legs) * 252 / 365) - 1,
        )
        return min(
            before_expiration,
            super().planned_holding_sessions(candidate, request, configured_maximum),
        )


__all__ = ["ButterflyStrategy"]
