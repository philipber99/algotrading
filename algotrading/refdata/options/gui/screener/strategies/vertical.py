"""Bounded call and put debit/credit vertical generation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..models import CandidateDraft, CandidateLeg, ContractRecord, ScreeningRequest
from .base import PayoffMetrics, StrategyDefinition, exact_defined_risk_payoff


class VerticalStrategy(StrategyDefinition):
    name = "vertical"

    def generate(
        self,
        contracts: Iterable[ContractRecord],
        spot: float,
        request: ScreeningRequest,
        limit: int,
    ) -> list[CandidateDraft]:
        groups: dict[tuple[object, str], list[ContractRecord]] = defaultdict(list)
        for contract in contracts:
            if contract.dte > min(60, request.dte.maximum):
                continue
            if abs(contract.strike / spot - 1.0) > request.strikes.maximum_moneyness_fraction:
                continue
            groups[(contract.expiration_at, contract.put_call)].append(contract)
        output: list[CandidateDraft] = []
        seen: set[tuple[object, ...]] = set()
        minimum_width = request.strikes.minimum_vertical_width_fraction * spot
        maximum_width = request.strikes.maximum_vertical_width_fraction * spot
        for key in sorted(groups, key=lambda item: (item[0], item[1])):
            chain = sorted(groups[key], key=lambda item: (item.strike, item.contract_id))
            for low_index, low in enumerate(chain):
                for high in chain[low_index + 1 :]:
                    width = high.strike - low.strike
                    if width < minimum_width:
                        continue
                    if width > maximum_width:
                        break
                    if low.put_call == "CALL":
                        variants = (
                            ("call_debit_vertical", "bullish", "long", "short"),
                            ("call_credit_vertical", "bearish", "short", "long"),
                        )
                    else:
                        variants = (
                            ("put_debit_vertical", "bearish", "short", "long"),
                            ("put_credit_vertical", "bullish", "long", "short"),
                        )
                    for variant, outlook, low_side, high_side in variants:
                        candidate = CandidateDraft(
                            strategy="vertical",
                            variant=variant,
                            outlook=outlook,
                            legs=(CandidateLeg(low_side, low), CandidateLeg(high_side, high)),
                        )
                        identity = tuple(
                            sorted(
                                (leg.contract.contract_id, leg.side, leg.quantity)
                                for leg in candidate.legs
                            )
                        )
                        if identity in seen:
                            continue
                        seen.add(identity)
                        if request.outlook != "any" and request.outlook != outlook:
                            continue
                        output.append(candidate)
                        if len(output) >= limit:
                            return output
        return output

    def structurally_valid(self, candidate: CandidateDraft) -> bool:
        if len(candidate.legs) != 2:
            return False
        first, second = candidate.legs
        return (
            first.side != second.side
            and first.contract.put_call == second.contract.put_call
            and first.contract.expiration_at == second.contract.expiration_at
            and first.contract.strike != second.contract.strike
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


__all__ = ["VerticalStrategy"]
