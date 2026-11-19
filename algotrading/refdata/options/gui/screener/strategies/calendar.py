"""Bounded same-strike calendar generation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..models import CandidateDraft, CandidateLeg, ContractRecord, ScreeningRequest
from .base import PayoffMetrics, StrategyDefinition, net_delta, outlook_from_delta


class CalendarStrategy(StrategyDefinition):
    name = "calendar"

    def generate(
        self,
        contracts: Iterable[ContractRecord],
        spot: float,
        request: ScreeningRequest,
        limit: int,
    ) -> list[CandidateDraft]:
        groups: dict[tuple[str, float], list[ContractRecord]] = defaultdict(list)
        for contract in contracts:
            if abs(contract.strike / spot - 1.0) <= request.strikes.maximum_moneyness_fraction:
                groups[(contract.put_call, contract.strike)].append(contract)
        output: list[CandidateDraft] = []
        seen: set[tuple[int, int]] = set()
        for key in sorted(groups):
            chain = sorted(groups[key], key=lambda item: (item.expiration_at, item.contract_id))
            for near_index, near in enumerate(chain):
                if not request.dte.front_minimum <= near.dte <= request.dte.front_maximum:
                    continue
                for far in chain[near_index + 1 :]:
                    if not request.dte.back_minimum <= far.dte <= request.dte.back_maximum:
                        continue
                    separation = (far.expiration_at - near.expiration_at).total_seconds() / 86400
                    if separation < request.dte.minimum_expiration_separation_days:
                        continue
                    identity = (near.contract_id, far.contract_id)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    provisional = CandidateDraft(
                        strategy="calendar",
                        variant=f"{near.put_call.lower()}_calendar",
                        outlook="neutral",
                        legs=(CandidateLeg("short", near), CandidateLeg("long", far)),
                    )
                    outlook = outlook_from_delta(net_delta(provisional))
                    candidate = CandidateDraft(
                        strategy=provisional.strategy,
                        variant=provisional.variant,
                        outlook=outlook,
                        legs=provisional.legs,
                    )
                    if request.outlook != "any" and request.outlook != outlook:
                        continue
                    output.append(candidate)
                    if len(output) >= limit:
                        return output
        return output

    def structurally_valid(self, candidate: CandidateDraft) -> bool:
        if len(candidate.legs) != 2:
            return False
        short = next((leg for leg in candidate.legs if leg.side == "short"), None)
        long = next((leg for leg in candidate.legs if leg.side == "long"), None)
        return bool(
            short
            and long
            and short.contract.put_call == long.contract.put_call
            and short.contract.strike == long.contract.strike
            and short.contract.expiration_at < long.contract.expiration_at
        )

    def payoff_metrics(
        self,
        candidate: CandidateDraft,
        entry_value: float,
    ) -> PayoffMetrics:
        multiplier = min(leg.contract.multiplier for leg in candidate.legs)
        capital = max(0.0, entry_value) * multiplier if entry_value > 0 else None
        return PayoffMetrics(
            exact_maximum_profit=None,
            exact_maximum_loss=None,
            breakeven=None,
            basis="modeled_pre_front_expiration_range",
            capital_at_risk=capital,
        )

    def planned_holding_sessions(
        self,
        candidate: CandidateDraft,
        request: ScreeningRequest,
        configured_maximum: int,
    ) -> int:
        short = next(leg for leg in candidate.legs if leg.side == "short")
        before_front = max(1, int(short.contract.dte * 252 / 365) - 1)
        return min(
            before_front,
            super().planned_holding_sessions(candidate, request, configured_maximum),
        )


__all__ = ["CalendarStrategy"]
