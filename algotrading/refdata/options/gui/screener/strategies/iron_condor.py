"""Bounded standard short iron-condor generation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..models import CandidateDraft, CandidateLeg, ContractRecord, ScreeningRequest
from .base import (
    PayoffMetrics,
    StrategyDefinition,
    exact_defined_risk_payoff,
    net_delta,
    outlook_from_delta,
)


class IronCondorStrategy(StrategyDefinition):
    name = "iron_condor"

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
        minimum_width = request.strikes.minimum_vertical_width_fraction * spot
        maximum_width = request.strikes.maximum_vertical_width_fraction * spot
        expirations = sorted({expiration for expiration, _ in groups})
        admitted: list[tuple[tuple[object, ...], CandidateDraft]] = []
        for expiration in expirations:
            puts = sorted(
                groups.get((expiration, "PUT"), ()),
                key=lambda item: (item.strike, item.contract_id),
            )
            calls = sorted(
                groups.get((expiration, "CALL"), ()),
                key=lambda item: (item.strike, item.contract_id),
            )
            put_wings: list[tuple[ContractRecord, ContractRecord]] = []
            for low_index, long_put in enumerate(puts):
                for short_put in puts[low_index + 1 :]:
                    width = short_put.strike - long_put.strike
                    if width < minimum_width:
                        continue
                    if width > maximum_width:
                        break
                    if short_put.strike <= spot:
                        put_wings.append((long_put, short_put))
            call_wings: list[tuple[ContractRecord, ContractRecord]] = []
            for low_index, short_call in enumerate(calls):
                if short_call.strike < spot:
                    continue
                for long_call in calls[low_index + 1 :]:
                    width = long_call.strike - short_call.strike
                    if width < minimum_width:
                        continue
                    if width > maximum_width:
                        break
                    call_wings.append((short_call, long_call))
            for long_put, short_put in put_wings:
                for short_call, long_call in call_wings:
                    if short_put.strike >= short_call.strike:
                        continue
                    provisional = CandidateDraft(
                        strategy="iron_condor",
                        variant="short_iron_condor",
                        outlook="neutral",
                        legs=(
                            CandidateLeg("long", long_put),
                            CandidateLeg("short", short_put),
                            CandidateLeg("short", short_call),
                            CandidateLeg("long", long_call),
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
                    put_width = short_put.strike - long_put.strike
                    call_width = long_call.strike - short_call.strike
                    center = (short_put.strike + short_call.strike) / 2.0
                    identity = tuple(
                        leg.contract.contract_id for leg in candidate.legs
                    )
                    priority = (
                        abs(delta),
                        abs(center / spot - 1.0),
                        abs(put_width - call_width) / spot,
                        (short_call.strike - short_put.strike) / spot,
                        expiration,
                        *identity,
                    )
                    admitted.append((priority, candidate))
        admitted.sort(key=lambda item: item[0])
        return [candidate for _, candidate in admitted[:limit]]

    def structurally_valid(self, candidate: CandidateDraft) -> bool:
        if len(candidate.legs) != 4:
            return False
        if len({leg.contract.expiration_at for leg in candidate.legs}) != 1:
            return False
        puts = sorted(
            (leg for leg in candidate.legs if leg.contract.put_call == "PUT"),
            key=lambda leg: leg.contract.strike,
        )
        calls = sorted(
            (leg for leg in candidate.legs if leg.contract.put_call == "CALL"),
            key=lambda leg: leg.contract.strike,
        )
        return bool(
            len(puts) == len(calls) == 2
            and puts[0].side == "long"
            and puts[1].side == "short"
            and calls[0].side == "short"
            and calls[1].side == "long"
            and all(leg.quantity == 1 for leg in candidate.legs)
            and puts[0].contract.strike
            < puts[1].contract.strike
            < calls[0].contract.strike
            < calls[1].contract.strike
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


__all__ = ["IronCondorStrategy"]
