"""Bounded different-strike, different-expiration diagonal generation."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
import heapq
from typing import Iterable

from ..models import CandidateDraft, CandidateLeg, ContractRecord, ScreeningRequest
from .base import PayoffMetrics, StrategyDefinition, net_delta, outlook_from_delta


class DiagonalStrategy(StrategyDefinition):
    name = "diagonal"

    def generate(
        self,
        contracts: Iterable[ContractRecord],
        spot: float,
        request: ScreeningRequest,
        limit: int,
    ) -> list[CandidateDraft]:
        groups: dict[tuple[str, object], list[ContractRecord]] = defaultdict(list)
        for contract in contracts:
            if abs(contract.strike / spot - 1.0) <= request.strikes.maximum_moneyness_fraction:
                groups[(contract.put_call, contract.expiration_at)].append(contract)
        maximum_separation = request.strikes.maximum_diagonal_separation_fraction * spot
        for members in groups.values():
            members.sort(key=lambda item: (item.strike, item.contract_id))
        expirations_by_type: dict[str, list[object]] = defaultdict(list)
        for put_call, expiration in groups:
            expirations_by_type[put_call].append(expiration)
        for put_call in expirations_by_type:
            expirations_by_type[put_call] = sorted(
                set(expirations_by_type[put_call])
            )

        # A bounded max-heap admits candidates by stable structural priority so
        # later expiration groups can replace weaker early natural-order pairs.
        admitted: list[tuple[float, float, float, float, int, int, CandidateDraft]] = []
        for put_call in sorted(expirations_by_type):
            expirations = expirations_by_type[put_call]
            for near_expiration in expirations:
                near_contracts = [
                    item
                    for item in groups[(put_call, near_expiration)]
                    if request.dte.front_minimum
                    <= item.dte
                    <= request.dte.front_maximum
                ]
                if not near_contracts:
                    continue
                for far_expiration in expirations:
                    if far_expiration <= near_expiration:
                        continue
                    far_contracts = [
                        item
                        for item in groups[(put_call, far_expiration)]
                        if request.dte.back_minimum
                        <= item.dte
                        <= request.dte.back_maximum
                    ]
                    if not far_contracts:
                        continue
                    separation_days = (
                        far_expiration - near_expiration
                    ).total_seconds() / 86400
                    if separation_days < request.dte.minimum_expiration_separation_days:
                        continue
                    far_strikes = [item.strike for item in far_contracts]
                    for near in near_contracts:
                        first = bisect_left(
                            far_strikes, near.strike - maximum_separation
                        )
                        last = bisect_right(
                            far_strikes, near.strike + maximum_separation
                        )
                        for far in far_contracts[first:last]:
                            if far.strike == near.strike:
                                continue
                            provisional = CandidateDraft(
                                strategy="diagonal",
                                variant=f"{put_call.lower()}_diagonal",
                                outlook="neutral",
                                legs=(
                                    CandidateLeg("short", near),
                                    CandidateLeg("long", far),
                                ),
                            )
                            delta = net_delta(provisional)
                            outlook = outlook_from_delta(delta)
                            candidate = CandidateDraft(
                                strategy=provisional.strategy,
                                variant=provisional.variant,
                                outlook=outlook,
                                legs=provisional.legs,
                            )
                            if (
                                request.outlook != "any"
                                and request.outlook != outlook
                            ):
                                continue
                            priority = (
                                abs(delta),
                                abs((near.strike + far.strike) / (2.0 * spot) - 1.0),
                                abs(far.strike - near.strike) / spot,
                                -far.dte,
                                near.contract_id,
                                far.contract_id,
                            )
                            entry = (
                                -priority[0],
                                -priority[1],
                                -priority[2],
                                -priority[3],
                                -priority[4],
                                -priority[5],
                                candidate,
                            )
                            if len(admitted) < limit:
                                heapq.heappush(admitted, entry)
                                continue
                            worst = admitted[0]
                            worst_priority = tuple(-value for value in worst[:6])
                            if priority < worst_priority:
                                heapq.heapreplace(admitted, entry)
        output_entries = sorted(
            admitted,
            key=lambda entry: tuple(-value for value in entry[:6]),
        )
        return [entry[-1] for entry in output_entries]

    def structurally_valid(self, candidate: CandidateDraft) -> bool:
        if len(candidate.legs) != 2:
            return False
        short = next((leg for leg in candidate.legs if leg.side == "short"), None)
        long = next((leg for leg in candidate.legs if leg.side == "long"), None)
        return bool(
            short
            and long
            and short.contract.put_call == long.contract.put_call
            and short.contract.strike != long.contract.strike
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


__all__ = ["DiagonalStrategy"]
