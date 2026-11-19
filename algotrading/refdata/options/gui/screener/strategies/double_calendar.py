"""Bounded four-leg double-calendar generation."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

from ..models import CandidateDraft, CandidateLeg, ContractRecord, ScreeningRequest
from .base import PayoffMetrics, StrategyDefinition, net_delta, outlook_from_delta


class DoubleCalendarStrategy(StrategyDefinition):
    name = "double_calendar"

    def generate(
        self,
        contracts: Iterable[ContractRecord],
        spot: float,
        request: ScreeningRequest,
        limit: int,
    ) -> list[CandidateDraft]:
        by_contract: dict[tuple[str, float], list[ContractRecord]] = defaultdict(list)
        for contract in contracts:
            if abs(contract.strike / spot - 1.0) <= (
                request.strikes.maximum_moneyness_fraction + 1.0e-12
            ):
                by_contract[(contract.put_call, contract.strike)].append(contract)
        paired_terms: dict[
            tuple[object, object],
            list[tuple[ContractRecord, ContractRecord]],
        ] = defaultdict(list)
        for key in sorted(by_contract):
            chain = sorted(
                by_contract[key], key=lambda item: (item.expiration_at, item.contract_id)
            )
            for near_index, near in enumerate(chain):
                if not request.dte.front_minimum <= near.dte <= request.dte.front_maximum:
                    continue
                for far in chain[near_index + 1 :]:
                    if not request.dte.back_minimum <= far.dte <= request.dte.back_maximum:
                        continue
                    separation = (
                        far.expiration_at - near.expiration_at
                    ).total_seconds() / 86400
                    if separation < request.dte.minimum_expiration_separation_days:
                        continue
                    paired_terms[(near.expiration_at, far.expiration_at)].append(
                        (near, far)
                    )

        admitted: list[tuple[tuple[object, ...], CandidateDraft]] = []
        for term_key in sorted(paired_terms):
            pairs = sorted(
                paired_terms[term_key],
                key=lambda pair: (
                    pair[0].strike,
                    pair[0].put_call,
                    pair[0].contract_id,
                    pair[1].contract_id,
                ),
            )
            for first, second in combinations(pairs, 2):
                first_near, first_far = first
                second_near, second_far = second
                if first_near.strike == second_near.strike:
                    continue
                if first_near.put_call != second_near.put_call:
                    put = first_near if first_near.put_call == "PUT" else second_near
                    call = first_near if first_near.put_call == "CALL" else second_near
                    if put.strike >= call.strike:
                        continue
                    variant = "mixed_double_calendar"
                    type_priority = 0
                else:
                    variant = f"{first_near.put_call.lower()}_double_calendar"
                    type_priority = 1
                provisional = CandidateDraft(
                    strategy="double_calendar",
                    variant=variant,
                    outlook="neutral",
                    legs=(
                        CandidateLeg("short", first_near),
                        CandidateLeg("short", second_near),
                        CandidateLeg("long", first_far),
                        CandidateLeg("long", second_far),
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
                strikes = sorted((first_near.strike, second_near.strike))
                identity = tuple(leg.contract.contract_id for leg in candidate.legs)
                priority = (
                    type_priority,
                    abs(delta),
                    abs(sum(strikes) / (2.0 * spot) - 1.0),
                    abs(strikes[1] - strikes[0]) / spot,
                    *term_key,
                    *identity,
                )
                admitted.append((priority, candidate))
        admitted.sort(key=lambda item: item[0])
        return [candidate for _, candidate in admitted[:limit]]

    def structurally_valid(self, candidate: CandidateDraft) -> bool:
        if len(candidate.legs) != 4 or any(
            leg.quantity != 1 for leg in candidate.legs
        ):
            return False
        short_legs = [leg for leg in candidate.legs if leg.side == "short"]
        long_legs = [leg for leg in candidate.legs if leg.side == "long"]
        if len(short_legs) != 2 or len(long_legs) != 2:
            return False
        near_expirations = {leg.contract.expiration_at for leg in short_legs}
        far_expirations = {leg.contract.expiration_at for leg in long_legs}
        if len(near_expirations) != 1 or len(far_expirations) != 1:
            return False
        near = next(iter(near_expirations))
        far = next(iter(far_expirations))
        if near >= far:
            return False
        short_contracts = {
            (leg.contract.put_call, leg.contract.strike) for leg in short_legs
        }
        long_contracts = {
            (leg.contract.put_call, leg.contract.strike) for leg in long_legs
        }
        return len(short_contracts) == 2 and short_contracts == long_contracts

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
        short_dte = min(
            leg.contract.dte for leg in candidate.legs if leg.side == "short"
        )
        before_front = max(1, int(short_dte * 252 / 365) - 1)
        return min(
            before_front,
            super().planned_holding_sessions(candidate, request, configured_maximum),
        )


__all__ = ["DoubleCalendarStrategy"]
