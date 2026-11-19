"""Contract normalization gates and explicit rejection diagnostics."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Callable

import numpy as np

from ..cache.pricing import CompiledSurfaceIvEvaluator
from ..errors import ScreenerTimeoutError
from .models import ContractRecord, RejectionDiagnostics, ScreeningRequest, Universe


STALE_QUOTE = "STALE_QUOTE"
MISSING_BID_OR_ASK = "MISSING_BID_OR_ASK"
INVALID_MARKET = "INVALID_MARKET"
EXCESSIVE_SPREAD_WIDTH = "EXCESSIVE_SPREAD_WIDTH"
INSUFFICIENT_OPEN_INTEREST = "INSUFFICIENT_OPEN_INTEREST"
INSUFFICIENT_VOLUME = "INSUFFICIENT_VOLUME"
MISSING_IV_OR_GREEKS = "MISSING_IV_OR_GREEKS"
NONSTANDARD_CONTRACT = "NONSTANDARD_CONTRACT"
INVALID_EXPIRATION_RELATIONSHIP = "INVALID_EXPIRATION_RELATIONSHIP"
INVALID_STRIKE_RELATIONSHIP = "INVALID_STRIKE_RELATIONSHIP"
MISSING_ACTIVE_CALCULATION_STATE = "MISSING_ACTIVE_CALCULATION_STATE"
MISSING_SURFACE = "MISSING_SURFACE"
OUTSIDE_FITTED_SURFACE_COVERAGE = "OUTSIDE_FITTED_SURFACE_COVERAGE"
PRICING_FAILURE = "PRICING_FAILURE"
NONPOSITIVE_EXPECTED_VALUE = "NONPOSITIVE_EXPECTED_VALUE"
CAPITAL_AT_RISK_LIMIT_EXCEEDED = "CAPITAL_AT_RISK_LIMIT_EXCEEDED"
DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
TIMEOUT_OR_CAPACITY_LIMIT = "TIMEOUT_OR_CAPACITY_LIMIT"


class RejectionTracker:
    def __init__(self, maximum_examples_per_reason: int = 5) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = defaultdict(list)
        self.maximum_examples_per_reason = maximum_examples_per_reason

    def add(self, reason: str, example: str | int | None = None) -> None:
        self.counts[reason] += 1
        if example is not None and len(self.examples[reason]) < self.maximum_examples_per_reason:
            self.examples[reason].append(str(example))

    def diagnostics(self) -> RejectionDiagnostics:
        dominant = None
        if self.counts:
            dominant = sorted(self.counts, key=lambda key: (-self.counts[key], key))[0]
        return RejectionDiagnostics(
            by_reason=dict(sorted(self.counts.items())),
            examples={key: values for key, values in sorted(self.examples.items())},
            dominant_reason=dominant,
        )


def eligible_contracts(
    universe: Universe,
    request: ScreeningRequest,
    tracker: RejectionTracker,
    *,
    require_surface: bool | None = None,
    deadline: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[ContractRecord, ...]:
    """Apply fail-closed rules before any combination generation."""

    allowed_types = {value.upper() for value in request.option_types}
    surface_required = (
        request.surface_relative_value if require_surface is None else require_surface
    )
    ordinary: list[ContractRecord] = []
    for index, contract in enumerate(universe.contracts):
        if index % 256 == 0 and deadline is not None and clock() > deadline:
            raise ScreenerTimeoutError("The spread screen exceeded its deadline")
        reason = _ineligible_reason(contract, universe, request, allowed_types)
        if reason is not None:
            tracker.add(reason, contract.option_symbol)
            continue
        ordinary.append(contract)
    if not surface_required:
        return tuple(ordinary)

    enriched_by_id: dict[int, ContractRecord] = {}
    grouped: dict[object, list[ContractRecord]] = defaultdict(list)
    for contract in ordinary:
        grouped[contract.expiration_at].append(contract)
    for expiration_at in sorted(grouped):
        if deadline is not None and clock() > deadline:
            raise ScreenerTimeoutError("The spread screen exceeded its deadline")
        contracts = grouped[expiration_at]
        fit = universe.surface_fits.get((expiration_at, request.price_basis))
        expiration = universe.expiration_inputs.get(expiration_at)
        if fit is None or expiration is None:
            for contract in contracts:
                tracker.add(MISSING_SURFACE, contract.option_symbol)
            continue
        if fit.calculation_definition_id not in {
            0,
            expiration.calculation_definition_id,
        }:
            for contract in contracts:
                tracker.add(MISSING_ACTIVE_CALCULATION_STATE, contract.option_symbol)
            continue
        fit_key = (expiration_at, request.price_basis)
        evaluator = universe.compiled_surface_evaluators.get(fit_key)
        if evaluator is None:
            evaluator = CompiledSurfaceIvEvaluator.from_fit(
                fit, expiration.time_to_expiry_seconds
            )
            universe.compiled_surface_evaluators[fit_key] = evaluator
        targets = np.log(
            np.asarray([contract.strike for contract in contracts], dtype=np.float64)
            / expiration.forward_price
        )
        fitted_values = evaluator.evaluate(targets)
        for contract, fitted in zip(contracts, fitted_values):
            if not math.isfinite(float(fitted)):
                tracker.add(
                    OUTSIDE_FITTED_SURFACE_COVERAGE, contract.option_symbol
                )
                continue
            enriched_by_id[contract.contract_id] = replace(
                contract,
                fitted_iv=float(fitted),
                iv_residual=float(contract.effective_iv or 0) - float(fitted),
                pricing_method=fit.pricing_method,
            )
    return tuple(
        enriched_by_id[contract.contract_id]
        for contract in ordinary
        if contract.contract_id in enriched_by_id
    )


def _ineligible_reason(
    contract: ContractRecord,
    universe: Universe,
    request: ScreeningRequest,
    allowed_types: set[str],
) -> str | None:
    if contract.put_call not in allowed_types:
        return "OPTION_TYPE_FILTERED"
    if contract.is_mini or contract.is_nonstandard:
        return NONSTANDARD_CONTRACT
    if contract.symbol_map_mismatch:
        return "SYMBOL_MAP_MISMATCH"
    if contract.active_failure_reason:
        return "FATAL_CALCULATION_FAILURE"
    if not math.isfinite(contract.strike) or contract.strike <= 0:
        return INVALID_STRIKE_RELATIONSHIP
    if not math.isfinite(contract.multiplier) or contract.multiplier <= 0:
        return INVALID_MARKET
    if not request.dte.minimum <= contract.dte <= request.dte.maximum:
        return "DTE_OUTSIDE_REQUEST"
    if contract.bid is None or contract.ask is None:
        return MISSING_BID_OR_ASK
    if contract.bid < 0 or contract.ask < contract.bid:
        return INVALID_MARKET
    midpoint = (contract.bid + contract.ask) / 2.0
    if not math.isfinite(midpoint) or midpoint <= 0:
        return INVALID_MARKET
    relative_width = (contract.ask - contract.bid) / midpoint
    if relative_width > request.liquidity.maximum_relative_bid_ask_width:
        return EXCESSIVE_SPREAD_WIDTH
    if contract.quote_time is None:
        return STALE_QUOTE
    age = (universe.snapshot.source_collected_at - contract.quote_time).total_seconds()
    if age < -5:
        return INVALID_MARKET
    if age > request.maximum_quote_age_seconds:
        return STALE_QUOTE
    required = (
        contract.effective_iv,
        contract.delta,
        contract.gamma,
        contract.theta,
        contract.vega,
    )
    if any(value is None or not math.isfinite(value) for value in required):
        return MISSING_IV_OR_GREEKS
    if not request.deltas.minimum_absolute <= abs(contract.delta or 0) <= request.deltas.maximum_absolute:
        return "DELTA_OUTSIDE_REQUEST"
    if (contract.open_interest or 0) < request.liquidity.minimum_open_interest:
        return INSUFFICIENT_OPEN_INTEREST
    if (contract.volume or 0) < request.liquidity.minimum_volume:
        return INSUFFICIENT_VOLUME
    if request.liquidity.require_sizes and (
        contract.bid_size is None or contract.ask_size is None
    ):
        return "MISSING_QUOTE_SIZE"
    return None


__all__ = [name for name in globals() if name.isupper()] + [
    "RejectionTracker",
    "eligible_contracts",
]
