"""Leg and synthetic-spread execution quality calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .config import ScreenerConfig
from .models import CandidateDraft, ContractRecord, ExecutionResult


@dataclass(frozen=True)
class SpreadPrices:
    synthetic_bid: float
    synthetic_midpoint: float
    synthetic_ask: float

    def entry(self, basis: Literal["natural", "midpoint"]) -> float:
        return self.synthetic_ask if basis == "natural" else self.synthetic_midpoint


def spread_prices(candidate: CandidateDraft) -> SpreadPrices:
    """Apply the single net-liquidation-value convention from the API spec."""

    synthetic_bid = 0.0
    synthetic_mid = 0.0
    synthetic_ask = 0.0
    for leg in candidate.legs:
        contract = leg.contract
        if contract.bid is None or contract.ask is None:
            raise ValueError("spread price requires bid and ask on every leg")
        quantity = leg.quantity
        midpoint = (contract.bid + contract.ask) / 2.0
        if leg.side == "long":
            synthetic_bid += quantity * contract.bid
            synthetic_ask += quantity * contract.ask
            synthetic_mid += quantity * midpoint
        else:
            synthetic_bid -= quantity * contract.ask
            synthetic_ask -= quantity * contract.bid
            synthetic_mid -= quantity * midpoint
    return SpreadPrices(synthetic_bid, synthetic_mid, synthetic_ask)


def execution_quality(
    candidate: CandidateDraft,
    prices: SpreadPrices,
    config: ScreenerConfig,
    *,
    as_of: datetime | None = None,
) -> ExecutionResult:
    per_leg: dict[str, list[str]] = {}
    scores: list[float] = []
    for leg in candidate.legs:
        score, reasons = leg_execution_quality(
            leg.contract, config, as_of=as_of
        )
        per_leg[leg.contract.option_symbol] = reasons
        scores.append(score)
    spread_reasons: list[str] = []
    spread_width = max(0.0, prices.synthetic_ask - prices.synthetic_bid)
    denominator = max(abs(prices.synthetic_midpoint), 0.01)
    relative_width = spread_width / denominator
    spread_score = max(0.0, 1.0 - relative_width / max(config.maximum_relative_spread_width, 1.0e-9))
    if relative_width > config.maximum_relative_spread_width:
        spread_reasons.append("WIDE_SYNTHETIC_MARKET")
    elif relative_width > config.maximum_relative_spread_width * 0.5:
        spread_reasons.append("MODERATE_SYNTHETIC_MARKET")
    else:
        spread_reasons.append("TIGHT_SYNTHETIC_MARKET")
    score = max(0.0, min(1.0, 0.70 * min(scores or [0.0]) + 0.30 * spread_score))
    rating: Literal["good", "questionable", "poor"]
    if score >= 0.75:
        rating = "good"
    elif score >= 0.45:
        rating = "questionable"
    else:
        rating = "poor"
    return ExecutionResult(
        rating=rating,
        score=score,
        estimated_entry_slippage=abs(prices.synthetic_ask - prices.synthetic_midpoint),
        spread_reasons=spread_reasons,
        per_leg_reasons=per_leg,
    )


def leg_execution_quality(
    contract: ContractRecord,
    config: ScreenerConfig,
    *,
    as_of: datetime | None = None,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if as_of is not None:
        if contract.quote_time is None:
            return 0.0, ["STALE_QUOTE"]
        quote_age = (as_of - contract.quote_time).total_seconds()
        if quote_age > config.maximum_quote_age_seconds or quote_age < -5:
            return 0.0, ["STALE_QUOTE"]
        if quote_age > config.maximum_quote_age_seconds * 0.75:
            reasons.append("AGING_QUOTE")
    if contract.bid is None or contract.ask is None or contract.midpoint <= 0:
        return 0.0, ["MISSING_OR_INVALID_MARKET"]
    relative_width = (contract.ask - contract.bid) / contract.midpoint
    score = max(0.0, 1.0 - relative_width / max(config.maximum_relative_leg_width, 1.0e-9))
    if relative_width <= 0.10:
        reasons.append("TIGHT_LEG_MARKET")
    elif relative_width <= config.maximum_relative_leg_width:
        reasons.append("MODERATE_LEG_MARKET")
    else:
        reasons.append("WIDE_LEG_MARKET")
    if contract.bid_size is None or contract.ask_size is None:
        score -= 0.08
        reasons.append("MISSING_QUOTE_SIZE")
    elif min(contract.bid_size, contract.ask_size) <= 0:
        score -= 0.12
        reasons.append("NO_DISPLAYED_SIZE")
    if (contract.open_interest or 0) < config.minimum_open_interest:
        score -= 0.15
        reasons.append("LOW_OPEN_INTEREST")
    if (contract.volume or 0) < config.minimum_volume:
        score -= 0.10
        reasons.append("LOW_VOLUME")
    if contract.quality_flags:
        score -= 0.15
        reasons.append("CALCULATION_QUALITY_FLAGS")
    if contract.result_origin == "source_only":
        score -= 0.08
        reasons.append("SOURCE_ONLY_VALUES")
    if contract.is_mini or contract.is_nonstandard:
        score -= 0.35
        reasons.append("NONSTANDARD_CONTRACT")
    return max(0.0, min(1.0, score)), reasons


def quality_at_least(actual: str, minimum: str) -> bool:
    order = {"poor": 0, "questionable": 1, "good": 2}
    return order[actual] >= order[minimum]


__all__ = [
    "SpreadPrices",
    "execution_quality",
    "leg_execution_quality",
    "quality_at_least",
    "spread_prices",
]
