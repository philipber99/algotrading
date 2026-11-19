"""Exact piecewise payoff certification for separable listed positions."""

from __future__ import annotations

import math
from typing import Any, Sequence

from algotrading.risk.models import NormalizedPosition


METHODOLOGY_VERSION = "exact_payoff_certification_v4"
_LINEAR = {"EQUITY", "ETF", "MUTUAL_FUND"}


def certify_maximum_loss(
    positions: Sequence[NormalizedPosition],
) -> dict[str, Any]:
    relevant = [item for item in positions if item.asset_type != "CASH"]
    if not relevant:
        return _insufficient("empty_payoff_position_set")
    unsupported = [
        item.position_id
        for item in relevant
        if item.asset_type not in {*_LINEAR, "OPTION", "FUTURE"}
    ]
    if unsupported:
        return _insufficient(
            "unsupported_instrument_terminal_payoff", excluded=unsupported
        )
    futures = [item.position_id for item in relevant if item.asset_type == "FUTURE"]
    if futures:
        return {
            **_base("unbounded"),
            "reason": "futures_price_domain_not_bounded",
            "maximum_loss_usd": None,
            "minimum_pnl_usd": None,
            "unbounded_loss": True,
            "excluded_position_ids": [],
            "unbounded_position_ids": futures,
            "warnings": ["terminal_loss_is_mathematically_unbounded"],
        }
    grouped: dict[str, list[NormalizedPosition]] = {}
    for item in relevant:
        underlying = item.underlying_symbol or item.exact_symbol
        grouped.setdefault(underlying, []).append(item)

    results: list[dict[str, Any]] = []
    for underlying, rows in sorted(grouped.items()):
        result = _certify_one_underlying(underlying, rows)
        if result["status"] == "insufficient_data":
            return {
                **result,
                "underlying_results": [*results, result],
                "excluded_position_ids": sorted(
                    set(result.get("excluded_position_ids") or ())
                ),
            }
        if result["status"] == "bounded_stress_only":
            return {
                **result,
                "underlying_results": [*results, result],
                "reason": (
                    result["reason"]
                    if len(grouped) == 1
                    else "one_or_more_underlyings_require_path_dependent_analysis"
                ),
            }
        results.append(result)

    if any(item["status"] == "unbounded" for item in results):
        return {
            **_base("unbounded"),
            "reason": "one_or_more_underlyings_have_unbounded_loss",
            "maximum_loss_usd": None,
            "minimum_pnl_usd": None,
            "unbounded_loss": True,
            "underlying_results": results,
            "position_contributions": [
                row for result in results for row in result["position_contributions"]
            ],
            "excluded_position_ids": [],
            "warnings": ["terminal_loss_is_mathematically_unbounded"],
        }
    minimum = sum(float(item["minimum_pnl_usd"]) for item in results)
    return {
        **_base("exact_bounded"),
        "reason": None,
        "minimum_pnl_usd": minimum,
        "maximum_loss_usd": max(0.0, -minimum),
        "unbounded_loss": False,
        "underlying_results": results,
        "terminal_payoff": results[0]["terminal_payoff"] if len(results) == 1 else [],
        "position_contributions": [
            row for result in results for row in result["position_contributions"]
        ],
        "excluded_position_ids": [],
        "warnings": [],
    }


def _certify_one_underlying(
    underlying: str, positions: Sequence[NormalizedPosition]
) -> dict[str, Any]:
    for item in positions:
        if item.mark is None or not math.isfinite(float(item.mark)):
            return _insufficient(
                "missing_current_mark_for_terminal_payoff",
                excluded=[item.position_id],
            )
        if item.asset_type != "OPTION":
            continue
        if any(
            value is None
            for value in (
                item.strike,
                item.multiplier,
                item.put_call,
                item.expiration_at,
                item.signed_liquidation_value_usd,
                item.instrument_terms,
            )
        ):
            return _insufficient(
                "incomplete_option_terminal_payoff_inputs",
                excluded=[item.position_id],
            )
        terms = item.instrument_terms
        assert terms is not None
        if (
            "option_terms_assumed_standard_occ" in terms.warnings
            or terms.quality_status not in {"standard", "adjusted"}
            or terms.deliverable is None
            or not terms.deliverable.fully_supported
            or terms.deliverable.aggregate_exercise_amount is None
        ):
            return _insufficient(
                "option_deliverable_not_certified",
                excluded=[item.position_id],
            )
    known_expirations = {
        item.expiration_at
        for item in positions
        if item.asset_type == "OPTION" and item.expiration_at is not None
    }
    if len(known_expirations) > 1:
        return _bounded_stress(underlying, positions)
    strikes = sorted(
        float(value)
        for item in positions
        if item.asset_type == "OPTION"
        and (value := _effective_breakpoint(item)) is not None
    )
    spot_candidates = [
        float(value)
        for item in positions
        if (
            value := item.raw_broker_fields.get("underlying_price")
            if item.asset_type == "OPTION"
            else item.mark
        )
        is not None
        and math.isfinite(float(value))
        and float(value) > 0
    ]
    if not spot_candidates:
        return _insufficient("underlying_spot_unavailable_for_terminal_payoff")
    current_spot = spot_candidates[0]
    breakpoints = sorted({0.0, *strikes})
    high = max(current_spot * 2.0, (strikes[-1] * 1.25 if strikes else current_spot * 2.0))

    def total(spot: float) -> float:
        return sum(_position_terminal_pnl(item, spot) for item in positions)

    # A negative high-price slope proves unlimited loss beyond all strike breaks.
    high_slope = sum(
        item.signed_quantity
        for item in positions
        if item.asset_type in _LINEAR
    ) + sum(
        item.signed_quantity * _delivered_underlying_quantity(item)
        for item in positions
        if item.asset_type == "OPTION" and item.put_call == "CALL"
    )
    points = sorted({*breakpoints, current_spot, high})
    values = [(spot, total(spot)) for spot in points]
    unbounded = high_slope < -1e-12
    minimum = None if unbounded else min(total(spot) for spot in breakpoints)
    contributions = []
    for item in sorted(positions, key=lambda row: row.position_id):
        item_slope = (
            item.signed_quantity
            if item.asset_type in _LINEAR
            else item.signed_quantity * _delivered_underlying_quantity(item)
            if item.put_call == "CALL"
            else 0.0
        )
        item_minimum = (
            None
            if item_slope < -1e-12
            else min(_position_terminal_pnl(item, spot) for spot in breakpoints)
        )
        contributions.append(
            {
                "position_id": item.position_id,
                "economic_position_id": item.economic_position_id,
                "minimum_pnl_usd": item_minimum,
                "unbounded_loss": item_slope < -1e-12,
            }
        )
    return {
        **_base("unbounded" if unbounded else "exact_bounded"),
        "underlying_symbol": underlying,
        "economic_domain": {"minimum_price": 0.0, "maximum_price": None},
        "evaluated_breakpoints": breakpoints,
        "limiting_high_price_slope_usd_per_price_unit": high_slope,
        "terminal_payoff": [
            {"underlying_price": spot, "pnl_usd": value}
            for spot, value in values
        ],
        "minimum_pnl_usd": minimum,
        "maximum_loss_usd": None if unbounded else max(0.0, -float(minimum)),
        "unbounded_loss": unbounded,
        "position_contributions": contributions,
        "excluded_position_ids": [],
        "warnings": ["terminal_loss_is_mathematically_unbounded"] if unbounded else [],
    }


def _bounded_stress(
    underlying: str, positions: Sequence[NormalizedPosition]
) -> dict[str, Any]:
    spots = [
        float(item.raw_broker_fields.get("underlying_price") or item.mark or 0.0)
        for item in positions
    ]
    current = next((value for value in spots if value > 0), 0.0)
    # Mixed expirations permit a bounded stress grid but no exact loss certificate.
    lower, upper = 0.25 * current, 2.0 * current
    points = [lower + (upper - lower) * index / 20 for index in range(21)]
    total_values = [
        sum(_position_terminal_pnl(item, spot) for item in positions)
        for spot in points
    ]
    stress_minimum = min(total_values) if total_values else None
    contributions = [
        {
            "position_id": item.position_id,
            "economic_position_id": item.economic_position_id,
            "minimum_pnl_usd": min(
                _position_terminal_pnl(item, spot) for spot in points
            ),
            "unbounded_loss": None,
            "certification_state": "bounded_stress_only",
        }
        for item in sorted(positions, key=lambda row: row.position_id)
    ]
    return {
        **_base("bounded_stress_only"),
        "reason": "mixed_expirations_path_dependent_terminal_payoff",
        "underlying_symbol": underlying,
        "stress_domain": {"minimum_price": lower, "maximum_price": upper},
        "terminal_payoff": [
            {
                "underlying_price": spot,
                "pnl_usd": value,
            }
            for spot, value in zip(points, total_values)
        ],
        "minimum_pnl_usd": None,
        "maximum_loss_usd": None,
        "stress_minimum_pnl_usd": stress_minimum,
        "stress_maximum_loss_usd": (
            max(0.0, -stress_minimum) if stress_minimum is not None else None
        ),
        "unbounded_loss": None,
        "position_contributions": contributions,
        "excluded_position_ids": [],
        "warnings": ["maximum_loss_not_certified_mixed_expirations"],
    }


def _position_terminal_pnl(item: NormalizedPosition, spot: float) -> float:
    if item.asset_type in _LINEAR:
        return item.signed_quantity * (spot - float(item.mark))
    intrinsic = option_intrinsic_per_contract(item, spot)
    current = item.signed_liquidation_value_usd
    if current is None:
        current = item.signed_market_value_usd
    return item.signed_quantity * intrinsic - float(current)


def option_intrinsic_per_contract(item: NormalizedPosition, spot: float) -> float:
    terms = item.instrument_terms
    if terms is None or terms.deliverable is None:
        raise ValueError("option deliverable is unavailable")
    deliverable = terms.deliverable
    if not deliverable.fully_supported or deliverable.aggregate_exercise_amount is None:
        raise ValueError("option deliverable is not payoff-certifiable")
    if terms.settlement_type == "cash":
        if terms.price_multiplier is None or terms.strike is None:
            raise ValueError("cash-settled option multiplier or strike is unavailable")
        intrinsic = (
            max(spot - terms.strike, 0.0)
            if item.put_call == "CALL"
            else max(terms.strike - spot, 0.0)
        )
        return terms.price_multiplier * intrinsic
    # Value the actual delivery basket before subtracting the exercise amount.
    market_value = 0.0
    for component in deliverable.components:
        if component.component_type == "security":
            if component.security_identifier != terms.underlying_identifier or component.quantity is None:
                raise ValueError("unsupported security deliverable component")
            market_value += component.quantity * spot
        elif component.component_type == "cash" and component.cash_amount is not None:
            market_value += component.cash_amount
        else:
            raise ValueError("unsupported option deliverable component")
    exercise = deliverable.aggregate_exercise_amount
    return (
        max(market_value - exercise, 0.0)
        if item.put_call == "CALL"
        else max(exercise - market_value, 0.0)
    )


def _delivered_underlying_quantity(item: NormalizedPosition) -> float:
    terms = item.instrument_terms
    if terms is None or terms.deliverable is None:
        return 0.0
    if terms.settlement_type == "cash":
        return float(terms.price_multiplier or 0.0)
    return sum(
        float(component.quantity or 0.0)
        for component in terms.deliverable.components
        if component.component_type == "security"
        and component.security_identifier == terms.underlying_identifier
    )


def _effective_breakpoint(item: NormalizedPosition) -> float | None:
    terms = item.instrument_terms
    if terms is None or terms.deliverable is None:
        return None
    if terms.settlement_type == "cash":
        return float(terms.strike) if terms.strike is not None else None
    deliverable = terms.deliverable
    shares = _delivered_underlying_quantity(item)
    if shares <= 0 or deliverable.aggregate_exercise_amount is None:
        return None
    cash = sum(
        float(component.cash_amount or 0.0)
        for component in deliverable.components
        if component.component_type == "cash"
    )
    return max(0.0, (deliverable.aggregate_exercise_amount - cash) / shares)


def _base(status: str) -> dict[str, Any]:
    return {"status": status, "methodology_version": METHODOLOGY_VERSION}


def _insufficient(
    reason: str, *, excluded: Sequence[str] = ()
) -> dict[str, Any]:
    return {
        **_base("insufficient_data"),
        "reason": reason,
        "terminal_payoff": [],
        "minimum_pnl_usd": None,
        "maximum_loss_usd": None,
        "unbounded_loss": None,
        "position_contributions": [],
        "excluded_position_ids": list(excluded),
        "warnings": ["maximum_loss_unavailable"],
    }


__all__ = [
    "METHODOLOGY_VERSION",
    "certify_maximum_loss",
    "option_intrinsic_per_contract",
]
