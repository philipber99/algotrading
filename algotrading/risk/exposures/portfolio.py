"""Explicit-unit portfolio KPI and sector exposure calculations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from ..materiality import assess_omission_materiality
from ..models import (
    CoverageMeasure,
    NormalizedPosition,
    PortfolioKpis,
    RiskCoverage,
)


def delta_equivalent_usd(position: NormalizedPosition) -> float | None:
    if position.delta_exposure_usd is not None:
        return position.delta_exposure_usd
    if position.asset_type in {"EQUITY", "ETF", "MUTUAL_FUND"}:
        if position.mark is None:
            return None
        return position.signed_quantity * position.mark
    if position.asset_type == "OPTION":
        if position.mark is None or position.multiplier is None or position.delta is None:
            return None
        spot = position.mark
        # Option marks are not underlying spots. Normalized snapshots store the
        # provider underlying spot in raw audit fields when available.
        raw_spot = position.raw_broker_fields.get("underlying_price")
        if raw_spot is None:
            return None
        return position.signed_quantity * position.multiplier * position.delta * float(raw_spot)
    if position.asset_type == "FUTURE":
        return position.signed_notional_exposure_usd
    return 0.0 if position.asset_type == "CASH" else None


def portfolio_kpis(
    positions: Sequence[NormalizedPosition],
    *,
    portfolio_value_usd: float | None,
    maintenance_requirement_usd: float | None = None,
    margin_loan_balance_usd: float | None = None,
    context: Mapping[str, Any] | None = None,
) -> PortfolioKpis:
    context = context or {}
    coverage = portfolio_risk_coverage(
        positions, portfolio_value_usd=portfolio_value_usd, context=context
    )
    delta_eligible = [item for item in positions if item.asset_type != "CASH"]
    delta_pairs = [
        (position, delta_equivalent_usd(position)) for position in delta_eligible
    ]
    delta_values = [
        float(value)
        for position, value in delta_pairs
        if _field_valid(position, "delta") and value is not None
    ]
    missing_delta = [
        position.position_id
        for position, value in delta_pairs
        if not _field_valid(position, "delta") or value is None
    ]
    # Futures contribute exposure but are excluded from liquidation-value totals.
    liquidation_positions = [item for item in positions if item.asset_type != "FUTURE"]
    market_values = [
        float(position.signed_market_value_usd)
        for position in liquidation_positions
        if _field_valid(position, "mark")
        and position.signed_market_value_usd is not None
    ]
    missing_market = [
        position.position_id
        for position in liquidation_positions
        if not _field_valid(position, "mark")
        or position.signed_market_value_usd is None
    ]
    # Scale per-unit Greeks by signed contract quantity and price multiplier.
    options = [item for item in positions if item.asset_type == "OPTION"]
    vega_values = [
        position.signed_quantity * position.multiplier * position.vega_per_1pct
        for position in options
        if position.multiplier is not None
        and position.vega_per_1pct is not None
        and _field_valid(position, "vega")
    ]
    theta_values = [
        position.signed_quantity * position.multiplier * position.theta_per_day
        for position in options
        if position.multiplier is not None
        and position.theta_per_day is not None
        and _field_valid(position, "theta")
    ]
    missing_vega = [
        item.position_id
        for item in options
        if item.multiplier is None
        or item.vega_per_1pct is None
        or not _field_valid(item, "vega")
    ]
    missing_theta = [
        item.position_id
        for item in options
        if item.multiplier is None
        or item.theta_per_day is None
        or not _field_valid(item, "theta")
    ]
    # Incomplete totals are usable only when the missing value is immaterial.
    decisions = {
        "delta": assess_omission_materiality(
            missing_delta, positions, portfolio_value_usd=portfolio_value_usd,
            context=context,
        ),
        "market": assess_omission_materiality(
            missing_market, positions, portfolio_value_usd=portfolio_value_usd,
            context=context,
        ),
        "vega": assess_omission_materiality(
            missing_vega, positions, portfolio_value_usd=portfolio_value_usd,
            context=context,
        ),
        "theta": assess_omission_materiality(
            missing_theta, positions, portfolio_value_usd=portfolio_value_usd,
            context=context,
        ),
    }
    delta_complete = not missing_delta or decisions["delta"].may_exclude
    market_complete = not missing_market or decisions["market"].may_exclude
    vega_complete = not missing_vega or decisions["vega"].may_exclude
    theta_complete = not missing_theta or decisions["theta"].may_exclude
    gross_delta = sum(abs(value) for value in delta_values) if delta_complete else None
    net_delta = sum(delta_values) if delta_complete else None
    gross_mv = (
        sum(abs(value) for value in market_values)
        if liquidation_positions and market_complete
        else None
    )
    warnings = []
    if missing_delta and not delta_complete:
        warnings.append("some_positions_excluded_from_delta_totals")
    if missing_market and not market_complete:
        warnings.append("some_positions_excluded_from_market_value_totals")
    if missing_vega and not vega_complete:
        warnings.append("some_options_excluded_from_vega_total")
    if missing_theta and not theta_complete:
        warnings.append("some_options_excluded_from_theta_total")
    warnings.extend(
        warning
        for decision in decisions.values()
        if (warning := decision.warning) is not None
    )
    margin = None
    if maintenance_requirement_usd is not None and portfolio_value_usd is not None and portfolio_value_usd > 0 and maintenance_requirement_usd >= 0:
        margin = maintenance_requirement_usd / portfolio_value_usd
    elif maintenance_requirement_usd is None:
        warnings.append("margin_unavailable_broker_field_missing")
    return PortfolioKpis(
        portfolio_value_usd=portfolio_value_usd,
        gross_delta_exposure_usd=gross_delta,
        net_delta_exposure_usd=net_delta,
        gross_market_value_usd=gross_mv,
        leverage=(gross_delta / portfolio_value_usd if gross_delta is not None and portfolio_value_usd and portfolio_value_usd > 0 else None),
        options_vega_per_1pct_usd=(sum(vega_values) if vega_complete else None),
        options_theta_per_day_usd=(sum(theta_values) if theta_complete else None),
        margin_utilization=margin,
        margin_loan_balance_usd=margin_loan_balance_usd,
        maintenance_requirement_usd=maintenance_requirement_usd,
        net_liquidating_value_usd=portfolio_value_usd,
        modeled_market_value_percent=coverage.market_value.percent,
        modeled_gross_delta_percent=coverage.delta.percent,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def sector_exposures(
    positions: Sequence[NormalizedPosition],
    *,
    portfolio_value_usd: float | None = None,
    context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    context = context or {}
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "delta_values": [],
            "market_values": [],
            "notional_values": [],
            "delta_eligible": [],
            "market_eligible": [],
            "notional_eligible": [],
            "contributors": [],
            "warnings": [],
        }
    )
    for position in positions:
        row = grouped[position.sector or "Unclassified"]
        if position.asset_type != "CASH":
            row["delta_eligible"].append(position.position_id)
            delta = delta_equivalent_usd(position)
            if delta is not None and _field_valid(position, "delta"):
                row["delta_values"].append(float(delta))
            else:
                row["warnings"].append(f"{position.position_id}:delta_unavailable")
        if position.asset_type != "FUTURE":
            row["market_eligible"].append(position.position_id)
            if position.signed_market_value_usd is not None and _field_valid(position, "mark"):
                row["market_values"].append(float(position.signed_market_value_usd))
            else:
                row["warnings"].append(f"{position.position_id}:market_value_unavailable")
        if position.asset_type != "CASH":
            row["notional_eligible"].append(position.position_id)
            if position.signed_notional_exposure_usd is not None and _field_valid(position, "instrument_terms"):
                row["notional_values"].append(float(position.signed_notional_exposure_usd))
            else:
                row["warnings"].append(f"{position.position_id}:notional_unavailable")
        row["contributors"].append(position.position_id)
    output: list[dict[str, Any]] = []
    delta_missing = [
        item.position_id
        for item in positions
        if item.asset_type != "CASH"
        and (delta_equivalent_usd(item) is None or not _field_valid(item, "delta"))
    ]
    market_missing = [
        item.position_id
        for item in positions
        if item.asset_type != "FUTURE"
        and (item.signed_market_value_usd is None or not _field_valid(item, "mark"))
    ]
    notional_missing = [
        item.position_id
        for item in positions
        if item.asset_type != "CASH"
        and (item.signed_notional_exposure_usd is None or not _field_valid(item, "instrument_terms"))
    ]
    # Assess omissions across the portfolio before publishing sector subtotals.
    omission_decisions = {
        "delta": assess_omission_materiality(
            delta_missing, positions, portfolio_value_usd=portfolio_value_usd,
            context=context,
        ),
        "market": assess_omission_materiality(
            market_missing, positions, portfolio_value_usd=portfolio_value_usd,
            context=context,
        ),
        "notional": assess_omission_materiality(
            notional_missing, positions, portfolio_value_usd=portfolio_value_usd,
            context=context,
        ),
    }
    for sector, values in sorted(grouped.items()):
        delta_complete = (
            len(values["delta_values"]) == len(values["delta_eligible"])
            or omission_decisions["delta"].may_exclude
        )
        market_complete = (
            len(values["market_values"]) == len(values["market_eligible"])
            or omission_decisions["market"].may_exclude
        )
        notional_complete = (
            len(values["notional_values"]) == len(values["notional_eligible"])
            or omission_decisions["notional"].may_exclude
        )
        row_warnings = list(dict.fromkeys(values["warnings"]))
        row_warnings.extend(
            warning
            for decision in omission_decisions.values()
            if (warning := decision.warning) is not None
        )
        output.append(
            {
                "sector": sector,
                "net_delta_exposure_usd": sum(values["delta_values"]) if delta_complete else None,
                "gross_delta_exposure_usd": sum(abs(value) for value in values["delta_values"]) if delta_complete else None,
                "signed_market_value_usd": sum(values["market_values"]) if market_complete else None,
                "gross_liquidation_value_usd": sum(abs(value) for value in values["market_values"]) if market_complete else None,
                "signed_notional_exposure_usd": sum(values["notional_values"]) if notional_complete else None,
                "coverage": {
                    "delta": _count_coverage(values["delta_eligible"], len(values["delta_values"])),
                    "market_value": _count_coverage(values["market_eligible"], len(values["market_values"])),
                    "notional": _count_coverage(values["notional_eligible"], len(values["notional_values"])),
                },
                "contributors": values["contributors"],
                "warnings": list(dict.fromkeys(row_warnings)),
            }
        )
    return output


def _count_coverage(eligible_ids: Sequence[str], modeled: int) -> dict[str, Any]:
    return {
        "eligible_positions": len(eligible_ids),
        "modeled_positions": modeled,
        "count_percent": 100.0 * modeled / len(eligible_ids) if eligible_ids else 100.0,
    }


def portfolio_risk_coverage(
    positions: Sequence[NormalizedPosition],
    *,
    portfolio_value_usd: float | None = None,
    context: Mapping[str, Any] | None = None,
) -> RiskCoverage:
    context = context or {}
    noncash = [item for item in positions if item.asset_type != "CASH"]
    derivatives = [
        item for item in positions if item.asset_type in {"OPTION", "FUTURE"}
    ]
    # Measure derivative term coverage independently from mark and Greek coverage.
    terms_missing = [
        item.position_id
        for item in derivatives
        if item.instrument_terms is None
        or item.instrument_terms.quality_status in {"partial", "unsupported"}
        or item.multiplier is None
    ]
    terms = _measure(
        derivatives,
        [item for item in derivatives if item.position_id not in terms_missing],
        missing=terms_missing,
        weight_basis="resolved_derivative_notional_usd",
        dollar_value=_derivative_notional,
    )
    liquidation_eligible = [item for item in positions if item.asset_type != "FUTURE"]
    market_modeled = [
        item
        for item in liquidation_eligible
        if item.signed_liquidation_value_usd is not None and _field_valid(item, "mark")
    ]
    market = _measure(
        liquidation_eligible,
        market_modeled,
        missing=[
            item.position_id
            for item in liquidation_eligible
            if item.signed_liquidation_value_usd is None or not _field_valid(item, "mark")
        ],
        weight_basis="absolute_valid_liquidation_value_usd",
        dollar_value=lambda item: (
            item.signed_liquidation_value_usd if _field_valid(item, "mark") else None
        ),
    )
    delta_modeled = [
        item
        for item in noncash
        if delta_equivalent_usd(item) is not None and _field_valid(item, "delta")
    ]
    delta = _measure(
        noncash,
        delta_modeled,
        missing=[item.position_id for item in noncash if item not in delta_modeled],
        weight_basis="absolute_valid_delta_exposure_usd",
        dollar_value=lambda item: (
            delta_equivalent_usd(item) if _field_valid(item, "delta") else None
        ),
    )
    market_decision = assess_omission_materiality(
        market.missing_position_ids,
        positions,
        portfolio_value_usd=portfolio_value_usd,
        context=context,
    )
    if market_decision.may_exclude:
        market = _materiality_coverage(market, market_decision)
    delta_decision = assess_omission_materiality(
        delta.missing_position_ids,
        positions,
        portfolio_value_usd=portfolio_value_usd,
        context=context,
    )
    if delta_decision.may_exclude:
        delta = _materiality_coverage(delta, delta_decision)
    options = [item for item in positions if item.asset_type == "OPTION"]

    def greek(name: str) -> CoverageMeasure:
        modeled = [
            item
            for item in options
            if item.multiplier is not None
            and getattr(item, name) is not None
            and _field_valid(item, _validity_name(name))
        ]
        return CoverageMeasure(
            eligible_positions=len(options),
            modeled_positions=len(modeled),
            percent=None,
            count_percent=(100.0 * len(modeled) / len(options) if options else 100.0),
            weight_basis="eligible_option_count",
            missing_position_ids=tuple(
                sorted(item.position_id for item in options if item not in modeled)
            ),
        )

    return RiskCoverage(
        instrument_terms=terms,
        market_value=market,
        delta=delta,
        gamma=greek("gamma"),
        vega=greek("vega_per_1pct"),
        theta=greek("theta_per_day"),
        rho=greek("rho_per_1pct"),
    )


def _materiality_coverage(
    original: CoverageMeasure, decision: Any
) -> CoverageMeasure:
    denominator = abs(float(decision.portfolio_value_usd))
    omitted = float(decision.gross_marked_value_usd or 0.0)
    return original.model_copy(
        update={
            "denominator_usd": denominator,
            "modeled_usd": max(0.0, denominator - omitted),
            "percent": _bounded_percent(100.0 * (1.0 - omitted / denominator)),
            "weight_basis": "portfolio_value_less_excluded_gross_marked_value_usd",
            "unavailable_denominator_position_ids": (),
        }
    )


def _measure(
    eligible: Sequence[NormalizedPosition],
    modeled: Sequence[NormalizedPosition],
    *,
    missing: Sequence[str],
    weight_basis: str,
    dollar_value: Any,
) -> CoverageMeasure:
    # Fix the eligible denominator before measuring the modeled subset.
    dollar_values = [dollar_value(item) for item in eligible]
    denominator_missing = [
        item.position_id
        for item, value in zip(eligible, dollar_values)
        if value is None
    ]
    # Unknown denominator values keep dollar-weighted coverage unavailable.
    denominator_missing = sorted(set(denominator_missing))
    denominator = (
        None
        if denominator_missing
        else sum(abs(float(value)) for value in dollar_values if value is not None)
    )
    modeled_usd = (
        None
        if denominator is None
        else sum(
            abs(float(dollar_value(item)))
            for item in modeled
            if dollar_value(item) is not None
        )
    )
    percent = (
        100.0 * len(modeled) / len(eligible)
        if eligible and denominator == 0
        else 100.0 * modeled_usd / denominator
        if denominator not in (None, 0) and modeled_usd is not None
        else 100.0
        if not eligible
        else None
    )
    return CoverageMeasure(
        eligible_positions=len(eligible),
        modeled_positions=len(modeled),
        denominator_usd=denominator,
        modeled_usd=modeled_usd,
        percent=_bounded_percent(percent),
        count_percent=(100.0 * len(modeled) / len(eligible) if eligible else 100.0),
        weight_basis=weight_basis,
        missing_position_ids=tuple(sorted(set(missing))),
        unavailable_denominator_position_ids=tuple(sorted(set(denominator_missing))),
    )


def _derivative_notional(item: NormalizedPosition) -> float | None:
    if item.asset_type == "FUTURE":
        return item.signed_notional_exposure_usd
    if item.asset_type != "OPTION" or item.instrument_terms is None:
        return None
    if not _field_valid(item, "payoff_deliverable"):
        return None
    deliverable = item.instrument_terms.deliverable
    spot = item.raw_broker_fields.get("underlying_price")
    if (
        deliverable is None
        or deliverable.shares_per_contract is None
        or spot is None
        or item.instrument_terms.price_multiplier is None
    ):
        return None
    return item.signed_quantity * deliverable.shares_per_contract * float(spot)


def _bounded_percent(value: float | None) -> float | None:
    """Keep floating-point coverage arithmetic inside the API's [0, 100] contract."""

    return None if value is None else min(100.0, max(0.0, float(value)))


def _field_valid(item: NormalizedPosition, field: str) -> bool:
    state = item.field_validity.get(field)
    if state is not None:
        return state.valid_for_risk
    # Compatibility for deterministic fixtures created before field metadata
    # became part of the additive snapshot contract.
    if field == "mark":
        return item.mark is not None
    if field == "delta":
        return delta_equivalent_usd(item) is not None
    attribute = {
        "vega": "vega_per_1pct",
        "theta": "theta_per_day",
    }.get(field, field)
    return getattr(item, attribute, None) is not None


def _validity_name(field: str) -> str:
    return {"vega_per_1pct": "vega", "theta_per_day": "theta", "rho_per_1pct": "rho"}.get(field, field)


__all__ = [
    "delta_equivalent_usd",
    "portfolio_kpis",
    "portfolio_risk_coverage",
    "sector_exposures",
]
