"""Normalize broker positions while preserving missingness and raw audit data."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
import re
from typing import Any, Mapping

from algotrading.refdata.api.option_symbols import parse_occ_symbol

from ..instruments.terms import build_instrument_terms
from ..market_inputs.freshness import evaluate_quote_freshness
from ..models import (
    Coverage,
    DataQuality,
    FieldValidity,
    InstrumentTerms,
    NormalizedPosition,
)


_FUTURES = re.compile(r"^/(?P<root>[A-Z0-9]+)[FGHJKMNQUVXZ]\d{1,2}$")

_SECTOR_ALIASES = {
    "basic materials": "Basic Materials",
    "materials": "Basic Materials",
    "communication services": "Communication Services",
    "consumer cyclical": "Consumer Cyclical",
    "consumer discretionary": "Consumer Cyclical",
    "consumer defensive": "Consumer Defensive",
    "consumer staples": "Consumer Defensive",
    "energy": "Energy",
    "financial services": "Financial Services",
    "financials": "Financial Services",
    "health care": "Healthcare",
    "healthcare": "Healthcare",
    "industrials": "Industrials",
    "real estate": "Real Estate",
    "technology": "Technology",
    "information technology": "Technology",
    "utilities": "Utilities",
}


def canonical_sector(value: Any) -> str:
    source = str(value or "").strip()
    return _SECTOR_ALIASES.get(source.casefold(), source) or "Unclassified"


def normalize_broker_position(
    position: Mapping[str, Any],
    *,
    account_id: str,
    quote: Mapping[str, Any] | None = None,
    option_quote: Mapping[str, Any] | None = None,
    classification: Mapping[str, Any] | None = None,
    futures_spec: Mapping[str, Any] | None = None,
    option_terms: Mapping[str, Any] | None = None,
    instrument_terms: InstrumentTerms | None = None,
    greek: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    as_of: datetime | None = None,
) -> NormalizedPosition:
    context = context or {}
    instrument = position.get("instrument")
    if not isinstance(instrument, Mapping):
        instrument = {}
    symbol = str(instrument.get("symbol") or "").strip().upper()
    broker_type = str(instrument.get("type") or instrument.get("assetType") or "").upper()
    occ = None
    try:
        occ = parse_occ_symbol(symbol)
    except ValueError:
        pass
    future_match = _FUTURES.fullmatch(symbol)
    reference_type = str((classification or {}).get("security_type") or "").upper()
    inferred_asset_type = _asset_type(
        broker_type or reference_type, occ is not None, future_match is not None
    )
    terms = instrument_terms or build_instrument_terms(
        symbol,
        broker_type=broker_type or reference_type,
        option_terms=option_terms,
        futures_spec=futures_spec,
        context=context,
    )
    asset_type = terms.asset_type if terms.asset_type != "UNKNOWN" else inferred_asset_type
    # Represent long and short holdings with one signed quantity.
    raw_long_quantity = position.get("long_quantity", position.get("longQuantity"))
    raw_short_quantity = position.get("short_quantity", position.get("shortQuantity"))
    parsed_long_quantity = _number(raw_long_quantity)
    parsed_short_quantity = _number(raw_short_quantity)
    quantity_invalid = (
        raw_long_quantity is not None and parsed_long_quantity is None
    ) or (
        raw_short_quantity is not None and parsed_short_quantity is None
    )
    long_quantity = parsed_long_quantity or 0.0
    short_quantity = parsed_short_quantity or 0.0
    quantity = long_quantity - short_quantity
    warnings: list[str] = []
    if quantity_invalid:
        warnings.append("position_quantity_nonfinite_or_invalid")
    multiplier = terms.price_multiplier
    multiplier_source = terms.source
    # Keep the old public provenance value while clients migrate to the richer
    # InstrumentTerms source/version/quality contract.
    if "option_premium_multiplier_assumed_standard_occ" in terms.warnings:
        multiplier_source = "standard_us_equity_occ_v1"
    warnings.extend(terms.warnings)
    if multiplier is None:
        warnings.append("multiplier_unavailable")

    source_quote = option_quote if asset_type == "OPTION" else quote
    mark = _nonnegative_number(source_quote.get("mark")) if source_quote else None
    quote_time = _datetime(
        source_quote.get("quote_time")
        or source_quote.get("trade_time")
        or source_quote.get("as_of")
    ) if source_quote else None
    freshness = None
    if as_of is not None:
        source_kind = (
            "mutual_fund_nav"
            if asset_type == "MUTUAL_FUND" and source_quote
            else str(source_quote.get("source_kind") or "intraday_quote")
            if source_quote
            else "intraday_quote"
        )
        freshness = evaluate_quote_freshness(
            quote_time,
            as_of=as_of,
            context=context,
            exchange_timezone=terms.exchange_timezone or "America/New_York",
            source_kind=source_kind,
            calendar_id=terms.exchange_calendar,
            option_delta=(
                _number(greek.get("delta"))
                if asset_type == "OPTION" and greek
                else _number(option_quote.get("provider_delta"))
                if asset_type == "OPTION" and option_quote
                else None
            ),
        )
        if not freshness.valid_for_risk:
            warnings.append(f"quote_not_valid_for_risk:{freshness.reason}")
    if source_quote:
        bid = _nonnegative_number(source_quote.get("bid"))
        ask = _nonnegative_number(source_quote.get("ask"))
        if bid is not None and ask is not None and bid > ask:
            warnings.append("invalid_crossed_bid_ask")
            mark = None
    if mark is None:
        warnings.append("current_mark_unavailable")
    # Accept broker cost prices only when their per-unit scale is known.
    raw_average = _number(position.get("average_price", position.get("averagePrice")))
    verified_average_units = asset_type in {
        "EQUITY", "ETF", "MUTUAL_FUND", "CASH"
    } or (asset_type in {"OPTION", "FUTURE"} and multiplier is not None)
    normalized_average = raw_average if verified_average_units else None
    normalization_method = "broker_price_per_unit" if normalized_average is not None else None
    if raw_average is not None and normalized_average is None:
        warnings.append("average_price_units_unverified")
    cost_basis = (
        quantity * normalized_average * multiplier
        if normalized_average is not None and multiplier is not None
        else None
    )
    broker_market_value = _number(position.get("market_value", position.get("marketValue")))
    # Keep futures notional separate from their liquidation value and P&L.
    if asset_type == "FUTURE":
        market_value = None
        notional = (
            quantity * multiplier * mark
            if mark is not None and multiplier is not None
            else None
        )
    else:
        market_value = quantity * multiplier * mark if mark is not None and multiplier is not None else broker_market_value
        notional = market_value if asset_type in {"EQUITY", "ETF", "MUTUAL_FUND"} else None
    if asset_type != "FUTURE" and market_value is broker_market_value and broker_market_value is not None:
        warnings.append("market_value_from_broker_without_live_mark")
    if asset_type == "FUTURE":
        unrealized = (
            quantity * multiplier * (mark - normalized_average)
            if mark is not None
            and normalized_average is not None
            and multiplier is not None
            else None
        )
        cost_basis = None
    else:
        unrealized = market_value - cost_basis if market_value is not None and cost_basis is not None else None
    delta = _number(greek.get("delta")) if greek else None
    gamma = _number(greek.get("gamma")) if greek else None
    vega = _number(greek.get("vega_per_1pct")) if greek else None
    theta = _number(greek.get("theta_per_day")) if greek else None
    rho = _number(greek.get("rho_per_1pct")) if greek else None
    iv = _number(greek.get("implied_volatility")) if greek else None
    greek_source = (
        str(greek.get("method") or "tian")
        if greek and greek.get("status") in {"ok", "usable_with_warnings"}
        else None
    )
    current_option_greeks_supported = (
        asset_type != "OPTION" or terms.exercise_style in {"american", "european"}
    )
    # Unknown exercise style prevents trustworthy current option Greeks.
    if asset_type == "OPTION" and not current_option_greeks_supported:
        delta = gamma = vega = theta = rho = iv = None
        greek_source = None
        warnings.append("exercise_style_unknown_current_greeks_unavailable")
    elif asset_type == "OPTION" and delta is None:
        provider_delta = _number(option_quote.get("provider_delta")) if option_quote else None
        if provider_delta is not None:
            delta = provider_delta
            gamma = _number(option_quote.get("provider_gamma"))
            vega = _number(option_quote.get("provider_vega_per_1pct"))
            theta = _number(option_quote.get("provider_theta_per_day"))
            rho = _number(option_quote.get("provider_rho_per_1pct"))
            iv = _positive_number(option_quote.get("provider_iv"))
            if option_quote.get("provider_iv") is not None and iv is None:
                warnings.append("provider_iv_unusable")
            greek_source = "provider_fallback"
            warnings.append("tian_unavailable_provider_greeks_used")
        else:
            warnings.append("option_greeks_unavailable")
    underlying = terms.underlying_identifier or (
        occ.root if occ else (future_match.group("root") if future_match else symbol)
    )
    expiration = terms.expiration_at
    underlying_spot = (
        _positive_number(option_quote.get("underlying_price"))
        if option_quote
        else None
    ) or (_positive_number(quote.get("mark")) if quote else None)
    delta_exposure = None
    if asset_type in {"EQUITY", "ETF", "MUTUAL_FUND"} and mark is not None:
        delta_exposure = quantity * mark
    elif asset_type == "FUTURE":
        delta_exposure = notional
    elif (
        asset_type == "OPTION"
        and multiplier is not None
        and delta is not None
        and underlying_spot is not None
    ):
        delta_exposure = quantity * multiplier * delta * underlying_spot
    # Stable identifiers preserve position identity across snapshot refreshes.
    instrument_id = occ.canonical if occ else symbol
    position_id = _stable_id(account_id, asset_type, instrument_id)
    economic_id = _stable_id(account_id, underlying or instrument_id, "auto_underlying_v1")
    sector = canonical_sector((classification or {}).get("sector"))
    if sector == "Unclassified":
        warnings.append("classification_unavailable")
        if asset_type == "ETF":
            reason = str((classification or {}).get("classification_reason") or "insufficient_evidence")
            warnings.append(f"etf_exposure_unclassified:{reason}")
    status = "usable" if not warnings else "usable_with_warnings"
    if (
        not symbol
        or quantity_invalid
        or mark is None
        or (asset_type in {"OPTION", "FUTURE"} and multiplier is None)
        or (freshness is not None and not freshness.valid_for_risk)
    ):
        status = "insufficient_data"
        if not symbol:
            warnings.append("instrument_symbol_missing")
    quote_valid = freshness.valid_for_risk if freshness is not None else mark is not None
    freshness_state = freshness.state if freshness is not None else "not_evaluated"
    freshness_reason = freshness.reason if freshness is not None else (
        "timestamp_policy_not_requested" if mark is not None else "mark_unavailable"
    )
    quote_method = (
        freshness.calendar_id if freshness is not None else None
    )
    # Track validity per field so partial terms do not validate every calculation.
    terms_warnings = set(terms.warnings)
    premium_multiplier_valid = (
        terms.price_multiplier is not None
        and "option_premium_multiplier_assumed_standard_occ" not in terms_warnings
        and terms.quality_status != "unsupported"
    )
    complete_terms_valid = (
        terms.quality_status in {"standard", "adjusted"}
        and terms.price_multiplier is not None
    )
    exercise_style_valid = (
        asset_type != "OPTION"
        or terms.exercise_style in {"american", "european"}
    )
    settlement_valid = (
        asset_type != "OPTION"
        or terms.settlement_type in {"physical", "cash"}
    )
    payoff_deliverable_valid = (
        asset_type != "OPTION"
        or (
            terms.deliverable is not None
            and terms.deliverable.fully_supported
            and terms.deliverable.aggregate_exercise_amount is not None
        )
    )
    field_validity = {
        "instrument_terms": FieldValidity(
            source_at=terms.source_as_of,
            freshness_state="authoritative_or_audited_terms",
            valid_for_risk=complete_terms_valid,
            method=terms.source_version,
            reason=None if complete_terms_valid else "terms_incomplete",
        ),
        "premium_multiplier": FieldValidity(
            source_at=terms.source_as_of,
            freshness_state="authoritative_contract_economics",
            valid_for_risk=(premium_multiplier_valid if asset_type == "OPTION" else complete_terms_valid),
            method=terms.source_version,
            reason=None if premium_multiplier_valid else "premium_multiplier_unavailable_or_assumed",
        ),
        "historical_mark_revaluation_terms": FieldValidity(
            source_at=terms.source_as_of,
            freshness_state="authoritative_contract_economics",
            valid_for_risk=(premium_multiplier_valid if asset_type == "OPTION" else complete_terms_valid),
            method=terms.source_version,
            reason=None if premium_multiplier_valid else "exact_premium_multiplier_required",
        ),
        "exercise_style": FieldValidity(
            source_at=terms.source_as_of,
            freshness_state="authoritative_contract_terms",
            valid_for_risk=exercise_style_valid,
            method=terms.source_version,
            reason=None if exercise_style_valid else "exercise_style_unknown",
        ),
        "settlement": FieldValidity(
            source_at=terms.source_as_of,
            freshness_state="authoritative_contract_terms",
            valid_for_risk=settlement_valid,
            method=terms.source_version,
            reason=None if settlement_valid else "settlement_unknown",
        ),
        "payoff_deliverable": FieldValidity(
            source_at=terms.source_as_of,
            freshness_state="authoritative_contract_terms",
            valid_for_risk=payoff_deliverable_valid,
            method=terms.source_version,
            reason=None if payoff_deliverable_valid else "payoff_deliverable_incomplete",
        ),
        "mark": FieldValidity(
            source_at=quote_time,
            freshness_state=freshness_state,
            valid_for_risk=quote_valid and mark is not None,
            method=quote_method or (str(source_quote.get("source") or "refdata_api") if source_quote else None),
            reason=freshness_reason,
        ),
    }
    # Each Greek inherits quote freshness as well as its own availability.
    for field_name, value in (
        ("delta", delta_exposure),
        ("gamma", gamma),
        ("vega", vega),
        ("theta", theta),
        ("rho", rho),
    ):
        field_validity[field_name] = FieldValidity(
            source_at=quote_time,
            freshness_state=freshness_state,
            valid_for_risk=quote_valid and value is not None,
            method=greek_source,
            reason=(freshness_reason if not quote_valid else None if value is not None else f"{field_name}_unavailable"),
        )
    return NormalizedPosition(
        position_id=position_id,
        economic_position_id=economic_id,
        account_id=account_id,
        instrument_id=instrument_id or position_id,
        asset_type=asset_type,
        exact_symbol=symbol,
        canonical_occ_symbol=occ.canonical if occ else None,
        provider_option_symbol=occ.provider if occ else None,
        underlying_symbol=underlying if asset_type == "OPTION" else (symbol if asset_type in {"EQUITY", "ETF", "MUTUAL_FUND"} else None),
        futures_root=f"/{future_match.group('root')}" if future_match else None,
        put_call=occ.put_call if occ else None,
        strike=float(occ.strike) if occ else None,
        expiration_at=expiration,
        signed_quantity=quantity,
        multiplier=multiplier,
        multiplier_source=multiplier_source,
        instrument_terms=terms,
        raw_average_price=raw_average,
        normalized_average_price=normalized_average,
        normalization_method=normalization_method,
        signed_cost_basis_usd=cost_basis,
        mark=mark,
        signed_market_value_usd=market_value,
        signed_liquidation_value_usd=market_value,
        signed_notional_exposure_usd=notional,
        delta_exposure_usd=delta_exposure,
        unrealized_pnl_usd=unrealized,
        current_day_pnl_usd=_number(position.get("current_day_profit_loss", position.get("currentDayProfitLoss"))),
        sector=sector,
        industry=str((classification or {}).get("industry") or "") or None,
        delta=delta,
        gamma=gamma,
        vega_per_1pct=vega,
        theta_per_day=theta,
        rho_per_1pct=rho,
        implied_volatility=iv,
        greek_source=greek_source,
        quote_as_of=quote_time,
        quote_source=str(source_quote.get("source") or "refdata_api") if source_quote else None,
        field_validity=field_validity,
        data_quality=DataQuality(
            status=status,
            coverage=Coverage(
                positions_total=1,
                positions_modeled=(
                    1
                    if mark is not None
                    and not quantity_invalid
                    and (asset_type not in {"OPTION", "FUTURE"} or multiplier is not None)
                    and (freshness is None or freshness.valid_for_risk)
                    else 0
                ),
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        ),
        raw_broker_fields={
            "average_price": raw_average,
            "market_value": broker_market_value,
            "long_quantity": long_quantity,
            "short_quantity": short_quantity,
            "instrument_type": broker_type,
            "underlying_price": (
                underlying_spot
            ),
            "bid": _number(source_quote.get("bid")) if source_quote else None,
            "ask": _number(source_quote.get("ask")) if source_quote else None,
            "net_change": _number(source_quote.get("net_change")) if source_quote else None,
            "mark_change": _number(source_quote.get("mark_change")) if source_quote else None,
            "open_interest": _number(option_quote.get("open_interest")) if option_quote else None,
            "risk_free_rate_continuous": (
                _number(greek.get("risk_free_rate_continuous")) if greek else None
            ),
            "dividend_yield_continuous": (
                _number(greek.get("dividend_yield_continuous")) if greek else None
            ),
            "dividend_inputs": list((greek or {}).get("dividend_inputs") or ()),
            "surface_state": (
                dict(option_quote.get("surface_state"))
                if option_quote
                and isinstance(option_quote.get("surface_state"), Mapping)
                else dict(greek.get("surface_state"))
                if greek and isinstance(greek.get("surface_state"), Mapping)
                else None
            ),
            "economic_exposure_type": (classification or {}).get("economic_exposure_type"),
            "quote_freshness": freshness.as_dict() if freshness else None,
            "broker_market_value_ignored_for_future": (
                broker_market_value if asset_type == "FUTURE" else None
            ),
        },
    )


def _asset_type(broker_type: str, option: bool, future: bool) -> str:
    if option or "OPTION" in broker_type:
        return "OPTION"
    if future or "FUTURE" in broker_type:
        return "FUTURE"
    if broker_type in {"ETF", "EXCHANGE_TRADED_FUND"}:
        return "ETF"
    if broker_type in {
        "EQUITY", "STOCK", "COMMON_STOCK", "ADR", "DEPOSITARY_RECEIPT"
    }:
        return "EQUITY"
    if broker_type in {
        "MUTUAL_FUND",
        "FUND",
        "COLLECTIVE_INVESTMENT",
        "NO_LOAD_TAXABLE",
        "NO_LOAD_NONTAXABLE",
        "MONEY_MARKET_FUND",
    }:
        return "MUTUAL_FUND"
    if "CASH" in broker_type or "MONEY_MARKET" in broker_type:
        return "CASH"
    return "UNKNOWN"


def _stable_id(*parts: str) -> str:
    return "sha256:" + hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _nonnegative_number(value: Any) -> float | None:
    converted = _number(value)
    return converted if converted is not None and converted >= 0 else None


def _positive_number(value: Any) -> float | None:
    converted = _number(value)
    return (
        converted
        if converted is not None and math.isfinite(converted) and converted > 0
        else None
    )


def _datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


__all__ = ["canonical_sector", "normalize_broker_position"]
