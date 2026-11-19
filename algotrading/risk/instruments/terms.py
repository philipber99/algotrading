"""Authoritative and audited-fallback instrument terms.

Reference Data owns the raw facts.  This module owns Risk's deterministic
normalization of those facts and the deliberately small, versioned futures
fallback registry used when entitled CME reference data is unavailable.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
import math
import re
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from algotrading.refdata.api.option_symbols import parse_occ_symbol
from algotrading.refdata.api.market_calendars import (
    CALENDAR_VERSION,
    calendar_for_futures_root,
)

from ..models import InstrumentTerms, OptionDeliverable, OptionDeliverableComponent


FUTURES_TERMS_VERSION = "risk_static_futures_terms_2026_08_v1"
_FUTURE = re.compile(
    r"^/?(?P<root>[A-Z0-9]{1,6})(?P<month>[FGHJKMNQUVXZ])(?P<year>[0-9]{1,2})$"
)


def _future_row(
    multiplier: float,
    tick: float,
    *,
    quote: str,
    asset_class: str,
    currency: str = "USD",
    exchange_timezone: str = "America/Chicago",
) -> dict[str, Any]:
    return {
        "price_multiplier": multiplier,
        "minimum_tick": tick,
        "tick_value": multiplier * tick,
        "quote_convention": quote,
        "asset_class": asset_class,
        "currency": currency,
        "exchange_timezone": exchange_timezone,
        "source_version": FUTURES_TERMS_VERSION,
    }


DEFAULT_FUTURES_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    "ES": _future_row(50, 0.25, quote="index_points", asset_class="equity_index"),
    "MES": _future_row(5, 0.25, quote="index_points", asset_class="equity_index"),
    "NQ": _future_row(20, 0.25, quote="index_points", asset_class="equity_index"),
    "MNQ": _future_row(2, 0.25, quote="index_points", asset_class="equity_index"),
    "RTY": _future_row(50, 0.10, quote="index_points", asset_class="equity_index"),
    "M2K": _future_row(5, 0.10, quote="index_points", asset_class="equity_index"),
    "YM": _future_row(5, 1.0, quote="index_points", asset_class="equity_index"),
    "CL": _future_row(1000, 0.01, quote="usd_per_barrel", asset_class="energy"),
    "MCL": _future_row(100, 0.01, quote="usd_per_barrel", asset_class="energy"),
    "GC": _future_row(100, 0.10, quote="usd_per_troy_ounce", asset_class="metals"),
    "MGC": _future_row(10, 0.10, quote="usd_per_troy_ounce", asset_class="metals"),
    "SI": _future_row(5000, 0.005, quote="usd_per_troy_ounce", asset_class="metals"),
    "SIL": _future_row(1000, 0.005, quote="usd_per_troy_ounce", asset_class="metals"),
    "ZN": _future_row(1000, 1 / 64, quote="treasury_points_32nds", asset_class="rates"),
    "ZF": _future_row(1000, 1 / 128, quote="treasury_points_32nds", asset_class="rates"),
    "ZB": _future_row(1000, 1 / 32, quote="treasury_points_32nds", asset_class="rates"),
    "6E": _future_row(125000, 0.00005, quote="usd_per_eur", asset_class="fx"),
    "M6E": _future_row(12500, 0.0001, quote="usd_per_eur", asset_class="fx"),
}


def validate_futures_specification_table(
    table: Mapping[str, Mapping[str, Any]],
) -> None:
    if not table:
        raise ValueError("static futures specification table must not be empty")
    for root, raw in table.items():
        multiplier = _positive(raw.get("price_multiplier"))
        tick = _positive(raw.get("minimum_tick"))
        tick_value = _positive(raw.get("tick_value"))
        if multiplier is None or tick is None or tick_value is None:
            raise ValueError(f"futures specification {root} has invalid multiplier/tick data")
        if not math.isclose(multiplier * tick, tick_value, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError(
                f"futures specification {root} has inconsistent tick value"
            )


def resolve_instrument_terms(
    symbols: Sequence[tuple[str, str]],
    *,
    option_terms: Mapping[str, Mapping[str, Any]] | None = None,
    futures_specs: Mapping[str, Mapping[str, Any]] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, InstrumentTerms]:
    """Resolve a deterministic symbol/type universe without fabricating terms."""

    return {
        symbol: build_instrument_terms(
            symbol,
            broker_type=broker_type,
            option_terms=(option_terms or {}).get(_canonical_option(symbol)),
            futures_spec=(futures_specs or {}).get(_canonical_future(symbol)),
            context=context or {},
        )
        for symbol, broker_type in sorted(set(symbols))
        if symbol
    }


def build_instrument_terms(
    symbol: str,
    *,
    broker_type: str = "",
    option_terms: Mapping[str, Any] | None = None,
    futures_spec: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> InstrumentTerms:
    context = context or {}
    normalized = str(symbol).strip().upper()
    broker = str(broker_type).strip().upper()
    try:
        occ = parse_occ_symbol(normalized)
    except ValueError:
        occ = None
    future_match = _FUTURE.fullmatch(normalized)

    # Resolve derivative contracts before applying ordinary security defaults.
    if occ is not None:
        return _option_instrument_terms(occ, broker, option_terms, context)
    if future_match is not None or "FUTURE" in broker:
        return _future_instrument_terms(
            normalized, future_match, futures_spec, context
        )
    if broker in {"ETF", "EXCHANGE_TRADED_FUND"}:
        subtype, asset = "etf", "ETF"
    elif broker in {
        "MUTUAL_FUND",
        "FUND",
        "COLLECTIVE_INVESTMENT",
        "NO_LOAD_TAXABLE",
        "NO_LOAD_NONTAXABLE",
        "MONEY_MARKET_FUND",
    }:
        subtype, asset = "mutual_fund", "MUTUAL_FUND"
    elif "CASH" in broker or "MONEY_MARKET" in broker:
        subtype, asset = "cash", "CASH"
    elif broker in {"EQUITY", "STOCK", "COMMON_STOCK", "ADR", "DEPOSITARY_RECEIPT"}:
        subtype, asset = "equity", "EQUITY"
    else:
        return InstrumentTerms(
            canonical_instrument_id=normalized or "UNKNOWN",
            asset_type="UNKNOWN",
            contract_subtype="unknown",
            quantity_unit="unknown",
            source="broker_classification",
            source_version="canonical_instrument_terms_v4",
            quality_status="unsupported",
            warnings=("instrument_terms_unsupported",),
        )
    return InstrumentTerms(
        canonical_instrument_id=normalized,
        asset_type=asset,
        contract_subtype=subtype,
        underlying_identifier=normalized if asset in {"EQUITY", "ETF", "MUTUAL_FUND"} else None,
        price_multiplier=1.0,
        quantity_unit="shares" if asset in {"EQUITY", "ETF"} else "units",
        quote_convention="usd_per_unit",
        economic_exposure_type=("equity" if asset == "EQUITY" else "unknown"),
        source="unit_instrument",
        source_version="canonical_instrument_terms_v4",
        quality_status="standard",
    )


def _option_instrument_terms(
    occ: Any,
    broker_type: str,
    raw: Mapping[str, Any] | None,
    context: Mapping[str, Any],
) -> InstrumentTerms:
    source = raw or {}
    raw_subtype = str((raw or {}).get("asset_subtype") or "").lower()
    subtype = (
        raw_subtype
        if raw_subtype in {"equity_option", "etf_option", "index_option", "futures_option"}
        else "futures_option"
        if "FUTURE" in broker_type
        else "index_option"
        if "INDEX" in broker_type
        else "unknown"
    )
    if subtype == "futures_option":
        return InstrumentTerms(
            canonical_instrument_id=occ.canonical,
            asset_type="OPTION",
            contract_subtype=subtype,
            underlying_identifier=occ.root,
            futures_root=f"/{occ.root}",
            quantity_unit="contracts",
            strike=float(occ.strike),
            put_call=occ.put_call,
            source=str(source.get("source") or "occ_parse"),
            source_version=str(source.get("source_version") or "canonical_instrument_terms_v4"),
            quality_status="unsupported",
            warnings=("futures_option_terms_unsupported",),
        )

    status = str(source.get("terms_status") or source.get("quality_status") or "").lower()
    multiplier = _positive(source.get("price_multiplier") or source.get("multiplier"))
    adjusted = source.get("adjusted_contract")
    deliverable_text = str(source.get("deliverable") or "").strip()
    shares = _nonnegative(
        source.get("shares_per_contract") or source.get("deliverable_shares")
    )
    shares_were_supplied = shares is not None
    expiration = _timestamp(source.get("expiration_at"))
    warnings = [str(item) for item in source.get("warnings") or ()]
    confirmed = status in {"standard", "adjusted"} and multiplier is not None
    # A standard OCC assumption remains partial and must disclose its provenance.
    if not confirmed:
        allow = _as_bool(
            context.get("risk.instrument_terms.allow_standard_occ_assumption", True)
        )
        evidence_adjusted = bool(adjusted) or (
            multiplier is not None and not math.isclose(multiplier, 100.0)
        ) or bool(_finite(source.get("cash_per_contract")))
        if allow and not evidence_adjusted:
            if multiplier is None:
                multiplier = 100.0
                warnings.append("option_premium_multiplier_assumed_standard_occ")
            if shares is None and not source.get("deliverable_components"):
                shares = 100.0
                warnings.append("option_deliverable_assumed_standard_occ")
            if adjusted is None:
                adjusted = False
            status = "partial"
            warnings.append("option_terms_assumed_standard_occ")
        else:
            status = "partial" if source else "unsupported"
            warnings.append("option_contract_terms_unavailable")
    # Fallback expiry times use the configured exchange timezone.
    if expiration is None:
        timezone_name = str(
            source.get("exchange_timezone")
            or context.get("risk.instrument_terms.standard_option_timezone", "America/New_York")
        )
        local_time = _configured_time(
            source.get("expiration_local_time")
            or context.get("risk.instrument_terms.standard_option_expiration_local_time", "16:00")
        )
        expiration = datetime.combine(
            occ.expiration, local_time, ZoneInfo(timezone_name)
        ).astimezone(timezone.utc)
        warnings.append("option_expiration_assumed_configured_standard_session")
    else:
        timezone_name = str(source.get("exchange_timezone") or "America/New_York")
    # Normalize the delivery basket before deciding which payoff calculations are supported.
    component_rows = source.get("deliverable_components") or ()
    components: list[OptionDeliverableComponent] = []
    for item in component_rows if isinstance(component_rows, Sequence) and not isinstance(component_rows, (str, bytes)) else ():
        if not isinstance(item, Mapping):
            continue
        try:
            components.append(OptionDeliverableComponent.model_validate(item))
        except ValueError:
            warnings.append("option_deliverable_component_invalid")
    if not components and shares is not None:
        components.append(
            OptionDeliverableComponent(
                component_type="security",
                security_identifier=occ.root,
                quantity=shares,
                currency=str(source.get("currency") or "USD"),
                description=f"{shares:g} {occ.root} shares",
            )
        )
    cash_per_contract = _finite(source.get("cash_per_contract"))
    if cash_per_contract and not any(item.component_type == "cash" for item in components):
        components.append(
            OptionDeliverableComponent(
                component_type="cash",
                cash_amount=cash_per_contract,
                currency=str(source.get("currency") or "USD"),
                description=f"{cash_per_contract:g} cash",
            )
        )
    aggregate_exercise_amount = _nonnegative(source.get("aggregate_exercise_amount"))
    settlement_type = _settlement_type(source.get("settlement_type"))
    if (
        aggregate_exercise_amount is None
        and multiplier is not None
        and settlement_type == "cash"
    ):
        aggregate_exercise_amount = float(occ.strike) * multiplier
    elif (
        aggregate_exercise_amount is None
        and shares is not None
        and shares_were_supplied
        and (status in {"standard", "adjusted"} or adjusted is not True)
    ):
        aggregate_exercise_amount = float(occ.strike) * shares
    deliverable = None
    if multiplier is not None:
        if settlement_type == "cash":
            deliverable = OptionDeliverable(
                description=(
                    deliverable_text
                    or f"cash-settled intrinsic value x {multiplier:g}"
                ),
                currency=str(source.get("currency") or "USD"),
                aggregate_exercise_amount=aggregate_exercise_amount,
                exercise_currency=str(
                    source.get("exercise_currency")
                    or source.get("currency")
                    or "USD"
                ),
                fully_supported=aggregate_exercise_amount is not None,
            )
        else:
            fully_supported = bool(components) and all(
                item.component_type in {"security", "cash"}
                and (item.component_type != "security" or item.security_identifier == occ.root)
                for item in components
            )
            deliverable = OptionDeliverable(
                description=deliverable_text or f"{shares or multiplier:g} shares",
                shares_per_contract=shares,
                cash_per_contract=cash_per_contract,
                currency=str(source.get("currency") or "USD"),
                components=tuple(components),
                aggregate_exercise_amount=aggregate_exercise_amount,
                exercise_currency=str(source.get("exercise_currency") or source.get("currency") or "USD"),
                fully_supported=(
                    fully_supported
                    and (shares_were_supplied or bool(component_rows))
                    and aggregate_exercise_amount is not None
                ),
            )
    return InstrumentTerms(
        canonical_instrument_id=occ.canonical,
        asset_type="OPTION",
        contract_subtype=subtype,
        underlying_identifier=occ.root,
        currency=str(source.get("currency") or "USD"),
        price_multiplier=multiplier,
        quantity_unit="contracts",
        minimum_tick=_positive(source.get("minimum_tick")),
        quote_convention=str(source.get("quote_convention") or "usd_per_share"),
        strike=float(occ.strike),
        put_call=occ.put_call,
        exercise_style=_exercise_style(source.get("exercise_style")),
        deliverable=deliverable,
        adjusted_contract=bool(adjusted) if adjusted is not None else None,
        listing_date=_date(source.get("listing_date")),
        expiration_at=expiration,
        last_trade_at=_timestamp(source.get("last_trade_at")),
        settlement_at=_timestamp(source.get("settlement_at")),
        settlement_style=str(source.get("settlement_style") or "") or None,
        exchange_timezone=timezone_name,
        settlement_type=settlement_type,
        exercise_cutoff_at=_timestamp(source.get("exercise_cutoff_at")),
        exchange_calendar=str(source.get("exchange_calendar") or "") or None,
        calendar_version=str(source.get("calendar_version") or "") or None,
        economic_exposure_type=(
            "equity" if subtype in {"equity_option", "etf_option", "index_option"} else "unknown"
        ),
        source=str(source.get("source") or "audited_occ_standard_assumption"),
        source_as_of=_timestamp(source.get("source_as_of")),
        source_version=str(source.get("source_version") or "canonical_instrument_terms_v4"),
        quality_status=status if status in {"standard", "adjusted", "partial", "unsupported"} else "partial",
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _future_instrument_terms(
    symbol: str,
    match: re.Match[str] | None,
    raw: Mapping[str, Any] | None,
    context: Mapping[str, Any],
) -> InstrumentTerms:
    canonical = _canonical_future(symbol)
    root = match.group("root") if match is not None else _future_root(canonical)
    source = raw or {}
    # Prefer authoritative futures terms and disclose any audited static fallback.
    remote_usable = str(source.get("status") or "").lower() == "ok"
    row: Mapping[str, Any] = source if remote_usable else {}
    warnings = [str(item) for item in source.get("warnings") or ()]
    if not remote_usable:
        configured = context.get(
            "risk.instrument_terms.static_futures_specifications",
            DEFAULT_FUTURES_SPECIFICATIONS,
        )
        table = configured if isinstance(configured, Mapping) else DEFAULT_FUTURES_SPECIFICATIONS
        candidate = table.get(root)
        if isinstance(candidate, Mapping):
            row = candidate
            warnings.append("futures_terms_static_fallback_used")
        else:
            warnings.append("futures_contract_terms_unavailable")
    multiplier = _positive(row.get("price_multiplier"))
    tick = _positive(row.get("minimum_tick"))
    tick_value = _positive(row.get("tick_value"))
    if tick_value is None and multiplier is not None and tick is not None:
        tick_value = multiplier * tick
    complete = multiplier is not None and tick is not None and tick_value is not None
    # Inconsistent tick economics invalidate the entire multiplier-and-tick set.
    if complete and not math.isclose(multiplier * tick, tick_value, rel_tol=1e-9, abs_tol=1e-8):
        complete = False
        warnings.append("futures_tick_value_inconsistent")
        multiplier = tick = tick_value = None
    return InstrumentTerms(
        canonical_instrument_id=canonical,
        asset_type="FUTURE",
        contract_subtype="future",
        underlying_identifier=canonical,
        futures_root=f"/{root}" if root else None,
        currency=str(row.get("currency") or "USD"),
        price_multiplier=multiplier,
        quantity_unit="contracts",
        minimum_tick=tick,
        tick_value=tick_value,
        quote_convention=str(row.get("quote_convention") or "") or None,
        listing_date=_date(row.get("listing_date") or row.get("first_trade_date")),
        expiration_at=_timestamp(row.get("expiration_at") or row.get("last_trade_at")),
        last_trade_at=_timestamp(row.get("last_trade_at")),
        settlement_at=_timestamp(row.get("settlement_at")),
        settlement_style=str(row.get("settlement_type") or "") or None,
        exchange_timezone=str(row.get("trading_timezone") or row.get("exchange_timezone") or "America/Chicago"),
        settlement_type=_settlement_type(row.get("delivery_type") or row.get("settlement_type")),
        exchange_calendar=str(
            row.get("exchange_calendar") or calendar_for_futures_root(root)
        ),
        calendar_version=str(row.get("calendar_version") or CALENDAR_VERSION),
        economic_exposure_type=(
            "equity" if str(row.get("asset_class") or "") == "equity_index"
            else "rates" if str(row.get("asset_class") or "") == "rates"
            else "currency" if str(row.get("asset_class") or "") == "fx"
            else "commodity" if str(row.get("asset_class") or "") in {"energy", "metals", "agriculture"}
            else "unknown"
        ),
        source=str(row.get("source") or ("configured_static_registry" if row else "unavailable")),
        source_as_of=_timestamp(row.get("provider_as_of") or row.get("retrieved_at")),
        source_version=str(row.get("source_version") or (FUTURES_TERMS_VERSION if row else "canonical_instrument_terms_v4")),
        quality_status="standard" if complete else "unsupported",
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _settlement_type(value: Any) -> str:
    text = str(value or "").lower()
    if "cash" in text or "financial" in text:
        return "cash"
    if any(token in text for token in ("physical", "delivery")):
        return "physical"
    return "unknown"


def _exercise_style(value: Any) -> str:
    text = str(value or "unknown").lower()
    if text in {"a", "american"}:
        return "american"
    if text in {"e", "european"}:
        return "european"
    return "unknown"


def _canonical_option(symbol: str) -> str:
    try:
        return parse_occ_symbol(symbol).canonical
    except ValueError:
        return symbol.strip().upper()


def _canonical_future(symbol: str) -> str:
    # Broker symbols may carry an exchange routing suffix (for example
    # /VXMQ26:XCBF). RefData and the static registry key the actual contract.
    return symbol.strip().upper().removeprefix("/").split(":", 1)[0]


def _future_root(symbol: str) -> str:
    match = _FUTURE.fullmatch(symbol)
    return match.group("root") if match is not None else ""


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _positive(value: Any) -> float | None:
    converted = _finite(value)
    return converted if converted is not None and converted > 0 else None


def _nonnegative(value: Any) -> float | None:
    converted = _finite(value)
    return converted if converted is not None and converted >= 0 else None


def _timestamp(value: Any) -> datetime | None:
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


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _configured_time(value: Any) -> time:
    try:
        hour, minute = str(value).split(":", 1)
        return time(int(hour), int(minute))
    except (TypeError, ValueError):
        raise ValueError("configured option expiration local time must be HH:MM")


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
