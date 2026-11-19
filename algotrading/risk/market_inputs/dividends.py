"""Cash-dividend inputs for continuous-yield and escrowed-dividend models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DerivedDividendYield:
    trailing_cash_per_share: float
    spot: float
    simple_yield: float
    continuous_yield: float
    event_count: int
    window_start: date
    window_end: date
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class EscrowedDividendAdjustment:
    original_spot: float
    adjusted_spot: float
    present_value: float
    event_count: int
    event_dates: tuple[date, ...]
    algorithm_version: str = "escrowed_discrete_dividend_spot_v1"


def derive_trailing_cash_yield(
    events: Sequence[Mapping[str, Any]],
    *,
    spot: float,
    valuation_date: date,
    trailing_days: int = 365,
) -> DerivedDividendYield:
    if not math.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be positive")
    window_start = valuation_date - timedelta(days=trailing_days)
    total = 0.0
    count = 0
    classification_unverified = False
    seen: set[tuple[str, float]] = set()
    for event in events:
        raw_date = event.get("ex_dividend_date") or event.get("date")
        try:
            event_date = date.fromisoformat(str(raw_date)[0:10])
        except ValueError:
            continue
        if not window_start <= event_date <= valuation_date:
            continue
        status = str(event.get("status") or "").lower()
        if status in {"cancelled", "canceled", "deleted"}:
            continue
        classification = str(event.get("distribution_type") or event.get("type") or "").lower()
        if classification and any(
            marker in classification
            for marker in ("special", "return of capital", "stock dividend")
        ):
            continue
        if not classification:
            classification_unverified = True
        try:
            amount = float(event.get("amount"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(amount) or amount < 0:
            continue
        # Deduplicate identical cash events before estimating the trailing yield.
        key = (event_date.isoformat(), amount)
        if key in seen:
            continue
        seen.add(key)
        total += amount
        count += 1
    simple = total / spot
    if not math.isfinite(simple) or simple < 0 or simple >= 1:
        raise ValueError("derived dividend yield is outside [0, 1)")
    warnings = ("dividend_classification_unverified",) if classification_unverified else ()
    return DerivedDividendYield(
        trailing_cash_per_share=total,
        spot=spot,
        simple_yield=simple,
        continuous_yield=-math.log1p(-simple),
        event_count=count,
        window_start=window_start,
        window_end=valuation_date,
        warnings=warnings,
    )


def escrowed_dividend_adjustment(
    events: Sequence[Mapping[str, Any]],
    *,
    spot: float,
    valuation_at: datetime,
    expiration_at: datetime,
    continuous_rate: float,
) -> EscrowedDividendAdjustment:
    """Subtract the PV of known ordinary cash dividends from Tian's spot input."""

    if not math.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be positive")
    if not math.isfinite(continuous_rate):
        raise ValueError("continuous_rate must be finite")
    valuation = _aware_utc(valuation_at)
    expiration = _aware_utc(expiration_at)
    if expiration <= valuation:
        raise ValueError("expiration_at must be after valuation_at")
    present_value = 0.0
    included: list[date] = []
    seen: set[tuple[date, float]] = set()
    for event in events:
        event_date = _event_date(event)
        if event_date is None or not valuation.date() <= event_date <= expiration.date():
            continue
        if not _ordinary_cash_dividend(event):
            continue
        amount = _cash_amount(event)
        if amount is None:
            raise ValueError(f"cash dividend amount is unavailable for {event_date.isoformat()}")
        key = (event_date, amount)
        if key in seen:
            continue
        seen.add(key)
        # Discount each included cash payment to the valuation timestamp.
        event_at = datetime.combine(event_date, datetime.min.time(), tzinfo=timezone.utc)
        years = max(0.0, (event_at - valuation).total_seconds() / (365.0 * 86400.0))
        present_value += amount * math.exp(-continuous_rate * years)
        included.append(event_date)
    # Remove dividend present value from spot for escrowed-dividend pricing.
    adjusted = spot - present_value
    if not math.isfinite(adjusted) or adjusted <= 0:
        raise ValueError("escrowed dividends consume the entire spot value")
    return EscrowedDividendAdjustment(
        original_spot=float(spot),
        adjusted_spot=adjusted,
        present_value=present_value,
        event_count=len(included),
        event_dates=tuple(included),
    )


def _event_date(event: Mapping[str, Any]) -> date | None:
    raw = event.get("ex_dividend_date") or event.get("date")
    try:
        return date.fromisoformat(str(raw)[0:10])
    except ValueError:
        return None


def _ordinary_cash_dividend(event: Mapping[str, Any]) -> bool:
    status = str(event.get("status") or "").lower()
    if status in {"cancelled", "canceled", "deleted"}:
        return False
    classification = str(
        event.get("distribution_type") or event.get("type") or "cash"
    ).lower()
    return not any(
        marker in classification
        for marker in ("special", "return of capital", "stock dividend")
    )


def _cash_amount(event: Mapping[str, Any]) -> float | None:
    raw = next(
        (
            event.get(key)
            for key in ("amount", "cash_amount", "dividend")
            if event.get(key) is not None
        ),
        None,
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "DerivedDividendYield",
    "EscrowedDividendAdjustment",
    "derive_trailing_cash_yield",
    "escrowed_dividend_adjustment",
]
