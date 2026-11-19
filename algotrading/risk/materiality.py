"""Marked-value materiality policy for explicitly disclosed risk omissions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from .models import NormalizedPosition


@dataclass(frozen=True)
class OmissionMateriality:
    position_ids: tuple[str, ...]
    symbols: tuple[str, ...]
    gross_marked_value_usd: float | None
    portfolio_value_usd: float | None
    portfolio_fraction: float | None
    maximum_fraction: float
    may_exclude: bool
    unbounded_position_ids: tuple[str, ...] = ()

    @property
    def warning(self) -> str | None:
        if not self.position_ids or not self.may_exclude:
            return None
        gross = float(self.gross_marked_value_usd or 0.0)
        percent = 100.0 * float(self.portfolio_fraction or 0.0)
        threshold = 100.0 * self.maximum_fraction
        return (
            "positions_excluded_below_marked_value_materiality_limit:"
            f"symbols={','.join(self.symbols)}:gross_marked_value_usd={gross:.2f}:"
            f"portfolio_percent={percent:.4f}:limit_percent={threshold:.2f}"
        )


def marked_position_value(position: NormalizedPosition) -> float | None:
    """Return gross current marked value without requiring the mark to be fresh."""

    mark = _finite(position.mark)
    multiplier = _finite(position.multiplier)
    quantity = _finite(position.signed_quantity)
    if mark is None or multiplier is None or quantity is None:
        return None
    return abs(quantity * multiplier * mark)


def assess_omission_materiality(
    position_ids: Iterable[str],
    positions: Sequence[NormalizedPosition],
    *,
    portfolio_value_usd: float | None,
    context: Mapping[str, Any],
) -> OmissionMateriality:
    ids = tuple(sorted(set(str(item) for item in position_ids)))
    maximum = float(
        context.get("risk.calculations.maximum_excluded_marked_value_fraction", 0.03)
    )
    if not math.isfinite(maximum) or not 0.0 <= maximum < 1.0:
        raise ValueError(
            "risk.calculations.maximum_excluded_marked_value_fraction must be between 0 and 1"
        )
    by_id = {item.position_id: item for item in positions}
    selected = [by_id[item] for item in ids if item in by_id]
    unbounded = tuple(
        sorted(item.position_id for item in selected if marked_position_value(item) is None)
    )
    portfolio_value = _finite(portfolio_value_usd)
    values = [marked_position_value(item) for item in selected]
    # Unknown positions or marks prevent a bounded omission estimate.
    gross = (
        sum(float(value) for value in values if value is not None)
        if not unbounded and len(selected) == len(ids)
        else None
    )
    fraction = (
        gross / abs(portfolio_value)
        if gross is not None and portfolio_value not in (None, 0.0)
        else None
    )
    return OmissionMateriality(
        position_ids=ids,
        symbols=tuple(sorted(item.exact_symbol for item in selected)),
        gross_marked_value_usd=gross,
        portfolio_value_usd=portfolio_value,
        portfolio_fraction=fraction,
        maximum_fraction=maximum,
        may_exclude=bool(ids) and fraction is not None and fraction < maximum,
        unbounded_position_ids=unbounded,
    )


def _finite(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


__all__ = [
    "OmissionMateriality",
    "assess_omission_materiality",
    "marked_position_value",
]
