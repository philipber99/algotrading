"""One-year sector-beta and cross-sector-correlation shock construction."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..models import NormalizedPosition


SECTOR_PROXY_SYMBOLS: dict[str, str] = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}


def _aligned_values(
    left: Mapping[str, float],
    right: Mapping[str, float],
    maximum_observations: int,
) -> tuple[list[float], list[float]]:
    dates = sorted(set(left).intersection(right))[-maximum_observations:]
    return [float(left[day]) for day in dates], [float(right[day]) for day in dates]


def _beta(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    variance = sum((y - right_mean) ** 2 for y in right)
    if variance <= 0:
        return None
    value = covariance / variance
    return value if math.isfinite(value) else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = sum((x - left_mean) ** 2 for x in left)
    right_scale = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_scale * right_scale)
    if denominator <= 0:
        return None
    value = numerator / denominator
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else None


def build_sector_rotation_shocks(
    positions: Sequence[NormalizedPosition],
    returns: Mapping[str, Mapping[str, float]],
    *,
    selected_sector: str,
    sector_move: float,
    default_equity_shock: float | None = None,
    explicit_shocks: Mapping[str, float] | None = None,
    minimum_observations: int = 126,
    maximum_observations: int = 252,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Combine a sector rotation with broad and symbol-specific shocks.

    A position's own-sector beta scales its move. Positions outside the selected
    sector are additionally scaled by the Pearson correlation between their
    sector proxy and the selected-sector proxy.
    """
    selected_proxy = SECTOR_PROXY_SYMBOLS.get(selected_sector)
    if selected_proxy is None:
        raise ValueError(f"unsupported sector rotation sector: {selected_sector}")
    explicit = dict(explicit_shocks or {})
    shocks = dict(explicit)
    diagnostics: list[dict[str, Any]] = []
    unmodeled_symbols: list[str] = []
    selected_returns = returns.get(selected_proxy)
    for position in positions:
        if position.asset_type not in {"EQUITY", "ETF", "MUTUAL_FUND", "OPTION", "FUTURE"}:
            continue
        symbol = position.underlying_symbol or position.exact_symbol
        if symbol in shocks and any(item.get("symbol") == symbol for item in diagnostics):
            continue
        own_proxy = SECTOR_PROXY_SYMBOLS.get(position.sector)
        symbol_returns = returns.get(symbol)
        own_returns = returns.get(own_proxy or "")
        if own_proxy is None or selected_returns is None or symbol_returns is None or own_returns is None:
            unmodeled_symbols.append(symbol)
            continue
        stock_values, sector_values = _aligned_values(symbol_returns, own_returns, maximum_observations)
        if len(stock_values) < minimum_observations:
            unmodeled_symbols.append(symbol)
            continue
        sector_beta = _beta(stock_values, sector_values)
        correlation = 1.0
        correlation_observations = len(stock_values)
        # Transmit the selected sector's move through cross-sector correlation.
        if position.sector != selected_sector:
            own_values, selected_values = _aligned_values(own_returns, selected_returns, maximum_observations)
            correlation_observations = len(own_values)
            if len(own_values) < minimum_observations:
                unmodeled_symbols.append(symbol)
                continue
            correlation = _pearson(own_values, selected_values)
        if sector_beta is None or correlation is None:
            unmodeled_symbols.append(symbol)
            continue
        rotation_shock = float(sector_move) * sector_beta * correlation
        # Add the rotation to the caller's explicit or broad market shock.
        base_shock = explicit.get(symbol, default_equity_shock or 0.0)
        shocks[symbol] = float(base_shock) + rotation_shock
        diagnostics.append(
            {
                "symbol": symbol,
                "sector": position.sector,
                "sector_proxy": own_proxy,
                "sector_beta_1y": sector_beta,
                "selected_sector_correlation_1y": correlation,
                "rotation_shock": rotation_shock,
                "combined_shock": shocks[symbol],
                "beta_observations": len(stock_values),
                "correlation_observations": correlation_observations,
            }
        )
    return shocks, {
        "selected_sector": selected_sector,
        "selected_sector_proxy": selected_proxy,
        "sector_move": float(sector_move),
        "lookback_sessions": maximum_observations,
        "minimum_observations": minimum_observations,
        "positions": diagnostics,
        "unmodeled_symbols": sorted(set(unmodeled_symbols)),
    }


__all__ = ["SECTOR_PROXY_SYMBOLS", "build_sector_rotation_shocks"]
