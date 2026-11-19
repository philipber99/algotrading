"""History-derived ETF response mapping for economic scenario shocks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..models import NormalizedPosition
from .sector_rotation import SECTOR_PROXY_SYMBOLS


ETF_RESPONSE_PROXY_SYMBOLS = tuple(
    dict.fromkeys(("SPY", "QQQ", *SECTOR_PROXY_SYMBOLS.values()))
)


def build_etf_history_shocks(
    positions: Sequence[NormalizedPosition],
    returns: Mapping[str, Mapping[str, float]],
    *,
    broad_equity_shock: float | None,
    sector_shocks: Mapping[str, float],
    explicit_shocks: Mapping[str, float] | None = None,
    minimum_observations: int = 126,
    maximum_observations: int = 252,
    ridge_alpha: float = 1e-3,
    minimum_r_squared: float = 0.20,
    maximum_absolute_shock: float = 0.50,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Regress each held ETF on market and sector ETF returns and shock the fit."""

    shocks = dict(explicit_shocks or {})
    proxy_shocks = _proxy_shocks(broad_equity_shock, sector_shocks)
    modeled: list[dict[str, Any]] = []
    unmodeled: list[dict[str, Any]] = []
    symbols = sorted(
        {
            position.exact_symbol
            for position in positions
            if position.asset_type == "ETF" and position.exact_symbol not in shocks
        }
    )
    for symbol in symbols:
        target = returns.get(symbol)
        available_proxies = [
            proxy
            for proxy in ETF_RESPONSE_PROXY_SYMBOLS
            if proxy != symbol and returns.get(proxy)
        ]
        if target is None or "SPY" not in available_proxies:
            unmodeled.append({"symbol": symbol, "reason": "required_history_unavailable"})
            continue
        # Fit only sessions shared by the ETF and all selected proxy histories.
        dates = sorted(
            set(target).intersection(
                *(set(returns[proxy]) for proxy in available_proxies)
            )
        )[-maximum_observations:]
        if len(dates) < minimum_observations:
            unmodeled.append(
                {
                    "symbol": symbol,
                    "reason": "insufficient_aligned_history",
                    "observations": len(dates),
                }
            )
            continue
        y = np.asarray([target[day] for day in dates], dtype=np.float64)
        x = np.asarray(
            [[returns[proxy][day] for proxy in available_proxies] for day in dates],
            dtype=np.float64,
        )
        valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        y = y[valid]
        x = x[valid]
        if len(y) < minimum_observations:
            unmodeled.append(
                {
                    "symbol": symbol,
                    "reason": "insufficient_finite_history",
                    "observations": int(len(y)),
                }
            )
            continue
        means = x.mean(axis=0)
        scales = x.std(axis=0, ddof=0)
        nonzero = scales > np.finfo(np.float64).eps
        x = x[:, nonzero]
        means = means[nonzero]
        scales = scales[nonzero]
        proxy_names = [
            proxy for proxy, keep in zip(available_proxies, nonzero) if keep
        ]
        if "SPY" not in proxy_names:
            unmodeled.append({"symbol": symbol, "reason": "market_proxy_has_no_variance"})
            continue
        # Standardization keeps the ridge penalty comparable across proxy scales.
        design = (x - means) / scales
        centered_y = y - y.mean()
        gram = design.T @ design + float(ridge_alpha) * np.eye(design.shape[1])
        try:
            standardized_beta = np.linalg.solve(gram, design.T @ centered_y)
        except np.linalg.LinAlgError:
            unmodeled.append({"symbol": symbol, "reason": "response_regression_failed"})
            continue
        beta = standardized_beta / scales
        fitted = y.mean() + design @ standardized_beta
        total = float(np.sum(centered_y**2))
        r_squared = (
            1.0 - float(np.sum((y - fitted) ** 2)) / total if total > 0 else None
        )
        if r_squared is None or r_squared < minimum_r_squared:
            unmodeled.append(
                {
                    "symbol": symbol,
                    "reason": "response_fit_inadequate",
                    "observations": int(len(y)),
                    "r_squared": r_squared,
                }
            )
            continue
        predicted = float(
            sum(beta[index] * proxy_shocks[name] for index, name in enumerate(proxy_names))
        )
        # Cap extrapolated moves before handing them to scenario valuation.
        predicted = max(
            -float(maximum_absolute_shock),
            min(float(maximum_absolute_shock), predicted),
        )
        shocks[symbol] = predicted
        modeled.append(
            {
                "symbol": symbol,
                "response_shock": predicted,
                "observations": int(len(y)),
                "r_squared": r_squared,
                "coefficients": {
                    name: float(value) for name, value in zip(proxy_names, beta)
                },
                "proxy_shocks": {name: proxy_shocks[name] for name in proxy_names},
            }
        )
    return shocks, {
        "methodology_version": "etf_market_sector_ridge_response_v1",
        "minimum_observations": minimum_observations,
        "lookback_sessions": maximum_observations,
        "ridge_alpha": ridge_alpha,
        "minimum_r_squared": minimum_r_squared,
        "positions": modeled,
        "unmodeled": unmodeled,
        "unmodeled_symbols": sorted({item["symbol"] for item in unmodeled}),
    }


def _proxy_shocks(
    broad_equity_shock: float | None,
    sector_shocks: Mapping[str, float],
) -> dict[str, float]:
    broad = float(broad_equity_shock or 0.0)
    output = {"SPY": broad, "QQQ": float(sector_shocks.get("Technology", broad))}
    output.update(
        {
            proxy: float(sector_shocks.get(sector, broad))
            for sector, proxy in SECTOR_PROXY_SYMBOLS.items()
        }
    )
    return output


__all__ = ["ETF_RESPONSE_PROXY_SYMBOLS", "build_etf_history_shocks"]
