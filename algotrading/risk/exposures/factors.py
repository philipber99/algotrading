"""Versioned factor transformations and deterministic ridge exposures."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


FACTOR_PROXY_SYMBOLS = (
    "SIZE", "VLUE", "MTUM", "QUAL", "USMV",
    "TLT", "SHY", "IEF", "GLD", "SLV", "USO", "UNG", "UUP", "QQQ", "SPY",
)


def factor_columns(proxy_returns: Mapping[str, Sequence[float]]) -> tuple[list[str], np.ndarray]:
    required = set(FACTOR_PROXY_SYMBOLS)
    missing = sorted(required - set(proxy_returns))
    if missing:
        raise ValueError("missing factor proxy returns: " + ",".join(missing))
    arrays = {key: np.asarray(proxy_returns[key], dtype=np.float64) for key in required}
    lengths = {array.shape for array in arrays.values()}
    if len(lengths) != 1 or next(iter(lengths), ()) == ():
        raise ValueError("factor proxy returns must be aligned one-dimensional arrays")
    names = [
        "Market", "Technology/growth tilt", "Size", "Value", "Momentum",
        "Quality", "Low volatility", "Long duration", "Intermediate duration",
        "Gold", "Silver", "Oil", "Natural gas", "U.S. dollar",
    ]
    # Relative-return factors isolate style and duration moves from their baselines.
    columns = [
        arrays["SPY"], arrays["QQQ"] - arrays["SPY"],
        arrays["SIZE"] - arrays["SPY"], arrays["VLUE"] - arrays["SPY"],
        arrays["MTUM"] - arrays["SPY"], arrays["QUAL"] - arrays["SPY"],
        arrays["USMV"] - arrays["SPY"], arrays["TLT"] - arrays["SHY"],
        arrays["IEF"] - arrays["SHY"], arrays["GLD"], arrays["SLV"],
        arrays["USO"], arrays["UNG"], arrays["UUP"],
    ]
    return names, np.column_stack(columns)


def ridge_factor_exposure(
    unit_returns: Sequence[float],
    proxy_returns: Mapping[str, Sequence[float]],
    *,
    current_delta_exposure_usd: float | None = None,
    unit_pnl_by_id: Mapping[str, Sequence[float]] | None = None,
    ridge_alpha: float = 1e-3,
    minimum_observations: int = 126,
) -> dict[str, Any]:
    names, factors = factor_columns(proxy_returns)
    target = np.asarray(unit_returns, dtype=np.float64)
    if current_delta_exposure_usd is not None:
        if not np.isfinite(float(current_delta_exposure_usd)):
            raise ValueError("current_delta_exposure_usd must be finite")
        target = target * float(current_delta_exposure_usd)
    if target.ndim != 1 or len(target) != factors.shape[0]:
        raise ValueError("unit and factor returns must be aligned")
    # Use the same finite observations for the target and every factor.
    valid = np.isfinite(target) & np.all(np.isfinite(factors), axis=1)
    target = target[valid]
    factors = factors[valid]
    if len(target) < minimum_observations:
        return {
            "status": "insufficient_data",
            "observations": int(len(target)),
            "minimum_observations": minimum_observations,
            "warnings": ["factor_history_insufficient"],
        }
    means = factors.mean(axis=0)
    std = factors.std(axis=0, ddof=0)
    nonzero = std > np.finfo(np.float64).eps
    if not np.all(nonzero):
        factors = factors[:, nonzero]
        means = means[nonzero]
        std = std[nonzero]
        names = [name for name, keep in zip(names, nonzero) if keep]
    # Standardize the fit, then convert coefficients back to raw return units.
    design = (factors - means) / std
    centered_target = target - target.mean()
    gram = design.T @ design + ridge_alpha * np.eye(design.shape[1])
    standardized_beta = np.linalg.solve(gram, design.T @ centered_target)
    beta = standardized_beta / std
    fitted = target.mean() + design @ standardized_beta
    residual = target - fitted
    total = np.sum((target - target.mean()) ** 2)
    r_squared = 1 - float(np.sum(residual**2) / total) if total > 0 else None
    factor_rows = [
        {
            "name": name,
            "beta": (
                float(value) / float(current_delta_exposure_usd)
                if current_delta_exposure_usd not in (None, 0)
                else None
            ),
            "coefficient_usd_per_unit_factor_return": float(value),
            "coefficient_units": "USD P&L per 1.00 factor return",
            "estimated_pnl_for_plus_1pct_factor_move_usd": float(value) * 0.01,
            "response_curve": [
                {
                    "factor_move_percent": move / 2.0,
                    "pnl_usd": float(value) * (move / 200.0),
                }
                for move in range(-8, 9)
            ],
            "factor_definition": _factor_definition(name),
        }
        for name, value in zip(names, beta)
    ]
    contributions: list[dict[str, Any]] = []
    # Reuse the portfolio design so unit contributions share the same factor basis.
    if unit_pnl_by_id:
        for unit_id in sorted(unit_pnl_by_id):
            unit_target = np.asarray(unit_pnl_by_id[unit_id], dtype=np.float64)
            if unit_target.ndim != 1 or len(unit_target) != len(valid):
                raise ValueError("unit P&L contribution series must align with factor history")
            unit_target = unit_target[valid]
            unit_centered = unit_target - unit_target.mean()
            unit_beta = np.linalg.solve(gram, design.T @ unit_centered) / std
            contributions.append(
                {
                    "unit_id": unit_id,
                    "factors": [
                        {
                            "name": name,
                            "coefficient_usd_per_unit_factor_return": float(value),
                            "estimated_pnl_for_plus_1pct_factor_move_usd": float(value) * 0.01,
                        }
                        for name, value in zip(names, unit_beta)
                    ],
                }
            )
    return {
        "status": "usable",
        "methodology_version": "dollar_factor_regression_v3",
        "calculation_method": "ridge_regression_of_daily_dollar_pnl_on_factor_returns",
        "observations": int(len(target)),
        "ridge_alpha": ridge_alpha,
        "r_squared": r_squared,
        "factors": factor_rows,
        "unit_contributions": contributions,
        "warnings": [],
    }


def _factor_definition(name: str) -> dict[str, str]:
    definitions = {
        "Market": {"return": "SPY", "sign": "positive means SPY price return is positive"},
        "Technology/growth tilt": {"return": "QQQ - SPY", "sign": "positive means QQQ outperforms SPY"},
        "Size": {"return": "SIZE - SPY", "sign": "positive means the size proxy outperforms SPY"},
        "Value": {"return": "VLUE - SPY", "sign": "positive means VLUE outperforms SPY"},
        "Momentum": {"return": "MTUM - SPY", "sign": "positive means MTUM outperforms SPY"},
        "Quality": {"return": "QUAL - SPY", "sign": "positive means QUAL outperforms SPY"},
        "Low volatility": {"return": "USMV - SPY", "sign": "positive means USMV outperforms SPY"},
        "Long duration": {"return": "TLT - SHY", "sign": "positive is a long-duration price outperformance, not a positive yield shock"},
        "Intermediate duration": {"return": "IEF - SHY", "sign": "positive is an intermediate-duration price outperformance, not a positive yield shock"},
    }
    return definitions.get(name, {"return": name, "sign": "positive means the named proxy return is positive"})


__all__ = ["FACTOR_PROXY_SYMBOLS", "factor_columns", "ridge_factor_exposure"]
