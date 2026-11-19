"""Exact fitted-surface evaluation and on-demand option pricing."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr


SECONDS_PER_YEAR = 365.25 * 86400.0


class CompiledSurfaceIvEvaluator:
    """Validate and compile one fitted surface for repeated array evaluation."""

    def __init__(
        self,
        *,
        fit_method: str,
        minimum_log_forward_moneyness: float,
        maximum_log_forward_moneyness: float,
        time_to_expiry_seconds: float,
        svi_a: float | None = None,
        svi_b: float | None = None,
        svi_rho: float | None = None,
        svi_m: float | None = None,
        svi_sigma: float | None = None,
        pchip_k: Sequence[float] = (),
        pchip_total_variance: Sequence[float] = (),
    ) -> None:
        self.fit_method = str(fit_method).strip().upper()
        self.minimum = float(minimum_log_forward_moneyness)
        self.maximum = float(maximum_log_forward_moneyness)
        self.tenor_years = float(time_to_expiry_seconds) / SECONDS_PER_YEAR
        self._valid = bool(
            self.fit_method in {"SVI", "PCHIP"}
            and np.isfinite(self.minimum)
            and np.isfinite(self.maximum)
            and self.minimum < self.maximum
            and np.isfinite(self.tenor_years)
            and self.tenor_years > 0.0
        )
        self._svi: tuple[float, float, float, float, float] | None = None
        self._pchip: PchipInterpolator | None = None
        self._knot_minimum = math.nan
        self._knot_maximum = math.nan
        if not self._valid:
            return
        if self.fit_method == "SVI":
            parameters = (svi_a, svi_b, svi_rho, svi_m, svi_sigma)
            try:
                values = tuple(float(value) for value in parameters)
            except (TypeError, ValueError, OverflowError):
                self._valid = False
                return
            if not all(np.isfinite(value) for value in values) or values[4] <= 0:
                self._valid = False
                return
            self._svi = values
            return
        knots = np.asarray(pchip_k, dtype=np.float64)
        variances = np.asarray(pchip_total_variance, dtype=np.float64)
        if (
            len(knots) < 2
            or len(knots) != len(variances)
            or not np.all(np.isfinite(knots))
            or not np.all(np.isfinite(variances))
            or np.any(np.diff(knots) <= 0.0)
        ):
            self._valid = False
            return
        try:
            self._pchip = PchipInterpolator(knots, variances, extrapolate=False)
        except (TypeError, ValueError, FloatingPointError):
            self._valid = False
            return
        self._knot_minimum = float(knots[0])
        self._knot_maximum = float(knots[-1])

    @classmethod
    def from_fit(cls, fit: Any, time_to_expiry_seconds: float) -> "CompiledSurfaceIvEvaluator":
        def value(name: str, default: Any = None) -> Any:
            if isinstance(fit, Mapping):
                return fit.get(name, default)
            return getattr(fit, name, default)

        return cls(
            fit_method=value("fit_method", ""),
            minimum_log_forward_moneyness=value(
                "minimum_log_forward_moneyness", math.nan
            ),
            maximum_log_forward_moneyness=value(
                "maximum_log_forward_moneyness", math.nan
            ),
            time_to_expiry_seconds=time_to_expiry_seconds,
            svi_a=value("svi_a"),
            svi_b=value("svi_b"),
            svi_rho=value("svi_rho"),
            svi_m=value("svi_m"),
            svi_sigma=value("svi_sigma"),
            pchip_k=value("pchip_k", ()),
            pchip_total_variance=value("pchip_total_variance", ()),
        )

    def evaluate(self, target_moneyness: Any) -> np.ndarray:
        target = np.asarray(target_moneyness, dtype=np.float64)
        shape = target.shape
        flat = target.reshape(-1)
        output = np.full(len(flat), np.nan, dtype=np.float64)
        if not self._valid or not len(flat):
            return output.reshape(shape)
        covered = (
            np.isfinite(flat)
            & (flat >= self.minimum)
            & (flat <= self.maximum)
        )
        if self.fit_method == "SVI" and self._svi is not None:
            a, b, rho, center, sigma = self._svi
            shifted = flat[covered] - center
            total_variance = a + b * (
                rho * shifted + np.sqrt(shifted * shifted + sigma * sigma)
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                output[covered] = np.sqrt(total_variance / self.tenor_years)
        elif self._pchip is not None:
            pchip_covered = (
                covered
                & (flat >= self._knot_minimum)
                & (flat <= self._knot_maximum)
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                output[pchip_covered] = np.sqrt(
                    np.asarray(self._pchip(flat[pchip_covered]), dtype=np.float64)
                    / self.tenor_years
                )
        output[~np.isfinite(output) | (output <= 0.0)] = np.nan
        return output.reshape(shape)


def _column(surface_columns: Any, name: str) -> Any:
    if isinstance(surface_columns, pa.Table):
        return surface_columns[name]
    return surface_columns[name]


def _numeric_column(surface_columns: Any, name: str, size: int) -> np.ndarray:
    value = _column(surface_columns, name)
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        if isinstance(value, pa.ChunkedArray):
            value = value.combine_chunks()
        return np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float64)
    return array.reshape(-1)


def _string_column(surface_columns: Any, name: str, size: int) -> np.ndarray:
    value = _column(surface_columns, name)
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return np.asarray(value.to_pylist(), dtype=object)
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        array = np.full(size, str(array), dtype=object)
    return array.reshape(-1)


def _list_value(column: Any, index: int) -> np.ndarray:
    if isinstance(column, pa.ChunkedArray):
        scalar = column[index]
        value = scalar.as_py() if scalar.is_valid else ()
    elif isinstance(column, pa.Array):
        scalar = column[index]
        value = scalar.as_py() if scalar.is_valid else ()
    else:
        value = column[index]
    return np.asarray(value or (), dtype=np.float64)


def evaluate_surface_iv_targets(
    surface_columns: Mapping[str, Any] | pa.Table,
    target_moneyness: Sequence[float] | np.ndarray | float,
) -> np.ndarray:
    """Evaluate one exact log-forward-moneyness target per surface row.

    SVI rows are evaluated in one indexed NumPy operation. PCHIP list columns
    remain Arrow-native and only the selected PCHIP row's two small knot arrays
    cross the Python boundary. Invalid, nonpositive, or uncovered results are
    returned as ``NaN``; fitted surfaces are never extrapolated.
    """

    if isinstance(surface_columns, Mapping) and isinstance(
        surface_columns.get("fit_method"), str
    ):
        # Convenient single-row form used by narrow control-plane summaries.
        surface_columns = {
            name: [value]
            for name, value in surface_columns.items()
        }
    target = np.asarray(target_moneyness, dtype=np.float64)
    if target.ndim == 0:
        try:
            size = len(_column(surface_columns, "fit_method"))
        except TypeError:
            size = 1
        target = np.full(size, float(target), dtype=np.float64)
    else:
        target = target.reshape(-1)
        size = len(target)
    if size == 0:
        return np.asarray([], dtype=np.float64)

    methods = np.char.upper(
        np.char.strip(_string_column(surface_columns, "fit_method", size).astype(str))
    )
    minimum = _numeric_column(
        surface_columns, "minimum_log_forward_moneyness", size
    )
    maximum = _numeric_column(
        surface_columns, "maximum_log_forward_moneyness", size
    )
    tenor_years = (
        _numeric_column(surface_columns, "time_to_expiry_seconds", size)
        / SECONDS_PER_YEAR
    )
    if not all(len(value) == size for value in (methods, minimum, maximum, tenor_years)):
        raise ValueError("Surface columns and target_moneyness are not aligned")

    covered = (
        np.isfinite(target)
        & np.isfinite(minimum)
        & np.isfinite(maximum)
        & (minimum < maximum)
        & (target >= minimum)
        & (target <= maximum)
        & np.isfinite(tenor_years)
        & (tenor_years > 0.0)
    )
    total_variance = np.full(size, np.nan, dtype=np.float64)

    svi = covered & (methods == "SVI")
    if np.any(svi):
        indices = np.flatnonzero(svi)
        a = _numeric_column(surface_columns, "svi_a", size)[indices]
        b = _numeric_column(surface_columns, "svi_b", size)[indices]
        rho = _numeric_column(surface_columns, "svi_rho", size)[indices]
        center = _numeric_column(surface_columns, "svi_m", size)[indices]
        sigma = _numeric_column(surface_columns, "svi_sigma", size)[indices]
        shifted = target[indices] - center
        values = a + b * (
            rho * shifted + np.sqrt(shifted * shifted + sigma * sigma)
        )
        valid_parameters = (
            np.isfinite(a)
            & np.isfinite(b)
            & np.isfinite(rho)
            & np.isfinite(center)
            & np.isfinite(sigma)
            & (sigma > 0.0)
        )
        total_variance[indices[valid_parameters]] = values[valid_parameters]

    pchip = np.flatnonzero(covered & (methods == "PCHIP"))
    if len(pchip):
        knots_column = _column(surface_columns, "pchip_k")
        variance_column = _column(surface_columns, "pchip_total_variance")
        # PCHIP fits have short, variable-size list columns. A narrow loop is
        # both bounded and avoids materializing the complete Arrow table.
        for index in pchip:
            knots = _list_value(knots_column, int(index))
            variances = _list_value(variance_column, int(index))
            if (
                len(knots) < 2
                or len(knots) != len(variances)
                or not np.all(np.isfinite(knots))
                or not np.all(np.isfinite(variances))
                or np.any(np.diff(knots) <= 0.0)
            ):
                continue
            # Coverage bounds may be narrower than the fit knots, but never
            # allow SciPy to extrapolate beyond the knots themselves.
            if target[index] < knots[0] or target[index] > knots[-1]:
                continue
            try:
                total_variance[index] = float(
                    PchipInterpolator(knots, variances, extrapolate=False)(
                        target[index]
                    )
                )
            except (TypeError, ValueError, FloatingPointError):
                continue

    with np.errstate(invalid="ignore", divide="ignore"):
        iv = np.sqrt(total_variance / tenor_years)
    iv[~covered | ~np.isfinite(iv) | (iv <= 0.0)] = np.nan
    return iv


def european_prices(
    *,
    spot: np.ndarray,
    strike: np.ndarray,
    tenor_seconds: np.ndarray,
    rate: np.ndarray,
    dividend_yield: np.ndarray,
    volatility: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """Vectorized Black-Scholes prices used only for European source rows."""

    tenor = tenor_seconds / SECONDS_PER_YEAR
    sqrt_t = np.sqrt(tenor)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * tenor
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discounted_spot = spot * np.exp(-dividend_yield * tenor)
    discounted_strike = strike * np.exp(-rate * tenor)
    call = discounted_spot * ndtr(d1) - discounted_strike * ndtr(d2)
    put = discounted_strike * ndtr(-d2) - discounted_spot * ndtr(-d1)
    return np.where(is_call, call, put)


def create_solver_registry(context: Mapping[str, Any]) -> Any:
    from algotrading.refdata.options.solvers.Registry import SolverRegistry

    return SolverRegistry(context)


def _normalized_method(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "cranknicholson": "crank_nicolson",
        "crank_nicholson": "crank_nicolson",
        "cranknicolson": "crank_nicolson",
        "european": "none",
        "black_scholes": "none",
    }
    return aliases.get(normalized, normalized)


def price_iv_only_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    registry: Any | None = None,
) -> np.ndarray:
    """Price arbitrary IV-only request jobs with the canonical solver route."""

    if not jobs:
        return np.asarray([], dtype=np.float64)
    registry = registry or create_solver_registry(context)
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, job in enumerate(jobs):
        grouped[_normalized_method(job["pricing_method"])].append((index, job))
    output = np.full(len(jobs), np.nan, dtype=np.float64)
    for method, members in grouped.items():
        if method == "none":
            member_jobs = [item[1] for item in members]
            values = european_prices(
                spot=np.asarray([item["spot"] for item in member_jobs], dtype=float),
                strike=np.asarray([item["strike"] for item in member_jobs], dtype=float),
                tenor_seconds=np.asarray(
                    [item["tenor_seconds"] for item in member_jobs], dtype=float
                ),
                rate=np.asarray([item["rate"] for item in member_jobs], dtype=float),
                dividend_yield=np.asarray(
                    [item["dividend_yield"] for item in member_jobs], dtype=float
                ),
                volatility=np.asarray(
                    [item["volatility"] for item in member_jobs], dtype=float
                ),
                is_call=np.asarray(
                    [item["put_call"] == "CALL" for item in member_jobs], dtype=bool
                ),
            )
        else:
            from algotrading.refdata.options.solvers.Types import PricingRequest

            requests = [
                PricingRequest(
                    option_symbol=f"cache-api:{index}",
                    snapshot_id=str(job.get("snapshot_id", "cache-api")),
                    put_call=str(job["put_call"]),
                    exercise_type="AMERICAN",
                    spot=float(job["spot"]),
                    strike=float(job["strike"]),
                    tenor_seconds=float(job["tenor_seconds"]),
                    rate=float(job["rate"]),
                    dividend_yield=float(job["dividend_yield"]),
                    source_collected_at=job.get("source_collected_at"),
                    expiration_at=job.get("expiration_at"),
                    requested_dtype="float64",
                )
                for index, job in members
            ]
            values = registry.get(method).price_many(
                requests,
                [float(job["volatility"]) for _, job in members],
                requested_dtype="float64",
            )
        for (index, _), value in zip(members, values):
            output[index] = float(value)
    return output
