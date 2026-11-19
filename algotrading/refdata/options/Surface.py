"""Deterministic numerical volatility-surface calculations.

This module has no ClickHouse or application-loop dependency.  It owns the
shared precision policy, market-IV inversion, per-expiration SVI/PCHIP fits,
static-arbitrage diagnostics, and standardized surface-node solving.
"""

from __future__ import annotations

import json
import hashlib
import math
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr

from algotrading.refdata.options.Common import (
    OPTIONS_CALCS_CONFIG_PREFIX,
    OPTIONS_CALCS_GREEKS_PREFIX,
    OPTIONS_CALCS_SURFACE_PREFIX,
)


SECONDS_PER_YEAR = 365.25 * 86400.0
SUPPORTED_DTYPES = frozenset({"float32", "float64"})
IV_FAILURE_REASONS = frozenset(
    {
        "crr_iv_lower_bracket_invalid",
        "nonfinite_iv_price",
        "iv_root_not_bracketed",
        "objective_stagnation",
        "iv_price_residual_exceeded",
        "iv_solver_convergence_failure",
    }
)

# Surface quality flags.  These are intentionally stable persisted bits.
QUALITY_SOURCE_TRUNCATED = 1
QUALITY_SOURCE_IV_FALLBACK = 2
QUALITY_PCHIP_FALLBACK = 4
QUALITY_BUTTERFLY_WARNING = 8
QUALITY_CALENDAR_WARNING = 16
QUALITY_INSUFFICIENT_WING = 32
QUALITY_NODE_ORDERING = 64
QUALITY_PRECISION_ESCALATED = 128
QUALITY_INVALID_FITTED_IV_GRID = 256
QUALITY_NONFINITE_CRR_DELTA = 512
QUALITY_EXACT_ROOT_FALLBACK = 1024
QUALITY_NODE_PRECISION_ESCALATED = 2048
QUALITY_PARTIAL_NODE_SET = 4096
QUALITY_CRR_DELTA_SCALAR_RECOVERY = 8192
QUALITY_CRR_DELTA_PRICE_FALLBACK = 16384
QUALITY_CRR_DELTA_STEP_ESCALATED = 32768
QUALITY_CRR_TREE_INPUT_INFEASIBLE = 65536

SURFACE_NODE_REASONS = frozenset(
    {
        "none",
        "outside_smile_coverage",
        "target_delta_outside_coverage",
        "crr_tree_input_infeasible",
        "nonfinite_direct_crr_delta",
        "scalar_direct_delta_recovered",
        "crr_price_finite_difference_recovered",
        "crr_price_nonfinite",
        "delta_root_residual_exceeded",
        "delta_root_solver_failed",
        "solver_delta_failed",
        "solver_circuit_open",
        "solver_infrastructure_failure",
        "solver_unknown_outcome",
        "solver_price_finite_difference_recovered",
        "optional_node_outside_coverage",
    }
)

BUTTERFLY_NEGATIVE_VARIANCE = 1
BUTTERFLY_NEGATIVE_DENSITY = 2
BUTTERFLY_INDETERMINATE = 4
CALENDAR_VARIANCE_INVERSION = 1


class NumericalInstability(RuntimeError):
    """A calculation should be retried at higher precision."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


class SurfaceStageNumericalInstability(NumericalInstability):
    """Retryable numerical instability with an explicit persisted stage."""

    def __init__(self, stage: str, reason: str):
        self.stage = str(stage)
        super().__init__(reason)


class PrecisionEscalationFailure(RuntimeError):
    """A required float64 retry failed and must become a terminal fit row."""

    def __init__(self, escalation_reason: str, retry_error: Exception):
        self.escalation_reason = str(escalation_reason)
        self.retry_error = retry_error
        super().__init__(
            f"float64_retry_failed_after_{self.escalation_reason}: {retry_error}"
        )


class SurfaceStageError(RuntimeError):
    """Typed terminal exception at a persisted surface stage boundary."""

    def __init__(
        self, stage: str, reason: str, cause: Optional[Exception] = None
    ):
        self.stage = str(stage)
        self.reason = str(reason)
        self.cause = cause
        super().__init__(self.reason)


@dataclass(frozen=True)
class SurfaceNumericalRuntime:
    """Resolved shared tree device/chunk settings for legacy diagnostics."""

    device: Any
    chunk_size: int = 50_000

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "SurfaceNumericalRuntime":
        import torch

        # Production surface pricing is registry-routed.  This compatibility
        # runtime is retained only for isolated legacy diagnostics, so it must
        # not inspect or instantiate an inactive CRR configuration.
        return cls(torch.device("cpu"), 50_000)


def _legacy_crr_module(*, steps: int = 256, chunk_size: int = 50_000, device: str = "cpu") -> Any:
    """Resolve the compatibility CRR kernel through the shared registry."""
    from algotrading.refdata.options.solvers.Registry import SolverRegistry

    context = {
        OPTIONS_CALCS_GREEKS_PREFIX + "crr.steps": int(steps),
        OPTIONS_CALCS_GREEKS_PREFIX + "crr.chunk_size": int(chunk_size),
        OPTIONS_CALCS_GREEKS_PREFIX + "crr.device": str(device),
    }
    return SolverRegistry(context).get("crr").module


@dataclass
class IVInversionDiagnostics:
    pricing_batch_calls: int = 0
    contract_price_evaluations: int = 0
    iterations: list[int] = field(default_factory=list)
    solved_at_source_candidate: int = 0
    global_bracket_count: int = 0
    float64_escalation_count: int = 0
    rejected_by_reason: Counter[str] = field(default_factory=Counter)
    float64_retry_contract_count: int = 0
    float64_recovered_contract_count: int = 0
    float64_failed_contract_count: int = 0
    complete_result_count: int = 0
    rejected_result_count: int = 0

    def summary(self) -> dict[str, Any]:
        ordered = sorted(self.iterations)

        def percentile(fraction: float) -> float:
            if not ordered:
                return 0.0
            index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
            return float(ordered[index])

        return {
            "pricing_batch_calls": self.pricing_batch_calls,
            "contract_price_evaluations": self.contract_price_evaluations,
            "mean_iv_iterations": (
                float(np.mean(ordered)) if ordered else 0.0
            ),
            "p50_iv_iterations": percentile(0.50),
            "p95_iv_iterations": percentile(0.95),
            "max_iv_iterations": max(ordered, default=0),
            "solved_at_source_candidate": self.solved_at_source_candidate,
            "global_bracket_count": self.global_bracket_count,
            "float64_escalation_count": self.float64_escalation_count,
            "rejected_by_reason": dict(sorted(self.rejected_by_reason.items())),
            "float64_retry_contract_count": self.float64_retry_contract_count,
            "float64_recovered_contract_count": self.float64_recovered_contract_count,
            "float64_failed_contract_count": self.float64_failed_contract_count,
            "complete_result_count": self.complete_result_count,
            "rejected_result_count": self.rejected_result_count,
        }


@dataclass(frozen=True)
class SVIFitDiagnostics:
    initialization_source: str
    nfev: int
    objective: float
    optimizer_success: bool
    fallback_path: str = ""


class SVIWarmStartCache:
    """Bounded advisory cache; invalid entries never block deterministic fits."""

    def __init__(self, maximum_size: int = 2048):
        if int(maximum_size) <= 0:
            raise ValueError("surface.svi_warm_start_cache_size must be positive")
        self.maximum_size = int(maximum_size)
        self._values: OrderedDict[
            tuple[str, Any, str], Mapping[str, float]
        ] = OrderedDict()

    def get(
        self, key: tuple[str, Any, str]
    ) -> Optional[Mapping[str, float]]:
        value = self._values.get(key)
        if value is None:
            return None
        self._values.move_to_end(key)
        return value

    def put(
        self, key: tuple[str, Any, str], parameters: Mapping[str, Any]
    ) -> None:
        try:
            normalized = {
                name: float(parameters[name])
                for name in ("a", "b", "rho", "m", "sigma")
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        if not all(math.isfinite(value) for value in normalized.values()):
            return
        self._values[key] = MappingProxyType(normalized)
        self._values.move_to_end(key)
        while len(self._values) > self.maximum_size:
            self._values.popitem(last=False)

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True)
class NumericPrecisionPolicy:
    short_tenor_max_seconds: float = 86400.0
    short_tenor_dtype: str = "float64"
    other_tenor_dtype: str = "float32"
    auto_promote_to_float64: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.short_tenor_max_seconds)) or float(
            self.short_tenor_max_seconds
        ) <= 0:
            raise ValueError("short_tenor_max_seconds must be positive and finite")
        for name, value in (
            ("short_tenor_dtype", self.short_tenor_dtype),
            ("other_tenor_dtype", self.other_tenor_dtype),
        ):
            normalized = str(value).strip().lower()
            if normalized not in SUPPORTED_DTYPES:
                raise ValueError(f"{name} must be float32 or float64, got {value!r}")

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "NumericPrecisionPolicy":
        prefix = OPTIONS_CALCS_CONFIG_PREFIX
        return cls(
            short_tenor_max_seconds=float(
                context.get(prefix + "short_tenor_max_seconds", 86400)
            ),
            short_tenor_dtype=str(
                context.get(prefix + "short_tenor_dtype", "float64")
            ).lower(),
            other_tenor_dtype=str(
                context.get(prefix + "other_tenor_dtype", "float32")
            ).lower(),
            auto_promote_to_float64=_as_bool(
                context.get(prefix + "auto_promote_to_float64", True)
            ),
        )


@dataclass(frozen=True)
class PrecisionExecution:
    value: Any
    requested_dtype: str
    actual_dtype: str
    escalated: bool
    escalation_reason: str


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def resolve_numeric_dtype(
    tenor_seconds: float,
    short_tenor_max_seconds: float = 86400.0,
    short_tenor_dtype: str = "float64",
    other_tenor_dtype: str = "float32",
) -> np.dtype:
    """Resolve the explicit shared dtype policy at a numerical boundary."""
    tenor = float(tenor_seconds)
    threshold = float(short_tenor_max_seconds)
    if not math.isfinite(tenor) or tenor < 0:
        raise ValueError("tenor_seconds must be non-negative and finite")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("short_tenor_max_seconds must be positive and finite")
    short_name = str(short_tenor_dtype).strip().lower()
    other_name = str(other_tenor_dtype).strip().lower()
    if short_name not in SUPPORTED_DTYPES or other_name not in SUPPORTED_DTYPES:
        raise ValueError("numeric dtypes must be float32 or float64")
    return np.dtype(short_name if tenor <= threshold else other_name)


def execute_with_precision_policy(
    tenor_seconds: float,
    policy: NumericPrecisionPolicy,
    operation: Callable[[np.dtype], Any],
) -> PrecisionExecution:
    requested = resolve_numeric_dtype(
        tenor_seconds,
        policy.short_tenor_max_seconds,
        policy.short_tenor_dtype,
        policy.other_tenor_dtype,
    ).name
    try:
        value = operation(np.dtype(requested))
        _assert_finite_result(value)
        return PrecisionExecution(value, requested, requested, False, "")
    except NumericalInstability as exc:
        if (
            requested != "float32"
            or not policy.auto_promote_to_float64
        ):
            raise
        try:
            value = operation(np.dtype("float64"))
            _assert_finite_result(value)
        except Exception as retry_error:
            raise PrecisionEscalationFailure(exc.reason, retry_error) from retry_error
        return PrecisionExecution(value, requested, "float64", True, exc.reason)


def _assert_finite_result(value: Any) -> None:
    checker = getattr(value, "assert_finite", None)
    if callable(checker):
        checker()
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise NumericalInstability("nonfinite_output")
    if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
        raise NumericalInstability("nonfinite_output")


def _dtype_scalar(dtype: np.dtype, value: Any) -> np.floating:
    return dtype.type(value)


def european_option_price(
    *,
    spot: float,
    strike: float,
    tenor_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    dtype: np.dtype | str = np.float64,
) -> float:
    dtype = np.dtype(dtype)
    s = _dtype_scalar(dtype, spot)
    k = _dtype_scalar(dtype, strike)
    t = _dtype_scalar(dtype, tenor_years)
    r = _dtype_scalar(dtype, rate)
    q = _dtype_scalar(dtype, dividend_yield)
    sigma = _dtype_scalar(dtype, volatility)
    if not all(
        math.isfinite(float(value)) for value in (s, k, t, r, q, sigma)
    ) or s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return math.nan
    sqrt_t = dtype.type(np.sqrt(t, dtype=dtype))
    log_sk = dtype.type(np.log(s / k, dtype=dtype))
    d1 = dtype.type((log_sk + (r - q + dtype.type(0.5) * sigma * sigma) * t) / (sigma * sqrt_t))
    d2 = dtype.type(d1 - sigma * sqrt_t)
    disc_q = dtype.type(np.exp(-q * t, dtype=dtype))
    disc_r = dtype.type(np.exp(-r * t, dtype=dtype))
    if is_call:
        price = disc_q * s * dtype.type(ndtr(float(d1))) - disc_r * k * dtype.type(ndtr(float(d2)))
    else:
        price = disc_r * k * dtype.type(ndtr(float(-d2))) - disc_q * s * dtype.type(ndtr(float(-d1)))
    return float(price)


def european_option_delta(
    *,
    spot: float,
    strike: float,
    tenor_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    dtype: np.dtype | str = np.float64,
) -> float:
    dtype = np.dtype(dtype)
    s, k, t = (dtype.type(value) for value in (spot, strike, tenor_years))
    r, q, sigma = (dtype.type(value) for value in (rate, dividend_yield, volatility))
    if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return math.nan
    sqrt_t = dtype.type(np.sqrt(t, dtype=dtype))
    d1 = dtype.type(
        (np.log(s / k, dtype=dtype) + (r - q + dtype.type(0.5) * sigma * sigma) * t)
        / (sigma * sqrt_t)
    )
    disc_q = dtype.type(np.exp(-q * t, dtype=dtype))
    call_delta = disc_q * dtype.type(ndtr(float(d1)))
    return float(call_delta if is_call else call_delta - disc_q)


def option_price_bounds(
    *,
    spot: float,
    strike: float,
    tenor_years: float,
    rate: float,
    dividend_yield: float,
    is_call: bool,
    exercise_type: str,
) -> tuple[float, float]:
    s, k, t, r, q = map(
        float, (spot, strike, tenor_years, rate, dividend_yield)
    )
    if not all(math.isfinite(value) for value in (s, k, t, r, q)):
        return math.nan, math.nan
    exercise = str(exercise_type).strip().upper()
    disc_s = s * math.exp(-q * t)
    disc_k = k * math.exp(-r * t)
    if is_call:
        european_lower = max(0.0, disc_s - disc_k)
        if exercise in {"A", "AMERICAN"}:
            return max(european_lower, s - k, 0.0), s
        return european_lower, disc_s
    european_lower = max(0.0, disc_k - disc_s)
    if exercise in {"A", "AMERICAN"}:
        return max(european_lower, k - s, 0.0), k
    return european_lower, disc_k


def american_option_price(
    *,
    spot: float,
    strike: float,
    tenor_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    steps: int,
    dtype: np.dtype | str,
    runtime: Optional[SurfaceNumericalRuntime] = None,
) -> float:
    """Dispatch to the repository's shared CRR price kernel."""
    import torch
    np_dtype = np.dtype(dtype)
    torch_dtype = torch.float32 if np_dtype == np.dtype("float32") else torch.float64
    runtime = runtime or SurfaceNumericalRuntime(torch.device("cpu"), 1)
    module = _legacy_crr_module(
        steps=steps,
        chunk_size=runtime.chunk_size,
        device=str(runtime.device),
    )
    output = module.compute_prices_crr_from_iv(
        S=np.asarray([spot], dtype=np.float64),
        K=np.asarray([strike], dtype=np.float64),
        T=np.asarray([tenor_years], dtype=np.float64),
        r=np.asarray([rate], dtype=np.float64),
        q=np.asarray([dividend_yield], dtype=np.float64),
        is_call=np.asarray([is_call], dtype=bool),
        sigma=np.asarray([volatility], dtype=np.float64),
        steps=int(steps),
        dtype=torch_dtype,
        device=runtime.device,
        chunk_size=max(1, min(runtime.chunk_size, 1)),
        to_numpy=True,
    )
    return float(output[0])


def model_option_price(
    *,
    exercise_type: str,
    spot: float,
    strike: float,
    tenor_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    dtype: np.dtype | str,
    crr_steps: int = 256,
    runtime: Optional[SurfaceNumericalRuntime] = None,
) -> float:
    exercise = str(exercise_type).strip().upper()
    if exercise in {"E", "EUROPEAN"}:
        return european_option_price(
            spot=spot,
            strike=strike,
            tenor_years=tenor_years,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            is_call=is_call,
            dtype=dtype,
        )
    if exercise in {"A", "AMERICAN"}:
        return american_option_price(
            spot=spot,
            strike=strike,
            tenor_years=tenor_years,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            is_call=is_call,
            steps=crr_steps,
            dtype=dtype,
            runtime=runtime,
        )
    raise ValueError(f"unsupported_exercise_style:{exercise_type}")


@dataclass(frozen=True)
class IVInversionResult:
    iv: Optional[float]
    status: str
    reason: str
    model_price: Optional[float]
    price_residual: Optional[float]
    iterations: int
    requested_dtype: str = "float64"
    actual_dtype: str = "float64"
    precision_escalated: bool = False
    precision_escalation_reason: str = ""


def american_crr_sigma_floor(
    *,
    tenor_years: np.ndarray,
    rate: np.ndarray,
    dividend_yield: np.ndarray,
    steps: int,
    probability_epsilon: float = 1.0e-3,
    dtype: np.dtype | str = np.float64,
) -> np.ndarray:
    """Return the shared stable CRR probability-feasibility floor."""
    import torch

    module = _legacy_crr_module(steps=steps)

    dtype = np.dtype(dtype)
    if dtype.name not in SUPPORTED_DTYPES:
        raise ValueError("dtype must be float32 or float64")
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    epsilon = float(probability_epsilon)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("probability_epsilon must be finite and between zero and 0.5")
    tenor, rates, dividends = np.broadcast_arrays(
        np.asarray(tenor_years, dtype=np.float64),
        np.asarray(rate, dtype=np.float64),
        np.asarray(dividend_yield, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(tenor))
        or np.any(tenor <= 0.0)
        or np.any(~np.isfinite(rates))
        or np.any(~np.isfinite(dividends))
    ):
        raise ValueError("CRR floor inputs require finite positive tenor and finite rates")
    torch_dtype = torch.float32 if dtype == np.dtype("float32") else torch.float64
    floors = module.crr_sigma_floor(
        torch.as_tensor(tenor, dtype=torch_dtype),
        torch.as_tensor(rates, dtype=torch_dtype),
        torch.as_tensor(dividends, dtype=torch_dtype),
        steps=int(steps),
        sigma_lo=0.0,
        p_epsilon=epsilon,
        min_sigma_abs=0.0,
    )
    return floors.detach().cpu().numpy().astype(dtype, copy=False)


def invert_implied_volatility(
    *,
    price: float,
    spot: float,
    strike: float,
    tenor_years: float,
    rate: float,
    dividend_yield: float,
    is_call: bool,
    exercise_type: str,
    minimum_iv: float = 1.0e-4,
    maximum_iv: float = 10.0,
    price_tolerance: float = 1.0e-7,
    max_iterations: int = 100,
    source_iv: Optional[float] = None,
    dtype: np.dtype | str = np.float64,
    crr_steps: int = 256,
    runtime: Optional[SurfaceNumericalRuntime] = None,
) -> IVInversionResult:
    dtype = np.dtype(dtype)
    values = (price, spot, strike, tenor_years, rate, dividend_yield)
    if not all(math.isfinite(float(value)) for value in values):
        return IVInversionResult(None, "rejected", "nonfinite_input", None, None, 0)
    if price < 0:
        return IVInversionResult(None, "rejected", "negative_price", None, None, 0)
    if spot <= 0 or strike <= 0 or tenor_years <= 0:
        return IVInversionResult(None, "rejected", "invalid_model_input", None, None, 0)
    exercise = str(exercise_type).strip().upper()
    if exercise not in {"A", "AMERICAN", "E", "EUROPEAN"}:
        return IVInversionResult(None, "rejected", "unsupported_exercise_style", None, None, 0)
    lower_bound, upper_bound = option_price_bounds(
        spot=spot,
        strike=strike,
        tenor_years=tenor_years,
        rate=rate,
        dividend_yield=dividend_yield,
        is_call=is_call,
        exercise_type=exercise,
    )
    scaled_tolerance = max(
        float(price_tolerance),
        np.finfo(dtype).eps * max(abs(price), abs(upper_bound), 1.0) * 16.0,
    )
    if price < lower_bound - scaled_tolerance:
        return IVInversionResult(None, "rejected", "price_below_lower_bound", None, None, 0)
    if price > upper_bound + scaled_tolerance:
        return IVInversionResult(None, "rejected", "price_above_upper_bound", None, None, 0)

    evaluations = 0

    def objective(sigma: float) -> float:
        nonlocal evaluations
        evaluations += 1
        model = model_option_price(
            exercise_type=exercise,
            spot=spot,
            strike=strike,
            tenor_years=tenor_years,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=sigma,
            is_call=is_call,
            dtype=dtype,
            crr_steps=crr_steps,
            runtime=runtime,
        )
        if not math.isfinite(model):
            raise NumericalInstability("nonfinite_iv_price")
        return model - price

    lo, hi = float(minimum_iv), float(maximum_iv)
    if not 0 < lo < hi or not math.isfinite(hi):
        raise ValueError("IV bounds must be finite and satisfy 0 < minimum < maximum")
    try:
        f_lo = objective(lo)
    except NumericalInstability:
        if exercise not in {"A", "AMERICAN"}:
            raise
        # CRR requires a tenor/rate-dependent minimum sigma to keep its
        # risk-neutral probability strictly inside (0, 1).
        found = False
        for candidate in np.geomspace(max(lo, 1.0e-3), min(hi, 1.0), 32):
            try:
                f_lo = objective(float(candidate))
                lo = float(candidate)
                found = True
                break
            except NumericalInstability:
                continue
        if not found:
            raise NumericalInstability("crr_iv_lower_bracket_invalid")
    f_hi = objective(hi)
    if abs(f_lo) <= scaled_tolerance:
        model = f_lo + price
        return IVInversionResult(lo, "complete", "", model, model - price, evaluations)
    if abs(f_hi) <= scaled_tolerance:
        model = f_hi + price
        return IVInversionResult(hi, "complete", "", model, model - price, evaluations)
    if f_lo * f_hi > 0:
        # A valid theoretical price without a numerical bracket is a precision
        # or model-grid failure, not a market-input rejection.
        raise NumericalInstability("iv_root_not_bracketed")

    bracket_lo, bracket_hi = lo, hi
    source_value = None
    try:
        source_value = float(source_iv) if source_iv is not None else None
    except (TypeError, ValueError):
        source_value = None
    if source_value is not None and math.isfinite(source_value) and lo < source_value < hi:
        candidates = sorted(
            {
                lo,
                max(lo, source_value * 0.5),
                source_value,
                min(hi, source_value * 1.5 + 0.05),
                hi,
            }
        )
        previous_x, previous_f = candidates[0], f_lo
        for candidate in candidates[1:]:
            candidate_f = f_hi if candidate == hi else objective(candidate)
            if previous_f * candidate_f <= 0:
                bracket_lo, bracket_hi = previous_x, candidate
                break
            previous_x, previous_f = candidate, candidate_f
    try:
        root, result = brentq(
            objective,
            bracket_lo,
            bracket_hi,
            xtol=max(1.0e-12, np.finfo(dtype).eps * 4.0),
            rtol=max(1.0e-12, np.finfo(dtype).eps * 8.0),
            maxiter=int(max_iterations),
            full_output=True,
            disp=False,
        )
    except NumericalInstability:
        raise
    except Exception as exc:
        raise NumericalInstability(f"iv_solver_failure:{type(exc).__name__}") from exc
    if not result.converged or not math.isfinite(float(root)):
        raise NumericalInstability("iv_solver_convergence_failure")
    model_price = objective(float(root)) + price
    residual = model_price - price
    if not math.isfinite(model_price) or abs(residual) > scaled_tolerance * 4.0:
        raise NumericalInstability("iv_price_residual_exceeded")
    return IVInversionResult(
        float(root), "complete", "", float(model_price), float(residual), evaluations
    )


def invert_american_implied_volatilities_batch(
    inputs: Sequence[Mapping[str, Any]],
    *,
    minimum_iv: float = 1.0e-4,
    maximum_iv: float = 10.0,
    price_tolerance: float = 1.0e-7,
    max_iterations: int = 100,
    residual_tolerance: Optional[float] = None,
    dtype: np.dtype | str = np.float64,
    crr_steps: int = 256,
    source_bracket_multipliers: Sequence[float] = (0.50, 0.75, 1.0, 1.25, 1.50),
    runtime: Optional[SurfaceNumericalRuntime] = None,
    diagnostics: Optional[IVInversionDiagnostics] = None,
    auto_promote_to_float64: bool = False,
    _record_terminal_diagnostics: bool = True,
) -> tuple[IVInversionResult, ...]:
    """Vectorized bracketed American IV inversion with contract-local failures."""
    if not inputs:
        return ()
    dtype = np.dtype(dtype)
    if dtype.name not in SUPPORTED_DTYPES:
        raise ValueError("dtype must be float32 or float64")
    if (
        not 0.0 < float(minimum_iv) < float(maximum_iv)
        or not math.isfinite(float(maximum_iv))
    ):
        raise ValueError("IV bounds must be finite and satisfy 0 < minimum < maximum")
    if int(max_iterations) <= 0 or int(crr_steps) <= 0:
        raise ValueError("max_iterations and crr_steps must be positive")
    if not math.isfinite(float(price_tolerance)) or float(price_tolerance) < 0.0:
        raise ValueError("price_tolerance must be finite and non-negative")
    if residual_tolerance is not None and (
        not math.isfinite(float(residual_tolerance))
        or float(residual_tolerance) < 0.0
    ):
        raise ValueError("residual_tolerance must be finite and non-negative")
    count = len(inputs)
    normalized_multipliers = [
        float(value) for value in source_bracket_multipliers
    ]
    if (
        not normalized_multipliers
        or any(
            not math.isfinite(value) or value <= 0
            for value in normalized_multipliers
        )
        or len(normalized_multipliers) != len(set(normalized_multipliers))
        or 1.0 not in normalized_multipliers
    ):
        raise ValueError(
            "source IV bracket multipliers must be finite, positive, unique, and include 1.0"
        )
    multipliers = sorted(normalized_multipliers)
    diagnostics = diagnostics or IVInversionDiagnostics()
    names = ("price", "spot", "strike", "tenor_years", "rate", "dividend_yield")
    arrays = {
        name: np.full(count, np.nan, dtype=np.float64) for name in names
    }
    calls = np.zeros(count, dtype=bool)
    results: list[Optional[IVInversionResult]] = [None] * count
    valid = np.zeros(count, dtype=bool)

    def reject(
        index: int,
        reason: str,
        *,
        model_price: Optional[float] = None,
        price_residual: Optional[float] = None,
        iterations: int = 0,
    ) -> None:
        results[index] = IVInversionResult(
            None,
            "rejected",
            reason,
            model_price if model_price is not None and math.isfinite(model_price) else None,
            (
                price_residual
                if price_residual is not None and math.isfinite(price_residual)
                else None
            ),
            int(iterations),
            requested_dtype=dtype.name,
            actual_dtype=dtype.name,
        )
        valid[index] = False

    for index, item in enumerate(inputs):
        try:
            for name in names:
                arrays[name][index] = float(item[name])
            calls[index] = bool(item["is_call"])
        except (KeyError, TypeError, ValueError, OverflowError):
            reject(index, "nonfinite_input")
            continue
        if any(not math.isfinite(arrays[name][index]) for name in names):
            reject(index, "nonfinite_input")
            continue
        if arrays["price"][index] < 0.0:
            reject(index, "negative_price")
            continue
        if (
            arrays["spot"][index] <= 0.0
            or arrays["strike"][index] <= 0.0
            or arrays["tenor_years"][index] <= 0.0
        ):
            reject(index, "invalid_model_input")
            continue
        valid[index] = True

    lower_bounds = np.zeros(count, dtype=np.float64)
    upper_bounds = np.zeros(count, dtype=np.float64)
    for index in np.flatnonzero(valid):
        lower, upper = option_price_bounds(
            spot=arrays["spot"][index],
            strike=arrays["strike"][index],
            tenor_years=arrays["tenor_years"][index],
            rate=arrays["rate"][index],
            dividend_yield=arrays["dividend_yield"][index],
            is_call=bool(calls[index]),
            exercise_type="A",
        )
        if not math.isfinite(lower) or not math.isfinite(upper):
            reject(index, "invalid_model_input")
            continue
        lower_bounds[index], upper_bounds[index] = lower, upper
    tolerances = np.maximum(
        float(price_tolerance),
        np.finfo(dtype).eps
        * np.maximum.reduce(
            (np.abs(arrays["price"]), np.abs(upper_bounds), np.ones(count))
        )
        * 16.0,
    )
    solver_tolerances = tolerances.copy()
    if residual_tolerance is not None:
        solver_tolerances = np.maximum(
            float(price_tolerance),
            np.minimum(solver_tolerances, float(residual_tolerance)),
        )
    for index in np.flatnonzero(valid):
        price = arrays["price"][index]
        if price < lower_bounds[index] - tolerances[index]:
            reject(index, "price_below_lower_bound")
        elif price > upper_bounds[index] + tolerances[index]:
            reject(index, "price_above_upper_bound")

    # Mirror the shared CRR probability floor so every lower bracket is a
    # valid tree input, with a small dtype-aware margin above the boundary.
    probability_epsilon = 1.0e-3
    sigma_floor = np.full(count, np.nan, dtype=np.float64)
    floor_indices = np.flatnonzero(valid)
    if len(floor_indices):
        sigma_floor[floor_indices] = american_crr_sigma_floor(
            tenor_years=arrays["tenor_years"][floor_indices],
            rate=arrays["rate"][floor_indices],
            dividend_yield=arrays["dividend_yield"][floor_indices],
            steps=int(crr_steps),
            probability_epsilon=probability_epsilon,
            dtype=dtype,
        )
    lo = np.maximum.reduce(
        (
            np.full(count, float(minimum_iv)),
            np.full(count, 1.0e-2),
            sigma_floor * (1.0 + max(1.0e-7, np.finfo(dtype).eps * 16.0)),
        )
    )
    hi = np.full(count, float(maximum_iv), dtype=np.float64)
    for index in np.flatnonzero(valid):
        if not math.isfinite(sigma_floor[index]) or lo[index] >= hi[index]:
            reject(index, "crr_iv_lower_bracket_invalid")
    global_lo = lo.copy()
    global_hi = hi.copy()

    import torch

    torch_dtype = torch.float32 if dtype == np.dtype("float32") else torch.float64
    runtime = runtime or SurfaceNumericalRuntime(torch.device("cpu"), 50_000)
    module = _legacy_crr_module(
        steps=crr_steps,
        chunk_size=runtime.chunk_size,
        device=str(runtime.device),
    )

    def price_jobs(indices: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        if not len(indices):
            return np.empty(0, dtype=np.float64)
        diagnostics.pricing_batch_calls += 1
        diagnostics.contract_price_evaluations += len(indices)
        output = module.compute_prices_crr_from_iv(
            S=arrays["spot"][indices],
            K=arrays["strike"][indices],
            T=arrays["tenor_years"][indices],
            r=arrays["rate"][indices],
            q=arrays["dividend_yield"][indices],
            is_call=calls[indices],
            sigma=np.asarray(sigmas, dtype=np.float64),
            steps=int(crr_steps),
            dtype=torch_dtype,
            device=runtime.device,
            chunk_size=max(1, min(runtime.chunk_size, len(indices))),
            to_numpy=True,
        )
        converted = np.asarray(output, dtype=np.float64).reshape(-1)
        if len(converted) != len(indices):
            raise RuntimeError(
                "CRR price kernel output shape mismatch: "
                f"expected {len(indices)}, received {len(converted)}"
            )
        return converted

    def prices(sigmas: np.ndarray, mask: np.ndarray) -> np.ndarray:
        indices = np.flatnonzero(mask)
        return price_jobs(indices, np.asarray(sigmas[indices], dtype=np.float64))

    f_lo = np.zeros(count, dtype=np.float64)
    f_hi = np.zeros(count, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        boundary_indices = np.concatenate((valid_indices, valid_indices))
        boundary_sigmas = np.concatenate((lo[valid], hi[valid]))
        boundary_prices = price_jobs(boundary_indices, boundary_sigmas)
        split = len(valid_indices)
        low_prices, high_prices = boundary_prices[:split], boundary_prices[split:]
        for position, index in enumerate(valid_indices):
            if not math.isfinite(low_prices[position]) or not math.isfinite(
                high_prices[position]
            ):
                reject(index, "nonfinite_iv_price")
                continue
            f_lo[index] = low_prices[position] - arrays["price"][index]
            f_hi[index] = high_prices[position] - arrays["price"][index]
    active = valid.copy()
    iterations = np.zeros(count, dtype=np.int64)
    roots = np.full(count, np.nan, dtype=np.float64)
    model_prices = np.full(count, np.nan, dtype=np.float64)
    for index in np.flatnonzero(valid):
        if abs(f_lo[index]) <= solver_tolerances[index]:
            roots[index], model_prices[index], active[index] = (
                lo[index], f_lo[index] + arrays["price"][index], False
            )
        elif abs(f_hi[index]) <= solver_tolerances[index]:
            roots[index], model_prices[index], active[index] = (
                hi[index], f_hi[index] + arrays["price"][index], False
            )
        elif f_lo[index] * f_hi[index] > 0:
            reject(index, "iv_root_not_bracketed")
            active[index] = False

    # Evaluate the complete source-IV multiplier grid in one flattened CRR
    # batch.  Candidates are hints only and are never accepted without a
    # model-price residual check.
    guesses = np.asarray(
        [float(item.get("source_iv") or math.nan) for item in inputs],
        dtype=np.float64,
    )
    candidate_indices: list[int] = []
    candidate_sigmas: list[float] = []
    for index in np.flatnonzero(
        active & np.isfinite(guesses) & (guesses > 0)
    ):
        values = sorted(
            {
                min(max(float(guesses[index]) * multiplier, lo[index]), hi[index])
                for multiplier in multipliers
            }
        )
        candidate_indices.extend([int(index)] * len(values))
        candidate_sigmas.extend(values)
    candidate_prices_by_index: dict[int, list[tuple[float, float]]] = {}
    if candidate_indices:
        candidate_prices = price_jobs(
            np.asarray(candidate_indices, dtype=np.int64),
            np.asarray(candidate_sigmas, dtype=np.float64),
        )
        for index, sigma, model_price in zip(
            candidate_indices, candidate_sigmas, candidate_prices
        ):
            if not math.isfinite(model_price):
                continue
            candidate_prices_by_index.setdefault(index, []).append(
                (sigma, model_price)
            )
    for index in np.flatnonzero(active):
        points = [(lo[index], f_lo[index])]
        solved = False
        for sigma, model_price in candidate_prices_by_index.get(int(index), []):
            error = model_price - arrays["price"][index]
            if abs(error) <= solver_tolerances[index]:
                roots[index], model_prices[index], active[index] = (
                    sigma,
                    model_price,
                    False,
                )
                diagnostics.solved_at_source_candidate += 1
                solved = True
                break
            points.append((sigma, error))
        if solved:
            continue
        points.append((hi[index], f_hi[index]))
        points = sorted(set(points), key=lambda item: item[0])
        brackets = [
            (left, right)
            for left, right in zip(points, points[1:])
            if left[1] * right[1] <= 0
        ]
        if brackets:
            source_value = guesses[index]
            left, right = min(
                brackets,
                key=lambda pair: (
                    not (
                        math.isfinite(source_value)
                        and pair[0][0] <= source_value <= pair[1][0]
                    ),
                    pair[1][0] - pair[0][0],
                ),
            )
            lo[index], f_lo[index] = left
            hi[index], f_hi[index] = right
            if (
                left[0] == global_lo[index]
                and right[0] == global_hi[index]
            ):
                diagnostics.global_bracket_count += 1
        else:
            diagnostics.global_bracket_count += 1

    for iteration in range(1, int(max_iterations) + 1):
        if not np.any(active):
            break
        midpoint = (lo + hi) / 2.0
        stalled = active & ((midpoint == lo) | (midpoint == hi))
        if np.any(stalled):
            stalled_indices = np.flatnonzero(stalled)
            stalled_prices = price_jobs(
                np.concatenate((stalled_indices, stalled_indices)),
                np.concatenate((lo[stalled], hi[stalled])),
            )
            split = len(stalled_indices)
            low_prices, high_prices = stalled_prices[:split], stalled_prices[split:]
            for position, index in enumerate(stalled_indices):
                if not math.isfinite(low_prices[position]) or not math.isfinite(
                    high_prices[position]
                ):
                    reject(index, "nonfinite_iv_price", iterations=int(iterations[index]))
                    active[index] = False
                    continue
                low_error = abs(low_prices[position] - arrays["price"][index])
                high_error = abs(high_prices[position] - arrays["price"][index])
                if min(low_error, high_error) <= solver_tolerances[index]:
                    use_low = low_error <= high_error
                    roots[index] = lo[index] if use_low else hi[index]
                    model_prices[index] = (
                        low_prices[position] if use_low else high_prices[position]
                    )
                    active[index] = False
                else:
                    reject(
                        index,
                        "objective_stagnation",
                        iterations=int(iterations[index]),
                    )
                    active[index] = False
            if not np.any(active):
                break
            midpoint = (lo + hi) / 2.0
        midpoint_prices = prices(midpoint, active)
        for position, index in enumerate(np.flatnonzero(active)):
            iterations[index] = iteration
            model_price = midpoint_prices[position]
            if not math.isfinite(model_price):
                reject(index, "nonfinite_iv_price", iterations=iteration)
                active[index] = False
                continue
            error = model_price - arrays["price"][index]
            if abs(error) <= solver_tolerances[index]:
                roots[index] = midpoint[index]
                model_prices[index] = model_price
                active[index] = False
            elif error > 0:
                hi[index] = midpoint[index]
            else:
                lo[index] = midpoint[index]
    if np.any(active):
        roots[active] = (lo[active] + hi[active]) / 2.0
        terminal_indices = np.flatnonzero(active)
        terminal_prices = prices(roots, active)
        for position, index in enumerate(terminal_indices):
            if math.isfinite(terminal_prices[position]):
                model_prices[index] = terminal_prices[position]
            else:
                reject(
                    index,
                    "nonfinite_iv_price",
                    iterations=int(iterations[index]),
                )
                active[index] = False
    for index in np.flatnonzero(valid):
        if results[index] is not None:
            continue
        residual = model_prices[index] - arrays["price"][index]
        if not math.isfinite(model_prices[index]) or not math.isfinite(roots[index]):
            reject(
                index,
                "nonfinite_iv_price",
                iterations=int(iterations[index]),
            )
            continue
        if abs(residual) > solver_tolerances[index]:
            reject(
                index,
                "iv_price_residual_exceeded",
                model_price=float(model_prices[index]),
                price_residual=float(residual),
                iterations=int(iterations[index]),
            )
            continue
        results[index] = IVInversionResult(
            float(roots[index]),
            "complete",
            "",
            float(model_prices[index]),
            float(residual),
            int(iterations[index]),
            requested_dtype=dtype.name,
            actual_dtype=dtype.name,
        )
    if any(result is None for result in results):
        raise RuntimeError("American IV batch did not produce one terminal result per input")
    terminal_results = tuple(results)  # type: ignore[arg-type]

    retryable_reasons = {
        "crr_iv_lower_bracket_invalid",
        "nonfinite_iv_price",
        "iv_root_not_bracketed",
        "objective_stagnation",
        "iv_price_residual_exceeded",
    }
    retry_indices = [
        index
        for index, result in enumerate(terminal_results)
        if result.status == "rejected" and result.reason in retryable_reasons
    ]
    if (
        retry_indices
        and dtype == np.dtype("float32")
        and bool(auto_promote_to_float64)
    ):
        diagnostics.float64_escalation_count += 1
        diagnostics.float64_retry_contract_count += len(retry_indices)
        retry_results = invert_american_implied_volatilities_batch(
            [inputs[index] for index in retry_indices],
            minimum_iv=minimum_iv,
            maximum_iv=maximum_iv,
            price_tolerance=price_tolerance,
            max_iterations=max_iterations,
            residual_tolerance=residual_tolerance,
            dtype=np.dtype("float64"),
            crr_steps=crr_steps,
            source_bracket_multipliers=source_bracket_multipliers,
            runtime=runtime,
            diagnostics=diagnostics,
            auto_promote_to_float64=False,
            _record_terminal_diagnostics=False,
        )
        merged = list(terminal_results)
        for index, retried in zip(retry_indices, retry_results):
            original_reason = terminal_results[index].reason
            merged[index] = replace(
                retried,
                requested_dtype="float32",
                actual_dtype="float64",
                precision_escalated=True,
                precision_escalation_reason=original_reason,
            )
            if retried.status == "complete":
                diagnostics.float64_recovered_contract_count += 1
            else:
                diagnostics.float64_failed_contract_count += 1
        terminal_results = tuple(merged)

    if _record_terminal_diagnostics:
        complete = [result for result in terminal_results if result.status == "complete"]
        rejected = [result for result in terminal_results if result.status == "rejected"]
        diagnostics.complete_result_count += len(complete)
        diagnostics.rejected_result_count += len(rejected)
        diagnostics.rejected_by_reason.update(
            result.reason or "iv_inversion_failed" for result in rejected
        )
        diagnostics.iterations.extend(result.iterations for result in complete)
    return terminal_results


def invert_routed_american_implied_volatilities(
    inputs: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    dtype: np.dtype | str = np.float64,
    diagnostics: Optional[IVInversionDiagnostics] = None,
    solver_registry: Optional[Any] = None,
) -> tuple[IVInversionResult, ...]:
    """Invert each American target with the expiration's configured method."""
    from algotrading.refdata.options.Pricing import configured_methods, select_method
    from algotrading.refdata.options.solvers.Registry import SolverRegistry
    from algotrading.refdata.options.solvers.Types import (
        PricingRequest,
        SolverBatchJob,
        SolverOperation,
    )

    diagnostics = diagnostics or IVInversionDiagnostics()
    requested_dtype = np.dtype(dtype).name
    routing = configured_methods(context)
    registry = solver_registry or SolverRegistry(context)
    jobs: list[SolverBatchJob] = []
    for index, item in enumerate(inputs):
        tenor_seconds = float(item["tenor_years"]) * 365.25 * 86400.0
        selection = select_method(
            "A",
            tenor_seconds,
            routing=routing,
        )
        request = PricingRequest(
            option_symbol=str(item.get("option_symbol", index)),
            snapshot_id=str(item.get("snapshot_id", "surface")),
            put_call="CALL" if bool(item["is_call"]) else "PUT",
            exercise_type="A",
            spot=float(item["spot"]),
            strike=float(item["strike"]),
            tenor_seconds=tenor_seconds,
            rate=float(item["rate"]),
            dividend_yield=float(item["dividend_yield"]),
            source_iv=_finite(item.get("source_iv")),
            source_collected_at=item.get("source_collected_at"),
            expiration_at=item.get("expiration_at"),
            requested_dtype=requested_dtype,
        )
        jobs.append(
            SolverBatchJob(
                job_id=f"surface-iv-{index}",
                owner_run_id=str(item.get("surface_run_id", "surface")),
                owner_snapshot_id=str(item.get("snapshot_id", "surface")),
                output_index=index,
                operation=SolverOperation.SOLVE_IV_ONLY,
                request=request,
                method=selection.method,
                method_selection_reason=selection.reason,
                target_price=float(item["price"]),
                requested_dtype=requested_dtype,
            )
        )
    diagnostics.contract_price_evaluations += len(jobs)
    if _as_bool(_surface_setting(context, "iv_batch.enabled", True)) and hasattr(
        registry, "solve_iv_only_many"
    ):
        solved_results = registry.solve_iv_only_many(jobs)
    else:
        solved_results = [
            registry.solve_iv_only(
                job.request,
                method=job.method,
                target_price=float(job.target_price),
                iv_hints=job.iv_hints,
            )
            for job in jobs
        ]
    results: list[IVInversionResult] = []
    for solved in solved_results:
        if solved.status == "complete":
            diagnostics.complete_result_count += 1
            diagnostics.iterations.append(solved.iterations)
        else:
            diagnostics.rejected_result_count += 1
            diagnostics.rejected_by_reason[
                solved.reason or "iv_inversion_failed"
            ] += 1
        results.append(
            IVInversionResult(
                solved.iv,
                "complete" if solved.status == "complete" else "rejected",
                solved.reason,
                solved.model_price,
                solved.price_residual,
                solved.iterations,
                requested_dtype=requested_dtype,
                actual_dtype=getattr(solved, "actual_dtype", requested_dtype),
                precision_escalated=bool(
                    getattr(solved, "precision_escalated", False)
                ),
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class SurfacePoint:
    option_symbol: str
    strike: float
    log_forward_moneyness: float
    iv: float
    total_variance: float
    weight: float
    delta: Optional[float]
    put_call: str
    price: Optional[float]
    price_residual: Optional[float]
    iv_origin: str = "none"
    used_source_fallback: bool = False
    precision_escalated: bool = False
    precision_escalation_reason: str = ""


@dataclass(frozen=True)
class FitPointSelectionPolicy:
    enabled: bool
    require_full_set_acceptance: bool
    initial_max_points_per_wing: int
    expansion_points_per_wing: int
    maximum_expansion_rounds: int
    minimum_points_per_wing: int
    minimum_total_points: int
    maximum_weighted_rmse_increase: float
    maximum_grid_iv_difference: float
    maximum_node_iv_difference: float
    maximum_node_strike_fraction_difference: float
    maximum_atm_iv_difference: float
    maximum_skew_difference: float
    maximum_curvature_difference: float
    shadow_enabled: bool
    shadow_sample_fraction: float
    shadow_seed: int

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "FitPointSelectionPolicy":
        values = cls(
            enabled=_as_bool(_surface_setting(context, "fit_points.enabled", False)),
            require_full_set_acceptance=_as_bool(
                _surface_setting(
                    context, "fit_points.require_full_set_acceptance", True
                )
            ),
            initial_max_points_per_wing=int(_surface_setting(context, "fit_points.initial_max_points_per_wing", 8)),
            expansion_points_per_wing=int(_surface_setting(context, "fit_points.expansion_points_per_wing", 4)),
            maximum_expansion_rounds=int(_surface_setting(context, "fit_points.maximum_expansion_rounds", 3)),
            minimum_points_per_wing=int(_surface_setting(context, "fit_points.minimum_points_per_wing", 3)),
            minimum_total_points=int(_surface_setting(context, "fit_points.minimum_total_points", 7)),
            maximum_weighted_rmse_increase=float(_surface_setting(context, "fit_points.maximum_weighted_rmse_increase", 0.0005)),
            maximum_grid_iv_difference=float(_surface_setting(context, "fit_points.maximum_grid_iv_difference", 0.0010)),
            maximum_node_iv_difference=float(_surface_setting(context, "fit_points.maximum_node_iv_difference", 0.0010)),
            maximum_node_strike_fraction_difference=float(_surface_setting(context, "fit_points.maximum_node_strike_fraction_difference", 0.0010)),
            maximum_atm_iv_difference=float(_surface_setting(context, "fit_points.maximum_atm_iv_difference", 0.0010)),
            maximum_skew_difference=float(_surface_setting(context, "fit_points.maximum_skew_difference", 0.0010)),
            maximum_curvature_difference=float(_surface_setting(context, "fit_points.maximum_curvature_difference", 0.0010)),
            shadow_enabled=_as_bool(_surface_setting(context, "fit_points.full_set_shadow.enabled", False)),
            shadow_sample_fraction=float(_surface_setting(context, "fit_points.full_set_shadow.sample_fraction", 0.05)),
            shadow_seed=int(_surface_setting(context, "fit_points.full_set_shadow.seed", 0)),
        )
        numeric = (
            values.maximum_weighted_rmse_increase,
            values.maximum_grid_iv_difference,
            values.maximum_node_iv_difference,
            values.maximum_node_strike_fraction_difference,
            values.maximum_atm_iv_difference,
            values.maximum_skew_difference,
            values.maximum_curvature_difference,
            values.shadow_sample_fraction,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("surface fit-point tolerances must be finite and nonnegative")
        if not 0 <= values.shadow_sample_fraction <= 1:
            raise ValueError("surface full-set shadow sample fraction must be in [0, 1]")
        if (
            values.initial_max_points_per_wing <= 0
            or values.expansion_points_per_wing <= 0
            or values.maximum_expansion_rounds <= 0
            or values.minimum_points_per_wing <= 0
            or values.minimum_total_points <= 0
            or values.minimum_points_per_wing > values.initial_max_points_per_wing
            or values.minimum_total_points > 2 * values.initial_max_points_per_wing
        ):
            raise ValueError("invalid surface fit-point count configuration")
        return values


def _point_origin_rank(point: SurfacePoint) -> int:
    return {
        "recalculated_current": 0,
        "calculated": 0,
        "calculated_seeded": 0,
        "recalculated_reused": 1,
        "reused": 1,
        "source": 2,
    }.get(point.iv_origin, 3)


def select_surface_coreset(
    points: Sequence[SurfacePoint], *, maximum_points_per_wing: int
) -> tuple[SurfacePoint, ...]:
    """Deterministically preserve ATM, outer, delta, quality, and spacing coverage."""
    selected: list[SurfacePoint] = []
    for wing in ("PUT", "CALL"):
        candidates = sorted(
            (point for point in points if point.put_call == wing),
            key=lambda point: (point.log_forward_moneyness, point.option_symbol),
        )
        if len(candidates) <= maximum_points_per_wing:
            selected.extend(candidates)
            continue
        chosen: list[SurfacePoint] = []

        def add(point: SurfacePoint) -> None:
            if point not in chosen and len(chosen) < maximum_points_per_wing:
                chosen.append(point)

        add(min(candidates, key=lambda point: (abs(point.log_forward_moneyness), point.option_symbol)))
        add(max(candidates, key=lambda point: (abs(point.log_forward_moneyness), point.option_symbol)))
        for target in (0.40, 0.25, 0.10):
            with_delta = [point for point in candidates if point.delta is not None and math.isfinite(float(point.delta))]
            if with_delta:
                add(min(with_delta, key=lambda point: (abs(abs(float(point.delta)) - target), point.option_symbol)))
        while len(chosen) < maximum_points_per_wing:
            remaining = [point for point in candidates if point not in chosen]
            if not remaining:
                break
            add(max(
                remaining,
                key=lambda point: (
                    min(abs(point.log_forward_moneyness - item.log_forward_moneyness) for item in chosen),
                    -_point_origin_rank(point),
                    point.weight,
                    -abs(point.log_forward_moneyness),
                    tuple(-ord(char) for char in point.option_symbol),
                ),
            ))
        selected.extend(chosen)
    return tuple(sorted(selected, key=lambda point: (point.log_forward_moneyness, point.put_call, point.option_symbol)))


@dataclass(frozen=True)
class PreparedSurfaceContract:
    row_index: int
    option_symbol: str
    put_call: str
    exercise_type: str
    strike: float
    log_forward_moneyness: float
    spot: float
    forward: float
    tenor_years: float
    rate: float
    dividend_yield: float
    bid: Optional[float]
    mid: Optional[float]
    ask: Optional[float]
    source_iv: Optional[float]
    preferred_delta: Optional[float]
    preferred_vega: Optional[float]
    spread: float
    source_row: Mapping[str, Any]
    preferred_iv: Optional[float] = None
    preferred_iv_origin: str = "none"
    recalc_price_residual: Optional[float] = None


@dataclass(frozen=True)
class PreparedExpirationInputs:
    contracts: tuple[PreparedSurfaceContract, ...]
    input_count: int
    common_rejected_reasons: Mapping[str, int]
    source_truncated: bool
    tenor_seconds: float
    spot: float
    forward: float
    rate: float
    dividend_yield: float
    exercise_type: str
    source_collected_at: Optional[object] = None
    expiration_at: Optional[object] = None
    snapshot_id: str = "surface"


@dataclass(frozen=True)
class SurfaceInputSelection:
    points: tuple[SurfacePoint, ...]
    input_count: int
    rejected_reasons: Mapping[str, int]
    source_fallback_count: int
    source_truncated: bool
    recalculated_current_iv_point_count: int = 0
    recalculated_reused_iv_point_count: int = 0
    source_iv_point_count: int = 0
    no_iv_excluded_count: int = 0
    source_iv_weight_fraction: float = 0.0
    iv_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    float64_retry_accepted_count: int = 0


@dataclass(frozen=True)
class AmericanIVJob:
    row_index: int
    price_basis: str
    inversion_input: Mapping[str, Any]


def american_iv_job_key(
    prepared: PreparedExpirationInputs, job: AmericanIVJob
) -> tuple[str, str, str, int]:
    """Stable owner/expiration/basis key for cross-run IV fan-out."""
    expiration = prepared.expiration_at
    if isinstance(expiration, datetime):
        expiration_text = expiration.isoformat()
    else:
        expiration_text = str(expiration)
    return (
        str(
            job.inversion_input.get(
                "source_calculation_run_id", prepared.snapshot_id
            )
        ),
        expiration_text,
        str(job.price_basis),
        int(job.row_index),
    )


def _surface_setting(context: Mapping[str, Any], name: str, default: Any) -> Any:
    return context.get(OPTIONS_CALCS_SURFACE_PREFIX + name, default)


def _iv_setting(context: Mapping[str, Any], name: str, default: Any) -> Any:
    return context.get(OPTIONS_CALCS_GREEKS_PREFIX + "iv." + name, default)


def _minimum_surface_tenor(context: Mapping[str, Any]) -> float:
    value = float(_surface_setting(context, "minimum_tenor_seconds", 60.0))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "surface.minimum_tenor_seconds must be finite and non-negative"
        )
    return value


def _finite(value: Any) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _sequence_setting(
    context: Mapping[str, Any], name: str, default: Sequence[float]
) -> tuple[float, ...]:
    configured = _surface_setting(context, name, default)
    if isinstance(configured, str):
        configured = [
            value.strip() for value in configured.split(",") if value.strip()
        ]
    return tuple(float(value) for value in configured)


def surface_contract_exclusion_reason(
    row: Mapping[str, Any], *, context: Mapping[str, Any]
) -> Optional[str]:
    """Check contract eligibility before selecting a surface's pricing style.

    Provider nonstandard flags are insufficient: a chain can also contain
    share-plus-cash or basket deliverables flagged as standard.  The vanilla
    surface engine cannot price those payoffs, regardless of exercise style.
    Empty deliverable metadata remains supported for ordinary/cash-settled
    contracts whose provider does not populate it.
    """
    if _as_bool(row.get("is_nonstandard")) and not _as_bool(
        _surface_setting(context, "include_nonstandard", False)
    ):
        return "nonstandard_contract"
    if _as_bool(row.get("is_mini")) and not _as_bool(
        _surface_setting(context, "include_minis", False)
    ):
        return "mini_contract"
    multiplier = _finite(row.get("multiplier"))
    expected_multiplier = float(
        context.get(OPTIONS_CALCS_CONFIG_PREFIX + "expected_multiplier", 100.0)
    )
    if multiplier is None or not math.isclose(multiplier, expected_multiplier):
        return "unexpected_multiplier"
    try:
        deliverables = json.loads(row.get("option_deliverables_json") or "[]")
    except (TypeError, ValueError):
        return "invalid_deliverable_metadata"
    if not isinstance(deliverables, list) or any(
        not isinstance(item, dict) for item in deliverables
    ):
        return "invalid_deliverable_metadata"
    if len(deliverables) > 1:
        return "unsupported_deliverable"
    if deliverables and deliverables[0].get("assetType") == "STOCK":
        item = deliverables[0]
        units = _finite(item.get("deliverableUnits"))
        underlying = str(row.get("underlying_symbol") or "")
        if (
            units is None
            or not math.isclose(units, multiplier)
            or (underlying and item.get("symbol") != underlying)
        ):
            return "unsupported_deliverable"
    return None


def prepare_surface_expiration_inputs(
    rows: Sequence[Mapping[str, Any]], *, context: Mapping[str, Any]
) -> PreparedExpirationInputs:
    """Apply every basis-independent expiration filter exactly once."""
    if not rows:
        raise ValueError("Cannot prepare an empty expiration")
    rejected: Counter[str] = Counter()
    contracts: list[PreparedSurfaceContract] = []
    source_truncated = any(_as_bool(row.get("chain_is_truncated")) for row in rows)
    allow_truncated = _as_bool(
        _surface_setting(context, "allow_truncated_snapshots", False)
    )
    max_quote_age = float(
        _surface_setting(context, "maximum_quote_age_seconds", 60)
    )
    max_spread = float(
        _surface_setting(context, "maximum_spread_fraction_of_mid", 0.75)
    )
    max_abs_k = float(
        _surface_setting(context, "maximum_abs_log_forward_moneyness", 0.50)
    )
    atm_width = float(_surface_setting(context, "near_atm_log_moneyness", 0.025))

    for row_index, row in enumerate(rows):
        exclusion = surface_contract_exclusion_reason(row, context=context)
        if exclusion is not None:
            rejected[exclusion] += 1
            continue
        if str(row.get("calculation_status", "")).lower() != "complete":
            rejected["failed_contract_calculation"] += 1
            continue
        if int(row.get("contract_quality_flags") or 0) & 262144:
            rejected["inconsistent_call_put_map"] += 1
            continue
        if source_truncated and not allow_truncated:
            rejected["truncated_snapshot"] += 1
            continue
        quote_age = _finite(row.get("quote_age_seconds"))
        underlying_age = _finite(row.get("underlying_quote_age_seconds"))
        if (quote_age is not None and quote_age > max_quote_age) or (
            underlying_age is not None and underlying_age > max_quote_age
        ):
            rejected["stale_quote"] += 1
            continue
        bid, ask = _finite(row.get("bid")), _finite(row.get("ask"))
        if bid is not None and ask is not None and ask < bid:
            rejected["crossed_market"] += 1
            continue
        spread_fraction = _finite(row.get("spread_fraction_of_mid"))
        if spread_fraction is not None and spread_fraction > max_spread:
            rejected["excessive_spread"] += 1
            continue
        strike = _finite(row.get("strike_price"))
        forward = _finite(row.get("forward_price"))
        tenor_years = _finite(row.get("time_to_expiry_years"))
        spot = _finite(row.get("spot_price"))
        rate = _finite(row.get("risk_free_rate"))
        dividend = _finite(row.get("dividend_yield"))
        if (
            strike is None
            or forward is None
            or tenor_years is None
            or spot is None
            or rate is None
            or dividend is None
            or strike <= 0
            or forward <= 0
            or tenor_years <= 0
            or spot <= 0
        ):
            rejected["invalid_model_input"] += 1
            continue
        k = float(np.log(np.float64(strike) / np.float64(forward)))
        if not math.isfinite(k) or abs(k) > max_abs_k:
            rejected["outside_moneyness_filter"] += 1
            continue
        put_call = str(row.get("put_call", "")).upper()
        if put_call not in {"CALL", "PUT"}:
            rejected["invalid_put_call"] += 1
            continue
        exercise_type = str(row.get("exercise_type", "")).strip().upper()
        if exercise_type not in {"A", "AMERICAN", "E", "EUROPEAN"}:
            rejected["invalid_exercise_type"] += 1
            continue
        if abs(k) > atm_width and (
            (put_call == "CALL" and k < 0) or (put_call == "PUT" and k > 0)
        ):
            rejected["in_the_money_input"] += 1
            continue
        source_iv = _finite(row.get("source_iv"))
        preferred_iv = _finite(row.get("preferred_iv"))
        if preferred_iv is None:
            preferred_iv = source_iv
        preferred_iv_origin = str(row.get("preferred_iv_origin") or "").lower()
        if (
            str(row.get("recalc_iv_identifiability_status") or "").lower()
            == "source_anchor_flat"
            and source_iv is not None
        ):
            preferred_iv = source_iv
            preferred_iv_origin = "source"
        if preferred_iv_origin not in {
            "recalculated_current",
            "recalculated_reused",
            "source",
            "none",
        }:
            preferred_iv_origin = "none"
        if preferred_iv_origin == "none" and preferred_iv is not None:
            preferred_iv_origin = "source" if source_iv is not None else "none"
        spread = max(
            (ask - bid) if bid is not None and ask is not None else 1.0,
            1.0e-8,
        )
        contracts.append(
            PreparedSurfaceContract(
                row_index=row_index,
                option_symbol=str(row.get("option_symbol", "")),
                put_call=put_call,
                exercise_type=exercise_type,
                strike=strike,
                log_forward_moneyness=k,
                spot=spot,
                forward=forward,
                tenor_years=tenor_years,
                rate=rate,
                dividend_yield=dividend,
                bid=bid,
                mid=_finite(row.get("mid_price")),
                ask=ask,
                source_iv=source_iv,
                preferred_iv=preferred_iv,
                preferred_iv_origin=preferred_iv_origin,
                recalc_price_residual=_finite(row.get("recalc_price_residual")),
                preferred_delta=_finite(row.get("preferred_delta")),
                preferred_vega=_finite(row.get("preferred_vega")),
                spread=spread,
                source_row=row,
            )
        )
    first = next(
        (
            row for row in rows
            if surface_contract_exclusion_reason(row, context=context) is None
        ),
        rows[0],
    )
    first_dividend_yield = _finite(first.get("dividend_yield"))
    return PreparedExpirationInputs(
        contracts=tuple(contracts),
        input_count=len(rows),
        common_rejected_reasons=MappingProxyType(dict(sorted(rejected.items()))),
        source_truncated=source_truncated,
        tenor_seconds=float(_finite(first.get("time_to_expiry_seconds")) or 0.0),
        spot=float(_finite(first.get("spot_price")) or math.nan),
        forward=float(_finite(first.get("forward_price")) or math.nan),
        rate=float(_finite(first.get("risk_free_rate")) or math.nan),
        # A zero dividend yield is a valid model input.  Boolean fallback used
        # to turn exactly 0.0 into NaN after contract selection had succeeded.
        dividend_yield=(
            float(first_dividend_yield)
            if first_dividend_yield is not None
            else math.nan
        ),
        exercise_type=str(first.get("exercise_type", "")),
        source_collected_at=first.get("source_collected_at"),
        expiration_at=first.get("expiration_at"),
        snapshot_id=str(first.get("snapshot_id", "surface")),
    )


def _selected_basis_price(
    contract: PreparedSurfaceContract, price_basis: str
) -> Optional[float]:
    if price_basis == "mid":
        return contract.mid
    if price_basis == "bid":
        return contract.bid if contract.bid is not None and contract.bid > 0 else None
    if price_basis == "ask":
        return contract.ask
    if price_basis == "source":
        return None
    raise ValueError(f"Unsupported price basis: {price_basis}")


def american_iv_jobs_for_basis(
    prepared: PreparedExpirationInputs,
    *,
    price_basis: str,
    context: Mapping[str, Any],
) -> tuple[AmericanIVJob, ...]:
    if price_basis in {"source", "mid"}:
        return ()
    minimum_price = float(_surface_setting(context, "minimum_option_price", 0.01))
    jobs = []
    for contract in prepared.contracts:
        price = _selected_basis_price(contract, price_basis)
        if price is None or price < minimum_price:
            continue
        if contract.exercise_type.strip().upper() not in {"A", "AMERICAN"}:
            continue
        jobs.append(
            AmericanIVJob(
                contract.row_index,
                price_basis,
                {
                    "price": price,
                    "spot": contract.spot,
                    "strike": contract.strike,
                    "tenor_years": contract.tenor_years,
                    "rate": contract.rate,
                    "dividend_yield": contract.dividend_yield,
                    "is_call": contract.put_call == "CALL",
                    "source_iv": contract.source_iv,
                    "option_symbol": contract.option_symbol,
                    "snapshot_id": contract.source_row.get("snapshot_id", "surface"),
                    "source_calculation_run_id": contract.source_row.get(
                        "_source_calculation_run_id",
                        contract.source_row.get("snapshot_id", "surface"),
                    ),
                    "source_collected_at": contract.source_row.get("source_collected_at"),
                    "expiration_at": contract.source_row.get("expiration_at"),
                },
            )
        )
    return tuple(jobs)


def select_surface_inputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    price_basis: str,
    context: Mapping[str, Any],
    dtype: np.dtype,
    prepared: Optional[PreparedExpirationInputs] = None,
    american_inversions: Optional[Mapping[int, IVInversionResult]] = None,
    runtime: Optional[SurfaceNumericalRuntime] = None,
    diagnostics: Optional[IVInversionDiagnostics] = None,
    solver_registry: Optional[Any] = None,
) -> SurfaceInputSelection:
    prepared = prepared or prepare_surface_expiration_inputs(rows, context=context)
    rejected: Counter[str] = Counter(prepared.common_rejected_reasons)
    candidates: list[SurfacePoint] = []
    source_fallback_count = 0
    minimum_price = float(_surface_setting(context, "minimum_option_price", 0.01))
    minimum_iv = float(_iv_setting(context, "minimum", 1.0e-4))
    maximum_iv = float(_iv_setting(context, "maximum", 10.0))
    weighting_method = str(
        _surface_setting(context, "weighting_method", "inverse_spread_vega")
    ).lower()
    diagnostics = diagnostics or IVInversionDiagnostics()
    pending_american: list[tuple[PreparedSurfaceContract, float]] = []

    def accept_candidate(
        *,
        contract: PreparedSurfaceContract,
        selected_price: Optional[float],
        inversion: Optional[IVInversionResult],
        iv: Optional[float],
        used_fallback: bool,
        iv_origin: str,
        direct_residual: Optional[float] = None,
    ) -> None:
        nonlocal source_fallback_count
        if iv is None or not minimum_iv <= iv <= maximum_iv or not math.isfinite(iv):
            rejected["invalid_iv"] += 1
            return
        if used_fallback:
            source_fallback_count += 1
        total_variance = float(np.float64(iv) * np.float64(iv) * np.float64(contract.tenor_years))
        if not math.isfinite(total_variance) or total_variance < 0:
            raise NumericalInstability("negative_or_nonfinite_total_variance")
        vega = abs(contract.preferred_vega or 1.0)
        if weighting_method == "uniform":
            weight = 1.0
        elif weighting_method == "inverse_spread":
            weight = 1.0 / contract.spread
        elif weighting_method == "inverse_spread_vega":
            weight = max(vega, 1.0e-6) / contract.spread
        else:
            raise ValueError(
                f"Unsupported surface weighting_method: {weighting_method}"
            )
        weight /= 1.0 + abs(contract.log_forward_moneyness)
        candidates.append(
            SurfacePoint(
                option_symbol=contract.option_symbol,
                strike=contract.strike,
                log_forward_moneyness=contract.log_forward_moneyness,
                iv=float(iv),
                total_variance=total_variance,
                weight=float(weight),
                delta=contract.preferred_delta,
                put_call=contract.put_call,
                price=selected_price,
                price_residual=(
                    inversion.price_residual if inversion else direct_residual
                ),
                iv_origin=iv_origin,
                used_source_fallback=used_fallback,
                precision_escalated=bool(
                    inversion is not None
                    and inversion.iv is not None
                    and inversion.precision_escalated
                    and not used_fallback
                ),
                precision_escalation_reason=(
                    inversion.precision_escalation_reason
                    if inversion is not None
                    and inversion.iv is not None
                    and inversion.precision_escalated
                    and not used_fallback
                    else ""
                ),
            )
        )

    for contract in prepared.contracts:
        source_iv = contract.source_iv
        used_fallback = False
        inversion: Optional[IVInversionResult] = None
        selected_price: Optional[float] = None
        if contract.exercise_type.strip().upper() in {"E", "EUROPEAN"}:
            if source_iv is None or not minimum_iv <= source_iv <= maximum_iv:
                rejected["invalid_source_iv"] += 1
                continue
            accept_candidate(
                contract=contract,
                selected_price=None,
                inversion=None,
                iv=source_iv,
                used_fallback=False,
                iv_origin="source",
            )
            continue
        if price_basis == "source":
            rejected["american_requires_market_price"] += 1
            continue
        elif price_basis == "mid":
            selected_price = contract.mid
            if selected_price is None or selected_price < minimum_price:
                rejected["missing_or_small_price"] += 1
                continue
            preferred_iv = contract.preferred_iv
            if (
                preferred_iv is None
                or not minimum_iv <= preferred_iv <= maximum_iv
                or contract.preferred_iv_origin == "none"
            ):
                rejected["no_valid_preferred_iv"] += 1
                continue
            origin = contract.preferred_iv_origin
            accept_candidate(
                contract=contract,
                selected_price=selected_price,
                inversion=None,
                iv=preferred_iv,
                used_fallback=origin == "source",
                iv_origin=origin,
                direct_residual=(
                    contract.recalc_price_residual
                    if origin == "recalculated_current"
                    else None
                ),
            )
            continue
        else:
            selected_price = _selected_basis_price(contract, price_basis)
            if price_basis == "bid" and selected_price is None:
                rejected["zero_or_missing_bid"] += 1
                continue
            if selected_price is None or selected_price < minimum_price:
                rejected["missing_or_small_price"] += 1
                continue
            else:
                if contract.exercise_type.strip().upper() in {
                    "A",
                    "AMERICAN",
                }:
                    if american_inversions is None:
                        pending_american.append((contract, selected_price))
                        continue
                    inversion = american_inversions.get(contract.row_index)
                    if inversion is None:
                        raise RuntimeError(
                            f"Missing combined American IV result for row {contract.row_index} {price_basis}"
                        )
                if inversion is None:
                    raise RuntimeError(
                        "American market IV must be produced by the routed solver"
                    )
                if inversion.iv is None:
                    rejected[inversion.reason or "iv_inversion_failed"] += 1
                    continue
                else:
                    iv = inversion.iv
        accept_candidate(
            contract=contract,
            selected_price=selected_price,
            inversion=inversion,
            iv=iv,
            used_fallback=used_fallback,
            iv_origin=f"market_{price_basis}_inverted",
        )

    if pending_american:
        inversions = invert_routed_american_implied_volatilities(
            [
                {
                    "price": price,
                    "spot": contract.spot,
                    "strike": contract.strike,
                    "tenor_years": contract.tenor_years,
                    "rate": contract.rate,
                    "dividend_yield": contract.dividend_yield,
                    "is_call": contract.put_call == "CALL",
                    "source_iv": contract.source_iv,
                    "option_symbol": contract.option_symbol,
                    "snapshot_id": contract.source_row.get(
                        "snapshot_id", "surface"
                    ),
                    "source_collected_at": contract.source_row.get(
                        "source_collected_at"
                    ),
                    "expiration_at": contract.source_row.get("expiration_at"),
                }
                for contract, price in pending_american
            ],
            context=context,
            dtype=dtype,
            diagnostics=diagnostics,
            solver_registry=solver_registry,
        )
        for (contract, selected_price), inversion in zip(pending_american, inversions):
            source_iv = contract.source_iv
            used_fallback = False
            iv = inversion.iv
            if iv is None:
                rejected[inversion.reason or "iv_inversion_failed"] += 1
                continue
            accept_candidate(
                contract=contract,
                selected_price=selected_price,
                inversion=inversion,
                iv=iv,
                used_fallback=used_fallback,
                iv_origin=f"market_{price_basis}_inverted",
            )

    # Prefer the tightest quote for duplicate option side/strike coordinates.
    deduplicated: dict[tuple[float, str], SurfacePoint] = {}
    for point in sorted(
        candidates,
        key=lambda p: (
            p.strike,
            p.put_call,
            -p.weight,
            p.option_symbol,
        ),
    ):
        key = (point.strike, point.put_call)
        if key not in deduplicated:
            deduplicated[key] = point
        else:
            rejected["duplicate_strike_side"] += 1
    points = tuple(
        sorted(
            deduplicated.values(),
            key=lambda p: (p.log_forward_moneyness, p.put_call, p.option_symbol),
        )
    )
    origin_counts = Counter(point.iv_origin for point in points)
    total_weight = sum(point.weight for point in points)
    source_weight = sum(
        point.weight for point in points if point.iv_origin == "source"
    )
    return SurfaceInputSelection(
        points=points,
        input_count=prepared.input_count,
        rejected_reasons=dict(sorted(rejected.items())),
        source_fallback_count=source_fallback_count,
        source_truncated=prepared.source_truncated,
        recalculated_current_iv_point_count=origin_counts["recalculated_current"],
        recalculated_reused_iv_point_count=origin_counts["recalculated_reused"],
        source_iv_point_count=origin_counts["source"],
        no_iv_excluded_count=int(rejected.get("no_valid_preferred_iv", 0)),
        source_iv_weight_fraction=(source_weight / total_weight if total_weight else 0.0),
        iv_diagnostics=diagnostics.summary(),
        float64_retry_accepted_count=sum(
            point.precision_escalated for point in points
        ),
    )


def svi_total_variance(
    k: np.ndarray | float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    k_values = np.asarray(k, dtype=np.float64)
    shifted = k_values - np.float64(m)
    return np.float64(a) + np.float64(b) * (
        np.float64(rho) * shifted
        + np.sqrt(np.square(shifted) + np.float64(sigma) ** 2)
    )


def _softplus(value: float) -> float:
    return float(np.logaddexp(0.0, np.float64(value)))


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1.0e-12)
    return float(np.log(np.expm1(value))) if value < 50.0 else value


def _transformed_svi_parameters(
    values: np.ndarray, *, m_center: float, m_half_width: float
) -> dict[str, float]:
    b = _softplus(float(values[1]))
    rho = math.tanh(float(values[2])) * (1.0 - 1.0e-8)
    sigma = _softplus(float(values[4])) + 1.0e-8
    minimum_variance = _softplus(float(values[0]))
    a = minimum_variance - b * sigma * math.sqrt(max(1.0 - rho * rho, 0.0))
    m = m_center + m_half_width * math.tanh(float(values[3]))
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma}


def svi_parameters_to_optimizer(
    parameters: Mapping[str, Any],
    *,
    m_center: float,
    m_half_width: float,
) -> np.ndarray:
    """Safely invert the constrained SVI parameter transform."""
    a, b, rho, m, sigma = (
        float(parameters[name]) for name in ("a", "b", "rho", "m", "sigma")
    )
    if (
        not all(math.isfinite(value) for value in (a, b, rho, m, sigma))
        or b <= 0
        or sigma <= 1.0e-8
        or m_half_width <= 0
    ):
        raise ValueError("invalid constrained SVI warm start")
    rho_scale = 1.0 - 1.0e-8
    clipped_rho = float(np.clip(rho / rho_scale, -1.0 + 1.0e-10, 1.0 - 1.0e-10))
    normalized_m = float(
        np.clip(
            (m - m_center) / m_half_width,
            -1.0 + 1.0e-10,
            1.0 - 1.0e-10,
        )
    )
    minimum_variance = a + b * sigma * math.sqrt(max(1.0 - rho * rho, 0.0))
    if minimum_variance < -1.0e-10:
        raise ValueError("SVI warm start has negative minimum variance")
    return np.asarray(
        [
            _inverse_softplus(max(minimum_variance, 1.0e-12)),
            _inverse_softplus(b),
            math.atanh(clipped_rho),
            math.atanh(normalized_m),
            _inverse_softplus(max(sigma - 1.0e-8, 1.0e-12)),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class SmileFit:
    price_basis: str
    method: str
    status: str
    error: str
    parameters: Mapping[str, Any]
    points: tuple[SurfacePoint, ...]
    input_count: int
    rejected_reasons: Mapping[str, int]
    source_fallback_count: int
    source_truncated: bool
    tenor_seconds: float
    tenor_years: float
    spot: float
    forward: float
    rate: float
    dividend_yield: float
    exercise_type: str
    weighted_rmse: Optional[float]
    maximum_absolute_error: Optional[float]
    butterfly_flags: int
    calendar_flags: int
    quality_flags: int
    failure_stage: str = "none"
    requested_dtype: str = "float64"
    actual_dtype: str = "float64"
    precision_escalated: bool = False
    precision_escalation_reason: str = ""
    source_collected_at: Optional[object] = None
    expiration_at: Optional[object] = None
    snapshot_id: str = "surface"
    fit_candidate_contract_count: int = 0
    fit_selected_contract_count: int = 0
    fit_point_selection_method: str = "full_set_feature_disabled"
    fit_point_selection_rounds: int = 0
    fit_full_set_fallback: bool = False
    fit_full_set_shadow_checked: bool = False
    fit_full_set_shadow_status: str = "not_checked"
    fit_full_set_shadow_max_grid_iv_difference: Optional[float] = None
    fit_full_set_shadow_max_node_iv_difference: Optional[float] = None
    fit_full_set_shadow_max_node_strike_fraction_difference: Optional[float] = None
    fit_full_set_shadow_weighted_rmse_difference: Optional[float] = None
    fit_full_set_acceptance_required: bool = False
    fit_full_set_acceptance_passed: bool = False
    fit_full_set_acceptance_reason: str = "not_required"
    fit_full_set_required_nodes_match: bool = False
    fit_full_set_strategy_diagnostics_match: Optional[bool] = None
    fit_full_set_atm_iv_difference: Optional[float] = None
    fit_full_set_skew_difference: Optional[float] = None
    fit_full_set_curvature_difference: Optional[float] = None
    fit_full_set_arbitrage_flags_changed: Optional[bool] = None
    candidate_recalculated_current_iv_point_count: int = 0
    candidate_recalculated_reused_iv_point_count: int = 0
    candidate_source_iv_point_count: int = 0
    candidate_no_iv_excluded_count: int = 0
    candidate_source_iv_weight_fraction: float = 0.0

    @property
    def coverage(self) -> tuple[Optional[float], Optional[float]]:
        if not self.points:
            return None, None
        values = [point.log_forward_moneyness for point in self.points]
        return min(values), max(values)

    @property
    def successful(self) -> bool:
        return self.status in {"complete", "complete_warning"}

    def evaluate_total_variance(self, k: float | np.ndarray) -> np.ndarray:
        k_values = np.asarray(k, dtype=np.float64)
        if self.method == "svi":
            return svi_total_variance(k_values, **{
                name: float(self.parameters[name])
                for name in ("a", "b", "rho", "m", "sigma")
            })
        if self.method == "pchip":
            x = np.asarray(self.parameters["k"], dtype=np.float64)
            y = np.asarray(self.parameters["total_variance"], dtype=np.float64)
            return np.asarray(PchipInterpolator(x, y, extrapolate=False)(k_values), dtype=np.float64)
        return np.full_like(k_values, np.nan, dtype=np.float64)

    def assert_finite(self) -> None:
        if not self.successful:
            return
        for value in (self.weighted_rmse, self.maximum_absolute_error):
            if value is None or not math.isfinite(float(value)):
                raise NumericalInstability("nonfinite_fit_diagnostic")
        for value in self.parameters.values():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise NumericalInstability("nonfinite_fit_parameter")


def _aggregate_k_points(points: Sequence[SurfacePoint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[float, list[SurfacePoint]] = {}
    for point in points:
        grouped.setdefault(float(point.log_forward_moneyness), []).append(point)
    k_values, variances, weights = [], [], []
    for k in sorted(grouped):
        members = grouped[k]
        member_weights = np.asarray([max(point.weight, 1.0e-12) for point in members], dtype=np.float64)
        member_variances = np.asarray([point.total_variance for point in members], dtype=np.float64)
        k_values.append(k)
        variances.append(float(np.average(member_variances, weights=member_weights)))
        weights.append(float(np.sum(member_weights)))
    return (
        np.asarray(k_values, dtype=np.float64),
        np.asarray(variances, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def fit_svi(
    points: Sequence[SurfacePoint],
    *,
    max_iterations: int = 2000,
    warm_starts: Sequence[tuple[str, Mapping[str, Any]]] = (),
    return_diagnostics: bool = False,
) -> Any:
    k, observed, weights = _aggregate_k_points(points)
    if len(k) < 5:
        raise RuntimeError("SVI requires at least five distinct strikes")
    if np.any(~np.isfinite(observed)) or np.any(observed < 0):
        raise RuntimeError("invalid_total_variance")
    weight_scale = weights / max(float(np.mean(weights)), 1.0e-12)
    sqrt_weights = np.sqrt(weight_scale)
    center = float(np.median(k))
    half_width = max(float(np.ptp(k)) * 1.5, 0.25)
    variance_floor = max(float(np.min(observed)), 1.0e-10)
    variance_range = max(float(np.ptp(observed)), variance_floor * 0.1, 1.0e-8)

    deterministic_guesses = (
        (
            "deterministic_primary",
            np.asarray(
            [
                _inverse_softplus(variance_floor),
                _inverse_softplus(variance_range / max(float(np.ptp(k)), 0.1)),
                0.0,
                0.0,
                _inverse_softplus(max(float(np.ptp(k)) / 4.0, 0.03)),
            ],
            dtype=np.float64,
            ),
        ),
        (
            "deterministic_secondary",
            np.asarray(
            [
                _inverse_softplus(variance_floor * 0.8),
                _inverse_softplus(variance_range / max(float(np.ptp(k)) * 2.0, 0.1)),
                -0.5,
                0.0,
                _inverse_softplus(max(float(np.ptp(k)) / 2.0, 0.05)),
            ],
            dtype=np.float64,
            ),
        ),
    )

    def residuals(unconstrained: np.ndarray) -> np.ndarray:
        parameters = _transformed_svi_parameters(
            unconstrained, m_center=center, m_half_width=half_width
        )
        predicted = svi_total_variance(k, **parameters)
        if np.any(~np.isfinite(predicted)):
            return np.full_like(observed, 1.0e12)
        return sqrt_weights * (predicted - observed)

    warm_guesses = []
    for source, parameters in warm_starts:
        try:
            warm_guesses.append(
                (
                    str(source),
                    svi_parameters_to_optimizer(
                        parameters,
                        m_center=center,
                        m_half_width=half_width,
                    ),
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

    best = None
    total_nfev = 0
    warm_accepted = False
    for source, initial in warm_guesses:
        result = least_squares(
            residuals,
            initial,
            method="trf",
            max_nfev=int(max_iterations),
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
        )
        total_nfev += int(result.nfev)
        if not result.success or np.any(~np.isfinite(result.x)):
            continue
        objective = float(np.dot(result.fun, result.fun))
        if best is None or objective < best[0]:
            best = (objective, result, source)
        # A converged warm solution with a very small variance-space objective
        # is advisory but sufficient; deterministic guesses remain the fallback.
        if objective <= 1.0e-6:
            warm_accepted = True
            break
    if not warm_accepted:
        for source, initial in deterministic_guesses:
            result = least_squares(
                residuals,
                initial,
                method="trf",
                max_nfev=int(max_iterations),
                ftol=1.0e-12,
                xtol=1.0e-12,
                gtol=1.0e-12,
            )
            total_nfev += int(result.nfev)
            if not result.success or np.any(~np.isfinite(result.x)):
                continue
            objective = float(np.dot(result.fun, result.fun))
            if best is None or objective < best[0]:
                best = (objective, result, source)
    if best is None:
        raise RuntimeError("SVI optimizer failed")
    objective, result, initialization_source = best
    parameters = _transformed_svi_parameters(
        result.x, m_center=center, m_half_width=half_width
    )
    predicted = svi_total_variance(k, **parameters)
    if np.any(~np.isfinite(predicted)) or np.any(predicted < -1.0e-12):
        raise RuntimeError("SVI produced invalid total variance")
    errors = predicted - observed
    weighted_rmse = math.sqrt(float(np.average(np.square(errors), weights=weights)))
    maximum_error = float(np.max(np.abs(errors)))
    diagnostics = SVIFitDiagnostics(
        initialization_source=initialization_source,
        nfev=total_nfev,
        objective=objective,
        optimizer_success=True,
        fallback_path=(
            "" if warm_accepted else ("deterministic_after_warm" if warm_guesses else "cold_start")
        ),
    )
    output = (parameters, weighted_rmse, maximum_error)
    return (*output, diagnostics) if return_diagnostics else output


def butterfly_arbitrage_diagnostics(
    evaluator: Callable[[np.ndarray], np.ndarray],
    minimum_k: float,
    maximum_k: float,
    *,
    grid_size: int = 401,
    tolerance: float = 1.0e-10,
) -> tuple[int, dict[str, float]]:
    grid = np.linspace(float(minimum_k), float(maximum_k), max(51, int(grid_size)), dtype=np.float64)
    variance = np.asarray(evaluator(grid), dtype=np.float64)
    if np.any(~np.isfinite(variance)):
        return BUTTERFLY_INDETERMINATE, {"minimum_density": math.nan}
    flags = 0
    if float(np.min(variance)) < -abs(float(tolerance)):
        flags |= BUTTERFLY_NEGATIVE_VARIANCE
    safe_variance = np.maximum(variance, max(abs(float(tolerance)), 1.0e-14))
    edge_order = 2 if len(grid) >= 5 else 1
    first = np.gradient(safe_variance, grid, edge_order=edge_order)
    second = np.gradient(first, grid, edge_order=edge_order)
    density = (
        np.square(1.0 - grid * first / (2.0 * safe_variance))
        - (np.square(first) / 4.0) * (1.0 / safe_variance + 0.25)
        + second / 2.0
    )
    if np.any(~np.isfinite(density)):
        flags |= BUTTERFLY_INDETERMINATE
    minimum_density = float(np.nanmin(density))
    if minimum_density < -abs(float(tolerance)):
        flags |= BUTTERFLY_NEGATIVE_DENSITY
    return flags, {
        "minimum_total_variance": float(np.min(variance)),
        "minimum_density": minimum_density,
    }


def fit_smile(
    selection: SurfaceInputSelection,
    *,
    price_basis: str,
    tenor_seconds: float,
    spot: float,
    forward: float,
    rate: float,
    dividend_yield: float,
    exercise_type: str,
    context: Mapping[str, Any],
    source_collected_at: Optional[object] = None,
    expiration_at: Optional[object] = None,
    snapshot_id: str = "surface",
    dtype: np.dtype = np.dtype("float64"),
    warm_starts: Sequence[tuple[str, Mapping[str, Any]]] = (),
) -> SmileFit:
    dtype = np.dtype(dtype)
    points = selection.points
    minimum_points = int(_surface_setting(context, "minimum_input_points", 7))
    minimum_puts = int(_surface_setting(context, "minimum_otm_put_points", 2))
    minimum_calls = int(_surface_setting(context, "minimum_otm_call_points", 2))
    put_count = sum(
        point.put_call == "PUT" and point.log_forward_moneyness < 0 for point in points
    )
    call_count = sum(
        point.put_call == "CALL" and point.log_forward_moneyness > 0 for point in points
    )
    quality = QUALITY_SOURCE_TRUNCATED if selection.source_truncated else 0
    if selection.source_fallback_count:
        quality |= QUALITY_SOURCE_IV_FALLBACK
    common = dict(
        price_basis=price_basis,
        points=points,
        input_count=selection.input_count,
        rejected_reasons=selection.rejected_reasons,
        source_fallback_count=selection.source_fallback_count,
        source_truncated=selection.source_truncated,
        tenor_seconds=float(tenor_seconds),
        tenor_years=float(tenor_seconds) / SECONDS_PER_YEAR,
        spot=float(spot),
        forward=float(forward),
        rate=float(rate),
        dividend_yield=float(dividend_yield),
        exercise_type=str(exercise_type),
        source_collected_at=source_collected_at,
        expiration_at=expiration_at,
        snapshot_id=str(snapshot_id),
        calendar_flags=0,
    )
    if len(points) < minimum_points:
        inversion_failures = sum(
            count
            for reason, count in selection.rejected_reasons.items()
            if reason in IV_FAILURE_REASONS
            or reason.startswith("iv_solver_failure:")
            or reason in {
                "price_below_lower_bound",
                "price_above_upper_bound",
            }
        )
        return SmileFit(
            method="none",
            status=("failed_iv_inversion" if not points and inversion_failures else "skipped_insufficient_points"),
            error="insufficient accepted inputs",
            failure_stage=(
                "iv_inversion"
                if not points and inversion_failures
                else "input_selection"
            ),
            parameters={"rejected_reasons": dict(selection.rejected_reasons)},
            weighted_rmse=None,
            maximum_absolute_error=None,
            butterfly_flags=0,
            quality_flags=quality,
            **common,
        )
    if put_count < minimum_puts or call_count < minimum_calls:
        return SmileFit(
            method="none",
            status="skipped_insufficient_coverage",
            error=f"wing coverage put={put_count}, call={call_count}",
            failure_stage="input_selection",
            parameters={"rejected_reasons": dict(selection.rejected_reasons)},
            weighted_rmse=None,
            maximum_absolute_error=None,
            butterfly_flags=0,
            quality_flags=quality | QUALITY_INSUFFICIENT_WING,
            **common,
        )

    maximum_iterations = int(_surface_setting(context, "svi_max_iterations", 2000))
    fit_tolerance = float(_surface_setting(context, "maximum_weighted_rmse", 0.02))
    fallback_allowed = _as_bool(_surface_setting(context, "allow_pchip_fallback", True))
    method = "svi"
    status = "complete"
    fit_error = ""
    optimizer_diagnostics: Mapping[str, Any] = {}
    try:
        parameters, weighted_rmse, maximum_error, svi_diagnostics = fit_svi(
            points,
            max_iterations=maximum_iterations,
            warm_starts=warm_starts,
            return_diagnostics=True,
        )
        optimizer_diagnostics = {
            "initialization_source": svi_diagnostics.initialization_source,
            "nfev": svi_diagnostics.nfev,
            "objective": svi_diagnostics.objective,
            "optimizer_success": svi_diagnostics.optimizer_success,
            "fallback_path": svi_diagnostics.fallback_path,
        }
        if weighted_rmse > fit_tolerance:
            raise RuntimeError(
                f"SVI weighted RMSE {weighted_rmse:.6g} exceeds {fit_tolerance:.6g}"
            )
    except Exception as exc:
        # PCHIP is a data/model fallback, not a substitute for the required
        # precision retry.  An unstable optimizer or invalid fitted variance
        # first gets one localized float64 attempt.
        precision_failure = any(
            marker in str(exc).lower()
            for marker in (
                "optimizer failed",
                "invalid total variance",
                "produced invalid total variance",
                "nonfinite",
                "stagnation",
                "machine precision",
            )
        )
        if dtype == np.dtype("float32") and precision_failure:
            raise NumericalInstability(
                "svi_" + str(exc).lower().replace(" ", "_")[:96]
            ) from exc
        if not fallback_allowed:
            return SmileFit(
                method="none",
                status="failed_calibration",
                error=str(exc),
                failure_stage="smile_fit",
                parameters={"rejected_reasons": dict(selection.rejected_reasons)},
                weighted_rmse=None,
                maximum_absolute_error=None,
                butterfly_flags=0,
                quality_flags=quality,
                **common,
            )
        k, observed, weights = _aggregate_k_points(points)
        if len(k) < 3 or np.any(observed < 0):
            return SmileFit(
                method="none",
                status="failed_calibration",
                error=f"SVI failed ({exc}); PCHIP unavailable",
                failure_stage="smile_fit",
                parameters={"rejected_reasons": dict(selection.rejected_reasons)},
                weighted_rmse=None,
                maximum_absolute_error=None,
                butterfly_flags=0,
                quality_flags=quality,
                **common,
            )
        interpolator = PchipInterpolator(k, observed, extrapolate=False)
        predicted = np.asarray(interpolator(k), dtype=np.float64)
        errors = predicted - observed
        weighted_rmse = math.sqrt(float(np.average(np.square(errors), weights=weights)))
        maximum_error = float(np.max(np.abs(errors)))
        parameters = {
            "k": [float(value) for value in k],
            "total_variance": [float(value) for value in observed],
        }
        optimizer_diagnostics = {
            "initialization_source": "none",
            "nfev": 0,
            "objective": float(np.dot(errors, errors)),
            "optimizer_success": False,
            "fallback_path": "pchip_after_svi_failure",
        }
        method = "pchip"
        status = "complete_warning"
        fit_error = f"SVI fallback: {exc}"
        quality |= QUALITY_PCHIP_FALLBACK

    candidate = SmileFit(
        method=method,
        status=status,
        error=fit_error,
        parameters={
            **parameters,
            "rejected_reasons": dict(selection.rejected_reasons),
            "source_iv_fallback_count": selection.source_fallback_count,
            "iv_inversion_diagnostics": dict(selection.iv_diagnostics),
            "optimizer_diagnostics": dict(optimizer_diagnostics),
        },
        weighted_rmse=float(weighted_rmse),
        maximum_absolute_error=float(maximum_error),
        butterfly_flags=0,
        quality_flags=quality,
        **common,
    )
    minimum_k, maximum_k = candidate.coverage
    try:
        butterfly_flags, metrics = butterfly_arbitrage_diagnostics(
            candidate.evaluate_total_variance,
            minimum_k,
            maximum_k,
            grid_size=int(_surface_setting(context, "arbitrage_grid_size", 401)),
            tolerance=float(
                _surface_setting(context, "butterfly_tolerance", 1.0e-9)
            ),
        )
    except Exception as exc:
        return replace(
            candidate,
            status="failed_arbitrage",
            error=str(exc),
            failure_stage="butterfly_diagnostic",
        )
    parameters_with_diagnostics = dict(candidate.parameters)
    parameters_with_diagnostics["butterfly_metrics"] = metrics
    if butterfly_flags:
        candidate = replace(
            candidate,
            status="complete_warning",
            butterfly_flags=butterfly_flags,
            quality_flags=candidate.quality_flags | QUALITY_BUTTERFLY_WARNING,
            parameters=parameters_with_diagnostics,
        )
    else:
        candidate = replace(candidate, parameters=parameters_with_diagnostics)
    candidate.assert_finite()
    return candidate


@dataclass(frozen=True)
class SurfaceNode:
    node_type: str
    node_value: float
    put_call: str
    fitted_strike: Optional[float]
    fitted_delta: Optional[float]
    fitted_iv_bid: Optional[float]
    fitted_iv_mid: Optional[float]
    fitted_iv_ask: Optional[float]
    fit_confidence: Optional[float]
    status: str
    quality_flags: int = 0
    node_reason: str = "none"

    def assert_finite(self) -> None:
        if self.node_reason not in SURFACE_NODE_REASONS:
            raise RuntimeError(f"Invalid surface node reason: {self.node_reason}")
        for value in (
            self.fitted_strike,
            self.fitted_delta,
            self.fitted_iv_bid,
            self.fitted_iv_mid,
            self.fitted_iv_ask,
            self.fit_confidence,
        ):
            if value is not None and not math.isfinite(float(value)):
                raise NumericalInstability("nonfinite_node_output")


@dataclass(frozen=True)
class NodeGenerationResult:
    nodes: tuple[SurfaceNode, ...]
    status: str
    error: str = ""
    requested_dtype: str = "none"
    actual_dtype: str = "none"
    precision_escalated: bool = False
    precision_escalation_reason: str = ""
    required_status: str = "not_requested"
    optional_status: str = "not_requested"

    @classmethod
    def not_requested(cls) -> "NodeGenerationResult":
        return cls((), "not_requested")

    def assert_terminal(self) -> None:
        if self.status not in {
            "not_requested",
            "complete",
            "partial",
            "unavailable",
            "failed",
        }:
            raise RuntimeError(f"Invalid node-generation status: {self.status}")
        if self.requested_dtype not in {"none", *SUPPORTED_DTYPES}:
            raise RuntimeError(
                f"Invalid requested node dtype: {self.requested_dtype}"
            )
        if self.actual_dtype not in {"none", *SUPPORTED_DTYPES}:
            raise RuntimeError(f"Invalid actual node dtype: {self.actual_dtype}")
        terminal_statuses = {
            "not_requested",
            "complete",
            "partial",
            "unavailable",
            "failed",
        }
        if self.required_status not in terminal_statuses:
            raise RuntimeError(
                f"Invalid required node status: {self.required_status}"
            )
        if self.optional_status not in terminal_statuses:
            raise RuntimeError(
                f"Invalid optional node status: {self.optional_status}"
            )
        for node in self.nodes:
            node.assert_finite()


@dataclass(frozen=True)
class ExpirationCalibration:
    fit: SmileFit
    node_result: NodeGenerationResult = field(
        default_factory=NodeGenerationResult.not_requested
    )

    @property
    def nodes(self) -> tuple[SurfaceNode, ...]:
        """Compatibility view for callers that consume canonical nodes."""
        return self.node_result.nodes


@dataclass(frozen=True)
class AmericanDeltaGridDiagnostics:
    """Bounded summary of one American delta grid and its recovery probes."""

    grid_point_count: int
    valid_iv_point_count: int
    finite_direct_delta_count: int
    nonfinite_direct_delta_count: int
    finite_scalar_retry_count: int
    finite_price_fallback_count: int
    direct_scalar_mismatch_count: int
    price_kernel_nonfinite_count: int
    minimum_strike: Optional[float]
    maximum_strike: Optional[float]
    minimum_iv: Optional[float]
    maximum_iv: Optional[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def _fit_iv(fit: SmileFit, k: float) -> Optional[float]:
    minimum_k, maximum_k = fit.coverage
    if minimum_k is None or k < minimum_k or k > maximum_k:
        return None
    variance = float(np.asarray(fit.evaluate_total_variance(k)).reshape(()))
    if not math.isfinite(variance) or variance < 0 or fit.tenor_years <= 0:
        return None
    return math.sqrt(variance / fit.tenor_years)


def _american_deltas(
    fit: SmileFit,
    strikes: np.ndarray,
    ivs: np.ndarray,
    dtype: np.dtype,
    steps: int,
    *,
    is_call: bool,
    runtime: Optional[SurfaceNumericalRuntime] = None,
) -> np.ndarray:
    import torch

    torch_dtype = torch.float32 if dtype == np.dtype("float32") else torch.float64
    runtime = runtime or SurfaceNumericalRuntime(torch.device("cpu"), 50_000)
    module = _legacy_crr_module(
        steps=steps,
        chunk_size=runtime.chunk_size,
        device=str(runtime.device),
    )
    output = module.compute_price_deltas_crr_from_iv(
        S=np.full(len(strikes), fit.spot, dtype=np.float64),
        K=np.asarray(strikes, dtype=np.float64),
        T=np.full(len(strikes), fit.tenor_years, dtype=np.float64),
        r=np.full(len(strikes), fit.rate, dtype=np.float64),
        q=np.full(len(strikes), fit.dividend_yield, dtype=np.float64),
        is_call=np.full(len(strikes), is_call, dtype=bool),
        sigma=np.asarray(ivs, dtype=np.float64),
        steps=steps,
        dtype=torch_dtype,
        device=runtime.device,
        chunk_size=max(1, min(runtime.chunk_size, len(strikes))),
        to_numpy=True,
    )
    deltas = np.asarray(output["delta"], dtype=np.float64).reshape(-1)
    if len(deltas) != len(strikes):
        raise RuntimeError(
            "CRR delta kernel output shape mismatch: "
            f"expected {len(strikes)}, received {len(deltas)}"
        )
    return deltas


def _american_delta(
    fit: SmileFit,
    strike: float,
    iv: float,
    dtype: np.dtype,
    steps: int,
    *,
    is_call: bool,
    runtime: Optional[SurfaceNumericalRuntime] = None,
) -> float:
    return float(
        _american_deltas(
            fit,
            np.asarray([strike], dtype=np.float64),
            np.asarray([iv], dtype=np.float64),
            dtype,
            steps,
            is_call=is_call,
            runtime=runtime,
        )[0]
    )


@dataclass(frozen=True)
class AmericanDeltaRecoveryBatch:
    deltas: np.ndarray
    quality_flags: np.ndarray
    reasons: tuple[str, ...]
    actual_steps: np.ndarray
    diagnostics: Mapping[str, int]


@dataclass(frozen=True)
class SurfaceExactDeltaPoint:
    """One owner-local exact-delta request exposed to the wave coordinator."""

    cache_key: tuple[str, float]
    log_strike: float
    put_call: str
    strike: float
    volatility: float


@dataclass(frozen=True)
class DeltaSearchGrid:
    """One shared Tian proposal grid for all delta targets in an expiration."""

    log_moneyness: np.ndarray
    strikes: np.ndarray
    ivs: np.ndarray
    valid: np.ndarray
    call_deltas: np.ndarray
    put_deltas: np.ndarray
    call_quality_flags: np.ndarray
    put_quality_flags: np.ndarray
    call_reasons: tuple[str, ...]
    put_reasons: tuple[str, ...]
    quality_flags: int


@dataclass(frozen=True)
class SurfaceNodeSearchConfig:
    """Validated rollout and numerical budgets for delta-node generation."""

    hybrid_enabled: bool
    search_method: str
    validation_method: str
    tian_grid_size: int
    validation_max_evaluations: int
    validation_initial_candidates: int
    legacy_fallback_enabled: bool

    @classmethod
    def from_context(
        cls, context: Mapping[str, Any]
    ) -> "SurfaceNodeSearchConfig":
        hybrid_enabled = _as_bool(
            _surface_setting(context, "node_hybrid_enabled", True)
        )
        search_method = str(
            _surface_setting(context, "node_search_method", "tian")
        ).strip().lower()
        allowed_search = {"tian"} if hybrid_enabled else {
            "tian", "legacy", "observed_then_analytic"
        }
        if search_method not in allowed_search:
            raise ValueError(
                "options_calcs.surface.node_search_method must be "
                + ("tian" if hybrid_enabled else "tian or a legacy search method")
                + f", got {search_method!r}"
            )
        validation_method = str(
            _surface_setting(context, "node_validation_method", "configured")
        ).strip().lower()
        if validation_method != "configured":
            from algotrading.refdata.options.solvers.Types import RepricingMethod

            if RepricingMethod.parse(validation_method) is RepricingMethod.NONE:
                raise ValueError(
                    "options_calcs.surface.node_validation_method cannot be none"
                )

        def positive_integer(name: str, default: int) -> int:
            raw = _surface_setting(context, name, default)
            if isinstance(raw, bool) or (
                isinstance(raw, float) and not raw.is_integer()
            ):
                raise ValueError(
                    f"options_calcs.surface.{name} must be a positive integer"
                )
            try:
                value = int(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"options_calcs.surface.{name} must be a positive integer"
                ) from exc
            if value <= 0:
                raise ValueError(
                    f"options_calcs.surface.{name} must be a positive integer"
                )
            return value

        grid_size = positive_integer("node_tian_grid_size", 81)
        if grid_size < 9:
            raise ValueError(
                "options_calcs.surface.node_tian_grid_size must be at least 9"
            )
        budget = positive_integer("node_validation_max_evaluations", 4)
        initial = positive_integer("node_validation_initial_candidates", 1)
        if initial > budget:
            raise ValueError(
                "options_calcs.surface.node_validation_initial_candidates cannot "
                "exceed options_calcs.surface.node_validation_max_evaluations"
            )
        return cls(
            hybrid_enabled=hybrid_enabled,
            search_method=search_method,
            validation_method=validation_method,
            tian_grid_size=grid_size,
            validation_max_evaluations=budget,
            validation_initial_candidates=initial,
            legacy_fallback_enabled=_as_bool(
                _surface_setting(
                    context, "node_legacy_exact_fallback_enabled", True
                )
            ),
        )


class PendingSurfaceExactDeltaWave(RuntimeError):
    """Cooperative suspension at the next exact numerical node wave."""

    def __init__(
        self,
        fit: SmileFit,
        points: Sequence[SurfaceExactDeltaPoint],
        *,
        dense_fallback: bool,
        diagnostics: Optional[Mapping[str, int]] = None,
        timings: Optional[Mapping[str, float]] = None,
    ):
        self.fit = fit
        self.points = tuple(points)
        self.dense_fallback = bool(dense_fallback)
        self.diagnostics = MappingProxyType(dict(diagnostics or {}))
        self.timings = MappingProxyType(dict(timings or {}))
        super().__init__(f"surface exact-delta wave has {len(self.points)} points")


def _surface_validation_method(
    fit: SmileFit,
    context: Mapping[str, Any],
    *,
    method_override: Optional[Any] = None,
) -> Any:
    """Resolve the authoritative node method, or an explicit search method."""
    from algotrading.refdata.options.Pricing import configured_methods, select_method
    from algotrading.refdata.options.solvers.Types import RepricingMethod

    if method_override is not None:
        method = RepricingMethod.parse(method_override)
    else:
        configured = str(
            _surface_setting(context, "node_validation_method", "configured")
        ).strip().lower()
        if configured == "configured":
            method = select_method(
                "A", fit.tenor_seconds, routing=configured_methods(context)
            ).method
        else:
            method = RepricingMethod.parse(configured)
    if method is RepricingMethod.NONE:
        raise ValueError("American surface node validation method cannot be none")
    return method


def _routed_american_delta_waves(
    waves: Sequence[tuple[SmileFit, Sequence[SurfaceExactDeltaPoint]]],
    *,
    context: Mapping[str, Any],
    solver_registry: Any,
    method_override: Optional[Any] = None,
) -> list[AmericanDeltaRecoveryBatch]:
    """Execute several owners' ready exact-node points as explicit typed waves.

    Delta and eligible lower/upper price work are each submitted once across
    every active owner.  Registry grouping remains responsible for separating
    incompatible methods/devices/dtypes and for exact ordered fan-out.
    """
    from datetime import datetime, timedelta, timezone

    from algotrading.refdata.options.solvers.Types import (
        CircuitCallOutcome,
        PricingRequest,
    )

    output: list[dict[str, Any]] = []
    owned: list[tuple[int, int, PricingRequest, float, Any]] = []
    for wave_index, (fit, points) in enumerate(waves):
        method = _surface_validation_method(
            fit, context, method_override=method_override
        )
        _version, parameters = solver_registry.describe(method)
        source_at = fit.source_collected_at
        expiration_at = fit.expiration_at
        if source_at is None or expiration_at is None:
            source_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
            expiration_at = source_at + timedelta(seconds=fit.tenor_seconds)
        output.append(
            {
                "fit": fit,
                "points": tuple(points),
                "method": method,
                "configured_steps": int(
                    parameters.get("steps", parameters.get("time_grid", 0))
                ),
                "deltas": np.full(len(points), np.nan, dtype=np.float64),
                "quality": np.zeros(len(points), dtype=np.uint32),
                "reasons": np.full(len(points), "solver_delta_failed", dtype=object),
                "diagnostics": Counter(routed_delta_points=len(points)),
            }
        )
        for local_index, point in enumerate(points):
            if not (
                math.isfinite(point.strike)
                and point.strike > 0.0
                and math.isfinite(point.volatility)
                and point.volatility > 0.0
            ):
                output[wave_index]["quality"][local_index] |= (
                    QUALITY_INVALID_FITTED_IV_GRID
                )
                output[wave_index]["diagnostics"][
                    "routed_delta_failure_invalid_fitted_iv"
                ] += 1
                continue
            request = PricingRequest(
                option_symbol=(
                    f"surface-node-{fit.snapshot_id}-{point.put_call}-{local_index}"
                ),
                snapshot_id=fit.snapshot_id,
                put_call=point.put_call,
                exercise_type="A",
                spot=fit.spot,
                strike=point.strike,
                tenor_seconds=fit.tenor_seconds,
                rate=fit.rate,
                dividend_yield=fit.dividend_yield,
                source_collected_at=source_at,
                expiration_at=expiration_at,
                requested_dtype="float64",
            )
            owned.append(
                (wave_index, local_index, request, point.volatility, method)
            )

    if owned:
        direct_results = solver_registry.delta_many_results(
            [item[2] for item in owned],
            [item[3] for item in owned],
            [item[4] for item in owned],
            ["float64"] * len(owned),
        )
    else:
        direct_results = []

    fallbacks: list[tuple[int, int, PricingRequest, float, Any]] = []
    for item, result in zip(owned, direct_results):
        wave_index, local_index, request, volatility, method = item
        state = output[wave_index]
        value = result.value
        if value is not None and _delta_in_model_range(
            float(value),
            is_call=request.put_call == "CALL",
            dtype=np.dtype("float64"),
        ):
            state["deltas"][local_index] = float(value)
            state["reasons"][local_index] = "none"
            continue
        state["quality"][local_index] |= QUALITY_NONFINITE_CRR_DELTA
        state["diagnostics"]["node_direct_delta_failures"] += 1
        if result.circuit_outcome is CircuitCallOutcome.SUCCESS:
            fallbacks.append(item)
        elif result.circuit_outcome is CircuitCallOutcome.SUPPRESSED:
            state["reasons"][local_index] = "solver_circuit_open"
            state["diagnostics"]["surface_circuit_suppressed_delta_calls"] += 1
            state["diagnostics"]["surface_circuit_open_node_count"] += 1
            state["diagnostics"][
                f"routed_delta_failure_{method.value}:solver_circuit_open"
            ] += 1
        elif result.circuit_outcome is CircuitCallOutcome.INFRASTRUCTURE_FAILURE:
            state["reasons"][local_index] = "solver_infrastructure_failure"
            state["diagnostics"][
                "surface_circuit_actual_infrastructure_failures"
            ] += 1
            state["diagnostics"][
                f"routed_delta_failure_{method.value}:infrastructure"
            ] += 1
        else:
            state["reasons"][local_index] = "solver_unknown_outcome"
            state["diagnostics"][
                f"routed_delta_failure_{method.value}:unknown_outcome"
            ] += 1

    if fallbacks:
        lower_requests = []
        upper_requests = []
        for wave_index, _local_index, request, _volatility, _method in fallbacks:
            bump = max(abs(output[wave_index]["fit"].spot) * 1.0e-4, 1.0e-4)
            lower_requests.append(replace(request, spot=request.spot - bump))
            upper_requests.append(replace(request, spot=request.spot + bump))
        fallback_volatilities = [item[3] for item in fallbacks]
        fallback_methods = [item[4] for item in fallbacks]
        lower_results = solver_registry.price_many_results(
            lower_requests,
            fallback_volatilities,
            fallback_methods,
            ["float64"] * len(fallbacks),
        )
        upper_results = solver_registry.price_many_results(
            upper_requests,
            fallback_volatilities,
            fallback_methods,
            ["float64"] * len(fallbacks),
        )
        for item, lower_result, upper_result in zip(
            fallbacks, lower_results, upper_results
        ):
            wave_index, local_index, request, _volatility, method = item
            state = output[wave_index]
            failed = next(
                (
                    result
                    for result in (lower_result, upper_result)
                    if result.status != "complete"
                    or result.value is None
                    or not math.isfinite(float(result.value))
                ),
                None,
            )
            for result in (lower_result, upper_result):
                if result.status == "complete" and result.value is not None:
                    continue
                if result.circuit_outcome is CircuitCallOutcome.SUPPRESSED:
                    state["diagnostics"][
                        "surface_circuit_suppressed_price_calls"
                    ] += 1
                    state["diagnostics"]["surface_circuit_open_node_count"] += 1
                    state["diagnostics"][
                        f"routed_delta_failure_{method.value}:solver_circuit_open"
                    ] += 1
                elif result.circuit_outcome is CircuitCallOutcome.INFRASTRUCTURE_FAILURE:
                    state["diagnostics"][
                        "surface_circuit_actual_infrastructure_failures"
                    ] += 1
                    state["diagnostics"][
                        f"routed_delta_failure_{method.value}:infrastructure"
                    ] += 1
                elif result.circuit_outcome is CircuitCallOutcome.UNKNOWN_OUTCOME:
                    state["diagnostics"][
                        f"routed_delta_failure_{method.value}:unknown_outcome"
                    ] += 1
            if failed is not None:
                if failed.circuit_outcome is CircuitCallOutcome.SUPPRESSED:
                    state["reasons"][local_index] = "solver_circuit_open"
                elif failed.circuit_outcome is CircuitCallOutcome.INFRASTRUCTURE_FAILURE:
                    state["reasons"][local_index] = "solver_infrastructure_failure"
                elif failed.circuit_outcome is CircuitCallOutcome.UNKNOWN_OUTCOME:
                    state["reasons"][local_index] = "solver_unknown_outcome"
                else:
                    state["reasons"][local_index] = "crr_price_nonfinite"
                    state["diagnostics"][
                        f"routed_delta_failure_{method.value}:nonfinite_price"
                    ] += 1
                continue
            bump = max(abs(state["fit"].spot) * 1.0e-4, 1.0e-4)
            value = (float(upper_result.value) - float(lower_result.value)) / (
                2.0 * bump
            )
            if _delta_in_model_range(
                value,
                is_call=request.put_call == "CALL",
                dtype=np.dtype("float64"),
            ):
                state["deltas"][local_index] = value
                state["quality"][local_index] |= QUALITY_CRR_DELTA_PRICE_FALLBACK
                state["reasons"][local_index] = (
                    "solver_price_finite_difference_recovered"
                )
                state["diagnostics"]["node_price_delta_fallback_count"] += 1
            else:
                state["diagnostics"][
                    f"routed_delta_failure_{method.value}:nonfinite_delta:price_delta"
                ] += 1

    batches = []
    for state in output:
        failures = int(np.count_nonzero(~np.isfinite(state["deltas"])))
        state["diagnostics"]["routed_delta_failures"] = failures
        if state["method"].value == "crank_nicolson":
            state["diagnostics"]["node_crank_nicolson_fdm_delta_valuations"] = len(
                state["points"]
            )
            state["diagnostics"]["node_crank_nicolson_fdm_price_valuations"] = 2 * sum(
                1 for item in fallbacks if item[0] == len(batches)
            )
        batches.append(
            AmericanDeltaRecoveryBatch(
                deltas=state["deltas"],
                quality_flags=state["quality"],
                reasons=tuple(str(value) for value in state["reasons"]),
                actual_steps=np.full(
                    len(state["points"]), state["configured_steps"], dtype=np.int64
                ),
                diagnostics=MappingProxyType(dict(state["diagnostics"])),
            )
        )
    return batches


def _routed_american_delta_points(
    fit: SmileFit,
    strikes: np.ndarray,
    ivs: np.ndarray,
    *,
    is_call: bool,
    context: Mapping[str, Any],
    solver_registry: Optional[Any] = None,
    method_override: Optional[Any] = None,
) -> AmericanDeltaRecoveryBatch:
    """Calculate node deltas with the exact method routed for this tenor."""
    from datetime import datetime, timedelta, timezone

    from algotrading.refdata.options.solvers.Registry import SolverRegistry
    from algotrading.refdata.options.solvers.Types import (
        CircuitCallOutcome,
        PricingRequest,
        SolverCircuitOpenError,
        SolverInfrastructureError,
        SolverNumericalError,
    )

    strikes = np.asarray(strikes, dtype=np.float64).reshape(-1)
    ivs = np.asarray(ivs, dtype=np.float64).reshape(-1)
    if len(strikes) != len(ivs):
        raise ValueError("American delta strikes and IVs must be aligned")
    method = _surface_validation_method(
        fit, context, method_override=method_override
    )
    registry = solver_registry or SolverRegistry(context)
    _algorithm_version, solver_parameters = registry.describe(method)
    circuit_before = registry.circuit(method).snapshot()
    source_at = fit.source_collected_at
    expiration_at = fit.expiration_at
    if source_at is None or expiration_at is None:
        # Pure numerical callers need stable dates; production fits carry the
        # exact source and expiration timestamps from the calculation row.
        source_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
        expiration_at = source_at + timedelta(seconds=fit.tenor_seconds)
    deltas = np.full(len(strikes), np.nan, dtype=np.float64)
    quality = np.zeros(len(strikes), dtype=np.uint32)
    reasons = np.full(len(strikes), "solver_delta_failed", dtype=object)
    failures = Counter()
    direct_failures = 0
    price_fallbacks = 0
    fdm_delta_valuations = 0
    fdm_price_valuations = 0
    surface_infrastructure_failures = 0
    suppressed_delta_calls = 0
    suppressed_price_calls = 0
    circuit_open_nodes = 0
    if _as_bool(_surface_setting(context, "delta_batch.enabled", True)) and hasattr(
        registry, "delta_many"
    ) and hasattr(registry, "price_many"):
        valid_indices = [
            index
            for index, (strike, volatility) in enumerate(zip(strikes, ivs))
            if math.isfinite(float(strike))
            and float(strike) > 0.0
            and math.isfinite(float(volatility))
            and float(volatility) > 0.0
        ]
        invalid_indices = sorted(set(range(len(strikes))) - set(valid_indices))
        for index in invalid_indices:
            quality[index] |= QUALITY_INVALID_FITTED_IV_GRID
            failures["invalid_fitted_iv"] += 1
        requests = [
            PricingRequest(
                option_symbol=f"surface-node-{index}",
                snapshot_id=fit.snapshot_id,
                put_call="CALL" if is_call else "PUT",
                exercise_type="A",
                spot=fit.spot,
                strike=float(strikes[index]),
                tenor_seconds=fit.tenor_seconds,
                rate=fit.rate,
                dividend_yield=fit.dividend_yield,
                source_collected_at=source_at,
                expiration_at=expiration_at,
                requested_dtype="float64",
            )
            for index in valid_indices
        ]
        typed_valuation_api = hasattr(registry, "delta_many_results")
        if typed_valuation_api:
            direct_results = registry.delta_many_results(
                requests,
                [float(ivs[index]) for index in valid_indices],
                method,
                ["float64"] * len(valid_indices),
            )
        else:
            # Compatibility for narrowly scoped test doubles. Production
            # SolverRegistry always exposes the typed result API.
            direct_results = registry.delta_many(
                requests,
                [float(ivs[index]) for index in valid_indices],
                method,
                ["float64"] * len(valid_indices),
            )
        fallback_positions: list[int] = []
        for position, (index, result) in enumerate(zip(valid_indices, direct_results)):
            typed = hasattr(result, "circuit_outcome")
            value = result.value if typed else result
            if _delta_in_model_range(
                float(value), is_call=is_call, dtype=np.dtype("float64")
            ) if value is not None else False:
                deltas[index] = float(value)
                reasons[index] = "none"
            else:
                quality[index] |= QUALITY_NONFINITE_CRR_DELTA
                direct_failures += 1
                outcome = (
                    result.circuit_outcome
                    if typed
                    else CircuitCallOutcome.SUCCESS
                )
                if outcome is CircuitCallOutcome.SUCCESS:
                    fallback_positions.append(position)
                elif outcome is CircuitCallOutcome.SUPPRESSED:
                    reasons[index] = "solver_circuit_open"
                    failures[f"{method.value}:solver_circuit_open"] += 1
                    suppressed_delta_calls += 1
                    circuit_open_nodes += 1
                elif outcome is CircuitCallOutcome.INFRASTRUCTURE_FAILURE:
                    reasons[index] = "solver_infrastructure_failure"
                    failures[f"{method.value}:infrastructure"] += 1
                    surface_infrastructure_failures += 1
                else:
                    reasons[index] = "solver_unknown_outcome"
                    failures[f"{method.value}:unknown_outcome"] += 1
        if fallback_positions:
            bump = max(abs(fit.spot) * 1.0e-4, 1.0e-4)
            fallback_requests = [requests[position] for position in fallback_positions]
            fallback_volatilities = [
                float(ivs[valid_indices[position]]) for position in fallback_positions
            ]
            price_function = (
                registry.price_many_results
                if hasattr(registry, "price_many_results")
                else registry.price_many
            )
            lower_results = price_function(
                [replace(request, spot=fit.spot - bump) for request in fallback_requests],
                fallback_volatilities,
                method,
                ["float64"] * len(fallback_requests),
            )
            upper_results = price_function(
                [replace(request, spot=fit.spot + bump) for request in fallback_requests],
                fallback_volatilities,
                method,
                ["float64"] * len(fallback_requests),
            )
            for position, lower_result, upper_result in zip(
                fallback_positions, lower_results, upper_results
            ):
                index = valid_indices[position]
                lower_typed = hasattr(lower_result, "circuit_outcome")
                upper_typed = hasattr(upper_result, "circuit_outcome")
                lower_value = lower_result.value if lower_typed else lower_result
                upper_value = upper_result.value if upper_typed else upper_result
                failed_result = next(
                    (
                        result
                        for result, typed_result in (
                            (lower_result, lower_typed),
                            (upper_result, upper_typed),
                        )
                        if typed_result and result.status != "complete"
                    ),
                    None,
                )
                typed_failures = [
                    result
                    for result, typed_result in (
                        (lower_result, lower_typed),
                        (upper_result, upper_typed),
                    )
                    if typed_result and result.status != "complete"
                ]
                for typed_failure in typed_failures:
                    if typed_failure.circuit_outcome is CircuitCallOutcome.SUPPRESSED:
                        suppressed_price_calls += 1
                        circuit_open_nodes += 1
                        failures[f"{method.value}:solver_circuit_open"] += 1
                    elif typed_failure.circuit_outcome is CircuitCallOutcome.INFRASTRUCTURE_FAILURE:
                        surface_infrastructure_failures += 1
                        failures[f"{method.value}:infrastructure"] += 1
                    elif typed_failure.circuit_outcome is CircuitCallOutcome.UNKNOWN_OUTCOME:
                        failures[f"{method.value}:unknown_outcome"] += 1
                if failed_result is not None:
                    if failed_result.circuit_outcome is CircuitCallOutcome.SUPPRESSED:
                        reasons[index] = "solver_circuit_open"
                    elif failed_result.circuit_outcome is CircuitCallOutcome.INFRASTRUCTURE_FAILURE:
                        reasons[index] = "solver_infrastructure_failure"
                    elif failed_result.circuit_outcome is CircuitCallOutcome.UNKNOWN_OUTCOME:
                        reasons[index] = "solver_unknown_outcome"
                    else:
                        reasons[index] = "crr_price_nonfinite"
                        failures[f"{method.value}:nonfinite_price"] += 1
                    continue
                if lower_value is None or upper_value is None:
                    reasons[index] = "crr_price_nonfinite"
                    failures[f"{method.value}:nonfinite_price"] += 1
                    continue
                value = (float(upper_value) - float(lower_value)) / (2.0 * bump)
                if _delta_in_model_range(
                    value, is_call=is_call, dtype=np.dtype("float64")
                ):
                    deltas[index] = value
                    quality[index] |= QUALITY_CRR_DELTA_PRICE_FALLBACK
                    reasons[index] = "solver_price_finite_difference_recovered"
                    price_fallbacks += 1
                else:
                    failures[f"{method.value}:nonfinite_delta:price_delta"] += 1
        if method.value == "crank_nicolson":
            fdm_delta_valuations = len(valid_indices)
            fdm_price_valuations = len(fallback_positions) * 2
        configured_steps = int(
            solver_parameters.get("steps", solver_parameters.get("time_grid", 0))
        )
        circuit_after = registry.circuit(method).snapshot()
        if not typed_valuation_api:
            suppressed_delta_calls = max(
                suppressed_delta_calls,
                max(
                    0,
                    int(circuit_after.get("suppressed_calls", 0))
                    - int(circuit_before.get("suppressed_calls", 0)),
                ),
            )
        circuit_open_nodes = max(circuit_open_nodes, suppressed_delta_calls)
        return AmericanDeltaRecoveryBatch(
            deltas=deltas,
            quality_flags=quality,
            reasons=tuple(str(reason) for reason in reasons),
            actual_steps=np.full(len(strikes), configured_steps, dtype=np.int64),
            diagnostics=MappingProxyType(
                {
                    "routed_delta_points": len(strikes),
                    "routed_delta_failures": int(np.count_nonzero(~np.isfinite(deltas))),
                    "node_direct_delta_failures": direct_failures,
                    "node_price_delta_fallback_count": price_fallbacks,
                    "node_crank_nicolson_fdm_delta_valuations": fdm_delta_valuations,
                    "node_crank_nicolson_fdm_price_valuations": fdm_price_valuations,
                    "surface_circuit_actual_infrastructure_failures": max(
                        0,
                        int(circuit_after.get("actual_infrastructure_failures", 0))
                        - int(circuit_before.get("actual_infrastructure_failures", 0)),
                    ),
                    "surface_circuit_suppressed_delta_calls": suppressed_delta_calls,
                    "surface_circuit_suppressed_price_calls": suppressed_price_calls,
                    "surface_circuit_open_node_count": circuit_open_nodes,
                    "surface_circuit_half_open_probes": max(
                        0,
                        int(circuit_after.get("half_open_probes", 0))
                        - int(circuit_before.get("half_open_probes", 0)),
                    ),
                    **{
                        f"routed_delta_failure_{key}": value
                        for key, value in failures.items()
                    },
                }
            ),
        )
    for index, (strike, volatility) in enumerate(zip(strikes, ivs)):
        if not (
            math.isfinite(float(strike))
            and float(strike) > 0.0
            and math.isfinite(float(volatility))
            and float(volatility) > 0.0
        ):
            quality[index] |= QUALITY_INVALID_FITTED_IV_GRID
            failures["invalid_fitted_iv"] += 1
            continue
        request = PricingRequest(
            option_symbol=f"surface-node-{index}",
            snapshot_id=fit.snapshot_id,
            put_call="CALL" if is_call else "PUT",
            exercise_type="A",
            spot=fit.spot,
            strike=float(strike),
            tenor_seconds=fit.tenor_seconds,
            rate=fit.rate,
            dividend_yield=fit.dividend_yield,
            source_collected_at=source_at,
            expiration_at=expiration_at,
        )
        direct_error = "nonfinite_delta"
        direct_performance: dict[str, float | int] = {}
        try:
            value = float(
                registry.delta(
                    request,
                    float(volatility),
                    method=method,
                    performance=direct_performance,
                )
            )
            fdm_delta_valuations += int(
                direct_performance.get("fdm_valuations", 0)
            )
            if _delta_in_model_range(
                value, is_call=is_call, dtype=np.dtype("float64")
            ):
                deltas[index] = value
                reasons[index] = "none"
                continue
        except SolverCircuitOpenError:
            fdm_delta_valuations += int(
                direct_performance.get("fdm_valuations", 0)
            )
            quality[index] |= QUALITY_NONFINITE_CRR_DELTA
            reasons[index] = "solver_circuit_open"
            failures[f"{method.value}:solver_circuit_open"] += 1
            suppressed_delta_calls += 1
            circuit_open_nodes += 1
            direct_failures += 1
            continue
        except SolverInfrastructureError as exc:
            fdm_delta_valuations += int(
                direct_performance.get("fdm_valuations", 0)
            )
            quality[index] |= QUALITY_NONFINITE_CRR_DELTA
            reasons[index] = "solver_infrastructure_failure"
            failures[
                f"{method.value}:infrastructure:{type(exc).__name__}"
            ] += 1
            surface_infrastructure_failures += 1
            direct_failures += 1
            continue
        except (SolverNumericalError, FloatingPointError) as exc:
            fdm_delta_valuations += int(
                direct_performance.get("fdm_valuations", 0)
            )
            direct_error = type(exc).__name__
        except Exception as exc:
            fdm_delta_valuations += int(
                direct_performance.get("fdm_valuations", 0)
            )
            quality[index] |= QUALITY_NONFINITE_CRR_DELTA
            reasons[index] = "solver_unknown_outcome"
            failures[
                f"{method.value}:unknown:{type(exc).__name__}"
            ] += 1
            direct_failures += 1
            continue
        quality[index] |= QUALITY_NONFINITE_CRR_DELTA
        direct_failures += 1
        bump = max(abs(fit.spot) * 1.0e-4, 1.0e-4)
        lower_performance: dict[str, float | int] = {}
        try:
            lower = registry.price(
                replace(request, spot=fit.spot - bump),
                float(volatility),
                method=method,
                performance=lower_performance,
            )
            fdm_price_valuations += int(
                lower_performance.get("fdm_valuations", 0)
            )
        except SolverCircuitOpenError:
            fdm_price_valuations += int(
                lower_performance.get("fdm_valuations", 0)
            )
            reasons[index] = "solver_circuit_open"
            suppressed_price_calls += 1
            circuit_open_nodes += 1
            failures[f"{method.value}:solver_circuit_open"] += 1
            continue
        except SolverInfrastructureError as exc:
            fdm_price_valuations += int(
                lower_performance.get("fdm_valuations", 0)
            )
            reasons[index] = "solver_infrastructure_failure"
            surface_infrastructure_failures += 1
            failures[
                f"{method.value}:infrastructure:{type(exc).__name__}"
            ] += 1
            continue
        except (SolverNumericalError, FloatingPointError) as exc:
            fdm_price_valuations += int(
                lower_performance.get("fdm_valuations", 0)
            )
            failures[
                f"{method.value}:{direct_error}:{type(exc).__name__}"
            ] += 1
            continue
        except Exception as exc:
            fdm_price_valuations += int(
                lower_performance.get("fdm_valuations", 0)
            )
            reasons[index] = "solver_unknown_outcome"
            failures[
                f"{method.value}:unknown:{type(exc).__name__}"
            ] += 1
            continue
        upper_performance: dict[str, float | int] = {}
        try:
            upper = registry.price(
                replace(request, spot=fit.spot + bump),
                float(volatility),
                method=method,
                performance=upper_performance,
            )
            fdm_price_valuations += int(
                upper_performance.get("fdm_valuations", 0)
            )
            value = (upper - lower) / (2.0 * bump)
        except SolverCircuitOpenError:
            fdm_price_valuations += int(
                upper_performance.get("fdm_valuations", 0)
            )
            reasons[index] = "solver_circuit_open"
            suppressed_price_calls += 1
            circuit_open_nodes += 1
            failures[f"{method.value}:solver_circuit_open"] += 1
            continue
        except SolverInfrastructureError as exc:
            fdm_price_valuations += int(
                upper_performance.get("fdm_valuations", 0)
            )
            reasons[index] = "solver_infrastructure_failure"
            surface_infrastructure_failures += 1
            failures[
                f"{method.value}:infrastructure:{type(exc).__name__}"
            ] += 1
            continue
        except (SolverNumericalError, FloatingPointError) as exc:
            fdm_price_valuations += int(
                upper_performance.get("fdm_valuations", 0)
            )
            failures[
                f"{method.value}:{direct_error}:{type(exc).__name__}"
            ] += 1
            continue
        except Exception as exc:
            fdm_price_valuations += int(
                upper_performance.get("fdm_valuations", 0)
            )
            reasons[index] = "solver_unknown_outcome"
            failures[
                f"{method.value}:unknown:{type(exc).__name__}"
            ] += 1
            continue
        if not _delta_in_model_range(
            value, is_call=is_call, dtype=np.dtype("float64")
        ):
            failures[f"{method.value}:{direct_error}:price_delta"] += 1
            continue
        deltas[index] = value
        quality[index] |= QUALITY_CRR_DELTA_PRICE_FALLBACK
        reasons[index] = "solver_price_finite_difference_recovered"
        price_fallbacks += 1
    configured_steps = int(
        solver_parameters.get("steps", solver_parameters.get("time_grid", 0))
    )
    circuit_after = registry.circuit(method).snapshot()
    return AmericanDeltaRecoveryBatch(
        deltas=deltas,
        quality_flags=quality,
        reasons=tuple(str(reason) for reason in reasons),
        actual_steps=np.full(len(strikes), configured_steps, dtype=np.int64),
        diagnostics=MappingProxyType(
            {
                "routed_delta_points": len(strikes),
                "routed_delta_failures": int(np.count_nonzero(~np.isfinite(deltas))),
                "node_direct_delta_failures": direct_failures,
                "node_price_delta_fallback_count": price_fallbacks,
                "node_crank_nicolson_fdm_delta_valuations": (
                    fdm_delta_valuations
                    if method.value == "crank_nicolson"
                    else 0
                ),
                "node_crank_nicolson_fdm_price_valuations": (
                    fdm_price_valuations
                    if method.value == "crank_nicolson"
                    else 0
                ),
                "surface_circuit_actual_infrastructure_failures": surface_infrastructure_failures,
                "surface_circuit_suppressed_delta_calls": suppressed_delta_calls,
                "surface_circuit_suppressed_price_calls": suppressed_price_calls,
                "surface_circuit_open_node_count": circuit_open_nodes,
                "surface_circuit_half_open_probes": max(
                    0,
                    int(circuit_after.get("half_open_probes", 0))
                    - int(circuit_before.get("half_open_probes", 0)),
                ),
                **{f"routed_delta_failure_{key}": value for key, value in failures.items()},
            }
        ),
    )


def _delta_in_model_range(value: float, *, is_call: bool, dtype: np.dtype) -> bool:
    if not math.isfinite(value):
        return False
    tolerance = 1.0e-3 if dtype == np.dtype("float32") else 2.0e-4
    if is_call:
        return -tolerance <= value <= 1.0 + tolerance
    return -1.0 - tolerance <= value <= tolerance


def _american_price_finite_difference_deltas(
    fit: SmileFit,
    strikes: np.ndarray,
    ivs: np.ndarray,
    dtype: np.dtype,
    steps: int,
    *,
    is_call: bool,
    context: Mapping[str, Any],
    runtime: Optional[SurfaceNumericalRuntime] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-IV central CRR price deltas and finite-price indicators."""
    import torch

    count = len(strikes)
    output = np.full(count, np.nan, dtype=np.float64)
    finite_prices = np.zeros(count, dtype=bool)
    if not count:
        return output, finite_prices
    relative_bump = float(
        _surface_setting(context, "delta_relative_bump", 1.0e-4)
    )
    minimum_bump = float(
        _surface_setting(
            context, "delta_minimum_absolute_bump", 1.0e-4
        )
    )
    maximum_relative_bump = float(
        _surface_setting(context, "delta_maximum_relative_bump", 1.0e-2)
    )
    bump = min(
        max(abs(float(fit.spot)) * relative_bump, minimum_bump),
        abs(float(fit.spot)) * maximum_relative_bump,
    )
    if (
        not math.isfinite(bump)
        or bump <= 0.0
        or not math.isfinite(float(fit.spot))
        or fit.spot - bump <= 0.0
    ):
        return output, finite_prices
    runtime = runtime or SurfaceNumericalRuntime(torch.device("cpu"), 50_000)
    torch_dtype = torch.float32 if dtype == np.dtype("float32") else torch.float64
    module = _legacy_crr_module(
        steps=steps,
        chunk_size=runtime.chunk_size,
        device=str(runtime.device),
    )
    prices = np.asarray(
        module.compute_prices_crr_from_iv(
            S=np.concatenate(
                (
                    np.full(count, fit.spot - bump),
                    np.full(count, fit.spot + bump),
                )
            ),
            K=np.tile(np.asarray(strikes, dtype=np.float64), 2),
            T=np.full(count * 2, fit.tenor_years),
            r=np.full(count * 2, fit.rate),
            q=np.full(count * 2, fit.dividend_yield),
            is_call=np.full(count * 2, is_call, dtype=bool),
            sigma=np.tile(np.asarray(ivs, dtype=np.float64), 2),
            steps=int(steps),
            dtype=torch_dtype,
            device=runtime.device,
            chunk_size=max(1, min(runtime.chunk_size, count * 2)),
            to_numpy=True,
        ),
        dtype=np.float64,
    )
    lower, upper = prices[:count], prices[count:]
    finite_prices = np.isfinite(lower) & np.isfinite(upper)
    output[finite_prices] = (
        upper[finite_prices] - lower[finite_prices]
    ) / (2.0 * bump)
    valid_delta = np.asarray(
        [
            _delta_in_model_range(value, is_call=is_call, dtype=dtype)
            for value in output
        ],
        dtype=bool,
    )
    output[~valid_delta] = np.nan
    return output, finite_prices


def _recover_american_delta_points(
    fit: SmileFit,
    strikes: np.ndarray,
    ivs: np.ndarray,
    dtype: np.dtype,
    steps: int,
    *,
    is_call: bool,
    context: Mapping[str, Any],
    runtime: Optional[SurfaceNumericalRuntime] = None,
) -> AmericanDeltaRecoveryBatch:
    """Apply the bounded point-local direct/scalar/price recovery chain."""
    dtype = np.dtype(dtype)
    strikes = np.asarray(strikes, dtype=np.float64).reshape(-1)
    ivs = np.asarray(ivs, dtype=np.float64).reshape(-1)
    if len(strikes) != len(ivs):
        raise ValueError("American delta strikes and IVs must be aligned")
    count = len(strikes)
    deltas = np.full(count, np.nan, dtype=np.float64)
    quality = np.zeros(count, dtype=np.uint32)
    reasons = np.full(count, "nonfinite_direct_crr_delta", dtype=object)
    actual_steps = np.full(count, int(steps), dtype=np.int64)
    stats = Counter(
        direct_grid_points=count,
        nonfinite_direct_points=0,
        scalar_recoveries=0,
        price_fallback_recoveries=0,
        unrecovered_points=0,
        price_kernel_calls=0,
        delta_kernel_calls=0,
        step_escalations=0,
        infeasible_tree_inputs=0,
    )
    try:
        common_model_finite = all(
            math.isfinite(float(value))
            for value in (
                fit.spot,
                fit.tenor_years,
                fit.rate,
                fit.dividend_yield,
            )
        )
    except (TypeError, ValueError, OverflowError):
        common_model_finite = False
    feasible = (
        np.isfinite(strikes)
        & (strikes > 0.0)
        & np.isfinite(ivs)
        & (ivs > 0.0)
    )
    if common_model_finite and np.any(feasible):
        floors = american_crr_sigma_floor(
            tenor_years=np.full(count, fit.tenor_years),
            rate=np.full(count, fit.rate),
            dividend_yield=np.full(count, fit.dividend_yield),
            steps=int(steps),
            dtype=dtype,
        )
        margin = 1.0 + max(1.0e-7, np.finfo(dtype).eps * 16.0)
        feasible &= np.isfinite(floors) & (ivs >= floors * margin)
    else:
        feasible[:] = False
    infeasible = ~feasible
    quality[infeasible] |= QUALITY_CRR_TREE_INPUT_INFEASIBLE
    reasons[infeasible] = "crr_tree_input_infeasible"
    stats["infeasible_tree_inputs"] = int(np.count_nonzero(infeasible))

    feasible_indices = np.flatnonzero(feasible)
    if len(feasible_indices):
        primary = _american_deltas(
            fit,
            strikes[feasible_indices],
            ivs[feasible_indices],
            dtype,
            int(steps),
            is_call=is_call,
            runtime=runtime,
        )
        stats["delta_kernel_calls"] += 1
        primary_valid = np.asarray(
            [
                _delta_in_model_range(value, is_call=is_call, dtype=dtype)
                for value in primary
            ]
        )
        direct_indices = feasible_indices[primary_valid]
        deltas[direct_indices] = primary[primary_valid]
        reasons[direct_indices] = "none"
        failed_indices = feasible_indices[~primary_valid]
        quality[failed_indices] |= QUALITY_NONFINITE_CRR_DELTA
        stats["nonfinite_direct_points"] += len(failed_indices)
    else:
        failed_indices = np.asarray([], dtype=np.int64)

    def scalar_recover(indices: np.ndarray, retry_steps: int) -> np.ndarray:
        unresolved = []
        for index in indices:
            value = _american_delta(
                fit,
                float(strikes[index]),
                float(ivs[index]),
                dtype,
                retry_steps,
                is_call=is_call,
                runtime=runtime,
            )
            stats["delta_kernel_calls"] += 1
            used_float64 = False
            if (
                not _delta_in_model_range(value, is_call=is_call, dtype=dtype)
                and dtype == np.dtype("float32")
                and _as_bool(
                    context.get(
                        OPTIONS_CALCS_CONFIG_PREFIX + "auto_promote_to_float64",
                        True,
                    )
                )
            ):
                value = _american_delta(
                    fit,
                    float(strikes[index]),
                    float(ivs[index]),
                    np.dtype("float64"),
                    retry_steps,
                    is_call=is_call,
                    runtime=runtime,
                )
                stats["delta_kernel_calls"] += 1
                used_float64 = True
            validation_dtype = np.dtype("float64") if used_float64 else dtype
            if _delta_in_model_range(
                value, is_call=is_call, dtype=validation_dtype
            ):
                deltas[index] = value
                quality[index] |= QUALITY_CRR_DELTA_SCALAR_RECOVERY
                if used_float64:
                    quality[index] |= QUALITY_NODE_PRECISION_ESCALATED
                reasons[index] = "scalar_direct_delta_recovered"
                actual_steps[index] = retry_steps
                stats["scalar_recoveries"] += 1
            else:
                unresolved.append(index)
        return np.asarray(unresolved, dtype=np.int64)

    unresolved = scalar_recover(failed_indices, int(steps))

    def price_recover(indices: np.ndarray, retry_steps: int) -> np.ndarray:
        if not len(indices):
            return indices
        values, finite_prices = _american_price_finite_difference_deltas(
            fit,
            strikes[indices],
            ivs[indices],
            dtype,
            retry_steps,
            is_call=is_call,
            context=context,
            runtime=runtime,
        )
        stats["price_kernel_calls"] += 1
        recovered = np.asarray(
            [
                _delta_in_model_range(value, is_call=is_call, dtype=dtype)
                for value in values
            ],
            dtype=bool,
        )
        recovered_indices = indices[recovered]
        deltas[recovered_indices] = values[recovered]
        quality[recovered_indices] |= QUALITY_CRR_DELTA_PRICE_FALLBACK
        reasons[recovered_indices] = "crr_price_finite_difference_recovered"
        actual_steps[recovered_indices] = retry_steps
        stats["price_fallback_recoveries"] += len(recovered_indices)
        failed = indices[~recovered]
        reasons[failed[~finite_prices[~recovered]]] = "crr_price_nonfinite"
        return failed

    if _as_bool(
        _surface_setting(context, "delta_recovery_enabled", True)
    ):
        multiplier = int(
            _surface_setting(context, "delta_recovery_steps_multiplier", 1)
        )
        fallback_steps = int(steps) * multiplier
        maximum_steps = int(
            _surface_setting(context, "delta_maximum_steps", int(steps) * 4)
        )
        if fallback_steps <= maximum_steps:
            if fallback_steps > int(steps):
                quality[unresolved] |= QUALITY_CRR_DELTA_STEP_ESCALATED
                stats["step_escalations"] += len(unresolved)
            unresolved = price_recover(unresolved, fallback_steps)

    maximum_steps = int(
        _surface_setting(context, "delta_maximum_steps", int(steps) * 4)
    )
    doubled_steps = int(steps) * 2
    if len(unresolved) and doubled_steps <= maximum_steps:
        quality[unresolved] |= QUALITY_CRR_DELTA_STEP_ESCALATED
        stats["step_escalations"] += len(unresolved)
        retried = _american_deltas(
            fit,
            strikes[unresolved],
            ivs[unresolved],
            dtype,
            doubled_steps,
            is_call=is_call,
            runtime=runtime,
        )
        stats["delta_kernel_calls"] += 1
        valid_retry = np.asarray(
            [
                _delta_in_model_range(value, is_call=is_call, dtype=dtype)
                for value in retried
            ]
        )
        direct_retry_indices = unresolved[valid_retry]
        deltas[direct_retry_indices] = retried[valid_retry]
        reasons[direct_retry_indices] = "none"
        actual_steps[direct_retry_indices] = doubled_steps
        unresolved = scalar_recover(unresolved[~valid_retry], doubled_steps)
        if _as_bool(
            _surface_setting(context, "delta_recovery_enabled", True)
        ):
            unresolved = price_recover(unresolved, doubled_steps)
    stats["unrecovered_points"] = int(
        np.count_nonzero(feasible & ~np.isfinite(deltas))
    )
    return AmericanDeltaRecoveryBatch(
        deltas=deltas,
        quality_flags=quality,
        reasons=tuple(str(reason) for reason in reasons),
        actual_steps=actual_steps,
        diagnostics=MappingProxyType(dict(stats)),
    )


def diagnose_american_delta_grid(
    fit: SmileFit,
    *,
    context: Mapping[str, Any],
    is_call: bool,
    dtype: np.dtype | str,
    runtime: Optional[SurfaceNumericalRuntime] = None,
) -> Mapping[str, Any]:
    """Reproduce one stored-fit CRR delta grid without mutating the fit.

    The returned summary is suitable for a single INFO/WARNING record.  The
    point records are intentionally separate so callers can keep them at
    DEBUG level.
    """
    import torch

    dtype = np.dtype(dtype)
    if dtype.name not in SUPPORTED_DTYPES:
        raise ValueError("dtype must be float32 or float64")
    if fit.exercise_type.strip().upper() not in {"A", "AMERICAN"}:
        raise ValueError("American delta diagnostics require an American fit")
    try:
        diagnostic_dividend_yield = float(fit.dividend_yield)
    except (TypeError, ValueError, OverflowError):
        diagnostic_dividend_yield = math.nan
    minimum_k, maximum_k = fit.coverage
    if minimum_k is None or maximum_k is None:
        raise ValueError("American delta diagnostics require finite fit coverage")
    steps = int(context.get(OPTIONS_CALCS_GREEKS_PREFIX + "crr.steps", 256))
    runtime = runtime or SurfaceNumericalRuntime(torch.device("cpu"), 50_000)
    module = _legacy_crr_module(
        steps=steps,
        chunk_size=runtime.chunk_size,
        device=str(runtime.device),
    )
    grid = np.linspace(minimum_k, maximum_k, 81, dtype=np.float64)
    strikes = fit.forward * np.exp(grid)
    ivs = np.asarray(
        [
            _fit_iv(fit, float(log_strike))
            if _fit_iv(fit, float(log_strike)) is not None
            else math.nan
            for log_strike in grid
        ],
        dtype=np.float64,
    )
    valid = (
        np.isfinite(strikes)
        & (strikes > 0.0)
        & np.isfinite(ivs)
        & (ivs > 0.0)
    )
    direct = np.full(len(grid), np.nan, dtype=np.float64)
    indices = np.flatnonzero(valid)
    if len(indices):
        direct[indices] = _american_deltas(
            fit,
            strikes[indices],
            ivs[indices],
            dtype,
            steps,
            is_call=is_call,
            runtime=runtime,
        )
    scalar = np.full(len(grid), np.nan, dtype=np.float64)
    for index in indices:
        scalar[index] = _american_delta(
            fit,
            float(strikes[index]),
            float(ivs[index]),
            dtype,
            steps,
            is_call=is_call,
            runtime=runtime,
        )

    dtype_results: dict[str, list[Optional[float]]] = {}
    for probe_dtype in (np.dtype("float32"), np.dtype("float64")):
        values = np.full(len(grid), np.nan, dtype=np.float64)
        if len(indices):
            values[indices] = _american_deltas(
                fit,
                strikes[indices],
                ivs[indices],
                probe_dtype,
                steps,
                is_call=is_call,
                runtime=runtime,
            )
        dtype_results[probe_dtype.name] = [
            float(value) if math.isfinite(float(value)) else None
            for value in values
        ]

    doubled = np.full(len(grid), np.nan, dtype=np.float64)
    if len(indices):
        doubled[indices] = _american_deltas(
            fit,
            strikes[indices],
            ivs[indices],
            dtype,
            steps * 2,
            is_call=is_call,
            runtime=runtime,
        )

    relative_bump = float(
        _surface_setting(context, "delta_relative_bump", 1.0e-4)
    )
    minimum_bump = float(
        _surface_setting(
            context, "delta_minimum_absolute_bump", 1.0e-4
        )
    )
    bump = max(abs(float(fit.spot)) * relative_bump, minimum_bump)
    if not math.isfinite(bump) or bump <= 0.0 or fit.spot - bump <= 0.0:
        bump = math.nan
    lower_prices = np.full(len(grid), np.nan, dtype=np.float64)
    center_prices = np.full(len(grid), np.nan, dtype=np.float64)
    upper_prices = np.full(len(grid), np.nan, dtype=np.float64)
    if len(indices) and math.isfinite(bump):
        count = len(indices)
        spot_values = np.concatenate(
            (
                np.full(count, fit.spot - bump),
                np.full(count, fit.spot),
                np.full(count, fit.spot + bump),
            )
        )
        tiled_strikes = np.tile(strikes[indices], 3)
        tiled_ivs = np.tile(ivs[indices], 3)
        torch_dtype = (
            torch.float32 if dtype == np.dtype("float32") else torch.float64
        )
        prices = np.asarray(
            module.compute_prices_crr_from_iv(
                S=spot_values,
                K=tiled_strikes,
                T=np.full(count * 3, fit.tenor_years),
                r=np.full(count * 3, fit.rate),
                q=np.full(count * 3, diagnostic_dividend_yield),
                is_call=np.full(count * 3, is_call, dtype=bool),
                sigma=tiled_ivs,
                steps=steps,
                dtype=torch_dtype,
                device=runtime.device,
                chunk_size=max(1, min(runtime.chunk_size, count * 3)),
                to_numpy=True,
            ),
            dtype=np.float64,
        )
        lower_prices[indices] = prices[:count]
        center_prices[indices] = prices[count : count * 2]
        upper_prices[indices] = prices[count * 2 :]
    finite_difference = np.divide(
        upper_prices - lower_prices,
        2.0 * bump,
        out=np.full(len(grid), np.nan, dtype=np.float64),
        where=math.isfinite(bump),
    )
    scalar_mismatch = (
        np.isfinite(direct)
        & np.isfinite(scalar)
        & ~np.isclose(
            direct,
            scalar,
            atol=2.0e-4 if dtype == np.dtype("float64") else 1.0e-3,
            rtol=0.0,
        )
    )
    finite_prices = np.isfinite(lower_prices) & np.isfinite(upper_prices)
    summary = AmericanDeltaGridDiagnostics(
        grid_point_count=len(grid),
        valid_iv_point_count=int(np.count_nonzero(valid)),
        finite_direct_delta_count=int(np.count_nonzero(valid & np.isfinite(direct))),
        nonfinite_direct_delta_count=int(
            np.count_nonzero(valid & ~np.isfinite(direct))
        ),
        finite_scalar_retry_count=int(
            np.count_nonzero(valid & ~np.isfinite(direct) & np.isfinite(scalar))
        ),
        finite_price_fallback_count=int(
            np.count_nonzero(valid & ~np.isfinite(direct) & np.isfinite(finite_difference))
        ),
        direct_scalar_mismatch_count=int(np.count_nonzero(scalar_mismatch)),
        price_kernel_nonfinite_count=int(np.count_nonzero(valid & ~finite_prices)),
        minimum_strike=(float(np.min(strikes[valid])) if np.any(valid) else None),
        maximum_strike=(float(np.max(strikes[valid])) if np.any(valid) else None),
        minimum_iv=(float(np.min(ivs[valid])) if np.any(valid) else None),
        maximum_iv=(float(np.max(ivs[valid])) if np.any(valid) else None),
    )
    points = []
    for index in range(len(grid)):
        european = european_option_delta(
            spot=fit.spot,
            strike=float(strikes[index]),
            tenor_years=fit.tenor_years,
            rate=fit.rate,
            dividend_yield=diagnostic_dividend_yield,
            volatility=float(ivs[index]),
            is_call=is_call,
            dtype=np.float64,
        )
        dt = fit.tenor_years / steps
        up = math.exp(float(ivs[index]) * math.sqrt(dt))
        down = 1.0 / up
        carry_growth = math.exp((fit.rate - diagnostic_dividend_yield) * dt)
        probability = (carry_growth - down) / (up - down)
        points.append(
            {
                "grid_strike": float(strikes[index]),
                "grid_log_moneyness": float(grid[index]),
                "fitted_iv": float(ivs[index]),
                "direct_crr_delta": (
                    float(direct[index]) if math.isfinite(direct[index]) else None
                ),
                "direct_crr_delta_finite": bool(math.isfinite(direct[index])),
                "scalar_crr_delta": (
                    float(scalar[index]) if math.isfinite(scalar[index]) else None
                ),
                "float32_crr_delta": dtype_results["float32"][index],
                "float64_crr_delta": dtype_results["float64"][index],
                "doubled_steps_crr_delta": (
                    float(doubled[index]) if math.isfinite(doubled[index]) else None
                ),
                "crr_price_spot_minus_h": (
                    float(lower_prices[index])
                    if math.isfinite(lower_prices[index])
                    else None
                ),
                "crr_price_spot": (
                    float(center_prices[index])
                    if math.isfinite(center_prices[index])
                    else None
                ),
                "crr_price_spot_plus_h": (
                    float(upper_prices[index])
                    if math.isfinite(upper_prices[index])
                    else None
                ),
                "finite_difference_crr_delta": (
                    float(finite_difference[index])
                    if math.isfinite(finite_difference[index])
                    else None
                ),
                "european_delta_diagnostic": (
                    float(european) if math.isfinite(european) else None
                ),
                "up_factor": up if math.isfinite(up) else None,
                "down_factor": down if math.isfinite(down) else None,
                "risk_neutral_probability": (
                    probability if math.isfinite(probability) else None
                ),
                "discount_factor": math.exp(-fit.rate * dt),
                "first_step_denominator": fit.spot * (up - down),
            }
        )
    return MappingProxyType(
        {
            "price_basis": fit.price_basis,
            "exercise_type": fit.exercise_type,
            "fit_method": fit.method,
            "fit_parameters": dict(fit.parameters),
            "spot": fit.spot,
            "forward": fit.forward,
            "rate": fit.rate,
            "dividend_yield": fit.dividend_yield,
            "tenor_years": fit.tenor_years,
            "tenor_seconds": fit.tenor_seconds,
            "requested_dtype": fit.requested_dtype,
            "actual_dtype": dtype.name,
            "crr_steps": steps,
            "doubled_crr_steps": steps * 2,
            "crr_device": str(runtime.device),
            "fit_coverage": [minimum_k, maximum_k],
            "put_call": "CALL" if is_call else "PUT",
            "spot_bump": bump if math.isfinite(bump) else None,
            "summary": summary.as_dict(),
            "points": tuple(points),
        }
    )


def _model_delta(
    fit: SmileFit,
    strike: float,
    *,
    is_call: bool,
    dtype: np.dtype,
    crr_steps: int,
    runtime: Optional[SurfaceNumericalRuntime] = None,
) -> float:
    k = math.log(strike / fit.forward)
    iv = _fit_iv(fit, k)
    if iv is None:
        return math.nan
    exercise = fit.exercise_type.strip().upper()
    if exercise in {"E", "EUROPEAN"}:
        return european_option_delta(
            spot=fit.spot,
            strike=strike,
            tenor_years=fit.tenor_years,
            rate=fit.rate,
            dividend_yield=fit.dividend_yield,
            volatility=iv,
            is_call=is_call,
            dtype=dtype,
        )
    return _american_delta(
        fit, strike, iv, dtype, crr_steps, is_call=is_call, runtime=runtime
    )


def _confidence(fit: SmileFit) -> float:
    rmse = float(fit.weighted_rmse or 0.0)
    coverage = fit.coverage
    width = max((coverage[1] - coverage[0]) if coverage[0] is not None else 0.0, 0.0)
    return float(max(0.0, min(1.0, math.exp(-50.0 * rmse) * min(1.0, width / 0.2))))


def _find_delta_bracket(
    x_values: Sequence[float],
    y_values: Sequence[float],
    target: float,
) -> Optional[tuple[float, float, float, float]]:
    """Return the best adjacent finite sign-changing interval."""
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    if len(x_array) < 2 or len(x_array) != len(y_array):
        return None
    order = np.argsort(x_array, kind="stable")
    x_array, y_array = x_array[order], y_array[order]
    candidates: list[tuple[float, float, float, float, float]] = []
    for index in range(len(x_array) - 1):
        left = float(x_array[index])
        right = float(x_array[index + 1])
        left_delta = float(y_array[index])
        right_delta = float(y_array[index + 1])
        if not all(
            math.isfinite(value)
            for value in (left, right, left_delta, right_delta)
        ):
            continue
        left_error = left_delta - target
        right_error = right_delta - target
        if left_error == 0.0 or left_error * right_error <= 0.0:
            candidates.append(
                (
                    abs(left_error) + abs(right_error),
                    left,
                    right,
                    left_delta,
                    right_delta,
                )
            )
    if not candidates:
        return None
    _score, left, right, left_delta, right_delta = min(candidates)
    return left, right, left_delta, right_delta


def _interpolated_delta_candidate(
    bracket: tuple[float, float, float, float],
    target: float,
    *,
    safeguard: bool = False,
) -> float:
    """Stable secant proposal with an optional interior bisection guard."""
    left, right, left_delta, right_delta = bracket
    denominator = right_delta - left_delta
    if not math.isfinite(denominator) or abs(denominator) < 1.0e-15:
        return (left + right) / 2.0
    candidate = left + (target - left_delta) * (right - left) / denominator
    if not math.isfinite(candidate) or candidate <= left or candidate >= right:
        return (left + right) / 2.0
    if safeguard:
        margin = max((right - left) * 1.0e-6, 1.0e-14)
        if candidate <= left + margin or candidate >= right - margin:
            return (left + right) / 2.0
    return min(right, max(left, candidate))


def _tian_search_cache_key(
    fit: SmileFit, config: SurfaceNodeSearchConfig
) -> tuple[Any, ...]:
    return (
        str(fit.snapshot_id),
        str(fit.price_basis),
        str(fit.expiration_at),
        float(fit.tenor_seconds),
        tuple(fit.coverage),
        int(config.tian_grid_size),
        str(fit.method),
        tuple(sorted((str(key), repr(value)) for key, value in fit.parameters.items())),
    )


def _build_tian_delta_search_grid(
    fit: SmileFit,
    *,
    context: Mapping[str, Any],
    config: SurfaceNodeSearchConfig,
    solver_registry: Optional[Any],
    minimum_iv: float,
    maximum_iv: float,
    cache: Optional[MutableMapping[tuple[Any, ...], DeltaSearchGrid]],
) -> tuple[DeltaSearchGrid, bool, Mapping[str, int]]:
    """Build the two side-specific Tian arrays exactly once for this fit."""
    cache_key = _tian_search_cache_key(fit, config)
    if cache is not None and cache_key in cache:
        return cache[cache_key], False, MappingProxyType({})
    minimum_k, maximum_k = fit.coverage
    if minimum_k is None or maximum_k is None:
        raise ValueError("Tian node search requires finite smile coverage")
    log_moneyness = np.linspace(
        minimum_k, maximum_k, config.tian_grid_size, dtype=np.float64
    )
    strikes = fit.forward * np.exp(log_moneyness)
    ivs = np.full(len(log_moneyness), np.nan, dtype=np.float64)
    for index, value in enumerate(log_moneyness):
        fitted = _fit_iv(fit, float(value))
        if (
            fitted is not None
            and math.isfinite(fitted)
            and minimum_iv <= fitted <= maximum_iv
        ):
            ivs[index] = fitted
    valid = (
        np.isfinite(log_moneyness)
        & np.isfinite(strikes)
        & (strikes > 0.0)
        & np.isfinite(ivs)
        & (ivs >= minimum_iv)
        & (ivs <= maximum_iv)
    )
    call_deltas = np.full(len(log_moneyness), np.nan, dtype=np.float64)
    put_deltas = np.full(len(log_moneyness), np.nan, dtype=np.float64)
    call_quality = np.zeros(len(log_moneyness), dtype=np.uint32)
    put_quality = np.zeros(len(log_moneyness), dtype=np.uint32)
    call_reasons = np.full(len(log_moneyness), "outside_smile_coverage", dtype=object)
    put_reasons = np.full(len(log_moneyness), "outside_smile_coverage", dtype=object)
    valid_indices = np.flatnonzero(valid)
    diagnostics: Counter[str] = Counter()
    if len(valid_indices):
        from algotrading.refdata.options.solvers.Registry import SolverRegistry

        registry = solver_registry or SolverRegistry(context)
        for put_call, output, quality_output, reason_output in (
            ("CALL", call_deltas, call_quality, call_reasons),
            ("PUT", put_deltas, put_quality, put_reasons),
        ):
            points = tuple(
                SurfaceExactDeltaPoint(
                    cache_key=(f"TIAN:{put_call}", round(float(log_moneyness[index]), 14)),
                    log_strike=float(log_moneyness[index]),
                    put_call=put_call,
                    strike=float(strikes[index]),
                    volatility=float(ivs[index]),
                )
                for index in valid_indices
            )
            recovered = _routed_american_delta_waves(
                ((fit, points),),
                context=context,
                solver_registry=registry,
                method_override="tian",
            )[0]
            output[valid_indices] = recovered.deltas
            quality_output[valid_indices] = recovered.quality_flags
            reason_output[valid_indices] = recovered.reasons
            for name, value in recovered.diagnostics.items():
                diagnostics[f"node_tian_search_solver_{name}"] += int(value)
    grid = DeltaSearchGrid(
        log_moneyness=log_moneyness,
        strikes=strikes,
        ivs=ivs,
        valid=valid,
        call_deltas=call_deltas,
        put_deltas=put_deltas,
        call_quality_flags=call_quality,
        put_quality_flags=put_quality,
        call_reasons=tuple(str(value) for value in call_reasons),
        put_reasons=tuple(str(value) for value in put_reasons),
        quality_flags=(QUALITY_INVALID_FITTED_IV_GRID if np.any(~valid) else 0),
    )
    if cache is not None:
        cache[cache_key] = grid
    diagnostics["surface.node_tian_search_evaluations"] += 2 * len(valid_indices)
    diagnostics["surface.node_search_method_tian"] += 1
    return grid, True, MappingProxyType(dict(diagnostics))


def _generate_surface_nodes_legacy(
    fit: SmileFit,
    *,
    context: Mapping[str, Any],
    dtype: np.dtype,
    runtime: Optional[SurfaceNumericalRuntime] = None,
    diagnostics: Optional[MutableMapping[str, int]] = None,
    solver_registry: Optional[Any] = None,
    stage_timings: Optional[MutableMapping[str, float]] = None,
    exact_delta_cache: Optional[
        Mapping[tuple[str, float], tuple[float, int, str]]
    ] = None,
    defer_exact_delta_waves: bool = False,
    precomputed_exact_diagnostics: Optional[Mapping[str, int]] = None,
) -> tuple[SurfaceNode, ...]:
    if not fit.successful:
        return ()
    minimum_k, maximum_k = fit.coverage
    confidence = _confidence(fit)
    nodes: list[SurfaceNode] = []
    american_delta_diagnostics: Counter[str] = Counter(
        precomputed_exact_diagnostics or {}
    )
    crr_steps = int(
        context.get(OPTIONS_CALCS_GREEKS_PREFIX + "crr.steps", 256)
    )
    delta_tolerance = float(_surface_setting(context, "node_delta_tolerance", 1.0e-5))
    minimum_iv = float(_iv_setting(context, "minimum", 1.0e-4))
    maximum_iv = float(_iv_setting(context, "maximum", 10.0))
    delta_values = _surface_setting(context, "delta_nodes", [0.40, 0.25, 0.10])
    if isinstance(delta_values, str):
        delta_values = [float(value.strip()) for value in delta_values.split(",") if value.strip()]
    delta_values = [float(value) for value in delta_values]
    log_nodes = _surface_setting(
        context, "log_forward_moneyness_nodes", [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
    )
    if isinstance(log_nodes, str):
        log_nodes = [float(value.strip()) for value in log_nodes.split(",") if value.strip()]
    log_nodes = [float(value) for value in log_nodes]

    def unavailable_delta(
        delta_value: float,
        put_call: str,
        status: str,
        quality_flags: int = 0,
        node_reason: str = "nonfinite_direct_crr_delta",
    ) -> SurfaceNode:
        return SurfaceNode(
            "delta", delta_value, put_call, None, None, None, None, None,
            None, status, quality_flags, node_reason
        )

    if minimum_k is None or maximum_k is None:
        nodes.extend(
            unavailable_delta(
                delta_value,
                put_call,
                "unavailable",
                node_reason="outside_smile_coverage",
            )
            for delta_value in delta_values
            for put_call in ("PUT", "CALL")
        )
        nodes.append(
            SurfaceNode(
                "atm_forward", 0.0, "NONE", None, None, None, None, None,
                None, "unavailable", node_reason="outside_smile_coverage"
            )
        )
        nodes.extend(
            SurfaceNode(
                "log_forward_moneyness", k, "NONE", None, None, None, None,
                None, None, "unavailable",
                node_reason="optional_node_outside_coverage",
            )
            for k in log_nodes
        )
        return tuple(nodes)

    cheap_search_started = time.perf_counter()
    delta_cache: dict[tuple[str, float], float] = {
        key: float(value[0]) for key, value in (exact_delta_cache or {}).items()
    }
    delta_quality_cache: dict[tuple[str, float], int] = {
        key: int(value[1]) for key, value in (exact_delta_cache or {}).items()
    }
    delta_reason_cache: dict[tuple[str, float], str] = {
        key: str(value[2]) for key, value in (exact_delta_cache or {}).items()
    }
    cheap_grid_size = int(_surface_setting(context, "node_search_grid_size", 21))
    exact_budget = int(_surface_setting(context, "node_exact_max_evaluations", 8))
    dense_fallback_enabled = _as_bool(
        _surface_setting(context, "node_dense_exact_fallback_enabled", True)
    )
    dense_grid_size = int(
        _surface_setting(context, "node_dense_exact_fallback_grid_size", 81)
    )
    delta_search_grid = np.linspace(
        minimum_k, maximum_k, cheap_grid_size, dtype=np.float64
    )
    grid_strikes = fit.forward * np.exp(delta_search_grid)
    grid_ivs = np.full(len(delta_search_grid), np.nan, dtype=np.float64)
    for index, log_strike in enumerate(delta_search_grid):
        fitted = _fit_iv(fit, float(log_strike))
        if (
            fitted is not None
            and math.isfinite(fitted)
            and minimum_iv <= fitted <= maximum_iv
        ):
            grid_ivs[index] = fitted
    valid_grid = (
        np.isfinite(delta_search_grid)
        & np.isfinite(grid_strikes)
        & (grid_strikes > 0.0)
        & np.isfinite(grid_ivs)
        & (grid_ivs > 0.0)
        & (grid_ivs >= minimum_iv)
        & (grid_ivs <= maximum_iv)
        & (delta_search_grid >= minimum_k)
        & (delta_search_grid <= maximum_k)
    )
    grid_quality = (
        QUALITY_INVALID_FITTED_IV_GRID if np.any(~valid_grid) else 0
    )
    grid_deltas_by_side: dict[str, np.ndarray] = {}
    grid_failure_quality_by_side: dict[str, int] = {}
    grid_failure_reason_by_side: dict[str, str] = {}

    def cached_model_deltas(
        points: Sequence[tuple[float, str]],
    ) -> list[float]:
        """Populate the exact-node cache in aligned put/call waves.

        Root-finding remains owner-local and deterministic.  Only the
        independent numerical evaluations for a search wave are combined.
        """

        missing: dict[tuple[str, float], tuple[float, str]] = {}
        for log_strike, put_call in points:
            key = (put_call, round(float(log_strike), 14))
            missing.setdefault(key, (float(log_strike), put_call))
        missing = {key: point for key, point in missing.items() if key not in delta_cache}
        if missing:
            exact_started = time.perf_counter()
            valid_by_side: dict[str, list[tuple[tuple[str, float], float, float]]] = {
                "PUT": [],
                "CALL": [],
            }
            for key, (log_strike, put_call) in missing.items():
                strike = fit.forward * math.exp(log_strike)
                iv = _fit_iv(fit, log_strike)
                if iv is None:
                    delta_cache[key] = math.nan
                    delta_quality_cache[key] = QUALITY_INVALID_FITTED_IV_GRID
                    delta_reason_cache[key] = "outside_smile_coverage"
                else:
                    valid_by_side[put_call].append((key, strike, float(iv)))

            deferred_points = [
                SurfaceExactDeltaPoint(
                    cache_key=key,
                    log_strike=float(log_strike),
                    put_call=put_call,
                    strike=float(strike),
                    volatility=float(volatility),
                )
                for put_call in ("PUT", "CALL")
                for key, strike, volatility in valid_by_side[put_call]
                for log_strike in (missing[key][0],)
            ]
            if defer_exact_delta_waves and deferred_points:
                raise PendingSurfaceExactDeltaWave(
                    fit,
                    deferred_points,
                    dense_fallback=in_dense_fallback,
                )

            american_delta_diagnostics["node_exact_delta_evaluations"] += len(missing)

            for put_call in ("PUT", "CALL"):
                valid = valid_by_side[put_call]
                if not valid:
                    continue
                if fit.exercise_type.strip().upper() in {"A", "AMERICAN"}:
                    recovered = _routed_american_delta_points(
                        fit,
                        np.asarray([point[1] for point in valid], dtype=np.float64),
                        np.asarray([point[2] for point in valid], dtype=np.float64),
                        is_call=put_call == "CALL",
                        context=context,
                        solver_registry=solver_registry,
                    )
                    for index, (key, _strike, _iv) in enumerate(valid):
                        delta_cache[key] = float(recovered.deltas[index])
                        delta_quality_cache[key] = int(recovered.quality_flags[index])
                        delta_reason_cache[key] = recovered.reasons[index]
                    american_delta_diagnostics.update(recovered.diagnostics)
                else:
                    for key, strike, _iv in valid:
                        delta_cache[key] = _model_delta(
                            fit,
                            strike,
                            is_call=put_call == "CALL",
                            dtype=dtype,
                            crr_steps=crr_steps,
                            runtime=runtime,
                        )
                        delta_quality_cache[key] = 0
                        delta_reason_cache[key] = "none"
            if stage_timings is not None and not in_dense_fallback:
                stage_timings["surface.node_exact_refinement_seconds"] = (
                    stage_timings.get(
                        "surface.node_exact_refinement_seconds", 0.0
                    )
                    + time.perf_counter()
                    - exact_started
                )
        return [
            delta_cache[(put_call, round(float(log_strike), 14))]
            for log_strike, put_call in points
        ]

    def cached_model_delta(log_strike: float, put_call: str) -> float:
        return cached_model_deltas([(log_strike, put_call)])[0]

    # This grid is deliberately cheap: observed deltas are preferred below;
    # European analytic deltas seed only those targets lacking an observed
    # bracket.  Published American nodes are still exact-solver validated.
    for put_call in ("PUT", "CALL"):
        full_deltas = np.full(len(delta_search_grid), np.nan, dtype=np.float64)
        for index in np.flatnonzero(valid_grid):
            full_deltas[index] = european_option_delta(
                spot=fit.spot,
                strike=float(grid_strikes[index]),
                tenor_years=fit.tenor_years,
                rate=fit.rate,
                dividend_yield=fit.dividend_yield,
                volatility=float(grid_ivs[index]),
                is_call=put_call == "CALL",
                dtype=np.dtype("float64"),
            )
            american_delta_diagnostics["node_analytic_delta_evaluations"] += 1
        grid_deltas_by_side[put_call] = full_deltas
        grid_failure_quality_by_side[put_call] = 0
        grid_failure_reason_by_side[put_call] = "target_delta_outside_coverage"
    if stage_timings is not None:
        stage_timings["surface.node_cheap_search_seconds"] = stage_timings.get(
            "surface.node_cheap_search_seconds", 0.0
        ) + (time.perf_counter() - cheap_search_started)

    def contiguous_segments(put_call: str) -> list[np.ndarray]:
        deltas = grid_deltas_by_side[put_call]
        indices = np.flatnonzero(valid_grid & np.isfinite(deltas))
        if not len(indices):
            return []
        return [
            segment
            for segment in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
            if len(segment) >= 2
        ]

    tolerance = max(delta_tolerance, np.finfo(dtype).eps * 64)

    def find_bracket(
        x_values: np.ndarray, y_values: np.ndarray, target: float
    ) -> Optional[tuple[float, float, float, float]]:
        x_values = np.asarray(x_values, dtype=np.float64)
        y_values = np.asarray(y_values, dtype=np.float64)
        if len(x_values) < 2:
            return None
        order = np.argsort(x_values, kind="stable")
        x_values, y_values = x_values[order], y_values[order]
        candidates = []
        for left_index in range(len(x_values) - 1):
            if not (
                math.isfinite(float(x_values[left_index]))
                and math.isfinite(float(x_values[left_index + 1]))
                and math.isfinite(float(y_values[left_index]))
                and math.isfinite(float(y_values[left_index + 1]))
            ):
                continue
            left_error = y_values[left_index] - target
            right_error = y_values[left_index + 1] - target
            if left_error == 0.0 or left_error * right_error <= 0.0:
                candidates.append(
                    (
                        abs(left_error) + abs(right_error),
                        float(x_values[left_index]),
                        float(x_values[left_index + 1]),
                        float(y_values[left_index]),
                        float(y_values[left_index + 1]),
                    )
                )
        if not candidates:
            return None
        _, left, right, left_delta, right_delta = min(candidates)
        return left, right, left_delta, right_delta

    def interpolated_candidate(
        bracket: tuple[float, float, float, float], target: float
    ) -> float:
        left, right, left_delta, right_delta = bracket
        denominator = right_delta - left_delta
        if not math.isfinite(denominator) or abs(denominator) < 1.0e-15:
            return (left + right) / 2.0
        return min(
            right,
            max(left, left + (target - left_delta) * (right - left) / denominator),
        )

    dense_exact_bracket_found = False
    in_dense_fallback = False

    def dense_exact_fallback(
        put_call: str, target: float
    ) -> Optional[tuple[float, float, int, str]]:
        nonlocal dense_exact_bracket_found, in_dense_fallback
        dense_exact_bracket_found = False
        if not dense_fallback_enabled:
            return None
        american_delta_diagnostics["node_dense_exact_fallback_count"] += 1
        in_dense_fallback = True
        grid = np.linspace(minimum_k, maximum_k, dense_grid_size, dtype=np.float64)
        strikes = fit.forward * np.exp(grid)
        ivs = np.asarray([_fit_iv(fit, float(value)) for value in grid], dtype=np.float64)
        valid = np.isfinite(ivs) & (ivs >= minimum_iv) & (ivs <= maximum_iv)
        deltas = np.full(len(grid), np.nan, dtype=np.float64)
        qualities = np.zeros(len(grid), dtype=np.uint32)
        reasons = np.full(len(grid), "none", dtype=object)
        indices = np.flatnonzero(valid)
        if fit.exercise_type.strip().upper() in {"A", "AMERICAN"} and len(indices):
            recovered = _routed_american_delta_points(
                fit,
                strikes[indices],
                ivs[indices],
                is_call=put_call == "CALL",
                context=context,
                solver_registry=solver_registry,
            )
            deltas[indices] = recovered.deltas
            qualities[indices] = recovered.quality_flags
            reasons[indices] = recovered.reasons
            american_delta_diagnostics.update(recovered.diagnostics)
            american_delta_diagnostics["node_exact_delta_evaluations"] += len(indices)
            american_delta_diagnostics["node_dense_exact_fallback_evaluations"] += len(indices)
        else:
            for index in indices:
                deltas[index] = cached_model_delta(float(grid[index]), put_call)
        bracket = find_bracket(grid, deltas, target)
        if bracket is None:
            return None
        dense_exact_bracket_found = True
        left, right, left_delta, right_delta = bracket
        root = interpolated_candidate(bracket, target)
        solved = cached_model_delta(root, put_call)
        # Dense fallback is a correctness path, so retain bounded bisection
        # after the legacy grid interpolation until the configured tolerance.
        for _ in range(int(_surface_setting(context, "node_max_iterations", 100))):
            if math.isfinite(solved) and abs(solved - target) <= tolerance:
                break
            if not math.isfinite(solved):
                root = (left + right) / 2.0
                solved = cached_model_delta(root, put_call)
            error = solved - target
            if (left_delta - target) * error <= 0.0:
                right, right_delta = root, solved
            else:
                left, left_delta = root, solved
            root = interpolated_candidate(
                (left, right, left_delta, right_delta), target
            )
            solved = cached_model_delta(root, put_call)
            american_delta_diagnostics["node_dense_exact_fallback_evaluations"] += 1
        if not math.isfinite(solved) or abs(solved - target) > tolerance:
            return None
        key = (put_call, round(float(root), 14))
        return (
            root,
            solved,
            int(qualities[np.nanargmin(np.abs(grid - root))])
            | delta_quality_cache.get(key, 0),
            delta_reason_cache.get(key, "none"),
        )

    target_states: list[dict[str, Any]] = []
    for delta_value in delta_values:
        for put_call in ("PUT", "CALL"):
            target = -delta_value if put_call == "PUT" else delta_value
            observed = sorted(
                (point.log_forward_moneyness, float(point.delta))
                for point in fit.points
                if point.put_call == put_call
                and point.delta is not None
                and math.isfinite(float(point.delta))
            )
            bracket = (
                find_bracket(
                    np.asarray([value[0] for value in observed]),
                    np.asarray([value[1] for value in observed]),
                    target,
                )
                if observed
                else None
            )
            if bracket is not None:
                american_delta_diagnostics["node_observed_delta_seed_count"] += 1
            else:
                bracket = find_bracket(
                    delta_search_grid, grid_deltas_by_side[put_call], target
                )
            target_states.append(
                {
                    "delta_value": delta_value,
                    "put_call": put_call,
                    "target": target,
                    "bracket": bracket,
                    "evaluations": 0,
                    "solved_result": None,
                    "used_dense_fallback": False,
                    "dense_bracket_found": False,
                }
            )

    # Each refinement iteration is one aligned numerical wave.  The search
    # state and the exact-evaluation budget remain separate for every node.
    initial_states = [state for state in target_states if state["bracket"] is not None]
    initial_points = [
        (interpolated_candidate(state["bracket"], state["target"]), state["put_call"])
        for state in initial_states
    ]
    for state, (root, _put_call), solved in zip(
        initial_states,
        initial_points,
        cached_model_deltas(initial_points) if initial_points else (),
    ):
        state["evaluations"] = 1
        state["root"] = root
        if math.isfinite(solved) and abs(solved - state["target"]) <= tolerance:
            key = (state["put_call"], round(float(root), 14))
            state["solved_result"] = (
                root,
                solved,
                delta_quality_cache.get(key, 0),
                delta_reason_cache.get(key, "none"),
            )

    endpoint_points: list[tuple[float, str]] = []
    endpoint_owners: list[tuple[dict[str, Any], str]] = []
    for state in initial_states:
        if state["solved_result"] is not None:
            continue
        left, right, _left_seed, _right_seed = state["bracket"]
        state["left"], state["right"] = left, right
        for side, value in (("left_delta", left), ("right_delta", right)):
            if state["evaluations"] >= exact_budget:
                state[side] = math.nan
                continue
            endpoint_points.append((value, state["put_call"]))
            endpoint_owners.append((state, side))
            state["evaluations"] += 1
    for (state, side), value in zip(
        endpoint_owners,
        cached_model_deltas(endpoint_points) if endpoint_points else (),
    ):
        state[side] = value

    while True:
        wave_states: list[dict[str, Any]] = []
        wave_points: list[tuple[float, str]] = []
        for state in initial_states:
            if state["solved_result"] is not None or state["evaluations"] >= exact_budget:
                continue
            left_delta = state.get("left_delta", math.nan)
            right_delta = state.get("right_delta", math.nan)
            target = state["target"]
            if not (
                math.isfinite(left_delta)
                and math.isfinite(right_delta)
                and (left_delta - target) * (right_delta - target) <= 0.0
            ):
                continue
            candidate = interpolated_candidate(
                (state["left"], state["right"], left_delta, right_delta), target
            )
            if candidate in {state["left"], state["right"]}:
                candidate = (state["left"] + state["right"]) / 2.0
            state["candidate"] = candidate
            state["evaluations"] += 1
            wave_states.append(state)
            wave_points.append((candidate, state["put_call"]))
            american_delta_diagnostics["node_exact_refinement_iterations"] += 1
        if not wave_states:
            break
        values = cached_model_deltas(wave_points)
        for state, candidate_delta in zip(wave_states, values):
            candidate = state["candidate"]
            target = state["target"]
            if not math.isfinite(candidate_delta):
                state["evaluations"] = exact_budget
                continue
            if abs(candidate_delta - target) <= tolerance:
                key = (state["put_call"], round(float(candidate), 14))
                state["solved_result"] = (
                    candidate,
                    candidate_delta,
                    delta_quality_cache.get(key, 0),
                    delta_reason_cache.get(key, "none"),
                )
            elif (state["left_delta"] - target) * (candidate_delta - target) <= 0.0:
                state["right"], state["right_delta"] = candidate, candidate_delta
            else:
                state["left"], state["left_delta"] = candidate, candidate_delta

    for state in initial_states:
        if state["solved_result"] is not None:
            american_delta_diagnostics["node_exact_refinement_successes"] += 1
        else:
            american_delta_diagnostics["node_exact_refinement_failures"] += 1

    for state in target_states:
        delta_value = state["delta_value"]
        put_call = state["put_call"]
        target = state["target"]
        bracket = state["bracket"]
        solved_result = state["solved_result"]
        if solved_result is None:
            dense_started = time.perf_counter()
            solved_result = dense_exact_fallback(put_call, target)
            state["dense_bracket_found"] = dense_exact_bracket_found
            in_dense_fallback = False
            if stage_timings is not None:
                stage_timings["surface.node_dense_fallback_seconds"] = (
                    stage_timings.get("surface.node_dense_fallback_seconds", 0.0)
                    + time.perf_counter()
                    - dense_started
                )
            state["used_dense_fallback"] = solved_result is not None
        if solved_result is None:
            dense_bracket_found = bool(state["dense_bracket_found"])
            nodes.append(
                unavailable_delta(
                    delta_value,
                    put_call,
                    (
                        "outside_coverage"
                        if bracket is None
                        or (dense_fallback_enabled and not dense_bracket_found)
                        else "failed"
                    ),
                    grid_quality | grid_failure_quality_by_side[put_call],
                    (
                        "target_delta_outside_coverage"
                        if bracket is None
                        or (dense_fallback_enabled and not dense_bracket_found)
                        else "delta_root_solver_failed"
                    ),
                )
            )
            continue
        root, solved_delta, point_quality, point_reason = solved_result
        strike = fit.forward * math.exp(float(root))
        iv = _fit_iv(fit, float(root))
        if iv is None or not minimum_iv <= iv <= maximum_iv:
            nodes.append(
                unavailable_delta(
                    delta_value,
                    put_call,
                    "failed",
                    grid_quality | point_quality,
                    "delta_root_residual_exceeded",
                )
            )
            continue
        nodes.append(
            SurfaceNode(
                "delta",
                delta_value,
                put_call,
                float(strike),
                float(solved_delta),
                None,
                float(iv),
                None,
                confidence,
                "complete",
                grid_quality
                | point_quality
                | (
                    QUALITY_EXACT_ROOT_FALLBACK
                    if state["used_dense_fallback"]
                    else 0
                ),
                point_reason,
            )
        )

    atm_iv = _fit_iv(fit, 0.0)
    nodes.append(
        SurfaceNode(
            "atm_forward",
            0.0,
            "NONE",
            float(fit.forward) if atm_iv is not None else None,
            None,
            None,
            atm_iv,
            None,
            confidence if atm_iv is not None else None,
            "complete" if atm_iv is not None else "outside_coverage",
            node_reason=("none" if atm_iv is not None else "outside_smile_coverage"),
        )
    )
    for k in log_nodes:
        iv = _fit_iv(fit, k)
        nodes.append(
            SurfaceNode(
                "log_forward_moneyness",
                k,
                "NONE",
                fit.forward * math.exp(k) if iv is not None else None,
                None,
                None,
                iv,
                None,
                confidence if iv is not None else None,
                "complete" if iv is not None else "outside_coverage",
                node_reason=(
                    "none" if iv is not None else "optional_node_outside_coverage"
                ),
            )
        )
    for node in nodes:
        node.assert_finite()
    if diagnostics is not None:
        for name, value in american_delta_diagnostics.items():
            diagnostics[name] = int(diagnostics.get(name, 0)) + int(value)
    return tuple(nodes)


@dataclass
class DeltaNodeValidationState:
    delta_value: float
    put_call: str
    target_delta: float
    observed_bracket: Optional[tuple[float, float, float, float]]
    tian_bracket: Optional[tuple[float, float, float, float]]
    primary_bracket: Optional[tuple[float, float, float, float]]
    bracket_source: str
    evaluations: int = 0
    correction_iterations: int = 0
    samples: dict[float, tuple[float, float, int, str]] = field(
        default_factory=dict
    )
    solved_result: Optional[tuple[float, float, int, str]] = None
    used_legacy_fallback: bool = False


def _generate_surface_nodes_hybrid(
    fit: SmileFit,
    *,
    context: Mapping[str, Any],
    dtype: np.dtype,
    config: SurfaceNodeSearchConfig,
    runtime: Optional[SurfaceNumericalRuntime] = None,
    diagnostics: Optional[MutableMapping[str, int]] = None,
    solver_registry: Optional[Any] = None,
    stage_timings: Optional[MutableMapping[str, float]] = None,
    exact_delta_cache: Optional[
        Mapping[tuple[str, float], tuple[float, int, str]]
    ] = None,
    search_delta_cache: Optional[
        MutableMapping[tuple[Any, ...], DeltaSearchGrid]
    ] = None,
    defer_exact_delta_waves: bool = False,
    precomputed_exact_diagnostics: Optional[Mapping[str, int]] = None,
) -> tuple[SurfaceNode, ...]:
    """Tian proposal search followed by strictly bounded exact validation."""
    if not fit.successful:
        return ()
    minimum_k, maximum_k = fit.coverage
    confidence = _confidence(fit)
    delta_tolerance = float(
        _surface_setting(context, "node_delta_tolerance", 1.0e-5)
    )
    if not math.isfinite(delta_tolerance) or delta_tolerance <= 0.0:
        raise ValueError(
            "options_calcs.surface.node_delta_tolerance must be positive and finite"
        )
    minimum_iv = float(_iv_setting(context, "minimum", 1.0e-4))
    maximum_iv = float(_iv_setting(context, "maximum", 10.0))
    delta_values = _surface_setting(context, "delta_nodes", [0.40, 0.25, 0.10])
    if isinstance(delta_values, str):
        delta_values = [
            float(value.strip())
            for value in delta_values.split(",")
            if value.strip()
        ]
    delta_values = [float(value) for value in delta_values]
    log_nodes = _surface_setting(
        context,
        "log_forward_moneyness_nodes",
        [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20],
    )
    if isinstance(log_nodes, str):
        log_nodes = [
            float(value.strip())
            for value in log_nodes.split(",")
            if value.strip()
        ]
    log_nodes = [float(value) for value in log_nodes]
    nodes: list[SurfaceNode] = []
    counters: Counter[str] = Counter(precomputed_exact_diagnostics or {})
    required_metric_names = (
        "surface.node_tian_search_evaluations",
        "surface.node_tian_bracket_hits",
        "surface.node_tian_bracket_misses",
        "surface.node_validation_evaluations",
        "surface.node_validation_initial_successes",
        "surface.node_validation_correction_successes",
        "surface.node_validation_failures",
        "surface.node_validation_budget_exhaustions",
        "surface.node_legacy_fallback_count",
        "surface.node_legacy_fallback_evaluations",
        "surface.node_dense_exact_fallback_evaluations",
    )
    for name in required_metric_names:
        counters[name] += 0

    def unavailable_delta(
        delta_value: float,
        put_call: str,
        status: str,
        quality_flags: int = 0,
        node_reason: str = "nonfinite_direct_crr_delta",
    ) -> SurfaceNode:
        return SurfaceNode(
            "delta", delta_value, put_call, None, None, None, None, None,
            None, status, quality_flags, node_reason
        )

    if minimum_k is None or maximum_k is None:
        nodes.extend(
            unavailable_delta(
                value, side, "unavailable", node_reason="outside_smile_coverage"
            )
            for value in delta_values
            for side in ("PUT", "CALL")
        )
        nodes.append(
            SurfaceNode(
                "atm_forward", 0.0, "NONE", None, None, None, None, None,
                None, "unavailable", node_reason="outside_smile_coverage"
            )
        )
        nodes.extend(
            SurfaceNode(
                "log_forward_moneyness", value, "NONE", None, None, None,
                None, None, None, "unavailable",
                node_reason="optional_node_outside_coverage",
            )
            for value in log_nodes
        )
        return tuple(nodes)

    search_started = time.perf_counter()
    search_grid, search_built, search_diagnostics = (
        _build_tian_delta_search_grid(
            fit,
            context=context,
            config=config,
            solver_registry=solver_registry,
            minimum_iv=minimum_iv,
            maximum_iv=maximum_iv,
            cache=search_delta_cache,
        )
    )
    if search_built:
        counters.update(search_diagnostics)
        elapsed = time.perf_counter() - search_started
        if stage_timings is not None:
            stage_timings["surface.node_tian_search_seconds"] = (
                stage_timings.get("surface.node_tian_search_seconds", 0.0)
                + elapsed
            )
            # Compatibility timing for existing dashboards.
            stage_timings["surface.node_cheap_search_seconds"] = (
                stage_timings.get("surface.node_cheap_search_seconds", 0.0)
                + elapsed
            )
    validation_method = _surface_validation_method(fit, context)
    counters[f"surface.node_validation_method_{validation_method.value}"] = 1

    delta_cache: dict[tuple[str, float], float] = {
        key: float(value[0]) for key, value in (exact_delta_cache or {}).items()
    }
    delta_quality_cache: dict[tuple[str, float], int] = {
        key: int(value[1]) for key, value in (exact_delta_cache or {}).items()
    }
    delta_reason_cache: dict[tuple[str, float], str] = {
        key: str(value[2]) for key, value in (exact_delta_cache or {}).items()
    }
    in_legacy_fallback = False

    def cached_authoritative_deltas(
        points: Sequence[tuple[float, str]],
    ) -> list[float]:
        missing: dict[tuple[str, float], tuple[float, str]] = {}
        for log_strike, put_call in points:
            key = (put_call, round(float(log_strike), 14))
            if key not in delta_cache:
                missing.setdefault(key, (float(log_strike), put_call))
        if missing:
            valid_by_side: dict[
                str, list[tuple[tuple[str, float], float, float]]
            ] = {"PUT": [], "CALL": []}
            for key, (log_strike, put_call) in missing.items():
                strike = fit.forward * math.exp(log_strike)
                iv = _fit_iv(fit, log_strike)
                if (
                    iv is None
                    or not math.isfinite(iv)
                    or not minimum_iv <= iv <= maximum_iv
                ):
                    delta_cache[key] = math.nan
                    delta_quality_cache[key] = QUALITY_INVALID_FITTED_IV_GRID
                    delta_reason_cache[key] = "outside_smile_coverage"
                else:
                    valid_by_side[put_call].append((key, strike, float(iv)))
            deferred_points = tuple(
                SurfaceExactDeltaPoint(
                    cache_key=key,
                    log_strike=missing[key][0],
                    put_call=put_call,
                    strike=strike,
                    volatility=volatility,
                )
                for put_call in ("PUT", "CALL")
                for key, strike, volatility in valid_by_side[put_call]
            )
            if defer_exact_delta_waves and deferred_points:
                if in_legacy_fallback:
                    counters[
                        "surface.node_legacy_fallback_evaluations"
                    ] += len(deferred_points)
                raise PendingSurfaceExactDeltaWave(
                    fit,
                    deferred_points,
                    dense_fallback=False,
                    diagnostics=counters,
                    timings=stage_timings,
                )
            validation_started = time.perf_counter()
            if deferred_points:
                from algotrading.refdata.options.solvers.Registry import SolverRegistry

                registry = solver_registry or SolverRegistry(context)
                recovered = _routed_american_delta_waves(
                    ((fit, deferred_points),),
                    context=context,
                    solver_registry=registry,
                )[0]
                counters.update(recovered.diagnostics)
                for point, value, quality, reason in zip(
                    deferred_points,
                    recovered.deltas,
                    recovered.quality_flags,
                    recovered.reasons,
                ):
                    delta_cache[point.cache_key] = float(value)
                    delta_quality_cache[point.cache_key] = int(quality)
                    delta_reason_cache[point.cache_key] = str(reason)
            evaluated = len(deferred_points)
            counters["surface.node_validation_evaluations"] += evaluated
            counters["node_exact_delta_evaluations"] += evaluated
            if in_legacy_fallback:
                counters["surface.node_legacy_fallback_evaluations"] += evaluated
            elapsed = time.perf_counter() - validation_started
            if stage_timings is not None:
                stage_timings["surface.node_validation_seconds"] = (
                    stage_timings.get("surface.node_validation_seconds", 0.0)
                    + elapsed
                )
                stage_timings["surface.node_exact_refinement_seconds"] = (
                    stage_timings.get(
                        "surface.node_exact_refinement_seconds", 0.0
                    )
                    + elapsed
                )
        return [
            delta_cache[(put_call, round(float(log_strike), 14))]
            for log_strike, put_call in points
        ]

    def bracket_for_side(
        put_call: str, target: float
    ) -> Optional[tuple[float, float, float, float]]:
        deltas = (
            search_grid.call_deltas
            if put_call == "CALL"
            else search_grid.put_deltas
        )
        return _find_delta_bracket(search_grid.log_moneyness, deltas, target)

    states: list[DeltaNodeValidationState] = []
    for delta_value in delta_values:
        for put_call in ("PUT", "CALL"):
            target = delta_value if put_call == "CALL" else -delta_value
            observed = sorted(
                (point.log_forward_moneyness, float(point.delta))
                for point in fit.points
                if point.put_call == put_call
                and point.delta is not None
                and math.isfinite(float(point.delta))
            )
            observed_bracket = (
                _find_delta_bracket(
                    [value[0] for value in observed],
                    [value[1] for value in observed],
                    target,
                )
                if observed
                else None
            )
            tian_bracket = bracket_for_side(put_call, target)
            primary_bracket = observed_bracket or tian_bracket
            states.append(
                DeltaNodeValidationState(
                    delta_value=delta_value,
                    put_call=put_call,
                    target_delta=target,
                    observed_bracket=observed_bracket,
                    tian_bracket=tian_bracket,
                    primary_bracket=primary_bracket,
                    bracket_source=(
                        "observed"
                        if observed_bracket is not None
                        else "tian" if tian_bracket is not None else "none"
                    ),
                )
            )

    def candidate_key(value: float) -> float:
        return round(float(value), 14)

    def initial_candidates(state: DeltaNodeValidationState) -> list[float]:
        bracket = state.primary_bracket
        if bracket is None:
            return []
        candidates: list[float] = []

        def add(value: float) -> None:
            value = min(max(float(value), float(minimum_k)), float(maximum_k))
            if math.isfinite(value) and candidate_key(value) not in {
                candidate_key(existing) for existing in candidates
            }:
                candidates.append(value)

        add(_interpolated_delta_candidate(bracket, state.target_delta))
        if state.tian_bracket is not None:
            add(
                _interpolated_delta_candidate(
                    state.tian_bracket, state.target_delta
                )
            )
        left, right = bracket[:2]
        add((left + right) / 2.0)
        for fraction in (0.25, 0.75, 0.125, 0.875):
            add(left + fraction * (right - left))
        return candidates[: config.validation_initial_candidates]

    def record_sample(
        state: DeltaNodeValidationState, log_strike: float, value: float
    ) -> None:
        key = (state.put_call, candidate_key(log_strike))
        quality = delta_quality_cache.get(key, 0)
        reason = delta_reason_cache.get(key, "solver_delta_failed")
        state.samples[candidate_key(log_strike)] = (
            float(log_strike), float(value), int(quality), str(reason)
        )
        if (
            state.solved_result is None
            and math.isfinite(value)
            and abs(value - state.target_delta) <= delta_tolerance
        ):
            state.solved_result = (
                float(log_strike), float(value), int(quality), str(reason)
            )

    initial_owners: list[tuple[DeltaNodeValidationState, float]] = []
    initial_points: list[tuple[float, str]] = []
    for state in states:
        candidates = initial_candidates(state)
        state.evaluations += len(candidates)
        for candidate in candidates:
            initial_owners.append((state, candidate))
            initial_points.append((candidate, state.put_call))
    initial_values = (
        cached_authoritative_deltas(initial_points) if initial_points else ()
    )
    for (state, candidate), value in zip(initial_owners, initial_values):
        record_sample(state, candidate, value)
    initial_success_ids = {
        id(state) for state in states if state.solved_result is not None
    }

    def authoritative_bracket(
        state: DeltaNodeValidationState,
    ) -> Optional[tuple[float, float, float, float]]:
        finite = sorted(
            (sample[0], sample[1])
            for sample in state.samples.values()
            if math.isfinite(sample[1])
        )
        return (
            _find_delta_bracket(
                [value[0] for value in finite],
                [value[1] for value in finite],
                state.target_delta,
            )
            if finite
            else None
        )

    def unused(value: float, state: DeltaNodeValidationState) -> bool:
        return candidate_key(value) not in state.samples

    def endpoint_probe(
        state: DeltaNodeValidationState,
    ) -> Optional[float]:
        bracket = state.tian_bracket or state.primary_bracket
        if bracket is None:
            return None
        candidates = [
            (bracket[0], bracket[2]),
            (bracket[1], bracket[3]),
        ]
        candidates = [item for item in candidates if unused(item[0], state)]
        if not candidates:
            return None
        finite_samples = [
            sample for sample in state.samples.values() if math.isfinite(sample[1])
        ]
        if finite_samples:
            best = min(
                finite_samples,
                key=lambda sample: abs(sample[1] - state.target_delta),
            )
            best_error = best[1] - state.target_delta
            opposite = [
                item
                for item in candidates
                if (item[1] - state.target_delta) * best_error <= 0.0
            ]
            if opposite:
                candidates = opposite
        return float(
            min(candidates, key=lambda item: abs(item[1] - state.target_delta))[0]
        )

    def narrow_fallback_probe(
        state: DeltaNodeValidationState,
    ) -> Optional[float]:
        if not config.legacy_fallback_enabled or state.tian_bracket is None:
            return None
        left, right = state.tian_bracket[:2]
        grid = search_grid.log_moneyness
        left_index = int(np.argmin(np.abs(grid - left)))
        right_index = int(np.argmin(np.abs(grid - right)))
        indices = (
            left_index - 1,
            right_index + 1,
            (left_index + right_index) // 2,
        )
        side_deltas = (
            search_grid.call_deltas
            if state.put_call == "CALL"
            else search_grid.put_deltas
        )
        candidates = [
            (float(grid[index]), float(side_deltas[index]))
            for index in indices
            if 0 <= index < len(grid)
            and bool(search_grid.valid[index])
            and math.isfinite(float(side_deltas[index]))
            and unused(float(grid[index]), state)
        ]
        if not candidates:
            return None
        finite_samples = [
            sample for sample in state.samples.values() if math.isfinite(sample[1])
        ]
        if finite_samples:
            best = min(
                finite_samples,
                key=lambda sample: abs(sample[1] - state.target_delta),
            )
            best_error = best[1] - state.target_delta
            opposite = [
                item
                for item in candidates
                if (item[1] - state.target_delta) * best_error <= 0.0
            ]
            if opposite:
                candidates = opposite
        state.used_legacy_fallback = True
        return min(
            candidates,
            key=lambda item: (
                abs(item[1] - state.target_delta),
                min(abs(item[0] - left), abs(item[0] - right)),
            ),
        )[0]

    while True:
        wave_owners: list[tuple[DeltaNodeValidationState, float, str]] = []
        wave_points: list[tuple[float, str]] = []
        for state in states:
            if (
                state.solved_result is not None
                or state.primary_bracket is None
                or state.evaluations >= config.validation_max_evaluations
            ):
                continue
            exact_bracket = authoritative_bracket(state)
            action = "correction"
            if exact_bracket is not None:
                candidate = _interpolated_delta_candidate(
                    exact_bracket, state.target_delta, safeguard=True
                )
                if not unused(candidate, state):
                    candidate = (exact_bracket[0] + exact_bracket[1]) / 2.0
                if not unused(candidate, state):
                    candidate = None
            else:
                candidate = endpoint_probe(state)
                action = "endpoint"
                finite_sample_count = sum(
                    math.isfinite(sample[1]) for sample in state.samples.values()
                )
                if finite_sample_count >= 2:
                    fallback_candidate = narrow_fallback_probe(state)
                    if fallback_candidate is not None:
                        candidate = fallback_candidate
                        action = "fallback"
            if candidate is None:
                candidate = narrow_fallback_probe(state)
                action = "fallback"
            if candidate is None:
                continue
            state.evaluations += 1
            if action == "correction":
                state.correction_iterations += 1
            wave_owners.append((state, candidate, action))
            wave_points.append((candidate, state.put_call))
        if not wave_points:
            break
        in_legacy_fallback = any(action == "fallback" for _, _, action in wave_owners)
        values = cached_authoritative_deltas(wave_points)
        in_legacy_fallback = False
        for (state, candidate, _action), value in zip(wave_owners, values):
            record_sample(state, candidate, value)

    counters["surface.node_tian_bracket_hits"] = sum(
        state.tian_bracket is not None for state in states
    )
    counters["surface.node_tian_bracket_misses"] = sum(
        state.tian_bracket is None for state in states
    )
    counters["surface.node_observed_bracket_count"] = sum(
        state.bracket_source == "observed" for state in states
    )
    counters["surface.node_tian_bracket_selected_count"] = sum(
        state.bracket_source == "tian" for state in states
    )
    counters["surface.node_validation_initial_successes"] = len(
        initial_success_ids
    )
    counters["surface.node_validation_correction_successes"] = sum(
        state.solved_result is not None and id(state) not in initial_success_ids
        for state in states
    )
    counters["surface.node_validation_failures"] = sum(
        state.primary_bracket is not None and state.solved_result is None
        for state in states
    )
    counters["surface.node_validation_budget_exhaustions"] = sum(
        state.primary_bracket is not None
        and state.solved_result is None
        and state.evaluations >= config.validation_max_evaluations
        for state in states
    )
    counters["surface.node_validation_correction_iterations"] = sum(
        state.correction_iterations for state in states
    )
    counters["surface.node_legacy_fallback_count"] = sum(
        state.used_legacy_fallback for state in states
    )
    counters["surface.node_dense_exact_fallback_evaluations"] = 0
    # Compatibility aliases retained while dashboards migrate.
    counters["node_dense_exact_fallback_evaluations"] = 0
    counters["node_dense_exact_fallback_count"] = 0
    counters["node_exact_refinement_successes"] = sum(
        state.solved_result is not None for state in states
    )
    counters["node_exact_refinement_failures"] = counters[
        "surface.node_validation_failures"
    ]

    def search_failure(
        state: DeltaNodeValidationState,
    ) -> tuple[bool, int, Optional[str]]:
        deltas = (
            search_grid.call_deltas
            if state.put_call == "CALL"
            else search_grid.put_deltas
        )
        qualities = (
            search_grid.call_quality_flags
            if state.put_call == "CALL"
            else search_grid.put_quality_flags
        )
        reasons = (
            search_grid.call_reasons
            if state.put_call == "CALL"
            else search_grid.put_reasons
        )
        valid_indices = np.flatnonzero(search_grid.valid)
        finite_count = int(np.count_nonzero(np.isfinite(deltas[valid_indices])))
        failed = len(valid_indices) >= 2 and finite_count < 2
        quality = int(
            np.bitwise_or.reduce(qualities[valid_indices], initial=np.uint32(0))
        )
        reason = next(
            (
                str(reasons[index])
                for index in valid_indices
                if str(reasons[index]) in SURFACE_NODE_REASONS
                and str(reasons[index]) not in {"none", "outside_smile_coverage"}
            ),
            None,
        )
        return failed, quality, reason

    for state in states:
        if state.solved_result is None:
            has_bracket = state.primary_bracket is not None
            point_reasons = [sample[3] for sample in state.samples.values()]
            actionable_reason = next(
                (
                    reason
                    for reason in reversed(point_reasons)
                    if reason in SURFACE_NODE_REASONS and reason != "none"
                ),
                None,
            )
            tian_failed, tian_quality, tian_reason = search_failure(state)
            nodes.append(
                unavailable_delta(
                    state.delta_value,
                    state.put_call,
                    "failed" if has_bracket or tian_failed else "outside_coverage",
                    search_grid.quality_flags | tian_quality,
                    (
                        actionable_reason
                        if has_bracket and actionable_reason is not None
                        else (
                            "delta_root_solver_failed"
                            if has_bracket
                            else (
                                tian_reason or "solver_delta_failed"
                                if tian_failed
                                else "target_delta_outside_coverage"
                            )
                        )
                    ),
                )
            )
            continue
        root, solved_delta, quality, reason = state.solved_result
        strike = fit.forward * math.exp(root)
        iv = _fit_iv(fit, root)
        if (
            iv is None
            or not math.isfinite(strike)
            or strike <= 0.0
            or not minimum_iv <= iv <= maximum_iv
            or abs(solved_delta - state.target_delta) > delta_tolerance
        ):
            nodes.append(
                unavailable_delta(
                    state.delta_value,
                    state.put_call,
                    "failed",
                    quality,
                    "delta_root_residual_exceeded",
                )
            )
            continue
        nodes.append(
            SurfaceNode(
                "delta",
                state.delta_value,
                state.put_call,
                float(strike),
                float(solved_delta),
                None,
                float(iv),
                None,
                confidence,
                "complete",
                int(quality)
                | (
                    QUALITY_EXACT_ROOT_FALLBACK
                    if state.used_legacy_fallback
                    else 0
                ),
                reason,
            )
        )

    atm_iv = _fit_iv(fit, 0.0)
    nodes.append(
        SurfaceNode(
            "atm_forward",
            0.0,
            "NONE",
            float(fit.forward) if atm_iv is not None else None,
            None,
            None,
            atm_iv,
            None,
            confidence if atm_iv is not None else None,
            "complete" if atm_iv is not None else "outside_coverage",
            node_reason=("none" if atm_iv is not None else "outside_smile_coverage"),
        )
    )
    for value in log_nodes:
        iv = _fit_iv(fit, value)
        nodes.append(
            SurfaceNode(
                "log_forward_moneyness",
                value,
                "NONE",
                fit.forward * math.exp(value) if iv is not None else None,
                None,
                None,
                iv,
                None,
                confidence if iv is not None else None,
                "complete" if iv is not None else "outside_coverage",
                node_reason=(
                    "none" if iv is not None else "optional_node_outside_coverage"
                ),
            )
        )
    for node in nodes:
        node.assert_finite()
    if diagnostics is not None:
        for name, value in counters.items():
            diagnostics[name] = int(diagnostics.get(name, 0)) + int(value)
    return tuple(nodes)


def generate_surface_nodes(
    fit: SmileFit,
    *,
    context: Mapping[str, Any],
    dtype: np.dtype,
    runtime: Optional[SurfaceNumericalRuntime] = None,
    diagnostics: Optional[MutableMapping[str, int]] = None,
    solver_registry: Optional[Any] = None,
    stage_timings: Optional[MutableMapping[str, float]] = None,
    exact_delta_cache: Optional[
        Mapping[tuple[str, float], tuple[float, int, str]]
    ] = None,
    search_delta_cache: Optional[
        MutableMapping[tuple[Any, ...], DeltaSearchGrid]
    ] = None,
    defer_exact_delta_waves: bool = False,
    precomputed_exact_diagnostics: Optional[Mapping[str, int]] = None,
) -> tuple[SurfaceNode, ...]:
    """Generate nodes using hybrid validation for American fits by default."""
    config = SurfaceNodeSearchConfig.from_context(context)
    if (
        fit.exercise_type.strip().upper() in {"A", "AMERICAN"}
        and config.hybrid_enabled
    ):
        return _generate_surface_nodes_hybrid(
            fit,
            context=context,
            dtype=dtype,
            config=config,
            runtime=runtime,
            diagnostics=diagnostics,
            solver_registry=solver_registry,
            stage_timings=stage_timings,
            exact_delta_cache=exact_delta_cache,
            search_delta_cache=search_delta_cache,
            defer_exact_delta_waves=defer_exact_delta_waves,
            precomputed_exact_diagnostics=precomputed_exact_diagnostics,
        )
    return _generate_surface_nodes_legacy(
        fit,
        context=context,
        dtype=dtype,
        runtime=runtime,
        diagnostics=diagnostics,
        solver_registry=solver_registry,
        stage_timings=stage_timings,
        exact_delta_cache=exact_delta_cache,
        defer_exact_delta_waves=defer_exact_delta_waves,
        precomputed_exact_diagnostics=precomputed_exact_diagnostics,
    )


def _terminal_semantic_nodes(
    context: Mapping[str, Any],
    *,
    status: str,
    quality_flags: int = 0,
    node_reason: str = "delta_root_solver_failed",
) -> tuple[SurfaceNode, ...]:
    delta_values = _surface_setting(context, "delta_nodes", [0.40, 0.25, 0.10])
    if isinstance(delta_values, str):
        delta_values = [
            float(value.strip())
            for value in delta_values.split(",")
            if value.strip()
        ]
    log_values = _surface_setting(
        context,
        "log_forward_moneyness_nodes",
        [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20],
    )
    if isinstance(log_values, str):
        log_values = [
            float(value.strip())
            for value in log_values.split(",")
            if value.strip()
        ]
    nodes = [
        SurfaceNode(
            "delta", float(delta), put_call, None, None, None, None, None,
            None, status, quality_flags, node_reason
        )
        for delta in delta_values
        for put_call in ("PUT", "CALL")
    ]
    nodes.append(
        SurfaceNode(
            "atm_forward", 0.0, "NONE", None, None, None, None, None,
            None, status, quality_flags, node_reason
        )
    )
    nodes.extend(
        SurfaceNode(
            "log_forward_moneyness", float(value), "NONE", None, None, None,
            None, None, None, status, quality_flags, node_reason
        )
        for value in log_values
    )
    return tuple(nodes)


def _classify_node_generation(nodes: Sequence[SurfaceNode]) -> str:
    if nodes and all(node.status == "complete" for node in nodes):
        return "complete"
    if any(node.status == "complete" for node in nodes):
        return "partial"
    return "unavailable"


REQUIRED_DELTA_NODE_VALUES = frozenset({0.10, 0.25, 0.40})


def is_required_surface_node(node_type: str, node_value: float) -> bool:
    """Central required-node definition used by persistence and health."""
    if str(node_type) == "atm_forward":
        return True
    return str(node_type) == "delta" and any(
        math.isclose(float(node_value), value, abs_tol=1.0e-12, rel_tol=0.0)
        for value in REQUIRED_DELTA_NODE_VALUES
    )


def _classify_node_subset(nodes: Sequence[SurfaceNode]) -> str:
    if not nodes:
        return "not_requested"
    if all(node.status == "complete" for node in nodes):
        return "complete"
    if any(node.status == "complete" for node in nodes):
        return "partial"
    if any(node.status == "failed" for node in nodes):
        return "failed"
    return "unavailable"


def classify_required_optional_nodes(
    nodes: Sequence[SurfaceNode],
) -> tuple[str, str]:
    required = [
        node
        for node in nodes
        if is_required_surface_node(node.node_type, node.node_value)
    ]
    optional = [
        node
        for node in nodes
        if not is_required_surface_node(node.node_type, node.node_value)
    ]
    required_status = _classify_node_subset(required)
    required_delta_nodes = [
        node for node in required if node.node_type == "delta"
    ]
    if required_delta_nodes and not any(
        node.status == "complete" for node in required_delta_nodes
    ):
        # ATM can remain usable even when the complete required delta semantic
        # set is not.  Preserve that distinction instead of reporting a
        # misleading partial result solely because ATM completed.
        required_status = (
            "failed"
            if any(node.status == "failed" for node in required_delta_nodes)
            else "unavailable"
        )
    return required_status, _classify_node_subset(optional)


def generate_calibration_nodes(
    calibration: ExpirationCalibration,
    *,
    context: Mapping[str, Any],
    runtime: Optional[SurfaceNumericalRuntime] = None,
    diagnostics: Optional[MutableMapping[str, int]] = None,
    solver_registry: Optional[Any] = None,
    stage_timings: Optional[MutableMapping[str, float]] = None,
    exact_delta_cache: Optional[
        Mapping[tuple[str, float], tuple[float, int, str]]
    ] = None,
    search_delta_cache: Optional[
        MutableMapping[tuple[Any, ...], DeltaSearchGrid]
    ] = None,
    defer_exact_delta_waves: bool = False,
    precomputed_exact_diagnostics: Optional[Mapping[str, int]] = None,
) -> ExpirationCalibration:
    """Generate/retry nodes without allowing node failures to erase the fit."""
    fit = calibration.fit
    if not fit.successful:
        return replace(
            calibration, node_result=NodeGenerationResult.not_requested()
        )
    requested_dtype = fit.actual_dtype
    if requested_dtype not in SUPPORTED_DTYPES:
        requested_dtype = fit.requested_dtype
    if requested_dtype not in SUPPORTED_DTYPES:
        requested_dtype = "float64"
    policy = NumericPrecisionPolicy.from_context(context)

    def attempt(dtype_name: str) -> tuple[SurfaceNode, ...]:
        return generate_surface_nodes(
            fit,
            context=context,
            dtype=np.dtype(dtype_name),
            runtime=runtime,
            diagnostics=diagnostics,
            solver_registry=solver_registry,
            stage_timings=stage_timings,
            exact_delta_cache=exact_delta_cache,
            search_delta_cache=search_delta_cache,
            defer_exact_delta_waves=defer_exact_delta_waves,
            precomputed_exact_diagnostics=precomputed_exact_diagnostics,
        )

    first_error: Optional[Exception] = None
    try:
        nodes = attempt(requested_dtype)
        retry_reason = ""
        # Point-local American recovery already performs its permitted
        # float64 retry.  Do not rerun an otherwise terminal full grid merely
        # because a recovered node retains the primary-path quality bit.
        retry_needed = (
            fit.exercise_type.strip().upper() not in {"A", "AMERICAN"}
            and any(node.status == "failed" for node in nodes)
        )
    except PendingSurfaceExactDeltaWave:
        raise
    except Exception as exc:
        first_error = exc
        nodes = ()
        retry_reason = (
            exc.reason if isinstance(exc, NumericalInstability) else type(exc).__name__
        )
        retry_needed = True

    escalated = (
        requested_dtype == "float32"
        and policy.auto_promote_to_float64
        and retry_needed
    )
    actual_dtype = requested_dtype
    error = ""
    if escalated:
        actual_dtype = "float64"
        if not retry_reason:
            retry_reason = (
                "node_local_numerical_failure"
                if any(node.status == "failed" for node in nodes)
                else "node_grid_numerical_failure"
            )
        try:
            nodes = attempt("float64")
            first_error = None
        except PendingSurfaceExactDeltaWave:
            raise
        except Exception as exc:
            first_error = exc
            error = str(exc)

    if first_error is not None:
        error = error or str(first_error)
        nodes = _terminal_semantic_nodes(
            context,
            status="failed",
            quality_flags=(
                QUALITY_NODE_PRECISION_ESCALATED if escalated else 0
            ),
        )
        result = NodeGenerationResult(
            nodes,
            "failed",
            error,
            requested_dtype,
            actual_dtype,
            escalated,
            retry_reason,
            *classify_required_optional_nodes(nodes),
        )
        result.assert_terminal()
        return replace(calibration, node_result=result)

    status = _classify_node_generation(nodes)
    required_status, optional_status = classify_required_optional_nodes(nodes)
    common_quality = 0
    if escalated:
        common_quality |= QUALITY_NODE_PRECISION_ESCALATED
    if status in {"partial", "unavailable"}:
        common_quality |= QUALITY_PARTIAL_NODE_SET
    if common_quality:
        nodes = tuple(
            replace(node, quality_flags=node.quality_flags | common_quality)
            for node in nodes
        )
    result = NodeGenerationResult(
        tuple(nodes),
        status,
        "",
        requested_dtype,
        actual_dtype,
        escalated,
        retry_reason if escalated else "",
        required_status,
        optional_status,
    )
    result.assert_terminal()
    return replace(calibration, node_result=result)


def generate_calibration_nodes_many(
    calibrations: Sequence[ExpirationCalibration],
    *,
    context: Mapping[str, Any],
    solver_registry: Any,
    runtimes: Optional[Sequence[Optional[SurfaceNumericalRuntime]]] = None,
    diagnostics: Optional[Sequence[MutableMapping[str, int]]] = None,
    stage_timings: Optional[Sequence[MutableMapping[str, float]]] = None,
) -> list[ExpirationCalibration]:
    """Advance all owners' exact node searches through explicit aligned waves.

    Search code remains the single owner-local authority.  Cooperative
    suspension occurs only when a state reaches its next missing exact-delta
    collection; completed cache entries let it resume deterministically at
    the same bracket, evaluation count, and dense-fallback order.
    """
    count = len(calibrations)
    runtime_values = list(runtimes or [None] * count)
    diagnostic_values = list(diagnostics or [Counter() for _ in range(count)])
    timing_values = list(stage_timings or [{} for _ in range(count)])
    if not (
        len(runtime_values) == len(diagnostic_values) == len(timing_values) == count
    ):
        raise ValueError("surface node owner metadata must align with calibrations")
    caches: list[dict[tuple[str, float], tuple[float, int, str]]] = [
        {} for _ in range(count)
    ]
    search_caches: list[dict[tuple[Any, ...], DeltaSearchGrid]] = [
        {} for _ in range(count)
    ]
    precomputed: list[Counter[str]] = [Counter() for _ in range(count)]
    results: list[Optional[ExpirationCalibration]] = [None] * count
    pending = list(range(count))
    while pending:
        wave_indices: list[int] = []
        waves: list[tuple[SmileFit, Sequence[SurfaceExactDeltaPoint]]] = []
        wave_dense_fallback: list[bool] = []
        completed: list[int] = []
        for index in pending:
            local_diagnostics: Counter[str] = Counter()
            local_timings: dict[str, float] = {}
            try:
                results[index] = generate_calibration_nodes(
                    calibrations[index],
                    context=context,
                    runtime=runtime_values[index],
                    diagnostics=local_diagnostics,
                    solver_registry=solver_registry,
                    stage_timings=local_timings,
                    exact_delta_cache=caches[index],
                    search_delta_cache=search_caches[index],
                    defer_exact_delta_waves=True,
                    precomputed_exact_diagnostics=precomputed[index],
                )
            except PendingSurfaceExactDeltaWave as request:
                if request.diagnostics:
                    precomputed[index] = Counter(request.diagnostics)
                for name, value in request.timings.items():
                    timing_values[index][name] = (
                        timing_values[index].get(name, 0.0) + float(value)
                    )
                wave_indices.append(index)
                waves.append((request.fit, request.points))
                wave_dense_fallback.append(request.dense_fallback)
            else:
                completed.append(index)
                diagnostic_values[index].update(local_diagnostics)
                for name, value in local_timings.items():
                    timing_values[index][name] = (
                        timing_values[index].get(name, 0.0) + value
                    )
        if waves:
            numerical_started = time.perf_counter()
            wave_results = _routed_american_delta_waves(
                waves, context=context, solver_registry=solver_registry
            )
            numerical_elapsed = time.perf_counter() - numerical_started
            total_points = sum(len(points) for _fit, points in waves)
            for index, (_fit, points), dense_fallback, recovered in zip(
                wave_indices, waves, wave_dense_fallback, wave_results
            ):
                timing_name = (
                    "surface.node_dense_fallback_seconds"
                    if dense_fallback
                    else "surface.node_exact_refinement_seconds"
                )
                timing_values[index][timing_name] = timing_values[index].get(
                    timing_name, 0.0
                ) + numerical_elapsed * len(points) / max(1, total_points)
                timing_values[index]["surface.node_validation_seconds"] = (
                    timing_values[index].get(
                        "surface.node_validation_seconds", 0.0
                    )
                    + numerical_elapsed * len(points) / max(1, total_points)
                )
                precomputed[index]["node_exact_delta_evaluations"] += len(points)
                precomputed[index]["surface.node_validation_evaluations"] += len(
                    points
                )
                if dense_fallback:
                    precomputed[index][
                        "surface.node_legacy_fallback_evaluations"
                    ] += len(points)
                precomputed[index].update(recovered.diagnostics)
                for point, value, quality, reason in zip(
                    points,
                    recovered.deltas,
                    recovered.quality_flags,
                    recovered.reasons,
                ):
                    caches[index][point.cache_key] = (
                        float(value),
                        int(quality),
                        str(reason),
                    )
        if not completed and not waves:
            raise AssertionError("surface node wave coordinator made no progress")
        pending = [index for index in pending if index not in completed]
    if any(result is None for result in results):
        raise AssertionError("surface node wave fan-out left an unowned calibration")
    return [result for result in results if result is not None]


def _calibrate_prepared_operation(
    rows: Sequence[Mapping[str, Any]],
    prepared: PreparedExpirationInputs,
    *,
    price_basis: str,
    context: Mapping[str, Any],
    dtype: np.dtype,
    runtime: SurfaceNumericalRuntime,
    american_inversions: Optional[Mapping[int, IVInversionResult]] = None,
    diagnostics: Optional[IVInversionDiagnostics] = None,
    warm_starts: Sequence[tuple[str, Mapping[str, Any]]] = (),
    performance_timings: Optional[MutableMapping[str, float]] = None,
    solver_registry: Optional[Any] = None,
) -> ExpirationCalibration:
    stage_started = time.perf_counter()
    try:
        selection = select_surface_inputs(
            rows,
            price_basis=price_basis,
            context=context,
            dtype=dtype,
            prepared=prepared,
            american_inversions=american_inversions,
            runtime=runtime,
            diagnostics=diagnostics,
            solver_registry=solver_registry,
        )
    except SurfaceStageNumericalInstability:
        raise
    except NumericalInstability as exc:
        raise SurfaceStageNumericalInstability(
            "iv_inversion", exc.reason
        ) from exc
    except SurfaceStageError:
        raise
    except Exception as exc:
        raise SurfaceStageError("iv_inversion", str(exc), exc) from exc
    if performance_timings is not None:
        elapsed = time.perf_counter() - stage_started
        performance_timings["iv_input_resolution_seconds"] = (
            performance_timings.get("iv_input_resolution_seconds", 0.0) + elapsed
        )
        if price_basis in {"bid", "ask"} and american_inversions is None:
            performance_timings["iv_inversion_seconds"] = performance_timings.get(
                "iv_inversion_seconds", 0.0
            ) + elapsed
    point_policy = FitPointSelectionPolicy.from_context(context)
    candidate_count = len(selection.points)
    selection_method = "full_set_feature_disabled"
    selection_rounds = 0
    full_set_fallback = False
    shadow_checked = False
    shadow_status = "not_checked"
    shadow_max_grid: Optional[float] = None
    shadow_max_node: Optional[float] = None
    shadow_max_node_strike: Optional[float] = None
    shadow_rmse: Optional[float] = None
    full_set_acceptance_required = bool(
        point_policy.enabled and point_policy.require_full_set_acceptance
    )
    full_set_acceptance_passed = False
    full_set_acceptance_reason = (
        "pending" if full_set_acceptance_required else "not_required"
    )
    full_set_required_nodes_match = False
    full_set_strategy_diagnostics_match: Optional[bool] = None
    full_set_atm_iv_difference: Optional[float] = None
    full_set_skew_difference: Optional[float] = None
    full_set_curvature_difference: Optional[float] = None
    full_set_arbitrage_flags_changed: Optional[bool] = None
    stage_started = time.perf_counter()
    fit_dtype = (
        np.dtype("float64")
        if selection.float64_retry_accepted_count
        else dtype
    )
    def with_points(points: Sequence[SurfacePoint]) -> SurfaceInputSelection:
        return replace(selection, points=tuple(points))

    fit_call_count = 0
    svi_function_evaluation_count = 0

    def execute_fit(selected: SurfaceInputSelection, starts=warm_starts) -> SmileFit:
        nonlocal fit_call_count, svi_function_evaluation_count
        fitted = fit_smile(
            selected,
            price_basis=price_basis,
            tenor_seconds=prepared.tenor_seconds,
            spot=prepared.spot,
            forward=prepared.forward,
            rate=prepared.rate,
            dividend_yield=prepared.dividend_yield,
            exercise_type=prepared.exercise_type,
            context=context,
            source_collected_at=prepared.source_collected_at,
            expiration_at=prepared.expiration_at,
            snapshot_id=prepared.snapshot_id,
            dtype=fit_dtype,
            warm_starts=starts,
        )
        fit_call_count += 1
        optimizer = fitted.parameters.get("optimizer_diagnostics", {})
        if isinstance(optimizer, Mapping):
            svi_function_evaluation_count += max(
                0, int(optimizer.get("nfev", 0) or 0)
            )
        return fitted

    def full_pool_rmse(candidate: SmileFit) -> float:
        if not candidate.successful or not selection.points:
            return math.inf
        k = np.asarray(
            [point.log_forward_moneyness for point in selection.points],
            dtype=np.float64,
        )
        observed = np.asarray(
            [point.total_variance for point in selection.points], dtype=np.float64
        )
        weights = np.asarray(
            [max(point.weight, 0.0) for point in selection.points], dtype=np.float64
        )
        predicted = candidate.evaluate_total_variance(k)
        if np.any(~np.isfinite(predicted)):
            return math.inf
        return math.sqrt(float(np.average(np.square(predicted - observed), weights=weights)))

    def full_domain_covered(candidate: SmileFit) -> bool:
        candidate_low, candidate_high = candidate.coverage
        full_values = [
            point.log_forward_moneyness for point in selection.points
        ]
        if candidate_low is None or candidate_high is None or not full_values:
            return False
        tolerance = 16.0 * np.finfo(np.float64).eps
        return (
            candidate_low <= min(full_values) + tolerance
            and candidate_high >= max(full_values) - tolerance
        )

    def generated_nodes(candidate: SmileFit) -> tuple[SurfaceNode, ...]:
        return generate_calibration_nodes(
            ExpirationCalibration(candidate),
            context=context,
            runtime=runtime,
            solver_registry=solver_registry,
        ).nodes

    def required_nodes_complete(candidate: SmileFit) -> bool:
        nodes = generated_nodes(candidate)
        expected = {
            ("atm_forward", 0.0, "NONE"),
            *{
                ("delta", value, put_call)
                for value in REQUIRED_DELTA_NODE_VALUES
                for put_call in ("PUT", "CALL")
            },
        }
        complete = {
            (node.node_type, float(node.node_value), node.put_call)
            for node in nodes
            if node.status == "complete"
            and is_required_surface_node(node.node_type, node.node_value)
        }
        return expected <= complete

    try:
        fit_selection = selection
        if point_policy.enabled and any(
            sum(point.put_call == wing for point in selection.points)
            > point_policy.initial_max_points_per_wing
            for wing in ("PUT", "CALL")
        ):
            selection_started = time.perf_counter()
            selected_points = select_surface_coreset(
                selection.points,
                maximum_points_per_wing=point_policy.initial_max_points_per_wing,
            )
            put_count = sum(point.put_call == "PUT" for point in selected_points)
            call_count = sum(point.put_call == "CALL" for point in selected_points)
            if (
                len(selected_points) >= point_policy.minimum_total_points
                and put_count >= point_policy.minimum_points_per_wing
                and call_count >= point_policy.minimum_points_per_wing
            ):
                fit_selection = with_points(selected_points)
                selection_method = "coreset_initial"
            if performance_timings is not None:
                performance_timings["fit_point_selection_seconds"] = (
                    performance_timings.get("fit_point_selection_seconds", 0.0)
                    + time.perf_counter() - selection_started
                )
        initial_fit_started = time.perf_counter()
        fit = execute_fit(fit_selection)
        if performance_timings is not None and fit_selection is not selection:
            performance_timings["fit_coreset_seconds"] = performance_timings.get(
                "fit_coreset_seconds", 0.0
            ) + (time.perf_counter() - initial_fit_started)
        candidate_pool_rmse = full_pool_rmse(fit)
        candidate_domain_complete = full_domain_covered(fit)
        candidate_required_nodes_complete = (
            required_nodes_complete(fit) if fit.successful else False
        )
        maximum_fit_rmse = float(_surface_setting(context, "maximum_weighted_rmse", 0.02))
        while (
            point_policy.enabled
            and fit_selection is not selection
            and (
                not fit.successful
                or fit.status == "complete_warning"
                or candidate_pool_rmse > maximum_fit_rmse
                or not candidate_domain_complete
                or not candidate_required_nodes_complete
            )
            and selection_rounds < point_policy.maximum_expansion_rounds
        ):
            selection_rounds += 1
            expanded_cap = (
                point_policy.initial_max_points_per_wing
                + selection_rounds * point_policy.expansion_points_per_wing
            )
            expanded = select_surface_coreset(
                selection.points, maximum_points_per_wing=expanded_cap
            )
            if len(expanded) == len(fit_selection.points):
                break
            fit_selection = with_points(expanded)
            expansion_started = time.perf_counter()
            fit = execute_fit(
                fit_selection,
                (("previous_coreset", fit.parameters),)
                if fit.method == "svi" else warm_starts,
            )
            if performance_timings is not None:
                performance_timings["fit_expansion_seconds"] = performance_timings.get(
                    "fit_expansion_seconds", 0.0
                ) + (time.perf_counter() - expansion_started)
            candidate_pool_rmse = full_pool_rmse(fit)
            candidate_domain_complete = full_domain_covered(fit)
            candidate_required_nodes_complete = (
                required_nodes_complete(fit) if fit.successful else False
            )
            selection_method = "coreset_expanded"
        if (
            point_policy.enabled
            and fit_selection is not selection
            and (
                not fit.successful
                or fit.status == "complete_warning"
                or candidate_pool_rmse > maximum_fit_rmse
                or not candidate_domain_complete
                or not candidate_required_nodes_complete
            )
        ):
            fallback_started = time.perf_counter()
            fit = execute_fit(selection)
            candidate_pool_rmse = full_pool_rmse(fit)
            fit_selection = selection
            selection_method = "full_set_fallback"
            full_set_fallback = True
            if performance_timings is not None:
                performance_timings["fit_full_set_fallback_seconds"] = (
                    performance_timings.get("fit_full_set_fallback_seconds", 0.0)
                    + time.perf_counter() - fallback_started
                )
        if point_policy.enabled and fit_selection is selection:
            if full_set_fallback:
                full_set_acceptance_reason = "coreset_precheck_failed_full_set_published"
            else:
                full_set_acceptance_passed = True
                full_set_acceptance_reason = "full_set_direct"
        if (
            point_policy.enabled
            and fit_selection is not selection
            and fit.successful
        ):
            payload = (
                f"{point_policy.shadow_seed}\0{prepared.snapshot_id}\0"
                f"{prepared.expiration_at}\0{price_basis}"
            ).encode()
            sampled = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") < int(
                point_policy.shadow_sample_fraction * (1 << 64)
            )
            compare_full_set = point_policy.require_full_set_acceptance or (
                point_policy.shadow_enabled and sampled
            )
            if compare_full_set:
                shadow_started = time.perf_counter()
                shadow_checked = True
                full_fit = execute_fit(selection)
                if full_fit.successful:
                    coverage_tolerance = 16.0 * np.finfo(np.float64).eps
                    fit_coverage = fit.coverage
                    full_coverage = full_fit.coverage
                    coverage_matches = bool(
                        fit_coverage[0] is not None
                        and fit_coverage[1] is not None
                        and full_coverage[0] is not None
                        and full_coverage[1] is not None
                        and abs(fit_coverage[0] - full_coverage[0])
                        <= coverage_tolerance
                        and abs(fit_coverage[1] - full_coverage[1])
                        <= coverage_tolerance
                    )
                    low = max(fit.coverage[0], full_fit.coverage[0])
                    high = min(fit.coverage[1], full_fit.coverage[1])
                    grid = np.linspace(low, high, 101, dtype=np.float64)
                    coreset_iv = np.sqrt(np.maximum(fit.evaluate_total_variance(grid), 0) / fit.tenor_years)
                    full_iv = np.sqrt(np.maximum(full_fit.evaluate_total_variance(grid), 0) / full_fit.tenor_years)
                    shadow_max_grid = float(np.max(np.abs(coreset_iv - full_iv)))
                    rmse_increase = candidate_pool_rmse - full_pool_rmse(full_fit)
                    shadow_rmse = rmse_increase
                    coreset_nodes = generated_nodes(fit)
                    full_nodes = generated_nodes(full_fit)
                    coreset_by_key = {
                        (node.node_type, float(node.node_value), node.put_call): node
                        for node in coreset_nodes
                        if is_required_surface_node(node.node_type, node.node_value)
                    }
                    full_by_key = {
                        (node.node_type, float(node.node_value), node.put_call): node
                        for node in full_nodes
                        if is_required_surface_node(node.node_type, node.node_value)
                    }
                    required_keys = set(coreset_by_key) | set(full_by_key)
                    required_keys |= {
                        ("atm_forward", 0.0, "NONE"),
                        *{
                            ("delta", value, put_call)
                            for value in REQUIRED_DELTA_NODE_VALUES
                            for put_call in ("PUT", "CALL")
                        },
                    }
                    iv_differences = []
                    strike_differences = []
                    full_set_required_nodes_match = True
                    for key in sorted(required_keys):
                        left_node = coreset_by_key.get(key)
                        right_node = full_by_key.get(key)
                        if (
                            left_node is None
                            or right_node is None
                            or left_node.status != "complete"
                            or right_node.status != "complete"
                        ):
                            full_set_required_nodes_match = False
                            iv_differences.append(math.inf)
                            strike_differences.append(math.inf)
                            continue
                        if (
                            left_node.fitted_iv_mid is None
                            or right_node.fitted_iv_mid is None
                        ):
                            full_set_required_nodes_match = False
                            iv_differences.append(math.inf)
                        else:
                            iv_differences.append(
                                abs(left_node.fitted_iv_mid - right_node.fitted_iv_mid)
                            )
                        if (
                            left_node.fitted_strike is None
                            or right_node.fitted_strike is None
                            or right_node.fitted_strike == 0
                        ):
                            full_set_required_nodes_match = False
                            strike_differences.append(math.inf)
                        else:
                            strike_differences.append(
                                abs(left_node.fitted_strike - right_node.fitted_strike)
                                / abs(right_node.fitted_strike)
                            )
                    shadow_max_node = max(iv_differences) if iv_differences else math.inf
                    shadow_max_node_strike = (
                        max(strike_differences) if strike_differences else math.inf
                    )

                    diagnostic_k = np.asarray([-0.10, 0.0, 0.10])
                    candidate_diagnostic_iv = np.sqrt(
                        np.maximum(fit.evaluate_total_variance(diagnostic_k), 0.0)
                        / fit.tenor_years
                    )
                    full_diagnostic_iv = np.sqrt(
                        np.maximum(full_fit.evaluate_total_variance(diagnostic_k), 0.0)
                        / full_fit.tenor_years
                    )
                    atm_difference = abs(
                        float(candidate_diagnostic_iv[1] - full_diagnostic_iv[1])
                    )
                    skew_difference = abs(
                        float(
                            (candidate_diagnostic_iv[2] - candidate_diagnostic_iv[0])
                            - (full_diagnostic_iv[2] - full_diagnostic_iv[0])
                        )
                    )
                    curvature_difference = abs(
                        float(
                            (
                                candidate_diagnostic_iv[2]
                                - 2.0 * candidate_diagnostic_iv[1]
                                + candidate_diagnostic_iv[0]
                            )
                            - (
                                full_diagnostic_iv[2]
                                - 2.0 * full_diagnostic_iv[1]
                                + full_diagnostic_iv[0]
                            )
                        )
                    )
                    full_set_strategy_diagnostics_match = bool(
                        math.isfinite(atm_difference)
                        and math.isfinite(skew_difference)
                        and math.isfinite(curvature_difference)
                        and atm_difference <= point_policy.maximum_atm_iv_difference
                        and skew_difference <= point_policy.maximum_skew_difference
                        and curvature_difference
                        <= point_policy.maximum_curvature_difference
                    )
                    full_set_atm_iv_difference = atm_difference
                    full_set_skew_difference = skew_difference
                    full_set_curvature_difference = curvature_difference
                    full_set_arbitrage_flags_changed = bool(
                        fit.butterfly_flags != full_fit.butterfly_flags
                        or fit.calendar_flags != full_fit.calendar_flags
                    )
                    breaches = []
                    if not coverage_matches:
                        breaches.append("fit_domain_coverage_changed")
                    if shadow_max_grid > point_policy.maximum_grid_iv_difference:
                        breaches.append("maximum_grid_iv_difference")
                    if rmse_increase > point_policy.maximum_weighted_rmse_increase:
                        breaches.append("weighted_rmse_increase")
                    if shadow_max_node > point_policy.maximum_node_iv_difference:
                        breaches.append("required_node_iv_difference")
                    if shadow_max_node_strike > point_policy.maximum_node_strike_fraction_difference:
                        breaches.append("required_node_strike_difference")
                    if not full_set_required_nodes_match:
                        breaches.append("required_node_status_mismatch")
                    if not full_set_strategy_diagnostics_match:
                        breaches.append("strategy_diagnostics_mismatch")
                    if fit.butterfly_flags != full_fit.butterfly_flags:
                        breaches.append("butterfly_flags_changed")
                    if fit.calendar_flags != full_fit.calendar_flags:
                        breaches.append("calendar_flags_changed")
                    if breaches:
                        fit = full_fit
                        candidate_pool_rmse = full_pool_rmse(full_fit)
                        fit_selection = selection
                        selection_method = "full_set_fallback"
                        full_set_fallback = True
                        shadow_status = "breach_full_set_published"
                        full_set_acceptance_reason = ",".join(breaches)
                    else:
                        shadow_status = "within_thresholds"
                        full_set_acceptance_passed = True
                        full_set_acceptance_reason = "within_thresholds"
                else:
                    fit = full_fit
                    fit_selection = selection
                    selection_method = "full_set_fallback"
                    full_set_fallback = True
                    shadow_status = "full_set_acceptance_fit_failed"
                    full_set_acceptance_reason = "full_set_fit_failed"
                if performance_timings is not None:
                    performance_timings["fit_full_set_acceptance_seconds"] = (
                        performance_timings.get("fit_full_set_acceptance_seconds", 0.0)
                        + time.perf_counter() - shadow_started
                    )
        fit = replace(
            fit,
            parameters={
                **fit.parameters,
                "fit_candidate_pool_weighted_rmse": (
                    candidate_pool_rmse if math.isfinite(candidate_pool_rmse) else None
                ),
                "fit_actual_call_count": fit_call_count,
                "fit_actual_svi_function_evaluations": (
                    svi_function_evaluation_count
                ),
            },
            fit_candidate_contract_count=candidate_count,
            fit_selected_contract_count=len(fit.points),
            fit_point_selection_method=selection_method,
            fit_point_selection_rounds=selection_rounds,
            fit_full_set_fallback=full_set_fallback,
            fit_full_set_shadow_checked=shadow_checked,
            fit_full_set_shadow_status=shadow_status,
            fit_full_set_shadow_max_grid_iv_difference=shadow_max_grid,
            fit_full_set_shadow_max_node_iv_difference=shadow_max_node,
            fit_full_set_shadow_max_node_strike_fraction_difference=shadow_max_node_strike,
            fit_full_set_shadow_weighted_rmse_difference=shadow_rmse,
            fit_full_set_acceptance_required=full_set_acceptance_required,
            fit_full_set_acceptance_passed=full_set_acceptance_passed,
            fit_full_set_acceptance_reason=full_set_acceptance_reason,
            fit_full_set_required_nodes_match=full_set_required_nodes_match,
            fit_full_set_strategy_diagnostics_match=full_set_strategy_diagnostics_match,
            fit_full_set_atm_iv_difference=full_set_atm_iv_difference,
            fit_full_set_skew_difference=full_set_skew_difference,
            fit_full_set_curvature_difference=full_set_curvature_difference,
            fit_full_set_arbitrage_flags_changed=full_set_arbitrage_flags_changed,
            candidate_recalculated_current_iv_point_count=selection.recalculated_current_iv_point_count,
            candidate_recalculated_reused_iv_point_count=selection.recalculated_reused_iv_point_count,
            candidate_source_iv_point_count=selection.source_iv_point_count,
            candidate_no_iv_excluded_count=selection.no_iv_excluded_count,
            candidate_source_iv_weight_fraction=selection.source_iv_weight_fraction,
        )
    except SurfaceStageNumericalInstability:
        raise
    except NumericalInstability as exc:
        raise SurfaceStageNumericalInstability(
            "smile_fit", exc.reason
        ) from exc
    except SurfaceStageError:
        raise
    except Exception as exc:
        raise SurfaceStageError("smile_fit", str(exc), exc) from exc
    if performance_timings is not None:
        performance_timings["smile_fit_seconds"] = performance_timings.get(
            "smile_fit_seconds", 0.0
        ) + (time.perf_counter() - stage_started)
    residual_threshold = float(
        _surface_setting(context, "float32_input_residual_threshold", 1.0e-4)
    )
    if fit_dtype == np.dtype("float32"):
        residuals = [
            abs(point.price_residual)
            for point in fit.points
            if point.price_residual is not None
        ]
        if residuals and max(residuals) > residual_threshold:
            raise SurfaceStageNumericalInstability(
                "iv_inversion", "float32_input_residual_exceeded"
            )
        if fit.butterfly_flags & BUTTERFLY_INDETERMINATE:
            raise SurfaceStageNumericalInstability(
                "butterfly_diagnostic",
                "butterfly_diagnostic_indeterminate",
            )
    if selection.float64_retry_accepted_count:
        fit = replace(
            fit,
            requested_dtype=dtype.name,
            actual_dtype="float64",
            precision_escalated=True,
            precision_escalation_reason="american_iv_contract_retry",
            quality_flags=fit.quality_flags | QUALITY_PRECISION_ESCALATED,
        )
    return ExpirationCalibration(fit)


def minimum_tenor_calibration(
    prepared: PreparedExpirationInputs,
    *,
    price_basis: str,
    context: Mapping[str, Any],
) -> ExpirationCalibration:
    minimum_tenor = _minimum_surface_tenor(context)
    requested_dtype = resolve_numeric_dtype(
        prepared.tenor_seconds,
        NumericPrecisionPolicy.from_context(context).short_tenor_max_seconds,
        NumericPrecisionPolicy.from_context(context).short_tenor_dtype,
        NumericPrecisionPolicy.from_context(context).other_tenor_dtype,
    ).name
    detail = (
        f"tenor_seconds={prepared.tenor_seconds:.9g} is below "
        f"minimum_tenor_seconds={minimum_tenor:.9g}"
    )
    fit = SmileFit(
        price_basis=price_basis,
        method="none",
        status="skipped_minimum_tenor",
        error=detail,
        parameters={
            "minimum_tenor_seconds": minimum_tenor,
            "actual_tenor_seconds": prepared.tenor_seconds,
            "rejected_reasons": {"minimum_tenor": prepared.input_count},
        },
        points=(),
        input_count=prepared.input_count,
        rejected_reasons={"minimum_tenor": prepared.input_count},
        source_fallback_count=0,
        source_truncated=prepared.source_truncated,
        tenor_seconds=prepared.tenor_seconds,
        tenor_years=prepared.tenor_seconds / SECONDS_PER_YEAR,
        spot=prepared.spot,
        forward=prepared.forward,
        rate=prepared.rate,
        dividend_yield=prepared.dividend_yield,
        exercise_type=prepared.exercise_type,
        weighted_rmse=None,
        maximum_absolute_error=None,
        butterfly_flags=0,
        calendar_flags=0,
        quality_flags=(
            QUALITY_SOURCE_TRUNCATED if prepared.source_truncated else 0
        ),
        failure_stage="minimum_tenor",
        requested_dtype=requested_dtype,
        actual_dtype=requested_dtype,
        source_collected_at=prepared.source_collected_at,
        expiration_at=prepared.expiration_at,
        snapshot_id=prepared.snapshot_id,
    )
    return ExpirationCalibration(fit)


def _apply_precision_execution(
    calibration: ExpirationCalibration,
    *,
    requested_dtype: str,
    actual_dtype: str,
    escalation_reason: str = "",
) -> ExpirationCalibration:
    contract_retry_escalated = calibration.fit.precision_escalated
    effective_actual_dtype = (
        calibration.fit.actual_dtype
        if contract_retry_escalated
        else actual_dtype
    )
    escalated = requested_dtype != effective_actual_dtype
    effective_reason = (
        calibration.fit.precision_escalation_reason
        if contract_retry_escalated
        else escalation_reason
    )
    fit = replace(
        calibration.fit,
        requested_dtype=requested_dtype,
        actual_dtype=effective_actual_dtype,
        precision_escalated=escalated,
        precision_escalation_reason=effective_reason,
        quality_flags=(
            calibration.fit.quality_flags
            | (QUALITY_PRECISION_ESCALATED if escalated else 0)
        ),
    )
    return replace(calibration, fit=fit)


def calibrate_expiration(
    rows: Sequence[Mapping[str, Any]],
    *,
    price_basis: str,
    context: Mapping[str, Any],
    generate_nodes: bool = True,
    force_float64_reason: Optional[str] = None,
    prepared: Optional[PreparedExpirationInputs] = None,
    runtime: Optional[SurfaceNumericalRuntime] = None,
    warm_starts: Sequence[tuple[str, Mapping[str, Any]]] = (),
    solver_registry: Optional[Any] = None,
) -> ExpirationCalibration:
    if not rows:
        raise ValueError("Cannot calibrate an empty expiration")
    prepared = prepared or prepare_surface_expiration_inputs(rows, context=context)
    tenor_seconds = prepared.tenor_seconds
    if tenor_seconds <= 0:
        raise ValueError("Expiration has no positive tenor")
    minimum_tenor = _minimum_surface_tenor(context)
    if tenor_seconds < minimum_tenor:
        return minimum_tenor_calibration(
            prepared, price_basis=price_basis, context=context
        )
    policy = NumericPrecisionPolicy.from_context(context)
    runtime = runtime or SurfaceNumericalRuntime.from_context(context)
    if (
        solver_registry is None
        and prepared.exercise_type.strip().upper() in {"A", "AMERICAN"}
    ):
        from algotrading.refdata.options.solvers.Registry import SolverRegistry

        solver_registry = SolverRegistry(context)

    def operation(dtype: np.dtype) -> ExpirationCalibration:
        return _calibrate_prepared_operation(
            rows,
            prepared,
            price_basis=price_basis,
            context=context,
            dtype=dtype,
            runtime=runtime,
            warm_starts=warm_starts,
            solver_registry=solver_registry,
        )

    requested_dtype = resolve_numeric_dtype(
        tenor_seconds,
        policy.short_tenor_max_seconds,
        policy.short_tenor_dtype,
        policy.other_tenor_dtype,
    ).name
    if force_float64_reason and requested_dtype == "float32":
        try:
            value = operation(np.dtype("float64"))
            value.fit.assert_finite()
        except Exception as retry_error:
            raise PrecisionEscalationFailure(
                str(force_float64_reason), retry_error
            ) from retry_error
        calibrated = _apply_precision_execution(
            value,
            requested_dtype=requested_dtype,
            actual_dtype="float64",
            escalation_reason=str(force_float64_reason),
        )
        return (
            generate_calibration_nodes(
                calibrated,
                context=context,
                runtime=runtime,
                solver_registry=solver_registry,
            )
            if generate_nodes
            else calibrated
        )
    execution = execute_with_precision_policy(tenor_seconds, policy, operation)
    calibrated = _apply_precision_execution(
        execution.value,
        requested_dtype=execution.requested_dtype,
        actual_dtype=execution.actual_dtype,
        escalation_reason=execution.escalation_reason,
    )
    return (
        generate_calibration_nodes(
            calibrated,
            context=context,
            runtime=runtime,
            solver_registry=solver_registry,
        )
        if generate_nodes
        else calibrated
    )


@dataclass(frozen=True)
class MultiBasisCalibrationResult:
    calibrations: Mapping[str, ExpirationCalibration]
    errors: Mapping[str, Exception]
    iv_diagnostics: Mapping[str, Any]
    timings: Mapping[str, float]


def calibrate_expiration_bases(
    rows: Sequence[Mapping[str, Any]],
    *,
    price_bases: Sequence[str],
    context: Mapping[str, Any],
    underlying_symbol: str = "",
    expiration_at: Any = None,
    warm_start_cache: Optional[SVIWarmStartCache] = None,
    force_float64_reason: Optional[str] = None,
    runtime: Optional[SurfaceNumericalRuntime] = None,
    generate_nodes: bool = True,
    solver_registry: Optional[Any] = None,
    preloaded_american_inversions: Optional[
        Mapping[tuple[str, str, str, int], IVInversionResult]
    ] = None,
) -> MultiBasisCalibrationResult:
    """Calibrate bases from one common filter and one combined American batch."""
    if not rows:
        raise ValueError("Cannot calibrate an empty expiration")
    bases = tuple(dict.fromkeys(str(value) for value in price_bases))
    performance_timings: dict[str, float] = {}
    prepared_started = time.perf_counter()
    prepared = prepare_surface_expiration_inputs(rows, context=context)
    performance_timings["input_prepare_seconds"] = (
        time.perf_counter() - prepared_started
    )
    if prepared.tenor_seconds <= 0:
        raise ValueError("Expiration has no positive tenor")
    minimum_tenor = _minimum_surface_tenor(context)
    if prepared.tenor_seconds < minimum_tenor:
        return MultiBasisCalibrationResult(
            MappingProxyType(
                {
                    basis: minimum_tenor_calibration(
                        prepared, price_basis=basis, context=context
                    )
                    for basis in bases
                }
            ),
            MappingProxyType({}),
            MappingProxyType(IVInversionDiagnostics().summary()),
            MappingProxyType(dict(sorted(performance_timings.items()))),
        )
    runtime = runtime or SurfaceNumericalRuntime.from_context(context)
    if (
        solver_registry is None
        and prepared.exercise_type.strip().upper() in {"A", "AMERICAN"}
    ):
        from algotrading.refdata.options.solvers.Registry import SolverRegistry

        solver_registry = SolverRegistry(context)
    policy = NumericPrecisionPolicy.from_context(context)
    requested_dtype = resolve_numeric_dtype(
        prepared.tenor_seconds,
        policy.short_tenor_max_seconds,
        policy.short_tenor_dtype,
        policy.other_tenor_dtype,
    ).name
    initial_dtype = (
        "float64"
        if force_float64_reason and requested_dtype == "float32"
        else requested_dtype
    )
    diagnostics = IVInversionDiagnostics()
    jobs = [
        job
        for basis in bases
        for job in american_iv_jobs_for_basis(
            prepared, price_basis=basis, context=context
        )
    ]
    combined: dict[str, dict[int, IVInversionResult]] = {
        basis: {} for basis in bases
    }
    combined_error: Optional[Exception] = None
    if jobs:
        try:
            inversion_started = time.perf_counter()
            if preloaded_american_inversions is None:
                inversions = invert_routed_american_implied_volatilities(
                    [job.inversion_input for job in jobs],
                    context=context,
                    dtype=np.dtype(initial_dtype),
                    diagnostics=diagnostics,
                    solver_registry=solver_registry,
                )
            else:
                inversions = tuple(
                    preloaded_american_inversions[american_iv_job_key(prepared, job)]
                    for job in jobs
                )
                complete = [item for item in inversions if item.status == "complete"]
                rejected = [item for item in inversions if item.status != "complete"]
                diagnostics.complete_result_count += len(complete)
                diagnostics.rejected_result_count += len(rejected)
                diagnostics.iterations.extend(item.iterations for item in complete)
                diagnostics.rejected_by_reason.update(
                    item.reason or "iv_inversion_failed" for item in rejected
                )
                diagnostics.float64_retry_contract_count += sum(
                    item.precision_escalated for item in inversions
                )
                diagnostics.float64_recovered_contract_count += sum(
                    item.precision_escalated and item.status == "complete"
                    for item in inversions
                )
            performance_timings["iv_inversion_seconds"] = (
                performance_timings.get("iv_inversion_seconds", 0.0)
                + time.perf_counter()
                - inversion_started
            )
            for job, inversion in zip(jobs, inversions):
                combined[job.price_basis][job.row_index] = inversion
        except NumericalInstability as exc:
            combined_error = exc

    calibrations: dict[str, ExpirationCalibration] = {}
    errors: dict[str, Exception] = {}
    current_mid_parameters: Optional[Mapping[str, Any]] = None
    ordered_bases = sorted(
        bases, key=lambda basis: (basis not in {"mid", "source"}, bases.index(basis))
    )
    for basis in ordered_bases:
        warm_starts: list[tuple[str, Mapping[str, Any]]] = []
        cache_key = (str(underlying_symbol), expiration_at, basis)
        prior = warm_start_cache.get(cache_key) if warm_start_cache is not None else None
        if basis in {"bid", "ask"} and current_mid_parameters is not None:
            warm_starts.append(("current_midpoint", current_mid_parameters))
        if prior is not None:
            warm_starts.append(("prior_same_basis", prior))
        def operation(dtype: np.dtype, *, use_combined: bool) -> ExpirationCalibration:
            return _calibrate_prepared_operation(
                rows,
                prepared,
                price_basis=basis,
                context=context,
                dtype=dtype,
                runtime=runtime,
                american_inversions=(combined[basis] if use_combined and jobs else None),
                diagnostics=diagnostics,
                warm_starts=warm_starts,
                performance_timings=performance_timings,
                solver_registry=solver_registry,
            )

        try:
            if combined_error is None:
                try:
                    value = operation(np.dtype(initial_dtype), use_combined=True)
                    value.fit.assert_finite()
                    actual_dtype = initial_dtype
                    reason = (
                        str(force_float64_reason or "")
                        if requested_dtype != initial_dtype
                        else ""
                    )
                except NumericalInstability as exc:
                    if initial_dtype != "float32" or not policy.auto_promote_to_float64:
                        raise
                    diagnostics.float64_escalation_count += 1
                    try:
                        value = operation(np.dtype("float64"), use_combined=False)
                        value.fit.assert_finite()
                    except Exception as retry_error:
                        raise PrecisionEscalationFailure(
                            exc.reason, retry_error
                        ) from retry_error
                    actual_dtype = "float64"
                    reason = exc.reason
            else:
                if force_float64_reason and requested_dtype == "float32":
                    try:
                        value = operation(np.dtype("float64"), use_combined=False)
                        value.fit.assert_finite()
                    except Exception as retry_error:
                        raise PrecisionEscalationFailure(
                            str(force_float64_reason), retry_error
                        ) from retry_error
                    actual_dtype = "float64"
                    reason = str(force_float64_reason)
                    diagnostics.float64_escalation_count += 1
                else:
                    execution = execute_with_precision_policy(
                        prepared.tenor_seconds,
                        policy,
                        lambda dtype: operation(dtype, use_combined=False),
                    )
                    value = execution.value
                    actual_dtype = execution.actual_dtype
                    reason = execution.escalation_reason
                    if execution.escalated:
                        diagnostics.float64_escalation_count += 1
            calibration = _apply_precision_execution(
                value,
                requested_dtype=requested_dtype,
                actual_dtype=actual_dtype,
                escalation_reason=reason,
            )
            calibrations[basis] = calibration
            if calibration.fit.successful and calibration.fit.method == "svi":
                if basis in {"mid", "source"}:
                    current_mid_parameters = calibration.fit.parameters
                if warm_start_cache is not None:
                    warm_start_cache.put(cache_key, calibration.fit.parameters)
        except Exception as exc:
            errors[basis] = exc
    if generate_nodes:
        primary_basis = "mid" if "mid" in calibrations else "source"
        primary = calibrations.get(primary_basis)
        if primary is not None:
            node_started = time.perf_counter()
            calibrations[primary_basis] = generate_calibration_nodes(
                primary,
                context=context,
                runtime=runtime,
                solver_registry=solver_registry,
            )
            performance_timings["node_generation_seconds"] = (
                performance_timings.get("node_generation_seconds", 0.0)
                + time.perf_counter()
                - node_started
            )
    return MultiBasisCalibrationResult(
        MappingProxyType(calibrations),
        MappingProxyType(errors),
        MappingProxyType(diagnostics.summary()),
        MappingProxyType(dict(sorted(performance_timings.items()))),
    )


def apply_calendar_arbitrage_diagnostics(
    calibrations: Sequence[ExpirationCalibration],
    *,
    tolerance: float = 1.0e-10,
    grid_size: int = 101,
) -> tuple[ExpirationCalibration, ...]:
    output = list(calibrations)
    by_basis: dict[str, list[int]] = {}
    for index, calibration in enumerate(output):
        if calibration.fit.successful:
            by_basis.setdefault(calibration.fit.price_basis, []).append(index)
    for indices in by_basis.values():
        indices.sort(key=lambda index: output[index].fit.tenor_seconds)
        for previous_index, current_index in zip(indices, indices[1:]):
            previous = output[previous_index].fit
            current = output[current_index].fit
            low = max(previous.coverage[0], current.coverage[0])
            high = min(previous.coverage[1], current.coverage[1])
            if low is None or high is None or high <= low:
                continue
            grid = np.linspace(low, high, max(21, int(grid_size)), dtype=np.float64)
            try:
                difference = (
                    current.evaluate_total_variance(grid)
                    - previous.evaluate_total_variance(grid)
                )
            except Exception as exc:
                failed = replace(
                    current,
                    status="failed_arbitrage",
                    error=str(exc),
                    failure_stage="calendar_diagnostic",
                )
                output[current_index] = replace(
                    output[current_index], fit=failed
                )
                continue
            if np.any(~np.isfinite(difference)):
                failed = replace(
                    current,
                    status="failed_arbitrage",
                    error="nonfinite calendar diagnostic",
                    failure_stage="calendar_diagnostic",
                )
                output[current_index] = replace(
                    output[current_index], fit=failed
                )
                continue
            if float(np.min(difference)) < -abs(float(tolerance)):
                updated = replace(
                    current,
                    status="complete_warning",
                    calendar_flags=current.calendar_flags | CALENDAR_VARIANCE_INVERSION,
                    quality_flags=current.quality_flags | QUALITY_CALENDAR_WARNING,
                )
                output[current_index] = replace(output[current_index], fit=updated)
    return tuple(output)


def merge_bid_mid_ask_nodes(
    calibrations: Mapping[str, ExpirationCalibration],
) -> tuple[SurfaceNode, ...]:
    primary = calibrations.get("mid") or calibrations.get("source")
    if primary is None:
        return ()
    bid_fit = calibrations.get("bid")
    ask_fit = calibrations.get("ask")
    merged = []
    for node in primary.nodes:
        # Midpoint owns canonical node geometry.  Bid and ask IVs are evaluated
        # at that exact fitted strike, never copied from independent delta roots.
        bid_iv = ask_iv = None
        if node.fitted_strike is not None:
            if bid_fit is not None and bid_fit.fit.successful:
                bid_iv = _fit_iv(
                    bid_fit.fit,
                    math.log(node.fitted_strike / bid_fit.fit.forward),
                )
            if ask_fit is not None and ask_fit.fit.successful:
                ask_iv = _fit_iv(
                    ask_fit.fit,
                    math.log(node.fitted_strike / ask_fit.fit.forward),
                )
        quality = node.quality_flags
        if (
            bid_iv is not None
            and node.fitted_iv_mid is not None
            and ask_iv is not None
            and not bid_iv <= node.fitted_iv_mid <= ask_iv
        ):
            tolerance = np.finfo(np.float64).eps * max(abs(ask_iv), 1.0) * 64.0
            if bid_iv > node.fitted_iv_mid + tolerance or node.fitted_iv_mid > ask_iv + tolerance:
                quality |= QUALITY_NODE_ORDERING
        merged.append(
            replace(
                node,
                fitted_iv_bid=bid_iv,
                fitted_iv_ask=ask_iv,
                quality_flags=quality,
            )
        )
    return tuple(merged)


def fit_parameters_json(fit: SmileFit) -> str:
    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, (np.floating, float)):
            converted = float(value)
            return converted if math.isfinite(converted) else None
        return value

    payload = {
        "model": fit.method,
        "parameters": json_safe(fit.parameters),
        "tenor_seconds": json_safe(fit.tenor_seconds),
        "tenor_years": json_safe(fit.tenor_years),
        "spot": json_safe(fit.spot),
        "forward": json_safe(fit.forward),
        "rate": json_safe(fit.rate),
        "dividend_yield": json_safe(fit.dividend_yield),
        "exercise_type": fit.exercise_type,
        "iv_provenance": {
            "recalculated_current_iv_point_count": sum(
                point.iv_origin == "recalculated_current" for point in fit.points
            ),
            "recalculated_reused_iv_point_count": sum(
                point.iv_origin == "recalculated_reused" for point in fit.points
            ),
            "source_iv_point_count": sum(
                point.iv_origin == "source" for point in fit.points
            ),
            "no_iv_excluded_count": int(
                fit.rejected_reasons.get("no_valid_preferred_iv", 0)
            ),
            "source_iv_weight_fraction": (
                sum(point.weight for point in fit.points if point.iv_origin == "source")
                / sum(point.weight for point in fit.points)
                if fit.points and sum(point.weight for point in fit.points) > 0
                else 0.0
            ),
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
