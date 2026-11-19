"""Worker-local lazy registry for all supported American repricing engines."""

from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from algotrading.refdata.options.Common import (
    OPTIONS_CALCS_CONFIG_PREFIX,
    OPTIONS_CALCS_GREEKS_PREFIX,
)
from algotrading.refdata.options.solvers.IvSolver import (
    IvBatchProblem,
    IvSolveResult,
    preload_iv_solver_dependencies,
    solve_iv,
    solve_iv_many,
)
from algotrading.refdata.options.solvers.Types import (
    CircuitCallOutcome,
    IvHint,
    PricingRequest,
    QUALITY_IV_PRICE_INDETERMINATE,
    RecalcRetryClass,
    RepricingMethod,
    RepricingResult,
    SolverBatchJob,
    SolverOperation,
    SolverCircuitOpenError,
    SolverInfrastructureError,
    SolverNumericalError,
    SolverResourceError,
    SolverUnavailableError,
    ValuationResult,
    classify_repricing_failure,
)


@dataclass(frozen=True)
class IvSolverConfig:
    minimum: float
    maximum: float
    price_tolerance: float
    max_iterations: int
    source_bracket_multipliers: tuple[float, ...]
    prior_seed_bracket_multipliers: tuple[float, ...]
    hint_identifiability_enabled: bool
    hint_identifiability_probe_iv_points: float
    hint_identifiability_minimum_price_span_absolute: float
    hint_identifiability_price_tolerance_multiplier: float


@dataclass(frozen=True)
class Float32GuardConfig:
    enabled: bool
    short_tenor_validation_enabled: bool
    short_tenor_max_seconds: float
    shadow_fraction: float
    minimum_signal_to_roundoff: float
    price_atol: float
    delta_atol: float
    gamma_atol: float
    theta_per_day_atol: float
    vega_per_1pct_atol: float
    rho_per_1pct_atol: float
    rtol: float


@dataclass(frozen=True)
class NumericalBatchConfig:
    enabled: bool
    maximum_jobs: int
    maximum_estimated_bytes: int
    float64_retry_enabled: bool
    exception_isolation_enabled: bool
    scalar_rollback_enabled: bool
    float32_guard: Float32GuardConfig

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "NumericalBatchConfig":
        from algotrading.refdata.options.Common import _as_bool

        prefix = OPTIONS_CALCS_GREEKS_PREFIX + "batch."

        def positive_integer(name: str, default: int) -> int:
            raw = context.get(prefix + name, default)
            if isinstance(raw, bool) or isinstance(raw, float) and not raw.is_integer():
                raise ValueError(f"{prefix + name} must be a positive integer")
            try:
                value = int(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{prefix + name} must be a positive integer") from exc
            if value <= 0:
                raise ValueError(f"{prefix + name} must be a positive integer")
            return value

        guard_prefix = prefix + "float32_guard."

        def finite_nonnegative(name: str, default: float) -> float:
            value = float(context.get(guard_prefix + name, default))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{guard_prefix + name} must be finite and nonnegative")
            return value

        shadow_fraction = finite_nonnegative("shadow_fraction", 0.01)
        if shadow_fraction > 1.0:
            raise ValueError(f"{guard_prefix}shadow_fraction must be between 0 and 1")
        minimum_signal = finite_nonnegative("minimum_signal_to_roundoff", 256.0)
        if minimum_signal <= 0.0:
            raise ValueError(
                f"{guard_prefix}minimum_signal_to_roundoff must be positive"
            )
        short_tenor_max_seconds = float(
            context.get(
                OPTIONS_CALCS_CONFIG_PREFIX + "short_tenor_max_seconds",
                86400.0,
            )
        )
        if not math.isfinite(short_tenor_max_seconds) or short_tenor_max_seconds <= 0:
            raise ValueError(
                f"{OPTIONS_CALCS_CONFIG_PREFIX}short_tenor_max_seconds must be positive and finite"
            )
        guard = Float32GuardConfig(
            enabled=_as_bool(context.get(guard_prefix + "enabled", True)),
            short_tenor_validation_enabled=_as_bool(
                context.get(guard_prefix + "short_tenor_validation_enabled", True)
            ),
            short_tenor_max_seconds=short_tenor_max_seconds,
            shadow_fraction=shadow_fraction,
            minimum_signal_to_roundoff=minimum_signal,
            price_atol=finite_nonnegative("price_atol", 2.5e-5),
            delta_atol=finite_nonnegative("delta_atol", 2.5e-4),
            gamma_atol=finite_nonnegative("gamma_atol", 2.5e-5),
            theta_per_day_atol=finite_nonnegative("theta_per_day_atol", 2.5e-4),
            vega_per_1pct_atol=finite_nonnegative("vega_per_1pct_atol", 2.5e-4),
            rho_per_1pct_atol=finite_nonnegative("rho_per_1pct_atol", 2.5e-4),
            rtol=finite_nonnegative("rtol", 5.0e-4),
        )
        return cls(
            enabled=_as_bool(context.get(prefix + "enabled", True)),
            maximum_jobs=positive_integer("maximum_jobs", 2048),
            maximum_estimated_bytes=positive_integer(
                "maximum_estimated_bytes", 536_870_912
            ),
            float64_retry_enabled=_as_bool(
                context.get(prefix + "float64_retry_enabled", True)
            ),
            exception_isolation_enabled=_as_bool(
                context.get(prefix + "exception_isolation_enabled", True)
            ),
            scalar_rollback_enabled=_as_bool(
                context.get(prefix + "scalar_rollback_enabled", True)
            ),
            float32_guard=guard,
        )


def _parse_iv_solver_config(context: Mapping[str, Any]) -> IvSolverConfig:
    from algotrading.refdata.options.Common import _as_bool

    prefix = OPTIONS_CALCS_GREEKS_PREFIX + "iv."
    multipliers = context.get(
        prefix + "source_bracket_multipliers", (0.5, 0.75, 1.0, 1.25, 1.5)
    )
    if isinstance(multipliers, str):
        parsed_multipliers = tuple(
            float(value.strip()) for value in multipliers.split(",") if value.strip()
        )
    else:
        parsed_multipliers = tuple(float(value) for value in multipliers)
    prior_multipliers = context.get(
        prefix + "prior_seed_bracket_multipliers", (0.75, 0.90, 1.0, 1.10, 1.25)
    )
    if isinstance(prior_multipliers, str):
        parsed_prior = tuple(
            float(value.strip()) for value in prior_multipliers.split(",") if value.strip()
        )
    else:
        parsed_prior = tuple(float(value) for value in prior_multipliers)
    config = IvSolverConfig(
        minimum=float(context.get(prefix + "minimum", 1.0e-4)),
        maximum=float(context.get(prefix + "maximum", 10.0)),
        price_tolerance=float(context.get(prefix + "price_tolerance", 1.0e-7)),
        max_iterations=int(context.get(prefix + "max_iterations", 100)),
        source_bracket_multipliers=parsed_multipliers,
        prior_seed_bracket_multipliers=parsed_prior,
        hint_identifiability_enabled=_as_bool(
            context.get(prefix + "hint_identifiability.enabled", True)
        ),
        hint_identifiability_probe_iv_points=float(
            context.get(prefix + "hint_identifiability.probe_iv_points", 1.0)
        ),
        hint_identifiability_minimum_price_span_absolute=float(
            context.get(
                prefix + "hint_identifiability.minimum_price_span_absolute",
                1.0e-6,
            )
        ),
        hint_identifiability_price_tolerance_multiplier=float(
            context.get(
                prefix + "hint_identifiability.price_tolerance_multiplier",
                10.0,
            )
        ),
    )
    ident_values = (
        config.hint_identifiability_probe_iv_points,
        config.hint_identifiability_minimum_price_span_absolute,
        config.hint_identifiability_price_tolerance_multiplier,
    )
    if any(not math.isfinite(value) for value in ident_values):
        raise ValueError("IV hint-identifiability configuration must be finite")
    if (
        config.hint_identifiability_probe_iv_points <= 0.0
        or config.hint_identifiability_minimum_price_span_absolute < 0.0
        or config.hint_identifiability_price_tolerance_multiplier <= 0.0
    ):
        raise ValueError(
            "IV hint-identifiability probe/multiplier must be positive and "
            "the absolute span floor must be nonnegative"
        )
    return config


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitBreakerConfig:
    enabled: bool
    window_seconds: float
    minimum_attempts: int
    minimum_infrastructure_failures: int
    failure_fraction: float
    initial_open_seconds: float
    maximum_open_seconds: float
    open_backoff_multiplier: float
    half_open_probe_count: int
    allow_worker_start_with_open_method: bool

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "CircuitBreakerConfig":
        prefix = OPTIONS_CALCS_GREEKS_PREFIX + "circuit_breaker."
        from algotrading.refdata.options.Common import _as_bool

        config = cls(
            enabled=_as_bool(context.get(prefix + "enabled", False)),
            window_seconds=float(context.get(prefix + "window_seconds", 60)),
            minimum_attempts=int(context.get(prefix + "minimum_attempts", 3)),
            minimum_infrastructure_failures=int(
                context.get(prefix + "minimum_infrastructure_failures", 3)
            ),
            failure_fraction=float(context.get(prefix + "failure_fraction", 0.75)),
            initial_open_seconds=float(
                context.get(prefix + "initial_open_seconds", 30)
            ),
            maximum_open_seconds=float(
                context.get(prefix + "maximum_open_seconds", 600)
            ),
            open_backoff_multiplier=float(
                context.get(prefix + "open_backoff_multiplier", 2.0)
            ),
            half_open_probe_count=int(
                context.get(prefix + "half_open_probe_count", 1)
            ),
            allow_worker_start_with_open_method=_as_bool(
                context.get(prefix + "allow_worker_start_with_open_method", True)
            ),
        )
        numeric = (
            config.window_seconds,
            config.failure_fraction,
            config.initial_open_seconds,
            config.maximum_open_seconds,
            config.open_backoff_multiplier,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("circuit-breaker values must be finite")
        if (
            config.window_seconds <= 0
            or config.minimum_attempts <= 0
            or config.minimum_infrastructure_failures <= 0
            or not 0 < config.failure_fraction <= 1
            or config.initial_open_seconds <= 0
            or config.maximum_open_seconds < config.initial_open_seconds
            or config.open_backoff_multiplier < 1
            or config.half_open_probe_count <= 0
        ):
            raise ValueError("invalid circuit-breaker configuration")
        return config


class MethodCircuitBreaker:
    """Worker-local breaker; only typed infrastructure outcomes affect state."""

    def __init__(
        self,
        method: RepricingMethod,
        config: CircuitBreakerConfig,
        *,
        monotonic: Any = time.monotonic,
        utcnow: Any = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.method = method
        self.config = config
        self._monotonic = monotonic
        self._utcnow = utcnow
        self.state = CircuitState.CLOSED
        self._events: deque[tuple[float, bool]] = deque()
        self._open_until_mono = 0.0
        self._open_duration = config.initial_open_seconds
        self._half_open_probes = 0
        self._lock = threading.RLock()
        self.metrics: dict[str, int] = {
            "open_transitions": 0,
            "close_transitions": 0,
            "half_open_transitions": 0,
            "half_open_probes": 0,
            "actual_infrastructure_failures": 0,
            "suppressed_calls": 0,
        }

    @property
    def open_until(self) -> datetime:
        with self._lock:
            remaining = max(0.0, self._open_until_mono - self._monotonic())
            return self._utcnow() + timedelta(seconds=remaining)

    def _prune(self, now: float) -> None:
        threshold = now - self.config.window_seconds
        while self._events and self._events[0][0] < threshold:
            self._events.popleft()

    def _open(self, now: float, *, backoff: bool) -> None:
        if backoff:
            self._open_duration = min(
                self.config.maximum_open_seconds,
                self._open_duration * self.config.open_backoff_multiplier,
            )
        self.state = CircuitState.OPEN
        self._open_until_mono = now + self._open_duration
        self._half_open_probes = 0
        self.metrics["open_transitions"] += 1

    def force_open(self) -> None:
        with self._lock:
            if not self.config.enabled:
                return
            if self.state is CircuitState.OPEN:
                return
            self._open(self._monotonic(), backoff=False)

    def record_unknown_outcome(self) -> None:
        """Release a half-open probe without changing circuit health state."""
        with self._lock:
            if self.config.enabled and self.state is CircuitState.HALF_OPEN:
                self._half_open_probes = max(0, self._half_open_probes - 1)

    def before_call(self) -> None:
        with self._lock:
            if not self.config.enabled:
                return
            now = self._monotonic()
            if self.state is CircuitState.OPEN:
                if now < self._open_until_mono:
                    self.metrics["suppressed_calls"] += 1
                    raise SolverCircuitOpenError(self.method.value, self.open_until)
                self.state = CircuitState.HALF_OPEN
                self._half_open_probes = 0
                self.metrics["half_open_transitions"] += 1
            if self.state is CircuitState.HALF_OPEN:
                if self._half_open_probes >= self.config.half_open_probe_count:
                    self.metrics["suppressed_calls"] += 1
                    raise SolverCircuitOpenError(self.method.value, self.open_until)
                self._half_open_probes += 1
                self.metrics["half_open_probes"] += 1

    def record_infrastructure_success(self, logical_count: int = 1) -> None:
        """Record one successful infrastructure outcome per permitted job."""
        count = max(0, int(logical_count))
        if not count:
            return
        with self._lock:
            if not self.config.enabled:
                return
            now = self._monotonic()
            if self.state is CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self._events.clear()
                self._open_duration = self.config.initial_open_seconds
                self._half_open_probes = 0
                self.metrics["close_transitions"] += 1
                count -= 1
            if self.state is CircuitState.CLOSED:
                self._events.extend((now, False) for _index in range(count))
                self._prune(now)

    def record_infrastructure_failure(self, logical_count: int = 1) -> None:
        """Record one physical failure affecting ``logical_count`` jobs."""
        count = max(1, int(logical_count))
        with self._lock:
            if not self.config.enabled:
                return
            now = self._monotonic()
            self.metrics["actual_infrastructure_failures"] += 1
            if self.state is CircuitState.HALF_OPEN:
                self._open(now, backoff=True)
                return
            self._events.extend((now, True) for _index in range(count))
            self._prune(now)
            failures = sum(1 for _when, failed in self._events if failed)
            if (
                len(self._events) >= self.config.minimum_attempts
                and failures >= self.config.minimum_infrastructure_failures
                and failures / len(self._events) >= self.config.failure_fraction
            ):
                self._open(now, backoff=False)

    def snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "method": self.method.value,
                "state": self.state.value,
                "open_until": (
                    self.open_until if self.state is not CircuitState.CLOSED else None
                ),
                "open_seconds": (
                    max(0.0, self._open_until_mono - self._monotonic())
                    if self.state is CircuitState.OPEN else 0.0
                ),
                **self.metrics,
            }


class SolverAdapter:
    def __init__(
        self,
        method: RepricingMethod,
        module: Any,
        parameters: Mapping[str, Any],
    ) -> None:
        self.method = method
        self.module = module
        self.parameters = dict(parameters)
        self.algorithm_version = str(module.ALGORITHM_VERSION)
        payload = json.dumps(self.parameters, sort_keys=True, separators=(",", ":"))
        self.configuration_hash = hashlib.sha256(payload.encode()).hexdigest()

    def _compute_dtype(self, requested_dtype: str) -> Any:
        name = str(requested_dtype or "float64").strip().lower()
        if self.method is RepricingMethod.CRANK_NICOLSON:
            return np.float64
        if name not in {"float32", "float64"}:
            raise ValueError(f"unsupported requested dtype {requested_dtype!r}")
        return getattr(self.module.torch, name)

    def _aligned_batch_kwargs(
        self,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float] | np.ndarray,
        *,
        requested_dtype: str,
    ) -> dict[str, Any]:
        count = len(requests)
        volatility_array = np.ascontiguousarray(
            np.asarray(volatilities, dtype=np.float64).reshape(-1)
        )
        if len(volatility_array) != count:
            raise ValueError("requests and volatilities must have the same length")
        kwargs = {
            "S": np.ascontiguousarray(
                np.fromiter((request.spot for request in requests), np.float64, count=count)
            ),
            "K": np.ascontiguousarray(
                np.fromiter((request.strike for request in requests), np.float64, count=count)
            ),
            "T": np.ascontiguousarray(
                np.fromiter(
                    (request.tenor_seconds / (365.25 * 86400.0) for request in requests),
                    np.float64,
                    count=count,
                )
            ),
            "r": np.ascontiguousarray(
                np.fromiter((request.rate for request in requests), np.float64, count=count)
            ),
            "q": np.ascontiguousarray(
                np.fromiter(
                    (request.dividend_yield for request in requests), np.float64, count=count
                )
            ),
            "is_call": np.ascontiguousarray(
                np.fromiter(
                    (request.put_call.upper() == "CALL" for request in requests),
                    bool,
                    count=count,
                )
            ),
            "sigma": volatility_array,
            **{
                name: value
                for name, value in self.parameters.items()
                if name not in {
                    "fused_greeks_enabled",
                    "fused_greeks_maximum_contracts",
                }
            },
        }
        if self.method is RepricingMethod.CRANK_NICOLSON:
            kwargs.update(
                source_collected_at=[request.source_collected_at for request in requests],
                expiration_at=[request.expiration_at for request in requests],
            )
        else:
            kwargs["dtype"] = self._compute_dtype(requested_dtype)
        return kwargs

    def _batch_kwargs(self, request: PricingRequest, volatility: float) -> dict[str, Any]:
        return self._aligned_batch_kwargs(
            (request,),
            (volatility,),
            requested_dtype=request.requested_dtype,
        )

    def _scalar_kwargs(self, request: PricingRequest) -> dict[str, Any]:
        return {
            "spot": float(request.spot),
            "strike": float(request.strike),
            "tenor_seconds": float(request.tenor_seconds),
            "rate": float(request.rate),
            "dividend_yield": float(request.dividend_yield),
            "is_call": request.put_call.upper() == "CALL",
            "source_collected_at": request.source_collected_at,
            "expiration_at": request.expiration_at,
            **self.parameters,
        }

    def _greek_precision_diagnostics(
        self,
        request: PricingRequest,
        volatility: float,
        values: Mapping[str, float],
        *,
        requested_dtype: str,
    ) -> dict[str, float]:
        """Expose per-item finite-difference conditioning at the adapter edge."""
        if self.method not in (RepricingMethod.CRR, RepricingMethod.TIAN):
            return {}
        dtype = np.dtype(requested_dtype)
        epsilon = float(np.finfo(dtype).eps)
        price = float(values.get("model_price", math.nan))
        delta = float(values.get("delta", math.nan))
        gamma = float(values.get("gamma", math.nan))
        theta = float(values.get("theta_per_day", math.nan))
        raw_vega = float(values.get("vega", math.nan))
        raw_rho = float(values.get("rho", math.nan))
        tenor_years = float(request.tenor_seconds) / (365.25 * 86400.0)
        steps = max(1, int(self.parameters.get("steps", 1)))
        tree_spot_width = max(
            abs(float(request.spot))
            * max(abs(float(volatility)), epsilon)
            * math.sqrt(max(tenor_years, epsilon) / steps),
            epsilon,
        )
        vega_bump = (
            1.0e-4
            if self.method is RepricingMethod.CRR
            else min(0.001, max(0.00025, abs(float(volatility)) * 0.0025))
        )
        signals = {
            "gamma": abs(gamma) * tree_spot_width,
            "theta": abs(theta) * min(1.0, max(tenor_years * 365.25, epsilon)),
            "vega": abs(raw_vega) * vega_bump * 2.0,
            "rho": abs(raw_rho) * 2.0e-4,
        }
        scales = {
            "gamma": max(1.0, abs(delta), abs(delta) + signals["gamma"]),
            "theta": max(1.0, abs(price), abs(price) + signals["theta"]),
            "vega": max(1.0, abs(price), abs(price) + signals["vega"]),
            "rho": max(1.0, abs(price), abs(price) + signals["rho"]),
        }
        ratios = {
            name: signal / max(epsilon * scales[name], np.finfo(np.float64).tiny)
            for name, signal in signals.items()
        }
        invalid = not all(
            math.isfinite(value)
            for value in (
                request.spot,
                request.strike,
                request.tenor_seconds,
                request.rate,
                request.dividend_yield,
                volatility,
                price,
                delta,
                gamma,
                theta,
                raw_vega,
                raw_rho,
            )
        )
        diagnostics = {
            "precision_dtype_epsilon": epsilon,
            "precision_invalid_parameter_flags": float(invalid),
            "precision_min_signal_to_roundoff": min(ratios.values()),
        }
        for name in ("gamma", "theta", "vega", "rho"):
            diagnostics[f"precision_{name}_absolute_bump_numerator"] = signals[name]
            diagnostics[f"precision_{name}_maximum_constituent_scale"] = scales[name]
            diagnostics[f"precision_{name}_signal_to_roundoff"] = ratios[name]
        return diagnostics

    def create_pricing_session(
        self, request: PricingRequest, *, initial_iv: Optional[float] = None
    ) -> Optional[Any]:
        if self.method is not RepricingMethod.CRANK_NICOLSON:
            return None
        source_iv = initial_iv if initial_iv is not None else request.source_iv
        initial_volatility = (
            float(source_iv)
            if source_iv is not None
            and math.isfinite(float(source_iv))
            and float(source_iv) > 0.0
            else 0.20
        )
        return self.module.create_pricing_session(
            **self._scalar_kwargs(request),
            initial_volatility=initial_volatility,
        )

    def price(
        self,
        request: PricingRequest,
        volatility: float,
        *,
        pricing_session: Optional[Any] = None,
    ) -> float:
        if pricing_session is not None:
            return float(pricing_session.price(volatility))
        output = self.module.price_from_iv_batch(**self._batch_kwargs(request, volatility))
        return float(np.asarray(output).reshape(-1)[0])

    def price_many(
        self,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float] | np.ndarray,
        *,
        requested_dtype: str,
    ) -> np.ndarray:
        output = self.module.price_from_iv_batch(
            **self._aligned_batch_kwargs(
                requests, volatilities, requested_dtype=requested_dtype
            )
        )
        values = np.asarray(output).reshape(-1)
        if len(values) != len(requests):
            raise ValueError("solver price batch returned a misaligned result")
        return values

    def greeks(
        self,
        request: PricingRequest,
        volatility: float,
        *,
        pricing_session: Optional[Any] = None,
    ) -> dict[str, float]:
        if pricing_session is not None:
            output = pricing_session.greeks(volatility)
            return {
                name: float(getattr(output, name))
                for name in (
                    "model_price",
                    "delta",
                    "gamma",
                    "theta_per_day",
                    "vega_per_1pct",
                    "rho_per_1pct",
                )
            }
        greek_kwargs = self._batch_kwargs(request, volatility)
        if self.method in (RepricingMethod.CRR, RepricingMethod.TIAN):
            greek_kwargs["fused_greeks_enabled"] = self.parameters.get(
                "fused_greeks_enabled", True
            )
            greek_kwargs["fused_greeks_maximum_contracts"] = self.parameters.get(
                "fused_greeks_maximum_contracts", 0
            )
        output = self.module.greeks_from_iv_batch(**greek_kwargs)
        values = {name: float(np.asarray(value).reshape(-1)[0]) for name, value in output.items()}
        values.update(
            self._greek_precision_diagnostics(
                request,
                volatility,
                values,
                requested_dtype=request.requested_dtype,
            )
        )
        if self.method in (RepricingMethod.CRR, RepricingMethod.TIAN):
            values["vega_per_1pct"] = values.pop("vega") * 0.01
            values["rho_per_1pct"] = values.pop("rho") * 0.01
        return values

    def greeks_many(
        self,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float] | np.ndarray,
        *,
        requested_dtype: str,
    ) -> list[dict[str, float]]:
        if len(requests) == 1:
            # Scalar compatibility and isolated rollback path. Multi-item CRR
            # and Tian groups always use their aligned module kernels below.
            request = replace(requests[0], requested_dtype=requested_dtype)
            return [self.greeks(request, float(volatilities[0]))]
        if self.method is RepricingMethod.CRANK_NICOLSON:
            # Compatibility route is intentionally scalar-internal this release.
            return [
                self.greeks(request, float(volatility))
                for request, volatility in zip(requests, volatilities)
            ]
        greek_kwargs = self._aligned_batch_kwargs(
            requests, volatilities, requested_dtype=requested_dtype
        )
        if self.method in (RepricingMethod.CRR, RepricingMethod.TIAN):
            greek_kwargs["fused_greeks_enabled"] = self.parameters.get(
                "fused_greeks_enabled", True
            )
            greek_kwargs["fused_greeks_maximum_contracts"] = self.parameters.get(
                "fused_greeks_maximum_contracts", 0
            )
        output = self.module.greeks_from_iv_batch(**greek_kwargs)
        arrays = {name: np.asarray(value).reshape(-1) for name, value in output.items()}
        if any(len(value) != len(requests) for value in arrays.values()):
            raise ValueError("solver Greek batch returned a misaligned result")
        if self.method in (RepricingMethod.CRR, RepricingMethod.TIAN):
            arrays["vega_per_1pct"] = arrays.pop("vega") * 0.01
            arrays["rho_per_1pct"] = arrays.pop("rho") * 0.01
        results = [
            {name: float(values[index]) for name, values in arrays.items()}
            for index in range(len(requests))
        ]
        if self.method in (RepricingMethod.CRR, RepricingMethod.TIAN):
            for request, volatility, values in zip(requests, volatilities, results):
                raw_values = dict(values)
                raw_values["vega"] = raw_values["vega_per_1pct"] / 0.01
                raw_values["rho"] = raw_values["rho_per_1pct"] / 0.01
                values.update(
                    self._greek_precision_diagnostics(
                        request,
                        float(volatility),
                        raw_values,
                        requested_dtype=requested_dtype,
                    )
                )
        return results

    def delta(
        self,
        request: PricingRequest,
        volatility: float,
        *,
        pricing_session: Optional[Any] = None,
    ) -> float:
        """Return only delta without evaluating the remaining Greek tuple."""
        if pricing_session is not None:
            value = float(pricing_session.delta(volatility))
        else:
            output = self.module.price_delta_from_iv_batch(
                **self._batch_kwargs(request, volatility)
            )
            value = float(np.asarray(output["delta"]).reshape(-1)[0])
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite_solver_delta")
        return value

    def delta_many(
        self,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float] | np.ndarray,
        *,
        requested_dtype: str,
    ) -> np.ndarray:
        if self.method is RepricingMethod.CRANK_NICOLSON:
            return np.asarray(
                [
                    self.delta(request, float(volatility))
                    for request, volatility in zip(requests, volatilities)
                ],
                dtype=np.float64,
            )
        output = self.module.price_delta_from_iv_batch(
            **self._aligned_batch_kwargs(
                requests, volatilities, requested_dtype=requested_dtype
            )
        )
        values = np.asarray(output["delta"]).reshape(-1)
        if len(values) != len(requests):
            raise ValueError("solver delta batch returned a misaligned result")
        return values

    def solve_iv(
        self,
        request: PricingRequest,
        target_price: float,
        *,
        minimum: float,
        maximum: float,
        price_tolerance: float,
        max_iterations: int,
        source_bracket_multipliers: tuple[float, ...],
        iv_hints: tuple[IvHint, ...] = (),
        hint_bracket_multipliers: Optional[tuple[float, ...]] = None,
        hint_identifiability_enabled: bool = True,
        hint_identifiability_probe_iv_points: float = 1.0,
        hint_identifiability_minimum_price_span_absolute: float = 1.0e-6,
        hint_identifiability_price_tolerance_multiplier: float = 10.0,
        pricing_session: Optional[Any] = None,
    ) -> IvSolveResult:
        return solve_iv(
            target_price=target_price,
            price_function=lambda volatility: self.price(
                request, volatility, pricing_session=pricing_session
            ),
            source_iv=request.source_iv,
            iv_hints=iv_hints,
            minimum=minimum,
            maximum=maximum,
            price_tolerance=price_tolerance,
            max_iterations=max_iterations,
            source_bracket_multipliers=source_bracket_multipliers,
            hint_bracket_multipliers=hint_bracket_multipliers,
            hint_identifiability_enabled=hint_identifiability_enabled,
            hint_identifiability_probe_iv_points=hint_identifiability_probe_iv_points,
            hint_identifiability_minimum_price_span_absolute=(
                hint_identifiability_minimum_price_span_absolute
            ),
            hint_identifiability_price_tolerance_multiplier=(
                hint_identifiability_price_tolerance_multiplier
            ),
        )


class SolverRegistry:
    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        monotonic: Any = time.monotonic,
        utcnow: Any = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.context = context
        self.iv_config = _parse_iv_solver_config(context)
        self.batch_config = NumericalBatchConfig.from_context(context)
        self.circuit_config = CircuitBreakerConfig.from_context(context)
        self.batch_metrics: Counter[str] = Counter()
        self.batch_sizes: list[int] = []
        self._instances: dict[RepricingMethod, SolverAdapter] = {}
        self._circuits = {
            method: MethodCircuitBreaker(
                method, self.circuit_config, monotonic=monotonic, utcnow=utcnow
            )
            for method in RepricingMethod
            if method is not RepricingMethod.NONE
        }

    def preload(self) -> None:
        """Load configured numerical dependencies and adapter modules once."""
        from algotrading.refdata.options.Pricing import configured_methods

        routing = configured_methods(self.context)
        selected_methods = tuple(
            dict.fromkeys((routing.default_method, routing.short_tenor_method))
        )
        try:
            preload_iv_solver_dependencies()
        except (ImportError, ModuleNotFoundError) as exc:
            error = SolverUnavailableError("configured IV root solver is unavailable")
            if not self.circuit_config.enabled:
                raise error from exc
            for selected in selected_methods:
                self._circuits[selected].record_infrastructure_failure()
                self._circuits[selected].force_open()
            if not self.circuit_config.allow_worker_start_with_open_method:
                raise error from exc
            return
        for selected in selected_methods:
            try:
                self.get(selected)
            except SolverInfrastructureError:
                if not self.circuit_config.enabled:
                    raise
                circuit = self._circuits[selected]
                circuit.record_infrastructure_failure()
                circuit.force_open()
                if not self.circuit_config.allow_worker_start_with_open_method:
                    raise

    def _parameters(self, method: RepricingMethod) -> dict[str, Any]:
        prefix = OPTIONS_CALCS_GREEKS_PREFIX
        if method in (RepricingMethod.CRR, RepricingMethod.TIAN):
            method_prefix = prefix + method.value + "."
            legacy = method is RepricingMethod.CRR
            parameters = {
                "steps": int(
                    self.context.get(
                        method_prefix + "steps",
                        self.context.get(prefix + "steps", 256) if legacy else 256,
                    )
                ),
                "chunk_size": int(
                    self.context.get(
                        method_prefix + "chunk_size",
                        self.context.get(prefix + "chunk_size", 50000) if legacy else 50000,
                    )
                ),
                "device": str(
                    self.context.get(
                        method_prefix + "device",
                        self.context.get(prefix + "device", "cpu") if legacy else "cpu",
                    )
                ),
            }
            from algotrading.refdata.options.Common import _as_bool

            parameters["fused_greeks_enabled"] = _as_bool(
                self.context.get(method_prefix + "fused_greeks_enabled", True)
            )
            parameters["fused_greeks_maximum_contracts"] = int(
                self.context.get(
                    method_prefix + "fused_greeks_maximum_contracts",
                    160 if method is RepricingMethod.CRR else 0,
                )
            )
            if parameters["fused_greeks_maximum_contracts"] < 0:
                raise ValueError(
                    f"{method_prefix}fused_greeks_maximum_contracts must be nonnegative"
                )
            if method is RepricingMethod.TIAN:
                parameters.update(
                    vectorized_enabled=_as_bool(
                        self.context.get(method_prefix + "vectorized_enabled", True)
                    ),
                    scalar_fallback_enabled=_as_bool(
                        self.context.get(method_prefix + "scalar_fallback_enabled", True)
                    ),
                )
            return parameters
        method_prefix = prefix + "crank_nicolson."
        return {
            "time_grid": int(self.context.get(method_prefix + "time_grid", 100)),
            "space_grid": int(self.context.get(method_prefix + "space_grid", 200)),
            "damping_steps": int(self.context.get(method_prefix + "damping_steps", 4)),
            "vega_relative_bump": float(
                self.context.get(method_prefix + "vega_relative_bump", 0.0025)
            ),
            "vega_minimum_bump": float(
                self.context.get(method_prefix + "vega_minimum_bump", 0.00025)
            ),
            "vega_maximum_bump": float(
                self.context.get(method_prefix + "vega_maximum_bump", 0.001)
            ),
            "rho_bump": float(self.context.get(method_prefix + "rho_bump", 0.0001)),
        }

    def _module(self, selected: RepricingMethod, *, initialize: bool) -> Any:
        try:
            if selected is RepricingMethod.CRR:
                from algotrading.refdata.options.solvers import Crr as module
            elif selected is RepricingMethod.TIAN:
                from algotrading.refdata.options.solvers import Tian as module
            else:
                from algotrading.refdata.options.solvers import CrankNicolson as module
                if initialize:
                    module.load_quantlib()
            return module
        except (ImportError, ModuleNotFoundError) as exc:
            raise SolverUnavailableError(
                f"configured {selected.value} dependency is unavailable"
            ) from exc
        except MemoryError as exc:
            raise SolverResourceError(
                f"configured {selected.value} initialization exhausted memory"
            ) from exc
        except RuntimeError as exc:
            # This wrapper is deliberately limited to the known dependency-load
            # boundary. Generic RuntimeError elsewhere remains unknown.
            if selected is RepricingMethod.CRANK_NICOLSON and initialize:
                raise SolverUnavailableError(
                    "configured crank_nicolson dependency is unavailable"
                ) from exc
            raise

    def describe(
        self, method: RepricingMethod | str
    ) -> tuple[str, Mapping[str, Any]]:
        selected = method if isinstance(method, RepricingMethod) else RepricingMethod.parse(method)
        if selected is RepricingMethod.NONE:
            raise ValueError("No solver exists for repricing method none")
        module = self._module(selected, initialize=False)
        return str(module.ALGORITHM_VERSION), self._parameters(selected)

    def get(self, method: RepricingMethod | str) -> SolverAdapter:
        selected = method if isinstance(method, RepricingMethod) else RepricingMethod.parse(method)
        if selected is RepricingMethod.NONE:
            raise ValueError("No solver exists for repricing method none")
        cached = self._instances.get(selected)
        if cached is not None:
            return cached
        module = self._module(selected, initialize=True)
        adapter = SolverAdapter(selected, module, self._parameters(selected))
        self._instances[selected] = adapter
        return adapter

    def circuit(self, method: RepricingMethod | str) -> MethodCircuitBreaker:
        selected = method if isinstance(method, RepricingMethod) else RepricingMethod.parse(method)
        return self._circuits[selected]

    def circuit_snapshots(self) -> Mapping[str, Mapping[str, Any]]:
        return {
            method.value: dict(circuit.snapshot())
            for method, circuit in self._circuits.items()
        }

    def _actual_dtype(self, method: RepricingMethod, requested_dtype: str) -> str:
        if method is RepricingMethod.CRANK_NICOLSON:
            return "float64"
        if (
            method is RepricingMethod.TIAN
            and not bool(self._parameters(method).get("vectorized_enabled", True))
        ):
            # The retained scalar Tian implementation computes in NumPy float64.
            return "float64"
        name = str(requested_dtype).strip().lower()
        if name not in {"float32", "float64"}:
            raise ValueError(f"unsupported requested dtype {requested_dtype!r}")
        return name

    def _maximum_chunk_jobs(
        self,
        adapter: SolverAdapter,
        operation: str,
        requested_dtype: str,
    ) -> int:
        width = 4 if requested_dtype == "float32" else 8
        steps = int(adapter.parameters.get("steps", 1))
        buffers = {
            "price": 3,
            "delta": 5,
            "greeks": 12,
            "solve_iv_only": 4,
            "solve_iv_and_greeks": 12,
        }.get(operation, 12)
        estimated_per_job = max(1, (steps + 1) * width * buffers)
        memory_limit = max(
            1, self.batch_config.maximum_estimated_bytes // estimated_per_job
        )
        return max(1, min(self.batch_config.maximum_jobs, memory_limit))

    def _job_groups(
        self, jobs: Sequence[SolverBatchJob]
    ) -> list[tuple[tuple[Any, ...], list[tuple[int, SolverBatchJob]]]]:
        grouped: dict[tuple[Any, ...], list[tuple[int, SolverBatchJob]]] = {}
        for index, job in enumerate(jobs):
            adapter = self.get(job.method)
            actual_dtype = self._actual_dtype(job.method, job.requested_dtype)
            device = str(adapter.parameters.get("device", "cpu"))
            key = (
                job.operation.value,
                job.method.value,
                actual_dtype,
                job.solver_configuration_hash or adapter.configuration_hash,
                adapter.algorithm_version,
                device,
            )
            grouped.setdefault(key, []).append((index, job))
        return list(grouped.items())

    def _iv_problem(self, job: SolverBatchJob) -> IvBatchProblem:
        return IvBatchProblem(
            target_price=float(job.target_price),
            source_iv=job.request.source_iv,
            iv_hints=job.iv_hints,
            minimum=self.iv_config.minimum,
            maximum=self.iv_config.maximum,
            price_tolerance=self.iv_config.price_tolerance,
            max_iterations=self.iv_config.max_iterations,
            source_bracket_multipliers=self.iv_config.source_bracket_multipliers,
            hint_bracket_multipliers=(
                self.iv_config.prior_seed_bracket_multipliers
                if job.iv_hints
                and job.iv_hints[0].origin in {"prior_success", "prior_partial_failure"}
                else self.iv_config.source_bracket_multipliers
            ),
            hint_identifiability_enabled=self.iv_config.hint_identifiability_enabled,
            hint_identifiability_probe_iv_points=(
                self.iv_config.hint_identifiability_probe_iv_points
            ),
            hint_identifiability_minimum_price_span_absolute=(
                self.iv_config.hint_identifiability_minimum_price_span_absolute
            ),
            hint_identifiability_price_tolerance_multiplier=(
                self.iv_config.hint_identifiability_price_tolerance_multiplier
            ),
        )

    @staticmethod
    def _failed_iv_for_exception(exc: BaseException) -> IvSolveResult:
        if isinstance(exc, SolverCircuitOpenError):
            reason = "solver_circuit_open"
        elif isinstance(exc, SolverNumericalError):
            reason = "numerical_evaluation_failed"
        elif isinstance(exc, SolverInfrastructureError):
            reason = "method_unavailable"
        else:
            reason = f"solver_exception:{type(exc).__name__}"
        return IvSolveResult("failed", reason, None, 0, None, None)

    def _solve_iv_vector_group(
        self,
        jobs: Sequence[SolverBatchJob],
        *,
        actual_dtype: str,
    ) -> list[IvSolveResult]:
        if not jobs:
            return []
        adapter = self.get(jobs[0].method)
        problems = [self._iv_problem(job) for job in jobs]

        def price_wave(indices: Sequence[int], sigmas: np.ndarray) -> np.ndarray:
            requests = [jobs[index].request for index in indices]
            self.batch_metrics["iv_price_physical_calls"] += 1
            self.batch_metrics["physical_calls"] += 1
            self.batch_sizes.append(len(indices))
            values = adapter.price_many(
                requests, sigmas, requested_dtype=actual_dtype
            )
            self.batch_metrics["successful_physical_calls"] += 1
            return values

        return solve_iv_many(problems, price_wave)

    def _solve_iv_group_isolated(
        self,
        jobs: Sequence[SolverBatchJob],
        *,
        actual_dtype: str,
    ) -> list[IvSolveResult]:
        try:
            return self._solve_iv_vector_group(jobs, actual_dtype=actual_dtype)
        except BaseException as exc:
            self.batch_metrics["physical_failures"] += 1
            if isinstance(exc, SolverInfrastructureError):
                self.batch_metrics["infrastructure_failures"] += 1
            elif isinstance(exc, SolverNumericalError):
                self.batch_metrics["item_numerical_failures"] += len(jobs)
            else:
                self.batch_metrics["unknown_outcome_failures"] += 1
            if (
                self.batch_config.exception_isolation_enabled
                and len(jobs) > 1
                and not isinstance(
                    exc, (SolverUnavailableError, SolverCircuitOpenError)
                )
            ):
                self.batch_metrics["exception_splits"] += 1
                middle = len(jobs) // 2
                return self._solve_iv_group_isolated(
                    jobs[:middle], actual_dtype=actual_dtype
                ) + self._solve_iv_group_isolated(
                    jobs[middle:], actual_dtype=actual_dtype
                )
            if (
                len(jobs) == 1
                and self.batch_config.scalar_rollback_enabled
                and not isinstance(
                    exc, (SolverUnavailableError, SolverCircuitOpenError)
                )
            ):
                job = jobs[0]
                adapter = self.get(job.method)
                try:
                    self.batch_metrics["scalar_rollback_calls"] += 1
                    solved = adapter.solve_iv(
                        replace(job.request, requested_dtype=actual_dtype),
                        float(job.target_price),
                        minimum=self.iv_config.minimum,
                        maximum=self.iv_config.maximum,
                        price_tolerance=self.iv_config.price_tolerance,
                        max_iterations=self.iv_config.max_iterations,
                        source_bracket_multipliers=(
                            self.iv_config.source_bracket_multipliers
                        ),
                        iv_hints=job.iv_hints,
                        hint_bracket_multipliers=(
                            self.iv_config.prior_seed_bracket_multipliers
                            if job.iv_hints
                            and job.iv_hints[0].origin
                            in {"prior_success", "prior_partial_failure"}
                            else self.iv_config.source_bracket_multipliers
                        ),
                        hint_identifiability_enabled=(
                            self.iv_config.hint_identifiability_enabled
                        ),
                        hint_identifiability_probe_iv_points=(
                            self.iv_config.hint_identifiability_probe_iv_points
                        ),
                        hint_identifiability_minimum_price_span_absolute=(
                            self.iv_config.hint_identifiability_minimum_price_span_absolute
                        ),
                        hint_identifiability_price_tolerance_multiplier=(
                            self.iv_config.hint_identifiability_price_tolerance_multiplier
                        ),
                    )
                    self.batch_metrics["successful_scalar_rollbacks"] += 1
                    return [solved]
                except BaseException:
                    self.batch_metrics["scalar_rollback_failures"] += 1
            return [self._failed_iv_for_exception(exc) for _job in jobs]

    def _repricing_from_iv(
        self,
        job: SolverBatchJob,
        solved: IvSolveResult,
        values: Optional[Mapping[str, float]],
        *,
        actual_dtype: str,
        precision_escalated: bool = False,
    ) -> RepricingResult:
        adapter = self.get(job.method)
        base = dict(
            method=job.method.value,
            method_selection_reason=job.method_selection_reason,
            algorithm_version=adapter.algorithm_version,
            input_basis="market_mid",
            target_price=float(job.target_price),
            input_iv=solved.iv,
            implied_volatility=solved.iv,
            iv_iterations=solved.iterations,
            price_residual=solved.price_residual,
            requested_dtype=job.requested_dtype,
            actual_dtype=actual_dtype,
            precision_escalated=precision_escalated,
            precision_diagnostics=(
                {
                    str(name): float(value)
                    for name, value in (values or {}).items()
                    if str(name).startswith("precision_")
                    and value is not None
                    and math.isfinite(float(value))
                }
            ),
            solver_parameters=adapter.parameters,
            iv_hint_origin_used=solved.hint_origin_used,
            iv_hint_value_used=solved.hint_value_used,
            iv_hint_exact_match=solved.hint_exact_match,
            iv_hint_local_bracket_found=solved.hint_local_bracket_found,
            iv_hint_evaluations=solved.hint_evaluations,
            iv_fallback_global_probes_used=solved.fallback_global_probes_used,
            iv_price_evaluations=solved.price_evaluations,
            iv_identifiability_status=solved.identifiability_status,
            iv_identifiability_price_span=solved.identifiability_price_span,
            iv_identifiability_slope=solved.identifiability_slope,
            iv_hint_flat_rejection_count=solved.hint_flat_rejection_count,
            quality_flags=(
                QUALITY_IV_PRICE_INDETERMINATE
                if solved.identifiability_status == "source_anchor_flat"
                else 0
            ),
        )
        if solved.status != "complete" or solved.iv is None:
            return RepricingResult(
                status="failed",
                reason=solved.reason,
                failure_stage=(
                    "initialization"
                    if solved.reason in {"solver_circuit_open", "method_unavailable"}
                    or solved.reason.startswith("solver_exception:")
                    else "iv_inversion"
                ),
                model_price=solved.model_price,
                delta=None,
                gamma=None,
                theta_per_day=None,
                vega_per_1pct=None,
                rho_per_1pct=None,
                **base,
            )
        required_names = (
            "model_price", "delta", "gamma", "theta_per_day",
            "vega_per_1pct", "rho_per_1pct",
        )
        if values is None or not all(
            values.get(name) is not None and math.isfinite(float(values[name]))
            for name in required_names
        ):
            return RepricingResult(
                status="failed", reason="nonfinite_solver_output",
                failure_stage="greeks", model_price=solved.model_price,
                delta=None, gamma=None, theta_per_day=None,
                vega_per_1pct=None, rho_per_1pct=None, **base,
            )
        residual = float(values["model_price"]) - float(job.target_price)
        status = "complete" if abs(residual) <= self.iv_config.price_tolerance else "failed"
        return RepricingResult(
            status=status,
            reason="" if status == "complete" else "price_residual_exceeds_tolerance",
            failure_stage="none" if status == "complete" else "residual_validation",
            price_residual=residual,
            model_price=float(values["model_price"]),
            delta=float(values["delta"]),
            gamma=float(values["gamma"]),
            theta_per_day=float(values["theta_per_day"]),
            vega_per_1pct=float(values["vega_per_1pct"]),
            rho_per_1pct=float(values["rho_per_1pct"]),
            **{key: value for key, value in base.items() if key != "price_residual"},
        )

    def _solve_combined_group(
        self,
        jobs: Sequence[SolverBatchJob],
        *,
        actual_dtype: str,
    ) -> list[RepricingResult]:
        adapter = self.get(jobs[0].method)
        solved = self._solve_iv_group_isolated(jobs, actual_dtype=actual_dtype)
        greek_values: list[Optional[Mapping[str, float]]] = [None] * len(jobs)
        greek_errors: list[Optional[BaseException]] = [None] * len(jobs)
        successful = [
            index for index, item in enumerate(solved)
            if item.status == "complete" and item.iv is not None
        ]

        def evaluate_greeks(indices: Sequence[int]) -> None:
            if not indices:
                return
            requests = [jobs[index].request for index in indices]
            volatilities = np.ascontiguousarray(
                [float(solved[index].iv) for index in indices], dtype=np.float64
            )
            try:
                self.batch_metrics["greeks_physical_calls"] += 1
                self.batch_metrics["physical_calls"] += 1
                self.batch_sizes.append(len(indices))
                returned = adapter.greeks_many(
                    requests, volatilities, requested_dtype=actual_dtype
                )
                self.batch_metrics["successful_physical_calls"] += 1
                for index, values in zip(indices, returned):
                    greek_values[index] = values
            except BaseException as exc:
                self.batch_metrics["physical_failures"] += 1
                if isinstance(exc, SolverInfrastructureError):
                    self.batch_metrics["infrastructure_failures"] += 1
                elif isinstance(exc, SolverNumericalError):
                    self.batch_metrics["item_numerical_failures"] += len(indices)
                else:
                    self.batch_metrics["unknown_outcome_failures"] += 1
                if (
                    self.batch_config.exception_isolation_enabled
                    and len(indices) > 1
                    and not isinstance(
                        exc, SolverCircuitOpenError
                    )
                ):
                    self.batch_metrics["exception_splits"] += 1
                    middle = len(indices) // 2
                    evaluate_greeks(indices[:middle])
                    evaluate_greeks(indices[middle:])
                    return
                if len(indices) == 1 and self.batch_config.scalar_rollback_enabled:
                    index = indices[0]
                    try:
                        self.batch_metrics["scalar_rollback_calls"] += 1
                        greek_values[index] = adapter.greeks(
                            replace(
                                jobs[index].request,
                                requested_dtype=actual_dtype,
                            ),
                            float(solved[index].iv),
                        )
                        self.batch_metrics["successful_scalar_rollbacks"] += 1
                        return
                    except BaseException as rollback_exc:
                        self.batch_metrics["scalar_rollback_failures"] += 1
                        greek_errors[index] = rollback_exc
                        return
                for index in indices:
                    greek_errors[index] = exc

        evaluate_greeks(successful)
        return [
            self._repricing_from_iv(
                job,
                (
                    self._failed_iv_for_exception(greek_errors[index])
                    if greek_errors[index] is not None
                    else item
                ),
                greek_values[index],
                actual_dtype=actual_dtype,
            )
            for index, (job, item) in enumerate(zip(jobs, solved))
        ]

    def _float32_guard_conditioning_flag(
        self, job: SolverBatchJob, result: RepricingResult
    ) -> bool:
        """Select finite results whose per-item bump signal is cancellation-prone."""
        if result.status != "complete":
            return True
        diagnostics = result.precision_diagnostics
        minimum_ratio = diagnostics.get("precision_min_signal_to_roundoff")
        invalid_flags = diagnostics.get("precision_invalid_parameter_flags")
        if invalid_flags is not None and float(invalid_flags) != 0.0:
            return True
        if minimum_ratio is not None:
            return (
                float(minimum_ratio)
                < self.batch_config.float32_guard.minimum_signal_to_roundoff
            )
        scale = max(
            1.0,
            abs(float(result.model_price or 0.0)),
            abs(float(job.request.spot)),
        )
        roundoff_floor = (
            np.finfo(np.float32).eps
            * scale
            * self.batch_config.float32_guard.minimum_signal_to_roundoff
        )
        return any(
            abs(float(value or 0.0)) <= roundoff_floor
            for value in (
                result.gamma,
                result.theta_per_day,
                result.vega_per_1pct,
                result.rho_per_1pct,
            )
        )

    def _float32_guard_shadow_selected(self, job: SolverBatchJob) -> bool:
        fraction = self.batch_config.float32_guard.shadow_fraction
        if fraction <= 0.0:
            return False
        digest = hashlib.sha256(
            (
                f"{job.owner_snapshot_id}|{job.request.option_symbol}|"
                f"{job.method.value}|{job.output_index}"
            ).encode("utf-8")
        ).digest()
        sample = int.from_bytes(digest[:8], "big") / float(2**64)
        return sample < fraction

    def _float32_guard_reason(
        self, requested: RepricingResult, reference: RepricingResult
    ) -> str:
        if requested.status != "complete":
            return "float32_failed_or_nonfinite"
        if reference.status != "complete":
            return "none"
        guard = self.batch_config.float32_guard
        tolerances = {
            "model_price": guard.price_atol,
            "delta": guard.delta_atol,
            "gamma": guard.gamma_atol,
            "theta_per_day": guard.theta_per_day_atol,
            "vega_per_1pct": guard.vega_per_1pct_atol,
            "rho_per_1pct": guard.rho_per_1pct_atol,
        }
        for name, atol in tolerances.items():
            left = getattr(requested, name)
            right = getattr(reference, name)
            if left is None or right is None:
                return f"float32_guard_{name}_missing"
            if not math.isclose(
                float(left), float(right), rel_tol=guard.rtol, abs_tol=atol
            ):
                return f"float32_guard_{name}_mismatch"
        return "none"

    def _apply_float32_precision_guard(
        self,
        jobs: Sequence[SolverBatchJob],
        results: list[RepricingResult],
        *,
        already_retried: set[int],
    ) -> None:
        guard = self.batch_config.float32_guard
        if not guard.enabled:
            return
        candidates: list[int] = []
        for index, (job, result) in enumerate(zip(jobs, results)):
            if index in already_retried:
                continue
            if (
                job.method not in (RepricingMethod.CRR, RepricingMethod.TIAN)
                or self._actual_dtype(job.method, job.requested_dtype) != "float32"
                or result.status != "complete"
            ):
                continue
            short_tenor = (
                guard.short_tenor_validation_enabled
                and float(job.request.tenor_seconds) <= guard.short_tenor_max_seconds
            )
            if (
                short_tenor
                or self._float32_guard_conditioning_flag(job, result)
                or self._float32_guard_shadow_selected(job)
            ):
                candidates.append(index)
        if not candidates:
            return
        self.batch_metrics["float32_guard_validation_jobs"] += len(candidates)
        references = self._solve_combined_group(
            [jobs[index] for index in candidates], actual_dtype="float64"
        )
        for index, reference in zip(candidates, references):
            reason = self._float32_guard_reason(results[index], reference)
            if reason == "none":
                self.batch_metrics["float32_guard_stable_jobs"] += 1
                continue
            results[index] = replace(
                reference,
                requested_dtype=jobs[index].requested_dtype,
                actual_dtype="float64",
                precision_escalated=True,
                precision_escalation_reason=reason,
                precision_diagnostics=results[index].precision_diagnostics,
            )
            self.batch_metrics["float32_guard_escalated_jobs"] += 1

    def _account_repricing_circuit_results(
        self,
        jobs: Sequence[SolverBatchJob],
        results: list[RepricingResult],
    ) -> None:
        """Record one circuit outcome for each logical IV+Greek item."""
        for index, (job, result) in enumerate(zip(jobs, results)):
            if not job.consult_circuit:
                continue
            circuit = self.circuit(job.method)
            infrastructure = result.reason == "method_unavailable"
            unknown = result.reason.startswith("solver_exception:")
            if (infrastructure or unknown) and (
                circuit.snapshot()["state"] == CircuitState.OPEN.value
            ):
                results[index] = replace(
                    result,
                    status="failed",
                    reason="solver_circuit_open",
                    failure_stage="initialization",
                    model_price=None,
                    delta=None,
                    gamma=None,
                    theta_per_day=None,
                    vega_per_1pct=None,
                    rho_per_1pct=None,
                )
                self.batch_metrics["circuit_suppressions"] += 1
                self.batch_metrics["logical_jobs_suppressed"] += 1
            elif infrastructure:
                circuit.record_infrastructure_failure(1)
                self.batch_metrics["logical_infrastructure_failures"] += 1
            elif unknown:
                circuit.record_unknown_outcome()
                self.batch_metrics["logical_unknown_outcomes"] += 1
            else:
                # Deterministic numerical failures still prove that the
                # dependency was reached for this logical item.
                circuit.record_infrastructure_success(1)
                self.batch_metrics["logical_circuit_successes"] += 1

    def _account_iv_circuit_results(
        self,
        jobs: Sequence[SolverBatchJob],
        results: list[IvSolveResult],
    ) -> None:
        """IV-only counterpart of per-logical-item circuit accounting."""
        for index, (job, result) in enumerate(zip(jobs, results)):
            if not job.consult_circuit:
                continue
            circuit = self.circuit(job.method)
            infrastructure = result.reason == "method_unavailable"
            unknown = result.reason.startswith("solver_exception:")
            if (infrastructure or unknown) and (
                circuit.snapshot()["state"] == CircuitState.OPEN.value
            ):
                results[index] = replace(result, status="failed", reason="solver_circuit_open")
                self.batch_metrics["circuit_suppressions"] += 1
                self.batch_metrics["logical_jobs_suppressed"] += 1
            elif infrastructure:
                circuit.record_infrastructure_failure(1)
                self.batch_metrics["logical_infrastructure_failures"] += 1
            elif unknown:
                circuit.record_unknown_outcome()
                self.batch_metrics["logical_unknown_outcomes"] += 1
            else:
                circuit.record_infrastructure_success(1)
                self.batch_metrics["logical_circuit_successes"] += 1

    def solve_iv_and_greeks_many(
        self, jobs: Sequence[SolverBatchJob]
    ) -> list[RepricingResult]:
        """Execute ordered, independently owned IV+Greek jobs in aligned groups."""
        if not jobs:
            return []
        if not self.batch_config.enabled:
            results: list[RepricingResult] = []
            for job in jobs:
                try:
                    results.append(
                        self.solve_iv_and_greeks(
                            job.request,
                            method=job.method,
                            method_selection_reason=job.method_selection_reason,
                            target_price=float(job.target_price),
                            iv_hints=job.iv_hints,
                        )
                    )
                except BaseException as exc:
                    results.append(
                        self._repricing_from_iv(
                            job,
                            self._failed_iv_for_exception(exc),
                            None,
                            actual_dtype=self._actual_dtype(
                                job.method, job.requested_dtype
                            ),
                        )
                    )
            self.batch_metrics["scalar_internal_calls"] += len(jobs)
            return results
        output: list[Optional[RepricingResult]] = [None] * len(jobs)
        for _key, group in self._job_groups(jobs):
            adapter = self.get(group[0][1].method)
            actual_dtype = self._actual_dtype(
                group[0][1].method, group[0][1].requested_dtype
            )
            maximum = self._maximum_chunk_jobs(
                adapter, "solve_iv_and_greeks", actual_dtype
            )
            if group[0][1].method is RepricingMethod.CRANK_NICOLSON:
                # QuantLib's useful unit of reuse is one contract session, not
                # an aligned contract wave.  Keep IV probes and final Greeks
                # inside the scalar combined lifecycle so every attempted
                # contract constructs exactly one FDM session.  That lifecycle
                # also owns circuit authorization/accounting; doing either in
                # this outer batch layer would double-count logical calls.
                for start in range(0, len(group), maximum):
                    chunk = group[start : start + maximum]
                    self.batch_metrics["chunks"] += 1
                    for index, job in chunk:
                        try:
                            result = self._solve_iv_and_greeks_scalar(
                                job.request,
                                method=job.method,
                                method_selection_reason=job.method_selection_reason,
                                target_price=float(job.target_price),
                                iv_hints=job.iv_hints,
                                consult_circuit=job.consult_circuit,
                            )
                        except BaseException as exc:
                            result = self._repricing_from_iv(
                                job,
                                self._failed_iv_for_exception(exc),
                                None,
                                actual_dtype=actual_dtype,
                            )
                            if isinstance(exc, SolverCircuitOpenError):
                                self.batch_metrics["circuit_suppressions"] += 1
                                self.batch_metrics["logical_jobs_suppressed"] += 1
                        else:
                            self.batch_metrics["logical_jobs_authorized"] += 1
                        output[index] = result
                        self.batch_metrics["scalar_internal_calls"] += 1
                continue
            for start in range(0, len(group), maximum):
                chunk = group[start : start + maximum]
                allowed: list[tuple[int, SolverBatchJob]] = []
                for index, job in chunk:
                    if job.consult_circuit:
                        try:
                            self.circuit(job.method).before_call()
                        except SolverCircuitOpenError as exc:
                            output[index] = self._repricing_from_iv(
                                job,
                                self._failed_iv_for_exception(exc),
                                None,
                                actual_dtype=actual_dtype,
                            )
                            self.batch_metrics["circuit_suppressions"] += 1
                            self.batch_metrics["logical_jobs_suppressed"] += 1
                            continue
                    allowed.append((index, job))
                    self.batch_metrics["logical_jobs_authorized"] += 1
                if not allowed:
                    continue
                chunk_jobs = [job for _index, job in allowed]
                self.batch_metrics["chunks"] += 1
                chunk_results = self._solve_combined_group(
                    chunk_jobs, actual_dtype=actual_dtype
                )
                retry_positions = [
                    position
                    for position, result in enumerate(chunk_results)
                    if actual_dtype == "float32"
                    and chunk_jobs[position].method
                    in (RepricingMethod.CRR, RepricingMethod.TIAN)
                    and result.status == "failed"
                    and result.reason
                    in {
                        "nonfinite_model_price",
                        "nonfinite_solver_output",
                        "numerical_evaluation_failed",
                        "iv_price_tolerance_not_reached",
                        "price_residual_exceeds_tolerance",
                    }
                ]
                if retry_positions and self.batch_config.float64_retry_enabled:
                    retry_jobs = [chunk_jobs[position] for position in retry_positions]
                    self.batch_metrics["float64_retry_jobs"] += len(retry_jobs)
                    retried = self._solve_combined_group(
                        retry_jobs, actual_dtype="float64"
                    )
                    for position, retried_result in zip(retry_positions, retried):
                        chunk_results[position] = replace(
                            retried_result,
                            requested_dtype=chunk_jobs[position].requested_dtype,
                            actual_dtype="float64",
                            precision_escalated=True,
                            precision_escalation_reason=(
                                "float32_failed_or_nonfinite"
                            ),
                        )
                self._apply_float32_precision_guard(
                    chunk_jobs,
                    chunk_results,
                    already_retried=set(retry_positions),
                )
                self._account_repricing_circuit_results(
                    chunk_jobs, chunk_results
                )
                for (index, _job), result in zip(allowed, chunk_results):
                    output[index] = result
        if any(result is None for result in output):
            raise AssertionError("solver batch fan-out left an unowned result")
        return [result for result in output if result is not None]

    def solve_iv_only_many(
        self, jobs: Sequence[SolverBatchJob]
    ) -> list[IvSolveResult]:
        """Ordered IV-only execution; CRR/Tian probes are aligned in waves."""
        if not jobs:
            return []
        if not self.batch_config.enabled:
            results = []
            for job in jobs:
                try:
                    solved = self.solve_iv_only(
                        job.request,
                        method=job.method,
                        target_price=float(job.target_price),
                        iv_hints=job.iv_hints,
                        consult_circuit=job.consult_circuit,
                    )
                except BaseException as exc:
                    solved = self._failed_iv_for_exception(exc)
                results.append(
                    replace(
                        solved,
                        requested_dtype=job.requested_dtype,
                        actual_dtype=(
                            solved.actual_dtype
                            if solved.precision_escalated
                            else self._actual_dtype(
                                job.method, job.requested_dtype
                            )
                        ),
                    )
                )
            self.batch_metrics["scalar_internal_calls"] += len(jobs)
            return results
        output: list[Optional[IvSolveResult]] = [None] * len(jobs)
        for _key, group in self._job_groups(jobs):
            adapter = self.get(group[0][1].method)
            actual_dtype = self._actual_dtype(
                group[0][1].method, group[0][1].requested_dtype
            )
            maximum = self._maximum_chunk_jobs(adapter, "solve_iv_only", actual_dtype)
            for start in range(0, len(group), maximum):
                chunk = group[start : start + maximum]
                allowed: list[tuple[int, SolverBatchJob]] = []
                for index, job in chunk:
                    if job.consult_circuit:
                        try:
                            self.circuit(job.method).before_call()
                        except SolverCircuitOpenError as exc:
                            output[index] = replace(
                                self._failed_iv_for_exception(exc),
                                requested_dtype=job.requested_dtype,
                                actual_dtype=actual_dtype,
                            )
                            self.batch_metrics["circuit_suppressions"] += 1
                            self.batch_metrics["logical_jobs_suppressed"] += 1
                            continue
                    allowed.append((index, job))
                    self.batch_metrics["logical_jobs_authorized"] += 1
                if not allowed:
                    continue
                if allowed[0][1].method is RepricingMethod.CRANK_NICOLSON:
                    solved = []
                    for _index, job in allowed:
                        try:
                            solved.append(
                                self.solve_iv_only(
                                    job.request,
                                    method=job.method,
                                    target_price=float(job.target_price),
                                    iv_hints=job.iv_hints,
                                    consult_circuit=False,
                                )
                            )
                        except BaseException as exc:
                            solved.append(self._failed_iv_for_exception(exc))
                    self.batch_metrics["scalar_internal_calls"] += len(solved)
                else:
                    solved = self._solve_iv_group_isolated(
                        [job for _index, job in allowed], actual_dtype=actual_dtype
                    )
                solved = [
                    replace(
                        item,
                        requested_dtype=job.requested_dtype,
                        actual_dtype=actual_dtype,
                        precision_escalated=False,
                    )
                    for (_index, job), item in zip(allowed, solved)
                ]
                retry_positions = [
                    position
                    for position, item in enumerate(solved)
                    if actual_dtype == "float32"
                    and allowed[position][1].method
                    in (RepricingMethod.CRR, RepricingMethod.TIAN)
                    and item.status == "failed"
                    and item.reason in {
                        "nonfinite_model_price",
                        "numerical_evaluation_failed",
                        "iv_price_tolerance_not_reached",
                    }
                ]
                if retry_positions and self.batch_config.float64_retry_enabled:
                    retry_jobs = [allowed[position][1] for position in retry_positions]
                    retried = self._solve_iv_group_isolated(
                        retry_jobs, actual_dtype="float64"
                    )
                    self.batch_metrics["float64_retry_jobs"] += len(retry_jobs)
                    for position, item in zip(retry_positions, retried):
                        solved[position] = replace(
                            item,
                            requested_dtype=allowed[position][1].requested_dtype,
                            actual_dtype="float64",
                            precision_escalated=True,
                        )
                self._account_iv_circuit_results(
                    [job for _index, job in allowed], solved
                )
                for (index, job), item in zip(allowed, solved):
                    output[index] = item
        if any(result is None for result in output):
            raise AssertionError("IV batch fan-out left an unowned result")
        return [result for result in output if result is not None]

    @staticmethod
    def _aligned_argument(
        value: RepricingMethod | str | Sequence[RepricingMethod | str], count: int
    ) -> list[RepricingMethod]:
        if isinstance(value, (RepricingMethod, str)):
            parsed = value if isinstance(value, RepricingMethod) else RepricingMethod.parse(value)
            return [parsed] * count
        parsed = [
            item if isinstance(item, RepricingMethod) else RepricingMethod.parse(item)
            for item in value
        ]
        if len(parsed) != count:
            raise ValueError("methods must align with requests")
        return parsed

    @staticmethod
    def _failed_valuation_value(operation: str) -> Any:
        if operation == "greeks":
            return {
                name: math.nan
                for name in (
                    "model_price",
                    "delta",
                    "gamma",
                    "theta_per_day",
                    "vega_per_1pct",
                    "rho_per_1pct",
                )
            }
        return math.nan

    def _valuation_result_for_value(
        self,
        *,
        operation: str,
        value: Any,
        method: RepricingMethod,
        requested_dtype: str,
        actual_dtype: str,
        performance: Optional[Mapping[str, Any]] = None,
    ) -> ValuationResult:
        solver_operation = SolverOperation(operation)
        if operation == "greeks":
            required = (
                "model_price",
                "delta",
                "gamma",
                "theta_per_day",
                "vega_per_1pct",
                "rho_per_1pct",
            )
            usable = isinstance(value, Mapping) and all(
                value.get(name) is not None
                and math.isfinite(float(value[name]))
                for name in required
            )
            normalized = (
                {name: float(value[name]) for name in required}
                if usable
                else None
            )
        else:
            try:
                normalized = float(value)
                usable = math.isfinite(normalized)
            except (TypeError, ValueError, OverflowError):
                normalized = None
                usable = False
        if usable:
            return ValuationResult(
                operation=solver_operation,
                status="complete",
                reason="",
                failure_stage="none",
                retry_class=RecalcRetryClass.NONE,
                method=method.value,
                requested_dtype=requested_dtype,
                actual_dtype=actual_dtype,
                precision_escalated=False,
                circuit_outcome=CircuitCallOutcome.SUCCESS,
                value=normalized,
                performance=dict(performance or {}),
            )
        reason = "nonfinite_solver_output"
        return ValuationResult(
            operation=solver_operation,
            status="failed",
            reason=reason,
            failure_stage="valuation",
            retry_class=classify_repricing_failure(
                status="failed", reason=reason, failure_stage="valuation"
            ),
            method=method.value,
            requested_dtype=requested_dtype,
            actual_dtype=actual_dtype,
            precision_escalated=False,
            circuit_outcome=CircuitCallOutcome.SUCCESS,
            value=None,
            performance=dict(performance or {}),
        )

    def _valuation_result_for_exception(
        self,
        *,
        operation: str,
        exc: BaseException,
        method: RepricingMethod,
        requested_dtype: str,
        actual_dtype: str,
        performance: Optional[Mapping[str, Any]] = None,
    ) -> ValuationResult:
        if isinstance(exc, SolverCircuitOpenError):
            reason = "solver_circuit_open"
            stage = "initialization"
            circuit_outcome = CircuitCallOutcome.SUPPRESSED
        elif isinstance(exc, SolverUnavailableError):
            reason = "method_unavailable"
            stage = "initialization"
            circuit_outcome = CircuitCallOutcome.INFRASTRUCTURE_FAILURE
        elif isinstance(exc, (SolverInfrastructureError, MemoryError)):
            reason = "solver_infrastructure_failure"
            stage = "initialization"
            circuit_outcome = CircuitCallOutcome.INFRASTRUCTURE_FAILURE
        elif isinstance(exc, (SolverNumericalError, FloatingPointError)):
            reason = "numerical_evaluation_failed"
            stage = "valuation"
            circuit_outcome = CircuitCallOutcome.SUCCESS
        else:
            reason = f"solver_exception:{type(exc).__name__}"
            stage = "execution"
            circuit_outcome = CircuitCallOutcome.UNKNOWN_OUTCOME
        return ValuationResult(
            operation=SolverOperation(operation),
            status="failed",
            reason=reason,
            failure_stage=stage,
            retry_class=classify_repricing_failure(
                status="failed",
                reason=reason,
                failure_stage=stage,
                exception=exc,
            ),
            method=method.value,
            requested_dtype=requested_dtype,
            actual_dtype=actual_dtype,
            precision_escalated=False,
            circuit_outcome=circuit_outcome,
            value=None,
            exception_type=type(exc).__name__,
            performance=dict(performance or {}),
        )

    def _record_valuation_circuit_outcome(
        self, method: RepricingMethod, result: ValuationResult
    ) -> None:
        circuit = self.circuit(method)
        if result.circuit_outcome is CircuitCallOutcome.SUCCESS:
            circuit.record_infrastructure_success(1)
            self.batch_metrics["logical_circuit_successes"] += 1
        elif result.circuit_outcome is CircuitCallOutcome.INFRASTRUCTURE_FAILURE:
            circuit.record_infrastructure_failure(1)
            self.batch_metrics["logical_infrastructure_failures"] += 1
        elif result.circuit_outcome is CircuitCallOutcome.UNKNOWN_OUTCOME:
            circuit.record_unknown_outcome()
            self.batch_metrics["logical_unknown_outcomes"] += 1

    def _valuation_many_results(
        self,
        operation: str,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float],
        methods: RepricingMethod | str | Sequence[RepricingMethod | str],
        requested_dtypes: Optional[Sequence[str]] = None,
    ) -> list[ValuationResult]:
        count = len(requests)
        if len(volatilities) != count:
            raise ValueError("requests and volatilities must align")
        parsed_methods = self._aligned_argument(methods, count)
        dtypes = (
            [str(request.requested_dtype or "float64") for request in requests]
            if requested_dtypes is None
            else [str(value) for value in requested_dtypes]
        )
        if len(dtypes) != count:
            raise ValueError("requested_dtypes must align with requests")
        if not self.batch_config.enabled:
            scalar_output: list[ValuationResult] = []
            for request, volatility, method, requested_dtype in zip(
                requests, volatilities, parsed_methods, dtypes
            ):
                request = replace(request, requested_dtype=requested_dtype)
                actual_dtype = self._actual_dtype(method, requested_dtype)
                try:
                    self.circuit(method).before_call()
                except SolverCircuitOpenError as exc:
                    scalar_output.append(
                        self._valuation_result_for_exception(
                            operation=operation,
                            exc=exc,
                            method=method,
                            requested_dtype=requested_dtype,
                            actual_dtype=actual_dtype,
                        )
                    )
                    self.batch_metrics["circuit_suppressions"] += 1
                    self.batch_metrics["logical_jobs_suppressed"] += 1
                    continue
                try:
                    adapter = self.get(method)
                    if operation == "price":
                        value = adapter.price(request, volatility)
                    elif operation == "delta":
                        value = adapter.delta(request, volatility)
                    else:
                        value = adapter.greeks(request, volatility)
                    result = self._valuation_result_for_value(
                        operation=operation,
                        value=value,
                        method=method,
                        requested_dtype=requested_dtype,
                        actual_dtype=actual_dtype,
                    )
                except BaseException as exc:
                    result = self._valuation_result_for_exception(
                        operation=operation,
                        exc=exc,
                        method=method,
                        requested_dtype=requested_dtype,
                        actual_dtype=actual_dtype,
                    )
                self._record_valuation_circuit_outcome(method, result)
                scalar_output.append(result)
            self.batch_metrics["scalar_internal_calls"] += count
            return scalar_output
        output: list[Optional[ValuationResult]] = [None] * count
        grouped: dict[tuple[Any, ...], list[int]] = {}
        for index, (method, requested_dtype) in enumerate(zip(parsed_methods, dtypes)):
            adapter = self.get(method)
            actual_dtype = self._actual_dtype(method, requested_dtype)
            grouped.setdefault(
                (
                    method,
                    actual_dtype,
                    adapter.configuration_hash,
                    adapter.algorithm_version,
                    str(adapter.parameters.get("device", "cpu")),
                ),
                [],
            ).append(index)

        def invoke(
            adapter: SolverAdapter, indices: Sequence[int], actual_dtype: str
        ) -> list[Any]:
            selected_requests = [requests[index] for index in indices]
            selected_volatilities = np.ascontiguousarray(
                [float(volatilities[index]) for index in indices], dtype=np.float64
            )
            self.batch_metrics["physical_calls"] += 1
            self.batch_metrics[f"{operation}_physical_calls"] += 1
            self.batch_sizes.append(len(indices))
            if operation == "price":
                values = adapter.price_many(
                    selected_requests, selected_volatilities,
                    requested_dtype=actual_dtype,
                )
                return [float(value) for value in values]
            if operation == "delta":
                values = adapter.delta_many(
                    selected_requests, selected_volatilities,
                    requested_dtype=actual_dtype,
                )
                return [float(value) for value in values]
            return adapter.greeks_many(
                selected_requests, selected_volatilities,
                requested_dtype=actual_dtype,
            )

        def suppressed_result(
            *, index: int, method: RepricingMethod, actual_dtype: str,
            exc: SolverCircuitOpenError,
        ) -> ValuationResult:
            self.batch_metrics["circuit_suppressions"] += 1
            self.batch_metrics["logical_jobs_suppressed"] += 1
            return self._valuation_result_for_exception(
                operation=operation,
                exc=exc,
                method=method,
                requested_dtype=dtypes[index],
                actual_dtype=actual_dtype,
            )

        def suppress_open_pending(
            method: RepricingMethod,
            indices: Sequence[int],
            actual_dtype: str,
        ) -> tuple[list[int], dict[int, ValuationResult]]:
            if self.circuit(method).snapshot()["state"] != CircuitState.OPEN.value:
                return list(indices), {}
            remaining: list[int] = []
            suppressed: dict[int, ValuationResult] = {}
            for index in indices:
                try:
                    self.circuit(method).before_call()
                except SolverCircuitOpenError as exc:
                    suppressed[index] = suppressed_result(
                        index=index,
                        method=method,
                        actual_dtype=actual_dtype,
                        exc=exc,
                    )
                else:
                    remaining.append(index)
            return remaining, suppressed

        def isolated(
            adapter: SolverAdapter,
            method: RepricingMethod,
            indices: Sequence[int],
            actual_dtype: str,
        ) -> dict[int, ValuationResult]:
            if not indices:
                return {}
            try:
                values = invoke(adapter, indices, actual_dtype)
                self.batch_metrics["successful_physical_calls"] += 1
                results = {
                    index: self._valuation_result_for_value(
                        operation=operation,
                        value=value,
                        method=method,
                        requested_dtype=dtypes[index],
                        actual_dtype=actual_dtype,
                    )
                    for index, value in zip(indices, values)
                }
                for result in results.values():
                    self._record_valuation_circuit_outcome(method, result)
                return results
            except BaseException as exc:
                self.batch_metrics["physical_failures"] += 1
                outcome = (
                    "infrastructure"
                    if isinstance(exc, SolverInfrastructureError)
                    else "numerical"
                    if isinstance(exc, SolverNumericalError)
                    else "unknown"
                )
                self.batch_metrics[f"{outcome}_failures"] += 1
                if (
                    self.batch_config.exception_isolation_enabled
                    and len(indices) > 1
                    and not isinstance(exc, SolverCircuitOpenError)
                ):
                    self.batch_metrics["exception_splits"] += 1
                    middle = len(indices) // 2
                    results = isolated(
                        adapter, method, indices[:middle], actual_dtype
                    )
                    right_indices, newly_suppressed = suppress_open_pending(
                        method, indices[middle:], actual_dtype
                    )
                    results.update(newly_suppressed)
                    results.update(
                        isolated(adapter, method, right_indices, actual_dtype)
                    )
                    return results
                if len(indices) == 1 and self.batch_config.scalar_rollback_enabled:
                    index = indices[0]
                    try:
                        self.batch_metrics["scalar_rollback_calls"] += 1
                        if operation == "price":
                            value = adapter.price(
                                replace(
                                    requests[index], requested_dtype=actual_dtype
                                ),
                                float(volatilities[index]),
                            )
                        elif operation == "delta":
                            value = adapter.delta(
                                replace(
                                    requests[index], requested_dtype=actual_dtype
                                ),
                                float(volatilities[index]),
                            )
                        else:
                            value = adapter.greeks(
                                replace(
                                    requests[index], requested_dtype=actual_dtype
                                ),
                                float(volatilities[index]),
                            )
                        self.batch_metrics["successful_scalar_rollbacks"] += 1
                        result = self._valuation_result_for_value(
                            operation=operation,
                            value=value,
                            method=method,
                            requested_dtype=dtypes[index],
                            actual_dtype=actual_dtype,
                            performance={"scalar_rollback": True},
                        )
                        self._record_valuation_circuit_outcome(method, result)
                        return {index: result}
                    except BaseException as rollback_exc:
                        self.batch_metrics["scalar_rollback_failures"] += 1
                        exc = rollback_exc
                results: dict[int, ValuationResult] = {}
                for position, index in enumerate(indices):
                    if position:
                        remaining, newly_suppressed = suppress_open_pending(
                            method, (index,), actual_dtype
                        )
                        results.update(newly_suppressed)
                        if not remaining:
                            continue
                    result = self._valuation_result_for_exception(
                        operation=operation,
                        exc=exc,
                        method=method,
                        requested_dtype=dtypes[index],
                        actual_dtype=actual_dtype,
                    )
                    self._record_valuation_circuit_outcome(method, result)
                    results[index] = result
                return results

        for (method, actual_dtype, _hash, _version, _device), indices in grouped.items():
            adapter = self.get(method)
            maximum = self._maximum_chunk_jobs(adapter, operation, actual_dtype)
            for start in range(0, len(indices), maximum):
                candidate_indices = indices[start : start + maximum]
                allowed: list[int] = []
                for index in candidate_indices:
                    try:
                        self.circuit(method).before_call()
                    except SolverCircuitOpenError as exc:
                        output[index] = suppressed_result(
                            index=index,
                            method=method,
                            actual_dtype=actual_dtype,
                            exc=exc,
                        )
                    else:
                        allowed.append(index)
                        self.batch_metrics["logical_jobs_authorized"] += 1
                if allowed:
                    for index, result in isolated(
                        adapter, method, allowed, actual_dtype
                    ).items():
                        output[index] = result
        if any(value is None for value in output):
            raise AssertionError("valuation batch fan-out left an unowned result")
        return [value for value in output if value is not None]

    def price_many_results(
        self,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float],
        methods: RepricingMethod | str | Sequence[RepricingMethod | str],
        requested_dtypes: Optional[Sequence[str]] = None,
    ) -> list[ValuationResult]:
        return self._valuation_many_results(
            "price", requests, volatilities, methods, requested_dtypes
        )

    def delta_many_results(
        self,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float],
        methods: RepricingMethod | str | Sequence[RepricingMethod | str],
        requested_dtypes: Optional[Sequence[str]] = None,
    ) -> list[ValuationResult]:
        return self._valuation_many_results(
            "delta", requests, volatilities, methods, requested_dtypes
        )

    def greeks_many_results(
        self,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float],
        methods: RepricingMethod | str | Sequence[RepricingMethod | str],
        requested_dtypes: Optional[Sequence[str]] = None,
    ) -> list[ValuationResult]:
        return self._valuation_many_results(
            "greeks", requests, volatilities, methods, requested_dtypes
        )

    def price_many(
        self,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float],
        methods: RepricingMethod | str | Sequence[RepricingMethod | str],
        requested_dtypes: Optional[Sequence[str]] = None,
    ) -> list[float]:
        results = self.price_many_results(
            requests, volatilities, methods, requested_dtypes
        )
        return [
            float(result.value)
            if result.status == "complete" and result.value is not None
            else math.nan
            for result in results
        ]

    def delta_many(
        self,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float],
        methods: RepricingMethod | str | Sequence[RepricingMethod | str],
        requested_dtypes: Optional[Sequence[str]] = None,
    ) -> list[float]:
        results = self.delta_many_results(
            requests, volatilities, methods, requested_dtypes
        )
        return [
            float(result.value)
            if result.status == "complete" and result.value is not None
            else math.nan
            for result in results
        ]

    def greeks_many(
        self,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float],
        methods: RepricingMethod | str | Sequence[RepricingMethod | str],
        requested_dtypes: Optional[Sequence[str]] = None,
    ) -> list[Mapping[str, float]]:
        results = self.greeks_many_results(
            requests, volatilities, methods, requested_dtypes
        )
        return [
            result.value
            if result.status == "complete" and isinstance(result.value, Mapping)
            else self._failed_valuation_value("greeks")
            for result in results
        ]

    def solve_iv_and_greeks(
        self,
        request: PricingRequest,
        *,
        method: RepricingMethod,
        method_selection_reason: str,
        target_price: float,
        performance: Optional[MutableMapping[str, float | int]] = None,
        iv_hints: tuple[IvHint, ...] = (),
    ) -> RepricingResult:
        """Scalar compatibility wrapper around the aligned tree coordinator."""
        if not self.batch_config.enabled or method is RepricingMethod.CRANK_NICOLSON:
            result = self._solve_iv_and_greeks_scalar(
                request,
                method=method,
                method_selection_reason=method_selection_reason,
                target_price=target_price,
                performance=performance,
                iv_hints=iv_hints,
            )
            if (
                not self.batch_config.enabled
                and self.batch_config.float64_retry_enabled
                and method in (RepricingMethod.CRR, RepricingMethod.TIAN)
                and request.requested_dtype == "float32"
                and result.status == "failed"
                and result.reason
                in {
                    "nonfinite_model_price",
                    "nonfinite_solver_output",
                    "numerical_evaluation_failed",
                    "iv_price_tolerance_not_reached",
                    "price_residual_exceeds_tolerance",
                }
            ):
                retried = self._solve_iv_and_greeks_scalar(
                    replace(request, requested_dtype="float64"),
                    method=method,
                    method_selection_reason=method_selection_reason,
                    target_price=target_price,
                    performance=performance,
                    iv_hints=iv_hints,
                )
                return replace(
                    retried,
                    requested_dtype="float32",
                    actual_dtype="float64",
                    precision_escalated=True,
                    precision_escalation_reason="float32_failed_or_nonfinite",
                )
            return result
        started = time.perf_counter()
        job = SolverBatchJob(
            job_id=str(uuid.uuid4()),
            owner_run_id=request.snapshot_id,
            owner_snapshot_id=request.snapshot_id,
            output_index=0,
            operation=SolverOperation.SOLVE_IV_AND_GREEKS,
            request=request,
            method=method,
            method_selection_reason=method_selection_reason,
            target_price=float(target_price),
            iv_hints=iv_hints,
            requested_dtype=request.requested_dtype,
        )
        result = self.solve_iv_and_greeks_many((job,))[0]
        if performance is not None:
            suppressed = result.reason == "solver_circuit_open"
            performance.update(
                solver_call_permitted=int(not suppressed),
                solver_adapter_invoked=int(not suppressed),
                solver_session_created=0,
                circuit_suppressed=int(suppressed),
                iv_price_evaluations=result.iv_price_evaluations,
                iv_hint_evaluations=result.iv_hint_evaluations,
                total_seconds=time.perf_counter() - started,
            )
        if result.reason == "solver_circuit_open":
            raise SolverCircuitOpenError(method.value, self.circuit(method).open_until)
        return result

    def _solve_iv_and_greeks_scalar(
        self,
        request: PricingRequest,
        *,
        method: RepricingMethod,
        method_selection_reason: str,
        target_price: float,
        performance: Optional[MutableMapping[str, float | int]] = None,
        iv_hints: tuple[IvHint, ...] = (),
        consult_circuit: bool = True,
    ) -> RepricingResult:
        started_total = time.perf_counter()
        pricing_session: Optional[Any] = None
        fdm_valuations_iv = 0

        def record(name: str, value: float | int) -> None:
            if performance is not None:
                performance[name] = value

        def record_fdm_counts() -> None:
            if performance is None or pricing_session is None:
                return
            total = int(getattr(pricing_session, "fdm_valuations", 0))
            record("fdm_valuations_total", total)
            record("fdm_valuations_iv", fdm_valuations_iv)
            record("fdm_valuations_greeks", max(0, total - fdm_valuations_iv))

        circuit = self.circuit(method)
        if performance is not None:
            performance.setdefault("solver_call_permitted", 0)
            performance.setdefault("solver_adapter_invoked", 0)
            performance.setdefault("solver_session_created", 0)
            performance.setdefault("circuit_suppressed", 0)
        if consult_circuit:
            try:
                circuit.before_call()
            except SolverCircuitOpenError:
                record("circuit_suppressed", 1)
                raise
        record("solver_call_permitted", 1)

        def record_circuit_success() -> None:
            if consult_circuit:
                circuit.record_infrastructure_success()

        def record_circuit_infrastructure_failure() -> None:
            if consult_circuit:
                circuit.record_infrastructure_failure()

        def record_circuit_unknown_outcome() -> None:
            if consult_circuit:
                circuit.record_unknown_outcome()

        try:
            if performance is not None:
                for name in (
                    "session_init_seconds",
                    "iv_seconds",
                    "greeks_seconds",
                    "total_seconds",
                    "iv_price_evaluations",
                ):
                    performance.setdefault(name, 0.0)
                if method is RepricingMethod.CRANK_NICOLSON:
                    for name in (
                        "fdm_valuations_total",
                        "fdm_valuations_iv",
                        "fdm_valuations_greeks",
                    ):
                        performance.setdefault(name, 0)
            adapter = self.get(method)
            record("solver_adapter_invoked", 1)
            session_started = time.perf_counter()
            try:
                pricing_session = adapter.create_pricing_session(
                    request,
                    initial_iv=(iv_hints[0].value if iv_hints else None),
                )
                record("solver_session_created", 1)
                for name, value in getattr(
                    pricing_session, "tenor_diagnostics", {}
                ).items():
                    if isinstance(value, (int, float, bool)):
                        record(name, value)
            finally:
                record("session_init_seconds", time.perf_counter() - session_started)
            iv_started = time.perf_counter()
            try:
                try:
                    solved = adapter.solve_iv(
                        request,
                        target_price,
                        minimum=self.iv_config.minimum,
                        maximum=self.iv_config.maximum,
                        price_tolerance=self.iv_config.price_tolerance,
                        max_iterations=self.iv_config.max_iterations,
                        source_bracket_multipliers=self.iv_config.source_bracket_multipliers,
                        iv_hints=iv_hints,
                        hint_bracket_multipliers=(
                            self.iv_config.prior_seed_bracket_multipliers
                            if iv_hints and iv_hints[0].origin in {
                                "prior_success", "prior_partial_failure"
                            }
                            else self.iv_config.source_bracket_multipliers
                        ),
                        hint_identifiability_enabled=(
                            self.iv_config.hint_identifiability_enabled
                        ),
                        hint_identifiability_probe_iv_points=(
                            self.iv_config.hint_identifiability_probe_iv_points
                        ),
                        hint_identifiability_minimum_price_span_absolute=(
                            self.iv_config.hint_identifiability_minimum_price_span_absolute
                        ),
                        hint_identifiability_price_tolerance_multiplier=(
                            self.iv_config.hint_identifiability_price_tolerance_multiplier
                        ),
                        pricing_session=pricing_session,
                    )
                except SolverNumericalError:
                    solved = IvSolveResult(
                        "failed",
                        "numerical_evaluation_failed",
                        None,
                        0,
                        None,
                        None,
                    )
            finally:
                record("iv_seconds", time.perf_counter() - iv_started)
            record("iv_price_evaluations", solved.price_evaluations)
            record("iv_hint_evaluations", solved.hint_evaluations)
            if pricing_session is not None:
                fdm_valuations_iv = int(
                    getattr(pricing_session, "fdm_valuations", 0)
                )
                record_fdm_counts()
            base = dict(
                method=method.value,
                method_selection_reason=method_selection_reason,
                algorithm_version=adapter.algorithm_version,
                input_basis="market_mid",
                target_price=float(target_price),
                input_iv=solved.iv,
                implied_volatility=solved.iv,
                iv_iterations=solved.iterations,
                price_residual=solved.price_residual,
                requested_dtype=str(request.requested_dtype or "float64"),
                actual_dtype=self._actual_dtype(method, request.requested_dtype),
                precision_escalated=False,
                solver_parameters=adapter.parameters,
                iv_hint_origin_used=solved.hint_origin_used,
                iv_hint_value_used=solved.hint_value_used,
                iv_hint_exact_match=solved.hint_exact_match,
                iv_hint_local_bracket_found=solved.hint_local_bracket_found,
                iv_hint_evaluations=solved.hint_evaluations,
                iv_fallback_global_probes_used=solved.fallback_global_probes_used,
                iv_price_evaluations=solved.price_evaluations,
                iv_identifiability_status=solved.identifiability_status,
                iv_identifiability_price_span=solved.identifiability_price_span,
                iv_identifiability_slope=solved.identifiability_slope,
                iv_hint_flat_rejection_count=solved.hint_flat_rejection_count,
                quality_flags=(
                    QUALITY_IV_PRICE_INDETERMINATE
                    if solved.identifiability_status == "source_anchor_flat"
                    else 0
                ),
            )
            if solved.status != "complete" or solved.iv is None:
                record_circuit_success()
                return RepricingResult(
                    status="failed",
                    reason=solved.reason,
                    failure_stage="iv_inversion",
                    model_price=solved.model_price,
                    delta=None,
                    gamma=None,
                    theta_per_day=None,
                    vega_per_1pct=None,
                    rho_per_1pct=None,
                    **base,
                )
            greeks_started = time.perf_counter()
            try:
                values = (
                    adapter.greeks(
                        request, solved.iv, pricing_session=pricing_session
                    )
                    if pricing_session is not None
                    else adapter.greeks(request, solved.iv)
                )
                required = (
                    values.get("model_price"), values.get("delta"), values.get("gamma"),
                    values.get("theta_per_day"), values.get("vega_per_1pct"),
                    values.get("rho_per_1pct"),
                )
                if not all(
                    value is not None and math.isfinite(float(value))
                    for value in required
                ):
                    raise FloatingPointError("nonfinite_solver_output")
                residual = float(values["model_price"]) - float(target_price)
                if abs(residual) > self.iv_config.price_tolerance:
                    record_circuit_success()
                    return RepricingResult(
                        status="failed", reason="price_residual_exceeds_tolerance",
                        failure_stage="residual_validation", price_residual=residual,
                        model_price=values["model_price"], delta=values["delta"],
                        gamma=values["gamma"], theta_per_day=values["theta_per_day"],
                        vega_per_1pct=values["vega_per_1pct"],
                        rho_per_1pct=values["rho_per_1pct"],
                        **{
                            key: value
                            for key, value in base.items()
                            if key != "price_residual"
                        },
                    )
                record_circuit_success()
                return RepricingResult(
                    status="complete", reason="", failure_stage="none",
                    price_residual=residual, model_price=values["model_price"],
                    delta=values["delta"], gamma=values["gamma"],
                    theta_per_day=values["theta_per_day"],
                    vega_per_1pct=values["vega_per_1pct"],
                    rho_per_1pct=values["rho_per_1pct"],
                    **{
                        key: value
                        for key, value in base.items()
                        if key != "price_residual"
                    },
                )
            except SolverInfrastructureError:
                raise
            except SolverNumericalError:
                record_circuit_success()
                return RepricingResult(
                    status="failed", reason="numerical_evaluation_failed",
                    failure_stage="greeks", model_price=solved.model_price,
                    delta=None, gamma=None, theta_per_day=None,
                    vega_per_1pct=None, rho_per_1pct=None, **base,
                )
            except FloatingPointError as exc:
                record_circuit_success()
                return RepricingResult(
                    status="failed",
                    reason=(
                        "nonfinite_solver_output"
                        if str(exc) == "nonfinite_solver_output"
                        else "numerical_evaluation_failed"
                    ),
                    failure_stage="greeks", model_price=solved.model_price,
                    delta=None, gamma=None, theta_per_day=None,
                    vega_per_1pct=None, rho_per_1pct=None, **base,
                )
            finally:
                record("greeks_seconds", time.perf_counter() - greeks_started)
                record_fdm_counts()
        except MemoryError as exc:
            record_circuit_infrastructure_failure()
            raise SolverResourceError("solver execution exhausted memory") from exc
        except SolverInfrastructureError:
            record_circuit_infrastructure_failure()
            raise
        except Exception:
            record_circuit_unknown_outcome()
            raise
        finally:
            record_fdm_counts()
            record("total_seconds", time.perf_counter() - started_total)

    def solve_iv_only(
        self,
        request: PricingRequest,
        *,
        method: RepricingMethod,
        target_price: float,
        iv_hints: tuple[IvHint, ...] = (),
        performance: Optional[MutableMapping[str, float | int]] = None,
        consult_circuit: bool = True,
    ) -> IvSolveResult:
        """Run one fresh-session inversion without calculating Greeks."""
        circuit = self.circuit(method)
        if consult_circuit:
            circuit.before_call()
        try:
            adapter = self.get(method)
            session = adapter.create_pricing_session(
                request, initial_iv=(iv_hints[0].value if iv_hints else None)
            )
            try:
                solved = adapter.solve_iv(
                    request,
                    target_price,
                    minimum=self.iv_config.minimum,
                    maximum=self.iv_config.maximum,
                    price_tolerance=self.iv_config.price_tolerance,
                    max_iterations=self.iv_config.max_iterations,
                    source_bracket_multipliers=self.iv_config.source_bracket_multipliers,
                    iv_hints=iv_hints,
                    hint_bracket_multipliers=(
                        self.iv_config.prior_seed_bracket_multipliers
                        if iv_hints and iv_hints[0].origin in {
                            "prior_success", "prior_partial_failure"
                        }
                        else self.iv_config.source_bracket_multipliers
                    ),
                    hint_identifiability_enabled=(
                        self.iv_config.hint_identifiability_enabled
                    ),
                    hint_identifiability_probe_iv_points=(
                        self.iv_config.hint_identifiability_probe_iv_points
                    ),
                    hint_identifiability_minimum_price_span_absolute=(
                        self.iv_config.hint_identifiability_minimum_price_span_absolute
                    ),
                    hint_identifiability_price_tolerance_multiplier=(
                        self.iv_config.hint_identifiability_price_tolerance_multiplier
                    ),
                    pricing_session=session,
                )
            except SolverNumericalError:
                solved = IvSolveResult(
                    "failed",
                    "numerical_evaluation_failed",
                    None,
                    0,
                    None,
                    None,
                )
            if (
                not self.batch_config.enabled
                and self.batch_config.float64_retry_enabled
                and method in (RepricingMethod.CRR, RepricingMethod.TIAN)
                and request.requested_dtype == "float32"
                and solved.status == "failed"
                and solved.reason
                in {
                    "nonfinite_model_price",
                    "numerical_evaluation_failed",
                    "iv_price_tolerance_not_reached",
                }
            ):
                retry_request = replace(request, requested_dtype="float64")
                solved = replace(
                    adapter.solve_iv(
                        retry_request,
                        target_price,
                        minimum=self.iv_config.minimum,
                        maximum=self.iv_config.maximum,
                        price_tolerance=self.iv_config.price_tolerance,
                        max_iterations=self.iv_config.max_iterations,
                        source_bracket_multipliers=(
                            self.iv_config.source_bracket_multipliers
                        ),
                        iv_hints=iv_hints,
                        hint_bracket_multipliers=(
                            self.iv_config.prior_seed_bracket_multipliers
                            if iv_hints
                            and iv_hints[0].origin
                            in {"prior_success", "prior_partial_failure"}
                            else self.iv_config.source_bracket_multipliers
                        ),
                        hint_identifiability_enabled=(
                            self.iv_config.hint_identifiability_enabled
                        ),
                        hint_identifiability_probe_iv_points=(
                            self.iv_config.hint_identifiability_probe_iv_points
                        ),
                        hint_identifiability_minimum_price_span_absolute=(
                            self.iv_config.hint_identifiability_minimum_price_span_absolute
                        ),
                        hint_identifiability_price_tolerance_multiplier=(
                            self.iv_config.hint_identifiability_price_tolerance_multiplier
                        ),
                    ),
                    requested_dtype="float32",
                    actual_dtype="float64",
                    precision_escalated=True,
                )
            if performance is not None:
                performance["iv_price_evaluations"] = solved.price_evaluations
                performance["iv_iterations"] = solved.iterations
                performance["fdm_valuations"] = int(
                    getattr(session, "fdm_valuations", 0)
                )
            if consult_circuit:
                circuit.record_infrastructure_success()
            return solved
        except MemoryError as exc:
            if consult_circuit:
                circuit.record_infrastructure_failure()
            raise SolverResourceError("solver execution exhausted memory") from exc
        except SolverInfrastructureError:
            if consult_circuit:
                circuit.record_infrastructure_failure()
            raise
        except Exception:
            if consult_circuit:
                circuit.record_unknown_outcome()
            raise

    def _scalar_operation(
        self,
        operation: str,
        request: PricingRequest,
        volatility: float,
        *,
        method: RepricingMethod,
        performance: Optional[MutableMapping[str, float | int]] = None,
    ) -> float:
        """Execute one circuit-aware scalar valuation with actual-work facts."""
        if operation not in {"price", "delta"}:
            raise ValueError(f"unsupported scalar solver operation: {operation}")

        def record(name: str, value: float | int) -> None:
            if performance is not None:
                performance[name] = value

        if performance is not None:
            for name in (
                "solver_call_permitted",
                "solver_adapter_invoked",
                "solver_session_created",
                "circuit_suppressed",
                "fdm_valuations",
            ):
                performance.setdefault(name, 0)
        circuit = self.circuit(method)
        try:
            circuit.before_call()
        except SolverCircuitOpenError:
            record("circuit_suppressed", 1)
            raise
        record("solver_call_permitted", 1)
        session: Optional[Any] = None
        try:
            adapter = self.get(method)
            record("solver_adapter_invoked", 1)
            session = adapter.create_pricing_session(
                request, initial_iv=float(volatility)
            )
            record("solver_session_created", 1)
            value = float(
                getattr(adapter, operation)(
                    request,
                    float(volatility),
                    pricing_session=session,
                )
            )
            if not math.isfinite(value):
                raise FloatingPointError(f"nonfinite_solver_{operation}")
            circuit.record_infrastructure_success()
            return value
        except MemoryError as exc:
            circuit.record_infrastructure_failure()
            raise SolverResourceError("solver scalar valuation exhausted memory") from exc
        except SolverInfrastructureError:
            circuit.record_infrastructure_failure()
            raise
        except (SolverNumericalError, FloatingPointError):
            circuit.record_infrastructure_success()
            raise
        except Exception:
            circuit.record_unknown_outcome()
            raise
        finally:
            if session is not None:
                record("fdm_valuations", int(getattr(session, "fdm_valuations", 0)))

    def price(
        self,
        request: PricingRequest,
        volatility: float,
        *,
        method: RepricingMethod,
        performance: Optional[MutableMapping[str, float | int]] = None,
    ) -> float:
        return self._scalar_operation(
            "price", request, volatility, method=method, performance=performance
        )

    def delta(
        self,
        request: PricingRequest,
        volatility: float,
        *,
        method: RepricingMethod,
        performance: Optional[MutableMapping[str, float | int]] = None,
    ) -> float:
        return self._scalar_operation(
            "delta", request, volatility, method=method, performance=performance
        )


@dataclass
class _QueuedSolverCall:
    operation: str
    payload: Any
    event: threading.Event
    result: Any = None
    exception: Optional[BaseException] = None


class MicroBatchingSolverRegistry:
    """Coalesce numerical calls made by independently prepared owner threads.

    This wrapper owns no solver state: the wrapped registry remains the single
    circuit/provenance authority.  Calls are flattened only for the physical
    numerical invocation and are fanned back to their waiting durable owner.
    """

    cross_owner_batching = True

    def __init__(
        self,
        registry: SolverRegistry,
        *,
        coalesce_seconds: float = 0.0005,
    ) -> None:
        self.registry = registry
        self.coalesce_seconds = max(0.0, float(coalesce_seconds))
        self._queue: queue.Queue[Optional[_QueuedSolverCall]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._dispatch,
            name="options-numerical-micro-batch",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.registry, name)

    def _submit(self, operation: str, payload: Any) -> Any:
        if self._closed:
            raise RuntimeError("numerical micro-batch registry is closed")
        call = _QueuedSolverCall(operation, payload, threading.Event())
        self._queue.put(call)
        call.event.wait()
        if call.exception is not None:
            raise call.exception
        return call.result

    def _dispatch(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            calls = [first]
            deadline = time.monotonic() + self.coalesce_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self._queue.put(None)
                    break
                calls.append(item)
            by_operation: dict[str, list[_QueuedSolverCall]] = {}
            for call in calls:
                by_operation.setdefault(call.operation, []).append(call)
            for operation, operation_calls in by_operation.items():
                try:
                    self._execute_calls(operation, operation_calls)
                except BaseException as exc:
                    for call in operation_calls:
                        call.exception = exc
                finally:
                    for call in operation_calls:
                        call.event.set()

    def _execute_calls(
        self, operation: str, calls: Sequence[_QueuedSolverCall]
    ) -> None:
        if operation in {"solve_iv_and_greeks", "solve_iv_only"}:
            owned_jobs = [
                (call_index, item_index, job)
                for call_index, call in enumerate(calls)
                for item_index, job in enumerate(call.payload)
            ]
            owned_jobs.sort(
                key=lambda item: (
                    item[2].method.value,
                    str(item[2].owner_snapshot_id),
                    str(item[2].owner_run_id),
                    int(item[2].output_index),
                    item[2].request.option_symbol,
                    item[2].job_id,
                )
            )
            ordered_jobs = [item[2] for item in owned_jobs]
            results = (
                self.registry.solve_iv_and_greeks_many(ordered_jobs)
                if operation == "solve_iv_and_greeks"
                else self.registry.solve_iv_only_many(ordered_jobs)
            )
            fanned: list[list[Any]] = [
                [None] * len(call.payload) for call in calls
            ]
            for (call_index, item_index, _job), result in zip(owned_jobs, results):
                fanned[call_index][item_index] = result
            for call, owned_results in zip(calls, fanned):
                call.result = owned_results
            return
        owned_values = []
        for call_index, call in enumerate(calls):
            call_requests, call_volatilities, call_methods, call_dtypes = call.payload
            owned_values.extend(
                (
                    call_index, item_index, request, volatility, method, dtype
                )
                for item_index, (request, volatility, method, dtype) in enumerate(
                    zip(
                        call_requests, call_volatilities, call_methods, call_dtypes
                    )
                )
            )
        owned_values.sort(
            key=lambda item: (
                item[4].value,
                item[5],
                item[2].snapshot_id,
                item[2].option_symbol,
                item[1],
            )
        )
        requests = [item[2] for item in owned_values]
        volatilities = [item[3] for item in owned_values]
        methods = [item[4] for item in owned_values]
        dtypes = [item[5] for item in owned_values]
        function = getattr(self.registry, f"{operation}_many")
        results = function(requests, volatilities, methods, dtypes)
        fanned = [
            [None] * len(call.payload[0]) for call in calls
        ]
        for (call_index, item_index, *_rest), result in zip(owned_values, results):
            fanned[call_index][item_index] = result
        for call, owned_results in zip(calls, fanned):
            call.result = owned_results

    def solve_iv_and_greeks_many(
        self, jobs: Sequence[SolverBatchJob]
    ) -> list[RepricingResult]:
        return self._submit("solve_iv_and_greeks", tuple(jobs))

    def solve_iv_only_many(
        self, jobs: Sequence[SolverBatchJob]
    ) -> list[IvSolveResult]:
        return self._submit("solve_iv_only", tuple(jobs))

    def solve_iv_and_greeks(
        self,
        request: PricingRequest,
        *,
        method: RepricingMethod,
        method_selection_reason: str,
        target_price: float,
        performance: Optional[MutableMapping[str, float | int]] = None,
        iv_hints: tuple[IvHint, ...] = (),
    ) -> RepricingResult:
        job = SolverBatchJob(
            job_id=str(uuid.uuid4()),
            owner_run_id=request.snapshot_id,
            owner_snapshot_id=request.snapshot_id,
            output_index=0,
            operation=SolverOperation.SOLVE_IV_AND_GREEKS,
            request=request,
            method=method,
            method_selection_reason=method_selection_reason,
            target_price=target_price,
            iv_hints=iv_hints,
            requested_dtype=request.requested_dtype,
        )
        result = self.solve_iv_and_greeks_many((job,))[0]
        if performance is not None:
            performance.update(
                solver_call_permitted=int(result.reason != "solver_circuit_open"),
                solver_adapter_invoked=int(result.reason != "solver_circuit_open"),
                solver_session_created=int(method is RepricingMethod.CRANK_NICOLSON),
                circuit_suppressed=int(result.reason == "solver_circuit_open"),
                iv_price_evaluations=result.iv_price_evaluations,
            )
        if result.reason == "solver_circuit_open":
            raise SolverCircuitOpenError(method.value, self.registry.circuit(method).open_until)
        return result

    def solve_iv_only(
        self,
        request: PricingRequest,
        *,
        method: RepricingMethod,
        target_price: float,
        iv_hints: tuple[IvHint, ...] = (),
        performance: Optional[MutableMapping[str, float | int]] = None,
        consult_circuit: bool = True,
    ) -> IvSolveResult:
        if not consult_circuit:
            return self.registry.solve_iv_only(
                request,
                method=method,
                target_price=target_price,
                iv_hints=iv_hints,
                performance=performance,
                consult_circuit=False,
            )
        job = SolverBatchJob(
            job_id=str(uuid.uuid4()), owner_run_id=request.snapshot_id,
            owner_snapshot_id=request.snapshot_id, output_index=0,
            operation=SolverOperation.SOLVE_IV_ONLY, request=request, method=method,
            target_price=target_price, iv_hints=iv_hints,
            requested_dtype=request.requested_dtype,
        )
        result = self.solve_iv_only_many((job,))[0]
        if performance is not None:
            performance.update(
                iv_price_evaluations=result.price_evaluations,
                iv_iterations=result.iterations,
            )
        return result

    def _valuation_submit(
        self,
        operation: str,
        requests: Sequence[PricingRequest],
        volatilities: Sequence[float],
        methods: RepricingMethod | str | Sequence[RepricingMethod | str],
        requested_dtypes: Optional[Sequence[str]],
    ) -> list[Any]:
        if len(volatilities) != len(requests):
            raise ValueError("requests and volatilities must align")
        parsed = SolverRegistry._aligned_argument(methods, len(requests))
        dtypes = (
            [request.requested_dtype for request in requests]
            if requested_dtypes is None
            else list(requested_dtypes)
        )
        if len(dtypes) != len(requests):
            raise ValueError("requested_dtypes must align with requests")
        return self._submit(
            operation,
            (tuple(requests), tuple(volatilities), tuple(parsed), tuple(dtypes)),
        )

    def price_many(self, requests: Sequence[PricingRequest], volatilities: Sequence[float], methods: Any, requested_dtypes: Optional[Sequence[str]] = None) -> list[float]:
        return self._valuation_submit("price", requests, volatilities, methods, requested_dtypes)

    def delta_many(self, requests: Sequence[PricingRequest], volatilities: Sequence[float], methods: Any, requested_dtypes: Optional[Sequence[str]] = None) -> list[float]:
        return self._valuation_submit("delta", requests, volatilities, methods, requested_dtypes)

    def greeks_many(self, requests: Sequence[PricingRequest], volatilities: Sequence[float], methods: Any, requested_dtypes: Optional[Sequence[str]] = None) -> list[Mapping[str, float]]:
        return self._valuation_submit("greeks", requests, volatilities, methods, requested_dtypes)

    # Typed valuation calls intentionally bypass the legacy timing-driven
    # coalescer. Production orchestration prepares and invokes these batches
    # explicitly; these methods keep this retained test/rollback helper safe.
    def price_many_results(self, requests: Sequence[PricingRequest], volatilities: Sequence[float], methods: Any, requested_dtypes: Optional[Sequence[str]] = None) -> list[ValuationResult]:
        return self.registry.price_many_results(requests, volatilities, methods, requested_dtypes)

    def delta_many_results(self, requests: Sequence[PricingRequest], volatilities: Sequence[float], methods: Any, requested_dtypes: Optional[Sequence[str]] = None) -> list[ValuationResult]:
        return self.registry.delta_many_results(requests, volatilities, methods, requested_dtypes)

    def greeks_many_results(self, requests: Sequence[PricingRequest], volatilities: Sequence[float], methods: Any, requested_dtypes: Optional[Sequence[str]] = None) -> list[ValuationResult]:
        return self.registry.greeks_many_results(requests, volatilities, methods, requested_dtypes)

    def price(self, request: PricingRequest, volatility: float, *, method: RepricingMethod, performance: Optional[MutableMapping[str, float | int]] = None) -> float:
        value = float(self.price_many((request,), (volatility,), method)[0])
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite_solver_price")
        return value

    def delta(self, request: PricingRequest, volatility: float, *, method: RepricingMethod, performance: Optional[MutableMapping[str, float | int]] = None) -> float:
        value = float(self.delta_many((request,), (volatility,), method)[0])
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite_solver_delta")
        return value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()

    def __enter__(self) -> "MicroBatchingSolverRegistry":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
