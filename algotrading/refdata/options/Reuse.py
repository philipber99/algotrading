"""Pure contracts and conservative policy for successful repricing reuse."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Optional

from algotrading.refdata.options.solvers.Types import RecalcRetryClass


UTC = timezone.utc
REQUIRED_REUSED_OUTPUTS = (
    "implied_volatility",
    "delta",
    "gamma",
    "theta_per_day",
    "vega_per_1pct",
    "rho_per_1pct",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _finite(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class RepricingReuseTenorBand:
    max_tenor_seconds: Optional[float]
    max_age_seconds: float
    max_spot_move_bps: float
    max_elapsed_fraction: float


@dataclass(frozen=True)
class ShadowRepricingPolicy:
    enabled: bool
    sample_fraction: float
    short_tenor_seconds: float
    short_tenor_sample_fraction: float
    seed: int
    maximum_iv_error: float
    maximum_delta_error: float


@dataclass(frozen=True)
class RepricingReusePolicy:
    enabled: bool
    target_price_absolute_tolerance: float
    maximum_rate_change: float
    maximum_dividend_yield_change: float
    maximum_estimated_price_drift: float
    maximum_estimated_iv_change_points: float
    vega_floor: float
    lookup_allowance_seconds: float
    tenor_bands: tuple[RepricingReuseTenorBand, ...]
    shadow: ShadowRepricingPolicy

    @property
    def maximum_age_seconds(self) -> float:
        return max(band.max_age_seconds for band in self.tenor_bands)

    def band_for(self, tenor_seconds: float) -> RepricingReuseTenorBand:
        for band in self.tenor_bands:
            if band.max_tenor_seconds is None or tenor_seconds <= band.max_tenor_seconds:
                return band
        raise AssertionError("validated reuse policy has no catch-all tenor band")


@dataclass(frozen=True)
class CurrentRepricingProblem:
    option_symbol: str
    snapshot_id: str
    source_collected_at: datetime
    calculation_definition_id: int
    spot: float
    target_price: float
    time_to_expiry_seconds: float
    rate: float
    dividend_yield: float
    method: str
    algorithm_version: str
    solver_config_hash: str
    input_basis: str
    dividend_model: str


@dataclass(frozen=True)
class PriorRepricingOutcome:
    option_symbol: str
    source_snapshot_id: str
    source_row_collected_at: datetime
    calculation_definition_id: int
    source_calculated_at: datetime
    origin_snapshot_id: str
    origin_calculated_at: datetime
    valuation_at: datetime
    valuation_spot: float
    valuation_target_price: float
    valuation_time_to_expiry_seconds: float
    valuation_rate: float
    valuation_dividend_yield: float
    method: str
    method_selection_reason: str
    algorithm_version: str
    solver_config_hash: str
    solver_parameters_json: str
    input_basis: str
    input_iv: Optional[float]
    implied_volatility: Optional[float]
    iv_iterations: int
    price_residual: Optional[float]
    model_price: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta_per_day: Optional[float]
    vega_per_1pct: Optional[float]
    rho_per_1pct: Optional[float]
    requested_dtype: str
    actual_dtype: str
    precision_escalated: bool
    quality_flags: int
    time_steps: Optional[int]
    space_steps: Optional[int]
    damping_steps: Optional[int]
    dividend_model: str
    precision_escalation_reason: str = "none"
    status: str = "complete"
    reason: str = ""
    failure_stage: str = "none"
    retry_class: str = RecalcRetryClass.NONE.value
    retry_after: Optional[datetime] = None
    failure_fingerprint: str = "0" * 64
    result_origin: str = "calculated"


@dataclass(frozen=True)
class PriorOutcomeCandidates:
    success: Optional[PriorRepricingOutcome] = None
    failure: Optional[PriorRepricingOutcome] = None
    failure_seed: Optional[PriorRepricingOutcome] = None


@dataclass(frozen=True)
class PriorOutcomeLookupPolicy:
    """Explicit database demand derived from active reuse/seed policies."""

    maximum_age_seconds: float
    lookup_allowance_seconds: float
    need_success_reuse: bool
    need_success_seed: bool
    need_failure_reuse: bool
    need_failure_seed: bool
    minimum_seed_iv: float
    maximum_seed_iv: float


@dataclass(frozen=True)
class ReuseDecision:
    reusable: bool
    reason: str
    age_seconds: Optional[float] = None
    spot_move_bps: Optional[float] = None
    rate_change: Optional[float] = None
    absolute_dividend_yield_change: Optional[float] = None
    elapsed_fraction_of_prior_remaining_tenor: Optional[float] = None
    estimated_price_drift: Optional[float] = None
    estimated_iv_change_points: Optional[float] = None


def parse_tenor_bands(value: Any) -> tuple[RepricingReuseTenorBand, ...]:
    """Parse ordered mappings or their JSON representation, failing closed."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("reuse.tenor_bands must be valid JSON") from exc
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("reuse.tenor_bands must be a non-empty list or tuple")
    bands: list[RepricingReuseTenorBand] = []
    previous = 0.0
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("each reuse tenor band must be a mapping")
        required = {
            "max_tenor_seconds",
            "max_age_seconds",
            "max_spot_move_bps",
            "max_elapsed_fraction",
        }
        if set(item) != required:
            raise ValueError(
                "reuse tenor bands require exactly " + ", ".join(sorted(required))
            )
        raw_tenor = item["max_tenor_seconds"]
        maximum_tenor = None if raw_tenor is None else _finite(raw_tenor)
        if raw_tenor is not None and (maximum_tenor is None or maximum_tenor <= 0):
            raise ValueError("max_tenor_seconds must be positive or null")
        if maximum_tenor is None and index != len(value) - 1:
            raise ValueError("only the final reuse tenor band may be catch-all")
        if maximum_tenor is not None and maximum_tenor <= previous:
            raise ValueError("reuse max_tenor_seconds values must be strictly ascending")
        parsed_limits = [
            _finite(item[name])
            for name in (
                "max_age_seconds",
                "max_spot_move_bps",
                "max_elapsed_fraction",
            )
        ]
        if any(limit is None or limit < 0 for limit in parsed_limits):
            raise ValueError("reuse age and movement limits must be finite and nonnegative")
        bands.append(
            RepricingReuseTenorBand(
                maximum_tenor,
                float(parsed_limits[0]),
                float(parsed_limits[1]),
                float(parsed_limits[2]),
            )
        )
        if maximum_tenor is not None:
            previous = maximum_tenor
    if bands[-1].max_tenor_seconds is not None:
        raise ValueError("reuse.tenor_bands requires a final catch-all band")
    return tuple(bands)


def parse_repricing_reuse_policy(context: Mapping[str, Any]) -> RepricingReusePolicy:
    prefix = "options_calcs.greeks.reuse."
    defaults = default_tenor_bands()

    def nonnegative(name: str, default: float, *, positive: bool = False) -> float:
        value = _finite(context.get(prefix + name, default))
        if value is None or value < 0 or (positive and value <= 0):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{prefix + name} must be finite and {qualifier}")
        return value

    shadow_prefix = prefix + "shadow."
    sample_fraction = _finite(context.get(shadow_prefix + "sample_fraction", 0.05))
    short_fraction = _finite(
        context.get(shadow_prefix + "short_tenor_sample_fraction", 0.25)
    )
    if sample_fraction is None or not 0.0 <= sample_fraction <= 1.0:
        raise ValueError("reuse.shadow.sample_fraction must be between zero and one")
    if short_fraction is None or not 0.0 <= short_fraction <= 1.0:
        raise ValueError(
            "reuse.shadow.short_tenor_sample_fraction must be between zero and one"
        )
    shadow = ShadowRepricingPolicy(
        enabled=_as_bool(context.get(shadow_prefix + "enabled", True)),
        sample_fraction=sample_fraction,
        short_tenor_seconds=nonnegative(
            "shadow.short_tenor_seconds", 86400.0, positive=True
        ),
        short_tenor_sample_fraction=short_fraction,
        seed=int(context.get(shadow_prefix + "seed", 0)),
        maximum_iv_error=nonnegative("shadow.maximum_iv_error", 0.0005),
        maximum_delta_error=nonnegative("shadow.maximum_delta_error", 0.0025),
    )
    return RepricingReusePolicy(
        enabled=_as_bool(context.get(prefix + "enabled", False)),
        target_price_absolute_tolerance=nonnegative(
            "target_price_absolute_tolerance", 1.0e-8
        ),
        maximum_rate_change=nonnegative("maximum_rate_change", 1.0e-6),
        maximum_dividend_yield_change=nonnegative(
            "maximum_dividend_yield_change", 1.0e-8
        ),
        maximum_estimated_price_drift=nonnegative(
            "maximum_estimated_price_drift", 0.005
        ),
        maximum_estimated_iv_change_points=nonnegative(
            "maximum_estimated_iv_change_points", 0.02
        ),
        vega_floor=nonnegative("vega_floor", 1.0e-6, positive=True),
        lookup_allowance_seconds=nonnegative("lookup_allowance_seconds", 60.0),
        tenor_bands=parse_tenor_bands(context.get(prefix + "tenor_bands", defaults)),
        shadow=shadow,
    )


def default_tenor_bands() -> list[dict[str, Optional[float]]]:
    values = (
        (1800, 15, 0.5, 0.01),
        (7200, 30, 1.0, 0.01),
        (21600, 60, 2.0, 0.01),
        (86400, 90, 3.0, 0.005),
        (604800, 180, 5.0, 0.002),
        (2592000, 300, 10.0, 0.001),
        (7776000, 600, 20.0, 0.001),
        (None, 900, 30.0, 0.001),
    )
    return [
        {
            "max_tenor_seconds": tenor,
            "max_age_seconds": age,
            "max_spot_move_bps": spot,
            "max_elapsed_fraction": elapsed,
        }
        for tenor, age, spot, elapsed in values
    ]


def assess_success_reuse(
    current: CurrentRepricingProblem,
    prior: PriorRepricingOutcome,
    policy: RepricingReusePolicy,
) -> ReuseDecision:
    """Assess one original successful valuation without database dependencies."""
    reason = "successful_result_reusable"
    if current.option_symbol != prior.option_symbol:
        return ReuseDecision(False, "contract_mismatch")
    if current.calculation_definition_id != prior.calculation_definition_id:
        return ReuseDecision(False, "calculation_definition_mismatch")
    if current.method != prior.method:
        return ReuseDecision(False, "method_mismatch")
    if current.algorithm_version != prior.algorithm_version:
        return ReuseDecision(False, "algorithm_mismatch")
    if current.solver_config_hash != prior.solver_config_hash:
        return ReuseDecision(False, "solver_config_mismatch")
    if current.input_basis != prior.input_basis:
        return ReuseDecision(False, "input_basis_mismatch")
    if current.dividend_model != prior.dividend_model:
        return ReuseDecision(False, "dividend_model_mismatch")
    required = [_finite(getattr(prior, name)) for name in REQUIRED_REUSED_OUTPUTS]
    original_inputs = [
        _finite(prior.valuation_spot),
        _finite(prior.valuation_target_price),
        _finite(prior.valuation_time_to_expiry_seconds),
        _finite(prior.valuation_rate),
        _finite(prior.valuation_dividend_yield),
    ]
    if any(value is None for value in (*required, *original_inputs)):
        return ReuseDecision(False, "nonfinite_prior")
    if original_inputs[0] <= 0 or original_inputs[2] <= 0:
        return ReuseDecision(False, "nonfinite_prior")
    if not math.isclose(
        current.target_price,
        prior.valuation_target_price,
        rel_tol=0.0,
        abs_tol=policy.target_price_absolute_tolerance,
    ):
        return ReuseDecision(False, "target_price_changed")

    age = (current.source_collected_at - prior.valuation_at).total_seconds()
    source_age = (
        current.source_collected_at - prior.source_row_collected_at
    ).total_seconds()
    if age < 0 or source_age <= 0:
        return ReuseDecision(False, "prior_not_strictly_earlier")
    band = policy.band_for(current.time_to_expiry_seconds)
    spot_change = current.spot - prior.valuation_spot
    spot_move_bps = abs(spot_change) / abs(prior.valuation_spot) * 10_000.0
    elapsed_fraction = age / prior.valuation_time_to_expiry_seconds
    rate_change = abs(current.rate - prior.valuation_rate)
    dividend_change = abs(
        current.dividend_yield - prior.valuation_dividend_yield
    )
    elapsed_days = age / 86400.0
    estimated_price_drift = (
        prior.delta * spot_change
        + 0.5 * prior.gamma * spot_change * spot_change
        + prior.theta_per_day * elapsed_days
        + prior.rho_per_1pct * ((current.rate - prior.valuation_rate) / 0.01)
    )
    estimated_iv_change = abs(estimated_price_drift) / max(
        abs(prior.vega_per_1pct), policy.vega_floor
    )
    metrics = dict(
        age_seconds=age,
        spot_move_bps=spot_move_bps,
        rate_change=rate_change,
        absolute_dividend_yield_change=dividend_change,
        elapsed_fraction_of_prior_remaining_tenor=elapsed_fraction,
        estimated_price_drift=estimated_price_drift,
        estimated_iv_change_points=estimated_iv_change,
    )
    guards = (
        (age > band.max_age_seconds, "age_exceeded"),
        (spot_move_bps > band.max_spot_move_bps, "spot_move_exceeded"),
        (elapsed_fraction > band.max_elapsed_fraction, "elapsed_fraction_exceeded"),
        (rate_change > policy.maximum_rate_change, "rate_change_exceeded"),
        (
            dividend_change > policy.maximum_dividend_yield_change,
            "dividend_yield_change_exceeded",
        ),
        (
            abs(estimated_price_drift) > policy.maximum_estimated_price_drift,
            "estimated_price_drift_exceeded",
        ),
        (
            estimated_iv_change > policy.maximum_estimated_iv_change_points,
            "estimated_iv_drift_exceeded",
        ),
    )
    for invalid, invalid_reason in guards:
        if invalid:
            return ReuseDecision(False, invalid_reason, **metrics)
    return ReuseDecision(True, reason, **metrics)


def deterministic_shadow_sample(
    current: CurrentRepricingProblem, policy: ShadowRepricingPolicy
) -> bool:
    if not policy.enabled:
        return False
    fraction = (
        policy.short_tenor_sample_fraction
        if current.time_to_expiry_seconds <= policy.short_tenor_seconds
        else policy.sample_fraction
    )
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    payload = "\0".join(
        (
            str(policy.seed),
            current.snapshot_id,
            current.option_symbol,
            str(current.calculation_definition_id),
        )
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value < int(fraction * (1 << 64))


REUSE_INVALIDATION_DIAGNOSTIC = MappingProxyType(
    {
        "target_price_changed": "recalc_reuse_invalidated_target_price",
        "age_exceeded": "recalc_reuse_invalidated_age",
        "spot_move_exceeded": "recalc_reuse_invalidated_spot",
        "elapsed_fraction_exceeded": "recalc_reuse_invalidated_elapsed_fraction",
        "rate_change_exceeded": "recalc_reuse_invalidated_rate",
        "dividend_yield_change_exceeded": "recalc_reuse_invalidated_dividend",
        "estimated_price_drift_exceeded": "recalc_reuse_invalidated_estimated_price_drift",
        "estimated_iv_drift_exceeded": "recalc_reuse_invalidated_estimated_iv_drift",
        "method_mismatch": "recalc_reuse_invalidated_method",
        "algorithm_mismatch": "recalc_reuse_invalidated_algorithm",
        "solver_config_mismatch": "recalc_reuse_invalidated_solver_config",
        "nonfinite_prior": "recalc_reuse_invalidated_nonfinite_prior",
    }
)


@dataclass(frozen=True)
class IvSeedTenorBand:
    max_tenor_seconds: Optional[float]
    max_seed_age_seconds: float
    max_spot_move_bps: float


@dataclass(frozen=True)
class SeedShadowPolicy:
    enabled: bool
    sample_fraction: float
    seed: int
    maximum_iv_difference: float
    maximum_price_residual_difference: float


@dataclass(frozen=True)
class IvSeedPolicy:
    enabled: bool
    minimum_iv: float
    maximum_iv: float
    maximum_target_price_absolute_change: Optional[float]
    maximum_target_price_fractional_change: Optional[float]
    maximum_rate_change: Optional[float]
    maximum_dividend_yield_change: Optional[float]
    tenor_bands: tuple[IvSeedTenorBand, ...]
    shadow: SeedShadowPolicy

    @property
    def maximum_age_seconds(self) -> float:
        return max(band.max_seed_age_seconds for band in self.tenor_bands)

    def band_for(self, tenor_seconds: float) -> IvSeedTenorBand:
        for band in self.tenor_bands:
            if band.max_tenor_seconds is None or tenor_seconds <= band.max_tenor_seconds:
                return band
        raise AssertionError("validated seed policy has no catch-all tenor band")


@dataclass(frozen=True)
class IvSeedDecision:
    eligible: bool
    reason: str
    seed_value: Optional[float] = None
    seed_origin: str = "none"
    seed_snapshot_id: Optional[str] = None
    seed_calculated_at: Optional[datetime] = None
    age_seconds: Optional[float] = None
    spot_move_bps: Optional[float] = None
    target_price_absolute_change: Optional[float] = None
    target_price_fractional_change: Optional[float] = None


@dataclass(frozen=True)
class FailureReuseTenorBand:
    max_tenor_seconds: Optional[float]
    retry_seconds: float
    max_spot_move_bps: float
    max_elapsed_fraction: float


@dataclass(frozen=True)
class FailureReusePolicy:
    enabled: bool
    allowed_retry_classes: frozenset[str]
    allowed_reasons: frozenset[str]
    maximum_target_price_absolute_change: float
    maximum_target_price_fractional_change: float
    maximum_rate_change: float
    maximum_dividend_yield_change: float
    maximum_failure_age_seconds: float
    transient_retry_seconds: float
    lookup_allowance_seconds: float
    tenor_bands: tuple[FailureReuseTenorBand, ...]

    def band_for(self, tenor_seconds: float) -> FailureReuseTenorBand:
        for band in self.tenor_bands:
            if band.max_tenor_seconds is None or tenor_seconds <= band.max_tenor_seconds:
                return band
        raise AssertionError("validated failure policy has no catch-all tenor band")


@dataclass(frozen=True)
class FailureReuseDecision:
    reusable: bool
    reason: str
    retry_after: Optional[datetime] = None
    age_seconds: Optional[float] = None
    spot_move_bps: Optional[float] = None
    elapsed_fraction: Optional[float] = None
    target_price_absolute_change: Optional[float] = None
    target_price_fractional_change: Optional[float] = None


class RepricingAction(StrEnum):
    REUSE_SUCCESS = "reuse_success"
    SOLVE_PRIOR_IV_SEED = "solve_prior_iv_seed"
    REUSE_FAILURE_BACKOFF = "reuse_failure_backoff"
    SOLVE_NORMAL = "solve_normal"
    SUPPRESS_CIRCUIT_OPEN = "suppress_circuit_open"

    @property
    def requires_solver(self) -> bool:
        return self in {self.SOLVE_PRIOR_IV_SEED, self.SOLVE_NORMAL}


@dataclass(frozen=True)
class RepricingActionDecision:
    action: RepricingAction
    reason: str
    success_reuse_decision: Optional[ReuseDecision] = None
    seed_decision: Optional[IvSeedDecision] = None
    failure_reuse_decision: Optional[FailureReuseDecision] = None
    selected_iv_seed: Optional[float] = None
    selected_iv_seed_origin: str = "none"
    selected_iv_seed_snapshot_id: Optional[str] = None
    retry_after: Optional[datetime] = None


def default_seed_tenor_bands() -> list[dict[str, Optional[float]]]:
    return [
        {"max_tenor_seconds": tenor, "max_seed_age_seconds": age,
         "max_spot_move_bps": spot}
        for tenor, age, spot in (
            (1800, 30, 2.0), (7200, 60, 5.0), (21600, 120, 10.0),
            (86400, 300, 20.0), (604800, 900, 50.0), (None, 1800, 100.0),
        )
    ]


def default_failure_tenor_bands() -> list[dict[str, Optional[float]]]:
    return [
        {"max_tenor_seconds": tenor, "retry_seconds": retry,
         "max_spot_move_bps": spot, "max_elapsed_fraction": elapsed}
        for tenor, retry, spot, elapsed in (
            (1800, 30, 0.5, 0.01), (7200, 60, 1.0, 0.01),
            (86400, 120, 2.0, 0.005), (604800, 300, 5.0, 0.002),
            (None, 300, 10.0, 0.001),
        )
    ]


def _parse_ordered_bands(
    value: Any, *, required: frozenset[str], label: str
) -> list[Mapping[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    previous = 0.0
    output: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != set(required):
            raise ValueError(f"{label} entries require exactly {sorted(required)}")
        raw = item["max_tenor_seconds"]
        tenor = None if raw is None else _finite(raw)
        if raw is not None and (tenor is None or tenor <= previous):
            raise ValueError(f"{label} tenors must be positive and strictly ascending")
        if tenor is None and index != len(value) - 1:
            raise ValueError(f"only the final {label} entry may be catch-all")
        if tenor is not None:
            previous = tenor
        output.append(item)
    if output[-1]["max_tenor_seconds"] is not None:
        raise ValueError(f"{label} requires a final catch-all")
    return output


def _optional_nonnegative(context: Mapping[str, Any], key: str, default: Any) -> Optional[float]:
    raw = context.get(key, default)
    if raw is None:
        return None
    value = _finite(raw)
    if value is None or value < 0:
        raise ValueError(f"{key} must be null or finite and nonnegative")
    return value


def parse_iv_seed_policy(context: Mapping[str, Any]) -> IvSeedPolicy:
    prefix = "options_calcs.greeks.reuse.seeded_refresh."
    raw_bands = _parse_ordered_bands(
        context.get(prefix + "tenor_bands", default_seed_tenor_bands()),
        required=frozenset({"max_tenor_seconds", "max_seed_age_seconds", "max_spot_move_bps"}),
        label="seeded_refresh.tenor_bands",
    )
    bands = []
    for item in raw_bands:
        age = _finite(item["max_seed_age_seconds"])
        spot = _finite(item["max_spot_move_bps"])
        if age is None or age <= 0 or spot is None or spot < 0:
            raise ValueError("seed age must be positive and spot movement nonnegative")
        bands.append(IvSeedTenorBand(
            None if item["max_tenor_seconds"] is None else float(item["max_tenor_seconds"]),
            age, spot,
        ))
    fraction = _finite(context.get(prefix + "shadow.sample_fraction", 0.02))
    if fraction is None or not 0 <= fraction <= 1:
        raise ValueError("seeded_refresh.shadow.sample_fraction must be in [0, 1]")
    minimum = _finite(context.get("options_calcs.greeks.iv.minimum", 1.0e-4))
    maximum = _finite(context.get("options_calcs.greeks.iv.maximum", 10.0))
    if minimum is None or maximum is None or not 0 < minimum < maximum:
        raise ValueError("IV bounds must satisfy 0 < minimum < maximum")
    maximum_iv_difference = _finite(
        context.get(prefix + "shadow.maximum_iv_difference", 0.0005)
    )
    maximum_price_residual_difference = _finite(
        context.get(
            prefix + "shadow.maximum_price_residual_difference", 1.0e-7
        )
    )
    if maximum_iv_difference is None or maximum_iv_difference <= 0.0:
        raise ValueError(
            "seeded_refresh.shadow.maximum_iv_difference must be positive and finite"
        )
    if (
        maximum_price_residual_difference is None
        or maximum_price_residual_difference < 0.0
    ):
        raise ValueError(
            "seeded_refresh.shadow.maximum_price_residual_difference must be "
            "finite and nonnegative"
        )
    return IvSeedPolicy(
        enabled=_as_bool(context.get(prefix + "enabled", False)),
        minimum_iv=minimum,
        maximum_iv=maximum,
        maximum_target_price_absolute_change=_optional_nonnegative(
            context, prefix + "maximum_target_price_absolute_change", 0.25
        ),
        maximum_target_price_fractional_change=_optional_nonnegative(
            context, prefix + "maximum_target_price_fractional_change", 0.20
        ),
        maximum_rate_change=_optional_nonnegative(
            context, prefix + "maximum_rate_change", 0.0005
        ),
        maximum_dividend_yield_change=_optional_nonnegative(
            context, prefix + "maximum_dividend_yield_change", 0.0005
        ),
        tenor_bands=tuple(bands),
        shadow=SeedShadowPolicy(
            _as_bool(context.get(prefix + "shadow.enabled", False)),
            fraction,
            int(context.get(prefix + "shadow.seed", 0)),
            maximum_iv_difference,
            maximum_price_residual_difference,
        ),
    )


def _unique_string_set(value: Any, *, key: str) -> frozenset[str]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = [str(part).strip() for part in value]
    else:
        raise ValueError(f"{key} must be a sequence")
    if not values or any(not part for part in values) or len(values) != len(set(values)):
        raise ValueError(f"{key} must contain unique nonempty values")
    return frozenset(values)


def parse_failure_reuse_policy(context: Mapping[str, Any]) -> FailureReusePolicy:
    prefix = "options_calcs.greeks.failure_reuse."
    allowed_classes = _unique_string_set(
        context.get(prefix + "allowed_retry_classes", [
            RecalcRetryClass.DETERMINISTIC_NUMERICAL.value,
            RecalcRetryClass.TRANSIENT_SOLVER.value,
        ]), key=prefix + "allowed_retry_classes"
    )
    reusable_classes = {
        RecalcRetryClass.DETERMINISTIC_NUMERICAL.value,
        RecalcRetryClass.TRANSIENT_SOLVER.value,
    }
    if not allowed_classes <= reusable_classes:
        raise ValueError(
            "failure reuse retry classes are limited to "
            "deterministic_numerical and transient_solver"
        )
    allowed_reasons = _unique_string_set(
        context.get(prefix + "allowed_reasons", [
            "iv_root_not_bracketed", "iv_max_iterations",
            "iv_price_tolerance_not_reached",
            "price_residual_exceeds_tolerance", "nonfinite_solver_output",
        ]), key=prefix + "allowed_reasons"
    )
    raw_bands = _parse_ordered_bands(
        context.get(prefix + "tenor_bands", default_failure_tenor_bands()),
        required=frozenset({"max_tenor_seconds", "retry_seconds", "max_spot_move_bps", "max_elapsed_fraction"}),
        label="failure_reuse.tenor_bands",
    )
    bands = []
    for item in raw_bands:
        retry = _finite(item["retry_seconds"])
        spot = _finite(item["max_spot_move_bps"])
        elapsed = _finite(item["max_elapsed_fraction"])
        if retry is None or retry <= 0 or spot is None or spot < 0 or elapsed is None or elapsed < 0:
            raise ValueError("failure retry must be positive and guards nonnegative")
        bands.append(FailureReuseTenorBand(
            None if item["max_tenor_seconds"] is None else float(item["max_tenor_seconds"]),
            retry, spot, elapsed,
        ))
    def required_positive(name: str, default: float) -> float:
        value = _finite(context.get(prefix + name, default))
        if value is None or value <= 0:
            raise ValueError(f"{prefix + name} must be positive")
        return value
    def nonnegative(name: str, default: float) -> float:
        value = _finite(context.get(prefix + name, default))
        if value is None or value < 0:
            raise ValueError(f"{prefix + name} must be nonnegative")
        return value
    return FailureReusePolicy(
        enabled=_as_bool(context.get(prefix + "enabled", False)),
        allowed_retry_classes=allowed_classes,
        allowed_reasons=allowed_reasons,
        maximum_target_price_absolute_change=nonnegative("maximum_target_price_absolute_change", 1.0e-8),
        maximum_target_price_fractional_change=nonnegative("maximum_target_price_fractional_change", 0.0),
        maximum_rate_change=nonnegative("maximum_rate_change", 1.0e-6),
        maximum_dividend_yield_change=nonnegative("maximum_dividend_yield_change", 1.0e-8),
        maximum_failure_age_seconds=required_positive("maximum_failure_age_seconds", 900),
        transient_retry_seconds=required_positive("transient_retry_seconds", 30),
        lookup_allowance_seconds=nonnegative("lookup_allowance_seconds", 60),
        tenor_bands=tuple(bands),
    )


def assess_iv_seed(
    current: CurrentRepricingProblem,
    prior: PriorRepricingOutcome,
    policy: IvSeedPolicy,
) -> IvSeedDecision:
    if not policy.enabled:
        return IvSeedDecision(False, "seed_disabled")
    checks = (
        (current.option_symbol != prior.option_symbol, "contract_mismatch"),
        (
            current.calculation_definition_id != prior.calculation_definition_id,
            "calculation_definition_mismatch",
        ),
        (current.method != prior.method, "method_mismatch"),
        (current.algorithm_version != prior.algorithm_version, "algorithm_mismatch"),
        (current.solver_config_hash != prior.solver_config_hash, "solver_config_mismatch"),
        (current.input_basis != prior.input_basis, "input_basis_mismatch"),
        (current.dividend_model != prior.dividend_model, "dividend_model_mismatch"),
    )
    for invalid, reason in checks:
        if invalid:
            return IvSeedDecision(False, reason)
    iv = _finite(prior.implied_volatility)
    if iv is None or not policy.minimum_iv <= iv <= policy.maximum_iv or iv <= 0:
        return IvSeedDecision(False, "invalid_prior_iv")
    age = (current.source_collected_at - prior.valuation_at).total_seconds()
    source_age = (current.source_collected_at - prior.source_row_collected_at).total_seconds()
    if age < 0 or source_age <= 0:
        return IvSeedDecision(False, "prior_not_strictly_earlier")
    spot = _finite(prior.valuation_spot)
    target = _finite(prior.valuation_target_price)
    rate = _finite(prior.valuation_rate)
    dividend = _finite(prior.valuation_dividend_yield)
    if spot is None or spot <= 0 or target is None or rate is None or dividend is None:
        return IvSeedDecision(False, "missing_seed_provenance")
    band = policy.band_for(current.time_to_expiry_seconds)
    spot_move = abs(current.spot - spot) / spot * 10_000.0
    absolute_target = abs(current.target_price - target)
    fractional_target = absolute_target / abs(target) if target != 0 else math.inf
    metrics = dict(
        seed_value=iv,
        seed_origin=("prior_success" if prior.status == "complete" else "prior_partial_failure"),
        seed_snapshot_id=prior.origin_snapshot_id,
        seed_calculated_at=prior.origin_calculated_at,
        age_seconds=age,
        spot_move_bps=spot_move,
        target_price_absolute_change=absolute_target,
        target_price_fractional_change=fractional_target,
    )
    guards = (
        (age > band.max_seed_age_seconds, "seed_age_exceeded"),
        (spot_move > band.max_spot_move_bps, "seed_spot_move_exceeded"),
        (policy.maximum_target_price_absolute_change is not None and absolute_target > policy.maximum_target_price_absolute_change, "seed_target_absolute_change_exceeded"),
        (policy.maximum_target_price_fractional_change is not None and fractional_target > policy.maximum_target_price_fractional_change, "seed_target_fractional_change_exceeded"),
        (policy.maximum_rate_change is not None and abs(current.rate - rate) > policy.maximum_rate_change, "seed_rate_change_exceeded"),
        (policy.maximum_dividend_yield_change is not None and abs(current.dividend_yield - dividend) > policy.maximum_dividend_yield_change, "seed_dividend_change_exceeded"),
    )
    for invalid, reason in guards:
        if invalid:
            return IvSeedDecision(False, reason, **metrics)
    return IvSeedDecision(True, "prior_iv_seed_eligible", **metrics)


def retry_after_for_failure(
    current: CurrentRepricingProblem,
    retry_class: RecalcRetryClass | str,
    policy: FailureReusePolicy,
    attempted_at: datetime,
) -> datetime:
    selected = RecalcRetryClass(str(retry_class))
    if selected not in {
        RecalcRetryClass.DETERMINISTIC_NUMERICAL,
        RecalcRetryClass.TRANSIENT_SOLVER,
    }:
        raise ValueError(
            f"retry class {selected.value} cannot receive a failure-reuse deadline"
        )
    seconds = (
        policy.transient_retry_seconds
        if selected is RecalcRetryClass.TRANSIENT_SOLVER
        else policy.band_for(current.time_to_expiry_seconds).retry_seconds
    )
    return _timestamp(attempted_at) + timedelta(seconds=seconds)


def failure_fingerprint(
    current: CurrentRepricingProblem, *, failure_stage: str, reason: str
) -> str:
    payload = {
        "option_symbol": current.option_symbol,
        "method": current.method,
        "algorithm_version": current.algorithm_version,
        "solver_config_hash": current.solver_config_hash,
        "input_basis": current.input_basis,
        "dividend_model": current.dividend_model,
        "target_price": current.target_price,
        "spot": current.spot,
        "tenor_seconds": current.time_to_expiry_seconds,
        "rate": current.rate,
        "dividend_yield": current.dividend_yield,
        "failure_stage": str(failure_stage),
        "reason": str(reason),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def assess_failure_reuse(
    current: CurrentRepricingProblem,
    prior: PriorRepricingOutcome,
    policy: FailureReusePolicy,
    now: datetime,
) -> FailureReuseDecision:
    if not policy.enabled:
        return FailureReuseDecision(False, "failure_reuse_disabled")
    if prior.status == "complete":
        return FailureReuseDecision(False, "prior_not_failed")
    if prior.retry_class not in policy.allowed_retry_classes:
        return FailureReuseDecision(False, "retry_class_not_allowed", prior.retry_after)
    if prior.reason not in policy.allowed_reasons:
        return FailureReuseDecision(False, "failure_reason_not_allowed", prior.retry_after)
    checks = (
        (current.option_symbol != prior.option_symbol, "contract_mismatch"),
        (
            current.calculation_definition_id != prior.calculation_definition_id,
            "calculation_definition_mismatch",
        ),
        (current.method != prior.method, "method_mismatch"),
        (current.algorithm_version != prior.algorithm_version, "algorithm_mismatch"),
        (current.solver_config_hash != prior.solver_config_hash, "solver_config_mismatch"),
        (current.input_basis != prior.input_basis, "input_basis_mismatch"),
        (current.dividend_model != prior.dividend_model, "dividend_model_mismatch"),
    )
    for invalid, reason in checks:
        if invalid:
            return FailureReuseDecision(False, reason, prior.retry_after)
    if prior.retry_after is None:
        return FailureReuseDecision(False, "missing_retry_after")
    current_now = _timestamp(now)
    retry_after = _timestamp(prior.retry_after)
    age = (current_now - prior.origin_calculated_at).total_seconds()
    market_age = (current.source_collected_at - prior.valuation_at).total_seconds()
    if market_age < 0 or current.source_collected_at <= prior.source_row_collected_at:
        return FailureReuseDecision(False, "prior_not_strictly_earlier", retry_after)
    spot = _finite(prior.valuation_spot)
    target = _finite(prior.valuation_target_price)
    tenor = _finite(prior.valuation_time_to_expiry_seconds)
    rate = _finite(prior.valuation_rate)
    dividend = _finite(prior.valuation_dividend_yield)
    if any(value is None for value in (spot, target, tenor, rate, dividend)) or spot <= 0 or tenor <= 0:
        return FailureReuseDecision(False, "missing_failure_provenance", retry_after)
    absolute_target = abs(current.target_price - target)
    fractional_target = absolute_target / abs(target) if target != 0 else math.inf
    spot_move = abs(current.spot - spot) / spot * 10_000.0
    elapsed = market_age / tenor
    metrics = dict(
        retry_after=retry_after,
        age_seconds=age,
        spot_move_bps=spot_move,
        elapsed_fraction=elapsed,
        target_price_absolute_change=absolute_target,
        target_price_fractional_change=fractional_target,
    )
    band = policy.band_for(current.time_to_expiry_seconds)
    guards = (
        (current_now >= retry_after, "retry_due"),
        (age < 0 or age > policy.maximum_failure_age_seconds, "failure_age_exceeded"),
        (absolute_target > policy.maximum_target_price_absolute_change, "failure_target_absolute_change_exceeded"),
        (fractional_target > policy.maximum_target_price_fractional_change, "failure_target_fractional_change_exceeded"),
        (spot_move > band.max_spot_move_bps, "failure_spot_move_exceeded"),
        (elapsed > band.max_elapsed_fraction, "failure_elapsed_fraction_exceeded"),
        (abs(current.rate - rate) > policy.maximum_rate_change, "failure_rate_change_exceeded"),
        (abs(current.dividend_yield - dividend) > policy.maximum_dividend_yield_change, "failure_dividend_change_exceeded"),
    )
    for invalid, reason in guards:
        if invalid:
            return FailureReuseDecision(False, reason, **metrics)
    return FailureReuseDecision(True, "failed_outcome_in_backoff", **metrics)


def deterministic_seed_shadow_sample(
    current: CurrentRepricingProblem, policy: SeedShadowPolicy
) -> bool:
    if not policy.enabled or policy.sample_fraction <= 0:
        return False
    if policy.sample_fraction >= 1:
        return True
    payload = f"{policy.seed}\0{current.snapshot_id}\0{current.option_symbol}\0seed".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value < int(policy.sample_fraction * (1 << 64))


def decide_repricing_action(
    *,
    current: CurrentRepricingProblem,
    success_prior: Optional[PriorRepricingOutcome],
    failed_prior: Optional[PriorRepricingOutcome],
    reuse_policy: RepricingReusePolicy,
    seed_policy: IvSeedPolicy,
    failure_policy: FailureReusePolicy,
    now: datetime,
    failed_seed_prior: Optional[PriorRepricingOutcome] = None,
) -> RepricingActionDecision:
    reuse_decision = None
    if reuse_policy.enabled and success_prior is not None:
        reuse_decision = assess_success_reuse(current, success_prior, reuse_policy)
        if reuse_decision.reusable:
            return RepricingActionDecision(
                RepricingAction.REUSE_SUCCESS,
                reuse_decision.reason,
                success_reuse_decision=reuse_decision,
            )
    success_seed = None
    if success_prior is not None:
        success_seed = assess_iv_seed(current, success_prior, seed_policy)
        if success_seed.eligible:
            return RepricingActionDecision(
                RepricingAction.SOLVE_PRIOR_IV_SEED,
                success_seed.reason,
                success_reuse_decision=reuse_decision,
                seed_decision=success_seed,
                selected_iv_seed=success_seed.seed_value,
                selected_iv_seed_origin=success_seed.seed_origin,
                selected_iv_seed_snapshot_id=success_seed.seed_snapshot_id,
            )
    failure_decision = None
    if failed_prior is not None:
        failure_decision = assess_failure_reuse(current, failed_prior, failure_policy, now)
        if failure_decision.reusable:
            return RepricingActionDecision(
                RepricingAction.REUSE_FAILURE_BACKOFF,
                failure_decision.reason,
                success_reuse_decision=reuse_decision,
                seed_decision=success_seed,
                failure_reuse_decision=failure_decision,
                retry_after=failure_decision.retry_after,
            )
        failure_seed = assess_iv_seed(
            current, failed_seed_prior or failed_prior, seed_policy
        )
        if failure_seed.eligible:
            return RepricingActionDecision(
                RepricingAction.SOLVE_PRIOR_IV_SEED,
                (
                    "retry_due_with_partial_failure_seed"
                    if failure_decision.reason == "retry_due"
                    else "eligible_partial_failure_seed"
                ),
                success_reuse_decision=reuse_decision,
                seed_decision=failure_seed,
                failure_reuse_decision=failure_decision,
                selected_iv_seed=failure_seed.seed_value,
                selected_iv_seed_origin=failure_seed.seed_origin,
                selected_iv_seed_snapshot_id=failure_seed.seed_snapshot_id,
            )
    return RepricingActionDecision(
        RepricingAction.SOLVE_NORMAL,
        "normal_current_solve",
        success_reuse_decision=reuse_decision,
        seed_decision=success_seed,
        failure_reuse_decision=failure_decision,
    )


def suppress_for_open_circuit(
    decision: RepricingActionDecision, *, open_until: datetime
) -> RepricingActionDecision:
    if not decision.action.requires_solver:
        return decision
    return RepricingActionDecision(
        RepricingAction.SUPPRESS_CIRCUIT_OPEN,
        "solver_circuit_open",
        success_reuse_decision=decision.success_reuse_decision,
        seed_decision=decision.seed_decision,
        failure_reuse_decision=decision.failure_reuse_decision,
        selected_iv_seed=decision.selected_iv_seed,
        selected_iv_seed_origin=decision.selected_iv_seed_origin,
        selected_iv_seed_snapshot_id=decision.selected_iv_seed_snapshot_id,
        retry_after=_timestamp(open_until),
    )
