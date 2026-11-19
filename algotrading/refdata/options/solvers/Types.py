"""Picklable, engine-neutral request and result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from datetime import datetime
from typing import Any, Mapping, Optional


QUALITY_IV_PRICE_INDETERMINATE = 16


class SolverError(Exception):
    """Base class for failures with explicit solver semantics."""


class SolverInfrastructureError(SolverError):
    """The configured pricing method could not be reached or initialized."""


class SolverUnavailableError(SolverInfrastructureError):
    """A configured solver dependency or method is unavailable."""


class SolverResourceError(SolverInfrastructureError):
    """A known solver resource limit prevented execution."""


class SolverNumericalError(SolverError):
    """A known numerical evaluation failure after the solver was reached."""


class SolverNumericalEvaluationError(SolverNumericalError):
    """A pricing callback deliberately reported an unusable numerical value."""


class SolverTransientError(SolverError):
    """A typed, retryable solver condition that is not method-wide."""


class SolverCircuitOpenError(SolverInfrastructureError):
    """A worker-local method circuit denied a solver-required call."""

    def __init__(self, method: str, open_until: datetime):
        self.method = str(method)
        self.open_until = open_until
        super().__init__(
            f"solver circuit for {self.method} is open until {open_until.isoformat()}"
        )


class RecalcRetryClass(StrEnum):
    NONE = "none"
    DETERMINISTIC_INPUT = "deterministic_input"
    DETERMINISTIC_NUMERICAL = "deterministic_numerical"
    TRANSIENT_SOLVER = "transient_solver"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


DETERMINISTIC_INPUT_REASONS = frozenset(
    {
        "invalid_iv_configuration",
        "invalid_pricing_request",
        "invalid_target_price",
        "target_price_outside_bounds",
        "invalid_expiration",
        "missing_repricing_input",
    }
)

DETERMINISTIC_NUMERICAL_REASONS = frozenset(
    {
        "iv_root_not_bracketed",
        "iv_max_iterations",
        "iv_price_tolerance_not_reached",
        "nonfinite_model_price",
        "price_residual_exceeds_tolerance",
        "nonfinite_solver_output",
        "numerical_evaluation_failed",
        "iv_price_not_identifiable",
    }
)


def classify_repricing_failure(
    *,
    status: str,
    reason: str,
    failure_stage: str,
    exception: Optional[BaseException] = None,
) -> RecalcRetryClass:
    """Classify retry semantics without guessing from exception text."""
    if str(status) in {"complete", "disabled", "skipped"}:
        return RecalcRetryClass.NONE
    if isinstance(exception, SolverInfrastructureError):
        return RecalcRetryClass.INFRASTRUCTURE
    if isinstance(exception, SolverNumericalError):
        return RecalcRetryClass.DETERMINISTIC_NUMERICAL
    if isinstance(exception, SolverTransientError):
        return RecalcRetryClass.TRANSIENT_SOLVER
    stable_reason = str(reason or "").strip().lower()
    if stable_reason in DETERMINISTIC_INPUT_REASONS:
        return RecalcRetryClass.DETERMINISTIC_INPUT
    if stable_reason in DETERMINISTIC_NUMERICAL_REASONS:
        return RecalcRetryClass.DETERMINISTIC_NUMERICAL
    # Circuit rows are explicitly infrastructure even when reconstructed from
    # persisted stable fields and no live exception object is available.
    if stable_reason == "solver_circuit_open" and failure_stage == "initialization":
        return RecalcRetryClass.INFRASTRUCTURE
    return RecalcRetryClass.UNKNOWN


@dataclass(frozen=True)
class IvHint:
    value: float
    origin: str
    source_snapshot_id: Optional[str] = None
    source_calculated_at: Optional[datetime] = None


class RepricingMethod(StrEnum):
    NONE = "none"
    CRR = "crr"
    TIAN = "tian"
    CRANK_NICOLSON = "crank_nicolson"

    @classmethod
    def parse(cls, value: object) -> "RepricingMethod":
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        normalized = {
            "cranknicholson": cls.CRANK_NICOLSON.value,
            "crank_nicholson": cls.CRANK_NICOLSON.value,
            "cranknicolson": cls.CRANK_NICOLSON.value,
        }.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(method.value for method in cls if method is not cls.NONE)
            raise ValueError(f"Unsupported repricing method {value!r}; use {allowed}") from exc


class SolverOperation(StrEnum):
    SOLVE_IV_AND_GREEKS = "solve_iv_and_greeks"
    SOLVE_IV_ONLY = "solve_iv_only"
    GREEKS = "greeks"
    DELTA = "delta"
    PRICE = "price"


class CircuitCallOutcome(StrEnum):
    """Per-item circuit accounting outcome for a valuation call."""

    SUCCESS = "success"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    UNKNOWN_OUTCOME = "unknown_outcome"
    SUPPRESSED = "suppressed"


@dataclass(frozen=True)
class PricingRequest:
    option_symbol: str
    snapshot_id: str
    put_call: str
    exercise_type: str
    spot: float
    strike: float
    tenor_seconds: float
    rate: float
    dividend_yield: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    mark: Optional[float] = None
    source_iv: Optional[float] = None
    source_collected_at: Optional[object] = None
    expiration_at: Optional[object] = None
    requested_dtype: str = "float64"


@dataclass(frozen=True)
class SolverBatchJob:
    """Pickle-safe numerical work item with explicit durable ownership.

    The owner fields are routing metadata only.  They are deliberately absent
    from the numerical grouping key so compatible work from different durable
    runs can share one physical kernel call.
    """

    job_id: str
    owner_run_id: str
    owner_snapshot_id: str
    output_index: int
    operation: SolverOperation
    request: PricingRequest
    method: RepricingMethod
    method_selection_reason: str = ""
    target_price: Optional[float] = None
    iv_hints: tuple[IvHint, ...] = ()
    requested_dtype: str = "float64"
    solver_configuration_hash: str = ""
    attempt_category: str = "normal"
    shadow: bool = False
    retry_metadata: tuple[tuple[str, str], ...] = ()
    consult_circuit: bool = True


@dataclass(frozen=True)
class ValuationResult:
    """Typed, ordered result for price, delta, or Greek valuation work."""

    operation: SolverOperation
    status: str
    reason: str
    failure_stage: str
    retry_class: RecalcRetryClass
    method: str
    requested_dtype: str
    actual_dtype: str
    precision_escalated: bool
    circuit_outcome: CircuitCallOutcome
    value: Any = None
    quality_flags: int = 0
    exception_type: str = ""
    performance: Mapping[str, Any] = field(default_factory=dict)
    precision_escalation_reason: str = "none"


@dataclass(frozen=True)
class RepricingResult:
    status: str
    reason: str
    failure_stage: str
    method: str
    method_selection_reason: str
    algorithm_version: str
    input_basis: str
    target_price: Optional[float]
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
    precision_escalation_reason: str = "none"
    precision_diagnostics: Mapping[str, float] = field(default_factory=dict)
    solver_parameters: Mapping[str, Any] = field(
        default_factory=dict
    )
    quality_flags: int = 0
    time_steps: Optional[int] = None
    space_steps: Optional[int] = None
    damping_steps: Optional[int] = None
    iv_hint_origin_used: str = "none"
    iv_hint_value_used: Optional[float] = None
    iv_hint_exact_match: bool = False
    iv_hint_local_bracket_found: bool = False
    iv_hint_evaluations: int = 0
    iv_fallback_global_probes_used: bool = False
    iv_price_evaluations: int = 0
    iv_identifiability_status: str = "not_checked"
    iv_identifiability_price_span: Optional[float] = None
    iv_identifiability_slope: Optional[float] = None
    iv_hint_flat_rejection_count: int = 0
