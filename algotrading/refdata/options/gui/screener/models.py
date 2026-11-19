"""Bounded REST models and immutable in-memory screener records."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


StrategyName = Literal[
    "vertical",
    "calendar",
    "diagonal",
    "butterfly",
    "iron_condor",
    "double_calendar",
]
PutCallName = Literal["CALL", "PUT"]


class DteConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum: int = Field(default=7, ge=0, le=365)
    maximum: int = Field(default=90, ge=1, le=365)
    front_minimum: int = Field(default=7, ge=0, le=365)
    front_maximum: int = Field(default=35, ge=1, le=365)
    back_minimum: int = Field(default=21, ge=1, le=365)
    back_maximum: int = Field(default=90, ge=1, le=365)
    minimum_expiration_separation_days: int = Field(default=7, ge=1, le=180)

    @model_validator(mode="after")
    def ordered(self) -> "DteConstraints":
        if self.minimum > self.maximum:
            raise ValueError("DTE minimum cannot exceed maximum")
        if self.front_minimum > self.front_maximum:
            raise ValueError("front DTE minimum cannot exceed maximum")
        if self.back_minimum > self.back_maximum:
            raise ValueError("back DTE minimum cannot exceed maximum")
        return self


class StrikeConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    maximum_moneyness_fraction: float = Field(default=0.10, ge=0, le=0.50)
    minimum_vertical_width_fraction: float = Field(default=0.005, gt=0, le=0.25)
    maximum_vertical_width_fraction: float = Field(default=0.07, gt=0, le=0.50)
    maximum_diagonal_separation_fraction: float = Field(default=0.10, ge=0, le=0.50)

    @model_validator(mode="after")
    def ordered(self) -> "StrikeConstraints":
        if self.minimum_vertical_width_fraction > self.maximum_vertical_width_fraction:
            raise ValueError("minimum vertical width cannot exceed maximum")
        return self


class DeltaConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum_absolute: float = Field(default=0.15, ge=0, le=1)
    maximum_absolute: float = Field(default=0.80, ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> "DeltaConstraints":
        if self.minimum_absolute > self.maximum_absolute:
            raise ValueError("minimum delta cannot exceed maximum delta")
        return self


class LiquidityThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum_open_interest: int = Field(default=10, ge=0, le=10_000_000)
    minimum_volume: int = Field(default=0, ge=0, le=10_000_000)
    maximum_relative_bid_ask_width: float = Field(default=0.50, ge=0, le=5)
    require_sizes: bool = False


class ProbabilityAssumptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drift_mode: Literal["zero", "risk_neutral", "explicit"] = "zero"
    annualized_drift: float | None = Field(default=None, ge=-2, le=2)
    simulation_paths: int | None = Field(default=None, ge=100, le=25_000)
    iv_residual_half_life_sessions: float | None = Field(default=None, ge=1, le=60)
    seed_salt: str = Field(default="", max_length=64)

    @model_validator(mode="after")
    def explicit_drift(self) -> "ProbabilityAssumptions":
        if self.drift_mode == "explicit" and self.annualized_drift is None:
            raise ValueError("annualized_drift is required for explicit drift mode")
        return self


class ExitPolicyOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")
    debit_vertical_target_fraction: float | None = Field(default=None, gt=0, le=1)
    credit_vertical_target_fraction: float | None = Field(default=None, gt=0, le=1)
    debit_calendar_diagonal_target_fraction: float | None = Field(
        default=None, gt=0, le=1
    )
    credit_calendar_diagonal_target_fraction: float | None = Field(
        default=None, gt=0, le=1
    )
    maximum_holding_sessions: int | None = Field(default=None, ge=1, le=252)


class ScreeningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    underlying_symbol: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9$^][A-Za-z0-9.$^_-]*$")
    snapshot_id: str | None = Field(default=None, max_length=64)
    enabled_strategies: list[StrategyName] = Field(
        default_factory=lambda: [
            "vertical",
            "calendar",
            "diagonal",
            "butterfly",
            "iron_condor",
            "double_calendar",
        ],
        min_length=1,
        max_length=6,
    )
    option_types: list[Literal["call", "put"]] = Field(
        default_factory=lambda: ["call", "put"], min_length=1, max_length=2
    )
    outlook: Literal["any", "bullish", "bearish", "neutral"] = "any"
    dte: DteConstraints = Field(default_factory=DteConstraints)
    strikes: StrikeConstraints = Field(default_factory=StrikeConstraints)
    deltas: DeltaConstraints = Field(default_factory=DeltaConstraints)
    liquidity: LiquidityThresholds = Field(default_factory=LiquidityThresholds)
    minimum_execution_quality: Literal["poor", "questionable", "good"] = "questionable"
    maximum_quote_age_seconds: int = Field(default=120, ge=1, le=86_400)
    maximum_debit: float | None = Field(default=None, gt=0, le=10_000_000)
    maximum_capital_at_risk: float | None = Field(default=None, gt=0, le=100_000_000)
    result_count_per_strategy: int = Field(default=20, ge=1, le=50)
    delta_hedge_mode: Literal["none", "static", "both"] = "none"
    price_basis: Literal["mid", "bid", "ask"] = "mid"
    entry_basis: Literal["natural", "midpoint"] = "natural"
    surface_relative_value: bool = True
    include_nonpositive_ev: bool = False
    include_poor_execution: bool = False
    probability: ProbabilityAssumptions = Field(default_factory=ProbabilityAssumptions)
    exit_policy: ExitPolicyOverrides = Field(default_factory=ExitPolicyOverrides)

    @field_validator("underlying_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(UUID(value))

    @field_validator("enabled_strategies", "option_types")
    @classmethod
    def unique_values(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("duplicate values are not allowed")
        return value


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    source_collected_at: datetime
    underlying_symbol: str
    underlying_bid: float | None
    underlying_ask: float | None
    underlying_mark: float
    calculation_definition_id: int
    calculation_run_id: str | None = None
    calculation_attempt_id: str | None = None


@dataclass(frozen=True)
class ExpirationInput:
    expiration_at: datetime
    time_to_expiry_seconds: int
    risk_free_rate: float
    dividend_yield: float
    forward_price: float
    calculation_definition_id: int
    quality_flags: int = 0


@dataclass(frozen=True)
class SurfaceFit:
    expiration_at: datetime
    price_basis: str
    pricing_method: str
    fit_method: str
    svi_a: float | None
    svi_b: float | None
    svi_rho: float | None
    svi_m: float | None
    svi_sigma: float | None
    pchip_k: tuple[float, ...]
    pchip_total_variance: tuple[float, ...]
    minimum_log_forward_moneyness: float
    maximum_log_forward_moneyness: float
    weighted_rmse: float
    quality_flags: int
    calculation_definition_id: int
    surface_definition_id: int


@dataclass(frozen=True)
class ContractRecord:
    contract_id: int
    option_symbol: str
    put_call: PutCallName
    strike: float
    expiration_at: datetime
    dte: float
    exercise_type: str
    multiplier: float
    is_mini: bool
    is_nonstandard: bool
    symbol_map_mismatch: bool
    bid: float | None
    ask: float | None
    bid_size: int | None
    ask_size: int | None
    volume: int | None
    open_interest: int | None
    quote_time: datetime | None
    effective_iv: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    rho: float | None
    iv_value_source: str
    greeks_value_source: str
    result_origin: str
    quality_flags: int
    active_failure_reason: str | None = None
    fitted_iv: float | None = None
    iv_residual: float | None = None
    pricing_method: str | None = None

    @property
    def midpoint(self) -> float:
        if self.bid is None or self.ask is None:
            raise ValueError("midpoint is unavailable without bid and ask")
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class CandidateLeg:
    side: Literal["long", "short"]
    contract: ContractRecord
    quantity: int = 1

    @property
    def sign(self) -> int:
        return 1 if self.side == "long" else -1


@dataclass(frozen=True)
class CandidateDraft:
    strategy: StrategyName
    variant: str
    outlook: Literal["bullish", "bearish", "neutral"]
    legs: tuple[CandidateLeg, ...]
    hedge_mode: Literal["none", "static"] = "none"
    hedge_shares: int = 0
    prescore: float = 0.0
    reason_factors: tuple[tuple[str, float], ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Universe:
    snapshot: SnapshotRecord
    contracts: tuple[ContractRecord, ...]
    expiration_inputs: dict[datetime, ExpirationInput]
    surface_fits: dict[tuple[datetime, str], SurfaceFit]
    atm_volatility: float
    forward_variance: tuple[dict[str, Any], ...] = ()
    compiled_surface_evaluators: dict[tuple[datetime, str], Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )


class LegResult(BaseModel):
    side: Literal["long", "short"]
    quantity: int
    contract_id: int
    option_symbol: str
    put_call: PutCallName
    strike: float
    expiration_at: datetime
    dte: float
    exercise_type: str
    multiplier: float
    bid: float
    midpoint: float
    ask: float
    bid_size: int | None
    ask_size: int | None
    open_interest: int | None
    volume: int | None
    quote_time: datetime
    effective_iv: float
    fitted_surface_iv: float | None
    iv_residual: float | None
    delta: float
    gamma: float
    theta: float
    vega: float
    iv_value_source: str
    greeks_value_source: str
    execution_reasons: list[str]


class ExecutionResult(BaseModel):
    rating: Literal["good", "questionable", "poor"]
    score: float
    estimated_entry_slippage: float
    spread_reasons: list[str]
    per_leg_reasons: dict[str, list[str]]


class ScoreResult(BaseModel):
    overall: float
    expected_value_return: float
    probability: float
    execution_quality: float
    robustness: float
    capital_efficiency: float


class RationaleResult(BaseModel):
    sentence: str
    reason_codes: list[str]
    values: dict[str, float]


class CandidateResult(BaseModel):
    candidate_id: str
    strategy: StrategyName
    variant: str
    outlook: Literal["bullish", "bearish", "neutral"]
    legs: list[LegResult]
    synthetic_bid: float
    synthetic_midpoint: float
    synthetic_ask: float
    entry_value: float
    opening_classification: Literal["debit", "credit"]
    target_exit_value: float
    target_profit: float
    expected_holding_sessions: float | None
    expected_holding_time_basis: Literal["target_hit_median", "planned_maximum"]
    planned_maximum_holding_sessions: int
    model_probability_of_profit: float
    probability_of_reaching_target: float
    expected_pnl_after_costs: float
    return_on_capital_at_risk: float | None
    exact_maximum_profit: float | None
    exact_maximum_loss: float | None
    maximum_profit_loss_basis: str
    breakeven: float | None
    breakevens: list[float] = Field(default_factory=list)
    modeled_pnl_range: tuple[float, float]
    pnl_percentiles: dict[str, float]
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    static_hedge_shares: int | None
    execution_quality: ExecutionResult
    surface_relative_factors: dict[str, float]
    scores: ScoreResult
    rationale: RationaleResult
    warnings: list[str]


class ScreeningFunnel(BaseModel):
    contracts_loaded: int = 0
    contracts_eligible: int = 0
    contracts_rejected: int = 0
    structural_combinations_generated: int = 0
    candidates_passing_structural_validation: int = 0
    candidates_passing_liquidity_filters: int = 0
    candidates_pre_scored: int = 0
    candidates_fully_evaluated: int = 0
    candidates_rejected_after_scenario_evaluation: int = 0
    results_returned: int = 0


class RejectionDiagnostics(BaseModel):
    by_reason: dict[str, int] = Field(default_factory=dict)
    examples: dict[str, list[str]] = Field(default_factory=dict)
    dominant_reason: str | None = None


class ScreeningResponse(BaseModel):
    request_id: str
    model_version: str
    snapshot_id: str
    snapshot_timestamp: datetime
    underlying_bid: float | None
    underlying_ask: float | None
    underlying_mark: float
    applied_assumptions: dict[str, Any]
    candidates: list[CandidateResult]
    funnel: ScreeningFunnel
    rejections: RejectionDiagnostics
    cache_hit: bool
    elapsed_ms: float
    warnings: list[str]
    performance_diagnostics: dict[str, Any] = Field(default_factory=dict)


class CapabilitiesResponse(BaseModel):
    strategies: list[StrategyName]
    static_delta_hedging: bool
    dynamic_delta_hedging: bool
    price_bases: list[str]
    directional_assumption_modes: list[str]
    minimum_dte: int
    maximum_dte: int
    maximum_results_per_strategy: int
    maximum_simulation_paths: int
    maximum_holding_sessions: int
    model_version: str


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


__all__ = [name for name in globals() if not name.startswith("_")]
