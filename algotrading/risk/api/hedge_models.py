"""Strict public and objective-specific Hedge Lab request contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from .portfolio_models import ContractModel


HedgeObjective = Literal[
    "Scenario Loss", "VaR / Tail Risk", "Greeks", "Factors", "Maximum Loss"
]
BoundedIdentifier = Annotated[str, Field(min_length=1, max_length=160)]
BoundedAccountId = Annotated[str, Field(min_length=1, max_length=128)]


def _normalized_ids(values: list[str], *, field_name: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item:
            raise ValueError(f"{field_name} must contain non-empty identifiers")
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


class HedgeTarget(ContractModel):
    kind: Literal["Portfolio", "Sector", "Security", "Custom", "Expiration", "Cluster"]
    label: str = Field(min_length=1, max_length=160)
    position_ids: list[BoundedIdentifier] = Field(min_length=1, max_length=5000)
    economic_position_ids: list[BoundedIdentifier] = Field(default_factory=list, max_length=5000)

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("label must not be blank")
        return normalized

    @field_validator("position_ids")
    @classmethod
    def normalize_position_ids(cls, value: list[str]) -> list[str]:
        return _normalized_ids(value, field_name="position_ids")

    @field_validator("economic_position_ids")
    @classmethod
    def normalize_economic_position_ids(cls, value: list[str]) -> list[str]:
        return _normalized_ids(value, field_name="economic_position_ids")


class ScenarioLossParameters(ContractModel):
    default_equity_shock: float = Field(default=0.0, ge=-0.50, le=0.50)
    iv_points: float = Field(default=0.0, ge=-100.0, le=100.0)
    rate_basis_points: float = Field(default=0.0, ge=-5000.0, le=5000.0)
    horizon_calendar_days: float = Field(default=0.0, ge=0.0, le=365.0)
    sector_rotation_sector: str | None = Field(default=None, min_length=1, max_length=160)
    sector_rotation_move: float | None = Field(default=None, ge=-0.50, le=0.50)

    @model_validator(mode="after")
    def paired_sector_rotation(self) -> "ScenarioLossParameters":
        if (self.sector_rotation_sector is None) != (self.sector_rotation_move is None):
            raise ValueError(
                "sector_rotation_sector and sector_rotation_move must be supplied together"
            )
        return self


class TailRiskParameters(ContractModel):
    near_confidence: float = 0.95
    far_confidence: float = 0.975
    hedge_region: Literal["near_curve", "marginal", "far_tail"] = "marginal"

    @model_validator(mode="after")
    def valid_confidences(self) -> "TailRiskParameters":
        for field_name, value in (
            ("near_confidence", self.near_confidence),
            ("far_confidence", self.far_confidence),
        ):
            if not math.isfinite(value) or not 0.50 < value < 1.0:
                raise ValueError(f"{field_name} must be greater than 0.50 and less than 1.0")
            steps = value / 0.005
            if not math.isclose(steps, round(steps), abs_tol=1e-9):
                raise ValueError(f"{field_name} must move in 0.005 steps")
        if self.near_confidence >= self.far_confidence:
            raise ValueError("near_confidence must be less than far_confidence")
        return self


class GreekTargets(ContractModel):
    delta: float | None = Field(default=None, ge=-1e15, le=1e15)
    gamma: float | None = Field(default=None, ge=-1e15, le=1e15)
    vega: float | None = Field(default=None, ge=-1e15, le=1e15)
    theta: float | None = Field(default=None, ge=-1e15, le=1e15)
    rho: float | None = Field(default=None, ge=-1e15, le=1e15)


class GreeksParameters(ContractModel):
    selected_greeks: list[Literal["delta", "gamma", "vega", "theta", "rho"]] = Field(
        default_factory=lambda: ["delta"], min_length=1, max_length=5
    )
    targets: GreekTargets = Field(default_factory=GreekTargets)

    @field_validator("selected_greeks")
    @classmethod
    def unique_greeks(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("selected_greeks must not contain duplicates")
        return value


class FactorsParameters(ContractModel):
    factor: str = Field(default="Market", min_length=1, max_length=160)
    target_exposure: float = Field(default=0.0, ge=-1e9, le=1e9)


class MaximumLossParameters(ContractModel):
    desired_loss_floor_usd: float = Field(default=0.0, ge=-1e15, le=1e15)


ObjectiveParameters = (
    ScenarioLossParameters
    | TailRiskParameters
    | GreeksParameters
    | FactorsParameters
    | MaximumLossParameters
)


def validate_objective_parameters(
    objective: HedgeObjective, values: dict[str, Any]
) -> ObjectiveParameters:
    # Validate objective parameters with the selected objective's strict schema.
    model = {
        "Scenario Loss": ScenarioLossParameters,
        "VaR / Tail Risk": TailRiskParameters,
        "Greeks": GreeksParameters,
        "Factors": FactorsParameters,
        "Maximum Loss": MaximumLossParameters,
    }[objective]
    return model.model_validate(values)


class HedgeAssessmentRequest(ContractModel):
    account_ids: list[BoundedAccountId] = Field(default_factory=lambda: ["all"], min_length=1, max_length=100)
    portfolio_snapshot_id: str = Field(min_length=1, max_length=160)
    target: HedgeTarget
    objective: HedgeObjective
    objective_parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("account_ids")
    @classmethod
    def normalize_account_ids(cls, value: list[str]) -> list[str]:
        normalized = _normalized_ids(value, field_name="account_ids")
        if any(item.casefold() == "all" for item in normalized):
            return ["all"]
        return sorted(normalized)

    @field_validator("portfolio_snapshot_id")
    @classmethod
    def normalize_snapshot_id(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def objective_contract(self) -> "HedgeAssessmentRequest":
        validated = validate_objective_parameters(self.objective, self.objective_parameters)
        self.objective_parameters = validated.model_dump(mode="python", exclude_none=True)
        return self

    def parsed_objective_parameters(self) -> ObjectiveParameters:
        return validate_objective_parameters(self.objective, self.objective_parameters)


class HedgeCandidateQueryRequest(HedgeAssessmentRequest):
    limit: int = Field(default=20, ge=1, le=50)
    evaluation_mode: Literal["full", "effects"] = "full"


class HedgeLeg(ContractModel):
    client_leg_id: str = Field(min_length=1, max_length=128)
    instrument_type: Literal["EQUITY", "OPTION", "FUTURE"]
    side: Literal["BUY", "SELL"]
    quantity: float = Field(gt=0, le=1e9)
    symbol: str = Field(min_length=1, max_length=32)
    canonical_occ_symbol: str | None = Field(default=None, min_length=1, max_length=32)
    expiration_at: datetime | None = None
    put_call: Literal["CALL", "PUT"] | None = None
    strike: float | None = Field(default=None, gt=0)
    mark: float | None = Field(default=None, ge=0)
    multiplier: float | None = Field(default=None, gt=0)
    hidden: bool = False

    @field_validator("client_leg_id")
    @classmethod
    def normalize_leg_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("client_leg_id must not be blank")
        return normalized

    @field_validator("symbol", "canonical_occ_symbol")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("symbol fields must not be blank")
        if not re.fullmatch(r"/?[A-Z0-9][A-Z0-9.\-]{0,30}", normalized):
            raise ValueError("symbol fields contain unsupported characters")
        return normalized

    @field_validator("expiration_at")
    @classmethod
    def aware_expiration(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expiration_at must be timezone aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def complete_instrument(self) -> "HedgeLeg":
        # Hidden proposal legs may remain incomplete because they are excluded from valuation.
        if self.hidden:
            return self
        if self.mark == 0:
            raise ValueError("an active leg mark must be greater than zero")
        # Require derivative-specific facts while rejecting them on incompatible instruments.
        if self.instrument_type == "OPTION":
            missing = [
                name
                for name, value in (
                    ("expiration_at", self.expiration_at),
                    ("put_call", self.put_call),
                    ("strike", self.strike),
                )
                if value is None
            ]
            if missing:
                raise ValueError("active option leg is incomplete: " + ", ".join(missing))
        elif self.instrument_type == "EQUITY" and any(
            value is not None
            for value in (
                self.canonical_occ_symbol,
                self.expiration_at,
                self.put_call,
                self.strike,
            )
        ):
            raise ValueError("option and expiration fields are not allowed on equity legs")
        elif self.instrument_type == "FUTURE" and any(
            value is not None
            for value in (self.canonical_occ_symbol, self.put_call, self.strike)
        ):
            raise ValueError("option-only fields are not allowed on future legs")
        return self


class HedgeProposalRequest(HedgeAssessmentRequest):
    legs: list[HedgeLeg] = Field(min_length=1, max_length=50)
    evaluation_mode: Literal["build", "full"] = "full"

    @field_validator("legs")
    @classmethod
    def unique_leg_ids(cls, value: list[HedgeLeg]) -> list[HedgeLeg]:
        identifiers = [item.client_leg_id for item in value]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("client_leg_id values must be unique")
        return value


__all__ = [
    "FactorsParameters",
    "GreeksParameters",
    "HedgeAssessmentRequest",
    "HedgeCandidateQueryRequest",
    "HedgeLeg",
    "HedgeObjective",
    "HedgeProposalRequest",
    "HedgeTarget",
    "MaximumLossParameters",
    "ScenarioLossParameters",
    "TailRiskParameters",
    "validate_objective_parameters",
]
