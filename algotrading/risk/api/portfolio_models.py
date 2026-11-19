"""Public portfolio-analysis response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from algotrading.risk.models import DataQuality, NormalizedPosition, PortfolioKpis


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CacheMeta(ContractModel):
    state: Literal["hit", "miss", "stale", "hard_stale", "suppressed"]
    age_seconds: float
    source_as_of: datetime
    refresh_suppressed_until: datetime | None = None


class PortfolioSnapshotData(ContractModel):
    account_ids: list[str]
    positions: list[NormalizedPosition]
    kpis: PortfolioKpis
    balances: list[dict[str, Any]]
    capabilities: dict[str, Any]


class PortfolioSnapshotResponse(ContractModel):
    schema_version: int = 1
    portfolio_snapshot_id: str
    configuration_version: str
    as_of: datetime
    display_timezone: str
    data_quality: DataQuality
    data: PortfolioSnapshotData
    cache: CacheMeta


class PortfolioAnalysisResponse(ContractModel):
    schema_version: int = 1
    portfolio_snapshot_id: str
    configuration_version: str
    as_of: datetime
    display_timezone: str
    data_quality: DataQuality
    methodology_version: str
    data: dict[str, Any]


class ScenarioEvaluateRequest(ContractModel):
    portfolio_snapshot_id: str
    scenario_id: str | None = None
    name: str | None = None
    underlying_shocks: dict[str, float] = Field(default_factory=dict)
    exact_underlying_shocks: dict[str, float] = Field(default_factory=dict)
    broad_equity_shock: float | None = Field(default=None, ge=-0.50, le=0.50)
    sector_shocks: dict[str, float] = Field(default_factory=dict)
    futures_root_shocks: dict[str, float] = Field(default_factory=dict)
    commodity_root_shocks: dict[str, float] = Field(default_factory=dict)
    fx_pair_shocks: dict[str, float] = Field(default_factory=dict)
    yield_curve_basis_points: dict[str, float] = Field(default_factory=dict)
    default_equity_shock: float | None = None
    iv_points: float = 0.0
    rate_basis_points: float = 0.0
    horizon_calendar_days: float = Field(default=0.0, ge=0, le=365)
    path_rule: Literal[
        "instantaneous_and_held", "log_linear_to_horizon", "explicit_dated_points"
    ] | None = "instantaneous_and_held"
    dated_path_points: dict[datetime, float] = Field(default_factory=dict)
    sector_rotation_sector: str | None = None
    sector_rotation_move: float | None = Field(default=None, ge=-0.50, le=0.50)

    @field_validator("dated_path_points")
    @classmethod
    def ordered_aware_path_points(
        cls, value: dict[datetime, float]
    ) -> dict[datetime, float]:
        for at, spot in value.items():
            if at.tzinfo is None or at.utcoffset() is None:
                raise ValueError("dated path points must use timezone-aware timestamps")
            if spot <= 0:
                raise ValueError("dated path point prices must be positive")
        # Keep explicit path points chronological for expiry and horizon evaluation.
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def explicit_path_contract(self) -> "ScenarioEvaluateRequest":
        if self.path_rule == "explicit_dated_points" and not self.dated_path_points:
            raise ValueError("explicit_dated_points requires dated_path_points")
        if self.path_rule != "explicit_dated_points" and self.dated_path_points:
            raise ValueError("dated_path_points require path_rule=explicit_dated_points")
        return self


class HedgeChainRequest(ContractModel):
    account_ids: list[str] = Field(default_factory=lambda: ["all"])
    portfolio_snapshot_id: str
    symbol: str
    contract_type: Literal["CALL", "PUT", "ALL"] = "ALL"
    strike_count: int = Field(default=20, ge=1, le=120)
    from_date: str
    to_date: str
    include_underlying_quote: bool = True


class HedgeExpirationRequest(ContractModel):
    account_ids: list[str] = Field(default_factory=lambda: ["all"])
    portfolio_snapshot_id: str
    symbol: str
    from_date: str
    to_date: str
    include_underlying_quote: bool = True


class PositionAlertPredicate(ContractModel):
    duration: Literal["1D", "1W", "1M", "3M", "6M", "1Y", "3Y", "5Y"]
    signal: Literal["Channel Position", "MACD", "RSI", "FINRA Short Volume %"]
    comparator: Literal["<", ">", ">="]
    target: float = Field(ge=-1_000_000, le=1_000_000)
    join: Literal["AND", "OR"] = "AND"

    @model_validator(mode="after")
    def signal_window_contract(self) -> "PositionAlertPredicate":
        # FINRA thresholds use fractions and a bounded published-session lookback.
        if self.signal == "FINRA Short Volume %":
            if self.duration not in {"1D", "1W", "1M"}:
                raise ValueError(
                    "FINRA Short Volume % supports only 1D, 1W, and 1M"
                )
            if not 0 <= self.target <= 1:
                raise ValueError(
                    "FINRA Short Volume % targets must be between 0 and 1"
                )
        elif self.duration == "1D":
            raise ValueError("1D is supported only for FINRA Short Volume %")
        return self


class PositionAlertCondition(ContractModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9 ._()-]*$")
    predicates: list[PositionAlertPredicate] = Field(min_length=1, max_length=12)


class PositionAlertRule(ContractModel):
    rule_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
    side: Literal["Long", "Short"]
    position_type: Literal[
        "Call Vertical", "Put Vertical", "Put Calendar", "Call Calendar",
        "Put", "Call", "Equity", "Future",
    ]
    condition_name: str = Field(min_length=1, max_length=80)
    once: bool = False


class RiskConfigurationPatch(ContractModel):
    greek_source: Literal["recalculated", "provider"] | None = None
    fallback_to_provider: bool | None = None
    tail_confidence: float | None = Field(default=None, gt=0.5, lt=1.0)
    tail_lookback_sessions: int | None = Field(default=None, ge=126, le=2520)
    cluster_minimum_correlation: float | None = Field(default=None, ge=-1, le=1)
    apply_stressed_correlations: bool | None = None
    margin_warning_utilization: float | None = Field(default=None, ge=0, le=1)
    margin_critical_utilization: float | None = Field(default=None, ge=0, le=1)
    position_alert_conditions: list[PositionAlertCondition] | None = Field(default=None, max_length=100)
    position_alert_rules: list[PositionAlertRule] | None = Field(default=None, max_length=250)
