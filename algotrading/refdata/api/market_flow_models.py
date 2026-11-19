"""Typed public contracts for tracked-ETF market-flow analytics."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .analytics_models import AnalyticsMeta


MetricStatus = Literal["available", "unavailable", "not_meaningful"]
WindowId = Literal["1d", "1w", "1m", "3m", "6m"]


def _validate_nested_metrics(value: Any) -> None:
    if isinstance(value, dict):
        # Apply the metric contract recursively inside otherwise flexible analytics payloads.
        if "status" in value and "unit" in value:
            FlowMetric.model_validate(value)
        for child in value.values():
            _validate_nested_metrics(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_nested_metrics(child)


class FlowMetric(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    status: MetricStatus
    value: float | None = None
    unit: str
    reason: str | None = None
    observation_count: int = Field(default=0, ge=0)
    data_as_of: date | None = None
    anchor_as_of: date | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_availability(self) -> "FlowMetric":
        # Missing or meaningless metrics carry null values with an explicit reason.
        if self.status == "available" and self.value is None:
            raise ValueError("available metrics require a finite value")
        if self.status != "available" and self.value is not None:
            raise ValueError("unavailable/not-meaningful metrics must use a null value")
        if self.status != "available" and not self.reason:
            raise ValueError("unavailable/not-meaningful metrics require a stable reason")
        return self


class MetricTreeModel(BaseModel):
    """Validate every metric-shaped object, including metrics in flexible groups."""

    @model_validator(mode="before")
    @classmethod
    def validate_nested_metrics(cls, value: Any) -> Any:
        _validate_nested_metrics(value)
        return value


class FlowCoverage(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    etf_asset_coverage: FlowMetric
    etf_count_coverage: FlowMetric
    nav_share_coverage: FlowMetric
    consecutive_session_coverage: FlowMetric
    exposure_classification_coverage: FlowMetric
    price_coverage: FlowMetric
    finra_overlay_coverage: FlowMetric
    data_freshness: FlowMetric
    mandate_lookthrough_agreement: FlowMetric
    flow_price_agreement: FlowMetric


class WindowAvailability(BaseModel):
    model_config = ConfigDict(extra="allow")

    window: WindowId
    sessions: int = Field(ge=1, le=126)
    available: bool
    reason: str | None = None
    actual_start_date: date | None = None
    actual_end_date: date | None = None
    observation_count: int = Field(default=0, ge=0)


class MarketFlowState(BaseModel):
    code: Literal[
        "confirmed_inflow", "confirmed_outflow", "flow_price_divergence",
        "mixed_noisy", "neutral", "insufficient_data",
    ]
    label: str
    explanation: str | None = None


class MarketFlowOverviewData(MetricTreeModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    requested_window: WindowId
    actual_start_date: date | None
    actual_end_date: date
    window_availability: list[WindowAvailability]
    source_as_of: dict[str, date | None]
    scope: dict[str, Any]
    state: MarketFlowState
    headline: dict[str, Any]
    instruments: list[dict[str, Any]]
    sector_comparison: list[dict[str, Any]]
    series: list[dict[str, Any]]
    methodology: dict[str, Any]
    coverage: FlowCoverage
    warnings: list[str] = Field(default_factory=list)


class MarketFlowCategoryNode(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    node_id: str
    parent_id: str | None
    level: str
    name: str
    symbol: str | None = None
    has_children: bool
    sort_key: str
    view: Literal["all", "mandate", "economic"] = "all"
    capital_allocation: dict[str, Any] = Field(default_factory=dict)
    market_behavior: dict[str, Any] = Field(default_factory=dict)
    rotation: dict[str, Any] = Field(default_factory=dict)
    composition_and_contributors: dict[str, Any] = Field(default_factory=dict)
    short_volume_context: dict[str, Any] = Field(default_factory=dict)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    flow_series: list[dict[str, Any]] = Field(default_factory=list)


class MarketFlowCategoriesData(MetricTreeModel):
    model_config = ConfigDict(extra="allow")

    requested_window: WindowId
    actual_start_date: date | None
    actual_end_date: date
    view: Literal["all", "mandate", "economic"]
    parent_id: str | None = None
    include_constituents: bool = False
    nodes: list[MarketFlowCategoryNode]
    coverage: dict[str, Any]
    source_as_of: dict[str, date | None]
    warnings: list[str] = Field(default_factory=list)


class MarketFlowRotationPoint(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    node_id: str
    name: str
    dimension: Literal["sector", "industry", "capitalization", "fixed_income"]
    relative_flow_pressure: FlowMetric
    relative_performance: FlowMetric
    acceleration: FlowMetric
    persistence: FlowMetric
    breadth: FlowMetric
    abnormal_volume: FlowMetric
    tracked_assets: FlowMetric
    confidence: dict[str, Any]
    evidence_score: FlowMetric


class MarketFlowRotationEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int = Field(ge=1)
    evidence_type: Literal["destination", "source", "pair"]
    source_node_id: str | None = None
    source_name: str | None = None
    destination_node_id: str | None = None
    destination_name: str | None = None
    score: FlowMetric
    evidence_label: str
    contradiction: str | None = None
    reasons: list[str] = Field(default_factory=list)


class MarketFlowRotationsData(MetricTreeModel):
    model_config = ConfigDict(extra="allow")

    requested_window: WindowId
    actual_start_date: date | None
    actual_end_date: date
    dimension: Literal["sector", "industry", "capitalization", "fixed_income"]
    points: list[MarketFlowRotationPoint]
    destinations: list[MarketFlowRotationEvidence]
    sources: list[MarketFlowRotationEvidence]
    evidence_pairs: list[MarketFlowRotationEvidence]
    coverage: dict[str, Any]
    source_as_of: dict[str, date | None]
    warnings: list[str] = Field(default_factory=list)


class MarketFlowETFListItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    fund_name: str
    issuer: str
    asset_class: Literal["EQUITY", "FIXED_INCOME"]
    exposure_type: str
    exposure_name: str
    latest_reference_date: date | None
    latest_exposure_date: date | None
    earliest_reference_date: date | None
    available_windows: list[WindowAvailability]
    has_nav: bool
    has_shares: bool
    has_net_assets: bool
    has_market_price: bool
    has_exposure: bool
    has_fixed_income_characteristics: bool


class MarketFlowETFListData(MetricTreeModel):
    model_config = ConfigDict(extra="allow")

    as_of: date
    actual_start_date: date
    actual_end_date: date
    asset_class: Literal["EQUITY", "FIXED_INCOME"] | None = None
    exposure_type: str | None = None
    etfs: list[MarketFlowETFListItem]
    source_as_of: dict[str, date | None]
    warnings: list[str] = Field(default_factory=list)


class MarketFlowETFDetailData(MetricTreeModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    symbol: str
    identity: dict[str, Any]
    requested_window: WindowId
    actual_start_date: date | None
    actual_end_date: date
    window_availability: list[WindowAvailability]
    flow_series: list[dict[str, Any]]
    primary_market: dict[str, Any]
    secondary_market: dict[str, Any]
    metrics: dict[str, Any]
    exposures: dict[str, Any]
    fixed_income_characteristics: dict[str, Any] | None = None
    short_volume_context: dict[str, Any]
    data_quality: dict[str, Any]
    source_as_of: dict[str, date | None]
    warnings: list[str] = Field(default_factory=list)


T = TypeVar("T")


class MarketFlowResponse(BaseModel, Generic[T]):
    data: T
    meta: AnalyticsMeta


__all__ = [
    "FlowCoverage", "FlowMetric", "MarketFlowCategoriesData",
    "MarketFlowCategoryNode", "MarketFlowETFDetailData", "MarketFlowETFListData",
    "MarketFlowETFListItem", "MarketFlowOverviewData", "MarketFlowResponse",
    "MarketFlowRotationEvidence", "MarketFlowRotationPoint",
    "MarketFlowRotationsData", "MarketFlowState", "MetricStatus",
    "WindowAvailability", "WindowId",
]
