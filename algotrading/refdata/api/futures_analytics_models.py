"""Typed response contracts for futures analytics endpoints."""

from __future__ import annotations

from datetime import date
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .analytics_models import AnalyticsMeta


FuturesMetricStatus = Literal[
    "available", "insufficient_history", "unavailable", "not_meaningful"
]


class FuturesMetric(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    status: FuturesMetricStatus
    value: float | None = None
    unit: str
    reason: str | None = None
    observation_count: int = Field(default=0, ge=0)
    required_observation_count: int = Field(default=0, ge=0)
    data_as_of: date | None = None
    anchor_as_of: date | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_availability(self) -> "FuturesMetric":
        # Availability must agree with both the numeric value and the explanatory reason.
        if self.status == "available" and self.value is None:
            raise ValueError("available futures metrics require a finite value")
        if self.status != "available" and self.value is not None:
            raise ValueError("unavailable futures metrics must use a null value")
        if self.status != "available" and not self.reason:
            raise ValueError("unavailable futures metrics require a stable reason")
        return self


def _validate_nested_metrics(value: Any) -> None:
    if isinstance(value, dict):
        # Validate metric-shaped objects even inside flexible nested response groups.
        if "status" in value and "unit" in value:
            FuturesMetric.model_validate(value)
        for child in value.values():
            _validate_nested_metrics(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_nested_metrics(child)


class FuturesMetricTree(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def validate_metrics(cls, value: Any) -> Any:
        _validate_nested_metrics(value)
        return value


class FuturesPositioningData(FuturesMetricTree):
    requested_change_window: Literal["1w", "4w", "13w"]
    requested_history_window: Literal["1y", "3y", "5y"]
    resolved_as_of: date
    source_as_of: dict[str, Any]
    items: list[dict[str, Any]]
    coverage: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class FuturesParticipationData(FuturesMetricTree):
    requested_windows: list[Literal["1d", "5d", "20d"]]
    resolved_as_of: date
    source_as_of: dict[str, Any]
    items: list[dict[str, Any]]
    coverage: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class FuturesTermStructureData(FuturesMetricTree):
    product_symbol: str
    resolved_as_of: date
    data_date: date | None = None
    comparison_date: date | None = None
    current_curve: list[dict[str, Any]]
    comparison_curve: list[dict[str, Any]]
    m1_m2_pct: FuturesMetric
    m1_m3_pct: FuturesMetric
    curve_state: dict[str, Any]
    coverage: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class FuturesCrowdingData(FuturesMetricTree):
    requested_history_window: Literal["1y", "3y", "5y"]
    requested_change_window: Literal["1w", "4w", "13w"]
    resolved_as_of: date
    items: list[dict[str, Any]]
    coverage: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class FuturesMarketHealthData(FuturesMetricTree):
    requested_position_window: Literal["1w", "4w", "13w"]
    requested_history_window: Literal["1y", "3y", "5y"]
    resolved_as_of: date
    source_as_of: dict[str, Any]
    state: dict[str, Any]
    dimensions: dict[str, Any]
    headline_cards: list[dict[str, Any]]
    equity_summary: list[dict[str, Any]]
    exposure_changes: list[dict[str, Any]]
    participation_summary: list[dict[str, Any]]
    volatility_summary: dict[str, Any]
    cross_asset: list[dict[str, Any]]
    crowding_summary: dict[str, Any]
    coverage: dict[str, Any]
    methodology: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


T = TypeVar("T")


class FuturesAnalyticsResponse(BaseModel, Generic[T]):
    data: T
    meta: AnalyticsMeta


__all__ = [
    "FuturesAnalyticsResponse", "FuturesCrowdingData",
    "FuturesMarketHealthData", "FuturesMetric", "FuturesMetricStatus",
    "FuturesParticipationData", "FuturesPositioningData",
    "FuturesTermStructureData",
]
