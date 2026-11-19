"""Typed public contracts for market, sector, and company analytics."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class MetricStatus(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: Literal["available", "unavailable", "not_meaningful"]
    value: float | None = None
    reason: str | None = None


class AnalyticsCoverage(BaseModel):
    requested_symbols: list[str] = Field(default_factory=list)
    successful_symbols: list[str] = Field(default_factory=list)
    insufficient_history_symbols: list[str] = Field(default_factory=list)
    missing_price_symbols: list[str] = Field(default_factory=list)
    missing_weight_symbols: list[str] = Field(default_factory=list)
    stale_symbols: list[str] = Field(default_factory=list)
    intraday_unavailable_symbols: list[str] = Field(default_factory=list)


class AnalyticsMeta(BaseModel):
    as_of: datetime
    timezone: str
    cache: Literal["hit", "miss", "bypass"]
    warnings: list[str] = Field(default_factory=list)
    sql_query_count: int = Field(ge=0)


class MarketOverviewData(BaseModel):
    model_config = ConfigDict(extra="allow")
    universe: str
    benchmark: str
    indexes: list[dict[str, Any]]
    breadth: dict[str, Any]
    sectors: list[dict[str, Any]]
    leaders: list[dict[str, Any]]
    laggards: list[dict[str, Any]]
    market_short_volume: dict[str, Any] = Field(default_factory=dict)
    most_shorted: list[dict[str, Any]] = Field(default_factory=list)
    least_shorted: list[dict[str, Any]] = Field(default_factory=list)
    coverage: AnalyticsCoverage
    warnings: list[str]
    as_of: datetime
    timezone: str


class SectorSummaryData(BaseModel):
    model_config = ConfigDict(extra="allow")
    universe: str
    grouping: str
    sectors: list[dict[str, Any]]
    as_of: datetime
    timezone: str


class SectorConstituentsData(BaseModel):
    model_config = ConfigDict(extra="allow")
    classification_id: str
    universe: str
    matched_count: int = Field(ge=0, le=5000)
    constituents: list[dict[str, Any]]
    coverage: AnalyticsCoverage
    as_of: datetime
    timezone: str


class SectorPerformanceData(BaseModel):
    model_config = ConfigDict(extra="allow")
    universe: str
    grouping: str
    series: list[dict[str, Any]]
    as_of: datetime
    timezone: str


class SectorRelationshipsData(BaseModel):
    model_config = ConfigDict(extra="allow")
    universe: str
    grouping: str
    row_ids: list[str]
    column_ids: list[str]
    matrix: list[list[dict[str, Any]]]
    pair: dict[str, Any] | None = None
    as_of: datetime
    timezone: str


class CompanyAnalyticsData(BaseModel):
    model_config = ConfigDict(extra="allow")
    symbol: str
    benchmark: str
    selected_range: str
    price_action: dict[str, Any]
    valuation: dict[str, Any]
    financial_diagnostics: dict[str, Any]
    short_volume_statistics: dict[str, Any]
    dividend_and_earnings: dict[str, Any]
    economic_catalysts: list[dict[str, Any]]
    condition_labels: list[dict[str, Any]]
    coverage: AnalyticsCoverage
    warnings: list[str]
    as_of: datetime
    timezone: str


class FINRAShortInterestObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    settlement_date: date
    short_interest: int = Field(ge=0)
    previous_short_interest: int | None = Field(default=None, ge=0)
    change_percent: float | None = None
    change_amount: int | None = None
    average_daily_volume: int | None = Field(default=None, ge=0)
    days_to_cover: float | None = Field(default=None, ge=0)
    stock_split: bool
    revised: bool


class FINRAShortInterestData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    report_limit: int = Field(ge=1, le=130)
    source: Literal["FINRA_CONSOLIDATED_SHORT_INTEREST"]
    observations: list[FINRAShortInterestObservation] = Field(default_factory=list)
    latest: FINRAShortInterestObservation | None = None
    as_of: datetime
    timezone: str


class FINRAShortVolumeData(BaseModel):
    """Bounded daily FINRA observations for an explicit symbol set."""

    model_config = ConfigDict(extra="forbid")
    symbols: list[str]
    session_limit: int = Field(ge=1, le=252)
    sessions: list[datetime] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    as_of: datetime
    timezone: str


T = TypeVar("T")


class AnalyticsResponse(BaseModel, Generic[T]):
    data: T
    meta: AnalyticsMeta


class CapabilityAvailability(BaseModel):
    market_analytics: bool
    sector_analytics: bool
    company_analytics: bool
    finra_short_volume: bool
    finra_short_interest: bool
    news: bool
    calendars: bool
    prices: bool
    fundamentals: bool
    stock_screening: bool
    risk_free_curve: bool
    futures_contract_specifications: bool
    options_chain: bool
    exact_option_quotes: bool
    exact_option_contract_terms: bool = False
    exact_option_market_history: bool = False


class CapabilitiesData(BaseModel):
    service: str
    version: str
    capability_schema_version: str
    available: CapabilityAvailability
    supported: dict[str, Any]
    intraday_data_configured: bool


class CapabilitiesResponse(BaseModel):
    data: CapabilitiesData
    meta: AnalyticsMeta
