"""Typed public contracts for Stock Analysis endpoints."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class StockMeta(BaseModel):
    count: int = Field(ge=0)
    as_of: datetime
    timezone: str
    cache: str
    stale: bool = False
    warnings: list[str] = Field(default_factory=list)
    provider: str | None = None


T = TypeVar("T")


class StockCollectionResponse(BaseModel, Generic[T]):
    data: list[T]
    meta: StockMeta


class StockObjectResponse(BaseModel, Generic[T]):
    data: T
    meta: StockMeta


class UniverseItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    security_id: int
    symbol: str
    exchange: str | None = None
    security_type: str
    active: bool
    company_id: int | None = None
    company_name: str | None = None
    classification_id: int | None = None
    taxonomy: str | None = None
    sector: str | None = None
    industry: str | None = None
    sub_industry: str | None = None
    latest_daily_bar_date: date | None = None
    daily_bar_fresh: bool


class IndexItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    index_id: str
    name: str
    benchmark_security_id: int | None = None
    benchmark_symbol: str | None = None
    source_provider: str


class ConstituentItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    security_id: int
    symbol: str
    company_id: int | None = None
    company_name: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    weight: Decimal | None = None
    source_provider: str | None = None


class ClassificationItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    classification_id: int
    taxonomy: str
    sector: str | None = None
    industry_group: str | None = None
    industry: str | None = None
    sub_industry: str | None = None
    source_provider: str | None = None


class ProfileItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    security_id: int
    symbol: str
    company_id: int
    legal_name: str
    display_name: str
    description: str | None = None
    website: str | None = None
    fiscal_year_end: str | None = None
    reporting_currency: str | None = None
    taxonomy: str | None = None
    sector: str | None = None
    industry: str | None = None
    sub_industry: str | None = None
    index_memberships: list[dict[str, Any]] = Field(default_factory=list)
    market_cap: dict[str, Any] | None = None


class CompanyInformationItem(BaseModel):
    """Database-first company identity with a provider fallback shape."""
    model_config = ConfigDict(extra="allow")
    symbol: str
    legal_name: str | None = None
    display_name: str | None = None
    description: str | None = None
    website: str | None = None
    taxonomy: str | None = None
    sector: str | None = None
    industry_group: str | None = None
    industry: str | None = None
    sub_industry: str | None = None
    source_provider: str | None = None


class ETFWeightItem(BaseModel):
    sector: str | None = None
    country: str | None = None
    weight_percent: float


class ETFExposureItem(BaseModel):
    symbol: str
    status: str
    reason: str | None = None
    fund_name: str | None = None
    asset_class: str | None = None
    objective: str | None = None
    updated_at: datetime | None = None
    sector_weights: list[ETFWeightItem] = Field(default_factory=list)
    sector_weights_status: str | None = None
    country_weights: list[ETFWeightItem] = Field(default_factory=list)
    country_weights_status: str | None = None


class FinancialItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    financial_id: int
    filing_id: int
    period_type: str
    fiscal_year: int
    fiscal_quarter: int | None = None
    period_start: date | None = None
    period_end: date
    currency: str
    revenue: Decimal | None = None
    net_income: Decimal | None = None
    diluted_eps: Decimal | None = None
    profit_margin: Decimal | None = None
    accession: str
    form_type: str
    filed_at: date
    accepted_at: datetime | None = None
    filing_url: str
    source_provider: str


class EstimateItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    estimate_id: int
    period_type: str
    target_fiscal_year: int
    target_fiscal_quarter: int | None = None
    target_period_end: date
    currency: str
    eps_mean: Decimal | None = None
    eps_analyst_count: int | None = None
    revenue_mean: Decimal | None = None
    revenue_analyst_count: int | None = None
    source_provider: str
    provider_as_of: datetime | None = None


class GuidanceItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    guidance_id: int
    issued_at: datetime
    target_period_end: date
    period_type: str
    metric: str
    value_low: Decimal | None = None
    value_high: Decimal | None = None
    unit: str
    basis: str | None = None
    status: str
    source_url: str
    source_provider: str


class NewsItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    provider_article_id: str | None = None
    title: str
    url: str
    publisher: str
    published_at: datetime
    symbols: list[str]
    snippet: str | None = None
    category: str | None = None
    sentiment: str | None = None
    source_provider: str


class ForwardPEItem(BaseModel):
    status: str
    value: Decimal | None = None
    target_fiscal_year: int | None = None
    target_period_end: date | None = None
    estimate_provider: str | None = None
    estimate_as_of: datetime | None = None
    analyst_count: int | None = None
    price_as_of: datetime | None = None
    reason: str | None = None


class OverviewItem(BaseModel):
    profile: ProfileItem
    index_memberships: list[dict[str, Any]]
    market_cap: dict[str, Any] | None = None
    annual_financials: list[FinancialItem]
    estimates: list[EstimateItem]
    guidance: list[GuidanceItem]
    events: dict[str, Any]
    current_forward_pe: ForwardPEItem


class DataStatusItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    category: str
    provider: str
    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    next_refresh_at: datetime | None = None
    result_count: int = 0
    due_count: int = 0
    error_count: int = 0
    last_error_code: str | None = None
