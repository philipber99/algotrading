"""Immutable, explicit-unit models consumed by risk calculators."""

from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AssetType = Literal[
    "EQUITY", "ETF", "MUTUAL_FUND", "OPTION", "FUTURE", "CASH", "UNKNOWN"
]
QualityStatus = Literal["usable", "usable_with_warnings", "insufficient_data"]


class ImmutableModel(BaseModel):
    # Shared snapshots reject unknown fields and cannot be mutated by calculators.
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SourceState(ImmutableModel):
    name: str
    as_of: datetime | None = None
    age_seconds: float | None = None
    state: str
    method_version: str | None = None


class Coverage(ImmutableModel):
    positions_total: int = 0
    positions_modeled: int = 0
    market_value_percent: float | None = None
    gross_delta_percent: float | None = None


class CoverageMeasure(ImmutableModel):
    """Coverage for one economic field, with the denominator fixed up front."""

    eligible_positions: int = Field(default=0, ge=0)
    modeled_positions: int = Field(default=0, ge=0)
    denominator_usd: float | None = Field(default=None, ge=0)
    modeled_usd: float | None = Field(default=None, ge=0)
    percent: float | None = Field(default=None, ge=0, le=100)
    count_percent: float | None = Field(default=None, ge=0, le=100)
    weight_basis: str = "eligible_position_count"
    missing_position_ids: tuple[str, ...] = ()
    unavailable_denominator_position_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def consistent_counts(self) -> "CoverageMeasure":
        if self.modeled_positions > self.eligible_positions:
            raise ValueError("modeled_positions cannot exceed eligible_positions")
        return self


class RiskCoverage(ImmutableModel):
    instrument_terms: CoverageMeasure = CoverageMeasure()
    market_value: CoverageMeasure = CoverageMeasure()
    delta: CoverageMeasure = CoverageMeasure()
    gamma: CoverageMeasure = CoverageMeasure()
    vega: CoverageMeasure = CoverageMeasure()
    theta: CoverageMeasure = CoverageMeasure()
    rho: CoverageMeasure = CoverageMeasure()
    scenario: CoverageMeasure | None = None
    tail: CoverageMeasure | None = None
    option_price_history: CoverageMeasure | None = None
    option_surface: CoverageMeasure | None = None


class DataQuality(ImmutableModel):
    status: QualityStatus
    sources: tuple[SourceState, ...] = ()
    coverage: Coverage = Coverage()
    risk_coverage: RiskCoverage | None = None
    warnings: tuple[str, ...] = ()


class OptionDeliverableComponent(ImmutableModel):
    component_type: Literal["security", "cash", "other"]
    security_identifier: str | None = None
    quantity: float | None = Field(default=None, ge=0)
    cash_amount: float | None = None
    currency: str = "USD"
    description: str | None = None


class OptionDeliverable(ImmutableModel):
    description: str
    shares_per_contract: float | None = Field(default=None, ge=0)
    cash_per_contract: float | None = None
    currency: str = "USD"
    components: tuple[OptionDeliverableComponent, ...] = ()
    aggregate_exercise_amount: float | None = Field(default=None, ge=0)
    exercise_currency: str = "USD"
    fully_supported: bool = True


class FieldValidity(ImmutableModel):
    source_at: datetime | None = None
    freshness_state: str = "not_evaluated"
    valid_for_risk: bool = False
    method: str | None = None
    reason: str | None = None


class InstrumentTerms(ImmutableModel):
    """Canonical economic terms used by every Risk calculation."""

    methodology_version: str = "canonical_instrument_terms_v4"
    canonical_instrument_id: str
    asset_type: AssetType
    contract_subtype: Literal[
        "equity",
        "etf",
        "mutual_fund",
        "equity_option",
        "etf_option",
        "index_option",
        "future",
        "futures_option",
        "cash",
        "unknown",
    ]
    underlying_identifier: str | None = None
    futures_root: str | None = None
    currency: str = "USD"
    price_multiplier: float | None = Field(default=None, gt=0)
    quantity_unit: str
    minimum_tick: float | None = Field(default=None, gt=0)
    tick_value: float | None = Field(default=None, gt=0)
    quote_convention: str | None = None
    strike: float | None = Field(default=None, gt=0)
    put_call: Literal["CALL", "PUT"] | None = None
    exercise_style: Literal["american", "european", "unknown"] | None = None
    deliverable: OptionDeliverable | None = None
    adjusted_contract: bool | None = None
    listing_date: date | None = None
    expiration_at: datetime | None = None
    last_trade_at: datetime | None = None
    settlement_at: datetime | None = None
    settlement_style: str | None = None
    exchange_timezone: str | None = None
    settlement_type: Literal["physical", "cash", "unknown"] | None = None
    exercise_cutoff_at: datetime | None = None
    exchange_calendar: str | None = None
    calendar_version: str | None = None
    economic_exposure_type: Literal[
        "equity", "rates", "commodity", "currency", "mixed", "unknown"
    ] = "unknown"
    source: str
    source_as_of: datetime | None = None
    source_version: str
    quality_status: Literal["standard", "adjusted", "partial", "unsupported"]
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def coherent_tick_economics(self) -> "InstrumentTerms":
        # Tick value must reconcile with the contract's price-to-dollar conversion.
        if (
            self.minimum_tick is not None
            and self.price_multiplier is not None
            and self.tick_value is not None
            and not math.isclose(
                self.minimum_tick * self.price_multiplier,
                self.tick_value,
                rel_tol=1e-9,
                abs_tol=1e-8,
            )
        ):
            raise ValueError("minimum_tick * price_multiplier must equal tick_value")
        return self


class NormalizedPosition(ImmutableModel):
    position_id: str
    economic_position_id: str
    account_id: str
    instrument_id: str
    asset_type: AssetType
    exact_symbol: str
    canonical_occ_symbol: str | None = None
    provider_option_symbol: str | None = None
    underlying_symbol: str | None = None
    futures_root: str | None = None
    put_call: Literal["CALL", "PUT"] | None = None
    strike: float | None = None
    expiration_at: datetime | None = None
    signed_quantity: float
    multiplier: float | None = None
    multiplier_source: str | None = None
    instrument_terms: InstrumentTerms | None = None
    raw_average_price: float | None = None
    normalized_average_price: float | None = None
    normalization_method: str | None = None
    signed_cost_basis_usd: float | None = None
    mark: float | None = None
    signed_market_value_usd: float | None = None
    signed_liquidation_value_usd: float | None = None
    signed_notional_exposure_usd: float | None = None
    delta_exposure_usd: float | None = None
    unrealized_pnl_usd: float | None = None
    current_day_pnl_usd: float | None = None
    sector: str = "Unclassified"
    industry: str | None = None
    delta: float | None = None
    gamma: float | None = None
    vega_per_1pct: float | None = None
    theta_per_day: float | None = None
    rho_per_1pct: float | None = None
    implied_volatility: float | None = None
    greek_source: str | None = None
    quote_as_of: datetime | None = None
    quote_source: str | None = None
    field_validity: dict[str, FieldValidity] = Field(default_factory=dict)
    data_quality: DataQuality
    raw_broker_fields: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def populate_additive_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        output = dict(value)
        # Populate either legacy value alias before enforcing their equality.
        if "signed_liquidation_value_usd" not in output:
            output["signed_liquidation_value_usd"] = output.get(
                "signed_market_value_usd"
            )
        if "signed_market_value_usd" not in output:
            output["signed_market_value_usd"] = output.get(
                "signed_liquidation_value_usd"
            )
        return output

    @model_validator(mode="after")
    def legacy_fields_match_terms(self) -> "NormalizedPosition":
        if self.instrument_terms is not None:
            if self.multiplier != self.instrument_terms.price_multiplier:
                raise ValueError("legacy multiplier must equal instrument_terms.price_multiplier")
            if self.expiration_at != self.instrument_terms.expiration_at:
                raise ValueError("legacy expiration_at must equal instrument_terms.expiration_at")
        if self.signed_market_value_usd != self.signed_liquidation_value_usd:
            raise ValueError(
                "legacy signed_market_value_usd must equal signed_liquidation_value_usd"
            )
        return self


class PortfolioSnapshot(ImmutableModel):
    schema_version: int = 1
    portfolio_snapshot_id: str
    configuration_version: str
    as_of: datetime
    display_timezone: str = "America/Los_Angeles"
    account_ids: tuple[str, ...]
    portfolio_value_usd: float | None = None
    positions: tuple[NormalizedPosition, ...]
    data_quality: DataQuality


class PortfolioKpis(ImmutableModel):
    portfolio_value_usd: float | None
    gross_delta_exposure_usd: float | None
    net_delta_exposure_usd: float | None
    gross_market_value_usd: float | None
    leverage: float | None
    options_vega_per_1pct_usd: float | None
    options_theta_per_day_usd: float | None
    margin_utilization: float | None
    margin_loan_balance_usd: float | None = None
    maintenance_requirement_usd: float | None = None
    net_liquidating_value_usd: float | None = None
    modeled_market_value_percent: float | None
    modeled_gross_delta_percent: float | None
    warnings: tuple[str, ...] = ()
