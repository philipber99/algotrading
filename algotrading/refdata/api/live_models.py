"""Typed contracts for live option and external reference-data ingress."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class OptionsQuoteRequest(StrictModel):
    symbols: list[str] = Field(min_length=1)

    @field_validator("symbols")
    @classmethod
    def non_blank_symbols(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("symbols must contain non-empty strings")
        return value


class OptionQuoteItem(StrictModel):
    requested_symbol: str
    canonical_occ_symbol: str | None = None
    provider_option_symbol: str | None = None
    status: Literal["ok", "missing", "invalid", "provider_error"]
    bid: float | None = None
    ask: float | None = None
    mark: float | None = None
    last: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    last_size: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    quote_time: datetime | None = None
    trade_time: datetime | None = None
    source_kind: Literal["exchange_quote", "completed_session_mark"] = "exchange_quote"
    underlying_price: float | None = None
    intrinsic_value: float | None = None
    time_value: float | None = None
    provider_iv: float | None = None
    provider_iv_raw: float | None = None
    provider_delta: float | None = None
    provider_gamma: float | None = None
    provider_theta_per_day: float | None = None
    provider_vega_per_1pct: float | None = None
    provider_rho_per_1pct: float | None = None
    theoretical_option_value: float | None = None
    implied_yield_raw: float | None = None
    open_interest: float | None = None
    volume: float | None = None
    security_status: str | None = None
    net_change: float | None = None
    mark_change: float | None = None
    mark_percent_change: float | None = None
    percent_change: float | None = None
    multiplier: float | None = Field(default=None, gt=0)
    deliverable: str | None = None
    deliverable_components: list[dict[str, Any]] = Field(default_factory=list)
    aggregate_exercise_amount: float | None = Field(default=None, ge=0)
    asset_subtype: str | None = None
    adjusted_contract: bool | None = None
    shares_per_contract: float | None = Field(default=None, ge=0)
    cash_per_contract: float | None = None
    exercise_cutoff_at: datetime | None = None
    last_trade_at: datetime | None = None
    settlement_at: datetime | None = None
    exchange_calendar: str | None = None
    exchange_timezone: str | None = None
    calendar_version: str | None = None
    exercise_style: str | None = None
    settlement_style: str | None = None
    settlement_type: str | None = None
    listing_date: date | None = None
    expiration_at: datetime | None = None
    source: str = "TDAReferenceData"
    source_units: dict[str, str] = Field(default_factory=dict)
    stale_after: datetime | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class LiveResponseMeta(StrictModel):
    as_of: datetime
    source: str
    cache: Literal["hit", "miss", "bypass"]
    warnings: list[str] = Field(default_factory=list)


class OptionsQuoteResponse(StrictModel):
    data: list[OptionQuoteItem]
    missing_symbols: list[str] = Field(default_factory=list)
    meta: LiveResponseMeta


class OptionContractTermsRequest(StrictModel):
    symbols: list[str] = Field(min_length=1, max_length=500)

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, value: list[str]) -> list[str]:
        from .option_symbols import parse_occ_symbol

        output = [parse_occ_symbol(item).canonical for item in value]
        if len(set(output)) != len(output):
            raise ValueError("symbols must not contain duplicates after normalization")
        return output


class OptionDeliverableComponentItem(StrictModel):
    component_type: Literal["security", "cash", "other"]
    security_identifier: str | None = None
    quantity: float | None = Field(default=None, ge=0)
    cash_amount: float | None = None
    currency: str = "USD"
    description: str | None = None


class OptionContractTermsItem(StrictModel):
    requested_symbol: str
    canonical_occ_symbol: str | None = None
    terms_status: Literal["standard", "adjusted", "partial", "unsupported"]
    asset_subtype: Literal["equity_option", "etf_option", "index_option", "futures_option", "unknown"]
    underlying_identifier: str | None = None
    currency: str = "USD"
    price_multiplier: float | None = Field(default=None, gt=0)
    quantity_unit: Literal["contracts"] = "contracts"
    minimum_tick: float | None = Field(default=None, gt=0)
    quote_convention: str | None = None
    strike: float | None = Field(default=None, gt=0)
    put_call: Literal["CALL", "PUT"] | None = None
    exercise_style: Literal["american", "european", "unknown"] = "unknown"
    deliverable: str | None = None
    shares_per_contract: float | None = Field(default=None, ge=0)
    cash_per_contract: float | None = None
    deliverable_components: list[OptionDeliverableComponentItem] = Field(default_factory=list)
    aggregate_exercise_amount: float | None = Field(default=None, ge=0)
    exercise_currency: str = "USD"
    adjusted_contract: bool | None = None
    listing_date: date | None = None
    expiration_at: datetime | None = None
    last_trade_at: datetime | None = None
    settlement_at: datetime | None = None
    settlement_style: str | None = None
    exchange_timezone: str | None = None
    settlement_type: Literal["physical", "cash", "unknown"] = "unknown"
    exercise_cutoff_at: datetime | None = None
    exchange_calendar: str | None = None
    calendar_version: str | None = None
    source: str
    source_as_of: datetime | None = None
    source_version: str = "refdata_exact_option_terms_v4"
    warnings: list[str] = Field(default_factory=list)


class OptionContractTermsResponse(StrictModel):
    data: list[OptionContractTermsItem]
    meta: LiveResponseMeta


class OptionMarketHistoryRequest(StrictModel):
    symbols: list[str] = Field(min_length=1, max_length=500)
    start_at: datetime
    end_at: datetime
    interval: Literal["1d"] = "1d"
    include_greeks: bool = True

    @field_validator("symbols")
    @classmethod
    def normalize_occ_symbols(cls, value: list[str]) -> list[str]:
        from .option_symbols import parse_occ_symbol

        normalized: list[str] = []
        for symbol in value:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("symbols must contain non-empty OCC symbols")
            try:
                # Compare exact-contract requests in canonical OCC form before detecting duplicates.
                normalized.append(parse_occ_symbol(symbol).canonical)
            except ValueError as exc:
                raise ValueError(f"invalid OCC option symbol: {symbol!r}") from exc
        if len(set(normalized)) != len(normalized):
            raise ValueError("symbols must not contain duplicates after normalization")
        return normalized

    @field_validator("start_at", "end_at")
    @classmethod
    def aware_utc(cls, value: datetime) -> datetime:
        from datetime import timezone

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def ordered_interval(self) -> "OptionMarketHistoryRequest":
        if self.end_at <= self.start_at:
            raise ValueError("end_at must be after start_at")
        return self


class OptionMarketHistoryItem(StrictModel):
    requested_symbol: str
    canonical_occ_symbol: str | None = None
    at: datetime
    session_date: date
    observed_at: datetime | None = None
    status: Literal["ok", "missing", "invalid", "provider_error"]
    bid: float | None = None
    ask: float | None = None
    mark: float | None = None
    last: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    underlying_price: float | None = None
    provider_iv: float | None = None
    provider_delta: float | None = None
    provider_gamma: float | None = None
    provider_theta_per_day: float | None = None
    provider_vega_per_1pct: float | None = None
    provider_rho_per_1pct: float | None = None
    surface_state: dict[str, Any] | None = None
    mark_basis: str | None = None
    source: str
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    @field_validator("at", "observed_at")
    @classmethod
    def aware_utc(cls, value: datetime | None) -> datetime | None:
        from datetime import timezone

        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("history timestamps must be timezone aware")
        return value.astimezone(timezone.utc)


class OptionMarketHistoryResponse(StrictModel):
    data: list[OptionMarketHistoryItem]
    missing_symbols: list[str] = Field(default_factory=list)
    meta: LiveResponseMeta
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class OptionChainRequest(StrictModel):
    symbol: str = Field(min_length=1, max_length=16, pattern=r"^[A-Za-z0-9.\-]+$")
    contract_type: Literal["CALL", "PUT", "ALL"] = "ALL"
    strike_count: int = Field(default=20, ge=1)
    from_date: date
    to_date: date
    include_underlying_quote: bool = True

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def ordered_dates(self) -> "OptionChainRequest":
        if self.to_date < self.from_date:
            raise ValueError("to_date must be on or after from_date")
        return self


class OptionChainContract(StrictModel):
    canonical_occ_symbol: str | None = None
    provider_option_symbol: str | None = None
    put_call: Literal["CALL", "PUT"]
    expiration: date
    strike: float
    multiplier: float | None = None
    deliverable: str | None = None
    bid: float | None = None
    ask: float | None = None
    mark: float | None = None
    last: float | None = None
    quote_time: datetime | None = None
    trade_time: datetime | None = None
    volume: float | None = None
    open_interest: float | None = None
    provider_iv_raw: float | None = None
    provider_delta: float | None = None
    provider_gamma: float | None = None
    provider_theta_raw: float | None = None
    provider_vega_raw: float | None = None
    provider_rho_raw: float | None = None
    warnings: list[str] = Field(default_factory=list)


class OptionChainData(StrictModel):
    symbol: str
    contract_type: Literal["CALL", "PUT", "ALL"]
    strategy: Literal["SINGLE"] = "SINGLE"
    strike_count: int
    from_date: date
    to_date: date
    underlying_quote: dict[str, Any] | None = None
    contracts: list[OptionChainContract]
    provider_snapshot_time: datetime | None = None
    retrieved_at: datetime


class OptionChainResponse(StrictModel):
    data: OptionChainData
    meta: LiveResponseMeta


class OptionExpirationRequest(StrictModel):
    symbol: str = Field(min_length=1, max_length=16, pattern=r"^[A-Za-z0-9.\-]+$")
    from_date: date
    to_date: date

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def ordered_dates(self) -> "OptionExpirationRequest":
        if self.to_date < self.from_date:
            raise ValueError("to_date must be on or after from_date")
        return self


class OptionExpirationItem(StrictModel):
    expiration: date
    days_to_expiration: int | None = Field(default=None, ge=0)
    expiration_type: str | None = None
    settlement_type: str | None = None
    option_roots: list[str] = Field(default_factory=list)
    standard: bool | None = None


class OptionExpirationData(StrictModel):
    symbol: str
    from_date: date
    to_date: date
    expirations: list[OptionExpirationItem]
    retrieved_at: datetime


class OptionExpirationResponse(StrictModel):
    data: OptionExpirationData
    meta: LiveResponseMeta


class RiskFreeAnchor(StrictModel):
    series_id: str
    tenor_days: int = Field(ge=0)
    observation_date: date | None = None
    quoted_percent: float | None = None
    simple_annual_rate_decimal: float | None = None
    status: Literal["ok", "missing", "provider_error"]
    warning: str | None = None


class RiskFreeCurveData(StrictModel):
    requested_as_of: date
    curve_id: str
    curve_version: str
    source: Literal["fred"] = "fred"
    anchors: list[RiskFreeAnchor]


class RiskFreeCurveResponse(StrictModel):
    data: RiskFreeCurveData
    meta: LiveResponseMeta


class RiskFreeCurveHistoryRequest(StrictModel):
    as_of_dates: list[date] = Field(min_length=1, max_length=400)

    @field_validator("as_of_dates")
    @classmethod
    def normalize_dates(cls, value: list[date]) -> list[date]:
        if len(set(value)) != len(value):
            raise ValueError("as_of_dates must not contain duplicates")
        # Sort curve dates so historical observation selection is deterministic.
        return sorted(value)


class RiskFreeCurveHistoryResponse(StrictModel):
    data: list[RiskFreeCurveData]
    meta: LiveResponseMeta


class FuturesSpecificationRequest(StrictModel):
    contracts: list[str] = Field(min_length=1)

    @field_validator("contracts")
    @classmethod
    def normalize_contracts(cls, value: list[str]) -> list[str]:
        normalized = [
            str(item).strip().upper().removeprefix("/") for item in value
        ]
        if any(not item for item in normalized):
            raise ValueError("contracts must contain non-empty strings")
        return normalized


class FuturesSpecificationItem(StrictModel):
    requested_contract: str
    canonical_contract: str
    status: Literal["ok", "not_found", "ambiguous", "partial", "provider_error"]
    provider_instrument_id: str | None = None
    provider_product_id: str | None = None
    root: str | None = None
    product_code: str | None = None
    product_name: str | None = None
    exchange: str | None = None
    mic: str | None = None
    asset_class: str | None = None
    currency: str | None = None
    contract_unit: float | None = None
    price_multiplier: float | None = None
    quote_convention: str | None = None
    minimum_tick: float | None = None
    tick_value: float | None = None
    tick_value_derivation: str | None = None
    trading_timezone: str | None = None
    trading_schedule: str | None = None
    listing_date: date | None = None
    first_trade_date: date | None = None
    last_trade_at: datetime | None = None
    first_notice_date: date | None = None
    delivery_type: str | None = None
    settlement_type: str | None = None
    delivery_start_date: date | None = None
    delivery_end_date: date | None = None
    settlement_date: date | None = None
    source: Literal["cme_reference"] = "cme_reference"
    provider_as_of: datetime | None = None
    retrieved_at: datetime
    warnings: list[str] = Field(default_factory=list)


class FuturesSpecificationResponse(StrictModel):
    data: list[FuturesSpecificationItem]
    meta: LiveResponseMeta
