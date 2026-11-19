"""Strict public contracts for P&L lifecycle queries and replay analysis."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from algotrading.refdata.api.option_symbols import parse_occ_symbol
from algotrading.risk.transactions.models import TransactionCoverage


class PnlContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone aware")
    return value.astimezone(timezone.utc)


def _nonblank_unique(values: list[str], label: str, maximum: int) -> list[str]:
    if len(values) > maximum:
        raise ValueError(f"{label} may contain at most {maximum} values")
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value:
            raise ValueError(f"{label} must contain nonblank values")
        key = value.casefold()
        if key not in seen:
            output.append(value)
            seen.add(key)
    return output


class PnlTradeQueryRequest(PnlContractModel):
    account_ids: list[str] = Field(min_length=1)
    portfolio_snapshot_id: str
    start_at: datetime
    end_at: datetime
    include_closing_after_end: bool
    lookahead_end_at: datetime | None
    sectors: list[str]
    instrument_types: list[str]
    statuses: list[Literal["closed", "closed_after_period", "partial", "open"]] = Field(
        default_factory=list
    )

    @field_validator("account_ids")
    @classmethod
    def valid_accounts(cls, value: list[str]) -> list[str]:
        return _nonblank_unique(value, "account_ids", 50)

    @field_validator("portfolio_snapshot_id")
    @classmethod
    def nonblank_snapshot(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("portfolio_snapshot_id must not be blank")
        return normalized

    @field_validator("start_at", "end_at", "lookahead_end_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return _aware_utc(value) if value is not None else None

    @field_validator("sectors", "instrument_types", "statuses")
    @classmethod
    def normalize_filters(cls, value: list[str], info: Any) -> list[str]:
        return _nonblank_unique(value, info.field_name, 100)

    @model_validator(mode="after")
    def valid_interval(self) -> "PnlTradeQueryRequest":
        if self.start_at >= self.end_at:
            raise ValueError("start_at must be before end_at")
        # Lookahead history is permitted only when the caller explicitly requests it.
        if not self.include_closing_after_end and self.lookahead_end_at is not None:
            raise ValueError(
                "lookahead_end_at must be null when include_closing_after_end is false"
            )
        if self.include_closing_after_end:
            if self.lookahead_end_at is None:
                raise ValueError(
                    "lookahead_end_at is required when include_closing_after_end is true"
                )
            if self.lookahead_end_at < self.end_at:
                raise ValueError("lookahead_end_at must be on or after end_at")
        return self


class PnlDataQuality(PnlContractModel):
    status: Literal["usable", "usable_with_warnings", "insufficient_data"]
    transaction_coverage: list[TransactionCoverage]
    mark_coverage_percent: float | None
    warnings: list[str]


class PnlSummary(PnlContractModel):
    period_pnl_usd: float
    realized_pnl_usd: float
    unrealized_pnl_usd: float
    fees_and_carry_usd: float
    fee_components_usd: dict[str, float] = Field(default_factory=dict)
    gross_profit_usd: float
    gross_loss_usd: float
    closed_trade_count: int = Field(ge=0)
    open_trade_count: int = Field(ge=0)


class PnlDailyPoint(PnlContractModel):
    at: datetime
    daily_pnl_usd: float
    cumulative_pnl_usd: float

    _normalize_at = field_validator("at")(_aware_utc)


class PnlAttributionRow(PnlContractModel):
    name: str
    pnl_usd: float
    realized_pnl_usd: float
    unrealized_pnl_usd: float
    trade_count: int = Field(ge=0)


class PnlTransaction(PnlContractModel):
    transaction_id: str
    broker_transaction_id: str | None
    account_id: str
    timestamp: datetime
    role: str
    instrument: str
    symbol: str | None
    asset_type: str | None
    action: str
    quantity: float | None
    fill_price: float | None
    fees_usd: float
    fee_components_usd: dict[str, float] = Field(default_factory=dict)
    cash_effect_usd: float | None
    period_pnl_usd: float | None
    leg_status: str
    position_effect: str | None
    is_unrealized: bool

    _normalize_timestamp = field_validator("timestamp")(_aware_utc)


class PnlTrade(PnlContractModel):
    trade_id: str
    name: str
    description: str
    underlying_symbol: str
    sector: str
    instrument_type: str
    opened_at: datetime
    closed_at: datetime | None
    status: Literal["closed", "closed_after_period", "partial", "open"]
    leg_count: int = Field(ge=1)
    period_pnl_usd: float
    lifecycle_pnl_usd: float | None
    realized_pnl_usd: float
    unrealized_pnl_usd: float
    fees_and_carry_usd: float
    fee_components_usd: dict[str, float] = Field(default_factory=dict)
    capital_at_risk_usd: float | None
    return_percent: float | None
    match_confidence_percent: float = Field(ge=0, le=100)
    primary_driver: str | None
    transaction_ids: list[str]
    transactions: list[PnlTransaction]

    @field_validator("opened_at", "closed_at")
    @classmethod
    def aware_trade_timestamps(cls, value: datetime | None) -> datetime | None:
        return _aware_utc(value) if value is not None else None


class PnlAvailableFilters(PnlContractModel):
    sectors: list[str]
    instrument_types: list[str]


class PnlExcludedTrade(PnlContractModel):
    trade_id: str
    name: str
    underlying_symbol: str
    sector: str
    instrument_type: str
    missing_symbols: list[str]
    transaction_ids: list[str]
    reason: str


class PnlTradeQueryData(PnlContractModel):
    summary: PnlSummary
    daily_pnl: list[PnlDailyPoint]
    sector_attribution: list[PnlAttributionRow]
    instrument_attribution: list[PnlAttributionRow]
    available_filters: PnlAvailableFilters
    trades: list[PnlTrade]
    excluded_trades: list[PnlExcludedTrade]


class PnlTradeQueryResponse(PnlContractModel):
    schema_version: Literal[1] = 1
    portfolio_snapshot_id: str
    configuration_version: str
    as_of: datetime
    methodology_version: str
    data_quality: PnlDataQuality
    data: PnlTradeQueryData

    _normalize_as_of = field_validator("as_of")(_aware_utc)


class PnlMarketHistoryRequest(PnlContractModel):
    symbols: list[str] = Field(min_length=1, max_length=8)
    start_at: datetime
    end_at: datetime

    @field_validator("symbols")
    @classmethod
    def valid_symbols(cls, value: list[str]) -> list[str]:
        normalized = _nonblank_unique(value, "symbols", 8)
        for symbol in normalized:
            if re.fullmatch(r"/?[A-Z0-9][A-Z0-9.\-]{0,15}", symbol.upper()) is None:
                raise ValueError(f"invalid comparison symbol: {symbol!r}")
        return [symbol.upper() for symbol in normalized]

    @field_validator("start_at", "end_at")
    @classmethod
    def normalize_history_timestamp(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @model_validator(mode="after")
    def ordered_history_interval(self) -> "PnlMarketHistoryRequest":
        if self.end_at <= self.start_at:
            raise ValueError("end_at must be after start_at")
        if self.end_at - self.start_at > timedelta(days=367):
            raise ValueError("market-history interval may not exceed 367 days")
        return self


class PnlMarketHistoryPoint(PnlContractModel):
    at: datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    volume: float | None = None

    _normalize_at = field_validator("at")(_aware_utc)


class PnlMarketHistorySeries(PnlContractModel):
    symbol: str
    points: list[PnlMarketHistoryPoint]


class PnlMarketHistoryResponse(PnlContractModel):
    schema_version: Literal[1] = 1
    data: list[PnlMarketHistorySeries]
    warnings: list[str] = Field(default_factory=list)


class PnlModification(PnlContractModel):
    action: Literal["add", "remove"]
    instrument_kind: Literal["underlying", "etf", "future", "option"]
    symbol: str
    quantity: float
    at: datetime | None = None
    fill_price: float | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    vega_per_1pct: float | None = None
    theta_per_day: float | None = None

    @field_validator("symbol")
    @classmethod
    def valid_symbol(cls, value: str, info: Any) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("symbol must not be blank")
        return symbol

    @field_validator("quantity")
    @classmethod
    def finite_nonzero_quantity(cls, value: float) -> float:
        if not math.isfinite(value) or value == 0:
            raise ValueError("quantity must be finite and nonzero")
        return value

    @field_validator(
        "fill_price",
        "implied_volatility",
        "delta",
        "gamma",
        "vega_per_1pct",
        "theta_per_day",
    )
    @classmethod
    def finite_optional_metric(cls, value: float | None, info: Any) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite")
        if info.field_name in {"fill_price", "implied_volatility"} and value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @field_validator("at")
    @classmethod
    def normalize_at(cls, value: datetime | None) -> datetime | None:
        return _aware_utc(value) if value is not None else None

    @model_validator(mode="after")
    def canonical_symbol(self) -> "PnlModification":
        if self.instrument_kind == "option":
            try:
                canonical = parse_occ_symbol(self.symbol).canonical
            except ValueError as exc:
                raise ValueError("option modification requires an exact OCC symbol") from exc
            # Canonicalize the immutable modification before matching exact-contract history.
            object.__setattr__(self, "symbol", canonical)
        elif re.fullmatch(r"/?[A-Z0-9][A-Z0-9.\-]{0,15}", self.symbol) is None:
            raise ValueError("equity and future symbols must be exact security symbols")
        return self


class PnlExitRule(PnlContractModel):
    kind: Literal[
        "actual",
        "timestamp",
        "profit_target",
        "drawdown_from_peak",
        "days_before_short_expiry",
    ]
    value: str | float | None

    @model_validator(mode="after")
    def compatible_value(self) -> "PnlExitRule":
        if self.kind == "actual":
            if self.value is not None:
                raise ValueError("actual exit rule requires a null value")
            return self
        if self.value is None:
            raise ValueError(f"{self.kind} exit rule requires a value")
        # Exit-rule values follow the selected rule's timestamp or numeric contract.
        if self.kind == "timestamp":
            if not isinstance(self.value, str):
                raise ValueError("timestamp exit-rule value must be an RFC 3339 string")
            try:
                parsed = datetime.fromisoformat(self.value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("timestamp exit-rule value must be RFC 3339") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("timestamp exit-rule value must be timezone aware")
            return self
        if isinstance(self.value, bool):
            raise ValueError(f"{self.kind} exit-rule value must be numeric")
        try:
            number = float(self.value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.kind} exit-rule value must be numeric") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{self.kind} exit-rule value must be finite and positive")
        object.__setattr__(self, "value", number)
        return self


class PnlAnalyzerRequest(PnlContractModel):
    account_ids: list[str] = Field(min_length=1)
    portfolio_snapshot_id: str
    trade_ids: list[str]
    transaction_ids: list[str]
    decision_at: datetime | None
    fill_model: Literal[
        "historical_mid", "historical_touch_plus_tick", "conservative_touch"
    ]
    modifications: list[PnlModification]
    exit_rule: PnlExitRule

    @field_validator("account_ids")
    @classmethod
    def valid_accounts(cls, value: list[str]) -> list[str]:
        return _nonblank_unique(value, "account_ids", 50)

    @field_validator("portfolio_snapshot_id")
    @classmethod
    def valid_snapshot(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("portfolio_snapshot_id must not be blank")
        return normalized

    @field_validator("trade_ids", "transaction_ids")
    @classmethod
    def valid_selection_ids(cls, value: list[str], info: Any) -> list[str]:
        return _nonblank_unique(value, info.field_name, 500)

    @field_validator("decision_at")
    @classmethod
    def normalize_decision_at(cls, value: datetime | None) -> datetime | None:
        return _aware_utc(value) if value is not None else None

    @model_validator(mode="after")
    def valid_selection(self) -> "PnlAnalyzerRequest":
        if not self.trade_ids and not self.transaction_ids:
            raise ValueError("at least one trade_id or transaction_id is required")
        # Every modification needs a decision time for causal fill selection.
        if self.modifications and self.decision_at is None and any(
            item.at is None for item in self.modifications
        ):
            raise ValueError("each modification requires at when decision_at is omitted")
        return self


class PnlReplayPoint(PnlContractModel):
    at: datetime
    pnl_usd: float
    underlying_price: float | None
    delta: float | None
    gamma: float | None
    vega_per_1pct_usd: float | None
    theta_per_day_usd: float | None
    event: str | None

    _normalize_at = field_validator("at")(_aware_utc)


class PnlChronologyEvent(PnlContractModel):
    at: datetime
    title: str
    detail: str
    kind: Literal["transaction", "market", "risk"]

    _normalize_at = field_validator("at")(_aware_utc)


class PnlFactorAttribution(PnlContractModel):
    factor: str
    pnl_usd: float


class PnlActiveLeg(PnlContractModel):
    leg_id: str
    label: str
    symbol: str
    quantity: float


class PnlScenarioOutcome(PnlContractModel):
    scenario_id: str
    name: str
    net_pnl_usd: float
    max_drawdown_usd: float
    capital_at_risk_usd: float | None
    interpretation: str
    path: list[PnlReplayPoint]


class PnlAnalyzerDataQuality(PnlContractModel):
    status: Literal["usable", "usable_with_warnings", "insufficient_data"]
    historical_option_marks_available: bool
    attribution_method: Literal[
        "surface_revaluation",
        "price_history_fallback",
        "mark_revaluation",
    ] = "mark_revaluation"
    surface_coverage_percent: float | None = Field(default=None, ge=0, le=100)
    option_price_history_coverage_percent: float | None = Field(
        default=None, ge=0, le=100
    )
    calculation_method: str | None = None
    input_mode: str | None = None
    warnings: list[str]


class PnlAnalyzerData(PnlContractModel):
    selection_name: str
    match_confidence_percent: float = Field(ge=0, le=100)
    lifecycle_pnl_usd: float
    peak_open_pnl_usd: float | None
    maximum_drawdown_usd: float | None
    fees_and_slippage_usd: float
    actual_path: list[PnlReplayPoint]
    chronology: list[PnlChronologyEvent]
    attribution: list[PnlFactorAttribution]
    active_legs: list[PnlActiveLeg]
    scenarios: list[PnlScenarioOutcome]


class PnlAnalyzerResponse(PnlContractModel):
    schema_version: Literal[1] = 1
    portfolio_snapshot_id: str
    as_of: datetime
    methodology_version: str
    data_quality: PnlAnalyzerDataQuality
    data: PnlAnalyzerData

    _normalize_as_of = field_validator("as_of")(_aware_utc)


__all__ = [name for name in globals() if name.startswith("Pnl")]
