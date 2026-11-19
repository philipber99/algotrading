"""Parquet schemas, manifest helpers, and REST request models."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
from pydantic import BaseModel, Field


UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")

SESSION_INDEX_SCHEMA = pa.schema(
    [
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("session_date", pa.date32(), nullable=False),
        pa.field("session_open_utc", UTC_TIMESTAMP, nullable=False),
        pa.field("session_close_utc", UTC_TIMESTAMP, nullable=False),
        pa.field("regular_session_minutes", pa.uint16(), nullable=False),
        pa.field("is_early_close", pa.bool_(), nullable=False),
        pa.field("day_of_week", pa.uint8(), nullable=False),
        pa.field("first_sample_minute", pa.uint16(), nullable=False),
        pa.field("last_sample_minute", pa.uint16(), nullable=False),
        pa.field("expected_bin_count", pa.uint16(), nullable=False),
        pa.field("available_bin_count", pa.uint16(), nullable=False),
        pa.field("coverage_fraction", pa.float32(), nullable=False),
        pa.field("opening_spot", pa.float64()),
        pa.field("opening_atm_iv", pa.float32()),
        pa.field("quality_flags", pa.uint32(), nullable=False),
        pa.field("rejected_snapshot_count", pa.uint32(), nullable=False),
        pa.field("stale_sample_count", pa.uint32(), nullable=False),
    ]
)

SURFACE_OBSERVATION_SCHEMA = pa.schema(
    [
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("session_date", pa.date32(), nullable=False),
        pa.field("session_open_utc", UTC_TIMESTAMP, nullable=False),
        pa.field("session_close_utc", UTC_TIMESTAMP, nullable=False),
        pa.field("session_minute", pa.uint16(), nullable=False),
        pa.field("sample_target_utc", UTC_TIMESTAMP, nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", UTC_TIMESTAMP, nullable=False),
        pa.field("sample_age_seconds", pa.float32(), nullable=False),
        pa.field("expiration_at", UTC_TIMESTAMP, nullable=False),
        pa.field("price_basis", pa.string(), nullable=False),
        pa.field("surface_definition_id", pa.uint32(), nullable=False),
        pa.field("calculation_definition_id", pa.uint32(), nullable=False),
        pa.field("pricing_method", pa.string(), nullable=False),
        pa.field("fit_method", pa.string(), nullable=False),
        pa.field("svi_a", pa.float64()),
        pa.field("svi_b", pa.float64()),
        pa.field("svi_rho", pa.float64()),
        pa.field("svi_m", pa.float64()),
        pa.field("svi_sigma", pa.float64()),
        pa.field("pchip_k", pa.list_(pa.float64()), nullable=False),
        pa.field("pchip_total_variance", pa.list_(pa.float64()), nullable=False),
        pa.field("spot_price", pa.float64(), nullable=False),
        pa.field("forward_price", pa.float64(), nullable=False),
        pa.field("time_to_expiry_seconds", pa.uint32(), nullable=False),
        pa.field("risk_free_rate", pa.float64(), nullable=False),
        pa.field("dividend_yield", pa.float64(), nullable=False),
        pa.field("minimum_log_forward_moneyness", pa.float64(), nullable=False),
        pa.field("maximum_log_forward_moneyness", pa.float64(), nullable=False),
        pa.field("input_contract_count", pa.uint32(), nullable=False),
        pa.field("accepted_contract_count", pa.uint32(), nullable=False),
        pa.field("rejected_contract_count", pa.uint32(), nullable=False),
        pa.field("weighted_rmse", pa.float64(), nullable=False),
        pa.field("maximum_absolute_error", pa.float64(), nullable=False),
        pa.field("butterfly_arbitrage_flags", pa.uint32(), nullable=False),
        pa.field("calendar_arbitrage_flags", pa.uint32(), nullable=False),
        pa.field("quality_flags", pa.uint32(), nullable=False),
    ]
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_fingerprint(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


class StrategyLegRequest(BaseModel):
    side: Literal["long", "short"]
    put_call: Literal["CALL", "PUT"]
    quantity: float = Field(gt=0)
    strike: float = Field(gt=0)
    expiration_at: datetime
    anchor_forward: float = Field(gt=0)
    strike_group: str | None = None
    shared_strike_group: str | None = None


class CohortRequest(BaseModel):
    historical_window: str | None = None
    dte_tolerance_days: float = Field(default=2.0, ge=0, le=31)
    minimum_session_coverage: float = Field(default=0.90, gt=0, le=1)
    minimum_sample_count: int = Field(default=5, ge=1)
    quality_flags_excluded: list[int | str] = Field(
        default_factory=list,
        max_length=64,
    )
    days_of_week: list[int] | None = Field(default=None, max_length=7)


class AggregationRequest(BaseModel):
    interval_minutes: int = Field(default=5, ge=1, le=60)
    quantiles: list[float] = Field(
        default_factory=lambda: [0.10, 0.90],
        min_length=2,
        max_length=2,
    )
    normalization: Literal[
        "none", "spread_width", "capital_at_risk", "underlying"
    ] = "none"
    normalization_value: float | None = Field(default=None, gt=0)
    center_statistic: Literal["mean", "median"] = "median"


class ProfileRequest(BaseModel):
    underlying_symbol: str
    as_of_utc: datetime
    mode: Literal["comparable_entry", "enter_and_hold"] = "comparable_entry"
    value_component: Literal["total_value", "iv_only"] = "total_value"
    price_basis: Literal["mid", "bid", "ask"] = "mid"
    legs: list[StrategyLegRequest] = Field(min_length=1, max_length=12)
    cohort: CohortRequest = Field(default_factory=CohortRequest)
    aggregation: AggregationRequest = Field(default_factory=AggregationRequest)
    entry_session_minute: int | None = Field(default=None, ge=0, le=1440)
    path_end_session_minute: int | None = Field(default=None, ge=0, le=1440)
    output_metric: Literal[
        "strategy_value", "pnl_from_entry", "return_from_entry"
    ] = "strategy_value"
    current_strategy_value: float | None = None


class NormalizeRequest(BaseModel):
    underlying_symbol: str
    as_of_utc: datetime
    legs: list[StrategyLegRequest] = Field(min_length=1, max_length=12)
