"""Deterministic split of a flattened vendor chain into canonical source facts."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import pyarrow as pa

from algotrading.refdata.options.Common import (
    _parse_timestamp,
    option_contract_id,
    sha256_bytes,
)


UINT32_MAX = 2**32 - 1
UINT64_MAX = 2**64 - 1
FLOAT32_MAX = 3.4028235e38

IMMUTABLE_CONTRACT_FIELDS = (
    "option_symbol",
    "underlying_symbol",
    "put_call",
    "strike_price",
    "expiration_at",
    "settlement_type",
    "exercise_type",
    "multiplier",
    "is_mini",
    "is_nonstandard",
    "option_deliverables_json",
)

DESCRIPTIVE_CONTRACT_FIELDS = (
    "option_root",
    "ssid",
    "description",
    "exchange_name",
    "expiration_type",
    "last_trading_day",
    "is_penny_pilot",
    "deliverable_note",
)


def _canonical(value: Any) -> Any:
    if isinstance(value, datetime):
        parsed = _parse_timestamp(value)
        return parsed.isoformat(timespec="microseconds") if parsed else None
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Contract identity cannot contain nonfinite numbers")
        return value
    return value


def _hash_fields(row: Mapping[str, Any], fields: Sequence[str]) -> bytes:
    payload = json.dumps(
        {name: _canonical(row.get(name)) for name in fields},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def contract_identity_hash(row: Mapping[str, Any]) -> bytes:
    """Hash a prepared canonical contract's immutable economic identity."""
    return _hash_fields(row, IMMUTABLE_CONTRACT_FIELDS)


def contract_metadata_hash(row: Mapping[str, Any]) -> bytes:
    """Hash a prepared canonical contract's refreshable vendor metadata."""
    return _hash_fields(row, DESCRIPTIVE_CONTRACT_FIELDS)


def _uint32(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    result = int(value)
    if result < 0 or result > UINT32_MAX:
        raise OverflowError(f"{name}={result} does not fit UInt32")
    return result


def _float32(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite when present")
    if abs(result) > FLOAT32_MAX:
        raise OverflowError(f"{name}={result} does not fit Float32")
    return result


def _uint64(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    result = int(value)
    if result < 0 or result > UINT64_MAX:
        raise OverflowError(f"{name}={result} does not fit UInt64")
    return result


def _float64(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite when present")
    return result


SNAPSHOT_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("request_started_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("response_received_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("response_latency_ms", pa.uint32(), nullable=False),
        pa.field("payload_sha256", pa.binary(32), nullable=False),
        pa.field("chain_status", pa.string(), nullable=False),
        pa.field("strategy", pa.string(), nullable=False),
        pa.field("chain_interval", pa.float32()),
        pa.field("chain_is_delayed", pa.bool_(), nullable=False),
        pa.field("chain_is_index", pa.bool_(), nullable=False),
        pa.field("chain_is_truncated", pa.bool_(), nullable=False),
        pa.field("underlying_is_delayed", pa.bool_(), nullable=False),
        pa.field("source_contract_count", pa.uint32(), nullable=False),
        pa.field("stored_observation_count", pa.uint32(), nullable=False),
        pa.field("stored_detail_count", pa.uint32(), nullable=False),
        pa.field("stored_distinct_contract_count", pa.uint32(), nullable=False),
        pa.field("asset_main_type", pa.string(), nullable=False),
        pa.field("asset_sub_type", pa.string(), nullable=False),
        pa.field("chain_interest_rate", pa.float64()),
        pa.field("chain_dividend_yield", pa.float64()),
        pa.field("chain_volatility", pa.float64()),
        pa.field("chain_underlying_price", pa.float64()),
        pa.field("underlying_bid", pa.float64()),
        pa.field("underlying_ask", pa.float64()),
        pa.field("underlying_last", pa.float64()),
        pa.field("underlying_mark", pa.float64()),
        pa.field("underlying_bid_size", pa.uint64()),
        pa.field("underlying_ask_size", pa.uint64()),
        pa.field("underlying_quote_time", pa.timestamp("ms", tz="UTC")),
        pa.field("underlying_trade_time", pa.timestamp("ms", tz="UTC")),
        pa.field("underlying_total_volume", pa.uint64()),
        pa.field("collection_definition_id", pa.uint32(), nullable=False),
        pa.field("normalization_definition_id", pa.uint32(), nullable=False),
        pa.field("normalized_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

CONTRACT_SCHEMA = pa.schema(
    [
        pa.field("contract_id", pa.uint64(), nullable=False),
        pa.field("option_symbol", pa.string(), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("option_root", pa.string(), nullable=False),
        pa.field("ssid", pa.uint64()),
        pa.field("description", pa.string(), nullable=False),
        pa.field("exchange_name", pa.string(), nullable=False),
        pa.field("put_call", pa.string(), nullable=False),
        pa.field("strike_price", pa.float64(), nullable=False),
        pa.field("expiration_at", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("expiration_type", pa.string(), nullable=False),
        pa.field("last_trading_day", pa.timestamp("ms", tz="UTC")),
        pa.field("settlement_type", pa.string(), nullable=False),
        pa.field("exercise_type", pa.string(), nullable=False),
        pa.field("multiplier", pa.float32(), nullable=False),
        pa.field("is_mini", pa.bool_(), nullable=False),
        pa.field("is_nonstandard", pa.bool_(), nullable=False),
        pa.field("is_penny_pilot", pa.bool_(), nullable=False),
        pa.field("deliverable_note", pa.string(), nullable=False),
        pa.field("option_deliverables_json", pa.string(), nullable=False),
        pa.field("identity_hash", pa.binary(32), nullable=False),
        pa.field("metadata_hash", pa.binary(32), nullable=False),
        pa.field("first_seen_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("last_seen_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("metadata_updated_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

OBSERVATION_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("contract_id", pa.uint64(), nullable=False),
        pa.field("contract_ordinal", pa.uint32(), nullable=False),
        pa.field("symbol_map_mismatch", pa.bool_(), nullable=False),
        *(pa.field(name, pa.float32()) for name in ("bid", "ask", "last", "mark")),
        *(pa.field(name, pa.uint32()) for name in ("bid_size", "ask_size", "last_size")),
        *(pa.field(name, pa.float32()) for name in ("open_price", "high_price", "low_price", "close_price")),
        pa.field("total_volume", pa.uint32()),
        pa.field("open_interest", pa.uint32()),
        pa.field("quote_time", pa.timestamp("ms", tz="UTC")),
        pa.field("trade_time", pa.timestamp("ms", tz="UTC")),
        *(pa.field(name, pa.float32()) for name in ("source_iv", "source_delta", "source_gamma", "source_theta", "source_vega", "source_rho")),
    ]
)

DETAIL_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("contract_id", pa.uint64(), nullable=False),
        *(pa.field(name, pa.float32()) for name in (
            "high_52_week", "low_52_week", "net_change", "percent_change",
            "mark_change", "mark_percent_change", "intrinsic_value",
            "extrinsic_value", "time_value", "break_even",
            "theoretical_option_value", "theoretical_volatility",
        )),
    ]
)


@dataclass(frozen=True)
class NormalizedSnapshot:
    snapshot: Mapping[str, Any]
    contracts: tuple[Mapping[str, Any], ...]
    observations: tuple[Mapping[str, Any], ...]
    details: tuple[Mapping[str, Any], ...]

    def validate(self) -> None:
        expected = int(self.snapshot["stored_observation_count"])
        if expected <= 0:
            raise ValueError("A normalized snapshot must contain observations")
        natural_keys = {
            (str(row["snapshot_id"]), int(row["contract_id"]))
            for row in self.observations
        }
        if len(self.observations) != expected or len(natural_keys) != expected:
            raise ValueError("Observation count or natural-key uniqueness mismatch")
        if len(self.details) != int(self.snapshot["stored_detail_count"]):
            raise ValueError("Observation-detail count does not match manifest")
        if int(self.snapshot["stored_distinct_contract_count"]) != expected:
            raise ValueError("Stored distinct-contract count does not match observations")
        detail_keys = {
            (str(row["snapshot_id"]), int(row["contract_id"]))
            for row in self.details
        }
        if len(detail_keys) != len(self.details) or detail_keys != natural_keys:
            raise ValueError("Observation-detail natural keys do not match observations")
        if len({int(row["contract_id"]) for row in self.contracts}) != expected:
            raise ValueError("Contract identity count does not match observations")
        contract_ids = {int(row["contract_id"]) for row in self.contracts}
        if contract_ids != {int(row["contract_id"]) for row in self.observations}:
            raise ValueError("Observation references an absent contract")


def build_normalized_snapshot(
    raw_record: Mapping[str, Any],
    flat_rows: Sequence[Mapping[str, Any]],
    *,
    collection_definition_id: int,
    normalization_definition_id: int,
    normalized_at: datetime,
) -> NormalizedSnapshot:
    if not flat_rows:
        raise ValueError("A normalized snapshot must contain at least one observation")
    first = flat_rows[0]
    source_at = _parse_timestamp(raw_record["collected_at_utc"])
    if source_at is None:
        raise ValueError("Raw collected timestamp is invalid")
    snapshot_id = str(raw_record["snapshot_id"])
    uuid.UUID(snapshot_id)
    if len(bytes(raw_record["payload_sha256"])) != 32:
        raise ValueError("Raw payload digest must contain exactly 32 bytes")
    if int(collection_definition_id) <= 0 or int(normalization_definition_id) <= 0:
        raise ValueError("Normalized snapshots require nonzero definition IDs")
    underlying = str(first["underlying_symbol"]).upper()
    snapshot_fields = (
        "chain_status", "strategy", "chain_interval", "chain_is_delayed",
        "chain_is_index", "chain_is_truncated", "underlying_is_delayed",
        "asset_main_type", "asset_sub_type", "chain_interest_rate",
        "chain_dividend_yield", "chain_volatility_raw",
        "chain_underlying_price", "underlying_bid", "underlying_ask",
        "underlying_last", "underlying_mark", "underlying_bid_size",
        "underlying_ask_size", "underlying_quote_time", "underlying_trade_time",
        "underlying_total_volume", "ingested_at",
    )
    contracts: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    seen_ids: dict[int, str] = {}

    for flat in flat_rows:
        if str(flat.get("underlying_symbol") or "").upper() != underlying:
            raise ValueError("Normalized snapshot mixes underlying symbols")
        if str(flat.get("snapshot_id") or snapshot_id) != snapshot_id:
            raise ValueError("Normalized snapshot mixes snapshot IDs")
        if any(_canonical(flat.get(name)) != _canonical(first.get(name)) for name in snapshot_fields):
            raise ValueError("Normalized snapshot mixes snapshot-level metadata")
        option_symbol = str(flat["option_symbol"])
        contract_id = option_contract_id(underlying, option_symbol)
        previous_symbol = seen_ids.setdefault(contract_id, option_symbol)
        if previous_symbol != option_symbol:
            raise RuntimeError(
                f"Contract-ID collision: {previous_symbol!r} and {option_symbol!r}"
            )
        contract = {
            "contract_id": contract_id,
            "option_symbol": option_symbol,
            "underlying_symbol": underlying,
            "option_root": str(flat.get("option_root") or ""),
            "ssid": flat.get("ssid"),
            "description": str(flat.get("description") or ""),
            "exchange_name": str(flat.get("exchange_name") or ""),
            "put_call": str(flat["put_call"]),
            "strike_price": _float64(flat["strike_price"], "strike_price"),
            "expiration_at": _parse_timestamp(flat["expiration_at"]),
            "expiration_type": str(flat.get("expiration_type") or ""),
            "last_trading_day": _parse_timestamp(flat.get("last_trading_day")),
            "settlement_type": str(flat.get("settlement_type") or ""),
            "exercise_type": str(flat.get("exercise_type") or ""),
            "multiplier": _float32(flat.get("multiplier") or 0.0, "multiplier"),
            "is_mini": bool(flat.get("is_mini")),
            "is_nonstandard": bool(flat.get("is_nonstandard")),
            "is_penny_pilot": bool(flat.get("is_penny_pilot")),
            "deliverable_note": str(flat.get("deliverable_note") or ""),
            "option_deliverables_json": str(flat.get("option_deliverables_json") or "[]"),
            "first_seen_at": source_at,
            "last_seen_at": source_at,
            "metadata_updated_at": source_at,
        }
        if contract["put_call"] not in {"CALL", "PUT"}:
            raise ValueError(f"Invalid contract side: {contract['put_call']!r}")
        if contract["expiration_at"] is None:
            raise ValueError("Contract expiration_at is required")
        if contract["multiplier"] is None or contract["multiplier"] <= 0:
            raise ValueError("Contract multiplier must be positive")
        contract["identity_hash"] = contract_identity_hash(contract)
        contract["metadata_hash"] = contract_metadata_hash(contract)
        contracts.append(contract)

        observation = {
            "snapshot_id": snapshot_id,
            "source_collected_at": source_at,
            "underlying_symbol": underlying,
            "contract_id": contract_id,
            "contract_ordinal": _uint32(flat["contract_ordinal"], "contract_ordinal"),
            "symbol_map_mismatch": bool(flat.get("symbol_map_mismatch")),
        }
        for name in (
            "bid", "ask", "last", "mark", "open_price", "high_price",
            "low_price", "close_price", "source_iv", "source_delta",
            "source_gamma", "source_theta", "source_vega", "source_rho",
        ):
            observation[name] = _float32(flat.get(name), name)
        for name in ("bid_size", "ask_size", "last_size", "total_volume", "open_interest"):
            observation[name] = _uint32(flat.get(name), name)
        observation["quote_time"] = _parse_timestamp(flat.get("quote_time"))
        observation["trade_time"] = _parse_timestamp(flat.get("trade_time"))
        observations.append(observation)

        detail = {
            "snapshot_id": snapshot_id,
            "source_collected_at": source_at,
            "underlying_symbol": underlying,
            "contract_id": contract_id,
            "theoretical_volatility": _float32(
                flat.get("theoretical_volatility_raw"),
                "theoretical_volatility",
            ),
        }
        for name in (
            "high_52_week", "low_52_week", "net_change", "percent_change",
            "mark_change", "mark_percent_change", "intrinsic_value",
            "extrinsic_value", "time_value", "break_even",
            "theoretical_option_value",
        ):
            detail[name] = _float32(flat.get(name), name)
        details.append(detail)

    count = len(observations)
    snapshot = {
        "snapshot_id": snapshot_id,
        "source": str(raw_record["source"]),
        "underlying_symbol": underlying,
        "source_collected_at": source_at,
        "request_started_at": _parse_timestamp(raw_record["request_started_at_utc"]),
        "response_received_at": _parse_timestamp(raw_record["response_received_at_utc"]),
        "ingested_at": _parse_timestamp(first["ingested_at"]),
        "response_latency_ms": _uint32(raw_record["response_latency_ms"], "response_latency_ms"),
        "payload_sha256": bytes(raw_record["payload_sha256"]),
        "chain_status": str(first.get("chain_status") or ""),
        "strategy": str(first.get("strategy") or ""),
        "chain_interval": _float32(first.get("chain_interval"), "chain_interval"),
        "chain_is_delayed": bool(first.get("chain_is_delayed")),
        "chain_is_index": bool(first.get("chain_is_index")),
        "chain_is_truncated": bool(first.get("chain_is_truncated")),
        "underlying_is_delayed": bool(first.get("underlying_is_delayed")),
        "source_contract_count": _uint32(raw_record["source_contract_count"], "source_contract_count"),
        "stored_observation_count": count,
        "stored_detail_count": len(details),
        "stored_distinct_contract_count": len(seen_ids),
        "asset_main_type": str(first.get("asset_main_type") or ""),
        "asset_sub_type": str(first.get("asset_sub_type") or ""),
        "chain_interest_rate": _float64(first.get("chain_interest_rate"), "chain_interest_rate"),
        "chain_dividend_yield": _float64(first.get("chain_dividend_yield"), "chain_dividend_yield"),
        "chain_volatility": _float64(first.get("chain_volatility_raw"), "chain_volatility"),
        "chain_underlying_price": _float64(first.get("chain_underlying_price"), "chain_underlying_price"),
        "underlying_bid": _float64(first.get("underlying_bid"), "underlying_bid"),
        "underlying_ask": _float64(first.get("underlying_ask"), "underlying_ask"),
        "underlying_last": _float64(first.get("underlying_last"), "underlying_last"),
        "underlying_mark": _float64(first.get("underlying_mark"), "underlying_mark"),
        "underlying_bid_size": _uint64(first.get("underlying_bid_size"), "underlying_bid_size"),
        "underlying_ask_size": _uint64(first.get("underlying_ask_size"), "underlying_ask_size"),
        "underlying_quote_time": _parse_timestamp(first.get("underlying_quote_time")),
        "underlying_trade_time": _parse_timestamp(first.get("underlying_trade_time")),
        "underlying_total_volume": _uint64(first.get("underlying_total_volume"), "underlying_total_volume"),
        "collection_definition_id": int(collection_definition_id),
        "normalization_definition_id": int(normalization_definition_id),
        "normalized_at": _parse_timestamp(normalized_at),
    }
    for timestamp_name in (
        "request_started_at", "response_received_at", "ingested_at", "normalized_at"
    ):
        if snapshot[timestamp_name] is None:
            raise ValueError(f"Snapshot {timestamp_name} is required")
    result = NormalizedSnapshot(
        snapshot=snapshot,
        contracts=tuple(contracts),
        observations=tuple(observations),
        details=tuple(details),
    )
    result.validate()
    return result


def merge_contract_metadata(
    candidate: Mapping[str, Any], existing: Optional[Mapping[str, Any]]
) -> tuple[str, Mapping[str, Any]]:
    """Validate identity and classify a batched current-dimension update."""
    if existing is None:
        return "insert", candidate
    if int(existing["contract_id"]) != int(candidate["contract_id"]):
        raise RuntimeError("Contract lookup returned the wrong contract ID")
    if (
        str(existing["option_symbol"]) != str(candidate["option_symbol"])
        or str(existing["underlying_symbol"]) != str(candidate["underlying_symbol"])
    ):
        raise RuntimeError(f"Contract-ID collision for {candidate['contract_id']}")
    if bytes(existing["identity_hash"]) != bytes(candidate["identity_hash"]):
        raise RuntimeError(
            f"Immutable contract identity mismatch for {candidate['option_symbol']}"
        )
    merged = dict(candidate)
    merged["first_seen_at"] = existing["first_seen_at"]
    if bytes(existing["metadata_hash"]) == bytes(candidate["metadata_hash"]):
        # last_seen_at is part of the current dimension contract even when no
        # descriptive vendor field changed, so publish a narrow replacement.
        return "touch", merged
    return "refresh", merged


__all__ = [
    "CONTRACT_SCHEMA",
    "DESCRIPTIVE_CONTRACT_FIELDS",
    "DETAIL_SCHEMA",
    "IMMUTABLE_CONTRACT_FIELDS",
    "NormalizedSnapshot",
    "OBSERVATION_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "build_normalized_snapshot",
    "contract_identity_hash",
    "contract_metadata_hash",
    "merge_contract_metadata",
]
