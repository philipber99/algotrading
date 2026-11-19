"""Authoritative generation-3 ClickHouse schema and bounded reset support.

The registry contains exactly the 23 canonical operational tables.  It creates
no compatibility or result-ranking views, and reset plans cannot touch durable
normalized source or archive state.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from algotrading.refdata.options.Common import (
    create_clickhouse_client,
    validate_database_identifier,
)


@dataclass(frozen=True)
class ResetPlan:
    reset_level: str
    database: str
    objects: tuple[str, ...]
    drop_statements: tuple[str, ...]
    create_statements: tuple[str, ...]

    @property
    def statements(self) -> tuple[str, ...]:
        return self.drop_statements + self.create_statements


SCHEMA_GENERATION = 3


@dataclass(frozen=True)
class CanonicalTableSpec:
    name: str
    ddl: str
    owner: str
    engine_expression: str
    column_types: Mapping[str, str]
    partition_expression: Optional[str]
    order_expression: str
    durable: bool = True
    skipping_indexes: Mapping[str, str] = MappingProxyType({})

    @property
    def schema_hash(self) -> str:
        normalized = " ".join(self.ddl.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SchemaValidationError(RuntimeError):
    """The installed ClickHouse schema is not the canonical generation."""


def _arrow_type_compatible(arrow_type: Any, clickhouse_type: str) -> bool:
    """Compare the supported Arrow representations to a canonical CH type."""
    actual = str(arrow_type).replace(" ", "").lower()
    expected = str(clickhouse_type).replace(" ", "").lower()
    while expected.startswith(("nullable(", "lowcardinality(")) and expected.endswith(")"):
        expected = expected[expected.index("(") + 1 : -1]
    if expected.startswith("enum8(") or expected.startswith("enum16("):
        return actual in {"string", "large_string"}
    if expected == "uuid":
        return actual in {"string", "large_string"}
    if expected.startswith("fixedstring("):
        length = expected.removeprefix("fixedstring(").removesuffix(")")
        return actual in {f"fixed_size_binary[{length}]", f"binary[{length}]"}
    if expected.startswith("datetime64("):
        precision = expected.removeprefix("datetime64(").split(",", 1)[0]
        unit = {"3": "ms", "6": "us"}.get(precision)
        return unit is not None and actual.startswith(f"timestamp[{unit},tz=utc]")
    if expected == "date":
        return actual.startswith("date32")
    if expected.startswith("array(") and expected.endswith(")"):
        nested = expected[6:-1]
        nested_arrow = {
            "float64": "double",
            "float32": "float",
            "uint32": "uint32",
            "uint64": "uint64",
        }.get(nested)
        return actual in {
            f"list<item:{nested_arrow}>",
            f"list<item: {nested_arrow}>",
        }
    primitives = {
        "string": {"string", "large_string"},
        "bool": {"bool"},
        "float32": {"float"},
        "float64": {"double"},
        "uint16": {"uint16"},
        "uint32": {"uint32"},
        "uint64": {"uint64"},
    }
    return actual in primitives.get(expected, {expected})


def validate_arrow_schema_parity(
    table_name: str,
    arrow_schema: Any,
    *,
    nullable_omission_allowlist: Sequence[str] = (),
) -> None:
    """Fail closed when an Arrow insert contract drifts from canonical DDL.

    Runtime-native nullable lineage columns are intentionally present in the
    Arrow schemas.  The allowlist parameter exists only for explicitly
    documented future exceptions and defaults to no omissions.
    """
    registry = CANONICAL_SCHEMA
    if table_name not in registry:
        raise SchemaValidationError(f"Unknown canonical Arrow table: {table_name}")
    spec = registry[table_name]
    actual_fields = {field.name: field for field in arrow_schema}
    expected = dict(spec.column_types)
    unknown = sorted(set(actual_fields).difference(expected))
    missing = sorted(set(expected).difference(actual_fields))
    allowed = set(nullable_omission_allowlist)
    errors: list[str] = []
    if unknown:
        errors.append(f"unknown Arrow columns {unknown}")
    missing_without_allowlist = [name for name in missing if name not in allowed]
    if missing_without_allowlist:
        errors.append(f"missing canonical columns {missing_without_allowlist}")
    for name in sorted(set(expected).intersection(actual_fields)):
        field = actual_fields[name]
        if not _arrow_type_compatible(field.type, expected[name]):
            errors.append(
                f"{name}: Arrow type {field.type!s} is incompatible with "
                f"ClickHouse {expected[name]!r}"
            )
        canonical_nullable = "nullable(" in str(expected[name]).lower()
        if not canonical_nullable and field.nullable:
            errors.append(f"{name}: nonnullable canonical column is nullable in Arrow")
        if canonical_nullable and not field.nullable:
            errors.append(f"{name}: nullable canonical column is nonnullable in Arrow")
    if errors:
        raise SchemaValidationError(
            f"Arrow/canonical schema mismatch for {table_name}: "
            + "; ".join(errors)
        )


@dataclass(frozen=True)
class SchemaValidationReport:
    database: str
    table_count: int
    column_count: int


def _split_top_level(value: str) -> list[str]:
    """Split a DDL fragment on commas outside strings and parentheses."""
    parts: list[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    escaped = False
    for index, character in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def _column_type_registry(columns: str) -> Mapping[str, str]:
    output: dict[str, str] = {}
    for definition in _split_top_level(columns):
        if definition.upper().startswith(("INDEX ", "CONSTRAINT ", "PROJECTION ")):
            continue
        name, separator, type_expression = definition.partition(" ")
        if not separator:
            raise RuntimeError(f"Invalid canonical column definition: {definition!r}")
        type_expression = re.split(
            r"\s+(?:CODEC|DEFAULT|MATERIALIZED|ALIAS|COMMENT)\b",
            type_expression,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        output[name.strip("`")] = " ".join(type_expression.split())
    return MappingProxyType(output)


def _skipping_index_registry(columns: str) -> Mapping[str, str]:
    output: dict[str, str] = {}
    for definition in _split_top_level(columns):
        if not definition.upper().startswith("INDEX "):
            continue
        match = re.match(
            r"INDEX\s+([^\s]+)\s+(.+?)\s+TYPE\s+([^\s]+)\s+GRANULARITY\s+(\d+)\s*$",
            definition,
            flags=re.IGNORECASE,
        )
        if not match:
            raise RuntimeError(f"Invalid canonical skipping-index definition: {definition!r}")
        name, expression, index_type, granularity = match.groups()
        output[name.strip("`")] = re.sub(
            r"\s+", "", f"{expression}|{index_type}|{granularity}"
        ).replace("`", "").lower()
    return MappingProxyType(output)


def _normalized_index_metadata(
    expression: Any, index_type: Any, granularity: Any
) -> str:
    return re.sub(
        r"\s+", "", f"{expression}|{index_type}|{granularity}"
    ).replace("`", "").lower()


def _index_metadata_matches(expected: str, actual: str) -> bool:
    """Compare skip-index metadata across ClickHouse ``type`` variants.

    Recent ClickHouse versions expose the parameterized type through
    ``system.data_skipping_indices.type_full`` while ``type`` contains only
    the base name (for example ``bloom_filter``).  Older versions may expose
    only the base name, so the probability parameter cannot be validated there
    but expression, base type, and granularity still are.
    """
    expected_expression, expected_type, expected_granularity = expected.split("|", 2)
    actual_expression, actual_type, actual_granularity = actual.split("|", 2)
    if (
        expected_expression != actual_expression
        or expected_granularity != actual_granularity
    ):
        return False
    return actual_type == expected_type or actual_type == expected_type.split("(", 1)[0]


def _table(
    name: str,
    columns: str,
    engine: str,
    partition: Optional[str],
    order: str,
    *,
    owner: str,
    durable: bool = True,
    ttl: Optional[str] = None,
) -> CanonicalTableSpec:
    partition_sql = f"\nPARTITION BY {partition}" if partition else ""
    ttl_sql = f"\nTTL {ttl}" if ttl else ""
    ddl = f"""
CREATE TABLE IF NOT EXISTS {{database}}.{name}
(
{columns.strip()}
)
ENGINE = {engine}{partition_sql}
ORDER BY {order}{ttl_sql}
SETTINGS index_granularity = 8192
"""
    return CanonicalTableSpec(
        name=name,
        ddl=ddl,
        owner=owner,
        engine_expression=engine,
        column_types=_column_type_registry(columns),
        partition_expression=partition,
        order_expression=order,
        durable=durable,
        skipping_indexes=_skipping_index_registry(columns),
    )


_SOURCE_MONTH = "toYYYYMM(source_collected_at)"
_EVENT_WEEK = "toMonday(toDate({column}))"
_TTL_30_DAYS = "{column} + INTERVAL 30 DAY DELETE"


_CANONICAL_SPECS = (
    _table(
        "options_chain_raw",
        """
    snapshot_id UUID,
    source LowCardinality(String),
    symbol LowCardinality(String),
    request_started_at_utc DateTime64(6, 'UTC'),
    response_received_at_utc DateTime64(6, 'UTC'),
    collected_at_utc DateTime64(6, 'UTC'),
    response_latency_ms UInt32,
    request_parameters_json String CODEC(ZSTD(3)),
    response_json String CODEC(ZSTD(3)),
    payload_sha256 FixedString(32),
    source_contract_count UInt32,
    is_delayed Bool,
    is_chain_truncated Bool,
    status LowCardinality(String),
    INDEX idx_raw_snapshot snapshot_id TYPE bloom_filter(0.001) GRANULARITY 1
        """,
        "MergeTree",
        "toMonday(toDate(collected_at_utc))",
        "(symbol, collected_at_utc, snapshot_id)",
        owner="collector",
    ),
    _table(
        "options_chain_archive_manifest",
        """
    archive_id UUID,
    archive_kind Enum8('raw_week' = 1, 'normalized_month' = 2),
    range_start DateTime64(6, 'UTC'),
    range_end DateTime64(6, 'UTC'),
    schema_generation UInt16,
    dataset_generation_id UUID,
    state Enum8('planned' = 1, 'writing' = 2, 'written' = 3,
                'verifying' = 4, 'verified' = 5, 'purging' = 6,
                'purged' = 7, 'failed' = 8, 'ignore' = 9),
    archive_path String,
    semantic_content_hash Nullable(FixedString(32)),
    archive_file_hash Nullable(FixedString(32)),
    row_count UInt64,
    distinct_snapshot_count UInt64,
    uncompressed_bytes UInt64,
    compressed_bytes UInt64,
    compression_settings_json String CODEC(ZSTD(3)),
    owner_id String,
    lease_expires_at Nullable(DateTime64(6, 'UTC')),
    attempt_count UInt16,
    created_at DateTime64(6, 'UTC'),
    written_at Nullable(DateTime64(6, 'UTC')),
    verified_at Nullable(DateTime64(6, 'UTC')),
    purged_at Nullable(DateTime64(6, 'UTC')),
    last_error String CODEC(ZSTD(6)),
    updated_at DateTime64(6, 'UTC')
        """,
        "ReplacingMergeTree(updated_at)",
        None,
        "(archive_kind, range_start, range_end)",
        owner="archive",
    ),
    _table(
        "options_chain_snapshots",
        """
    snapshot_id UUID,
    source LowCardinality(String),
    underlying_symbol LowCardinality(String),
    source_collected_at DateTime64(6, 'UTC'),
    request_started_at DateTime64(6, 'UTC'),
    response_received_at DateTime64(6, 'UTC'),
    ingested_at DateTime64(6, 'UTC'),
    response_latency_ms UInt32,
    payload_sha256 FixedString(32),
    chain_status LowCardinality(String),
    strategy LowCardinality(String),
    chain_interval Nullable(Float32),
    chain_is_delayed Bool,
    chain_is_index Bool,
    chain_is_truncated Bool,
    underlying_is_delayed Bool,
    source_contract_count UInt32,
    stored_observation_count UInt32,
    stored_detail_count UInt32,
    stored_distinct_contract_count UInt32,
    asset_main_type LowCardinality(String),
    asset_sub_type LowCardinality(String),
    chain_interest_rate Nullable(Float64),
    chain_dividend_yield Nullable(Float64),
    chain_volatility Nullable(Float64),
    chain_underlying_price Nullable(Float64),
    underlying_bid Nullable(Float64),
    underlying_ask Nullable(Float64),
    underlying_last Nullable(Float64),
    underlying_mark Nullable(Float64),
    underlying_bid_size Nullable(UInt64),
    underlying_ask_size Nullable(UInt64),
    underlying_quote_time Nullable(DateTime64(3, 'UTC')),
    underlying_trade_time Nullable(DateTime64(3, 'UTC')),
    underlying_total_volume Nullable(UInt64),
    collection_definition_id UInt32,
    normalization_definition_id UInt32,
    normalized_at DateTime64(6, 'UTC'),
    INDEX idx_snapshot_id snapshot_id TYPE bloom_filter(0.001) GRANULARITY 1
        """,
        "MergeTree",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id)",
        owner="collector",
    ),
    _table(
        "options_contracts",
        """
    contract_id UInt64,
    option_symbol String,
    underlying_symbol LowCardinality(String),
    option_root LowCardinality(String),
    ssid Nullable(UInt64),
    description String,
    exchange_name LowCardinality(String),
    put_call Enum8('CALL' = 1, 'PUT' = 2),
    strike_price Float64,
    expiration_at DateTime64(3, 'UTC'),
    expiration_type LowCardinality(String),
    last_trading_day Nullable(DateTime64(3, 'UTC')),
    settlement_type LowCardinality(String),
    exercise_type LowCardinality(String),
    multiplier Float32,
    is_mini Bool,
    is_nonstandard Bool,
    is_penny_pilot Bool,
    deliverable_note String,
    option_deliverables_json String CODEC(ZSTD(3)),
    identity_hash FixedString(32),
    metadata_hash FixedString(32),
    first_seen_at DateTime64(6, 'UTC'),
    last_seen_at DateTime64(6, 'UTC'),
    metadata_updated_at DateTime64(6, 'UTC'),
    INDEX idx_contract_symbol option_symbol TYPE bloom_filter(0.001) GRANULARITY 1
        """,
        "ReplacingMergeTree(metadata_updated_at)",
        None,
        "(contract_id)",
        owner="collector",
    ),
    _table(
        "options_chain_observations",
        """
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    underlying_symbol LowCardinality(String),
    contract_id UInt64,
    contract_ordinal UInt32,
    symbol_map_mismatch Bool,
    bid Nullable(Float32),
    ask Nullable(Float32),
    last Nullable(Float32),
    mark Nullable(Float32),
    bid_size Nullable(UInt32),
    ask_size Nullable(UInt32),
    last_size Nullable(UInt32),
    open_price Nullable(Float32),
    high_price Nullable(Float32),
    low_price Nullable(Float32),
    close_price Nullable(Float32),
    total_volume Nullable(UInt32),
    open_interest Nullable(UInt32),
    quote_time Nullable(DateTime64(3, 'UTC')),
    trade_time Nullable(DateTime64(3, 'UTC')),
    source_iv Nullable(Float32),
    source_delta Nullable(Float32),
    source_gamma Nullable(Float32),
    source_theta Nullable(Float32),
    source_vega Nullable(Float32),
    source_rho Nullable(Float32),
    INDEX idx_observation_snapshot snapshot_id TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_observation_contract contract_id TYPE bloom_filter(0.01) GRANULARITY 4
        """,
        "MergeTree",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id, contract_ordinal, contract_id)",
        owner="collector",
    ),
    _table(
        "options_chain_observation_details",
        """
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    underlying_symbol LowCardinality(String),
    contract_id UInt64,
    high_52_week Nullable(Float32),
    low_52_week Nullable(Float32),
    net_change Nullable(Float32),
    percent_change Nullable(Float32),
    mark_change Nullable(Float32),
    mark_percent_change Nullable(Float32),
    intrinsic_value Nullable(Float32),
    extrinsic_value Nullable(Float32),
    time_value Nullable(Float32),
    break_even Nullable(Float32),
    theoretical_option_value Nullable(Float32),
    theoretical_volatility Nullable(Float32),
    INDEX idx_detail_snapshot snapshot_id TYPE bloom_filter(0.001) GRANULARITY 1
        """,
        "MergeTree",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id, contract_id)",
        owner="collector",
    ),
    _table(
        "options_chain_expiration_inputs",
        """
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    underlying_symbol LowCardinality(String),
    expiration_at DateTime64(3, 'UTC'),
    calculation_definition_id UInt32,
    time_to_expiry_seconds UInt32,
    risk_free_rate Float64,
    dividend_yield Float64,
    forward_price Float64,
    fred_observation_date Date,
    fred_lower_series LowCardinality(String),
    fred_upper_series LowCardinality(String),
    quality_flags UInt32,
    calculated_at DateTime64(6, 'UTC')
        """,
        "MergeTree",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id, expiration_at)",
        owner="calculation",
    ),
    _table(
        "options_chain_reference_rates",
        """
    series_id LowCardinality(String),
    observation_date Date,
    value_percent Float64,
    retrieved_at DateTime64(6, 'UTC'),
    source LowCardinality(String)
        """,
        "ReplacingMergeTree(retrieved_at)",
        "toYear(observation_date)",
        "(series_id, observation_date)",
        owner="reference",
    ),
    _table(
        "options_chain_definitions",
        """
    definition_id UInt32,
    definition_type Enum8('collection' = 1, 'normalization' = 2,
                           'calculation' = 3, 'surface' = 4),
    parent_definition_id Nullable(UInt32),
    code_version String,
    configuration_hash FixedString(32),
    solver_configuration_hash FixedString(32),
    configuration_json String CODEC(ZSTD(6)),
    activated_at DateTime64(6, 'UTC'),
    deactivated_at Nullable(DateTime64(6, 'UTC')),
    created_at DateTime64(6, 'UTC'),
    updated_at DateTime64(6, 'UTC'),
    is_active Bool
        """,
        "ReplacingMergeTree(updated_at)",
        None,
        "(definition_type, definition_id)",
        owner="shared",
    ),
    _table(
        "options_chain_processing_attempts",
        """
    attempt_id UUID,
    process_type Enum8('calculation' = 1, 'surface' = 2),
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    definition_id UInt32,
    worker_id String,
    started_at DateTime64(6, 'UTC'),
    finished_at Nullable(DateTime64(6, 'UTC')),
    status LowCardinality(String),
    expected_count UInt32,
    written_success_count UInt32,
    failure_count UInt32,
    error_summary String CODEC(ZSTD(6)),
    created_at DateTime64(6, 'UTC')
        """,
        "MergeTree",
        _EVENT_WEEK.format(column="started_at"),
        "(process_type, started_at, snapshot_id, attempt_id)",
        owner="shared",
        durable=False,
        ttl=_TTL_30_DAYS.format(column="started_at"),
    ),
    _table(
        "options_chain_calculation_results",
        """
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    underlying_symbol LowCardinality(String),
    contract_id UInt64,
    calculation_definition_id UInt32,
    source_calculation_run_id Nullable(UUID),
    calculated_at DateTime64(6, 'UTC'),
    recalc_method LowCardinality(String),
    result_origin Enum8('fresh' = 1, 'reused' = 2),
    origin_snapshot_id UUID,
    input_basis LowCardinality(String),
    dividend_model LowCardinality(String),
    iv_selection Enum8('source' = 1, 'result' = 2),
    greeks_selection Enum8('source' = 1, 'result' = 2),
    target_price Float64,
    iv Float64,
    iv_iterations UInt16,
    price_residual Float64,
    model_price Float64,
    delta Float64,
    gamma Float64,
    theta_per_day Float64,
    vega_per_1pct Float64,
    rho_per_1pct Float64,
    actual_dtype Enum8('float32' = 1, 'float64' = 2),
    precision_escalated Bool,
    quality_flags UInt32
        """,
        "MergeTree",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id, contract_id)",
        owner="calculation",
    ),
    _table(
        "options_chain_calculation_failures",
        """
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    underlying_symbol LowCardinality(String),
    calculation_definition_id UInt32,
    source_calculation_run_id Nullable(UUID),
    failure_scope Enum8('snapshot' = 1, 'expiration' = 2, 'contract' = 3),
    expiration_at Nullable(DateTime64(3, 'UTC')),
    contract_id Nullable(UInt64),
    failure_stage LowCardinality(String),
    reason_code LowCardinality(String),
    error_class LowCardinality(String),
    error_detail String CODEC(ZSTD(6)),
    retry_class LowCardinality(String),
    retry_after Nullable(DateTime64(6, 'UTC')),
    failure_fingerprint FixedString(32),
    first_failed_at DateTime64(6, 'UTC'),
    last_failed_at DateTime64(6, 'UTC'),
    failure_count UInt16,
    updated_at DateTime64(6, 'UTC'),
    is_active Bool,
    valuation_at Nullable(DateTime64(6, 'UTC')),
    spot Nullable(Float64),
    target_price Nullable(Float64),
    time_to_expiry_seconds Nullable(UInt32),
    risk_free_rate Nullable(Float64),
    dividend_yield Nullable(Float64),
    method LowCardinality(Nullable(String)),
    input_basis LowCardinality(Nullable(String)),
    dividend_model LowCardinality(Nullable(String)),
    partial_iv Nullable(Float64),
    partial_model_price Nullable(Float64),
    partial_price_residual Nullable(Float64),
    seed_iv Nullable(Float64),
    seed_origin_snapshot_id Nullable(UUID),
    quality_flags UInt32
        """,
        "ReplacingMergeTree(updated_at)",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id, failure_scope, "
        "ifNull(expiration_at, toDateTime64('1970-01-01 00:00:00', 3, 'UTC')), "
        "ifNull(contract_id, toUInt64(0)))",
        owner="calculation",
    ),
    _table(
        "options_chain_calculation_reuse_provenance",
        """
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    underlying_symbol LowCardinality(String),
    contract_id UInt64,
    calculation_definition_id UInt32,
    origin_snapshot_id UUID,
    origin_calculated_at DateTime64(6, 'UTC'),
    origin_valuation_at DateTime64(6, 'UTC'),
    origin_spot Float64,
    origin_target_price Float64,
    origin_time_to_expiry_seconds UInt32,
    origin_risk_free_rate Float64,
    origin_dividend_yield Float64,
    origin_input_basis LowCardinality(String),
    origin_method LowCardinality(String),
    origin_dividend_model LowCardinality(String),
    reuse_age_seconds UInt32,
    spot_movement_bps Float32,
    target_price_absolute_change Float32,
    target_price_fractional_change Float32,
    elapsed_tenor_fraction Float32,
    rate_change Float32,
    dividend_yield_change Float32,
    estimated_iv_drift Float32,
    estimated_delta_drift Float32,
    reuse_mode LowCardinality(String),
    policy_reason LowCardinality(String),
    created_at DateTime64(6, 'UTC')
        """,
        "MergeTree",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id, contract_id)",
        owner="calculation",
    ),
    _table(
        "options_chain_calculation_diagnostics",
        """
    diagnostic_id UUID,
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    contract_id Nullable(UInt64),
    calculation_definition_id UInt32,
    diagnostic_type LowCardinality(String),
    diagnostic_json String CODEC(ZSTD(6)),
    created_at DateTime64(6, 'UTC')
        """,
        "MergeTree",
        _EVENT_WEEK.format(column="created_at"),
        "(created_at, snapshot_id, diagnostic_id)",
        owner="calculation",
        durable=False,
        ttl=_TTL_30_DAYS.format(column="created_at"),
    ),
    _table(
        "options_chain_calculation_reuse_validation",
        """
    validation_id UUID,
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    contract_id UInt64,
    calculation_definition_id UInt32,
    validation_type LowCardinality(String),
    metrics_json String CODEC(ZSTD(6)),
    created_at DateTime64(6, 'UTC')
        """,
        "MergeTree",
        _EVENT_WEEK.format(column="created_at"),
        "(created_at, snapshot_id, validation_id)",
        owner="calculation",
        durable=False,
        ttl=_TTL_30_DAYS.format(column="created_at"),
    ),
    _table(
        "options_chain_calculation_state",
        """
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    underlying_symbol LowCardinality(String),
    calculation_definition_id UInt32,
    source_calculation_run_id Nullable(UUID),
    status Enum8('pending' = 1, 'running' = 2, 'complete' = 3,
                 'complete_with_failures' = 4, 'failed' = 5,
                 'stale_running' = 6, 'abandoned' = 7),
    expected_observations UInt32,
    eligible_contracts UInt32,
    successful_results UInt32,
    source_only_contracts UInt32,
    terminal_contract_failures UInt32,
    terminal_expiration_failures UInt32,
    snapshot_failure_count UInt32,
    started_at Nullable(DateTime64(6, 'UTC')),
    finished_at Nullable(DateTime64(6, 'UTC')),
    updated_at DateTime64(6, 'UTC'),
    active_attempt_id Nullable(UUID),
    error_summary String CODEC(ZSTD(6)),
    is_active Bool
        """,
        "ReplacingMergeTree(updated_at)",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id)",
        owner="calculation",
    ),
    _table(
        "options_chain_surface_fits",
        """
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    underlying_symbol LowCardinality(String),
    expiration_at DateTime64(3, 'UTC'),
    price_basis LowCardinality(String),
    surface_definition_id UInt32,
    source_surface_run_id Nullable(UUID),
    calculated_at DateTime64(6, 'UTC'),
    pricing_method LowCardinality(String),
    fit_method Enum8('SVI' = 1, 'PCHIP' = 2),
    svi_a Nullable(Float64),
    svi_b Nullable(Float64),
    svi_rho Nullable(Float64),
    svi_m Nullable(Float64),
    svi_sigma Nullable(Float64),
    pchip_k Array(Float64),
    pchip_total_variance Array(Float64),
    input_contract_count UInt32,
    accepted_contract_count UInt32,
    rejected_contract_count UInt32,
    selected_contract_count UInt32,
    source_current_iv_point_count UInt32,
    source_reused_iv_point_count UInt32,
    source_vendor_iv_point_count UInt32,
    no_iv_excluded_count UInt32,
    source_iv_weight_fraction Float64,
    minimum_input_delta Nullable(Float64),
    maximum_input_delta Nullable(Float64),
    minimum_log_forward_moneyness Float64,
    maximum_log_forward_moneyness Float64,
    weighted_rmse Float64,
    maximum_absolute_error Float64,
    butterfly_arbitrage_flags UInt32,
    calendar_arbitrage_flags UInt32,
    quality_flags UInt32,
    actual_dtype Enum8('float32' = 1, 'float64' = 2),
    precision_escalated Bool,
    required_node_status LowCardinality(String),
    optional_node_status LowCardinality(String)
        """,
        "MergeTree",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id, expiration_at, price_basis)",
        owner="surface",
    ),
    _table(
        "options_chain_surface_nodes",
        """
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    underlying_symbol LowCardinality(String),
    expiration_at DateTime64(3, 'UTC'),
    price_basis LowCardinality(String),
    surface_definition_id UInt32,
    source_surface_run_id Nullable(UUID),
    node_type LowCardinality(String),
    node_value Float64,
    put_call Enum8('NONE' = 0, 'CALL' = 1, 'PUT' = 2),
    fitted_strike Float64,
    fitted_delta Nullable(Float64),
    fitted_iv_bid Nullable(Float64),
    fitted_iv_mid Float64,
    fitted_iv_ask Nullable(Float64),
    fit_confidence Float64,
    quality_flags UInt32,
    calculated_at DateTime64(6, 'UTC')
        """,
        "MergeTree",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id, expiration_at, "
        "price_basis, node_type, node_value, put_call)",
        owner="surface",
    ),
    _table(
        "options_chain_surface_failures",
        """
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    underlying_symbol LowCardinality(String),
    surface_definition_id UInt32,
    source_surface_run_id Nullable(UUID),
    failure_scope Enum8('fit' = 1, 'node' = 2),
    expiration_at DateTime64(3, 'UTC'),
    price_basis LowCardinality(String),
    node_type LowCardinality(Nullable(String)),
    node_value Nullable(Float64),
    put_call Nullable(Enum8('NONE' = 0, 'CALL' = 1, 'PUT' = 2)),
    failure_stage LowCardinality(String),
    reason_code LowCardinality(String),
    error_class LowCardinality(String),
    error_detail String CODEC(ZSTD(6)),
    retry_class LowCardinality(String),
    retry_after Nullable(DateTime64(6, 'UTC')),
    failure_fingerprint FixedString(32),
    input_contract_count Nullable(UInt32),
    accepted_contract_count Nullable(UInt32),
    rejected_contract_count Nullable(UInt32),
    first_failed_at DateTime64(6, 'UTC'),
    last_failed_at DateTime64(6, 'UTC'),
    failure_count UInt16,
    updated_at DateTime64(6, 'UTC'),
    quality_flags UInt32,
    is_active Bool
        """,
        "ReplacingMergeTree(updated_at)",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id, expiration_at, "
        "price_basis, failure_scope, ifNull(node_type, ''), "
        "ifNull(node_value, toFloat64(-1e308)), "
        "ifNull(put_call, CAST('NONE', 'Enum8(\\'NONE\\' = 0, \\'CALL\\' = 1, \\'PUT\\' = 2)')))",
        owner="surface",
    ),
    _table(
        "options_chain_surface_diagnostics",
        """
    diagnostic_id UUID,
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    surface_definition_id UInt32,
    diagnostic_type LowCardinality(String),
    diagnostic_json String CODEC(ZSTD(6)),
    created_at DateTime64(6, 'UTC')
        """,
        "MergeTree",
        _EVENT_WEEK.format(column="created_at"),
        "(created_at, snapshot_id, diagnostic_id)",
        owner="surface",
        durable=False,
        ttl=_TTL_30_DAYS.format(column="created_at"),
    ),
    _table(
        "options_chain_surface_validation",
        """
    validation_id UUID,
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    surface_definition_id UInt32,
    validation_type LowCardinality(String),
    metrics_json String CODEC(ZSTD(6)),
    created_at DateTime64(6, 'UTC')
        """,
        "MergeTree",
        _EVENT_WEEK.format(column="created_at"),
        "(created_at, snapshot_id, validation_id)",
        owner="surface",
        durable=False,
        ttl=_TTL_30_DAYS.format(column="created_at"),
    ),
    _table(
        "options_chain_surface_state",
        """
    snapshot_id UUID,
    source_collected_at DateTime64(6, 'UTC'),
    underlying_symbol LowCardinality(String),
    surface_definition_id UInt32,
    source_calculation_definition_id UInt32,
    source_surface_run_id Nullable(UUID),
    status Enum8('pending' = 1, 'running' = 2, 'complete' = 3,
                 'complete_with_failures' = 4, 'failed' = 5,
                 'stale_running' = 6, 'abandoned' = 7),
    expected_expirations UInt32,
    successful_expirations UInt32,
    source_only_or_skipped_expirations UInt32,
    terminal_fit_failures UInt32,
    expected_required_nodes UInt32,
    successful_required_nodes UInt32,
    terminal_required_node_failures UInt32,
    successful_optional_nodes UInt32,
    terminal_optional_node_failures UInt32,
    started_at Nullable(DateTime64(6, 'UTC')),
    finished_at Nullable(DateTime64(6, 'UTC')),
    updated_at DateTime64(6, 'UTC'),
    active_attempt_id Nullable(UUID),
    error_summary String CODEC(ZSTD(6)),
    is_active Bool
        """,
        "ReplacingMergeTree(updated_at)",
        _SOURCE_MONTH,
        "(underlying_symbol, source_collected_at, snapshot_id)",
        owner="surface",
    ),
    _table(
        "options_chain_dataset_metadata",
        """
    dataset_generation_id UUID,
    schema_generation UInt16,
    migration_id UUID,
    created_at DateTime64(6, 'UTC'),
    activated_at Nullable(DateTime64(6, 'UTC')),
    deactivated_at Nullable(DateTime64(6, 'UTC')),
    default_collection_definition_id UInt32,
    default_normalization_definition_id UInt32,
    default_calculation_definition_id UInt32,
    default_surface_definition_id UInt32,
    source_start_at DateTime64(6, 'UTC'),
    source_end_at DateTime64(6, 'UTC'),
    validation_summary_json String CODEC(ZSTD(6)),
    migration_phase LowCardinality(String),
    migration_checkpoint_json String CODEC(ZSTD(6)),
    updated_at DateTime64(6, 'UTC'),
    is_active Bool
        """,
        "ReplacingMergeTree(updated_at)",
        None,
        "(dataset_generation_id)",
        owner="shared",
    ),
)


CANONICAL_SCHEMA = MappingProxyType({spec.name: spec for spec in _CANONICAL_SPECS})
CANONICAL_TABLE_NAMES = tuple(CANONICAL_SCHEMA)
if len(CANONICAL_TABLE_NAMES) != 23:  # fail at import if the operational set drifts
    raise RuntimeError(
        f"Canonical option-chain schema must contain 23 tables, found "
        f"{len(CANONICAL_TABLE_NAMES)}"
    )

CANONICAL_SCHEMA_HASHES = MappingProxyType(
    {name: spec.schema_hash for name, spec in CANONICAL_SCHEMA.items()}
)

# Migration control-plane tables deliberately do not belong to the 23-table
# operational registry.  They are retained across a failed migration and are
# never exposed through application-facing compatibility views.
MIGRATION_SUPPORT_SCHEMA = MappingProxyType(
    {
        "options_chain_migration_leases": _table(
            "options_chain_migration_leases",
            """
    migration_id UUID,
    owner String,
    phase LowCardinality(String),
    status LowCardinality(String),
    created_at DateTime64(6, 'UTC'),
    heartbeat_at DateTime64(6, 'UTC'),
    expires_at DateTime64(6, 'UTC'),
    takeover_of Nullable(UUID),
    audit_message String CODEC(ZSTD(6)),
    updated_at DateTime64(6, 'UTC')
            """,
            "ReplacingMergeTree(updated_at)",
            None,
            "(migration_id)",
            owner="migration",
        ),
        "options_chain_migration_quarantine": _table(
            "options_chain_migration_quarantine",
            """
    quarantine_id UUID,
    migration_id UUID,
    source_table LowCardinality(String),
    source_row_identity String,
    field_name LowCardinality(String),
    raw_value String CODEC(ZSTD(6)),
    conversion_error String CODEC(ZSTD(6)),
    created_at DateTime64(6, 'UTC'),
    updated_at DateTime64(6, 'UTC')
            """,
            "ReplacingMergeTree(updated_at)",
            _EVENT_WEEK.format(column="created_at"),
            "(migration_id, source_table, source_row_identity, quarantine_id)",
            owner="migration",
        ),
        "options_chain_migration_validation": _table(
            "options_chain_migration_validation",
            """
    validation_id UUID,
    migration_id UUID,
    phase LowCardinality(String),
    check_name LowCardinality(String),
    severity LowCardinality(String),
    status LowCardinality(String),
    expected String CODEC(ZSTD(6)),
    actual String CODEC(ZSTD(6)),
    details String CODEC(ZSTD(6)),
    checked_at DateTime64(6, 'UTC'),
    updated_at DateTime64(6, 'UTC')
            """,
            "ReplacingMergeTree(updated_at)",
            _EVENT_WEEK.format(column="checked_at"),
            "(migration_id, checked_at, validation_id)",
            owner="migration",
        ),
    }
)


def migration_support_schema_statements(database: str) -> list[str]:
    database = validate_database_identifier(database)
    return [spec.ddl.format(database=database) for spec in MIGRATION_SUPPORT_SCHEMA.values()]


def _normalized_metadata_expression(value: Any) -> str:
    normalized = re.sub(r"\s+", "", str(value or "")).replace("`", "").lower()
    if normalized.startswith("(") and normalized.endswith(")"):
        depth = 0
        encloses_entire_value = True
        for index, character in enumerate(normalized):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0 and index != len(normalized) - 1:
                    encloses_entire_value = False
                    break
        if encloses_entire_value:
            normalized = normalized[1:-1]
    return normalized


def validate_canonical_table_metadata(
    client: Any,
    database: str,
    table_names: Sequence[str],
) -> SchemaValidationReport:
    """Validate live table metadata directly against the canonical registry.

    The registry is derived from the same column and engine fragments used to
    render DDL, so preflight cannot drift into a second hand-maintained schema.
    """
    database = validate_database_identifier(database)
    names = tuple(dict.fromkeys(str(name) for name in table_names))
    registry = {**CANONICAL_SCHEMA, **MIGRATION_SUPPORT_SCHEMA}
    unknown = sorted(set(names).difference(registry))
    if unknown:
        raise ValueError(f"Tables are not present in the canonical registry: {unknown}")

    table_result = client.query(
        "SELECT name, engine, engine_full, partition_key, sorting_key "
        "FROM system.tables WHERE database = {database:String} "
        "AND name IN {names:Array(String)}",
        parameters={"database": database, "names": list(names)},
    )
    live_tables = {
        str(name): (str(engine), str(engine_full), str(partition), str(sorting))
        for name, engine, engine_full, partition, sorting in table_result.result_rows
    }
    missing_tables = sorted(set(names).difference(live_tables))
    if missing_tables:
        raise SchemaValidationError(
            f"Generation-{SCHEMA_GENERATION} schema is missing tables: {missing_tables}"
        )

    column_result = client.query(
        "SELECT table, name, type FROM system.columns "
        "WHERE database = {database:String} AND table IN {names:Array(String)} "
        "ORDER BY table, position",
        parameters={"database": database, "names": list(names)},
    )
    live_columns: dict[str, dict[str, str]] = {name: {} for name in names}
    for table, name, column_type in column_result.result_rows:
        live_columns[str(table)][str(name)] = str(column_type)

    try:
        index_result = client.query(
            "SELECT table, name, type, type_full, expr, granularity "
            "FROM system.data_skipping_indices "
            "WHERE database = {database:String} AND table IN {names:Array(String)} "
            "ORDER BY table, name",
            parameters={"database": database, "names": list(names)},
        )
        index_rows_include_type_full = True
    except Exception:
        # ``type_full`` is not present on some older ClickHouse versions.
        # Keep the metadata check useful there using the base type.
        index_result = client.query(
            "SELECT table, name, type, expr, granularity "
            "FROM system.data_skipping_indices "
            "WHERE database = {database:String} AND table IN {names:Array(String)} "
            "ORDER BY table, name",
            parameters={"database": database, "names": list(names)},
        )
        index_rows_include_type_full = False
    live_indexes: dict[str, dict[str, str]] = {name: {} for name in names}
    if index_rows_include_type_full:
        for table, index_name, index_type, type_full, expression, granularity in index_result.result_rows:
            live_indexes[str(table)][str(index_name)] = _normalized_index_metadata(
                expression, type_full or index_type, granularity
            )
    else:
        for table, index_name, index_type, expression, granularity in index_result.result_rows:
            live_indexes[str(table)][str(index_name)] = _normalized_index_metadata(
                expression, index_type, granularity
            )

    errors: list[str] = []
    for name in names:
        spec = registry[name]
        engine, engine_full, partition_key, sorting_key = live_tables[name]
        expected_engine = spec.engine_expression.split("(", 1)[0]
        if engine != expected_engine:
            errors.append(f"{name}: engine {engine!r} != {expected_engine!r}")
        expected_engine_expression = _normalized_metadata_expression(
            spec.engine_expression
        )
        if not _normalized_metadata_expression(engine_full).startswith(
            expected_engine_expression
        ):
            errors.append(
                f"{name}: engine expression does not start with "
                f"{spec.engine_expression!r}"
            )
        expected_partition = _normalized_metadata_expression(
            spec.partition_expression or ""
        )
        if _normalized_metadata_expression(partition_key) != expected_partition:
            errors.append(
                f"{name}: partition key {partition_key!r} != "
                f"{(spec.partition_expression or '')!r}"
            )
        if _normalized_metadata_expression(
            sorting_key
        ) != _normalized_metadata_expression(spec.order_expression):
            errors.append(
                f"{name}: sorting key {sorting_key!r} != {spec.order_expression!r}"
            )

        expected_columns = dict(spec.column_types)
        actual_columns = live_columns[name]
        missing_columns = sorted(set(expected_columns).difference(actual_columns))
        unexpected_columns = sorted(set(actual_columns).difference(expected_columns))
        if missing_columns:
            errors.append(f"{name}: missing columns {missing_columns}")
        if unexpected_columns:
            errors.append(f"{name}: unexpected columns {unexpected_columns}")
        for column in sorted(set(expected_columns).intersection(actual_columns)):
            if _normalized_metadata_expression(
                actual_columns[column]
            ) != _normalized_metadata_expression(expected_columns[column]):
                errors.append(
                    f"{name}.{column}: type {actual_columns[column]!r} != "
                    f"{expected_columns[column]!r}"
                )
        expected_indexes = dict(spec.skipping_indexes)
        actual_indexes = live_indexes[name]
        missing_indexes = sorted(set(expected_indexes).difference(actual_indexes))
        unexpected_indexes = sorted(set(actual_indexes).difference(expected_indexes))
        if missing_indexes:
            errors.append(f"{name}: missing skipping indexes {missing_indexes}")
        if unexpected_indexes:
            errors.append(f"{name}: unexpected skipping indexes {unexpected_indexes}")
        for index_name in sorted(set(expected_indexes).intersection(actual_indexes)):
            if not _index_metadata_matches(
                expected_indexes[index_name], actual_indexes[index_name]
            ):
                errors.append(
                    f"{name}.{index_name}: index definition "
                    f"{actual_indexes[index_name]!r} != {expected_indexes[index_name]!r}"
                )
    if errors:
        raise SchemaValidationError(
            f"Generation-{SCHEMA_GENERATION} schema metadata mismatch: "
            + "; ".join(errors)
        )
    return SchemaValidationReport(
        database=database,
        table_count=len(names),
        column_count=sum(len(registry[name].column_types) for name in names),
    )

DURABLE_SOURCE_TABLES = frozenset(
    {
        "options_chain_raw",
        "options_chain_archive_manifest",
        "options_chain_snapshots",
        "options_contracts",
        "options_chain_observations",
        "options_chain_observation_details",
        "options_chain_definitions",
        "options_chain_dataset_metadata",
    }
)

SURFACE_RESET_TABLES = (
    "options_chain_surface_fits",
    "options_chain_surface_nodes",
    "options_chain_surface_failures",
    "options_chain_surface_diagnostics",
    "options_chain_surface_validation",
    "options_chain_surface_state",
)
CALCULATION_RESET_TABLES = (
    "options_chain_expiration_inputs",
    "options_chain_calculation_results",
    "options_chain_calculation_failures",
    "options_chain_calculation_reuse_provenance",
    "options_chain_calculation_diagnostics",
    "options_chain_calculation_reuse_validation",
    "options_chain_calculation_state",
    *SURFACE_RESET_TABLES,
)
REFERENCE_RATE_RESET_TABLES = ("options_chain_reference_rates",)
RESET_TABLES = MappingProxyType(
    {
        "surfaces": SURFACE_RESET_TABLES,
        "contract_calculations": CALCULATION_RESET_TABLES,
        "reference_rates": REFERENCE_RATE_RESET_TABLES,
        "all_derived": CALCULATION_RESET_TABLES + REFERENCE_RATE_RESET_TABLES,
    }
)


def canonical_schema_statements(database: str) -> list[str]:
    database = validate_database_identifier(database)
    return [
        f"CREATE DATABASE IF NOT EXISTS {database}",
        *(spec.ddl.format(database=database) for spec in _CANONICAL_SPECS),
    ]


def calculation_schema_statements(database: str) -> list[str]:
    """Return the canonical schema plus safe in-place schema evolution."""
    database = validate_database_identifier(database)
    return [
        *canonical_schema_statements(database),
        f"ALTER TABLE {database}.options_chain_surface_nodes "
        "MODIFY COLUMN fitted_delta Nullable(Float64)",
    ]


def plan_calculation_reset(database: str, reset_level: str) -> ResetPlan:
    database = validate_database_identifier(database)
    if reset_level not in RESET_TABLES:
        raise ValueError(f"Unknown calculation reset level: {reset_level!r}")
    objects = tuple(RESET_TABLES[reset_level])
    if DURABLE_SOURCE_TABLES.intersection(objects):
        raise RuntimeError("Reset plan intersects durable source ownership")
    drops = tuple(
        f"DROP TABLE IF EXISTS {database}.{name} SYNC" for name in reversed(objects)
    )
    attempts = []
    if reset_level in {"surfaces", "contract_calculations", "all_derived"}:
        process_types = (
            "('surface')"
            if reset_level == "surfaces"
            else "('calculation', 'surface')"
        )
        attempts.append(
            f"ALTER TABLE {database}.options_chain_processing_attempts "
            f"DELETE WHERE process_type IN {process_types} "
            "SETTINGS mutations_sync = 2"
        )
    creates = tuple(
        CANONICAL_SCHEMA[name].ddl.format(database=database) for name in objects
    )
    return ResetPlan(
        reset_level,
        database,
        objects,
        drops + tuple(attempts),
        creates,
    )


def _source_table_identities(client: Any, database: str) -> dict[str, str]:
    names = sorted(DURABLE_SOURCE_TABLES)
    result = client.query(
        "SELECT name, toString(uuid) FROM system.tables "
        "WHERE database = {database:String} AND name IN {names:Array(String)} "
        "ORDER BY name",
        parameters={"database": database, "names": names},
    )
    identities = {str(name): str(value) for name, value in result.result_rows}
    if set(identities) != DURABLE_SOURCE_TABLES:
        raise RuntimeError(
            "Durable source tables are missing before/after reset: "
            f"found {sorted(identities)}"
        )
    return identities


def execute_reset_plan(
    client: Any,
    plan: ResetPlan,
    *,
    logger: Optional[logging.Logger] = None,
    verify_source_uuids: bool = True,
) -> None:
    """Execute a literal derived-table reset and prove source UUID stability."""
    logger = logger or logging.getLogger(__name__)
    before = (
        _source_table_identities(client, plan.database)
        if verify_source_uuids
        else None
    )
    for statement in plan.statements:
        logger.warning("Calculation reset plan: %s", " ".join(statement.split()))
    for statement in plan.statements:
        client.command(statement)
    if before is not None:
        after = _source_table_identities(client, plan.database)
        if after != before:
            raise RuntimeError("Durable source table UUID changed during derived reset")


def reset_calculation_schema(
    context: Mapping[str, Any],
    *,
    reset_level: str,
    confirm_database: Optional[str],
    dry_run: bool = False,
    client_factory: Any = create_clickhouse_client,
    logger: Optional[logging.Logger] = None,
) -> ResetPlan:
    database = validate_database_identifier(
        str(context["options_chain.clickhouse.database"])
    )
    plan = plan_calculation_reset(database, reset_level)
    logger = logger or logging.getLogger(__name__)
    if dry_run:
        if confirm_database is not None and confirm_database != database:
            raise ValueError(
                f"Reset confirmation {confirm_database!r} does not match {database!r}"
            )
        for statement in plan.statements:
            logger.warning("Calculation reset plan: %s", " ".join(statement.split()))
        return plan
    if confirm_database != database:
        raise ValueError(f"Destructive reset requires --confirm-reset {database}")
    client = client_factory(context)
    try:
        execute_reset_plan(client, plan, logger=logger)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return plan


def create_calculation_schema(
    context: Mapping[str, Any],
    *,
    client_factory: Any = create_clickhouse_client,
    source_validator: Optional[Any] = None,
) -> None:
    database = validate_database_identifier(
        str(context["options_chain.clickhouse.database"])
    )
    client = client_factory(context, database="default")
    try:
        # Migration support is outside the 23-table operational registry but
        # the runtime preflight requires the lease table to exist before any
        # worker can discover or claim work.
        for statement in (
            *calculation_schema_statements(database),
            *migration_support_schema_statements(database),
        ):
            client.command(statement)
        if source_validator is not None:
            source_validator(client, database)
    finally:
        client.close()
