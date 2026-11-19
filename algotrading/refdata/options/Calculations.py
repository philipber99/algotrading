"""Reusable normalized option-chain numerical functions and CLI helpers.

The worker has no collector, raw-payload, or filesystem dependency.  Its market
data boundary is the normalized snapshot, observation, and current-contract
tables; reference rates are durably cached in ClickHouse.

Future Phase 2 underlying-path integration is intentionally database-only::

    MarketData.prices_sql(
        context, symbol, start_dt, end_dt, frequency="1min",
        mkt_hours_only=True, skip_tm_adjust=True,
    )

That later deployment will need MySQL access and should set
``refdata.load_intraday_for_today = False``.  Phase 1 does not call it.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import logging
import math
import os
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Collection, Mapping, Optional, Sequence

import pyarrow as pa
import requests

from algotrading.Common import load_configuration
from algotrading.ContextFields import (
    PARAM_ALGO_HOME,
    PARAM_DEFAULT_LOG_LEVEL,
    PARAM_LOGGER,
    PARAM_LOG_PATH,
)
from algotrading.refdata.options.Common import (
    OPTIONS_CALCS_CONFIG_PREFIX,
    OPTIONS_CALCS_FRED_PREFIX,
    OPTIONS_CALCS_GREEKS_PREFIX,
    OPTIONS_CALCS_SURFACE_PREFIX,
    InconsistentSnapshotError,
    SnapshotDataError,
    _as_bool,
    _compact_json,
    _finite_float,
    has_required_source_greeks,
    _parse_timestamp,
    _rows_to_arrow,
    _utc_now,
    create_clickhouse_client,
    is_operational_options_calcs_key,
    solver_tenor_seconds,
    validate_options_calcs_configuration,
    validate_database_identifier,
)
from algotrading.refdata.options.Schema import (
    CANONICAL_TABLE_NAMES,
    SCHEMA_GENERATION,
    SchemaValidationError,
    calculation_schema_statements as _owned_schema_statements,
    create_calculation_schema as _create_owned_schema,
    reset_calculation_schema,
    validate_canonical_table_metadata,
)
from algotrading.refdata.api.market_calendars import (
    MARKET_CALENDARS,
    exchange_sessions,
)
from algotrading.refdata.options.Definitions import (
    canonical_configuration_json,
    digest32,
)
from algotrading.refdata.options.Runtime import (
    ATTEMPT_SCHEMA,
    CALCULATION_FAILURE_SCHEMA,
    CALCULATION_STATE_SCHEMA,
    RESULT_SCHEMA,
    CalculationRepository,
    CalculationWork,
    SurfaceWork,
)
from algotrading.refdata.options.Queries import (
    CALCULATION_SOURCE_FIELD_EXPRESSIONS,
    authoritative_calculation_fact_subquery,
    calculation_source_projection_columns,
)
from algotrading.refdata.options.Surface import (
    ExpirationCalibration,
    NumericPrecisionPolicy,
    PrecisionEscalationFailure,
    QUALITY_NODE_ORDERING,
    SurfaceStageError,
    SVIWarmStartCache,
    SurfaceNumericalRuntime,
    apply_calendar_arbitrage_diagnostics,
    calibrate_expiration_bases,
    generate_calibration_nodes_many,
    merge_bid_mid_ask_nodes,
    resolve_numeric_dtype,
    surface_contract_exclusion_reason,
)


FRED_ANCHORS = (
    (0.0, "SOFR"),
    (30.0, "DGS1MO"),
    (91.0, "DGS3MO"),
    (182.0, "DGS6MO"),
    (365.0, "DGS1"),
)


REFERENCE_RATE_SCHEMA = pa.schema(
    [
        pa.field("series_id", pa.string(), nullable=False),
        pa.field("observation_date", pa.date32(), nullable=False),
        pa.field("value_percent", pa.float64(), nullable=False),
        pa.field("retrieved_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)

REUSE_VALIDATION_SCHEMA = pa.schema(
    [
        pa.field("validation_id", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("contract_id", pa.uint64(), nullable=False),
        pa.field("calculation_definition_id", pa.uint32(), nullable=False),
        pa.field("validation_type", pa.string(), nullable=False),
        pa.field("metrics_json", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

SURFACE_VALIDATION_SCHEMA = pa.schema(
    [
        pa.field("validation_id", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("surface_definition_id", pa.uint32(), nullable=False),
        pa.field("validation_type", pa.string(), nullable=False),
        pa.field("metrics_json", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


CONTRACT_TIMING_NAMES = (
    "contract.raw_validation_seconds",
    "contract.base_load_seconds",
    "contract.fred_prepare_seconds",
    "contract.row_calculation_seconds",
    "contract.recalc_seconds",
    "contract.prior_outcome_lookup_seconds",
    "contract.failure_outcome_lookup_seconds",
    "contract.reuse_assessment_seconds",
    "contract.action_decision_seconds",
    "contract.seed_assessment_seconds",
    "contract.failure_reuse_assessment_seconds",
    "contract.reuse_copy_seconds",
    "contract.shadow_solver_seconds",
    "contract.seed_baseline_seconds",
    "contract.validation_insert_seconds",
    "contract.arrow_build_seconds",
    "contract.insert_seconds",
    "contract.verify_seconds",
    "contract.total_seconds",
)

SURFACE_TIMING_NAMES = (
    "surface.source_load_seconds",
    "surface.input_prepare_seconds",
    "surface.iv_input_resolution_seconds",
    "surface.iv_inversion_seconds",
    "surface.fit_point_selection_seconds",
    "surface.fit_coreset_seconds",
    "surface.fit_expansion_seconds",
    "surface.fit_full_set_fallback_seconds",
    "surface.fit_full_set_shadow_seconds",
    "surface.fit_full_set_acceptance_seconds",
    "surface.smile_fit_seconds",
    "surface.arbitrage_seconds",
    "surface.node_generation_seconds",
    "surface.node_tian_search_seconds",
    "surface.node_validation_seconds",
    "surface.node_cheap_search_seconds",
    "surface.node_exact_refinement_seconds",
    "surface.node_dense_fallback_seconds",
    "surface.arrow_build_seconds",
    "surface.insert_seconds",
    "surface.verify_seconds",
    "surface.total_seconds",
)

CONTRACT_LOG_TIMING_NAMES = (
    "contract.total_seconds",
    "contract.row_calculation_seconds",
    "contract.recalc_seconds",
    "contract.insert_seconds",
    "contract.verify_seconds",
)

SURFACE_LOG_TIMING_NAMES = (
    "surface.total_seconds",
    "surface.source_load_seconds",
    "surface.iv_input_resolution_seconds",
    "surface.iv_inversion_seconds",
    "surface.smile_fit_seconds",
    "surface.node_generation_seconds",
    "surface.node_tian_search_seconds",
    "surface.node_validation_seconds",
    "surface.node_cheap_search_seconds",
    "surface.node_exact_refinement_seconds",
    "surface.node_dense_fallback_seconds",
    "surface.insert_seconds",
)


@dataclass
class StageTimings:
    """Negligible-overhead in-memory stage timings for one durable run."""

    values: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] = self.values.get(name, 0.0) + (
                time.perf_counter() - started
            )

    def initialize(self, names: Sequence[str]) -> None:
        for name in names:
            self.values.setdefault(name, 0.0)

    def snapshot(self) -> Mapping[str, float]:
        # Run results cross a spawned-process boundary.  A detached dict keeps
        # snapshot semantics while remaining pickle-compatible; mappingproxy
        # cannot be serialized by multiprocessing's ForkingPickler.
        return dict(sorted(self.values.items()))


@dataclass(frozen=True)
class PreparedFredRate:
    rate: float
    observation_date: date
    lower_series: str
    upper_series: str


@dataclass(frozen=True)
class PreparedFredSnapshot:
    snapshot_date: date
    by_expiration: Mapping[datetime, PreparedFredRate]

    def rate_for(self, expiration_at: datetime) -> PreparedFredRate:
        return self.by_expiration[expiration_at]


NUMERICAL_CALCULATION_INPUT_COLUMNS = (
    "snapshot_id",
    "collected_at",
    "underlying_symbol",
    "chain_is_delayed",
    "chain_is_truncated",
    "underlying_is_delayed",
    "symbol_map_mismatch",
    "chain_dividend_yield",
    "chain_underlying_price",
    "underlying_bid",
    "underlying_ask",
    "underlying_last",
    "underlying_mark",
    "underlying_quote_time",
    "contract_id",
    "option_symbol",
    "put_call",
    "strike_price",
    "expiration_at",
    "exercise_type",
    "multiplier",
    "is_mini",
    "is_nonstandard",
    "bid",
    "ask",
    "quote_time",
    "source_iv",
    "source_delta",
    "source_gamma",
    "source_theta",
    "source_vega",
    "source_rho",
    "stored_contract_count",
)

_NUMERICAL_TO_CANONICAL_SOURCE_FIELDS = MappingProxyType(
    {
        **{
            name: name
            for name, _expression in CALCULATION_SOURCE_FIELD_EXPRESSIONS
            if name not in {"source_collected_at", "stored_observation_count"}
        },
        "collected_at": "source_collected_at",
        "stored_contract_count": "stored_observation_count",
    }
)


def project_canonical_calculation_rows(
    work: Any,
    source_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project immutable canonical source rows into the numerical input shape.

    This is intentionally the only persistence-to-solver adapter.  It is
    strict about generation-3 names so a retry cannot silently revive a legacy
    flattened row or publish against the wrong snapshot owner.
    """
    expected_count = int(work.expected_observations)
    if len(source_rows) != expected_count:
        raise InconsistentSnapshotError(
            f"Snapshot {work.snapshot_id} returned {len(source_rows)} source rows; "
            f"expected {expected_count}"
        )
    work_snapshot = str(work.snapshot_id)
    work_timestamp = _parse_timestamp(work.source_collected_at)
    if work_timestamp is None:
        raise SnapshotDataError("CalculationWork has an invalid source_collected_at")
    work_symbol = str(work.underlying_symbol).strip().upper()
    required_canonical = {
        canonical for canonical in _NUMERICAL_TO_CANONICAL_SOURCE_FIELDS.values()
    }
    seen_contract_ids: set[int] = set()
    projected_rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        if "collected_at" in source or "stored_contract_count" in source:
            raise SnapshotDataError(
                f"Source row {index} uses a legacy numerical field; canonical "
                "persistence requires source_collected_at and "
                "stored_observation_count"
            )
        missing = sorted(name for name in required_canonical if name not in source)
        if missing:
            raise SnapshotDataError(
                f"Source row {index} is missing canonical calculation fields: "
                + ", ".join(missing)
            )
        source_snapshot = str(source["snapshot_id"])
        if source_snapshot != work_snapshot:
            raise InconsistentSnapshotError(
                f"Source row {index} snapshot_id={source_snapshot!r} does not "
                f"match requested snapshot {work_snapshot!r}"
            )
        source_timestamp = _parse_timestamp(source["source_collected_at"])
        if source_timestamp is None or source_timestamp != work_timestamp:
            raise InconsistentSnapshotError(
                f"Source row {index} source_collected_at={source.get('source_collected_at')!r} "
                f"does not match CalculationWork source_collected_at="
                f"{work_timestamp.isoformat()}"
            )
        source_symbol = str(source["underlying_symbol"]).strip().upper()
        if source_symbol != work_symbol:
            raise InconsistentSnapshotError(
                f"Source row {index} underlying_symbol={source_symbol!r} does not "
                f"match CalculationWork underlying_symbol={work_symbol!r}"
            )
        try:
            contract_id = int(source["contract_id"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise SnapshotDataError(
                f"Source row {index} has invalid contract_id={source.get('contract_id')!r}"
            ) from exc
        if contract_id in seen_contract_ids:
            raise InconsistentSnapshotError(
                f"Snapshot {work_snapshot} contains duplicate contract_id={contract_id}"
            )
        seen_contract_ids.add(contract_id)
        try:
            stored_count = int(source["stored_observation_count"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise SnapshotDataError(
                f"Source row {index} has invalid stored_observation_count="
                f"{source.get('stored_observation_count')!r}"
            ) from exc
        if stored_count != expected_count:
            raise InconsistentSnapshotError(
                f"Source row {index} stored_observation_count={stored_count} "
                f"does not equal expected_observations={expected_count}"
            )
        put_call = str(source["put_call"]).strip().upper()
        if put_call not in {"CALL", "PUT"}:
            raise SnapshotDataError(
                f"Source row {index} has invalid put_call={source['put_call']!r}; "
                "expected CALL or PUT"
            )
        output: dict[str, Any] = {}
        for numerical_name in NUMERICAL_CALCULATION_INPUT_COLUMNS:
            canonical_name = _NUMERICAL_TO_CANONICAL_SOURCE_FIELDS[numerical_name]
            output[numerical_name] = source[canonical_name]
        output["snapshot_id"] = work_snapshot
        output["collected_at"] = work_timestamp
        output["put_call"] = put_call
        output["stored_contract_count"] = stored_count
        projected_rows.append(output)
    return projected_rows

SOURCE_REQUIRED_TABLES = (
    "options_chain_snapshots",
    "options_chain_observations",
    "options_contracts",
)


def calculation_schema_statements(database: str) -> list[str]:
    return _owned_schema_statements(database)


def validate_source_schema(client: Any, database: str) -> None:
    """Fail early against canonical types, engines, and keys."""
    validate_canonical_table_metadata(client, database, SOURCE_REQUIRED_TABLES)


@dataclass(frozen=True)
class CalculationPreflightReport:
    database: str
    server_version: str
    schema_generation: int
    validated_table_count: int
    validated_column_count: int
    active_dataset_generation_id: str
    calculation_definition_ids: tuple[int, ...]
    surface_definition_ids: tuple[int, ...]
    sampled_snapshot_id: Optional[uuid.UUID]


def calculations_preflight(
    context: Mapping[str, Any],
    client: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> CalculationPreflightReport:
    """Validate the complete generation-3 calculation read/write boundary.

    This function is deliberately read-only.  It is called before repository
    construction and definition resolution so schema drift cannot claim a
    snapshot and fail during publication.
    """
    database = validate_database_identifier(
        str(context["options_chain.clickhouse.database"])
    )
    logger = logger or logging.getLogger(__name__)
    errors: list[str] = []
    try:
        version_rows = client.query("SELECT version()").result_rows
        server_version = str(version_rows[0][0]) if len(version_rows) == 1 else ""
        if not server_version.strip():
            errors.append("ClickHouse did not return exactly one server version")
    except Exception as exc:
        server_version = "unknown"
        errors.append(f"server version probe failed: {exc}")

    schema_report = None
    try:
        schema_report = validate_canonical_table_metadata(
            client,
            database,
            (*CANONICAL_TABLE_NAMES, "options_chain_migration_leases"),
        )
    except Exception as exc:
        errors.append(f"canonical metadata mismatch: {exc}")

    dataset_generation_id = ""
    default_ids: tuple[int, int, int, int] = (0, 0, 0, 0)
    try:
        rows = client.query(
            f"""
            SELECT toString(dataset_generation_id), schema_generation,
                   default_collection_definition_id,
                   default_normalization_definition_id,
                   default_calculation_definition_id,
                   default_surface_definition_id
            FROM {database}.options_chain_dataset_metadata FINAL
            WHERE is_active
            """
        ).result_rows
        if len(rows) != 1:
            errors.append(
                "active dataset metadata cardinality is "
                f"{len(rows)}; expected exactly one"
            )
        else:
            row = rows[0]
            dataset_generation_id = str(row[0])
            if dataset_generation_id.replace("-", "").strip("0") == "":
                errors.append("active dataset_generation_id is zero")
            if int(row[1]) != SCHEMA_GENERATION:
                errors.append(
                    f"active dataset schema_generation={row[1]} != {SCHEMA_GENERATION}"
                )
            default_ids = tuple(int(value or 0) for value in row[2:6])  # type: ignore[assignment]
        overlap = client.query(
            f"""
            SELECT count()
            FROM (SELECT * FROM {database}.options_chain_dataset_metadata FINAL
                  WHERE is_active) AS left_dataset
            INNER JOIN (SELECT * FROM {database}.options_chain_dataset_metadata FINAL
                        WHERE is_active) AS right_dataset
              ON left_dataset.dataset_generation_id != right_dataset.dataset_generation_id
             AND left_dataset.source_start_at < right_dataset.source_end_at
             AND right_dataset.source_start_at < left_dataset.source_end_at
            """
        ).result_rows
        if overlap and int(overlap[0][0] or 0):
            errors.append(f"active dataset source ranges overlap: {overlap[0][0]}")
    except Exception as exc:
        errors.append(f"dataset metadata validation failed: {exc}")

    calculation_definition_ids: tuple[int, ...] = ()
    surface_definition_ids: tuple[int, ...] = ()
    try:
        definition_rows = client.query(
            f"""
            SELECT definition_id, toString(definition_type), parent_definition_id,
                   activated_at, deactivated_at, is_active
            FROM {database}.options_chain_definitions FINAL
            WHERE is_active
            ORDER BY definition_type, definition_id
            """
        ).result_rows
        active_definitions = {
            int(row[0]): (str(row[1]), row[2], row[3], row[4])
            for row in definition_rows
        }
        calculation_definition_ids = tuple(
            sorted(identifier for identifier, row in active_definitions.items()
                   if row[0] == "calculation")
        )
        surface_definition_ids = tuple(
            sorted(identifier for identifier, row in active_definitions.items()
                   if row[0] == "surface")
        )
        for identifier in surface_definition_ids:
            parent_id = active_definitions[identifier][1]
            if (
                parent_id is None
                or int(parent_id) not in active_definitions
                or active_definitions[int(parent_id)][0] != "calculation"
            ):
                errors.append(
                    f"active surface definition {identifier} has no active "
                    f"calculation parent: {parent_id!r}"
                )
        expected_default_types = (
            (default_ids[0], "collection"),
            (default_ids[1], "normalization"),
            (default_ids[2], "calculation"),
            (default_ids[3], "surface"),
        )
        for identifier, expected_type in expected_default_types:
            if identifier and (
                identifier not in active_definitions
                or active_definitions[identifier][0] != expected_type
            ):
                errors.append(
                    f"dataset default definition {identifier} is not an active "
                    f"{expected_type} definition"
                )
        overlap_rows = client.query(
            f"""
            SELECT count()
            FROM (SELECT definition_id, definition_type, activated_at,
                         deactivated_at FROM {database}.options_chain_definitions FINAL
                  WHERE is_active) AS left_definition
            INNER JOIN (SELECT definition_id, definition_type, activated_at,
                               deactivated_at FROM {database}.options_chain_definitions FINAL
                        WHERE is_active) AS right_definition
              ON left_definition.definition_type = right_definition.definition_type
             AND left_definition.definition_id != right_definition.definition_id
             AND left_definition.activated_at < ifNull(right_definition.deactivated_at,
                                                        toDateTime64('2100-01-01 00:00:00', 6, 'UTC'))
             AND right_definition.activated_at < ifNull(left_definition.deactivated_at,
                                                        toDateTime64('2100-01-01 00:00:00', 6, 'UTC'))
            """
        ).result_rows
        if overlap_rows and int(overlap_rows[0][0] or 0):
            errors.append(f"active definition intervals overlap: {overlap_rows[0][0]}")
    except Exception as exc:
        errors.append(f"definition validation failed: {exc}")

    validation_queries = {
        "snapshot_definition_applicability": f"""
            SELECT count() FROM {database}.options_chain_snapshots AS s
            LEFT JOIN (SELECT * FROM {database}.options_chain_definitions FINAL) AS collection
              ON collection.definition_id = s.collection_definition_id
             AND collection.definition_type = 'collection' AND collection.is_active
             AND collection.activated_at <= s.source_collected_at
             AND (isNull(collection.deactivated_at) OR s.source_collected_at < collection.deactivated_at)
            LEFT JOIN (SELECT * FROM {database}.options_chain_definitions FINAL) AS normalization
              ON normalization.definition_id = s.normalization_definition_id
             AND normalization.definition_type = 'normalization' AND normalization.is_active
             AND normalization.activated_at <= s.source_collected_at
             AND (isNull(normalization.deactivated_at) OR s.source_collected_at < normalization.deactivated_at)
            WHERE isNull(collection.definition_id) OR isNull(normalization.definition_id)
        """,
        "calculation_state_definition_applicability": f"""
            SELECT count() FROM (SELECT * FROM {database}.options_chain_calculation_state FINAL) AS state
            LEFT JOIN (SELECT * FROM {database}.options_chain_definitions FINAL) AS definition
              ON definition.definition_id = state.calculation_definition_id
             AND definition.definition_type = 'calculation' AND definition.is_active
             AND definition.activated_at <= state.source_collected_at
             AND (isNull(definition.deactivated_at) OR state.source_collected_at < definition.deactivated_at)
            WHERE state.is_active AND isNull(definition.definition_id)
        """,
        "surface_state_definition_applicability": f"""
            SELECT count() FROM (SELECT * FROM {database}.options_chain_surface_state FINAL) AS state
            LEFT JOIN (SELECT * FROM {database}.options_chain_definitions FINAL) AS definition
              ON definition.definition_id = state.surface_definition_id
             AND definition.definition_type = 'surface' AND definition.is_active
             AND definition.activated_at <= state.source_collected_at
             AND (isNull(definition.deactivated_at) OR state.source_collected_at < definition.deactivated_at)
            WHERE state.is_active AND isNull(definition.definition_id)
        """,
        "surface_source_calculation_definition_applicability": f"""
            SELECT count()
            FROM (SELECT * FROM {database}.options_chain_surface_state FINAL) AS surface
            LEFT JOIN (SELECT * FROM {database}.options_chain_definitions FINAL) AS definition
              ON definition.definition_id = surface.source_calculation_definition_id
             AND definition.definition_type = 'calculation' AND definition.is_active
             AND definition.activated_at <= surface.source_collected_at
             AND (isNull(definition.deactivated_at)
                  OR surface.source_collected_at < definition.deactivated_at)
            WHERE surface.is_active AND isNull(definition.definition_id)
        """,
    }
    for name, sql in validation_queries.items():
        try:
            rows = client.query(sql).result_rows
            if rows and int(rows[0][0] or 0):
                errors.append(f"{name}: {rows[0][0]} invalid rows")
        except Exception as exc:
            errors.append(f"{name} probe failed: {exc}")

    sampled_snapshot_id: Optional[uuid.UUID] = None
    try:
        rows = client.query(
            f"SELECT toString(snapshot_id) FROM {database}.options_chain_snapshots "
            "GROUP BY snapshot_id ORDER BY max(source_collected_at) DESC LIMIT 1"
        ).result_rows
        if rows:
            sampled_snapshot_id = uuid.UUID(str(rows[0][0]))
    except Exception as exc:
        errors.append(f"sample snapshot probe failed: {exc}")

    # These are schema-only representative probes.  LIMIT 0 keeps them bounded
    # and, critically, incapable of claiming or publishing work.
    representative_queries = {
        "normalized_source_loading": (
            f"SELECT {', '.join(calculation_source_projection_columns())} "
            f"FROM {database}.options_chain_snapshots AS s "
            f"INNER JOIN {database}.options_chain_observations AS o "
            "ON o.snapshot_id = s.snapshot_id AND o.source_collected_at = s.source_collected_at "
            f"INNER JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c "
            "ON c.contract_id = o.contract_id LIMIT 0"
        ),
        "calculation_work_discovery": (
            f"SELECT snapshot_id, source_collected_at, calculation_definition_id "
            f"FROM {database}.options_chain_calculation_state FINAL LIMIT 0"
        ),
        "calculation_result_reads": (
            f"SELECT * FROM {database}.options_chain_calculation_results LIMIT 0"
        ),
        "calculation_failure_reads": (
            f"SELECT * FROM {database}.options_chain_calculation_failures FINAL LIMIT 0"
        ),
        "calculation_state_reads": (
            f"SELECT * FROM {database}.options_chain_calculation_state FINAL LIMIT 0"
        ),
        "expiration_inputs": (
            f"SELECT * FROM {database}.options_chain_expiration_inputs LIMIT 0"
        ),
        "surface_state": (
            f"SELECT * FROM {database}.options_chain_surface_state FINAL LIMIT 0"
        ),
        "surface_fits": (
            f"SELECT * FROM {database}.options_chain_surface_fits LIMIT 0"
        ),
        "surface_nodes": (
            f"SELECT * FROM {database}.options_chain_surface_nodes LIMIT 0"
        ),
        "surface_failures": (
            f"SELECT * FROM {database}.options_chain_surface_failures FINAL LIMIT 0"
        ),
        "migration_lease_lookup": (
            f"SELECT * FROM {database}.options_chain_migration_leases FINAL LIMIT 0"
        ),
    }
    try:
        from algotrading.refdata.options.Queries import build_effective_chain_query

        representative_queries["effective_chain"] = (
            f"SELECT * FROM ({build_effective_chain_query(database, uuid.uuid4()).sql}) "
            "AS bounded_effective_chain LIMIT 0"
        )
    except Exception as exc:
        errors.append(f"effective-chain query construction failed: {exc}")
    for name, sql in representative_queries.items():
        try:
            client.query(sql, parameters={"snapshot_id": str(sampled_snapshot_id or uuid.uuid4())})
        except Exception as exc:
            errors.append(f"representative query {name} failed: {exc}")

    if errors:
        raise SchemaValidationError(
            f"Calculations preflight failed for database={database} "
            f"server_version={server_version}: " + "; ".join(errors)
        )
    assert schema_report is not None
    report = CalculationPreflightReport(
        database=database,
        server_version=server_version,
        schema_generation=SCHEMA_GENERATION,
        validated_table_count=schema_report.table_count,
        validated_column_count=schema_report.column_count,
        active_dataset_generation_id=dataset_generation_id,
        calculation_definition_ids=calculation_definition_ids,
        surface_definition_ids=surface_definition_ids,
        sampled_snapshot_id=sampled_snapshot_id,
    )
    logger.info(
        "event=calculations_preflight status=success database=%s server_version=%s "
        "schema_generation=%s validated_table_count=%s validated_column_count=%s "
        "active_dataset_generation_id=%s calculation_definition_ids=%s "
        "surface_definition_ids=%s sampled_snapshot_id=%s",
        report.database,
        report.server_version,
        report.schema_generation,
        report.validated_table_count,
        report.validated_column_count,
        report.active_dataset_generation_id,
        report.calculation_definition_ids,
        report.surface_definition_ids,
        report.sampled_snapshot_id,
    )
    return report


def create_calculation_schema(
    context: Mapping[str, Any], *, client_factory: Any = create_clickhouse_client
) -> None:
    validate_options_calcs_configuration(context)
    _create_owned_schema(
        context,
        client_factory=client_factory,
        source_validator=validate_source_schema,
    )


class FredCurve:
    """FRED curve backed only by ClickHouse durable observations."""

    endpoint = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        context: Mapping[str, Any],
        client: Any,
        logger: logging.Logger,
        *,
        http_session: Any = None,
        allow_network: bool = True,
    ):
        validate_options_calcs_configuration(context)
        self.context = context
        self.client = client
        self.logger = logger
        self.allow_network = bool(allow_network)
        self._owns_http_session = bool(self.allow_network and http_session is None)
        self.http_session = (
            (http_session or requests.Session()) if self.allow_network else None
        )
        self.database = validate_database_identifier(
            str(context["options_chain.clickhouse.database"])
        )
        prefix = OPTIONS_CALCS_FRED_PREFIX
        self.api_key = str(context.get(prefix + "api_key", ""))
        configured = context.get(prefix + "series", "SOFR,DGS1MO,DGS3MO,DGS6MO,DGS1")
        if isinstance(configured, str):
            configured = configured.split(",")
        self.series = [
            str(item).strip().upper() for item in configured if str(item).strip()
        ]
        required = {series for _, series in FRED_ANCHORS}
        if not required.issubset(self.series):
            raise ValueError(f"FRED series must contain {sorted(required)}")
        self.max_age_days = int(context.get(prefix + "max_curve_age_days", 5))
        self.history_days = int(context.get(prefix + "history_days", 3650))
        self.trading_calendar = str(
            context.get(prefix + "trading_calendar", "XNYS")
        ).strip().upper()
        if self.trading_calendar not in MARKET_CALENDARS:
            raise ValueError(
                f"{prefix}trading_calendar is unknown: {self.trading_calendar!r}"
            )
        try:
            self.refresh_retry_seconds = float(
                context.get(prefix + "refresh_retry_seconds", 60)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{prefix}refresh_retry_seconds must be finite and positive"
            ) from exc
        if (
            not math.isfinite(self.refresh_retry_seconds)
            or self.refresh_retry_seconds <= 0
        ):
            raise ValueError(
                f"{prefix}refresh_retry_seconds must be finite and positive"
            )
        self.convert_to_continuous = _as_bool(
            context.get(prefix + "convert_to_continuous", True)
        )
        self.observations: dict[str, list[tuple[date, float]]] = {}
        self.observation_dates: dict[str, list[date]] = {}
        self._selection_cache: dict[
            date, Mapping[str, tuple[date, float]]
        ] = {}
        self._prepared_snapshot_cache: dict[
            tuple[datetime, tuple[datetime, ...]], PreparedFredSnapshot
        ] = {}
        self.last_refresh_local_date: Optional[date] = None
        self.refresh_failures = 0
        self._next_refresh_retry_monotonic = 0.0
        self._session_age_cache: dict[tuple[date, date], int] = {}
        self.load_observations()

    def close(self) -> None:
        """Release the worker-owned HTTP session, if one was constructed."""
        if not self._owns_http_session:
            return
        close = getattr(self.http_session, "close", None)
        if callable(close):
            close()
        self.http_session = None

    def _invalidate_caches(self) -> None:
        self._selection_cache.clear()
        self._prepared_snapshot_cache.clear()

    def load_observations(self) -> None:
        result = self.client.query(
            f"SELECT series_id, observation_date, value_percent, retrieved_at "
            f"FROM {self.database}.options_chain_reference_rates FINAL "
            "WHERE series_id IN {series:Array(String)} "
            "ORDER BY series_id, observation_date DESC",
            parameters={"series": self.series},
        )
        loaded: dict[str, list[tuple[date, float]]] = {}
        latest_retrieval: dict[str, datetime] = {}
        for series, observation_date, value, retrieved_at in result.result_rows:
            parsed_date = (
                observation_date
                if isinstance(observation_date, date)
                else date.fromisoformat(str(observation_date))
            )
            loaded.setdefault(str(series), []).append((parsed_date, float(value)))
            parsed_retrieval = _parse_timestamp(retrieved_at)
            if parsed_retrieval is not None:
                latest_retrieval[str(series)] = max(
                    parsed_retrieval,
                    latest_retrieval.get(str(series), parsed_retrieval),
                )
        # The current-row query is ordered descending for efficient ClickHouse
        # reads, while selection uses ascending arrays and bisect_right.
        self.observations = {
            series: sorted(observations, key=lambda item: item[0])
            for series, observations in loaded.items()
        }
        self.observation_dates = {
            series: [observation_date for observation_date, _value in observations]
            for series, observations in self.observations.items()
        }
        self._invalidate_caches()
        local_date = datetime.now().date()
        if all(
            series in latest_retrieval
            and latest_retrieval[series].astimezone().date() == local_date
            for series in self.series
        ):
            self.last_refresh_local_date = local_date
            self._next_refresh_retry_monotonic = 0.0

    def refresh(self, *, required_from_date: Optional[date] = None) -> None:
        if not self.allow_network:
            raise RuntimeError("Worker-local FRED curve is durable-read-only")
        if not self.api_key:
            raise RuntimeError(f"{OPTIONS_CALCS_FRED_PREFIX}api_key is empty")
        retrieved: list[dict[str, Any]] = []
        retrieved_at = _utc_now()
        local_date = datetime.now().date()
        history_start = local_date - timedelta(days=self.history_days)
        if required_from_date is not None:
            history_start = min(
                history_start,
                required_from_date - timedelta(days=self.max_age_days + 7),
            )
        timeout = int(
            self.context.get(OPTIONS_CALCS_FRED_PREFIX + "request_timeout", 15)
        )
        for series in self.series:
            try:
                response = self.http_session.get(
                    self.endpoint,
                    params={
                        "series_id": series,
                        "api_key": self.api_key,
                        "file_type": "json",
                        "sort_order": "asc",
                        "observation_start": history_start.isoformat(),
                        "limit": 100000,
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                # Provider exceptions can include authenticated request URLs.
                raise RuntimeError(
                    f"FRED request failed ({type(exc).__name__})"
                ) from None
            valid_count = 0
            for item in payload.get("observations", []):
                if item.get("value") == ".":
                    continue
                try:
                    observation_date = date.fromisoformat(item["date"])
                    value = float(item["value"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                retrieved.append(
                    {
                        "series_id": series,
                        "observation_date": observation_date,
                        "value_percent": value,
                        "retrieved_at": retrieved_at,
                        "source": "fred",
                    }
                )
                valid_count += 1
            if not valid_count:
                raise RuntimeError(f"FRED returned no valid observations for {series}")
        retrieved.sort(key=lambda row: (row["series_id"], row["observation_date"]))
        existing = {
            (series, observation_date): value
            for series, observations in self.observations.items()
            for observation_date, value in observations
        }
        newest_retrieved = {
            series: max(
                row["observation_date"]
                for row in retrieved
                if row["series_id"] == series
            )
            for series in self.series
        }
        # Persist revisions and missing history, plus one newest observation per
        # series so another process can see that today's refresh already ran.
        retrieved = [
            row
            for row in retrieved
            if row["observation_date"] == newest_retrieved[row["series_id"]]
            or existing.get((row["series_id"], row["observation_date"]))
            != row["value_percent"]
        ]
        token_payload = {
            "refresh_local_date": datetime.now().date().isoformat(),
            "history_start": history_start.isoformat(),
            "observations": [
                (
                    row["series_id"],
                    row["observation_date"].isoformat(),
                    row["value_percent"],
                )
                for row in retrieved
            ],
        }
        token_hash = hashlib.sha256(
            _compact_json(token_payload).encode("utf-8")
        ).hexdigest()
        try:
            self.client.insert_arrow(
                f"{self.database}.options_chain_reference_rates",
                _rows_to_arrow(retrieved, REFERENCE_RATE_SCHEMA),
                settings={"insert_deduplication_token": f"fred:{token_hash}"},
            )
        except Exception:
            self.load_observations()
            if self.last_refresh_local_date != datetime.now().date():
                raise
        self.load_observations()
        self.refresh_failures = 0
        self._next_refresh_retry_monotonic = 0.0

    def ensure_current(self) -> None:
        local_date = datetime.now().date()
        if self.last_refresh_local_date == local_date:
            return
        monotonic_now = time.monotonic()
        if monotonic_now < getattr(self, "_next_refresh_retry_monotonic", 0.0):
            if not self.observations:
                raise RuntimeError(
                    "FRED refresh retry is pending and no ClickHouse "
                    "observations are available"
                )
            return
        try:
            self.refresh()
        except Exception as exc:
            self._handle_refresh_failure(exc, monotonic_now=monotonic_now)

    def _handle_refresh_failure(
        self, exc: Exception, *, monotonic_now: float
    ) -> None:
        """Retain last-good data without suppressing retries for the day."""
        self.refresh_failures += 1
        # A failed network attempt must not mark the whole local day as
        # refreshed.  Keep using the durable last-good curve, but allow this
        # process to retry after a bounded cooldown.  This also lets a worker
        # observe a curve published later by a sibling process.
        retry_seconds = getattr(self, "refresh_retry_seconds", 60.0)
        self._next_refresh_retry_monotonic = monotonic_now + retry_seconds
        try:
            self.load_observations()
        except Exception:
            self.logger.warning(
                "Unable to reload ClickHouse reference rates", exc_info=True
            )
        if not self.observations:
            raise RuntimeError(
                "FRED retrieval failed and no ClickHouse observations are "
                f"available: {exc}"
            ) from exc
        self.logger.warning(
            "FRED retrieval failed; evaluating last-good ClickHouse "
            "observations and retrying in %.1f seconds: %s",
            retry_seconds,
            exc,
        )

    def _trading_session_age(
        self, observation_date: date, snapshot_date: date
    ) -> int:
        """Return exchange sessions after an observation through a snapshot."""
        if observation_date > snapshot_date:
            raise RuntimeError(
                "FRED observation date is after the option snapshot date: "
                f"{observation_date} > {snapshot_date}"
            )
        key = (observation_date, snapshot_date)
        cache = getattr(self, "_session_age_cache", None)
        if cache is None:
            cache = self._session_age_cache = {}
        cached = cache.get(key)
        if cached is not None:
            return cached
        age = len(
            exchange_sessions(
                self.trading_calendar,
                observation_date + timedelta(days=1),
                snapshot_date,
            )
        )
        cache[key] = age
        return age

    def _selected_observations(
        self, snapshot_date: date
    ) -> Mapping[str, tuple[date, float]]:
        cached = self._selection_cache.get(snapshot_date)
        if cached is not None:
            return cached
        selected: dict[str, tuple[date, float]] = {}
        for _, series in FRED_ANCHORS:
            observations = self.observations.get(series, [])
            dates = self.observation_dates.get(series, [])
            index = bisect.bisect_right(dates, snapshot_date) - 1
            if index < 0:
                raise RuntimeError(
                    f"No {series} observation on or before {snapshot_date}"
                )
            selected[series] = observations[index]
        frozen = MappingProxyType(selected)
        self._selection_cache[snapshot_date] = frozen
        return frozen

    def _ensure_snapshot_date_ready(self, snapshot_date: date) -> None:
        if not self.allow_network:
            if not self.observations:
                raise RuntimeError("No durable FRED observations are available")
            return
        local_date = datetime.now().date()
        if (
            snapshot_date < local_date - timedelta(days=self.history_days)
            and self.last_refresh_local_date != local_date
        ):
            monotonic_now = time.monotonic()
            if monotonic_now < getattr(
                self, "_next_refresh_retry_monotonic", 0.0
            ):
                if not self.observations:
                    raise RuntimeError(
                        "FRED refresh retry is pending and no ClickHouse "
                        "observations are available"
                    )
                return
            try:
                self.refresh(required_from_date=snapshot_date)
            except Exception as exc:
                self._handle_refresh_failure(exc, monotonic_now=monotonic_now)
        else:
            self.ensure_current()

    def _interpolate_selected(
        self,
        snapshot_date: date,
        days_to_expiry: float,
        selected: Mapping[str, tuple[date, float]],
    ) -> PreparedFredRate:
        days = min(max(float(days_to_expiry), 0.0), 365.0)
        lower_anchor, lower_series = FRED_ANCHORS[0]
        upper_anchor, upper_series = FRED_ANCHORS[-1]
        for index in range(1, len(FRED_ANCHORS)):
            if days <= FRED_ANCHORS[index][0]:
                lower_anchor, lower_series = FRED_ANCHORS[index - 1]
                upper_anchor, upper_series = FRED_ANCHORS[index]
                break
        lower_date, lower_pct = selected[lower_series]
        upper_date, upper_pct = selected[upper_series]
        stale = [
            (series, observation_date, session_age)
            for series, observation_date in (
                (lower_series, lower_date),
                (upper_series, upper_date),
            )
            if (
                session_age := self._trading_session_age(
                    observation_date, snapshot_date
                )
            ) > self.max_age_days
        ]
        if stale:
            # A request can succeed while FRED still serves an observation
            # that is too old for the current session (notably near the open
            # after a long closure).  Treat that as refreshable state rather
            # than allowing this process's successful retrieval marker to
            # suppress every later refresh for the day.
            if self.allow_network:
                monotonic_now = time.monotonic()
                if monotonic_now >= getattr(
                    self, "_next_refresh_retry_monotonic", 0.0
                ):
                    retry_seconds = getattr(self, "refresh_retry_seconds", 60.0)
                    self.last_refresh_local_date = None
                    self._next_refresh_retry_monotonic = (
                        monotonic_now + retry_seconds
                    )
            detail = ", ".join(
                f"{series}={observation_date} ({session_age} "
                f"{self.trading_calendar} sessions; max={self.max_age_days})"
                for series, observation_date, session_age in stale
            )
            raise RuntimeError(f"FRED curve anchors are stale: {detail}")
        weight = (
            0.0
            if upper_anchor == lower_anchor
            else (days - lower_anchor) / (upper_anchor - lower_anchor)
        )
        annual_decimal = (lower_pct + weight * (upper_pct - lower_pct)) / 100.0
        rate = (
            math.log1p(annual_decimal) if self.convert_to_continuous else annual_decimal
        )
        return PreparedFredRate(
            rate,
            min(lower_date, upper_date),
            lower_series,
            upper_series,
        )

    def prepare_snapshot(
        self, base_rows: Sequence[Mapping[str, Any]]
    ) -> PreparedFredSnapshot:
        """Resolve refresh, anchors, and rates once for one immutable snapshot."""
        if not base_rows:
            raise RuntimeError("Cannot validate FRED readiness for an empty snapshot")
        collected = _parse_timestamp(base_rows[0].get("collected_at"))
        if collected is None:
            raise RuntimeError("Snapshot has no valid collection timestamp")
        expirations: set[datetime] = set()
        for row in base_rows:
            expiration = _parse_timestamp(row.get("expiration_at"))
            if expiration is not None and (expiration - collected).total_seconds() > 0:
                expirations.add(expiration)
        return self.prepare_snapshot_metadata(collected, tuple(sorted(expirations)))

    def prepare_snapshot_metadata(
        self, collected_at: datetime, expirations: Sequence[datetime]
    ) -> PreparedFredSnapshot:
        """Prepare a detached worker payload without loading base contract rows."""
        collected = _parse_timestamp(collected_at)
        if collected is None:
            raise RuntimeError("Snapshot has no valid collection timestamp")
        snapshot_date = collected.date()
        ordered_expirations = tuple(
            sorted(
                {
                    expiration
                    for value in expirations
                    if (expiration := _parse_timestamp(value)) is not None
                    and (expiration - collected).total_seconds() > 0.0
                }
            )
        )
        if not ordered_expirations:
            return PreparedFredSnapshot(snapshot_date=snapshot_date, by_expiration={})
        # Collection time, not only its date, is part of the tenor used for
        # interpolation.  Keep same-day snapshots with identical expiration
        # sets isolated in the prepared cache.
        cache_key = (collected, ordered_expirations)
        cached = self._prepared_snapshot_cache.get(cache_key)
        if cached is not None:
            return cached

        # FRED refresh and observation selection are deliberately snapshot
        # scoped; contract arithmetic consumes only this immutable lookup.
        self._ensure_snapshot_date_ready(snapshot_date)
        selected = self._selected_observations(snapshot_date)
        prepared = PreparedFredSnapshot(
            snapshot_date=snapshot_date,
            by_expiration={
                expiration: self._interpolate_selected(
                    snapshot_date,
                    (expiration - collected).total_seconds() / 86400.0,
                    selected,
                )
                for expiration in ordered_expirations
            },
        )
        self._prepared_snapshot_cache[cache_key] = prepared
        return prepared

    def ensure_snapshot_ready(self, base_rows: Sequence[Mapping[str, Any]]) -> None:
        """Require durable rates for every tenor before a batch can be published."""
        self.prepare_snapshot(base_rows)

    def interpolate(
        self, snapshot_date: date, days_to_expiry: float
    ) -> tuple[float, date, str, str]:
        self.ensure_current()
        selected = self._selected_observations(snapshot_date)
        prepared = self._interpolate_selected(
            snapshot_date, days_to_expiry, selected
        )
        return (
            prepared.rate,
            prepared.observation_date,
            prepared.lower_series,
            prepared.upper_series,
        )


def _base_select_columns(
    missing_optional: Collection[str] = (),
) -> list[str]:
    """Project normalized source tables into the numerical engine row shape."""
    del missing_optional
    return calculation_source_projection_columns(numerical=True)


def _normalized_source_from_clause(database: str) -> str:
    return (
        f"{database}.options_chain_snapshots AS s "
        f"INNER JOIN {database}.options_chain_observations AS o "
        "ON o.snapshot_id = s.snapshot_id "
        "AND o.source_collected_at = s.source_collected_at "
        f"INNER JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c "
        "ON c.contract_id = o.contract_id"
    )


def _base_rows_for_snapshot(
    client: Any, database: str, snapshot_id: uuid.UUID
) -> list[dict[str, Any]]:
    columns = _base_select_columns()
    table = client.query_arrow(
        f"SELECT {', '.join(columns)} FROM {_normalized_source_from_clause(database)} "
        "WHERE s.snapshot_id = {snapshot_id:UUID} ORDER BY o.contract_ordinal, o.contract_id",
        parameters={"snapshot_id": str(snapshot_id)},
    )
    rows = table.to_pylist()
    return rows


BULK_OWNER_QUERY_CHUNK_SIZE = 256
BULK_SYMBOL_QUERY_CHUNK_SIZE = 10_000


def _symbol_external_data(symbols: Sequence[str]) -> Any:
    """Carry a large symbol set in the HTTP body, not URL form fields."""
    from clickhouse_connect.driver.external import ExternalData

    payload = b"".join(
        (_compact_json({"option_symbol": symbol}) + "\n").encode("utf-8")
        for symbol in symbols
    )
    return ExternalData(
        data=payload,
        file_name="prior_symbols.jsonl",
        fmt="JSONEachRow",
        structure="option_symbol String",
    )


def _base_rows_for_snapshots(
    client: Any, database: str, snapshot_ids: Sequence[uuid.UUID]
) -> Mapping[uuid.UUID, list[dict[str, Any]]]:
    """Load wide base rows for several owners with explicit detached grouping."""
    ordered_ids = tuple(dict.fromkeys(uuid.UUID(str(value)) for value in snapshot_ids))
    grouped: dict[uuid.UUID, list[dict[str, Any]]] = {value: [] for value in ordered_ids}
    columns = _base_select_columns()
    for start in range(0, len(ordered_ids), BULK_OWNER_QUERY_CHUNK_SIZE):
        chunk = ordered_ids[start : start + BULK_OWNER_QUERY_CHUNK_SIZE]
        parameters = {"snapshot_ids": [str(value) for value in chunk]}
        table = client.query_arrow(
            f"SELECT {', '.join(columns)} FROM {_normalized_source_from_clause(database)} "
            "WHERE s.snapshot_id IN {snapshot_ids:Array(UUID)} "
            "ORDER BY s.snapshot_id, o.contract_ordinal, o.contract_id",
            parameters=parameters,
        )
        for row in table.to_pylist():
            owner = uuid.UUID(str(row["snapshot_id"]))
            if owner not in grouped:
                raise InconsistentSnapshotError(
                    f"Bulk base query returned unexpected snapshot {owner}"
                )
            grouped[owner].append(row)
    return grouped


def _row_spot(row: Mapping[str, Any]) -> tuple[Optional[float], str]:
    bid = _finite_float(row.get("underlying_bid"))
    ask = _finite_float(row.get("underlying_ask"))
    if bid is not None and ask is not None and bid >= 0 and ask >= bid and ask > 0:
        return (bid + ask) / 2.0, "underlying_mid"
    mark = _finite_float(row.get("underlying_mark"))
    if mark is not None and mark > 0:
        return mark, "underlying_mark"
    chain = _finite_float(row.get("chain_underlying_price"))
    if chain is not None and chain > 0:
        return chain, "chain_underlying"
    last = _finite_float(row.get("underlying_last"))
    if last is not None and last > 0:
        return last, "underlying_last"
    return None, "unavailable"


def _age_seconds(
    collected: Optional[datetime], event: Optional[datetime]
) -> Optional[float]:
    collected = _parse_timestamp(collected)
    event = _parse_timestamp(event)
    if collected is None or event is None:
        return None
    return max(0.0, (collected - event).total_seconds())


def _source_greek_missing(row: Mapping[str, Any]) -> bool:
    return any(
        _finite_float(row.get(name)) is None
        for name in (
            "source_delta",
            "source_gamma",
            "source_theta",
            "source_vega",
            "source_rho",
        )
    )


@dataclass(frozen=True)
class _ContractSnapshotConstants:
    snapshot_id: str
    collected_at: datetime
    underlying_symbol: str
    spot: Optional[float]
    spot_source: str
    underlying_quote_age_seconds: Optional[float]
    dividend_yield: float
    common_quality_flags: int


@dataclass(frozen=True)
class _ContractExpirationConstants:
    expiration_at: Optional[datetime]
    tenor_seconds: Optional[float]
    tenor_years: Optional[float]
    rate: Optional[float]
    observation_date: Optional[date]
    lower_series: str
    upper_series: str
    forward_price: Optional[float]


def _fallback_prepared_fred_snapshot(
    base_rows: Sequence[Mapping[str, Any]], fred_curve: Any
) -> PreparedFredSnapshot:
    """Prepare alternate/test curves without requiring the new curve API."""
    collected = _parse_timestamp(base_rows[0].get("collected_at"))
    if collected is None:
        raise RuntimeError("Snapshot has no valid collection timestamp")
    ensure_ready = getattr(fred_curve, "ensure_snapshot_ready", None)
    if callable(ensure_ready):
        ensure_ready(base_rows)
    else:
        fred_curve.ensure_current()
    expirations = sorted(
        {
            expiration
            for row in base_rows
            if (expiration := _parse_timestamp(row.get("expiration_at"))) is not None
            and (expiration - collected).total_seconds() > 0
        }
    )
    if not expirations:
        raise RuntimeError("Snapshot has no positive expiration tenors")
    rates: dict[datetime, PreparedFredRate] = {}
    for expiration in expirations:
        rate, observation_date, lower_series, upper_series = fred_curve.interpolate(
            collected.date(), (expiration - collected).total_seconds() / 86400.0
        )
        rates[expiration] = PreparedFredRate(
            float(rate), observation_date, str(lower_series), str(upper_series)
        )
    return PreparedFredSnapshot(collected.date(), dict(rates))


def _prepare_contract_snapshot_constants(
    context: Mapping[str, Any],
    base_rows: Sequence[Mapping[str, Any]],
    raw_diagnostics: Mapping[str, Any],
) -> _ContractSnapshotConstants:
    if not base_rows:
        raise SnapshotDataError("Cannot calculate an empty base snapshot")
    first = base_rows[0]
    collected = _parse_timestamp(first.get("collected_at"))
    if collected is None:
        raise SnapshotDataError("Snapshot has no valid collection timestamp")
    snapshot_id = str(first.get("snapshot_id"))
    underlying_symbol = str(first.get("underlying_symbol") or "")
    spot, spot_source = _row_spot(first)
    underlying_quote_time = _parse_timestamp(first.get("underlying_quote_time"))
    chain_delayed = _as_bool(first.get("chain_is_delayed"))
    chain_truncated = _as_bool(first.get("chain_is_truncated"))
    first_dividend = _finite_float(first.get("chain_dividend_yield"))
    for row in base_rows[1:]:
        row_collected = _parse_timestamp(row.get("collected_at"))
        row_spot = _row_spot(row)
        row_dividend = _finite_float(row.get("chain_dividend_yield"))
        if (
            str(row.get("snapshot_id")) != snapshot_id
            or row_collected != collected
            or str(row.get("underlying_symbol") or "") != underlying_symbol
            or _parse_timestamp(row.get("underlying_quote_time"))
            != underlying_quote_time
            or _as_bool(row.get("chain_is_delayed")) != chain_delayed
            or _as_bool(row.get("chain_is_truncated")) != chain_truncated
            or row_spot != (spot, spot_source)
            or row_dividend != first_dividend
        ):
            raise InconsistentSnapshotError(
                "Base snapshot mixes snapshot-level timestamps, underlying data, or flags"
            )
    prefix = OPTIONS_CALCS_CONFIG_PREFIX
    dividend_yield = first_dividend
    if dividend_yield is None:
        dividend_yield = _finite_float(
            context.get(prefix + f"dividend_yield.{underlying_symbol}")
        )
    if dividend_yield is None:
        dividend_yield = float(context.get(prefix + "dividend_yield_default", 0.0))
    common_flags = 0
    if chain_delayed:
        common_flags |= 1
    if _as_bool(raw_diagnostics.get("underlying_delayed", False)):
        common_flags |= 2
    if chain_truncated:
        common_flags |= 131072
    return _ContractSnapshotConstants(
        snapshot_id,
        collected,
        underlying_symbol,
        spot,
        spot_source,
        _age_seconds(collected, underlying_quote_time),
        float(dividend_yield),
        common_flags,
    )


def calculate_rows(
    context: Mapping[str, Any],
    base_rows: list[dict[str, Any]],
    fred_curve: FredCurve,
    *,
    calculated_at: Optional[datetime] = None,
    raw_diagnostics: Optional[Mapping[str, Any]] = None,
    prepared_fred: Optional[PreparedFredSnapshot] = None,
    timings: Optional[StageTimings] = None,
    recalc_diagnostics: Optional[dict[str, int]] = None,
    solver_registry: Optional[Any] = None,
    prior_outcome_loader: Optional[Any] = None,
    defer_repricing: bool = False,
) -> list[dict[str, Any]]:
    timings = timings or StageTimings()
    timings.initialize(CONTRACT_TIMING_NAMES)
    recalc_diagnostics = (
        recalc_diagnostics if recalc_diagnostics is not None else Counter()
    )
    calculated_at = calculated_at or _utc_now()
    prefix = OPTIONS_CALCS_CONFIG_PREFIX
    version = int(context.get(prefix + "calculation_definition_id", 1))
    quote_max_age = float(context.get(prefix + "quote_max_age_seconds", 60))
    max_spread = float(context.get(prefix + "max_spread_fraction_of_mid", 0.50))
    expected_multiplier = float(context.get(prefix + "expected_multiplier", 100))
    recalculate = _as_bool(
        context.get(OPTIONS_CALCS_GREEKS_PREFIX + "enabled", False)
    )
    raw_diagnostics = raw_diagnostics or {}
    mismatch_symbols = set(raw_diagnostics.get("map_mismatch_symbols", ()))
    results: list[dict[str, Any]] = []
    if prepared_fred is None:
        with timings.measure("contract.fred_prepare_seconds"):
            prepare = getattr(fred_curve, "prepare_snapshot", None)
            prepared_fred = (
                prepare(base_rows)
                if callable(prepare)
                else _fallback_prepared_fred_snapshot(base_rows, fred_curve)
            )

    with timings.measure("contract.row_calculation_seconds"):
        snapshot = _prepare_contract_snapshot_constants(
            context, base_rows, raw_diagnostics
        )
        expiration_constants: dict[
            Optional[datetime], _ContractExpirationConstants
        ] = {}
        for base in base_rows:
            expiration = _parse_timestamp(base.get("expiration_at"))
            if expiration in expiration_constants:
                continue
            tenor_seconds = (
                None
                if expiration is None
                else solver_tenor_seconds(expiration, snapshot.collected_at)
            )
            tenor_years = (
                None
                if tenor_seconds is None
                else tenor_seconds / (365.25 * 86400.0)
            )
            fred_rate = (
                prepared_fred.by_expiration.get(expiration)
                if expiration is not None and tenor_seconds is not None and tenor_seconds > 0
                else None
            )
            forward = (
                snapshot.spot
                * math.exp(
                    (fred_rate.rate - snapshot.dividend_yield) * tenor_years
                )
                if snapshot.spot is not None
                and fred_rate is not None
                and tenor_years is not None
                else None
            )
            expiration_constants[expiration] = _ContractExpirationConstants(
                expiration,
                tenor_seconds,
                tenor_years,
                fred_rate.rate if fred_rate is not None else None,
                fred_rate.observation_date if fred_rate is not None else None,
                fred_rate.lower_series if fred_rate is not None else "",
                fred_rate.upper_series if fred_rate is not None else "",
                forward,
            )

        for base in base_rows:
            flags = snapshot.common_quality_flags
            errors: list[str] = []
            if _as_bool(base.get("is_nonstandard")):
                flags |= 1024
            multiplier = _finite_float(base.get("multiplier"))
            if multiplier is None or not math.isclose(multiplier, expected_multiplier):
                flags |= 2048
            if base.get("option_symbol") in mismatch_symbols:
                flags |= 262144

            bid = _finite_float(base.get("bid"))
            ask = _finite_float(base.get("ask"))
            mid = spread = spread_fraction = None
            if bid is None:
                flags |= 4
            if ask is None:
                flags |= 8
            if bid is not None and ask is not None:
                if ask < bid:
                    flags |= 16
                else:
                    mid = (bid + ask) / 2.0
                    spread = ask - bid
                    if mid <= 0:
                        flags |= 32
                    else:
                        spread_fraction = spread / mid
                        if spread_fraction > max_spread:
                            flags |= 8192

            quote_age = _age_seconds(snapshot.collected_at, base.get("quote_time"))
            if quote_age is not None and quote_age > quote_max_age:
                flags |= 64
            if (
                snapshot.underlying_quote_age_seconds is not None
                and snapshot.underlying_quote_age_seconds > quote_max_age
            ):
                flags |= 128
            if snapshot.spot is None:
                flags |= 16384
                errors.append("no_valid_spot")

            expiration = _parse_timestamp(base.get("expiration_at"))
            expiration_data = expiration_constants[expiration]
            if (
                expiration_data.tenor_seconds is None
                or expiration_data.tenor_seconds <= 0
            ):
                flags |= 4096
                errors.append("invalid_expiration")
            elif expiration_data.rate is None:
                flags |= 32768
                errors.append("fred:prepared rate unavailable")
            strike = _finite_float(base.get("strike_price"))
            source_iv = _finite_float(base.get("source_iv"))
            if source_iv is None or source_iv <= 0:
                flags |= 256
            if _source_greek_missing(base):
                flags |= 512

            strike_distance = relative_strike = log_moneyness = None
            log_forward_moneyness = None
            if (
                snapshot.spot is not None
                and strike is not None
                and snapshot.spot > 0
                and strike > 0
            ):
                strike_distance = strike - snapshot.spot
                relative_strike = strike / snapshot.spot
                log_moneyness = math.log(relative_strike)
                if expiration_data.forward_price is not None and expiration_data.forward_price > 0:
                    log_forward_moneyness = math.log(
                        strike / expiration_data.forward_price
                    )
            results.append(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "source_collected_at": snapshot.collected_at,
                    "option_symbol": str(base["option_symbol"]),
                    "calculation_definition_id": version,
                    "calculated_at": calculated_at,
                    "calculation_status": "failed" if errors else "complete",
                    "calculation_error": "; ".join(errors),
                    "quality_flags": flags,
                    "spot_price": snapshot.spot,
                    "spot_source": snapshot.spot_source,
                    "mid_price": mid,
                    "bid_ask_spread": spread,
                    "spread_fraction_of_mid": spread_fraction,
                    "quote_age_seconds": quote_age,
                    "underlying_quote_age_seconds": snapshot.underlying_quote_age_seconds,
                    "strike_distance": strike_distance,
                    "relative_strike": relative_strike,
                    "log_moneyness": log_moneyness,
                    "forward_price": expiration_data.forward_price,
                    "log_forward_moneyness": log_forward_moneyness,
                    "time_to_expiry_seconds": expiration_data.tenor_seconds,
                    "time_to_expiry_years": expiration_data.tenor_years,
                    "recalc_total_implied_variance": None,
                    "risk_free_rate": expiration_data.rate,
                    "dividend_yield": snapshot.dividend_yield,
                    "fred_observation_date": expiration_data.observation_date,
                    "fred_lower_series": expiration_data.lower_series,
                    "fred_upper_series": expiration_data.upper_series,
                    "recalc_status": "skipped" if recalculate else "disabled",
                    "recalc_reason": (
                        "source_only_european"
                        if recalculate
                        and str(base.get("exercise_type") or "").upper() == "E"
                        else (
                            "not_selected"
                            if recalculate
                            else "disabled_by_configuration"
                        )
                    ),
                    "recalc_failure_stage": "none",
                    "recalc_method": "none",
                    "recalc_method_selection_reason": "",
                    "recalc_algorithm_version": "",
                    "recalc_solver_config_hash": "0" * 64,
                    "recalc_solver_parameters_json": "{}",
                    "recalc_input_basis": "market_mid",
                    "recalc_target_price": None,
                    "recalc_input_iv": None,
                    "recalc_iv": None,
                    "recalc_iv_iterations": 0,
                    "recalc_price_residual": None,
                    "recalc_model_price": None,
                    "recalc_delta": None,
                    "recalc_gamma": None,
                    "recalc_theta_per_day": None,
                    "recalc_vega_per_1pct": None,
                    "recalc_rho_per_1pct": None,
                    "recalc_requested_dtype": "",
                    "recalc_actual_dtype": "",
                    "recalc_precision_escalated": False,
                    "recalc_precision_escalation_reason": "none",
                    "recalc_quality_flags": 0,
                    "recalc_time_steps": None,
                    "recalc_space_steps": None,
                    "recalc_damping_steps": None,
                    "recalc_dividend_model": str(
                        context.get(
                            OPTIONS_CALCS_GREEKS_PREFIX + "dividend_model",
                            "continuous_yield",
                        )
                    ),
                    "recalc_result_origin": "none",
                    "recalc_reuse_mode": "none",
                    "recalc_origin_snapshot_id": None,
                    "recalc_origin_calculated_at": None,
                    "recalc_valuation_at": None,
                    "recalc_valuation_spot": None,
                    "recalc_valuation_target_price": None,
                    "recalc_valuation_time_to_expiry_seconds": None,
                    "recalc_valuation_rate": None,
                    "recalc_valuation_dividend_yield": None,
                    "recalc_reuse_age_seconds": None,
                    "recalc_reuse_spot_move_bps": None,
                    "recalc_reuse_elapsed_tenor_fraction": None,
                    "recalc_reuse_estimated_price_drift": None,
                    "recalc_reuse_estimated_iv_change_points": None,
                    "recalc_shadow_checked": False,
                    "recalc_shadow_status": "none",
                    "recalc_shadow_iv_error": None,
                    "recalc_shadow_delta_error": None,
                    "recalc_iv_seed_origin": "none",
                    "recalc_iv_seed_snapshot_id": None,
                    "recalc_iv_seed_calculated_at": None,
                    "recalc_iv_seed_value": None,
                    "recalc_iv_seed_age_seconds": None,
                    "recalc_iv_seed_hint_evaluations": 0,
                    "recalc_iv_seed_exact_match": False,
                    "recalc_iv_seed_local_bracket_found": False,
                    "recalc_iv_seed_global_fallback_used": False,
                    "recalc_iv_identifiability_status": "not_checked",
                    "recalc_iv_identifiability_price_span": None,
                    "recalc_iv_identifiability_slope": None,
                    "recalc_iv_hint_flat_rejection_count": 0,
                    "recalc_retry_class": "none",
                    "recalc_retry_after": None,
                    "recalc_failure_fingerprint": "0" * 64,
                    "recalc_attempt_suppressed": False,
                    "_base": base,
                }
            )

    if recalculate and not defer_repricing:
        with timings.measure("contract.recalc_seconds"):
            selected_for_repricing = sorted(
                _select_repricing_indices(context, results)
            )
            if not selected_for_repricing:
                return results
            _apply_repricing(
                context,
                results,
                recalc_diagnostics,
                solver_registry=solver_registry,
                timings=timings,
                prior_outcome_loader=prior_outcome_loader,
                preselected_indices=selected_for_repricing,
            )
    return results


def _select_repricing_indices(
    context: Mapping[str, Any], rows: list[dict[str, Any]]
) -> set[int]:
    """Validate final market inputs, route methods, and apply the strike limit."""
    from algotrading.refdata.options.Pricing import configured_methods, select_method

    prefix = OPTIONS_CALCS_GREEKS_PREFIX
    routing = configured_methods(context)
    limit = max(
        0, int(context.get(prefix + "num_contracts_per_expiration", 20))
    )
    quote_max_age = float(
        context.get(OPTIONS_CALCS_CONFIG_PREFIX + "quote_max_age_seconds", 60)
    )
    maximum_spread = float(
        context.get(
            OPTIONS_CALCS_CONFIG_PREFIX + "max_spread_fraction_of_mid", 0.50
        )
    )
    minimum_tenor = float(
        context.get(OPTIONS_CALCS_SURFACE_PREFIX + "minimum_tenor_seconds", 60)
    )
    grouped: dict[datetime, list[int]] = {}
    for index, row in enumerate(rows):
        base = row["_base"]
        exercise = str(base.get("exercise_type") or "").upper()
        tenor_seconds = _finite_float(row.get("time_to_expiry_seconds"))
        if exercise == "E":
            row.update(
                recalc_status="skipped",
                recalc_reason="source_only_european",
                recalc_method="none",
                recalc_method_selection_reason="source_only_european",
            )
            continue
        if exercise != "A":
            row.update(
                recalc_status="skipped",
                recalc_reason="unsupported_exercise_style",
                recalc_method="none",
            )
            continue
        if tenor_seconds is None or tenor_seconds <= 0:
            row.update(
                recalc_status="skipped",
                recalc_reason="expired_or_invalid_tenor",
                recalc_method="none",
            )
            continue
        selection = select_method(
            exercise,
            tenor_seconds,
            routing=routing,
        )
        row["recalc_method"] = selection.method.value
        row["recalc_method_selection_reason"] = selection.reason
        if tenor_seconds < minimum_tenor:
            row["recalc_reason"] = "below_minimum_tenor"
            continue
        spot = _finite_float(row.get("spot_price"))
        strike = _finite_float(base.get("strike_price"))
        rate = _finite_float(row.get("risk_free_rate"))
        dividend = _finite_float(row.get("dividend_yield"))
        if (
            spot is None
            or spot <= 0
            or strike is None
            or strike <= 0
            or rate is None
            or dividend is None
        ):
            row["recalc_reason"] = "missing_repricing_input"
            continue
        bid = _finite_float(base.get("bid"))
        ask = _finite_float(base.get("ask"))
        if bid is None or ask is None:
            row["recalc_reason"] = "missing_market_price"
            continue
        if ask < bid:
            row["recalc_reason"] = "invalid_bid_ask"
            continue
        target = (bid + ask) / 2.0
        if not math.isfinite(target) or target <= 0:
            row["recalc_reason"] = "missing_market_price"
            continue
        quote_age = _finite_float(row.get("quote_age_seconds"))
        if quote_age is not None and quote_age > quote_max_age:
            row["recalc_reason"] = "stale_quote"
            continue
        spread_fraction = _finite_float(row.get("spread_fraction_of_mid"))
        if spread_fraction is not None and spread_fraction > maximum_spread:
            row["recalc_reason"] = "spread_too_wide"
            continue
        intrinsic = (
            max(spot - strike, 0.0)
            if str(base.get("put_call")).upper() == "CALL"
            else max(strike - spot, 0.0)
        )
        upper = spot if str(base.get("put_call")).upper() == "CALL" else strike
        tolerance = max(1.0e-10, upper * 1.0e-12)
        if target < intrinsic - tolerance or target > upper + tolerance:
            row["recalc_reason"] = "target_price_outside_bounds"
            continue
        row["recalc_target_price"] = target
        grouped.setdefault(_parse_timestamp(base["expiration_at"]), []).append(index)

    selected: set[int] = set()
    for indices in grouped.values():
        by_strike: dict[float, list[int]] = {}
        for index in indices:
            by_strike.setdefault(float(rows[index]["_base"]["strike_price"]), []).append(index)
        ranked = sorted(
            by_strike.values(),
            key=lambda members: min(
                abs(rows[index]["log_forward_moneyness"] or 0.0) for index in members
            ),
        )
        used = 0
        for members in ranked:
            calls = [
                index for index in members
                if rows[index]["_base"]["put_call"] == "CALL"
            ]
            puts = [
                index for index in members
                if rows[index]["_base"]["put_call"] == "PUT"
            ]
            ordered = calls[:1] + puts[:1] if calls and puts else members[:1]
            if used + len(ordered) > limit:
                continue
            selected.update(ordered)
            used += len(ordered)
            if used >= limit:
                break
        for index in indices:
            if index not in selected and rows[index]["recalc_reason"] == "not_selected":
                rows[index]["recalc_reason"] = "outside_contract_selection_limit"
    return selected


@dataclass(frozen=True)
class _RepricingProvenance:
    algorithm_version: str
    solver_config_hash: str
    solver_parameters_json: str
    time_steps: Optional[int]
    space_steps: Optional[int]
    damping_steps: Optional[int]


def _apply_repricing(
    context: Mapping[str, Any],
    rows: list[dict[str, Any]],
    diagnostics: Optional[dict[str, int]] = None,
    *,
    solver_registry: Optional[Any] = None,
    timings: Optional[StageTimings] = None,
    prior_outcome_loader: Optional[Any] = None,
    preselected_indices: Optional[Sequence[int]] = None,
    record_circuit_metrics: bool = True,
) -> None:
    from algotrading.refdata.options.solvers.Registry import SolverRegistry
    from algotrading.refdata.options.solvers.Types import (
        IvHint,
        PricingRequest,
        RecalcRetryClass,
        RepricingMethod,
        SolverBatchJob,
        SolverCircuitOpenError,
        SolverInfrastructureError,
        SolverOperation,
        classify_repricing_failure,
    )
    from algotrading.refdata.options.Reuse import (
        CurrentRepricingProblem,
        PriorOutcomeCandidates,
        PriorOutcomeLookupPolicy,
        REUSE_INVALIDATION_DIAGNOSTIC,
        RepricingAction,
        decide_repricing_action,
        deterministic_seed_shadow_sample,
        deterministic_shadow_sample,
        failure_fingerprint,
        parse_failure_reuse_policy,
        parse_iv_seed_policy,
        parse_repricing_reuse_policy,
        retry_after_for_failure,
    )

    diagnostics = diagnostics if diagnostics is not None else Counter()
    for name in (
        "recalc_attempted_count",
        "recalc_complete_count",
        "recalc_recoverable_failure_count",
        "recalc_unrecoverable_failure_count",
        "recalc_batch_exception_count",
        "recalc_infrastructure_failure_count",
        "recalc_selected_count",
        "recalc_solver_attempted_count",
        "recalc_solver_attempted_prior_seed",
        "recalc_solver_attempted_normal",
        "recalc_solver_attempted_retry",
        "recalc_solver_attempted_shadow",
        "recalc_solver_attempted_mismatch_fallback",
        "recalc_successes_calculated",
        "recalc_successes_reused",
        "recalc_shadow_attempted",
        "recalc_shadow_complete",
        "recalc_shadow_failed",
        "recalc_reuse_miss_no_prior",
        "recalc_shadow_iv_threshold_breaches",
        "recalc_shadow_delta_threshold_breaches",
        "recalc_action_reuse_success",
        "recalc_action_solve_prior_iv_seed",
        "recalc_action_reuse_failure_backoff",
        "recalc_action_solve_normal",
        "recalc_action_suppress_circuit_open",
        "recalc_prior_iv_seed_candidates",
        "recalc_prior_iv_seed_used",
        "recalc_prior_partial_failure_seed_used",
        "recalc_source_iv_seed_used",
        "recalc_seed_exact_matches",
        "recalc_seed_local_brackets",
        "recalc_seed_global_fallbacks",
        "recalc_seed_price_evaluations",
        "recalc_seed_baseline_attempted",
        "recalc_seed_baseline_completed",
        "recalc_seed_baseline_evaluations_saved",
        "recalc_seed_baseline_evaluations_increased",
        "recalc_seed_baseline_iv_mismatches",
        "recalc_failures_reused",
        "recalc_failure_retry_due",
        "recalc_failure_retry_attempted",
        "recalc_failure_retry_succeeded",
        "recalc_failure_retry_failed",
        "recalc_circuit_suppressed_calls",
    ):
        diagnostics.setdefault(name, 0)
    selected = (
        sorted(int(index) for index in preselected_indices)
        if preselected_indices is not None
        else sorted(_select_repricing_indices(context, rows))
    )
    diagnostics["recalc_attempted_count"] += len(selected)
    diagnostics["recalc_selected_count"] += len(selected)
    if not selected:
        return
    registry = solver_registry or SolverRegistry(context)
    snapshot_circuits = getattr(registry, "circuit_snapshots", None)
    circuit_before = (
        snapshot_circuits()
        if record_circuit_metrics and callable(snapshot_circuits)
        else {}
    )
    precision_policy = NumericPrecisionPolicy.from_context(context)
    reuse_policy = parse_repricing_reuse_policy(context)
    seed_policy = parse_iv_seed_policy(context)
    failure_policy = parse_failure_reuse_policy(context)
    provenance_by_method: dict[tuple[RepricingMethod, str], _RepricingProvenance] = {}

    def accumulate_performance(
        method: RepricingMethod, performance: Mapping[str, float | int]
    ) -> None:
        method_name = method.value
        for name in (
            "session_init_seconds",
            "iv_seconds",
            "greeks_seconds",
            "total_seconds",
        ):
            if timings is not None:
                timing_name = f"contract.recalc.{method_name}.{name}"
                timings.values[timing_name] = timings.values.get(timing_name, 0.0) + float(
                    performance.get(name, 0.0)
                )
        names = ["iv_price_evaluations"]
        if method is RepricingMethod.CRANK_NICOLSON:
            names.extend(
                (
                    "fdm_valuations_total",
                    "fdm_valuations_iv",
                    "fdm_valuations_greeks",
                )
            )
        for name in names:
            diagnostics[f"recalc_{method_name}_{name}"] = diagnostics.get(
                f"recalc_{method_name}_{name}", 0
            ) + int(performance.get(name, 0))

    def categorize_permitted_attempt(
        action: RepricingAction,
        *,
        shadow: bool,
        retry_due: bool,
        mismatch_fallback: bool = False,
    ) -> None:
        if mismatch_fallback:
            category = "mismatch_fallback"
        elif shadow:
            category = "shadow"
        elif retry_due:
            category = "retry"
        elif action is RepricingAction.SOLVE_PRIOR_IV_SEED:
            category = "prior_seed"
        else:
            category = "normal"
        key = f"recalc_solver_attempted_{category}"
        diagnostics[key] = diagnostics.get(key, 0) + 1

    current_by_index: dict[int, CurrentRepricingProblem] = {}
    request_by_index: dict[int, PricingRequest] = {}
    for index in selected:
        row = rows[index]
        base = row["_base"]
        method = RepricingMethod.parse(row["recalc_method"])
        requested_dtype = resolve_numeric_dtype(
            float(row["time_to_expiry_seconds"]),
            precision_policy.short_tenor_max_seconds,
            precision_policy.short_tenor_dtype,
            precision_policy.other_tenor_dtype,
        ).name
        row["recalc_requested_dtype"] = requested_dtype
        row["recalc_actual_dtype"] = requested_dtype
        try:
            provenance_key = (method, requested_dtype)
            provenance = provenance_by_method.get(provenance_key)
            if provenance is None:
                describe = getattr(registry, "describe", None)
                if callable(describe):
                    algorithm_version, parameters = describe(method)
                else:
                    adapter = registry.get(method)
                    algorithm_version, parameters = (
                        adapter.algorithm_version,
                        adapter.parameters,
                    )
                provenance = _RepricingProvenance(
                    algorithm_version=algorithm_version,
                    solver_config_hash=_solver_configuration_hash(
                        context, method.value, requested_dtype
                    ),
                    solver_parameters_json=_compact_json(parameters),
                    time_steps=(
                        int(parameters["steps"])
                        if "steps" in parameters
                        else int(parameters["time_grid"])
                    ),
                    space_steps=parameters.get("space_grid"),
                    damping_steps=parameters.get("damping_steps"),
                )
                provenance_by_method[provenance_key] = provenance
            row.update(
                recalc_algorithm_version=provenance.algorithm_version,
                recalc_solver_config_hash=provenance.solver_config_hash,
                recalc_solver_parameters_json=provenance.solver_parameters_json,
                recalc_time_steps=provenance.time_steps,
                recalc_space_steps=provenance.space_steps,
                recalc_damping_steps=provenance.damping_steps,
            )
            request = PricingRequest(
                option_symbol=str(base["option_symbol"]),
                snapshot_id=str(row["snapshot_id"]),
                put_call=str(base["put_call"]),
                exercise_type="A",
                spot=float(row["spot_price"]),
                strike=float(base["strike_price"]),
                tenor_seconds=float(row["time_to_expiry_seconds"]),
                rate=float(row["risk_free_rate"]),
                dividend_yield=float(row["dividend_yield"]),
                bid=_finite_float(base.get("bid")),
                ask=_finite_float(base.get("ask")),
                mark=_finite_float(base.get("mark")),
                source_iv=_finite_float(base.get("source_iv")),
                source_collected_at=row["source_collected_at"],
                expiration_at=_parse_timestamp(base.get("expiration_at")),
                requested_dtype=requested_dtype,
            )
            request_by_index[index] = request
            current_by_index[index] = CurrentRepricingProblem(
                option_symbol=request.option_symbol,
                snapshot_id=str(row["snapshot_id"]),
                source_collected_at=_parse_timestamp(row["source_collected_at"]),
                calculation_definition_id=int(row["calculation_definition_id"]),
                spot=request.spot,
                target_price=float(row["recalc_target_price"]),
                time_to_expiry_seconds=request.tenor_seconds,
                rate=request.rate,
                dividend_yield=request.dividend_yield,
                method=method.value,
                algorithm_version=provenance.algorithm_version,
                solver_config_hash=provenance.solver_config_hash,
                input_basis=str(row["recalc_input_basis"]),
                dividend_model=str(row["recalc_dividend_model"]),
            )
        except Exception as exc:
            retry_class = classify_repricing_failure(
                status="failed",
                reason="",
                failure_stage="initialization",
                exception=exc,
            )
            row.update(
                recalc_status="failed",
                recalc_reason=(
                    "method_unavailable"
                    if retry_class is RecalcRetryClass.INFRASTRUCTURE
                    else f"solver_exception:{type(exc).__name__}"
                ),
                recalc_failure_stage="initialization",
                recalc_retry_class=retry_class.value,
                recalc_quality_flags=row["recalc_quality_flags"] | 1,
            )
            diagnostics["recalc_batch_exception_count"] += 1
            diagnostics["recalc_action_solve_normal"] += 1
            if retry_class is RecalcRetryClass.INFRASTRUCTURE:
                diagnostics["recalc_infrastructure_failure_count"] += 1
            diagnostics["recalc_unrecoverable_failure_count"] += 1
            row["calculation_error"] = "; ".join(
                filter(None, (row.get("calculation_error", ""), f"recalc:{type(exc).__name__}"))
            )

    prior_outcomes: Mapping[str, Any] = {}
    lookup_enabled = (
        reuse_policy.enabled or seed_policy.enabled or failure_policy.enabled
    )
    if lookup_enabled and prior_outcome_loader is not None and current_by_index:
        started = time.perf_counter()
        prior_outcomes = prior_outcome_loader(
            tuple(rows[index] for index in current_by_index),
            PriorOutcomeLookupPolicy(
                maximum_age_seconds=max(
                    reuse_policy.maximum_age_seconds
                    if reuse_policy.enabled
                    else 0.0,
                    seed_policy.maximum_age_seconds
                    if seed_policy.enabled
                    else 0.0,
                    failure_policy.maximum_failure_age_seconds
                    if failure_policy.enabled
                    else 0.0,
                ),
                lookup_allowance_seconds=max(
                    reuse_policy.lookup_allowance_seconds
                    if reuse_policy.enabled
                    else 0.0,
                    failure_policy.lookup_allowance_seconds
                    if failure_policy.enabled
                    else 0.0,
                ),
                need_success_reuse=reuse_policy.enabled,
                need_success_seed=seed_policy.enabled,
                need_failure_reuse=failure_policy.enabled,
                need_failure_seed=seed_policy.enabled,
                minimum_seed_iv=seed_policy.minimum_iv,
                maximum_seed_iv=seed_policy.maximum_iv,
            ),
        )
        if timings is not None:
            elapsed = time.perf_counter() - started
            timings.values["contract.prior_outcome_lookup_seconds"] = timings.values.get(
                "contract.prior_outcome_lookup_seconds", 0.0
            ) + elapsed
            timings.values["contract.failure_outcome_lookup_seconds"] = timings.values.get(
                "contract.failure_outcome_lookup_seconds", 0.0
            ) + elapsed

    def publish_origin(row: dict[str, Any], origin: str) -> None:
        row.update(
            recalc_result_origin=origin,
            recalc_reuse_mode="none",
            recalc_origin_snapshot_id=row["snapshot_id"],
            recalc_origin_calculated_at=row["calculated_at"],
            recalc_valuation_at=row["source_collected_at"],
            recalc_valuation_spot=row["spot_price"],
            recalc_valuation_target_price=row["recalc_target_price"],
            recalc_valuation_time_to_expiry_seconds=row["time_to_expiry_seconds"],
            recalc_valuation_rate=row["risk_free_rate"],
            recalc_valuation_dividend_yield=row["dividend_yield"],
        )

    def publish_reused(row: dict[str, Any], prior: Any, decision: Any) -> None:
        started = time.perf_counter()
        row.update(
            recalc_status="complete",
            recalc_reason="successful_result_reused",
            recalc_failure_stage="none",
            recalc_method=prior.method,
            recalc_method_selection_reason=prior.method_selection_reason,
            recalc_algorithm_version=prior.algorithm_version,
            recalc_solver_config_hash=prior.solver_config_hash,
            recalc_solver_parameters_json=prior.solver_parameters_json,
            recalc_input_basis=prior.input_basis,
            recalc_input_iv=prior.input_iv,
            recalc_iv=prior.implied_volatility,
            recalc_iv_iterations=prior.iv_iterations,
            recalc_price_residual=prior.price_residual,
            recalc_model_price=prior.model_price,
            recalc_delta=prior.delta,
            recalc_gamma=prior.gamma,
            recalc_theta_per_day=prior.theta_per_day,
            recalc_vega_per_1pct=prior.vega_per_1pct,
            recalc_rho_per_1pct=prior.rho_per_1pct,
            recalc_requested_dtype=prior.requested_dtype,
            recalc_actual_dtype=prior.actual_dtype,
            recalc_precision_escalated=prior.precision_escalated,
            recalc_precision_escalation_reason=(
                prior.precision_escalation_reason
            ),
            recalc_quality_flags=prior.quality_flags,
            recalc_time_steps=prior.time_steps,
            recalc_space_steps=prior.space_steps,
            recalc_damping_steps=prior.damping_steps,
            recalc_dividend_model=prior.dividend_model,
            recalc_total_implied_variance=(
                prior.implied_volatility * prior.implied_volatility
                * float(row["time_to_expiry_years"])
            ),
            recalc_result_origin="reused",
            recalc_reuse_mode="bounded",
            recalc_origin_snapshot_id=prior.origin_snapshot_id,
            recalc_origin_calculated_at=prior.origin_calculated_at,
            recalc_valuation_at=prior.valuation_at,
            recalc_valuation_spot=prior.valuation_spot,
            recalc_valuation_target_price=prior.valuation_target_price,
            recalc_valuation_time_to_expiry_seconds=prior.valuation_time_to_expiry_seconds,
            recalc_valuation_rate=prior.valuation_rate,
            recalc_valuation_dividend_yield=prior.valuation_dividend_yield,
            recalc_reuse_age_seconds=decision.age_seconds,
            recalc_reuse_spot_move_bps=decision.spot_move_bps,
            recalc_reuse_elapsed_tenor_fraction=(
                decision.elapsed_fraction_of_prior_remaining_tenor
            ),
            recalc_reuse_estimated_price_drift=decision.estimated_price_drift,
            recalc_reuse_estimated_iv_change_points=(
                decision.estimated_iv_change_points
            ),
        )
        if timings is not None:
            timings.values["contract.reuse_copy_seconds"] += time.perf_counter() - started

    def publish_reused_failure(
        row: dict[str, Any], prior: Any, decision: Any
    ) -> None:
        row.update(
            recalc_status="failed",
            recalc_reason=prior.reason,
            recalc_failure_stage=prior.failure_stage,
            recalc_method=prior.method,
            recalc_method_selection_reason=prior.method_selection_reason,
            recalc_algorithm_version=prior.algorithm_version,
            recalc_solver_config_hash=prior.solver_config_hash,
            recalc_solver_parameters_json=prior.solver_parameters_json,
            recalc_input_basis=prior.input_basis,
            recalc_input_iv=prior.input_iv,
            recalc_iv=prior.implied_volatility,
            recalc_iv_iterations=prior.iv_iterations,
            recalc_price_residual=prior.price_residual,
            recalc_model_price=prior.model_price,
            recalc_delta=prior.delta,
            recalc_gamma=prior.gamma,
            recalc_theta_per_day=prior.theta_per_day,
            recalc_vega_per_1pct=prior.vega_per_1pct,
            recalc_rho_per_1pct=prior.rho_per_1pct,
            recalc_result_origin="reused_failure",
            recalc_reuse_mode="failure_backoff",
            recalc_origin_snapshot_id=prior.origin_snapshot_id,
            recalc_origin_calculated_at=prior.origin_calculated_at,
            recalc_valuation_at=prior.valuation_at,
            recalc_valuation_spot=prior.valuation_spot,
            recalc_valuation_target_price=prior.valuation_target_price,
            recalc_valuation_time_to_expiry_seconds=prior.valuation_time_to_expiry_seconds,
            recalc_valuation_rate=prior.valuation_rate,
            recalc_valuation_dividend_yield=prior.valuation_dividend_yield,
            recalc_retry_class=prior.retry_class,
            recalc_retry_after=decision.retry_after,
            recalc_failure_fingerprint=prior.failure_fingerprint,
            recalc_attempt_suppressed=True,
            recalc_quality_flags=int(row["recalc_quality_flags"]) | 1,
        )

    def publish_circuit_suppressed(
        row: dict[str, Any], open_until: datetime
    ) -> None:
        row.update(
            recalc_status="failed",
            recalc_reason="solver_circuit_open",
            recalc_failure_stage="initialization",
            recalc_retry_class=RecalcRetryClass.INFRASTRUCTURE.value,
            recalc_retry_after=open_until,
            recalc_result_origin="suppressed",
            recalc_reuse_mode="circuit_open",
            recalc_attempt_suppressed=True,
            recalc_quality_flags=int(row["recalc_quality_flags"]) | 1,
        )

    def loaded_priors(current: CurrentRepricingProblem) -> tuple[Any, Any, Any]:
        loaded = prior_outcomes.get(
            (current.snapshot_id, current.option_symbol),
            prior_outcomes.get(current.option_symbol),
        ) if lookup_enabled else None
        if isinstance(loaded, PriorOutcomeCandidates):
            return loaded.success, loaded.failure, loaded.failure_seed
        if loaded is not None and getattr(loaded, "status", "complete") == "complete":
            return loaded, None, None
        return None, loaded, loaded

    # Prepare the complete primary wave before numerical execution. This is
    # deterministic and lets CRR/Tian see genuinely aligned work instead of
    # relying on thread timing to make singleton calls overlap.
    prepared_results: dict[int, Any] = {}
    prepared_errors: dict[int, BaseException] = {}
    solve_many = getattr(registry, "solve_iv_and_greeks_many", None)
    if callable(solve_many):
        prepared_jobs: list[tuple[int, SolverBatchJob]] = []
        for index in selected:
            if index not in current_by_index:
                continue
            row = rows[index]
            current = current_by_index[index]
            success_prior, failed_prior, failed_seed_prior = loaded_priors(current)
            action = decide_repricing_action(
                current=current,
                success_prior=success_prior,
                failed_prior=failed_prior,
                reuse_policy=reuse_policy,
                seed_policy=seed_policy,
                failure_policy=failure_policy,
                now=_parse_timestamp(row["calculated_at"]),
                failed_seed_prior=failed_seed_prior,
            )
            if action.action is RepricingAction.REUSE_FAILURE_BACKOFF:
                continue
            shadow = (
                action.action is RepricingAction.REUSE_SUCCESS
                and deterministic_shadow_sample(current, reuse_policy.shadow)
            )
            if action.action is RepricingAction.REUSE_SUCCESS and not shadow:
                continue
            hints: tuple[IvHint, ...] = ()
            if action.action is RepricingAction.SOLVE_PRIOR_IV_SEED:
                seed = action.seed_decision
                hints = (
                    IvHint(
                        float(action.selected_iv_seed),
                        action.selected_iv_seed_origin,
                        action.selected_iv_seed_snapshot_id,
                        seed.seed_calculated_at if seed is not None else None,
                    ),
                )
            request = request_by_index[index]
            method = RepricingMethod.parse(current.method)
            retry_due = (
                action.failure_reuse_decision is not None
                and action.failure_reuse_decision.reason == "retry_due"
            )
            category = (
                "shadow"
                if shadow
                else "retry"
                if retry_due
                else "prior_seed"
                if action.action is RepricingAction.SOLVE_PRIOR_IV_SEED
                else "normal"
            )
            prepared_jobs.append(
                (
                    index,
                    SolverBatchJob(
                        job_id=(
                            f"contract:{current.snapshot_id}:"
                            f"{current.option_symbol}:{index}"
                        ),
                        owner_run_id=current.snapshot_id,
                        owner_snapshot_id=current.snapshot_id,
                        output_index=index,
                        operation=SolverOperation.SOLVE_IV_AND_GREEKS,
                        request=request,
                        method=method,
                        method_selection_reason=str(
                            row["recalc_method_selection_reason"]
                        ),
                        target_price=float(row["recalc_target_price"]),
                        iv_hints=hints,
                        requested_dtype=request.requested_dtype,
                        solver_configuration_hash=str(
                            row["recalc_solver_config_hash"]
                        ),
                        attempt_category=category,
                        shadow=shadow,
                    ),
                )
            )
        if prepared_jobs:
            ordered_jobs = sorted(
                prepared_jobs,
                key=lambda item: (
                    item[1].method.value,
                    item[1].requested_dtype,
                    item[1].owner_snapshot_id,
                    item[1].request.option_symbol,
                    item[1].output_index,
                ),
            )
            grouped_jobs: dict[tuple[str, str, str], list[tuple[int, SolverBatchJob]]] = {}
            for index, job in ordered_jobs:
                grouped_jobs.setdefault(
                    (
                        job.method.value,
                        job.requested_dtype,
                        job.solver_configuration_hash,
                    ),
                    [],
                ).append((index, job))
            for group in grouped_jobs.values():
                try:
                    batch_results = solve_many([job for _index, job in group])
                    if len(batch_results) != len(group):
                        raise RuntimeError(
                            "solver primary wave returned a misaligned result"
                        )
                    prepared_results.update(
                        (index, result)
                        for (index, _job), result in zip(group, batch_results)
                    )
                except BaseException as exc:
                    prepared_errors.update((index, exc) for index, _job in group)

    for index in selected:
        if index not in current_by_index:
            continue
        row = rows[index]
        request = request_by_index[index]
        current = current_by_index[index]
        method = RepricingMethod.parse(current.method)
        success_prior, failed_prior, failed_seed_prior = loaded_priors(current)
        if success_prior is None and failed_prior is None and lookup_enabled:
            diagnostics["recalc_reuse_miss_no_prior"] += 1
        started = time.perf_counter()
        action = decide_repricing_action(
            current=current,
            success_prior=success_prior,
            failed_prior=failed_prior,
            reuse_policy=reuse_policy,
            seed_policy=seed_policy,
            failure_policy=failure_policy,
            now=_parse_timestamp(row["calculated_at"]),
            failed_seed_prior=failed_seed_prior,
        )
        action_elapsed = time.perf_counter() - started
        if timings is not None:
            timings.values["contract.action_decision_seconds"] = timings.values.get(
                "contract.action_decision_seconds", 0.0
            ) + action_elapsed
            if success_prior is not None:
                timings.values["contract.seed_assessment_seconds"] = timings.values.get(
                    "contract.seed_assessment_seconds", 0.0
                ) + action_elapsed
            if failed_prior is not None:
                timings.values["contract.failure_reuse_assessment_seconds"] = timings.values.get(
                    "contract.failure_reuse_assessment_seconds", 0.0
                ) + action_elapsed
        diagnostics[f"recalc_action_{action.action.value}"] += 1
        decision = action.success_reuse_decision
        shadow = False
        if decision is not None and not decision.reusable:
            name = REUSE_INVALIDATION_DIAGNOSTIC.get(decision.reason)
            if name:
                diagnostics[name] = diagnostics.get(name, 0) + 1
        if action.action is RepricingAction.REUSE_SUCCESS:
            shadow = deterministic_shadow_sample(current, reuse_policy.shadow)
            if not shadow:
                publish_reused(row, success_prior, decision)
                diagnostics["recalc_successes_reused"] += 1
                continue
        if action.action is RepricingAction.REUSE_FAILURE_BACKOFF:
            publish_reused_failure(row, failed_prior, action.failure_reuse_decision)
            diagnostics["recalc_failures_reused"] += 1
            if has_required_source_greeks(row):
                diagnostics["recalc_failure_affected_with_source_fallback"] = (
                    diagnostics.get("recalc_failure_affected_with_source_fallback", 0) + 1
                )
            else:
                diagnostics["recalc_failure_affected_without_source_fallback"] = (
                    diagnostics.get("recalc_failure_affected_without_source_fallback", 0) + 1
                )
            continue
        retry_due = (
            action.failure_reuse_decision is not None
            and action.failure_reuse_decision.reason == "retry_due"
        )
        if retry_due:
            diagnostics["recalc_failure_retry_due"] += 1
        if action.failure_reuse_decision is not None and not action.failure_reuse_decision.reusable:
            failure_reason = action.failure_reuse_decision.reason
            metric = {
                "failure_reason_not_allowed": "recalc_failure_reuse_invalidated_reason",
                "failure_target_absolute_change_exceeded": "recalc_failure_reuse_invalidated_target_price",
                "failure_target_fractional_change_exceeded": "recalc_failure_reuse_invalidated_target_price",
                "failure_spot_move_exceeded": "recalc_failure_reuse_invalidated_spot",
                "failure_age_exceeded": "recalc_failure_reuse_invalidated_age",
                "retry_due": "recalc_failure_reuse_invalidated_retry_due",
            }.get(failure_reason)
            if metric:
                diagnostics[metric] = diagnostics.get(metric, 0) + 1

        iv_hints: tuple[IvHint, ...] = ()
        if action.action is RepricingAction.SOLVE_PRIOR_IV_SEED:
            diagnostics["recalc_prior_iv_seed_candidates"] += 1
            seed = action.seed_decision
            iv_hints = (
                IvHint(
                    float(action.selected_iv_seed),
                    action.selected_iv_seed_origin,
                    action.selected_iv_seed_snapshot_id,
                    seed.seed_calculated_at if seed is not None else None,
                ),
            )
            diagnostics["recalc_prior_iv_seed_used"] += 1
            if action.selected_iv_seed_origin == "prior_partial_failure":
                diagnostics["recalc_prior_partial_failure_seed_used"] += 1
            row.update(
                recalc_iv_seed_origin=action.selected_iv_seed_origin,
                recalc_iv_seed_snapshot_id=action.selected_iv_seed_snapshot_id,
                recalc_iv_seed_calculated_at=(seed.seed_calculated_at if seed else None),
                recalc_iv_seed_value=action.selected_iv_seed,
                recalc_iv_seed_age_seconds=(seed.age_seconds if seed else None),
            )

        if shadow:
            row["recalc_shadow_checked"] = True
        performance: dict[str, float | int] = {}
        attempt_recorded = False
        seed_mismatch_fallback = False
        solver_started = time.perf_counter()
        try:
            solve_kwargs = dict(
                method=method,
                method_selection_reason=row["recalc_method_selection_reason"],
                target_price=float(row["recalc_target_price"]),
                performance=performance,
            )
            if iv_hints:
                solve_kwargs["iv_hints"] = iv_hints
            if index in prepared_errors:
                raise prepared_errors[index]
            result = prepared_results.get(index)
            if result is None:
                result = registry.solve_iv_and_greeks(request, **solve_kwargs)
            else:
                performance.update(
                    solver_call_permitted=int(
                        result.reason != "solver_circuit_open"
                    ),
                    solver_adapter_invoked=int(
                        result.reason != "solver_circuit_open"
                    ),
                    circuit_suppressed=int(
                        result.reason == "solver_circuit_open"
                    ),
                    iv_price_evaluations=int(result.iv_price_evaluations),
                )
                if result.reason == "solver_circuit_open":
                    raise SolverCircuitOpenError(
                        method.value, registry.circuit(method).open_until
                    )
            if int(performance.get("solver_call_permitted", 0)):
                attempt_recorded = True
                diagnostics["recalc_solver_attempted_count"] += 1
                categorize_permitted_attempt(
                    action.action, shadow=shadow, retry_due=retry_due
                )
                diagnostics[f"recalc_{method.value}_attempted"] = diagnostics.get(
                    f"recalc_{method.value}_attempted", 0
                ) + 1
                if shadow:
                    diagnostics["recalc_shadow_attempted"] += 1
                if retry_due:
                    diagnostics["recalc_failure_retry_attempted"] += 1
            accumulate_performance(method, performance)
            if shadow and timings is not None:
                timings.values["contract.shadow_solver_seconds"] += (
                    time.perf_counter() - solver_started
                )
            row.update(
                recalc_status=result.status,
                recalc_reason=result.reason,
                recalc_failure_stage=result.failure_stage,
                recalc_input_iv=result.input_iv,
                recalc_iv=result.implied_volatility,
                recalc_iv_iterations=max(0, int(result.iv_iterations)),
                recalc_price_residual=result.price_residual,
                recalc_model_price=result.model_price,
                recalc_delta=result.delta,
                recalc_gamma=result.gamma,
                recalc_theta_per_day=result.theta_per_day,
                recalc_vega_per_1pct=result.vega_per_1pct,
                recalc_rho_per_1pct=result.rho_per_1pct,
                recalc_requested_dtype=result.requested_dtype,
                recalc_actual_dtype=result.actual_dtype,
                recalc_precision_escalated=result.precision_escalated,
                recalc_precision_escalation_reason=(
                    result.precision_escalation_reason
                ),
                recalc_quality_flags=(
                    int(row["recalc_quality_flags"]) | int(result.quality_flags)
                ),
                recalc_total_implied_variance=(
                    result.implied_volatility * result.implied_volatility
                    * float(row["time_to_expiry_years"])
                    if result.implied_volatility is not None
                    else None
                ),
                recalc_iv_seed_hint_evaluations=int(result.iv_hint_evaluations),
                recalc_iv_seed_exact_match=bool(result.iv_hint_exact_match),
                recalc_iv_seed_local_bracket_found=bool(
                    result.iv_hint_local_bracket_found
                ),
                recalc_iv_seed_global_fallback_used=bool(
                    result.iv_fallback_global_probes_used
                ),
                recalc_iv_identifiability_status=(
                    result.iv_identifiability_status
                ),
                recalc_iv_identifiability_price_span=(
                    result.iv_identifiability_price_span
                ),
                recalc_iv_identifiability_slope=(
                    result.iv_identifiability_slope
                ),
                recalc_iv_hint_flat_rejection_count=int(
                    result.iv_hint_flat_rejection_count
                ),
            )
            if result.iv_hint_origin_used == "source":
                row.update(
                    recalc_iv_seed_origin="source",
                    recalc_iv_seed_snapshot_id=None,
                    recalc_iv_seed_calculated_at=None,
                    recalc_iv_seed_value=result.iv_hint_value_used,
                    recalc_iv_seed_age_seconds=None,
                )
                diagnostics["recalc_source_iv_seed_used"] += 1
            elif result.iv_hint_origin_used not in {
                "prior_success", "prior_partial_failure"
            }:
                row.update(
                    recalc_iv_seed_origin="none",
                    recalc_iv_seed_snapshot_id=None,
                    recalc_iv_seed_calculated_at=None,
                    recalc_iv_seed_value=None,
                    recalc_iv_seed_age_seconds=None,
                )
            if action.action is RepricingAction.SOLVE_PRIOR_IV_SEED:
                diagnostics["recalc_seed_price_evaluations"] += int(
                    result.iv_price_evaluations
                )
                diagnostics["recalc_seed_exact_matches"] += int(
                    result.iv_hint_exact_match
                )
                diagnostics["recalc_seed_local_brackets"] += int(
                    result.iv_hint_local_bracket_found
                )
                diagnostics["recalc_seed_global_fallbacks"] += int(
                    result.iv_fallback_global_probes_used
                )
                if deterministic_seed_shadow_sample(current, seed_policy.shadow):
                    diagnostics["recalc_seed_baseline_attempted"] += 1
                    baseline_performance: dict[str, float | int] = {}
                    baseline_started = time.perf_counter()
                    mismatch_fallback_started = False
                    seeded_result = result
                    try:
                        baseline = registry.solve_iv_only(
                            request,
                            method=method,
                            target_price=float(row["recalc_target_price"]),
                            performance=baseline_performance,
                            consult_circuit=False,
                        )
                        if baseline.status == "complete" and baseline.iv is not None:
                            diagnostics["recalc_seed_baseline_completed"] += 1
                            difference = int(baseline.price_evaluations) - int(
                                seeded_result.iv_price_evaluations
                            )
                            if difference > 0:
                                diagnostics["recalc_seed_baseline_evaluations_saved"] += difference
                                comparison = "reduced_evaluations"
                            elif difference < 0:
                                diagnostics["recalc_seed_baseline_evaluations_increased"] += -difference
                                comparison = "increased_evaluations"
                            else:
                                comparison = "no_effect"
                            mismatch = abs(
                                float(baseline.iv)
                                - float(seeded_result.implied_volatility or math.nan)
                            )
                            residual_difference = abs(
                                float(baseline.price_residual or 0.0)
                                - float(seeded_result.price_residual or 0.0)
                            )
                            mismatch_breach = (
                                mismatch > seed_policy.shadow.maximum_iv_difference
                                or residual_difference
                                > seed_policy.shadow.maximum_price_residual_difference
                            )
                            if mismatch_breach:
                                diagnostics["recalc_seed_baseline_iv_mismatches"] += 1
                                comparison = "iv_mismatch"
                                fallback_performance: dict[str, float | int] = {}
                                mismatch_fallback_started = True
                                try:
                                    fallback_result = registry.solve_iv_and_greeks(
                                        request,
                                        method=method,
                                        method_selection_reason=row[
                                            "recalc_method_selection_reason"
                                        ],
                                        target_price=float(row["recalc_target_price"]),
                                        performance=fallback_performance,
                                    )
                                finally:
                                    if int(
                                        fallback_performance.get(
                                            "solver_call_permitted", 0
                                        )
                                    ):
                                        diagnostics[
                                            "recalc_solver_attempted_count"
                                        ] += 1
                                        categorize_permitted_attempt(
                                            action.action,
                                            shadow=False,
                                            retry_due=False,
                                            mismatch_fallback=True,
                                        )
                                        diagnostics[
                                            f"recalc_{method.value}_attempted"
                                        ] = diagnostics.get(
                                            f"recalc_{method.value}_attempted", 0
                                        ) + 1
                                    accumulate_performance(
                                        method, fallback_performance
                                    )
                                diagnostics[
                                    "recalc_seed_mismatch_fallback_attempted"
                                ] = diagnostics.get(
                                    "recalc_seed_mismatch_fallback_attempted", 0
                                ) + 1
                                # Once a material mismatch is proven, never
                                # publish the seeded result.  The current
                                # source-first solve is authoritative even if
                                # it returns a stable failure.
                                result = fallback_result
                                seed_mismatch_fallback = True
                                row.update(
                                    recalc_status=result.status,
                                    recalc_reason=(
                                        "seeded_mismatch_fallback"
                                        if result.status == "complete"
                                        else result.reason
                                    ),
                                    recalc_failure_stage=result.failure_stage,
                                    recalc_input_iv=result.input_iv,
                                    recalc_iv=result.implied_volatility,
                                    recalc_iv_iterations=max(0, int(result.iv_iterations)),
                                    recalc_price_residual=result.price_residual,
                                    recalc_model_price=result.model_price,
                                    recalc_delta=result.delta,
                                    recalc_gamma=result.gamma,
                                    recalc_theta_per_day=result.theta_per_day,
                                    recalc_vega_per_1pct=result.vega_per_1pct,
                                    recalc_rho_per_1pct=result.rho_per_1pct,
                                    recalc_requested_dtype=result.requested_dtype,
                                    recalc_actual_dtype=result.actual_dtype,
                                    recalc_precision_escalated=result.precision_escalated,
                                    recalc_precision_escalation_reason=(
                                        result.precision_escalation_reason
                                    ),
                                    recalc_total_implied_variance=(
                                        result.implied_volatility
                                        * result.implied_volatility
                                        * float(row["time_to_expiry_years"])
                                        if result.implied_volatility is not None
                                        else None
                                    ),
                                    recalc_iv_identifiability_status=result.iv_identifiability_status,
                                    recalc_iv_identifiability_price_span=result.iv_identifiability_price_span,
                                    recalc_iv_identifiability_slope=result.iv_identifiability_slope,
                                    recalc_iv_hint_flat_rejection_count=int(
                                        result.iv_hint_flat_rejection_count
                                    ),
                                    recalc_iv_seed_origin=result.iv_hint_origin_used,
                                    recalc_iv_seed_snapshot_id=None,
                                    recalc_iv_seed_calculated_at=None,
                                    recalc_iv_seed_value=result.iv_hint_value_used,
                                    recalc_iv_seed_age_seconds=None,
                                )
                                if result.status == "complete":
                                    diagnostics[
                                        "recalc_seed_mismatch_fallback_published"
                                    ] = diagnostics.get(
                                        "recalc_seed_mismatch_fallback_published", 0
                                    ) + 1
                            diagnostics.setdefault("_recalc_validation_records", []).append(
                                {
                                    "validation_type": "seeded_refresh_baseline",
                                    "snapshot_id": str(row["snapshot_id"]),
                                    "source_collected_at": row["source_collected_at"],
                                    "option_symbol": current.option_symbol,
                                    "put_call": str(row["_base"].get("put_call") or ""),
                                    "expiration_at": _parse_timestamp(row["_base"].get("expiration_at")),
                                    "time_to_expiry_seconds": current.time_to_expiry_seconds,
                                    "log_forward_moneyness": row.get("log_forward_moneyness"),
                                    "pricing_method": method.value,
                                    "pricing_algorithm_version": current.algorithm_version,
                                    "pricing_solver_config_hash": current.solver_config_hash,
                                    "candidate_origin_snapshot_id": action.selected_iv_seed_snapshot_id,
                                    "candidate_age_seconds": (
                                        seed.age_seconds if seed is not None else None
                                    ),
                                    "candidate_iv": action.selected_iv_seed,
                                    "current_iv": seeded_result.implied_volatility,
                                    "absolute_iv_error": mismatch,
                                    "absolute_price_residual_difference": residual_difference,
                                    "production_iv_price_evaluations": seeded_result.iv_price_evaluations,
                                    "baseline_iv_price_evaluations": baseline.price_evaluations,
                                    "production_iv_identifiability_status": seeded_result.iv_identifiability_status,
                                    "production_iv_identifiability_price_span": seeded_result.iv_identifiability_price_span,
                                    "production_iv_identifiability_slope": seeded_result.iv_identifiability_slope,
                                    "production_iv_hint_flat_rejection_count": int(
                                        seeded_result.iv_hint_flat_rejection_count
                                    ),
                                    "baseline_iv_identifiability_status": baseline.identifiability_status,
                                    "baseline_iv_identifiability_price_span": baseline.identifiability_price_span,
                                    "baseline_iv_identifiability_slope": baseline.identifiability_slope,
                                    "baseline_iv_hint_flat_rejection_count": int(
                                        baseline.hint_flat_rejection_count
                                    ),
                                    "production_fdm_valuations": performance.get("fdm_valuations_iv", 0),
                                    "baseline_fdm_valuations": baseline_performance.get("fdm_valuations", 0),
                                    "mismatch_fallback_status": (
                                        "published_complete"
                                        if mismatch_breach
                                        and result.status == "complete"
                                        else "published_failure"
                                        if mismatch_breach
                                        else "not_needed"
                                    ),
                                    "comparison_status": comparison,
                                    "comparison_reason": baseline.reason,
                                }
                            )
                    except Exception as baseline_exc:
                        if mismatch_fallback_started:
                            raise
                        diagnostics.setdefault("_recalc_validation_records", []).append(
                            {
                                "validation_type": "seeded_refresh_baseline",
                                "snapshot_id": str(row["snapshot_id"]),
                                "source_collected_at": row["source_collected_at"],
                                "option_symbol": current.option_symbol,
                                "put_call": str(row["_base"].get("put_call") or ""),
                                "expiration_at": _parse_timestamp(row["_base"].get("expiration_at")),
                                "time_to_expiry_seconds": current.time_to_expiry_seconds,
                                "log_forward_moneyness": row.get("log_forward_moneyness"),
                                "pricing_method": method.value,
                                "pricing_algorithm_version": current.algorithm_version,
                                "pricing_solver_config_hash": current.solver_config_hash,
                                "comparison_status": "baseline_failed",
                                "comparison_reason": type(baseline_exc).__name__,
                                "mismatch_fallback_status": "baseline_failed",
                            }
                        )
                    finally:
                        if timings is not None:
                            timings.values["contract.seed_baseline_seconds"] += (
                                time.perf_counter() - baseline_started
                            )
            if result.status == "complete":
                diagnostics["recalc_complete_count"] += 1
                diagnostics["recalc_successes_calculated"] += 1
                key = f"recalc_{method.value}_complete"
                diagnostics[key] = diagnostics.get(key, 0) + 1
                publish_origin(
                    row,
                    "calculated_shadow"
                    if shadow
                    else (
                        "calculated"
                        if seed_mismatch_fallback
                        else "calculated_seeded"
                        if action.action is RepricingAction.SOLVE_PRIOR_IV_SEED
                        else "calculated"
                    ),
                )
                if (
                    action.action is RepricingAction.SOLVE_PRIOR_IV_SEED
                    and not seed_mismatch_fallback
                ):
                    row["recalc_reuse_mode"] = "seeded_refresh"
                    if action.failure_reuse_decision is not None and action.failure_reuse_decision.reason == "retry_due":
                        diagnostics["recalc_failure_retry_succeeded"] += 1
                if shadow:
                    iv_error = abs(float(result.implied_volatility) - success_prior.implied_volatility)
                    delta_error = abs(float(result.delta) - success_prior.delta)
                    absolute_k = abs(float(row.get("log_forward_moneyness") or 0.0))
                    shadow_record = {
                        "tenor_bucket": (
                            "short"
                            if current.time_to_expiry_seconds
                            <= reuse_policy.shadow.short_tenor_seconds
                            else "other"
                        ),
                        "method": method.value,
                        "put_call": str(row["_base"].get("put_call", "")),
                        "moneyness_bucket": (
                            "atm" if absolute_k < 0.05 else "middle" if absolute_k < 0.15 else "wing"
                        ),
                        "absolute_iv_error": iv_error,
                        "absolute_delta_error": delta_error,
                        "absolute_gamma_error": abs(float(result.gamma) - success_prior.gamma),
                        "absolute_theta_error": abs(
                            float(result.theta_per_day) - success_prior.theta_per_day
                        ),
                        "absolute_vega_error": abs(
                            float(result.vega_per_1pct) - success_prior.vega_per_1pct
                        ),
                        "absolute_rho_error": abs(
                            float(result.rho_per_1pct) - success_prior.rho_per_1pct
                        ),
                        "current_model_price_difference": (
                            abs(float(result.model_price) - success_prior.model_price)
                            if result.model_price is not None
                            and success_prior.model_price is not None
                            and math.isfinite(float(result.model_price))
                            and math.isfinite(float(success_prior.model_price))
                            else None
                        ),
                    }
                    diagnostics.setdefault("_recalc_shadow_records", []).append(
                        shadow_record
                    )
                    diagnostics.setdefault("_recalc_validation_records", []).append(
                        {
                            "validation_type": "success_reuse_shadow",
                            "snapshot_id": str(row["snapshot_id"]),
                            "source_collected_at": row["source_collected_at"],
                            "option_symbol": current.option_symbol,
                            "put_call": str(row["_base"].get("put_call") or ""),
                            "expiration_at": _parse_timestamp(row["_base"].get("expiration_at")),
                            "time_to_expiry_seconds": current.time_to_expiry_seconds,
                            "log_forward_moneyness": row.get("log_forward_moneyness"),
                            "pricing_method": method.value,
                            "pricing_algorithm_version": current.algorithm_version,
                            "pricing_solver_config_hash": current.solver_config_hash,
                            "candidate_origin_snapshot_id": success_prior.origin_snapshot_id,
                            "candidate_valuation_at": success_prior.valuation_at,
                            "candidate_age_seconds": decision.age_seconds,
                            "spot_move_bps": decision.spot_move_bps,
                            "elapsed_tenor_fraction": decision.elapsed_fraction_of_prior_remaining_tenor,
                            "estimated_price_drift": decision.estimated_price_drift,
                            "estimated_iv_change_points": decision.estimated_iv_change_points,
                            "candidate_iv": success_prior.implied_volatility,
                            "current_iv": result.implied_volatility,
                            "absolute_iv_error": iv_error,
                            "candidate_delta": success_prior.delta,
                            "current_delta": result.delta,
                            "absolute_delta_error": delta_error,
                            "absolute_gamma_error": shadow_record["absolute_gamma_error"],
                            "absolute_theta_error": shadow_record["absolute_theta_error"],
                            "absolute_vega_error": shadow_record["absolute_vega_error"],
                            "absolute_rho_error": shadow_record["absolute_rho_error"],
                            "comparison_status": (
                                "threshold_breach"
                                if iv_error > reuse_policy.shadow.maximum_iv_error
                                or delta_error > reuse_policy.shadow.maximum_delta_error
                                else "complete"
                            ),
                            "comparison_reason": (
                                "reuse_error_slo_breach"
                                if iv_error > reuse_policy.shadow.maximum_iv_error
                                or delta_error > reuse_policy.shadow.maximum_delta_error
                                else "within_thresholds"
                            ),
                        }
                    )
                    row.update(
                        recalc_shadow_status="complete",
                        recalc_shadow_iv_error=iv_error,
                        recalc_shadow_delta_error=delta_error,
                    )
                    diagnostics["recalc_shadow_complete"] += 1
                    if iv_error > reuse_policy.shadow.maximum_iv_error:
                        diagnostics["recalc_shadow_iv_threshold_breaches"] += 1
                        row["recalc_quality_flags"] |= 2
                    if delta_error > reuse_policy.shadow.maximum_delta_error:
                        diagnostics["recalc_shadow_delta_threshold_breaches"] += 1
                        row["recalc_quality_flags"] |= 4
            else:
                if shadow:
                    publish_reused(row, success_prior, decision)
                    row.update(
                        recalc_reason="successful_result_reused_shadow_failed",
                        recalc_shadow_checked=True,
                        recalc_shadow_status="failed",
                        recalc_quality_flags=int(row["recalc_quality_flags"]) | 8,
                    )
                    diagnostics["recalc_shadow_failed"] += 1
                    diagnostics["recalc_successes_reused"] += 1
                    continue
                row["recalc_quality_flags"] |= 1
                retry_class = classify_repricing_failure(
                    status=result.status,
                    reason=result.reason,
                    failure_stage=result.failure_stage,
                )
                row["recalc_retry_class"] = retry_class.value
                row["recalc_failure_fingerprint"] = failure_fingerprint(
                    current,
                    failure_stage=result.failure_stage,
                    reason=result.reason,
                )
                if (
                    retry_class.value in failure_policy.allowed_retry_classes
                    and result.reason in failure_policy.allowed_reasons
                ):
                    row["recalc_retry_after"] = retry_after_for_failure(
                        current, retry_class, failure_policy, row["calculated_at"]
                    )
                publish_origin(row, "calculated")
                diagnostics[f"recalc_fresh_failures_{retry_class.value}"] = (
                    diagnostics.get(f"recalc_fresh_failures_{retry_class.value}", 0) + 1
                )
                if action.failure_reuse_decision is not None and action.failure_reuse_decision.reason == "retry_due":
                    diagnostics["recalc_failure_retry_failed"] += 1
                key = f"recalc_{method.value}_failed"
                diagnostics[key] = diagnostics.get(key, 0) + 1
                if has_required_source_greeks(row):
                    diagnostics["recalc_recoverable_failure_count"] += 1
                    diagnostics["recalc_failure_affected_with_source_fallback"] = diagnostics.get(
                        "recalc_failure_affected_with_source_fallback", 0
                    ) + 1
                else:
                    diagnostics["recalc_unrecoverable_failure_count"] += 1
                    diagnostics["recalc_failure_affected_without_source_fallback"] = diagnostics.get(
                        "recalc_failure_affected_without_source_fallback", 0
                    ) + 1
        except SolverCircuitOpenError as exc:
            accumulate_performance(method, performance)
            diagnostics[f"recalc_action_{action.action.value}"] -= 1
            diagnostics["recalc_action_suppress_circuit_open"] += 1
            diagnostics["recalc_circuit_suppressed_calls"] += 1
            publish_circuit_suppressed(row, exc.open_until)
            if has_required_source_greeks(row):
                diagnostics["recalc_recoverable_failure_count"] += 1
                diagnostics["recalc_failure_affected_with_source_fallback"] = diagnostics.get(
                    "recalc_failure_affected_with_source_fallback", 0
                ) + 1
            else:
                diagnostics["recalc_unrecoverable_failure_count"] += 1
                diagnostics["recalc_failure_affected_without_source_fallback"] = diagnostics.get(
                    "recalc_failure_affected_without_source_fallback", 0
                ) + 1
            continue
        except Exception as exc:
            if (
                not attempt_recorded
                and int(performance.get("solver_call_permitted", 0))
            ):
                diagnostics["recalc_solver_attempted_count"] += 1
                categorize_permitted_attempt(
                    action.action, shadow=shadow, retry_due=retry_due
                )
                diagnostics[f"recalc_{method.value}_attempted"] = diagnostics.get(
                    f"recalc_{method.value}_attempted", 0
                ) + 1
                if shadow:
                    diagnostics["recalc_shadow_attempted"] += 1
                if retry_due:
                    diagnostics["recalc_failure_retry_attempted"] += 1
            if not attempt_recorded:
                accumulate_performance(method, performance)
            if shadow and timings is not None:
                timings.values["contract.shadow_solver_seconds"] += (
                    time.perf_counter() - solver_started
                )
            if shadow:
                publish_reused(row, success_prior, decision)
                row.update(
                    recalc_reason="successful_result_reused_shadow_failed",
                    recalc_shadow_checked=True,
                    recalc_shadow_status="failed",
                    recalc_quality_flags=int(row["recalc_quality_flags"]) | 8,
                )
                diagnostics["recalc_shadow_failed"] += 1
                diagnostics["recalc_successes_reused"] += 1
                continue
            retry_class = classify_repricing_failure(
                status="failed",
                reason="",
                failure_stage="initialization",
                exception=exc,
            )
            reason = (
                "method_unavailable"
                if isinstance(exc, SolverInfrastructureError)
                else f"solver_exception:{type(exc).__name__}"
            )
            row.update(
                recalc_status="failed",
                recalc_reason=reason,
                recalc_failure_stage="initialization",
                recalc_retry_class=retry_class.value,
                recalc_quality_flags=row["recalc_quality_flags"] | 1,
            )
            if retry_class is not RecalcRetryClass.UNKNOWN:
                row["recalc_failure_fingerprint"] = failure_fingerprint(
                    current, failure_stage="initialization", reason=reason
                )
            if (
                retry_class.value in failure_policy.allowed_retry_classes
                and reason in failure_policy.allowed_reasons
            ):
                row["recalc_retry_after"] = retry_after_for_failure(
                    current, retry_class, failure_policy, row["calculated_at"]
                )
            publish_origin(row, "calculated")
            diagnostics["recalc_batch_exception_count"] += 1
            key = f"recalc_{method.value}_failed"
            diagnostics[key] = diagnostics.get(key, 0) + 1
            if retry_class is RecalcRetryClass.INFRASTRUCTURE:
                diagnostics["recalc_infrastructure_failure_count"] += 1
            if has_required_source_greeks(row):
                diagnostics["recalc_recoverable_failure_count"] += 1
                diagnostics["recalc_failure_affected_with_source_fallback"] = diagnostics.get(
                    "recalc_failure_affected_with_source_fallback", 0
                ) + 1
            else:
                diagnostics["recalc_unrecoverable_failure_count"] += 1
                diagnostics["recalc_failure_affected_without_source_fallback"] = diagnostics.get(
                    "recalc_failure_affected_without_source_fallback", 0
                ) + 1
            row["calculation_error"] = "; ".join(
                filter(None, (row.get("calculation_error", ""), f"recalc:{type(exc).__name__}"))
            )


    if record_circuit_metrics and callable(snapshot_circuits):
        for method_name, current_snapshot in snapshot_circuits().items():
            previous = circuit_before.get(method_name, {})
            for metric in (
                "open_transitions", "close_transitions", "half_open_transitions",
                "half_open_probes", "actual_infrastructure_failures", "suppressed_calls",
            ):
                difference = int(current_snapshot.get(metric, 0)) - int(previous.get(metric, 0))
                diagnostics[f"recalc_circuit_{metric}"] = diagnostics.get(
                    f"recalc_circuit_{metric}", 0
                ) + max(0, difference)
                diagnostics[f"recalc_circuit_{method_name}_{metric}"] = max(0, difference)
            diagnostics[f"recalc_circuit_{method_name}_open_seconds"] = int(
                float(current_snapshot.get("open_seconds", 0.0))
            )

    reconciled_actions = sum(
        int(diagnostics[f"recalc_action_{name}"])
        for name in (
            "reuse_success",
            "solve_prior_iv_seed",
            "reuse_failure_backoff",
            "solve_normal",
            "suppress_circuit_open",
        )
    )
    diagnostics["recalc_action_reconciled_count"] = reconciled_actions
    if reconciled_actions != len(selected):
        raise AssertionError(
            "repricing action counts do not reconcile to selected contracts: "
            f"selected={len(selected)} actions={reconciled_actions}"
        )
    categorized_attempts = sum(
        int(diagnostics[f"recalc_solver_attempted_{category}"])
        for category in (
            "prior_seed", "normal", "retry", "shadow", "mismatch_fallback"
        )
    )
    diagnostics["recalc_solver_attempt_reconciled_count"] = categorized_attempts
    if categorized_attempts != int(diagnostics["recalc_solver_attempted_count"]):
        raise AssertionError(
            "repricing permitted-call categories do not reconcile to actual "
            f"solver attempts: actual={diagnostics['recalc_solver_attempted_count']} "
            f"categories={categorized_attempts}"
        )


def _resolve_crr_torch_dtype(torch_module: Any, configured: Any) -> Any:
    name = str(configured).strip().lower()
    if name.startswith("torch."):
        name = name[6:]
    aliases = {
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "float64": torch_module.float64,
        "fp64": torch_module.float64,
        "double": torch_module.float64,
    }
    if name not in aliases:
        raise ValueError(
            f"Unsupported CRR dtype {configured!r}; use float32 or float64"
        )
    return aliases[name]


def load_normalized_diagnostics(
    client: Any, database: str, snapshot_id: uuid.UUID
) -> dict[str, Any]:
    """Load normalized map-side and nested-delay diagnostics directly."""
    rows = client.query(
        f"""
        SELECT any(s.underlying_is_delayed),
               groupUniqArrayIf(c.option_symbol, o.symbol_map_mismatch)
        FROM {database}.options_chain_snapshots AS s
        INNER JOIN {database}.options_chain_observations AS o
          ON o.snapshot_id = s.snapshot_id
         AND o.source_collected_at = s.source_collected_at
        INNER JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c
          ON c.contract_id = o.contract_id
        WHERE s.snapshot_id = {{snapshot_id:UUID}}
        """,
        parameters={"snapshot_id": str(snapshot_id)},
    ).result_rows
    if not rows:
        raise SnapshotDataError(f"Missing normalized diagnostics for {snapshot_id}")
    underlying_delayed, mismatch_symbols = rows[0]
    return {
        "map_mismatch_symbols": set(mismatch_symbols or ()),
        "underlying_delayed": bool(underlying_delayed),
    }


def _normalized_source_diagnostics(
    base_rows: Sequence[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """Return normalized diagnostics, or ``None`` for legacy null rows."""

    def normalized_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return None

    if not base_rows or any(
        any(
            name not in row or row.get(name) is None
            for name in (
                "underlying_is_delayed",
                "symbol_map_mismatch",
                "option_symbol",
            )
        )
        for row in base_rows
    ):
        return None
    parsed = [
        (
            normalized_bool(row["underlying_is_delayed"]),
            normalized_bool(row["symbol_map_mismatch"]),
        )
        for row in base_rows
    ]
    if any(delayed is None or mismatch is None for delayed, mismatch in parsed):
        return None
    delayed_values = {delayed for delayed, _mismatch in parsed}
    if len(delayed_values) != 1:
        raise InconsistentSnapshotError(
            "Base snapshot mixes normalized underlying delay diagnostics"
        )
    return {
        "underlying_delayed": delayed_values.pop(),
        "map_mismatch_symbols": {
            str(row.get("option_symbol") or "")
            for row, (_delayed, mismatch) in zip(base_rows, parsed)
            if mismatch
        },
    }


def load_normalized_diagnostics_many(
    client: Any, database: str, snapshot_ids: Sequence[uuid.UUID]
) -> Mapping[uuid.UUID, dict[str, Any]]:
    """Load normalized diagnostics for owners with one query per safe chunk."""
    ordered_ids = tuple(dict.fromkeys(uuid.UUID(str(value)) for value in snapshot_ids))
    output: dict[uuid.UUID, dict[str, Any]] = {}
    for start in range(0, len(ordered_ids), BULK_OWNER_QUERY_CHUNK_SIZE):
        chunk = ordered_ids[start : start + BULK_OWNER_QUERY_CHUNK_SIZE]
        result = client.query(
            f"""
            SELECT toString(s.snapshot_id), any(s.underlying_is_delayed),
                   groupUniqArrayIf(c.option_symbol, o.symbol_map_mismatch)
            FROM {database}.options_chain_snapshots AS s
            INNER JOIN {database}.options_chain_observations AS o
              ON o.snapshot_id = s.snapshot_id
             AND o.source_collected_at = s.source_collected_at
            INNER JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c
              ON c.contract_id = o.contract_id
            WHERE s.snapshot_id IN {{snapshot_ids:Array(UUID)}}
            GROUP BY s.snapshot_id ORDER BY s.snapshot_id
            """,
            parameters={"snapshot_ids": [str(value) for value in chunk]},
        )
        seen: set[uuid.UUID] = set()
        for owner_text, underlying_delayed, mismatch_symbols in result.result_rows:
            owner = uuid.UUID(str(owner_text))
            if owner in seen or owner not in ordered_ids:
                raise SnapshotDataError(
                    f"Normalized diagnostic ownership is duplicate or unexpected: {owner}"
                )
            seen.add(owner)
            output[owner] = {
                "map_mismatch_symbols": set(mismatch_symbols or ()),
                "underlying_delayed": bool(underlying_delayed),
            }
        missing = set(chunk) - seen
        if missing:
            raise SnapshotDataError(
                "Missing normalized diagnostics for "
                + ", ".join(str(value) for value in sorted(missing, key=str))
            )
    return output


@dataclass
class PreparedSurfaceOwner:
    """Owner-local fits held between explicit batch numerical stages."""

    claim: Any
    source_rows: list[dict[str, Any]]
    by_expiration: dict[datetime, list[dict[str, Any]]]
    price_bases: tuple[str, ...]
    mode: str
    entries: list[tuple[datetime, str, ExpirationCalibration]]
    failed_entries: list[dict[str, Any]]
    numerical_diagnostics: list[dict[str, Any]]
    american_delta_diagnostics: Counter[str]
    timings: StageTimings
    numerical_runtime: SurfaceNumericalRuntime
    solver_registry: Any
    total_started: float

    @property
    def fallback_fits(self) -> int:
        """Deprecated compatibility alias."""
        return self.svi_fallback_fits


@dataclass(frozen=True)
class SurfaceNumericalOwnerRef:
    """Generation-3 snapshot identity passed to reusable surface code."""

    work: SurfaceWork

    @property
    def snapshot_id(self) -> uuid.UUID:
        return self.work.snapshot_id

    @property
    def symbol(self) -> str:
        return self.work.underlying_symbol

    @property
    def source_collected_at(self) -> datetime:
        return self.work.source_collected_at

    @property
    def expected_calculation_rows(self) -> int:
        return self.work.expected_observations


@dataclass(frozen=True)
class SurfaceNumericalOwner:
    """Ephemeral owner context for one generation-3 surface snapshot."""

    work: SurfaceWork
    started_at: datetime
    ref: SurfaceNumericalOwnerRef


def _surface_numerical_owner_claim(
    work: SurfaceWork, started_at: datetime
) -> SurfaceNumericalOwner:
    return SurfaceNumericalOwner(work, started_at, SurfaceNumericalOwnerRef(work))


def _surface_configuration_hash(context: Mapping[str, Any]) -> str:
    from algotrading.refdata.options.Pricing import configured_methods

    calculation_prefix = OPTIONS_CALCS_CONFIG_PREFIX
    greeks_prefix = OPTIONS_CALCS_GREEKS_PREFIX
    surface_prefix = OPTIONS_CALCS_SURFACE_PREFIX
    shared_precision_names = {
        calculation_prefix + "calculation_definition_id",
        calculation_prefix + "expected_multiplier",
        calculation_prefix + "short_tenor_max_seconds",
        calculation_prefix + "short_tenor_dtype",
        calculation_prefix + "other_tenor_dtype",
        calculation_prefix + "auto_promote_to_float64",
        calculation_prefix + "surface_definition_id",
    }
    affecting = {
        key: context[key]
        for key in sorted(context)
        if (
            key.startswith(surface_prefix)
            and not is_operational_options_calcs_key(key)
        )
        or key in shared_precision_names
    }
    routing = configured_methods(context)
    affecting.update(
        {
            "routing.method": routing.default_method.value,
            "routing.short_tenor_method": routing.short_tenor_method.value,
            "routing.short_tenor_threshold_days": routing.short_tenor_threshold_days,
            "routing.price_basis": context.get(
                greeks_prefix + "price_basis", "market_mid"
            ),
        }
    )
    active_prefixes = {
        greeks_prefix + "iv.",
        greeks_prefix + routing.default_method.value + ".",
        greeks_prefix + routing.short_tenor_method.value + ".",
    }
    affecting.update(
        {
            key: context[key]
            for key in sorted(context)
            if any(key.startswith(active_prefix) for active_prefix in active_prefixes)
            and not is_operational_options_calcs_key(key)
        }
    )
    return hashlib.sha256(_compact_json(affecting).encode("utf-8")).hexdigest()


def _solver_configuration_hash(
    context: Mapping[str, Any], method: str, requested_dtype: str = "float64"
) -> str:
    """Hash numerical inputs for one engine, excluding process orchestration."""
    from algotrading.refdata.options.Pricing import configured_methods

    prefix = OPTIONS_CALCS_CONFIG_PREFIX
    method_prefix = OPTIONS_CALCS_GREEKS_PREFIX + method + "."
    shared = {
        prefix + "auto_promote_to_float64",
        OPTIONS_CALCS_GREEKS_PREFIX + "price_basis",
        OPTIONS_CALCS_GREEKS_PREFIX + "dividend_model",
    }
    affecting = {
        key: context[key]
        for key in sorted(context)
        if key in shared
        or key.startswith(method_prefix)
        or key.startswith(OPTIONS_CALCS_GREEKS_PREFIX + "iv.")
        or key.startswith(OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.")
    }
    routing = configured_methods(context)
    affecting.update(
        {
            "effective_requested_dtype": str(requested_dtype),
            "routing.method": routing.default_method.value,
            "routing.short_tenor_method": routing.short_tenor_method.value,
            "routing.short_tenor_threshold_days": routing.short_tenor_threshold_days,
        }
    )
    return hashlib.sha256(_compact_json(affecting).encode("utf-8")).hexdigest()


def _surface_solver_provenance(
    context: Mapping[str, Any],
    *,
    exercise_type: str,
    tenor_seconds: float,
    solver_registry: Optional[Any] = None,
) -> tuple[str, str, str]:
    from algotrading.refdata.options.Pricing import configured_methods, select_method
    from algotrading.refdata.options.solvers.Types import RepricingMethod

    routing = configured_methods(context)
    selection = select_method(
        exercise_type,
        tenor_seconds,
        routing=routing,
    )
    validation_method = str(
        context.get(
            OPTIONS_CALCS_CONFIG_PREFIX + "surface.node_validation_method",
            "configured",
        )
    ).strip().lower()
    if (
        str(exercise_type).strip().upper() in {"A", "AMERICAN"}
        and validation_method != "configured"
    ):
        selection = replace(
            selection,
            method=RepricingMethod.parse(validation_method),
            reason="surface_node_validation_override",
        )
    if selection.method is RepricingMethod.NONE:
        digest = hashlib.sha256(b"source_only_european").hexdigest()
        return "none", "source_provider_v1", digest
    if solver_registry is None:
        raise ValueError(
            "surface solver provenance requires the process-local SolverRegistry"
        )
    adapter = solver_registry.get(selection.method)
    return (
        selection.method.value,
        adapter.algorithm_version,
        _solver_configuration_hash(context, selection.method.value),
    )


def _terminal_failed_fit_row(
    *,
    work: SurfaceWork,
    expiration_at: datetime,
    price_basis: str,
    input_count: int,
    error: Exception,
    context: Mapping[str, Any],
    exercise_type: str = "A",
    calculated_at: datetime,
    solver_registry: Optional[Any] = None,
) -> dict[str, Any]:
    tenor_seconds = solver_tenor_seconds(
        expiration_at, work.source_collected_at
    )
    requested_dtype = resolve_numeric_dtype(
        tenor_seconds,
        NumericPrecisionPolicy.from_context(context).short_tenor_max_seconds,
        NumericPrecisionPolicy.from_context(context).short_tenor_dtype,
        NumericPrecisionPolicy.from_context(context).other_tenor_dtype,
    ).name
    pricing_method, pricing_version, solver_hash = _surface_solver_provenance(
        context,
        exercise_type=exercise_type,
        tenor_seconds=tenor_seconds,
        solver_registry=solver_registry,
    )
    escalation_failed = isinstance(error, PrecisionEscalationFailure)
    staged_error: Optional[Exception] = None
    if isinstance(error, SurfaceStageError) or hasattr(error, "stage"):
        staged_error = error
    elif escalation_failed and (
        isinstance(error.retry_error, SurfaceStageError)
        or hasattr(error.retry_error, "stage")
    ):
        staged_error = error.retry_error
    failure_stage = (
        str(getattr(staged_error, "stage"))
        if staged_error is not None
        else "unknown"
    )
    return {
        "snapshot_id": str(work.snapshot_id),
        "underlying_symbol": work.underlying_symbol,
        "source_collected_at": work.source_collected_at,
        "expiration_at": expiration_at,
        "surface_definition_id": work.surface_definition_id,
        "pricing_method": pricing_method,
        "pricing_algorithm_version": pricing_version,
        "pricing_solver_config_hash": solver_hash,
        "price_basis": price_basis,
        "fit_method": "none",
        "fit_status": (
            "failed_iv_inversion"
            if failure_stage == "iv_inversion"
            else (
                "failed_arbitrage"
                if failure_stage
                in {"butterfly_diagnostic", "calendar_diagnostic"}
                else "failed_calibration"
            )
        ),
        "fit_error": str(error),
        "failure_stage": failure_stage,
        "fit_parameters_json": "{}",
        "input_contract_count": input_count,
        "accepted_contract_count": 0,
        "rejected_contract_count": input_count,
        "fit_candidate_contract_count": 0,
        "fit_selected_contract_count": 0,
        "fit_point_selection_method": "full_set_feature_disabled",
        "fit_point_selection_rounds": 0,
        "fit_full_set_fallback": False,
        "fit_full_set_shadow_checked": False,
        "fit_full_set_shadow_status": "not_checked",
        "fit_full_set_shadow_max_grid_iv_difference": None,
        "fit_full_set_shadow_max_node_iv_difference": None,
        "fit_full_set_shadow_max_node_strike_fraction_difference": None,
        "fit_full_set_shadow_weighted_rmse_difference": None,
        "fit_full_set_acceptance_required": False,
        "fit_full_set_acceptance_passed": False,
        "fit_full_set_acceptance_reason": "not_required",
        "fit_full_set_required_nodes_match": False,
        "fit_full_set_strategy_diagnostics_match": None,
        "fit_full_set_atm_iv_difference": None,
        "fit_full_set_skew_difference": None,
        "fit_full_set_curvature_difference": None,
        "fit_full_set_arbitrage_flags_changed": None,
        "recalculated_current_iv_point_count": 0,
        "recalculated_reused_iv_point_count": 0,
        "source_iv_point_count": 0,
        "no_iv_excluded_count": input_count,
        "source_iv_weight_fraction": 0.0,
        "minimum_input_delta": None,
        "maximum_input_delta": None,
        "minimum_log_forward_moneyness": None,
        "maximum_log_forward_moneyness": None,
        "weighted_rmse": None,
        "maximum_absolute_error": None,
        "butterfly_arbitrage_flags": 0,
        "calendar_arbitrage_flags": 0,
        "quality_flags": 0,
        "requested_dtype": requested_dtype,
        "actual_dtype": "float64" if escalation_failed else requested_dtype,
        "precision_escalated": escalation_failed,
        "precision_escalation_reason": (
            error.escalation_reason if escalation_failed else ""
        ),
        "node_generation_status": "not_requested",
        "required_node_status": "not_requested",
        "optional_node_status": "not_requested",
        "node_generation_error": "",
        "node_requested_dtype": "none",
        "node_actual_dtype": "none",
        "node_precision_escalated": False,
        "node_precision_escalation_reason": "",
        "calculated_at": calculated_at,
    }


def _prepare_surface_owner_for_batch(
    context: Mapping[str, Any],
    claim: Any,
    source_rows: list[dict[str, Any]],
    logger: logging.Logger,
    *,
    warm_start_cache: SVIWarmStartCache,
    numerical_runtime: SurfaceNumericalRuntime,
    solver_registry: Any,
    preloaded_american_inversions: Mapping[Any, Any],
) -> PreparedSurfaceOwner:
    """Build owner-local fits without performing final exact-node work."""
    total_started = time.perf_counter()
    timings = StageTimings()
    timings.initialize(SURFACE_TIMING_NAMES)
    work = claim.work
    if len(source_rows) != work.expected_observations:
        raise InconsistentSnapshotError(
            f"Exact source snapshot {work.snapshot_id} contains "
            f"{len(source_rows)} rows, expected {work.expected_observations}"
        )
    with timings.measure("surface.input_prepare_seconds"):
        by_expiration: dict[datetime, list[dict[str, Any]]] = {}
        for row in source_rows:
            expiration = _parse_timestamp(row.get("expiration_at"))
            if expiration is None:
                raise SnapshotDataError("Surface source row has invalid expiration_at")
            by_expiration.setdefault(expiration, []).append(row)
        if not by_expiration:
            raise SnapshotDataError("Surface source run has no expirations")
        mode = str(
            context.get(
                OPTIONS_CALCS_CONFIG_PREFIX + "surface.iv_inversion_mode",
                "market_mid",
            )
        ).lower()
        configured = {
            "market_mid": ("mid",),
            "market_bid_mid_ask": ("bid", "mid", "ask"),
        }.get(mode)
        if configured is None:
            raise ValueError(f"Unsupported surface IV inversion mode: {mode}")
        eligible_rows = [
            row
            for row in source_rows
            if surface_contract_exclusion_reason(row, context=context) is None
        ]
        # Keep excluded rows in each expiration for rejection counts, but do
        # not let their exercise style determine the vanilla surface's basis.
        # Put eligible rows first for fit-failure provenance as well.
        for expiration_rows in by_expiration.values():
            expiration_rows.sort(
                key=lambda row: surface_contract_exclusion_reason(
                    row, context=context
                ) is not None
            )
        exercise_types = {
            str(row.get("exercise_type", "")).strip().upper()
            for row in (eligible_rows or source_rows[:1])
        }
        if exercise_types.issubset({"E", "EUROPEAN"}):
            price_bases = ("source",)
        elif exercise_types.issubset({"A", "AMERICAN"}):
            price_bases = configured
        elif exercise_types.issubset({"A", "AMERICAN", "E", "EUROPEAN"}):
            raise SnapshotDataError(
                "Surface source run mixes American and European exercise styles"
            )
        else:
            raise SnapshotDataError(
                f"Surface source run has unsupported exercise styles: {sorted(exercise_types)!r}"
            )
        excluded_reasons = Counter(
            reason
            for row in source_rows
            if (reason := surface_contract_exclusion_reason(
                row, context=context
            )) is not None
        )
        if excluded_reasons:
            logger.info(
                "event=surface_contract_exclusions snapshot_id=%s reasons=%s",
                work.snapshot_id,
                dict(sorted(excluded_reasons.items())),
            )

    entries: list[tuple[datetime, str, ExpirationCalibration]] = []
    failed_entries: list[dict[str, Any]] = []
    numerical_diagnostics: list[dict[str, Any]] = []
    for expiration_at in sorted(by_expiration):
        expiration_rows = by_expiration[expiration_at]
        multi = calibrate_expiration_bases(
            expiration_rows,
            price_bases=price_bases,
            context=context,
            underlying_symbol=work.underlying_symbol,
            expiration_at=expiration_at,
            warm_start_cache=warm_start_cache,
            runtime=numerical_runtime,
            generate_nodes=False,
            solver_registry=solver_registry,
            preloaded_american_inversions=preloaded_american_inversions,
        )
        for stage, value in multi.timings.items():
            timing_name = "surface." + stage
            timings.values[timing_name] = timings.values.get(timing_name, 0.0) + value
        numerical_diagnostics.append(
            {"expiration_at": expiration_at.isoformat(), **multi.iv_diagnostics}
        )
        for price_basis in price_bases:
            calibration = multi.calibrations.get(price_basis)
            if calibration is not None:
                entries.append((expiration_at, price_basis, calibration))
                continue
            exc = multi.errors.get(
                price_basis,
                RuntimeError("surface basis produced no terminal result"),
            )
            logger.warning(
                "Surface expiration calibration failed for %s %s: %s",
                expiration_at,
                price_basis,
                exc,
            )
            failed_entries.append(
                _terminal_failed_fit_row(
                    work=work,
                    expiration_at=expiration_at,
                    price_basis=price_basis,
                    input_count=len(expiration_rows),
                    error=exc,
                    context=context,
                    exercise_type=str(expiration_rows[0].get("exercise_type", "A")),
                    calculated_at=claim.started_at,
                    solver_registry=solver_registry,
                )
            )
    with timings.measure("surface.arbitrage_seconds"):
        diagnosed = apply_calendar_arbitrage_diagnostics(
            [entry[2] for entry in entries],
            tolerance=float(
                context.get(
                    OPTIONS_CALCS_CONFIG_PREFIX + "surface.calendar_tolerance",
                    1.0e-9,
                )
            ),
            grid_size=int(
                context.get(
                    OPTIONS_CALCS_CONFIG_PREFIX + "surface.calendar_grid_size", 101
                )
            ),
        )
    entries = [
        (entry[0], entry[1], calibration)
        for entry, calibration in zip(entries, diagnosed)
    ]
    return PreparedSurfaceOwner(
        claim=claim,
        source_rows=source_rows,
        by_expiration=by_expiration,
        price_bases=tuple(price_bases),
        mode=mode,
        entries=entries,
        failed_entries=failed_entries,
        numerical_diagnostics=numerical_diagnostics,
        american_delta_diagnostics=Counter(),
        timings=timings,
        numerical_runtime=numerical_runtime,
        solver_registry=solver_registry,
        total_started=total_started,
    )


def _diagnose_prepared_surface_calendar(
    context: Mapping[str, Any], owner: PreparedSurfaceOwner
) -> None:
    with owner.timings.measure("surface.arbitrage_seconds"):
        diagnosed = apply_calendar_arbitrage_diagnostics(
            [entry[2] for entry in owner.entries],
            tolerance=float(
                context.get(
                    OPTIONS_CALCS_CONFIG_PREFIX + "surface.calendar_tolerance",
                    1.0e-9,
                )
            ),
            grid_size=int(
                context.get(
                    OPTIONS_CALCS_CONFIG_PREFIX + "surface.calendar_grid_size", 101
                )
            ),
        )
    owner.entries = [
        (entry[0], entry[1], calibration)
        for entry, calibration in zip(owner.entries, diagnosed)
    ]


def _generate_prepared_surface_nodes_many(
    context: Mapping[str, Any],
    owners: Sequence[PreparedSurfaceOwner],
    logger: logging.Logger,
    *,
    warm_start_cache: SVIWarmStartCache,
) -> None:
    """Run ordering probes and final node searches as cross-owner waves."""
    ordering_refs: list[tuple[int, datetime]] = []
    ordering_calibrations: list[ExpirationCalibration] = []
    for owner_index, owner in enumerate(owners):
        if owner.mode != "market_bid_mid_ask":
            continue
        indexed = {
            (expiration_at, basis): calibration
            for expiration_at, basis, calibration in owner.entries
        }
        for expiration_at in sorted(owner.by_expiration):
            calibrations = {
                basis: indexed[(expiration_at, basis)]
                for basis in owner.price_bases
                if (expiration_at, basis) in indexed
            }
            if set(calibrations) != set(owner.price_bases):
                continue
            ordering_refs.append((owner_index, expiration_at))
            ordering_calibrations.append(calibrations["mid"])
    if ordering_calibrations:
        probed = generate_calibration_nodes_many(
            ordering_calibrations,
            context=context,
            solver_registry=owners[0].solver_registry,
            runtimes=[owners[index].numerical_runtime for index, _ in ordering_refs],
            diagnostics=[
                owners[index].american_delta_diagnostics for index, _ in ordering_refs
            ],
            stage_timings=[owners[index].timings.values for index, _ in ordering_refs],
        )
    else:
        probed = []
    retried_owners: set[int] = set()
    for (owner_index, expiration_at), primary in zip(ordering_refs, probed):
        owner = owners[owner_index]
        indexed = {
            (entry_expiration, basis): calibration
            for entry_expiration, basis, calibration in owner.entries
        }
        calibrations = {
            basis: indexed[(expiration_at, basis)] for basis in owner.price_bases
        }
        calibrations["mid"] = primary
        merged = merge_bid_mid_ask_nodes(calibrations)
        ordering_failed = any(
            node.quality_flags & QUALITY_NODE_ORDERING for node in merged
        )
        lower_precision_used = any(
            calibration.fit.actual_dtype == "float32"
            for calibration in calibrations.values()
        )
        if not ordering_failed or not lower_precision_used:
            continue
        retried_owners.add(owner_index)
        retry = calibrate_expiration_bases(
            owner.by_expiration[expiration_at],
            price_bases=owner.price_bases,
            context=context,
            underlying_symbol=owner.claim.work.underlying_symbol,
            expiration_at=expiration_at,
            warm_start_cache=warm_start_cache,
            force_float64_reason="bid_mid_ask_node_ordering",
            runtime=owner.numerical_runtime,
            generate_nodes=False,
            solver_registry=owner.solver_registry,
        )
        for stage, value in retry.timings.items():
            name = "surface." + stage
            owner.timings.values[name] = owner.timings.values.get(name, 0.0) + value
        owner.numerical_diagnostics.append(
            {
                "expiration_at": expiration_at.isoformat(),
                "retry": "bid_mid_ask_node_ordering",
                **retry.iv_diagnostics,
            }
        )
        for basis in owner.price_bases:
            key = (expiration_at, basis)
            calibration = retry.calibrations.get(basis)
            if calibration is not None:
                indexed[key] = calibration
                continue
            exc = retry.errors.get(
                basis, RuntimeError("float64 ordering retry produced no result")
            )
            logger.warning(
                "Float64 surface ordering retry failed for %s %s: %s",
                expiration_at,
                basis,
                exc,
            )
            indexed.pop(key, None)
            owner.failed_entries.append(
                _terminal_failed_fit_row(
                    work=owner.claim.work,
                    expiration_at=expiration_at,
                    price_basis=basis,
                    input_count=len(owner.by_expiration[expiration_at]),
                    error=exc,
                    context=context,
                    exercise_type=str(
                        owner.by_expiration[expiration_at][0].get(
                            "exercise_type", "A"
                        )
                    ),
                    calculated_at=owner.claim.started_at,
                    solver_registry=owner.solver_registry,
                )
            )
        owner.entries = [
            (entry_expiration, basis, indexed[(entry_expiration, basis)])
            for entry_expiration in sorted(owner.by_expiration)
            for basis in owner.price_bases
            if (entry_expiration, basis) in indexed
        ]
    for owner_index in sorted(retried_owners):
        _diagnose_prepared_surface_calendar(context, owners[owner_index])

    final_refs: list[tuple[int, int]] = []
    final_calibrations: list[ExpirationCalibration] = []
    for owner_index, owner in enumerate(owners):
        for entry_index, (_expiration, basis, calibration) in enumerate(owner.entries):
            if basis in {"mid", "source"} and calibration.fit.successful:
                final_refs.append((owner_index, entry_index))
                final_calibrations.append(calibration)
    if final_calibrations:
        generated = generate_calibration_nodes_many(
            final_calibrations,
            context=context,
            solver_registry=owners[0].solver_registry,
            runtimes=[owners[index].numerical_runtime for index, _ in final_refs],
            diagnostics=[
                owners[index].american_delta_diagnostics for index, _ in final_refs
            ],
            stage_timings=[owners[index].timings.values for index, _ in final_refs],
        )
        for (owner_index, entry_index), calibration in zip(final_refs, generated):
            expiration_at, basis, _old = owners[owner_index].entries[entry_index]
            owners[owner_index].entries[entry_index] = (
                expiration_at,
                basis,
                calibration,
            )


def prepare_surface_owner(
    context: Mapping[str, Any],
    work: SurfaceWork,
    source_rows: list[dict[str, Any]],
    logger: logging.Logger,
    *,
    warm_start_cache: SVIWarmStartCache,
    numerical_runtime: SurfaceNumericalRuntime,
    solver_registry: Any,
    preloaded_american_inversions: Optional[Mapping[Any, Any]] = None,
) -> PreparedSurfaceOwner:
    """Prepare one generation-3 surface owner for node generation.

    This is the canonical numerical entry point for the process worker.  It
    receives a snapshot owner directly and never creates a legacy scheduler
    claim or performs persistence.
    """
    runtime_context = dict(context)
    runtime_context[OPTIONS_CALCS_CONFIG_PREFIX + "calculation_definition_id"] = (
        work.calculation_definition_id
    )
    runtime_context[OPTIONS_CALCS_CONFIG_PREFIX + "surface_definition_id"] = (
        work.surface_definition_id
    )
    owner = _surface_numerical_owner_claim(work, _utc_now())
    return _prepare_surface_owner_for_batch(
        runtime_context,
        owner,
        source_rows,
        logger,
        warm_start_cache=warm_start_cache,
        numerical_runtime=numerical_runtime,
        solver_registry=solver_registry,
        preloaded_american_inversions=preloaded_american_inversions or {},
    )


def generate_surface_owner_nodes(
    context: Mapping[str, Any],
    owner: PreparedSurfaceOwner,
    logger: logging.Logger,
    *,
    warm_start_cache: SVIWarmStartCache,
) -> None:
    """Generate all exact nodes for one already-prepared owner."""
    with owner.timings.measure("surface.node_generation_seconds"):
        _generate_prepared_surface_nodes_many(
            context, [owner], logger, warm_start_cache=warm_start_cache
        )


class DailyAppendFileHandler(logging.Handler):
    def __init__(self, directory: Path):
        super().__init__()
        self.directory = directory
        self._date: Optional[date] = None
        self._stream = None
        self._lock = threading.RLock()

    def _ensure_stream(self) -> None:
        current_date = datetime.now().date()
        if current_date == self._date and self._stream is not None:
            return
        if self._stream is not None:
            self._stream.close()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = (
            self.directory / f"options_chain_calc_{current_date:%Y_%m_%d}.log"
        )
        self._stream = path.open("a", encoding="utf-8")
        self._date = current_date

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with self._lock:
                self._ensure_stream()
                self._stream.write(self.format(record) + "\n")
                self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        super().close()


def configure_logging(context: Mapping[str, Any]) -> logging.Logger:
    logger = logging.getLogger("OptionsChainWorker")
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s|%(levelname)s|%(processName)s|%(process)d|%(threadName)s| %(message)s"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = DailyAppendFileHandler(Path(context[PARAM_LOG_PATH]))
    file_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.setLevel(context.get(PARAM_DEFAULT_LOG_LEVEL, logging.INFO))
    return logger


def default_context(algo_home: str) -> dict[str, Any]:
    from algotrading.refdata.options.Reuse import (
        default_failure_tenor_bands,
        default_seed_tenor_bands,
        default_tenor_bands,
    )

    prefix = OPTIONS_CALCS_CONFIG_PREFIX
    return {
        PARAM_ALGO_HOME: algo_home,
        PARAM_LOG_PATH: os.path.join(algo_home, "logs"),
        PARAM_DEFAULT_LOG_LEVEL: logging.INFO,
        "options_chain.clickhouse.host": "localhost",
        "options_chain.clickhouse.port": 8123,
        "options_chain.clickhouse.username": "default",
        "options_chain.clickhouse.password": "",
        "options_chain.clickhouse.database": "algotrading",
        "options_chain.clickhouse.secure": False,
        "options_chain.clickhouse.connect_timeout": 10,
        "options_chain.clickhouse.send_receive_timeout": 120,
        prefix + "create_schema": True,
        prefix + "code_version": "options-calculations-v2",
        prefix + "poll_interval_seconds": 10,
        prefix + "contract.pending_batch_size": 100,
        prefix + "surface.pending_batch_size": 16,
        # Fit one surface for every N snapshots per symbol; 1 preserves every snapshot.
        prefix + "surface.cadence_multiplier": 1,
        # Do not backfill a newly eligible cadence gap from stale chains.
        prefix + "surface.max_minutes_for_calculation": 120,
        prefix + "reconcile_days": 7,
        prefix + "retry_failed": False,
        prefix + "running_stale_seconds": 900,
        prefix + "max_calculation_lag_seconds": 600,
        prefix + "quote_max_age_seconds": 60,
        prefix + "max_spread_fraction_of_mid": 0.50,
        prefix + "expected_multiplier": 100,
        prefix + "dividend_yield_default": 0.0,
        prefix + "worker_id": socket.gethostname(),
        prefix + "workers": 1,
        prefix + "max_inflight": 0,
        prefix + "surface.max_inflight": 0,
        prefix + "worker.max_tasks_per_child": 500,
        prefix + "worker.shutdown_timeout_seconds": 60,
        prefix + "worker.cpu_threads_per_process": 1,
        prefix + "monitor_interval": 300,
        prefix + "short_tenor_max_seconds": 86400,
        prefix + "short_tenor_dtype": "float64",
        prefix + "other_tenor_dtype": "float32",
        prefix + "auto_promote_to_float64": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "enabled": False,
        OPTIONS_CALCS_GREEKS_PREFIX + "num_contracts_per_expiration": 20,
        OPTIONS_CALCS_GREEKS_PREFIX + "method": "crank_nicolson",
        OPTIONS_CALCS_GREEKS_PREFIX + "short_tenor_method": "tian",
        OPTIONS_CALCS_GREEKS_PREFIX + "short_tenor_threshold_days": 3.0,
        OPTIONS_CALCS_GREEKS_PREFIX + "price_basis": "market_mid",
        OPTIONS_CALCS_GREEKS_PREFIX + "dividend_model": "continuous_yield",
        OPTIONS_CALCS_GREEKS_PREFIX + "iv.minimum": 0.0001,
        OPTIONS_CALCS_GREEKS_PREFIX + "iv.maximum": 10.0,
        OPTIONS_CALCS_GREEKS_PREFIX + "iv.price_tolerance": 1.0e-7,
        OPTIONS_CALCS_GREEKS_PREFIX + "iv.max_iterations": 100,
        OPTIONS_CALCS_GREEKS_PREFIX + "iv.source_bracket_multipliers": [
            0.50, 0.75, 1.00, 1.25, 1.50
        ],
        OPTIONS_CALCS_GREEKS_PREFIX + "iv.prior_seed_bracket_multipliers": [
            0.75, 0.90, 1.00, 1.10, 1.25
        ],
        OPTIONS_CALCS_GREEKS_PREFIX + "iv.hint_identifiability.enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "iv.hint_identifiability.probe_iv_points": 1.0,
        OPTIONS_CALCS_GREEKS_PREFIX + "iv.hint_identifiability.minimum_price_span_absolute": 1.0e-6,
        OPTIONS_CALCS_GREEKS_PREFIX + "iv.hint_identifiability.price_tolerance_multiplier": 10.0,
        OPTIONS_CALCS_GREEKS_PREFIX + "crr.device": "cpu",
        OPTIONS_CALCS_GREEKS_PREFIX + "crr.steps": 256,
        OPTIONS_CALCS_GREEKS_PREFIX + "crr.chunk_size": 50000,
        OPTIONS_CALCS_GREEKS_PREFIX + "crr.fused_greeks_enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "crr.fused_greeks_maximum_contracts": 160,
        OPTIONS_CALCS_GREEKS_PREFIX + "tian.device": "cpu",
        OPTIONS_CALCS_GREEKS_PREFIX + "tian.steps": 256,
        OPTIONS_CALCS_GREEKS_PREFIX + "tian.chunk_size": 50000,
        OPTIONS_CALCS_GREEKS_PREFIX + "tian.fused_greeks_enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "tian.fused_greeks_maximum_contracts": 0,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.maximum_jobs": 2048,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.maximum_estimated_bytes": 536870912,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float64_retry_enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.exception_isolation_enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.scalar_rollback_enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.short_tenor_validation_enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.shadow_fraction": 0.01,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.minimum_signal_to_roundoff": 256.0,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.price_atol": 2.5e-5,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.delta_atol": 2.5e-4,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.gamma_atol": 2.5e-5,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.theta_per_day_atol": 2.5e-4,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.vega_per_1pct_atol": 2.5e-4,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.rho_per_1pct_atol": 2.5e-4,
        OPTIONS_CALCS_GREEKS_PREFIX + "batch.float32_guard.rtol": 5.0e-4,
        OPTIONS_CALCS_GREEKS_PREFIX + "tian.vectorized_enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "tian.scalar_fallback_enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "crank_nicolson.time_grid": 100,
        OPTIONS_CALCS_GREEKS_PREFIX + "crank_nicolson.space_grid": 200,
        OPTIONS_CALCS_GREEKS_PREFIX + "crank_nicolson.damping_steps": 4,
        OPTIONS_CALCS_GREEKS_PREFIX + "crank_nicolson.vega_relative_bump": 0.0025,
        OPTIONS_CALCS_GREEKS_PREFIX + "crank_nicolson.vega_minimum_bump": 0.00025,
        OPTIONS_CALCS_GREEKS_PREFIX + "crank_nicolson.vega_maximum_bump": 0.001,
        OPTIONS_CALCS_GREEKS_PREFIX + "crank_nicolson.rho_bump": 0.0001,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.enabled": False,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.target_price_absolute_tolerance": 1.0e-8,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.maximum_rate_change": 1.0e-6,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.maximum_dividend_yield_change": 1.0e-8,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.maximum_estimated_price_drift": 0.005,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.maximum_estimated_iv_change_points": 0.02,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.vega_floor": 1.0e-6,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.lookup_allowance_seconds": 60,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.tenor_bands": default_tenor_bands(),
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.shadow.enabled": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.shadow.sample_fraction": 0.05,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.shadow.short_tenor_seconds": 86400,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.shadow.short_tenor_sample_fraction": 0.25,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.shadow.seed": 0,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.shadow.maximum_iv_error": 0.0005,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.shadow.maximum_delta_error": 0.0025,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.seeded_refresh.enabled": False,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.seeded_refresh.maximum_target_price_absolute_change": 0.25,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.seeded_refresh.maximum_target_price_fractional_change": 0.20,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.seeded_refresh.maximum_rate_change": 0.0005,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.seeded_refresh.maximum_dividend_yield_change": 0.0005,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.seeded_refresh.tenor_bands": default_seed_tenor_bands(),
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.seeded_refresh.shadow.enabled": False,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.seeded_refresh.shadow.sample_fraction": 0.02,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.seeded_refresh.shadow.seed": 0,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.seeded_refresh.shadow.maximum_iv_difference": 0.0005,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.seeded_refresh.shadow.maximum_price_residual_difference": 1.0e-7,
        OPTIONS_CALCS_GREEKS_PREFIX + "failure_reuse.enabled": False,
        OPTIONS_CALCS_GREEKS_PREFIX + "failure_reuse.allowed_retry_classes": [
            "deterministic_numerical", "transient_solver"
        ],
        OPTIONS_CALCS_GREEKS_PREFIX + "failure_reuse.allowed_reasons": [
            "iv_root_not_bracketed", "iv_max_iterations",
            "iv_price_tolerance_not_reached",
            "price_residual_exceeds_tolerance", "nonfinite_solver_output",
        ],
        OPTIONS_CALCS_GREEKS_PREFIX + "failure_reuse.maximum_target_price_absolute_change": 1.0e-8,
        OPTIONS_CALCS_GREEKS_PREFIX + "failure_reuse.maximum_target_price_fractional_change": 0.0,
        OPTIONS_CALCS_GREEKS_PREFIX + "failure_reuse.maximum_rate_change": 1.0e-6,
        OPTIONS_CALCS_GREEKS_PREFIX + "failure_reuse.maximum_dividend_yield_change": 1.0e-8,
        OPTIONS_CALCS_GREEKS_PREFIX + "failure_reuse.maximum_failure_age_seconds": 900,
        OPTIONS_CALCS_GREEKS_PREFIX + "failure_reuse.transient_retry_seconds": 30,
        OPTIONS_CALCS_GREEKS_PREFIX + "failure_reuse.lookup_allowance_seconds": 60,
        OPTIONS_CALCS_GREEKS_PREFIX + "failure_reuse.tenor_bands": default_failure_tenor_bands(),
        OPTIONS_CALCS_GREEKS_PREFIX + "circuit_breaker.enabled": False,
        OPTIONS_CALCS_GREEKS_PREFIX + "circuit_breaker.window_seconds": 60,
        OPTIONS_CALCS_GREEKS_PREFIX + "circuit_breaker.minimum_attempts": 3,
        OPTIONS_CALCS_GREEKS_PREFIX + "circuit_breaker.minimum_infrastructure_failures": 3,
        OPTIONS_CALCS_GREEKS_PREFIX + "circuit_breaker.failure_fraction": 0.75,
        OPTIONS_CALCS_GREEKS_PREFIX + "circuit_breaker.initial_open_seconds": 30,
        OPTIONS_CALCS_GREEKS_PREFIX + "circuit_breaker.maximum_open_seconds": 600,
        OPTIONS_CALCS_GREEKS_PREFIX + "circuit_breaker.open_backoff_multiplier": 2.0,
        OPTIONS_CALCS_GREEKS_PREFIX + "circuit_breaker.half_open_probe_count": 1,
        OPTIONS_CALCS_GREEKS_PREFIX + "circuit_breaker.allow_worker_start_with_open_method": True,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.calibration.maximum_p99_iv_error": 0.0005,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.calibration.maximum_p99_delta_error": 0.0025,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.calibration.maximum_max_iv_error": 0.0020,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.calibration.minimum_holdout_observations": 100,
        OPTIONS_CALCS_GREEKS_PREFIX + "reuse.calibration.maximum_candidate_combinations": 4096,
        OPTIONS_CALCS_GREEKS_PREFIX + "nagios_window_seconds": 1800,
        OPTIONS_CALCS_GREEKS_PREFIX + "nagios_min_failure_count": 5,
        OPTIONS_CALCS_GREEKS_PREFIX + "max_unrecoverable_failure_fraction": 0.02,
        OPTIONS_CALCS_GREEKS_PREFIX + "max_recoverable_failure_fraction": 0.20,
        OPTIONS_CALCS_GREEKS_PREFIX + "max_batch_exceptions": 0,
        OPTIONS_CALCS_GREEKS_PREFIX + "consecutive_failed_snapshot_threshold": 3,
        prefix + "surface.enabled": True,
        prefix + "surface.iv_batch.enabled": True,
        prefix + "surface.delta_batch.enabled": True,
        prefix + "surface.iv_inversion_mode": "market_mid",
        prefix + "surface.minimum_input_points": 7,
        prefix + "surface.minimum_otm_put_points": 2,
        prefix + "surface.minimum_otm_call_points": 2,
        prefix + "surface.minimum_option_price": 0.01,
        prefix + "surface.minimum_tenor_seconds": 60,
        prefix + "surface.maximum_quote_age_seconds": 60,
        prefix + "surface.maximum_spread_fraction_of_mid": 0.75,
        prefix + "surface.maximum_abs_log_forward_moneyness": 0.50,
        prefix + "surface.near_atm_log_moneyness": 0.025,
        prefix + "surface.include_nonstandard": False,
        prefix + "surface.include_minis": False,
        prefix + "surface.allow_truncated_snapshots": False,
        prefix + "surface.weighting_method": "inverse_spread_vega",
        prefix + "surface.svi_max_iterations": 2000,
        prefix + "surface.maximum_weighted_rmse": 0.02,
        prefix + "surface.allow_pchip_fallback": True,
        prefix + "surface.arbitrage_grid_size": 401,
        prefix + "surface.butterfly_tolerance": 1.0e-9,
        prefix + "surface.calendar_grid_size": 101,
        prefix + "surface.calendar_tolerance": 1.0e-9,
        prefix + "surface.delta_nodes": [0.40, 0.25, 0.10],
        prefix + "surface.log_forward_moneyness_nodes": [
            -0.20,
            -0.10,
            -0.05,
            0.0,
            0.05,
            0.10,
            0.20,
        ],
        prefix + "surface.node_delta_tolerance": 1.0e-5,
        prefix + "surface.node_max_iterations": 100,
        # American delta nodes use Tian only to propose strikes.  The normal
        # configured American method remains authoritative for publication.
        prefix + "surface.node_hybrid_enabled": True,
        prefix + "surface.node_search_method": "tian",
        prefix + "surface.node_validation_method": "configured",
        prefix + "surface.node_tian_grid_size": 81,
        prefix + "surface.node_validation_max_evaluations": 4,
        prefix + "surface.node_validation_initial_candidates": 1,
        prefix + "surface.node_legacy_exact_fallback_enabled": True,
        # Legacy-only A/B controls.  Hybrid mode never evaluates this dense
        # authoritative grid.
        prefix + "surface.node_search_grid_size": 21,
        prefix + "surface.node_exact_refinement": True,
        prefix + "surface.node_exact_max_evaluations": 8,
        prefix + "surface.node_dense_exact_fallback_enabled": True,
        prefix + "surface.node_dense_exact_fallback_grid_size": 81,
        prefix + "surface.svi_warm_start_cache_size": 2048,
        prefix + "surface.fit_points.enabled": False,
        prefix + "surface.fit_points.require_full_set_acceptance": True,
        prefix + "surface.fit_points.initial_max_points_per_wing": 8,
        prefix + "surface.fit_points.expansion_points_per_wing": 4,
        prefix + "surface.fit_points.maximum_expansion_rounds": 3,
        prefix + "surface.fit_points.minimum_points_per_wing": 3,
        prefix + "surface.fit_points.minimum_total_points": 7,
        prefix + "surface.fit_points.maximum_weighted_rmse_increase": 0.0005,
        prefix + "surface.fit_points.maximum_grid_iv_difference": 0.0010,
        prefix + "surface.fit_points.maximum_node_iv_difference": 0.0010,
        prefix + "surface.fit_points.maximum_node_strike_fraction_difference": 0.0010,
        prefix + "surface.fit_points.maximum_atm_iv_difference": 0.0010,
        prefix + "surface.fit_points.maximum_skew_difference": 0.0010,
        prefix + "surface.fit_points.maximum_curvature_difference": 0.0010,
        prefix + "surface.fit_points.full_set_shadow.enabled": False,
        prefix + "surface.fit_points.full_set_shadow.sample_fraction": 0.05,
        prefix + "surface.fit_points.full_set_shadow.seed": 0,
        prefix + "surface.float32_input_residual_threshold": 1.0e-4,
        prefix + "surface.max_surface_lag_seconds": 600,
        prefix + "surface.max_failed_expiration_fraction": 0.10,
        prefix + "surface.max_node_failure_fraction": 0.10,
        prefix + "surface.max_required_node_numerical_failure_fraction": 0.02,
        prefix + "surface.max_required_node_unavailable_fraction": 0.05,
        prefix + "surface.max_required_node_outside_coverage_fraction": 0.35,
        prefix + "surface.warn_on_optional_node_outside_coverage": False,
        prefix + "surface.max_symbol_required_node_unavailable_fraction": 0.25,
        prefix + "surface.warn_on_zero_nodes_for_successful_run": True,
        prefix + "fred.api_key": "",
        prefix + "fred.max_curve_age_days": 5,
        prefix + "fred.history_days": 3650,
        prefix + "fred.convert_to_continuous": True,
        prefix + "fred.series": "SOFR,DGS1MO,DGS3MO,DGS6MO,DGS1",
        prefix + "fred.request_timeout": 15,
        prefix + "fred.refresh_retry_seconds": 60,
        prefix + "fred.trading_calendar": "XNYS",
    }


@dataclass(frozen=True)
class ContractCoverageExpansionReport:
    snapshots_examined: int
    selected_contracts: int
    existing_results: int
    existing_contract_failures: int
    candidate_contracts: int
    successful_results_written: int
    contract_failures_written: int
    states_reconciled: int
    surface_snapshot_ids: tuple[uuid.UUID, ...]
    dry_run: bool


def _definition_configuration_for_coverage(
    context: Mapping[str, Any], *, surface: bool
) -> dict[str, Any]:
    prefix = OPTIONS_CALCS_SURFACE_PREFIX if surface else OPTIONS_CALCS_CONFIG_PREFIX
    return {
        str(key): context[key]
        for key in sorted(context)
        if str(key).startswith(prefix)
        and not is_operational_options_calcs_key(str(key))
        and (
            surface
            or (
                not str(key).startswith(OPTIONS_CALCS_SURFACE_PREFIX)
                and not str(key).startswith(OPTIONS_CALCS_GREEKS_PREFIX)
            )
        )
    }


def _solver_configuration_for_coverage(
    context: Mapping[str, Any], *, contract_limit: int
) -> dict[str, Any]:
    configuration = {
        str(key): context[key]
        for key in sorted(context)
        if str(key).startswith(OPTIONS_CALCS_GREEKS_PREFIX)
        and not is_operational_options_calcs_key(str(key))
    }
    configuration[
        OPTIONS_CALCS_GREEKS_PREFIX + "num_contracts_per_expiration"
    ] = int(contract_limit)
    return configuration


def _validate_coverage_definition_identity(
    client: Any,
    database: str,
    context: Mapping[str, Any],
    calculation_definition_ids: Collection[int],
    *,
    previous_contract_limit: int,
) -> None:
    definition_ids = sorted({int(value) for value in calculation_definition_ids})
    if not definition_ids:
        return
    rows = client.query_arrow(
        f"""
        SELECT definition_id, code_version, configuration_hash,
               solver_configuration_hash
        FROM {database}.options_chain_definitions FINAL
        WHERE definition_type = 'calculation'
          AND definition_id IN {{definition_ids:Array(UInt32)}}
          AND is_active
        """,
        parameters={"definition_ids": definition_ids},
    ).to_pylist()
    stored = {int(row["definition_id"]): row for row in rows}
    missing = sorted(set(definition_ids) - set(stored))
    if missing:
        raise RuntimeError(
            f"Calculation definitions are missing or inactive: {missing}"
        )
    code_version = str(
        context.get(OPTIONS_CALCS_CONFIG_PREFIX + "code_version") or ""
    )
    configuration_hash = digest32(
        canonical_configuration_json(
            _definition_configuration_for_coverage(context, surface=False)
        )
    )
    solver_hash = digest32(
        canonical_configuration_json(
            _solver_configuration_for_coverage(
                context, contract_limit=previous_contract_limit
            )
        )
    )
    current_solver_hash = digest32(
        canonical_configuration_json(
            _solver_configuration_for_coverage(
                context,
                contract_limit=int(
                    context.get(
                        OPTIONS_CALCS_GREEKS_PREFIX
                        + "num_contracts_per_expiration",
                        20,
                    )
                ),
            )
        )
    )
    for definition_id in definition_ids:
        row = stored[definition_id]
        if str(row["code_version"]) != code_version:
            raise RuntimeError(
                f"Calculation definition {definition_id} uses code version "
                f"{row['code_version']!r}, not configured version {code_version!r}"
            )
        if bytes(row["configuration_hash"]) != configuration_hash:
            raise RuntimeError(
                f"Calculation definition {definition_id} has non-Greeks "
                "configuration changes; coverage expansion is not safe"
            )
        if bytes(row["solver_configuration_hash"]) not in {
            solver_hash,
            current_solver_hash,
        }:
            raise RuntimeError(
                f"Calculation definition {definition_id} does not match the "
                "configured solver settings with either the previous or "
                "current num_contracts_per_expiration value. "
                "Only the contract-selection limit may change during expansion."
            )


def _validate_surface_definition_identity(
    client: Any,
    database: str,
    context: Mapping[str, Any],
    snapshot_ids: Sequence[uuid.UUID],
) -> tuple[uuid.UUID, ...]:
    if not snapshot_ids:
        return ()
    rows = client.query_arrow(
        f"""
        SELECT toString(state.snapshot_id) AS snapshot_id,
               state.surface_definition_id,
               state.source_calculation_definition_id,
               definition.code_version, definition.configuration_hash,
               definition.parent_definition_id
        FROM (SELECT * FROM {database}.options_chain_surface_state FINAL) AS state
        INNER JOIN (SELECT * FROM {database}.options_chain_definitions FINAL)
                   AS definition
          ON definition.definition_id = state.surface_definition_id
         AND definition.definition_type = 'surface'
         AND definition.is_active
        WHERE state.snapshot_id IN {{snapshot_ids:Array(UUID)}}
          AND state.is_active
          AND state.status IN ('complete', 'complete_with_failures')
        ORDER BY state.source_collected_at, state.snapshot_id
        """,
        parameters={"snapshot_ids": [str(value) for value in snapshot_ids]},
    ).to_pylist()
    code_version = str(
        context.get(OPTIONS_CALCS_CONFIG_PREFIX + "code_version") or ""
    )
    configuration_hash = digest32(
        canonical_configuration_json(
            _definition_configuration_for_coverage(context, surface=True)
        )
    )
    output: list[uuid.UUID] = []
    for row in rows:
        if str(row["code_version"]) != code_version:
            raise RuntimeError(
                f"Surface definition {row['surface_definition_id']} uses code "
                f"version {row['code_version']!r}, not {code_version!r}"
            )
        if bytes(row["configuration_hash"]) != configuration_hash:
            raise RuntimeError(
                f"Surface definition {row['surface_definition_id']} has "
                "configuration changes; historical surface replacement is not safe"
            )
        parent = row.get("parent_definition_id")
        if parent is None or int(parent) != int(
            row["source_calculation_definition_id"]
        ):
            raise RuntimeError(
                f"Surface definition {row['surface_definition_id']} has an "
                "invalid calculation-definition parent"
            )
        output.append(uuid.UUID(str(row["snapshot_id"])))
    return tuple(output)


def _load_coverage_states(
    client: Any,
    database: str,
    *,
    snapshot_id: Optional[uuid.UUID],
    symbols: Sequence[str],
    from_date: Optional[date],
    to_date: Optional[date],
) -> list[dict[str, Any]]:
    clauses = [
        "state.is_active",
        "state.status IN ('complete', 'complete_with_failures')",
    ]
    parameters: dict[str, Any] = {}
    if snapshot_id is not None:
        clauses.append("state.snapshot_id = {snapshot_id:UUID}")
        parameters["snapshot_id"] = str(snapshot_id)
    if symbols:
        clauses.append("state.underlying_symbol IN {symbols:Array(String)}")
        parameters["symbols"] = [str(value).upper() for value in symbols]
    if from_date is not None:
        start_at = datetime.combine(from_date, datetime.min.time(), tzinfo=timezone.utc)
        clauses.append(
            "toUnixTimestamp64Micro(state.source_collected_at) >= {start_us:Int64}"
        )
        parameters["start_us"] = int(start_at.timestamp() * 1_000_000)
    if to_date is not None:
        end_at = datetime.combine(
            to_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
        )
        clauses.append(
            "toUnixTimestamp64Micro(state.source_collected_at) < {end_us:Int64}"
        )
        parameters["end_us"] = int(end_at.timestamp() * 1_000_000)
    return client.query_arrow(
        f"""
        SELECT toString(state.snapshot_id) AS snapshot_id,
               state.source_collected_at, state.underlying_symbol,
               state.calculation_definition_id,
               toString(state.source_calculation_run_id)
                   AS source_calculation_run_id,
               toString(state.status) AS status, state.expected_observations,
               state.eligible_contracts, state.successful_results,
               state.source_only_contracts, state.terminal_contract_failures,
               state.terminal_expiration_failures,
               state.snapshot_failure_count, state.started_at,
               state.finished_at, state.error_summary
        FROM
        (
            SELECT *
            FROM {database}.options_chain_calculation_state FINAL
        ) AS state
        WHERE {' AND '.join(clauses)}
        ORDER BY state.source_collected_at, state.snapshot_id
        """,
        parameters=parameters,
    ).to_pylist()


def _load_prepared_fred_for_coverage(
    client: Any, database: str, snapshot_id: uuid.UUID, snapshot_date: date
) -> PreparedFredSnapshot:
    rows = client.query_arrow(
        f"""
        SELECT expiration_at, risk_free_rate, fred_observation_date,
               fred_lower_series, fred_upper_series
        FROM {authoritative_calculation_fact_subquery(database, 'options_chain_expiration_inputs', fact_alias='i', state_alias='calculation_state')}
        WHERE i.snapshot_id = {{snapshot_id:UUID}}
        ORDER BY expiration_at
        """,
        parameters={"snapshot_id": str(snapshot_id)},
    ).to_pylist()
    rates = {
        _parse_timestamp(row["expiration_at"]): PreparedFredRate(
            float(row["risk_free_rate"]),
            row["fred_observation_date"],
            str(row["fred_lower_series"]),
            str(row["fred_upper_series"]),
        )
        for row in rows
    }
    return PreparedFredSnapshot(snapshot_date, rates)


def _load_contract_outcome_ids(
    client: Any, database: str, snapshot_id: uuid.UUID
) -> tuple[set[int], set[int]]:
    result_rows = client.query(
        f"""
        SELECT groupUniqArray(toUInt64(r.contract_id))
        FROM {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_results', fact_alias='r', state_alias='calculation_state')}
        WHERE r.snapshot_id = {{snapshot_id:UUID}}
        """,
        parameters={"snapshot_id": str(snapshot_id)},
    ).result_rows
    failure_rows = client.query(
        f"""
        SELECT groupUniqArray(toUInt64(assumeNotNull(f.contract_id)))
        FROM {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_failures', fact_alias='f', state_alias='calculation_state', require_active_fact=True)}
        WHERE f.snapshot_id = {{snapshot_id:UUID}}
          AND f.failure_scope = 'contract' AND f.is_active
          AND isNotNull(f.contract_id)
        """,
        parameters={"snapshot_id": str(snapshot_id)},
    ).result_rows
    results = {
        int(value) for value in ((result_rows[0][0] if result_rows else None) or [])
    }
    failures = {
        int(value) for value in ((failure_rows[0][0] if failure_rows else None) or [])
    }
    return results, failures


def _validate_expandable_state(
    state: Mapping[str, Any],
    source_row_count: int,
    result_ids: Collection[int],
    failure_ids: Collection[int],
) -> tuple[bool, int]:
    snapshot_id = state["snapshot_id"]
    if int(state["terminal_expiration_failures"]) or int(
        state["snapshot_failure_count"]
    ):
        raise RuntimeError(
            f"Snapshot {snapshot_id} has expiration/snapshot failures; "
            "coverage expansion only supports source-only contract outcomes"
        )
    expected = int(state["expected_observations"])
    inferred_source_only = expected - len(result_ids) - len(failure_ids)
    stored_successes = int(state["successful_results"])
    stored_failures = int(state["terminal_contract_failures"])
    stored_source_only = int(state["source_only_contracts"])
    if (
        expected != source_row_count
        or int(state["eligible_contracts"]) != expected
        or stored_successes > len(result_ids)
        or stored_failures > len(failure_ids)
        or stored_source_only
        != expected - stored_successes - stored_failures
        or inferred_source_only < 0
    ):
        raise RuntimeError(
            f"Snapshot {snapshot_id} has inconsistent calculation counters; "
            "refusing additive coverage expansion"
        )
    needs_reconciliation = any(
        (
            stored_successes != len(result_ids),
            stored_failures != len(failure_ids),
            stored_source_only != inferred_source_only,
        )
    )
    return needs_reconciliation, len(result_ids) - stored_successes


def _write_contract_coverage_results(
    client: Any,
    database: str,
    state: Mapping[str, Any],
    *,
    target_contract_ids: set[int],
    results: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    worker_id: str,
    configured_limit: int,
) -> tuple[int, int, bool]:
    snapshot_id = uuid.UUID(str(state["snapshot_id"]))
    lineage_text = str(state.get("source_calculation_run_id") or "").strip()
    lineage = lineage_text if lineage_text.lower() not in {"", "none", "null"} else None
    new_results = [
        {**dict(row), "source_calculation_run_id": lineage}
        for row in results
        if int(row["contract_id"]) in target_contract_ids
    ]
    new_failures = [
        {**dict(row), "source_calculation_run_id": lineage}
        for row in failures
        if row.get("failure_scope") == "contract"
        and row.get("contract_id") is not None
        and int(row["contract_id"]) in target_contract_ids
    ]
    accounted = {
        int(row["contract_id"]) for row in new_results
    } | {
        int(row["contract_id"]) for row in new_failures
    }
    if accounted != target_contract_ids:
        missing = sorted(target_contract_ids - accounted)
        raise RuntimeError(
            f"Coverage expansion did not produce a terminal outcome for "
            f"snapshot {snapshot_id}: {missing[:20]}"
        )
    token_suffix = hashlib.sha256(
        ",".join(str(value) for value in sorted(target_contract_ids)).encode()
    ).hexdigest()
    if new_results:
        client.insert_arrow(
            f"{database}.options_chain_calculation_results",
            _rows_to_arrow(new_results, RESULT_SCHEMA),
            settings={
                "insert_deduplication_token": (
                    f"coverage-results:{snapshot_id}:{configured_limit}:{token_suffix}"
                )
            },
        )
    if new_failures:
        client.insert_arrow(
            f"{database}.options_chain_calculation_failures",
            _rows_to_arrow(new_failures, CALCULATION_FAILURE_SCHEMA),
        )

    result_ids, failure_ids = _load_contract_outcome_ids(
        client, database, snapshot_id
    )
    expected = int(state["expected_observations"])
    source_only = expected - len(result_ids) - len(failure_ids)
    if source_only < 0:
        raise RuntimeError(
            f"Expanded outcomes exceed expected contracts for {snapshot_id}"
        )
    status = "complete_with_failures" if failure_ids else "complete"
    counters_changed = any(
        (
            int(state["successful_results"]) != len(result_ids),
            int(state["terminal_contract_failures"]) != len(failure_ids),
            int(state["source_only_contracts"]) != source_only,
            str(state["status"]) != status,
        )
    )
    if target_contract_ids or counters_changed:
        now = _utc_now()
        state_row = {
            "snapshot_id": str(snapshot_id),
            "source_collected_at": state["source_collected_at"],
            "underlying_symbol": str(state["underlying_symbol"]),
            "calculation_definition_id": int(
                state["calculation_definition_id"]
            ),
            "source_calculation_run_id": lineage,
            "status": status,
            "expected_observations": expected,
            "eligible_contracts": expected,
            "successful_results": len(result_ids),
            "source_only_contracts": source_only,
            "terminal_contract_failures": len(failure_ids),
            "terminal_expiration_failures": 0,
            "snapshot_failure_count": 0,
            "started_at": state.get("started_at"),
            "finished_at": state.get("finished_at") or now,
            "updated_at": now,
            "active_attempt_id": None,
            "error_summary": str(state.get("error_summary") or ""),
            "is_active": True,
        }
        attempt = {
            "attempt_id": str(uuid.uuid4()),
            "process_type": "calculation",
            "snapshot_id": str(snapshot_id),
            "source_collected_at": state["source_collected_at"],
            "definition_id": int(state["calculation_definition_id"]),
            "worker_id": worker_id,
            "started_at": now,
            "finished_at": now,
            "status": (
                "complete_with_failures" if new_failures else "complete"
            ),
            "expected_count": len(target_contract_ids),
            "written_success_count": len(new_results),
            "failure_count": len(new_failures),
            "error_summary": "contract_coverage_expansion",
            "created_at": now,
        }
        client.insert_arrow(
            f"{database}.options_chain_processing_attempts",
            _rows_to_arrow([attempt], ATTEMPT_SCHEMA),
        )
        client.insert_arrow(
            f"{database}.options_chain_calculation_state",
            _rows_to_arrow([state_row], CALCULATION_STATE_SCHEMA),
        )
    return len(new_results), len(new_failures), counters_changed


def expand_contract_coverage(
    context: Mapping[str, Any],
    client: Any,
    *,
    snapshot_id: Optional[uuid.UUID] = None,
    symbols: Sequence[str] = (),
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    previous_contract_limit: int,
    dry_run: bool,
    confirm_database: Optional[str],
    recalculate_surfaces: bool,
    logger: Optional[logging.Logger] = None,
) -> ContractCoverageExpansionReport:
    """Add newly selected historical contract results without resetting facts."""
    logger = logger or logging.getLogger(__name__)
    database = validate_database_identifier(
        str(context["options_chain.clickhouse.database"])
    )
    if not dry_run and confirm_database != database:
        raise ValueError(
            f"Contract coverage expansion requires --confirm-expand {database}"
        )
    if previous_contract_limit <= 0:
        raise ValueError("--previous-contract-limit must be positive")
    configured_limit = int(
        context.get(
            OPTIONS_CALCS_GREEKS_PREFIX
            + "num_contracts_per_expiration",
            20,
        )
    )
    if configured_limit < previous_contract_limit:
        raise ValueError(
            "Configured num_contracts_per_expiration cannot be smaller than "
            "--previous-contract-limit"
        )
    if not _as_bool(
        context.get(OPTIONS_CALCS_GREEKS_PREFIX + "enabled", False)
    ):
        raise ValueError("Contract IV/Greeks recalculation is disabled")
    running = client.query(
        f"""
        SELECT
          (SELECT count()
           FROM {database}.options_chain_calculation_state FINAL
           WHERE is_active AND status = 'running')
          +
          (SELECT count()
           FROM {database}.options_chain_surface_state FINAL
           WHERE is_active AND status = 'running')
        """
    ).result_rows
    if running and int(running[0][0]):
        raise RuntimeError(
            "Stop the normal Calculations worker before expanding contract coverage; "
            "running calculation or surface claims were found"
        )
    states = _load_coverage_states(
        client,
        database,
        snapshot_id=snapshot_id,
        symbols=symbols,
        from_date=from_date,
        to_date=to_date,
    )
    _validate_coverage_definition_identity(
        client,
        database,
        context,
        {int(state["calculation_definition_id"]) for state in states},
        previous_contract_limit=previous_contract_limit,
    )
    repository = CalculationRepository(
        client,
        database,
        worker_id=str(
            context.get(
                OPTIONS_CALCS_CONFIG_PREFIX + "worker_id",
                socket.gethostname(),
            )
        ),
        stale_after_seconds=int(
            context.get(
                OPTIONS_CALCS_CONFIG_PREFIX + "running_stale_seconds", 900
            )
        ),
    )
    solver_registry = None
    if not dry_run:
        from algotrading.refdata.options.solvers.Registry import SolverRegistry

        solver_registry = SolverRegistry(context)
        solver_registry.preload()

    totals = Counter()
    surface_candidates: list[uuid.UUID] = []
    worker_id = str(
        context.get(
            OPTIONS_CALCS_CONFIG_PREFIX + "worker_id", socket.gethostname()
        )
    )
    for state in states:
        sid = uuid.UUID(str(state["snapshot_id"]))
        source_collected_at = _parse_timestamp(state["source_collected_at"])
        if source_collected_at is None:
            raise RuntimeError(
                f"Snapshot {sid} has no valid source collection timestamp"
            )
        work = CalculationWork(
            sid,
            source_collected_at,
            str(state["underlying_symbol"]),
            int(state["expected_observations"]),
            (),
            int(state["calculation_definition_id"]),
        )
        source_rows = repository.load_source_rows(sid)
        result_ids, failure_ids = _load_contract_outcome_ids(
            client, database, sid
        )
        needs_reconciliation, unaccounted_results = _validate_expandable_state(
            state, len(source_rows), result_ids, failure_ids
        )
        numerical_source = project_canonical_calculation_rows(work, source_rows)
        runtime_context = dict(context)
        runtime_context[
            OPTIONS_CALCS_CONFIG_PREFIX + "calculation_definition_id"
        ] = work.calculation_definition_id
        prepared_fred = _load_prepared_fred_for_coverage(
            client, database, sid, work.source_collected_at.date()
        )
        numerical = calculate_rows(
            runtime_context,
            numerical_source,
            None,
            raw_diagnostics=load_normalized_diagnostics(
                client, database, sid
            ),
            prepared_fred=prepared_fred,
            defer_repricing=True,
        )
        selected_indices = sorted(
            _select_repricing_indices(runtime_context, numerical)
        )
        target_indices = [
            index
            for index in selected_indices
            if int(numerical[index]["_base"]["contract_id"])
            not in result_ids | failure_ids
        ]
        target_ids = {
            int(numerical[index]["_base"]["contract_id"])
            for index in target_indices
        }
        totals["snapshots_examined"] += 1
        totals["selected_contracts"] += len(selected_indices)
        totals["existing_results"] += len(
            {
                int(numerical[index]["_base"]["contract_id"])
                for index in selected_indices
            }
            & result_ids
        )
        totals["existing_contract_failures"] += len(
            {
                int(numerical[index]["_base"]["contract_id"])
                for index in selected_indices
            }
            & failure_ids
        )
        totals["candidate_contracts"] += len(target_indices)
        totals["states_reconciled"] += int(
            dry_run and needs_reconciliation
        )
        logger.info(
            "event=contract_coverage_%s snapshot_id=%s symbol=%s "
            "configured_limit=%s selected=%s existing_results=%s "
            "existing_failures=%s candidates=%s",
            "plan" if dry_run else "start",
            sid,
            work.underlying_symbol,
            configured_limit,
            len(selected_indices),
            len(result_ids),
            len(failure_ids),
            len(target_indices),
        )
        if dry_run:
            if target_indices or unaccounted_results:
                surface_candidates.append(sid)
            continue
        successes = failures_written = 0
        if target_indices:
            diagnostics: dict[str, int] = Counter()
            _apply_repricing(
                runtime_context,
                numerical,
                diagnostics,
                solver_registry=solver_registry,
                prior_outcome_loader=None,
                preselected_indices=target_indices,
            )
            from algotrading.refdata.options.Worker import (
                compose_calculation_publication,
            )

            publication = compose_calculation_publication(
                work, source_rows, numerical
            )
            if publication[5]:
                raise RuntimeError(
                    "Coverage expansion unexpectedly produced reused results"
                )
            results_to_write = publication[0]
            failures_to_write = publication[3]
        else:
            results_to_write = ()
            failures_to_write = ()
        if target_indices or needs_reconciliation:
            successes, failures_written, reconciled = (
                _write_contract_coverage_results(
                    client,
                    database,
                    state,
                    target_contract_ids=target_ids,
                    results=results_to_write,
                    failures=failures_to_write,
                    worker_id=worker_id,
                    configured_limit=configured_limit,
                )
            )
            totals["successful_results_written"] += successes
            totals["contract_failures_written"] += failures_written
            totals["states_reconciled"] += int(reconciled)
            if successes or unaccounted_results:
                surface_candidates.append(sid)
        logger.info(
            "event=contract_coverage_complete snapshot_id=%s symbol=%s "
            "successes=%s failures=%s",
            sid,
            work.underlying_symbol,
            successes,
            failures_written,
        )

    surface_snapshot_ids: tuple[uuid.UUID, ...] = ()
    if recalculate_surfaces:
        surface_snapshot_ids = _validate_surface_definition_identity(
            client,
            database,
            context,
            surface_candidates,
        )
    report = ContractCoverageExpansionReport(
        snapshots_examined=totals["snapshots_examined"],
        selected_contracts=totals["selected_contracts"],
        existing_results=totals["existing_results"],
        existing_contract_failures=totals["existing_contract_failures"],
        candidate_contracts=totals["candidate_contracts"],
        successful_results_written=totals["successful_results_written"],
        contract_failures_written=totals["contract_failures_written"],
        states_reconciled=totals["states_reconciled"],
        surface_snapshot_ids=surface_snapshot_ids,
        dry_run=bool(dry_run),
    )
    logger.info(
        "event=contract_coverage_summary dry_run=%s snapshots=%s "
        "selected=%s candidates=%s successes=%s failures=%s "
        "surface_recalculations=%s recalculate_surfaces=%s",
        report.dry_run,
        report.snapshots_examined,
        report.selected_contracts,
        report.candidate_contracts,
        report.successful_results_written,
        report.contract_failures_written,
        len(report.surface_snapshot_ids),
        bool(recalculate_surfaces),
    )
    return report


def _parse_date(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


def _parse_cli_timestamp(value: str) -> datetime:
    parsed = _parse_timestamp(value)
    if parsed is None:
        raise argparse.ArgumentTypeError(
            "timestamp must be an ISO-8601 date/time"
        )
    return parsed


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate option-chain analytics from normalized ClickHouse snapshots"
    )
    parser.add_argument("-e", required=True, help="configuration environment to use")
    parser.add_argument(
        "--once", action="store_true", help="process the current pending batch and exit"
    )
    parser.add_argument(
        "--create-schema-only",
        action="store_true",
        help="validate source tables, create calculation-owned schema, and exit",
    )
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--contract-only", action="store_true", help="run only contract calculations"
    )
    stage_group.add_argument(
        "--surface-only", action="store_true", help="run only surface calculations"
    )
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--reset-surfaces",
        action="store_true",
        help="drop and recreate only calculation-owned surface objects",
    )
    reset_group.add_argument(
        "--reset-contract-calculations",
        action="store_true",
        help="drop and recreate contract calculations and dependent surfaces",
    )
    reset_group.add_argument(
        "--reset-reference-rates",
        action="store_true",
        help="drop and recreate only durable reference-rate objects",
    )
    reset_group.add_argument(
        "--reset-all-derived",
        action="store_true",
        help="drop and recreate every calculation-owned object",
    )
    parser.add_argument(
        "--dry-run-reset",
        action="store_true",
        help="print the exact ordered reset/recreate plan without executing DDL",
    )
    parser.add_argument(
        "--confirm-reset",
        metavar="DATABASE_NAME",
        help="exact configured database name required for destructive reset execution",
    )
    parser.add_argument(
        "--expand-contract-coverage",
        action="store_true",
        help=(
            "append recalculated IV/Greeks for newly selected contracts in "
            "completed historical snapshots"
        ),
    )
    parser.add_argument(
        "--previous-contract-limit",
        type=int,
        metavar="COUNT",
        help=(
            "historical num_contracts_per_expiration value used to validate "
            "that no other numerical setting changed"
        ),
    )
    parser.add_argument(
        "--dry-run-expand",
        action="store_true",
        help="report contract and surface expansion work without writing",
    )
    parser.add_argument(
        "--confirm-expand",
        metavar="DATABASE_NAME",
        help="exact configured database name required for coverage expansion",
    )
    parser.add_argument(
        "--recalculate-surfaces",
        action="store_true",
        help=(
            "after contract expansion, replace only existing completed "
            "surfaces for snapshots whose contract results changed"
        ),
    )
    parser.add_argument("--snapshot-id", type=uuid.UUID)
    parser.add_argument("--symbols", help="comma-separated symbol filter")
    parser.add_argument("--from-date", type=_parse_date)
    parser.add_argument("--to-date", type=_parse_date)
    parser.add_argument(
        "--retry-failed", action="store_true", help="retry failed attempts"
    )
    options = parser.parse_args(argv)
    reset_selected = any(
        (
            options.reset_surfaces,
            options.reset_contract_calculations,
            options.reset_reference_rates,
            options.reset_all_derived,
        )
    )
    expand_selected = bool(options.expand_contract_coverage)
    if options.dry_run_reset and not reset_selected:
        parser.error("--dry-run-reset requires one reset level")
    if options.confirm_reset and not reset_selected:
        parser.error("--confirm-reset requires one reset level")
    if options.dry_run_expand and not expand_selected:
        parser.error("--dry-run-expand requires --expand-contract-coverage")
    if options.confirm_expand and not expand_selected:
        parser.error("--confirm-expand requires --expand-contract-coverage")
    if options.recalculate_surfaces and not expand_selected:
        parser.error(
            "--recalculate-surfaces requires --expand-contract-coverage"
        )
    if options.previous_contract_limit is not None and not expand_selected:
        parser.error(
            "--previous-contract-limit requires --expand-contract-coverage"
        )
    if expand_selected:
        if options.previous_contract_limit is None:
            parser.error(
                "--expand-contract-coverage requires --previous-contract-limit"
            )
        if options.previous_contract_limit <= 0:
            parser.error("--previous-contract-limit must be positive")
        if not options.dry_run_expand and not options.confirm_expand:
            parser.error(
                "coverage expansion requires --confirm-expand DATABASE_NAME "
                "unless --dry-run-expand is used"
            )
        if options.snapshot_id is None and options.from_date is None:
            parser.error(
                "coverage expansion requires --snapshot-id or --from-date"
            )
        if options.snapshot_id is not None and (
            options.from_date is not None or options.to_date is not None
        ):
            parser.error(
                "--snapshot-id cannot be combined with expansion date filters"
            )
        if any(
            (
                reset_selected,
                options.once,
                options.create_schema_only,
                options.retry_failed,
                options.contract_only,
                options.surface_only,
            )
        ):
            parser.error(
                "coverage expansion cannot be combined with reset or normal "
                "processing options"
            )
    if reset_selected and any(
        (
            expand_selected,
            options.once,
            options.create_schema_only,
            options.snapshot_id,
            options.symbols,
            options.from_date,
            options.to_date,
            options.retry_failed,
            options.contract_only,
            options.surface_only,
        )
    ):
        parser.error("reset operations cannot be combined with processing options")
    if options.create_schema_only and any(
        (
            expand_selected,
            options.once,
            options.snapshot_id,
            options.symbols,
            options.from_date,
            options.to_date,
            options.retry_failed,
            options.contract_only,
            options.surface_only,
        )
    ):
        parser.error("--create-schema-only cannot be combined with processing options")
    if options.from_date and options.to_date and options.from_date > options.to_date:
        parser.error("--from-date cannot be after --to-date")
    return options


def main(argv: Optional[list[str]] = None) -> int:
    options = _parse_args(argv)
    algo_home = os.getenv("ALGO_HOME")
    if not algo_home:
        print("ALGO_HOME environment variable not set", file=sys.stderr)
        return 2
    algo_home = algo_home.rstrip("/") + "/"
    context = load_configuration(
        default_context(algo_home), algo_home, options.e, quiet=True
    )
    validate_options_calcs_configuration(context)

    if 'nagios.service_id' in context:
        context['nagios.service_id'] = context['nagios.service_id'] + " Calculations"

    logger = configure_logging(context)
    context[PARAM_LOGGER] = logger
    application: Optional[Any] = None
    symbols = (
        [
            item.strip().upper()
            for item in options.symbols.split(",")
            if item.strip()
        ]
        if options.symbols
        else []
    )
    try:
        reset_level = next(
            (
                level
                for selected, level in (
                    (options.reset_surfaces, "surfaces"),
                    (
                        options.reset_contract_calculations,
                        "contract_calculations",
                    ),
                    (options.reset_reference_rates, "reference_rates"),
                    (options.reset_all_derived, "all_derived"),
                )
                if selected
            ),
            None,
        )
        if reset_level is not None:
            plan = reset_calculation_schema(
                context,
                reset_level=reset_level,
                confirm_database=options.confirm_reset,
                dry_run=options.dry_run_reset,
                logger=logger,
            )
            logger.warning(
                "%s reset plan for %s (%s objects, %s statements)",
                "Dry-run" if options.dry_run_reset else "Completed",
                plan.database,
                len(plan.objects),
                len(plan.statements),
            )
            return 0
        if options.create_schema_only:
            create_calculation_schema(context)
            logger.info("Calculation ClickHouse schema created successfully")
            return 0
        if _as_bool(
            context.get(OPTIONS_CALCS_CONFIG_PREFIX + "create_schema", True)
        ):
            create_calculation_schema(context)
        if options.expand_contract_coverage:
            client = create_clickhouse_client(context)
            try:
                calculations_preflight(context, client, logger=logger)
                report = expand_contract_coverage(
                    context,
                    client,
                    snapshot_id=options.snapshot_id,
                    symbols=symbols,
                    from_date=options.from_date,
                    to_date=options.to_date,
                    previous_contract_limit=options.previous_contract_limit,
                    dry_run=options.dry_run_expand,
                    confirm_database=options.confirm_expand,
                    recalculate_surfaces=options.recalculate_surfaces,
                    logger=logger,
                )
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            if options.dry_run_expand:
                return 0
            if options.recalculate_surfaces and report.surface_snapshot_ids:
                from algotrading.refdata.options.Worker import OptionsChainWorker

                application = OptionsChainWorker(context, logger=logger)
                signal.signal(signal.SIGINT, application.request_stop)
                signal.signal(signal.SIGTERM, application.request_stop)
                completed_surfaces = 0
                for surface_snapshot_id in report.surface_snapshot_ids:
                    if application.stop_event.is_set():
                        raise RuntimeError(
                            "Surface replacement interrupted by a stop request"
                        )
                    reset = (
                        application.surfaces.reset_terminal_for_recalculation(
                            surface_snapshot_id
                        )
                    )
                    if not reset:
                        raise RuntimeError(
                            f"Expected an existing terminal surface for "
                            f"{surface_snapshot_id}"
                        )
                    application.run(
                        once=True,
                        snapshot_id=surface_snapshot_id,
                        surface_only=True,
                    )
                    rows = application.client.query(
                        f"""
                        SELECT toString(status)
                        FROM {application.database}.options_chain_surface_state FINAL
                        WHERE snapshot_id = {{snapshot_id:UUID}} AND is_active
                        LIMIT 1
                        """,
                        parameters={"snapshot_id": str(surface_snapshot_id)},
                    ).result_rows
                    if not rows or str(rows[0][0]) not in {
                        "complete",
                        "complete_with_failures",
                    }:
                        raise RuntimeError(
                            f"Surface replacement did not complete for "
                            f"{surface_snapshot_id}"
                        )
                    completed_surfaces += 1
                logger.info(
                    "event=contract_coverage_surface_summary "
                    "surfaces_recalculated=%s",
                    completed_surfaces,
                )
            return 0
        from algotrading.refdata.options.Worker import OptionsChainWorker

        application = OptionsChainWorker(context, logger=logger)
        signal.signal(signal.SIGINT, application.request_stop)
        signal.signal(signal.SIGTERM, application.request_stop)
        application.run(
            once=options.once,
            snapshot_id=options.snapshot_id,
            symbols=symbols or None,
            from_date=options.from_date,
            to_date=options.to_date,
            retry_failed=(True if options.retry_failed else None),
            contract_only=options.contract_only,
            surface_only=options.surface_only,
        )
        return 0
    except Exception as exc:
        if application is not None:
            application.health.update(
                worker_alive=False,
                terminal_failure=True,
                last_error=str(exc),
            )
            try:
                application._report_health(final=True)
            except Exception:
                logger.error(
                    "Unable to publish terminal calculation health\n%s",
                    traceback.format_exc(),
                )
        logger.critical(
            "Options calculations command terminated\n%s",
            traceback.format_exc(),
        )
        return 1
    finally:
        if application is not None:
            close = getattr(application, "close", None)
            if callable(close):
                close()
        logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
