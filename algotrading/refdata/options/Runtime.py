"""Canonical calculation and surface work/state publication repositories.

Numerical solvers remain in Calculations.py and Surface.py.  This module is the
only supported persistence boundary for their successful outputs, sparse
failures, assigned definitions, attempts, and terminal state.
"""

from __future__ import annotations

import math
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Sequence

import pyarrow as pa

from algotrading.refdata.options.Common import (
    _datetime64_microseconds,
    _finite_float,
    _parse_timestamp,
    _rows_to_arrow,
    _utc_now,
    validate_database_identifier,
)
from algotrading.refdata.options.Definitions import (
    DefinitionRepository,
    DefinitionType,
)
from algotrading.refdata.options.Queries import (
    authoritative_calculation_fact_subquery,
    build_effective_chain_query,
    calculation_source_projection_columns,
)
from algotrading.refdata.options.Queries import validate_effective_selection
from algotrading.refdata.options.Schema import validate_canonical_table_metadata


TERMINAL_STATES = frozenset({"complete", "complete_with_failures"})


@dataclass(frozen=True)
class CalculationWork:
    snapshot_id: uuid.UUID
    source_collected_at: datetime
    underlying_symbol: str
    expected_observations: int
    expirations: tuple[datetime, ...]
    calculation_definition_id: int
    retry_failed: bool = False


@dataclass(frozen=True)
class SurfaceWork:
    snapshot_id: uuid.UUID
    source_collected_at: datetime
    underlying_symbol: str
    calculation_definition_id: int
    surface_definition_id: int
    expected_observations: int
    retry_failed: bool = False


@dataclass(frozen=True)
class SurfaceCadenceHealth:
    pending: int
    failed: int
    definition_ids: tuple[int, ...]


@dataclass(frozen=True)
class SurfaceWorkAgeReport:
    """The newest automatic surface candidate excluded by an age limit."""

    skipped_count: int
    newest_snapshot_id: uuid.UUID
    newest_underlying_symbol: str
    newest_source_collected_at: datetime


@dataclass(frozen=True)
class InterruptedCalculationWork:
    """A retryable calculation owner found during unbounded startup recovery."""

    work: CalculationWork
    status: str
    attempt_id: Optional[uuid.UUID]


@dataclass(frozen=True)
class InterruptedSurfaceWork:
    """A retryable surface owner found during unbounded startup recovery."""

    work: SurfaceWork
    status: str
    attempt_id: Optional[uuid.UUID]


@dataclass(frozen=True)
class ProcessingClaim:
    attempt_id: uuid.UUID
    process_type: str
    snapshot_id: uuid.UUID
    source_collected_at: datetime
    definition_id: int
    started_at: datetime


@dataclass(frozen=True)
class SnapshotReadiness:
    manifest_rows: int
    expected_observations: int
    observation_rows: int
    distinct_contracts: int
    stored_distinct_contracts: int
    detail_rows: int
    stored_detail_count: int
    missing_contracts: int
    missing_definitions: int

    @property
    def ready(self) -> bool:
        return (
            self.manifest_rows == 1
            and self.expected_observations > 0
            and self.observation_rows == self.expected_observations
            and self.distinct_contracts == self.expected_observations
            and self.stored_distinct_contracts == self.expected_observations
            and self.detail_rows == self.stored_detail_count
            and self.missing_contracts == 0
            and self.missing_definitions == 0
        )


RESULT_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("contract_id", pa.uint64(), nullable=False),
        pa.field("calculation_definition_id", pa.uint32(), nullable=False),
        pa.field("source_calculation_run_id", pa.string()),
        pa.field("calculated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("recalc_method", pa.string(), nullable=False),
        pa.field("result_origin", pa.string(), nullable=False),
        pa.field("origin_snapshot_id", pa.string(), nullable=False),
        pa.field("input_basis", pa.string(), nullable=False),
        pa.field("dividend_model", pa.string(), nullable=False),
        pa.field("iv_selection", pa.string(), nullable=False),
        pa.field("greeks_selection", pa.string(), nullable=False),
        pa.field("target_price", pa.float64(), nullable=False),
        pa.field("iv", pa.float64(), nullable=False),
        pa.field("iv_iterations", pa.uint16(), nullable=False),
        pa.field("price_residual", pa.float64(), nullable=False),
        pa.field("model_price", pa.float64(), nullable=False),
        pa.field("delta", pa.float64(), nullable=False),
        pa.field("gamma", pa.float64(), nullable=False),
        pa.field("theta_per_day", pa.float64(), nullable=False),
        pa.field("vega_per_1pct", pa.float64(), nullable=False),
        pa.field("rho_per_1pct", pa.float64(), nullable=False),
        pa.field("actual_dtype", pa.string(), nullable=False),
        pa.field("precision_escalated", pa.bool_(), nullable=False),
        pa.field("quality_flags", pa.uint32(), nullable=False),
    ]
)

EXPIRATION_INPUT_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("expiration_at", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("calculation_definition_id", pa.uint32(), nullable=False),
        pa.field("time_to_expiry_seconds", pa.uint32(), nullable=False),
        pa.field("risk_free_rate", pa.float64(), nullable=False),
        pa.field("dividend_yield", pa.float64(), nullable=False),
        pa.field("forward_price", pa.float64(), nullable=False),
        pa.field("fred_observation_date", pa.date32(), nullable=False),
        pa.field("fred_lower_series", pa.string(), nullable=False),
        pa.field("fred_upper_series", pa.string(), nullable=False),
        pa.field("quality_flags", pa.uint32(), nullable=False),
        pa.field("calculated_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

CALCULATION_FAILURE_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("calculation_definition_id", pa.uint32(), nullable=False),
        pa.field("source_calculation_run_id", pa.string()),
        pa.field("failure_scope", pa.string(), nullable=False),
        pa.field("expiration_at", pa.timestamp("ms", tz="UTC")),
        pa.field("contract_id", pa.uint64()),
        pa.field("failure_stage", pa.string(), nullable=False),
        pa.field("reason_code", pa.string(), nullable=False),
        pa.field("error_class", pa.string(), nullable=False),
        pa.field("error_detail", pa.string(), nullable=False),
        pa.field("retry_class", pa.string(), nullable=False),
        pa.field("retry_after", pa.timestamp("us", tz="UTC")),
        pa.field("failure_fingerprint", pa.binary(32), nullable=False),
        pa.field("first_failed_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("last_failed_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("failure_count", pa.uint16(), nullable=False),
        pa.field("updated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("is_active", pa.bool_(), nullable=False),
        pa.field("valuation_at", pa.timestamp("us", tz="UTC")),
        pa.field("spot", pa.float64()),
        pa.field("target_price", pa.float64()),
        pa.field("time_to_expiry_seconds", pa.uint32()),
        pa.field("risk_free_rate", pa.float64()),
        pa.field("dividend_yield", pa.float64()),
        pa.field("method", pa.string()),
        pa.field("input_basis", pa.string()),
        pa.field("dividend_model", pa.string()),
        pa.field("partial_iv", pa.float64()),
        pa.field("partial_model_price", pa.float64()),
        pa.field("partial_price_residual", pa.float64()),
        pa.field("seed_iv", pa.float64()),
        pa.field("seed_origin_snapshot_id", pa.string()),
        pa.field("quality_flags", pa.uint32(), nullable=False),
    ]
)

REUSE_PROVENANCE_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("contract_id", pa.uint64(), nullable=False),
        pa.field("calculation_definition_id", pa.uint32(), nullable=False),
        # Reuse provenance is itself a runtime-owned derived fact and has no
        # source-run lineage column in the canonical table.
        pa.field("origin_snapshot_id", pa.string(), nullable=False),
        pa.field("origin_calculated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("origin_valuation_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("origin_spot", pa.float64(), nullable=False),
        pa.field("origin_target_price", pa.float64(), nullable=False),
        pa.field("origin_time_to_expiry_seconds", pa.uint32(), nullable=False),
        pa.field("origin_risk_free_rate", pa.float64(), nullable=False),
        pa.field("origin_dividend_yield", pa.float64(), nullable=False),
        pa.field("origin_input_basis", pa.string(), nullable=False),
        pa.field("origin_method", pa.string(), nullable=False),
        pa.field("origin_dividend_model", pa.string(), nullable=False),
        pa.field("reuse_age_seconds", pa.uint32(), nullable=False),
        *(pa.field(name, pa.float32(), nullable=False) for name in (
            "spot_movement_bps", "target_price_absolute_change",
            "target_price_fractional_change", "elapsed_tenor_fraction",
            "rate_change", "dividend_yield_change", "estimated_iv_drift",
            "estimated_delta_drift",
        )),
        pa.field("reuse_mode", pa.string(), nullable=False),
        pa.field("policy_reason", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

CALCULATION_STATE_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("calculation_definition_id", pa.uint32(), nullable=False),
        pa.field("source_calculation_run_id", pa.string()),
        pa.field("status", pa.string(), nullable=False),
        *(pa.field(name, pa.uint32(), nullable=False) for name in (
            "expected_observations", "eligible_contracts", "successful_results",
            "source_only_contracts", "terminal_contract_failures",
            "terminal_expiration_failures", "snapshot_failure_count",
        )),
        pa.field("started_at", pa.timestamp("us", tz="UTC")),
        pa.field("finished_at", pa.timestamp("us", tz="UTC")),
        pa.field("updated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("active_attempt_id", pa.string()),
        pa.field("error_summary", pa.string(), nullable=False),
        pa.field("is_active", pa.bool_(), nullable=False),
    ]
)

ATTEMPT_SCHEMA = pa.schema(
    [
        pa.field("attempt_id", pa.string(), nullable=False),
        pa.field("process_type", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("definition_id", pa.uint32(), nullable=False),
        pa.field("worker_id", pa.string(), nullable=False),
        pa.field("started_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("finished_at", pa.timestamp("us", tz="UTC")),
        pa.field("status", pa.string(), nullable=False),
        pa.field("expected_count", pa.uint32(), nullable=False),
        pa.field("written_success_count", pa.uint32(), nullable=False),
        pa.field("failure_count", pa.uint32(), nullable=False),
        pa.field("error_summary", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

SURFACE_FIT_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("expiration_at", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("price_basis", pa.string(), nullable=False),
        pa.field("surface_definition_id", pa.uint32(), nullable=False),
        pa.field("source_surface_run_id", pa.string()),
        pa.field("calculated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("pricing_method", pa.string(), nullable=False),
        pa.field("fit_method", pa.string(), nullable=False),
        *(pa.field(f"svi_{name}", pa.float64()) for name in ("a", "b", "rho", "m", "sigma")),
        pa.field("pchip_k", pa.list_(pa.float64()), nullable=False),
        pa.field("pchip_total_variance", pa.list_(pa.float64()), nullable=False),
        *(pa.field(name, pa.uint32(), nullable=False) for name in (
            "input_contract_count", "accepted_contract_count", "rejected_contract_count",
            "selected_contract_count", "source_current_iv_point_count",
            "source_reused_iv_point_count", "source_vendor_iv_point_count",
            "no_iv_excluded_count",
        )),
        pa.field("source_iv_weight_fraction", pa.float64(), nullable=False),
        pa.field("minimum_input_delta", pa.float64()),
        pa.field("maximum_input_delta", pa.float64()),
        *(pa.field(name, pa.float64(), nullable=False) for name in (
            "minimum_log_forward_moneyness", "maximum_log_forward_moneyness",
            "weighted_rmse", "maximum_absolute_error",
        )),
        pa.field("butterfly_arbitrage_flags", pa.uint32(), nullable=False),
        pa.field("calendar_arbitrage_flags", pa.uint32(), nullable=False),
        pa.field("quality_flags", pa.uint32(), nullable=False),
        pa.field("actual_dtype", pa.string(), nullable=False),
        pa.field("precision_escalated", pa.bool_(), nullable=False),
        pa.field("required_node_status", pa.string(), nullable=False),
        pa.field("optional_node_status", pa.string(), nullable=False),
    ]
)

SURFACE_NODE_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("expiration_at", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("price_basis", pa.string(), nullable=False),
        pa.field("surface_definition_id", pa.uint32(), nullable=False),
        pa.field("source_surface_run_id", pa.string()),
        pa.field("node_type", pa.string(), nullable=False),
        pa.field("node_value", pa.float64(), nullable=False),
        pa.field("put_call", pa.string(), nullable=False),
        pa.field("fitted_strike", pa.float64(), nullable=False),
        pa.field("fitted_delta", pa.float64()),
        pa.field("fitted_iv_bid", pa.float64()),
        pa.field("fitted_iv_mid", pa.float64(), nullable=False),
        pa.field("fitted_iv_ask", pa.float64()),
        pa.field("fit_confidence", pa.float64(), nullable=False),
        pa.field("quality_flags", pa.uint32(), nullable=False),
        pa.field("calculated_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

SURFACE_FAILURE_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("surface_definition_id", pa.uint32(), nullable=False),
        pa.field("source_surface_run_id", pa.string()),
        pa.field("failure_scope", pa.string(), nullable=False),
        pa.field("expiration_at", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("price_basis", pa.string(), nullable=False),
        pa.field("node_type", pa.string()),
        pa.field("node_value", pa.float64()),
        pa.field("put_call", pa.string()),
        pa.field("failure_stage", pa.string(), nullable=False),
        pa.field("reason_code", pa.string(), nullable=False),
        pa.field("error_class", pa.string(), nullable=False),
        pa.field("error_detail", pa.string(), nullable=False),
        pa.field("retry_class", pa.string(), nullable=False),
        pa.field("retry_after", pa.timestamp("us", tz="UTC")),
        pa.field("failure_fingerprint", pa.binary(32), nullable=False),
        pa.field("input_contract_count", pa.uint32()),
        pa.field("accepted_contract_count", pa.uint32()),
        pa.field("rejected_contract_count", pa.uint32()),
        pa.field("first_failed_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("last_failed_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("failure_count", pa.uint16(), nullable=False),
        pa.field("updated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("quality_flags", pa.uint32(), nullable=False),
        pa.field("is_active", pa.bool_(), nullable=False),
    ]
)

CALCULATION_DIAGNOSTIC_SCHEMA = pa.schema(
    [
        pa.field("diagnostic_id", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("contract_id", pa.uint64()),
        pa.field("calculation_definition_id", pa.uint32(), nullable=False),
        pa.field("diagnostic_type", pa.string(), nullable=False),
        pa.field("diagnostic_json", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

SURFACE_DIAGNOSTIC_SCHEMA = pa.schema(
    [
        pa.field("diagnostic_id", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("surface_definition_id", pa.uint32(), nullable=False),
        pa.field("diagnostic_type", pa.string(), nullable=False),
        pa.field("diagnostic_json", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

DATASET_METADATA_SCHEMA = pa.schema(
    [
        pa.field("dataset_generation_id", pa.string(), nullable=False),
        pa.field("schema_generation", pa.uint16(), nullable=False),
        pa.field("migration_id", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("activated_at", pa.timestamp("us", tz="UTC")),
        pa.field("deactivated_at", pa.timestamp("us", tz="UTC")),
        *(pa.field(name, pa.uint32(), nullable=False) for name in (
            "default_collection_definition_id",
            "default_normalization_definition_id",
            "default_calculation_definition_id",
            "default_surface_definition_id",
        )),
        pa.field("source_start_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("source_end_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("validation_summary_json", pa.string(), nullable=False),
        pa.field("migration_phase", pa.string(), nullable=False),
        pa.field("migration_checkpoint_json", pa.string(), nullable=False),
        pa.field("updated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("is_active", pa.bool_(), nullable=False),
    ]
)

SURFACE_STATE_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source_collected_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("surface_definition_id", pa.uint32(), nullable=False),
        pa.field("source_calculation_definition_id", pa.uint32(), nullable=False),
        pa.field("source_surface_run_id", pa.string()),
        pa.field("status", pa.string(), nullable=False),
        *(pa.field(name, pa.uint32(), nullable=False) for name in (
            "expected_expirations", "successful_expirations",
            "source_only_or_skipped_expirations", "terminal_fit_failures",
            "expected_required_nodes", "successful_required_nodes",
            "terminal_required_node_failures", "successful_optional_nodes",
            "terminal_optional_node_failures",
        )),
        pa.field("started_at", pa.timestamp("us", tz="UTC")),
        pa.field("finished_at", pa.timestamp("us", tz="UTC")),
        pa.field("updated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("active_attempt_id", pa.string()),
        pa.field("error_summary", pa.string(), nullable=False),
        pa.field("is_active", pa.bool_(), nullable=False),
    ]
)


def _runtime_rows_with_native_lineage(
    rows: Sequence[Mapping[str, Any]], lineage_column: str
) -> list[dict[str, Any]]:
    """Materialize runtime-native rows with explicit NULL migration lineage."""
    return [{**dict(row), lineage_column: None} for row in rows]


def validate_calculation_failure(row: Mapping[str, Any]) -> None:
    scope = str(row.get("failure_scope") or "")
    expiration = row.get("expiration_at")
    contract = row.get("contract_id")
    if scope == "snapshot" and (expiration is not None or contract is not None):
        raise ValueError("Snapshot failure scope cannot have expiration/contract keys")
    if scope == "expiration" and (expiration is None or contract is not None):
        raise ValueError("Expiration failure scope requires only expiration_at")
    if scope == "contract" and contract is None:
        raise ValueError("Contract failure scope requires contract_id")
    if scope not in {"snapshot", "expiration", "contract"}:
        raise ValueError(f"Unknown calculation failure scope: {scope!r}")


def validate_success_result(
    row: Mapping[str, Any], observation: Mapping[str, Any]
) -> None:
    required = (
        "target_price", "iv", "price_residual", "model_price", "delta", "gamma",
        "theta_per_day", "vega_per_1pct", "rho_per_1pct",
    )
    missing = [name for name in required if _finite_float(row.get(name)) is None]
    if missing:
        raise ValueError("Successful result has nonfinite outputs: " + ", ".join(missing))
    if str(row.get("result_origin")) not in {"fresh", "reused"}:
        raise ValueError("Successful result origin must be fresh or reused")
    if str(row.get("iv_selection")) not in {"source", "result"}:
        raise ValueError("Invalid IV selection")
    if str(row.get("greeks_selection")) not in {"source", "result"}:
        raise ValueError("Invalid Greeks selection")
    validate_effective_selection(
        {
            **observation,
            "result_iv": row.get("iv"),
            "result_delta": row.get("delta"),
            "result_gamma": row.get("gamma"),
            "result_theta_per_day": row.get("theta_per_day"),
            "result_vega_per_1pct": row.get("vega_per_1pct"),
            "result_rho_per_1pct": row.get("rho_per_1pct"),
            "iv_selection": row.get("iv_selection"),
            "greeks_selection": row.get("greeks_selection"),
        }
    )


def snapshot_readiness(client: Any, database: str, snapshot_id: uuid.UUID) -> SnapshotReadiness:
    database = validate_database_identifier(database)
    result = client.query(
        f"""
        SELECT
          (SELECT count() FROM {database}.options_chain_snapshots
           WHERE snapshot_id = {{snapshot_id:UUID}}),
          (SELECT any(stored_observation_count) FROM {database}.options_chain_snapshots
           WHERE snapshot_id = {{snapshot_id:UUID}}),
          (SELECT count() FROM {database}.options_chain_observations
           WHERE snapshot_id = {{snapshot_id:UUID}}
             AND source_collected_at = (SELECT source_collected_at
               FROM {database}.options_chain_snapshots
               WHERE snapshot_id = {{snapshot_id:UUID}} LIMIT 1)),
          (SELECT countDistinct(contract_id) FROM {database}.options_chain_observations
           WHERE snapshot_id = {{snapshot_id:UUID}}
             AND source_collected_at = (SELECT source_collected_at
               FROM {database}.options_chain_snapshots
               WHERE snapshot_id = {{snapshot_id:UUID}} LIMIT 1)),
          (SELECT any(stored_distinct_contract_count) FROM {database}.options_chain_snapshots
           WHERE snapshot_id = {{snapshot_id:UUID}}),
          (SELECT count() FROM {database}.options_chain_observation_details
           WHERE snapshot_id = {{snapshot_id:UUID}}
             AND source_collected_at = (SELECT source_collected_at
               FROM {database}.options_chain_snapshots
               WHERE snapshot_id = {{snapshot_id:UUID}} LIMIT 1)),
          (SELECT any(stored_detail_count) FROM {database}.options_chain_snapshots
           WHERE snapshot_id = {{snapshot_id:UUID}}),
          (SELECT count() FROM {database}.options_chain_observations AS o
           LEFT JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c ON c.contract_id = o.contract_id
           WHERE o.snapshot_id = {{snapshot_id:UUID}}
             AND o.source_collected_at = (SELECT source_collected_at
               FROM {database}.options_chain_snapshots
               WHERE snapshot_id = {{snapshot_id:UUID}} LIMIT 1)
             AND isNull(c.contract_id)),
          (SELECT count() FROM {database}.options_chain_snapshots AS s
           LEFT JOIN (SELECT * FROM {database}.options_chain_definitions FINAL) AS cd
             ON cd.definition_id = s.collection_definition_id AND cd.is_active
           LEFT JOIN (SELECT * FROM {database}.options_chain_definitions FINAL) AS nd
             ON nd.definition_id = s.normalization_definition_id AND nd.is_active
           WHERE s.snapshot_id = {{snapshot_id:UUID}}
             AND (isNull(cd.definition_id) OR isNull(nd.definition_id)))
        """,
        parameters={"snapshot_id": str(snapshot_id)},
    )
    values = result.result_rows[0] if result.result_rows else (0,) * 9
    return SnapshotReadiness(*(int(value or 0) for value in values))


def validate_normalized_source_schema(client: Any, database: str) -> None:
    validate_canonical_table_metadata(
        client,
        database,
        (
            "options_chain_snapshots",
            "options_contracts",
            "options_chain_observations",
        ),
    )


def discover_calculation_work(
    client: Any,
    database: str,
    *,
    definition_repository: Optional[DefinitionRepository] = None,
    limit: int = 100,
    stale_after_seconds: int = 900,
    snapshot_id: Optional[uuid.UUID] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    symbols: Optional[Sequence[str]] = None,
    retry_failed: bool = False,
) -> list[CalculationWork]:
    database = validate_database_identifier(database)
    if limit <= 0:
        raise ValueError("calculation work limit must be positive")
    filters = []
    parameters: dict[str, Any] = {"page_size": limit}
    if snapshot_id:
        filters.append("s.snapshot_id = {snapshot_id:UUID}")
        parameters["snapshot_id"] = str(snapshot_id)
    if start_at:
        filters.append("toUnixTimestamp64Micro(s.source_collected_at) >= {start_us:Int64}")
        parameters["start_us"] = _datetime64_microseconds(start_at)
    if end_at:
        filters.append("toUnixTimestamp64Micro(s.source_collected_at) < {end_us:Int64}")
        parameters["end_us"] = _datetime64_microseconds(end_at)
    if symbols:
        filters.append("s.underlying_symbol IN {symbols:Array(String)}")
        parameters["symbols"] = [str(value).upper() for value in symbols]
    if not retry_failed:
        filters.append("(isNull(state.snapshot_id) OR state.status != 'failed')")
    where = " AND " + " AND ".join(filters) if filters else ""
    candidate_sql = f"""
        SELECT s.snapshot_id, s.source_collected_at, s.underlying_symbol,
               s.stored_observation_count, state.calculation_definition_id,
               toString(state.status), state.started_at
        FROM {database}.options_chain_snapshots AS s
        LEFT JOIN (SELECT * FROM {database}.options_chain_calculation_state FINAL) AS state
          ON state.snapshot_id = s.snapshot_id
         AND state.source_collected_at = s.source_collected_at
         AND state.is_active
        WHERE (isNull(state.snapshot_id)
               OR state.status NOT IN ('complete', 'complete_with_failures')){where}
        GROUP BY s.snapshot_id, s.source_collected_at, s.underlying_symbol,
                 s.stored_observation_count, state.calculation_definition_id,
                 state.status, state.started_at
        ORDER BY s.source_collected_at, s.snapshot_id,
                 state.calculation_definition_id, state.status, state.started_at
        LIMIT {{page_size:UInt32}} OFFSET {{offset:UInt64}}
    """
    # The original INNER JOIN omitted candidates with no matching observation
    # and contract. Page metadata until the same SQL-level limit is filled.
    selected_rows: list[tuple[Any, ...]] = []
    offset = 0
    while len(selected_rows) < limit:
        candidates = client.query(
            candidate_sql, parameters={**parameters, "offset": offset}
        ).result_rows
        if not candidates:
            break
        offset += len(candidates)
        unique_keys = dict.fromkeys(
            (str(sid), source_at) for sid, source_at, *_ in candidates
        )
        pair_sql = []
        expiration_parameters: dict[str, Any] = {}
        for index, (sid, source_at) in enumerate(unique_keys):
            pair_sql.append(
                f"({{candidate_at_{index}:DateTime64(6, 'UTC')}}, "
                f"{{candidate_id_{index}:UUID}})"
            )
            # ClickHouse returns UTC DateTime64 values as naive datetimes. Bind
            # an aware UTC value so the driver does not apply the host timezone.
            expiration_parameters[f"candidate_at_{index}"] = _parse_timestamp(source_at)
            expiration_parameters[f"candidate_id_{index}"] = sid
        expiration_rows = client.query(
            f"""
            SELECT toString(o.snapshot_id), o.source_collected_at,
                   groupUniqArray(c.expiration_at)
            FROM {database}.options_chain_observations AS o
            INNER JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c
              ON c.contract_id = o.contract_id
            WHERE (o.source_collected_at, o.snapshot_id)
                  IN ({', '.join(pair_sql)})
            GROUP BY o.snapshot_id, o.source_collected_at
            """,
            parameters=expiration_parameters,
        ).result_rows
        expirations_by_key = {
            (str(sid), _parse_timestamp(source_at)): expirations
            for sid, source_at, expirations in expiration_rows
        }
        for candidate in candidates:
            sid, source_at, symbol, expected, assigned, status, started_at = candidate
            expirations = expirations_by_key.get(
                (str(sid), _parse_timestamp(source_at))
            )
            if expirations is None:
                continue
            selected_rows.append(
                (sid, source_at, symbol, expected, expirations,
                 assigned, status, started_at)
            )
            if len(selected_rows) >= limit:
                break
        if len(candidates) < limit:
            break

    repository = definition_repository or DefinitionRepository(client, database)
    now = _utc_now()
    output = []
    for sid, source_at, symbol, expected, expirations, assigned, status, started_at in selected_rows:
        status = str(status or "").lower()
        if status == "running":
            started = _parse_timestamp(started_at)
            if started and (now - started).total_seconds() < stale_after_seconds:
                continue
        source_at = _parse_timestamp(source_at)
        definition_id = int(assigned or 0)
        if not definition_id:
            definition_id = repository.resolve(
                DefinitionType.CALCULATION, source_at
            ).definition_id
        output.append(
            CalculationWork(
                uuid.UUID(str(sid)),
                source_at,
                str(symbol),
                int(expected),
                tuple(sorted(_parse_timestamp(value) for value in expirations)),
                definition_id,
                bool(retry_failed),
            )
        )
    return output


def discover_interrupted_calculation_work(
    client: Any,
    database: str,
    *,
    definition_repository: Optional[DefinitionRepository] = None,
    snapshot_id: Optional[uuid.UUID] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    symbols: Optional[Sequence[str]] = None,
) -> list[InterruptedCalculationWork]:
    """Discover only retryable calculation owners across all retained history.

    This deliberately does not apply ``reconcile_days``.  It is a narrow
    startup-recovery scan for states left by an interrupted coordinator, not a
    replacement for bounded normal pending-work discovery.
    """
    database = validate_database_identifier(database)
    filters = ["state.is_active", "state.status IN ('running', 'abandoned', 'stale_running')"]
    parameters: dict[str, Any] = {}
    if snapshot_id is not None:
        filters.append("s.snapshot_id = {snapshot_id:UUID}")
        parameters["snapshot_id"] = str(snapshot_id)
    if start_at is not None:
        filters.append(
            "toUnixTimestamp64Micro(s.source_collected_at) >= {start_us:Int64}"
        )
        parameters["start_us"] = _datetime64_microseconds(start_at)
    if end_at is not None:
        filters.append(
            "toUnixTimestamp64Micro(s.source_collected_at) < {end_us:Int64}"
        )
        parameters["end_us"] = _datetime64_microseconds(end_at)
    if symbols:
        filters.append("s.underlying_symbol IN {symbols:Array(String)}")
        parameters["symbols"] = [str(value).upper() for value in symbols]
    rows = client.query(
        f"""
        SELECT toString(state.snapshot_id), state.source_collected_at,
               state.underlying_symbol, state.expected_observations,
               groupUniqArray(c.expiration_at), state.calculation_definition_id,
               toString(state.status), toString(state.active_attempt_id)
        FROM (SELECT * FROM {database}.options_chain_calculation_state FINAL) AS state
        INNER JOIN {database}.options_chain_snapshots AS s
          ON s.snapshot_id = state.snapshot_id
         AND s.source_collected_at = state.source_collected_at
        INNER JOIN {database}.options_chain_observations AS o
          ON o.snapshot_id = state.snapshot_id
         AND o.source_collected_at = state.source_collected_at
        INNER JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c
          ON c.contract_id = o.contract_id
        WHERE {' AND '.join(filters)}
        GROUP BY state.snapshot_id, state.source_collected_at,
                 state.underlying_symbol, state.expected_observations,
                 state.calculation_definition_id, state.status,
                 state.active_attempt_id
        ORDER BY state.source_collected_at
        """,
        parameters=parameters,
    ).result_rows
    repository = definition_repository or DefinitionRepository(client, database)
    output: list[InterruptedCalculationWork] = []
    for (
        sid, source_at, symbol, expected, expirations, assigned, status, active
    ) in rows:
        source_timestamp = _parse_timestamp(source_at)
        if source_timestamp is None:
            raise RuntimeError(f"Calculation recovery row has no source timestamp: {sid}")
        definition_id = int(assigned or 0)
        if not definition_id:
            definition_id = repository.resolve(
                DefinitionType.CALCULATION, source_timestamp
            ).definition_id
        attempt = None
        if active and str(active).lower() not in {"", "none", "null"}:
            attempt = uuid.UUID(str(active))
        output.append(
            InterruptedCalculationWork(
                CalculationWork(
                    uuid.UUID(str(sid)), source_timestamp, str(symbol), int(expected),
                    tuple(sorted(_parse_timestamp(value) for value in expirations)),
                    definition_id,
                ),
                str(status).lower(),
                attempt,
            )
        )
    return output


class CalculationRepository:
    def __init__(
        self,
        client: Any,
        database: str,
        *,
        worker_id: Optional[str] = None,
        stale_after_seconds: int = 900,
    ):
        self.client = client
        self.database = validate_database_identifier(database)
        self.worker_id = worker_id or f"{socket.gethostname()}:{uuid.uuid4()}"
        self.stale_after_seconds = int(stale_after_seconds)
        if self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")

    def claim(
        self, work: CalculationWork, *, attempt_id: Optional[uuid.UUID] = None
    ) -> ProcessingClaim:
        if maintenance_lease_active(self.client, self.database):
            raise RuntimeError("Calculation claims are disabled by an active migration lease")
        readiness = snapshot_readiness(self.client, self.database, work.snapshot_id)
        if not readiness.ready:
            raise ValueError(f"Snapshot {work.snapshot_id} is not ready: {readiness}")
        existing = self.client.query(
            f"SELECT toString(status), started_at FROM "
            f"{self.database}.options_chain_calculation_state FINAL "
            "WHERE snapshot_id = {snapshot_id:UUID} AND is_active LIMIT 1",
            parameters={"snapshot_id": str(work.snapshot_id)},
        ).result_rows
        if existing and str(existing[0][0]) in TERMINAL_STATES:
            raise ValueError("A terminal historical calculation state is authoritative")
        if existing and str(existing[0][0]) == "failed" and not work.retry_failed:
            raise ValueError("Failed calculation requires explicit retry_failed selection")
        if existing and str(existing[0][0]) == "running":
            started = _parse_timestamp(existing[0][1])
            if started and (
                _utc_now() - started
            ).total_seconds() < self.stale_after_seconds:
                raise ValueError("Calculation owner already has a live running claim")
        # Expired running, stale-running, abandoned, and explicitly selected
        # failed work all replay from a clean snapshot publication boundary.
        if existing:
            self.repair_partial_publication(work.snapshot_id)
        now = _utc_now()
        claim = ProcessingClaim(
            attempt_id or uuid.uuid4(), "calculation", work.snapshot_id,
            work.source_collected_at, work.calculation_definition_id, now,
        )
        state = {
            "snapshot_id": str(work.snapshot_id),
            "source_collected_at": work.source_collected_at,
            "underlying_symbol": work.underlying_symbol,
            "calculation_definition_id": work.calculation_definition_id,
            "status": "running",
            "expected_observations": work.expected_observations,
            "eligible_contracts": 0,
            "successful_results": 0,
            "source_only_contracts": 0,
            "terminal_contract_failures": 0,
            "terminal_expiration_failures": 0,
            "snapshot_failure_count": 0,
            "started_at": now,
            "finished_at": None,
            "updated_at": now,
            "active_attempt_id": str(claim.attempt_id),
            "error_summary": "",
            "is_active": True,
        }
        attempt = {
            "attempt_id": str(claim.attempt_id),
            "process_type": "calculation",
            "snapshot_id": str(work.snapshot_id),
            "source_collected_at": work.source_collected_at,
            "definition_id": work.calculation_definition_id,
            "worker_id": self.worker_id,
            "started_at": now,
            "finished_at": None,
            "status": "running",
            "expected_count": work.expected_observations,
            "written_success_count": 0,
            "failure_count": 0,
            "error_summary": "",
            "created_at": now,
        }
        self.client.insert_arrow(
            f"{self.database}.options_chain_calculation_state",
            _rows_to_arrow(
                _runtime_rows_with_native_lineage([state], "source_calculation_run_id"),
                CALCULATION_STATE_SCHEMA,
            ),
        )
        self.client.insert_arrow(
            f"{self.database}.options_chain_processing_attempts",
            _rows_to_arrow([attempt], ATTEMPT_SCHEMA),
        )
        owner = self.client.query(
            f"SELECT toString(active_attempt_id) FROM "
            f"{self.database}.options_chain_calculation_state FINAL "
            "WHERE snapshot_id = {snapshot_id:UUID} AND is_active LIMIT 1",
            parameters={"snapshot_id": str(work.snapshot_id)},
        ).result_rows
        if not owner or str(owner[0][0]) != str(claim.attempt_id):
            raise RuntimeError("Calculation claim lost concurrent ownership")
        return claim

    def fail_attempt_if_running(
        self,
        *,
        snapshot_id: uuid.UUID,
        attempt_id: uuid.UUID,
        error: BaseException | str,
        underlying_symbol: Optional[str] = None,
        expected_observations: Optional[int] = None,
    ) -> bool:
        """Fail only the still-active attempt identified by ``attempt_id``."""
        rows = self.client.query(
            f"SELECT toString(status), toString(active_attempt_id), "
            f"source_collected_at, underlying_symbol, calculation_definition_id, "
            f"expected_observations, started_at FROM "
            f"{self.database}.options_chain_calculation_state FINAL "
            "WHERE snapshot_id = {snapshot_id:UUID} AND is_active LIMIT 1",
            parameters={"snapshot_id": str(snapshot_id)},
        ).result_rows
        if not rows:
            return False
        status, active, source_at, symbol, definition_id, expected, started_at = rows[0]
        if str(status) in TERMINAL_STATES or str(status) != "running":
            return False
        if str(active) != str(attempt_id):
            return False
        claim = ProcessingClaim(
            uuid.UUID(str(attempt_id)), "calculation", snapshot_id,
            _parse_timestamp(source_at) or _utc_now(), int(definition_id),
            _parse_timestamp(started_at) or _utc_now(),
        )
        return self.fail(
            claim,
            underlying_symbol=str(underlying_symbol or symbol),
            expected_observations=int(
                expected_observations if expected_observations is not None else expected
            ),
            error=error,
        )

    def abandon_attempt_if_running(
        self,
        *,
        snapshot_id: uuid.UUID,
        attempt_id: uuid.UUID,
        reason: str = "coordinator_shutdown",
    ) -> bool:
        """Mark one owned calculation attempt as retryable infrastructure loss.

        Abandonment is deliberately distinct from :meth:`fail`: it records
        that a coordinator/worker was interrupted, preserves the partial
        counters for diagnosis, and never writes a numerical calculation
        failure row or increments snapshot failure counts.  The authoritative
        ``FINAL`` ownership check makes the operation idempotent and prevents
        a late coordinator from overwriting a terminal publication.
        """
        if maintenance_lease_active(self.client, self.database):
            raise RuntimeError("Calculation abandonment is disabled by an active migration lease")
        rows = self.client.query(
            f"""
            SELECT toString(status), toString(active_attempt_id),
                   source_collected_at, underlying_symbol,
                   calculation_definition_id, expected_observations,
                   eligible_contracts, successful_results, source_only_contracts,
                   terminal_contract_failures, terminal_expiration_failures,
                   snapshot_failure_count, started_at
            FROM {self.database}.options_chain_calculation_state FINAL
            WHERE snapshot_id = {{snapshot_id:UUID}} AND is_active
            LIMIT 1
            """,
            parameters={"snapshot_id": str(snapshot_id)},
        ).result_rows
        if not rows:
            return False
        (
            status, active, source_at, symbol, definition_id, expected,
            eligible, successes, source_only, contract_failures,
            expiration_failures, snapshot_failures, started_at,
        ) = rows[0]
        if str(status).lower() != "running" or str(active) != str(attempt_id):
            return False
        now = _utc_now()
        started = _parse_timestamp(started_at) or now
        detail = str(reason).strip() or "coordinator_shutdown"
        attempt = {
            "attempt_id": str(attempt_id),
            "process_type": "calculation",
            "snapshot_id": str(snapshot_id),
            "source_collected_at": _parse_timestamp(source_at) or now,
            "definition_id": int(definition_id),
            "worker_id": self.worker_id,
            "started_at": started,
            "finished_at": now,
            "status": "abandoned",
            "expected_count": int(expected),
            "written_success_count": int(successes),
            "failure_count": int(contract_failures) + int(expiration_failures),
            "error_summary": detail,
            "created_at": started,
        }
        state = {
            "snapshot_id": str(snapshot_id),
            "source_collected_at": _parse_timestamp(source_at) or now,
            "underlying_symbol": str(symbol),
            "calculation_definition_id": int(definition_id),
            "status": "abandoned",
            "expected_observations": int(expected),
            "eligible_contracts": int(eligible),
            "successful_results": int(successes),
            "source_only_contracts": int(source_only),
            "terminal_contract_failures": int(contract_failures),
            "terminal_expiration_failures": int(expiration_failures),
            "snapshot_failure_count": int(snapshot_failures),
            "started_at": started,
            "finished_at": now,
            "updated_at": now,
            "active_attempt_id": None,
            "error_summary": detail,
            "is_active": True,
        }
        self.client.insert_arrow(
            f"{self.database}.options_chain_processing_attempts",
            _rows_to_arrow([attempt], ATTEMPT_SCHEMA),
        )
        self.client.insert_arrow(
            f"{self.database}.options_chain_calculation_state",
            _rows_to_arrow(
                _runtime_rows_with_native_lineage([state], "source_calculation_run_id"),
                CALCULATION_STATE_SCHEMA,
            ),
        )
        verified = self.client.query(
            f"SELECT toString(status), toString(active_attempt_id) "
            f"FROM {self.database}.options_chain_calculation_state FINAL "
            "WHERE snapshot_id = {snapshot_id:UUID} AND is_active LIMIT 1",
            parameters={"snapshot_id": str(snapshot_id)},
        ).result_rows
        if (
            not verified
            or str(verified[0][0]).lower() != "abandoned"
            or str(verified[0][1]).lower() not in {"", "none", "null"}
        ):
            raise RuntimeError("Calculation abandonment could not be verified")
        return True

    def repair_partial_publication(self, snapshot_id: uuid.UUID) -> None:
        """Remove only one nonterminal owner's derived rows before replay."""
        parameters = {"snapshot_id": str(snapshot_id)}
        for table in (
            "options_chain_expiration_inputs",
            "options_chain_calculation_results",
            "options_chain_calculation_failures",
            "options_chain_calculation_reuse_provenance",
            "options_chain_calculation_diagnostics",
            "options_chain_calculation_reuse_validation",
        ):
            self.client.command(
                f"ALTER TABLE {self.database}.{table} DELETE "
                "WHERE snapshot_id = {snapshot_id:UUID}",
                parameters=parameters,
                settings={"mutations_sync": 2},
            )

    def fail(
        self,
        claim: ProcessingClaim,
        *,
        underlying_symbol: str,
        expected_observations: int,
        error: BaseException | str,
        failure_row: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Publish an infrastructure/incomplete-snapshot failure and finish its attempt.

        A failed state is intentionally not presented as processed numerical work.
        It remains visible to reconciliation and may be retried by policy.
        """
        if claim.process_type != "calculation":
            raise ValueError("Calculation failure received a non-calculation claim")
        owner = self.client.query(
            f"SELECT toString(status), toString(active_attempt_id) FROM "
            f"{self.database}.options_chain_calculation_state FINAL "
            "WHERE snapshot_id = {snapshot_id:UUID} AND is_active LIMIT 1",
            parameters={"snapshot_id": str(claim.snapshot_id)},
        ).result_rows
        if (
            not owner
            or str(owner[0][0]) != "running"
            or str(owner[0][1]) != str(claim.attempt_id)
        ):
            # A publication may have terminalized successfully just before a
            # caller observed an exception.  Never overwrite that outcome.
            return False
        now = _utc_now()
        detail = str(error)
        if failure_row is not None:
            row = dict(failure_row)
            validate_calculation_failure(row)
            if row["failure_scope"] != "snapshot":
                raise ValueError("Infrastructure publication requires snapshot failure scope")
            self.client.insert_arrow(
                f"{self.database}.options_chain_calculation_failures",
                _rows_to_arrow(
                    _runtime_rows_with_native_lineage([row], "source_calculation_run_id"),
                    CALCULATION_FAILURE_SCHEMA,
                ),
            )
        attempt = {
            "attempt_id": str(claim.attempt_id),
            "process_type": "calculation",
            "snapshot_id": str(claim.snapshot_id),
            "source_collected_at": claim.source_collected_at,
            "definition_id": claim.definition_id,
            "worker_id": self.worker_id,
            "started_at": claim.started_at,
            "finished_at": now,
            "status": "failed",
            "expected_count": int(expected_observations),
            "written_success_count": 0,
            "failure_count": 1,
            "error_summary": detail,
            "created_at": claim.started_at,
        }
        state = {
            "snapshot_id": str(claim.snapshot_id),
            "source_collected_at": claim.source_collected_at,
            "underlying_symbol": underlying_symbol,
            "calculation_definition_id": claim.definition_id,
            "status": "failed",
            "expected_observations": int(expected_observations),
            "eligible_contracts": 0,
            "successful_results": 0,
            "source_only_contracts": 0,
            "terminal_contract_failures": 0,
            "terminal_expiration_failures": 0,
            "snapshot_failure_count": 1,
            "started_at": claim.started_at,
            "finished_at": now,
            "updated_at": now,
            "active_attempt_id": None,
            "error_summary": detail,
            "is_active": True,
        }
        self.client.insert_arrow(
            f"{self.database}.options_chain_processing_attempts",
            _rows_to_arrow([attempt], ATTEMPT_SCHEMA),
        )
        self.client.insert_arrow(
            f"{self.database}.options_chain_calculation_state",
            _rows_to_arrow(
                _runtime_rows_with_native_lineage([state], "source_calculation_run_id"),
                CALCULATION_STATE_SCHEMA,
            ),
        )
        return True

    def load_source_rows(self, snapshot_id: uuid.UUID) -> list[dict[str, Any]]:
        table = self.client.query_arrow(
            f"""
            SELECT {', '.join(calculation_source_projection_columns())}
            FROM {self.database}.options_chain_snapshots AS s
            INNER JOIN {self.database}.options_chain_observations AS o
              ON o.snapshot_id = s.snapshot_id
             AND o.source_collected_at = s.source_collected_at
            INNER JOIN (SELECT * FROM {self.database}.options_contracts FINAL) AS c
              ON c.contract_id = o.contract_id
            PREWHERE s.source_collected_at =
              (SELECT source_collected_at
               FROM {self.database}.options_chain_snapshots
               WHERE snapshot_id = {{snapshot_id:UUID}}
               LIMIT 1)
            WHERE s.snapshot_id = {{snapshot_id:UUID}}
            ORDER BY o.contract_ordinal, o.contract_id
            """,
            parameters={"snapshot_id": str(snapshot_id)},
        )
        return table.to_pylist()

    def publish(
        self,
        claim: ProcessingClaim,
        *,
        results: Sequence[Mapping[str, Any]],
        observations: Mapping[int, Mapping[str, Any]],
        source_only_contract_ids: Sequence[int],
        failures: Sequence[Mapping[str, Any]],
        expiration_inputs: Sequence[Mapping[str, Any]],
        reuse_provenance: Sequence[Mapping[str, Any]] = (),
    ) -> Mapping[str, int]:
        if claim.process_type != "calculation":
            raise ValueError("Calculation publication received a non-calculation claim")
        result_ids = [int(row["contract_id"]) for row in results]
        source_only_ids = [int(value) for value in source_only_contract_ids]
        contract_failure_ids = [
            int(row["contract_id"])
            for row in failures
            if row.get("failure_scope") == "contract"
        ]
        failure_keys = [
            (
                str(row.get("failure_scope")),
                row.get("expiration_at"),
                int(row["contract_id"]) if row.get("contract_id") is not None else 0,
            )
            for row in failures
        ]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("Duplicate successful result natural key")
        if len(source_only_ids) != len(set(source_only_ids)):
            raise ValueError("Duplicate source-only contract outcome")
        if len(contract_failure_ids) != len(set(contract_failure_ids)):
            raise ValueError("Duplicate active contract failure natural key")
        if len(failure_keys) != len(set(failure_keys)):
            raise ValueError("Duplicate calculation failure natural key")
        if set(result_ids) & set(source_only_ids):
            raise ValueError("A contract cannot be both successful and source-only")
        if (set(result_ids) | set(source_only_ids)) & set(contract_failure_ids):
            raise ValueError("A contract cannot be successful/source-only and failed")
        for row in results:
            contract_id = int(row["contract_id"])
            observation = observations.get(contract_id)
            if observation is None:
                raise ValueError(f"Successful result has no observation: {contract_id}")
            validate_success_result(row, observation)
            if int(row["calculation_definition_id"]) != claim.definition_id:
                raise ValueError("Result definition differs from assigned state definition")
        for row in failures:
            validate_calculation_failure(row)
            if int(row["calculation_definition_id"]) != claim.definition_id:
                raise ValueError("Failure definition differs from assigned state definition")
        reused_ids = {
            int(row["contract_id"])
            for row in results
            if str(row["result_origin"]) == "reused"
        }
        provenance_ids = {int(row["contract_id"]) for row in reuse_provenance}
        if len(provenance_ids) != len(reuse_provenance):
            raise ValueError("Duplicate reuse-provenance natural key")
        if reused_ids != provenance_ids:
            raise ValueError("Every reused success requires exactly one provenance row")
        required_origin = {
            "origin_valuation_at", "origin_spot", "origin_target_price",
            "origin_time_to_expiry_seconds", "origin_risk_free_rate",
            "origin_dividend_yield", "origin_input_basis", "origin_method",
            "origin_dividend_model",
        }
        for row in reuse_provenance:
            if any(row.get(name) is None for name in required_origin):
                raise ValueError("Reuse provenance is not self-contained")
            if int(row["calculation_definition_id"]) != claim.definition_id:
                raise ValueError("Reuse definition differs from assigned state definition")
            for name in (
                "origin_spot", "origin_target_price", "origin_risk_free_rate",
                "origin_dividend_yield", "spot_movement_bps",
                "target_price_absolute_change", "target_price_fractional_change",
                "elapsed_tenor_fraction", "rate_change", "dividend_yield_change",
                "estimated_iv_drift", "estimated_delta_drift",
            ):
                value = _finite_float(row.get(name))
                if value is None:
                    raise ValueError(f"Reuse provenance has nonfinite {name}")
                if name not in {
                    "origin_spot", "origin_target_price", "origin_risk_free_rate",
                    "origin_dividend_yield",
                } and abs(value) > 3.4028235e38:
                    raise ValueError(f"Reuse provenance {name} exceeds Float32 range")
        expiration_keys = [row["expiration_at"] for row in expiration_inputs]
        if len(expiration_keys) != len(set(expiration_keys)):
            raise ValueError("Duplicate expiration-input natural key")
        for row in expiration_inputs:
            if int(row["calculation_definition_id"]) != claim.definition_id:
                raise ValueError("Expiration input definition differs from assigned state")
            if not 0 < int(row["time_to_expiry_seconds"]) <= 2**32 - 1:
                raise ValueError("Expiration input tenor is outside UInt32 range")
            for name in ("risk_free_rate", "dividend_yield", "forward_price"):
                if _finite_float(row.get(name)) is None:
                    raise ValueError(f"Expiration input has nonfinite {name}")
        expiration_failures = sum(
            row.get("failure_scope") == "expiration" for row in failures
        )
        snapshot_failures = sum(row.get("failure_scope") == "snapshot" for row in failures)
        terminal_contract_failures = len(contract_failure_ids)
        eligible = len(result_ids) + len(source_only_ids) + terminal_contract_failures
        # Expiration/snapshot failures may cover multiple contracts.  Callers
        # must supply covered_contract_count to make accounting explicit.
        covered = sum(
            int(row.get("covered_contract_count", 0))
            for row in failures
            if row.get("failure_scope") in {"expiration", "snapshot"}
        )
        eligible += covered
        expected = len(observations)
        if expected == 0:
            raise ValueError("Cannot publish an empty calculation owner")
        direct_ids = set(result_ids) | set(source_only_ids) | set(contract_failure_ids)
        if not direct_ids.issubset(observations):
            raise ValueError("Calculation outcome references a non-observation contract")
        if eligible != expected:
            raise ValueError(
                f"Calculation publication counts do not reconcile: {eligible} != {expected}"
            )
        sid = str(claim.snapshot_id)
        publication_token = f"{sid}:{claim.attempt_id}"
        now = _utc_now()
        status = "complete_with_failures" if failures else "complete"
        state = {
            "snapshot_id": sid,
            "source_collected_at": claim.source_collected_at,
            "underlying_symbol": next(iter(observations.values()))["underlying_symbol"],
            "calculation_definition_id": claim.definition_id,
            "status": status,
            "expected_observations": expected,
            "eligible_contracts": eligible,
            "successful_results": len(results),
            "source_only_contracts": len(source_only_ids),
            "terminal_contract_failures": terminal_contract_failures,
            "terminal_expiration_failures": expiration_failures,
            "snapshot_failure_count": snapshot_failures,
            "started_at": claim.started_at,
            "finished_at": now,
            "updated_at": now,
            "active_attempt_id": None,
            "error_summary": "",
            "is_active": True,
        }
        attempt = {
            "attempt_id": str(claim.attempt_id),
            "process_type": "calculation",
            "snapshot_id": sid,
            "source_collected_at": claim.source_collected_at,
            "definition_id": claim.definition_id,
            "worker_id": self.worker_id,
            "started_at": claim.started_at,
            "finished_at": now,
            "status": status,
            "expected_count": expected,
            "written_success_count": len(results),
            "failure_count": len(failures),
            "error_summary": "",
            "created_at": claim.started_at,
        }
        # All invariants above are checked before the first durable fact insert.
        # If an insert is interrupted, the nonterminal owner is repaired as one
        # bounded snapshot on the next claim.
        if expiration_inputs:
            self.client.insert_arrow(
                f"{self.database}.options_chain_expiration_inputs",
                _rows_to_arrow([dict(row) for row in expiration_inputs], EXPIRATION_INPUT_SCHEMA),
                settings={
                    "insert_deduplication_token": f"expiration-inputs:{publication_token}"
                },
            )
        if results:
            self.client.insert_arrow(
                f"{self.database}.options_chain_calculation_results",
                _rows_to_arrow(
                    _runtime_rows_with_native_lineage(
                        results, "source_calculation_run_id"
                    ),
                    RESULT_SCHEMA,
                ),
                settings={
                    "insert_deduplication_token": f"calculation-results:{publication_token}"
                },
            )
        if failures:
            self.client.insert_arrow(
                f"{self.database}.options_chain_calculation_failures",
                _rows_to_arrow(
                    _runtime_rows_with_native_lineage(
                        failures, "source_calculation_run_id"
                    ),
                    CALCULATION_FAILURE_SCHEMA,
                ),
            )
        if reuse_provenance:
            self.client.insert_arrow(
                f"{self.database}.options_chain_calculation_reuse_provenance",
                _rows_to_arrow([dict(row) for row in reuse_provenance], REUSE_PROVENANCE_SCHEMA),
                settings={
                    "insert_deduplication_token": f"reuse-provenance:{publication_token}"
                },
            )
        # Terminal state is the final publication barrier.
        self.client.insert_arrow(
            f"{self.database}.options_chain_processing_attempts",
            _rows_to_arrow([attempt], ATTEMPT_SCHEMA),
        )
        self.client.insert_arrow(
            f"{self.database}.options_chain_calculation_state",
            _rows_to_arrow(
                _runtime_rows_with_native_lineage([state], "source_calculation_run_id"),
                CALCULATION_STATE_SCHEMA,
            ),
        )
        return {
            "successful_results": len(results),
            "source_only_contracts": len(source_only_ids),
            "failures": len(failures),
        }


def _validate_surface_cadence_multiplier(value: Any) -> int:
    if isinstance(value, bool) or isinstance(value, float) and not value.is_integer():
        raise ValueError("surface cadence multiplier must be a positive integer")
    try:
        multiplier = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "surface cadence multiplier must be a positive integer"
        ) from exc
    if multiplier <= 0:
        raise ValueError("surface cadence multiplier must be a positive integer")
    return multiplier


def _validate_surface_max_minutes_for_calculation(value: Any) -> int:
    if isinstance(value, bool) or isinstance(value, float) and not value.is_integer():
        raise ValueError(
            "surface max_minutes_for_calculation must be a positive integer"
        )
    try:
        minutes = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "surface max_minutes_for_calculation must be a positive integer"
        ) from exc
    if minutes <= 0:
        raise ValueError(
            "surface max_minutes_for_calculation must be a positive integer"
        )
    return minutes


def surface_cadence_ctes(
    database: str,
    *,
    timeline_filter: str = "",
    snapshots_table: str = "options_chain_snapshots",
    surface_state_table: str = "options_chain_surface_state",
) -> str:
    """Return the canonical cadence timeline and selection CTEs.

    Active surface-state rows are durable cadence anchors, regardless of their
    status or whether they came from automatic or explicit processing.  The
    timeline intentionally contains snapshots with incomplete calculations so
    positions continue to mean every Nth snapshot.  Callers apply readiness
    and requested date ranges only after these CTEs have ranked the timeline.

    Table parameters allow migration and archival callers to apply the same
    cadence semantics to generation-scoped physical tables.
    """
    database = validate_database_identifier(database)
    snapshots_table = validate_database_identifier(snapshots_table)
    surface_state_table = validate_database_identifier(surface_state_table)
    return f"""
        surface_cadence_timeline AS
        (
            SELECT s.snapshot_id, s.source_collected_at, s.underlying_symbol,
                   s.stored_observation_count,
                   toUInt8(ifNull(u.is_active, false)) AS has_surface_state,
                   ifNull(u.surface_definition_id, 0) AS surface_definition_id,
                   ifNull(toString(u.status), '') AS surface_status,
                   u.started_at AS surface_started_at
            FROM {database}.{snapshots_table} AS s
            LEFT JOIN
            (
                SELECT *
                FROM {database}.{surface_state_table} FINAL
            ) AS u
              ON u.snapshot_id = s.snapshot_id
             AND u.source_collected_at = s.source_collected_at
             AND u.is_active
            {timeline_filter}
        ),
        surface_cadence_segmented AS
        (
            SELECT *,
                   sum(has_surface_state) OVER
                   (
                       PARTITION BY underlying_symbol
                       ORDER BY source_collected_at, snapshot_id
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS cadence_segment
            FROM surface_cadence_timeline
        ),
        surface_cadence_unowned AS
        (
            SELECT *,
                   row_number() OVER
                   (
                       PARTITION BY underlying_symbol, cadence_segment
                       ORDER BY source_collected_at, snapshot_id
                   ) AS position_in_segment
            FROM surface_cadence_segmented
            WHERE has_surface_state = 0
        ),
        surface_cadence_selected AS
        (
            SELECT *
            FROM surface_cadence_unowned
            WHERE (cadence_segment = 0 AND
                   modulo(position_in_segment - 1,
                          {{cadence_multiplier:UInt32}}) = 0)
               OR (cadence_segment > 0 AND
                   modulo(position_in_segment,
                          {{cadence_multiplier:UInt32}}) = 0)
        )
    """


def query_surface_cadence_health(
    client: Any,
    database: str,
    *,
    cadence_multiplier: int = 1,
    surface_enabled: bool = True,
) -> SurfaceCadenceHealth:
    """Return cadence-aware durable surface backlog and failure counts."""
    database = validate_database_identifier(database)
    cadence_multiplier = _validate_surface_cadence_multiplier(cadence_multiplier)
    parameters: dict[str, Any] = {"surface_enabled": bool(surface_enabled)}
    if cadence_multiplier == 1:
        rows = client.query(
            f"""
            SELECT countIf({{surface_enabled:Bool}} AND
                           (isNull(u.snapshot_id) OR u.status NOT IN
                            ('complete', 'complete_with_failures'))),
                   countIf({{surface_enabled:Bool}} AND u.status = 'failed'),
                   groupUniqArrayIf(u.surface_definition_id,
                                    NOT isNull(u.snapshot_id))
            FROM {database}.options_chain_snapshots AS s
            LEFT JOIN
            (
                SELECT * FROM {database}.options_chain_surface_state FINAL
            ) AS u
              ON u.snapshot_id = s.snapshot_id
             AND u.source_collected_at = s.source_collected_at
             AND u.is_active
            """,
            parameters=parameters,
        ).result_rows
    else:
        parameters["cadence_multiplier"] = cadence_multiplier
        cadence_ctes = surface_cadence_ctes(database)
        rows = client.query(
            f"""
            WITH {cadence_ctes},
            surface_pending_candidates AS
            (
                SELECT snapshot_id, source_collected_at
                FROM surface_cadence_selected
                UNION ALL
                SELECT snapshot_id, source_collected_at
                FROM surface_cadence_segmented
                WHERE has_surface_state = 1
                  AND surface_status NOT IN
                      ('complete', 'complete_with_failures')
            )
            SELECT if({{surface_enabled:Bool}},
                      (SELECT count() FROM surface_pending_candidates), 0),
                   if({{surface_enabled:Bool}},
                      (SELECT count()
                       FROM surface_cadence_timeline
                       WHERE has_surface_state = 1
                         AND surface_status = 'failed'), 0),
                   (SELECT groupUniqArrayIf(surface_definition_id,
                                            has_surface_state = 1)
                    FROM surface_cadence_timeline)
            """,
            parameters=parameters,
        ).result_rows
    if not rows:
        return SurfaceCadenceHealth(0, 0, ())
    pending, failed, definitions = rows[0]
    return SurfaceCadenceHealth(
        int(pending or 0),
        int(failed or 0),
        tuple(sorted(int(value) for value in (definitions or []))),
    )


def query_expired_surface_work(
    client: Any,
    database: str,
    *,
    max_minutes_for_calculation: int,
    snapshot_id: Optional[uuid.UUID] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    symbols: Optional[Sequence[str]] = None,
    retry_failed: bool = False,
    cadence_multiplier: int = 1,
) -> Optional[SurfaceWorkAgeReport]:
    """Return the newest automatic surface candidate beyond its age limit.

    The candidate selection intentionally mirrors :func:`discover_surface_work`.
    In particular, cadence is ranked over the complete symbol timeline before
    the requested range and age limit are applied.
    """
    database = validate_database_identifier(database)
    cadence_multiplier = _validate_surface_cadence_multiplier(cadence_multiplier)
    max_minutes = _validate_surface_max_minutes_for_calculation(
        max_minutes_for_calculation
    )
    cutoff_us = _datetime64_microseconds(
        _utc_now() - timedelta(minutes=max_minutes)
    )
    use_fast_path = cadence_multiplier == 1 or snapshot_id is not None
    filters: list[str] = []
    parameters: dict[str, Any] = {"surface_age_cutoff_us": cutoff_us}
    if snapshot_id is not None:
        filters.append("s.snapshot_id = {snapshot_id:UUID}")
        parameters["snapshot_id"] = str(snapshot_id)
    if start_at is not None:
        filters.append(
            "toUnixTimestamp64Micro(s.source_collected_at) >= {start_us:Int64}"
        )
        parameters["start_us"] = _datetime64_microseconds(start_at)
    if end_at is not None:
        filters.append(
            "toUnixTimestamp64Micro(s.source_collected_at) < {end_us:Int64}"
        )
        parameters["end_us"] = _datetime64_microseconds(end_at)
    if symbols:
        filters.append("s.underlying_symbol IN {symbols:Array(String)}")
        parameters["symbols"] = [str(value).upper() for value in symbols]
    if not retry_failed:
        filters.append("(isNull(u.snapshot_id) OR u.status != 'failed')")

    if use_fast_path:
        filters.append(
            "toUnixTimestamp64Micro(s.source_collected_at) "
            "< {surface_age_cutoff_us:Int64}"
        )
        where = " AND " + " AND ".join(filters) if filters else ""
        rows = client.query(
            f"""
            WITH expired_surface_candidates AS
            (
                SELECT s.snapshot_id AS snapshot_id,
                       s.source_collected_at AS source_collected_at,
                       s.underlying_symbol AS underlying_symbol
                FROM {database}.options_chain_snapshots AS s
                INNER JOIN
                (
                    SELECT * FROM {database}.options_chain_calculation_state FINAL
                ) AS c
                  ON c.snapshot_id = s.snapshot_id
                 AND c.source_collected_at = s.source_collected_at
                 AND c.is_active
                 AND c.status IN ('complete', 'complete_with_failures')
                LEFT JOIN
                (
                    SELECT * FROM {database}.options_chain_surface_state FINAL
                ) AS u
                  ON u.snapshot_id = s.snapshot_id
                 AND u.source_collected_at = s.source_collected_at
                 AND u.is_active
                WHERE (isNull(u.snapshot_id)
                       OR u.status NOT IN ('complete', 'complete_with_failures')){where}
            )
            SELECT toString(expired.snapshot_id), expired.underlying_symbol,
                   expired.source_collected_at, count() OVER () AS skipped_count
            FROM expired_surface_candidates AS expired
            ORDER BY expired.source_collected_at DESC, expired.snapshot_id DESC
            LIMIT 1
            """,
            parameters=parameters,
        ).result_rows
    else:
        parameters["cadence_multiplier"] = cadence_multiplier
        timeline_filter = ""
        if symbols:
            timeline_filter = "WHERE s.underlying_symbol IN {symbols:Array(String)}"
        cadence_ctes = surface_cadence_ctes(
            database, timeline_filter=timeline_filter
        )
        output_filters: list[str] = [
            "toUnixTimestamp64Micro(candidate.source_collected_at) "
            "< {surface_age_cutoff_us:Int64}"
        ]
        if start_at is not None:
            output_filters.append(
                "toUnixTimestamp64Micro(candidate.source_collected_at) "
                ">= {start_us:Int64}"
            )
        if end_at is not None:
            output_filters.append(
                "toUnixTimestamp64Micro(candidate.source_collected_at) "
                "< {end_us:Int64}"
            )
        output_where = "WHERE " + " AND ".join(output_filters)
        retryable_failed = "" if retry_failed else "AND surface_status != 'failed'"
        rows = client.query(
            f"""
            WITH {cadence_ctes},
            surface_candidates AS
            (
                SELECT snapshot_id, source_collected_at, underlying_symbol,
                       surface_status
                FROM surface_cadence_selected
                UNION ALL
                SELECT snapshot_id, source_collected_at, underlying_symbol,
                       surface_status
                FROM surface_cadence_segmented
                WHERE has_surface_state = 1
                  AND surface_status NOT IN
                      ('complete', 'complete_with_failures')
                  {retryable_failed}
            ),
            expired_surface_candidates AS
            (
                SELECT candidate.snapshot_id AS snapshot_id,
                       candidate.source_collected_at AS source_collected_at,
                       candidate.underlying_symbol AS underlying_symbol
                FROM surface_candidates AS candidate
                INNER JOIN
                (
                    SELECT * FROM {database}.options_chain_calculation_state FINAL
                ) AS c
                  ON c.snapshot_id = candidate.snapshot_id
                 AND c.source_collected_at = candidate.source_collected_at
                 AND c.is_active
                 AND c.status IN ('complete', 'complete_with_failures')
                {output_where}
            )
            SELECT toString(expired.snapshot_id), expired.underlying_symbol,
                   expired.source_collected_at, count() OVER () AS skipped_count
            FROM expired_surface_candidates AS expired
            ORDER BY expired.source_collected_at DESC, expired.snapshot_id DESC
            LIMIT 1
            """,
            parameters=parameters,
        ).result_rows
    if not rows:
        return None
    snapshot, symbol, source_at, skipped_count = rows[0]
    source_timestamp = _parse_timestamp(source_at)
    if source_timestamp is None:
        raise RuntimeError(
            f"Expired surface candidate has no source timestamp: {snapshot}"
        )
    return SurfaceWorkAgeReport(
        int(skipped_count),
        uuid.UUID(str(snapshot)),
        str(symbol),
        source_timestamp,
    )


def discover_surface_work(
    client: Any,
    database: str,
    *,
    definition_repository: Optional[DefinitionRepository] = None,
    limit: int = 100,
    stale_after_seconds: int = 900,
    snapshot_id: Optional[uuid.UUID] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    symbols: Optional[Sequence[str]] = None,
    retry_failed: bool = False,
    cadence_multiplier: int = 1,
    max_minutes_for_calculation: Optional[int] = None,
) -> list[SurfaceWork]:
    database = validate_database_identifier(database)
    cadence_multiplier = _validate_surface_cadence_multiplier(cadence_multiplier)
    minimum_source_collected_at_us: Optional[int] = None
    if max_minutes_for_calculation is not None:
        max_minutes = _validate_surface_max_minutes_for_calculation(
            max_minutes_for_calculation
        )
        minimum_source_collected_at_us = _datetime64_microseconds(
            _utc_now() - timedelta(minutes=max_minutes)
        )
    if maintenance_lease_active(client, database):
        return []
    # An explicit snapshot is operator-directed work and deliberately bypasses
    # automatic cadence.  Its durable claim becomes an anchor for later polls.
    use_fast_path = cadence_multiplier == 1 or snapshot_id is not None
    filters: list[str] = []
    parameters: dict[str, Any] = {"limit": int(limit)}
    if snapshot_id is not None:
        filters.append("s.snapshot_id = {snapshot_id:UUID}")
        parameters["snapshot_id"] = str(snapshot_id)
    if start_at is not None:
        filters.append(
            "toUnixTimestamp64Micro(s.source_collected_at) >= {start_us:Int64}"
        )
        parameters["start_us"] = _datetime64_microseconds(start_at)
    if end_at is not None:
        filters.append(
            "toUnixTimestamp64Micro(s.source_collected_at) < {end_us:Int64}"
        )
        parameters["end_us"] = _datetime64_microseconds(end_at)
    if symbols:
        filters.append("s.underlying_symbol IN {symbols:Array(String)}")
        parameters["symbols"] = [str(value).upper() for value in symbols]
    if not retry_failed:
        filters.append("(isNull(u.snapshot_id) OR u.status != 'failed')")
    if minimum_source_collected_at_us is not None:
        filters.append(
            "toUnixTimestamp64Micro(s.source_collected_at) "
            ">= {minimum_surface_source_collected_at_us:Int64}"
        )
        parameters["minimum_surface_source_collected_at_us"] = (
            minimum_source_collected_at_us
        )
    where = " AND " + " AND ".join(filters) if filters else ""
    if use_fast_path:
        rows = client.query(
            f"""
            SELECT toString(s.snapshot_id), s.source_collected_at, s.underlying_symbol,
                   s.stored_observation_count, c.calculation_definition_id,
                   u.surface_definition_id, toString(u.status), u.started_at
            FROM {database}.options_chain_snapshots AS s
            INNER JOIN (SELECT * FROM {database}.options_chain_calculation_state FINAL) AS c
              ON c.snapshot_id = s.snapshot_id
             AND c.source_collected_at = s.source_collected_at
             AND c.is_active
             AND c.status IN ('complete', 'complete_with_failures')
            LEFT JOIN (SELECT * FROM {database}.options_chain_surface_state FINAL) AS u
              ON u.snapshot_id = s.snapshot_id
             AND u.source_collected_at = s.source_collected_at
             AND u.is_active
            WHERE (isNull(u.snapshot_id)
                   OR u.status NOT IN ('complete', 'complete_with_failures')){where}
            ORDER BY s.source_collected_at LIMIT {{limit:UInt32}}
            """,
            parameters=parameters,
        ).result_rows
    else:
        parameters["cadence_multiplier"] = cadence_multiplier
        timeline_filter = ""
        if symbols:
            timeline_filter = (
                "WHERE s.underlying_symbol IN {symbols:Array(String)}"
            )
        cadence_ctes = surface_cadence_ctes(
            database, timeline_filter=timeline_filter
        )
        output_filters: list[str] = []
        if start_at is not None:
            output_filters.append(
                "toUnixTimestamp64Micro(candidate.source_collected_at) "
                ">= {start_us:Int64}"
            )
        if end_at is not None:
            output_filters.append(
                "toUnixTimestamp64Micro(candidate.source_collected_at) "
                "< {end_us:Int64}"
            )
        if minimum_source_collected_at_us is not None:
            output_filters.append(
                "toUnixTimestamp64Micro(candidate.source_collected_at) "
                ">= {minimum_surface_source_collected_at_us:Int64}"
            )
        output_where = (
            "WHERE " + " AND ".join(output_filters) if output_filters else ""
        )
        retryable_failed = "" if retry_failed else "AND surface_status != 'failed'"
        rows = client.query(
            f"""
            WITH {cadence_ctes},
            surface_candidates AS
            (
                SELECT snapshot_id, source_collected_at, underlying_symbol,
                       stored_observation_count, surface_definition_id,
                       surface_status, surface_started_at
                FROM surface_cadence_selected
                UNION ALL
                SELECT snapshot_id, source_collected_at, underlying_symbol,
                       stored_observation_count, surface_definition_id,
                       surface_status, surface_started_at
                FROM surface_cadence_segmented
                WHERE has_surface_state = 1
                  AND surface_status NOT IN
                      ('complete', 'complete_with_failures')
                  {retryable_failed}
            )
            SELECT toString(candidate.snapshot_id),
                   candidate.source_collected_at,
                   candidate.underlying_symbol,
                   candidate.stored_observation_count,
                   c.calculation_definition_id,
                   candidate.surface_definition_id,
                   candidate.surface_status,
                   candidate.surface_started_at
            FROM surface_candidates AS candidate
            INNER JOIN
            (
                SELECT *
                FROM {database}.options_chain_calculation_state FINAL
            ) AS c
              ON c.snapshot_id = candidate.snapshot_id
             AND c.source_collected_at = candidate.source_collected_at
             AND c.is_active
             AND c.status IN ('complete', 'complete_with_failures')
            {output_where}
            ORDER BY candidate.source_collected_at, candidate.snapshot_id
            LIMIT {{limit:UInt32}}
            """,
            parameters=parameters,
        ).result_rows
    repository = definition_repository or DefinitionRepository(client, database)
    now = _utc_now()
    output = []
    for sid, source_at, symbol, expected, calc_definition, surface_definition, status, started_at in rows:
        if str(status or "").lower() == "running":
            started = _parse_timestamp(started_at)
            if started and (now - started).total_seconds() < stale_after_seconds:
                continue
        source_at = _parse_timestamp(source_at)
        assigned = int(surface_definition or 0)
        if not assigned:
            assigned = repository.resolve(
                DefinitionType.SURFACE,
                source_at,
                parent_definition_id=int(calc_definition),
            ).definition_id
        output.append(
            SurfaceWork(
                uuid.UUID(str(sid)), source_at, str(symbol), int(calc_definition),
                assigned, int(expected), bool(retry_failed),
            )
        )
    return output


def discover_interrupted_surface_work(
    client: Any,
    database: str,
    *,
    definition_repository: Optional[DefinitionRepository] = None,
    snapshot_id: Optional[uuid.UUID] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    symbols: Optional[Sequence[str]] = None,
) -> list[InterruptedSurfaceWork]:
    """Discover retryable surface owners across all retained history.

    Surface recovery is restricted to snapshots whose source contract
    calculation is already terminal.  Like the calculation counterpart, this
    is intentionally status-scoped and unbounded only by the retained state
    table; it does not broaden normal backlog discovery.
    """
    database = validate_database_identifier(database)
    filters = ["surface.is_active", "surface.status IN ('running', 'abandoned', 'stale_running')"]
    parameters: dict[str, Any] = {}
    if snapshot_id is not None:
        filters.append("s.snapshot_id = {snapshot_id:UUID}")
        parameters["snapshot_id"] = str(snapshot_id)
    if start_at is not None:
        filters.append(
            "toUnixTimestamp64Micro(s.source_collected_at) >= {start_us:Int64}"
        )
        parameters["start_us"] = _datetime64_microseconds(start_at)
    if end_at is not None:
        filters.append(
            "toUnixTimestamp64Micro(s.source_collected_at) < {end_us:Int64}"
        )
        parameters["end_us"] = _datetime64_microseconds(end_at)
    if symbols:
        filters.append("s.underlying_symbol IN {symbols:Array(String)}")
        parameters["symbols"] = [str(value).upper() for value in symbols]
    rows = client.query(
        f"""
        SELECT toString(surface.snapshot_id), surface.source_collected_at,
               surface.underlying_symbol, surface.source_calculation_definition_id,
               surface.surface_definition_id, s.stored_observation_count,
               toString(surface.status), toString(surface.active_attempt_id)
        FROM (SELECT * FROM {database}.options_chain_surface_state FINAL) AS surface
        INNER JOIN {database}.options_chain_snapshots AS s
          ON s.snapshot_id = surface.snapshot_id
         AND s.source_collected_at = surface.source_collected_at
        INNER JOIN (SELECT * FROM {database}.options_chain_calculation_state FINAL) AS calculation
          ON calculation.snapshot_id = surface.snapshot_id
         AND calculation.source_collected_at = surface.source_collected_at
         AND calculation.is_active
         AND calculation.status IN ('complete', 'complete_with_failures')
        WHERE {' AND '.join(filters)}
        ORDER BY surface.source_collected_at
        """,
        parameters=parameters,
    ).result_rows
    repository = definition_repository or DefinitionRepository(client, database)
    output: list[InterruptedSurfaceWork] = []
    for (
        sid, source_at, symbol, calculation_definition, surface_definition,
        expected, status, active,
    ) in rows:
        source_timestamp = _parse_timestamp(source_at)
        if source_timestamp is None:
            raise RuntimeError(f"Surface recovery row has no source timestamp: {sid}")
        calculation_id = int(calculation_definition or 0)
        if not calculation_id:
            raise RuntimeError(f"Surface recovery row has no source definition: {sid}")
        surface_id = int(surface_definition or 0)
        if not surface_id:
            surface_id = repository.resolve(
                DefinitionType.SURFACE,
                source_timestamp,
                parent_definition_id=calculation_id,
            ).definition_id
        attempt = None
        if active and str(active).lower() not in {"", "none", "null"}:
            attempt = uuid.UUID(str(active))
        output.append(
            InterruptedSurfaceWork(
                SurfaceWork(
                    uuid.UUID(str(sid)), source_timestamp, str(symbol),
                    calculation_id, surface_id, int(expected),
                ),
                str(status).lower(),
                attempt,
            )
        )
    return output


def typed_fit_columns(fit: Mapping[str, Any]) -> dict[str, Any]:
    method = str(fit.get("fit_method") or fit.get("method") or "").upper()
    parameters = fit.get("parameters", fit)
    if method == "SVI":
        names = ("a", "b", "rho", "m", "sigma")
        values = {
            name: float(
                parameters[name]
                if name in parameters
                else parameters[f"svi_{name}"]
            )
            for name in names
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("SVI parameters must be finite")
        if values["b"] < 0 or values["sigma"] <= 0 or not -1 < values["rho"] < 1:
            raise ValueError("SVI parameters violate b/rho/sigma constraints")
        return {
            "fit_method": "SVI",
            **{f"svi_{name}": value for name, value in values.items()},
            "pchip_k": [],
            "pchip_total_variance": [],
        }
    if method == "PCHIP":
        k = [
            float(value)
            for value in parameters.get("k", parameters.get("pchip_k", ()))
        ]
        variance = [
            float(value)
            for value in parameters.get(
                "total_variance", parameters.get("pchip_total_variance", ())
            )
        ]
        if len(k) < 2 or len(k) != len(variance):
            raise ValueError("PCHIP arrays must have at least two equally sized points")
        if not all(math.isfinite(value) for value in (*k, *variance)):
            raise ValueError("PCHIP parameters must be finite")
        if any(right <= left for left, right in zip(k, k[1:])):
            raise ValueError("PCHIP knots must be strictly increasing")
        if any(value < 0 for value in variance):
            raise ValueError("PCHIP total variance cannot be negative")
        return {
            "fit_method": "PCHIP",
            "svi_a": None,
            "svi_b": None,
            "svi_rho": None,
            "svi_m": None,
            "svi_sigma": None,
            "pchip_k": k,
            "pchip_total_variance": variance,
        }
    raise ValueError(f"Unsupported fit method: {method!r}")


def validate_surface_failure(row: Mapping[str, Any]) -> None:
    scope = str(row.get("failure_scope") or "")
    has_node_key = all(
        row.get(name) is not None for name in ("node_type", "node_value", "put_call")
    )
    if scope == "fit" and any(
        row.get(name) is not None for name in ("node_type", "node_value", "put_call")
    ):
        raise ValueError("Fit-scope failure cannot contain a node key")
    if scope == "node" and not has_node_key:
        raise ValueError("Node-scope failure requires the complete requested node key")
    if scope not in {"fit", "node"}:
        raise ValueError(f"Unknown surface failure scope: {scope!r}")


def maintenance_lease_active(client: Any, database: str) -> bool:
    """Return whether SchemaMigration blocks writes, failing closed on errors.

    Generation 3 requires the migration control table.  A missing table,
    malformed schema, permission error, or unavailable server therefore cannot
    be interpreted as proof that no lease is active.
    """
    rows = client.query(
        f"SELECT count() FROM {validate_database_identifier(database)}."
        "options_chain_migration_leases FINAL "
        "WHERE status = 'active' AND expires_at > now64(6)"
    ).result_rows
    if len(rows) != 1:
        raise RuntimeError("Migration lease query did not return exactly one row")
    return bool(int(rows[0][0] or 0))


class SurfaceRepository:
    def __init__(
        self,
        client: Any,
        database: str,
        *,
        worker_id: Optional[str] = None,
        stale_after_seconds: int = 900,
    ):
        self.client = client
        self.database = validate_database_identifier(database)
        self.worker_id = worker_id or f"{socket.gethostname()}:{uuid.uuid4()}"
        self.stale_after_seconds = int(stale_after_seconds)
        if self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")

    def claim(
        self,
        work: SurfaceWork,
        *,
        expected_expirations: int,
        expected_required_nodes: int,
        attempt_id: Optional[uuid.UUID] = None,
    ) -> ProcessingClaim:
        if maintenance_lease_active(self.client, self.database):
            raise RuntimeError("Surface claims are disabled by an active migration lease")
        existing = self.client.query(
            f"SELECT toString(status), started_at FROM "
            f"{self.database}.options_chain_surface_state FINAL "
            "WHERE snapshot_id = {snapshot_id:UUID} AND is_active LIMIT 1",
            parameters={"snapshot_id": str(work.snapshot_id)},
        ).result_rows
        if existing and str(existing[0][0]) in TERMINAL_STATES:
            raise ValueError("A terminal historical surface state is authoritative")
        if existing and str(existing[0][0]) == "failed" and not work.retry_failed:
            raise ValueError("Failed surface requires explicit retry_failed selection")
        if existing and str(existing[0][0]) == "running":
            started = _parse_timestamp(existing[0][1])
            if started and (
                _utc_now() - started
            ).total_seconds() < self.stale_after_seconds:
                raise ValueError("Surface owner already has a live running claim")
        # Expired running, stale-running, abandoned, and explicitly selected
        # failed work all replay from a clean snapshot publication boundary.
        if existing:
            self.repair_partial_publication(work.snapshot_id)
        now = _utc_now()
        claim = ProcessingClaim(
            attempt_id or uuid.uuid4(), "surface", work.snapshot_id,
            work.source_collected_at, work.surface_definition_id, now,
        )
        state = {
            "snapshot_id": str(work.snapshot_id),
            "source_collected_at": work.source_collected_at,
            "underlying_symbol": work.underlying_symbol,
            "surface_definition_id": work.surface_definition_id,
            "source_calculation_definition_id": work.calculation_definition_id,
            "status": "running",
            "expected_expirations": int(expected_expirations),
            "successful_expirations": 0,
            "source_only_or_skipped_expirations": 0,
            "terminal_fit_failures": 0,
            "expected_required_nodes": int(expected_required_nodes),
            "successful_required_nodes": 0,
            "terminal_required_node_failures": 0,
            "successful_optional_nodes": 0,
            "terminal_optional_node_failures": 0,
            "started_at": now,
            "finished_at": None,
            "updated_at": now,
            "active_attempt_id": str(claim.attempt_id),
            "error_summary": "",
            "is_active": True,
        }
        attempt = {
            "attempt_id": str(claim.attempt_id),
            "process_type": "surface",
            "snapshot_id": str(work.snapshot_id),
            "source_collected_at": work.source_collected_at,
            "definition_id": work.surface_definition_id,
            "worker_id": self.worker_id,
            "started_at": now,
            "finished_at": None,
            "status": "running",
            "expected_count": int(expected_expirations),
            "written_success_count": 0,
            "failure_count": 0,
            "error_summary": "",
            "created_at": now,
        }
        self.client.insert_arrow(
            f"{self.database}.options_chain_surface_state",
            _rows_to_arrow(
                _runtime_rows_with_native_lineage([state], "source_surface_run_id"),
                SURFACE_STATE_SCHEMA,
            ),
        )
        self.client.insert_arrow(
            f"{self.database}.options_chain_processing_attempts",
            _rows_to_arrow([attempt], ATTEMPT_SCHEMA),
        )
        owner = self.client.query(
            f"SELECT toString(active_attempt_id) FROM "
            f"{self.database}.options_chain_surface_state FINAL "
            "WHERE snapshot_id = {snapshot_id:UUID} AND is_active LIMIT 1",
            parameters={"snapshot_id": str(work.snapshot_id)},
        ).result_rows
        if not owner or str(owner[0][0]) != str(claim.attempt_id):
            raise RuntimeError("Surface claim lost concurrent ownership")
        return claim

    def fail_attempt_if_running(
        self,
        *,
        snapshot_id: uuid.UUID,
        attempt_id: uuid.UUID,
        error: BaseException | str,
        underlying_symbol: Optional[str] = None,
        source_calculation_definition_id: Optional[int] = None,
        expected_expirations: Optional[int] = None,
        expected_required_nodes: Optional[int] = None,
    ) -> bool:
        """Fail only the still-active surface attempt identified by its UUID."""
        rows = self.client.query(
            f"SELECT toString(status), toString(active_attempt_id), "
            f"source_collected_at, underlying_symbol, source_calculation_definition_id, "
            f"surface_definition_id, expected_expirations, expected_required_nodes, "
            f"started_at FROM {self.database}.options_chain_surface_state FINAL "
            "WHERE snapshot_id = {snapshot_id:UUID} AND is_active LIMIT 1",
            parameters={"snapshot_id": str(snapshot_id)},
        ).result_rows
        if not rows:
            return False
        (
            status, active, source_at, symbol, source_definition, definition_id,
            expected, expected_required, started_at,
        ) = rows[0]
        if str(status) in TERMINAL_STATES or str(status) != "running":
            return False
        if str(active) != str(attempt_id):
            return False
        claim = ProcessingClaim(
            uuid.UUID(str(attempt_id)), "surface", snapshot_id,
            _parse_timestamp(source_at) or _utc_now(), int(definition_id),
            _parse_timestamp(started_at) or _utc_now(),
        )
        return self.fail(
            claim,
            underlying_symbol=str(underlying_symbol or symbol),
            source_calculation_definition_id=int(
                source_calculation_definition_id
                if source_calculation_definition_id is not None
                else source_definition
            ),
            expected_expirations=int(
                expected_expirations if expected_expirations is not None else expected
            ),
            expected_required_nodes=int(
                expected_required_nodes
                if expected_required_nodes is not None
                else expected_required
            ),
            error=error,
        )

    def abandon_attempt_if_running(
        self,
        *,
        snapshot_id: uuid.UUID,
        attempt_id: uuid.UUID,
        reason: str = "coordinator_shutdown",
    ) -> bool:
        """Mark one owned surface attempt as retryable infrastructure loss.

        Abandonment is not a numerical failure: it preserves the current
        surface counters, appends an abandonment event for the same attempt,
        and leaves the owner eligible for a new complete-snapshot replay.
        The authoritative ``FINAL`` ownership check makes the operation
        idempotent and ownership-aware.
        """
        if maintenance_lease_active(self.client, self.database):
            raise RuntimeError("Surface abandonment is disabled by an active migration lease")
        rows = self.client.query(
            f"""
            SELECT toString(status), toString(active_attempt_id),
                   source_collected_at, underlying_symbol,
                   source_calculation_definition_id, surface_definition_id,
                   expected_expirations, successful_expirations,
                   source_only_or_skipped_expirations, terminal_fit_failures,
                   expected_required_nodes, successful_required_nodes,
                   terminal_required_node_failures, successful_optional_nodes,
                   terminal_optional_node_failures, started_at
            FROM {self.database}.options_chain_surface_state FINAL
            WHERE snapshot_id = {{snapshot_id:UUID}} AND is_active
            LIMIT 1
            """,
            parameters={"snapshot_id": str(snapshot_id)},
        ).result_rows
        if not rows:
            return False
        (
            status, active, source_at, symbol, source_definition, definition_id,
            expected, successful, skipped, fit_failures, expected_required,
            successful_required, required_failures, successful_optional,
            optional_failures, started_at,
        ) = rows[0]
        if str(status).lower() != "running" or str(active) != str(attempt_id):
            return False
        now = _utc_now()
        started = _parse_timestamp(started_at) or now
        detail = str(reason).strip() or "coordinator_shutdown"
        attempt = {
            "attempt_id": str(attempt_id),
            "process_type": "surface",
            "snapshot_id": str(snapshot_id),
            "source_collected_at": _parse_timestamp(source_at) or now,
            "definition_id": int(definition_id),
            "worker_id": self.worker_id,
            "started_at": started,
            "finished_at": now,
            "status": "abandoned",
            "expected_count": int(expected),
            "written_success_count": int(successful) + int(successful_required) + int(successful_optional),
            "failure_count": int(fit_failures) + int(required_failures) + int(optional_failures),
            "error_summary": detail,
            "created_at": started,
        }
        state = {
            "snapshot_id": str(snapshot_id),
            "source_collected_at": _parse_timestamp(source_at) or now,
            "underlying_symbol": str(symbol),
            "surface_definition_id": int(definition_id),
            "source_calculation_definition_id": int(source_definition),
            "status": "abandoned",
            "expected_expirations": int(expected),
            "successful_expirations": int(successful),
            "source_only_or_skipped_expirations": int(skipped),
            "terminal_fit_failures": int(fit_failures),
            "expected_required_nodes": int(expected_required),
            "successful_required_nodes": int(successful_required),
            "terminal_required_node_failures": int(required_failures),
            "successful_optional_nodes": int(successful_optional),
            "terminal_optional_node_failures": int(optional_failures),
            "started_at": started,
            "finished_at": now,
            "updated_at": now,
            "active_attempt_id": None,
            "error_summary": detail,
            "is_active": True,
        }
        self.client.insert_arrow(
            f"{self.database}.options_chain_processing_attempts",
            _rows_to_arrow([attempt], ATTEMPT_SCHEMA),
        )
        self.client.insert_arrow(
            f"{self.database}.options_chain_surface_state",
            _rows_to_arrow(
                _runtime_rows_with_native_lineage([state], "source_surface_run_id"),
                SURFACE_STATE_SCHEMA,
            ),
        )
        verified = self.client.query(
            f"SELECT toString(status), toString(active_attempt_id) "
            f"FROM {self.database}.options_chain_surface_state FINAL "
            "WHERE snapshot_id = {snapshot_id:UUID} AND is_active LIMIT 1",
            parameters={"snapshot_id": str(snapshot_id)},
        ).result_rows
        if (
            not verified
            or str(verified[0][0]).lower() != "abandoned"
            or str(verified[0][1]).lower() not in {"", "none", "null"}
        ):
            raise RuntimeError("Surface abandonment could not be verified")
        return True

    def repair_partial_publication(self, snapshot_id: uuid.UUID) -> None:
        """Remove one nonterminal surface owner's derived rows before replay."""
        parameters = {"snapshot_id": str(snapshot_id)}
        for table in (
            "options_chain_surface_fits",
            "options_chain_surface_nodes",
            "options_chain_surface_failures",
            "options_chain_surface_diagnostics",
            "options_chain_surface_validation",
        ):
            self.client.command(
                f"ALTER TABLE {self.database}.{table} DELETE "
                "WHERE snapshot_id = {snapshot_id:UUID}",
                parameters=parameters,
                settings={"mutations_sync": 2},
            )

    def reset_terminal_for_recalculation(self, snapshot_id: uuid.UUID) -> bool:
        """Remove one completed surface so an explicit replay can reclaim it.

        Contract-coverage expansion uses this only when the operator explicitly
        asks to refresh existing surfaces.  An inactive replacement state is
        written after the owned facts are synchronously removed, so normal
        discovery cannot treat the old surface as authoritative.
        """
        if maintenance_lease_active(self.client, self.database):
            raise RuntimeError(
                "Surface recalculation reset is disabled by an active migration lease"
            )
        rows = self.client.query_arrow(
            f"""
            SELECT toString(snapshot_id) AS snapshot_id, source_collected_at,
                   underlying_symbol, surface_definition_id,
                   source_calculation_definition_id,
                   toString(source_surface_run_id) AS source_surface_run_id,
                   toString(status) AS status, expected_expirations,
                   successful_expirations,
                   source_only_or_skipped_expirations, terminal_fit_failures,
                   expected_required_nodes, successful_required_nodes,
                   terminal_required_node_failures, successful_optional_nodes,
                   terminal_optional_node_failures, started_at, finished_at,
                   error_summary
            FROM {self.database}.options_chain_surface_state FINAL
            WHERE snapshot_id = {{snapshot_id:UUID}} AND is_active
            LIMIT 1
            """,
            parameters={"snapshot_id": str(snapshot_id)},
        ).to_pylist()
        if not rows:
            return False
        state = dict(rows[0])
        if str(state["status"]) not in TERMINAL_STATES:
            raise ValueError(
                f"Surface {snapshot_id} is not terminal and cannot be replaced"
            )
        self.repair_partial_publication(snapshot_id)
        lineage = str(state.get("source_surface_run_id") or "").strip()
        state.update(
            source_surface_run_id=lineage or None,
            updated_at=_utc_now(),
            active_attempt_id=None,
            is_active=False,
        )
        self.client.insert_arrow(
            f"{self.database}.options_chain_surface_state",
            _rows_to_arrow([state], SURFACE_STATE_SCHEMA),
        )
        return True

    def load_source_rows(self, snapshot_id: uuid.UUID) -> list[dict[str, Any]]:
        """Load surface inputs through the shared effective-value query policy."""
        query = build_effective_chain_query(self.database, snapshot_id)
        table = query.execute(self.client)
        rows = table.to_pylist()
        expiration_inputs = self.client.query_arrow(
            f"""
            SELECT expiration_at, time_to_expiry_seconds, risk_free_rate,
                   dividend_yield, forward_price, fred_observation_date,
                   fred_lower_series, fred_upper_series, quality_flags
            FROM {authoritative_calculation_fact_subquery(self.database, 'options_chain_expiration_inputs', fact_alias='i', state_alias='calculation_state')}
            WHERE i.snapshot_id = {{snapshot_id:UUID}}
              AND i.source_collected_at = (SELECT source_collected_at
                FROM {self.database}.options_chain_snapshots
                WHERE snapshot_id = {{snapshot_id:UUID}} LIMIT 1)
            ORDER BY expiration_at
            """,
            parameters={"snapshot_id": str(snapshot_id)},
        ).to_pylist()
        by_expiration = {row["expiration_at"]: row for row in expiration_inputs}
        for row in rows:
            expiration = row.get("expiration_at")
            inputs = by_expiration.get(expiration)
            if inputs is None:
                raise ValueError(
                    f"Surface source has no expiration inputs for {expiration!r}"
                )
            row.update({f"expiration_{name}": value for name, value in inputs.items()})
        return rows

    def fail(
        self,
        claim: ProcessingClaim,
        *,
        underlying_symbol: str,
        source_calculation_definition_id: int,
        expected_expirations: int,
        expected_required_nodes: int,
        error: BaseException | str,
    ) -> bool:
        """Finish an incomplete surface attempt without fabricating fit failures."""
        if claim.process_type != "surface":
            raise ValueError("Surface failure received a non-surface claim")
        owner = self.client.query(
            f"SELECT toString(status), toString(active_attempt_id) FROM "
            f"{self.database}.options_chain_surface_state FINAL "
            "WHERE snapshot_id = {snapshot_id:UUID} AND is_active LIMIT 1",
            parameters={"snapshot_id": str(claim.snapshot_id)},
        ).result_rows
        if (
            not owner
            or str(owner[0][0]) != "running"
            or str(owner[0][1]) != str(claim.attempt_id)
        ):
            return False
        now = _utc_now()
        detail = str(error)
        attempt = {
            "attempt_id": str(claim.attempt_id),
            "process_type": "surface",
            "snapshot_id": str(claim.snapshot_id),
            "source_collected_at": claim.source_collected_at,
            "definition_id": claim.definition_id,
            "worker_id": self.worker_id,
            "started_at": claim.started_at,
            "finished_at": now,
            "status": "failed",
            "expected_count": int(expected_expirations),
            "written_success_count": 0,
            "failure_count": 1,
            "error_summary": detail,
            "created_at": claim.started_at,
        }
        state = {
            "snapshot_id": str(claim.snapshot_id),
            "source_collected_at": claim.source_collected_at,
            "underlying_symbol": underlying_symbol,
            "surface_definition_id": claim.definition_id,
            "source_calculation_definition_id": int(
                source_calculation_definition_id
            ),
            "status": "failed",
            "expected_expirations": int(expected_expirations),
            "successful_expirations": 0,
            "source_only_or_skipped_expirations": 0,
            "terminal_fit_failures": 0,
            "expected_required_nodes": int(expected_required_nodes),
            "successful_required_nodes": 0,
            "terminal_required_node_failures": 0,
            "successful_optional_nodes": 0,
            "terminal_optional_node_failures": 0,
            "started_at": claim.started_at,
            "finished_at": now,
            "updated_at": now,
            "active_attempt_id": None,
            "error_summary": detail,
            "is_active": True,
        }
        self.client.insert_arrow(
            f"{self.database}.options_chain_processing_attempts",
            _rows_to_arrow([attempt], ATTEMPT_SCHEMA),
        )
        self.client.insert_arrow(
            f"{self.database}.options_chain_surface_state",
            _rows_to_arrow(
                _runtime_rows_with_native_lineage([state], "source_surface_run_id"),
                SURFACE_STATE_SCHEMA,
            ),
        )
        return True

    def publish(
        self,
        claim: ProcessingClaim,
        *,
        source_calculation_definition_id: int,
        fits: Sequence[Mapping[str, Any]],
        nodes: Sequence[Mapping[str, Any]],
        failures: Sequence[Mapping[str, Any]],
        expected_expirations: int,
        skipped_expirations: int,
        expected_required_nodes: int,
        underlying_symbol: Optional[str] = None,
    ) -> Mapping[str, int]:
        if claim.process_type != "surface":
            raise ValueError("Surface publication received a non-surface claim")
        fit_keys = [
            (row["expiration_at"], str(row["price_basis"])) for row in fits
        ]
        node_keys = [
            (
                row["expiration_at"], str(row["price_basis"]), str(row["node_type"]),
                float(row["node_value"]), str(row["put_call"]),
            )
            for row in nodes
        ]
        if len(fit_keys) != len(set(fit_keys)) or len(node_keys) != len(set(node_keys)):
            raise ValueError("Surface success natural keys are not unique")
        for row in fits:
            if int(row["surface_definition_id"]) != claim.definition_id:
                raise ValueError("Fit definition differs from assigned surface state")
            typed = typed_fit_columns(row)
            if typed["fit_method"] != str(row["fit_method"]).upper():
                raise ValueError("Persisted fit method disagrees with typed parameters")
            for name in (
                "source_iv_weight_fraction", "minimum_log_forward_moneyness",
                "maximum_log_forward_moneyness", "weighted_rmse",
                "maximum_absolute_error",
            ):
                if _finite_float(row.get(name)) is None:
                    raise ValueError(f"Successful surface fit has no finite {name}")
        for row in nodes:
            if int(row["surface_definition_id"]) != claim.definition_id:
                raise ValueError("Node definition differs from assigned surface state")
            for name in ("fitted_strike", "fitted_iv_mid", "fit_confidence"):
                if _finite_float(row.get(name)) is None:
                    raise ValueError(f"Successful surface node has no finite {name}")
            fitted_delta = row.get("fitted_delta")
            if str(row.get("node_type")) == "delta":
                if _finite_float(fitted_delta) is None:
                    raise ValueError("Successful delta node has no finite fitted_delta")
            elif fitted_delta is not None and _finite_float(fitted_delta) is None:
                raise ValueError("Successful surface node has nonfinite fitted_delta")
        for row in failures:
            validate_surface_failure(row)
            if int(row["surface_definition_id"]) != claim.definition_id:
                raise ValueError("Surface failure definition differs from assigned state")
        fit_failure_keys = {
            (row["expiration_at"], str(row["price_basis"]))
            for row in failures
            if row["failure_scope"] == "fit"
        }
        fit_failure_key_list = [
            (row["expiration_at"], str(row["price_basis"]))
            for row in failures
            if row["failure_scope"] == "fit"
        ]
        if len(fit_failure_key_list) != len(set(fit_failure_key_list)):
            raise ValueError("Duplicate fit failure natural key")
        if set(fit_keys) & fit_failure_keys:
            raise ValueError("A fit natural key cannot be both success and failure")
        node_failure_keys = {
            (
                row["expiration_at"], str(row["price_basis"]), str(row["node_type"]),
                float(row["node_value"]), str(row["put_call"]),
            )
            for row in failures
            if row["failure_scope"] == "node"
        }
        node_failure_key_list = [
            (
                row["expiration_at"],
                str(row["price_basis"]),
                str(row["node_type"]),
                float(row["node_value"]),
                str(row["put_call"]),
            )
            for row in failures
            if row["failure_scope"] == "node"
        ]
        if len(node_failure_key_list) != len(set(node_failure_key_list)):
            raise ValueError("Duplicate node failure natural key")
        if set(node_keys) & node_failure_keys:
            raise ValueError("A node natural key cannot be both success and failure")
        fit_failures = sum(row["failure_scope"] == "fit" for row in failures)
        primary_successes = {
            row["expiration_at"]
            for row in fits
            if str(row["price_basis"]) in {"mid", "source"}
        }
        primary_failure_expirations = {
            row["expiration_at"]
            for row in failures
            if row["failure_scope"] == "fit"
            and str(row["price_basis"]) in {"mid", "source"}
            and row["expiration_at"] not in primary_successes
        }
        successful_expirations = len(primary_successes)
        if (
            successful_expirations
            + int(skipped_expirations)
            + len(primary_failure_expirations)
            != int(expected_expirations)
        ):
            raise ValueError("Surface expiration counts do not reconcile")
        required_successes = sum(bool(row.get("is_required", False)) for row in nodes)
        required_failures = sum(
            row["failure_scope"] == "node" and bool(row.get("is_required", False))
            for row in failures
        )
        if required_successes + required_failures != int(expected_required_nodes):
            raise ValueError("Required surface-node counts do not reconcile")
        optional_successes = len(nodes) - required_successes
        optional_failures = sum(
            row["failure_scope"] == "node" and not bool(row.get("is_required", False))
            for row in failures
        )
        now = _utc_now()
        status = "complete_with_failures" if failures else "complete"
        symbol = underlying_symbol or (
            str(fits[0]["underlying_symbol"])
            if fits
            else str(nodes[0]["underlying_symbol"])
            if nodes
            else str(failures[0]["underlying_symbol"])
            if failures
            else ""
        )
        if not symbol:
            raise ValueError("Surface publication requires an underlying symbol")
        sid = str(claim.snapshot_id)
        publication_token = f"{sid}:{claim.attempt_id}"
        state = {
            "snapshot_id": sid,
            "source_collected_at": claim.source_collected_at,
            "underlying_symbol": symbol,
            "surface_definition_id": claim.definition_id,
            "source_calculation_definition_id": int(source_calculation_definition_id),
            "status": status,
            "expected_expirations": int(expected_expirations),
            "successful_expirations": successful_expirations,
            "source_only_or_skipped_expirations": int(skipped_expirations),
            "terminal_fit_failures": fit_failures,
            "expected_required_nodes": int(expected_required_nodes),
            "successful_required_nodes": required_successes,
            "terminal_required_node_failures": required_failures,
            "successful_optional_nodes": optional_successes,
            "terminal_optional_node_failures": optional_failures,
            "started_at": claim.started_at,
            "finished_at": now,
            "updated_at": now,
            "active_attempt_id": None,
            "error_summary": "",
            "is_active": True,
        }
        attempt = {
            "attempt_id": str(claim.attempt_id),
            "process_type": "surface",
            "snapshot_id": sid,
            "source_collected_at": claim.source_collected_at,
            "definition_id": claim.definition_id,
            "worker_id": self.worker_id,
            "started_at": claim.started_at,
            "finished_at": now,
            "status": status,
            "expected_count": int(expected_expirations),
            "written_success_count": len(fits) + len(nodes),
            "failure_count": len(failures),
            "error_summary": "",
            "created_at": claim.started_at,
        }
        # Validate state reconciliation before writing any successful fact.
        if fits:
            self.client.insert_arrow(
                f"{self.database}.options_chain_surface_fits",
                _rows_to_arrow(
                    _runtime_rows_with_native_lineage(fits, "source_surface_run_id"),
                    SURFACE_FIT_SCHEMA,
                ),
                settings={
                    "insert_deduplication_token": f"surface-fits:{publication_token}"
                },
            )
        if nodes:
            self.client.insert_arrow(
                f"{self.database}.options_chain_surface_nodes",
                _rows_to_arrow(
                    _runtime_rows_with_native_lineage(nodes, "source_surface_run_id"),
                    SURFACE_NODE_SCHEMA,
                ),
                settings={
                    "insert_deduplication_token": f"surface-nodes:{publication_token}"
                },
            )
        if failures:
            self.client.insert_arrow(
                f"{self.database}.options_chain_surface_failures",
                _rows_to_arrow(
                    _runtime_rows_with_native_lineage(
                        failures, "source_surface_run_id"
                    ),
                    SURFACE_FAILURE_SCHEMA,
                ),
            )
        self.client.insert_arrow(
            f"{self.database}.options_chain_processing_attempts",
            _rows_to_arrow([attempt], ATTEMPT_SCHEMA),
        )
        # A node failure does not touch a successful fit; terminal state is last.
        self.client.insert_arrow(
            f"{self.database}.options_chain_surface_state",
            _rows_to_arrow(
                _runtime_rows_with_native_lineage([state], "source_surface_run_id"),
                SURFACE_STATE_SCHEMA,
            ),
        )
        return {"fits": len(fits), "nodes": len(nodes), "failures": len(failures)}


__all__ = [
    "ATTEMPT_SCHEMA",
    "CALCULATION_DIAGNOSTIC_SCHEMA",
    "CALCULATION_FAILURE_SCHEMA",
    "CALCULATION_STATE_SCHEMA",
    "CalculationRepository",
    "CalculationWork",
    "InterruptedCalculationWork",
    "InterruptedSurfaceWork",
    "EXPIRATION_INPUT_SCHEMA",
    "ProcessingClaim",
    "RESULT_SCHEMA",
    "REUSE_PROVENANCE_SCHEMA",
    "DATASET_METADATA_SCHEMA",
    "SnapshotReadiness",
    "SURFACE_FAILURE_SCHEMA",
    "SURFACE_DIAGNOSTIC_SCHEMA",
    "SURFACE_FIT_SCHEMA",
    "SURFACE_NODE_SCHEMA",
    "SURFACE_STATE_SCHEMA",
    "maintenance_lease_active",
    "SurfaceRepository",
    "SurfaceCadenceHealth",
    "SurfaceWorkAgeReport",
    "SurfaceWork",
    "discover_calculation_work",
    "discover_interrupted_calculation_work",
    "discover_interrupted_surface_work",
    "discover_surface_work",
    "query_expired_surface_work",
    "query_surface_cadence_health",
    "surface_cadence_ctes",
    "snapshot_readiness",
    "typed_fit_columns",
    "validate_calculation_failure",
    "validate_normalized_source_schema",
    "validate_success_result",
    "validate_surface_failure",
]
