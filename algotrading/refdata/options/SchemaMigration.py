"""Resumable offline migration and reversible cutover to canonical storage.

Safe operator sequence::

    --plan -> --all -> --validate -> --cutover -> (soak) -> --cleanup-legacy

``--plan`` and all non-destructive data phases support ``--dry-run``.  Dry runs
render/count planned work without advancing checkpoints or mutating ClickHouse;
cutover, rollback, and cleanup reject dry-run mode.  The migration holds an
explicit maintenance lease while it mutates data and object names.  Workers
refuse new claims under that lease, and stale takeover requires an explicit
``--takeover-stale-lease`` flag.  Every cutover, rollback, and cleanup mutation
has a durable checkpoint boundary, so ``--resume`` reconciles actual UUID
ownership before continuing.  Cleanup uses immutable ``cutover_completed_at``
for the configured soak period.

Exit codes are 0 for success/no-op, 2 for argparse usage errors, 3 for
configuration errors, 4 for blocked work or a live lease, and 6 for migration
execution failures.  Legacy object names are deliberately isolated in this
one-use module; normal runtime modules use the canonical registry in
``Schema.py``.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pyarrow as pa

from algotrading.Common import load_configuration
from algotrading.ContextFields import PARAM_ALGO_HOME, PARAM_LOGGER
from algotrading.refdata.options.Archival import (
    ArchiveConfig,
    RawWeekArchiver,
    week_is_age_eligible,
)
from algotrading.refdata.options.Common import (
    _compact_json,
    _parse_timestamp,
    _rows_to_arrow,
    _utc_now,
    create_clickhouse_client,
    validate_database_identifier,
)
from algotrading.refdata.options.Definitions import (
    DEFINITION_SCHEMA,
    DefinitionType,
    build_definition,
)
from algotrading.refdata.options.Schema import (
    CANONICAL_SCHEMA,
    CANONICAL_TABLE_NAMES,
    SCHEMA_GENERATION,
    migration_support_schema_statements,
)
from algotrading.refdata.options.Runtime import (
    SURFACE_FAILURE_SCHEMA as CANONICAL_SURFACE_FAILURE_SCHEMA,
    SURFACE_FIT_SCHEMA as CANONICAL_SURFACE_FIT_SCHEMA,
    SURFACE_NODE_SCHEMA as CANONICAL_SURFACE_NODE_SCHEMA,
    SURFACE_STATE_SCHEMA as CANONICAL_SURFACE_STATE_SCHEMA,
)
from algotrading.refdata.options.Normalization import (
    CONTRACT_SCHEMA as CANONICAL_CONTRACT_SCHEMA,
    contract_identity_hash,
    contract_metadata_hash,
)
from algotrading.refdata.options.Surface import is_required_surface_node


UTC = timezone.utc

# These names are migration input only.  Their presence outside this module,
# migration fixtures, or rollback documentation is rejected by the static test.
LEGACY_TABLES = (
    "options_chain_raw",
    "options_chain_base",
    "options_chain_calculations",
    "options_chain_calculation_runs",
    "options_chain_reuse_validation",
    "options_chain_reference_rates",
    "options_chain_surface_runs",
    "options_chain_surface_fits",
    "options_chain_surface_nodes",
    "options_chain_surface_validation",
)
LEGACY_VIEWS = (
    "options_chain",
    "options_chain_calculations_latest",
    "options_chain_calculation_runs_latest",
    "options_chain_calculation_runs_latest_complete",
    "options_chain_surface_runs_latest",
    "options_chain_surface_runs_latest_complete",
    "options_chain_surface_fits_latest",
    "options_chain_surface_nodes_latest",
    "options_chain_reference_rates_latest",
)


SOURCE_COLUMN_DISPOSITION = {
    # snapshot
    "collected_at": "moved to snapshot/source_collected_at",
    "request_started_at": "moved to snapshot",
    "response_received_at": "moved to snapshot",
    "ingested_at": "moved to snapshot",
    "raw_archive_date": "derived",
    "source": "moved to snapshot",
    "underlying_symbol": "moved to snapshot and denormalized fact key",
    "chain_status": "moved to snapshot",
    "strategy": "moved to snapshot",
    "chain_interval": "moved to snapshot",
    "chain_is_delayed": "moved to snapshot",
    "chain_is_index": "moved to snapshot",
    "chain_is_truncated": "moved to snapshot",
    "underlying_is_delayed": "moved to snapshot",
    "source_contract_count": "moved to snapshot",
    "stored_contract_count": "moved to snapshot/stored_observation_count",
    "asset_main_type": "moved to snapshot",
    "asset_sub_type": "moved to snapshot",
    "chain_interest_rate_raw": "intentionally dropped normalized duplicate",
    "chain_interest_rate": "moved to snapshot",
    "chain_dividend_yield_raw": "intentionally dropped normalized duplicate",
    "chain_dividend_yield": "moved to snapshot",
    "chain_volatility_raw": "moved to snapshot/chain_volatility",
    "chain_days_to_expiration": "derived",
    "chain_underlying_price": "moved to snapshot",
    "underlying_bid": "moved to snapshot",
    "underlying_ask": "moved to snapshot",
    "underlying_last": "moved to snapshot",
    "underlying_mark": "moved to snapshot",
    "underlying_bid_size": "moved to snapshot",
    "underlying_ask_size": "moved to snapshot",
    "underlying_quote_time": "moved to snapshot",
    "underlying_trade_time": "moved to snapshot",
    "underlying_total_volume": "moved to snapshot",
    # contract
    "option_symbol": "moved to contract",
    "option_root": "moved to contract",
    "ssid": "moved to contract",
    "description": "moved to contract",
    "exchange_name": "moved to contract",
    "put_call": "moved to contract",
    "strike_price": "moved to contract",
    "expiration_at": "moved to contract and expiration input",
    "expiration_date": "derived",
    "source_days_to_expiration": "derived",
    "expiration_type": "moved to contract",
    "last_trading_day": "moved to contract",
    "settlement_type": "moved to contract",
    "exercise_type": "moved to contract",
    "multiplier": "moved to contract",
    "is_in_the_money": "derived",
    "is_mini": "moved to contract",
    "is_nonstandard": "moved to contract",
    "is_penny_pilot": "moved to contract",
    "deliverable_note": "moved to contract",
    "option_deliverables_json": "moved to contract",
    # hot observation
    "contract_ordinal": "retained hot",
    "symbol_map_mismatch": "retained hot",
    "bid": "retained hot",
    "ask": "retained hot",
    "last": "retained hot",
    "mark": "retained hot",
    "bid_size": "retained hot",
    "ask_size": "retained hot",
    "last_size": "retained hot",
    "source_bid_ask_size": "intentionally dropped string duplicate",
    "open_price": "retained hot",
    "high_price": "retained hot",
    "low_price": "retained hot",
    "close_price": "retained hot",
    "total_volume": "retained hot",
    "open_interest": "retained hot",
    "quote_time": "retained hot",
    "trade_time": "retained hot",
    "source_volatility_raw": "intentionally dropped normalized duplicate",
    "source_iv": "retained hot",
    "source_delta": "retained hot",
    "source_gamma": "retained hot",
    "source_theta": "retained hot",
    "source_vega": "retained hot",
    "source_rho": "retained hot",
    # details
    "high_52_week": "retained detail",
    "low_52_week": "retained detail",
    "net_change": "retained detail",
    "percent_change": "retained detail",
    "mark_change": "retained detail",
    "mark_percent_change": "retained detail",
    "intrinsic_value": "retained detail",
    "extrinsic_value": "retained detail",
    "time_value": "retained detail",
    "break_even": "retained detail",
    "theoretical_option_value": "retained detail",
    "theoretical_volatility_raw": "retained detail/theoretical_volatility",
}

FLOAT32_CAST_THRESHOLDS = {
    # A cast is accepted when either its absolute or relative bound is met.
    "bid": (1e-4, 2e-7),
    "ask": (1e-4, 2e-7),
    "last": (1e-4, 2e-7),
    "mark": (1e-4, 2e-7),
    "open_price": (1e-4, 2e-7),
    "high_price": (1e-4, 2e-7),
    "low_price": (1e-4, 2e-7),
    "close_price": (1e-4, 2e-7),
    "high_52_week": (1e-3, 2e-7),
    "low_52_week": (1e-3, 2e-7),
    "net_change": (1e-4, 2e-7),
    "mark_change": (1e-4, 2e-7),
    "intrinsic_value": (1e-4, 2e-7),
    "extrinsic_value": (1e-4, 2e-7),
    "time_value": (1e-4, 2e-7),
    "break_even": (1e-3, 2e-7),
    "theoretical_option_value": (1e-4, 2e-7),
    "percent_change": (1e-4, 2e-7),
    "mark_percent_change": (1e-4, 2e-7),
    "theoretical_volatility_raw": (1e-7, 2e-7),
    "source_iv": (1e-7, 2e-7),
    "source_delta": (1e-7, 2e-7),
    "source_gamma": (1e-7, 2e-7),
    "source_theta": (1e-7, 2e-7),
    "source_vega": (1e-7, 2e-7),
    "source_rho": (1e-7, 2e-7),
}

SOURCE_NULLABLE_TARGETS = {
    **{
        name: ("options_chain_observations", name)
        for name in (
            "bid", "ask", "last", "mark", "bid_size", "ask_size", "last_size",
            "open_price", "high_price", "low_price", "close_price", "total_volume",
            "open_interest", "quote_time", "trade_time", "source_iv", "source_delta",
            "source_gamma", "source_theta", "source_vega", "source_rho",
        )
    },
    **{
        name: ("options_chain_observation_details", name)
        for name in (
            "high_52_week", "low_52_week", "net_change", "percent_change",
            "mark_change", "mark_percent_change", "intrinsic_value",
            "extrinsic_value", "time_value", "break_even",
            "theoretical_option_value",
        )
    },
    "theoretical_volatility_raw": (
        "options_chain_observation_details",
        "theoretical_volatility",
    ),
}


class MigrationPhase(StrEnum):
    NEW = "new"
    PREFLIGHT_COMPLETE = "preflight_complete"
    PLANNED = "planned"
    TARGET_SCHEMA_CREATED = "target_schema_created"
    # These aliases keep callers written against the original implementation
    # source-compatible while the persisted values describe the actual barrier.
    TARGET_SCHEMA = "target_schema_created"
    SOURCES_MIGRATED = "sources_migrated"
    SOURCES = "sources_migrated"
    CALCULATIONS_MIGRATED = "calculations_migrated"
    CALCULATIONS = "calculations_migrated"
    SURFACES_MIGRATED = "surfaces_migrated"
    SURFACES = "surfaces_migrated"
    RAW_ARCHIVED = "raw_archived"
    RAW_ARCHIVE = "raw_archived"
    VALIDATED = "validated"
    PREPARING_CUTOVER = "preparing_cutover"
    CUTTING_OVER = "cutting_over"
    CUTOVER_RENAMES_COMPLETE = "cutover_renames_complete"
    CUTOVER = "cutover"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    CLEANING = "cleaning"
    CLEANED = "cleaned"
    FAILED = "failed"


class CalculationClassification(StrEnum):
    SUCCESS = "success"
    SOURCE_ONLY = "source_only"
    TERMINAL_FAILURE = "terminal_failure"
    PENDING = "pending"


class SurfaceExpirationClassification(StrEnum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILURE = "failure"


CALCULATION_SUCCESS_FIELDS = (
    "recalc_target_price", "recalc_iv", "recalc_price_residual",
    "recalc_model_price", "recalc_delta", "recalc_gamma",
    "recalc_theta_per_day", "recalc_vega_per_1pct", "recalc_rho_per_1pct",
)


def _finite_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def classify_calculation_record(row: Mapping[str, Any]) -> CalculationClassification:
    """Classify one legacy calculation row using runtime outcome semantics.

    The failure checks intentionally precede the source-only checks.  This is
    the important distinction that the old SQL lost for ``failed + skipped``
    and ``failed + disabled`` rows.
    """
    calculation_status = str(row.get("calculation_status") or "").lower()
    recalc_status = str(row.get("recalc_status") or "").lower()
    if calculation_status == "failed" or recalc_status == "failed":
        return CalculationClassification.TERMINAL_FAILURE
    if calculation_status == "complete" and recalc_status in {"complete", "reused"}:
        valid = all(_finite_value(row.get(name)) for name in CALCULATION_SUCCESS_FIELDS)
        try:
            iterations = int(row.get("recalc_iv_iterations"))
            valid = valid and 0 <= iterations <= 2**16 - 1
        except (TypeError, ValueError, OverflowError):
            valid = False
        return (
            CalculationClassification.SUCCESS
            if valid else CalculationClassification.TERMINAL_FAILURE
        )
    if calculation_status == "complete" and recalc_status in {"skipped", "disabled"}:
        return CalculationClassification.SOURCE_ONLY
    if calculation_status in {"pending", "running"} or recalc_status in {"pending", "running"}:
        return CalculationClassification.PENDING
    return CalculationClassification.TERMINAL_FAILURE


def classify_surface_expiration(status: Any) -> SurfaceExpirationClassification:
    """Map the legacy fit status exactly as the surface runtime does."""
    text = str(status or "").lower()
    if text in {"complete", "complete_warning"}:
        return SurfaceExpirationClassification.SUCCESS
    if text.startswith("skipped"):
        return SurfaceExpirationClassification.SKIPPED
    return SurfaceExpirationClassification.FAILURE


def classify_legacy_run_status(
    status: Any,
    started_at: Any = None,
    *,
    stale_after_seconds: int = 3600,
    now: Optional[datetime] = None,
) -> str:
    """Preserve run state, including a distinct stale-running classification."""
    text = str(status or "").lower()
    if text == "running":
        started = _parse_timestamp(started_at)
        current = now or _utc_now()
        if started and (current - started).total_seconds() >= stale_after_seconds:
            return "stale_running"
        return "running"
    if text in {"pending", "queued"}:
        return "pending"
    if text in {"failed", "abandoned"}:
        return "abandoned" if text == "abandoned" else "failed"
    return text or "failed"


def reconcile_calculation_counts(
    successful: int, source_only: int, failed: int, expected: int
) -> bool:
    return int(successful) + int(source_only) + int(failed) == int(expected)


def reconcile_surface_counts(
    successful: int, skipped: int, failed: int, expected: int
) -> bool:
    return int(successful) + int(skipped) + int(failed) == int(expected)


MAX_FAILURE_TEXT = 512


def _bounded_text(value: Any, limit: int = MAX_FAILURE_TEXT) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def safe_source_value(value: Any, limit: int = 256) -> str:
    """Bound malformed source values before they enter operator records."""
    try:
        return _bounded_text(repr(value), limit)
    except Exception:
        return "<unrepresentable>"


@dataclass(frozen=True)
class ObjectPlan:
    canonical_name: str
    target_name: str
    legacy_name: str
    source_name: Optional[str]
    source_uuid: Optional[str]
    source_schema_hash: Optional[str]
    target_uuid: Optional[str]
    target_schema_hash: str
    source_create_sql: Optional[str]


@dataclass(frozen=True)
class MigrationCheckpoint:
    migration_id: str
    database: str
    schema_generation: int
    phase: MigrationPhase
    completed_steps: tuple[str, ...]
    objects: tuple[ObjectPlan, ...]
    validation_summary: Mapping[str, Any]
    archive_hashes: Mapping[str, str]
    updated_at: str
    is_cutover: bool = False
    is_rolled_back: bool = False
    planned_at: Optional[str] = None
    execution_started_at: Optional[str] = None
    validated_at: Optional[str] = None
    cutover_started_at: Optional[str] = None
    cutover_completed_at: Optional[str] = None
    rollback_started_at: Optional[str] = None
    rollback_completed_at: Optional[str] = None
    cleanup_started_at: Optional[str] = None
    cleanup_completed_at: Optional[str] = None

    def with_step(
        self,
        step: str,
        phase: MigrationPhase,
        *,
        validation_summary: Optional[Mapping[str, Any]] = None,
        objects: Optional[Sequence[ObjectPlan]] = None,
        is_cutover: Optional[bool] = None,
        is_rolled_back: Optional[bool] = None,
    ) -> "MigrationCheckpoint":
        completed = self.completed_steps
        if step not in completed:
            completed = (*completed, step)
        now = _utc_now().isoformat(timespec="microseconds")
        timestamp_updates: dict[str, Optional[str]] = {}
        if phase is MigrationPhase.PLANNED and self.planned_at is None:
            timestamp_updates["planned_at"] = now
        if phase not in {MigrationPhase.NEW, MigrationPhase.PLANNED} and self.execution_started_at is None:
            timestamp_updates["execution_started_at"] = now
        if phase is MigrationPhase.VALIDATED and self.validated_at is None:
            timestamp_updates["validated_at"] = now
        if phase in {MigrationPhase.PREPARING_CUTOVER, MigrationPhase.CUTTING_OVER} and self.cutover_started_at is None:
            timestamp_updates["cutover_started_at"] = now
        if phase is MigrationPhase.CUTOVER and self.cutover_completed_at is None:
            timestamp_updates["cutover_completed_at"] = now
        if phase is MigrationPhase.ROLLING_BACK and self.rollback_started_at is None:
            timestamp_updates["rollback_started_at"] = now
        if phase is MigrationPhase.ROLLED_BACK and self.rollback_completed_at is None:
            timestamp_updates["rollback_completed_at"] = now
        if phase is MigrationPhase.CLEANING and self.cleanup_started_at is None:
            timestamp_updates["cleanup_started_at"] = now
        if phase is MigrationPhase.CLEANED and self.cleanup_completed_at is None:
            timestamp_updates["cleanup_completed_at"] = now
        return replace(
            self,
            phase=phase,
            completed_steps=completed,
            validation_summary=(
                validation_summary
                if validation_summary is not None
                else self.validation_summary
            ),
            objects=tuple(objects) if objects is not None else self.objects,
            updated_at=now,
            is_cutover=self.is_cutover if is_cutover is None else is_cutover,
            is_rolled_back=(
                self.is_rolled_back if is_rolled_back is None else is_rolled_back
            ),
            **timestamp_updates,
        )


def _checkpoint_json(checkpoint: MigrationCheckpoint) -> dict[str, Any]:
    value = asdict(checkpoint)
    value["phase"] = checkpoint.phase.value
    return value


def _checkpoint_from_json(value: Mapping[str, Any]) -> MigrationCheckpoint:
    phase_text = str(value["phase"])
    phase_aliases = {
        "target_schema": MigrationPhase.TARGET_SCHEMA_CREATED.value,
        "sources": MigrationPhase.SOURCES_MIGRATED.value,
        "calculations": MigrationPhase.CALCULATIONS_MIGRATED.value,
        "surfaces": MigrationPhase.SURFACES_MIGRATED.value,
        "raw_archive": MigrationPhase.RAW_ARCHIVED.value,
    }
    phase_text = phase_aliases.get(phase_text, phase_text)
    return MigrationCheckpoint(
        migration_id=str(value["migration_id"]),
        database=str(value["database"]),
        schema_generation=int(value["schema_generation"]),
        phase=MigrationPhase(phase_text),
        completed_steps=tuple(value.get("completed_steps", ())),
        objects=tuple(ObjectPlan(**item) for item in value.get("objects", ())),
        validation_summary=dict(value.get("validation_summary", {})),
        archive_hashes=dict(value.get("archive_hashes", {})),
        updated_at=str(value["updated_at"]),
        is_cutover=bool(value.get("is_cutover", False)),
        is_rolled_back=bool(value.get("is_rolled_back", False)),
        planned_at=value.get("planned_at"),
        execution_started_at=value.get("execution_started_at"),
        validated_at=value.get("validated_at"),
        cutover_started_at=value.get("cutover_started_at"),
        cutover_completed_at=value.get("cutover_completed_at"),
        rollback_started_at=value.get("rollback_started_at"),
        rollback_completed_at=value.get("rollback_completed_at"),
        cleanup_started_at=value.get("cleanup_started_at"),
        cleanup_completed_at=value.get("cleanup_completed_at"),
    )


class CheckpointStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Optional[MigrationCheckpoint]:
        if not self.path.exists():
            return None
        return _checkpoint_from_json(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, checkpoint: MigrationCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        payload = (
            json.dumps(
                _checkpoint_json(checkpoint),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def short_migration_id(migration_id: uuid.UUID | str) -> str:
    return uuid.UUID(str(migration_id)).hex[:12]


def target_name(canonical_name: str, migration_id: uuid.UUID | str) -> str:
    return f"{canonical_name}__migrating_{short_migration_id(migration_id)}"


def legacy_name(canonical_name: str, migration_id: uuid.UUID | str) -> str:
    return f"{canonical_name}__legacy_{short_migration_id(migration_id)}"


def normalize_schema_sql(sql: str) -> str:
    """Canonicalize harmless authored-DDL versus ``SHOW CREATE`` differences.

    ClickHouse may render a one-expression tuple key without its redundant
    parentheses.  For example, ``ORDER BY (contract_id)`` is returned as
    ``ORDER BY contract_id`` on some server versions.  Those forms are
    semantically identical and must not prevent an idempotent migration resume.
    """
    normalized = sql.replace("`", "")
    normalized = re.sub(
        r"\bCREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\b",
        "CREATE TABLE",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\s+UUID\s+'[0-9a-fA-F-]+'", "", normalized, flags=re.IGNORECASE
    )
    normalized = " ".join(normalized.split())
    # ClickHouse renders ``date + INTERVAL n DAY DELETE`` as the equivalent
    # ``date + toIntervalDay(n)`` and omits the default DELETE action in SHOW
    # CREATE output on some server versions.
    normalized = re.sub(
        r"\+\s*INTERVAL\s+(\d+)\s+DAY\b",
        r"+ toIntervalDay(\1)",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(\btoIntervalDay\(\d+\))\s+DELETE(?=\s+SETTINGS\b)",
        r"\1",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b(ORDER\s+BY|PRIMARY\s+KEY|PARTITION\s+BY|SAMPLE\s+BY)"
        r"\s*\(\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\)",
        r"\1 \2",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s*([(),=])\s*", r"\1", normalized)
    return normalized


def schema_hash(sql: str) -> str:
    normalized = normalize_schema_sql(sql)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonicalize_created_table_name(
    sql: str,
    *,
    database: str,
    actual_name: str,
    canonical_name: str,
) -> str:
    """Replace only the table identifier following ``CREATE TABLE``.

    A global string replacement can accidentally alter a comment, expression,
    or literal containing the generated migration table name.
    """
    qualified_actual = (
        rf"(?:`?{re.escape(database)}`?\.)?`?{re.escape(actual_name)}`?"
    )
    pattern = re.compile(
        rf"(\bCREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+)"
        rf"{qualified_actual}",
        flags=re.IGNORECASE,
    )
    canonicalized, substitutions = pattern.subn(
        rf"\1{database}.{canonical_name}", sql, count=1
    )
    if substitutions != 1:
        raise MigrationError(
            f"Could not identify CREATE TABLE name {database}.{actual_name}"
        )
    return canonicalized


def schema_mismatch_details(expected_sql: str, actual_sql: str) -> str:
    """Return a bounded token diff for an actionable schema mismatch error."""
    expected = normalize_schema_sql(expected_sql)
    actual = normalize_schema_sql(actual_sql)
    lines = list(
        difflib.unified_diff(
            expected.split(),
            actual.split(),
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )
    if not lines:
        return "normalized schemas are identical"
    limit = 80
    rendered = "\n".join(lines[:limit])
    if len(lines) > limit:
        rendered += f"\n... {len(lines) - limit} additional diff line(s) omitted"
    return rendered


def parse_fit_parameters_json(value: str) -> dict[str, Any]:
    """Map a legacy fit payload to the typed canonical SVI/PCHIP shape."""
    payload = json.loads(value)
    method = str(payload.get("method") or payload.get("fit_method") or "").upper()
    parameters = payload.get("parameters", payload)
    if method == "SVI" or all(name in parameters for name in ("a", "b", "rho", "m", "sigma")):
        result = {
            "fit_method": "SVI",
            "svi_a": float(parameters["a"]),
            "svi_b": float(parameters["b"]),
            "svi_rho": float(parameters["rho"]),
            "svi_m": float(parameters["m"]),
            "svi_sigma": float(parameters["sigma"]),
            "pchip_k": [],
            "pchip_total_variance": [],
        }
    elif method == "PCHIP" or "k" in parameters:
        k = [float(item) for item in parameters.get("k", parameters.get("x", []))]
        variance = [
            float(item)
            for item in parameters.get(
                "total_variance", parameters.get("y", parameters.get("w", []))
            )
        ]
        if len(k) < 2 or len(k) != len(variance):
            raise ValueError(
                "PCHIP fit arrays require at least two equally sized points"
            )
        result = {
            "fit_method": "PCHIP",
            "svi_a": None,
            "svi_b": None,
            "svi_rho": None,
            "svi_m": None,
            "svi_sigma": None,
            "pchip_k": k,
            "pchip_total_variance": variance,
        }
    else:
        raise ValueError("Legacy fit payload is neither valid SVI nor PCHIP")
    numeric = [
        value
        for name, value in result.items()
        if name != "fit_method" and value is not None
    ]
    for value in numeric:
        values = value if isinstance(value, list) else [value]
        if any(not (float("-inf") < item < float("inf")) for item in values):
            raise ValueError("Typed fit parameters must be finite")
    if result["fit_method"] == "SVI":
        if (
            result["svi_b"] < 0
            or result["svi_sigma"] <= 0
            or not -1 < result["svi_rho"] < 1
        ):
            raise ValueError("SVI parameters violate b/rho/sigma constraints")
    else:
        if any(
            right <= left
            for left, right in zip(result["pchip_k"], result["pchip_k"][1:])
        ):
            raise ValueError("PCHIP knots must be strictly increasing")
        if any(value < 0 for value in result["pchip_total_variance"]):
            raise ValueError("PCHIP total variance cannot be negative")
    return result


def _iter_arrow_rows(
    client: Any,
    sql: str,
    *,
    max_block_size: int = 25_000,
) -> Iterable[dict[str, Any]]:
    """Yield query rows without materializing a migration-sized Arrow table."""
    stream_query = getattr(client, "query_arrow_stream", None)
    if stream_query is None:
        # Small test doubles and older clients can still exercise the mapper.
        yield from client.query_arrow(sql).to_pylist()
        return
    with stream_query(sql, settings={"max_block_size": max_block_size}) as blocks:
        for block in blocks:
            yield from block.to_pylist()


def _contract_id_sql(alias: str = "b") -> str:
    # reverse() makes reinterpretAsUInt64 consume the first SHA-256 bytes as
    # the same big-endian integer as Common.option_contract_id().
    return (
        "reinterpretAsUInt64(reverse(substring(SHA256(concat(upper(trimBoth("
        f"{alias}.underlying_symbol)), '\\x1f', {alias}.option_symbol)), 1, 8)))"
    )


def _legacy_success_predicate(alias: str = "c") -> str:
    """One migration-only definition of a persistable legacy success."""
    finite = " AND ".join(f"isFinite({alias}.{name})" for name in CALCULATION_SUCCESS_FIELDS)
    return (
        f"{alias}.calculation_status = 'complete' "
        f"AND {alias}.recalc_status IN ('complete', 'reused') "
        f"AND {finite} AND {alias}.recalc_iv_iterations >= 0 "
        f"AND {alias}.recalc_iv_iterations <= 65535"
    )


def _legacy_terminal_failure_predicate(alias: str = "c") -> str:
    return (
        f"({alias}.calculation_status = 'failed' OR {alias}.recalc_status = 'failed' "
        f"OR ({alias}.calculation_status = 'complete' "
        f"AND {alias}.recalc_status NOT IN ('pending', 'running') "
        f"AND NOT ({_legacy_source_only_predicate(alias)}) "
        f"AND NOT ({_legacy_success_predicate(alias)})))"
    )


def _legacy_source_only_predicate(alias: str = "c") -> str:
    return (
        f"{alias}.calculation_status = 'complete' "
        f"AND {alias}.recalc_status IN ('skipped', 'disabled')"
    )


class MigrationError(RuntimeError):
    pass


class MigrationBlockedError(MigrationError):
    """Raised when a safe migration cannot proceed yet."""


@dataclass
class MigrationReporter:
    """Small stdout-first reporter used by the CLI and embedders."""

    migration_id: str
    database: str
    checkpoint_path: Path
    json_output: bool = False
    started_at: float = field(default_factory=time.monotonic)

    def emit(self, event: str, **values: Any) -> None:
        payload = {"event": event, **values}
        if self.json_output:
            print(json.dumps(payload, default=str, sort_keys=True), flush=True)
        else:
            fields = " ".join(f"{key}={value}" for key, value in payload.items())
            print(fields, flush=True)

    def start(self, command: str, phase: MigrationPhase, dry_run: bool) -> None:
        self.emit(
            "migration_start", command=command, migration_id=self.migration_id,
            database=self.database, phase=phase.value, dry_run=bool(dry_run),
        )

    def progress(self, phase: str, **values: Any) -> None:
        self.emit("migration_progress", phase=phase, **values)

    def finish(
        self,
        result: str,
        phase_before: str,
        phase_after: str,
        *,
        rows_read: int = 0,
        rows_inserted: int = 0,
        rows_skipped: int = 0,
        rows_failed: int = 0,
        database_changes: bool = False,
        recommended_next_command: str = "--resume",
    ) -> None:
        self.emit(
            "migration_result", result=result, phase_before=phase_before,
            phase_after=phase_after, rows_read=rows_read,
            rows_inserted=rows_inserted, rows_skipped=rows_skipped,
            rows_failed=rows_failed,
            elapsed_seconds=round(time.monotonic() - self.started_at, 3),
            checkpoint_path=str(self.checkpoint_path),
            validation_report_path=str(
                self.checkpoint_path.parent
                / f"migration_validation_{self.migration_id}.json"
            ),
            database_changes="yes" if database_changes else "no",
            recommended_next_command=recommended_next_command,
        )

    def error(self, message: str) -> None:
        print(f"migration_error: {message}", file=sys.stderr, flush=True)

    def plan_report(self, report: Mapping[str, Any]) -> None:
        if self.json_output:
            self.emit("migration_plan", **dict(report))
            return
        print("migration_plan", flush=True)
        for key in (
            "migration_id", "database", "checkpoint_path", "archive_directory",
            "phase", "status", "objects", "calculations", "surfaces", "archive",
            "blockers",
        ):
            print(f"{key}: {report.get(key)}", flush=True)
        print("No database changes were made.", flush=True)


class MigrationCoordinator:
    def __init__(
        self,
        context: Mapping[str, Any],
        client: Any,
        migration_id: uuid.UUID,
        archive_directory: Path,
        *,
        logger: Optional[logging.Logger] = None,
        dry_run: bool = False,
        reporter: Optional[MigrationReporter] = None,
        allow_stale_lease_takeover: bool = False,
    ):
        self.context = context
        self.client = client
        self.migration_id = migration_id
        self.short_id = short_migration_id(migration_id)
        self.database = validate_database_identifier(
            str(context["options_chain.clickhouse.database"])
        )
        self.archive_directory = archive_directory
        self.logger = logger or logging.getLogger(__name__)
        self.dry_run = dry_run
        self.reporter = reporter
        self.allow_stale_lease_takeover = bool(allow_stale_lease_takeover)
        self.planned_sql: list[str] = []
        self._quarantine_rows = 0
        self._lease_acquired = False
        self._streaming_write_client: Any = None
        self._lease_owner = f"{os.uname().nodename}:{os.getpid()}"
        for key in (
            "options_chain.migration.attempt_retention_days",
            "options_chain.migration.diagnostics_retention_days",
            "options_chain.migration.validation_retention_days",
            "options_chain.migration.soak_days",
        ):
            if int(context.get(key, 30 if "retention" in key else 14)) < 0:
                raise MigrationError(f"{key} cannot be negative")
        repetitions = int(
            context.get("options_chain.gui_benchmark.repetitions", 3)
        )
        if not 2 <= repetitions <= 20:
            raise MigrationError(
                "options_chain.gui_benchmark.repetitions must be between 2 and 20"
            )
        for key, default in (
            ("options_chain.gui_benchmark.warm_target_seconds", 0.250),
            ("options_chain.gui_benchmark.cold_target_seconds", 1.0),
        ):
            if float(context.get(key, default)) <= 0:
                raise MigrationError(f"{key} must be positive")
        parity_sample_size = int(
            context.get("options_chain.migration.numerical_parity_sample_size", 512)
        )
        if not 1 <= parity_sample_size <= 10_000:
            raise MigrationError(
                "options_chain.migration.numerical_parity_sample_size must be "
                "between 1 and 10000"
            )
        self.store = CheckpointStore(
            archive_directory / f"options_chain_migration_{migration_id}.json"
        )
        self.checkpoint = self._load_or_create_checkpoint()

    def _load_or_create_checkpoint(self) -> MigrationCheckpoint:
        existing = self.store.load()
        if existing is not None:
            if existing.migration_id != str(self.migration_id):
                raise MigrationError("Checkpoint migration ID does not match")
            if existing.database != self.database:
                raise MigrationError("Checkpoint database does not match")
            if existing.schema_generation != SCHEMA_GENERATION:
                raise MigrationError("Checkpoint schema generation does not match")
            return existing
        checkpoint = MigrationCheckpoint(
            migration_id=str(self.migration_id),
            database=self.database,
            schema_generation=SCHEMA_GENERATION,
            phase=MigrationPhase.NEW,
            completed_steps=(),
            objects=(),
            validation_summary={},
            archive_hashes={},
            updated_at=_utc_now().isoformat(timespec="microseconds"),
        )
        if not self.dry_run:
            self.store.save(checkpoint)
        return checkpoint

    def _checkpoint(
        self,
        step: str,
        phase: MigrationPhase,
        **updates: Any,
    ) -> None:
        if self.dry_run:
            self.logger.info(
                "migration_checkpoint_planned migration_id=%s phase=%s step=%s",
                self.migration_id, phase.value, step,
            )
            return
        self.checkpoint = self.checkpoint.with_step(step, phase, **updates)
        self.logger.info(
            "migration_phase migration_id=%s database=%s phase=%s step=%s "
            "dry_run=%s",
            self.migration_id,
            self.database,
            phase.value,
            step,
            self.dry_run,
        )
        if not self.dry_run:
            self.store.save(self.checkpoint)
            self._save_database_checkpoint()
            self.heartbeat_lease()
        if self.reporter is not None:
            self.reporter.progress(
                phase=phase.value, step=step, dry_run=self.dry_run
            )

    def mark_failed(self, error: BaseException) -> None:
        """Persist an actionable failure barrier without hiding resume intent."""
        if self.dry_run or not self._lease_acquired:
            return
        failed_from = self.checkpoint.phase.value
        try:
            self._checkpoint(
                "failed",
                MigrationPhase.FAILED,
                validation_summary={
                    **self.checkpoint.validation_summary,
                    "failed_from_phase": failed_from,
                    "last_error": _bounded_text(error),
                    "failed_at": _utc_now().isoformat(timespec="microseconds"),
                },
            )
        except Exception:
            self.logger.exception(
                "migration_failure_checkpoint_failed migration_id=%s phase=%s",
                self.migration_id,
                failed_from,
            )

    def _save_database_checkpoint(self) -> None:
        canonical = "options_chain_dataset_metadata"
        target = target_name(canonical, self.migration_id)
        failed = target + "__failed"
        # The local file remains authoritative before the target schema exists.
        target_plan = next(
            (
                item
                for item in self.checkpoint.objects
                if item.canonical_name == canonical
            ),
            None,
        )
        candidates = self.client.query(
            "SELECT name, toString(uuid) FROM system.tables "
            "WHERE database = {database:String} AND name IN {names:Array(String)}",
            parameters={
                "database": self.database,
                "names": [target, canonical, failed],
            },
        ).result_rows
        physical = next(
            (
                str(name)
                for name, object_uuid in candidates
                if target_plan is not None
                and target_plan.target_uuid is not None
                and str(object_uuid) == target_plan.target_uuid
            ),
            target if any(str(name) == target for name, _ in candidates) else None,
        )
        if physical is None:
            return
        now = _utc_now()
        generation_id = uuid.uuid5(self.migration_id, "dataset-generation")
        definition_ids = self.checkpoint.validation_summary.get(
            "definition_ids", {}
        )
        source_range = self.checkpoint.validation_summary.get("source_range", {})
        source_start = _parse_timestamp(source_range.get("start")) or datetime(
            1970, 1, 1, tzinfo=UTC
        )
        source_end = _parse_timestamp(source_range.get("end")) or source_start
        active = self.checkpoint.is_cutover and not self.checkpoint.is_rolled_back
        existing_metadata = self.client.query(
            f"SELECT created_at, activated_at, deactivated_at "
            f"FROM {self.database}.{physical} FINAL "
            "WHERE dataset_generation_id = {generation_id:UUID} LIMIT 1",
            parameters={"generation_id": str(generation_id)},
        ).result_rows
        prior = existing_metadata[0] if existing_metadata else (None, None, None)
        created_at = _parse_timestamp(prior[0]) or now
        activated_at = _parse_timestamp(prior[1])
        deactivated_at = _parse_timestamp(prior[2])
        if active and activated_at is None:
            activated_at = now
        if self.checkpoint.is_rolled_back and deactivated_at is None:
            deactivated_at = now
        self.client.insert(
            f"{self.database}.{physical}",
            [[
                str(generation_id), SCHEMA_GENERATION, str(self.migration_id), created_at,
                activated_at, deactivated_at,
                int(definition_ids.get("collection", 0)),
                int(definition_ids.get("normalization", 0)),
                int(definition_ids.get("calculation", 0)),
                int(definition_ids.get("surface", 0)),
                source_start, source_end,
                _compact_json(self.checkpoint.validation_summary),
                self.checkpoint.phase.value,
                _compact_json(_checkpoint_json(self.checkpoint)), now, active,
            ]],
            column_names=[
                "dataset_generation_id", "schema_generation", "migration_id",
                "created_at", "activated_at", "deactivated_at",
                "default_collection_definition_id",
                "default_normalization_definition_id",
                "default_calculation_definition_id", "default_surface_definition_id",
                "source_start_at", "source_end_at", "validation_summary_json",
                "migration_phase", "migration_checkpoint_json", "updated_at", "is_active",
            ],
        )

    def _command(self, sql: str, **kwargs: Any) -> None:
        compact = " ".join(sql.split())
        self.planned_sql.append(compact)
        self.logger.info("migration_sql %s", compact)
        if self.dry_run:
            # Rendering alone misses server-version-specific analyzer errors.
            # EXPLAIN fully parses and type-checks INSERT SELECT statements
            # without executing the insert or mutating ClickHouse.
            if re.match(r"^INSERT\s+INTO\b", compact, flags=re.IGNORECASE) and re.search(
                r"\bSELECT\b", compact, flags=re.IGNORECASE
            ):
                self.client.command(f"EXPLAIN SYNTAX {sql}", **kwargs)
            return
        self.client.command(sql, **kwargs)

    def _insert_arrow_while_streaming(self, table: str, arrow_table: Any) -> None:
        """Insert through a session distinct from an active Arrow read stream."""
        if self._streaming_write_client is None:
            self._streaming_write_client = create_clickhouse_client(self.context)
        self._streaming_write_client.insert_arrow(table, arrow_table)

    def close(self) -> None:
        if self._streaming_write_client is not None:
            self._streaming_write_client.close()
            self._streaming_write_client = None

    def _archive_parent(self) -> Path:
        """Return the nearest existing parent for free-space checks."""
        candidate = self.archive_directory
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate

    def archive_space_report(self) -> Mapping[str, Any]:
        parent = self._archive_parent()
        exists = self.archive_directory.exists()
        is_directory = self.archive_directory.is_dir()
        try:
            usage = shutil.disk_usage(parent)
            writable = bool(
                os.access(self.archive_directory, os.W_OK)
                if exists and is_directory
                else os.access(parent, os.W_OK)
            )
        except OSError as exc:
            return {
                "path": str(self.archive_directory), "nearest_existing_parent": str(parent),
                "free_bytes": 0, "writable": False, "creatable": False,
                "exists": exists, "is_directory": is_directory,
                "error": str(exc),
            }
        creatable = bool((exists and is_directory) or (not exists and writable))
        return {
            "path": str(self.archive_directory),
            "nearest_existing_parent": str(parent),
            "free_bytes": int(usage.free),
            "writable": bool(writable),
            "creatable": bool(creatable),
            "exists": exists,
            "is_directory": is_directory,
        }

    def _ensure_lease(self, phase: Optional[MigrationPhase] = None) -> None:
        """Acquire the migration-wide writer barrier before mutable work."""
        if self.dry_run or self._lease_acquired:
            return
        for statement in migration_support_schema_statements(self.database):
            self.client.command(statement)
        stale_seconds = int(
            self.context.get("options_chain.migration.lease_stale_seconds", 3600)
        )
        now = _utc_now()
        rows = self.client.query(
            f"SELECT toString(migration_id), owner, heartbeat_at, expires_at "
            f"FROM {self.database}.options_chain_migration_leases FINAL "
            "WHERE status = 'active' ORDER BY updated_at DESC LIMIT 1 BY migration_id"
        ).result_rows
        for other_id, owner, heartbeat_at, expires_at in rows:
            other_id = str(other_id)
            if other_id == str(self.migration_id):
                continue
            expiry = _parse_timestamp(expires_at)
            heartbeat = _parse_timestamp(heartbeat_at)
            stale = (
                expiry is None or expiry <= now or heartbeat is None
                or (now - heartbeat).total_seconds() > stale_seconds
            )
            if not stale or not self.allow_stale_lease_takeover:
                message = (
                    f"active migration lease held by migration_id={other_id} owner={owner}"
                )
                if stale:
                    message += "; lease is stale, rerun with --takeover-stale-lease"
                raise MigrationBlockedError(message)
            self.client.command(
                f"INSERT INTO {self.database}.options_chain_migration_leases "
                "(migration_id, owner, phase, status, created_at, heartbeat_at, "
                "expires_at, takeover_of, audit_message, updated_at) VALUES "
                "({migration_id:UUID}, {owner:String}, 'takeover', 'taken_over', "
                "{now:DateTime64(6, 'UTC')}, {now:DateTime64(6, 'UTC')}, "
                "{now:DateTime64(6, 'UTC')}, {other_id:UUID}, {message:String}, "
                "{now:DateTime64(6, 'UTC')})",
                parameters={
                    "migration_id": str(self.migration_id), "owner": self._lease_owner,
                    "now": now, "other_id": other_id,
                    "message": f"stale lease takeover by {self.migration_id}",
                },
            )
        expires = now + timedelta(seconds=max(stale_seconds, 60))
        current_phase = phase or self.checkpoint.phase
        self.client.command(
            f"INSERT INTO {self.database}.options_chain_migration_leases "
            "(migration_id, owner, phase, status, created_at, heartbeat_at, "
            "expires_at, takeover_of, audit_message, updated_at) VALUES "
            "({migration_id:UUID}, {owner:String}, {phase:String}, 'active', "
            "{now:DateTime64(6, 'UTC')}, {now:DateTime64(6, 'UTC')}, "
            "{expires:DateTime64(6, 'UTC')}, NULL, {message:String}, "
            "{now:DateTime64(6, 'UTC')})",
            parameters={
                "migration_id": str(self.migration_id), "owner": self._lease_owner,
                "phase": current_phase.value, "now": now, "expires": expires,
                "message": "migration lease acquired",
            },
        )
        self._lease_acquired = True

    def heartbeat_lease(self) -> None:
        if self.dry_run or not self._lease_acquired:
            return
        stale_seconds = int(
            self.context.get("options_chain.migration.lease_stale_seconds", 3600)
        )
        now = _utc_now()
        self.client.command(
            f"INSERT INTO {self.database}.options_chain_migration_leases "
            "(migration_id, owner, phase, status, created_at, heartbeat_at, "
            "expires_at, takeover_of, audit_message, updated_at) VALUES "
            "({migration_id:UUID}, {owner:String}, {phase:String}, 'active', "
            "{now:DateTime64(6, 'UTC')}, {now:DateTime64(6, 'UTC')}, "
            "{expires:DateTime64(6, 'UTC')}, NULL, 'heartbeat', "
            "{now:DateTime64(6, 'UTC')})",
            parameters={
                "migration_id": str(self.migration_id), "owner": self._lease_owner,
                "phase": self.checkpoint.phase.value, "now": now,
                "expires": now + timedelta(seconds=max(stale_seconds, 60)),
            },
        )

    def release_lease(self) -> None:
        if self.dry_run or not self._lease_acquired:
            return
        now = _utc_now()
        self.client.command(
            f"INSERT INTO {self.database}.options_chain_migration_leases "
            "(migration_id, owner, phase, status, created_at, heartbeat_at, "
            "expires_at, takeover_of, audit_message, updated_at) VALUES "
            "({migration_id:UUID}, {owner:String}, {phase:String}, 'released', "
            "{now:DateTime64(6, 'UTC')}, {now:DateTime64(6, 'UTC')}, "
            "{now:DateTime64(6, 'UTC')}, NULL, 'migration complete', "
            "{now:DateTime64(6, 'UTC')})",
            parameters={
                "migration_id": str(self.migration_id), "owner": self._lease_owner,
                "phase": self.checkpoint.phase.value, "now": now,
            },
        )
        self._lease_acquired = False

    def _current_attempt_relation(
        self,
        table: str,
        *,
        id_column: str,
        latest_view: Optional[str] = None,
    ) -> str:
        """Return a relation containing one current logical row per attempt.

        Legacy run tables are append-only status histories.  Reading them with a
        plain ``WHERE status = 'running'`` counts every historical running row,
        including attempts that subsequently completed or failed.  Prefer the
        legacy latest-state view while it exists; after views have been removed
        or tables have been renamed for cutover/rollback, reproduce the same
        current-row behavior directly from the physical table.
        """
        if latest_view is not None and self._table_exists(latest_view):
            return f"{self.database}.{latest_view}"
        return (
            "(SELECT * FROM "
            f"{self.database}.{table} "
            f"ORDER BY {id_column}, "
            "isNotNull(finished_at) DESC, "
            "ifNull(finished_at, started_at) DESC, "
            "(toString(status) != 'running') DESC, "
            "toString(status) DESC "
            f"LIMIT 1 BY {id_column})"
        )

    def offline_preflight(
        self, *, strict: bool = True, persist: bool = True
    ) -> Mapping[str, Any]:
        counts: dict[str, Any] = {}
        try:
            result = self.client.query(
                "SELECT count() FROM system.processes WHERE query_id != currentQueryID() "
                "AND database = {database:String}",
                parameters={"database": self.database},
            )
            counts["running_queries"] = int(result.result_rows[0][0])
        except Exception as exc:
            raise MigrationError(
                f"Offline preflight could not prove running queries: {exc}"
            ) from exc
        attempt_tables = (
            (
                "options_chain_calculation_runs",
                "run_id",
                "options_chain_calculation_runs_latest",
            ),
            (
                "options_chain_surface_runs",
                "surface_run_id",
                "options_chain_surface_runs_latest",
            ),
            ("options_chain_processing_attempts", "attempt_id", None),
            (
                target_name("options_chain_processing_attempts", self.migration_id),
                "attempt_id",
                None,
            ),
            (
                legacy_name("options_chain_calculation_runs", self.migration_id),
                "run_id",
                None,
            ),
            (
                legacy_name("options_chain_surface_runs", self.migration_id),
                "surface_run_id",
                None,
            ),
        )
        for table, id_column, latest_view in attempt_tables:
            if not self._table_exists(table):
                continue
            try:
                relation = self._current_attempt_relation(
                    table,
                    id_column=id_column,
                    latest_view=latest_view,
                )
                stale_seconds = int(
                    self.context.get(
                        "options_chain.migration.surface_stale_seconds"
                        if "surface" in table
                        else "options_chain.migration.calculation_stale_seconds",
                        3600,
                    )
                )
                result = self.client.query(
                    f"SELECT "
                    "countIf(toString(status) = 'running' AND isNull(finished_at) "
                    "  AND (isNull(started_at) OR "
                    f"dateDiff('second', started_at, now64(6)) < {stale_seconds})), "
                    "countIf(toString(status) = 'running' AND isNull(finished_at) "
                    "  AND isNotNull(started_at) "
                    f"  AND dateDiff('second', started_at, now64(6)) >= {stale_seconds}), "
                    "countIf(toString(status) IN ('stale_running', 'abandoned')) "
                    f"FROM {relation}"
                )
                live, stale, abandoned = result.result_rows[0]
                counts[f"running_attempts:{table}"] = int(live or 0)
                counts[f"stale_attempts:{table}"] = int(stale or 0)
                counts[f"abandoned_attempts:{table}"] = int(abandoned or 0)
            except Exception as exc:
                raise MigrationError(
                    f"Offline preflight could not prove attempts in {table}: {exc}"
                ) from exc
        blockers = {key: value for key, value in counts.items() if value}
        if blockers:
            if strict:
                raise MigrationBlockedError(f"Offline preflight blocked: {blockers}")
            self.logger.warning(
                "migration_preflight migration_id=%s database=%s "
                "status=report_only blockers=%s",
                self.migration_id,
                self.database,
                blockers,
            )
        space = self.archive_space_report()
        counts.update(space)
        # Keep the original preflight key as a compatibility alias for
        # embedders that used the first implementation's report shape.
        counts["archive_free_bytes"] = int(space.get("free_bytes", 0))
        if int(space.get("free_bytes", 0)) <= 0:
            raise MigrationBlockedError("Archive filesystem has no free space")
        if strict and not space.get("creatable"):
            raise MigrationBlockedError(
                f"Archive directory is not writable or creatable: {self.archive_directory}"
            )
        if persist:
            self._checkpoint("offline_preflight", MigrationPhase.PREFLIGHT_COMPLETE)
        return counts

    def plan(self) -> tuple[ObjectPlan, ...]:
        # Object ownership is immutable once a plan has been committed.  A
        # later --plan is an inspection operation, never a way to replace the
        # source UUIDs used by an executing migration.
        if self.checkpoint.objects and self.checkpoint.phase is not MigrationPhase.NEW:
            return self.checkpoint.objects
        rows = self.client.query(
            "SELECT name, toString(uuid), create_table_query FROM system.tables "
            "WHERE database = {database:String}",
            parameters={"database": self.database},
        ).result_rows
        existing = {
            str(name): (str(object_uuid), str(create_sql))
            for name, object_uuid, create_sql in rows
        }
        plans = []
        for canonical in CANONICAL_TABLE_NAMES:
            source = existing.get(canonical)
            migrating_name = target_name(canonical, self.migration_id)
            target = existing.get(migrating_name)
            target_uuid: Optional[str] = None
            target_schema_hash = schema_hash(
                CANONICAL_SCHEMA[canonical].ddl.format(database=self.database)
            )
            if target is not None:
                normalized_target = canonicalize_created_table_name(
                    target[1],
                    database=self.database,
                    actual_name=migrating_name,
                    canonical_name=canonical,
                )
                normalized_target_hash = schema_hash(normalized_target)
                if normalized_target_hash != target_schema_hash:
                    raise MigrationError(
                        f"Existing target schema mismatch for {migrating_name}:\n"
                        + schema_mismatch_details(
                            CANONICAL_SCHEMA[canonical].ddl.format(
                                database=self.database
                            ),
                            normalized_target,
                        )
                    )
                target_uuid = target[0]
                target_schema_hash = normalized_target_hash
            plans.append(
                ObjectPlan(
                    canonical_name=canonical,
                    target_name=migrating_name,
                    legacy_name=legacy_name(canonical, self.migration_id),
                    source_name=canonical if source else None,
                    source_uuid=source[0] if source else None,
                    source_schema_hash=schema_hash(source[1]) if source else None,
                    target_uuid=target_uuid,
                    target_schema_hash=target_schema_hash,
                    source_create_sql=source[1] if source else None,
                )
            )
        legacy_view_ddl = {
            name: existing[name][1]
            for name in LEGACY_VIEWS
            if name in existing
        }
        legacy_only_objects = {
            name: {
                "uuid": existing[name][0],
                "create_sql": existing[name][1],
                "rollback_name": legacy_name(name, self.migration_id),
            }
            for name in LEGACY_TABLES
            if name in existing and name not in CANONICAL_TABLE_NAMES
        }
        planned_validation_summary = {
            **self.checkpoint.validation_summary,
            "legacy_view_ddl": legacy_view_ddl,
            "legacy_only_objects": legacy_only_objects,
        }
        self._checkpoint(
            "plan",
            MigrationPhase.PLANNED,
            objects=plans,
            validation_summary=planned_validation_summary,
        )
        if self.dry_run:
            # Dry runs must not persist or advance checkpoints, but the report
            # still needs the just-computed inventory in memory.
            self.checkpoint = replace(
                self.checkpoint,
                objects=tuple(plans),
                validation_summary=planned_validation_summary,
            )
        return tuple(plans)

    def plan_report(self) -> Mapping[str, Any]:
        """Return a bounded operator-facing inventory for --plan."""
        objects = self.checkpoint.objects
        def count(
            sql: str, *, parameters: Optional[Mapping[str, Any]] = None
        ) -> int:
            try:
                return self._scalar(sql, parameters=parameters)
            except Exception:
                return -1
        calculation_run_relation = (
            f"{self.database}.options_chain_calculation_runs_latest"
        )
        surface_run_relation = f"{self.database}.options_chain_surface_runs_latest"
        calculation_stale_seconds = int(
            self.context.get("options_chain.migration.calculation_stale_seconds", 3600)
        )
        surface_stale_seconds = int(
            self.context.get("options_chain.migration.surface_stale_seconds", 3600)
        )
        raw_weeks: list[date] = []
        try:
            raw_week_rows = self.client.query(
                f"SELECT DISTINCT toMonday(toDate(collected_at_utc)) "
                f"FROM {self.database}.options_chain_raw ORDER BY 1"
            ).result_rows
            minimum_age_days = int(
                self.context.get("options_chain.archive.minimum_age_days", 14)
            )
            retain_raw_weeks = int(
                self.context.get("options_chain.archive.retain_raw_weeks", 2)
            )
            for (week,) in raw_week_rows:
                parsed_week = _parse_timestamp(week)
                if parsed_week is not None and week_is_age_eligible(
                    parsed_week.date(), _utc_now(),
                    minimum_age_days=minimum_age_days,
                    retain_raw_weeks=retain_raw_weeks,
                ):
                    raw_weeks.append(parsed_week.date())
        except Exception:
            raw_weeks = []
        if raw_weeks:
            week_sql = ", ".join(
                f"toDate('{week.isoformat()}')" for week in raw_weeks
            )
            raw_scope = f"toMonday(toDate(collected_at_utc)) IN ({week_sql})"
            raw_rows_in_scope = count(
                f"SELECT count() FROM {self.database}.options_chain_raw "
                f"WHERE {raw_scope}"
            )
            estimated_archive_bytes = count(
                f"SELECT coalesce(sum(length(response_json)), 0) "
                f"FROM {self.database}.options_chain_raw WHERE {raw_scope}"
            )
        else:
            raw_rows_in_scope = 0
            estimated_archive_bytes = 0
        safety_margin = max(
            1.0,
            float(
                self.context.get("options_chain.migration.archive_safety_margin", 1.25)
            ),
        )
        archive_space = dict(self.archive_space_report())
        estimated_target_bytes = count(
            "SELECT coalesce(sum(total_bytes), 0) FROM system.tables "
            "WHERE database = {database:String} AND name IN {names:Array(String)}",
            parameters={
                "database": self.database,
                "names": [item.target_name for item in objects],
            },
        )
        archive_report = {
            **archive_space,
            "raw_rows_in_scope": raw_rows_in_scope,
            "raw_weeks_in_scope": len(raw_weeks),
            "estimated_bytes_to_archive": estimated_archive_bytes,
            "archive_safety_margin": safety_margin,
            "estimated_required_archive_bytes": int(
                max(0, estimated_archive_bytes) * safety_margin
            ),
            "estimated_target_clickhouse_bytes": max(0, estimated_target_bytes),
            "space_sufficient": (
                estimated_archive_bytes < 0
                or archive_space.get("free_bytes", 0)
                >= max(0, estimated_archive_bytes) * safety_margin
            ),
        }
        unexpected_uuid_ownership: list[Mapping[str, Any]] = []
        for item in objects:
            try:
                actual_target = self._object_uuid_or_none(item.target_name)
                if actual_target is not None and actual_target != item.target_uuid:
                    unexpected_uuid_ownership.append(
                        {
                            "object": item.target_name,
                            "expected_uuid": item.target_uuid,
                            "actual_uuid": actual_target,
                        }
                    )
                actual_source = self._object_uuid_or_none(item.canonical_name)
                if (
                    item.source_uuid is not None
                    and actual_source is not None
                    and actual_source != item.source_uuid
                ):
                    unexpected_uuid_ownership.append(
                        {
                            "object": item.canonical_name,
                            "expected_uuid": item.source_uuid,
                            "actual_uuid": actual_source,
                        }
                    )
            except Exception:
                unexpected_uuid_ownership.append(
                    {"object": item.canonical_name, "error": "ownership unreadable"}
                )
        current_names = {item.canonical_name for item in objects if item.source_uuid}
        report: dict[str, Any] = {
            "migration_id": str(self.migration_id),
            "database": self.database,
            "checkpoint_path": str(self.store.path),
            "archive_directory": str(self.archive_directory),
            "phase": self.checkpoint.phase.value,
            "status": (
                "ALREADY COMPLETE" if self.checkpoint.phase in {
                    MigrationPhase.CUTOVER, MigrationPhase.CLEANED,
                } else "READY"
            ),
            "objects": {
                "canonical_objects_found": len(current_names),
                "target_objects_to_create": sum(item.target_uuid is None for item in objects),
                "legacy_objects_to_preserve": len(
                    self.checkpoint.validation_summary.get("legacy_only_objects", {})
                ) + sum(item.source_uuid is not None for item in objects),
                "views_to_replace": len(
                    self.checkpoint.validation_summary.get("legacy_view_ddl", {})
                ),
                "unexpected_uuid_ownership": unexpected_uuid_ownership,
            },
            "calculations": {
                "complete_snapshots": count(
                    f"SELECT count() FROM {calculation_run_relation} "
                    "WHERE status = 'complete'"
                ),
                "complete_with_failures_snapshots": count(
                    f"SELECT count() FROM {calculation_run_relation} "
                    "WHERE status = 'complete_with_failures'"
                ),
                "failed_snapshots": count(
                    f"SELECT count() FROM {calculation_run_relation} WHERE status = 'failed'"
                ),
                "pending_snapshots": count(
                    f"SELECT count() FROM {calculation_run_relation} WHERE status = 'pending'"
                ),
                "running_snapshots": count(
                    f"SELECT count() FROM {calculation_run_relation} WHERE status = 'running'"
                ),
                "stale_running_snapshots": count(
                    f"SELECT count() FROM {calculation_run_relation} "
                    "WHERE status = 'running' AND dateDiff('second', started_at, now64(6)) "
                    f">= {calculation_stale_seconds}"
                ),
                "source_rows": count(
                    f"SELECT count() FROM {self.database}.options_chain_calculations_latest"
                ),
                "successful_contract_rows": count(
                    f"SELECT count() FROM {self.database}.options_chain_calculations_latest AS c "
                    f"WHERE {_legacy_success_predicate('c')}"
                ),
                "source_only_rows": count(
                    f"SELECT count() FROM {self.database}.options_chain_calculations_latest AS c "
                    f"WHERE {_legacy_source_only_predicate('c')}"
                ),
                "failed_contract_rows": count(
                    f"SELECT count() FROM {self.database}.options_chain_calculations_latest AS c "
                    f"WHERE {_legacy_terminal_failure_predicate('c')}"
                ),
            },
            "surfaces": {
                "complete_runs": count(
                    f"SELECT count() FROM {surface_run_relation} WHERE status = 'complete'"
                ),
                "complete_with_failures_runs": count(
                    f"SELECT count() FROM {surface_run_relation} "
                    "WHERE status = 'complete_with_failures'"
                ),
                "failed_runs": count(
                    f"SELECT count() FROM {surface_run_relation} WHERE status = 'failed'"
                ),
                "pending_runs": count(
                    f"SELECT count() FROM {surface_run_relation} WHERE status = 'pending'"
                ),
                "running_runs": count(
                    f"SELECT count() FROM {surface_run_relation} WHERE status = 'running'"
                ),
                "stale_running_runs": count(
                    f"SELECT count() FROM {surface_run_relation} "
                    "WHERE status = 'running' AND dateDiff('second', started_at, now64(6)) "
                    f">= {surface_stale_seconds}"
                ),
                "successful_expirations": count(
                    f"SELECT count() FROM {self.database}.options_chain_surface_fits_latest "
                    "WHERE fit_status IN ('complete', 'complete_warning')"
                ),
                "skipped_expirations": count(
                    f"SELECT count() FROM {self.database}.options_chain_surface_fits_latest "
                    "WHERE startsWith(toString(fit_status), 'skipped')"
                ),
                "failed_expirations": count(
                    f"SELECT count() FROM {self.database}.options_chain_surface_fits_latest "
                    "WHERE NOT startsWith(toString(fit_status), 'skipped') "
                    "AND fit_status NOT IN ('complete', 'complete_warning')"
                ),
                "fit_rows": count(
                    f"SELECT count() FROM {self.database}.options_chain_surface_fits_latest"
                ),
                "node_rows": count(
                    f"SELECT count() FROM {self.database}.options_chain_surface_nodes_latest"
                ),
            },
            "archive": archive_report,
            "blockers": {},
            "database_changes_made": False,
        }
        blocker_counts = self.offline_preflight(strict=False, persist=False)
        report["blockers"] = {
            key: value for key, value in blocker_counts.items()
            if value and (
                key == "running_queries"
                or "running_attempts:" in key
                or "stale_attempts:" in key
            )
        }
        if not archive_report["space_sufficient"]:
            report["blockers"]["insufficient_archive_space"] = {
                "free_bytes": archive_space.get("free_bytes", 0),
                "estimated_required_bytes": archive_report[
                    "estimated_required_archive_bytes"
                ],
            }
        if not archive_space.get("creatable", False):
            report["blockers"]["archive_directory_not_creatable"] = str(
                self.archive_directory
            )
        if unexpected_uuid_ownership:
            report["blockers"]["unexpected_uuid_ownership"] = unexpected_uuid_ownership
        if report["blockers"]:
            report["status"] = "BLOCKED"
        return report

    def create_target_schema(self) -> None:
        self._ensure_lease(MigrationPhase.TARGET_SCHEMA_CREATED)
        if not self.checkpoint.objects:
            self.plan()
        plans = list(self.checkpoint.objects)
        for index, item in enumerate(plans):
            exists = self.client.query(
                "SELECT toString(uuid), create_table_query FROM system.tables "
                "WHERE database = {database:String} AND name = {name:String}",
                parameters={"database": self.database, "name": item.target_name},
            ).result_rows
            canonical_ddl = CANONICAL_SCHEMA[item.canonical_name].ddl.format(
                database=self.database
            )
            expected_target_schema_hash = schema_hash(canonical_ddl)
            ddl = canonical_ddl.replace(
                f".{item.canonical_name}\n", f".{item.target_name}\n", 1
            )
            if exists:
                # SHOW CREATE formatting includes the target name, so compare a
                # canonical-name-substituted form after normalizing harmless
                # ClickHouse rendering differences.
                normalized = canonicalize_created_table_name(
                    str(exists[0][1]),
                    database=self.database,
                    actual_name=item.target_name,
                    canonical_name=item.canonical_name,
                )
                normalized_hash = schema_hash(normalized)
                if normalized_hash != expected_target_schema_hash:
                    raise MigrationError(
                        f"Existing target schema mismatch for {item.target_name}:\n"
                        + schema_mismatch_details(
                            canonical_ddl,
                            normalized,
                        )
                    )
                plans[index] = replace(
                    item,
                    target_uuid=str(exists[0][0]),
                    target_schema_hash=normalized_hash,
                )
                continue
            self._command(ddl)
            if not self.dry_run:
                target_uuid, create_sql = self.client.query(
                    "SELECT toString(uuid), create_table_query FROM system.tables "
                    "WHERE database = {database:String} AND name = {name:String}",
                    parameters={"database": self.database, "name": item.target_name},
                ).result_rows[0]
                normalized = canonicalize_created_table_name(
                    str(create_sql),
                    database=self.database,
                    actual_name=item.target_name,
                    canonical_name=item.canonical_name,
                )
                normalized_hash = schema_hash(normalized)
                if normalized_hash != expected_target_schema_hash:
                    raise MigrationError(
                        f"Created target schema mismatch for {item.target_name}:\n"
                        + schema_mismatch_details(
                            canonical_ddl,
                            normalized,
                        )
                    )
                plans[index] = replace(
                    item,
                    target_uuid=str(target_uuid),
                    target_schema_hash=normalized_hash,
                )
            self._checkpoint(
                f"create:{item.canonical_name}",
                MigrationPhase.TARGET_SCHEMA,
                objects=plans,
            )
        self._checkpoint("target_schema", MigrationPhase.TARGET_SCHEMA, objects=plans)

    def validate_source_column_disposition(self) -> None:
        columns = {
            str(row[0])
            for row in self.client.query(
                "SELECT name FROM system.columns WHERE database = {database:String} "
                "AND table = 'options_chain_base'",
                parameters={"database": self.database},
            ).result_rows
        }
        missing = sorted(columns - set(SOURCE_COLUMN_DISPOSITION) - {"snapshot_id"})
        if missing:
            raise MigrationError(
                "Legacy source columns have no explicit disposition: " + ", ".join(missing)
            )

    def _insert_select(self, canonical: str, columns: str, select_sql: str) -> None:
        target = target_name(canonical, self.migration_id)
        self.logger.info(
            "migration_partition migration_id=%s database=%s table=%s "
            "partition=server_side_all",
            self.migration_id,
            self.database,
            canonical,
        )
        self._command(
            f"INSERT INTO {self.database}.{target} ({columns}) {select_sql}"
        )

    def _table_exists(self, name: str) -> bool:
        rows = self.client.query(
            "SELECT count() FROM system.tables WHERE database = {database:String} "
            "AND name = {name:String}",
            parameters={"database": self.database, "name": name},
        ).result_rows
        return bool(rows and int(rows[0][0]))

    def _quarantine_row(
        self,
        source_table: str,
        row: Mapping[str, Any],
        field_name: str,
        conversion_error: str,
    ) -> None:
        """Persist a bounded corruption record without re-parsing the bad row."""
        identity = safe_source_value(
            row.get("snapshot_id") or row.get("surface_run_id")
            or row.get("run_id") or row.get("option_symbol") or "unknown"
        )
        values = {
            "quarantine_id": str(uuid.uuid4()),
            "migration_id": str(self.migration_id),
            "source_table": source_table,
            "source_row_identity": identity,
            "field_name": field_name,
            "raw_value": safe_source_value(row.get(field_name)),
            "conversion_error": _bounded_text(conversion_error),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        self._quarantine_rows += 1
        if not self.dry_run:
            self.client.insert(
                f"{self.database}.options_chain_migration_quarantine",
                [[
                    values[name] for name in (
                        "quarantine_id", "migration_id", "source_table",
                        "source_row_identity", "field_name", "raw_value",
                        "conversion_error", "created_at", "updated_at",
                    )
                ]],
                column_names=list(values),
            )
        self.logger.warning(
            "migration_quarantine migration_id=%s source_table=%s field=%s error=%s",
            self.migration_id, source_table, field_name, values["conversion_error"],
        )
        if self.reporter is not None:
            self.reporter.progress(
                phase=self.checkpoint.phase.value,
                rows_quarantined=self._quarantine_rows,
            )
        if self.dry_run:
            raise MigrationError(
                "Dry-run encountered a quarantine row: "
                f"source_table={source_table} field={field_name} "
                f"error={values['conversion_error']}"
            )

    def _migrate_recent_attempts(
        self, definitions: Mapping[str, int], *, include_surface: bool
    ) -> None:
        target = target_name("options_chain_processing_attempts", self.migration_id)
        self._command(f"TRUNCATE TABLE {self.database}.{target}")
        retention_days = int(
            self.context.get("options_chain.migration.attempt_retention_days", 30)
        )
        cutoff = _utc_now() - timedelta(days=retention_days)
        if self._table_exists("options_chain_calculation_runs"):
            calculation_runs = self._current_attempt_relation(
                "options_chain_calculation_runs",
                id_column="run_id",
                latest_view="options_chain_calculation_runs_latest",
            )
            self._command(
                f"""INSERT INTO {self.database}.{target}
                (attempt_id, process_type, snapshot_id, source_collected_at,
                 definition_id, worker_id, started_at, finished_at, status,
                 expected_count, written_success_count, failure_count,
                 error_summary, created_at)
                SELECT run_id, 'calculation', snapshot_id, source_collected_at,
                       {int(definitions['calculation'])}, worker_id, started_at,
                       finished_at, toString(status), expected_base_rows,
                       written_calculation_rows, failed_contract_rows,
                       error_message, started_at
                FROM {calculation_runs}
                WHERE started_at >= {{cutoff:DateTime64(6, 'UTC')}}""",
                parameters={"cutoff": cutoff},
            )
        if include_surface and self._table_exists("options_chain_surface_runs"):
            surface_runs = self._current_attempt_relation(
                "options_chain_surface_runs",
                id_column="surface_run_id",
                latest_view="options_chain_surface_runs_latest",
            )
            self._command(
                f"""INSERT INTO {self.database}.{target}
                (attempt_id, process_type, snapshot_id, source_collected_at,
                 definition_id, worker_id, started_at, finished_at, status,
                 expected_count, written_success_count, failure_count,
                 error_summary, created_at)
                SELECT surface_run_id, 'surface', snapshot_id, source_collected_at,
                       {int(definitions['surface'])}, worker_id, started_at,
                       finished_at, toString(status), expected_expirations,
                       fit_rows_written + node_rows_written, failed_expirations,
                       error_message, started_at
                FROM {surface_runs}
                WHERE started_at >= {{cutoff:DateTime64(6, 'UTC')}}""",
                parameters={"cutoff": cutoff},
            )

    def _migrate_recent_validations(
        self, definitions: Mapping[str, int], *, include_surface: bool
    ) -> None:
        retention_days = int(
            self.context.get("options_chain.migration.validation_retention_days", 30)
        )
        cutoff = _utc_now() - timedelta(days=retention_days)
        if not include_surface and self._table_exists("options_chain_reuse_validation"):
            target = target_name(
                "options_chain_calculation_reuse_validation", self.migration_id
            )
            self._command(
                f"""INSERT INTO {self.database}.{target}
                (validation_id, snapshot_id, source_collected_at, contract_id,
                 calculation_definition_id, validation_type, metrics_json, created_at)
                SELECT v.validation_id, v.snapshot_id, v.source_collected_at,
                       {_contract_id_sql('b')}, {int(definitions['calculation'])},
                       v.validation_type,
                       toJSONString(map(
                         'comparison_status', toString(v.comparison_status),
                         'comparison_reason', toString(v.comparison_reason),
                         'absolute_iv_error', toString(v.absolute_iv_error),
                         'absolute_delta_error', toString(v.absolute_delta_error),
                         'candidate_age_seconds', toString(v.candidate_age_seconds))),
                       v.created_at
                FROM {self.database}.options_chain_reuse_validation AS v
                INNER JOIN {self.database}.options_chain_base AS b
                  ON b.snapshot_id = v.snapshot_id AND b.option_symbol = v.option_symbol
                WHERE v.created_at >= {{cutoff:DateTime64(6, 'UTC')}}""",
                parameters={"cutoff": cutoff},
            )
        if include_surface and self._table_exists("options_chain_surface_validation"):
            target = target_name("options_chain_surface_validation", self.migration_id)
            self._command(
                f"""INSERT INTO {self.database}.{target}
                (validation_id, snapshot_id, source_collected_at,
                 surface_definition_id, validation_type, metrics_json, created_at)
                SELECT validation_id, snapshot_id, source_collected_at,
                       {int(definitions['surface'])}, 'legacy_surface_comparison',
                       toJSONString(map(
                         'comparison_status', toString(comparison_status),
                         'weighted_rmse_difference', toString(weighted_rmse_difference),
                         'maximum_grid_iv_difference', toString(maximum_grid_iv_difference),
                         'maximum_node_iv_difference', toString(maximum_node_iv_difference),
                         'arbitrage_flags_changed', toString(arbitrage_flags_changed))),
                       created_at
                FROM {self.database}.options_chain_surface_validation
                WHERE created_at >= {{cutoff:DateTime64(6, 'UTC')}}""",
                parameters={"cutoff": cutoff},
            )

    def _ensure_migration_definitions(self) -> Mapping[str, int]:
        target = target_name("options_chain_definitions", self.migration_id)
        try:
            minimum = self.client.query(
                f"SELECT min(collected_at) FROM {self.database}.options_chain_base"
            ).result_rows[0][0]
        except Exception:
            minimum = None
        activated_at = _parse_timestamp(minimum) or datetime(1970, 1, 1, tzinfo=UTC)
        code_version = str(
            self.context.get("options_chain.migration.code_version", "legacy-selected")
        )
        collection = build_definition(
            DefinitionType.COLLECTION,
            code_version=code_version,
            configuration={
                key: self.context[key]
                for key in sorted(self.context)
                if str(key).startswith("options_chain.api_")
            },
            activated_at=activated_at,
        )
        normalization = build_definition(
            DefinitionType.NORMALIZATION,
            code_version=code_version,
            configuration={
                key: self.context[key]
                for key in sorted(self.context)
                if str(key).startswith("options_chain.")
                and any(
                    token in str(key)
                    for token in ("tenor", "strike", "nonstandard", "minis")
                )
            },
            activated_at=activated_at,
        )
        calculation = build_definition(
            DefinitionType.CALCULATION,
            code_version=code_version,
            configuration={
                key: self.context[key]
                for key in sorted(self.context)
                if str(key).startswith("options_calcs.")
            },
            solver_configuration={
                key: self.context[key]
                for key in sorted(self.context)
                if str(key).startswith("options_calcs.greeks.")
            },
            activated_at=activated_at,
        )
        surface = build_definition(
            DefinitionType.SURFACE,
            code_version=code_version,
            configuration={
                key: self.context[key]
                for key in sorted(self.context)
                if str(key).startswith("options_calcs.surface.")
            },
            parent_definition_id=calculation.definition_id,
            activated_at=activated_at,
        )
        definitions = (collection, normalization, calculation, surface)
        if len({item.definition_id for item in definitions}) != len(definitions):
            raise MigrationError(
                "Generated migration definitions have a cross-type UInt32 collision"
            )
        if self.dry_run:
            self.logger.info(
                "migration_phase migration_id=%s database=%s phase=definitions "
                "definition_ids=%s dry_run=true",
                self.migration_id,
                self.database,
                [item.definition_id for item in definitions],
            )
            return {
                "collection": collection.definition_id,
                "normalization": normalization.definition_id,
                "calculation": calculation.definition_id,
                "surface": surface.definition_id,
            }
        existing = {
            int(row[0]): (
                str(row[1]),
                int(row[2]) if row[2] is not None else None,
                str(row[3]),
                bytes(row[4]),
                bytes(row[5]),
            )
            for row in self.client.query(
                f"SELECT definition_id, toString(definition_type), "
                "parent_definition_id, code_version, configuration_hash, "
                "solver_configuration_hash "
                f"FROM {self.database}.{target} FINAL"
            ).result_rows
        }
        insert_rows = []
        for definition in definitions:
            hashes = existing.get(definition.definition_id)
            expected = (
                definition.definition_type.value,
                definition.parent_definition_id,
                definition.code_version,
                definition.configuration_hash,
                definition.solver_configuration_hash,
            )
            if hashes is not None and hashes != expected:
                raise MigrationError(
                    f"Definition collision for {definition.definition_id}"
                )
            if hashes is None:
                row = dict(definition.__dict__)
                row["definition_type"] = definition.definition_type.value
                insert_rows.append(row)
        if insert_rows:
            self.client.insert_arrow(
                f"{self.database}.{target}",
                _rows_to_arrow(insert_rows, DEFINITION_SCHEMA),
            )
        return {
            "collection": collection.definition_id,
            "normalization": normalization.definition_id,
            "calculation": calculation.definition_id,
            "surface": surface.definition_id,
        }

    def migrate_sources(self) -> None:
        self._ensure_lease(MigrationPhase.SOURCES_MIGRATED)
        if any(
            step in self.checkpoint.completed_steps
            for step in ("migrate_calculations", "migrate_surfaces", "archive_raw")
        ):
            raise MigrationError(
                "Source migration cannot be rerun after a downstream phase"
            )
        if not self.dry_run:
            for canonical in (
                "options_chain_raw",
                "options_chain_snapshots",
                "options_contracts",
                "options_chain_observations",
                "options_chain_observation_details",
                "options_chain_reference_rates",
                "options_chain_definitions",
            ):
                self._command(
                    f"TRUNCATE TABLE {self.database}."
                    f"{target_name(canonical, self.migration_id)}"
                )
        self.validate_source_column_disposition()
        definitions = self._ensure_migration_definitions()
        raw_initial_count: Any = 0
        raw_initial_high_water: Any = None
        raw_high_water_snapshot_ids: list[str] = []
        for _ in range(5):
            before = self.client.query(
                f"SELECT count(), max(collected_at_utc) "
                f"FROM {self.database}.options_chain_raw"
            ).result_rows[0]
            boundary_ids = []
            if before[1] is not None:
                boundary_ids = [
                    str(row[0])
                    for row in self.client.query(
                        f"SELECT toString(snapshot_id) FROM "
                        f"{self.database}.options_chain_raw "
                        "WHERE collected_at_utc = "
                        "parseDateTime64BestEffort({high_water:String}, 6, 'UTC') "
                        "ORDER BY snapshot_id",
                        parameters={
                            "high_water": _parse_timestamp(before[1]).isoformat(
                                timespec="microseconds"
                            )
                        },
                    ).result_rows
                ]
            after = self.client.query(
                f"SELECT count(), max(collected_at_utc) "
                f"FROM {self.database}.options_chain_raw"
            ).result_rows[0]
            if before == after:
                raw_initial_count, raw_initial_high_water = after
                raw_high_water_snapshot_ids = boundary_ids
                break
        else:
            raise MigrationError(
                "Raw capture boundary did not stabilize; use capture-only mode "
                "at a lower cadence or stop Collector"
            )
        source_bounds = self.client.query(
            f"SELECT min(collected_at), max(collected_at) "
            f"FROM {self.database}.options_chain_base"
        ).result_rows[0]
        integer_columns = (
            "contract_ordinal", "bid_size", "ask_size", "last_size",
            "total_volume", "open_interest",
        )
        integer_ranges = {}
        for column in integer_columns:
            minimum, maximum = self.client.query(
                f"SELECT min({column}), max({column}) "
                f"FROM {self.database}.options_chain_base WHERE isNotNull({column})"
            ).result_rows[0]
            integer_ranges[column] = {
                "minimum": int(minimum or 0),
                "maximum": int(maximum or 0),
            }
            if int(minimum or 0) < 0 or int(maximum or 0) > 2**32 - 1:
                raise MigrationError(
                    f"{column} range does not fit UInt32: {minimum}..{maximum}"
                )
        float32_columns = (
            "bid", "ask", "last", "mark", "open_price", "high_price",
            "low_price", "close_price", "high_52_week", "low_52_week",
            "net_change", "mark_change", "intrinsic_value", "extrinsic_value",
            "time_value", "break_even", "theoretical_option_value",
            "percent_change", "mark_percent_change",
            "theoretical_volatility_raw", "source_iv", "source_delta",
            "source_gamma", "source_theta", "source_vega", "source_rho",
        )
        cast_errors = {}
        for column in float32_columns:
            absolute, relative = self.client.query(
                f"SELECT max(abs(toFloat64({column}) - toFloat64(toFloat32({column})))), "
                f"max(abs(toFloat64({column}) - toFloat64(toFloat32({column}))) / "
                f"greatest(abs(toFloat64({column})), 1e-30)) "
                f"FROM {self.database}.options_chain_base WHERE isNotNull({column})"
            ).result_rows[0]
            cast_errors[column] = {
                "maximum_absolute": float(absolute or 0.0),
                "maximum_relative": float(relative or 0.0),
            }
            absolute_limit, relative_limit = FLOAT32_CAST_THRESHOLDS[column]
            if float(absolute or 0.0) > absolute_limit and float(
                relative or 0.0
            ) > relative_limit:
                raise MigrationError(
                    f"Float32 cast error exceeds threshold for {column}: "
                    f"absolute={absolute}, relative={relative}"
                )
        nonfinite_counts = {
            column: int(value or 0)
            for column, value in zip(
                float32_columns,
                self.client.query(
                    "SELECT "
                    + ", ".join(
                        f"countIf(isNotNull({column}) AND NOT isFinite({column}))"
                        for column in float32_columns
                    )
                    + f" FROM {self.database}.options_chain_base"
                ).result_rows[0],
            )
        }
        invalid_nonfinite = {
            name: count for name, count in nonfinite_counts.items() if count
        }
        if invalid_nonfinite:
            raise MigrationError(
                f"Nonfinite retained source values cannot be migrated: {invalid_nonfinite}"
            )
        source_null_counts = {
            column: int(value or 0)
            for column, value in zip(
                SOURCE_NULLABLE_TARGETS,
                self.client.query(
                    "SELECT "
                    + ", ".join(
                        f"countIf(isNull({column}))"
                        for column in SOURCE_NULLABLE_TARGETS
                    )
                    + f" FROM {self.database}.options_chain_base"
                ).result_rows[0],
            )
        }
        delayed_nulls = int(
            self.client.query(
                f"SELECT countIf(isNull(underlying_is_delayed)) "
                f"FROM {self.database}.options_chain_base"
            ).result_rows[0][0]
            or 0
        )
        # Older base schemas added this diagnostic as Nullable(Bool) without a
        # historical backfill.  Runtime normalization already maps a missing
        # value to False, so preserve that canonical behavior explicitly and
        # retain the affected-row count for operator validation.
        source_null_counts["underlying_is_delayed"] = delayed_nulls
        validation = {
            **self.checkpoint.validation_summary,
            "definition_ids": dict(definitions),
            "source_range": {
                "start": (
                    _parse_timestamp(source_bounds[0]).isoformat()
                    if source_bounds[0] is not None
                    else None
                ),
                "end": (
                    _parse_timestamp(source_bounds[1]).isoformat()
                    if source_bounds[1] is not None
                    else None
                ),
            },
            "source_integer_ranges": integer_ranges,
            "source_float32_cast_errors": cast_errors,
            "source_nonfinite_counts": nonfinite_counts,
            "source_null_counts": source_null_counts,
            "initial_raw_count": int(raw_initial_count or 0),
            "initial_raw_high_water": (
                _parse_timestamp(raw_initial_high_water).isoformat()
                if raw_initial_high_water is not None
                else None
            ),
            "initial_raw_high_water_snapshot_ids": raw_high_water_snapshot_ids,
        }
        contract_id = _contract_id_sql("b")
        self._insert_select(
            "options_chain_raw",
            "snapshot_id, source, symbol, request_started_at_utc, "
            "response_received_at_utc, collected_at_utc, response_latency_ms, "
            "request_parameters_json, response_json, payload_sha256, "
            "source_contract_count, is_delayed, is_chain_truncated, status",
            f"SELECT snapshot_id, source, symbol, request_started_at_utc, "
            "response_received_at_utc, collected_at_utc, response_latency_ms, "
            "request_parameters_json, response_json, unhex(payload_sha256_hex), "
            "source_contract_count, is_delayed, is_chain_truncated, status "
            f"FROM {self.database}.options_chain_raw",
        )
        # Fail before grouping if an economic identity changed historically.
        invalid_symbol_keys = self.client.query(
            f"SELECT count() FROM {self.database}.options_chain_base "
            "WHERE underlying_symbol != upper(trimBoth(underlying_symbol)) "
            "OR empty(underlying_symbol) OR empty(option_symbol)"
        ).result_rows[0][0]
        if int(invalid_symbol_keys):
            raise MigrationError(
                f"Found {invalid_symbol_keys} noncanonical contract symbol keys"
            )
        conflict = self.client.query(
            f"SELECT count() FROM (SELECT option_symbol FROM {self.database}.options_chain_base "
            "GROUP BY option_symbol HAVING uniqExact(tuple(underlying_symbol, put_call, "
            "strike_price, expiration_at, settlement_type, exercise_type, multiplier, "
            "is_mini, is_nonstandard, option_deliverables_json)) != 1)"
        ).result_rows[0][0]
        if int(conflict):
            raise MigrationError(f"Found {conflict} immutable contract identity conflicts")
        compact_id_collisions = self.client.query(
            f"SELECT count() FROM (SELECT {_contract_id_sql('k')} AS contract_id "
            f"FROM (SELECT underlying_symbol, option_symbol FROM "
            f"{self.database}.options_chain_base GROUP BY underlying_symbol, "
            "option_symbol) AS k GROUP BY contract_id HAVING count() != 1)"
        ).result_rows[0][0]
        if int(compact_id_collisions):
            raise MigrationError(
                f"Found {compact_id_collisions} deterministic UInt64 contract-ID collisions"
            )
        self.logger.info(
            "migration_partition migration_id=%s database=%s table=options_contracts "
            "partition=server_grouped_stream",
            self.migration_id,
            self.database,
        )
        contract_source_sql = f"""
            SELECT {_contract_id_sql('x')} AS contract_id,
                   option_symbol, underlying_symbol,
                   argMax(option_root, collected_at) AS option_root,
                   argMax(ssid, collected_at) AS ssid,
                   argMax(description, collected_at) AS description,
                   argMax(exchange_name, collected_at) AS exchange_name,
                   toString(any(put_call)) AS put_call,
                   any(strike_price) AS strike_price,
                   any(expiration_at) AS expiration_at,
                   argMax(expiration_type, collected_at) AS expiration_type,
                   argMax(last_trading_day, collected_at) AS last_trading_day,
                   any(settlement_type) AS settlement_type,
                   any(exercise_type) AS exercise_type,
                   toFloat32(any(multiplier)) AS multiplier,
                   any(is_mini) AS is_mini,
                   any(is_nonstandard) AS is_nonstandard,
                   argMax(is_penny_pilot, collected_at) AS is_penny_pilot,
                   argMax(deliverable_note, collected_at) AS deliverable_note,
                   any(option_deliverables_json) AS option_deliverables_json,
                   min(collected_at) AS first_seen_at,
                   max(collected_at) AS last_seen_at,
                   max(collected_at) AS metadata_updated_at
            FROM {self.database}.options_chain_base AS x
            GROUP BY option_symbol, underlying_symbol
            ORDER BY underlying_symbol, option_symbol
        """
        contract_target = target_name("options_contracts", self.migration_id)
        contract_rows: list[dict[str, Any]] = []
        for contract_row in _iter_arrow_rows(self.client, contract_source_sql):
            try:
                prepared = dict(contract_row)
                prepared["expiration_at"] = _parse_timestamp(
                    prepared["expiration_at"]
                )
                if prepared["expiration_at"] is None:
                    raise ValueError("expiration_at is invalid")
                prepared["last_trading_day"] = _parse_timestamp(
                    prepared.get("last_trading_day")
                )
                prepared["multiplier"] = float(prepared["multiplier"])
                if not math.isfinite(prepared["multiplier"]):
                    raise ValueError("multiplier must be finite")
                for name in ("is_mini", "is_nonstandard", "is_penny_pilot"):
                    prepared[name] = bool(prepared[name])
                prepared["identity_hash"] = contract_identity_hash(prepared)
                prepared["metadata_hash"] = contract_metadata_hash(prepared)
                contract_rows.append(prepared)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                self._quarantine_row(
                    "options_chain_base", contract_row, "contract", str(exc)
                )
                continue
            if len(contract_rows) >= 25_000:
                if not self.dry_run:
                    self.client.insert_arrow(
                        f"{self.database}.{contract_target}",
                        _rows_to_arrow(contract_rows, CANONICAL_CONTRACT_SCHEMA),
                    )
                contract_rows.clear()
        if contract_rows and not self.dry_run:
            self.client.insert_arrow(
                f"{self.database}.{contract_target}",
                _rows_to_arrow(contract_rows, CANONICAL_CONTRACT_SCHEMA),
            )
        hot = (
            "bid, ask, last, mark, bid_size, ask_size, last_size, open_price, high_price, "
            "low_price, close_price, total_volume, open_interest, quote_time, trade_time, "
            "source_iv, source_delta, source_gamma, source_theta, source_vega, source_rho"
        )
        self._insert_select(
            "options_chain_observations",
            "snapshot_id, source_collected_at, underlying_symbol, contract_id, "
            f"contract_ordinal, symbol_map_mismatch, {hot}",
            f"SELECT b.snapshot_id, b.collected_at, b.underlying_symbol, {contract_id}, "
            "toUInt32(b.contract_ordinal), assumeNotNull(b.symbol_map_mismatch), "
            "toFloat32(b.bid), toFloat32(b.ask), toFloat32(b.last), "
            "toFloat32(b.mark), toUInt32(b.bid_size), toUInt32(b.ask_size), "
            "toUInt32(b.last_size), toFloat32(b.open_price), "
            "toFloat32(b.high_price), toFloat32(b.low_price), "
            "toFloat32(b.close_price), toUInt32(b.total_volume), "
            "toUInt32(b.open_interest), b.quote_time, b.trade_time, b.source_iv, "
            "b.source_delta, b.source_gamma, b.source_theta, b.source_vega, b.source_rho "
            f"FROM {self.database}.options_chain_base AS b",
        )
        details = (
            "high_52_week, low_52_week, net_change, percent_change, mark_change, "
            "mark_percent_change, intrinsic_value, extrinsic_value, time_value, break_even, "
            "theoretical_option_value, theoretical_volatility"
        )
        self._insert_select(
            "options_chain_observation_details",
            f"snapshot_id, source_collected_at, underlying_symbol, contract_id, {details}",
            f"SELECT b.snapshot_id, b.collected_at, b.underlying_symbol, {contract_id}, "
            "toFloat32(b.high_52_week), toFloat32(b.low_52_week), "
            "toFloat32(b.net_change), b.percent_change, toFloat32(b.mark_change), "
            "b.mark_percent_change, toFloat32(b.intrinsic_value), "
            "toFloat32(b.extrinsic_value), toFloat32(b.time_value), "
            "toFloat32(b.break_even), toFloat32(b.theoretical_option_value), "
            "b.theoretical_volatility_raw "
            f"FROM {self.database}.options_chain_base AS b",
        )
        # Snapshot manifests are deliberately last.  Definition IDs are filled
        # from migration-generated defaults in the metadata phase (zero is an
        # invalid sentinel and validation blocks cutover until resolved).
        self._insert_select(
            "options_chain_snapshots",
            "snapshot_id, source, underlying_symbol, source_collected_at, request_started_at, "
            "response_received_at, ingested_at, response_latency_ms, payload_sha256, "
            "chain_status, strategy, chain_interval, chain_is_delayed, chain_is_index, "
            "chain_is_truncated, underlying_is_delayed, source_contract_count, "
            "stored_observation_count, stored_detail_count, stored_distinct_contract_count, "
            "asset_main_type, asset_sub_type, chain_interest_rate, chain_dividend_yield, "
            "chain_volatility, chain_underlying_price, underlying_bid, underlying_ask, "
            "underlying_last, underlying_mark, underlying_bid_size, underlying_ask_size, "
            "underlying_quote_time, underlying_trade_time, underlying_total_volume, "
            "collection_definition_id, normalization_definition_id, normalized_at",
            f"""SELECT b.snapshot_id, any(b.source), any(b.underlying_symbol), any(b.collected_at),
            any(b.request_started_at), any(b.response_received_at), any(b.ingested_at),
            any(r.response_latency_ms), any(unhex(r.payload_sha256_hex)), any(b.chain_status),
            any(b.strategy), any(b.chain_interval), any(b.chain_is_delayed), any(b.chain_is_index),
            any(b.chain_is_truncated), coalesce(any(b.underlying_is_delayed), false),
            any(b.source_contract_count), count(), count(), uniqExact({_contract_id_sql('b')}),
            any(b.asset_main_type), any(b.asset_sub_type), any(b.chain_interest_rate),
            any(b.chain_dividend_yield), any(b.chain_volatility_raw),
            any(b.chain_underlying_price), any(b.underlying_bid), any(b.underlying_ask),
            any(b.underlying_last), any(b.underlying_mark), any(b.underlying_bid_size),
            any(b.underlying_ask_size), any(b.underlying_quote_time), any(b.underlying_trade_time),
            any(b.underlying_total_volume), {definitions['collection']},
            {definitions['normalization']}, max(b.ingested_at)
            FROM {self.database}.options_chain_base AS b
            INNER JOIN {self.database}.options_chain_raw AS r ON r.snapshot_id = b.snapshot_id
            GROUP BY b.snapshot_id""",
        )
        self._insert_select(
            "options_chain_reference_rates",
            "series_id, observation_date, value_percent, retrieved_at, source",
            f"""SELECT series_id, observation_date,
                       argMax(value_percent, retrieved_at), max(retrieved_at),
                       argMax(source, retrieved_at)
                FROM {self.database}.options_chain_reference_rates
                GROUP BY series_id, observation_date""",
        )
        self._checkpoint(
            "migrate_sources",
            MigrationPhase.SOURCES,
            validation_summary=validation,
        )

    def migrate_calculations(self) -> None:
        self._ensure_lease(MigrationPhase.CALCULATIONS_MIGRATED)
        # Selected-publication reconstruction is server-side and intentionally
        # anchored to the legacy latest-complete view before cutover.
        if "migrate_surfaces" in self.checkpoint.completed_steps:
            raise MigrationError(
                "Calculation migration cannot be rerun after surface migration"
            )
        if not self.dry_run:
            for canonical in (
                "options_chain_expiration_inputs",
                "options_chain_calculation_results",
                "options_chain_calculation_failures",
                "options_chain_calculation_reuse_provenance",
                "options_chain_calculation_diagnostics",
                "options_chain_calculation_reuse_validation",
                "options_chain_calculation_state",
            ):
                self._command(
                    f"TRUNCATE TABLE {self.database}."
                    f"{target_name(canonical, self.migration_id)}"
                )
        definitions = self._ensure_migration_definitions()
        calculation_stale_seconds = int(
            self.context.get("options_chain.migration.calculation_stale_seconds", 3600)
        )
        success_predicate = _legacy_success_predicate("c")
        integer_ranges = {}
        for name, condition, maximum in (
            (
                "iv_iterations",
                "c.calculation_status = 'complete' "
                "AND c.recalc_status IN ('complete', 'reused') "
                "AND isNotNull(c.recalc_iv_iterations)",
                2**16 - 1,
            ),
            ("time_to_expiry_seconds", "isNotNull(c.time_to_expiry_seconds)", 2**32 - 1),
            (
                "recalc_valuation_time_to_expiry_seconds",
                "isNotNull(c.recalc_valuation_time_to_expiry_seconds)",
                2**32 - 1,
            ),
            (
                "recalc_reuse_age_seconds",
                "c.recalc_status = 'reused' AND isNotNull(c.recalc_reuse_age_seconds)",
                2**32 - 1,
            ),
        ):
            column = "recalc_iv_iterations" if name == "iv_iterations" else name
            minimum, largest = self.client.query(
                f"SELECT min(c.{column}), max(c.{column}) FROM "
                f"{self.database}.options_chain_calculations_latest AS c "
                f"WHERE {condition}"
            ).result_rows[0]
            integer_ranges[name] = {
                "minimum": int(minimum or 0),
                "maximum": int(largest or 0),
            }
            if int(minimum or 0) < 0 or int(largest or 0) > maximum:
                raise MigrationError(
                    f"{name} range does not fit target integer type: "
                    f"{minimum}..{largest}"
                )
        result_target = target_name("options_chain_calculation_results", self.migration_id)
        self._command(
            f"""INSERT INTO {self.database}.{result_target}
            (snapshot_id, source_collected_at, underlying_symbol, contract_id,
             calculation_definition_id, source_calculation_run_id, calculated_at,
             recalc_method, result_origin,
             origin_snapshot_id, input_basis, dividend_model, iv_selection,
             greeks_selection, target_price, iv, iv_iterations, price_residual,
             model_price, delta, gamma, theta_per_day, vega_per_1pct, rho_per_1pct,
             actual_dtype, precision_escalated, quality_flags)
            SELECT c.snapshot_id, c.source_collected_at, b.underlying_symbol,
                   {_contract_id_sql('b')}, {definitions['calculation']}, run.run_id,
                   c.calculated_at, c.recalc_method,
                   if(c.recalc_result_origin = 'reused', 'reused', 'fresh'),
                   coalesce(c.recalc_origin_snapshot_id, c.snapshot_id),
                   c.recalc_input_basis,
                   c.recalc_dividend_model,
                   if(c.recalc_iv_identifiability_status = 'source_anchor_flat',
                      'source', 'result'),
                   'result', c.recalc_target_price, c.recalc_iv, c.recalc_iv_iterations,
                   c.recalc_price_residual, c.recalc_model_price, c.recalc_delta,
                   c.recalc_gamma, c.recalc_theta_per_day, c.recalc_vega_per_1pct,
                   c.recalc_rho_per_1pct, c.recalc_actual_dtype,
                   c.recalc_precision_escalated, c.recalc_quality_flags
            FROM {self.database}.options_chain_calculations_latest AS c
            INNER JOIN {self.database}.options_chain_calculation_runs_latest AS run
              ON run.snapshot_id = c.snapshot_id
             AND run.calculation_version = c.calculation_version
            INNER JOIN {self.database}.options_chain_base AS b
              ON b.snapshot_id = c.snapshot_id AND b.option_symbol = c.option_symbol
            WHERE run.status IN ('complete', 'complete_with_failures')
              AND {success_predicate}"""
        )
        expiration_target = target_name(
            "options_chain_expiration_inputs", self.migration_id
        )
        inconsistent_expirations = self.client.query(
            f"""SELECT count() FROM
            (SELECT c.snapshot_id, b.expiration_at
             FROM {self.database}.options_chain_calculations_latest AS c
             INNER JOIN {self.database}.options_chain_calculation_runs_latest AS run
               ON run.snapshot_id = c.snapshot_id
              AND run.calculation_version = c.calculation_version
             INNER JOIN {self.database}.options_chain_base AS b
               ON b.snapshot_id = c.snapshot_id AND b.option_symbol = c.option_symbol
             WHERE run.status IN ('complete', 'complete_with_failures')
             GROUP BY c.snapshot_id, b.expiration_at
             HAVING uniqExact(tuple(c.time_to_expiry_seconds, c.risk_free_rate,
                                    c.dividend_yield, c.forward_price,
                                    c.fred_observation_date, c.fred_lower_series,
                                    c.fred_upper_series)) != 1)"""
        ).result_rows[0][0]
        if int(inconsistent_expirations):
            raise MigrationError(
                f"Found {inconsistent_expirations} inconsistent expiration inputs"
            )
        self._command(
            f"""INSERT INTO {self.database}.{expiration_target}
            (snapshot_id, source_collected_at, underlying_symbol, expiration_at,
             calculation_definition_id, time_to_expiry_seconds, risk_free_rate,
             dividend_yield, forward_price, fred_observation_date, fred_lower_series,
             fred_upper_series, quality_flags, calculated_at)
            SELECT c.snapshot_id, any(c.source_collected_at), any(b.underlying_symbol),
                   b.expiration_at, {definitions['calculation']},
                   toUInt32(intDiv(toInt64(any(c.time_to_expiry_seconds) * 1000000), 1000000)),
                   any(c.risk_free_rate), any(c.dividend_yield), any(c.forward_price),
                   any(c.fred_observation_date), any(c.fred_lower_series),
                   any(c.fred_upper_series), groupBitOr(c.quality_flags), max(c.calculated_at)
            FROM {self.database}.options_chain_calculations_latest AS c
            INNER JOIN {self.database}.options_chain_calculation_runs_latest AS run
              ON run.snapshot_id = c.snapshot_id
             AND run.calculation_version = c.calculation_version
            INNER JOIN {self.database}.options_chain_base AS b
              ON b.snapshot_id = c.snapshot_id AND b.option_symbol = c.option_symbol
            WHERE run.status IN ('complete', 'complete_with_failures')
              AND c.time_to_expiry_seconds >= 0
            GROUP BY c.snapshot_id, b.expiration_at"""
        )
        failure_target = target_name(
            "options_chain_calculation_failures", self.migration_id
        )
        self._command(
            f"""INSERT INTO {self.database}.{failure_target}
            (snapshot_id, source_collected_at, underlying_symbol,
             calculation_definition_id, source_calculation_run_id, failure_scope,
             expiration_at, contract_id,
             failure_stage, reason_code, error_class, error_detail, retry_class,
             retry_after, failure_fingerprint, first_failed_at, last_failed_at,
             failure_count, updated_at, is_active, valuation_at, spot, target_price,
             time_to_expiry_seconds, risk_free_rate, dividend_yield, method,
             input_basis, dividend_model, partial_iv, partial_model_price,
             partial_price_residual, seed_iv, seed_origin_snapshot_id, quality_flags)
            SELECT c.snapshot_id, c.source_collected_at, b.underlying_symbol,
                   {definitions['calculation']}, run.run_id, 'contract', b.expiration_at,
                   {_contract_id_sql('b')}, c.recalc_failure_stage,
                   if(c.recalc_reason = '', 'invalid_legacy_success', c.recalc_reason),
                   'LegacyCalculationFailure', left(toString(c.calculation_error), 512),
                   c.recalc_retry_class, c.recalc_retry_after,
                   if(length(c.recalc_failure_fingerprint) = 64,
                      unhex(c.recalc_failure_fingerprint),
                      SHA256(concat(c.recalc_failure_stage, '|', c.recalc_reason))),
                   c.calculated_at, c.calculated_at, 1, c.calculated_at, true,
                   c.recalc_valuation_at, c.recalc_valuation_spot,
                   c.recalc_valuation_target_price,
                   toUInt32(c.recalc_valuation_time_to_expiry_seconds),
                   c.recalc_valuation_rate, c.recalc_valuation_dividend_yield,
                   nullIf(c.recalc_method, ''), nullIf(c.recalc_input_basis, ''),
                   nullIf(c.recalc_dividend_model, ''), c.recalc_iv,
                   c.recalc_model_price, c.recalc_price_residual,
                   c.recalc_iv_seed_value,
                   c.recalc_iv_seed_snapshot_id,
                   c.recalc_quality_flags
            FROM {self.database}.options_chain_calculations_latest AS c
            INNER JOIN {self.database}.options_chain_calculation_runs_latest AS run
              ON run.snapshot_id = c.snapshot_id
             AND run.calculation_version = c.calculation_version
            INNER JOIN {self.database}.options_chain_base AS b
              ON b.snapshot_id = c.snapshot_id AND b.option_symbol = c.option_symbol
            WHERE run.status IN ('failed', 'complete', 'complete_with_failures')
              AND {_legacy_terminal_failure_predicate('c')}"""
        )
        provenance_target = target_name(
            "options_chain_calculation_reuse_provenance", self.migration_id
        )
        self._command(
            f"""INSERT INTO {self.database}.{provenance_target}
            (snapshot_id, source_collected_at, underlying_symbol, contract_id,
             calculation_definition_id, origin_snapshot_id, origin_calculated_at,
             origin_valuation_at, origin_spot, origin_target_price,
             origin_time_to_expiry_seconds, origin_risk_free_rate,
             origin_dividend_yield, origin_input_basis, origin_method,
             origin_dividend_model, reuse_age_seconds, spot_movement_bps,
             target_price_absolute_change, target_price_fractional_change,
             elapsed_tenor_fraction, rate_change, dividend_yield_change,
             estimated_iv_drift, estimated_delta_drift, reuse_mode, policy_reason,
             created_at)
            SELECT c.snapshot_id, c.source_collected_at, b.underlying_symbol,
                   {_contract_id_sql('b')}, {definitions['calculation']},
                   c.recalc_origin_snapshot_id, c.recalc_origin_calculated_at,
                   c.recalc_valuation_at, c.recalc_valuation_spot,
                   c.recalc_valuation_target_price,
                   toUInt32(c.recalc_valuation_time_to_expiry_seconds),
                   c.recalc_valuation_rate, c.recalc_valuation_dividend_yield,
                   c.recalc_input_basis, c.recalc_method, c.recalc_dividend_model,
                   toUInt32(c.recalc_reuse_age_seconds),
                   toFloat32(c.recalc_reuse_spot_move_bps),
                   toFloat32(abs(c.recalc_target_price - c.recalc_valuation_target_price)),
                   toFloat32(abs(c.recalc_target_price - c.recalc_valuation_target_price)
                             / greatest(abs(c.recalc_valuation_target_price), 1e-30)),
                   toFloat32(c.recalc_reuse_elapsed_tenor_fraction),
                   toFloat32(abs(c.risk_free_rate - c.recalc_valuation_rate)),
                   toFloat32(abs(c.dividend_yield - c.recalc_valuation_dividend_yield)),
                   toFloat32(c.recalc_reuse_estimated_iv_change_points), 0,
                   c.recalc_reuse_mode, c.recalc_reason, c.calculated_at
            FROM {self.database}.options_chain_calculations_latest AS c
            INNER JOIN {self.database}.options_chain_calculation_runs_latest AS run
              ON run.snapshot_id = c.snapshot_id
             AND run.calculation_version = c.calculation_version
            INNER JOIN {self.database}.options_chain_base AS b
              ON b.snapshot_id = c.snapshot_id AND b.option_symbol = c.option_symbol
            WHERE run.status IN ('complete', 'complete_with_failures')
              AND c.recalc_status = 'reused'
              AND isNotNull(c.recalc_origin_snapshot_id)
              AND isNotNull(c.recalc_valuation_at)
              AND isNotNull(c.recalc_valuation_spot)
              AND isNotNull(c.recalc_valuation_target_price)
              AND isNotNull(c.recalc_valuation_time_to_expiry_seconds)
              AND isNotNull(c.recalc_valuation_rate)
              AND isNotNull(c.recalc_valuation_dividend_yield)"""
        )
        state_target = target_name(
            "options_chain_calculation_state", self.migration_id
        )
        self._command(
            f"""INSERT INTO {self.database}.{state_target}
            (snapshot_id, source_collected_at, underlying_symbol,
             calculation_definition_id, source_calculation_run_id, status,
             expected_observations,
             eligible_contracts, successful_results, source_only_contracts,
             terminal_contract_failures, terminal_expiration_failures,
             snapshot_failure_count, started_at, finished_at, updated_at,
             active_attempt_id, error_summary, is_active)
            SELECT run.snapshot_id, run.source_collected_at, run.underlying_symbol,
                   {definitions['calculation']}, run.run_id,
                   multiIf(
                     run.status = 'running' AND isNotNull(run.started_at)
                      AND dateDiff('second', run.started_at, now64(6)) >= {calculation_stale_seconds},
                      'stale_running',
                     run.status IN ('pending', 'running', 'stale_running', 'abandoned'),
                      run.status,
                     run.status = 'failed', 'failed',
                     coalesce(f.failure_count, 0) > 0 OR run.failed_contract_rows > 0,
                      'complete_with_failures', 'complete'),
                   run.expected_base_rows, run.expected_base_rows,
                   coalesce(r.result_count, 0),
                   if(run.status IN ('pending', 'running', 'stale_running', 'abandoned'),
                      0,
                      greatest(0, run.expected_base_rows - coalesce(r.result_count, 0)
                                     - coalesce(f.failure_count, 0))),
                   coalesce(f.failure_count, 0), 0, 0,
                   run.started_at, run.finished_at, run.finished_at, NULL, '', true
            FROM {self.database}.options_chain_calculation_runs_latest AS run
            LEFT JOIN
              (SELECT snapshot_id, count() AS result_count
               FROM {self.database}.{result_target} GROUP BY snapshot_id) AS r
              ON r.snapshot_id = run.snapshot_id
            LEFT JOIN
              (SELECT snapshot_id, count() AS failure_count
               FROM {self.database}.{failure_target}
               WHERE is_active GROUP BY snapshot_id) AS f
              ON f.snapshot_id = run.snapshot_id"""
        )
        self._migrate_recent_attempts(definitions, include_surface=False)
        self._migrate_recent_validations(definitions, include_surface=False)
        self._checkpoint(
            "migrate_calculations",
            MigrationPhase.CALCULATIONS,
            validation_summary={
                **self.checkpoint.validation_summary,
                "calculation_integer_ranges": integer_ranges,
            },
        )

    def migrate_surfaces(self) -> None:
        self._ensure_lease(MigrationPhase.SURFACES_MIGRATED)
        # Typed parsing is performed in bounded Arrow batches because malformed
        # legacy JSON must become a fit-scope failure instead of invalid success.
        definitions = self._ensure_migration_definitions()
        fit_target = target_name("options_chain_surface_fits", self.migration_id)
        node_target = target_name("options_chain_surface_nodes", self.migration_id)
        failure_target = target_name(
            "options_chain_surface_failures", self.migration_id
        )
        state_target = target_name("options_chain_surface_state", self.migration_id)
        if not self.dry_run:
            # An interrupted, uncheckpointed phase is restarted from empty
            # migration-only targets, making a resumed run duplicate-free.
            for table in (
                fit_target,
                node_target,
                failure_target,
                state_target,
                target_name("options_chain_surface_diagnostics", self.migration_id),
                target_name("options_chain_surface_validation", self.migration_id),
            ):
                self._command(f"TRUNCATE TABLE {self.database}.{table}")

        fit_insert_schema = CANONICAL_SURFACE_FIT_SCHEMA.append(
            pa.field("source_surface_run_id", pa.string())
        )
        node_insert_schema = CANONICAL_SURFACE_NODE_SCHEMA.append(
            pa.field("source_surface_run_id", pa.string())
        )
        failure_insert_schema = CANONICAL_SURFACE_FAILURE_SCHEMA.append(
            pa.field("source_surface_run_id", pa.string())
        )
        state_insert_schema = CANONICAL_SURFACE_STATE_SCHEMA.append(
            pa.field("source_surface_run_id", pa.string())
        )

        def flush(
            table: str,
            rows: list[dict[str, Any]],
            schema: Any,
        ) -> None:
            if rows and not self.dry_run:
                self._insert_arrow_while_streaming(
                    f"{self.database}.{table}", _rows_to_arrow(rows, schema)
                )
            rows.clear()

        def checked_uint(value: Any, name: str, maximum: int = 2**32 - 1) -> int:
            result = int(value or 0)
            if result < 0 or result > maximum:
                raise ValueError(f"{name} is outside 0..{maximum}: {result}")
            return result

        def checked_optional_uint(value: Any, name: str) -> Optional[int]:
            return None if value is None else checked_uint(value, name)

        def failure_uint(value: Any) -> Optional[int]:
            try:
                return None if value is None else checked_uint(value, "failure_count")
            except (ValueError, TypeError, OverflowError):
                return None

        def checked_float(
            value: Any, name: str, *, nullable: bool = False
        ) -> Optional[float]:
            if value is None and nullable:
                return None
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"{name} must be finite")
            return result

        fit_source_sql = (
            f"SELECT f.* EXCEPT(surface_run_id, snapshot_id, source_calculation_run_id, "
            f"price_basis, fit_method, fit_status, failure_stage, requested_dtype, "
            f"actual_dtype, node_generation_status, required_node_status, "
            f"optional_node_status, node_requested_dtype, node_actual_dtype), "
            f"toString(f.surface_run_id) AS surface_run_id, "
            f"toString(f.snapshot_id) AS snapshot_id, "
            f"toString(f.source_calculation_run_id) AS source_calculation_run_id, "
            f"toString(f.price_basis) AS price_basis, "
            f"toString(f.fit_method) AS fit_method, "
            f"toString(f.fit_status) AS fit_status, "
            f"toString(f.failure_stage) AS failure_stage, "
            f"toString(f.requested_dtype) AS requested_dtype, "
            f"toString(f.actual_dtype) AS actual_dtype, "
            f"toString(f.node_generation_status) AS node_generation_status, "
            f"toString(f.required_node_status) AS required_node_status, "
            f"toString(f.optional_node_status) AS optional_node_status, "
            f"toString(f.node_requested_dtype) AS node_requested_dtype, "
            f"toString(f.node_actual_dtype) AS node_actual_dtype, "
            f"toString(r.surface_run_id) AS authoritative_surface_run_id "
            f"FROM {self.database}.options_chain_surface_fits AS f "
            f"INNER JOIN {self.database}.options_chain_surface_runs_latest AS r "
            "ON r.surface_run_id = f.surface_run_id "
            "AND r.snapshot_id = f.snapshot_id "
            "AND r.surface_version = f.surface_version "
            "AND r.source_calculation_run_id = f.source_calculation_run_id "
            "WHERE r.status IN ('complete', 'complete_with_failures') "
            "ORDER BY f.source_collected_at, f.snapshot_id, f.expiration_at, f.price_basis"
        )
        successes = 0
        failures = 0
        fit_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        fits_by_snapshot: dict[str, set[Any]] = {}
        successful_fit_keys: set[tuple[str, Any, str, str]] = set()
        fit_failures_by_snapshot: dict[str, set[Any]] = {}
        skipped_by_snapshot: dict[str, set[Any]] = {}
        for row in _iter_arrow_rows(self.client, fit_source_sql):
            try:
                uuid.UUID(str(row["snapshot_id"]))
            except (ValueError, TypeError, KeyError) as exc:
                self._quarantine_row(
                    "options_chain_surface_fits", row, "snapshot_id", str(exc)
                )
                continue
            source_collected_at = _parse_timestamp(row.get("source_collected_at"))
            natural_expiration_at = _parse_timestamp(row.get("expiration_at"))
            sid = str(row.get("snapshot_id") or "")
            status = str(row.get("fit_status") or "").lower()
            classification = classify_surface_expiration(status)
            if classification is SurfaceExpirationClassification.SKIPPED:
                if str(row.get("price_basis") or "") in {"mid", "source"}:
                    skipped_by_snapshot.setdefault(sid, set()).add(
                        _parse_timestamp(row.get("expiration_at"))
                    )
                continue
            if source_collected_at is None or natural_expiration_at is None:
                self._quarantine_row(
                    "options_chain_surface_fits", row, "expiration_at",
                    "invalid natural-key timestamp",
                )
                continue
            underlying_symbol = str(row.get("underlying_symbol") or "")
            source_run_id = str(
                row.get("authoritative_surface_run_id") or row.get("surface_run_id") or ""
            )
            if not underlying_symbol or not source_run_id:
                self._quarantine_row(
                    "options_chain_surface_fits", row, "source_identity",
                    "missing underlying symbol or surface run identity",
                )
                continue
            try:
                if classification is SurfaceExpirationClassification.FAILURE:
                    raise ValueError(str(row.get("fit_error") or status))
                typed = parse_fit_parameters_json(
                    str(row.get("fit_parameters_json") or "")
                )
                required_diagnostics = (
                    "minimum_log_forward_moneyness",
                    "maximum_log_forward_moneyness",
                    "weighted_rmse",
                    "maximum_absolute_error",
                )
                diagnostics = {
                    name: checked_float(row.get(name), name)
                    for name in required_diagnostics
                }
                expiration_at = natural_expiration_at
                calculated_at = _parse_timestamp(row.get("calculated_at"))
                if expiration_at is None or calculated_at is None:
                    raise ValueError("successful fit has an invalid timestamp")
                actual_dtype = str(row.get("actual_dtype") or "float64")
                if actual_dtype not in {"float32", "float64"}:
                    raise ValueError(f"unsupported successful fit dtype {actual_dtype!r}")
                fit_rows.append(
                    {
                        "snapshot_id": str(row["snapshot_id"]),
                        "source_collected_at": source_collected_at,
                        "underlying_symbol": underlying_symbol,
                        "expiration_at": expiration_at,
                        "price_basis": str(row["price_basis"]),
                        "surface_definition_id": definitions["surface"],
                        "source_surface_run_id": source_run_id,
                        "calculated_at": calculated_at,
                        "pricing_method": str(row.get("pricing_method") or "unknown"),
                        **typed,
                        "input_contract_count": checked_uint(
                            row.get("input_contract_count"), "input_contract_count"
                        ),
                        "accepted_contract_count": checked_uint(
                            row.get("accepted_contract_count"), "accepted_contract_count"
                        ),
                        "rejected_contract_count": checked_uint(
                            row.get("rejected_contract_count"), "rejected_contract_count"
                        ),
                        "selected_contract_count": checked_uint(
                            row.get("fit_selected_contract_count")
                            or row.get("accepted_contract_count")
                            or 0,
                            "selected_contract_count",
                        ),
                        "source_current_iv_point_count": checked_uint(
                            row.get("recalculated_current_iv_point_count"),
                            "source_current_iv_point_count",
                        ),
                        "source_reused_iv_point_count": checked_uint(
                            row.get("recalculated_reused_iv_point_count"),
                            "source_reused_iv_point_count",
                        ),
                        "source_vendor_iv_point_count": checked_uint(
                            row.get("source_iv_point_count"),
                            "source_vendor_iv_point_count",
                        ),
                        "no_iv_excluded_count": checked_uint(
                            row.get("no_iv_excluded_count"), "no_iv_excluded_count"
                        ),
                        "source_iv_weight_fraction": checked_float(
                            row.get("source_iv_weight_fraction") or 0.0,
                            "source_iv_weight_fraction",
                        ),
                        "minimum_input_delta": checked_float(
                            row.get("minimum_input_delta"),
                            "minimum_input_delta",
                            nullable=True,
                        ),
                        "maximum_input_delta": checked_float(
                            row.get("maximum_input_delta"),
                            "maximum_input_delta",
                            nullable=True,
                        ),
                        "minimum_log_forward_moneyness": diagnostics[
                            "minimum_log_forward_moneyness"
                        ],
                        "maximum_log_forward_moneyness": diagnostics[
                            "maximum_log_forward_moneyness"
                        ],
                        "weighted_rmse": diagnostics["weighted_rmse"],
                        "maximum_absolute_error": diagnostics[
                            "maximum_absolute_error"
                        ],
                        "butterfly_arbitrage_flags": checked_uint(
                            row.get("butterfly_arbitrage_flags"),
                            "butterfly_arbitrage_flags",
                        ),
                        "calendar_arbitrage_flags": checked_uint(
                            row.get("calendar_arbitrage_flags"),
                            "calendar_arbitrage_flags",
                        ),
                        "quality_flags": checked_uint(
                            row.get("quality_flags"), "quality_flags"
                        ),
                        "actual_dtype": actual_dtype,
                        "precision_escalated": bool(
                            row.get("precision_escalated")
                        ),
                        "required_node_status": str(
                            row.get("required_node_status") or "not_requested"
                        ),
                        "optional_node_status": str(
                            row.get("optional_node_status") or "not_requested"
                        ),
                    }
                )
                successes += 1
                if str(row["price_basis"]) in {"mid", "source"}:
                    fits_by_snapshot.setdefault(str(row["snapshot_id"]), set()).add(
                        expiration_at
                    )
                successful_fit_keys.add(
                    (
                        str(row["snapshot_id"]), expiration_at,
                        str(row["price_basis"]),
                        str(row.get("authoritative_surface_run_id") or row.get("surface_run_id")),
                    )
                )
            except (ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError) as exc:
                now = _parse_timestamp(row.get("calculated_at")) or _utc_now()
                fingerprint = hashlib.sha256(
                    f"fit|{row.get('snapshot_id')}|{row.get('expiration_at')}|"
                    f"{row.get('price_basis')}|{status}|{exc}".encode()
                ).digest()
                failure_rows.append(
                    {
                        "snapshot_id": sid,
                        "source_collected_at": source_collected_at,
                        "underlying_symbol": underlying_symbol,
                        "surface_definition_id": definitions["surface"],
                        "source_surface_run_id": source_run_id,
                        "failure_scope": "fit",
                        "expiration_at": natural_expiration_at,
                        "price_basis": str(row["price_basis"]),
                        "node_type": None,
                        "node_value": None,
                        "put_call": None,
                        "failure_stage": str(
                            row.get("failure_stage") or "migration_validation"
                        ),
                        "reason_code": status or "malformed_success",
                        "error_class": type(exc).__name__,
                        "error_detail": _bounded_text(exc),
                        "retry_class": "deterministic",
                        "retry_after": None,
                        "failure_fingerprint": fingerprint,
                        "input_contract_count": failure_uint(row.get("input_contract_count")),
                        "accepted_contract_count": failure_uint(row.get("accepted_contract_count")),
                        "rejected_contract_count": failure_uint(row.get("rejected_contract_count")),
                        "first_failed_at": now,
                        "last_failed_at": now,
                        "failure_count": 1,
                        "updated_at": now,
                        "quality_flags": failure_uint(row.get("quality_flags")) or 0,
                        "is_active": True,
                    }
                )
                failures += 1
                if str(row.get("price_basis") or "") in {"mid", "source"}:
                    fit_failures_by_snapshot.setdefault(sid, set()).add(
                        natural_expiration_at
                    )
            if len(fit_rows) >= 25_000:
                flush(fit_target, fit_rows, fit_insert_schema)
            if len(failure_rows) >= 25_000:
                flush(
                    failure_target,
                    failure_rows,
                    failure_insert_schema,
                )
        flush(fit_target, fit_rows, fit_insert_schema)
        flush(failure_target, failure_rows, failure_insert_schema)

        node_source_sql = (
            f"SELECT n.* EXCEPT(surface_run_id, snapshot_id, source_calculation_run_id, "
            f"node_type, put_call, node_status), "
            f"toString(n.surface_run_id) AS surface_run_id, "
            f"toString(n.snapshot_id) AS snapshot_id, "
            f"toString(n.source_calculation_run_id) AS source_calculation_run_id, "
            f"toString(n.node_type) AS node_type, "
            f"toString(n.put_call) AS put_call, "
            f"toString(n.node_status) AS node_status, "
            f"toString(r.surface_run_id) AS authoritative_surface_run_id "
            f"FROM {self.database}.options_chain_surface_nodes AS n "
            f"INNER JOIN {self.database}.options_chain_surface_runs_latest AS r "
            "ON r.surface_run_id = n.surface_run_id "
            "AND r.snapshot_id = n.snapshot_id "
            "AND r.surface_version = n.surface_version "
            "AND r.source_calculation_run_id = n.source_calculation_run_id "
            "WHERE r.status IN ('complete', 'complete_with_failures') "
            "ORDER BY n.source_collected_at, n.snapshot_id, n.expiration_at, "
            "n.node_type, n.node_value, n.put_call"
        )
        node_rows: list[dict[str, Any]] = []
        node_successes = 0
        node_failures = 0
        required_nodes_by_snapshot: dict[str, int] = {}
        required_node_successes_by_snapshot: dict[str, int] = {}
        required_node_failures_by_snapshot: dict[str, int] = {}
        optional_node_successes_by_snapshot: dict[str, int] = {}
        optional_node_failures_by_snapshot: dict[str, int] = {}
        for row in _iter_arrow_rows(self.client, node_source_sql):
            try:
                uuid.UUID(str(row["snapshot_id"]))
            except (ValueError, TypeError, KeyError) as exc:
                self._quarantine_row(
                    "options_chain_surface_nodes", row, "snapshot_id", str(exc)
                )
                continue
            status = str(row.get("node_status") or "").lower()
            if status.startswith("skipped"):
                continue
            sid = str(row["snapshot_id"])
            node_type = str(row.get("node_type") or "")
            try:
                node_value = checked_float(row.get("node_value"), "node_value")
            except (ValueError, TypeError, OverflowError) as exc:
                self._quarantine_row(
                    "options_chain_surface_nodes", row, "node_value", str(exc)
                )
                continue
            expiration_at = _parse_timestamp(row.get("expiration_at"))
            source_collected_at = _parse_timestamp(row.get("source_collected_at"))
            put_call = str(row.get("put_call") or "")
            if expiration_at is None or source_collected_at is None:
                self._quarantine_row(
                    "options_chain_surface_nodes", row, "expiration_at",
                    "invalid natural-key timestamp",
                )
                continue
            if not node_type or put_call not in {"NONE", "CALL", "PUT"}:
                self._quarantine_row(
                    "options_chain_surface_nodes", row, "node_key",
                    "invalid natural key",
                )
                continue
            underlying_symbol = str(row.get("underlying_symbol") or "")
            if not underlying_symbol:
                self._quarantine_row(
                    "options_chain_surface_nodes", row, "underlying_symbol",
                    "missing underlying symbol",
                )
                continue
            run_identity = str(
                row.get("authoritative_surface_run_id") or row.get("surface_run_id")
            )
            if (
                sid, expiration_at, str(row.get("price_basis") or "mid"), run_identity
            ) not in successful_fit_keys:
                self._quarantine_row(
                    "options_chain_surface_nodes", row, "parent_fit",
                    "node has no successful fit in the selected surface run",
                )
                continue
            required = is_required_surface_node(node_type, node_value)
            if required:
                required_nodes_by_snapshot[sid] = (
                    required_nodes_by_snapshot.get(sid, 0) + 1
                )
            success_row: Optional[dict[str, Any]] = None
            success_error: Optional[Exception] = None
            if status == "complete":
                try:
                    success_row = {
                        "snapshot_id": str(row["snapshot_id"]),
                        "source_collected_at": source_collected_at,
                        "underlying_symbol": underlying_symbol,
                        "expiration_at": expiration_at,
                        "price_basis": str(row.get("price_basis") or "mid"),
                        "surface_definition_id": definitions["surface"],
                        "source_surface_run_id": str(
                            row.get("authoritative_surface_run_id")
                            or row.get("surface_run_id")
                        ),
                        "node_type": node_type,
                        "node_value": node_value,
                        "put_call": put_call,
                        "fitted_strike": checked_float(
                            row.get("fitted_strike"), "fitted_strike"
                        ),
                        "fitted_delta": checked_float(
                            row.get("fitted_delta"), "fitted_delta"
                        ),
                        "fitted_iv_bid": checked_float(
                            row.get("fitted_iv_bid"), "fitted_iv_bid", nullable=True
                        ),
                        "fitted_iv_mid": checked_float(
                            row.get("fitted_iv_mid"), "fitted_iv_mid"
                        ),
                        "fitted_iv_ask": checked_float(
                            row.get("fitted_iv_ask"), "fitted_iv_ask", nullable=True
                        ),
                        "fit_confidence": checked_float(
                            row.get("fit_confidence"), "fit_confidence"
                        ),
                        "quality_flags": checked_uint(
                            row.get("quality_flags"), "quality_flags"
                        ),
                        "calculated_at": _parse_timestamp(row.get("calculated_at")),
                    }
                    if success_row["calculated_at"] is None:
                        raise ValueError("successful node has an invalid calculated_at")
                except (ValueError, TypeError, OverflowError) as exc:
                    success_row = None
                    success_error = exc
            if success_row is not None:
                node_rows.append(success_row)
                node_successes += 1
                success_counts = (
                    required_node_successes_by_snapshot
                    if required
                    else optional_node_successes_by_snapshot
                )
                success_counts[sid] = success_counts.get(sid, 0) + 1
                if len(node_rows) >= 25_000:
                    flush(node_target, node_rows, node_insert_schema)
                continue
            now = _parse_timestamp(row.get("calculated_at")) or _utc_now()
            reason = str(
                row.get("node_reason")
                or (
                    f"invalid_node_success:{success_error}"
                    if success_error is not None
                    else status or "invalid_node_success"
                )
            )
            failure_rows.append(
                {
                    "snapshot_id": str(row["snapshot_id"]),
                    "source_collected_at": source_collected_at,
                    "underlying_symbol": underlying_symbol,
                    "surface_definition_id": definitions["surface"],
                    "source_surface_run_id": str(
                        row.get("authoritative_surface_run_id")
                        or row.get("surface_run_id")
                    ),
                    "failure_scope": "node",
                    "expiration_at": expiration_at,
                    "price_basis": str(row.get("price_basis") or "mid"),
                    "node_type": node_type,
                    "node_value": node_value,
                    "put_call": put_call,
                    "failure_stage": "node_generation",
                    "reason_code": reason,
                    "error_class": "LegacyNodeFailure",
                    "error_detail": _bounded_text(reason),
                    "retry_class": "deterministic",
                    "retry_after": None,
                    "failure_fingerprint": hashlib.sha256(
                        f"node|{row.get('snapshot_id')}|{row.get('expiration_at')}|"
                        f"{row.get('node_type')}|{row.get('node_value')}|"
                        f"{row.get('put_call')}|{reason}".encode()
                    ).digest(),
                    "input_contract_count": None,
                    "accepted_contract_count": None,
                    "rejected_contract_count": None,
                    "first_failed_at": now,
                    "last_failed_at": now,
                    "failure_count": 1,
                    "updated_at": now,
                    "quality_flags": failure_uint(row.get("quality_flags")) or 0,
                    "is_active": True,
                }
            )
            node_failures += 1
            failure_counts = (
                required_node_failures_by_snapshot
                if required
                else optional_node_failures_by_snapshot
            )
            failure_counts[sid] = failure_counts.get(sid, 0) + 1
            if len(failure_rows) >= 25_000:
                flush(
                    failure_target,
                    failure_rows,
                    failure_insert_schema,
                )
        flush(node_target, node_rows, node_insert_schema)
        flush(failure_target, failure_rows, failure_insert_schema)

        # One current authoritative surface state per snapshot/definition.  A
        # failed or unfinished run is still a state row; omission must never
        # turn it into fresh pending work.
        states = []
        run_source_sql = (
            f"SELECT * EXCEPT(surface_run_id, snapshot_id, "
            f"source_calculation_run_id, status), "
            f"toString(surface_run_id) AS surface_run_id, "
            f"toString(snapshot_id) AS snapshot_id, "
            f"toString(source_calculation_run_id) AS source_calculation_run_id, "
            f"toString(status) AS status FROM "
            f"{self.database}.options_chain_surface_runs_latest "
            "ORDER BY source_collected_at, snapshot_id"
        )
        for run in _iter_arrow_rows(self.client, run_source_sql):
            sid = str(run["snapshot_id"])
            now = _parse_timestamp(run.get("finished_at")) or _utc_now()
            fit_failure_count = len(fit_failures_by_snapshot.get(sid, set()))
            required_node_failure_count = required_node_failures_by_snapshot.get(
                sid, 0
            )
            optional_node_failure_count = optional_node_failures_by_snapshot.get(
                sid, 0
            )
            run_status = classify_legacy_run_status(
                run.get("status"), run.get("started_at"),
                stale_after_seconds=int(
                    self.context.get(
                        "options_chain.migration.surface_stale_seconds", 3600
                    )
                ),
            )
            if run_status in {"complete", "complete_warning", "complete_with_failures"}:
                state_status = (
                    "complete_with_failures"
                    if fit_failure_count or required_node_failure_count
                    or optional_node_failure_count
                    or int(run.get("failed_expirations") or 0)
                    else "complete"
                )
            else:
                state_status = run_status
            states.append(
                {
                    "snapshot_id": sid,
                    "source_collected_at": _parse_timestamp(
                        run["source_collected_at"]
                    ),
                    "underlying_symbol": str(run["underlying_symbol"]),
                    "surface_definition_id": definitions["surface"],
                    "source_calculation_definition_id": definitions["calculation"],
                    "source_surface_run_id": str(run.get("surface_run_id")),
                    "status": state_status,
                    "expected_expirations": int(
                        run.get("expected_expirations") or 0
                    ),
                    "successful_expirations": len(
                        fits_by_snapshot.get(sid, set())
                    ),
                    "source_only_or_skipped_expirations": int(
                        run.get("skipped_expirations")
                        if run.get("skipped_expirations") is not None
                        else len(skipped_by_snapshot.get(sid, set()))
                    ),
                    "terminal_fit_failures": max(
                        fit_failure_count, int(run.get("failed_expirations") or 0)
                    ),
                    "expected_required_nodes": required_nodes_by_snapshot.get(
                        sid, 0
                    ),
                    "successful_required_nodes": required_node_successes_by_snapshot.get(
                        sid, 0
                    ),
                    "terminal_required_node_failures": required_node_failure_count,
                    "successful_optional_nodes": optional_node_successes_by_snapshot.get(
                        sid, 0
                    ),
                    "terminal_optional_node_failures": optional_node_failure_count,
                    "started_at": _parse_timestamp(run.get("started_at")),
                    "finished_at": now,
                    "updated_at": now,
                    "active_attempt_id": None,
                    "error_summary": "",
                    "is_active": True,
                }
            )
            if len(states) >= 25_000:
                flush(state_target, states, state_insert_schema)
        flush(state_target, states, state_insert_schema)
        self._migrate_recent_attempts(definitions, include_surface=True)
        self._migrate_recent_validations(definitions, include_surface=True)
        self._checkpoint(
            "migrate_surfaces",
            MigrationPhase.SURFACES,
            validation_summary={
                **self.checkpoint.validation_summary,
                "surface_typed_parse_successes": successes,
                "surface_typed_parse_failures": failures,
                "surface_node_successes": node_successes,
                "surface_failure_rows": failures + node_failures,
                "migration_quarantine_rows": self._quarantine_rows,
            },
        )

    def archive_raw(self) -> None:
        self._ensure_lease(MigrationPhase.RAW_ARCHIVED)
        archive_context = dict(self.context)
        archive_context["options_chain.archive.output_directory"] = str(
            self.archive_directory
        )
        archive_context["options_chain.dataset_generation_id"] = str(
            uuid.uuid5(self.migration_id, "dataset-generation")
        )
        config = None
        archiver = None
        if not self.dry_run:
            config = ArchiveConfig.from_context(archive_context)
            archiver = RawWeekArchiver(
                archive_context,
                self.client,
                config,
                logger=self.logger,
                table_names={
                    name: target_name(name, self.migration_id)
                    for name in CANONICAL_TABLE_NAMES
                },
            )
        rows = self.client.query(
            f"SELECT DISTINCT toMonday(toDate(collected_at_utc)) "
            f"FROM {self.database}.options_chain_raw ORDER BY 1"
        ).result_rows
        if self.dry_run:
            minimum_age_days = int(
                archive_context.get("options_chain.archive.minimum_age_days", 14)
            )
            retain_raw_weeks = int(
                archive_context.get("options_chain.archive.retain_raw_weeks", 2)
            )
            estimates = []
            for (week,) in rows:
                if week_is_age_eligible(
                    week, _utc_now(), minimum_age_days=minimum_age_days,
                    retain_raw_weeks=retain_raw_weeks,
                ):
                    sql = (
                        f"SELECT count(), sum(length(response_json)) FROM "
                        f"{self.database}.options_chain_raw "
                        "WHERE toMonday(toDate(collected_at_utc)) = {week:Date}"
                    )
                    self.planned_sql.append(sql)
                    try:
                        values = self.client.query(
                            sql, parameters={"week": week}
                        ).result_rows[0]
                    except Exception:
                        values = (0, 0)
                    estimates.append({
                        "week": str(week), "rows": int(values[0] or 0),
                        "estimated_bytes": int(values[1] or 0),
                    })
            self.logger.info("migration_archive_plan weeks=%s estimates=%s", len(estimates), estimates)
            return
        for (week,) in rows:
            if not week_is_age_eligible(
                week,
                _utc_now(),
                minimum_age_days=config.minimum_age_days,
                retain_raw_weeks=config.retain_raw_weeks,
            ):
                continue
            assert archiver is not None and config is not None
            archiver.archive_week(
                week,
                dry_run=self.dry_run,
                purge=True,
                confirm_database=self.database,
            )
        self._checkpoint("archive_raw", MigrationPhase.RAW_ARCHIVE)

    def migrate_final_raw_delta(self) -> int:
        """Copy the stopped capture-only raw tail without fabricating normalized rows."""
        self._ensure_lease(self.checkpoint.phase)
        if self.dry_run:
            sql = f"SELECT count() FROM {self.database}.options_chain_raw"
            self.planned_sql.append(sql)
            try:
                return self._scalar(sql)
            except Exception:
                return 0
        raw_target = target_name("options_chain_raw", self.migration_id)
        high_water = _parse_timestamp(
            self.checkpoint.validation_summary.get("initial_raw_high_water")
        )
        high_water_snapshot_ids = [
            str(value)
            for value in self.checkpoint.validation_summary.get(
                "initial_raw_high_water_snapshot_ids", []
            )
        ]
        if high_water is None:
            delta_predicate = "1"
            delta_parameters: dict[str, Any] = {}
        else:
            high_water_sql = (
                "parseDateTime64BestEffort({initial_raw_high_water:String}, 6, 'UTC')"
            )
            high_water_parameter = high_water.isoformat(timespec="microseconds")
            # Recover checkpoints created by clients that truncated DateTime64
            # parameters while discovering the boundary IDs.  Target rows at
            # the exact initial high-water mark are authoritative members of
            # the original capture boundary.
            recovered_boundary_ids = [
                str(row[0])
                for row in self.client.query(
                    f"SELECT toString(snapshot_id) FROM {self.database}.{raw_target} "
                    f"WHERE collected_at_utc = {high_water_sql} ORDER BY snapshot_id",
                    parameters={"initial_raw_high_water": high_water_parameter},
                ).result_rows
            ]
            high_water_snapshot_ids = sorted(
                set(high_water_snapshot_ids).union(recovered_boundary_ids)
            )
            delta_predicate = (
                f"(r.collected_at_utc > {high_water_sql} OR "
                f"(r.collected_at_utc = {high_water_sql} AND "
                "r.snapshot_id NOT IN {initial_high_water_ids:Array(UUID)}))"
            )
            delta_parameters = {
                "initial_raw_high_water": high_water_parameter,
                "initial_high_water_ids": high_water_snapshot_ids,
            }
        delta = self._scalar(
            f"SELECT count() FROM {self.database}.options_chain_raw AS r "
            f"LEFT JOIN {self.database}.{raw_target} AS t ON t.snapshot_id = r.snapshot_id "
            f"WHERE {delta_predicate} AND isNull(t.snapshot_id)",
            parameters=delta_parameters,
        )
        normalized_delta = self._scalar(
            f"SELECT count() FROM {self.database}.options_chain_raw AS r "
            f"LEFT JOIN {self.database}.{raw_target} AS t ON t.snapshot_id = r.snapshot_id "
            f"INNER JOIN {self.database}.options_chain_base AS b ON b.snapshot_id = r.snapshot_id "
            f"WHERE {delta_predicate} AND isNull(t.snapshot_id)",
            parameters=delta_parameters,
        )
        if normalized_delta:
            raise MigrationError(
                f"Final raw delta contains {normalized_delta} legacy-normalized snapshots"
            )
        if delta:
            self._command(
                f"""INSERT INTO {self.database}.{raw_target}
                (snapshot_id, source, symbol, request_started_at_utc,
                 response_received_at_utc, collected_at_utc, response_latency_ms,
                 request_parameters_json, response_json, payload_sha256,
                 source_contract_count, is_delayed, is_chain_truncated, status)
                SELECT r.snapshot_id, r.source, r.symbol, r.request_started_at_utc,
                       r.response_received_at_utc, r.collected_at_utc,
                       r.response_latency_ms, r.request_parameters_json,
                       r.response_json, unhex(r.payload_sha256_hex),
                       r.source_contract_count, r.is_delayed,
                       r.is_chain_truncated, r.status
                FROM {self.database}.options_chain_raw AS r
                LEFT JOIN {self.database}.{raw_target} AS t
                  ON t.snapshot_id = r.snapshot_id
                WHERE {delta_predicate} AND isNull(t.snapshot_id)""",
                parameters=delta_parameters,
            )
        remaining = self._scalar(
            f"SELECT count() FROM {self.database}.options_chain_raw AS r "
            f"LEFT JOIN {self.database}.{raw_target} AS t ON t.snapshot_id = r.snapshot_id "
            f"WHERE {delta_predicate} AND isNull(t.snapshot_id)",
            parameters=delta_parameters,
        )
        if remaining:
            raise MigrationError(f"Final raw delta still has {remaining} missing rows")
        total_delta = self._scalar(
            f"SELECT count() FROM {self.database}.options_chain_raw AS r "
            f"WHERE {delta_predicate}",
            parameters=delta_parameters,
        )
        self._checkpoint(
            "final_raw_delta",
            self.checkpoint.phase,
            validation_summary={
                **self.checkpoint.validation_summary,
                "initial_raw_high_water_snapshot_ids": high_water_snapshot_ids,
                "final_raw_delta_rows": total_delta,
                "pending_normalization_after_cutover": total_delta,
            },
        )
        self.logger.info(
            "migration_capture_delta migration_id=%s database=%s rows=%s "
            "initial_high_water=%s remaining=%s total_delta=%s",
            self.migration_id,
            self.database,
            delta,
            high_water,
            remaining,
            total_delta,
        )
        return delta

    def _object_storage_report(self, name: str) -> Mapping[str, Any]:
        rows = self.client.query(
            "SELECT toString(uuid), create_table_query, total_rows, total_bytes, "
            "sorting_key, partition_key, primary_key "
            "FROM system.tables WHERE database = {database:String} AND name = {name:String}",
            parameters={"database": self.database, "name": name},
        ).result_rows
        if not rows:
            raise MigrationError(f"Storage report object is absent: {name}")
        object_uuid, create_sql, total_rows, total_bytes, order_key, partition_key, primary_key = rows[0]
        part_rows = self.client.query(
            "SELECT coalesce(sum(rows), 0), coalesce(sum(data_compressed_bytes), 0), "
            "coalesce(sum(data_uncompressed_bytes), 0), uniqExact(partition_id) "
            "FROM system.parts WHERE active AND database = {database:String} "
            "AND table = {name:String}",
            parameters={"database": self.database, "name": name},
        ).result_rows
        stored_rows, compressed, uncompressed, partitions = (
            part_rows[0] if part_rows else (0, 0, 0, 0)
        )
        compressed = int(compressed or total_bytes or 0)
        uncompressed = int(uncompressed or 0)
        create_text = str(create_sql)
        ttl = ""
        if " TTL " in " ".join(create_text.split()):
            ttl = " ".join(create_text.split()).split(" TTL ", 1)[1].split(" SETTINGS ", 1)[0]
        return {
            "uuid": str(object_uuid),
            "rows": int(stored_rows or total_rows or 0),
            "compressed_bytes": compressed,
            "uncompressed_bytes": uncompressed,
            "compression_ratio": (
                float(uncompressed) / compressed if compressed else None
            ),
            "partition_count": int(partitions or 0),
            "order_expression": str(order_key or ""),
            "partition_expression": str(partition_key or ""),
            "primary_key": str(primary_key or ""),
            "ttl": ttl,
            "create_sql": create_text,
        }

    def _scalar(
        self, sql: str, *, parameters: Optional[Mapping[str, Any]] = None
    ) -> int:
        result = (
            self.client.query(sql)
            if parameters is None
            else self.client.query(sql, parameters=parameters)
        )
        rows = result.result_rows
        if not rows:
            raise MigrationError("Validation query returned no scalar row")
        return int(rows[0][0] or 0)

    def _dry_run_validation(self) -> Mapping[str, Any]:
        """Validate prerequisites and render planning metadata without targets."""
        expected: dict[str, int] = {}
        source_queries = {
            "raw_rows": f"SELECT count() FROM {self.database}.options_chain_raw",
            "source_rows": f"SELECT count() FROM {self.database}.options_chain_base",
        }
        for name, sql in source_queries.items():
            try:
                expected[name] = self._scalar(sql)
            except Exception as exc:
                expected[name] = -1
                self.logger.warning("dry-run count failed name=%s error=%s", name, exc)
        report = {
            "status": "planned",
            "migration_id": str(self.migration_id),
            "database": self.database,
            "phase": self.checkpoint.phase.value,
            "expected_rows": expected,
            "planned_sql_count": len(self.planned_sql),
            "planned_sql": list(self.planned_sql),
            "archive_space": dict(self.archive_space_report()),
            "database_changes_made": False,
            "errors": [],
        }
        if not self.archive_space_report().get("creatable", False):
            report["errors"].append(
                f"archive directory is not creatable: {self.archive_directory}"
            )
        return report

    def validate(self) -> Mapping[str, Any]:
        if self.dry_run:
            return self._dry_run_validation()
        self._ensure_lease(MigrationPhase.VALIDATED)
        summary: dict[str, Any] = {
            "schema_generation": SCHEMA_GENERATION,
            "definition_ids": dict(
                self.checkpoint.validation_summary.get("definition_ids", {})
            ),
            "source_range": dict(
                self.checkpoint.validation_summary.get("source_range", {})
            ),
            "initial_raw_count": int(
                self.checkpoint.validation_summary.get("initial_raw_count", 0)
            ),
            "initial_raw_high_water": self.checkpoint.validation_summary.get(
                "initial_raw_high_water"
            ),
            "initial_raw_high_water_snapshot_ids": list(
                self.checkpoint.validation_summary.get(
                    "initial_raw_high_water_snapshot_ids", []
                )
            ),
            "final_raw_delta_rows": int(
                self.checkpoint.validation_summary.get("final_raw_delta_rows", 0)
            ),
            "source_integer_ranges": dict(
                self.checkpoint.validation_summary.get("source_integer_ranges", {})
            ),
            "source_float32_cast_errors": dict(
                self.checkpoint.validation_summary.get(
                    "source_float32_cast_errors", {}
                )
            ),
            "source_nonfinite_counts": dict(
                self.checkpoint.validation_summary.get(
                    "source_nonfinite_counts", {}
                )
            ),
            "source_null_counts": dict(
                self.checkpoint.validation_summary.get("source_null_counts", {})
            ),
            "calculation_integer_ranges": dict(
                self.checkpoint.validation_summary.get(
                    "calculation_integer_ranges", {}
                )
            ),
            "surface_typed_parse_successes": int(
                self.checkpoint.validation_summary.get(
                    "surface_typed_parse_successes", 0
                )
            ),
            "surface_typed_parse_failures": int(
                self.checkpoint.validation_summary.get(
                    "surface_typed_parse_failures", 0
                )
            ),
            "surface_node_successes": int(
                self.checkpoint.validation_summary.get(
                    "surface_node_successes", 0
                )
            ),
            "surface_failure_rows": int(
                self.checkpoint.validation_summary.get(
                    "surface_failure_rows", 0
                )
            ),
            "legacy_view_ddl": dict(
                self.checkpoint.validation_summary.get("legacy_view_ddl", {})
            ),
            "legacy_only_objects": dict(
                self.checkpoint.validation_summary.get("legacy_only_objects", {})
            ),
            "tables": {},
            "legacy_tables": {},
            "count_parity": {},
            "errors": [],
        }
        for item in self.checkpoint.objects:
            try:
                report = dict(self._object_storage_report(item.target_name))
            except MigrationError:
                summary["errors"].append(f"missing target {item.target_name}")
                continue
            normalized = canonicalize_created_table_name(
                str(report.pop("create_sql")),
                database=self.database,
                actual_name=item.target_name,
                canonical_name=item.canonical_name,
            )
            if schema_hash(normalized) != item.target_schema_hash:
                summary["errors"].append(
                    f"schema hash mismatch {item.target_name}:\n"
                    + schema_mismatch_details(
                        CANONICAL_SCHEMA[item.canonical_name].ddl.format(
                            database=self.database
                        ),
                        normalized,
                    )
                )
            summary["tables"][item.canonical_name] = report
        for name in LEGACY_TABLES:
            exists = self.client.query(
                "SELECT count() FROM system.tables WHERE database = {database:String} "
                "AND name = {name:String}",
                parameters={"database": self.database, "name": name},
            ).result_rows[0][0]
            if int(exists):
                report = dict(self._object_storage_report(name))
                report.pop("create_sql", None)
                summary["legacy_tables"][name] = report

        parity_queries = {
            "options_chain_snapshots": (
                f"SELECT uniqExact(snapshot_id) FROM {self.database}.options_chain_base"
            ),
            "options_contracts": (
                f"SELECT uniqExact(tuple(underlying_symbol, option_symbol)) "
                f"FROM {self.database}.options_chain_base"
            ),
            "options_chain_observations": (
                f"SELECT count() FROM {self.database}.options_chain_base"
            ),
            "options_chain_observation_details": (
                f"SELECT count() FROM {self.database}.options_chain_base"
            ),
            "options_chain_reference_rates": (
                f"SELECT count() FROM (SELECT series_id, observation_date FROM "
                f"{self.database}.options_chain_reference_rates "
                "GROUP BY series_id, observation_date)"
            ),
            "options_chain_calculation_state": (
                f"SELECT count() FROM {self.database}.options_chain_calculation_runs_latest"
            ),
            "options_chain_calculation_results": (
                f"SELECT count() FROM {self.database}.options_chain_calculations_latest AS c "
                f"INNER JOIN {self.database}.options_chain_calculation_runs_latest AS r "
                "ON r.snapshot_id = c.snapshot_id AND r.calculation_version = c.calculation_version "
                f"WHERE r.status IN ('complete', 'complete_with_failures') "
                f"AND {_legacy_success_predicate('c')}"
            ),
            "options_chain_surface_state": (
                f"SELECT count() FROM {self.database}.options_chain_surface_runs_latest"
            ),
        }
        for canonical, source_sql in parity_queries.items():
            expected = self._scalar(source_sql)
            actual = int(summary["tables"].get(canonical, {}).get("rows", 0))
            summary["count_parity"][canonical] = {
                "legacy_selected": expected,
                "target": actual,
                "matches": expected == actual,
            }
            if expected != actual:
                summary["errors"].append(
                    f"row-count parity failed for {canonical}: {expected} != {actual}"
                )
        for canonical, expected_key in (
            ("options_chain_surface_fits", "surface_typed_parse_successes"),
            ("options_chain_surface_nodes", "surface_node_successes"),
            ("options_chain_surface_failures", "surface_failure_rows"),
        ):
            expected = int(summary[expected_key])
            actual = int(summary["tables"].get(canonical, {}).get("rows", 0))
            summary["count_parity"][canonical] = {
                "migration_selected": expected,
                "target": actual,
                "matches": expected == actual,
            }
            if expected != actual:
                summary["errors"].append(
                    f"row-count parity failed for {canonical}: {expected} != {actual}"
                )
        raw_target = target_name("options_chain_raw", self.migration_id)
        archive_target = target_name(
            "options_chain_archive_manifest", self.migration_id
        )
        raw_working = self._scalar(
            f"SELECT count() FROM {self.database}.{raw_target}"
        )
        raw_archived = self._scalar(
            f"SELECT coalesce(sum(row_count), 0) FROM "
            f"{self.database}.{archive_target} FINAL "
            "WHERE archive_kind = 'raw_week' AND state IN ('verified', 'purged')"
        )
        raw_expected = int(summary["initial_raw_count"]) + int(
            summary["final_raw_delta_rows"]
        )
        legacy_raw_current = self._scalar(
            f"SELECT count() FROM {self.database}.options_chain_raw"
        )
        raw_coverage = raw_working + raw_archived
        summary["count_parity"]["options_chain_raw"] = {
            "legacy_initial": int(summary["initial_raw_count"]),
            "capture_only_delta": int(summary["final_raw_delta_rows"]),
            "expected_coverage": raw_expected,
            "legacy_current": legacy_raw_current,
            "target_working": raw_working,
            "verified_archived": raw_archived,
            "target": raw_coverage,
            "matches": raw_expected == raw_coverage,
        }
        if raw_expected != raw_coverage:
            summary["errors"].append(
                "raw working-plus-archive parity failed: "
                f"{raw_expected} != {raw_working} + {raw_archived}"
            )
        if raw_expected != legacy_raw_current:
            summary["errors"].append(
                "legacy raw changed outside the captured final-delta boundary: "
                f"{raw_expected} != {legacy_raw_current}"
            )
        # Hard correctness guards that must hold before a resumable checkpoint
        # may become cutover-eligible.
        for canonical, natural_key in (
            ("options_chain_snapshots", "snapshot_id"),
            ("options_chain_observations", "snapshot_id, contract_id"),
            ("options_chain_observation_details", "snapshot_id, contract_id"),
            ("options_chain_expiration_inputs", "snapshot_id, expiration_at"),
            ("options_chain_calculation_results", "snapshot_id, contract_id"),
            (
                "options_chain_calculation_reuse_provenance",
                "snapshot_id, contract_id",
            ),
            (
                "options_chain_surface_fits",
                "snapshot_id, expiration_at, price_basis",
            ),
            (
                "options_chain_surface_nodes",
                "snapshot_id, expiration_at, price_basis, node_type, "
                "node_value, put_call",
            ),
        ):
            target = target_name(canonical, self.migration_id)
            duplicate_count = self.client.query(
                f"SELECT count() FROM (SELECT {natural_key} FROM {self.database}.{target} "
                f"GROUP BY {natural_key} HAVING count() != 1)"
            ).result_rows[0][0]
            if int(duplicate_count):
                summary["errors"].append(
                    f"{canonical} has {duplicate_count} duplicate natural keys"
                )
        for canonical, definition_column in (
            ("options_chain_snapshots", "collection_definition_id"),
            ("options_chain_snapshots", "normalization_definition_id"),
            ("options_chain_calculation_results", "calculation_definition_id"),
            ("options_chain_surface_fits", "surface_definition_id"),
        ):
            target = target_name(canonical, self.migration_id)
            zeros = self.client.query(
                f"SELECT count() FROM {self.database}.{target} "
                f"WHERE {definition_column} = 0"
            ).result_rows[0][0]
            if int(zeros):
                summary["errors"].append(
                    f"{canonical}.{definition_column} has {zeros} unresolved rows"
                )
        legacy_range = self.client.query(
            f"SELECT min(collected_at), max(collected_at) "
            f"FROM {self.database}.options_chain_base"
        ).result_rows[0]
        target_range = self.client.query(
            f"SELECT min(source_collected_at), max(source_collected_at) "
            f"FROM {self.database}.{target_name('options_chain_snapshots', self.migration_id)}"
        ).result_rows[0]
        legacy_parsed_range = tuple(_parse_timestamp(value) for value in legacy_range)
        target_parsed_range = tuple(_parse_timestamp(value) for value in target_range)
        range_matches = legacy_parsed_range == target_parsed_range
        summary["timestamp_coverage"] = {
            "legacy": [value.isoformat() if value else None for value in legacy_parsed_range],
            "target": [value.isoformat() if value else None for value in target_parsed_range],
            "matches": range_matches,
        }
        if not range_matches:
            summary["errors"].append(
                "snapshot timestamp-range coverage differs between legacy and target"
            )
        legacy_months = {
            int(row[0])
            for row in self.client.query(
                f"SELECT DISTINCT toYYYYMM(collected_at) "
                f"FROM {self.database}.options_chain_base"
            ).result_rows
        }
        target_months = {
            int(row[0])
            for row in self.client.query(
                f"SELECT DISTINCT toYYYYMM(source_collected_at) FROM "
                f"{self.database}.{target_name('options_chain_snapshots', self.migration_id)}"
            ).result_rows
        }
        summary["source_month_coverage"] = {
            "legacy": sorted(legacy_months),
            "target": sorted(target_months),
            "matches": legacy_months == target_months,
        }
        if legacy_months != target_months:
            summary["errors"].append(
                "source-month coverage differs between legacy and target"
            )
        target_null_counts: dict[str, int] = {}
        null_columns_by_table: dict[str, list[tuple[str, str]]] = {}
        for source_column, (canonical, target_column) in SOURCE_NULLABLE_TARGETS.items():
            null_columns_by_table.setdefault(canonical, []).append(
                (source_column, target_column)
            )
        for canonical, mappings in null_columns_by_table.items():
            row = self.client.query(
                "SELECT "
                + ", ".join(
                    f"countIf(isNull({target_column}))"
                    for _source_column, target_column in mappings
                )
                + f" FROM {self.database}.{target_name(canonical, self.migration_id)}"
            ).result_rows[0]
            for (source_column, _target_column), value in zip(mappings, row):
                target_null_counts[source_column] = int(value or 0)
        null_mismatches = {
            name: {
                "legacy": int(expected),
                "target": int(target_null_counts.get(name, -1)),
            }
            for name, expected in summary["source_null_counts"].items()
            if name in SOURCE_NULLABLE_TARGETS
            and int(expected) != int(target_null_counts.get(name, -1))
        }
        summary["target_null_counts"] = target_null_counts
        summary["normalized_nonnullable_source_null_counts"] = {
            name: int(value)
            for name, value in summary["source_null_counts"].items()
            if name not in SOURCE_NULLABLE_TARGETS
        }
        summary["null_count_mismatches"] = null_mismatches
        if null_mismatches:
            summary["errors"].append(
                f"retained source null-count parity failed: {null_mismatches}"
            )
        def target(name: str) -> str:
            return f"{self.database}.{target_name(name, self.migration_id)}"
        integrity_queries = {
            "snapshot_reconciliation": f"""
                SELECT count() FROM (
                  SELECT s.snapshot_id
                  FROM {target('options_chain_snapshots')} AS s
                  LEFT JOIN {target('options_chain_observations')} AS o
                    ON o.snapshot_id = s.snapshot_id
                  LEFT JOIN {target('options_chain_observation_details')} AS d
                    ON d.snapshot_id = s.snapshot_id
                  GROUP BY s.snapshot_id, s.stored_observation_count,
                           s.stored_detail_count, s.stored_distinct_contract_count
                  HAVING countDistinct(o.contract_id) != s.stored_distinct_contract_count
                     OR countDistinct(o.contract_id) != s.stored_observation_count
                     OR countDistinct(d.contract_id) != s.stored_detail_count
                )""",
            "observations_without_snapshots": f"""
                SELECT count() FROM {target('options_chain_observations')} AS o
                LEFT JOIN {target('options_chain_snapshots')} AS s
                  ON s.snapshot_id = o.snapshot_id
                 AND s.source_collected_at = o.source_collected_at
                WHERE isNull(s.snapshot_id)""",
            "missing_contracts": f"""
                SELECT count() FROM {target('options_chain_observations')} AS o
                LEFT JOIN (SELECT * FROM {target('options_contracts')} FINAL) AS c
                  ON c.contract_id = o.contract_id
                WHERE isNull(c.contract_id)""",
            "results_without_expiration_inputs": f"""
                SELECT count() FROM {target('options_chain_calculation_results')} AS r
                INNER JOIN (SELECT * FROM {target('options_contracts')} FINAL) AS c
                  ON c.contract_id = r.contract_id
                LEFT JOIN {target('options_chain_expiration_inputs')} AS e
                  ON e.snapshot_id = r.snapshot_id
                 AND e.source_collected_at = r.source_collected_at
                 AND e.expiration_at = c.expiration_at
                WHERE isNull(e.snapshot_id)""",
            "reused_without_provenance": f"""
                SELECT count() FROM {target('options_chain_calculation_results')} AS r
                LEFT JOIN {target('options_chain_calculation_reuse_provenance')} AS p
                  ON p.snapshot_id = r.snapshot_id AND p.contract_id = r.contract_id
                WHERE r.result_origin = 'reused' AND isNull(p.contract_id)""",
            "calculation_state_reconciliation": f"""
                SELECT count() FROM (
                  SELECT s.snapshot_id, s.successful_results,
                         s.terminal_contract_failures,
                         s.terminal_expiration_failures,
                         s.snapshot_failure_count,
                         coalesce(r.result_count, 0) AS result_count,
                         coalesce(f.contract_failures, 0) AS contract_failures,
                         coalesce(f.expiration_failures, 0) AS expiration_failures,
                         coalesce(f.snapshot_failures, 0) AS snapshot_failures
                  FROM (SELECT * FROM {target('options_chain_calculation_state')} FINAL) AS s
                  LEFT JOIN (
                    SELECT snapshot_id, count() AS result_count
                    FROM {target('options_chain_calculation_results')}
                    GROUP BY snapshot_id
                  ) AS r ON r.snapshot_id = s.snapshot_id
                  LEFT JOIN (
                    SELECT snapshot_id,
                           countIf(failure_scope = 'contract' AND is_active)
                             AS contract_failures,
                           countIf(failure_scope = 'expiration' AND is_active)
                             AS expiration_failures,
                           countIf(failure_scope = 'snapshot' AND is_active)
                             AS snapshot_failures
                    FROM {target('options_chain_calculation_failures')} FINAL
                    GROUP BY snapshot_id
                  ) AS f ON f.snapshot_id = s.snapshot_id
                  WHERE s.is_active AND s.status IN ('complete', 'complete_with_failures')
                    AND (s.successful_results != result_count
                      OR s.terminal_contract_failures != contract_failures
                      OR s.terminal_expiration_failures != expiration_failures
                      OR s.snapshot_failure_count != snapshot_failures)
                )""",
            "surface_state_reconciliation": f"""
                SELECT count() FROM (
                  SELECT s.snapshot_id, s.successful_expirations,
                         s.terminal_fit_failures,
                         s.successful_required_nodes,
                         s.terminal_required_node_failures,
                         s.successful_optional_nodes,
                         s.terminal_optional_node_failures,
                         coalesce(f.successful_expirations, 0) AS actual_expirations,
                         coalesce(x.fit_failures, 0) AS actual_fit_failures,
                         coalesce(n.required_nodes, 0) AS actual_required_nodes,
                         coalesce(x.required_node_failures, 0)
                           AS actual_required_node_failures,
                         coalesce(n.optional_nodes, 0) AS actual_optional_nodes,
                         coalesce(x.optional_node_failures, 0)
                           AS actual_optional_node_failures
                  FROM (SELECT * FROM {target('options_chain_surface_state')} FINAL) AS s
                  LEFT JOIN (
                    SELECT snapshot_id,
                           uniqExactIf(expiration_at, price_basis IN ('mid', 'source'))
                             AS successful_expirations
                    FROM {target('options_chain_surface_fits')}
                    GROUP BY snapshot_id
                  ) AS f ON f.snapshot_id = s.snapshot_id
                  LEFT JOIN (
                    SELECT snapshot_id,
                           countIf(node_type = 'atm_forward' OR
                             (node_type = 'delta' AND
                              (abs(node_value - 0.10) <= 1e-12 OR
                               abs(node_value - 0.25) <= 1e-12 OR
                               abs(node_value - 0.40) <= 1e-12))) AS required_nodes,
                           countIf(NOT (node_type = 'atm_forward' OR
                             (node_type = 'delta' AND
                              (abs(node_value - 0.10) <= 1e-12 OR
                               abs(node_value - 0.25) <= 1e-12 OR
                               abs(node_value - 0.40) <= 1e-12)))) AS optional_nodes
                    FROM {target('options_chain_surface_nodes')}
                    GROUP BY snapshot_id
                  ) AS n ON n.snapshot_id = s.snapshot_id
                  LEFT JOIN (
                    SELECT snapshot_id,
                           uniqExactIf(expiration_at,
                             failure_scope = 'fit' AND is_active
                             AND price_basis IN ('mid', 'source'))
                             AS fit_failures,
                           countIf(failure_scope = 'node' AND is_active AND
                             (node_type = 'atm_forward' OR
                              (node_type = 'delta' AND
                               (abs(node_value - 0.10) <= 1e-12 OR
                                abs(node_value - 0.25) <= 1e-12 OR
                                abs(node_value - 0.40) <= 1e-12))))
                             AS required_node_failures,
                           countIf(failure_scope = 'node' AND is_active AND
                             NOT (node_type = 'atm_forward' OR
                              (node_type = 'delta' AND
                               (abs(node_value - 0.10) <= 1e-12 OR
                                abs(node_value - 0.25) <= 1e-12 OR
                                abs(node_value - 0.40) <= 1e-12))))
                             AS optional_node_failures
                    FROM {target('options_chain_surface_failures')} FINAL
                    GROUP BY snapshot_id
                  ) AS x ON x.snapshot_id = s.snapshot_id
                  WHERE s.is_active AND s.status IN ('complete', 'complete_with_failures')
                    AND (s.successful_expirations != actual_expirations
                      OR s.terminal_fit_failures != actual_fit_failures
                      OR s.successful_required_nodes != actual_required_nodes
                      OR s.terminal_required_node_failures != actual_required_node_failures
                      OR s.successful_optional_nodes != actual_optional_nodes
                      OR s.terminal_optional_node_failures != actual_optional_node_failures)
                )""",
            "invalid_typed_fits": f"""
                SELECT count() FROM {target('options_chain_surface_fits')}
                WHERE (fit_method = 'SVI' AND
                       (isNull(svi_a) OR isNull(svi_b) OR isNull(svi_rho)
                        OR isNull(svi_m) OR isNull(svi_sigma)
                        OR NOT isFinite(svi_a) OR NOT isFinite(svi_b)
                        OR NOT isFinite(svi_rho) OR NOT isFinite(svi_m)
                        OR NOT isFinite(svi_sigma) OR svi_b < 0
                        OR svi_sigma <= 0 OR svi_rho <= -1 OR svi_rho >= 1
                        OR length(pchip_k) != 0 OR length(pchip_total_variance) != 0))
                   OR (fit_method = 'PCHIP' AND
                       (NOT isNull(svi_a) OR NOT isNull(svi_b)
                        OR NOT isNull(svi_rho) OR NOT isNull(svi_m)
                        OR NOT isNull(svi_sigma)
                        OR length(pchip_k) < 2
                        OR length(pchip_k) != length(pchip_total_variance)
                        OR NOT arrayAll(x -> isFinite(x), pchip_k)
                        OR NOT arrayAll(x -> isFinite(x) AND x >= 0,
                                        pchip_total_variance)
                        OR NOT arrayAll(i -> pchip_k[i] < pchip_k[i + 1],
                                        range(1, length(pchip_k)))))""",
            "calculation_result_source_authorization": f"""
                SELECT count() FROM {target('options_chain_calculation_results')} AS r
                LEFT JOIN (SELECT * FROM {target('options_chain_calculation_state')} FINAL) AS s
                  ON s.snapshot_id = r.snapshot_id
                 AND s.calculation_definition_id = r.calculation_definition_id
                 AND s.is_active
                WHERE isNull(s.snapshot_id)
                   OR isNull(r.source_calculation_run_id)
                   OR r.source_calculation_run_id != s.source_calculation_run_id""",
            "calculation_failure_source_authorization": f"""
                SELECT count() FROM {target('options_chain_calculation_failures')} AS f
                LEFT JOIN (SELECT * FROM {target('options_chain_calculation_state')} FINAL) AS s
                  ON s.snapshot_id = f.snapshot_id
                 AND s.calculation_definition_id = f.calculation_definition_id
                 AND s.is_active
                WHERE isNull(s.snapshot_id)
                   OR isNull(f.source_calculation_run_id)
                   OR f.source_calculation_run_id != s.source_calculation_run_id""",
            "surface_fit_source_authorization": f"""
                SELECT count() FROM {target('options_chain_surface_fits')} AS f
                LEFT JOIN (SELECT * FROM {target('options_chain_surface_state')} FINAL) AS s
                  ON s.snapshot_id = f.snapshot_id
                 AND s.surface_definition_id = f.surface_definition_id
                 AND s.is_active
                WHERE isNull(s.snapshot_id)
                   OR isNull(f.source_surface_run_id)
                   OR f.source_surface_run_id != s.source_surface_run_id""",
            "surface_node_parent_authorization": f"""
                SELECT count() FROM {target('options_chain_surface_nodes')} AS n
                LEFT JOIN {target('options_chain_surface_fits')} AS f
                  ON f.snapshot_id = n.snapshot_id
                 AND f.expiration_at = n.expiration_at
                 AND f.price_basis = n.price_basis
                 AND f.surface_definition_id = n.surface_definition_id
                 AND f.source_surface_run_id = n.source_surface_run_id
                WHERE isNull(f.snapshot_id)""",
            "surface_failure_source_authorization": f"""
                SELECT count() FROM {target('options_chain_surface_failures')} AS f
                LEFT JOIN (SELECT * FROM {target('options_chain_surface_state')} FINAL) AS s
                  ON s.snapshot_id = f.snapshot_id
                 AND s.surface_definition_id = f.surface_definition_id
                 AND s.is_active
                WHERE isNull(s.snapshot_id)
                   OR isNull(f.source_surface_run_id)
                   OR f.source_surface_run_id != s.source_surface_run_id""",
            "calculation_state_source_coverage": f"""
                SELECT count() FROM {self.database}.options_chain_calculation_runs_latest AS r
                LEFT JOIN (SELECT * FROM {target('options_chain_calculation_state')} FINAL) AS s
                  ON s.snapshot_id = r.snapshot_id
                 AND s.source_calculation_run_id = r.run_id
                 AND s.is_active
                WHERE isNull(s.snapshot_id)""",
            "surface_state_source_coverage": f"""
                SELECT count() FROM {self.database}.options_chain_surface_runs_latest AS r
                LEFT JOIN (SELECT * FROM {target('options_chain_surface_state')} FINAL) AS s
                  ON s.snapshot_id = r.snapshot_id
                 AND s.source_surface_run_id = r.surface_run_id
                 AND s.is_active
                WHERE isNull(s.snapshot_id)""",
            "calculation_success_source_coverage": f"""
                SELECT count() FROM {self.database}.options_chain_calculations_latest AS c
                INNER JOIN {self.database}.options_chain_calculation_runs_latest AS run
                  ON run.snapshot_id = c.snapshot_id
                 AND run.calculation_version = c.calculation_version
                INNER JOIN {self.database}.options_chain_base AS b
                  ON b.snapshot_id = c.snapshot_id AND b.option_symbol = c.option_symbol
                LEFT JOIN {target('options_chain_calculation_results')} AS r
                  ON r.snapshot_id = c.snapshot_id
                 AND r.contract_id = {_contract_id_sql('b')}
                 AND r.source_calculation_run_id = run.run_id
                WHERE run.status IN ('complete', 'complete_with_failures')
                  AND {_legacy_success_predicate('c')}
                  AND isNull(r.snapshot_id)""",
            "calculation_failure_source_coverage": f"""
                SELECT count() FROM {self.database}.options_chain_calculations_latest AS c
                INNER JOIN {self.database}.options_chain_calculation_runs_latest AS run
                  ON run.snapshot_id = c.snapshot_id
                 AND run.calculation_version = c.calculation_version
                INNER JOIN {self.database}.options_chain_base AS b
                  ON b.snapshot_id = c.snapshot_id AND b.option_symbol = c.option_symbol
                LEFT JOIN {target('options_chain_calculation_failures')} AS f
                  ON f.snapshot_id = c.snapshot_id
                 AND f.contract_id = {_contract_id_sql('b')}
                 AND f.source_calculation_run_id = run.run_id
                 AND f.is_active
                WHERE run.status NOT IN ('pending', 'running')
                  AND {_legacy_terminal_failure_predicate('c')}
                  AND isNull(f.snapshot_id)""",
            "surface_success_fit_source_coverage": f"""
                SELECT count() FROM {self.database}.options_chain_surface_fits AS source_fit
                INNER JOIN {self.database}.options_chain_surface_runs_latest AS run
                  ON run.surface_run_id = source_fit.surface_run_id
                 AND run.snapshot_id = source_fit.snapshot_id
                 AND run.surface_version = source_fit.surface_version
                 AND run.source_calculation_run_id = source_fit.source_calculation_run_id
                LEFT JOIN {target('options_chain_surface_fits')} AS fit
                  ON fit.snapshot_id = source_fit.snapshot_id
                 AND fit.expiration_at = source_fit.expiration_at
                 AND fit.price_basis = source_fit.price_basis
                 AND fit.source_surface_run_id = run.surface_run_id
                WHERE run.status IN ('complete', 'complete_with_failures')
                  AND source_fit.fit_status IN ('complete', 'complete_warning')
                  AND isNull(fit.snapshot_id)""",
            "calculation_count_conservation": f"""
                SELECT count() FROM {target('options_chain_calculation_state')} FINAL
                WHERE is_active
                  AND status IN ('complete', 'complete_with_failures', 'failed')
                  AND eligible_contracts != successful_results + source_only_contracts
                                                   + terminal_contract_failures""",
            "surface_expiration_count_conservation": f"""
                SELECT count() FROM {target('options_chain_surface_state')} FINAL
                WHERE is_active
                  AND status IN ('complete', 'complete_with_failures', 'failed')
                  AND expected_expirations != successful_expirations
                                                   + source_only_or_skipped_expirations
                                                   + terminal_fit_failures""",
        }
        summary["integrity"] = {}
        for name, sql in integrity_queries.items():
            violations = self._scalar(sql)
            summary["integrity"][name] = violations
            if violations:
                summary["errors"].append(f"{name} has {violations} violations")
        summary["checks"] = [
            {
                "check_name": name,
                "severity": "error",
                "status": "passed" if not int(value or 0) else "failed",
                "expected": 0,
                "actual": int(value or 0),
                "details": "bidirectional integrity check",
            }
            for name, value in summary["integrity"].items()
            if isinstance(value, (int, float))
        ]
        contract_sample_size = int(
            self.context.get(
                "options_chain.migration.numerical_parity_sample_size", 512
            )
        )
        contract_sample = self.client.query_arrow(
            f"SELECT * EXCEPT(put_call), toString(put_call) AS put_call "
            f"FROM {target('options_contracts')} FINAL "
            "ORDER BY cityHash64(toString(contract_id)) "
            "LIMIT {sample_size:UInt32}",
            parameters={"sample_size": contract_sample_size},
        ).to_pylist()
        contract_hash_mismatches = 0
        for row in contract_sample:
            if (
                bytes(row["identity_hash"]) != contract_identity_hash(row)
                or bytes(row["metadata_hash"]) != contract_metadata_hash(row)
            ):
                contract_hash_mismatches += 1
        summary["integrity"]["contract_hash_compatibility_sample"] = {
            "sample_rows": len(contract_sample),
            "mismatches": contract_hash_mismatches,
        }
        if contract_hash_mismatches:
            summary["errors"].append(
                f"contract hash compatibility has {contract_hash_mismatches} mismatches"
            )
        try:
            from scipy.interpolate import PchipInterpolator

            sample_size = int(
                self.context.get(
                    "options_chain.migration.numerical_parity_sample_size", 512
                )
            )
            fit_target = target_name(
                "options_chain_surface_fits", self.migration_id
            )
            fit_sample = self.client.query(
                f"""
                SELECT l.fit_parameters_json, toString(t.fit_method),
                       t.svi_a, t.svi_b, t.svi_rho, t.svi_m, t.svi_sigma,
                       t.pchip_k, t.pchip_total_variance
                FROM {self.database}.options_chain_surface_fits_latest AS l
                INNER JOIN {self.database}.{fit_target} AS t
                  ON t.snapshot_id = l.snapshot_id
                 AND t.expiration_at = l.expiration_at
                 AND t.price_basis = l.price_basis
                WHERE l.fit_status IN ('complete', 'complete_warning')
                ORDER BY cityHash64(toString(l.snapshot_id),
                                    toString(l.expiration_at), l.price_basis)
                LIMIT {{sample_size:UInt32}}
                """,
                parameters={"sample_size": sample_size},
            ).result_rows
            maximum_parameter_error = 0.0
            maximum_grid_error = 0.0
            grid = (-0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50)
            if summary["surface_typed_parse_successes"] and not fit_sample:
                raise MigrationError(
                    "Surface numerical parity sample found no migrated successful fits"
                )
            for row in fit_sample:
                expected = parse_fit_parameters_json(str(row[0]))
                actual = {
                    "fit_method": str(row[1]),
                    "svi_a": row[2],
                    "svi_b": row[3],
                    "svi_rho": row[4],
                    "svi_m": row[5],
                    "svi_sigma": row[6],
                    "pchip_k": list(row[7]),
                    "pchip_total_variance": list(row[8]),
                }
                if expected["fit_method"] != actual["fit_method"]:
                    raise MigrationError("Typed surface fit method changed in migration")
                if expected["fit_method"] == "SVI":
                    names = ("svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma")
                    for name in names:
                        error = abs(float(expected[name]) - float(actual[name]))
                        maximum_parameter_error = max(maximum_parameter_error, error)
                    for k in grid:
                        def evaluate(values: Mapping[str, Any]) -> float:
                            shifted = k - float(values["svi_m"])
                            return float(values["svi_a"]) + float(values["svi_b"]) * (
                                float(values["svi_rho"]) * shifted
                                + math.sqrt(
                                    shifted * shifted
                                    + float(values["svi_sigma"]) ** 2
                                )
                            )

                        maximum_grid_error = max(
                            maximum_grid_error,
                            abs(evaluate(expected) - evaluate(actual)),
                        )
                else:
                    if len(expected["pchip_k"]) != len(actual["pchip_k"]):
                        raise MigrationError("Typed PCHIP knot count changed in migration")
                    for expected_values, actual_values in (
                        (expected["pchip_k"], actual["pchip_k"]),
                        (
                            expected["pchip_total_variance"],
                            actual["pchip_total_variance"],
                        ),
                    ):
                        for expected_value, actual_value in zip(
                            expected_values, actual_values
                        ):
                            maximum_parameter_error = max(
                                maximum_parameter_error,
                                abs(float(expected_value) - float(actual_value)),
                            )
                    left = float(expected["pchip_k"][0])
                    right = float(expected["pchip_k"][-1])
                    evaluation_grid = [
                        left + (right - left) * index / 16.0
                        for index in range(17)
                    ]
                    expected_grid = PchipInterpolator(
                        expected["pchip_k"],
                        expected["pchip_total_variance"],
                        extrapolate=False,
                    )(evaluation_grid)
                    actual_grid = PchipInterpolator(
                        actual["pchip_k"],
                        actual["pchip_total_variance"],
                        extrapolate=False,
                    )(evaluation_grid)
                    for expected_value, actual_value in zip(
                        expected_grid, actual_grid
                    ):
                        maximum_grid_error = max(
                            maximum_grid_error,
                            abs(float(expected_value) - float(actual_value)),
                        )
            node_target = target_name(
                "options_chain_surface_nodes", self.migration_id
            )
            node_sample = self.client.query(
                f"""
                SELECT l.fitted_strike, t.fitted_strike,
                       l.fitted_delta, t.fitted_delta,
                       l.fitted_iv_bid, t.fitted_iv_bid,
                       l.fitted_iv_mid, t.fitted_iv_mid,
                       l.fitted_iv_ask, t.fitted_iv_ask,
                       l.fit_confidence, t.fit_confidence,
                       l.quality_flags, t.quality_flags
                FROM {self.database}.options_chain_surface_nodes_latest AS l
                INNER JOIN {self.database}.{node_target} AS t
                  ON t.snapshot_id = l.snapshot_id
                 AND t.expiration_at = l.expiration_at
                 AND t.price_basis = 'mid'
                 AND t.node_type = l.node_type
                 AND t.node_value = l.node_value
                 AND t.put_call = l.put_call
                WHERE l.node_status = 'complete'
                ORDER BY cityHash64(toString(l.snapshot_id),
                                    toString(l.expiration_at),
                                    l.node_type, toString(l.node_value), l.put_call)
                LIMIT {{sample_size:UInt32}}
                """,
                parameters={"sample_size": sample_size},
            ).result_rows
            if summary["surface_node_successes"] and not node_sample:
                raise MigrationError(
                    "Surface numerical parity sample found no migrated successful nodes"
                )
            maximum_node_error = 0.0
            for row in node_sample:
                for index in range(0, 12, 2):
                    old_value, new_value = row[index], row[index + 1]
                    if old_value is None or new_value is None:
                        if old_value is not None or new_value is not None:
                            raise MigrationError(
                                "Surface node nullability changed in migration"
                            )
                        continue
                    maximum_node_error = max(
                        maximum_node_error,
                        abs(float(old_value) - float(new_value)),
                    )
                if int(row[12] or 0) != int(row[13] or 0):
                    raise MigrationError(
                        "Surface node quality flags changed in migration"
                    )
            summary["surface_numerical_parity"] = {
                "sample_rows": len(fit_sample),
                "node_sample_rows": len(node_sample),
                "maximum_parameter_error": maximum_parameter_error,
                "maximum_grid_error": maximum_grid_error,
                "maximum_node_error": maximum_node_error,
                "matches": (
                    maximum_parameter_error <= 1e-14
                    and maximum_grid_error <= 1e-14
                    and maximum_node_error <= 1e-14
                ),
            }
            if not summary["surface_numerical_parity"]["matches"]:
                summary["errors"].append(
                    "surface typed-parameter numerical parity exceeded tolerance"
                )
        except Exception as exc:
            if isinstance(exc, MigrationError):
                summary["errors"].append(str(exc))
            else:
                summary["errors"].append(
                    f"surface numerical parity validation failed: {exc}"
                )
        try:
            sample_size = int(
                self.context.get(
                    "options_chain.migration.numerical_parity_sample_size", 512
                )
            )
            result_target = target_name(
                "options_chain_calculation_results", self.migration_id
            )
            calculation_sample = self.client.query(
                f"""
                SELECT c.recalc_target_price, t.target_price,
                       c.recalc_iv, t.iv,
                       c.recalc_price_residual, t.price_residual,
                       c.recalc_model_price, t.model_price,
                       c.recalc_delta, t.delta,
                       c.recalc_gamma, t.gamma,
                       c.recalc_theta_per_day, t.theta_per_day,
                       c.recalc_vega_per_1pct, t.vega_per_1pct,
                       c.recalc_rho_per_1pct, t.rho_per_1pct,
                       c.recalc_iv_iterations, t.iv_iterations,
                       c.recalc_iv_identifiability_status,
                       toString(t.iv_selection),
                       c.recalc_result_origin, toString(t.result_origin),
                       b.source_iv, o.source_iv
                FROM {self.database}.options_chain_calculations_latest AS c
                INNER JOIN {self.database}.options_chain_calculation_runs_latest AS r
                  ON r.snapshot_id = c.snapshot_id
                 AND r.calculation_version = c.calculation_version
                INNER JOIN {self.database}.options_chain_base AS b
                  ON b.snapshot_id = c.snapshot_id
                 AND b.option_symbol = c.option_symbol
                INNER JOIN {self.database}.{result_target} AS t
                  ON t.snapshot_id = c.snapshot_id
                 AND t.contract_id = {_contract_id_sql('b')}
                INNER JOIN {self.database}.{target_name('options_chain_observations', self.migration_id)} AS o
                  ON o.snapshot_id = c.snapshot_id
                 AND o.contract_id = t.contract_id
                WHERE {_legacy_success_predicate('c')}
                ORDER BY cityHash64(toString(c.snapshot_id), c.option_symbol)
                LIMIT {{sample_size:UInt32}}
                """,
                parameters={"sample_size": sample_size},
            ).result_rows
            result_count = int(
                summary["tables"]
                .get("options_chain_calculation_results", {})
                .get("rows", 0)
            )
            if result_count and not calculation_sample:
                raise MigrationError(
                    "Calculation numerical parity sample found no migrated successes"
                )
            maximum_solver_error = 0.0
            maximum_effective_iv_error = 0.0
            maximum_effective_iv_relative_error = 0.0
            effective_iv_matches = True
            for row in calculation_sample:
                for index in range(0, 18, 2):
                    maximum_solver_error = max(
                        maximum_solver_error,
                        abs(float(row[index]) - float(row[index + 1])),
                    )
                if int(row[18]) != int(row[19]):
                    raise MigrationError(
                        "Calculation IV iteration count changed in migration"
                    )
                expected_iv_selection = (
                    "source" if str(row[20]) == "source_anchor_flat" else "result"
                )
                if str(row[21]) != expected_iv_selection:
                    raise MigrationError(
                        "Calculation IV selection semantic changed in migration"
                    )
                expected_origin = (
                    "reused" if str(row[22]) == "reused" else "fresh"
                )
                if str(row[23]) != expected_origin:
                    raise MigrationError(
                        "Calculation result origin changed in migration"
                    )
                old_effective_iv = (
                    row[24] if expected_iv_selection == "source" else row[2]
                )
                new_effective_iv = row[25] if str(row[21]) == "source" else row[3]
                if old_effective_iv is None or new_effective_iv is None:
                    raise MigrationError(
                        "Calculation effective IV became unavailable in migration"
                    )
                effective_error = abs(
                    float(old_effective_iv) - float(new_effective_iv)
                )
                effective_relative_error = effective_error / max(
                    abs(float(old_effective_iv)), 1e-30
                )
                maximum_effective_iv_error = max(
                    maximum_effective_iv_error, effective_error
                )
                maximum_effective_iv_relative_error = max(
                    maximum_effective_iv_relative_error,
                    effective_relative_error,
                )
                absolute_limit, relative_limit = FLOAT32_CAST_THRESHOLDS[
                    "source_iv"
                ]
                if (
                    effective_error > absolute_limit
                    and effective_relative_error > relative_limit
                ):
                    effective_iv_matches = False
            summary["calculation_numerical_parity"] = {
                "sample_rows": len(calculation_sample),
                "maximum_solver_error": maximum_solver_error,
                "maximum_effective_iv_error": maximum_effective_iv_error,
                "maximum_effective_iv_relative_error": (
                    maximum_effective_iv_relative_error
                ),
                "matches": (
                    maximum_solver_error <= 1e-14
                    and effective_iv_matches
                ),
            }
            if not summary["calculation_numerical_parity"]["matches"]:
                summary["errors"].append(
                    "calculation numerical parity exceeded tolerance"
                )
        except Exception as exc:
            if isinstance(exc, MigrationError):
                summary["errors"].append(str(exc))
            else:
                summary["errors"].append(
                    f"calculation numerical parity validation failed: {exc}"
                )
        old_compressed = sum(
            int(value.get("compressed_bytes", 0))
            for value in summary["legacy_tables"].values()
        )
        new_compressed = sum(
            int(value.get("compressed_bytes", 0))
            for value in summary["tables"].values()
        )
        summary["disk"] = {
            "legacy_compressed_bytes": old_compressed,
            "canonical_compressed_bytes": new_compressed,
            "estimated_bytes_reclaimed": old_compressed - new_compressed,
            "bytes_reclaimed_immediately": 0,
            "bytes_retained_for_rollback": old_compressed,
            "operational_table_count": len(CANONICAL_TABLE_NAMES),
        }
        self.logger.info(
            "schema_disk_usage migration_id=%s database=%s old_bytes=%s "
            "new_bytes=%s estimated_reclaimed=%s",
            self.migration_id,
            self.database,
            old_compressed,
            new_compressed,
            old_compressed - new_compressed,
        )
        try:
            calculation_totals = self.client.query(
                f"SELECT sum(successful_results), sum(source_only_contracts), "
                "sum(terminal_contract_failures), sum(terminal_expiration_failures), "
                "sum(snapshot_failure_count) FROM "
                f"{target('options_chain_calculation_state')} FINAL WHERE is_active"
            ).result_rows[0]
            calculation_failure_rows = self.client.query(
                f"SELECT toString(failure_scope), reason_code, count() FROM "
                f"{target('options_chain_calculation_failures')} FINAL "
                "WHERE is_active GROUP BY failure_scope, reason_code "
                "ORDER BY failure_scope, reason_code"
            ).result_rows
            surface_totals = self.client.query(
                f"SELECT sum(successful_expirations), sum(terminal_fit_failures), "
                "sum(successful_required_nodes), sum(terminal_required_node_failures), "
                "sum(successful_optional_nodes), sum(terminal_optional_node_failures) "
                f"FROM {target('options_chain_surface_state')} FINAL WHERE is_active"
            ).result_rows[0]
            raw_archive_bytes = self._scalar(
                f"SELECT coalesce(sum(compressed_bytes), 0) FROM "
                f"{target('options_chain_archive_manifest')} FINAL "
                "WHERE archive_kind = 'raw_week' AND state IN ('verified', 'purged')"
            )
            summary["outcomes"] = {
                "calculation": {
                    "success": int(calculation_totals[0] or 0),
                    "source_only": int(calculation_totals[1] or 0),
                    "contract_failures": int(calculation_totals[2] or 0),
                    "expiration_failures": int(calculation_totals[3] or 0),
                    "snapshot_failures": int(calculation_totals[4] or 0),
                },
                "calculation_failures_by_scope_reason": [
                    {
                        "scope": str(scope),
                        "reason": str(reason),
                        "count": int(count),
                    }
                    for scope, reason, count in calculation_failure_rows
                ],
                "surface": {
                    "successful_expirations": int(surface_totals[0] or 0),
                    "fit_failures": int(surface_totals[1] or 0),
                    "successful_required_nodes": int(surface_totals[2] or 0),
                    "required_node_failures": int(surface_totals[3] or 0),
                    "successful_optional_nodes": int(surface_totals[4] or 0),
                    "optional_node_failures": int(surface_totals[5] or 0),
                },
            }
            summary["disk"]["raw_bytes_archived"] = raw_archive_bytes
        except Exception as exc:
            summary["errors"].append(f"outcome/disk summary failed: {exc}")
        try:
            from algotrading.refdata.options.Queries import (
                Query,
                benchmark_query,
                build_effective_chain_query,
            )

            snapshot_target = target_name(
                "options_chain_snapshots", self.migration_id
            )
            latest = self.client.query(
                f"SELECT toString(snapshot_id) FROM {self.database}.{snapshot_target} "
                "ORDER BY source_collected_at DESC LIMIT 1"
            ).result_rows
            if latest:
                query = build_effective_chain_query(
                    self.database, str(latest[0][0])
                )
                target_sql = query.sql
                for canonical in sorted(CANONICAL_TABLE_NAMES, key=len, reverse=True):
                    target_sql = target_sql.replace(
                        f"{self.database}.{canonical}",
                        f"{self.database}.{target_name(canonical, self.migration_id)}",
                    )
                summary["query_benchmark"] = benchmark_query(
                    self.client,
                    Query(target_sql, query.parameters),
                    repetitions=int(
                        self.context.get(
                            "options_chain.gui_benchmark.repetitions", 3
                        )
                    ),
                    warm_target_seconds=float(
                        self.context.get(
                            "options_chain.gui_benchmark.warm_target_seconds", 0.250
                        )
                    ),
                    cold_target_seconds=float(
                        self.context.get(
                            "options_chain.gui_benchmark.cold_target_seconds", 1.0
                        )
                    ),
                )
            else:
                summary["query_benchmark"] = {"status": "skipped_empty_dataset"}
        except Exception as exc:
            summary["query_benchmark"] = {"status": "failed", "error": str(exc)}
            summary["errors"].append(f"effective-chain query benchmark failed: {exc}")
        summary["operator"] = {
            "legacy_table_names": sorted(summary["legacy_tables"]),
            "rollback_command": (
                "python -m algotrading.refdata.options.SchemaMigration "
                f"-e <environment> --migration-id {self.migration_id} "
                f"--archive-directory {self.archive_directory} --rollback "
                f"--confirm-database {self.database}"
            ),
            "cleanup_command": (
                "python -m algotrading.refdata.options.SchemaMigration "
                f"-e <environment> --migration-id {self.migration_id} "
                f"--archive-directory {self.archive_directory} --cleanup-legacy "
                f"--confirm-database {self.database}"
            ),
            "cast_error_maxima": self.checkpoint.validation_summary.get(
                "source_float32_cast_errors", {}
            ),
        }
        summary["status"] = "passed" if not summary["errors"] else "failed"
        self.logger.info(
            "migration_validation migration_id=%s database=%s status=%s "
            "errors=%s",
            self.migration_id,
            self.database,
            summary["status"],
            len(summary["errors"]),
        )
        report_path = self.archive_directory / f"migration_validation_{self.migration_id}.json"
        if not self.dry_run and summary.get("checks"):
            check_rows = []
            for check in summary["checks"]:
                check_rows.append([
                    str(uuid.uuid4()), str(self.migration_id), self.checkpoint.phase.value,
                    str(check["check_name"]), str(check["severity"]),
                    str(check["status"]), safe_source_value(check["expected"]),
                    safe_source_value(check["actual"]), _bounded_text(check["details"]),
                    _utc_now(), _utc_now(),
                ])
            self.client.insert(
                f"{self.database}.options_chain_migration_validation",
                check_rows,
                column_names=[
                    "validation_id", "migration_id", "phase", "check_name", "severity",
                    "status", "expected", "actual", "details", "checked_at", "updated_at",
                ],
            )
        if not self.dry_run:
            report_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if summary["errors"]:
            self._checkpoint(
                "validation_failed",
                self.checkpoint.phase,
                validation_summary=summary,
            )
            raise MigrationError("Migration validation failed: " + "; ".join(summary["errors"]))
        self._checkpoint(
            "validate", MigrationPhase.VALIDATED, validation_summary=summary
        )
        return summary

    def _require_confirmation(
        self, confirm_database: Optional[str], operation: str = "destructive operation"
    ) -> None:
        if confirm_database != self.database:
            raise MigrationError(
                f"--{operation} targets configured database {self.database}; "
                f"repeat with --confirm-database {self.database}"
            )

    def _verify_object_uuid(self, name: str, expected: Optional[str]) -> None:
        if expected is None:
            return
        rows = self.client.query(
            "SELECT toString(uuid) FROM system.tables "
            "WHERE database = {database:String} AND name = {name:String}",
            parameters={"database": self.database, "name": name},
        ).result_rows
        if not rows or str(rows[0][0]) != expected:
            raise MigrationError(f"Object UUID mismatch for {name}")

    def _object_uuid_or_none(self, name: str) -> Optional[str]:
        rows = self.client.query(
            "SELECT toString(uuid) FROM system.tables "
            "WHERE database = {database:String} AND name = {name:String}",
            parameters={"database": self.database, "name": name},
        ).result_rows
        return str(rows[0][0]) if rows else None

    def _post_cutover_validation(self) -> Mapping[str, Any]:
        errors: list[str] = []
        for item in self.checkpoint.objects:
            actual = self._object_uuid_or_none(item.canonical_name)
            if actual != item.target_uuid:
                errors.append(
                    f"{item.canonical_name} UUID {actual!r} != target {item.target_uuid!r}"
                )
        remaining_views = self.client.query(
            "SELECT name FROM system.tables WHERE database = {database:String} "
            "AND name IN {names:Array(String)}",
            parameters={"database": self.database, "names": list(LEGACY_VIEWS)},
        ).result_rows
        if remaining_views:
            errors.append(
                "legacy views remain: " + ", ".join(str(row[0]) for row in remaining_views)
            )
        legacy_remaining = self.client.query(
            "SELECT name FROM system.tables WHERE database = {database:String} "
            "AND name IN {names:Array(String)}",
            parameters={"database": self.database, "names": list(LEGACY_TABLES)},
        ).result_rows
        canonical_overlap = set(CANONICAL_TABLE_NAMES)
        forbidden_remaining = [
            str(row[0]) for row in legacy_remaining if str(row[0]) not in canonical_overlap
        ]
        if forbidden_remaining:
            errors.append(
                "legacy-only tables remain: " + ", ".join(forbidden_remaining)
            )
        table_count = self.client.query(
            "SELECT count() FROM system.tables WHERE database = {database:String} "
            "AND name IN {names:Array(String)}",
            parameters={"database": self.database, "names": list(CANONICAL_TABLE_NAMES)},
        ).result_rows[0][0]
        if int(table_count) != len(CANONICAL_TABLE_NAMES):
            errors.append(
                f"canonical table count is {table_count}, expected {len(CANONICAL_TABLE_NAMES)}"
            )
        representative = {}
        for name, sql in {
            "ready_snapshots": (
                f"SELECT count() FROM {self.database}.options_chain_snapshots "
                "WHERE stored_observation_count > 0"
            ),
            "calculation_states": (
                f"SELECT count() FROM {self.database}.options_chain_calculation_state FINAL "
                "WHERE is_active"
            ),
            "surface_states": (
                f"SELECT count() FROM {self.database}.options_chain_surface_state FINAL "
                "WHERE is_active"
            ),
        }.items():
            try:
                representative[name] = self._scalar(sql)
            except Exception as exc:
                errors.append(f"representative query {name} failed: {exc}")
        query_benchmark: Mapping[str, Any] = {}
        try:
            from algotrading.refdata.options.Queries import benchmark_effective_chain

            latest = self.client.query(
                f"SELECT toString(snapshot_id) FROM "
                f"{self.database}.options_chain_snapshots "
                "ORDER BY source_collected_at DESC LIMIT 1"
            ).result_rows
            if latest:
                query_benchmark = benchmark_effective_chain(
                    self.client,
                    self.database,
                    str(latest[0][0]),
                    repetitions=int(
                        self.context.get(
                            "options_chain.gui_benchmark.repetitions", 3
                        )
                    ),
                )
        except Exception as exc:
            errors.append(f"effective-chain query benchmark failed: {exc}")
        report = {
            "status": "passed" if not errors else "failed",
            "errors": errors,
            "canonical_table_count": int(table_count),
            "representative_queries": representative,
            "query_benchmark": query_benchmark,
        }
        if errors:
            raise MigrationError("Post-cutover validation failed: " + "; ".join(errors))
        return report

    def cutover(self, confirm_database: Optional[str]) -> None:
        self._require_confirmation(confirm_database, "cutover")
        if self.checkpoint.phase is MigrationPhase.CUTOVER:
            return
        self._ensure_lease(MigrationPhase.PREPARING_CUTOVER)
        if self.checkpoint.phase is MigrationPhase.CUTOVER_RENAMES_COMPLETE:
            post_validation = self._post_cutover_validation()
            self._checkpoint(
                "cutover", MigrationPhase.CUTOVER, is_cutover=True,
                is_rolled_back=False,
                validation_summary={
                    **self.checkpoint.validation_summary,
                    "post_cutover": post_validation,
                },
            )
            return
        if self.checkpoint.phase in {
            MigrationPhase.VALIDATED, MigrationPhase.PREPARING_CUTOVER,
        }:
            self.offline_preflight()
            self.migrate_final_raw_delta()
            self.validate()
            if self.migrate_final_raw_delta():
                raise MigrationError(
                    "Raw capture advanced during final validation; stop Collector "
                    "and rerun --cutover so the new delta is revalidated"
                )
            self._checkpoint("preparing_cutover", MigrationPhase.PREPARING_CUTOVER)
            self._checkpoint("cutover_started", MigrationPhase.CUTTING_OVER)
        elif self.checkpoint.phase is not MigrationPhase.CUTTING_OVER:
            raise MigrationError("Cutover requires a current passing validation checkpoint")

        for view in LEGACY_VIEWS:
            self._command(f"DROP VIEW IF EXISTS {self.database}.{view} SYNC")
            self._checkpoint(f"drop_view:{view}", MigrationPhase.CUTTING_OVER)
        for name, metadata in self.checkpoint.validation_summary.get(
            "legacy_only_objects", {}
        ).items():
            rollback_name = str(metadata["rollback_name"])
            source_uuid = str(metadata["uuid"])
            current_uuid = self._object_uuid_or_none(str(name))
            rollback_uuid = self._object_uuid_or_none(rollback_name)
            if current_uuid == source_uuid and rollback_uuid is None:
                self._command(
                    f"RENAME TABLE {self.database}.{name} TO "
                    f"{self.database}.{rollback_name}"
                )
                if self._object_uuid_or_none(rollback_name) != source_uuid:
                    raise MigrationError(f"Legacy-only rename verification failed for {name}")
            elif not (current_uuid is None and rollback_uuid == source_uuid):
                raise MigrationError(f"Legacy-only cutover ownership is ambiguous for {name}")
            self._checkpoint(f"cutover_legacy:{name}", MigrationPhase.CUTTING_OVER)
        for item in self.checkpoint.objects:
            target_uuid = self._object_uuid_or_none(item.target_name)
            canonical_uuid = self._object_uuid_or_none(item.canonical_name)
            legacy_uuid = self._object_uuid_or_none(item.legacy_name)
            if canonical_uuid == item.target_uuid and target_uuid is None and (
                item.source_uuid is None or legacy_uuid == item.source_uuid
            ):
                self._checkpoint(f"cutover:{item.canonical_name}", MigrationPhase.CUTTING_OVER)
                continue
            if target_uuid != item.target_uuid:
                raise MigrationError(f"Cutover target ownership is ambiguous for {item.target_name}")
            if item.source_uuid is not None:
                if canonical_uuid == item.source_uuid and legacy_uuid is None:
                    self._command(
                        f"RENAME TABLE {self.database}.{item.canonical_name} TO "
                        f"{self.database}.{item.legacy_name}"
                    )
                    if (
                        self._object_uuid_or_none(item.legacy_name) != item.source_uuid
                        or self._object_uuid_or_none(item.canonical_name) is not None
                    ):
                        raise MigrationError(
                            f"Cutover source rename verification failed for {item.canonical_name}"
                        )
                    self._checkpoint(
                        f"cutover:source:{item.canonical_name}", MigrationPhase.CUTTING_OVER
                    )
                elif not (canonical_uuid is None and legacy_uuid == item.source_uuid):
                    raise MigrationError(f"Cutover source ownership is ambiguous for {item.canonical_name}")
                self._command(
                    f"RENAME TABLE {self.database}.{item.target_name} TO "
                    f"{self.database}.{item.canonical_name}"
                )
            else:
                if canonical_uuid is not None:
                    raise MigrationError(f"Unexpected canonical object exists for {item.canonical_name}")
                self._command(
                    f"RENAME TABLE {self.database}.{item.target_name} TO "
                    f"{self.database}.{item.canonical_name}"
                )
            if self._object_uuid_or_none(item.canonical_name) != item.target_uuid:
                raise MigrationError(f"Cutover rename verification failed for {item.canonical_name}")
            self._checkpoint(f"cutover:{item.canonical_name}", MigrationPhase.CUTTING_OVER)
        self._checkpoint(
            "cutover_renames", MigrationPhase.CUTOVER_RENAMES_COMPLETE,
            is_cutover=True, is_rolled_back=False,
        )
        post_validation = self._post_cutover_validation()
        self._checkpoint(
            "cutover", MigrationPhase.CUTOVER, is_cutover=True,
            is_rolled_back=False,
            validation_summary={
                **self.checkpoint.validation_summary,
                "post_cutover": post_validation,
            },
        )

    def rollback(self, confirm_database: Optional[str]) -> None:
        self._require_confirmation(confirm_database, "rollback")
        if self.checkpoint.phase is MigrationPhase.ROLLED_BACK:
            return
        self._ensure_lease(MigrationPhase.ROLLING_BACK)
        if not self.checkpoint.is_cutover and self.checkpoint.phase is not MigrationPhase.ROLLING_BACK:
            raise MigrationError("Checkpoint does not describe an active cutover")
        if self.checkpoint.phase is not MigrationPhase.ROLLING_BACK:
            self.offline_preflight()
            self._checkpoint("rollback_started", MigrationPhase.ROLLING_BACK)
        for item in reversed(self.checkpoint.objects):
            failed_name = target_name(item.canonical_name, self.migration_id) + "__failed"
            canonical_uuid = self._object_uuid_or_none(item.canonical_name)
            legacy_uuid = self._object_uuid_or_none(item.legacy_name)
            failed_uuid = self._object_uuid_or_none(failed_name)
            if item.source_uuid is None:
                if canonical_uuid == item.target_uuid and failed_uuid is None:
                    self._command(
                        f"RENAME TABLE {self.database}.{item.canonical_name} TO "
                        f"{self.database}.{failed_name}"
                    )
                    if (
                        self._object_uuid_or_none(failed_name) != item.target_uuid
                        or self._object_uuid_or_none(item.canonical_name) is not None
                    ):
                        raise MigrationError(
                            f"Rollback target rename verification failed for {item.canonical_name}"
                        )
                    self._checkpoint(f"rollback:target:{item.canonical_name}", MigrationPhase.ROLLING_BACK)
                elif canonical_uuid is not None or failed_uuid != item.target_uuid:
                    raise MigrationError(
                        f"Rollback ownership is ambiguous for new object {item.canonical_name}"
                    )
            elif canonical_uuid == item.target_uuid and legacy_uuid == item.source_uuid:
                if failed_uuid is not None:
                    raise MigrationError(f"Rollback failed target already exists: {failed_name}")
                self._command(
                    f"RENAME TABLE {self.database}.{item.canonical_name} TO "
                    f"{self.database}.{failed_name}"
                )
                if (
                    self._object_uuid_or_none(failed_name) != item.target_uuid
                    or self._object_uuid_or_none(item.canonical_name) is not None
                ):
                    raise MigrationError(
                        f"Rollback target rename verification failed for {item.canonical_name}"
                    )
                self._checkpoint(f"rollback:target:{item.canonical_name}", MigrationPhase.ROLLING_BACK)
                self._command(
                    f"RENAME TABLE {self.database}.{item.legacy_name} TO "
                    f"{self.database}.{item.canonical_name}"
                )
                if self._object_uuid_or_none(item.canonical_name) != item.source_uuid:
                    raise MigrationError(
                        f"Rollback source rename verification failed for {item.canonical_name}"
                    )
            elif (
                item.source_uuid is not None
                and canonical_uuid is None
                and legacy_uuid == item.source_uuid
                and failed_uuid == item.target_uuid
            ):
                self._command(
                    f"RENAME TABLE {self.database}.{item.legacy_name} TO "
                    f"{self.database}.{item.canonical_name}"
                )
            elif not (
                canonical_uuid == item.source_uuid
                and legacy_uuid is None
                and failed_uuid == item.target_uuid
            ):
                raise MigrationError(
                    f"Rollback ownership is ambiguous for {item.canonical_name}"
                )
            self._checkpoint(f"rollback:{item.canonical_name}", MigrationPhase.ROLLING_BACK)
        for name, metadata in self.checkpoint.validation_summary.get(
            "legacy_only_objects", {}
        ).items():
            rollback_name = str(metadata["rollback_name"])
            source_uuid = str(metadata["uuid"])
            current_uuid = self._object_uuid_or_none(str(name))
            rollback_uuid = self._object_uuid_or_none(rollback_name)
            if current_uuid is None and rollback_uuid == source_uuid:
                self._command(
                    f"RENAME TABLE {self.database}.{rollback_name} TO "
                    f"{self.database}.{name}"
                )
                if self._object_uuid_or_none(str(name)) != source_uuid:
                    raise MigrationError(f"Legacy-only rollback rename verification failed for {name}")
            elif not (current_uuid == source_uuid and rollback_uuid is None):
                raise MigrationError(
                    f"Legacy-only rollback ownership is ambiguous for {name}"
                )
            self._checkpoint(f"rollback_legacy:{name}", MigrationPhase.ROLLING_BACK)
        saved_views = self.checkpoint.validation_summary.get("legacy_view_ddl", {})
        for view in LEGACY_VIEWS:
            ddl = saved_views.get(view)
            if ddl and self._object_uuid_or_none(view) is None:
                self._command(str(ddl))
                if self._object_uuid_or_none(view) is None:
                    raise MigrationError(f"Rollback view restore verification failed for {view}")
                self._checkpoint(f"rollback:view:{view}", MigrationPhase.ROLLING_BACK)
        rollback_errors = []
        for item in self.checkpoint.objects:
            actual = self._object_uuid_or_none(item.canonical_name)
            if actual != item.source_uuid:
                rollback_errors.append(
                    f"{item.canonical_name} UUID {actual!r} != source {item.source_uuid!r}"
                )
        for view in saved_views:
            if self._object_uuid_or_none(view) is None:
                rollback_errors.append(f"view was not restored: {view}")
        for name, metadata in self.checkpoint.validation_summary.get(
            "legacy_only_objects", {}
        ).items():
            if self._object_uuid_or_none(str(name)) != str(metadata["uuid"]):
                rollback_errors.append(f"legacy-only table was not restored: {name}")
        rollback_report = {
            "status": "passed" if not rollback_errors else "failed",
            "errors": rollback_errors,
        }
        if rollback_errors:
            raise MigrationError("Rollback validation failed: " + "; ".join(rollback_errors))
        self._checkpoint(
            "rollback",
            MigrationPhase.ROLLED_BACK,
            is_cutover=False,
            is_rolled_back=True,
            validation_summary={
                **self.checkpoint.validation_summary,
                "rollback": rollback_report,
            },
        )
        self.logger.info(
            "migration_rollback migration_id=%s database=%s action=complete",
            self.migration_id,
            self.database,
        )

    def cleanup_legacy(
        self,
        confirm_database: Optional[str],
        *,
        waive_soak: bool = False,
    ) -> Mapping[str, int]:
        self._require_confirmation(confirm_database, "cleanup-legacy")
        if self.checkpoint.phase is MigrationPhase.CLEANED:
            return {}
        self._ensure_lease(MigrationPhase.CLEANING)
        if not self.checkpoint.is_cutover or self.checkpoint.is_rolled_back:
            raise MigrationError("Legacy cleanup requires an active successful cutover")
        soak_days = int(self.context.get("options_chain.migration.soak_days", 14))
        cutover_at = _parse_timestamp(self.checkpoint.cutover_completed_at)
        if (
            not waive_soak
            and (cutover_at is None or _utc_now() < cutover_at + timedelta(days=soak_days))
        ):
            raise MigrationError("Configured migration soak period has not elapsed")
        if waive_soak:
            self.logger.warning(
                "migration_cleanup_soak_waived migration_id=%s database=%s "
                "configured_soak_days=%s",
                self.migration_id,
                self.database,
                soak_days,
            )
        if self.checkpoint.phase is not MigrationPhase.CLEANING:
            self._checkpoint("cleanup_started", MigrationPhase.CLEANING)
        reclaimed: dict[str, int] = {}
        for item in self.checkpoint.objects:
            actual_uuid = self._object_uuid_or_none(item.legacy_name)
            if actual_uuid is None:
                continue
            if actual_uuid != item.source_uuid:
                raise MigrationError(
                    f"Legacy cleanup UUID mismatch for {item.legacy_name}: "
                    f"{actual_uuid!r} != checkpoint {item.source_uuid!r}"
                )
            rows = self.client.query(
                "SELECT total_bytes FROM system.tables WHERE database = {database:String} "
                "AND name = {name:String}",
                parameters={"database": self.database, "name": item.legacy_name},
            ).result_rows
            if not rows:
                continue
            reclaimed[item.legacy_name] = int(rows[0][0] or 0)
            self._command(
                f"DROP TABLE {self.database}.{item.legacy_name} SYNC"
            )
            if self._object_uuid_or_none(item.legacy_name) is not None:
                raise MigrationError(f"Cleanup drop verification failed for {item.legacy_name}")
            self._checkpoint(f"cleanup:{item.legacy_name}", MigrationPhase.CLEANING)
        for metadata in self.checkpoint.validation_summary.get(
            "legacy_only_objects", {}
        ).values():
            name = str(metadata["rollback_name"])
            actual_uuid = self._object_uuid_or_none(name)
            if actual_uuid is None:
                continue
            expected_uuid = str(metadata["uuid"])
            if actual_uuid != expected_uuid:
                raise MigrationError(
                    f"Legacy cleanup UUID mismatch for {name}: "
                    f"{actual_uuid!r} != checkpoint {expected_uuid!r}"
                )
            rows = self.client.query(
                "SELECT total_bytes FROM system.tables WHERE database = {database:String} "
                "AND name = {name:String}",
                parameters={"database": self.database, "name": name},
            ).result_rows
            if not rows:
                continue
            reclaimed[name] = int(rows[0][0] or 0)
            self._command(f"DROP TABLE {self.database}.{name} SYNC")
            if self._object_uuid_or_none(name) is not None:
                raise MigrationError(f"Cleanup drop verification failed for {name}")
            self._checkpoint(f"cleanup:{name}", MigrationPhase.CLEANING)
        remaining = []
        for item in self.checkpoint.objects:
            if self._object_uuid_or_none(item.legacy_name) is not None:
                remaining.append(item.legacy_name)
        if remaining:
            raise MigrationError("Cleanup verification failed: " + ", ".join(remaining))
        self._checkpoint("cleanup", MigrationPhase.CLEANED)
        return reclaimed

    def resume(self, confirm_database: Optional[str] = None) -> None:
        """Resume from durable step boundaries without widening destructive scope."""
        completed = set(self.checkpoint.completed_steps)
        if self.checkpoint.phase in {
            MigrationPhase.CUTOVER, MigrationPhase.ROLLED_BACK,
            MigrationPhase.CLEANED,
        }:
            return
        if self.checkpoint.phase is MigrationPhase.FAILED:
            failed_from = self.checkpoint.validation_summary.get(
                "failed_from_phase", MigrationPhase.NEW.value
            )
            try:
                resume_phase = MigrationPhase(str(failed_from))
            except ValueError:
                resume_phase = MigrationPhase.NEW
            if resume_phase is MigrationPhase.FAILED:
                resume_phase = MigrationPhase.NEW
            self._checkpoint(
                "resume_after_failure",
                resume_phase,
                validation_summary={
                    **self.checkpoint.validation_summary,
                    "resumed_from_failed_phase": str(failed_from),
                },
            )
            self.resume(confirm_database)
            return
        if self.checkpoint.phase in {
            MigrationPhase.PREPARING_CUTOVER, MigrationPhase.CUTTING_OVER,
            MigrationPhase.CUTOVER_RENAMES_COMPLETE,
        }:
            self.cutover(confirm_database)
            return
        if self.checkpoint.phase in {MigrationPhase.ROLLING_BACK}:
            self.rollback(confirm_database)
            return
        if self.checkpoint.phase is MigrationPhase.CLEANING:
            self.cleanup_legacy(confirm_database)
            return
        self.offline_preflight()
        if not self.checkpoint.objects or "plan" not in completed:
            self.plan()
        if "target_schema" not in completed:
            self.create_target_schema()
        if "migrate_sources" not in completed:
            self.migrate_sources()
        if "migrate_calculations" not in completed:
            self.migrate_calculations()
        if "migrate_surfaces" not in completed:
            self.migrate_surfaces()
        if "archive_raw" not in completed:
            self.archive_raw()
        if "validate" not in completed:
            self.validate()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Resumable options schema migration") +
        " Commands are mutually exclusive; destructive commands require exact database confirmation.",
        epilog=(
            "Exit codes: 0 success/no-op, 3 configuration error, 4 blocked by a live "
            "lease or preflight condition, 6 migration failure."
        ),
    )
    parser.add_argument("-e", required=True, help="configuration environment name")
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--plan", action="store_true", help="inspect and report a non-destructive plan")
    commands.add_argument("--create-target-schema", action="store_true")
    commands.add_argument("--migrate-sources", action="store_true")
    commands.add_argument("--migrate-calculations", action="store_true")
    commands.add_argument("--migrate-surfaces", action="store_true")
    commands.add_argument("--migrate-final-raw-delta", action="store_true")
    commands.add_argument("--archive-raw", action="store_true")
    commands.add_argument("--validate", action="store_true")
    commands.add_argument("--cutover", action="store_true")
    commands.add_argument("--rollback", action="store_true")
    commands.add_argument("--cleanup-legacy", action="store_true")
    commands.add_argument("--resume", action="store_true")
    commands.add_argument("--all", action="store_true", help="run all non-destructive phases through validation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-database", help="exact database name for destructive commands")
    parser.add_argument(
        "--waive-soak",
        action="store_true",
        help=(
            "waive the configured soak period for --cleanup-legacy; still requires "
            "exact database confirmation"
        ),
    )
    parser.add_argument(
        "--takeover-stale-lease", action="store_true",
        help="explicitly take over an expired migration lease",
    )
    parser.add_argument("--json", action="store_true", dest="json_output", help="emit machine-readable reports")
    parser.add_argument("--migration-id", type=uuid.UUID, required=True)
    parser.add_argument("--archive-directory", type=Path, required=True)
    options = parser.parse_args(argv)
    if options.dry_run and (options.cutover or options.rollback or options.cleanup_legacy):
        parser.error("--dry-run is not supported with --cutover, --rollback, or --cleanup-legacy")
    if options.resume and options.dry_run:
        parser.error("--dry-run is not supported with --resume")
    if options.waive_soak and not options.cleanup_legacy:
        parser.error("--waive-soak is only valid with --cleanup-legacy")
    return options


def default_context(algo_home: str) -> dict[str, Any]:
    return {
        PARAM_ALGO_HOME: algo_home,
        "options_chain.clickhouse.host": "localhost",
        "options_chain.clickhouse.port": 8123,
        "options_chain.clickhouse.username": "default",
        "options_chain.clickhouse.password": "",
        "options_chain.clickhouse.database": "algotrading",
        "options_chain.clickhouse.secure": False,
        "options_chain.migration.attempt_retention_days": 30,
        "options_chain.migration.diagnostics_retention_days": 30,
        "options_chain.migration.validation_retention_days": 30,
        "options_chain.migration.numerical_parity_sample_size": 512,
        "options_chain.migration.soak_days": 14,
        "options_chain.migration.lease_stale_seconds": 3600,
        "options_chain.migration.calculation_stale_seconds": 3600,
        "options_chain.migration.surface_stale_seconds": 3600,
        "options_chain.migration.archive_safety_margin": 1.25,
        "options_chain.gui_benchmark.repetitions": 3,
        "options_chain.gui_benchmark.warm_target_seconds": 0.250,
        "options_chain.gui_benchmark.cold_target_seconds": 1.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = _parse_args(argv)
    algo_home = os.getenv("ALGO_HOME")
    if not algo_home:
        print("configuration error: ALGO_HOME is not set; set it to the repository/configuration root", file=sys.stderr)
        return 3
    logger = logging.getLogger("OptionsChainSchemaMigration")
    command = next(
        name for name in (
            "plan", "create_target_schema", "migrate_sources", "migrate_calculations",
            "migrate_surfaces", "migrate_final_raw_delta", "archive_raw", "validate", "cutover", "rollback",
            "cleanup_legacy", "resume", "all",
        ) if getattr(options, name)
    )
    try:
        # Migration output is retained for operator review.  Keep configuration
        # values (including identifiers and credentials) out of that audit log.
        context = load_configuration(
            default_context(algo_home), algo_home, options.e, quiet=True
        )
        context[PARAM_LOGGER] = logger
        client = create_clickhouse_client(context)
        migration: Optional[MigrationCoordinator] = None
        try:
            reporter = MigrationReporter(
                str(options.migration_id),
                str(context["options_chain.clickhouse.database"]),
                options.archive_directory / f"options_chain_migration_{options.migration_id}.json",
                json_output=options.json_output,
            )
            migration = MigrationCoordinator(
                context,
                client,
                options.migration_id,
                options.archive_directory,
                logger=logger,
                dry_run=options.dry_run,
                reporter=reporter,
                allow_stale_lease_takeover=options.takeover_stale_lease,
            )
            phase_before = migration.checkpoint.phase.value
            reporter.start(command, migration.checkpoint.phase, options.dry_run)
            if options.plan or options.all:
                # A dry run is report-only and must not be blocked by active work.
                # Real execution remains fail-closed until all readers/writers and
                # logical attempts are offline.
                migration.offline_preflight(
                    strict=not options.dry_run,
                    persist=bool(options.all and not options.dry_run),
                )
                migration.plan()
                plan_summary = migration.plan_report()
                reporter.plan_report(plan_summary)
                if options.plan:
                    reporter.finish(
                        "BLOCKED" if plan_summary.get("blockers") else "SUCCESS",
                        phase_before, migration.checkpoint.phase.value,
                        database_changes=False, recommended_next_command="--all",
                    )
                    return 0
            if options.create_target_schema or options.all:
                migration.create_target_schema()
            if options.migrate_sources or options.all:
                migration.migrate_sources()
            if options.migrate_calculations or options.all:
                migration.migrate_calculations()
            if options.migrate_surfaces or options.all:
                migration.migrate_surfaces()
            if options.migrate_final_raw_delta:
                migration.migrate_final_raw_delta()
            if options.archive_raw or options.all:
                migration.archive_raw()
            if options.all:
                migration.migrate_final_raw_delta()
            if options.validate or options.all:
                migration.validate()
            if options.cutover:
                migration.cutover(options.confirm_database)
            if options.rollback:
                migration.rollback(options.confirm_database)
            if options.cleanup_legacy:
                migration.cleanup_legacy(
                    options.confirm_database,
                    waive_soak=options.waive_soak,
                )
            if options.resume:
                migration.resume(options.confirm_database)
            migration.release_lease()
            reporter.finish(
                "NO-OP" if phase_before == migration.checkpoint.phase.value and
                migration.checkpoint.phase.value in {
                    MigrationPhase.CUTOVER.value, MigrationPhase.ROLLED_BACK.value,
                    MigrationPhase.CLEANED.value,
                } and command in {"resume", "cutover", "rollback", "cleanup_legacy"}
                else "SUCCESS",
                phase_before, migration.checkpoint.phase.value,
                rows_failed=migration._quarantine_rows,
                database_changes=not options.dry_run and command not in {"plan", "validate"},
                recommended_next_command=(
                    "--cleanup-legacy" if migration.checkpoint.phase is MigrationPhase.CUTOVER
                    else "--resume"
                ),
            )
            return 0
        except Exception as exc:
            if migration is not None and not isinstance(exc, MigrationBlockedError):
                migration.mark_failed(exc)
            elif migration is not None:
                try:
                    migration.release_lease()
                except Exception:
                    logger.exception("migration_lease_release_after_block_failed")
            raise
        finally:
            if migration is not None:
                migration.close()
            client.close()
    except MigrationBlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
        return 4
    except MigrationError as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return 6
    except (ValueError, OSError) as exc:
        print(
            f"configuration error: {exc}; verify ALGO_HOME, -e, and archive paths",
            file=sys.stderr,
            flush=True,
        )
        return 3
    except Exception:
        print("migration execution error: inspect the database/checkpoint and rerun --resume", file=sys.stderr, flush=True)
        logger.exception("migration_critical")
        return 6


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALCULATION_SUCCESS_FIELDS",
    "CalculationClassification",
    "CheckpointStore",
    "FLOAT32_CAST_THRESHOLDS",
    "LEGACY_TABLES",
    "LEGACY_VIEWS",
    "MigrationCheckpoint",
    "MigrationBlockedError",
    "MigrationCoordinator",
    "MigrationError",
    "MigrationPhase",
    "MigrationReporter",
    "ObjectPlan",
    "SOURCE_COLUMN_DISPOSITION",
    "SurfaceExpirationClassification",
    "classify_calculation_record",
    "classify_legacy_run_status",
    "canonicalize_created_table_name",
    "classify_surface_expiration",
    "legacy_name",
    "normalize_schema_sql",
    "parse_fit_parameters_json",
    "reconcile_calculation_counts",
    "reconcile_surface_counts",
    "safe_source_value",
    "schema_hash",
    "schema_mismatch_details",
    "short_migration_id",
    "target_name",
]
