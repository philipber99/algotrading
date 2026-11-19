"""Verified weekly raw archival with optional monthly options-history compaction."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import logging
import os
import socket
import sys
import tarfile
import tempfile
import time as monotonic_time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import pyarrow as pa
import zstandard as zstd

from algotrading.Common import load_configuration
from algotrading.ContextFields import PARAM_ALGO_HOME, PARAM_LOGGER
from algotrading.refdata.options.Common import (
    _as_bool,
    _compact_json,
    _parse_timestamp,
    _rows_to_arrow,
    _utc_now,
    create_clickhouse_client,
    validate_database_identifier,
)
from algotrading.refdata.options.Schema import SCHEMA_GENERATION
from algotrading.refdata.options.Runtime import surface_cadence_ctes
from algotrading.util.NagiosSystemMonitor import NagiosSystemMonitor


UTC = timezone.utc
ARCHIVE_FORMAT_VERSION = 1
PAYLOAD_QUERY_BATCH_BYTES = 64 * 1024 * 1024
PAYLOAD_QUERY_BATCH_ROWS = 128


class ExitCode(IntEnum):
    OK = 0
    WARNING = 1
    CRITICAL = 2
    UNKNOWN = 3


class ManifestState(StrEnum):
    PLANNED = "planned"
    WRITING = "writing"
    WRITTEN = "written"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    PURGING = "purging"
    PURGED = "purged"
    FAILED = "failed"
    IGNORE = "ignore"


_TRANSITIONS = {
    ManifestState.PLANNED: {
        ManifestState.WRITING, ManifestState.FAILED, ManifestState.IGNORE
    },
    ManifestState.WRITING: {
        ManifestState.WRITTEN, ManifestState.FAILED, ManifestState.IGNORE
    },
    ManifestState.WRITTEN: {ManifestState.VERIFYING, ManifestState.FAILED},
    ManifestState.VERIFYING: {ManifestState.VERIFIED, ManifestState.FAILED},
    ManifestState.VERIFIED: {ManifestState.PURGING, ManifestState.FAILED},
    ManifestState.PURGING: {ManifestState.PURGED, ManifestState.FAILED},
    ManifestState.PURGED: set(),
    ManifestState.IGNORE: set(),
    ManifestState.FAILED: {
        ManifestState.WRITING,
        ManifestState.VERIFYING,
        ManifestState.PURGING,
        ManifestState.IGNORE,
    },
}


@dataclass(frozen=True)
class ArchiveConfig:
    output_directory: Path
    lock_filename: str = ".options_chain_archive.lock"
    minimum_age_days: int = 14
    retain_raw_weeks: int = 2
    zstd_level: int = 22
    zstd_long_window_log: int = 31
    compression_threads: int = 0
    require_surface_completion: bool = True
    surface_cadence_multiplier: int = 1
    allow_complete_with_failures: bool = True
    diagnostics_retention_days: int = 30
    validation_retention_days: int = 30
    attempt_retention_days: int = 30
    monitor_service_name: str = "OptionsChainArchival"
    shared_filesystem: bool = True
    lease_seconds: int = 300

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "ArchiveConfig":
        prefix = "options_chain.archive."
        value = cls(
            output_directory=Path(str(context.get(prefix + "output_directory", ""))),
            lock_filename=str(
                context.get(prefix + "lock_filename", ".options_chain_archive.lock")
            ),
            minimum_age_days=int(context.get(prefix + "minimum_age_days", 14)),
            retain_raw_weeks=int(context.get(prefix + "retain_raw_weeks", 2)),
            zstd_level=int(context.get(prefix + "zstd_level", 22)),
            zstd_long_window_log=int(
                context.get(prefix + "zstd_long_window_log", 31)
            ),
            compression_threads=int(
                context.get(prefix + "compression_threads", 0)
            ),
            require_surface_completion=_as_bool(
                context.get(prefix + "require_surface_completion", True)
            ),
            surface_cadence_multiplier=int(
                context.get("options_calcs.surface.cadence_multiplier", 1)
            ),
            allow_complete_with_failures=_as_bool(
                context.get(prefix + "allow_complete_with_failures", True)
            ),
            diagnostics_retention_days=int(
                context.get(prefix + "diagnostics_retention_days", 30)
            ),
            validation_retention_days=int(
                context.get(prefix + "validation_retention_days", 30)
            ),
            attempt_retention_days=int(
                context.get(prefix + "attempt_retention_days", 30)
            ),
            monitor_service_name=str(
                context.get(prefix + "monitor_service_name", "OptionsChainArchival")
            ),
            shared_filesystem=_as_bool(
                context.get(prefix + "shared_filesystem", True)
            ),
            lease_seconds=int(context.get(prefix + "lease_seconds", 300)),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if not str(self.output_directory):
            raise ValueError("options_chain.archive.output_directory is required")
        if Path(self.lock_filename).name != self.lock_filename:
            raise ValueError("archive lock_filename must be a simple filename")
        if self.minimum_age_days < 0 or self.retain_raw_weeks < 0:
            raise ValueError("archive retention values cannot be negative")
        if not 1 <= self.zstd_level <= 22:
            raise ValueError("archive zstd_level must be between 1 and 22")
        if not 10 <= self.zstd_long_window_log <= 31:
            raise ValueError("archive zstd_long_window_log must be between 10 and 31")
        if self.compression_threads < 0:
            raise ValueError("archive compression_threads cannot be negative")
        if self.surface_cadence_multiplier <= 0:
            raise ValueError("surface cadence multiplier must be a positive integer")
        if (
            self.diagnostics_retention_days < 0
            or self.validation_retention_days < 0
            or self.attempt_retention_days < 0
        ):
            raise ValueError("archive TTL retention cannot be negative")
        if self.lease_seconds < 30:
            raise ValueError("archive lease_seconds must be at least 30")
        if not self.shared_filesystem:
            raise ValueError(
                "archive output directory must be shared by every possible runner"
            )
        if not self.output_directory.exists() or not self.output_directory.is_dir():
            raise ValueError(
                f"archive output directory does not exist: {self.output_directory}"
            )
        if not os.access(self.output_directory, os.W_OK):
            raise ValueError(
                f"archive output directory is not writable: {self.output_directory}"
            )


@dataclass(frozen=True)
class RawArchiveRecord:
    snapshot_id: str
    source: str
    symbol: str
    request_started_at_utc: datetime
    response_received_at_utc: datetime
    collected_at_utc: datetime
    response_latency_ms: int
    request_parameters_json: str
    payload_sha256: bytes
    payload_byte_length: int
    source_contract_count: int
    is_delayed: bool
    is_chain_truncated: bool
    status: str

    @property
    def tar_path(self) -> str:
        return f"responses/{self.snapshot_id}.json"

    def json_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["payload_sha256"] = self.payload_sha256.hex()
        for name in (
            "request_started_at_utc",
            "response_received_at_utc",
            "collected_at_utc",
        ):
            value[name] = getattr(self, name).isoformat(timespec="microseconds")
        value["tar_path"] = self.tar_path
        return value


@dataclass(frozen=True)
class RestoredRawRecord:
    """One verified raw row recovered from an archive without JSON rewriting."""

    metadata: RawArchiveRecord
    response_json: bytes

    def insert_row(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.metadata.snapshot_id,
            "source": self.metadata.source,
            "symbol": self.metadata.symbol,
            "request_started_at_utc": self.metadata.request_started_at_utc,
            "response_received_at_utc": self.metadata.response_received_at_utc,
            "collected_at_utc": self.metadata.collected_at_utc,
            "response_latency_ms": self.metadata.response_latency_ms,
            "request_parameters_json": self.metadata.request_parameters_json,
            "response_json": self.response_json,
            "payload_sha256": self.metadata.payload_sha256,
            "source_contract_count": self.metadata.source_contract_count,
            "is_delayed": self.metadata.is_delayed,
            "is_chain_truncated": self.metadata.is_chain_truncated,
            "status": self.metadata.status,
        }


@dataclass(frozen=True)
class ArchiveManifest:
    archive_id: uuid.UUID
    archive_kind: str
    range_start: datetime
    range_end: datetime
    schema_generation: int
    dataset_generation_id: uuid.UUID
    state: ManifestState
    archive_path: str
    semantic_content_hash: Optional[bytes]
    archive_file_hash: Optional[bytes]
    row_count: int
    distinct_snapshot_count: int
    uncompressed_bytes: int
    compressed_bytes: int
    compression_settings_json: str
    owner_id: str
    lease_expires_at: Optional[datetime]
    attempt_count: int
    created_at: datetime
    written_at: Optional[datetime]
    verified_at: Optional[datetime]
    purged_at: Optional[datetime]
    last_error: str
    updated_at: datetime


MANIFEST_ARROW_SCHEMA = pa.schema(
    [
        pa.field("archive_id", pa.string(), nullable=False),
        pa.field("archive_kind", pa.string(), nullable=False),
        pa.field("range_start", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("range_end", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("schema_generation", pa.uint16(), nullable=False),
        pa.field("dataset_generation_id", pa.string(), nullable=False),
        pa.field("state", pa.string(), nullable=False),
        pa.field("archive_path", pa.string(), nullable=False),
        pa.field("semantic_content_hash", pa.binary(32)),
        pa.field("archive_file_hash", pa.binary(32)),
        pa.field("row_count", pa.uint64(), nullable=False),
        pa.field("distinct_snapshot_count", pa.uint64(), nullable=False),
        pa.field("uncompressed_bytes", pa.uint64(), nullable=False),
        pa.field("compressed_bytes", pa.uint64(), nullable=False),
        pa.field("compression_settings_json", pa.string(), nullable=False),
        pa.field("owner_id", pa.string(), nullable=False),
        pa.field("lease_expires_at", pa.timestamp("us", tz="UTC")),
        pa.field("attempt_count", pa.uint16(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("written_at", pa.timestamp("us", tz="UTC")),
        pa.field("verified_at", pa.timestamp("us", tz="UTC")),
        pa.field("purged_at", pa.timestamp("us", tz="UTC")),
        pa.field("last_error", pa.string(), nullable=False),
        pa.field("updated_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def week_interval(week_start: date) -> tuple[datetime, datetime]:
    if week_start.weekday() != 0:
        raise ValueError("--week-start must be a Monday")
    start = datetime.combine(week_start, time.min, UTC)
    return start, start + timedelta(days=7)


def _retention_boundaries(
    now: datetime,
    *,
    minimum_age_days: int,
    retain_raw_weeks: int,
) -> tuple[datetime, datetime]:
    """Return the age cutoff and newest-retained-week boundary."""
    now = _parse_timestamp(now)
    if now is None:
        raise ValueError("now is invalid")
    age_cutoff = now - timedelta(days=minimum_age_days)
    current_week_start = datetime.combine(
        (now.date() - timedelta(days=now.weekday())), time.min, UTC
    )
    newest_retained_boundary = current_week_start - timedelta(
        weeks=retain_raw_weeks
    )
    return age_cutoff, newest_retained_boundary


def week_is_age_eligible(
    week_start: date,
    now: datetime,
    *,
    minimum_age_days: int,
    retain_raw_weeks: int,
) -> bool:
    """Apply both the age cutoff and newest-closed-week retention boundary."""
    _, end = week_interval(week_start)
    age_cutoff, newest_retained_boundary = _retention_boundaries(
        now,
        minimum_age_days=minimum_age_days,
        retain_raw_weeks=retain_raw_weeks,
    )
    return end <= age_cutoff and end <= newest_retained_boundary


def _retention_ineligibility_reason(
    week_start: date,
    *,
    age_cutoff: datetime,
    newest_retained_boundary: datetime,
) -> str:
    """Explain which retention boundary excludes a raw week."""
    _, end = week_interval(week_start)
    too_young = end > age_cutoff
    retained = end > newest_retained_boundary
    if too_young and retained:
        return "minimum_age_and_retained_week_boundary"
    if too_young:
        return "minimum_age"
    if retained:
        return "retained_week_boundary"
    return "eligible"


@dataclass(frozen=True)
class RawWeekInventory:
    week_start: date
    row_count: int
    oldest_collected_at: datetime
    newest_collected_at: datetime


def _raw_week_inventory(
    client: Any,
    database: str,
    raw_table: str,
) -> tuple[RawWeekInventory, ...]:
    """Count raw rows by weekly archive partition without reading payloads."""
    database = validate_database_identifier(database)
    raw_table = validate_database_identifier(raw_table)
    result = client.query(
        f"""
        SELECT toMonday(toDate(collected_at_utc)) AS week_start,
               count() AS row_count,
               min(collected_at_utc) AS oldest_collected_at,
               max(collected_at_utc) AS newest_collected_at
        FROM {database}.{raw_table}
        GROUP BY week_start
        ORDER BY week_start
        """
    )
    inventory = []
    for row in result.result_rows:
        raw_week = row[0]
        if isinstance(raw_week, datetime):
            week_start = raw_week.date()
        elif isinstance(raw_week, date):
            week_start = raw_week
        else:
            week_start = date.fromisoformat(str(raw_week))
        oldest = _parse_timestamp(row[2])
        newest = _parse_timestamp(row[3])
        if oldest is None or newest is None:
            raise RuntimeError(
                f"Raw week inventory returned invalid timestamps for {week_start}"
            )
        inventory.append(
            RawWeekInventory(
                week_start=week_start,
                row_count=int(row[1]),
                oldest_collected_at=oldest,
                newest_collected_at=newest,
            )
        )
    return tuple(inventory)


class ArchiveLock:
    def __init__(self, path: Path):
        self.path = path
        self._file: Optional[io.TextIOWrapper] = None

    def __enter__(self) -> "ArchiveLock":
        self._file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file.close()
            self._file = None
            raise RuntimeError(f"Another archive runner owns {self.path}")
        self._file.seek(0)
        self._file.truncate()
        self._file.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        return self

    def __exit__(self, *_: Any) -> None:
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None


def validate_transition(previous: ManifestState, target: ManifestState) -> None:
    if target not in _TRANSITIONS[previous]:
        raise ValueError(f"Invalid archive state transition: {previous} -> {target}")


class ManifestRepository:
    def __init__(
        self,
        client: Any,
        database: str,
        table: str = "options_chain_archive_manifest",
    ):
        self.client = client
        self.database = validate_database_identifier(database)
        self.table = validate_database_identifier(table)

    @staticmethod
    def _from_row(row: Mapping[str, Any]) -> ArchiveManifest:
        optional_uuid = row.get("dataset_generation_id")
        return ArchiveManifest(
            archive_id=uuid.UUID(str(row["archive_id"])),
            archive_kind=str(row["archive_kind"]),
            range_start=_parse_timestamp(row["range_start"]),
            range_end=_parse_timestamp(row["range_end"]),
            schema_generation=int(row["schema_generation"]),
            dataset_generation_id=uuid.UUID(str(optional_uuid)),
            state=ManifestState(str(row["state"])),
            archive_path=str(row["archive_path"]),
            semantic_content_hash=(
                bytes(row["semantic_content_hash"])
                if row.get("semantic_content_hash") is not None
                else None
            ),
            archive_file_hash=(
                bytes(row["archive_file_hash"])
                if row.get("archive_file_hash") is not None
                else None
            ),
            row_count=int(row["row_count"]),
            distinct_snapshot_count=int(row["distinct_snapshot_count"]),
            uncompressed_bytes=int(row["uncompressed_bytes"]),
            compressed_bytes=int(row["compressed_bytes"]),
            compression_settings_json=str(row["compression_settings_json"]),
            owner_id=str(row["owner_id"]),
            lease_expires_at=_parse_timestamp(row.get("lease_expires_at")),
            attempt_count=int(row["attempt_count"]),
            created_at=_parse_timestamp(row["created_at"]),
            written_at=_parse_timestamp(row.get("written_at")),
            verified_at=_parse_timestamp(row.get("verified_at")),
            purged_at=_parse_timestamp(row.get("purged_at")),
            last_error=str(row["last_error"]),
            updated_at=_parse_timestamp(row["updated_at"]),
        )

    def load(
        self,
        start: datetime,
        end: datetime,
        *,
        kind: str = "raw_week",
        include_ignored: bool = False,
    ) -> Optional[ArchiveManifest]:
        if kind not in {"raw_week", "normalized_month"}:
            raise ValueError(f"Unsupported archive manifest kind: {kind}")
        table = self.client.query_arrow(
            f"SELECT toString(archive_id) AS archive_id, "
            "toString(archive_kind) AS archive_kind, range_start, "
            "range_end, schema_generation, "
            "toString(dataset_generation_id) AS dataset_generation_id, "
            "toString(state) AS state, archive_path, semantic_content_hash, "
            "archive_file_hash, row_count, distinct_snapshot_count, uncompressed_bytes, "
            "compressed_bytes, compression_settings_json, owner_id, lease_expires_at, "
            "attempt_count, created_at, written_at, verified_at, purged_at, last_error, "
            f"updated_at FROM {self.database}.{self.table} FINAL "
            "WHERE archive_kind = {kind:String} "
            "AND range_start = {start:DateTime64(6, 'UTC')} "
            "AND range_end = {end:DateTime64(6, 'UTC')} LIMIT 1",
            parameters={"start": start, "end": end, "kind": kind},
        )
        rows = table.to_pylist()
        manifest = self._from_row(rows[0]) if rows else None
        if manifest is not None and manifest.state is ManifestState.IGNORE and not include_ignored:
            return None
        return manifest

    def save(self, manifest: ArchiveManifest) -> None:
        row = asdict(manifest)
        row.update(
            archive_id=str(manifest.archive_id),
            dataset_generation_id=str(manifest.dataset_generation_id),
            state=manifest.state.value,
        )
        self.client.insert_arrow(
            f"{self.database}.{self.table}",
            _rows_to_arrow([row], MANIFEST_ARROW_SCHEMA),
        )

    def transition(
        self,
        manifest: ArchiveManifest,
        target: ManifestState,
        *,
        now: Optional[datetime] = None,
        **updates: Any,
    ) -> ArchiveManifest:
        validate_transition(manifest.state, target)
        now = _parse_timestamp(now or _utc_now())
        values = {"state": target, "updated_at": now, **updates}
        if target is ManifestState.WRITTEN:
            values.setdefault("written_at", now)
        elif target is ManifestState.VERIFIED:
            values.setdefault("verified_at", now)
        elif target is ManifestState.PURGED:
            values.setdefault("purged_at", now)
        value = replace(manifest, **values)
        self.save(value)
        return value


@dataclass(frozen=True)
class Preflight:
    records: tuple[RawArchiveRecord, ...]
    blocking_counts: Mapping[str, int]
    oldest_backlog_at: Optional[datetime]

    @property
    def eligible(self) -> bool:
        return bool(self.records) and not any(self.blocking_counts.values())


@dataclass(frozen=True)
class PurgeOnlyResult:
    """Outcome of an explicit raw-data purge without an archive file."""

    row_count: int
    ttl_tables_purged: tuple[str, ...]


def _raw_records(
    client: Any,
    database: str,
    start: datetime,
    end: datetime,
    *,
    raw_table: str = "options_chain_raw",
) -> tuple[RawArchiveRecord, ...]:
    raw_table = validate_database_identifier(raw_table)
    result = client.query(
        f"""
        SELECT toString(snapshot_id), source, symbol, request_started_at_utc,
               response_received_at_utc, collected_at_utc, response_latency_ms,
               request_parameters_json, payload_sha256, length(response_json),
               source_contract_count, is_delayed, is_chain_truncated, status
        FROM {database}.{raw_table}
        PREWHERE collected_at_utc >= {{start:DateTime64(6, 'UTC')}}
             AND collected_at_utc < {{end:DateTime64(6, 'UTC')}}
        ORDER BY symbol, collected_at_utc, snapshot_id
        """,
        parameters={"start": start, "end": end},
    )
    records = tuple(
        RawArchiveRecord(
            snapshot_id=str(row[0]),
            source=str(row[1]),
            symbol=str(row[2]),
            request_started_at_utc=_parse_timestamp(row[3]),
            response_received_at_utc=_parse_timestamp(row[4]),
            collected_at_utc=_parse_timestamp(row[5]),
            response_latency_ms=int(row[6]),
            request_parameters_json=str(row[7]),
            payload_sha256=bytes(row[8]),
            payload_byte_length=int(row[9]),
            source_contract_count=int(row[10]),
            is_delayed=bool(row[11]),
            is_chain_truncated=bool(row[12]),
            status=str(row[13]),
        )
        for row in result.result_rows
    )
    if len({record.snapshot_id for record in records}) != len(records):
        raise RuntimeError("Raw archive interval contains duplicate snapshot IDs")
    return records


def archive_preflight(
    client: Any,
    database: str,
    start: datetime,
    end: datetime,
    *,
    require_surface_completion: bool,
    surface_cadence_multiplier: int = 1,
    allow_complete_with_failures: bool,
    table_names: Optional[Mapping[str, str]] = None,
) -> Preflight:
    database = validate_database_identifier(database)
    table_names = dict(table_names or {})
    def table(canonical: str) -> str:
        return validate_database_identifier(table_names.get(canonical, canonical))
    raw = table("options_chain_raw")
    snapshots = table("options_chain_snapshots")
    observations = table("options_chain_observations")
    details = table("options_chain_observation_details")
    contracts = table("options_contracts")
    definitions = table("options_chain_definitions")
    results = table("options_chain_calculation_results")
    calculation_failures = table("options_chain_calculation_failures")
    calculation_state = table("options_chain_calculation_state")
    attempts = table("options_chain_processing_attempts")
    surface_fits = table("options_chain_surface_fits")
    surface_nodes = table("options_chain_surface_nodes")
    surface_failures = table("options_chain_surface_failures")
    surface_state = table("options_chain_surface_state")
    records = _raw_records(client, database, start, end, raw_table=raw)
    terminal = ["complete"]
    if allow_complete_with_failures:
        terminal.append("complete_with_failures")
    if surface_cadence_multiplier <= 0:
        raise ValueError("surface cadence multiplier must be a positive integer")
    if surface_cadence_multiplier == 1:
        surface_backlog_query = f"""
            SELECT count(), minOrNull(s.source_collected_at)
            FROM {database}.{snapshots} AS s
            LEFT JOIN (SELECT * FROM {database}.{surface_state} FINAL) AS u
              ON u.snapshot_id = s.snapshot_id
             AND u.source_collected_at = s.source_collected_at
             AND u.is_active
            WHERE {{require_surface:Bool}}
              AND s.source_collected_at >= {{start:DateTime64(6, 'UTC')}}
              AND s.source_collected_at < {{end:DateTime64(6, 'UTC')}}
              AND (isNull(u.snapshot_id)
                   OR u.status NOT IN {{terminal:Array(String)}})"""
    else:
        cadence_ctes = surface_cadence_ctes(
            database,
            snapshots_table=snapshots,
            surface_state_table=surface_state,
        )
        surface_backlog_query = f"""
            WITH {cadence_ctes},
            surface_pending_candidates AS
            (
                SELECT snapshot_id, source_collected_at
                FROM surface_cadence_selected
                UNION ALL
                SELECT snapshot_id, source_collected_at
                FROM surface_cadence_segmented
                WHERE has_surface_state = 1
                  AND surface_status NOT IN {{terminal:Array(String)}}
            )
            SELECT count(), minOrNull(source_collected_at)
            FROM surface_pending_candidates
            WHERE {{require_surface:Bool}}
              AND source_collected_at >= {{start:DateTime64(6, 'UTC')}}
              AND source_collected_at < {{end:DateTime64(6, 'UTC')}}"""
    guard_queries = {
        "normalization_or_hash_mismatch": f"""
            SELECT count() FROM {database}.{raw} AS r
            LEFT JOIN {database}.{snapshots} AS s ON s.snapshot_id = r.snapshot_id
            WHERE r.collected_at_utc >= {{start:DateTime64(6, 'UTC')}}
              AND r.collected_at_utc < {{end:DateTime64(6, 'UTC')}}
              AND (isNull(s.snapshot_id) OR s.payload_sha256 != r.payload_sha256)""",
        "partial_normalization": f"""
            SELECT count() FROM {database}.{snapshots} AS s
            LEFT JOIN
              (SELECT snapshot_id, count() AS rows,
                      countDistinct(contract_id) AS distinct_contracts
               FROM {database}.{observations}
               WHERE source_collected_at >= {{start:DateTime64(6, 'UTC')}}
                 AND source_collected_at < {{end:DateTime64(6, 'UTC')}}
               GROUP BY snapshot_id) AS o ON o.snapshot_id = s.snapshot_id
            LEFT JOIN
              (SELECT snapshot_id, count() AS rows
               FROM {database}.{details}
               WHERE source_collected_at >= {{start:DateTime64(6, 'UTC')}}
                 AND source_collected_at < {{end:DateTime64(6, 'UTC')}}
               GROUP BY snapshot_id) AS d ON d.snapshot_id = s.snapshot_id
            WHERE s.source_collected_at >= {{start:DateTime64(6, 'UTC')}}
              AND s.source_collected_at < {{end:DateTime64(6, 'UTC')}}
              AND (s.stored_observation_count != coalesce(o.rows, 0)
                OR s.stored_detail_count != coalesce(d.rows, 0)
                OR s.stored_detail_count != s.stored_observation_count
                OR s.stored_distinct_contract_count != coalesce(o.distinct_contracts, 0))""",
        "missing_contracts": f"""
            SELECT count() FROM {database}.{observations} AS o
            LEFT JOIN (SELECT * FROM {database}.{contracts} FINAL) AS c ON c.contract_id = o.contract_id
            WHERE o.source_collected_at >= {{start:DateTime64(6, 'UTC')}}
              AND o.source_collected_at < {{end:DateTime64(6, 'UTC')}}
              AND isNull(c.contract_id)""",
        "calculation_backlog": f"""
            SELECT count() FROM {database}.{snapshots} AS s
            LEFT JOIN (SELECT * FROM {database}.{calculation_state} FINAL) AS c
              ON c.snapshot_id = s.snapshot_id AND c.is_active
            LEFT JOIN (SELECT * FROM {database}.{definitions} FINAL) AS d
              ON d.definition_id = c.calculation_definition_id AND d.is_active
            WHERE s.source_collected_at >= {{start:DateTime64(6, 'UTC')}}
              AND s.source_collected_at < {{end:DateTime64(6, 'UTC')}}
              AND (isNull(c.snapshot_id) OR isNull(d.definition_id)
                   OR c.status NOT IN {{terminal:Array(String)}})""",
        "calculation_count_mismatch": f"""
            SELECT count() FROM (SELECT * FROM {database}.{calculation_state} FINAL) AS c
            LEFT JOIN
              (SELECT snapshot_id,
                      source_collected_at,
                      calculation_definition_id,
                      source_calculation_run_id,
                      count() AS rows
               FROM {database}.{results}
               WHERE source_collected_at >= {{start:DateTime64(6, 'UTC')}}
                 AND source_collected_at < {{end:DateTime64(6, 'UTC')}}
               GROUP BY snapshot_id,
                        source_collected_at,
                        calculation_definition_id,
                        source_calculation_run_id) AS r
              ON r.snapshot_id = c.snapshot_id
             AND r.source_collected_at = c.source_collected_at
             AND r.calculation_definition_id = c.calculation_definition_id
             AND ((isNull(r.source_calculation_run_id)
                   AND isNull(c.source_calculation_run_id))
                  OR r.source_calculation_run_id = c.source_calculation_run_id)
            LEFT JOIN
              (SELECT snapshot_id,
                      source_collected_at,
                      calculation_definition_id,
                      source_calculation_run_id,
                      countIf(failure_scope = 'contract') AS contracts,
                      countIf(failure_scope = 'expiration') AS expirations,
                      countIf(failure_scope = 'snapshot') AS snapshots
               FROM {database}.{calculation_failures} FINAL
               WHERE is_active
                 AND source_collected_at >= {{start:DateTime64(6, 'UTC')}}
                 AND source_collected_at < {{end:DateTime64(6, 'UTC')}}
               GROUP BY snapshot_id,
                        source_collected_at,
                        calculation_definition_id,
                        source_calculation_run_id) AS f
              ON f.snapshot_id = c.snapshot_id
             AND f.source_collected_at = c.source_collected_at
             AND f.calculation_definition_id = c.calculation_definition_id
             AND ((isNull(f.source_calculation_run_id)
                   AND isNull(c.source_calculation_run_id))
                  OR f.source_calculation_run_id = c.source_calculation_run_id)
            WHERE c.source_collected_at >= {{start:DateTime64(6, 'UTC')}}
              AND c.source_collected_at < {{end:DateTime64(6, 'UTC')}}
              AND c.is_active AND c.status IN {{terminal:Array(String)}}
              AND (c.successful_results != coalesce(r.rows, 0)
                OR c.terminal_contract_failures != coalesce(f.contracts, 0)
                OR c.terminal_expiration_failures != coalesce(f.expirations, 0)
                OR c.snapshot_failure_count != coalesce(f.snapshots, 0)
                OR c.eligible_contracts != c.expected_observations)""",
        "active_attempts": f"""
            SELECT count() FROM
              (SELECT attempt_id,
                      argMax(status, tuple(isNotNull(finished_at),
                             ifNull(finished_at, started_at))) AS current_status,
                      argMax(finished_at, tuple(isNotNull(finished_at),
                             ifNull(finished_at, started_at))) AS current_finished_at
               FROM {database}.{attempts}
               WHERE source_collected_at >= {{start:DateTime64(6, 'UTC')}}
                 AND source_collected_at < {{end:DateTime64(6, 'UTC')}}
               GROUP BY attempt_id)
            WHERE current_status = 'running' AND isNull(current_finished_at)""",
        "surface_backlog": surface_backlog_query,
        "surface_count_mismatch": f"""
            SELECT count() FROM (SELECT * FROM {database}.{surface_state} FINAL) AS u
            LEFT JOIN
              (SELECT snapshot_id,
                      source_collected_at,
                      surface_definition_id,
                      source_surface_run_id,
                      countDistinctIf(expiration_at,
                                      price_basis IN ('mid', 'source')) AS expirations
               FROM {database}.{surface_fits}
               WHERE source_collected_at >= {{start:DateTime64(6, 'UTC')}}
                 AND source_collected_at < {{end:DateTime64(6, 'UTC')}}
               GROUP BY snapshot_id,
                        source_collected_at,
                        surface_definition_id,
                        source_surface_run_id) AS sf
              ON sf.snapshot_id = u.snapshot_id
             AND sf.source_collected_at = u.source_collected_at
             AND sf.surface_definition_id = u.surface_definition_id
             AND ((isNull(sf.source_surface_run_id)
                   AND isNull(u.source_surface_run_id))
                  OR sf.source_surface_run_id = u.source_surface_run_id)
            LEFT JOIN
              (SELECT snapshot_id,
                      source_collected_at,
                      surface_definition_id,
                      source_surface_run_id,
                      countIf(node_type = 'atm_forward' OR
                        (node_type = 'delta' AND node_value IN (0.10, 0.25, 0.40)))
                          AS required,
                      countIf(NOT (node_type = 'atm_forward' OR
                        (node_type = 'delta' AND node_value IN (0.10, 0.25, 0.40))))
                          AS optional
               FROM {database}.{surface_nodes}
               WHERE source_collected_at >= {{start:DateTime64(6, 'UTC')}}
                 AND source_collected_at < {{end:DateTime64(6, 'UTC')}}
               GROUP BY snapshot_id,
                        source_collected_at,
                        surface_definition_id,
                        source_surface_run_id) AS sn
              ON sn.snapshot_id = u.snapshot_id
             AND sn.source_collected_at = u.source_collected_at
             AND sn.surface_definition_id = u.surface_definition_id
             AND ((isNull(sn.source_surface_run_id)
                   AND isNull(u.source_surface_run_id))
                  OR sn.source_surface_run_id = u.source_surface_run_id)
            LEFT JOIN
              (SELECT snapshot_id,
                      source_collected_at,
                      surface_definition_id,
                      source_surface_run_id,
                      countIf(failure_scope = 'fit') AS fits,
                      countIf(failure_scope = 'node' AND
                        (node_type = 'atm_forward' OR
                         (node_type = 'delta' AND node_value IN (0.10, 0.25, 0.40))))
                          AS required_nodes,
                      countIf(failure_scope = 'node' AND NOT
                        (node_type = 'atm_forward' OR
                         (node_type = 'delta' AND node_value IN (0.10, 0.25, 0.40))))
                          AS optional_nodes
               FROM {database}.{surface_failures} FINAL
               WHERE is_active
                 AND source_collected_at >= {{start:DateTime64(6, 'UTC')}}
                 AND source_collected_at < {{end:DateTime64(6, 'UTC')}}
               GROUP BY snapshot_id,
                        source_collected_at,
                        surface_definition_id,
                        source_surface_run_id) AS ff
              ON ff.snapshot_id = u.snapshot_id
             AND ff.source_collected_at = u.source_collected_at
             AND ff.surface_definition_id = u.surface_definition_id
             AND ((isNull(ff.source_surface_run_id)
                   AND isNull(u.source_surface_run_id))
                  OR ff.source_surface_run_id = u.source_surface_run_id)
            WHERE {{require_surface:Bool}} AND u.is_active
              AND u.source_collected_at >= {{start:DateTime64(6, 'UTC')}}
              AND u.source_collected_at < {{end:DateTime64(6, 'UTC')}}
              AND u.status IN {{terminal:Array(String)}}
              AND (u.successful_expirations != coalesce(sf.expirations, 0)
                OR u.terminal_fit_failures != coalesce(ff.fits, 0)
                OR u.successful_required_nodes != coalesce(sn.required, 0)
                OR u.terminal_required_node_failures != coalesce(ff.required_nodes, 0)
                OR u.successful_optional_nodes != coalesce(sn.optional, 0)
                OR u.terminal_optional_node_failures != coalesce(ff.optional_nodes, 0))""",
    }
    parameters = {
        "start": start,
        "end": end,
        "terminal": terminal,
        "require_surface": require_surface_completion,
        "cadence_multiplier": surface_cadence_multiplier,
    }
    counts = {}
    oldest_candidates: list[datetime] = []
    for name, sql in guard_queries.items():
        result = client.query(sql, parameters=parameters)
        if not result.result_rows:
            raise RuntimeError(f"Archive preflight guard returned no row: {name}")
        counts[name] = int(result.result_rows[0][0] or 0)
        if name == "surface_backlog" and len(result.result_rows[0]) > 1:
            surface_oldest = result.result_rows[0][1]
            if surface_oldest is not None:
                oldest_candidates.append(_parse_timestamp(surface_oldest))
    oldest_result = client.query(
        f"SELECT minOrNull(source_collected_at) FROM {database}.{calculation_state} FINAL "
        "WHERE source_collected_at >= {start:DateTime64(6, 'UTC')} "
        "AND source_collected_at < {end:DateTime64(6, 'UTC')} "
        "AND is_active AND status NOT IN {terminal:Array(String)}",
        parameters={"start": start, "end": end, "terminal": terminal},
    )
    if oldest_result.result_rows and oldest_result.result_rows[0][0] is not None:
        oldest_candidates.append(_parse_timestamp(oldest_result.result_rows[0][0]))
    oldest = min(oldest_candidates) if oldest_candidates else None
    return Preflight(records, counts, oldest)


def semantic_content_hash(records: Sequence[RawArchiveRecord]) -> bytes:
    digest = hashlib.sha256()
    for record in records:
        encoded = json.dumps(
            record.json_record(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _archive_manifest_json(
    records: Sequence[RawArchiveRecord],
    start: datetime,
    end: datetime,
    config: ArchiveConfig,
) -> dict[str, Any]:
    return {
        "schema_version": ARCHIVE_FORMAT_VERSION,
        "schema_generation": SCHEMA_GENERATION,
        "archive_kind": "raw_week",
        "range_start": start.isoformat(timespec="microseconds"),
        "range_end": end.isoformat(timespec="microseconds"),
        "row_count": len(records),
        "distinct_snapshot_count": len({record.snapshot_id for record in records}),
        "uncompressed_bytes": sum(record.payload_byte_length for record in records),
        "compression_settings": {
            "algorithm": "zstd",
            "level": config.zstd_level,
            "long_window_log": config.zstd_long_window_log,
            "threads": config.compression_threads,
        },
        # A closed interval is used as the deterministic format timestamp.  The
        # operational creation time remains in ClickHouse manifest state.
        "created_at": end.isoformat(timespec="microseconds"),
        "semantic_content_hash": semantic_content_hash(records).hex(),
    }


def _tar_info(name: str, size: int, timestamp: int, *, mode: int = 0o640) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = timestamp
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mode = mode
    return info


def _payload_rows(
    client: Any,
    database: str,
    records: Sequence[RawArchiveRecord],
    start: datetime,
    end: datetime,
    *,
    raw_table: str = "options_chain_raw",
    batch_bytes: int = PAYLOAD_QUERY_BATCH_BYTES,
    batch_rows: int = PAYLOAD_QUERY_BATCH_ROWS,
) -> Iterator[tuple[str, bytes]]:
    """Read bounded primary-key ranges without a long-lived HTTP stream."""
    raw_table = validate_database_identifier(raw_table)
    if batch_bytes <= 0 or batch_rows <= 0:
        raise ValueError("Archive payload batch limits must be positive")
    sql = f"""
        SELECT toString(snapshot_id), response_json
        FROM {database}.{raw_table}
        PREWHERE collected_at_utc >= {{start:DateTime64(6, 'UTC')}}
             AND collected_at_utc < {{end:DateTime64(6, 'UTC')}}
        WHERE tuple(symbol, collected_at_utc, snapshot_id) >= tuple(
                  {{first_symbol:String}},
                  {{first_collected_at:DateTime64(6, 'UTC')}},
                  {{first_snapshot_id:UUID}})
          AND tuple(symbol, collected_at_utc, snapshot_id) <= tuple(
                  {{last_symbol:String}},
                  {{last_collected_at:DateTime64(6, 'UTC')}},
                  {{last_snapshot_id:UUID}})
        ORDER BY symbol, collected_at_utc, snapshot_id
    """
    offset = 0
    while offset < len(records):
        limit = offset
        total_bytes = 0
        while limit < len(records) and limit - offset < batch_rows:
            record_bytes = records[limit].payload_byte_length
            if limit > offset and total_bytes + record_bytes > batch_bytes:
                break
            total_bytes += record_bytes
            limit += 1
        batch = records[offset:limit]
        first = batch[0]
        last = batch[-1]
        parameters = {
            "start": start,
            "end": end,
            "first_symbol": first.symbol,
            "first_collected_at": first.collected_at_utc,
            "first_snapshot_id": first.snapshot_id,
            "last_symbol": last.symbol,
            "last_collected_at": last.collected_at_utc,
            "last_snapshot_id": last.snapshot_id,
        }
        rows = client.query(sql, parameters=parameters).result_rows
        expected_ids = [record.snapshot_id for record in batch]
        actual_ids = [str(row[0]) for row in rows]
        if actual_ids != expected_ids:
            raise RuntimeError(
                "Archive payload range changed while reading "
                f"{expected_ids[0]} through {expected_ids[-1]}"
            )
        for snapshot_id, payload in rows:
            yield (
                str(snapshot_id),
                payload.encode() if isinstance(payload, str) else bytes(payload),
            )
        offset = limit


def _compressor(config: ArchiveConfig) -> zstd.ZstdCompressor:
    try:
        parameters = zstd.ZstdCompressionParameters.from_level(
            config.zstd_level,
            enable_ldm=True,
            window_log=config.zstd_long_window_log,
            threads=config.compression_threads,
        )
        return zstd.ZstdCompressor(compression_params=parameters, write_checksum=True)
    except (TypeError, ValueError):
        return zstd.ZstdCompressor(
            level=config.zstd_level,
            threads=config.compression_threads,
            write_checksum=True,
        )


def write_archive(
    client: Any,
    database: str,
    records: Sequence[RawArchiveRecord],
    start: datetime,
    end: datetime,
    destination: Path,
    config: ArchiveConfig,
    *,
    raw_table: str = "options_chain_raw",
    progress_callback: Optional[Any] = None,
) -> Path:
    """Stream one deterministic tar.zst to a same-filesystem temporary path."""
    if destination.exists():
        raise FileExistsError(f"Archive destination already exists: {destination}")
    temp_path = destination.with_name(destination.name + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    logical_manifest = _archive_manifest_json(records, start, end, config)
    manifest_bytes = (
        json.dumps(logical_manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    timestamp = int(start.timestamp())
    by_id = {record.snapshot_id: record for record in records}
    try:
        with tempfile.TemporaryFile(mode="w+b") as jsonl:
            for record in records:
                jsonl.write(
                    (
                        json.dumps(
                            record.json_record(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode("utf-8")
                )
            jsonl_size = jsonl.tell()
            jsonl.seek(0)
            with temp_path.open("xb") as output:
                with _compressor(config).stream_writer(output, closefd=False) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w|") as tar:
                        tar.addfile(
                            _tar_info("manifest.json", len(manifest_bytes), timestamp),
                            io.BytesIO(manifest_bytes),
                        )
                        tar.addfile(
                            _tar_info("manifest.jsonl", jsonl_size, timestamp), jsonl
                        )
                        seen: set[str] = set()
                        last_progress = monotonic_time.monotonic()
                        for snapshot_id, payload in _payload_rows(
                            client,
                            database,
                            records,
                            start,
                            end,
                            raw_table=raw_table,
                        ):
                            record = by_id.get(snapshot_id)
                            if record is None or snapshot_id in seen:
                                raise RuntimeError(
                                    f"Archive payload stream changed at {snapshot_id}"
                                )
                            if len(payload) != record.payload_byte_length:
                                raise RuntimeError(
                                    f"Archive payload length changed for {snapshot_id}"
                                )
                            if hashlib.sha256(payload).digest() != record.payload_sha256:
                                raise RuntimeError(
                                    f"Archive payload hash mismatch for {snapshot_id}"
                                )
                            seen.add(snapshot_id)
                            tar.addfile(
                                _tar_info(record.tar_path, len(payload), timestamp),
                                io.BytesIO(payload),
                            )
                            if (
                                progress_callback is not None
                                and monotonic_time.monotonic() - last_progress >= 30.0
                            ):
                                progress_callback()
                                last_progress = monotonic_time.monotonic()
                        if seen != set(by_id):
                            raise RuntimeError("Archive payload stream is incomplete")
                output.flush()
                os.fsync(output.fileno())
        verify_archive_file(temp_path, expected_records=records)
        os.replace(temp_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return destination
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise


@dataclass(frozen=True)
class ArchiveVerification:
    manifest: Mapping[str, Any]
    file_hash: bytes
    payload_hashes: Mapping[str, bytes]


def file_sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def verify_archive_file(
    path: Path, *, expected_records: Optional[Sequence[RawArchiveRecord]] = None
) -> ArchiveVerification:
    manifest: Optional[dict[str, Any]] = None
    jsonl: list[dict[str, Any]] = []
    payload_hashes: dict[str, bytes] = {}
    payload_lengths: dict[str, int] = {}
    paths: list[str] = []
    tar_metadata: list[tuple[int, int, str, str, int, int]] = []
    with path.open("rb") as source:
        with zstd.ZstdDecompressor().stream_reader(source) as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|") as tar:
                for member in tar:
                    if not member.isfile():
                        raise RuntimeError(f"Archive contains a non-file member: {member.name}")
                    paths.append(member.name)
                    tar_metadata.append(
                        (
                            int(member.uid),
                            int(member.gid),
                            str(member.uname),
                            str(member.gname),
                            int(member.mode),
                            int(member.mtime),
                        )
                    )
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"Unable to read archive member {member.name}")
                    if member.name == "manifest.json":
                        manifest = json.loads(extracted.read())
                    elif member.name == "manifest.jsonl":
                        jsonl = [json.loads(line) for line in extracted if line.strip()]
                    elif member.name.startswith("responses/") and member.name.endswith(".json"):
                        snapshot_id = member.name.removeprefix("responses/").removesuffix(".json")
                        digest = hashlib.sha256()
                        payload_length = 0
                        for block in iter(lambda: extracted.read(1024 * 1024), b""):
                            digest.update(block)
                            payload_length += len(block)
                        if snapshot_id in payload_hashes:
                            raise RuntimeError(f"Duplicate archived payload {snapshot_id}")
                        payload_hashes[snapshot_id] = digest.digest()
                        payload_lengths[snapshot_id] = payload_length
                    else:
                        raise RuntimeError(f"Unexpected archive member: {member.name}")
    if manifest is None or paths[:2] != ["manifest.json", "manifest.jsonl"]:
        raise RuntimeError("Archive has no canonical manifest prefix")
    if int(manifest.get("schema_version", 0)) != ARCHIVE_FORMAT_VERSION:
        raise RuntimeError("Unsupported archive manifest version")
    if int(manifest.get("schema_generation", 0)) != SCHEMA_GENERATION:
        raise RuntimeError("Archive schema generation mismatch")
    if manifest.get("archive_kind") != "raw_week":
        raise RuntimeError("Archive kind is not raw_week")
    range_start = _parse_timestamp(manifest.get("range_start"))
    range_end = _parse_timestamp(manifest.get("range_end"))
    created_at = _parse_timestamp(manifest.get("created_at"))
    if (
        range_start is None
        or range_end is None
        or range_start.weekday() != 0
        or range_end - range_start != timedelta(days=7)
        or created_at != range_end
    ):
        raise RuntimeError("Archive week boundaries or deterministic creation time are invalid")
    expected_tar_metadata = (0, 0, "root", "root", 0o640, int(range_start.timestamp()))
    if any(value != expected_tar_metadata for value in tar_metadata):
        raise RuntimeError("Archive tar metadata is not deterministic")
    compression = manifest.get("compression_settings")
    if not isinstance(compression, dict) or compression.get("algorithm") != "zstd":
        raise RuntimeError("Archive compression settings are invalid")
    try:
        compression_level = int(compression["level"])
        window_log = int(compression["long_window_log"])
        compression_threads = int(compression["threads"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Archive compression settings are incomplete") from exc
    if (
        not 1 <= compression_level <= 22
        or not 10 <= window_log <= 31
        or compression_threads < 0
    ):
        raise RuntimeError("Archive compression settings are outside supported bounds")
    records = tuple(_record_from_json(row) for row in jsonl)
    if len(records) != int(manifest["row_count"]):
        raise RuntimeError("Archive manifest row count mismatch")
    if len(payload_hashes) != int(manifest["distinct_snapshot_count"]):
        raise RuntimeError("Archive payload count mismatch")
    if len({record.snapshot_id for record in records}) != len(records):
        raise RuntimeError("Archive JSONL contains duplicate snapshot IDs")
    if tuple(sorted(records, key=lambda item: (
        item.symbol, item.collected_at_utc, item.snapshot_id
    ))) != records:
        raise RuntimeError("Archive JSONL record order is not deterministic")
    if sum(record.payload_byte_length for record in records) != int(
        manifest["uncompressed_bytes"]
    ):
        raise RuntimeError("Archive uncompressed byte count mismatch")
    if paths[2:] != [record.tar_path for record in records]:
        raise RuntimeError("Archive response paths are not deterministic")
    for record in records:
        if payload_hashes.get(record.snapshot_id) != record.payload_sha256:
            raise RuntimeError(f"Archived payload digest mismatch for {record.snapshot_id}")
        if payload_lengths.get(record.snapshot_id) != record.payload_byte_length:
            raise RuntimeError(f"Archived payload length mismatch for {record.snapshot_id}")
    if semantic_content_hash(records).hex() != manifest["semantic_content_hash"]:
        raise RuntimeError("Archive semantic content hash mismatch")
    if expected_records is not None:
        expected = [item.json_record() for item in expected_records]
        actual = [item.json_record() for item in records]
        if actual != expected:
            raise RuntimeError("Archive differs from the pre-write raw snapshot/hash set")
    return ArchiveVerification(manifest, file_sha256(path), payload_hashes)


def _record_from_json(row: Mapping[str, Any]) -> RawArchiveRecord:
    return RawArchiveRecord(
        snapshot_id=str(row["snapshot_id"]),
        source=str(row["source"]),
        symbol=str(row["symbol"]),
        request_started_at_utc=_parse_timestamp(row["request_started_at_utc"]),
        response_received_at_utc=_parse_timestamp(row["response_received_at_utc"]),
        collected_at_utc=_parse_timestamp(row["collected_at_utc"]),
        response_latency_ms=int(row["response_latency_ms"]),
        request_parameters_json=str(row["request_parameters_json"]),
        payload_sha256=bytes.fromhex(str(row["payload_sha256"])),
        payload_byte_length=int(row["payload_byte_length"]),
        source_contract_count=int(row["source_contract_count"]),
        is_delayed=bool(row["is_delayed"]),
        is_chain_truncated=bool(row["is_chain_truncated"]),
        status=str(row["status"]),
    )


def read_archive_records(path: Path) -> Iterator[RestoredRawRecord]:
    """Verify an archive, then stream every exact payload with its raw metadata."""
    verification = verify_archive_file(path)
    records: dict[str, RawArchiveRecord] = {}
    seen: set[str] = set()
    with path.open("rb") as source:
        with zstd.ZstdDecompressor().stream_reader(source) as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|") as tar:
                for member in tar:
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"Unable to read archive member {member.name}")
                    if member.name == "manifest.json":
                        extracted.read()
                        continue
                    if member.name == "manifest.jsonl":
                        for line in extracted:
                            if not line.strip():
                                continue
                            record = _record_from_json(json.loads(line))
                            records[record.tar_path] = record
                        continue
                    record = records.get(member.name)
                    if record is None:
                        raise RuntimeError(
                            f"Archive response has no metadata row: {member.name}"
                        )
                    payload = extracted.read()
                    if len(payload) != record.payload_byte_length:
                        raise RuntimeError(
                            f"Archived payload length mismatch for {record.snapshot_id}"
                        )
                    if hashlib.sha256(payload).digest() != record.payload_sha256:
                        raise RuntimeError(
                            f"Archived payload digest mismatch for {record.snapshot_id}"
                        )
                    seen.add(member.name)
                    yield RestoredRawRecord(record, payload)
    if seen != set(records):
        missing = sorted(set(records) - seen)
        raise RuntimeError(f"Archive payloads missing for metadata rows: {missing[:5]}")
    if file_sha256(path) != verification.file_hash:
        raise RuntimeError("Archive file changed during restore/read")


class BacklogError(RuntimeError):
    def __init__(self, preflight: Preflight):
        self.preflight = preflight
        super().__init__(f"Archive backlog: {dict(preflight.blocking_counts)}")


class RawWeekArchiver:
    def __init__(
        self,
        context: Mapping[str, Any],
        client: Any,
        config: ArchiveConfig,
        *,
        logger: Optional[logging.Logger] = None,
        table_names: Optional[Mapping[str, str]] = None,
    ):
        self.context = context
        self.client = client
        self.config = config
        self.database = validate_database_identifier(
            str(context["options_chain.clickhouse.database"])
        )
        self.logger = logger or logging.getLogger(__name__)
        self.table_names = {
            canonical: validate_database_identifier(physical)
            for canonical, physical in dict(table_names or {}).items()
        }
        self.raw_table = self.table_names.get(
            "options_chain_raw", "options_chain_raw"
        )
        self.manifest_table = self.table_names.get(
            "options_chain_archive_manifest", "options_chain_archive_manifest"
        )
        self.owner_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"
        self.repository = ManifestRepository(
            client, self.database, self.manifest_table
        )

    def _path(self, start: datetime, end: datetime) -> Path:
        stem = (
            f"options_chain_raw_{start.date().isoformat()}_"
            f"{end.date().isoformat()}"
        )
        path = self.config.output_directory / f"{stem}.tar.zst"
        sequence = 1
        while path.exists():
            path = self.config.output_directory / f"{stem}_{sequence}.tar.zst"
            sequence += 1
        return path

    def _dataset_generation_id(self) -> uuid.UUID:
        configured = self.context.get("options_chain.dataset_generation_id")
        if configured:
            return uuid.UUID(str(configured))
        metadata_table = self.table_names.get(
            "options_chain_dataset_metadata", "options_chain_dataset_metadata"
        )
        rows = self.client.query(
            f"SELECT toString(dataset_generation_id) FROM "
            f"{self.database}.{metadata_table} FINAL WHERE is_active"
        ).result_rows
        if len(rows) != 1:
            raise RuntimeError(
                "Raw archival requires exactly one active dataset generation "
                f"(found {len(rows)})"
            )
        return uuid.UUID(str(rows[0][0]))

    def _new_manifest(self, start: datetime, end: datetime, path: Path) -> ArchiveManifest:
        now = _utc_now()
        generation_id = self._dataset_generation_id()
        return ArchiveManifest(
            archive_id=uuid.uuid4(),
            archive_kind="raw_week",
            range_start=start,
            range_end=end,
            schema_generation=SCHEMA_GENERATION,
            dataset_generation_id=generation_id,
            state=ManifestState.PLANNED,
            archive_path=str(path),
            semantic_content_hash=None,
            archive_file_hash=None,
            row_count=0,
            distinct_snapshot_count=0,
            uncompressed_bytes=0,
            compressed_bytes=0,
            compression_settings_json=_compact_json(
                {
                    "algorithm": "zstd",
                    "level": self.config.zstd_level,
                    "long_window_log": self.config.zstd_long_window_log,
                    "threads": self.config.compression_threads,
                }
            ),
            owner_id=self.owner_id,
            lease_expires_at=now + timedelta(seconds=self.config.lease_seconds),
            attempt_count=0,
            created_at=now,
            written_at=None,
            verified_at=None,
            purged_at=None,
            last_error="",
            updated_at=now,
        )

    def _race_check(self, records: Sequence[RawArchiveRecord], start: datetime, end: datetime) -> None:
        current = _raw_records(
            self.client, self.database, start, end, raw_table=self.raw_table
        )
        expected = [(row.snapshot_id, row.payload_sha256) for row in records]
        actual = [(row.snapshot_id, row.payload_sha256) for row in current]
        if actual != expected:
            raise RuntimeError("Raw snapshot/hash set changed during archival")

    def _purge_ttl_partitions(
        self, week_start: date, end: datetime
    ) -> tuple[str, ...]:
        """Explicitly retire eligible weekly operational TTL partitions."""
        now = _utc_now()
        retention = {
            "options_chain_processing_attempts": self.config.attempt_retention_days,
            "options_chain_calculation_diagnostics": self.config.diagnostics_retention_days,
            "options_chain_calculation_reuse_validation": self.config.validation_retention_days,
            "options_chain_surface_diagnostics": self.config.diagnostics_retention_days,
            "options_chain_surface_validation": self.config.validation_retention_days,
        }
        purged_tables = []
        for canonical, days in retention.items():
            if end > now - timedelta(days=days):
                continue
            physical = self.table_names.get(canonical, canonical)
            self.client.command(
                f"ALTER TABLE {self.database}.{physical} DROP PARTITION "
                "{week_start:Date}",
                parameters={"week_start": week_start},
            )
            purged_tables.append(physical)
        return tuple(purged_tables)

    def purge_only_week(
        self,
        week_start: date,
        *,
        dry_run: bool = False,
        confirm_database: Optional[str] = None,
    ) -> PurgeOnlyResult:
        """Purge a retention-eligible week without producing an archive file.

        This intentionally does not create an archive manifest: the manifest's
        verified/purged states attest to a recoverable archive file, which a
        purge-only operation explicitly does not produce.
        """
        start, end = week_interval(week_start)
        if not week_is_age_eligible(
            week_start,
            _utc_now(),
            minimum_age_days=self.config.minimum_age_days,
            retain_raw_weeks=self.config.retain_raw_weeks,
        ):
            raise ValueError(f"Raw week {week_start} is not retention-eligible")
        if confirm_database != self.database:
            raise ValueError(
                f"Destructive purge requires --confirm-database {self.database}"
            )
        lock_path = self.config.output_directory / self.config.lock_filename
        self.logger.info(
            "archive_lock operation=purge_only owner_id=%s range_start=%s "
            "range_end=%s path=%s",
            self.owner_id, start.isoformat(), end.isoformat(), lock_path,
        )
        with ArchiveLock(lock_path):
            preflight = archive_preflight(
                self.client,
                self.database,
                start,
                end,
                require_surface_completion=self.config.require_surface_completion,
                surface_cadence_multiplier=self.config.surface_cadence_multiplier,
                allow_complete_with_failures=self.config.allow_complete_with_failures,
                table_names=self.table_names,
            )
            self.logger.info(
                "archive_preflight purge_only=true range_start=%s range_end=%s "
                "eligible=%s rows=%s blockers=%s oldest_pending=%s",
                start.isoformat(), end.isoformat(), preflight.eligible,
                len(preflight.records), dict(preflight.blocking_counts),
                preflight.oldest_backlog_at,
            )
            if not preflight.eligible:
                raise BacklogError(preflight)
            if dry_run:
                return PurgeOnlyResult(len(preflight.records), ())
            # Keep the same fail-closed checks used before a verified archive is
            # purged, except for the archive-file verification that does not
            # exist in this explicitly destructive mode.
            second = archive_preflight(
                self.client,
                self.database,
                start,
                end,
                require_surface_completion=self.config.require_surface_completion,
                surface_cadence_multiplier=self.config.surface_cadence_multiplier,
                allow_complete_with_failures=self.config.allow_complete_with_failures,
                table_names=self.table_names,
            )
            self.logger.info(
                "archive_preflight second_guard=true purge_only=true "
                "range_start=%s range_end=%s eligible=%s rows=%s blockers=%s "
                "oldest_pending=%s",
                start.isoformat(), end.isoformat(), second.eligible,
                len(second.records), dict(second.blocking_counts),
                second.oldest_backlog_at,
            )
            if not second.eligible:
                raise BacklogError(second)
            self._race_check(preflight.records, start, end)
            self.client.command(
                f"ALTER TABLE {self.database}.{self.raw_table} "
                "DROP PARTITION {week_start:Date}",
                parameters={"week_start": week_start},
            )
            remaining = _raw_records(
                self.client,
                self.database,
                start,
                end,
                raw_table=self.raw_table,
            )
            if remaining:
                raise RuntimeError("Raw weekly partition still contains rows after purge")
            ttl_tables = self._purge_ttl_partitions(week_start, end)
            self.logger.info(
                "archive_purge purge_only=true range_start=%s range_end=%s "
                "rows=%s ttl_tables=%s status=purged",
                start.isoformat(), end.isoformat(), len(preflight.records),
                list(ttl_tables),
            )
            return PurgeOnlyResult(len(preflight.records), ttl_tables)

    def _archive_week_unprotected(
        self,
        week_start: date,
        *,
        dry_run: bool = False,
        purge: bool = False,
        confirm_database: Optional[str] = None,
    ) -> ArchiveManifest:
        start, end = week_interval(week_start)
        if not week_is_age_eligible(
            week_start,
            _utc_now(),
            minimum_age_days=self.config.minimum_age_days,
            retain_raw_weeks=self.config.retain_raw_weeks,
        ):
            raise ValueError(f"Raw week {week_start} is not retention-eligible")
        if purge and confirm_database != self.database:
            raise ValueError(
                f"Destructive purge requires --confirm-database {self.database}"
            )
        lock_path = self.config.output_directory / self.config.lock_filename
        self.logger.info(
            "archive_lock owner_id=%s range_start=%s range_end=%s path=%s",
            self.owner_id, start.isoformat(), end.isoformat(), lock_path,
        )
        with ArchiveLock(lock_path):
            manifest = self.repository.load(start, end)
            # A retry must keep using the path durably recorded by the first
            # attempt. New archives select the first unused numeric suffix.
            path = (
                Path(manifest.archive_path)
                if manifest is not None
                else self._path(start, end)
            )
            if manifest and manifest.state is ManifestState.PURGED:
                verification = verify_archive_file(Path(manifest.archive_path))
                if verification.file_hash != manifest.archive_file_hash:
                    raise RuntimeError("Purged archive file hash no longer matches manifest")
                self.logger.info(
                    "archive_purge no_op=true archive_id=%s range_start=%s "
                    "range_end=%s rows=%s",
                    manifest.archive_id, start.isoformat(), end.isoformat(),
                    manifest.row_count,
                )
                return manifest
            if manifest and manifest.state is ManifestState.PURGING:
                if file_sha256(Path(manifest.archive_path)) != manifest.archive_file_hash:
                    raise RuntimeError("Archive file changed while resuming purge")
                remaining = _raw_records(
                    self.client,
                    self.database,
                    start,
                    end,
                    raw_table=self.raw_table,
                )
                # DROP PARTITION may have committed immediately before the
                # runner stopped. Empty source is conclusive only after the
                # stored archive hash is independently reverified above.
                if not remaining:
                    self._purge_ttl_partitions(week_start, end)
                    manifest = self.repository.transition(
                        manifest,
                        ManifestState.PURGED,
                        owner_id="",
                        lease_expires_at=None,
                    )
                    self.logger.info(
                        "archive_purge recovered=true archive_id=%s range_start=%s "
                        "range_end=%s rows=%s",
                        manifest.archive_id, start.isoformat(), end.isoformat(),
                        manifest.row_count,
                    )
                    return manifest
            preflight = archive_preflight(
                self.client,
                self.database,
                start,
                end,
                require_surface_completion=self.config.require_surface_completion,
                surface_cadence_multiplier=self.config.surface_cadence_multiplier,
                allow_complete_with_failures=self.config.allow_complete_with_failures,
                table_names=self.table_names,
            )
            self.logger.info(
                "archive_preflight archive_id=%s range_start=%s range_end=%s "
                "eligible=%s rows=%s blockers=%s oldest_pending=%s",
                manifest.archive_id if manifest else "pending",
                start.isoformat(), end.isoformat(), preflight.eligible,
                len(preflight.records), dict(preflight.blocking_counts),
                preflight.oldest_backlog_at,
            )
            if not preflight.eligible:
                raise BacklogError(preflight)
            if dry_run:
                return manifest or self._new_manifest(start, end, path)
            if manifest is None:
                manifest = self._new_manifest(start, end, path)
                self.repository.save(manifest)
            if manifest.lease_expires_at and manifest.lease_expires_at > _utc_now():
                if manifest.owner_id and manifest.owner_id != self.owner_id:
                    raise RuntimeError(f"Archive manifest is leased by {manifest.owner_id}")
            if manifest.state in {ManifestState.PLANNED, ManifestState.FAILED}:
                manifest = self.repository.transition(
                    manifest,
                    ManifestState.WRITING,
                    owner_id=self.owner_id,
                    lease_expires_at=_utc_now()
                    + timedelta(seconds=self.config.lease_seconds),
                    attempt_count=manifest.attempt_count + 1,
                    last_error="",
                )
            if manifest.state is ManifestState.WRITING:
                def refresh_lease() -> None:
                    nonlocal manifest
                    now = _utc_now()
                    manifest = replace(
                        manifest,
                        lease_expires_at=now
                        + timedelta(seconds=self.config.lease_seconds),
                        updated_at=now,
                    )
                    self.repository.save(manifest)

                if path.exists():
                    verification = verify_archive_file(path, expected_records=preflight.records)
                else:
                    self.logger.info(
                        "archive_write archive_id=%s range_start=%s range_end=%s "
                        "rows=%s path=%s",
                        manifest.archive_id, start.isoformat(), end.isoformat(),
                        len(preflight.records), path,
                    )
                    write_archive(
                        self.client,
                        self.database,
                        preflight.records,
                        start,
                        end,
                        path,
                        self.config,
                        raw_table=self.raw_table,
                        progress_callback=refresh_lease,
                    )
                    verification = verify_archive_file(path, expected_records=preflight.records)
                manifest = self.repository.transition(
                    manifest,
                    ManifestState.WRITTEN,
                    semantic_content_hash=bytes.fromhex(
                        str(verification.manifest["semantic_content_hash"])
                    ),
                    row_count=len(preflight.records),
                    distinct_snapshot_count=len(
                        {record.snapshot_id for record in preflight.records}
                    ),
                    uncompressed_bytes=sum(
                        record.payload_byte_length for record in preflight.records
                    ),
                    compressed_bytes=path.stat().st_size,
                )
            if manifest.state is ManifestState.WRITTEN:
                manifest = self.repository.transition(manifest, ManifestState.VERIFYING)
            if manifest.state is ManifestState.VERIFYING:
                verification = verify_archive_file(path, expected_records=preflight.records)
                self.logger.info(
                    "archive_verify archive_id=%s range_start=%s range_end=%s "
                    "rows=%s file_hash=%s",
                    manifest.archive_id, start.isoformat(), end.isoformat(),
                    len(preflight.records), verification.file_hash.hex(),
                )
                self._race_check(preflight.records, start, end)
                self.logger.info(
                    "archive_race_check archive_id=%s range_start=%s range_end=%s "
                    "rows=%s status=passed",
                    manifest.archive_id, start.isoformat(), end.isoformat(),
                    len(preflight.records),
                )
                manifest = self.repository.transition(
                    manifest,
                    ManifestState.VERIFIED,
                    archive_file_hash=verification.file_hash,
                )
            if not purge:
                return manifest
            if manifest.state is ManifestState.PURGING:
                # The prior process stopped before DROP PARTITION.  Reverification
                # below uses the same pre-write snapshot/hash set before retrying.
                manifest = replace(manifest, state=ManifestState.VERIFIED)
            if manifest.state is not ManifestState.VERIFIED:
                raise RuntimeError(f"Archive is not verified: {manifest.state}")
            # Second fail-closed guard immediately before the only destructive step.
            second = archive_preflight(
                self.client,
                self.database,
                start,
                end,
                require_surface_completion=self.config.require_surface_completion,
                surface_cadence_multiplier=self.config.surface_cadence_multiplier,
                allow_complete_with_failures=self.config.allow_complete_with_failures,
                table_names=self.table_names,
            )
            self.logger.info(
                "archive_preflight second_guard=true archive_id=%s range_start=%s "
                "range_end=%s eligible=%s rows=%s blockers=%s oldest_pending=%s",
                manifest.archive_id, start.isoformat(), end.isoformat(),
                second.eligible, len(second.records),
                dict(second.blocking_counts), second.oldest_backlog_at,
            )
            if not second.eligible:
                raise BacklogError(second)
            self._race_check(preflight.records, start, end)
            if file_sha256(path) != manifest.archive_file_hash:
                raise RuntimeError("Archive file changed before purge")
            manifest = self.repository.transition(manifest, ManifestState.PURGING)
            self.client.command(
                f"ALTER TABLE {self.database}.{self.raw_table} "
                "DROP PARTITION {week_start:Date}",
                parameters={"week_start": week_start},
            )
            remaining = _raw_records(
                self.client,
                self.database,
                start,
                end,
                raw_table=self.raw_table,
            )
            if remaining:
                raise RuntimeError("Raw weekly partition still contains rows after purge")
            self._purge_ttl_partitions(week_start, end)
            manifest = self.repository.transition(
                manifest,
                ManifestState.PURGED,
                owner_id="",
                lease_expires_at=None,
            )
            self.logger.info(
                "archive_purge archive_id=%s range_start=%s range_end=%s "
                "rows=%s status=purged",
                manifest.archive_id, start.isoformat(), end.isoformat(),
                manifest.row_count,
            )
            return manifest

    def archive_week(
        self,
        week_start: date,
        *,
        dry_run: bool = False,
        purge: bool = False,
        confirm_database: Optional[str] = None,
    ) -> ArchiveManifest:
        try:
            return self._archive_week_unprotected(
                week_start,
                dry_run=dry_run,
                purge=purge,
                confirm_database=confirm_database,
            )
        except BacklogError:
            # Backlog is an expected fail-closed warning, not an archive attempt
            # failure.  No final path or destructive action was taken.
            raise
        except Exception as exc:
            start, end = week_interval(week_start)
            with contextlib.suppress(Exception):
                manifest = self.repository.load(start, end)
                if manifest is not None and manifest.state not in {
                    ManifestState.PURGED,
                    ManifestState.FAILED,
                }:
                    self.repository.save(
                        replace(
                            manifest,
                            state=ManifestState.FAILED,
                            last_error=str(exc),
                            owner_id="",
                            lease_expires_at=None,
                            updated_at=_utc_now(),
                        )
                    )
            raise


def _parse_resample_month(value: str) -> date:
    try:
        month = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--resample-month must be YYYY-MM-01") from exc
    if month.day != 1 or month.isoformat() != value:
        raise argparse.ArgumentTypeError("--resample-month must be YYYY-MM-01")
    return month


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--week-start", type=date.fromisoformat)
    group.add_argument("--archive-all-eligible", action="store_true")
    group.add_argument("--verify-only", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    purge_group = parser.add_mutually_exclusive_group()
    purge_group.add_argument("--purge-verified", action="store_true")
    purge_group.add_argument(
        "--purge-only",
        action="store_true",
        help=(
            "purge retention-eligible raw and TTL partitions without writing "
            "an archive file"
        ),
    )
    parser.add_argument("--confirm-database")
    parser.add_argument(
        "--resample-old-records", action="store_true",
        help="after the weekly operation, compact eligible normalized months to five-minute snapshots",
    )
    parser.add_argument(
        "--resample-month", type=_parse_resample_month,
        help="with --resample-old-records, compact only this YYYY-MM-01 month",
    )
    options = parser.parse_args(argv)
    if options.resample_old_records and options.verify_only:
        parser.error("--resample-old-records cannot be used with --verify-only")
    if options.resample_month and not options.resample_old_records:
        parser.error("--resample-month requires --resample-old-records")
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
        "options_chain.archive.output_directory": os.path.join(
            algo_home, "archives", "options_chain"
        ),
        "options_chain.archive.lock_filename": ".options_chain_archive.lock",
        "options_chain.archive.minimum_age_days": 14,
        "options_chain.archive.retain_raw_weeks": 2,
        "options_chain.archive.zstd_level": 22,
        "options_chain.archive.zstd_long_window_log": 31,
        "options_chain.archive.compression_threads": 0,
        "options_chain.archive.require_surface_completion": True,
        "options_chain.archive.allow_complete_with_failures": True,
        "options_chain.archive.diagnostics_retention_days": 30,
        "options_chain.archive.validation_retention_days": 30,
        "options_chain.archive.attempt_retention_days": 30,
        "options_chain.archive.monitor_service_name": "OptionsChainArchival",
        "options_chain.archive.shared_filesystem": True,
    }


def _configure_logging() -> logging.Logger:
    """Configure deterministic INFO logging before configuration is loaded."""
    logger = logging.getLogger("OptionsChainArchival")
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s|%(levelname)s|%(name)s| %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = _parse_args(argv)
    logger = _configure_logging()
    monitor: Optional[Any] = None
    if options.verify_only:
        logger.info("archive_verify path=%s status=started", options.verify_only)
        try:
            verification = verify_archive_file(options.verify_only)
            logger.info(
                "archive_verify path=%s status=completed file_hash=%s",
                options.verify_only,
                verification.file_hash.hex(),
            )
            return int(ExitCode.OK)
        except Exception:
            logger.exception(
                "archive_verify path=%s status=failed", options.verify_only
            )
            return int(ExitCode.CRITICAL)
    algo_home = os.getenv("ALGO_HOME")
    if not algo_home:
        logger.error(
            "archive_configuration error=ALGO_HOME environment variable not set"
        )
        return int(ExitCode.UNKNOWN)
    try:
        context = load_configuration(
            default_context(algo_home.rstrip("/") + "/"),
            algo_home,
            options.e,
            quiet=True,
        )
        context[PARAM_LOGGER] = logger
        config = ArchiveConfig.from_context(context)
        if options.resample_old_records and not options.dry_run:
            database = validate_database_identifier(
                str(context["options_chain.clickhouse.database"])
            )
            if options.confirm_database != database:
                raise ValueError(
                    f"Monthly resampling requires --confirm-database {database}"
                )

        if 'nagios.service_id' in context:
            context['nagios.service_id'] = context['nagios.service_id'] + " Archival"

        try:
            monitor = NagiosSystemMonitor(context)
        except Exception:
            logger.warning("archive_monitor_initialization_failed", exc_info=True)
        client = create_clickhouse_client(context)
        try:
            archiver = RawWeekArchiver(context, client, config, logger=logger)
            now = _utc_now()
            age_cutoff, newest_retained_boundary = _retention_boundaries(
                now,
                minimum_age_days=config.minimum_age_days,
                retain_raw_weeks=config.retain_raw_weeks,
            )
            effective_cutoff = min(age_cutoff, newest_retained_boundary)
            terminal_statuses = (
                "complete,complete_with_failures"
                if config.allow_complete_with_failures
                else "complete"
            )
            mode = "all_eligible" if options.archive_all_eligible else "single_week"
            logger.info(
                "archive_start mode=%s dry_run=%s purge_verified=%s purge_only=%s "
                "database=%s raw_table=%s output_directory=%s",
                mode,
                options.dry_run,
                options.purge_verified,
                options.purge_only,
                archiver.database,
                archiver.raw_table,
                config.output_directory,
            )
            logger.info(
                "archive_criteria minimum_age_days=%s age_cutoff_utc=%s "
                "retain_raw_weeks=%s newest_retained_boundary_utc=%s "
                "effective_week_end_cutoff_utc=%s "
                "require_surface_completion=%s surface_cadence_multiplier=%s "
                "allowed_terminal_statuses=%s",
                config.minimum_age_days,
                age_cutoff.isoformat(),
                config.retain_raw_weeks,
                newest_retained_boundary.isoformat(),
                effective_cutoff.isoformat(),
                config.require_surface_completion,
                config.surface_cadence_multiplier,
                terminal_statuses,
            )
            logger.info(
                "archive_criteria retention_rule=\"archive only complete UTC weeks "
                "whose exclusive week_end is on or before both cutoffs\" "
                "preflight_checks=normalization_or_hash_mismatch,"
                "partial_normalization,missing_contracts,calculation_backlog,"
                "calculation_count_mismatch,active_attempts,surface_backlog,"
                "surface_count_mismatch"
            )

            inventory = _raw_week_inventory(
                client,
                archiver.database,
                archiver.raw_table,
            )
            inventory_by_week = {item.week_start: item for item in inventory}
            retention_eligible = [
                item
                for item in inventory
                if week_is_age_eligible(
                    item.week_start,
                    now,
                    minimum_age_days=config.minimum_age_days,
                    retain_raw_weeks=config.retain_raw_weeks,
                )
            ]
            retention_ineligible = [
                item for item in inventory if item not in retention_eligible
            ]
            total_records = sum(item.row_count for item in inventory)
            eligible_records = sum(item.row_count for item in retention_eligible)
            ineligible_records = total_records - eligible_records
            oldest_record = inventory[0].oldest_collected_at if inventory else None
            newest_record = inventory[-1].newest_collected_at if inventory else None
            logger.info(
                "archive_inventory total_records=%s total_weeks=%s "
                "oldest_record_utc=%s newest_record_utc=%s",
                total_records,
                len(inventory),
                oldest_record.isoformat() if oldest_record else None,
                newest_record.isoformat() if newest_record else None,
            )
            logger.info(
                "archive_eligibility retention_eligible_records=%s "
                "retention_ineligible_records=%s retention_eligible_weeks=%s "
                "retention_ineligible_weeks=%s",
                eligible_records,
                ineligible_records,
                len(retention_eligible),
                len(retention_ineligible),
            )

            if options.archive_all_eligible:
                weeks = [item.week_start for item in retention_eligible]
                skipped = retention_ineligible
            else:
                if week_is_age_eligible(
                    options.week_start,
                    now,
                    minimum_age_days=config.minimum_age_days,
                    retain_raw_weeks=config.retain_raw_weeks,
                ):
                    weeks = [options.week_start]
                    skipped = []
                else:
                    weeks = []
                    selected = inventory_by_week.get(options.week_start)
                    skipped = [selected] if selected is not None else []
                    if selected is None:
                        _, selected_end = week_interval(options.week_start)
                        logger.info(
                            "archive_week week_start=%s week_end=%s rows=0 "
                            "status=skipped reason=%s",
                            options.week_start,
                            selected_end.isoformat(),
                            _retention_ineligibility_reason(
                                options.week_start,
                                age_cutoff=age_cutoff,
                                newest_retained_boundary=newest_retained_boundary,
                            ),
                        )

            for item in skipped:
                _, week_end = week_interval(item.week_start)
                logger.info(
                    "archive_week week_start=%s week_end=%s rows=%s "
                    "oldest_record_utc=%s newest_record_utc=%s status=skipped "
                    "reason=%s",
                    item.week_start,
                    week_end.isoformat(),
                    item.row_count,
                    item.oldest_collected_at.isoformat(),
                    item.newest_collected_at.isoformat(),
                    _retention_ineligibility_reason(
                        item.week_start,
                        age_cutoff=age_cutoff,
                        newest_retained_boundary=newest_retained_boundary,
                    ),
                )

            completed_weeks = 0
            completed_records = 0
            for week in weeks:
                week_inventory = inventory_by_week.get(week)
                row_count = week_inventory.row_count if week_inventory else 0
                logger.info(
                    "archive_week week_start=%s rows=%s status=started "
                    "dry_run=%s purge_verified=%s purge_only=%s",
                    week,
                    row_count,
                    options.dry_run,
                    options.purge_verified,
                    options.purge_only,
                )
                if options.purge_only:
                    purge_result = archiver.purge_only_week(
                        week,
                        dry_run=options.dry_run,
                        confirm_database=options.confirm_database,
                    )
                else:
                    manifest = archiver.archive_week(
                        week,
                        dry_run=options.dry_run,
                        purge=options.purge_verified,
                        confirm_database=options.confirm_database,
                    )
                completed_weeks += 1
                completed_records += row_count
                if options.purge_only:
                    logger.info(
                        "archive_week week_start=%s rows=%s status=%s "
                        "purge_only=true ttl_tables=%s",
                        week,
                        row_count,
                        "dry_run_eligible" if options.dry_run else "purged",
                        list(purge_result.ttl_tables_purged),
                    )
                else:
                    logger.info(
                        "archive_week week_start=%s rows=%s status=%s "
                        "manifest_state=%s archive_path=%s",
                        week,
                        row_count,
                        "dry_run_eligible" if options.dry_run else "completed",
                        manifest.state.value,
                        manifest.archive_path,
                    )

            if options.resample_old_records:
                from algotrading.refdata.options.MonthlyResampling import MonthlyResampler

                resampler = MonthlyResampler(archiver, logger=logger)
                months = resampler.eligible_months(now, options.resample_month)
                month_results = [
                    resampler.process_month(month, dry_run=options.dry_run)
                    for month in months
                ]
                logger.info(
                    "resample_summary selected_months=%s resampled_months=%s "
                    "already_resampled_months=%s dry_run=%s",
                    len(months), month_results.count("resampled"),
                    month_results.count("already_resampled"), options.dry_run,
                )

            logger.info(
                "archive_summary status=completed mode=%s total_records=%s "
                "retention_eligible_records=%s retention_ineligible_records=%s "
                "selected_weeks=%s completed_weeks=%s completed_records=%s "
                "dry_run=%s purge_verified=%s purge_only=%s",
                mode,
                total_records,
                eligible_records,
                ineligible_records,
                len(weeks),
                completed_weeks,
                completed_records,
                options.dry_run,
                options.purge_verified,
                options.purge_only,
            )
            if monitor is not None:
                if not weeks:
                    scope = (
                        "no retention-eligible records"
                        if options.archive_all_eligible
                        else "selected week is not retention-eligible"
                    )
                    message = (
                        f"Option-chain raw archive completed: {scope}; "
                        f"total_records={total_records} "
                        f"ineligible_records={ineligible_records} "
                        f"effective_week_end_cutoff={effective_cutoff.isoformat()}"
                    )
                elif options.dry_run:
                    message = (
                        "Option-chain raw archive dry run completed: "
                        f"eligible_records={completed_records} "
                        f"eligible_weeks={completed_weeks}"
                    )
                else:
                    message = (
                        "Option-chain raw archive completed: "
                        f"processed_records={completed_records} "
                        f"processed_weeks={completed_weeks} "
                        f"purge_verified={options.purge_verified} "
                        f"purge_only={options.purge_only}"
                    )
                monitor.report_event(
                    NagiosSystemMonitor.MONITOR_OK,
                    message,
                )
            return int(ExitCode.OK)
        finally:
            client.close()
    except BacklogError as exc:
        logger.warning(
            "archive_backlog status=blocked records=%s blocker_counts=%s "
            "oldest_pending_utc=%s",
            len(exc.preflight.records),
            dict(exc.preflight.blocking_counts),
            exc.preflight.oldest_backlog_at,
        )
        if monitor is not None:
            monitor.report_event(
                NagiosSystemMonitor.MONITOR_WARNING,
                "Option-chain raw archive blocked by backlog "
                f"counts={dict(exc.preflight.blocking_counts)} "
                f"oldest={exc.preflight.oldest_backlog_at}",
            )
        return int(ExitCode.WARNING)
    except (ValueError, OSError) as exc:
        logger.error("archive_configuration error=%s", exc, exc_info=True)
        return int(ExitCode.UNKNOWN)
    except Exception:
        logger.exception("archive_failure")
        if monitor is not None:
            monitor.report_event(
                NagiosSystemMonitor.MONITOR_CRITICAL,
                (
                    "Option-chain archival or monthly resampling failed; inspect "
                    "manifest and partitions"
                    if options.resample_old_records
                    else "Option-chain raw archive failed; no unverified purge is allowed"
                ),
            )
        return int(ExitCode.CRITICAL)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_FORMAT_VERSION",
    "ArchiveConfig",
    "ArchiveLock",
    "ArchiveManifest",
    "ArchiveVerification",
    "BacklogError",
    "ExitCode",
    "ManifestRepository",
    "ManifestState",
    "Preflight",
    "PurgeOnlyResult",
    "RawArchiveRecord",
    "RestoredRawRecord",
    "RawWeekArchiver",
    "archive_preflight",
    "file_sha256",
    "read_archive_records",
    "semantic_content_hash",
    "validate_transition",
    "verify_archive_file",
    "week_interval",
    "week_is_age_eligible",
    "write_archive",
]
