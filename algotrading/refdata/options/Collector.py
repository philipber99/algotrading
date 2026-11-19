"""Schwab option-chain capture and normalized ClickHouse publication.

``options_chain_raw`` is the short-lived normalization-recovery source.  The
normalized snapshot manifest is the durable publication barrier.  The
in-process queue is only an accelerator: startup and periodic reconciliation
rediscover raw rows whose manifest has not been committed.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import logging
import math
import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import date, datetime, time as datetime_time, timedelta
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from algotrading.Common import import_module_attr, load_configuration
from algotrading.ContextFields import (
    PARAM_ALGO_HOME,
    PARAM_DEFAULT_LOG_LEVEL,
    PARAM_LOGGER,
    PARAM_LOG_PATH,
    PARAM_REFDATA_INTERFACE,
    PARAM_REFDATA_PROVIDER_CLASS,
)
from algotrading.refdata import MarketData
from algotrading.refdata.options.Common import (
    RAW_SCHEMA,
    InconsistentSnapshotError,
    SnapshotDataError,
    UTC,
    _as_bool,
    _compact_json,
    _finite_float,
    _iter_contracts,
    _nonnegative_int,
    _parse_timestamp,
    _rows_to_arrow,
    _utc_now,
    create_clickhouse_client,
    read_raw_snapshot,
    validate_database_identifier,
)
from algotrading.refdata.options.Definitions import (
    DefinitionRepository,
    DefinitionType,
    build_definition,
)
from algotrading.refdata.options.Normalization import (
    CONTRACT_SCHEMA,
    DETAIL_SCHEMA,
    OBSERVATION_SCHEMA,
    SNAPSHOT_SCHEMA,
    NormalizedSnapshot,
    build_normalized_snapshot,
    merge_contract_metadata,
)
from algotrading.refdata.options.Schema import (
    CANONICAL_SCHEMA,
    MIGRATION_SUPPORT_SCHEMA,
    SCHEMA_GENERATION,
    SchemaValidationError,
    SchemaValidationReport,
    validate_canonical_table_metadata,
)
from algotrading.refdata.options.Runtime import maintenance_lease_active
from algotrading.util.NagiosSystemMonitor import NagiosSystemMonitor


_SENTINEL = object()
_MANIFESTLESS_REPLAY_BATCH_SIZE = 1_000

COLLECTOR_REQUIRED_TABLES = (
    "options_chain_raw",
    "options_chain_archive_manifest",
    "options_chain_snapshots",
    "options_contracts",
    "options_chain_observations",
    "options_chain_observation_details",
    "options_chain_definitions",
    "options_chain_dataset_metadata",
)
COLLECTOR_PREFLIGHT_TABLES = (
    *COLLECTOR_REQUIRED_TABLES,
    "options_chain_migration_leases",
)


def _calendar_session_date(value: Any, fallback: datetime) -> date:
    """Return the session date without depending on pandas scalar types."""
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return fallback.date()


def _calendar_boundary_datetime(value: Any, session_date: date) -> datetime:
    """Normalize calendar datetimes or time-since-midnight values for comparison."""
    if hasattr(value, "to_pytimedelta"):
        value = value.to_pytimedelta()
    elif hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        boundary = value
    elif isinstance(value, timedelta):
        boundary = datetime.combine(session_date, datetime_time.min) + value
    elif isinstance(value, datetime_time):
        boundary = datetime.combine(session_date, value)
    else:
        raise TypeError(f"Unsupported market-calendar boundary: {value!r}")

    # MarketData's established contract is a naive datetime in the process-local
    # timezone. Preserve it while also accepting aware calendar implementations.
    if boundary.tzinfo is not None:
        boundary = boundary.astimezone().replace(tzinfo=None)
    return boundary


@dataclass(frozen=True)
class RawSnapshotRef:
    snapshot_id: uuid.UUID
    symbol: str
    collected_at_utc: datetime


@dataclass(frozen=True)
class ReconciliationWork:
    pending: tuple[RawSnapshotRef, ...]
    duplicate_raw: tuple[tuple[uuid.UUID, int], ...]
    duplicate_manifests: tuple[RawSnapshotRef, ...]

    @property
    def replayable(self) -> tuple[RawSnapshotRef, ...]:
        return self.pending + self.duplicate_manifests

    def __iter__(self):
        """Preserve the former two-value unpacking contract for callers."""
        yield list(self.pending)
        yield [
            *(snapshot_id for snapshot_id, _ in self.duplicate_raw),
            *(ref.snapshot_id for ref in self.duplicate_manifests),
        ]


@dataclass(frozen=True)
class DatabaseErrorClassification:
    classification: str
    retryable: bool
    clickhouse_code: Optional[int]


@dataclass(frozen=True)
class CollectorPreflightReport:
    database: str
    server_version: str
    schema_generation: int
    active_dataset_count: int
    schema: SchemaValidationReport
    sampled_snapshot_id: Optional[uuid.UUID]
    sampled_snapshot_state: Optional["NormalizedSnapshotState"]
    reconciliation_candidate_count: int
    maintenance_lease_active: bool


@dataclass(frozen=True)
class IntegrityAuditReport:
    database: str
    bounded_days: int
    maximum_snapshots: int
    audited_snapshots: int
    finding_counts: Mapping[str, int]
    sample_snapshot_ids: Mapping[str, tuple[str, ...]]

    @property
    def finding_total(self) -> int:
        return sum(int(value) for value in self.finding_counts.values())


@dataclass(frozen=True)
class NormalizedSnapshotState:
    manifest_rows: int
    expected_observations: int
    stored_detail_count: int
    stored_distinct_contracts: int
    observation_rows: int
    detail_rows: int
    distinct_contracts: int
    missing_contracts: int

    @property
    def complete(self) -> bool:
        return (
            self.manifest_rows == 1
            and self.expected_observations > 0
            and self.observation_rows == self.expected_observations
            and self.stored_detail_count == self.expected_observations
            and self.detail_rows == self.stored_detail_count
            and self.stored_distinct_contracts == self.expected_observations
            and self.distinct_contracts == self.expected_observations
            and self.missing_contracts == 0
        )

    @property
    def partial(self) -> bool:
        return any(
            (
                self.manifest_rows,
                self.observation_rows,
                self.detail_rows,
                self.distinct_contracts,
            )
        ) and not self.complete

    @property
    def empty(self) -> bool:
        return not any(
            (
                self.manifest_rows,
                self.expected_observations,
                self.stored_detail_count,
                self.stored_distinct_contracts,
                self.observation_rows,
                self.detail_rows,
                self.distinct_contracts,
                self.missing_contracts,
            )
        )


@dataclass
class HealthState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_raw_commit_time: dict[str, datetime] = field(default_factory=dict)
    last_normalized_commit_time: Optional[datetime] = None
    raw_request_failures: int = 0
    raw_database_failures: int = 0
    normalization_parsing_failures: int = 0
    normalization_database_failures: int = 0
    pending_raw_to_normalized: int = 0
    oldest_pending_raw_time: Optional[datetime] = None
    queue_depth: int = 0
    queue_nonempty_since: Optional[datetime] = None
    last_error: str = ""
    consecutive_failures: dict[str, int] = field(default_factory=dict)
    repair_required_errors: int = 0
    capture_only: bool = False

    def update(self, **values: Any) -> None:
        with self.lock:
            for name, value in values.items():
                setattr(self, name, value)

    def increment(self, name: str, amount: int = 1) -> None:
        with self.lock:
            previous = getattr(self, name)
            current = previous + amount
            setattr(self, name, current)
            if name == "queue_depth":
                if previous <= 0 < current:
                    self.queue_nonempty_since = _utc_now()
                elif current <= 0:
                    self.queue_nonempty_since = None

    def record_consecutive_failure(self, operation: str) -> None:
        with self.lock:
            self.consecutive_failures[operation] = (
                self.consecutive_failures.get(operation, 0) + 1
            )

    def clear_consecutive_failure(self, operation: str) -> None:
        with self.lock:
            self.consecutive_failures.pop(operation, None)

    def require_repair(self) -> None:
        with self.lock:
            self.repair_required_errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "last_raw_commit_time": dict(self.last_raw_commit_time),
                "last_normalized_commit_time": self.last_normalized_commit_time,
                "raw_request_failures": self.raw_request_failures,
                "raw_database_failures": self.raw_database_failures,
                "normalization_parsing_failures": self.normalization_parsing_failures,
                "normalization_database_failures": self.normalization_database_failures,
                "pending_raw_to_normalized": self.pending_raw_to_normalized,
                "oldest_pending_raw_time": self.oldest_pending_raw_time,
                "queue_depth": self.queue_depth,
                "queue_nonempty_since": self.queue_nonempty_since,
                "last_error": self.last_error,
                "consecutive_failures": dict(self.consecutive_failures),
                "repair_required_errors": self.repair_required_errors,
                "capture_only": self.capture_only,
            }


def collector_schema_statements(database: str) -> list[str]:
    database = validate_database_identifier(database)
    return [
        f"CREATE DATABASE IF NOT EXISTS {database}",
        *(
            CANONICAL_SCHEMA[name].ddl.format(database=database)
            for name in COLLECTOR_REQUIRED_TABLES
        ),
        MIGRATION_SUPPORT_SCHEMA["options_chain_migration_leases"].ddl.format(
            database=database
        ),
    ]


def _bootstrap_empty_dataset(client: Any, database: str) -> Optional[uuid.UUID]:
    """Create the generation row only for a provably empty canonical database."""
    database = validate_database_identifier(database)
    data_tables = tuple(
        name
        for name in COLLECTOR_REQUIRED_TABLES
        if name != "options_chain_dataset_metadata"
    )
    counts = client.query(
        "SELECT "
        + ", ".join(
            f"(SELECT count() FROM {database}.{name})" for name in data_tables
        )
        + f", (SELECT count() FROM {database}.options_chain_dataset_metadata)"
    ).result_rows
    if len(counts) != 1:
        raise SchemaValidationError("Unable to determine whether the database is empty")
    *data_counts, metadata_count = (int(value or 0) for value in counts[0])
    if metadata_count:
        return None
    if any(data_counts):
        raise SchemaValidationError(
            "Dataset metadata is missing from a non-empty generation-3 database"
        )

    generation_id = uuid.uuid4()
    now = _utc_now()
    client.command(
        f"""
        INSERT INTO {database}.options_chain_dataset_metadata
        (
            dataset_generation_id, schema_generation, migration_id, created_at,
            activated_at, deactivated_at, default_collection_definition_id,
            default_normalization_definition_id, default_calculation_definition_id,
            default_surface_definition_id, source_start_at, source_end_at,
            validation_summary_json, migration_phase, migration_checkpoint_json,
            updated_at, is_active
        )
        VALUES
        (
            {{generation_id:UUID}}, {SCHEMA_GENERATION}, {{migration_id:UUID}},
            {{now:DateTime64(6, 'UTC')}}, {{now:DateTime64(6, 'UTC')}}, NULL,
            0, 0, 0, 0, {{now:DateTime64(6, 'UTC')}},
            {{now:DateTime64(6, 'UTC')}}, {{summary:String}}, 'collector_bootstrap',
            '{{}}', {{now:DateTime64(6, 'UTC')}}, true
        )
        """,
        parameters={
            "generation_id": str(generation_id),
            "migration_id": str(uuid.UUID(int=0)),
            "now": now,
            "summary": json.dumps(
                {"bootstrap": "collector", "schema_generation": SCHEMA_GENERATION},
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    )
    return generation_id


def create_collector_schema(
    context: Mapping[str, Any], *, client_factory: Any = create_clickhouse_client
) -> None:
    database = validate_database_identifier(
        str(context["options_chain.clickhouse.database"])
    )
    client = client_factory(context, database="default")
    try:
        for statement in collector_schema_statements(database):
            client.command(statement)
        _bootstrap_empty_dataset(client, database)
    finally:
        client.close()


def _optional_positive_int(value: Any, setting_name: str) -> Optional[int]:
    if value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "none", "all"}
    ):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{setting_name} must be a positive integer or None") from exc
    if result <= 0:
        raise ValueError(f"{setting_name} must be a positive integer or None")
    return result


_DETERMINISTIC_CLICKHOUSE_CODES = frozenset(
    {
        20,   # number of columns does not match
        27,   # cannot parse input
        36,   # bad arguments
        43,   # illegal type
        47,   # unknown identifier
        53,   # type mismatch
        60,   # unknown table
        62,   # syntax error
        81,   # unknown database
        117,  # incorrect data
        170,  # bad get
        171,  # bad set
    }
)
_TRANSIENT_CLICKHOUSE_CODES = frozenset(
    {
        159,  # timeout exceeded
        202,  # too many simultaneous queries / temporarily unavailable
        203,  # no free connection
        209,  # network error
        210,  # network error while receiving
        279,  # all connection tries failed
        319,  # unknown status from server during restart
    }
)
_TRANSIENT_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ETIMEDOUT,
        errno.EPIPE,
    }
)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _clickhouse_error_code(chain: Sequence[BaseException]) -> Optional[int]:
    for error in chain:
        raw_code = getattr(error, "code", None)
        try:
            if raw_code is not None:
                return int(raw_code)
        except (TypeError, ValueError):
            pass
        match = re.search(r"\bCode:\s*(\d+)\b", str(error))
        if match:
            return int(match.group(1))
    return None


def classify_database_error(exc: BaseException) -> DatabaseErrorClassification:
    """Classify database failures with a conservative, retry-safe allowlist."""
    chain = _exception_chain(exc)
    code = _clickhouse_error_code(chain)
    if code in _DETERMINISTIC_CLICKHOUSE_CODES:
        return DatabaseErrorClassification("deterministic", False, code)
    if code in _TRANSIENT_CLICKHOUSE_CODES:
        return DatabaseErrorClassification("transient", True, code)
    if any(
        isinstance(
            error,
            (
                SchemaValidationError,
                SnapshotDataError,
                InconsistentSnapshotError,
                ValueError,
                TypeError,
                AssertionError,
                PermissionError,
            ),
        )
        for error in chain
    ):
        return DatabaseErrorClassification("deterministic", False, code)
    for error in chain:
        if isinstance(error, TimeoutError):
            return DatabaseErrorClassification("transient", True, code)
        if isinstance(error, ConnectionError):
            return DatabaseErrorClassification("transient", True, code)
        if isinstance(error, OSError) and getattr(error, "errno", None) in _TRANSIENT_ERRNOS:
            return DatabaseErrorClassification("transient", True, code)
        type_name = f"{type(error).__module__}.{type(error).__name__}".lower()
        if any(token in type_name for token in ("timeout", "connectionerror")):
            return DatabaseErrorClassification("transient", True, code)
        if type_name.startswith(("urllib3.", "requests.")) and any(
            token in type_name
            for token in ("maxretry", "protocolerror", "newconnection", "proxyerror")
        ):
            return DatabaseErrorClassification("transient", True, code)
        message = str(error).lower()
        http_status = re.search(
            r"\b(?:http\s+)?status(?:\s+code)?[ :=]+(\d{3})\b", message
        )
        if http_status and int(http_status.group(1)) in {408, 429, 502, 503, 504}:
            return DatabaseErrorClassification("transient", True, code)
        if any(
            token in message
            for token in (
                "connection reset",
                "connection refused",
                "temporarily unavailable",
                "timed out",
                "remote end closed connection",
                "server disconnected",
            )
        ):
            return DatabaseErrorClassification("transient", True, code)
    return DatabaseErrorClassification("deterministic", False, code)


def _log_database_error(
    logger: logging.Logger, phase: str, exc: BaseException
) -> DatabaseErrorClassification:
    classification = classify_database_error(exc)
    logger.error(
        "event=collector_database_error phase=%s classification=%s retryable=%s "
        "clickhouse_code=%s error_type=%s",
        phase,
        classification.classification,
        str(classification.retryable).lower(),
        classification.clickhouse_code,
        type(exc).__name__,
    )
    return classification


def _count_contracts(response: Mapping[str, Any]) -> int:
    count = 0
    for map_name in ("callExpDateMap", "putExpDateMap"):
        expiration_map = response.get(map_name)
        if not isinstance(expiration_map, dict):
            continue
        for strike_map in expiration_map.values():
            if not isinstance(strike_map, dict):
                continue
            for leaf in strike_map.values():
                if isinstance(leaf, list):
                    count += sum(isinstance(item, dict) for item in leaf)
                elif isinstance(leaf, dict):
                    count += 1
    return count


def _select_spot(response: Mapping[str, Any]) -> tuple[Optional[float], str]:
    underlying = response.get("underlying")
    underlying = underlying if isinstance(underlying, dict) else {}
    bid = _finite_float(underlying.get("bid", underlying.get("bidPrice")))
    ask = _finite_float(underlying.get("ask", underlying.get("askPrice")))
    if bid is not None and ask is not None and bid >= 0 and ask >= bid and ask > 0:
        return (bid + ask) / 2.0, "underlying_mid"
    mark = _finite_float(underlying.get("mark", underlying.get("markPrice")))
    if mark is not None and mark > 0:
        return mark, "underlying_mark"
    chain = _finite_float(response.get("underlyingPrice"))
    if chain is not None and chain > 0:
        return chain, "chain_underlying"
    last = _finite_float(underlying.get("last", underlying.get("lastPrice")))
    if last is not None and last > 0:
        return last, "underlying_last"
    return None, "unavailable"


def build_raw_record(
    *,
    snapshot_id: uuid.UUID,
    symbol: str,
    response: Mapping[str, Any],
    request_started_at_utc: datetime,
    response_received_at_utc: datetime,
    collected_at_utc: datetime,
    request_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize the response once and build the exact durable raw row."""
    response_bytes = json.dumps(
        response, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    latency_ms = max(
        0,
        int(
            round(
                (response_received_at_utc - request_started_at_utc).total_seconds()
                * 1000.0
            )
        ),
    )
    return {
        "snapshot_id": str(snapshot_id),
        "source": "schwab",
        "symbol": symbol,
        "request_started_at_utc": request_started_at_utc,
        "response_received_at_utc": response_received_at_utc,
        "collected_at_utc": collected_at_utc,
        "response_latency_ms": latency_ms,
        "request_parameters_json": _compact_json(request_parameters),
        # Keep the transient collector record byte-oriented for hashing and
        # parsing.  insert_raw_snapshot() makes a detached UTF-8 String row
        # for the canonical ClickHouse schema.
        "response_json": response_bytes,
        "payload_sha256": hashlib.sha256(response_bytes).digest(),
        "source_contract_count": _count_contracts(response),
        "is_delayed": _as_bool(response.get("isDelayed", False)),
        "is_chain_truncated": _as_bool(response.get("isChainTruncated", False)),
        "status": str(response.get("status") or ""),
    }


def verify_raw_snapshot(
    client: Any,
    database: str,
    snapshot_id: uuid.UUID,
    *,
    expected_hash: Optional[bytes] = None,
) -> dict[str, Any]:
    record, _ = read_raw_snapshot(client, database, snapshot_id)
    if expected_hash is not None and bytes(record["payload_sha256"]) != bytes(expected_hash):
        raise SnapshotDataError(
            f"Raw insert verification failed for {snapshot_id}: unexpected payload hash"
        )
    return record


def _ensure_raw_interval_open(
    client: Any,
    database: str,
    collected_at: datetime,
    *,
    allow_purged_range_backfill: bool,
    confirm_database: Optional[str],
    confirm_range_start: Optional[Any],
    confirm_range_end: Optional[Any],
) -> None:
    result = client.query(
        f"""
        SELECT toString(state), range_start, range_end
        FROM {database}.options_chain_archive_manifest FINAL
        WHERE archive_kind = 'raw_week'
          AND range_start <= {{collected_at:DateTime64(6, 'UTC')}}
          AND range_end > {{collected_at:DateTime64(6, 'UTC')}}
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        parameters={"collected_at": collected_at},
    )
    if not result.result_rows or str(result.result_rows[0][0]).lower() != "purged":
        return
    manifest_start = _parse_timestamp(result.result_rows[0][1])
    manifest_end = _parse_timestamp(result.result_rows[0][2])
    confirmed_start = _parse_timestamp(confirm_range_start)
    confirmed_end = _parse_timestamp(confirm_range_end)
    if (
        not allow_purged_range_backfill
        or confirm_database != database
        or confirmed_start != manifest_start
        or confirmed_end != manifest_end
    ):
        logging.getLogger(__name__).critical(
            "archive_late_arrival database=%s collected_at=%s range_start=%s "
            "range_end=%s override=%s confirmation_matches=%s action=rejected",
            database,
            collected_at.isoformat(),
            manifest_start,
            manifest_end,
            allow_purged_range_backfill,
            (
                confirm_database == database
                and confirmed_start == manifest_start
                and confirmed_end == manifest_end
            ),
        )
        raise SnapshotDataError(
            "Refusing a raw late arrival in a purged archive interval; an explicit "
            f"backfill override and exact confirmation of {database!r}, "
            f"{manifest_start}, and {manifest_end} are required"
        )
    # A restored interval cannot remain eligible under a verified/purged
    # manifest.  The archival tool will require a replacement archive.
    logging.getLogger(__name__).warning(
        "archive_late_arrival database=%s collected_at=%s range_start=%s "
        "range_end=%s override=true action=restore_and_reverify",
        database,
        collected_at.isoformat(),
        manifest_start,
        manifest_end,
    )
    client.command(
        f"ALTER TABLE {database}.options_chain_archive_manifest UPDATE "
        "state = 'failed', last_error = 'purged interval restored; reverification required', "
        "updated_at = now64(6) WHERE archive_kind = 'raw_week' "
        "AND range_start <= {collected_at:DateTime64(6, 'UTC')} "
        "AND range_end > {collected_at:DateTime64(6, 'UTC')}",
        parameters={"collected_at": collected_at},
    )


def insert_raw_snapshot(
    client: Any,
    database: str,
    record: Mapping[str, Any],
    *,
    allow_purged_range_backfill: bool = False,
    confirm_database: Optional[str] = None,
    confirm_range_start: Optional[Any] = None,
    confirm_range_end: Optional[Any] = None,
) -> RawSnapshotRef:
    database = validate_database_identifier(database)
    snapshot_id = uuid.UUID(str(record["snapshot_id"]))
    collected_at = _parse_timestamp(record["collected_at_utc"])
    if collected_at is None:
        raise ValueError("Raw row has an invalid collected_at_utc")
    _ensure_raw_interval_open(
        client,
        database,
        collected_at,
        allow_purged_range_backfill=allow_purged_range_backfill,
        confirm_database=confirm_database,
        confirm_range_start=confirm_range_start,
        confirm_range_end=confirm_range_end,
    )
    raw_row = dict(record)
    payload = raw_row.get("response_json")
    if isinstance(payload, (bytes, bytearray, memoryview)):
        raw_row["response_json"] = bytes(payload).decode("utf-8")
    table = _rows_to_arrow([raw_row], RAW_SCHEMA)
    try:
        client.insert_arrow(
            f"{database}.options_chain_raw",
            table,
            settings={"insert_deduplication_token": f"raw:{snapshot_id}"},
        )
    except Exception:
        # An interrupted HTTP response is ambiguous.  Treat the insert as
        # successful only when the exact row is independently verifiable.
        try:
            verify_raw_snapshot(
                client,
                database,
                snapshot_id,
                expected_hash=bytes(record["payload_sha256"]),
            )
        except Exception:
            raise
    verify_raw_snapshot(
        client,
        database,
        snapshot_id,
        expected_hash=bytes(record["payload_sha256"]),
    )
    return RawSnapshotRef(
        snapshot_id,
        str(record["symbol"]),
        _parse_timestamp(record["collected_at_utc"]),
    )


def flatten_and_filter_response(
    response: Mapping[str, Any],
    *,
    snapshot_id: uuid.UUID,
    collected_at_utc: datetime,
    request_started_at_utc: Optional[datetime] = None,
    response_received_at_utc: Optional[datetime] = None,
    ingested_at_utc: Optional[datetime] = None,
    max_tenor_age: int = 365,
    num_strikes_each_side: int = 20,
    include_nonstandard: bool = False,
    include_minis: bool = False,
    expiration_grace_seconds: int = 0,
) -> list[dict[str, Any]]:
    """Flatten a complete Schwab response and apply database filters."""
    if not isinstance(response, Mapping):
        raise TypeError("Option-chain response must be a mapping")
    collected = _parse_timestamp(collected_at_utc)
    if collected is None:
        raise ValueError("collected_at_utc is invalid")
    spot, _ = _select_spot(response)
    if spot is None:
        raise ValueError("Snapshot has no valid spot for the fixed strike filter")

    request_started = _parse_timestamp(request_started_at_utc) or collected
    response_received = _parse_timestamp(response_received_at_utc) or collected
    source_count = _count_contracts(response)
    nested = response.get("underlying")
    nested = nested if isinstance(nested, dict) else {}
    underlying_is_delayed = _as_bool(
        nested.get("isDelayed", nested.get("delayed", False))
    )
    chain_rate_raw = _finite_float(response.get("interestRate"))
    chain_div_raw = _finite_float(response.get("dividendYield"))
    max_seconds = float(max_tenor_age) * 86400.0
    ingested_at = _parse_timestamp(ingested_at_utc) or _utc_now()
    candidates: list[dict[str, Any]] = []

    for (
        ordinal,
        expiration_key,
        strike_key,
        contract,
        expected_side,
        map_mismatch,
    ) in _iter_contracts(response):
        if not include_nonstandard and _as_bool(contract.get("nonStandard", False)):
            continue
        if not include_minis and _as_bool(contract.get("mini", False)):
            continue
        expiration_at = _parse_timestamp(contract.get("expirationDate"))
        if expiration_at is None:
            fallback_date = str(expiration_key).split(":", 1)[0]
            expiration_at = _parse_timestamp(fallback_date + "T00:00:00+00:00")
        if expiration_at is None:
            continue
        tenor_seconds = (expiration_at - collected).total_seconds()
        if (
            tenor_seconds <= -float(expiration_grace_seconds)
            or tenor_seconds > max_seconds
        ):
            continue
        strike = _finite_float(contract.get("strikePrice"))
        if strike is None:
            strike = _finite_float(strike_key)
        if strike is None:
            continue
        source_vol_raw = _finite_float(contract.get("volatility"))
        row = {
            "snapshot_id": str(snapshot_id),
            "collected_at": collected,
            "request_started_at": request_started,
            "response_received_at": response_received,
            "ingested_at": ingested_at,
            "raw_archive_date": collected.date(),
            "contract_ordinal": ordinal,
            "source": "schwab",
            "underlying_symbol": str(response.get("symbol") or "").upper(),
            "chain_status": str(response.get("status") or ""),
            "strategy": str(response.get("strategy") or ""),
            "chain_interval": _finite_float(response.get("interval")) or 0.0,
            "chain_is_delayed": _as_bool(response.get("isDelayed", False)),
            "chain_is_index": _as_bool(response.get("isIndex", False)),
            "chain_is_truncated": _as_bool(response.get("isChainTruncated", False)),
            "underlying_is_delayed": underlying_is_delayed,
            "symbol_map_mismatch": bool(map_mismatch),
            "source_contract_count": source_count,
            "stored_contract_count": 0,
            "asset_main_type": str(response.get("assetMainType") or ""),
            "asset_sub_type": str(response.get("assetSubType") or ""),
            "chain_interest_rate_raw": chain_rate_raw,
            "chain_interest_rate": (
                None if chain_rate_raw is None else chain_rate_raw / 100.0
            ),
            "chain_dividend_yield_raw": chain_div_raw,
            "chain_dividend_yield": (
                None if chain_div_raw is None else chain_div_raw / 100.0
            ),
            "chain_volatility_raw": _finite_float(response.get("volatility")),
            "chain_days_to_expiration": _finite_float(response.get("daysToExpiration")),
            "chain_underlying_price": _finite_float(response.get("underlyingPrice")),
            "underlying_bid": _finite_float(nested.get("bid", nested.get("bidPrice"))),
            "underlying_ask": _finite_float(nested.get("ask", nested.get("askPrice"))),
            "underlying_last": _finite_float(
                nested.get("last", nested.get("lastPrice"))
            ),
            "underlying_mark": _finite_float(
                nested.get("mark", nested.get("markPrice"))
            ),
            "underlying_bid_size": _nonnegative_int(nested.get("bidSize")),
            "underlying_ask_size": _nonnegative_int(nested.get("askSize")),
            "underlying_quote_time": _parse_timestamp(
                nested.get("quoteTimeInLong", nested.get("quoteTime"))
            ),
            "underlying_trade_time": _parse_timestamp(
                nested.get("tradeTimeInLong", nested.get("tradeTime"))
            ),
            "underlying_total_volume": _nonnegative_int(nested.get("totalVolume")),
            "option_symbol": str(contract.get("symbol") or ""),
            "option_root": str(contract.get("optionRoot") or ""),
            "ssid": _nonnegative_int(contract.get("ssid")),
            "description": str(contract.get("description") or ""),
            "exchange_name": str(contract.get("exchangeName") or ""),
            "put_call": str(contract.get("putCall") or expected_side).upper(),
            "strike_price": strike,
            "expiration_at": expiration_at,
            "source_days_to_expiration": (
                _nonnegative_int(contract.get("daysToExpiration")) or 0
            ),
            "expiration_type": str(contract.get("expirationType") or ""),
            "last_trading_day": _parse_timestamp(contract.get("lastTradingDay")),
            "settlement_type": str(contract.get("settlementType") or ""),
            "exercise_type": str(contract.get("exerciseType") or ""),
            "multiplier": _finite_float(contract.get("multiplier")) or 0.0,
            "is_in_the_money": _as_bool(contract.get("inTheMoney", False)),
            "is_mini": _as_bool(contract.get("mini", False)),
            "is_nonstandard": _as_bool(contract.get("nonStandard", False)),
            "is_penny_pilot": _as_bool(contract.get("pennyPilot", False)),
            "deliverable_note": str(contract.get("deliverableNote") or ""),
            "option_deliverables_json": _compact_json(
                contract.get("optionDeliverablesList") or []
            ),
            "bid": _finite_float(contract.get("bid")),
            "ask": _finite_float(contract.get("ask")),
            "last": _finite_float(contract.get("last")),
            "mark": _finite_float(contract.get("mark")),
            "bid_size": _nonnegative_int(contract.get("bidSize")),
            "ask_size": _nonnegative_int(contract.get("askSize")),
            "last_size": _nonnegative_int(contract.get("lastSize")),
            "source_bid_ask_size": str(contract.get("bidAskSize") or ""),
            "open_price": _finite_float(contract.get("openPrice")),
            "high_price": _finite_float(contract.get("highPrice")),
            "low_price": _finite_float(contract.get("lowPrice")),
            "close_price": _finite_float(contract.get("closePrice")),
            "high_52_week": _finite_float(contract.get("high52Week")),
            "low_52_week": _finite_float(contract.get("low52Week")),
            "total_volume": _nonnegative_int(contract.get("totalVolume")),
            "open_interest": _nonnegative_int(contract.get("openInterest")),
            "quote_time": _parse_timestamp(contract.get("quoteTimeInLong")),
            "trade_time": _parse_timestamp(contract.get("tradeTimeInLong")),
            "net_change": _finite_float(contract.get("netChange")),
            "percent_change": _finite_float(contract.get("percentChange")),
            "mark_change": _finite_float(contract.get("markChange")),
            "mark_percent_change": _finite_float(contract.get("markPercentChange")),
            "intrinsic_value": _finite_float(contract.get("intrinsicValue")),
            "extrinsic_value": _finite_float(contract.get("extrinsicValue")),
            "time_value": _finite_float(contract.get("timeValue")),
            "break_even": _finite_float(contract.get("breakEven")),
            "theoretical_option_value": _finite_float(
                contract.get("theoreticalOptionValue")
            ),
            "theoretical_volatility_raw": _finite_float(
                contract.get("theoreticalVolatility")
            ),
            "source_volatility_raw": source_vol_raw,
            "source_iv": None if source_vol_raw is None else source_vol_raw / 100.0,
            "source_delta": _finite_float(contract.get("delta")),
            "source_gamma": _finite_float(contract.get("gamma")),
            "source_theta": _finite_float(contract.get("theta")),
            "source_vega": _finite_float(contract.get("vega")),
            "source_rho": _finite_float(contract.get("rho")),
        }
        candidates.append(row)

    strikes_by_expiration: dict[datetime, set[float]] = {}
    for row in candidates:
        strikes_by_expiration.setdefault(row["expiration_at"], set()).add(
            row["strike_price"]
        )
    selected: dict[datetime, set[float]] = {}
    for expiration_at, strike_set in strikes_by_expiration.items():
        strikes = sorted(strike_set)
        below = (
            [strike for strike in strikes if strike < spot][-num_strikes_each_side:]
            if num_strikes_each_side
            else []
        )
        above = [strike for strike in strikes if strike > spot][:num_strikes_each_side]
        exact = [strike for strike in strikes if strike == spot]
        selected[expiration_at] = set(below + exact + above)

    rows = [
        row
        for row in candidates
        if row["strike_price"] in selected[row["expiration_at"]]
    ]
    rows.sort(key=lambda row: row["option_symbol"])
    stored_count = len(rows)
    for row in rows:
        row["stored_contract_count"] = stored_count
    return rows


def _normalized_snapshot_state(
    client: Any, database: str, snapshot_id: uuid.UUID
) -> NormalizedSnapshotState:
    validate_database_identifier(database)
    # Child-row visibility must be independent of the manifest.  The manifest is
    # deliberately written only after the child rows pass the pre-publication
    # checks, and restart repair also needs to detect children left behind when
    # that final insert did not happen.  Snapshot IDs own all three row sets, so
    # counting every row for the ID also exposes stale or mismatched children
    # instead of incorrectly treating the publication as complete.
    result = client.query(
        f"""
        SELECT
            (SELECT count() FROM {database}.options_chain_snapshots
             WHERE snapshot_id = {{snapshot_id:UUID}}),
            (SELECT any(stored_observation_count)
             FROM {database}.options_chain_snapshots
             WHERE snapshot_id = {{snapshot_id:UUID}}),
            (SELECT any(stored_detail_count)
             FROM {database}.options_chain_snapshots
             WHERE snapshot_id = {{snapshot_id:UUID}}),
            (SELECT any(stored_distinct_contract_count)
             FROM {database}.options_chain_snapshots
             WHERE snapshot_id = {{snapshot_id:UUID}}),
            (SELECT count() FROM {database}.options_chain_observations
             WHERE snapshot_id = {{snapshot_id:UUID}}),
            (SELECT count() FROM {database}.options_chain_observation_details
             WHERE snapshot_id = {{snapshot_id:UUID}}),
            (SELECT countDistinct(contract_id)
             FROM {database}.options_chain_observations
             WHERE snapshot_id = {{snapshot_id:UUID}}),
            (SELECT count()
             FROM {database}.options_chain_observations AS o
             LEFT JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c
               ON c.contract_id = o.contract_id
             WHERE o.snapshot_id = {{snapshot_id:UUID}}
               AND isNull(c.contract_id))
        """,
        parameters={"snapshot_id": str(snapshot_id)},
    )
    if not result.result_rows:
        return NormalizedSnapshotState(0, 0, 0, 0, 0, 0, 0, 0)
    return NormalizedSnapshotState(
        *(int(value or 0) for value in result.result_rows[0])
    )


def _repair_partial_normalization(
    client: Any, database: str, snapshot_id: uuid.UUID
) -> None:
    """Remove only bounded snapshot-owned partial source rows before replay."""
    for table in (
        "options_chain_snapshots",
        "options_chain_observation_details",
        "options_chain_observations",
    ):
        client.command(
            f"ALTER TABLE {database}.{table} DELETE "
            "WHERE snapshot_id = {snapshot_id:UUID}",
            parameters={"snapshot_id": str(snapshot_id)},
            settings={"mutations_sync": 2},
        )
    repaired = _normalized_snapshot_state(client, database, snapshot_id)
    if not repaired.empty:
        raise InconsistentSnapshotError(
            f"Bounded normalization cleanup did not clear {snapshot_id}: {repaired}"
        )


def _prepare_manifestless_replays(
    client: Any, database: str, snapshot_ids: Sequence[uuid.UUID]
) -> None:
    """Bulk-clear child rows before startup replays begin.

    Startup reconciliation runs before the ingestion thread exists, so all raw
    snapshots with no manifest can be cleaned in two bounded mutations without
    racing publication.  The ordinary single-snapshot cleanup remains the
    fallback for partial writes discovered after startup.
    """
    validate_database_identifier(database)
    ids = tuple(dict.fromkeys(str(snapshot_id) for snapshot_id in snapshot_ids))
    if not ids:
        return
    batch_count = math.ceil(len(ids) / _MANIFESTLESS_REPLAY_BATCH_SIZE)
    logger = logging.getLogger(__name__)
    for offset in range(0, len(ids), _MANIFESTLESS_REPLAY_BATCH_SIZE):
        batch = ids[offset : offset + _MANIFESTLESS_REPLAY_BATCH_SIZE]
        batch_number = offset // _MANIFESTLESS_REPLAY_BATCH_SIZE + 1
        parameters = {"snapshot_ids": list(batch)}
        logger.info(
            "event=collector_startup_bulk_cleanup_batch batch=%s batches=%s "
            "snapshots=%s action=delete_children",
            batch_number,
            batch_count,
            len(batch),
        )
        for table in (
            "options_chain_observation_details",
            "options_chain_observations",
        ):
            client.command(
                f"ALTER TABLE {database}.{table} DELETE "
                "WHERE snapshot_id IN {snapshot_ids:Array(UUID)}",
                parameters=parameters,
                settings={"mutations_sync": 2},
            )
        result = client.query(
            f"""
            SELECT
                (SELECT count() FROM {database}.options_chain_observations
                 WHERE snapshot_id IN {{snapshot_ids:Array(UUID)}}),
                (SELECT count() FROM {database}.options_chain_observation_details
                 WHERE snapshot_id IN {{snapshot_ids:Array(UUID)}})
            """,
            parameters=parameters,
        )
        remaining = result.result_rows[0] if result.result_rows else (0, 0)
        if any(int(value or 0) for value in remaining):
            raise InconsistentSnapshotError(
                "Bulk startup cleanup left manifestless child rows: "
                f"batch={batch_number}/{batch_count}, "
                f"observations={int(remaining[0] or 0)}, "
                f"details={int(remaining[1] or 0)}"
            )
        logger.info(
            "event=collector_startup_bulk_cleanup_batch batch=%s batches=%s "
            "snapshots=%s status=success",
            batch_number,
            batch_count,
            len(batch),
        )


def _definition_configuration(
    context: Mapping[str, Any], definition_type: DefinitionType
) -> dict[str, Any]:
    semantic_keys = {
        DefinitionType.COLLECTION: frozenset(
            {
                "options_chain.symbols",
                "options_chain.query_interval",
                "options_chain.api_strike_count",
                "options_chain.api_filter_expirations",
                "options_chain.max_tenor_age",
            }
        ),
        DefinitionType.NORMALIZATION: frozenset(
            {
                "options_chain.max_tenor_age",
                "options_chain.num_strikes_each_side",
                "options_chain.include_nonstandard",
                "options_chain.include_minis",
                "options_chain.expiration_grace_seconds",
            }
        ),
    }[definition_type]
    return {
        key: context[key]
        for key in sorted(context)
        if key in semantic_keys
    }


def _collector_code_version(context: Mapping[str, Any]) -> str:
    """Return best-effort provenance metadata for collector definitions.

    Code version is descriptive metadata, not a processing-compatibility gate.
    Configuration hashes define whether an existing collection or
    normalization definition can process a snapshot.
    """
    explicit = str(context.get("options_chain.code_version") or "").strip()
    if explicit:
        return explicit
    try:
        installed = str(package_version("algotrading")).strip()
    except PackageNotFoundError:
        installed = ""
    if not installed:
        return "unavailable"
    revision = str(os.getenv("ALGOTRADING_BUILD_REVISION") or "").strip()
    return (
        f"algotrading-{installed}+{revision}"
        if revision
        else f"algotrading-{installed}"
    )


def _explicit_definition_activation(context: Mapping[str, Any]) -> Optional[datetime]:
    value = context.get("options_chain.definition_activation_at")
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "options_chain.definition_activation_at must be timezone-aware UTC"
            )
        if value.utcoffset() != timedelta(0):
            raise ValueError(
                "options_chain.definition_activation_at must use the UTC timezone"
            )
    elif isinstance(value, str):
        try:
            parsed_literal = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                "options_chain.definition_activation_at must be an ISO-8601 UTC timestamp"
            ) from exc
        if parsed_literal.tzinfo is None or parsed_literal.utcoffset() != timedelta(0):
            raise ValueError(
                "options_chain.definition_activation_at must be timezone-aware UTC"
            )
    else:
        raise ValueError(
            "options_chain.definition_activation_at must be an ISO-8601 UTC timestamp"
        )
    parsed = _parse_timestamp(value)
    if parsed is None:
        raise ValueError("options_chain.definition_activation_at is invalid")
    return parsed


def _definition_identity_matches(left: Any, right: Any) -> bool:
    # Code version is provenance only.  It must never prevent a raw snapshot
    # from using the configuration-compatible definition applicable to its
    # source timestamp.
    return (
        left.definition_type is right.definition_type
        and left.parent_definition_id == right.parent_definition_id
        and left.configuration_hash == right.configuration_hash
        and left.solver_configuration_hash == right.solver_configuration_hash
    )


def _latest_normalized_source_at(client: Any, database: str) -> Optional[datetime]:
    rows = client.query(
        f"SELECT maxOrNull(source_collected_at) "
        f"FROM {database}.options_chain_snapshots"
    ).result_rows
    if len(rows) != 1:
        raise RuntimeError("Unable to determine latest normalized source timestamp")
    return _parse_timestamp(rows[0][0]) if rows[0][0] is not None else None


def _resolve_collector_definition(
    context: Mapping[str, Any],
    client: Any,
    database: str,
    definition_type: DefinitionType,
    source_at: datetime,
) -> int:
    source_at = _parse_timestamp(source_at)
    if source_at is None:
        raise ValueError("Definition resolution requires a valid UTC source timestamp")
    repository = DefinitionRepository(client, database)
    history = repository.load(definition_type)
    expected = build_definition(
        definition_type,
        code_version=_collector_code_version(context),
        configuration=_definition_configuration(context, definition_type),
        activated_at=source_at,
    )
    applicable = [item for item in history if item.is_active and item.applies_at(source_at)]
    if len(applicable) > 1:
        raise ValueError(
            f"Overlapping {definition_type.value} definitions at {source_at.isoformat()}"
        )
    if applicable and _definition_identity_matches(applicable[0], expected):
        return applicable[0].definition_id

    latest_published = _latest_normalized_source_at(client, database)
    if latest_published is not None and source_at <= latest_published:
        raise ValueError(
            "historical definition mismatch: expected collector identity differs from "
            f"the {definition_type.value} definition applicable at "
            f"{source_at.isoformat()}"
        )
    matching_history = [
        item for item in history if _definition_identity_matches(item, expected)
    ]
    if matching_history:
        raise ValueError(
            f"Historical {definition_type.value} identity exists outside its applicable "
            "interval and cannot be reopened"
        )

    explicit_activation = _explicit_definition_activation(context)
    activation = explicit_activation or source_at
    if activation > source_at:
        raise ValueError(
            "options_chain.definition_activation_at cannot be after the source snapshot"
        )
    if latest_published is not None and activation <= latest_published:
        raise ValueError(
            "options_chain.definition_activation_at must be after already-published history"
        )
    candidate = build_definition(
        definition_type,
        code_version=expected.code_version,
        configuration=_definition_configuration(context, definition_type),
        activated_at=activation,
    )
    installed = repository.activate_prospective(candidate)
    resolved = repository.resolve(definition_type, source_at)
    if resolved.definition_id != installed.definition_id:
        raise RuntimeError(
            f"Activated {definition_type.value} definition did not resolve at source time"
        )
    logging.getLogger(__name__).info(
        "event=collector_definition_activation definition_type=%s definition_id=%s "
        "activated_at=%s",
        definition_type.value,
        installed.definition_id,
        installed.activated_at.isoformat(),
    )
    return resolved.definition_id


def _load_current_contracts(
    client: Any, database: str, contract_ids: Sequence[int]
) -> dict[int, Mapping[str, Any]]:
    if not contract_ids:
        return {}
    table = client.query_arrow(
        f"SELECT * FROM {database}.options_contracts FINAL "
        "WHERE contract_id IN {contract_ids:Array(UInt64)}",
        parameters={"contract_ids": [int(value) for value in contract_ids]},
    )
    return {int(row["contract_id"]): row for row in table.to_pylist()}


def _publish_normalized_snapshot(
    client: Any,
    database: str,
    normalized: NormalizedSnapshot,
    *,
    publication_id: uuid.UUID,
) -> None:
    normalized.validate()
    snapshot_id = str(normalized.snapshot["snapshot_id"])
    publication_token = f"{snapshot_id}:{publication_id}"
    existing = _load_current_contracts(
        client, database, [int(row["contract_id"]) for row in normalized.contracts]
    )
    changed_contracts = []
    for candidate in normalized.contracts:
        action, row = merge_contract_metadata(
            candidate, existing.get(int(candidate["contract_id"]))
        )
        if action != "unchanged":
            changed_contracts.append(dict(row))
    if changed_contracts:
        client.insert_arrow(
            f"{database}.options_contracts",
            _rows_to_arrow(changed_contracts, CONTRACT_SCHEMA),
            settings={"insert_deduplication_token": f"contracts:{snapshot_id}"},
        )
    client.insert_arrow(
        f"{database}.options_chain_observations",
        _rows_to_arrow([dict(row) for row in normalized.observations], OBSERVATION_SCHEMA),
        settings={
            "insert_deduplication_token": f"observations:{publication_token}"
        },
    )
    client.insert_arrow(
        f"{database}.options_chain_observation_details",
        _rows_to_arrow([dict(row) for row in normalized.details], DETAIL_SCHEMA),
        settings={
            "insert_deduplication_token": f"observation-details:{publication_token}"
        },
    )
    before_manifest = _normalized_snapshot_state(
        client, database, uuid.UUID(snapshot_id)
    )
    if (
        before_manifest.manifest_rows != 0
        or before_manifest.observation_rows
        != int(normalized.snapshot["stored_observation_count"])
        or before_manifest.detail_rows
        != int(normalized.snapshot["stored_detail_count"])
        or before_manifest.distinct_contracts
        != int(normalized.snapshot["stored_distinct_contract_count"])
        or before_manifest.missing_contracts
    ):
        raise InconsistentSnapshotError(
            f"Pre-manifest normalization verification failed for {snapshot_id}: "
            f"{before_manifest}"
        )
    # This is the publication barrier and must remain the final insert.
    client.insert_arrow(
        f"{database}.options_chain_snapshots",
        _rows_to_arrow([dict(normalized.snapshot)], SNAPSHOT_SCHEMA),
        settings={
            "insert_deduplication_token": f"snapshot-manifest:{publication_token}"
        },
    )


def ingest_raw_snapshot(
    context: Mapping[str, Any],
    client: Any,
    ref: RawSnapshotRef,
    *,
    ingested_at_utc: Optional[datetime] = None,
    initial_state: Optional[NormalizedSnapshotState] = None,
) -> int:
    database = str(context["options_chain.clickhouse.database"])
    state = initial_state if initial_state is not None else _normalized_snapshot_state(
        client, database, ref.snapshot_id
    )
    if state.complete:
        return state.observation_rows
    repaired_partial = state.partial
    if state.partial:
        logging.getLogger(__name__).warning(
            "event=collector_snapshot_repair snapshot_id=%s before=partial action=delete",
            ref.snapshot_id,
        )
        _repair_partial_normalization(client, database, ref.snapshot_id)
        logging.getLogger(__name__).info(
            "event=collector_snapshot_repair snapshot_id=%s after=empty action=replay",
            ref.snapshot_id,
        )
    try:
        record, response = read_raw_snapshot(client, database, ref.snapshot_id)
        if int(record["source_contract_count"]) != _count_contracts(response):
            raise ValueError(
                "raw metadata contract count does not match the JSON payload"
            )
        rows = flatten_and_filter_response(
            response,
            snapshot_id=ref.snapshot_id,
            collected_at_utc=record["collected_at_utc"],
            request_started_at_utc=record["request_started_at_utc"],
            response_received_at_utc=record["response_received_at_utc"],
            ingested_at_utc=ingested_at_utc,
            max_tenor_age=int(context.get("options_chain.max_tenor_age", 365)),
            num_strikes_each_side=int(
                context.get("options_chain.num_strikes_each_side", 20)
            ),
            include_nonstandard=_as_bool(
                context.get("options_chain.include_nonstandard", False)
            ),
            include_minis=_as_bool(context.get("options_chain.include_minis", False)),
            expiration_grace_seconds=int(
                context.get("options_chain.expiration_grace_seconds", 0)
            ),
        )
        if not rows:
            raise ValueError("snapshot produced no stored contracts")
        option_symbols = [row["option_symbol"] for row in rows]
        if any(not symbol for symbol in option_symbols):
            raise ValueError("snapshot contains a contract without an option symbol")
        if len(option_symbols) != len(set(option_symbols)):
            raise ValueError("snapshot contains duplicate option symbols")
        if any(row["put_call"] not in {"CALL", "PUT"} for row in rows):
            raise ValueError("snapshot contains an invalid putCall value")
        source_at = _parse_timestamp(record["collected_at_utc"])
        if source_at is None:
            raise ValueError("raw collected timestamp is invalid")
        collection_definition_id = _resolve_collector_definition(
            context, client, database, DefinitionType.COLLECTION, source_at
        )
        normalization_definition_id = _resolve_collector_definition(
            context, client, database, DefinitionType.NORMALIZATION, source_at
        )
        normalized = build_normalized_snapshot(
            record,
            rows,
            collection_definition_id=collection_definition_id,
            normalization_definition_id=normalization_definition_id,
            normalized_at=ingested_at_utc or _utc_now(),
        )
    except (SnapshotDataError, InconsistentSnapshotError):
        raise
    except Exception as exc:
        classification = classify_database_error(exc)
        if (
            classification.retryable
            or classification.clickhouse_code is not None
            or type(exc).__module__.startswith("clickhouse_connect")
        ):
            raise
        raise SnapshotDataError(
            f"Snapshot {ref.snapshot_id} data is not ingestible: {exc}"
        ) from exc

    try:
        _publish_normalized_snapshot(
            client,
            database,
            normalized,
            publication_id=uuid.uuid4(),
        )
    except Exception:
        ambiguous_state = _normalized_snapshot_state(client, database, ref.snapshot_id)
        if not ambiguous_state.complete or ambiguous_state.observation_rows != len(rows):
            raise
    state = _normalized_snapshot_state(client, database, ref.snapshot_id)
    if not state.complete or state.observation_rows != len(rows):
        raise RuntimeError(
            f"Normalized publication verification failed for {ref.snapshot_id}: "
            f"{state}, sent={len(rows)}"
        )
    if repaired_partial:
        logging.getLogger(__name__).info(
            "event=collector_snapshot_repair snapshot_id=%s replay=success",
            ref.snapshot_id,
        )
    return state.observation_rows


def reconciliation_query(database: str, *, explain: bool = False) -> str:
    database = validate_database_identifier(database)
    prefix = "EXPLAIN PLAN\n" if explain else ""
    return prefix + f"""
        WITH
        raw AS
        (
            SELECT
                snapshot_id,
                any(symbol) AS raw_symbol,
                any(collected_at_utc) AS raw_collected_at_utc,
                count() AS raw_rows
            FROM {database}.options_chain_raw
            WHERE collected_at_utc >= {{cutoff:DateTime64(6, 'UTC')}}
            GROUP BY snapshot_id
        ),
        manifests AS
        (
            SELECT snapshot_id, count() AS manifest_rows
            FROM {database}.options_chain_snapshots
            WHERE source_collected_at >= {{cutoff:DateTime64(6, 'UTC')}}
            GROUP BY snapshot_id
        )
        SELECT
            toString(r.snapshot_id), r.raw_symbol, r.raw_collected_at_utc, r.raw_rows,
            coalesce(m.manifest_rows, toUInt64(0)) AS manifest_rows
        FROM raw AS r
        LEFT JOIN manifests AS m ON m.snapshot_id = r.snapshot_id
        WHERE r.raw_rows != 1 OR manifest_rows != 1
        ORDER BY r.raw_collected_at_utc
        """


def discover_raw_to_normalized_work(
    context: Mapping[str, Any], client: Any
) -> ReconciliationWork:
    """Discover publication-barrier defects from raw rows and manifests only."""
    database = validate_database_identifier(
        str(context["options_chain.clickhouse.database"])
    )
    cutoff = _utc_now() - timedelta(
        days=int(context.get("options_chain.reconcile_days", 7))
    )
    result = client.query(
        reconciliation_query(database),
        parameters={"cutoff": cutoff},
    )
    pending: list[RawSnapshotRef] = []
    duplicate_raw: list[tuple[uuid.UUID, int]] = []
    duplicate_manifests: list[RawSnapshotRef] = []
    for (
        snapshot_id,
        symbol,
        collected_at,
        raw_rows,
        manifest_rows,
    ) in result.result_rows:
        parsed_id = uuid.UUID(str(snapshot_id))
        if int(raw_rows) != 1:
            duplicate_raw.append((parsed_id, int(raw_rows)))
            continue
        ref = RawSnapshotRef(
            parsed_id,
            str(symbol),
            _parse_timestamp(collected_at),
        )
        if int(manifest_rows or 0) == 0:
            pending.append(ref)
        elif int(manifest_rows or 0) > 1:
            duplicate_manifests.append(ref)
    return ReconciliationWork(
        tuple(pending), tuple(duplicate_raw), tuple(duplicate_manifests)
    )


def run_integrity_audit(
    context: Mapping[str, Any],
    client: Any,
    *,
    days: int,
    maximum_snapshots: int,
) -> IntegrityAuditReport:
    """Run a bounded, read-only audit outside the reconciliation hot path."""
    if days <= 0:
        raise ValueError("Integrity audit days must be positive")
    if maximum_snapshots <= 0:
        raise ValueError("Integrity audit maximum snapshots must be positive")
    database = validate_database_identifier(
        str(context["options_chain.clickhouse.database"])
    )
    cutoff = _utc_now() - timedelta(days=days)
    snapshot_rows = client.query(
        f"""
        SELECT toString(snapshot_id)
        FROM
        (
            SELECT snapshot_id, max(collected_at_utc) AS source_at
            FROM {database}.options_chain_raw
            WHERE collected_at_utc >= {{cutoff:DateTime64(6, 'UTC')}}
            GROUP BY snapshot_id
            UNION ALL
            SELECT snapshot_id, max(source_collected_at) AS source_at
            FROM {database}.options_chain_snapshots
            WHERE source_collected_at >= {{cutoff:DateTime64(6, 'UTC')}}
            GROUP BY snapshot_id
        )
        GROUP BY snapshot_id
        ORDER BY max(source_at) DESC
        LIMIT {{maximum_snapshots:UInt32}}
        """,
        parameters={"cutoff": cutoff, "maximum_snapshots": maximum_snapshots},
    ).result_rows

    finding_counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}

    def finding(name: str, snapshot_id: uuid.UUID) -> None:
        finding_counts[name] = finding_counts.get(name, 0) + 1
        values = samples.setdefault(name, [])
        if len(values) < 5:
            values.append(str(snapshot_id))

    for (raw_snapshot_id,) in snapshot_rows:
        snapshot_id = uuid.UUID(str(raw_snapshot_id))
        rows = client.query(
            f"""
            WITH manifest AS
            (
                SELECT
                    any(payload_sha256) AS payload_sha256,
                    any(underlying_symbol) AS underlying_symbol,
                    any(source_collected_at) AS source_collected_at,
                    any(stored_observation_count) AS expected_observations,
                    any(stored_detail_count) AS expected_details,
                    any(stored_distinct_contract_count) AS expected_contracts,
                    any(collection_definition_id) AS collection_definition_id,
                    any(normalization_definition_id) AS normalization_definition_id
                FROM {database}.options_chain_snapshots
                WHERE snapshot_id = {{snapshot_id:UUID}}
            )
            SELECT
                (SELECT count() FROM {database}.options_chain_raw
                 WHERE snapshot_id = {{snapshot_id:UUID}}),
                (SELECT count() FROM {database}.options_chain_snapshots
                 WHERE snapshot_id = {{snapshot_id:UUID}}),
                (SELECT any(payload_sha256) FROM {database}.options_chain_raw
                 WHERE snapshot_id = {{snapshot_id:UUID}}),
                (SELECT payload_sha256 FROM manifest),
                (SELECT any(symbol) FROM {database}.options_chain_raw
                 WHERE snapshot_id = {{snapshot_id:UUID}}),
                (SELECT any(collected_at_utc) FROM {database}.options_chain_raw
                 WHERE snapshot_id = {{snapshot_id:UUID}}),
                (SELECT underlying_symbol FROM manifest),
                (SELECT source_collected_at FROM manifest),
                (SELECT expected_observations FROM manifest),
                (SELECT expected_details FROM manifest),
                (SELECT expected_contracts FROM manifest),
                (SELECT count() FROM {database}.options_chain_observations
                 WHERE snapshot_id = {{snapshot_id:UUID}}
                   AND source_collected_at = (SELECT source_collected_at FROM manifest)),
                (SELECT count() FROM {database}.options_chain_observation_details
                 WHERE snapshot_id = {{snapshot_id:UUID}}
                   AND source_collected_at = (SELECT source_collected_at FROM manifest)),
                (SELECT countDistinct(contract_id)
                 FROM {database}.options_chain_observations
                 WHERE snapshot_id = {{snapshot_id:UUID}}
                   AND source_collected_at = (SELECT source_collected_at FROM manifest)),
                (SELECT count() FROM {database}.options_chain_observations AS o
                 LEFT JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c
                   ON c.contract_id = o.contract_id
                 WHERE o.snapshot_id = {{snapshot_id:UUID}}
                   AND o.source_collected_at = (SELECT source_collected_at FROM manifest)
                   AND isNull(c.contract_id)),
                (SELECT count() - countDistinct(contract_id)
                 FROM {database}.options_chain_observations
                 WHERE snapshot_id = {{snapshot_id:UUID}}
                   AND source_collected_at = (SELECT source_collected_at FROM manifest)),
                (SELECT count() FROM {database}.options_chain_observation_details AS d
                 LEFT JOIN {database}.options_chain_observations AS o
                   ON o.snapshot_id = d.snapshot_id
                  AND o.source_collected_at = d.source_collected_at
                  AND o.contract_id = d.contract_id
                 WHERE d.snapshot_id = {{snapshot_id:UUID}}
                   AND d.source_collected_at = (SELECT source_collected_at FROM manifest)
                   AND isNull(o.contract_id)),
                (SELECT count() FROM {database}.options_chain_observations
                 WHERE snapshot_id = {{snapshot_id:UUID}}
                   AND (underlying_symbol != (SELECT underlying_symbol FROM manifest)
                     OR source_collected_at != (SELECT source_collected_at FROM manifest))),
                (SELECT count() FROM {database}.options_chain_observation_details
                 WHERE snapshot_id = {{snapshot_id:UUID}}
                   AND (underlying_symbol != (SELECT underlying_symbol FROM manifest)
                     OR source_collected_at != (SELECT source_collected_at FROM manifest))),
                (SELECT count() FROM {database}.options_chain_definitions FINAL
                 WHERE definition_id = (SELECT collection_definition_id FROM manifest)
                   AND definition_type = 'collection' AND is_active
                   AND activated_at <= (SELECT source_collected_at FROM manifest)
                   AND (isNull(deactivated_at)
                     OR (SELECT source_collected_at FROM manifest) < deactivated_at)),
                (SELECT count() FROM {database}.options_chain_definitions FINAL
                 WHERE definition_id = (SELECT normalization_definition_id FROM manifest)
                   AND definition_type = 'normalization' AND is_active
                   AND activated_at <= (SELECT source_collected_at FROM manifest)
                   AND (isNull(deactivated_at)
                     OR (SELECT source_collected_at FROM manifest) < deactivated_at)),
                (SELECT count() FROM {database}.options_chain_observations AS o
                 INNER JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c
                   ON c.contract_id = o.contract_id
                 WHERE o.snapshot_id = {{snapshot_id:UUID}}
                   AND o.source_collected_at = (SELECT source_collected_at FROM manifest)
                   AND c.underlying_symbol != o.underlying_symbol)
            """,
            parameters={"snapshot_id": str(snapshot_id)},
        ).result_rows
        if len(rows) != 1:
            raise RuntimeError(f"Integrity audit returned no state for {snapshot_id}")
        (
            raw_count,
            manifest_count,
            raw_hash,
            manifest_hash,
            raw_symbol,
            raw_at,
            manifest_symbol,
            manifest_at,
            expected_observations,
            expected_details,
            expected_contracts,
            observation_count,
            detail_count,
            distinct_contract_count,
            missing_contracts,
            duplicate_observations,
            orphan_details,
            observation_owner_mismatches,
            detail_owner_mismatches,
            collection_definition_matches,
            normalization_definition_matches,
            contract_underlying_mismatches,
        ) = rows[0]
        raw_count = int(raw_count or 0)
        manifest_count = int(manifest_count or 0)
        if raw_count != 1:
            finding("raw_row_count", snapshot_id)
        if manifest_count != 1:
            finding("manifest_row_count", snapshot_id)
        if raw_count == 1 and manifest_count == 1:
            if bytes(raw_hash) != bytes(manifest_hash):
                finding("payload_hash_mismatch", snapshot_id)
            if str(raw_symbol) != str(manifest_symbol):
                finding("symbol_mismatch", snapshot_id)
            if _parse_timestamp(raw_at) != _parse_timestamp(manifest_at):
                finding("source_timestamp_mismatch", snapshot_id)
        if int(observation_count or 0) != int(expected_observations or 0):
            finding("observation_count_mismatch", snapshot_id)
        if int(detail_count or 0) != int(expected_details or 0):
            finding("detail_count_mismatch", snapshot_id)
        if int(distinct_contract_count or 0) != int(expected_contracts or 0):
            finding("distinct_contract_count_mismatch", snapshot_id)
        for name, value in (
            ("missing_contract", missing_contracts),
            ("duplicate_observation", duplicate_observations),
            ("orphan_detail", orphan_details),
            ("observation_owner_mismatch", observation_owner_mismatches),
            ("detail_owner_mismatch", detail_owner_mismatches),
            ("contract_underlying_mismatch", contract_underlying_mismatches),
        ):
            if int(value or 0):
                finding(name, snapshot_id)
        if manifest_count == 1 and int(collection_definition_matches or 0) != 1:
            finding("collection_definition_invalid", snapshot_id)
        if manifest_count == 1 and int(normalization_definition_matches or 0) != 1:
            finding("normalization_definition_invalid", snapshot_id)

    return IntegrityAuditReport(
        database=database,
        bounded_days=days,
        maximum_snapshots=maximum_snapshots,
        audited_snapshots=len(snapshot_rows),
        finding_counts=dict(sorted(finding_counts.items())),
        sample_snapshot_ids={
            name: tuple(values) for name, values in sorted(samples.items())
        },
    )


def collector_preflight(
    context: Mapping[str, Any], client: Any
) -> CollectorPreflightReport:
    """Prove generation-3 schema and representative Collector reads are valid."""
    database = validate_database_identifier(
        str(context["options_chain.clickhouse.database"])
    )
    version_rows = client.query("SELECT version()").result_rows
    if len(version_rows) != 1 or not str(version_rows[0][0]).strip():
        raise SchemaValidationError("ClickHouse did not return a server version")
    server_version = str(version_rows[0][0])
    try:
        schema_report = validate_canonical_table_metadata(
            client, database, COLLECTOR_PREFLIGHT_TABLES
        )
    except SchemaValidationError as exc:
        raise SchemaValidationError(
            f"Collector preflight failed for database={database} "
            f"server_version={server_version}: {exc}"
        ) from exc

    dataset_rows = client.query(
        f"""
        SELECT
            toString(dataset_generation_id), schema_generation,
            default_collection_definition_id,
            default_normalization_definition_id,
            default_calculation_definition_id,
            default_surface_definition_id
        FROM {database}.options_chain_dataset_metadata FINAL
        WHERE is_active
        """
    ).result_rows
    if len(dataset_rows) != 1:
        raise SchemaValidationError(
            "Generation-3 preflight requires exactly one active dataset row; "
            f"found {len(dataset_rows)}"
        )
    if int(dataset_rows[0][1]) != SCHEMA_GENERATION:
        raise SchemaValidationError(
            f"Dataset schema generation {dataset_rows[0][1]} does not match "
            f"Collector generation {SCHEMA_GENERATION}"
        )
    default_definition_ids = [
        int(value) for value in dataset_rows[0][2:] if int(value or 0) != 0
    ]
    if default_definition_ids:
        resolved_rows = client.query(
            f"SELECT countDistinct(definition_id) "
            f"FROM {database}.options_chain_definitions FINAL "
            "WHERE definition_id IN {definition_ids:Array(UInt32)} AND is_active",
            parameters={"definition_ids": default_definition_ids},
        ).result_rows
        resolved = int(resolved_rows[0][0] or 0) if resolved_rows else 0
        if resolved != len(set(default_definition_ids)):
            raise SchemaValidationError(
                "One or more active dataset default definitions do not resolve"
            )

    invalid_snapshot_definitions = client.query(
        f"""
        SELECT count()
        FROM {database}.options_chain_snapshots AS s
        LEFT JOIN (SELECT * FROM {database}.options_chain_definitions FINAL) AS cd
          ON cd.definition_id = s.collection_definition_id
         AND cd.definition_type = 'collection' AND cd.is_active
        LEFT JOIN (SELECT * FROM {database}.options_chain_definitions FINAL) AS nd
          ON nd.definition_id = s.normalization_definition_id
         AND nd.definition_type = 'normalization' AND nd.is_active
        WHERE isNull(cd.definition_id) OR isNull(nd.definition_id)
           OR cd.activated_at > s.source_collected_at
           OR (isNotNull(cd.deactivated_at)
               AND s.source_collected_at >= cd.deactivated_at)
           OR nd.activated_at > s.source_collected_at
           OR (isNotNull(nd.deactivated_at)
               AND s.source_collected_at >= nd.deactivated_at)
        """
    ).result_rows
    invalid_count = (
        int(invalid_snapshot_definitions[0][0] or 0)
        if invalid_snapshot_definitions
        else 0
    )
    if invalid_count:
        raise SchemaValidationError(
            f"{invalid_count} snapshots reference missing or inapplicable definitions"
        )

    # This query must be decidable before any capture write is permitted.
    lease_active = maintenance_lease_active(client, database)
    work = discover_raw_to_normalized_work(context, client)
    sample_rows = client.query(
        f"""
        SELECT toString(snapshot_id)
        FROM {database}.options_chain_snapshots
        GROUP BY snapshot_id
        HAVING count() = 1
        ORDER BY max(source_collected_at) DESC
        LIMIT 1
        """
    ).result_rows
    sampled_snapshot_id = (
        uuid.UUID(str(sample_rows[0][0])) if sample_rows else None
    )
    sampled_state = (
        _normalized_snapshot_state(client, database, sampled_snapshot_id)
        if sampled_snapshot_id is not None
        else None
    )
    return CollectorPreflightReport(
        database=database,
        server_version=server_version,
        schema_generation=SCHEMA_GENERATION,
        active_dataset_count=len(dataset_rows),
        schema=schema_report,
        sampled_snapshot_id=sampled_snapshot_id,
        sampled_snapshot_state=sampled_state,
        reconciliation_candidate_count=(
            len(work.pending)
            + len(work.duplicate_manifests)
            + len(work.duplicate_raw)
        ),
        maintenance_lease_active=lease_active,
    )


def _normalized_symbols(value: Any) -> list[str]:
    values = value.split(",") if isinstance(value, str) else value
    output: list[str] = []
    seen: set[str] = set()
    for item in values or []:
        symbol = str(item).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            output.append(symbol)
    return output


class DailyAppendFileHandler(logging.Handler):
    """Append handler whose filename changes at local midnight."""

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
        path = self.directory / f"options_chain_collector_{current_date:%Y_%m_%d}.log"
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
    logger = logging.getLogger("OptionsChainCollector")
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s|%(levelname)s|%(threadName)s| %(message)s"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = DailyAppendFileHandler(Path(context[PARAM_LOG_PATH]))
    file_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.setLevel(context.get(PARAM_DEFAULT_LOG_LEVEL, logging.INFO))
    return logger


class OptionsChainCollector:
    def __init__(
        self,
        context: dict[str, Any],
        *,
        refdata: Any,
        monitor: Any,
        client_factory: Any = create_clickhouse_client,
    ):
        self.context = context
        self.logger = context[PARAM_LOGGER]
        self.refdata = refdata
        self.monitor = monitor
        self.client_factory = client_factory
        self.raw_queue: queue.SimpleQueue = queue.SimpleQueue()
        self.health = HealthState()
        self.stop_capture = threading.Event()
        self.capture_thread: Optional[threading.Thread] = None
        self.ingestion_thread: Optional[threading.Thread] = None
        self._last_monitor = 0.0
        self._continuous_capture = False
        self._capture_started_at_utc: Optional[datetime] = None
        self._queued_ids: set[uuid.UUID] = set()
        self._queued_lock = threading.Lock()
        self._repair_ids: set[uuid.UUID] = set()
        self._reconciliation_totals = {
            "discovered": 0,
            "normalized_successfully": 0,
            "repaired_successfully": 0,
            "already_complete": 0,
            "quarantined_ambiguous": 0,
            "failed": 0,
        }

    def validate(self, *, require_refdata: bool = True) -> None:
        if require_refdata and not callable(
            getattr(self.refdata, "get_options_chain", None)
        ):
            raise TypeError(
                "Configured reference-data class does not implement get_options_chain"
            )
        symbols = _normalized_symbols(self.context.get("options_chain.symbols", "MSFT"))
        if require_refdata and not symbols:
            raise ValueError("options_chain.symbols must contain at least one symbol")
        self.context["options_chain.symbols"] = symbols
        if float(self.context.get("options_chain.query_interval", 30)) <= 0:
            raise ValueError("options_chain.query_interval must be positive")
        if int(self.context.get("options_chain.capture_workers", 4)) <= 0:
            raise ValueError("options_chain.capture_workers must be positive")
        if int(self.context.get("options_chain.max_tenor_age", 365)) <= 0:
            raise ValueError("options_chain.max_tenor_age must be positive")
        if int(self.context.get("options_chain.num_strikes_each_side", 20)) < 0:
            raise ValueError("options_chain.num_strikes_each_side cannot be negative")
        _optional_positive_int(
            self.context.get(
                "options_chain.api_strike_count",
                self.context.get("options_chain.num_strikes_each_side", 20),
            ),
            "options_chain.api_strike_count",
        )
        if int(self.context.get("options_chain.expiration_grace_seconds", 0)) < 0:
            raise ValueError(
                "options_chain.expiration_grace_seconds cannot be negative"
            )
        if int(self.context.get("options_chain.reconcile_days", 7)) <= 0:
            raise ValueError("options_chain.reconcile_days must be positive")
        _explicit_definition_activation(self.context)
        if float(self.context.get("options_chain.monitor_interval", 300)) <= 0:
            raise ValueError("options_chain.monitor_interval must be positive")
        if float(self.context.get("options_chain.reconcile_interval_seconds", 300)) <= 0:
            raise ValueError("options_chain.reconcile_interval_seconds must be positive")
        raw_stale_seconds = float(
            self.context.get("options_chain.raw_capture_stale_seconds", 0)
        )
        if not math.isfinite(raw_stale_seconds) or raw_stale_seconds < 0:
            raise ValueError(
                "options_chain.raw_capture_stale_seconds must be finite and nonnegative"
            )
        if int(self.context.get("options_chain.database_reconnect_attempts", 5)) <= 0:
            raise ValueError("options_chain.database_reconnect_attempts must be positive")
        if float(
            self.context.get("options_chain.database_reconnect_max_seconds", 30)
        ) <= 0:
            raise ValueError("options_chain.database_reconnect_max_seconds must be positive")
        if (
            int(
                self.context.get("options_chain.nagios_consecutive_error_threshold", 10)
            )
            <= 0
        ):
            raise ValueError(
                "options_chain.nagios_consecutive_error_threshold must be positive"
            )
        collector_schema_statements(
            str(self.context["options_chain.clickhouse.database"])
        )
        if _as_bool(self.context.get("options_chain.collector.capture_only", False)):
            if not self.context.get("options_chain.migration_id"):
                raise ValueError(
                    "options_chain.collector.capture_only requires options_chain.migration_id"
                )
            self.health.update(capture_only=True)

    def _report_reconciliation_totals(self, started: float) -> None:
        self.logger.info(
            "event=collector_reconcile_totals discovered=%s normalized_successfully=%s "
            "repaired_successfully=%s already_complete=%s quarantined_ambiguous=%s "
            "failed=%s elapsed_seconds=%.6f",
            self._reconciliation_totals["discovered"],
            self._reconciliation_totals["normalized_successfully"],
            self._reconciliation_totals["repaired_successfully"],
            self._reconciliation_totals["already_complete"],
            self._reconciliation_totals["quarantined_ambiguous"],
            self._reconciliation_totals["failed"],
            time.monotonic() - started,
        )

    def _enqueue(self, ref: RawSnapshotRef) -> bool:
        with self._queued_lock:
            if ref.snapshot_id in self._queued_ids:
                return False
            self._queued_ids.add(ref.snapshot_id)
        self.raw_queue.put(ref)
        self.health.increment("queue_depth")
        return True

    def _finish_queued(self, snapshot_id: uuid.UUID) -> None:
        with self._queued_lock:
            self._queued_ids.discard(snapshot_id)

    def _reconcile(self, client: Any, *, prepare_replays: bool = False) -> int:
        started = time.monotonic()
        work = discover_raw_to_normalized_work(self.context, client)
        for ref in work.duplicate_manifests:
            self.logger.warning(
                "event=collector_snapshot_repair snapshot_id=%s "
                "before=duplicate_manifest action=bounded_replay_queued",
                ref.snapshot_id,
            )
        for snapshot_id, raw_count in work.duplicate_raw:
            if snapshot_id not in self._repair_ids:
                self._repair_ids.add(snapshot_id)
                self.health.require_repair()
                self._reconciliation_totals["quarantined_ambiguous"] += 1
                self.logger.critical(
                    "event=collector_snapshot_quarantine snapshot_id=%s "
                    "reason=duplicate_or_missing_raw raw_rows=%s",
                    snapshot_id,
                    raw_count,
                )
        replayable = [
            ref for ref in work.replayable if ref.snapshot_id not in self._repair_ids
        ]
        if prepare_replays:
            manifestless_ids = [
                ref.snapshot_id
                for ref in work.pending
                if ref.snapshot_id not in self._repair_ids
            ]
            if manifestless_ids:
                self.logger.warning(
                    "event=collector_startup_bulk_cleanup snapshots=%s action=delete_children",
                    len(manifestless_ids),
                )
                _prepare_manifestless_replays(
                    client,
                    str(self.context["options_chain.clickhouse.database"]),
                    manifestless_ids,
                )
                self.logger.info(
                    "event=collector_startup_bulk_cleanup snapshots=%s status=success",
                    len(manifestless_ids),
                )
        enqueued = sum(self._enqueue(ref) for ref in replayable)
        self._reconciliation_totals["discovered"] += len(work.replayable) + len(
            work.duplicate_raw
        )
        self.health.update(pending_raw_to_normalized=len(replayable))
        self.health.update(
            oldest_pending_raw_time=min(
                (ref.collected_at_utc for ref in replayable), default=None
            )
        )
        self.health.clear_consecutive_failure("database:reconciliation")
        self.logger.info(
            "event=collector_reconciliation discovered_missing_manifests=%s "
            "duplicate_raw=%s duplicate_manifests=%s enqueued=%s elapsed_seconds=%.6f",
            len(work.pending),
            len(work.duplicate_raw),
            len(work.duplicate_manifests),
            enqueued,
            time.monotonic() - started,
        )
        return enqueued

    def _is_market_hours(self) -> bool:
        if not _as_bool(self.context.get("options_chain.market_hours_only", True)):
            return True
        try:
            now = datetime.now()
            calendar = MarketData.get_trading_calendar(self.context, now, now)
            if calendar is None or len(calendar) != 1:
                return False
            session = calendar.iloc[0]
            session_date = _calendar_session_date(session.get("date"), now)
            open_at = _calendar_boundary_datetime(session["open"], session_date)
            close_at = _calendar_boundary_datetime(session["close"], session_date)
            return open_at <= now <= close_at
        except Exception:
            self.logger.error(
                "Market-calendar check failed; suppressing poll\n%s",
                traceback.format_exc(),
            )
            return False

    def _capture_one(self, symbol: str) -> None:
        database = validate_database_identifier(
            str(self.context["options_chain.clickhouse.database"])
        )
        lease_client = self.client_factory(self.context)
        try:
            if maintenance_lease_active(lease_client, database):
                self.logger.warning("Collector capture suppressed by active migration lease")
                return
        finally:
            close = getattr(lease_client, "close", None)
            if callable(close):
                close()
        snapshot_id = uuid.uuid4()
        response = None
        request_started = response_received = None
        request_parameters: dict[str, Any] = {}
        request_backoffs = (1, 2, 4, 8)
        for attempt in range(len(request_backoffs) + 1):
            request_started = _utc_now()
            api_strike_count = _optional_positive_int(
                self.context.get(
                    "options_chain.api_strike_count",
                    self.context.get("options_chain.num_strikes_each_side", 20),
                ),
                "options_chain.api_strike_count",
            )
            filter_expirations = _as_bool(
                self.context.get("options_chain.api_filter_expirations", True)
            )
            from_date = request_started.date() if filter_expirations else None
            to_date = (
                from_date
                + timedelta(
                    days=int(self.context.get("options_chain.max_tenor_age", 365))
                )
                if from_date is not None
                else None
            )
            request_parameters = {
                "symbol": symbol,
                "contractType": "ALL",
                "strategy": "SINGLE",
                "includeUnderlyingQuote": True,
                "strikeCount": api_strike_count,
                "fromDate": from_date.isoformat() if from_date else None,
                "toDate": to_date.isoformat() if to_date else None,
            }
            request_parameters = {
                key: value
                for key, value in request_parameters.items()
                if value is not None
            }
            try:
                response = self.refdata.get_options_chain(
                    symbol,
                    contract_type="ALL",
                    strategy="SINGLE",
                    strike_count=api_strike_count,
                    include_underlying_quote=True,
                    from_date=from_date,
                    to_date=to_date,
                )
                response_received = _utc_now()
                valid = (
                    isinstance(response, dict)
                    and response.get("symbol") is not None
                    and (
                        isinstance(response.get("callExpDateMap"), dict)
                        or isinstance(response.get("putExpDateMap"), dict)
                    )
                )
                if not valid:
                    raise ValueError(
                        "reference-data interface returned an invalid option-chain response"
                    )
                break
            except Exception as exc:
                self.health.increment("raw_request_failures")
                self.health.update(last_error=str(exc))
                if attempt == len(request_backoffs):
                    self.health.record_consecutive_failure(f"raw_request:{symbol}")
                    self.logger.error(
                        "Schwab request/validation failed for %s after %s attempts\n%s",
                        symbol,
                        attempt + 1,
                        traceback.format_exc(),
                    )
                    return
                if self.stop_capture.wait(request_backoffs[attempt]):
                    return

        collected = response_received
        record = build_raw_record(
            snapshot_id=snapshot_id,
            symbol=symbol,
            response=response,
            request_started_at_utc=request_started,
            response_received_at_utc=response_received,
            collected_at_utc=collected,
            request_parameters=request_parameters,
        )
        database_attempts = int(
            self.context.get("options_chain.database_reconnect_attempts", 5)
        )
        database_max_backoff = float(
            self.context.get("options_chain.database_reconnect_max_seconds", 30)
        )
        for attempt in range(database_attempts):
            client = None
            try:
                client = self.client_factory(self.context)
                ref = insert_raw_snapshot(
                    client,
                    str(self.context["options_chain.clickhouse.database"]),
                    record,
                    allow_purged_range_backfill=_as_bool(
                        self.context.get(
                            "options_chain.allow_purged_range_backfill", False
                        )
                    ),
                    confirm_database=self.context.get(
                        "options_chain.confirm_backfill_database"
                    ),
                    confirm_range_start=self.context.get(
                        "options_chain.confirm_backfill_range_start"
                    ),
                    confirm_range_end=self.context.get(
                        "options_chain.confirm_backfill_range_end"
                    ),
                )
                if not _as_bool(
                    self.context.get("options_chain.collector.capture_only", False)
                ):
                    self._enqueue(ref)
                with self.health.lock:
                    self.health.last_raw_commit_time[symbol] = collected
                    self.health.consecutive_failures.pop(f"raw_request:{symbol}", None)
                    self.health.consecutive_failures.pop(f"raw_database:{symbol}", None)
                self.logger.info(
                    "Committed raw ClickHouse snapshot %s for %s (%s source contracts)",
                    snapshot_id,
                    symbol,
                    record["source_contract_count"],
                )
                return
            except Exception as exc:
                self.health.increment("raw_database_failures")
                self.health.update(last_error=str(exc))
                classification = _log_database_error(
                    self.logger, "raw_snapshot_commit", exc
                )
                if not classification.retryable:
                    self.health.record_consecutive_failure(f"raw_database:{symbol}")
                    self.stop_capture.set()
                    raise
                if attempt == database_attempts - 1:
                    self.health.record_consecutive_failure(f"raw_database:{symbol}")
                    self.logger.error(
                        "Raw ClickHouse commit failed for %s snapshot %s after %s attempts\n%s",
                        symbol,
                        snapshot_id,
                        attempt + 1,
                        traceback.format_exc(),
                    )
                    return
                if self.stop_capture.wait(min(2**attempt, database_max_backoff)):
                    return
            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        self.logger.debug(
                            "Unable to close raw commit client", exc_info=True
                        )

    def _capture_loop(self, once: bool) -> None:
        threading.current_thread().name = "OptionsCapture"
        self._capture_started_at_utc = _utc_now()
        interval = float(self.context.get("options_chain.query_interval", 30))
        symbols = self.context["options_chain.symbols"]
        worker_count = min(
            int(self.context.get("options_chain.capture_workers", 4)), len(symbols)
        )
        try:
            with ThreadPoolExecutor(
                max_workers=worker_count, thread_name_prefix="OptionsCaptureWorker"
            ) as executor:
                while not self.stop_capture.is_set():
                    cycle_start = time.monotonic()
                    if self._is_market_hours():
                        futures = {
                            executor.submit(self._capture_one, symbol): symbol
                            for symbol in symbols
                            if not self.stop_capture.is_set()
                        }
                        pending = set(futures)
                        while pending:
                            done, pending = wait(
                                pending, timeout=0.5, return_when=FIRST_COMPLETED
                            )
                            for future in done:
                                if future.cancelled():
                                    continue
                                symbol = futures[future]
                                try:
                                    future.result()
                                except Exception as exc:
                                    self.health.increment("raw_request_failures")
                                    self.health.record_consecutive_failure(
                                        f"raw_request:{symbol}"
                                    )
                                    self.health.update(last_error=str(exc))
                                    self.logger.error(
                                        "Snapshot worker failed unexpectedly for %s\n%s",
                                        symbol,
                                        "".join(
                                            traceback.format_exception(
                                                type(exc), exc, exc.__traceback__
                                            )
                                        ),
                                    )
                    if once:
                        break
                    remaining = interval - (time.monotonic() - cycle_start)
                    if remaining > 0:
                        self.stop_capture.wait(remaining)
        except Exception as exc:
            self.health.increment("raw_request_failures")
            self.health.update(last_error=str(exc))
            self.logger.critical(
                "Capture thread terminated unexpectedly\n%s", traceback.format_exc()
            )

    def _ingestion_loop(self) -> None:
        threading.current_thread().name = "OptionsIngestion"
        client = None
        try:
            while True:
                reconnect_delay = 1.0
                reconnect_attempt = 0
                while client is None and not self.stop_capture.is_set():
                    try:
                        client = self.client_factory(self.context)
                        self.health.clear_consecutive_failure(
                            "database:ingestion_connection"
                        )
                    except Exception as exc:
                        reconnect_attempt += 1
                        self.health.increment("normalization_database_failures")
                        self.health.record_consecutive_failure(
                            "database:ingestion_connection"
                        )
                        self.health.update(last_error=str(exc))
                        classification = _log_database_error(
                            self.logger, "ingestion_connection", exc
                        )
                        maximum_attempts = int(
                            self.context.get(
                                "options_chain.database_reconnect_attempts", 5
                            )
                        )
                        if (
                            not classification.retryable
                            or reconnect_attempt >= maximum_attempts
                        ):
                            self.logger.critical(
                                "Ingestion connection failed permanently after %s attempts",
                                reconnect_attempt,
                            )
                            self.stop_capture.set()
                            return
                        if self.stop_capture.wait(reconnect_delay):
                            return
                        reconnect_delay = min(
                            reconnect_delay * 2.0,
                            float(
                                self.context.get(
                                    "options_chain.database_reconnect_max_seconds", 30
                                )
                            ),
                        )
                if client is None:
                    return
                ref = self.raw_queue.get()
                if ref is _SENTINEL:
                    break
                self.health.increment("queue_depth", -1)
                try:
                    before = _normalized_snapshot_state(
                        client,
                        str(self.context["options_chain.clickhouse.database"]),
                        ref.snapshot_id,
                    )
                    count = ingest_raw_snapshot(
                        self.context, client, ref, initial_state=before
                    )
                    self.health.clear_consecutive_failure("database:ingestion")
                    self.health.update(last_normalized_commit_time=_utc_now())
                    if before.complete:
                        self._reconciliation_totals["already_complete"] += 1
                    elif before.partial:
                        self._reconciliation_totals["repaired_successfully"] += 1
                    else:
                        self._reconciliation_totals["normalized_successfully"] += 1
                    self.logger.info(
                        "Committed %s normalized observations for snapshot %s",
                        count,
                        ref.snapshot_id,
                    )
                except SnapshotDataError as exc:
                    self._reconciliation_totals["failed"] += 1
                    self._repair_ids.add(ref.snapshot_id)
                    self.health.increment("normalization_parsing_failures")
                    self.health.require_repair()
                    self.health.update(last_error=str(exc))
                    self.logger.critical(
                        "Snapshot %s requires data repair: %s", ref.snapshot_id, exc
                    )
                except InconsistentSnapshotError as exc:
                    self._reconciliation_totals["failed"] += 1
                    self._repair_ids.add(ref.snapshot_id)
                    self.health.increment("normalization_database_failures")
                    self.health.require_repair()
                    self.health.update(last_error=str(exc))
                    self.logger.critical(
                        "Snapshot %s has a partial normalized publication: %s",
                        ref.snapshot_id,
                        exc,
                    )
                except Exception as exc:
                    self._reconciliation_totals["failed"] += 1
                    self.health.increment("normalization_database_failures")
                    self.health.record_consecutive_failure("database:ingestion")
                    self.health.update(last_error=str(exc))
                    classification = _log_database_error(
                        self.logger, "normalization", exc
                    )
                    self.logger.error(
                        "Normalization failed for %s; durable raw row remains pending\n%s",
                        ref.snapshot_id,
                        traceback.format_exc(),
                    )
                    if not classification.retryable:
                        self.logger.critical(
                            "Deterministic normalization blocker terminated capture"
                        )
                        self.stop_capture.set()
                    try:
                        client.close()
                    except Exception:
                        self.logger.debug(
                            "Unable to close failed ingestion client", exc_info=True
                        )
                    client = None
                    if not classification.retryable:
                        break
                finally:
                    self._finish_queued(ref.snapshot_id)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    self.logger.error(
                        "Unable to close ingestion ClickHouse client\n%s",
                        traceback.format_exc(),
                    )

    def _report_health(self, final: bool = False) -> None:
        snapshot = self.health.snapshot()
        threshold = int(
            self.context.get("options_chain.nagios_consecutive_error_threshold", 10)
        )
        breached = {
            key: value
            for key, value in snapshot["consecutive_failures"].items()
            if value >= threshold
        }
        capture_alive = bool(self.capture_thread and self.capture_thread.is_alive())
        ingestion_alive = bool(
            self.ingestion_thread and self.ingestion_thread.is_alive()
        )
        stale_backlog = False
        if snapshot["pending_raw_to_normalized"] and snapshot["oldest_pending_raw_time"]:
            stale_backlog = (
                _utc_now() - snapshot["oldest_pending_raw_time"]
            ).total_seconds() > max(
                float(self.context.get("options_chain.monitor_interval", 300)) * 2,
                float(self.context.get("options_chain.query_interval", 30)) * threshold,
            )
        reasons: list[str] = []
        level = NagiosSystemMonitor.MONITOR_OK
        raw_database_breach = any(key.startswith("raw_database:") for key in breached)
        stale_capture = False
        if (
            not final
            and self._continuous_capture
            and capture_alive
            and self._is_market_hours()
        ):
            request_timeout = self.context.get("requests.timeout", 5)
            if isinstance(request_timeout, (tuple, list)):
                request_timeout = max(float(value) for value in request_timeout)
            else:
                request_timeout = float(request_timeout)
            stale_after = max(
                float(self.context.get("options_chain.raw_capture_stale_seconds", 0)),
                float(self.context.get("options_chain.query_interval", 30))
                * max(3, threshold),
                ((request_timeout * 5.0) + sum((1, 2, 4, 8))) * threshold,
            )
            baseline = self._capture_started_at_utc or _utc_now()
            stale_capture = any(
                (
                    _utc_now() - snapshot["last_raw_commit_time"].get(symbol, baseline)
                ).total_seconds()
                > stale_after
                for symbol in self.context["options_chain.symbols"]
            )
        if snapshot["repair_required_errors"]:
            level = NagiosSystemMonitor.MONITOR_CRITICAL
            reasons.append("snapshot repair required")
        if raw_database_breach:
            level = NagiosSystemMonitor.MONITOR_CRITICAL
            reasons.append("raw ClickHouse commit failure threshold reached")
        if stale_capture:
            level = NagiosSystemMonitor.MONITOR_CRITICAL
            reasons.append("raw capture stale")
        if not final and self._continuous_capture and not capture_alive:
            level = NagiosSystemMonitor.MONITOR_CRITICAL
            reasons.append("capture coordinator stopped")
        if breached and level < NagiosSystemMonitor.MONITOR_WARNING:
            level = NagiosSystemMonitor.MONITOR_WARNING
            reasons.append("consecutive failure threshold reached")
        if stale_backlog and level < NagiosSystemMonitor.MONITOR_WARNING:
            level = NagiosSystemMonitor.MONITOR_WARNING
            reasons.append("raw-to-normalized backlog stale")
        if (
            not final
            and self.ingestion_thread is not None
            and not ingestion_alive
            and (self._continuous_capture or snapshot["queue_depth"])
        ):
            level = max(level, NagiosSystemMonitor.MONITOR_WARNING)
            reasons.append("normalization ingestion thread not alive")
        status = {
            NagiosSystemMonitor.MONITOR_OK: "OK",
            NagiosSystemMonitor.MONITOR_WARNING: "WARNING",
            NagiosSystemMonitor.MONITOR_CRITICAL: "CRITICAL",
        }.get(level, "UNKNOWN")
        message = (
            f"OptionsChainCollector {'final ' if final else ''}health; status={status}; "
            f"reasons={'; '.join(reasons) or 'none'}; queue={snapshot['queue_depth']}; "
            f"mode={'capture_only' if snapshot['capture_only'] else 'normal'}; "
            f"pending_raw_to_normalized={snapshot['pending_raw_to_normalized']}; "
            f"failures(request={snapshot['raw_request_failures']}, "
            f"raw_db={snapshot['raw_database_failures']}, "
            f"parse={snapshot['normalization_parsing_failures']}, "
            f"normalized_db={snapshot['normalization_database_failures']}); breached={breached}; "
            f"repair_required={snapshot['repair_required_errors']}"
        )
        if level >= NagiosSystemMonitor.MONITOR_CRITICAL:
            self.logger.error(message)
        elif level >= NagiosSystemMonitor.MONITOR_WARNING:
            self.logger.warning(message)
        else:
            self.logger.info(message)
        try:
            self.monitor.report_event(level, message)
        except Exception:
            self.logger.error(
                "Nagios health reporting failed\n%s", traceback.format_exc()
            )

    def _maybe_report_health(self) -> None:
        now = time.monotonic()
        if now - self._last_monitor >= float(
            self.context.get("options_chain.monitor_interval", 300)
        ):
            self._report_health()
            self._last_monitor = now

    def _open_main_client(self) -> Any:
        attempts = int(self.context.get("options_chain.database_reconnect_attempts", 5))
        max_delay = float(
            self.context.get("options_chain.database_reconnect_max_seconds", 30)
        )
        delay = 1.0
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return self.client_factory(self.context)
            except Exception as exc:
                last_error = exc
                self.health.increment("normalization_database_failures")
                self.health.record_consecutive_failure("database:main_connection")
                self.health.update(last_error=str(exc))
                classification = _log_database_error(
                    self.logger, "connection", exc
                )
                self.logger.error(
                    "Collector ClickHouse connection failed (attempt %s/%s): %s",
                    attempt,
                    attempts,
                    exc,
                )
                self._report_health()
                if not classification.retryable:
                    raise
                if attempt < attempts and self.stop_capture.wait(delay):
                    break
                delay = min(delay * 2.0, max_delay)
        raise RuntimeError(
            f"Unable to establish collector ClickHouse connection after {attempts} attempts"
        ) from last_error

    def _preflight_with_reconnect(self, client: Any) -> tuple[CollectorPreflightReport, Any]:
        attempts = int(self.context.get("options_chain.database_reconnect_attempts", 5))
        for attempt in range(1, attempts + 1):
            try:
                return collector_preflight(self.context, client), client
            except Exception as exc:
                classification = _log_database_error(self.logger, "preflight", exc)
                try:
                    client.close()
                except Exception:
                    self.logger.debug("Unable to close failed preflight client", exc_info=True)
                if not classification.retryable or attempt == attempts:
                    raise
                self.logger.warning(
                    "Collector preflight failed transiently (attempt %s/%s); reconnecting",
                    attempt,
                    attempts,
                )
                if self.stop_capture.wait(min(2 ** (attempt - 1), 30)):
                    raise RuntimeError("Collector preflight interrupted during reconnect")
                client = self._open_main_client()
        raise AssertionError("unreachable preflight retry state")

    def run(self, *, once: bool = False, reconcile_only: bool = False) -> None:
        run_started = time.monotonic()
        self.validate(require_refdata=not reconcile_only)
        self._continuous_capture = not once and not reconcile_only
        capture_only = _as_bool(
            self.context.get("options_chain.collector.capture_only", False)
        )
        if capture_only and reconcile_only:
            raise ValueError("capture-only mode cannot run reconciliation")
        main_client = self._open_main_client()
        preflight, main_client = self._preflight_with_reconnect(main_client)
        self.logger.info(
            "event=collector_preflight status=success database=%s server_version=%s "
            "schema_generation=%s active_dataset_count=%s sampled_snapshot_id=%s "
            "reconciliation_candidates=%s",
            preflight.database,
            preflight.server_version,
            preflight.schema_generation,
            preflight.active_dataset_count,
            preflight.sampled_snapshot_id,
            preflight.reconciliation_candidate_count,
        )
        if preflight.maintenance_lease_active:
            raise RuntimeError("Collector writes are blocked by an active migration lease")
        if capture_only:
            main_client.close()
            main_client = None
            self.logger.warning(
                "Collector capture-only migration mode is active; raw payloads will not be normalized"
            )
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(once,),
                name="OptionsCapture",
                daemon=False,
            )
            self.capture_thread.start()
            while self.capture_thread.is_alive() and not self.stop_capture.is_set():
                self._maybe_report_health()
                self.stop_capture.wait(0.5)
            if self.capture_thread.is_alive():
                self.stop_capture.set()
                self.capture_thread.join(timeout=30)
            self._report_health(final=True)
            return
        reconciliation_interval = float(
            self.context.get("options_chain.reconcile_interval_seconds", 300)
        )
        last_reconciliation = 0.0
        try:
            startup_attempts = int(
                self.context.get("options_chain.database_reconnect_attempts", 5)
            )
            for attempt in range(1, startup_attempts + 1):
                try:
                    enqueued = self._reconcile(main_client, prepare_replays=True)
                    break
                except Exception as exc:
                    classification = _log_database_error(
                        self.logger, "startup_reconciliation", exc
                    )
                    try:
                        main_client.close()
                    except Exception:
                        self.logger.debug(
                            "Unable to close failed startup reconciliation client",
                            exc_info=True,
                        )
                    if not classification.retryable or attempt == startup_attempts:
                        raise
                    self.logger.error(
                        "Startup reconciliation failed (attempt %s/%s); reconnecting\n%s",
                        attempt,
                        startup_attempts,
                        traceback.format_exc(),
                    )
                    if self.stop_capture.wait(min(2 ** (attempt - 1), 30)):
                        return
                    main_client = self._open_main_client()
            self.logger.info("Reconciled %s raw-to-normalized snapshot references", enqueued)
            self.ingestion_thread = threading.Thread(
                target=self._ingestion_loop, name="OptionsIngestion", daemon=False
            )
            self.ingestion_thread.start()
            if reconcile_only:
                self.raw_queue.put(_SENTINEL)
                self.ingestion_thread.join()
                self._report_reconciliation_totals(run_started)
                if (
                    self._reconciliation_totals["failed"]
                    or self._reconciliation_totals["quarantined_ambiguous"]
                ):
                    raise RuntimeError(
                        "Reconciliation completed with failed or ambiguous snapshots"
                    )
                return
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(once,),
                name="OptionsCapture",
                daemon=False,
            )
            self.capture_thread.start()
            while not self.stop_capture.is_set():
                if not self.capture_thread.is_alive():
                    if once:
                        break
                    self.logger.critical("Capture coordinator stopped unexpectedly")
                    break
                now = time.monotonic()
                if now - last_reconciliation >= reconciliation_interval:
                    try:
                        if main_client is None:
                            main_client = self._open_main_client()
                        self._reconcile(main_client)
                    except Exception as exc:
                        self.health.increment("normalization_database_failures")
                        self.health.record_consecutive_failure(
                            "database:reconciliation"
                        )
                        self.health.update(last_error=str(exc))
                        classification = _log_database_error(
                            self.logger, "periodic_reconciliation", exc
                        )
                        self.logger.error(
                            "Raw-to-normalized reconciliation failed\n%s",
                            traceback.format_exc(),
                        )
                        if not classification.retryable:
                            raise
                        if main_client is not None:
                            try:
                                main_client.close()
                            except Exception:
                                self.logger.debug(
                                    "Unable to close failed reconciliation client",
                                    exc_info=True,
                                )
                        main_client = None
                    last_reconciliation = now
                self._maybe_report_health()
                self.stop_capture.wait(0.5)
            if self.capture_thread.is_alive():
                self.stop_capture.set()
                self.capture_thread.join(timeout=30)
            if once:
                try:
                    if main_client is None:
                        main_client = self._open_main_client()
                    self._reconcile(main_client)
                except Exception as exc:
                    classification = _log_database_error(
                        self.logger, "final_reconciliation", exc
                    )
                    self.logger.error(
                        "Final reconciliation failed\n%s", traceback.format_exc()
                    )
                    if not classification.retryable:
                        raise
            self.raw_queue.put(_SENTINEL)
            self.ingestion_thread.join(timeout=60)
        finally:
            self.stop_capture.set()
            try:
                if main_client is not None:
                    main_client.close()
            except Exception:
                self.logger.debug("Unable to close main client", exc_info=True)
            self._report_health(final=True)

    def request_stop(self, signum: Optional[int] = None, frame: Any = None) -> None:
        self.logger.info(
            "Shutdown requested%s", " by signal " + str(signum) if signum else ""
        )
        self.stop_capture.set()


def default_context(algo_home: str) -> dict[str, Any]:
    return {
        PARAM_ALGO_HOME: algo_home,
        PARAM_LOG_PATH: os.path.join(algo_home, "logs"),
        PARAM_DEFAULT_LOG_LEVEL: logging.INFO,
        PARAM_REFDATA_PROVIDER_CLASS: (
            "algotrading.tda.TDAReferenceData.TDAReferenceData"
        ),
        "options_chain.symbols": "MSFT",
        "options_chain.query_interval": 30,
        "options_chain.capture_workers": 4,
        "options_chain.market_hours_only": True,
        "options_chain.max_tenor_age": 365,
        "options_chain.num_strikes_each_side": 20,
        "options_chain.api_strike_count": 20,
        "options_chain.api_filter_expirations": True,
        "options_chain.include_nonstandard": False,
        "options_chain.include_minis": False,
        "options_chain.expiration_grace_seconds": 0,
        "options_chain.clickhouse.host": "localhost",
        "options_chain.clickhouse.port": 8123,
        "options_chain.clickhouse.username": "default",
        "options_chain.clickhouse.password": "",
        "options_chain.clickhouse.database": "algotrading",
        "options_chain.clickhouse.secure": False,
        "options_chain.clickhouse.connect_timeout": 10,
        "options_chain.clickhouse.send_receive_timeout": 120,
        "options_chain.clickhouse.create_schema": True,
        "options_chain.reconcile_days": 7,
        "options_chain.reconcile_interval_seconds": 300,
        "options_chain.raw_capture_stale_seconds": 0,
        "options_chain.collector.capture_only": False,
        "options_chain.allow_purged_range_backfill": False,
        "options_chain.confirm_backfill_database": None,
        "options_chain.confirm_backfill_range_start": None,
        "options_chain.confirm_backfill_range_end": None,
        "options_chain.database_reconnect_attempts": 5,
        "options_chain.database_reconnect_max_seconds": 30,
        "options_chain.monitor_interval": 300,
        "options_chain.nagios_consecutive_error_threshold": 10,
    }


def validate_fixture(path: str) -> dict[str, Any]:
    """Internal parser/filter regression check for the supplied trusted pickle."""
    import pickle

    with open(path, "rb") as handle:
        response = pickle.load(handle)
    quote_times = [
        contract.get("quoteTimeInLong")
        for _, _, _, contract, _, _ in _iter_contracts(response)
        if contract.get("quoteTimeInLong") is not None
    ]
    collected = datetime.fromtimestamp(max(quote_times) / 1000.0, UTC)
    source_count = _count_contracts(response)
    tenor_count = 0
    for _, expiration_key, _, contract, _, _ in _iter_contracts(response):
        expiration = _parse_timestamp(contract.get("expirationDate"))
        if expiration is None:
            expiration = _parse_timestamp(
                str(expiration_key).split(":", 1)[0] + "T00:00:00+00:00"
            )
        if (
            expiration is not None
            and 0 < (expiration - collected).total_seconds() <= 365 * 86400
        ):
            tenor_count += 1
    rows = flatten_and_filter_response(
        response,
        snapshot_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        collected_at_utc=collected,
        max_tenor_age=365,
        num_strikes_each_side=20,
    )
    result = {
        "source_contract_count": source_count,
        "dte_contract_count": tenor_count,
        "stored_contract_count": len(rows),
        "collected_at_utc": collected.isoformat(),
        "underlying_is_none": response.get("underlying") is None,
        "chain_interest_rate": rows[0]["chain_interest_rate"] if rows else None,
        "chain_dividend_yield": rows[0]["chain_dividend_yield"] if rows else None,
        "first_source_iv": rows[0]["source_iv"] if rows else None,
        "deliverables_json_valid": bool(rows)
        and isinstance(json.loads(rows[0]["option_deliverables_json"]), list),
        "utc_timestamps": bool(rows)
        and rows[0]["expiration_at"].tzinfo == UTC
        and rows[0]["quote_time"].tzinfo == UTC,
    }
    expected = (3754, 2866, 1500)
    actual = (source_count, tenor_count, len(rows))
    if actual != expected:
        raise AssertionError(
            f"MSFT fixture regression failed: expected {expected}, got {actual}"
        )
    if (
        not result["underlying_is_none"]
        or not result["deliverables_json_valid"]
        or not result["utc_timestamps"]
    ):
        raise AssertionError(f"MSFT fixture shape regression failed: {result}")
    if not math.isclose(result["chain_interest_rate"], 0.03695) or not math.isclose(
        result["chain_dividend_yield"], 0.00945
    ):
        raise AssertionError(f"MSFT percentage normalization failed: {result}")
    return result


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Schwab option chains into normalized ClickHouse tables"
    )
    parser.add_argument("-e", required=True, help="configuration environment to use")
    parser.add_argument(
        "--once", action="store_true", help="poll each selected symbol once and exit"
    )
    parser.add_argument(
        "--create-schema-only",
        action="store_true",
        help="create collector-owned ClickHouse schema and exit",
    )
    parser.add_argument(
        "--reconcile-only",
        action="store_true",
        help="process raw snapshots missing normalized manifests and exit",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the generation-3 schema and Collector reads without writes",
    )
    parser.add_argument(
        "--integrity-audit",
        action="store_true",
        help="run a bounded read-only normalized integrity audit and exit",
    )
    parser.add_argument(
        "--audit-days",
        type=int,
        help="bounded lookback for --integrity-audit (defaults to reconcile_days)",
    )
    parser.add_argument(
        "--audit-max-snapshots",
        type=int,
        default=10,
        help="maximum snapshots for --integrity-audit (default: 10)",
    )
    parser.add_argument("--symbols", help="comma-separated symbol override")
    parser.add_argument(
        "--database",
        help="validated ClickHouse database override (primarily for isolated validation)",
    )
    parser.add_argument(
        "--capture-workers", type=int, help="option snapshot REST worker override"
    )
    parser.add_argument("--validate-fixture", metavar="PICKLE", help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if sum(
        (
            options.once,
            options.create_schema_only,
            options.reconcile_only,
            options.preflight_only,
            options.integrity_audit,
        )
    ) > 1:
        parser.error(
            "--once, --create-schema-only, --reconcile-only, --preflight-only, and "
            "--integrity-audit are mutually exclusive"
        )
    if options.audit_days is not None and options.audit_days <= 0:
        parser.error("--audit-days must be positive")
    if options.audit_max_snapshots <= 0:
        parser.error("--audit-max-snapshots must be positive")
    return options


def _preflight_payload(report: CollectorPreflightReport) -> dict[str, Any]:
    return {
        "database": report.database,
        "server_version": report.server_version,
        "schema_generation": report.schema_generation,
        "active_dataset_count": report.active_dataset_count,
        "validated_tables": report.schema.table_count,
        "validated_columns": report.schema.column_count,
        "sampled_snapshot_id": (
            str(report.sampled_snapshot_id) if report.sampled_snapshot_id else None
        ),
        "sampled_snapshot_state": (
            report.sampled_snapshot_state.__dict__
            if report.sampled_snapshot_state is not None
            else None
        ),
        "reconciliation_candidate_count": report.reconciliation_candidate_count,
        "maintenance_lease_active": report.maintenance_lease_active,
    }


def main(argv: Optional[list[str]] = None) -> int:
    options = _parse_args(argv)
    algo_home = os.getenv("ALGO_HOME")
    if not algo_home:
        print("ALGO_HOME environment variable not set", file=sys.stderr)
        return 2
    algo_home = algo_home.rstrip("/") + "/"
    # Collector configuration contains API and database credentials.  The
    # shared loader's verbose mode is not safe for this process.
    context = load_configuration(
        default_context(algo_home), algo_home, options.e, quiet=True
    )

    if 'nagios.service_id' in context:
        context['nagios.service_id'] = context['nagios.service_id'] + " Collector"

    if options.symbols:
        context["options_chain.symbols"] = options.symbols
    if options.database:
        context["options_chain.clickhouse.database"] = validate_database_identifier(
            options.database
        )
    if options.capture_workers is not None:
        context["options_chain.capture_workers"] = options.capture_workers
    logger = configure_logging(context)
    context[PARAM_LOGGER] = logger
    try:
        if options.validate_fixture:
            print(
                json.dumps(
                    validate_fixture(options.validate_fixture), indent=2, sort_keys=True
                )
            )
            return 0
        if options.create_schema_only:
            create_collector_schema(context)
            client = create_clickhouse_client(context)
            try:
                report = collector_preflight(context, client)
            finally:
                client.close()
            logger.info(
                "event=collector_schema_create status=success database=%s "
                "schema_generation=%s",
                report.database,
                report.schema_generation,
            )
            print(json.dumps(_preflight_payload(report), indent=2, sort_keys=True))
            return 0
        if options.preflight_only or options.integrity_audit:
            client = create_clickhouse_client(context)
            try:
                report = collector_preflight(context, client)
                if options.preflight_only:
                    print(
                        json.dumps(_preflight_payload(report), indent=2, sort_keys=True)
                    )
                    return 0
                audit = run_integrity_audit(
                    context,
                    client,
                    days=options.audit_days
                    or int(context.get("options_chain.reconcile_days", 7)),
                    maximum_snapshots=options.audit_max_snapshots,
                )
                payload = {
                    "preflight": _preflight_payload(report),
                    "integrity_audit": {
                        "database": audit.database,
                        "bounded_days": audit.bounded_days,
                        "maximum_snapshots": audit.maximum_snapshots,
                        "audited_snapshots": audit.audited_snapshots,
                        "finding_total": audit.finding_total,
                        "finding_counts": dict(audit.finding_counts),
                        "sample_snapshot_ids": dict(audit.sample_snapshot_ids),
                    },
                }
                print(json.dumps(payload, indent=2, sort_keys=True))
                return 1 if audit.finding_total else 0
            finally:
                client.close()
        if _as_bool(context.get("options_chain.clickhouse.create_schema", True)):
            create_collector_schema(context)
        refdata = None
        if not options.reconcile_only:
            logger.info("Loading %s", context[PARAM_REFDATA_PROVIDER_CLASS])
            refdata = import_module_attr(context[PARAM_REFDATA_PROVIDER_CLASS])(
                context
            )
            context[PARAM_REFDATA_INTERFACE] = refdata
        monitor = NagiosSystemMonitor(context)
        collector = OptionsChainCollector(context, refdata=refdata, monitor=monitor)
        signal.signal(signal.SIGINT, collector.request_stop)
        signal.signal(signal.SIGTERM, collector.request_stop)
        collector.run(once=options.once, reconcile_only=options.reconcile_only)
        return 0
    except Exception:
        logger.critical("OptionsChainCollector terminated\n%s", traceback.format_exc())
        return 1
    finally:
        logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
