"""Shared ClickHouse contracts for the option-chain applications.

This module deliberately contains no application loop, rate/solver integration,
logging policy, or base-row parser.  It is safe for the collector and the
calculation worker to import when they are deployed on different machines.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, Iterable, Mapping, Optional

import pyarrow as pa


UTC = timezone.utc
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Configuration namespaces shared by contract calculations and surface code.
# Database object names intentionally remain ``options_chain_*``.
OPTIONS_CALCS_CONFIG_PREFIX = "options_calcs."
OPTIONS_CALCS_GREEKS_PREFIX = "options_calcs.greeks."
OPTIONS_CALCS_SURFACE_PREFIX = "options_calcs.surface."
OPTIONS_CALCS_FRED_PREFIX = "options_calcs.fred."


_OPERATIONAL_OPTIONS_CALCS_EXACT = frozenset(
    {
        "create_schema", "code_version", "definition.activated_at",
        "surface.definition.activated_at", "poll_interval_seconds",
        "pending_batch_size",
        "contract.pending_batch_size", "surface.pending_batch_size",
        "reconcile_days", "retry_failed", "running_stale_seconds",
        "max_calculation_lag_seconds", "worker_id", "workers", "max_inflight",
        "surface.max_inflight", "monitor_interval",
        "database_reconnect_attempts", "database_reconnect_max_seconds",
        "backfill_stall_seconds", "nagios_consecutive_error_threshold",
        "nagios_quality_alerts_enabled", "fred.api_key", "fred.history_days",
        "fred.request_timeout", "fred.refresh_retry_seconds",
        "worker.max_tasks_per_child",
        "worker.shutdown_timeout_seconds", "worker.cpu_threads_per_process",
        "surface.enabled", "surface.max_surface_lag_seconds",
        "surface.cadence_multiplier", "surface.max_minutes_for_calculation",
        "surface.svi_warm_start_cache_size",
        "surface.max_failed_expiration_fraction", "surface.max_node_failure_fraction",
        "surface.max_required_node_numerical_failure_fraction",
        "surface.max_required_node_unavailable_fraction",
        "surface.max_required_node_outside_coverage_fraction",
        "surface.warn_on_optional_node_outside_coverage",
        "surface.max_symbol_required_node_unavailable_fraction",
        "surface.warn_on_zero_nodes_for_successful_run",
        "surface.iv_batch.enabled", "surface.delta_batch.enabled",
        "greeks.max_unrecoverable_failure_fraction",
        "greeks.max_recoverable_failure_fraction",
        "greeks.max_batch_exceptions",
        "greeks.consecutive_failed_snapshot_threshold",
        "greeks.nagios_window_seconds",
        "greeks.nagios_min_failure_count",
    }
)


def is_operational_options_calcs_key(key: str) -> bool:
    """Return whether *key* is excluded from numerical definition identity."""
    if not isinstance(key, str) or not key.startswith(OPTIONS_CALCS_CONFIG_PREFIX):
        return False
    suffix = key[len(OPTIONS_CALCS_CONFIG_PREFIX) :]
    return (
        suffix in _OPERATIONAL_OPTIONS_CALCS_EXACT
        or suffix.startswith("worker.")
        or suffix.startswith("contract.micro_batch.")
        or suffix.startswith("surface.micro_batch.")
        or suffix.startswith("logging.")
        or suffix.startswith("nagios.")
        or suffix.startswith("cli.")
        or suffix.startswith("maintenance.")
        or suffix.startswith("greeks.batch.")
        or suffix.startswith("greeks.nagios.")
        or suffix.startswith("greeks.circuit_breaker.")
        or (
            suffix.startswith("greeks.")
            and (suffix.endswith(".device") or suffix.endswith(".chunk_size"))
        )
        or suffix.startswith("bulk_query.")
    )


def validate_options_calcs_configuration(context: Mapping[str, Any]) -> None:
    """Validate the canonical calculator namespace before processing work."""
    obsolete_replacements = {
        OPTIONS_CALCS_CONFIG_PREFIX + "pending_batch_size": (
            "options_calcs.contract.pending_batch_size and "
            "options_calcs.surface.pending_batch_size"
        ),
        OPTIONS_CALCS_GREEKS_PREFIX + "num_contracts": (
            OPTIONS_CALCS_GREEKS_PREFIX + "num_contracts_per_expiration"
        ),
        OPTIONS_CALCS_GREEKS_PREFIX + "include_european_options": (
            "European options are always source-only"
        ),
        OPTIONS_CALCS_GREEKS_PREFIX + "require_model_price_within_bid_ask": (
            OPTIONS_CALCS_GREEKS_PREFIX + "iv.price_tolerance"
        ),
        OPTIONS_CALCS_GREEKS_PREFIX + "device": OPTIONS_CALCS_GREEKS_PREFIX + "crr.device",
        OPTIONS_CALCS_GREEKS_PREFIX + "steps": OPTIONS_CALCS_GREEKS_PREFIX + "crr.steps",
        OPTIONS_CALCS_GREEKS_PREFIX + "chunk_size": OPTIONS_CALCS_GREEKS_PREFIX + "crr.chunk_size",
        OPTIONS_CALCS_SURFACE_PREFIX + "minimum_iv": OPTIONS_CALCS_GREEKS_PREFIX + "iv.minimum",
        OPTIONS_CALCS_SURFACE_PREFIX + "maximum_iv": OPTIONS_CALCS_GREEKS_PREFIX + "iv.maximum",
        OPTIONS_CALCS_SURFACE_PREFIX + "iv_price_tolerance": OPTIONS_CALCS_GREEKS_PREFIX + "iv.price_tolerance",
        OPTIONS_CALCS_SURFACE_PREFIX + "iv_max_iterations": OPTIONS_CALCS_GREEKS_PREFIX + "iv.max_iterations",
        OPTIONS_CALCS_SURFACE_PREFIX + "iv_source_bracket_multipliers": OPTIONS_CALCS_GREEKS_PREFIX + "iv.source_bracket_multipliers",
        OPTIONS_CALCS_SURFACE_PREFIX + "allow_source_iv_fallback": (
            "remove; American repricing never falls back to source IV"
        ),
    }
    obsolete_prefixes = (
        OPTIONS_CALCS_CONFIG_PREFIX + "contract.micro_batch.",
        OPTIONS_CALCS_CONFIG_PREFIX + "surface.micro_batch.",
    )
    obsolete_exact = {
        OPTIONS_CALCS_CONFIG_PREFIX + name
        for name in (
            "database_reconnect_attempts",
            "database_reconnect_max_seconds",
            "backfill_stall_seconds",
            "nagios_consecutive_error_threshold",
            "nagios_quality_alerts_enabled",
        )
    }
    obsolete = [
        str(key)
        for key in context
        if isinstance(key, str)
        and (
            key in obsolete_replacements
            or key in obsolete_exact
            or any(key.startswith(prefix) for prefix in obsolete_prefixes)
            or key.startswith(OPTIONS_CALCS_SURFACE_PREFIX + "crr_")
        )
    ]
    if obsolete:
        lines = ["Deprecated options calculation configuration keys detected:"]
        for key in sorted(obsolete):
            replacement = obsolete_replacements.get(
                key, "shared options_calcs.greeks.<method> solver settings"
            )
            lines.append(f"{key}\n    -> {replacement}")
        raise ValueError("\n".join(lines))

    def positive_integral(key: str, default: int) -> int:
        raw = context.get(key, default)
        if isinstance(raw, bool) or isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"{key} must be a positive integer")
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be a positive integer") from exc
        if value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        return value

    for key, default in (
        (OPTIONS_CALCS_GREEKS_PREFIX + "batch.maximum_jobs", 2048),
        (OPTIONS_CALCS_GREEKS_PREFIX + "batch.maximum_estimated_bytes", 536_870_912),
    ):
        positive_integral(key, default)

    hybrid_enabled = _as_bool(
        context.get(OPTIONS_CALCS_SURFACE_PREFIX + "node_hybrid_enabled", True)
    )
    search_method = str(
        context.get(OPTIONS_CALCS_SURFACE_PREFIX + "node_search_method", "tian")
    ).strip().lower()
    allowed_search_methods = {"tian"} if hybrid_enabled else {
        "tian", "legacy", "observed_then_analytic"
    }
    if search_method not in allowed_search_methods:
        expected = "tian" if hybrid_enabled else "tian or a legacy search method"
        raise ValueError(
            "options_calcs.surface.node_search_method must be "
            f"{expected}, got {search_method!r}"
        )
    validation_method = str(
        context.get(
            OPTIONS_CALCS_SURFACE_PREFIX + "node_validation_method",
            "configured",
        )
    ).strip().lower()
    if validation_method != "configured":
        from algotrading.refdata.options.solvers.Types import RepricingMethod

        parsed_validation_method = RepricingMethod.parse(validation_method)
        if parsed_validation_method is RepricingMethod.NONE:
            raise ValueError(
                "options_calcs.surface.node_validation_method cannot be none"
            )
    tian_grid_size = positive_integral(
        OPTIONS_CALCS_SURFACE_PREFIX + "node_tian_grid_size", 81
    )
    if tian_grid_size < 9:
        raise ValueError(
            "options_calcs.surface.node_tian_grid_size must be at least 9"
        )
    validation_budget = positive_integral(
        OPTIONS_CALCS_SURFACE_PREFIX + "node_validation_max_evaluations", 4
    )
    initial_candidates = positive_integral(
        OPTIONS_CALCS_SURFACE_PREFIX + "node_validation_initial_candidates", 1
    )
    if initial_candidates > validation_budget:
        raise ValueError(
            "options_calcs.surface.node_validation_initial_candidates cannot "
            "exceed options_calcs.surface.node_validation_max_evaluations"
        )

    def nonnegative_integral(key: str, default: int) -> int:
        raw = context.get(key, default)
        if isinstance(raw, bool) or isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"{key} must be a nonnegative integer")
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be a nonnegative integer") from exc
        if value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        return value

    positive_integral(OPTIONS_CALCS_CONFIG_PREFIX + "workers", 1)
    nonnegative_integral(OPTIONS_CALCS_CONFIG_PREFIX + "max_inflight", 0)
    nonnegative_integral(OPTIONS_CALCS_CONFIG_PREFIX + "surface.max_inflight", 0)
    positive_integral(OPTIONS_CALCS_CONFIG_PREFIX + "contract.pending_batch_size", 100)
    positive_integral(OPTIONS_CALCS_CONFIG_PREFIX + "surface.pending_batch_size", 16)
    positive_integral(OPTIONS_CALCS_SURFACE_PREFIX + "cadence_multiplier", 1)
    positive_integral(
        OPTIONS_CALCS_SURFACE_PREFIX + "max_minutes_for_calculation", 120
    )
    nonnegative_integral(OPTIONS_CALCS_CONFIG_PREFIX + "worker.max_tasks_per_child", 0)
    positive_integral(OPTIONS_CALCS_CONFIG_PREFIX + "worker.cpu_threads_per_process", 1)
    positive_integral(OPTIONS_CALCS_CONFIG_PREFIX + "worker.shutdown_timeout_seconds", 60)
    positive_integral(OPTIONS_CALCS_CONFIG_PREFIX + "running_stale_seconds", 900)
    nonnegative_integral(OPTIONS_CALCS_CONFIG_PREFIX + "reconcile_days", 7)
    nonnegative_integral(OPTIONS_CALCS_CONFIG_PREFIX + "monitor_interval", 300)
    for key, default in (
        (OPTIONS_CALCS_CONFIG_PREFIX + "poll_interval_seconds", 10.0),
        (OPTIONS_CALCS_CONFIG_PREFIX + "max_calculation_lag_seconds", 600.0),
    ):
        try:
            value = float(context.get(key, default))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be finite and nonnegative") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    # Keep numerical-guard parsing at the common preflight boundary as well as
    # at registry construction so invalid tolerances fail before any claims.
    from algotrading.refdata.options.solvers.Registry import NumericalBatchConfig

    NumericalBatchConfig.from_context(context)


class SnapshotDataError(RuntimeError):
    """A durable snapshot cannot be processed without operator/data repair."""


class InconsistentSnapshotError(RuntimeError):
    """ClickHouse contains a partial snapshot that must not be appended to."""


RAW_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field(
            "request_started_at_utc", pa.timestamp("us", tz="UTC"), nullable=False
        ),
        pa.field(
            "response_received_at_utc", pa.timestamp("us", tz="UTC"), nullable=False
        ),
        pa.field("collected_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("response_latency_ms", pa.uint32(), nullable=False),
        pa.field("request_parameters_json", pa.string(), nullable=False),
        # Canonical ClickHouse storage is String.  The raw read helper converts
        # it back to UTF-8 bytes before hashing, preserving the exact payload.
        pa.field("response_json", pa.string(), nullable=False),
        pa.field("payload_sha256", pa.binary(32), nullable=False),
        pa.field("source_contract_count", pa.uint32(), nullable=False),
        pa.field("is_delayed", pa.bool_(), nullable=False),
        pa.field("is_chain_truncated", pa.bool_(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
    ]
)
RAW_COLUMNS = list(RAW_SCHEMA.names)


CONTRACT_ID_DELIMITER = "\x1f"
CONTRACT_ID_HASH = "sha256-first-64-bits-big-endian-v1"


def option_contract_id(underlying_symbol: str, option_symbol: str) -> int:
    """Return the deterministic compact contract ID shared with migration."""
    underlying = str(underlying_symbol).strip().upper()
    option = str(option_symbol)
    if not underlying or not option:
        raise ValueError("Contract ID requires an underlying and exact option symbol")
    payload = f"{underlying}{CONTRACT_ID_DELIMITER}{option}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256_bytes(value: str | bytes) -> bytes:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).digest()


def solver_tenor_seconds(expiration_at: Any, valuation_at: Any) -> int:
    """Preserve the solver's positive-tenor truncation convention exactly."""
    expiration = _parse_timestamp(expiration_at)
    valuation = _parse_timestamp(valuation_at)
    if expiration is None or valuation is None:
        raise ValueError("Solver tenor requires valid expiration and valuation times")
    # Python int truncates toward zero; valid solver tenors are positive, so
    # this is equivalent to flooring sub-second expiration differences.
    seconds = int((expiration - valuation).total_seconds())
    if seconds < 0 or seconds > 2**32 - 1:
        raise ValueError(f"Solver tenor does not fit UInt32: {seconds}")
    return seconds


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def has_required_source_greeks(row: Mapping[str, Any]) -> bool:
    """Whether the collector-owned source tuple can serve consumers."""
    source = row.get("_base", row)
    return all(
        _finite_float(source.get(name)) is not None
        for name in (
            "source_delta",
            "source_gamma",
            "source_theta",
            "source_vega",
            "source_rho",
        )
    )


def _nonnegative_int(value: Any) -> Optional[int]:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, datetime):
            # Arrow normally supplies aware UTC datetimes.  This is the hot
            # path for collector/calculation reads, so avoid reparsing or
            # allocating a second datetime when it is already normalized.
            if value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value):
                return value
            result = value
        elif isinstance(value, (int, float)):
            result = datetime.fromtimestamp(float(value) / 1000.0, UTC)
        else:
            result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None:
            result = result.replace(tzinfo=UTC)
        return result.astimezone(UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _datetime64_microseconds(value: Any) -> int:
    """Return an absolute UTC epoch value for ClickHouse DateTime64 predicates.

    clickhouse-connect's HTTP parameter rendering can discard an aware
    ``datetime`` offset before ClickHouse interprets a ``DateTime64`` literal.
    Comparing integer epoch values keeps predicates independent of the client
    and server local time zones.
    """
    parsed = _parse_timestamp(value)
    if parsed is None:
        raise ValueError(f"Invalid timestamp: {value!r}")
    delta = parsed - datetime(1970, 1, 1, tzinfo=UTC)
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000
        + delta.microseconds
    )


def _rows_to_arrow(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    # PyArrow projects to the supplied schema itself: extra input keys are
    # ignored and absent nullable fields become null.  Avoiding the Python
    # projection eliminates a second full list of dictionaries for large
    # base/calculation batches.
    return pa.Table.from_pylist(rows, schema=schema)


def _iter_contracts(
    response: Mapping[str, Any],
) -> Iterable[tuple[int, Any, Any, dict[str, Any], str, bool]]:
    """Iterate Schwab contract maps while retaining map-side diagnostics."""
    ordinal = 0
    for map_name, expected_side in (
        ("callExpDateMap", "CALL"),
        ("putExpDateMap", "PUT"),
    ):
        expiration_map = response.get(map_name)
        if not isinstance(expiration_map, dict):
            continue
        for expiration_key, strike_map in expiration_map.items():
            if not isinstance(strike_map, dict):
                continue
            for strike_key, leaf in strike_map.items():
                contracts = leaf if isinstance(leaf, list) else [leaf]
                for contract in contracts:
                    if not isinstance(contract, dict):
                        continue
                    actual_side = str(contract.get("putCall", expected_side)).upper()
                    yield (
                        ordinal,
                        expiration_key,
                        strike_key,
                        contract,
                        expected_side,
                        actual_side != expected_side,
                    )
                    ordinal += 1


def validate_database_identifier(database: str) -> str:
    if not _IDENTIFIER.fullmatch(database):
        raise ValueError(f"Unsafe ClickHouse database identifier: {database!r}")
    return database


def _clickhouse_connect_module():
    try:
        import clickhouse_connect
    except ImportError as exc:
        raise RuntimeError("clickhouse-connect >= 1.2, < 2.0 is required") from exc
    version_text = version("clickhouse-connect")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version_text)
    version_parts = tuple(map(int, match.groups())) if match else (0, 0, 0)
    if version_parts < (1, 2, 0) or version_parts >= (2, 0, 0):
        raise RuntimeError(
            f"clickhouse-connect {version_text} is unsupported; require >= 1.2, < 2.0"
        )
    return clickhouse_connect


def create_clickhouse_client(
    context: Mapping[str, Any], *, database: Optional[str] = None
):
    clickhouse_connect = _clickhouse_connect_module()
    return clickhouse_connect.get_client(
        host=context["options_chain.clickhouse.host"],
        port=int(context["options_chain.clickhouse.port"]),
        username=context["options_chain.clickhouse.username"],
        password=context.get("options_chain.clickhouse.password", ""),
        database=database or context["options_chain.clickhouse.database"],
        secure=_as_bool(context.get("options_chain.clickhouse.secure", False)),
        connect_timeout=int(
            context.get("options_chain.clickhouse.connect_timeout", 10)
        ),
        send_receive_timeout=int(
            context.get("options_chain.clickhouse.send_receive_timeout", 120)
        ),
        # Archival streams raw payloads while periodically persisting its lease.
        # ClickHouse permits only one active query per HTTP session, so the
        # heartbeat INSERT must not share an auto-generated session with the
        # open streaming SELECT.  No options workflow relies on session state.
        autogenerate_session_id=False,
        # Canonical queries distinguish an absent LEFT JOIN row from valid
        # zero/empty values.  Make that semantic explicit for every runtime,
        # archival, migration, and operator client created by this package.
        settings={"join_use_nulls": 1},
    )


def query_scalar(
    client: Any, sql: str, parameters: Optional[dict[str, Any]] = None
) -> int:
    result = client.query(sql, parameters=parameters or {})
    return int(result.result_rows[0][0]) if result.result_rows else 0


def read_raw_snapshot(
    client: Any, database: str, snapshot_id: uuid.UUID
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read, hash-check, and decode exactly one ClickHouse raw snapshot."""
    validate_database_identifier(database)
    select_columns = [
        "toString(snapshot_id) AS snapshot_id_text",
        *RAW_COLUMNS[1:],
    ]
    result = client.query(
        f"SELECT {', '.join(select_columns)} FROM {database}.options_chain_raw "
        "WHERE snapshot_id = {snapshot_id:UUID}",
        parameters={"snapshot_id": str(snapshot_id)},
    )
    rows = list(result.result_rows)
    if len(rows) != 1:
        raise SnapshotDataError(
            f"Expected exactly one raw row for {snapshot_id}, found {len(rows)}"
        )
    column_names = list(getattr(result, "column_names", RAW_COLUMNS))
    record = dict(zip(column_names, rows[0]))
    if "snapshot_id_text" in record:
        record["snapshot_id"] = record.pop("snapshot_id_text")
    stored_id = str(record.get("snapshot_id"))
    if stored_id != str(snapshot_id):
        raise SnapshotDataError(
            f"Raw snapshot identity mismatch: expected {snapshot_id}, found {stored_id}"
        )
    payload = record.get("response_json")
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    elif isinstance(payload, (bytearray, memoryview)):
        payload = bytes(payload)
    if not isinstance(payload, bytes):
        raise SnapshotDataError(
            f"Raw snapshot {snapshot_id} has a non-byte JSON payload"
        )
    digest = hashlib.sha256(payload).digest()
    stored_digest_value = record.get("payload_sha256") or b""
    if isinstance(stored_digest_value, str):
        try:
            stored_digest = bytes.fromhex(stored_digest_value)
        except ValueError as exc:
            raise SnapshotDataError(
                f"Raw snapshot {snapshot_id} has an invalid stored digest"
            ) from exc
    else:
        stored_digest = bytes(stored_digest_value)
    if digest != stored_digest:
        raise SnapshotDataError(
            f"Raw snapshot {snapshot_id} hash mismatch: "
            f"stored={stored_digest.hex()}, actual={digest.hex()}"
        )
    try:
        response = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotDataError(
            f"Raw snapshot {snapshot_id} contains invalid UTF-8 JSON"
        ) from exc
    if not isinstance(response, dict):
        raise SnapshotDataError(f"Raw snapshot {snapshot_id} JSON is not an object")
    record["response_json"] = payload
    record["payload_sha256"] = stored_digest
    return record, response
