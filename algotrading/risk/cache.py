"""Bounded restart-persistent latest-result cache and remote-attempt ledger."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Mapping


class PersistentRiskCache:
    def __init__(self, path: str | Path, *, max_entries: int = 5000) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._connection.row_factory = sqlite3.Row
        self._max_entries = max_entries
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    source_as_of TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    soft_expires_at TEXT NOT NULL,
                    hard_expires_at TEXT NOT NULL,
                    last_access_at TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS remote_attempt_ledger (
                    resource_key TEXT PRIMARY KEY,
                    last_attempt_at TEXT NOT NULL,
                    last_success_at TEXT NULL,
                    last_error TEXT NULL
                )
                """
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def get(self, cache_key: str, *, schema_version: int = 1) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM risk_cache_entries WHERE cache_key = ? AND schema_version = ?",
                (cache_key, schema_version),
            ).fetchone()
            if row is None:
                return None
            payload_text = str(row["payload"])
            # Discard damaged entries before decoding or returning cached results.
            if hashlib.sha256(payload_text.encode("utf-8")).hexdigest() != row["checksum"]:
                self._connection.execute("DELETE FROM risk_cache_entries WHERE cache_key = ?", (cache_key,))
                return None
            self._connection.execute(
                "UPDATE risk_cache_entries SET last_access_at = ? WHERE cache_key = ?",
                (now, cache_key),
            )
        payload = json.loads(payload_text)
        return {
            "payload": payload,
            "source_as_of": row["source_as_of"],
            "fetched_at": row["fetched_at"],
            "soft_expires_at": row["soft_expires_at"],
            "hard_expires_at": row["hard_expires_at"],
        }

    def put(
        self,
        cache_key: str,
        *,
        namespace: str,
        request_fingerprint: str,
        payload: Mapping[str, Any],
        source_as_of: datetime,
        fetched_at: datetime,
        soft_expires_at: datetime,
        hard_expires_at: datetime,
        schema_version: int = 1,
    ) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
        checksum = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO risk_cache_entries (
                    cache_key, namespace, schema_version, request_fingerprint,
                    payload, checksum, source_as_of, fetched_at, soft_expires_at,
                    hard_expires_at, last_access_at, size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    namespace=excluded.namespace, schema_version=excluded.schema_version,
                    request_fingerprint=excluded.request_fingerprint,
                    payload=excluded.payload, checksum=excluded.checksum,
                    source_as_of=excluded.source_as_of, fetched_at=excluded.fetched_at,
                    soft_expires_at=excluded.soft_expires_at,
                    hard_expires_at=excluded.hard_expires_at,
                    last_access_at=excluded.last_access_at, size_bytes=excluded.size_bytes
                """,
                (
                    cache_key,
                    namespace,
                    schema_version,
                    request_fingerprint,
                    encoded,
                    checksum,
                    source_as_of.astimezone(timezone.utc).isoformat(),
                    fetched_at.astimezone(timezone.utc).isoformat(),
                    soft_expires_at.astimezone(timezone.utc).isoformat(),
                    hard_expires_at.astimezone(timezone.utc).isoformat(),
                    now,
                    len(encoded.encode("utf-8")),
                ),
            )
            # Evict the least recently accessed entries to bound persistent storage.
            self._connection.execute(
                "DELETE FROM risk_cache_entries WHERE cache_key IN ("
                "SELECT cache_key FROM risk_cache_entries ORDER BY last_access_at DESC LIMIT -1 OFFSET ?)",
                (self._max_entries,),
            )

    def attempt_state(self, resource_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM remote_attempt_ledger WHERE resource_key = ?",
                (resource_key,),
            ).fetchone()
        return dict(row) if row is not None else None

    def record_attempt(
        self,
        resource_key: str,
        *,
        at: datetime,
        success: bool,
        error: str | None = None,
    ) -> None:
        safe_error = str(error).replace("\n", " ")[:256] if error else None
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO remote_attempt_ledger (
                    resource_key, last_attempt_at, last_success_at, last_error
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(resource_key) DO UPDATE SET
                    last_attempt_at=excluded.last_attempt_at,
                    last_success_at=CASE WHEN excluded.last_success_at IS NOT NULL
                        THEN excluded.last_success_at ELSE remote_attempt_ledger.last_success_at END,
                    last_error=excluded.last_error
                """,
                (
                    resource_key,
                    at.astimezone(timezone.utc).isoformat(),
                    at.astimezone(timezone.utc).isoformat() if success else None,
                    safe_error,
                ),
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS entries, COALESCE(SUM(size_bytes), 0) AS size_bytes "
                "FROM risk_cache_entries"
            ).fetchone()
            attempts = self._connection.execute(
                "SELECT COUNT(*) AS count FROM remote_attempt_ledger"
            ).fetchone()
        return {
            "path": str(self.path),
            "entries": int(row["entries"]),
            "size_bytes": int(row["size_bytes"]),
            "attempt_ledger_entries": int(attempts["count"]),
            "persistence": "sqlite_wal",
        }


def default_risk_cache_path(context: Mapping[str, Any]) -> Path:
    configured = str(context.get("risk.cache.path", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "algotrading" / "risk-api-cache.sqlite3"


__all__ = ["PersistentRiskCache", "default_risk_cache_path"]
