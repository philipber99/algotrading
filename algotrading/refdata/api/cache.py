"""Small, bounded, thread-safe TTL cache with duplicate-load suppression."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from threading import Condition, RLock
from time import monotonic
from typing import Any, Generic, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CacheLookup(Generic[T]):
    value: T
    hit: bool


@dataclass(frozen=True)
class OptionBarsCacheLookup:
    rows: list[dict[str, Any]]
    missing_ranges: list[tuple[date, date]]
    hit: bool


@dataclass
class _Entry(Generic[T]):
    value: T
    expires_at: float


class SingleFlightTTLCache(Generic[T]):
    """
    A process-local cache intended for API response payloads.

    Only one thread loads a missing key. Other threads requesting the same key
    wait for that load and then use the cached value.
    """

    def __init__(
        self,
        max_entries: int = 1024,
        enabled: bool = True,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")

        self._max_entries = max_entries
        self._enabled = enabled
        self._clock = clock
        self._entries: OrderedDict[Hashable, _Entry[T]] = OrderedDict()
        self._loading: set[Hashable] = set()
        self._condition = Condition(RLock())
        self._hits = 0
        self._misses = 0

    def get_or_load(
        self,
        key: Hashable,
        ttl_seconds: float,
        loader: Callable[[], T],
    ) -> CacheLookup[T]:
        if not self._enabled or ttl_seconds <= 0:
            return CacheLookup(loader(), False)

        with self._condition:
            while True:
                entry = self._get_unlocked(key)
                if entry is not None:
                    self._hits += 1
                    return CacheLookup(entry.value, True)

                # Elect one loader per key while other requests wait on the condition.
                if key not in self._loading:
                    self._loading.add(key)
                    self._misses += 1
                    break

                self._condition.wait()

        # Run the slow loader outside the cache lock so unrelated keys can proceed.
        try:
            value = loader()
        # Wake waiting requests even when the active load fails.
        except BaseException:
            with self._condition:
                self._loading.discard(key)
                self._condition.notify_all()
            raise

        with self._condition:
            self._entries[key] = _Entry(
                value=value,
                expires_at=self._clock() + ttl_seconds,
            )
            self._entries.move_to_end(key)
            self._evict_unlocked()
            self._loading.discard(key)
            self._condition.notify_all()

        return CacheLookup(value, False)

    def clear(self) -> None:
        with self._condition:
            self._entries.clear()

    def delete_where(self, predicate: Callable[[Hashable], bool]) -> int:
        with self._condition:
            keys = [key for key in self._entries if predicate(key)]
            for key in keys:
                del self._entries[key]
            return len(keys)

    def stats(self) -> dict[str, int | bool]:
        with self._condition:
            self._purge_expired_unlocked()
            return {
                "enabled": self._enabled,
                "entries": len(self._entries),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "loads_in_progress": len(self._loading),
            }

    def _get_unlocked(self, key: Hashable) -> _Entry[T] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None

        if entry.expires_at <= self._clock():
            del self._entries[key]
            return None

        self._entries.move_to_end(key)
        return entry

    def _purge_expired_unlocked(self) -> None:
        now = self._clock()
        expired = [
            key for key, entry in self._entries.items() if entry.expires_at <= now
        ]
        for key in expired:
            del self._entries[key]

    def _evict_unlocked(self) -> None:
        self._purge_expired_unlocked()
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


class PersistentDailyBarsCache:
    """Small date-scoped SQLite cache for bulk daily-bar reads.

    Coverage is tracked only when at least one SQL row was cached for a symbol.
    Empty results are deliberately not negative-cached because the underlying
    price table can be repaired while the API is running. All rows from an
    earlier local calendar date are removed before each access; this keeps the
    cache useful across API restarts without allowing yesterday's snapshot to
    survive into a new day.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=10,
        )
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_bars (
                    cache_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (cache_date, symbol, timestamp)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_bar_coverage (
                    cache_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    range_start TEXT NOT NULL,
                    range_end TEXT NOT NULL,
                    PRIMARY KEY (cache_date, symbol, range_start, range_end)
                )
                """
            )

    def load(
        self,
        symbols: Sequence[str],
        *,
        start: date,
        end: date,
        cache_date: date,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        normalized = tuple(dict.fromkeys(str(item).upper() for item in symbols))
        if not normalized:
            return [], set()
        day = cache_date.isoformat()
        placeholders = ",".join("?" for _ in normalized)
        with self._lock, self._connection:
            self._invalidate_other_days(day)
            covered_rows = self._connection.execute(
                f"""
                SELECT DISTINCT symbol
                FROM daily_bar_coverage
                WHERE cache_date = ?
                  AND symbol IN ({placeholders})
                  AND range_start <= ? AND range_end >= ?
                """,
                (day, *normalized, start.isoformat(), end.isoformat()),
            ).fetchall()
            covered = {str(row["symbol"]) for row in covered_rows}
            if not covered:
                return [], set()
            covered_placeholders = ",".join("?" for _ in covered)
            rows = self._connection.execute(
                f"""
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM daily_bars
                WHERE cache_date = ?
                  AND symbol IN ({covered_placeholders})
                  AND timestamp >= ? AND timestamp < ?
                ORDER BY symbol, timestamp
                """,
                (
                    day,
                    *sorted(covered),
                    start.isoformat(),
                    date.fromordinal(end.toordinal() + 1).isoformat(),
                ),
            ).fetchall()
        materialized = [dict(row) for row in rows]
        # Older cache versions recorded coverage even when SQL returned no
        # rows. Treat those stale negative entries as misses so repaired or
        # newly backfilled database data becomes visible immediately.
        symbols_with_rows = {
            str(row.get("symbol") or "").upper() for row in materialized
        }
        return materialized, covered.intersection(symbols_with_rows)

    def store(
        self,
        symbols: Sequence[str],
        rows: Sequence[dict[str, Any]],
        *,
        start: date,
        end: date,
        cache_date: date,
    ) -> None:
        normalized = tuple(dict.fromkeys(str(item).upper() for item in symbols))
        if not normalized:
            return
        day = cache_date.isoformat()
        values = []
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            timestamp = row.get("timestamp")
            if symbol not in normalized or timestamp is None:
                continue
            values.append(
                (
                    day,
                    symbol,
                    str(timestamp),
                    self._number(row.get("open")),
                    self._number(row.get("high")),
                    self._number(row.get("low")),
                    self._number(row.get("close")),
                    self._number(row.get("volume")),
                )
            )
        symbols_with_rows = {value[1] for value in values}
        coverage = [
            (day, symbol, start.isoformat(), end.isoformat())
            for symbol in normalized
            if symbol in symbols_with_rows
        ]
        with self._lock, self._connection:
            self._invalidate_other_days(day)
            self._connection.executemany(
                """
                INSERT OR REPLACE INTO daily_bars
                    (cache_date, symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            self._connection.executemany(
                """
                INSERT OR REPLACE INTO daily_bar_coverage
                    (cache_date, symbol, range_start, range_end)
                VALUES (?, ?, ?, ?)
                """,
                coverage,
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _invalidate_other_days(self, cache_date: str) -> None:
        self._connection.execute(
            "DELETE FROM daily_bars WHERE cache_date <> ?", (cache_date,)
        )
        self._connection.execute(
            "DELETE FROM daily_bar_coverage WHERE cache_date <> ?", (cache_date,)
        )

    @staticmethod
    def _number(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class PersistentOptionBarsCache:
    """Restart-persistent daily OHLC cache with per-contract sliding TTL.

    Coverage is stored separately from rows so weekends, holidays, and valid
    empty provider responses do not cause the same interval to be requested
    repeatedly. Coverage intervals are inclusive dates; the REST service owns
    conversion from its exclusive ``end_at`` contract.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=10,
        )
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS option_bar_contracts (
                    symbol TEXT PRIMARY KEY,
                    last_access_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS option_daily_bars (
                    symbol TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, session_date)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS option_bar_coverage (
                    symbol TEXT NOT NULL,
                    range_start TEXT NOT NULL,
                    range_end TEXT NOT NULL,
                    PRIMARY KEY (symbol, range_start, range_end)
                )
                """
            )

    def load(
        self,
        symbol: str,
        *,
        start: date,
        end: date,
        now: datetime,
        ttl: timedelta,
    ) -> OptionBarsCacheLookup:
        if end < start:
            raise ValueError("option cache end must be on or after start")
        normalized = str(symbol).strip().upper()
        now_utc = self._utc(now)
        with self._lock, self._connection:
            self._purge_expired(now_utc)
            contract = self._connection.execute(
                "SELECT expires_at FROM option_bar_contracts WHERE symbol = ?",
                (normalized,),
            ).fetchone()
            active = contract is not None
            coverage_rows = self._connection.execute(
                """
                SELECT range_start, range_end
                FROM option_bar_coverage
                WHERE symbol = ? AND range_end >= ? AND range_start <= ?
                ORDER BY range_start, range_end
                """,
                (normalized, start.isoformat(), end.isoformat()),
            ).fetchall()
            # Merge stored coverage intervals before identifying the requested gaps.
            covered = self._merge_ranges(
                [
                    (date.fromisoformat(row["range_start"]), date.fromisoformat(row["range_end"]))
                    for row in coverage_rows
                ]
            )
            hit = active and bool(covered)
            # Extend the contract TTL when an active cache entry is reused.
            if hit:
                self._touch(normalized, now_utc, ttl)
            rows = self._connection.execute(
                """
                SELECT symbol, session_date, timestamp, open, high, low, close, volume
                FROM option_daily_bars
                WHERE symbol = ? AND session_date >= ? AND session_date <= ?
                ORDER BY session_date
                """,
                (normalized, start.isoformat(), end.isoformat()),
            ).fetchall()
        return OptionBarsCacheLookup(
            rows=[dict(row) for row in rows],
            missing_ranges=self._missing_ranges(start, end, covered),
            hit=hit,
        )

    def store(
        self,
        symbol: str,
        rows: Sequence[dict[str, Any]],
        *,
        start: date,
        end: date,
        now: datetime,
        ttl: timedelta,
    ) -> None:
        if end < start:
            raise ValueError("option cache end must be on or after start")
        normalized = str(symbol).strip().upper()
        now_utc = self._utc(now)
        values: list[tuple[Any, ...]] = []
        for row in rows:
            session_value = row.get("session_date")
            timestamp = row.get("timestamp")
            if session_value is None or timestamp is None:
                continue
            try:
                if isinstance(session_value, datetime):
                    session = session_value.date()
                elif isinstance(session_value, date):
                    session = session_value
                else:
                    session = date.fromisoformat(str(session_value))
            except (TypeError, ValueError):
                continue
            if session < start or session > end:
                continue
            values.append(
                (
                    normalized,
                    session.isoformat(),
                    str(timestamp),
                    self._number(row.get("open")),
                    self._number(row.get("high")),
                    self._number(row.get("low")),
                    self._number(row.get("close")),
                    self._number(row.get("volume")),
                )
            )

        with self._lock, self._connection:
            self._purge_expired(now_utc)
            self._connection.executemany(
                """
                INSERT OR REPLACE INTO option_daily_bars
                    (symbol, session_date, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            existing = self._connection.execute(
                """
                SELECT range_start, range_end
                FROM option_bar_coverage
                WHERE symbol = ?
                """,
                (normalized,),
            ).fetchall()
            # Remember queried ranges, including empty dates, to avoid repeating remote reads.
            merged = self._merge_ranges(
                [
                    (date.fromisoformat(row["range_start"]), date.fromisoformat(row["range_end"]))
                    for row in existing
                ]
                + [(start, end)]
            )
            self._connection.execute(
                "DELETE FROM option_bar_coverage WHERE symbol = ?",
                (normalized,),
            )
            self._connection.executemany(
                """
                INSERT INTO option_bar_coverage (symbol, range_start, range_end)
                VALUES (?, ?, ?)
                """,
                [
                    (normalized, range_start.isoformat(), range_end.isoformat())
                    for range_start, range_end in merged
                ],
            )
            self._touch(normalized, now_utc, ttl)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _touch(self, symbol: str, now: datetime, ttl: timedelta) -> None:
        expires = now + ttl
        self._connection.execute(
            """
            INSERT INTO option_bar_contracts (symbol, last_access_at, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                last_access_at=excluded.last_access_at,
                expires_at=excluded.expires_at
            """,
            (symbol, now.isoformat(), expires.isoformat()),
        )

    def _purge_expired(self, now: datetime) -> None:
        rows = self._connection.execute(
            "SELECT symbol FROM option_bar_contracts WHERE expires_at <= ?",
            (now.isoformat(),),
        ).fetchall()
        expired = [str(row["symbol"]) for row in rows]
        if not expired:
            return
        placeholders = ",".join("?" for _ in expired)
        self._connection.execute(
            f"DELETE FROM option_daily_bars WHERE symbol IN ({placeholders})",
            expired,
        )
        self._connection.execute(
            f"DELETE FROM option_bar_coverage WHERE symbol IN ({placeholders})",
            expired,
        )
        self._connection.execute(
            f"DELETE FROM option_bar_contracts WHERE symbol IN ({placeholders})",
            expired,
        )

    @staticmethod
    def _merge_ranges(ranges: Sequence[tuple[date, date]]) -> list[tuple[date, date]]:
        merged: list[tuple[date, date]] = []
        for start, end in sorted(ranges):
            if not merged or start > merged[-1][1] + timedelta(days=1):
                merged.append((start, end))
                continue
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    @staticmethod
    def _missing_ranges(
        start: date,
        end: date,
        covered: Sequence[tuple[date, date]],
    ) -> list[tuple[date, date]]:
        missing: list[tuple[date, date]] = []
        cursor = start
        for covered_start, covered_end in covered:
            covered_start = max(start, covered_start)
            covered_end = min(end, covered_end)
            if covered_end < cursor:
                continue
            if covered_start > cursor:
                missing.append((cursor, covered_start - timedelta(days=1)))
            cursor = max(cursor, covered_end + timedelta(days=1))
            if cursor > end:
                break
        if cursor <= end:
            missing.append((cursor, end))
        return missing

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _number(value: Any) -> float | None:
        if value is None:
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted


def default_option_bars_cache_path(context: Mapping[str, Any]) -> Path:
    configured = str(
        context.get("refdata.api.option_history_cache.path", "") or ""
    ).strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            base = context.get("AlgoHome") or os.environ.get("ALGO_HOME")
            path = Path(str(base)).expanduser() / path if base else Path.cwd() / path
        return path
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "algotrading" / "refdata-option-bars.sqlite3"
