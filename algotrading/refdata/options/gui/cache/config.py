"""Configuration parsing shared by the builder and REST service."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import ConfigurationError


CACHE_SCHEMA_VERSION = 1
CACHE_CONFIG_PREFIX = "options_chain.time_of_day_cache."
API_CONFIG_PREFIX = "options_chain.time_of_day_api."
_WINDOW_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
_SYMBOL = re.compile(r"^[A-Za-z0-9$^][A-Za-z0-9.$^_-]{0,31}$")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        raise ConfigurationError(f"Cannot parse boolean value {value!r}")
    return bool(value)


def _items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


@dataclass(frozen=True, order=True)
class HistoricalWindow:
    """One named, session-count based history selection."""

    sessions: int
    name: str
    label: str

    def __post_init__(self) -> None:
        if not _WINDOW_NAME.fullmatch(self.name):
            raise ConfigurationError(f"Invalid historical window name {self.name!r}")
        if self.sessions <= 0:
            raise ConfigurationError(
                f"Historical window {self.name!r} must contain at least one session"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "sessions": self.sessions,
        }


def _parse_window_mapping(raw: Any) -> dict[str, tuple[int, str]]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        parsed = raw
    else:
        text = str(raw).strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"{CACHE_CONFIG_PREFIX}windows is not valid JSON"
                ) from exc
        else:
            parsed = {}
            for item in _items(text):
                separator = ":" if ":" in item else "=" if "=" in item else None
                if separator is None:
                    raise ConfigurationError(
                        f"Historical window {item!r} must use name:sessions"
                    )
                name, sessions = (part.strip() for part in item.split(separator, 1))
                parsed[name] = sessions

    result: dict[str, tuple[int, str]] = {}
    for raw_name, raw_value in parsed.items():
        name = str(raw_name).strip()
        label = name
        sessions_value = raw_value
        if isinstance(raw_value, Mapping):
            sessions_value = raw_value.get("sessions")
            label = str(raw_value.get("label") or name)
        try:
            sessions = int(sessions_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(
                f"Historical window {name!r} has invalid session count "
                f"{sessions_value!r}"
            ) from exc
        result[name] = (sessions, label)
    return result


def parse_windows(context: Mapping[str, Any]) -> tuple[HistoricalWindow, ...]:
    """Parse named windows from a JSON/mapping value or per-window keys.

    Supported examples::

        options_chain.time_of_day_cache.windows = 10d:10,3m:63
        options_chain.time_of_day_cache.windows = {"10d": 10, "3m": 63}
        options_chain.time_of_day_cache.window.10d.sessions = 10
        options_chain.time_of_day_cache.window.10d.label = 10 Trading Days
    """

    values = _parse_window_mapping(context.get(CACHE_CONFIG_PREFIX + "windows"))
    prefix = CACHE_CONFIG_PREFIX + "window."
    for key, raw_value in context.items():
        key = str(key)
        if not key.startswith(prefix) or not key.endswith(".sessions"):
            continue
        name = key[len(prefix) : -len(".sessions")]
        label_key = prefix + name + ".label"
        try:
            sessions = int(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(
                f"Historical window {name!r} has invalid session count {raw_value!r}"
            ) from exc
        values[name] = (sessions, str(context.get(label_key, name)))

    if not values:
        values = {
            "10d": (10, "10 Trading Days"),
            "3m": (63, "3 Months"),
        }

    windows = tuple(
        sorted(
            (
                HistoricalWindow(sessions=sessions, name=name, label=label)
                for name, (sessions, label) in values.items()
            ),
            key=lambda item: (item.sessions, item.name),
        )
    )
    if len({item.name for item in windows}) != len(windows):
        raise ConfigurationError("Historical window names must be unique")
    return windows


@dataclass(frozen=True)
class CacheConfig:
    output_directory: Path
    symbols: tuple[str, ...]
    windows: tuple[HistoricalWindow, ...]
    default_window: str
    sample_interval_minutes: int
    max_snapshot_age_seconds: int
    price_bases: tuple[str, ...]
    require_surface_completion: bool
    allow_complete_with_failures: bool
    parquet_compression: str
    parquet_compression_level: int
    chunk_symbol_count: int
    keep_previous_generations: int
    staging_grace_hours: int
    minimum_session_coverage: float
    max_rejected_row_fraction: float
    monitor_service_suffix: str
    local_timezone: str
    exchange_timezone: str

    @property
    def maximum_window_sessions(self) -> int:
        return max(window.sessions for window in self.windows)

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "CacheConfig":
        windows = parse_windows(context)
        default_window = str(
            context.get(CACHE_CONFIG_PREFIX + "default_window", windows[0].name)
        ).strip()
        if default_window not in {window.name for window in windows}:
            raise ConfigurationError(
                f"default_window {default_window!r} is not one of the configured windows"
            )

        output = str(
            context.get(
                CACHE_CONFIG_PREFIX + "output_directory",
                os.path.join(
                    str(context.get("algo_home", os.getenv("ALGO_HOME", "."))),
                    "refdata",
                    "options_cache",
                    "time_of_day",
                ),
            )
        ).strip()
        if not output:
            raise ConfigurationError(
                f"{CACHE_CONFIG_PREFIX}output_directory is required"
            )

        raw_symbols = context.get(
            CACHE_CONFIG_PREFIX + "symbols",
            context.get("options_chain.symbols", ()),
        )
        symbols = tuple(dict.fromkeys(item.upper() for item in _items(raw_symbols)))
        if not symbols:
            raise ConfigurationError(
                f"{CACHE_CONFIG_PREFIX}symbols or options_chain.symbols is required"
            )
        invalid = [symbol for symbol in symbols if not _SYMBOL.fullmatch(symbol)]
        if invalid:
            raise ConfigurationError(f"Invalid cache symbols: {invalid}")

        bases = tuple(
            dict.fromkeys(
                item.lower()
                for item in _items(
                    context.get(CACHE_CONFIG_PREFIX + "price_bases", "mid")
                )
            )
        )
        invalid_bases = set(bases).difference({"mid", "bid", "ask"})
        if not bases or invalid_bases:
            raise ConfigurationError(
                f"price_bases must use mid, bid, and/or ask; invalid={sorted(invalid_bases)}"
            )

        config = cls(
            output_directory=Path(output).expanduser().resolve(),
            symbols=symbols,
            windows=windows,
            default_window=default_window,
            sample_interval_minutes=int(
                context.get(CACHE_CONFIG_PREFIX + "sample_interval_minutes", 5)
            ),
            max_snapshot_age_seconds=int(
                context.get(CACHE_CONFIG_PREFIX + "max_snapshot_age_seconds", 120)
            ),
            price_bases=bases,
            require_surface_completion=_as_bool(
                context.get(CACHE_CONFIG_PREFIX + "require_surface_completion", True)
            ),
            allow_complete_with_failures=_as_bool(
                context.get(
                    CACHE_CONFIG_PREFIX + "allow_complete_with_failures", True
                )
            ),
            parquet_compression=str(
                context.get(CACHE_CONFIG_PREFIX + "parquet_compression", "zstd")
            ).lower(),
            parquet_compression_level=int(
                context.get(CACHE_CONFIG_PREFIX + "parquet_compression_level", 6)
            ),
            chunk_symbol_count=int(
                context.get(CACHE_CONFIG_PREFIX + "chunk_symbol_count", 2)
            ),
            keep_previous_generations=int(
                context.get(CACHE_CONFIG_PREFIX + "keep_previous_generations", 1)
            ),
            staging_grace_hours=int(
                context.get(CACHE_CONFIG_PREFIX + "staging_grace_hours", 24)
            ),
            minimum_session_coverage=float(
                context.get(CACHE_CONFIG_PREFIX + "minimum_session_coverage", 0.90)
            ),
            max_rejected_row_fraction=float(
                context.get(CACHE_CONFIG_PREFIX + "max_rejected_row_fraction", 0.01)
            ),
            monitor_service_suffix=str(
                context.get(
                    CACHE_CONFIG_PREFIX + "nagios_service_suffix",
                    "TimeOfDayCache",
                )
            ).strip(),
            local_timezone=str(
                context.get(
                    CACHE_CONFIG_PREFIX + "local_timezone",
                    context.get("local_timezone", "America/Los_Angeles"),
                )
            ),
            exchange_timezone=str(
                context.get(
                    CACHE_CONFIG_PREFIX + "exchange_timezone",
                    context.get("exchange_timezone", "America/New_York"),
                )
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 1 <= self.sample_interval_minutes <= 60:
            raise ConfigurationError("sample_interval_minutes must be from 1 through 60")
        if self.max_snapshot_age_seconds < 0:
            raise ConfigurationError("max_snapshot_age_seconds cannot be negative")
        if self.parquet_compression not in {"zstd", "snappy", "gzip", "brotli", "lz4"}:
            raise ConfigurationError("Unsupported Parquet compression codec")
        if not 1 <= self.parquet_compression_level <= 22:
            raise ConfigurationError("parquet_compression_level must be from 1 through 22")
        if self.chunk_symbol_count <= 0:
            raise ConfigurationError("chunk_symbol_count must be positive")
        if self.keep_previous_generations < 1:
            raise ConfigurationError("keep_previous_generations must be at least one")
        if self.staging_grace_hours < 1:
            raise ConfigurationError("staging_grace_hours must be positive")
        if not 0 < self.minimum_session_coverage <= 1:
            raise ConfigurationError("minimum_session_coverage must be in (0, 1]")
        if not 0 <= self.max_rejected_row_fraction < 1:
            raise ConfigurationError("max_rejected_row_fraction must be in [0, 1)")
        if not self.monitor_service_suffix:
            raise ConfigurationError("nagios_service_suffix cannot be empty")


@dataclass(frozen=True)
class CacheQueryConfig:
    """Time-of-day manifest, query-cache, concurrency, and pricing settings."""

    manifest_poll_seconds: float
    result_cache_mb: int
    max_concurrent_queries: int
    query_timeout_seconds: float
    max_lookback_sessions: int
    pricing_workers: int
    pricing_batch_size: int
    max_pending_pricing_batches: int

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "CacheQueryConfig":
        config = cls(
            manifest_poll_seconds=float(
                context.get(API_CONFIG_PREFIX + "manifest_poll_seconds", 5)
            ),
            result_cache_mb=int(context.get(API_CONFIG_PREFIX + "result_cache_mb", 512)),
            max_concurrent_queries=int(
                context.get(API_CONFIG_PREFIX + "max_concurrent_queries", 4)
            ),
            query_timeout_seconds=float(
                context.get(API_CONFIG_PREFIX + "query_timeout_seconds", 5)
            ),
            max_lookback_sessions=int(
                context.get(API_CONFIG_PREFIX + "max_lookback_sessions", 1260)
            ),
            pricing_workers=int(context.get(API_CONFIG_PREFIX + "pricing_workers", 0)),
            pricing_batch_size=int(
                context.get(API_CONFIG_PREFIX + "pricing_batch_size", 256)
            ),
            max_pending_pricing_batches=int(
                context.get(API_CONFIG_PREFIX + "max_pending_pricing_batches", 0)
            ),
        )
        if config.manifest_poll_seconds < 0:
            raise ConfigurationError("manifest_poll_seconds cannot be negative")
        if config.result_cache_mb < 1:
            raise ConfigurationError("result_cache_mb must be positive")
        if config.max_concurrent_queries < 1:
            raise ConfigurationError("max_concurrent_queries must be positive")
        if config.query_timeout_seconds <= 0:
            raise ConfigurationError("query_timeout_seconds must be positive")
        if config.max_lookback_sessions < 1:
            raise ConfigurationError("max_lookback_sessions must be positive")
        if config.pricing_workers < 0:
            raise ConfigurationError("pricing_workers cannot be negative")
        if config.pricing_batch_size <= 0:
            raise ConfigurationError("pricing_batch_size must be positive")
        if config.max_pending_pricing_batches < 0:
            raise ConfigurationError("max_pending_pricing_batches cannot be negative")
        return config

    @property
    def resolved_pricing_workers(self) -> int:
        if self.pricing_workers:
            return self.pricing_workers
        try:
            available = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available = os.cpu_count() or 1
        return min(8, max(1, available - 1 if available > 1 else 1))

    @property
    def resolved_max_pending_pricing_batches(self) -> int:
        if self.max_pending_pricing_batches:
            return self.max_pending_pricing_batches
        return max(1, self.resolved_pricing_workers * 2)


def chunks(values: Sequence[Any], size: int) -> list[tuple[Any, ...]]:
    return [
        tuple(values[index : index + size])
        for index in range(0, len(values), size)
    ]
