"""Validated, non-secret server configuration for the Risk API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RiskApiConfig:
    host: str = "127.0.0.1"
    port: int = 8789
    log_level: str = "INFO"
    cors_origins: tuple[str, ...] = ()
    account_cache_ttl_seconds: float = 60.0
    max_concurrent_broker_calls: int = 1
    gzip_minimum_size: int = 1000

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "RiskApiConfig":
        host = str(context.get("risk.api.host", cls.host)).strip()
        if not host:
            raise ValueError("risk.api.host must not be empty")

        port = _integer(context.get("risk.api.port", cls.port), "risk.api.port")
        if not 1 <= port <= 65535:
            raise ValueError("risk.api.port must be between 1 and 65535")

        log_level = (
            str(context.get("risk.api.log_level", cls.log_level)).strip().upper()
        )
        if log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("risk.api.log_level is invalid")

        ttl = _number(
            context.get(
                "risk.api.account_cache_ttl_seconds",
                cls.account_cache_ttl_seconds,
            ),
            "risk.api.account_cache_ttl_seconds",
        )
        if ttl < 0:
            raise ValueError("risk.api.account_cache_ttl_seconds must be non-negative")

        maximum = _integer(
            context.get(
                "risk.api.max_concurrent_broker_calls",
                cls.max_concurrent_broker_calls,
            ),
            "risk.api.max_concurrent_broker_calls",
        )
        # The broker boundary supports one account request at a time.
        if maximum != 1:
            raise ValueError(
                "risk.api.max_concurrent_broker_calls must be 1 in this version"
            )

        gzip_minimum = _integer(
            context.get("risk.api.gzip_minimum_size", cls.gzip_minimum_size),
            "risk.api.gzip_minimum_size",
        )
        if gzip_minimum < 0:
            raise ValueError("risk.api.gzip_minimum_size must be non-negative")

        origins = _origins(context.get("risk.api.cors_origins", ()))
        # Require explicit browser origins for account-data access.
        if "*" in origins:
            raise ValueError("risk.api.cors_origins must contain explicit origins")

        return cls(
            host=host,
            port=port,
            log_level=log_level,
            cors_origins=origins,
            account_cache_ttl_seconds=ttl,
            max_concurrent_broker_calls=maximum,
            gzip_minimum_size=gzip_minimum,
        )


def _origins(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        raise ValueError("risk.api.cors_origins must be a list or comma-separated text")
    return tuple(
        dict.fromkeys(str(item).strip() for item in items if str(item).strip())
    )


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return converted


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted
