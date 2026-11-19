"""Server-wide host and middleware configuration for the Options GUI API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .cache.errors import ConfigurationError


API_CONFIG_PREFIX = "options_chain.time_of_day_api."


def _items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


@dataclass(frozen=True)
class ApiConfig:
    """Settings that apply to the complete REST server."""

    host: str
    port: int
    allowed_origins: tuple[str, ...]
    gzip_minimum_size: int
    max_request_bytes: int
    log_level: str

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "ApiConfig":
        config = cls(
            host=str(context.get(API_CONFIG_PREFIX + "host", "0.0.0.0")),
            port=int(context.get(API_CONFIG_PREFIX + "port", 8788)),
            allowed_origins=tuple(
                _items(context.get(API_CONFIG_PREFIX + "allowed_origins", ()))
            ),
            gzip_minimum_size=int(
                context.get(API_CONFIG_PREFIX + "gzip_minimum_size", 1000)
            ),
            max_request_bytes=int(
                context.get(API_CONFIG_PREFIX + "max_request_bytes", 1_048_576)
            ),
            log_level=str(context.get(API_CONFIG_PREFIX + "log_level", "INFO")).upper(),
        )
        if not 1 <= config.port <= 65535:
            raise ConfigurationError("time_of_day_api.port is invalid")
        if config.gzip_minimum_size < 0:
            raise ConfigurationError("gzip_minimum_size cannot be negative")
        if config.max_request_bytes < 1024:
            raise ConfigurationError("max_request_bytes must be at least 1024")
        return config


__all__ = ["API_CONFIG_PREFIX", "ApiConfig"]
