"""Stable exception and warning contracts for the cache package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..errors import ApiError


class CacheError(RuntimeError):
    """Base class for deterministic cache failures."""


class ConfigurationError(CacheError):
    """Configuration is missing, unsafe, or internally inconsistent."""


class SourceBacklogError(CacheError):
    """Canonical surface data is not complete enough to publish."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class ValidationError(CacheError):
    """A candidate or active generation failed an integrity check."""


class PublicationError(CacheError):
    """An immutable generation could not be atomically published."""


class QueryError(CacheError, ApiError):
    """Base class for REST query failures."""

    status_code = 500
    code = "QUERY_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        CacheError.__init__(self, message)
        self.message = message
        self.details = dict(details or {})


class InvalidRequestError(QueryError):
    status_code = 422
    code = "INVALID_REQUEST"


class CoverageUnavailableError(QueryError):
    status_code = 409
    code = "CACHE_COVERAGE_UNAVAILABLE"


class NotReadyError(QueryError):
    status_code = 503
    code = "CACHE_NOT_READY"


class BusyError(QueryError):
    status_code = 503
    code = "SERVICE_BUSY"


@dataclass(frozen=True)
class CacheWarning:
    code: str
    message: str
    details: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            result["details"] = dict(self.details)
        return result


WARNING_CACHE_STALE = "CACHE_STALE"
WARNING_LOW_SAMPLE_COUNT = "LOW_SAMPLE_COUNT"
WARNING_PARTIAL_SESSION_COVERAGE = "PARTIAL_SESSION_COVERAGE"
WARNING_STALE_SOURCE_SAMPLE = "STALE_SOURCE_SAMPLE"
WARNING_DTE_MATCH_RELAXED = "DTE_MATCH_RELAXED"
WARNING_MONEYNESS_INTERPOLATED = "MONEYNESS_INTERPOLATED"
WARNING_MONEYNESS_EXTRAPOLATED = "MONEYNESS_EXTRAPOLATED"
WARNING_FIT_QUALITY_DEGRADED = "FIT_QUALITY_DEGRADED"
WARNING_ARBITRAGE_FLAG_PRESENT = "ARBITRAGE_FLAG_PRESENT"
WARNING_PRICE_BASIS_UNAVAILABLE = "PRICE_BASIS_UNAVAILABLE"
