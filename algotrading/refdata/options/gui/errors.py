"""Sanitized REST error contracts shared by cache and screener routes."""

from __future__ import annotations

from typing import Any, Mapping


class ApiError(RuntimeError):
    status_code = 500
    code = "API_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})


class ScreenerError(ApiError):
    code = "SCREENER_ERROR"


class ScreenerInvalidRequestError(ScreenerError):
    status_code = 422
    code = "INVALID_REQUEST"


class ScreenerNotReadyError(ScreenerError):
    status_code = 503
    code = "SCREENER_NOT_READY"


class ScreenerBusyError(ScreenerError):
    status_code = 503
    code = "SCREENER_BUSY"


class ScreenerTimeoutError(ScreenerError):
    status_code = 504
    code = "SCREENER_TIMEOUT"


class SnapshotNotFoundError(ScreenerError):
    status_code = 404
    code = "SNAPSHOT_NOT_FOUND"


class SnapshotNotTerminalError(ScreenerError):
    status_code = 409
    code = "SNAPSHOT_NOT_TERMINAL"


class SurfaceNotAvailableError(ScreenerError):
    status_code = 409
    code = "SURFACE_NOT_AVAILABLE"


class NoEligibleContractsError(ScreenerError):
    status_code = 422
    code = "NO_ELIGIBLE_CONTRACTS"


class PricingUnavailableError(ScreenerError):
    status_code = 503
    code = "PRICING_UNAVAILABLE"


__all__ = [name for name in globals() if name.endswith("Error")]
