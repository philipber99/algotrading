"""Sanitized service errors exposed by the Risk API."""

from __future__ import annotations


class RiskApiError(Exception):
    """Base error carrying a safe public HTTP representation."""

    status_code = 500
    code = "INTERNAL_ERROR"
    public_message = "An unexpected server error occurred"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.public_message)
        self.message = message or self.public_message


class BrokerNotReadyError(RiskApiError):
    status_code = 503
    code = "BROKER_NOT_READY"
    public_message = "The account service is not ready"


class BrokerUpstreamError(RiskApiError):
    status_code = 502
    code = "BROKER_UPSTREAM_ERROR"
    public_message = "Account information is temporarily unavailable"


class UpstreamSchemaError(RiskApiError):
    status_code = 502
    code = "UPSTREAM_SCHEMA_ERROR"
    public_message = "The broker returned an invalid account response"


class AccountNotFoundError(RiskApiError):
    status_code = 404
    code = "ACCOUNT_NOT_FOUND"
    public_message = "Account not found"


class SnapshotUnavailableError(RiskApiError):
    status_code = 409
    code = "SNAPSHOT_NOT_LOADED"
    public_message = "No portfolio snapshot is loaded; use an explicit refresh"


class SnapshotMismatchError(RiskApiError):
    status_code = 409
    code = "SNAPSHOT_MISMATCH"
    public_message = "The requested portfolio snapshot is stale or belongs to another scope"


class TransactionRepositoryError(RiskApiError):
    status_code = 503
    code = "TRANSACTION_REPOSITORY_UNAVAILABLE"
    public_message = "The transaction repository is unavailable"


class InvalidRiskRequestError(RiskApiError):
    status_code = 422
    code = "INVALID_REQUEST"
    public_message = "The request is invalid"


class ReferenceDataUnavailableError(RiskApiError):
    status_code = 503
    code = "REFERENCE_DATA_UNAVAILABLE"
    public_message = "Reference market data is temporarily unavailable"


class AlertNotFoundError(RiskApiError):
    status_code = 404
    code = "ALERT_NOT_FOUND"
    public_message = "Alert not found"


class AlertVersionConflictError(RiskApiError):
    status_code = 409
    code = "ALERT_VERSION_CONFLICT"
    public_message = "The alert changed; reload it before applying this action"


class PnlSelectionNotFoundError(RiskApiError):
    status_code = 404
    code = "PNL_SELECTION_NOT_FOUND"
    public_message = "The selected P&L evidence was not found in the requested scope"


class PnlMarkHistoryUnavailableError(RiskApiError):
    status_code = 503
    code = "PNL_MARK_HISTORY_UNAVAILABLE"
    public_message = "Required historical market marks are unavailable"


class PnlReconstructionLimitError(RiskApiError):
    status_code = 422
    code = "PNL_RECONSTRUCTION_LIMIT_EXCEEDED"
    public_message = "The P&L reconstruction request exceeds a configured limit"
