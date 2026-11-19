"""The sole Risk API boundary around TDABrokerInterface."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from algotrading.tda.TDABrokerInterface import TDABrokerInterface

from .errors import BrokerNotReadyError, BrokerUpstreamError


@dataclass(frozen=True)
class SourceResponse:
    payload: Any
    obtained_at: datetime


def create_tda_broker(
    context: dict[str, Any],
) -> Any:
    """Construct the one application-lifespan broker with streaming disabled."""

    return TDABrokerInterface(context, streaming=False)


class BrokerAccountSource:
    """Serialize and sanitize access to the two allowed broker methods."""

    def __init__(
        self,
        context: dict[str, Any],
        *,
        broker_factory: Callable[[dict[str, Any]], Any] = create_tda_broker,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._logger = _logger(context)
        self._log_guard = threading.local()
        _install_broker_log_filter(self._logger, self._log_guard)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._call_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._broker: Any | None = None
        self.broker_constructed = False
        self.required_methods_callable = False
        self.diagnostic: str | None = None
        self.last_account_request_at: datetime | None = None
        self.last_account_request_outcome: str | None = None

        try:
            self._log_guard.muted = True
            try:
                self._broker = broker_factory(context)
            finally:
                self._log_guard.muted = False
            self.broker_constructed = True
        except Exception as exc:
            self.diagnostic = "Broker construction failed"
            self._logger.error(
                "Risk broker construction failed exception=%s",
                type(exc).__name__,
            )
            return

        get_accounts = getattr(self._broker, "get_accounts", None)
        get_account = getattr(self._broker, "get_account", None)
        missing = []
        if not callable(get_accounts):
            missing.append("get_accounts")
        if not callable(get_account):
            missing.append("get_account")
        if missing:
            self.diagnostic = "Required broker methods unavailable: " + ", ".join(
                missing
            )
            self._logger.error(
                "Risk broker is not ready missing_methods=%s", ",".join(missing)
            )
            return

        self.required_methods_callable = True

    @property
    def ready(self) -> bool:
        return self.broker_constructed and self.required_methods_callable

    def fetch_all(self) -> SourceResponse:
        if not self.ready or self._broker is None:
            raise BrokerNotReadyError()
        return self._invoke(
            lambda: self._broker.get_accounts(),
            allow_not_found=False,
        )

    def fetch_one(self, account_id: str) -> SourceResponse:
        if not self.ready or self._broker is None:
            raise BrokerNotReadyError()
        return self._invoke(
            lambda: self._broker.get_account(account=account_id),
            allow_not_found=True,
        )

    def readiness(self) -> dict[str, Any]:
        with self._status_lock:
            return {
                "broker_constructed": self.broker_constructed,
                "required_methods_callable": self.required_methods_callable,
                "diagnostic": self.diagnostic,
                "last_account_request_at": self.last_account_request_at,
                "last_account_request_outcome": self.last_account_request_outcome,
            }

    def _invoke(
        self,
        method: Callable[..., Any],
        *,
        allow_not_found: bool,
    ) -> SourceResponse:
        # Serialize access to the shared broker instance across account requests.
        with self._call_lock:
            try:
                # Mute legacy broker logging only while sensitive account data is being fetched.
                self._log_guard.muted = True
                try:
                    payload = method()
                finally:
                    self._log_guard.muted = False
            except Exception as exc:
                obtained_at = _aware_utc(self._now())
                self._record_request(obtained_at, "failure")
                self._logger.error(
                    "Risk broker account request failed exception=%s",
                    type(exc).__name__,
                )
                raise BrokerUpstreamError() from exc
            obtained_at = _aware_utc(self._now())
            # Translate broker sentinel failures into the sanitized public error contract.
            if payload is None or (
                type(payload) is int and not (allow_not_found and payload == 404)
            ):
                self._record_request(obtained_at, "failure")
                self._logger.error("Risk broker account request returned no data")
                raise BrokerUpstreamError()
            self._record_request(obtained_at, "success")
            return SourceResponse(payload=payload, obtained_at=obtained_at)

    def _record_request(self, at: datetime, outcome: str) -> None:
        with self._status_lock:
            self.last_account_request_at = at
            self.last_account_request_outcome = outcome


def _logger(context: dict[str, Any]) -> logging.Logger:
    candidate = context.get("Logger") or context.get("logger")
    if isinstance(candidate, logging.Logger):
        return candidate
    return logging.getLogger("algotrading.risk.api")


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Risk API clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


class _BrokerCallLogFilter(logging.Filter):
    """Suppress legacy broker logs while a sensitive account call is active."""

    def __init__(self, guard: threading.local) -> None:
        super().__init__()
        self._guard = guard

    def filter(self, record: logging.LogRecord) -> bool:
        return not bool(getattr(self._guard, "muted", False))


def _install_broker_log_filter(logger: logging.Logger, guard: threading.local) -> None:
    logger.addFilter(_BrokerCallLogFilter(guard))
