"""Account retrieval, normalization, caching, and position flattening."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Generic, Literal, Protocol, TypeVar

from .broker_source import SourceResponse
from .errors import BrokerUpstreamError, UpstreamSchemaError
from .models import Account, AccountSummary, Position
from .normalization import normalize_accounts, normalize_specific_account

SERVICE_NAME = "Algotrading Risk API"
SERVICE_VERSION = "1.0.0"

T = TypeVar("T")
CacheState = Literal["hit", "miss", "bypass"]


class AccountSource(Protocol):
    def fetch_all(self) -> SourceResponse: ...

    def fetch_one(self, account_id: str) -> SourceResponse: ...

    def readiness(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ServiceResult(Generic[T]):
    data: T
    as_of: datetime
    cache: CacheState


@dataclass(frozen=True)
class _CacheEntry(Generic[T]):
    value: T
    as_of: datetime
    expires_at: float


class RiskService:
    def __init__(
        self,
        source: AccountSource,
        *,
        cache_ttl_seconds: float = 2.0,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        self._source = source
        self._ttl = float(cache_ttl_seconds)
        self._monotonic = monotonic_fn
        self._lock = threading.RLock()
        self._all_accounts: _CacheEntry[tuple[Account, ...]] | None = None
        self._accounts: dict[str, _CacheEntry[Account]] = {}

    def accounts(
        self, *, include_positions: bool = True, refresh: bool = False
    ) -> ServiceResult[list[Account | AccountSummary]]:
        result = self._retrieve_all(refresh=refresh)
        data = [_account_view(account, include_positions) for account in result.data]
        return ServiceResult(data=data, as_of=result.as_of, cache=result.cache)

    def account(
        self,
        account_id: str,
        *,
        include_positions: bool = True,
        refresh: bool = False,
    ) -> ServiceResult[Account | AccountSummary]:
        result = self._retrieve_one(account_id, refresh=refresh)
        return ServiceResult(
            data=_account_view(result.data, include_positions),
            as_of=result.as_of,
            cache=result.cache,
        )

    def positions(
        self, *, account_id: str | None = None, refresh: bool = False
    ) -> ServiceResult[list[Position]]:
        if account_id is None:
            result = self._retrieve_all(refresh=refresh)
            positions = [
                position.model_copy(deep=True)
                for account in result.data
                for position in account.positions
            ]
        else:
            result = self._retrieve_one(account_id, refresh=refresh)
            positions = [
                position.model_copy(deep=True) for position in result.data.positions
            ]
        return ServiceResult(
            data=positions,
            as_of=result.as_of,
            cache=result.cache,
        )

    def readiness(self) -> dict[str, Any]:
        return self._source.readiness()

    def _retrieve_all(self, *, refresh: bool) -> ServiceResult[list[Account]]:
        # Serialize cache checks and loads to avoid duplicate broker requests.
        with self._lock:
            now = self._monotonic()
            entry = self._all_accounts
            if (
                not refresh
                and self._ttl > 0
                and entry is not None
                and entry.expires_at > now
            ):
                return ServiceResult(
                    data=_copy_accounts(entry.value),
                    as_of=entry.as_of,
                    cache="hit",
                )

            response = self._source.fetch_all()
            _reject_failed_response(response.payload, specific=False)
            accounts = normalize_accounts(response.payload)
            state: CacheState = "bypass" if refresh or self._ttl == 0 else "miss"
            if self._ttl > 0:
                expires_at = self._monotonic() + self._ttl
                stored = tuple(account.model_copy(deep=True) for account in accounts)
                self._all_accounts = _CacheEntry(
                    value=stored,
                    as_of=response.obtained_at,
                    expires_at=expires_at,
                )
                # Populate per-account entries from the same complete broker snapshot.
                for account in stored:
                    self._accounts[account.account_id] = _CacheEntry(
                        value=account.model_copy(deep=True),
                        as_of=response.obtained_at,
                        expires_at=expires_at,
                    )
            return ServiceResult(
                data=_copy_accounts(accounts),
                as_of=response.obtained_at,
                cache=state,
            )

    def _retrieve_one(
        self, account_id: str, *, refresh: bool
    ) -> ServiceResult[Account]:
        with self._lock:
            now = self._monotonic()
            entry = self._accounts.get(account_id)
            if (
                not refresh
                and self._ttl > 0
                and entry is not None
                and entry.expires_at > now
            ):
                return ServiceResult(
                    data=entry.value.model_copy(deep=True),
                    as_of=entry.as_of,
                    cache="hit",
                )

            response = self._source.fetch_one(account_id)
            _reject_failed_response(response.payload, specific=True)
            account = normalize_specific_account(response.payload)
            # Reject a broker response for any account other than the one requested.
            if account.account_id != account_id:
                raise UpstreamSchemaError()
            state: CacheState = "bypass" if refresh or self._ttl == 0 else "miss"
            if self._ttl > 0:
                self._accounts[account_id] = _CacheEntry(
                    value=account.model_copy(deep=True),
                    as_of=response.obtained_at,
                    expires_at=self._monotonic() + self._ttl,
                )
            return ServiceResult(
                data=account.model_copy(deep=True),
                as_of=response.obtained_at,
                cache=state,
            )


def _account_view(
    account: Account, include_positions: bool
) -> Account | AccountSummary:
    if include_positions:
        return account.model_copy(deep=True)
    values = account.model_dump(exclude={"positions"})
    return AccountSummary.model_validate(values)


def _copy_accounts(accounts: Any) -> list[Account]:
    return [account.model_copy(deep=True) for account in accounts]


def _reject_failed_response(payload: Any, *, specific: bool) -> None:
    if payload is None:
        raise BrokerUpstreamError()
    if type(payload) is int and not (specific and payload == 404):
        raise BrokerUpstreamError()
