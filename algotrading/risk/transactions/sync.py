"""Standalone, idempotent nightly broker-transaction importer."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import sys
from typing import Any
from zoneinfo import ZoneInfo

from algotrading.ContextFields import PARAM_LOCAL_TIMEZONE
from algotrading.risk.api.broker_source import create_tda_broker
from algotrading.risk.api.bootstrap import build_context

from .models import NormalizedTransaction, SUPPORTED_TRANSACTION_TYPES
from .normalization import normalize_transaction
from .repository import TransactionRepository


class ResponseCapUnresolved(RuntimeError):
    pass


@dataclass(frozen=True)
class SyncCounts:
    rows_received: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0
    chunks_completed: int = 0


class TransactionSynchronizer:
    def __init__(
        self,
        broker: Any,
        repository: TransactionRepository,
        context: Mapping[str, Any],
        *,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        logger: logging.Logger | None = None,
    ) -> None:
        self.broker = broker
        self.repository = repository
        self.context = context
        self._now = now_fn
        self.logger = logger or logging.getLogger("algotrading.risk.transactions.sync")

    def sync(
        self,
        *,
        accounts: Sequence[str],
        transaction_types: Sequence[str] = SUPPORTED_TRANSACTION_TYPES,
        start_at: datetime | None = None,
        through: datetime | None = None,
        symbol: str | None = None,
        dry_run: bool = False,
    ) -> tuple[dict[tuple[str, str], SyncCounts], list[str]]:
        end = _aware_utc(through or self._now())
        failures: list[str] = []
        results: dict[tuple[str, str], SyncCounts] = {}
        for account in dict.fromkeys(str(item).strip() for item in accounts):
            for transaction_type in dict.fromkeys(transaction_types):
                if transaction_type not in SUPPORTED_TRANSACTION_TYPES:
                    raise ValueError(f"unsupported transaction type: {transaction_type}")
                try:
                    counts = self._sync_stream(
                        account,
                        transaction_type,
                        start_at=start_at,
                        through=end,
                        symbol=symbol,
                        dry_run=dry_run,
                    )
                    results[(account, transaction_type)] = counts
                except NotImplementedError:
                    raise
                except Exception as exc:
                    masked = _mask_account(account)
                    failures.append(f"{masked}:{transaction_type}:{type(exc).__name__}")
                    self.logger.error(
                        "transaction sync failed account=%s type=%s error=%s",
                        masked,
                        transaction_type,
                        type(exc).__name__,
                    )
        return results, failures

    def _sync_stream(
        self,
        account: str,
        transaction_type: str,
        *,
        start_at: datetime | None,
        through: datetime,
        symbol: str | None,
        dry_run: bool,
    ) -> SyncCounts:
        overlap = timedelta(minutes=int(self.context.get("risk.transactions.sync.overlap_minutes", 10)))
        watermark = None
        if not dry_run:
            sync_lookup = getattr(self.repository, "last_sync_through", None)
            if callable(sync_lookup):
                watermark = sync_lookup(account, transaction_type)
            if watermark is None:
                watermark = self.repository.last_transaction_at(
                    account, transaction_type
                )
        start = _aware_utc(start_at) if start_at is not None else None
        # Replay a small overlap so corrected broker records are upserted safely.
        if watermark is not None:
            start = watermark - overlap
        if start is None:
            configured = str(self.context.get("risk.transactions.sync.initial_start_at", "") or "").strip()
            start = _parse_aware(configured) if configured else through - timedelta(
                days=int(self.context.get("risk.transactions.sync.initial_lookback_days", 30))
            )
        if start >= through:
            return SyncCounts()
        chunk_days = min(365, max(1, int(self.context.get("risk.transactions.sync.chunk_days", 30))))
        chunks = _time_chunks(start, through, timedelta(days=chunk_days))
        loaded: list[tuple[datetime, datetime, tuple[NormalizedTransaction, ...]]] = []
        # Provider work and all validation happen before the first write. This
        # makes the deliberate owner placeholder and malformed/capped responses
        # leave rows and watermarks untouched.
        for chunk_start, chunk_end in chunks:
            raw_rows = self._fetch_bounded(
                account,
                transaction_type,
                chunk_start,
                chunk_end,
                symbol=symbol,
            )
            normalized = tuple(
                normalize_transaction(
                    row,
                    account_id=account,
                    requested_type=transaction_type,
                )
                for row in raw_rows
            )
            loaded.append((chunk_start, chunk_end, normalized))
        if dry_run:
            return SyncCounts(
                rows_received=sum(len(rows) for _, _, rows in loaded),
                chunks_completed=len(loaded),
            )

        inserted = updated = received = completed = 0
        lock_wait = int(self.context.get("risk.transactions.sync.advisory_lock_timeout_seconds", 0))
        with self.repository.advisory_lock(account, lock_wait):
            for _, _, rows in loaded:
                chunk_inserted, chunk_updated = self.repository.upsert_chunk(rows)
                received += len(rows)
                inserted += chunk_inserted
                updated += chunk_updated
                completed += 1
            # Advance durable coverage only after all transaction writes succeed.
            record_success = getattr(self.repository, "record_sync_success", None)
            if callable(record_success):
                record_success(account, transaction_type, start, through)
        return SyncCounts(received, inserted, updated, completed)

    def _fetch_bounded(
        self,
        account: str,
        transaction_type: str,
        start: datetime,
        end: datetime,
        *,
        symbol: str | None,
    ) -> list[Mapping[str, Any]]:
        if end - start > timedelta(days=365):
            raise ValueError("a broker transaction request may not exceed one year")
        local_timezone = self.context.get(PARAM_LOCAL_TIMEZONE) or self.context.get(
            "risk.display_timezone", "America/Los_Angeles"
        )
        result = self.broker.get_transactions(
            account,
            _naive_local(start, local_timezone),
            _naive_local(end, local_timezone),
            transaction_type,
            symbol=symbol,
        )
        if not isinstance(result, list) or any(not isinstance(item, Mapping) for item in result):
            raise ValueError("broker transaction response must be a list of objects")
        if len(result) < 3000:
            return result
        minimum_seconds = max(1, int(self.context.get("risk.transactions.sync.minimum_split_seconds", 1)))
        if (end - start).total_seconds() <= minimum_seconds:
            raise ResponseCapUnresolved("response_cap_unresolved")
        # Bisect capped responses until each broker request is demonstrably complete.
        midpoint = start + (end - start) / 2
        if midpoint <= start or midpoint >= end:
            raise ResponseCapUnresolved("response_cap_unresolved")
        left = self._fetch_bounded(
            account, transaction_type, start, midpoint, symbol=symbol
        )
        right = self._fetch_bounded(
            account, transaction_type, midpoint, end, symbol=symbol
        )
        return left + right


def _time_chunks(
    start: datetime, end: datetime, size: timedelta
) -> list[tuple[datetime, datetime]]:
    chunks = []
    cursor = start
    while cursor < end:
        next_end = min(end, cursor + size)
        chunks.append((cursor, next_end))
        cursor = next_end
    return chunks


def _parse_aware(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be RFC 3339") from exc
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone aware")
    return value.astimezone(timezone.utc)


def _naive_local(value: datetime, local_timezone: Any) -> datetime:
    """Convert an aware internal timestamp to the broker's local-wallclock contract."""

    aware = _aware_utc(value)
    zone = local_timezone if hasattr(local_timezone, "utcoffset") else ZoneInfo(str(local_timezone))
    return aware.astimezone(zone).replace(tzinfo=None)


def _mask_account(value: str) -> str:
    text = str(value)
    return "***" + text[-4:] if len(text) > 4 else "***"


def _discover_accounts(broker: Any) -> list[str]:
    raw_accounts = broker.get_accounts()
    if not isinstance(raw_accounts, list):
        raise ValueError("broker accounts response must be a list")

    account_ids: list[str] = []
    for envelope in raw_accounts:
        if not isinstance(envelope, Mapping):
            raise ValueError("broker account response must contain objects")
        securities_account = envelope.get("securitiesAccount")
        if not isinstance(securities_account, Mapping):
            raise ValueError("broker account response is missing securitiesAccount")
        account_id = securities_account.get("accountNumber")
        if not isinstance(account_id, str) or not account_id.strip():
            raise ValueError("broker account response is missing accountNumber")
        account_ids.append(account_id.strip())

    discovered = list(dict.fromkeys(account_ids))
    if not discovered:
        raise ValueError("broker returned no accounts")
    return discovered


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import account transactions into Risk MySQL")
    parser.add_argument("-e", "--environment", default="refdata")
    parser.add_argument(
        "--account",
        action="append",
        help="account ID to update (repeatable); omitted discovers all broker accounts",
    )
    parser.add_argument("--type", dest="types", action="append", choices=SUPPORTED_TRANSACTION_TYPES)
    parser.add_argument("--start-at", type=_parse_aware)
    parser.add_argument("--through", type=_parse_aware)
    parser.add_argument("--symbol")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        context = build_context(args.environment, quiet_configuration=True)
        context["tda.oauthlib_logging"] = False
        context["tda.websockets_logging"] = False
        repository = TransactionRepository.from_context(context)
        broker = create_tda_broker(context)
        accounts = args.account or _discover_accounts(broker)
        synchronizer = TransactionSynchronizer(broker, repository, context)
        results, failures = synchronizer.sync(
            accounts=accounts,
            transaction_types=args.types or SUPPORTED_TRANSACTION_TYPES,
            start_at=args.start_at,
            through=args.through,
            symbol=args.symbol,
            dry_run=args.dry_run,
        )
    except NotImplementedError:
        print(
            "Transaction import unavailable: TDABrokerInterface.get_transactions "
            "must be implemented by the owner.",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(f"Transaction import failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    for (account, transaction_type), counts in results.items():
        print(
            f"account={_mask_account(account)} type={transaction_type} "
            f"received={counts.rows_received} inserted={counts.rows_inserted} "
            f"updated={counts.rows_updated} chunks={counts.chunks_completed}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
