"""MySQL repository for durable account transactions and stored-range queries."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import base64
import hashlib
import hmac
import json
from typing import Any
from zoneinfo import ZoneInfo

from algotrading.ContextFields import (
    PARAM_REFDATA_DB_DEFAULTDB,
    PARAM_REFDATA_DB_HOSTNAME,
    PARAM_REFDATA_DB_PASSWORD,
    PARAM_REFDATA_DB_USERNAME,
    PARAM_LOCAL_TIMEZONE,
)
from algotrading.refdata.repositories.connection import StockAnalysisConnectionFactory
from algotrading.refdata.repositories.base import row_dict

from .models import (
    NormalizedTransaction,
    SUPPORTED_TRANSACTION_TYPES,
    SyncStatusItem,
    TransactionCoverage,
    TransactionFeeModel,
    TransactionItemModel,
    TransactionMeta,
    TransactionModel,
    TransactionsResponse,
    TransactionStatusResponse,
)


class TransactionRepositoryUnavailable(RuntimeError):
    pass


class InvalidCursor(ValueError):
    pass


class TransactionResultLimitExceeded(ValueError):
    pass


class TransactionRepository:
    def __init__(
        self,
        connection_factory: StockAnalysisConnectionFactory,
        *,
        cursor_secret: bytes,
        now_fn=lambda: datetime.now(timezone.utc),
        feed_timezone: str = "America/Los_Angeles",
        daily_feed_complete_hour: int = 20,
    ) -> None:
        self.connection_factory = connection_factory
        self._cursor_secret = cursor_secret
        self._now = now_fn
        self._feed_timezone = ZoneInfo(feed_timezone)
        self._daily_feed_complete_hour = min(23, max(0, daily_feed_complete_hour))

    @classmethod
    def from_context(
        cls,
        context: Mapping[str, Any],
        *,
        connector: Any | None = None,
        now_fn=lambda: datetime.now(timezone.utc),
    ) -> "TransactionRepository":
        required = {
            "host": str(
                context.get("risk.transactions.sql.host")
                or context.get(PARAM_REFDATA_DB_HOSTNAME)
                or ""
            ).strip(),
            "database": str(
                context.get("risk.transactions.sql.database")
                or context.get(PARAM_REFDATA_DB_DEFAULTDB)
                or ""
            ).strip(),
            "user": str(
                context.get("risk.transactions.sql.user")
                or context.get(PARAM_REFDATA_DB_USERNAME)
                or ""
            ).strip(),
            "password": str(
                context.get("risk.transactions.sql.password")
                or context.get(PARAM_REFDATA_DB_PASSWORD)
                or ""
            ),
        }
        if not required["host"] or not required["database"] or not required["user"]:
            raise TransactionRepositoryUnavailable(
                "Risk transaction MySQL host, database, and user are not configured"
            )
        translated = dict(context)
        translated[PARAM_REFDATA_DB_HOSTNAME] = required["host"]
        translated[PARAM_REFDATA_DB_DEFAULTDB] = required["database"]
        translated[PARAM_REFDATA_DB_USERNAME] = required["user"]
        translated[PARAM_REFDATA_DB_PASSWORD] = required["password"]
        translated["refdata.stock_analysis.sql.port"] = int(
            context.get("risk.transactions.sql.port", 3306)
        )
        translated["refdata.stock_analysis.sql.connection_timeout_seconds"] = int(
            context.get("risk.transactions.sql.connection_timeout_seconds", 5)
        )
        translated["refdata.stock_analysis.sql.connection_attempts"] = 1
        translated["refdata.stock_analysis.sql.pool.enabled"] = True
        translated["refdata.stock_analysis.sql.pool.size"] = int(
            context.get("risk.transactions.sql.pool_size", 4)
        )
        factory = StockAnalysisConnectionFactory(translated, connector=connector)
        explicit_secret = str(context.get("risk.transactions.cursor_secret", "") or "")
        seed = explicit_secret or f"{required['database']}:risk-transaction-cursor-v1"
        return cls(
            factory,
            cursor_secret=hashlib.sha256(seed.encode("utf-8")).digest(),
            now_fn=now_fn,
            feed_timezone=str(
                context.get(PARAM_LOCAL_TIMEZONE)
                or context.get("risk.display_timezone")
                or "America/Los_Angeles"
            ),
            daily_feed_complete_hour=int(
                context.get("risk.transactions.daily_feed_complete_hour", 20)
            ),
        )

    def query(
        self,
        account_id: str,
        *,
        start_at: datetime,
        end_at: datetime,
        types: Sequence[str] | None = None,
        symbol: str | None = None,
        limit: int = 250,
        cursor: str | None = None,
    ) -> TransactionsResponse:
        normalized_types = tuple(types or SUPPORTED_TRANSACTION_TYPES)
        if any(item not in SUPPORTED_TRANSACTION_TYPES for item in normalized_types):
            raise ValueError("unsupported transaction type")
        start_utc, end_utc = _require_interval(start_at, end_at)
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        # Keep client values in bound SQL parameters while composing fixed predicates.
        params: list[Any] = [account_id, _mysql_utc(start_utc), _mysql_utc(end_utc)]
        predicates = [
            "t.account_id = %s",
            "t.transaction_at >= %s",
            "t.transaction_at < %s",
        ]
        if normalized_types:
            predicates.append(
                "t.transaction_type IN (" + ",".join("%s" for _ in normalized_types) + ")"
            )
            params.extend(normalized_types)
        if symbol:
            normalized_symbol = symbol.strip().upper()
            predicates.append(
                "(t.symbol = %s OR EXISTS ("
                "SELECT 1 FROM risk_transaction_items si "
                "WHERE si.transaction_id = t.id AND si.symbol = %s))"
            )
            params.extend([normalized_symbol, normalized_symbol])
        # Use timestamp and row ID together for stable descending keyset pagination.
        if cursor:
            cursor_at, cursor_id = self._decode_cursor(cursor)
            predicates.append("(t.transaction_at < %s OR (t.transaction_at = %s AND t.id < %s))")
            params.extend([_mysql_utc(cursor_at), _mysql_utc(cursor_at), cursor_id])
        sql = (
            "SELECT t.id, t.broker_transaction_id, t.transaction_at, "
            "t.transaction_type, t.transaction_subtype, t.description, t.symbol, "
            "t.asset_type, t.trade_date, t.settlement_date, t.amount, t.fees, "
            "t.net_amount, t.currency, t.order_id, t.source_updated_at, t.ingested_at "
            "FROM risk_transactions t WHERE "
            + " AND ".join(predicates)
            + " ORDER BY t.transaction_at DESC, t.id DESC LIMIT %s"
        )
        # Read one extra row to decide whether another page exists.
        params.append(limit + 1)
        try:
            with self.connection_factory.cursor(dictionary=True) as (_, db_cursor, _):
                db_cursor.execute(sql, tuple(params))
                rows = [row_dict(db_cursor, item) for item in db_cursor.fetchall()]
                page_rows = rows[:limit]
                items_by_parent = self._load_items(db_cursor, [int(row["id"]) for row in page_rows])
                fees_by_parent = self._load_fee_components(
                    db_cursor, [int(row["id"]) for row in page_rows]
                )
                sync_ranges = self._sync_ranges(db_cursor, account_id)
        except Exception as exc:
            raise TransactionRepositoryUnavailable("Transaction repository query failed") from exc
        data = [
            self._transaction_model(
                row,
                items_by_parent.get(int(row["id"]), []),
                fees_by_parent.get(int(row["id"]), []),
            )
            for row in page_rows
        ]
        next_cursor = None
        if len(rows) > limit and page_rows:
            tail = page_rows[-1]
            next_cursor = self._encode_cursor(_api_utc(tail["transaction_at"]), int(tail["id"]))
        coverage = self._coverage(
            normalized_types, start_utc, end_utc, sync_ranges
        )
        return TransactionsResponse(
            account_id=account_id,
            data=data,
            meta=TransactionMeta(
                as_of=max((item.ingested_at for item in data), default=self._now().astimezone(timezone.utc)),
                next_cursor=next_cursor,
                coverage=coverage,
                warnings=[] if coverage.complete else ["transaction_data_coverage_incomplete"],
            ),
        )

    def status(self, account_id: str) -> TransactionStatusResponse:
        try:
            with self.connection_factory.cursor(dictionary=True) as (_, cursor, _):
                rows = self._sync_ranges(cursor, account_id)
        except Exception as exc:
            raise TransactionRepositoryUnavailable("Transaction repository status query failed") from exc
        by_type = {str(row["transaction_type"]): row for row in rows}
        data = []
        for transaction_type in SUPPORTED_TRANSACTION_TYPES:
            row = by_type.get(transaction_type)
            if row is None:
                data.append(SyncStatusItem(transaction_type=transaction_type, status="never_run"))
            else:
                data.append(
                    SyncStatusItem(
                        transaction_type=transaction_type,
                        last_successful_start_at=_api_utc_optional(row.get("first_synced_at")),
                        last_successful_end_at=_api_utc_optional(row.get("synced_through_at")),
                        last_success_at=_api_utc_optional(row.get("last_success_at")),
                        status="available",
                    )
                )
        return TransactionStatusResponse(
            account_id=account_id,
            data=data,
            as_of=self._now().astimezone(timezone.utc),
        )

    def list_trade_accounts(self) -> list[str]:
        """Return accounts that have durable trade history."""
        try:
            with self.connection_factory.cursor(dictionary=True) as (_, cursor, _):
                cursor.execute(
                    "SELECT DISTINCT account_id FROM risk_transactions "
                    "WHERE transaction_type = 'TRADE' ORDER BY account_id"
                )
                rows = [row_dict(cursor, item) for item in cursor.fetchall()]
        except Exception as exc:
            raise TransactionRepositoryUnavailable(
                "Transaction repository account query failed"
            ) from exc
        return [
            value
            for row in rows
            if (value := str(row.get("account_id") or "").strip())
        ]

    def query_all_trade_history(
        self,
        account_id: str,
        *,
        through: datetime,
        max_rows: int,
    ) -> list[TransactionModel]:
        """Load all stored trades before ``through`` in chronological order.

        This bounded internal read is intended for one-time reconstruction jobs
        that need the inventory preceding their output window. It does not alter
        the paginated public transaction API.
        """
        end = _api_utc(through)
        if not 1 <= max_rows <= 1_000_000:
            raise ValueError("max_rows must be between 1 and 1000000")
        sql = (
            "SELECT t.id, t.broker_transaction_id, t.transaction_at, "
            "t.transaction_type, t.transaction_subtype, t.description, t.symbol, "
            "t.asset_type, t.trade_date, t.settlement_date, t.amount, t.fees, "
            "t.net_amount, t.currency, t.order_id, t.source_updated_at, t.ingested_at "
            "FROM risk_transactions t "
            "WHERE t.account_id = %s AND t.transaction_type = 'TRADE' "
            "AND t.transaction_at < %s "
            "ORDER BY t.transaction_at ASC, t.id ASC LIMIT %s"
        )
        try:
            with self.connection_factory.cursor(dictionary=True) as (_, cursor, _):
                cursor.execute(sql, (account_id, _mysql_utc(end), max_rows + 1))
                rows = [row_dict(cursor, item) for item in cursor.fetchall()]
                if len(rows) > max_rows:
                    raise TransactionResultLimitExceeded(
                        "transaction history exceeds the configured backfill limit"
                    )
                items_by_parent = self._load_items(
                    cursor, [int(row["id"]) for row in rows]
                )
                fees_by_parent = self._load_fee_components(
                    cursor, [int(row["id"]) for row in rows]
                )
        except TransactionResultLimitExceeded:
            raise
        except Exception as exc:
            raise TransactionRepositoryUnavailable(
                "Transaction repository full-history query failed"
            ) from exc
        return [
            self._transaction_model(
                row,
                items_by_parent.get(int(row["id"]), []),
                fees_by_parent.get(int(row["id"]), []),
            )
            for row in rows
        ]

    def last_transaction_at(self, account_id: str, transaction_type: str) -> datetime | None:
        """Return the latest durable broker timestamp for one account/type stream."""

        with self.connection_factory.cursor(dictionary=True) as (_, cursor, _):
            cursor.execute(
                "SELECT MAX(transaction_at) AS last_transaction_at FROM risk_transactions "
                "WHERE account_id = %s AND transaction_type = %s",
                (account_id, transaction_type),
            )
            row = cursor.fetchone()
            normalized = row_dict(cursor, row) if row is not None else None
        return _api_utc_optional(normalized.get("last_transaction_at")) if normalized else None

    def query_lifecycle_history(
        self,
        account_id: str,
        *,
        history_start_at: datetime,
        end_at: datetime,
        requested_start_at: datetime,
        requested_end_at: datetime,
        max_rows: int,
    ) -> tuple[list[TransactionModel], TransactionCoverage, datetime]:
        """Load a bounded chronological TRADE history and all child items.

        This internal analytics read leaves the public transaction route and
        its pagination contract unchanged. It only reads the MySQL repository.
        """
        history_start = _api_utc(history_start_at)
        history_end = _api_utc(end_at)
        requested_start = _api_utc(requested_start_at)
        requested_end = _api_utc(requested_end_at)
        if history_end <= history_start:
            raise ValueError("history end_at must be after history_start_at")
        if requested_end <= requested_start:
            raise ValueError("requested_end_at must be after requested_start_at")
        if not 1 <= max_rows <= 1_000_000:
            raise ValueError("max_rows must be between 1 and 1000000")
        sql = (
            "SELECT t.id, t.broker_transaction_id, t.transaction_at, "
            "t.transaction_type, t.transaction_subtype, t.description, t.symbol, "
            "t.asset_type, t.trade_date, t.settlement_date, t.amount, t.fees, "
            "t.net_amount, t.currency, t.order_id, t.source_updated_at, t.ingested_at "
            "FROM risk_transactions t "
            "WHERE t.account_id = %s AND t.transaction_type = 'TRADE' "
            "AND t.transaction_at >= %s AND t.transaction_at < %s "
            "ORDER BY t.transaction_at ASC, t.id ASC LIMIT %s"
        )
        try:
            with self.connection_factory.cursor(dictionary=True) as (_, cursor, _):
                cursor.execute(
                    sql,
                    (
                        account_id,
                        _mysql_utc(history_start),
                        _mysql_utc(history_end),
                        max_rows + 1,
                    ),
                )
                rows = [row_dict(cursor, item) for item in cursor.fetchall()]
                if len(rows) > max_rows:
                    raise TransactionResultLimitExceeded(
                        "transaction history exceeds the configured reconstruction limit"
                    )
                items_by_parent = self._load_items(
                    cursor, [int(row["id"]) for row in rows]
                )
                fees_by_parent = self._load_fee_components(
                    cursor, [int(row["id"]) for row in rows]
                )
                sync_ranges = self._sync_ranges(cursor, account_id)
        except TransactionResultLimitExceeded:
            raise
        except Exception as exc:
            raise TransactionRepositoryUnavailable(
                "Transaction repository lifecycle query failed"
            ) from exc
        data = [
            self._transaction_model(
                row,
                items_by_parent.get(int(row["id"]), []),
                fees_by_parent.get(int(row["id"]), []),
            )
            for row in rows
        ]
        coverage = self._coverage(
            ("TRADE",), requested_start, requested_end, sync_ranges
        )
        # A durable transaction at or before the requested boundary is direct
        # evidence that this repository contains history covering that boundary,
        # even when an older sync-state watermark was initialized later.
        if any(item.transaction_at <= requested_start for item in data):
            remaining = [
                item for item in coverage.missing_intervals
                if not item.endswith(":feed_history_starts_after_request")
            ]
            coverage = coverage.model_copy(
                update={"missing_intervals": remaining, "complete": not remaining}
            )
        as_of = max(
            (item.ingested_at for item in data),
            default=self._now().astimezone(timezone.utc),
        )
        return data, coverage, as_of

    def last_sync_through(
        self, account_id: str, transaction_type: str
    ) -> datetime | None:
        with self.connection_factory.cursor(dictionary=True) as (_, cursor, _):
            rows = self._sync_ranges(cursor, account_id)
        row = next(
            (
                item
                for item in rows
                if str(item.get("transaction_type") or "") == transaction_type
            ),
            None,
        )
        return _api_utc_optional(row.get("synced_through_at")) if row else None

    def record_sync_success(
        self,
        account_id: str,
        transaction_type: str,
        start_at: datetime,
        through_at: datetime,
    ) -> None:
        start = _api_utc(start_at)
        through = _api_utc(through_at)
        if through <= start:
            raise ValueError("sync through_at must be after start_at")
        now = _api_utc(self._now())
        with self.connection_factory.transaction(dictionary=True) as (_, cursor, _):
            self._ensure_sync_state_table(cursor)
            cursor.execute(
                """
                INSERT INTO risk_transaction_sync_state (
                    account_id, transaction_type, first_synced_at,
                    synced_through_at, last_success_at
                ) VALUES (%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    first_synced_at=LEAST(first_synced_at,VALUES(first_synced_at)),
                    synced_through_at=GREATEST(synced_through_at,VALUES(synced_through_at)),
                    last_success_at=VALUES(last_success_at)
                """,
                (
                    account_id,
                    transaction_type,
                    _mysql_utc(start),
                    _mysql_utc(through),
                    _mysql_utc(now),
                ),
            )

    def upsert_chunk(
        self,
        transactions: Sequence[NormalizedTransaction],
    ) -> tuple[int, int]:
        inserted = updated = 0
        # Commit each normalized chunk with its child items and fee breakdowns.
        with self.connection_factory.transaction(dictionary=True) as (_, cursor, _):
            for transaction in transactions:
                cursor.execute(
                    "SELECT id FROM risk_transactions "
                    "WHERE account_id = %s AND broker_transaction_id = %s",
                    (transaction.account_id, transaction.broker_transaction_id),
                )
                existing_raw = cursor.fetchone()
                existing = row_dict(cursor, existing_raw) if existing_raw is not None else None
                cursor.execute(
                    """
                    INSERT INTO risk_transactions (
                        account_id, broker_transaction_id, identifier_method,
                        transaction_type, transaction_subtype, transaction_at,
                        trade_date, settlement_date, symbol, asset_type, description,
                        amount, fees, net_amount, currency, order_id, source_updated_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                    )
                    ON DUPLICATE KEY UPDATE
                        identifier_method=VALUES(identifier_method),
                        transaction_type=VALUES(transaction_type),
                        transaction_subtype=VALUES(transaction_subtype),
                        transaction_at=VALUES(transaction_at), trade_date=VALUES(trade_date),
                        settlement_date=VALUES(settlement_date), symbol=VALUES(symbol),
                        asset_type=VALUES(asset_type), description=VALUES(description),
                        amount=VALUES(amount), fees=VALUES(fees), net_amount=VALUES(net_amount),
                        currency=VALUES(currency), order_id=VALUES(order_id),
                        source_updated_at=VALUES(source_updated_at),
                        id=LAST_INSERT_ID(id)
                    """,
                    self._parent_values(transaction),
                )
                transaction_id = int(cursor.lastrowid)
                if existing is None:
                    inserted += 1
                else:
                    updated += 1
                cursor.execute(
                    "DELETE FROM risk_transaction_items WHERE transaction_id = %s",
                    (transaction_id,),
                )
                cursor.execute(
                    "DELETE FROM risk_transaction_fees WHERE transaction_id = %s",
                    (transaction_id,),
                )
                if transaction.items:
                    cursor.executemany(
                        """
                        INSERT INTO risk_transaction_items (
                            transaction_id, item_index, symbol, asset_type, instruction,
                            position_effect, quantity, price, amount, cost, currency
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        [
                            (
                                transaction_id,
                                item.item_index,
                                item.symbol,
                                item.asset_type,
                                item.instruction,
                                item.position_effect,
                                item.quantity,
                                item.price,
                                item.amount,
                                item.cost,
                                item.currency,
                            )
                            for item in transaction.items
                        ],
                    )
                if transaction.fees_by_type:
                    cursor.executemany(
                        """
                        INSERT INTO risk_transaction_fees (
                            transaction_id, fee_index, fee_type, amount
                        ) VALUES (%s,%s,%s,%s)
                        """,
                        [
                            (
                                transaction_id,
                                fee.fee_index,
                                fee.fee_type,
                                fee.amount,
                            )
                            for fee in transaction.fees_by_type
                        ],
                    )
        return inserted, updated

    @contextmanager
    def advisory_lock(self, account_id: str, wait_seconds: int = 0) -> Iterator[None]:
        # Use a bounded account digest for the per-account database lock name.
        digest = hashlib.sha256(account_id.encode("utf-8")).hexdigest()[:40]
        lock_name = f"risk-transactions:{digest}"
        with self.connection_factory.connection() as connection:
            cursor = connection.cursor(dictionary=True)
            acquired = False
            try:
                cursor.execute("SELECT GET_LOCK(%s, %s) AS acquired", (lock_name, wait_seconds))
                result = cursor.fetchone()
                acquired = int(row_dict(cursor, result).get("acquired") or 0) == 1
                if not acquired:
                    raise TransactionRepositoryUnavailable("Transaction sync lock is busy")
                yield
            finally:
                if acquired:
                    cursor.execute("SELECT RELEASE_LOCK(%s) AS released", (lock_name,))
                    cursor.fetchone()
                cursor.close()

    def _parent_values(self, item: NormalizedTransaction) -> tuple[Any, ...]:
        return (
            item.account_id,
            item.broker_transaction_id,
            item.identifier_method,
            item.transaction_type,
            item.transaction_subtype,
            _mysql_utc(item.transaction_at),
            item.trade_date,
            item.settlement_date,
            item.symbol,
            item.asset_type,
            item.description,
            item.amount,
            item.fees,
            item.net_amount,
            item.currency,
            item.order_id,
            _mysql_utc(item.source_updated_at) if item.source_updated_at else None,
        )

    def _load_items(self, cursor: Any, parent_ids: Sequence[int]) -> dict[int, list[TransactionItemModel]]:
        if not parent_ids:
            return {}
        grouped: dict[int, list[TransactionItemModel]] = {}
        batch_size = 1000
        for offset in range(0, len(parent_ids), batch_size):
            batch = parent_ids[offset : offset + batch_size]
            cursor.execute(
                "SELECT transaction_id, item_index, symbol, asset_type, instruction, "
                "position_effect, quantity, price, amount, cost, currency "
                "FROM risk_transaction_items WHERE transaction_id IN ("
                + ",".join("%s" for _ in batch)
                + ") ORDER BY transaction_id, item_index",
                tuple(batch),
            )
            for raw in cursor.fetchall():
                row = row_dict(cursor, raw)
                parent = int(row.pop("transaction_id"))
                grouped.setdefault(parent, []).append(
                    TransactionItemModel.model_validate(row)
                )
        return grouped

    def _load_fee_components(
        self, cursor: Any, parent_ids: Sequence[int]
    ) -> dict[int, list[TransactionFeeModel]]:
        if not parent_ids:
            return {}
        grouped: dict[int, list[TransactionFeeModel]] = {}
        batch_size = 1000
        for offset in range(0, len(parent_ids), batch_size):
            batch = parent_ids[offset : offset + batch_size]
            cursor.execute(
                "SELECT transaction_id, fee_index, fee_type, amount "
                "FROM risk_transaction_fees WHERE transaction_id IN ("
                + ",".join("%s" for _ in batch)
                + ") ORDER BY transaction_id, fee_index",
                tuple(batch),
            )
            for raw in cursor.fetchall():
                row = row_dict(cursor, raw)
                parent = int(row.pop("transaction_id"))
                grouped.setdefault(parent, []).append(
                    TransactionFeeModel.model_validate(row)
                )
        return grouped

    def _transaction_ranges(self, cursor: Any, account_id: str) -> list[dict[str, Any]]:
        cursor.execute(
            "SELECT transaction_type, MIN(transaction_at) AS first_transaction_at, "
            "MAX(transaction_at) AS last_transaction_at, MAX(ingested_at) AS last_ingested_at "
            "FROM risk_transactions WHERE account_id = %s GROUP BY transaction_type",
            (account_id,),
        )
        return [row_dict(cursor, raw) for raw in cursor.fetchall()]

    def _sync_ranges(self, cursor: Any, account_id: str) -> list[dict[str, Any]]:
        try:
            cursor.execute(
                "SELECT transaction_type, first_synced_at, synced_through_at, "
                "last_success_at FROM risk_transaction_sync_state WHERE account_id = %s",
                (account_id,),
            )
        except Exception as exc:
            if getattr(exc, "errno", None) == 1146 or "doesn't exist" in str(exc):
                return []
            raise
        return [row_dict(cursor, raw) for raw in cursor.fetchall()]

    @staticmethod
    def _ensure_sync_state_table(cursor: Any) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_transaction_sync_state (
                account_id VARCHAR(128) NOT NULL,
                transaction_type VARCHAR(64) NOT NULL,
                first_synced_at DATETIME(6) NOT NULL,
                synced_through_at DATETIME(6) NOT NULL,
                last_success_at DATETIME(6) NOT NULL,
                PRIMARY KEY (account_id, transaction_type),
                KEY ix_risk_transaction_sync_state_success (last_success_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
            """
        )

    def _coverage(
        self,
        transaction_types: Sequence[str],
        requested_start: datetime,
        requested_end: datetime,
        sync_ranges: Sequence[Mapping[str, Any]],
    ) -> TransactionCoverage:
        by_type = {
            str(row.get("transaction_type") or ""): row for row in sync_ranges
        }
        through_by_type: dict[str, datetime | None] = {}
        missing: list[str] = []
        # Measure feed completeness against the last required broker feed session.
        required_session = self._required_feed_session(requested_end)
        for transaction_type in transaction_types:
            row = by_type.get(transaction_type)
            first = _api_utc_optional(row.get("first_synced_at")) if row else None
            through = (
                _api_utc_optional(row.get("synced_through_at")) if row else None
            )
            through_by_type[transaction_type] = through
            # Older installations have no explicit feed state. Absence of a
            # transaction is not evidence of a sync gap, so coverage remains
            # unverified-but-nonwarning until the next successful feed records
            # its checked interval.
            if row is None:
                continue
            if first is None or first > requested_start:
                missing.append(f"{transaction_type}:feed_history_starts_after_request")
            if (
                through is None
                or through.astimezone(self._feed_timezone).date()
                < required_session
            ):
                missing.append(f"{transaction_type}:daily_feed_stale")
        return TransactionCoverage(
            requested_start_at=requested_start,
            requested_end_at=requested_end,
            sync_through_by_type=through_by_type,
            complete=not missing,
            missing_intervals=missing,
        )

    def _required_feed_session(self, requested_end: datetime) -> date:
        now_local = _api_utc(self._now()).astimezone(self._feed_timezone)
        requested_local = (
            _api_utc(requested_end) - timedelta(microseconds=1)
        ).astimezone(self._feed_timezone)
        session = min(requested_local.date(), now_local.date())
        if (
            session == now_local.date()
            and now_local.hour < self._daily_feed_complete_hour
        ):
            session -= timedelta(days=1)
        while session.weekday() >= 5:
            session -= timedelta(days=1)
        return session

    def _transaction_model(
        self,
        row: Mapping[str, Any],
        items: list[TransactionItemModel],
        fee_components: list[TransactionFeeModel],
    ) -> TransactionModel:
        return TransactionModel(
            id=int(row["id"]),
            broker_transaction_id=str(row["broker_transaction_id"]),
            transaction_at=_api_utc(row["transaction_at"]),
            transaction_type=str(row["transaction_type"]),
            transaction_subtype=_text(row.get("transaction_subtype")),
            description=_text(row.get("description")),
            symbol=_text(row.get("symbol")),
            asset_type=_text(row.get("asset_type")),
            trade_date=row.get("trade_date"),
            settlement_date=row.get("settlement_date"),
            amount=_decimal(row.get("amount")),
            fees=_decimal(row.get("fees")),
            net_amount=_decimal(row.get("net_amount")),
            currency=_text(row.get("currency")),
            order_id=_text(row.get("order_id")),
            source_updated_at=_api_utc_optional(row.get("source_updated_at")),
            ingested_at=_api_utc(row["ingested_at"]),
            items=items,
            fee_components=fee_components,
        )

    def _encode_cursor(self, at: datetime, row_id: int) -> str:
        body = json.dumps(
            {"at": at.astimezone(timezone.utc).isoformat(), "id": row_id},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        signature = hmac.new(self._cursor_secret, body, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(body + signature).decode("ascii").rstrip("=")

    def _decode_cursor(self, value: str) -> tuple[datetime, int]:
        try:
            padded = value + "=" * (-len(value) % 4)
            decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
            body, signature = decoded[:-32], decoded[-32:]
            expected = hmac.new(self._cursor_secret, body, hashlib.sha256).digest()
            # Verify cursor integrity before trusting its pagination fields.
            if not hmac.compare_digest(signature, expected):
                raise InvalidCursor("invalid cursor signature")
            payload = json.loads(body.decode("utf-8"))
            return _api_utc(datetime.fromisoformat(str(payload["at"]))), int(payload["id"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise InvalidCursor("invalid transaction cursor") from exc


def _require_interval(start_at: datetime, end_at: datetime) -> tuple[datetime, datetime]:
    start = _api_utc(start_at)
    end = _api_utc(end_at)
    if end <= start:
        raise ValueError("end_at must be after start_at")
    if end - start > timedelta(days=366):
        raise ValueError("transaction interval may not exceed 366 days")
    return start, end


def _mysql_utc(value: datetime) -> datetime:
    return _api_utc(value).replace(tzinfo=None)


def _api_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("datetime value required")
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _api_utc_optional(value: Any) -> datetime | None:
    return _api_utc(value) if isinstance(value, datetime) else None


def _decimal(value: Any) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def _text(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


__all__ = [
    "InvalidCursor",
    "TransactionRepository",
    "TransactionRepositoryUnavailable",
    "TransactionResultLimitExceeded",
]
