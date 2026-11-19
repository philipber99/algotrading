"""Typed transaction-domain and public API contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SUPPORTED_TRANSACTION_TYPES = (
    "TRADE",
    "RECEIVE_AND_DELIVER",
    "DIVIDEND_OR_INTEREST",
    "ACH_RECEIPT",
    "ACH_DISBURSEMENT",
    "CASH_RECEIPT",
    "CASH_DISBURSEMENT",
    "ELECTRONIC_FUND",
    "WIRE_OUT",
    "WIRE_IN",
    "JOURNAL",
    "MEMORANDUM",
    "MARGIN_CALL",
    "MONEY_MARKET",
    "SMA_ADJUSTMENT",
)


@dataclass(frozen=True)
class NormalizedTransactionItem:
    item_index: int
    symbol: str | None
    asset_type: str | None
    instruction: str | None
    position_effect: str | None
    quantity: Decimal | None
    price: Decimal | None
    amount: Decimal | None
    cost: Decimal | None
    currency: str | None


@dataclass(frozen=True)
class NormalizedTransactionFee:
    """One broker-reported fee component from a transaction transfer item."""

    fee_index: int
    fee_type: str
    amount: Decimal


@dataclass(frozen=True)
class NormalizedTransaction:
    account_id: str
    broker_transaction_id: str
    identifier_method: str
    transaction_type: str
    transaction_subtype: str | None
    transaction_at: datetime
    trade_date: date | None
    settlement_date: date | None
    symbol: str | None
    asset_type: str | None
    description: str | None
    amount: Decimal | None
    fees: Decimal | None
    net_amount: Decimal | None
    currency: str | None
    order_id: str | None
    source_updated_at: datetime | None
    items: tuple[NormalizedTransactionItem, ...]
    fees_by_type: tuple[NormalizedTransactionFee, ...]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TransactionItemModel(ContractModel):
    item_index: int = Field(ge=0)
    symbol: str | None = None
    asset_type: str | None = None
    instruction: str | None = None
    position_effect: str | None = None
    quantity: Decimal | None = None
    price: Decimal | None = None
    amount: Decimal | None = None
    cost: Decimal | None = None
    currency: str | None = None


class TransactionFeeModel(ContractModel):
    """One non-zero broker fee component stored for a transaction."""

    fee_index: int = Field(ge=0)
    fee_type: str
    amount: Decimal


class TransactionModel(ContractModel):
    id: int
    broker_transaction_id: str
    transaction_at: datetime
    transaction_type: str
    transaction_subtype: str | None = None
    description: str | None = None
    symbol: str | None = None
    asset_type: str | None = None
    trade_date: date | None = None
    settlement_date: date | None = None
    amount: Decimal | None = None
    fees: Decimal | None = None
    net_amount: Decimal | None = None
    currency: str | None = None
    order_id: str | None = None
    source_updated_at: datetime | None = None
    ingested_at: datetime
    items: list[TransactionItemModel] = Field(default_factory=list)
    fee_components: list[TransactionFeeModel] = Field(default_factory=list)


class TransactionCoverage(ContractModel):
    requested_start_at: datetime
    requested_end_at: datetime
    sync_through_by_type: dict[str, datetime | None]
    complete: bool
    missing_intervals: list[str] = Field(default_factory=list)


class TransactionMeta(ContractModel):
    as_of: datetime
    next_cursor: str | None = None
    source: Literal["mysql_transaction_repository"] = "mysql_transaction_repository"
    coverage: TransactionCoverage
    warnings: list[str] = Field(default_factory=list)


class TransactionsResponse(ContractModel):
    schema_version: int = 1
    account_id: str
    data: list[TransactionModel]
    meta: TransactionMeta


class SyncStatusItem(ContractModel):
    transaction_type: str
    last_successful_start_at: datetime | None = None
    last_successful_end_at: datetime | None = None
    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    status: str
    last_run_id: str | None = None
    last_rows_received: int = 0
    last_rows_upserted: int = 0
    last_error_code: str | None = None
    last_error_message: str | None = None


class TransactionStatusResponse(ContractModel):
    schema_version: int = 1
    account_id: str
    data: list[SyncStatusItem]
    as_of: datetime
    source: Literal["mysql_transaction_repository"] = "mysql_transaction_repository"
    repository_status: Literal["available"] = "available"
