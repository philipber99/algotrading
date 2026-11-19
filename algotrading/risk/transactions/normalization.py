"""Normalize broker transaction dictionaries into durable transaction fields."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .models import (
    NormalizedTransaction,
    NormalizedTransactionFee,
    NormalizedTransactionItem,
    SUPPORTED_TRANSACTION_TYPES,
)


def normalize_transaction(
    raw: Mapping[str, Any],
    *,
    account_id: str,
    requested_type: str,
) -> NormalizedTransaction:
    if requested_type not in SUPPORTED_TRANSACTION_TYPES:
        raise ValueError("unsupported transaction type")
    if not isinstance(raw, Mapping):
        raise ValueError("transaction must be an object")
    identifier = _first_text(raw, "activityId", "transactionId", "id")
    if identifier is None:
        raise ValueError("broker transaction ID is required")
    transaction_type = (_first_text(raw, "type", "transactionType") or requested_type).upper()
    if transaction_type != requested_type:
        raise ValueError("broker transaction type differs from requested type")
    transaction_at = _aware_datetime(
        _first(raw, "transactionDate", "transactionAt", "time", "date")
    )
    transfer_items = raw.get("transferItems")
    if transfer_items is None:
        transfer_items = raw.get("transactionItem")
    if transfer_items is None:
        item_values: list[Any] = []
    elif isinstance(transfer_items, list):
        item_values = transfer_items
    elif isinstance(transfer_items, Mapping):
        item_values = [transfer_items]
    else:
        raise ValueError("transaction items must be an object or list")
    # Separate execution legs from fee entries before normalizing their fields.
    execution_item_values = [
        (index, item)
        for index, item in enumerate(item_values)
        if isinstance(item, Mapping) and _first_text(item, "feeType") is None
    ]
    items = tuple(
        _normalize_item(item, index)
        for index, item in execution_item_values
    )
    fee_components = tuple(
        _normalize_fee(item, index)
        for index, item in enumerate(item_values)
        if isinstance(item, Mapping) and _first_text(item, "feeType") is not None
    )
    fees_by_type = tuple(item for item in fee_components if item.amount != Decimal(0))
    parent_instrument = raw.get("instrument") if isinstance(raw.get("instrument"), Mapping) else {}
    first_item = execution_item_values[0][1] if execution_item_values else {}
    item_instrument = first_item.get("instrument") if isinstance(first_item.get("instrument"), Mapping) else {}
    fees = raw.get("fees")
    if isinstance(fees, Mapping):
        fees_value = sum(
            (value for value in (_decimal(item) for item in fees.values()) if value is not None),
            Decimal(0),
        )
    else:
        fees_value = _decimal(fees)
    if fees_value is None and fees_by_type:
        # The broker reports component fee amounts as positive charges while
        # the existing parent ``fees`` field represents signed cash effect.
        fees_value = -sum((item.amount for item in fees_by_type), Decimal(0))
    amount = _decimal(_first(raw, "amount", "cashBalanceEffect"))
    net_amount = _decimal(_first(raw, "netAmount", "netCashAmount"))
    # Derive missing net cash only when both gross amount and fees are known.
    if net_amount is None and amount is not None and fees_value is not None:
        net_amount = amount + fees_value
    return NormalizedTransaction(
        account_id=account_id,
        broker_transaction_id=identifier,
        identifier_method="broker_id",
        transaction_type=transaction_type,
        transaction_subtype=_first_text(raw, "subType", "transactionSubType"),
        transaction_at=transaction_at,
        trade_date=_date(_first(raw, "tradeDate")),
        settlement_date=_date(_first(raw, "settlementDate")),
        symbol=_normalize_symbol(
            _first_text(raw, "symbol")
            or _first_text(parent_instrument, "symbol")
            or _first_text(item_instrument, "symbol")
        ),
        asset_type=(
            _first_text(raw, "assetType")
            or _first_text(parent_instrument, "assetType")
            or _first_text(item_instrument, "assetType")
        ),
        description=_first_text(raw, "description"),
        amount=amount,
        fees=fees_value,
        net_amount=net_amount,
        currency=(_first_text(raw, "currency") or "USD").upper(),
        order_id=_first_text(raw, "orderId"),
        source_updated_at=_optional_datetime(_first(raw, "updatedAt", "sourceUpdatedAt")),
        items=items,
        fees_by_type=fees_by_type,
    )


def _normalize_item(raw: Mapping[str, Any], index: int) -> NormalizedTransactionItem:
    instrument = raw.get("instrument") if isinstance(raw.get("instrument"), Mapping) else {}
    return NormalizedTransactionItem(
        item_index=index,
        symbol=_normalize_symbol(_first_text(raw, "symbol") or _first_text(instrument, "symbol")),
        asset_type=_first_text(raw, "assetType") or _first_text(instrument, "assetType"),
        instruction=_first_text(raw, "instruction"),
        position_effect=_first_text(raw, "positionEffect"),
        quantity=_decimal(_first(raw, "amount", "quantity")),
        price=_decimal(_first(raw, "price")),
        amount=_decimal(_first(raw, "cashAmount", "netAmount")),
        cost=_decimal(_first(raw, "cost")),
        currency=(_first_text(raw, "currency") or "USD").upper(),
    )


def _normalize_fee(raw: Mapping[str, Any], index: int) -> NormalizedTransactionFee:
    fee_type = _first_text(raw, "feeType")
    amount = _decimal(_first(raw, "amount", "feeAmount"))
    if fee_type is None or amount is None:
        raise ValueError("fee transaction item requires a fee type and numeric amount")
    return NormalizedTransactionFee(
        fee_index=index,
        fee_type=fee_type.upper(),
        amount=amount,
    )


def _first(raw: Mapping[str, Any], *keys: str) -> Any:
    return next((raw[key] for key in keys if key in raw and raw[key] is not None), None)


def _first_text(raw: Mapping[str, Any], *keys: str) -> str | None:
    value = _first(raw, *keys)
    return str(value).strip() if value not in (None, "") else None


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return converted if converted.is_finite() else None


def _aware_datetime(value: Any) -> datetime:
    parsed = _optional_datetime(value)
    if parsed is None:
        raise ValueError("transaction timestamp is required")
    return parsed


def _optional_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        # Normalize provider epoch timestamps supplied in seconds or milliseconds.
        seconds = float(value) / 1000 if abs(float(value)) > 10_000_000_000 else float(value)
        parsed = datetime.fromtimestamp(seconds, timezone.utc)
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("transaction timestamp must be RFC 3339") from exc
    if parsed.tzinfo is None:
        raise ValueError("transaction timestamp must be timezone aware")
    return parsed.astimezone(timezone.utc)


def _date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[0:10])
    except ValueError:
        return None


def _normalize_symbol(value: str | None) -> str | None:
    return value.strip().upper() if value else None
