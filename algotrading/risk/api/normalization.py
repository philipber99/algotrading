"""Pure normalization of broker account dictionaries into public contracts."""

from __future__ import annotations

import math
from typing import Any

from .errors import AccountNotFoundError, UpstreamSchemaError
from .models import (
    Account,
    CurrentOrProjectedBalances,
    InitialBalances,
    Instrument,
    Position,
)


INITIAL_BALANCE_FIELDS = {
    "accrued_interest": "accruedInterest",
    "available_funds_non_marginable_trade": "availableFundsNonMarginableTrade",
    "bond_value": "bondValue",
    "buying_power": "buyingPower",
    "cash_balance": "cashBalance",
    "cash_available_for_trading": "cashAvailableForTrading",
    "cash_receipts": "cashReceipts",
    "day_trading_buying_power": "dayTradingBuyingPower",
    "day_trading_buying_power_call": "dayTradingBuyingPowerCall",
    "day_trading_equity_call": "dayTradingEquityCall",
    "equity": "equity",
    "equity_percentage": "equityPercentage",
    "liquidation_value": "liquidationValue",
    "long_margin_value": "longMarginValue",
    "long_option_market_value": "longOptionMarketValue",
    "long_stock_value": "longStockValue",
    "maintenance_call": "maintenanceCall",
    "maintenance_requirement": "maintenanceRequirement",
    "margin": "margin",
    "margin_equity": "marginEquity",
    "money_market_fund": "moneyMarketFund",
    "mutual_fund_value": "mutualFundValue",
    "reg_t_call": "regTCall",
    "short_margin_value": "shortMarginValue",
    "short_option_market_value": "shortOptionMarketValue",
    "short_stock_value": "shortStockValue",
    "total_cash": "totalCash",
    "unsettled_cash": "unsettledCash",
    "pending_deposits": "pendingDeposits",
    "margin_balance": "marginBalance",
    "short_balance": "shortBalance",
    "account_value": "accountValue",
}

CURRENT_BALANCE_FIELDS = {
    "available_funds": "availableFunds",
    "available_funds_non_marginable_trade": "availableFundsNonMarginableTrade",
    "buying_power": "buyingPower",
    "buying_power_non_marginable_trade": "buyingPowerNonMarginableTrade",
    "day_trading_buying_power": "dayTradingBuyingPower",
    "day_trading_buying_power_call": "dayTradingBuyingPowerCall",
    "equity": "equity",
    "equity_percentage": "equityPercentage",
    "long_margin_value": "longMarginValue",
    "maintenance_call": "maintenanceCall",
    "maintenance_requirement": "maintenanceRequirement",
    "margin_balance": "marginBalance",
    "reg_t_call": "regTCall",
    "short_balance": "shortBalance",
    "short_margin_value": "shortMarginValue",
    "sma": "sma",
    "stock_buying_power": "stockBuyingPower",
    "option_buying_power": "optionBuyingPower",
}

POSITION_FIELDS = {
    "short_quantity": "shortQuantity",
    "average_price": "averagePrice",
    "current_day_profit_loss": "currentDayProfitLoss",
    "current_day_profit_loss_percentage": "currentDayProfitLossPercentage",
    "long_quantity": "longQuantity",
    "settled_long_quantity": "settledLongQuantity",
    "settled_short_quantity": "settledShortQuantity",
    "aged_quantity": "agedQuantity",
    "market_value": "marketValue",
    "maintenance_requirement": "maintenanceRequirement",
    "average_long_price": "averageLongPrice",
    "average_short_price": "averageShortPrice",
    "tax_lot_average_long_price": "taxLotAverageLongPrice",
    "tax_lot_average_short_price": "taxLotAverageShortPrice",
    "long_open_profit_loss": "longOpenProfitLoss",
    "short_open_profit_loss": "shortOpenProfitLoss",
    "previous_session_long_quantity": "previousSessionLongQuantity",
    "previous_session_short_quantity": "previousSessionShortQuantity",
    "current_day_cost": "currentDayCost",
}


def normalize_accounts(raw: Any) -> list[Account]:
    """Normalize the documented list response from get_accounts()."""

    if not isinstance(raw, list):
        raise UpstreamSchemaError()
    return [_normalize_envelope(item) for item in raw]


def normalize_specific_account(raw: Any) -> Account:
    """Normalize get_account(), recognizing only an explicit HTTP 404."""

    if type(raw) is int and raw == 404:
        raise AccountNotFoundError()
    return _normalize_envelope(raw)


def _normalize_envelope(raw: Any) -> Account:
    if not isinstance(raw, dict):
        raise UpstreamSchemaError()
    securities_account = raw.get("securitiesAccount")
    if not isinstance(securities_account, dict):
        raise UpstreamSchemaError()
    return _normalize_account(securities_account)


def _normalize_account(raw: dict[str, Any]) -> Account:
    account_id = raw.get("accountNumber")
    if not isinstance(account_id, str) or not account_id:
        raise UpstreamSchemaError()

    positions_raw = raw.get("positions", [])
    if not isinstance(positions_raw, list):
        raise UpstreamSchemaError()
    # Project broker dictionaries into the explicit public account and position fields.
    positions = [_normalize_position(item, account_id) for item in positions_raw]

    return Account(
        account_id=account_id,
        round_trips=_optional_integer(raw, "roundTrips"),
        is_day_trader=_optional_boolean(raw, "isDayTrader"),
        is_closing_only_restricted=_optional_boolean(raw, "isClosingOnlyRestricted"),
        pfcb_flag=_optional_boolean(raw, "pfcbFlag"),
        position_count=len(positions_raw),
        positions=positions,
        initial_balances=_initial_balances(raw),
        current_balances=_current_balances(raw, "currentBalances"),
        projected_balances=_current_balances(raw, "projectedBalances"),
    )


def _initial_balances(raw: dict[str, Any]) -> InitialBalances | None:
    container = _optional_container(raw, "initialBalances")
    if container is None:
        return None
    values = {
        public: _optional_number(container, upstream)
        for public, upstream in INITIAL_BALANCE_FIELDS.items()
    }
    values["is_in_call"] = _optional_boolean(container, "isInCall")
    return InitialBalances(**values)


def _current_balances(
    raw: dict[str, Any], key: str
) -> CurrentOrProjectedBalances | None:
    container = _optional_container(raw, key)
    if container is None:
        return None
    values = {
        public: _optional_number(container, upstream)
        for public, upstream in CURRENT_BALANCE_FIELDS.items()
    }
    values["is_in_call"] = _optional_boolean(container, "isInCall")
    return CurrentOrProjectedBalances(**values)


def _normalize_position(raw: Any, account_id: str) -> Position:
    if not isinstance(raw, dict):
        raise UpstreamSchemaError()
    instrument_raw = raw.get("instrument")
    if not isinstance(instrument_raw, dict):
        raise UpstreamSchemaError()

    values = {
        public: _optional_number(raw, upstream)
        for public, upstream in POSITION_FIELDS.items()
    }
    return Position(
        account_id=account_id,
        instrument=_normalize_instrument(instrument_raw),
        **values,
    )


def _normalize_instrument(raw: dict[str, Any]) -> Instrument:
    return Instrument(
        cusip=_optional_text(raw, "cusip"),
        symbol=_optional_text(raw, "symbol"),
        description=_optional_text(raw, "description"),
        instrument_id=_optional_numeric_identifier(raw, "instrumentId"),
        net_change=_optional_number(raw, "netChange"),
        # Schwab account payloads use ``assetType`` while older TDA fixtures
        # used ``type``.  Preserve either spelling at this boundary so the
        # portfolio normalizer does not have to infer ordinary stocks.
        type=_optional_text(raw, "type") or _optional_text(raw, "assetType"),
    )


def _optional_container(raw: dict[str, Any], key: str) -> dict[str, Any] | None:
    if key not in raw:
        return None
    value = raw[key]
    if not isinstance(value, dict):
        raise UpstreamSchemaError()
    return value


def _optional_number(raw: dict[str, Any], key: str) -> float | None:
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise UpstreamSchemaError()
    return float(value)


def _optional_integer(raw: dict[str, Any], key: str) -> int | None:
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise UpstreamSchemaError()
    if int(value) != value:
        raise UpstreamSchemaError()
    return int(value)


def _optional_numeric_identifier(raw: dict[str, Any], key: str) -> int | float | None:
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise UpstreamSchemaError()
    return value


def _optional_boolean(raw: dict[str, Any], key: str) -> bool | None:
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if not isinstance(value, bool):
        raise UpstreamSchemaError()
    return value


def _optional_text(raw: dict[str, Any], key: str) -> str | None:
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if not isinstance(value, str):
        raise UpstreamSchemaError()
    return value
