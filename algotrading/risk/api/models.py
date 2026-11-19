"""Typed public contracts for accounts, positions, metadata, and errors."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InitialBalances(ContractModel):
    accrued_interest: float | None = None
    available_funds_non_marginable_trade: float | None = None
    bond_value: float | None = None
    buying_power: float | None = None
    cash_balance: float | None = None
    cash_available_for_trading: float | None = None
    cash_receipts: float | None = None
    day_trading_buying_power: float | None = None
    day_trading_buying_power_call: float | None = None
    day_trading_equity_call: float | None = None
    equity: float | None = None
    equity_percentage: float | None = None
    liquidation_value: float | None = None
    long_margin_value: float | None = None
    long_option_market_value: float | None = None
    long_stock_value: float | None = None
    maintenance_call: float | None = None
    maintenance_requirement: float | None = None
    margin: float | None = None
    margin_equity: float | None = None
    money_market_fund: float | None = None
    mutual_fund_value: float | None = None
    reg_t_call: float | None = None
    short_margin_value: float | None = None
    short_option_market_value: float | None = None
    short_stock_value: float | None = None
    total_cash: float | None = None
    is_in_call: bool | None = None
    unsettled_cash: float | None = None
    pending_deposits: float | None = None
    margin_balance: float | None = None
    short_balance: float | None = None
    account_value: float | None = None


class CurrentOrProjectedBalances(ContractModel):
    available_funds: float | None = None
    available_funds_non_marginable_trade: float | None = None
    buying_power: float | None = None
    buying_power_non_marginable_trade: float | None = None
    day_trading_buying_power: float | None = None
    day_trading_buying_power_call: float | None = None
    equity: float | None = None
    equity_percentage: float | None = None
    long_margin_value: float | None = None
    maintenance_call: float | None = None
    maintenance_requirement: float | None = None
    margin_balance: float | None = None
    reg_t_call: float | None = None
    short_balance: float | None = None
    short_margin_value: float | None = None
    sma: float | None = None
    is_in_call: bool | None = None
    stock_buying_power: float | None = None
    option_buying_power: float | None = None


class Instrument(ContractModel):
    cusip: str | None = None
    symbol: str | None = None
    description: str | None = None
    instrument_id: int | float | None = None
    net_change: float | None = None
    type: str | None = None


class Position(ContractModel):
    account_id: str
    short_quantity: float | None = None
    average_price: float | None = None
    current_day_profit_loss: float | None = None
    current_day_profit_loss_percentage: float | None = None
    long_quantity: float | None = None
    settled_long_quantity: float | None = None
    settled_short_quantity: float | None = None
    aged_quantity: float | None = None
    instrument: Instrument
    market_value: float | None = None
    maintenance_requirement: float | None = None
    average_long_price: float | None = None
    average_short_price: float | None = None
    tax_lot_average_long_price: float | None = None
    tax_lot_average_short_price: float | None = None
    long_open_profit_loss: float | None = None
    short_open_profit_loss: float | None = None
    previous_session_long_quantity: float | None = None
    previous_session_short_quantity: float | None = None
    current_day_cost: float | None = None


class AccountSummary(ContractModel):
    account_id: str
    round_trips: int | None = None
    is_day_trader: bool | None = None
    is_closing_only_restricted: bool | None = None
    pfcb_flag: bool | None = None
    position_count: int = Field(ge=0)
    initial_balances: InitialBalances | None = None
    current_balances: CurrentOrProjectedBalances | None = None
    projected_balances: CurrentOrProjectedBalances | None = None


class Account(AccountSummary):
    positions: list[Position]


class ResponseMeta(ContractModel):
    count: int = Field(ge=0)
    as_of: datetime
    source: Literal["TDABrokerInterface"] = "TDABrokerInterface"
    cache: Literal["hit", "miss", "bypass"]
    request_id: str


class AccountsResponse(ContractModel):
    data: list[Account | AccountSummary]
    meta: ResponseMeta


class AccountResponse(ContractModel):
    data: Account | AccountSummary
    meta: ResponseMeta


class PositionsResponse(ContractModel):
    data: list[Position]
    meta: ResponseMeta


class LiveHealthResponse(ContractModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class ReadyHealthResponse(ContractModel):
    status: Literal["ready", "not_ready"]
    service: str
    version: str
    configuration_loaded: bool
    broker_constructed: bool
    required_methods_callable: bool
    diagnostic: str | None = None
    last_account_request_at: datetime | None = None
    last_account_request_outcome: Literal["success", "failure"] | None = None


class ErrorDetail(ContractModel):
    code: str
    message: str
    request_id: str
    details: None = None


class ErrorResponse(ContractModel):
    error: ErrorDetail
