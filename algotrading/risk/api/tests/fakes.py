from __future__ import annotations

import copy
import threading
import time
from datetime import datetime, timezone
from typing import Any

from ..broker_source import SourceResponse

FIXED_NOW = datetime(2026, 8, 15, 18, 30, tzinfo=timezone.utc)
_DEFAULT = object()


def position(symbol: str = "FICTIONAL") -> dict[str, Any]:
    return {
        "shortQuantity": 0,
        "averagePrice": 12.5,
        "currentDayProfitLoss": -1.25,
        "currentDayProfitLossPercentage": -0.5,
        "longQuantity": 2,
        "settledLongQuantity": 2,
        "settledShortQuantity": 0,
        "agedQuantity": 0,
        "instrument": {
            "cusip": "000000000",
            "symbol": symbol,
            "description": "Fictional instrument",
            "instrumentId": 123,
            "netChange": 0,
            "type": "OPEN_BROKER_TYPE",
            "futureAdditiveField": "ignored",
        },
        "marketValue": 25,
        "maintenanceRequirement": 0,
        "averageLongPrice": 12.5,
        "averageShortPrice": 0,
        "taxLotAverageLongPrice": 12.5,
        "taxLotAverageShortPrice": 0,
        "longOpenProfitLoss": 0,
        "shortOpenProfitLoss": 0,
        "previousSessionLongQuantity": 2,
        "previousSessionShortQuantity": 0,
        "currentDayCost": 25,
        "futurePositionField": 999,
    }


def account(
    account_id: str = "FAKE-0001", *, include_positions: bool = True
) -> dict[str, Any]:
    securities_account: dict[str, Any] = {
        "accountNumber": account_id,
        "roundTrips": 0,
        "isDayTrader": False,
        "isClosingOnlyRestricted": False,
        "pfcbFlag": True,
        "initialBalances": {
            "accruedInterest": 0,
            "availableFundsNonMarginableTrade": 1,
            "bondValue": 2,
            "buyingPower": 3,
            "cashBalance": 4,
            "cashAvailableForTrading": 5,
            "cashReceipts": 6,
            "dayTradingBuyingPower": 7,
            "dayTradingBuyingPowerCall": 8,
            "dayTradingEquityCall": 9,
            "equity": 10,
            "equityPercentage": 11,
            "liquidationValue": 12,
            "longMarginValue": 13,
            "longOptionMarketValue": 14,
            "longStockValue": 15,
            "maintenanceCall": 16,
            "maintenanceRequirement": 17,
            "margin": 18,
            "marginEquity": 19,
            "moneyMarketFund": 20,
            "mutualFundValue": 21,
            "regTCall": 22,
            "shortMarginValue": 23,
            "shortOptionMarketValue": 24,
            "shortStockValue": 25,
            "totalCash": 26,
            "isInCall": False,
            "unsettledCash": 27,
            "pendingDeposits": 28,
            "marginBalance": 29,
            "shortBalance": 30,
            "accountValue": 31,
        },
        "currentBalances": {
            "availableFunds": 0,
            "availableFundsNonMarginableTrade": 1,
            "buyingPower": 2,
            "buyingPowerNonMarginableTrade": 3,
            "dayTradingBuyingPower": 4,
            "dayTradingBuyingPowerCall": 5,
            "equity": 6,
            "equityPercentage": 7,
            "longMarginValue": 8,
            "maintenanceCall": 9,
            "maintenanceRequirement": 10,
            "marginBalance": 11,
            "regTCall": 12,
            "shortBalance": 13,
            "shortMarginValue": 14,
            "sma": 15,
            "isInCall": True,
            "stockBuyingPower": 16,
            "optionBuyingPower": 17,
        },
        "projectedBalances": {
            "availableFunds": 100,
            "isInCall": False,
        },
        "futureAccountField": {"ignored": True},
    }
    if include_positions:
        securities_account["positions"] = [position()]
    return {"securitiesAccount": securities_account, "futureEnvelopeField": True}


class FakeSource:
    def __init__(
        self,
        *,
        all_payload: Any = _DEFAULT,
        one_payload: Any = _DEFAULT,
        delay_seconds: float = 0,
    ) -> None:
        self.all_payload = [account()] if all_payload is _DEFAULT else all_payload
        self.one_payload = account() if one_payload is _DEFAULT else one_payload
        self.delay_seconds = delay_seconds
        self.all_calls = 0
        self.one_calls: list[str] = []
        self.active_calls = 0
        self.maximum_active_calls = 0
        self._lock = threading.Lock()

    def fetch_all(self) -> SourceResponse:
        with self._lock:
            self.all_calls += 1
            self.active_calls += 1
            self.maximum_active_calls = max(
                self.maximum_active_calls, self.active_calls
            )
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            return SourceResponse(copy.deepcopy(self.all_payload), FIXED_NOW)
        finally:
            with self._lock:
                self.active_calls -= 1

    def fetch_one(self, account_id: str) -> SourceResponse:
        with self._lock:
            self.one_calls.append(account_id)
            self.active_calls += 1
            self.maximum_active_calls = max(
                self.maximum_active_calls, self.active_calls
            )
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            payload = self.one_payload
            if isinstance(payload, dict):
                payload = copy.deepcopy(payload)
                if "securitiesAccount" in payload:
                    payload["securitiesAccount"]["accountNumber"] = account_id
            return SourceResponse(payload, FIXED_NOW)
        finally:
            with self._lock:
                self.active_calls -= 1

    def readiness(self) -> dict[str, Any]:
        return {
            "broker_constructed": True,
            "required_methods_callable": True,
            "diagnostic": None,
            "last_account_request_at": None,
            "last_account_request_outcome": None,
        }
