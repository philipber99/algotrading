from __future__ import annotations

import copy

import pytest

from ..errors import AccountNotFoundError, UpstreamSchemaError
from ..normalization import normalize_accounts, normalize_specific_account
from .fakes import account


def test_all_documented_fields_normalize_and_unknown_fields_are_ignored():
    raw = account()
    untouched = copy.deepcopy(raw)

    result = normalize_accounts([raw])[0]

    assert raw == untouched
    assert result.account_id == "FAKE-0001"
    assert result.round_trips == 0
    assert result.is_day_trader is False
    assert result.is_closing_only_restricted is False
    assert result.pfcb_flag is True
    assert result.position_count == 1
    assert set(result.initial_balances.model_dump()) == {
        "accrued_interest",
        "available_funds_non_marginable_trade",
        "bond_value",
        "buying_power",
        "cash_balance",
        "cash_available_for_trading",
        "cash_receipts",
        "day_trading_buying_power",
        "day_trading_buying_power_call",
        "day_trading_equity_call",
        "equity",
        "equity_percentage",
        "liquidation_value",
        "long_margin_value",
        "long_option_market_value",
        "long_stock_value",
        "maintenance_call",
        "maintenance_requirement",
        "margin",
        "margin_equity",
        "money_market_fund",
        "mutual_fund_value",
        "reg_t_call",
        "short_margin_value",
        "short_option_market_value",
        "short_stock_value",
        "total_cash",
        "is_in_call",
        "unsettled_cash",
        "pending_deposits",
        "margin_balance",
        "short_balance",
        "account_value",
    }
    assert result.initial_balances.accrued_interest == 0
    assert result.initial_balances.account_value == 31
    assert result.initial_balances.is_in_call is False
    assert set(result.current_balances.model_dump()) == {
        "available_funds",
        "available_funds_non_marginable_trade",
        "buying_power",
        "buying_power_non_marginable_trade",
        "day_trading_buying_power",
        "day_trading_buying_power_call",
        "equity",
        "equity_percentage",
        "long_margin_value",
        "maintenance_call",
        "maintenance_requirement",
        "margin_balance",
        "reg_t_call",
        "short_balance",
        "short_margin_value",
        "sma",
        "is_in_call",
        "stock_buying_power",
        "option_buying_power",
    }
    assert result.current_balances.available_funds == 0
    assert result.current_balances.option_buying_power == 17
    assert result.projected_balances.available_funds == 100
    assert result.projected_balances.buying_power is None

    normalized_position = result.positions[0]
    assert set(normalized_position.model_dump()) == {
        "account_id",
        "short_quantity",
        "average_price",
        "current_day_profit_loss",
        "current_day_profit_loss_percentage",
        "long_quantity",
        "settled_long_quantity",
        "settled_short_quantity",
        "aged_quantity",
        "instrument",
        "market_value",
        "maintenance_requirement",
        "average_long_price",
        "average_short_price",
        "tax_lot_average_long_price",
        "tax_lot_average_short_price",
        "long_open_profit_loss",
        "short_open_profit_loss",
        "previous_session_long_quantity",
        "previous_session_short_quantity",
        "current_day_cost",
    }
    assert normalized_position.account_id == "FAKE-0001"
    assert normalized_position.short_quantity == 0
    assert normalized_position.current_day_cost == 25
    assert normalized_position.instrument.model_dump() == {
        "cusip": "000000000",
        "symbol": "FICTIONAL",
        "description": "Fictional instrument",
        "instrument_id": 123,
        "net_change": 0.0,
        "type": "OPEN_BROKER_TYPE",
    }


def test_missing_positions_is_an_empty_collection_and_missing_scalars_are_null():
    result = normalize_specific_account(account(include_positions=False))
    assert result.position_count == 0
    assert result.positions == []

    raw = account()
    del raw["securitiesAccount"]["initialBalances"]["cashBalance"]
    result = normalize_specific_account(raw)
    assert result.initial_balances.cash_balance is None


def test_schwab_asset_type_is_preserved_when_legacy_type_is_absent():
    raw = account()
    instrument = raw["securitiesAccount"]["positions"][0]["instrument"]
    instrument.pop("type")
    instrument["assetType"] = "EQUITY"

    result = normalize_specific_account(raw)

    assert result.positions[0].instrument.type == "EQUITY"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        [],
        {"securitiesAccount": []},
        {"securitiesAccount": {"positions": []}},
        {"securitiesAccount": {"accountNumber": "X", "positions": {}}},
        {
            "securitiesAccount": {
                "accountNumber": "X",
                "initialBalances": [],
            }
        },
        {
            "securitiesAccount": {
                "accountNumber": "X",
                "positions": [{"instrument": []}],
            }
        },
    ],
)
def test_malformed_specific_response_raises_schema_error(raw):
    with pytest.raises(UpstreamSchemaError):
        normalize_specific_account(raw)


def test_non_numeric_text_is_rejected_instead_of_coerced():
    raw = account()
    raw["securitiesAccount"]["currentBalances"]["equity"] = "100.00"
    with pytest.raises(UpstreamSchemaError):
        normalize_specific_account(raw)


def test_only_explicit_http_404_is_account_not_found():
    with pytest.raises(AccountNotFoundError):
        normalize_specific_account(404)
    with pytest.raises(UpstreamSchemaError):
        normalize_specific_account(None)


def test_all_accounts_requires_a_list():
    with pytest.raises(UpstreamSchemaError):
        normalize_accounts(account())
