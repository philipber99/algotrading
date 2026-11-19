from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import math
from types import SimpleNamespace

import pytest

from ..clusters.hierarchical import cluster_risk_units
from ..exposures.factors import FACTOR_PROXY_SYMBOLS, ridge_factor_exposure
from ..exposures.portfolio import delta_equivalent_usd, portfolio_kpis
from ..instruments.terms import build_instrument_terms
from ..market_inputs.dividends import derive_trailing_cash_yield
from ..market_inputs.freshness import evaluate_quote_freshness
from ..market_inputs.rates import interpolate_risk_free_rate
from ..positions.normalization import normalize_broker_position
from ..scenarios.sector_rotation import build_sector_rotation_shocks
from ..tail.historical import historical_var_es


def test_signed_equity_and_option_exposure_units_reconcile():
    equity = normalize_broker_position(
        {"longQuantity": 0, "shortQuantity": 4, "averagePrice": 100,
         "instrument": {"symbol": "XYZ", "type": "EQUITY"}},
        account_id="TEST", quote={"mark": 110, "quote_time": "2026-08-17T12:00:00Z"},
    )
    option = normalize_broker_position(
        {"longQuantity": 2, "shortQuantity": 0,
         "instrument": {"symbol": "XYZ260918C00100000", "type": "OPTION"}},
        account_id="TEST",
        option_quote={"mark": 12, "underlying_price": 110},
        greek={"status": "ok", "delta": 0.6, "gamma": 0.01, "vega_per_1pct": 0.2,
               "theta_per_day": -0.05, "rho_per_1pct": 0.1, "implied_volatility": 0.3},
        instrument_terms=build_instrument_terms(
            "XYZ260918C00100000",
            broker_type="OPTION",
            option_terms={
                "terms_status": "standard",
                "price_multiplier": 100,
                "exercise_style": "american",
                "settlement_type": "physical",
                "asset_subtype": "equity_option",
            },
        ),
    )
    assert equity.signed_quantity == -4
    assert equity.signed_cost_basis_usd == -400
    assert equity.signed_market_value_usd == -440
    assert delta_equivalent_usd(equity) == -440
    assert delta_equivalent_usd(option) == pytest.approx(13_200)
    result = portfolio_kpis([equity, option], portfolio_value_usd=20_000)
    assert result.net_delta_exposure_usd == pytest.approx(12_760)
    assert result.gross_delta_exposure_usd == pytest.approx(13_640)
    assert result.options_vega_per_1pct_usd == pytest.approx(40)
    assert result.options_theta_per_day_usd == pytest.approx(-10)


def test_gics_consumer_staples_etf_maps_to_consumer_defensive():
    position = normalize_broker_position(
        {
            "longQuantity": 10,
            "averagePrice": 70,
            "instrument": {"symbol": "XLP", "type": "ETF"},
        },
        account_id="TEST",
        quote={"mark": 80, "quote_time": "2026-08-17T12:00:00Z"},
        classification={"taxonomy": "GICS", "sector": "Consumer Staples"},
    )

    assert position.sector == "Consumer Defensive"
    assert "classification_unavailable" not in position.data_quality.warnings


def test_trade_time_is_used_when_a_quote_time_is_unavailable():
    as_of = datetime(2026, 8, 17, 14, 1, tzinfo=timezone.utc)
    equity = normalize_broker_position(
        {"longQuantity": 1, "instrument": {"symbol": "XYZ", "type": "EQUITY"}},
        account_id="TEST",
        quote={"mark": 110, "trade_time": "2026-08-17T14:00:00Z"},
        as_of=as_of,
    )

    assert equity.quote_as_of == datetime(2026, 8, 17, 14, 0, tzinfo=timezone.utc)
    assert equity.field_validity["mark"].valid_for_risk is True
    assert "quote_not_valid_for_risk:quote_timestamp_missing" not in equity.data_quality.warnings


def test_quote_freshness_allows_only_bounded_snapshot_acquisition_skew():
    as_of = datetime(2026, 8, 28, 16, 11, 57, tzinfo=timezone.utc)

    during_snapshot = evaluate_quote_freshness(
        as_of + timedelta(seconds=4),
        as_of=as_of,
        context={},
        calendar_id="XNYS",
    )
    meaningfully_future = evaluate_quote_freshness(
        as_of + timedelta(seconds=6),
        as_of=as_of,
        context={},
        calendar_id="XNYS",
    )

    assert during_snapshot.valid_for_risk is True
    assert during_snapshot.age_seconds == 0
    assert meaningfully_future.valid_for_risk is False
    assert meaningfully_future.reason == "quote_timestamp_in_future"


def test_low_delta_option_quote_uses_four_hour_intraday_limit():
    as_of = datetime(2026, 8, 28, 16, 0, tzinfo=timezone.utc)
    quote_at = as_of - timedelta(minutes=44)

    low_delta = evaluate_quote_freshness(
        quote_at,
        as_of=as_of,
        context={},
        calendar_id="XNYS",
        option_delta=0.08,
    )
    near_money = evaluate_quote_freshness(
        quote_at,
        as_of=as_of,
        context={},
        calendar_id="XNYS",
        option_delta=0.40,
    )

    assert low_delta.valid_for_risk is True
    assert near_money.valid_for_risk is False
    assert near_money.reason == "intraday_quote_too_old"


def test_exchange_quote_is_valid_as_the_latest_completed_session_mark():
    equity = normalize_broker_position(
        {"longQuantity": 1, "instrument": {"symbol": "XYZ", "type": "EQUITY"}},
        account_id="TEST",
        quote={
            "mark": 110,
            "quote_time": "2026-08-17T19:59:23Z",
            "source_kind": "exchange_quote",
        },
        as_of=datetime(2026, 8, 17, 20, 30, tzinfo=timezone.utc),
    )

    assert equity.field_validity["mark"].valid_for_risk is True
    assert equity.field_validity["mark"].freshness_state == "valid_completed_session_eod"


def test_premarket_option_uses_completed_close_and_underlying_close_for_risk():
    option = normalize_broker_position(
        {
            "longQuantity": 2,
            "instrument": {
                "symbol": "SPY260918P00500000",
                "type": "OPTION",
            },
        },
        account_id="TEST",
        quote={
            "mark": 510.0,
            "quote_time": "2026-08-14T20:00:00Z",
            "source_kind": "completed_session_mark",
        },
        option_quote={
            "mark": 8.25,
            "quote_time": "2026-08-14T20:00:00Z",
            "source_kind": "completed_session_mark",
        },
        greek={
            "status": "ok",
            "method": "tian",
            "delta": -0.4,
            "gamma": 0.01,
            "vega_per_1pct": 0.2,
            "theta_per_day": -0.05,
            "rho_per_1pct": -0.1,
            "implied_volatility": 0.25,
        },
        instrument_terms=build_instrument_terms(
            "SPY260918P00500000",
            broker_type="OPTION",
            option_terms={
                "terms_status": "standard",
                "price_multiplier": 100,
                "shares_per_contract": 100,
                "exercise_style": "american",
                "settlement_type": "physical",
                "asset_subtype": "etf_option",
                "expiration_at": "2026-09-18T20:00:00Z",
            },
        ),
        as_of=datetime(2026, 8, 17, 13, 0, tzinfo=timezone.utc),
    )

    assert option.mark == pytest.approx(8.25)
    assert option.signed_market_value_usd == pytest.approx(1650)
    assert option.delta_exposure_usd == pytest.approx(-40_800)
    assert option.field_validity["mark"].valid_for_risk is True
    assert option.field_validity["delta"].valid_for_risk is True
    assert (
        option.field_validity["mark"].freshness_state
        == "valid_completed_session_eod"
    )


def test_standard_occ_option_uses_verified_trade_price_and_multiplier():
    option = normalize_broker_position(
        {"longQuantity": 2, "shortQuantity": 0, "averagePrice": 3.25,
         "instrument": {"symbol": "TSM260918P00500000", "type": "OPTION"}},
        account_id="TEST",
        option_quote={"mark": 4.0, "bid": 3.9, "ask": 4.1, "net_change": 0.25,
                      "mark_change": 0.2, "open_interest": 1234, "underlying_price": 450},
    )
    assert option.multiplier == 100
    assert option.multiplier_source == "standard_us_equity_occ_v1"
    assert option.normalized_average_price == pytest.approx(3.25)
    assert option.signed_cost_basis_usd == pytest.approx(650)
    assert option.raw_broker_fields["bid"] == pytest.approx(3.9)
    assert option.raw_broker_fields["ask"] == pytest.approx(4.1)
    assert option.raw_broker_fields["net_change"] == pytest.approx(0.25)
    assert option.raw_broker_fields["mark_change"] == pytest.approx(0.2)
    assert option.raw_broker_fields["open_interest"] == pytest.approx(1234)
    assert "average_price_units_unverified" not in option.data_quality.warnings
    assert "option_expiration_time_assumed_1600_new_york" not in option.data_quality.warnings


def test_index_option_keeps_expiration_time_assumption_warning():
    option = normalize_broker_position(
        {"longQuantity": 1, "shortQuantity": 0,
         "instrument": {"symbol": "SPX260918P05000000", "type": "INDEX_OPTION"}},
        account_id="TEST",
        option_quote={"mark": 10.0, "underlying_price": 5000},
    )
    assert "option_expiration_assumed_configured_standard_session" in option.data_quality.warnings


def test_reference_classification_can_supply_missing_stock_asset_type():
    equity = normalize_broker_position(
        {"longQuantity": 3, "averagePrice": 100,
         "instrument": {"symbol": "AAPL"}},
        account_id="TEST",
        quote={"mark": 110},
        classification={
            "symbol": "AAPL", "security_type": "common_stock",
            "sector": "Technology", "industry": "Consumer Electronics",
        },
    )
    assert equity.asset_type == "EQUITY"
    assert equity.multiplier == 1
    assert equity.sector == "Technology"
    assert equity.signed_market_value_usd == pytest.approx(330)
    assert "classification_unavailable" not in equity.data_quality.warnings
    assert "multiplier_unavailable" not in equity.data_quality.warnings


def test_broker_no_load_taxable_position_is_a_unit_multiplier_mutual_fund():
    fund = normalize_broker_position(
        {"longQuantity": 37126.15, "averagePrice": 1.0, "marketValue": 37126.15,
         "instrument": {"symbol": "SWVXX", "type": "NO_LOAD_TAXABLE"}},
        account_id="TEST",
        quote={"mark": 1.0},
    )
    assert fund.asset_type == "MUTUAL_FUND"
    assert fund.multiplier == 1
    assert fund.normalized_average_price == pytest.approx(1.0)
    assert fund.signed_market_value_usd == pytest.approx(37126.15)
    assert "multiplier_unavailable" not in fund.data_quality.warnings
    assert "average_price_units_unverified" not in fund.data_quality.warnings


def test_mutual_fund_uses_provider_mark_as_latest_available_daily_nav():
    fund = normalize_broker_position(
        {
            "longQuantity": 37126.15,
            "averagePrice": 1.0,
            "instrument": {"symbol": "SWVXX", "type": "NO_LOAD_TAXABLE"},
        },
        account_id="TEST",
        quote={
            "mark": 1.0,
            "quote_time": "2026-08-26T22:45:00Z",
            "source_kind": "exchange_quote",
        },
        as_of=datetime(2026, 8, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert fund.mark == 1.0
    assert fund.signed_market_value_usd == pytest.approx(37126.15)
    assert fund.field_validity["mark"].valid_for_risk is True
    assert (
        fund.field_validity["mark"].freshness_state
        == "valid_latest_available_nav"
    )
    assert "quote_not_valid_for_risk:mutual_fund_nav_too_old" not in (
        fund.data_quality.warnings
    )


def test_mutual_fund_nav_older_than_one_completed_session_fails_closed():
    fund = normalize_broker_position(
        {
            "longQuantity": 10,
            "instrument": {"symbol": "SWVXX", "type": "MUTUAL_FUND"},
        },
        account_id="TEST",
        quote={"mark": 1.0, "quote_time": "2026-08-25T22:45:00Z"},
        as_of=datetime(2026, 8, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert fund.field_validity["mark"].valid_for_risk is False
    assert "quote_not_valid_for_risk:mutual_fund_nav_too_old" in (
        fund.data_quality.warnings
    )


def test_curve_and_dividend_continuous_conversions():
    curve = {"curve_id": "test", "curve_version": "v1", "anchors": [
        {"series_id": "A", "tenor_days": 30, "observation_date": "2026-08-15", "simple_annual_rate_decimal": 0.04, "status": "ok"},
        {"series_id": "B", "tenor_days": 90, "observation_date": "2026-08-14", "simple_annual_rate_decimal": 0.05, "status": "ok"},
    ]}
    rate = interpolate_risk_free_rate(curve, valuation_date=date(2026, 8, 17), tenor_days=60)
    assert rate.simple_annual_rate_decimal == pytest.approx(0.045)
    assert rate.continuous_annual_rate_decimal == pytest.approx(math.log1p(0.045))
    dividend = derive_trailing_cash_yield(
        [{"ex_dividend_date": "2026-04-01", "amount": 1.0, "distribution_type": "cash"},
         {"ex_dividend_date": "2026-07-01", "amount": 1.0, "distribution_type": "cash"}],
        spot=100, valuation_date=date(2026, 8, 17),
    )
    assert dividend.simple_yield == pytest.approx(0.02)
    assert dividend.continuous_yield == pytest.approx(-math.log1p(-0.02))


def test_tail_contributions_reconcile_and_cluster_is_deterministic():
    rows = [[-100, -50], [20, 10], [-200, -100], [5, -5], [-50, -25]]
    risk = historical_var_es(rows, confidence=0.60, half_life_sessions=10, unit_ids=["a", "b"])
    assert risk["contribution_sum_usd"] == pytest.approx(risk["expected_shortfall_usd"])
    vectors = {"b": [-5000, 2000, -7000, 1000], "a": [-10000, 4000, -14000, 2000], "hedge": [10000, -4000, 14000, -2000]}
    first = cluster_risk_units(vectors, minimum_cluster_loss_usd=1)
    second = cluster_risk_units(dict(reversed(list(vectors.items()))), minimum_cluster_loss_usd=1)
    assert first == second
    assert first[0]["members"] == ["a", "b"]
    assert first[0]["effective_correlation"] == pytest.approx(1.0)


def test_weighted_historical_tail_rejects_full_confidence():
    rows = [[-10.0], [0.0], [5.0]]
    median = historical_var_es(rows, confidence=0.50, half_life_sessions=10, unit_ids=["a"])
    assert median["confidence"] == 0.50
    with pytest.raises(ValueError, match="less than one"):
        historical_var_es(rows, confidence=1.0, half_life_sessions=10, unit_ids=["a"])


def test_factor_exposure_is_pnl_for_one_percent_factor_move():
    observations = 140
    proxy = {name: [0.001 * ((i % 11) - 5) + j * 1e-5 for i in range(observations)] for j, name in enumerate(FACTOR_PROXY_SYMBOLS)}
    proxy["QQQ"] = [proxy["SPY"][i] + 0.002 * ((i % 7) - 3) for i in range(observations)]
    target = [2 * value for value in proxy["SPY"]]
    result = ridge_factor_exposure(target, proxy, current_delta_exposure_usd=1_000_000)
    assert result["status"] == "usable"
    market = next(item for item in result["factors"] if item["name"] == "Market")
    assert market["estimated_pnl_for_plus_1pct_factor_move_usd"] == pytest.approx(market["beta"] * 10_000)


def test_sector_rotation_combines_sector_beta_correlation_and_broad_shock():
    dates = [f"2026-{1 + index // 28:02d}-{1 + index % 28:02d}" for index in range(140)]
    technology = [0.01 if index % 2 else -0.01 for index in range(140)]
    defensive = [-value for value in technology]
    returns = {
        "XLK": dict(zip(dates, technology)),
        "XLP": dict(zip(dates, defensive)),
        "MSFT": dict(zip(dates, [0.5 * value for value in technology])),
        "PG": dict(zip(dates, [0.7 * value for value in defensive])),
    }
    positions = [
        SimpleNamespace(asset_type="EQUITY", underlying_symbol=None, exact_symbol="MSFT", sector="Technology"),
        SimpleNamespace(asset_type="OPTION", underlying_symbol="PG", exact_symbol="PG260918C00150000", sector="Consumer Defensive"),
    ]
    shocks, diagnostics = build_sector_rotation_shocks(
        positions,
        returns,
        selected_sector="Technology",
        sector_move=0.03,
        default_equity_shock=-0.01,
    )
    assert shocks["MSFT"] == pytest.approx(0.005)
    assert shocks["PG"] == pytest.approx(-0.031)
    assert diagnostics["unmodeled_symbols"] == []
    pg = next(item for item in diagnostics["positions"] if item["symbol"] == "PG")
    assert pg["sector_beta_1y"] == pytest.approx(0.7)
    assert pg["selected_sector_correlation_1y"] == pytest.approx(-1.0)
