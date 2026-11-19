from __future__ import annotations

from algotrading.risk.market_inputs.etf_exposure import (
    CURATED_ETF_SECTORS, classify_etf_exposure,
)


def exposure(*, asset_class="Equity", objective="", weights=()):
    return {
        "status": "available", "asset_class": asset_class,
        "objective": objective, "fund_name": "Test ETF",
        "sector_weights": [
            {"sector": sector, "weight_percent": weight}
            for sector, weight in weights
        ],
    }


def test_verified_fund_buckets_cover_the_current_portfolio():
    assert CURATED_ETF_SECTORS == {
        "CIBR": "Technology", "SOXL": "Technology",
        "HEWJ": "International Equity",
        "TLH": "U.S. Treasuries", "TLT": "U.S. Treasuries",
    }


def test_concentrated_sector_uses_economic_weight():
    data = exposure(asset_class="Sector Equity", weights=[
        ("Technology", 95.05), ("Industrials", 3.08),
        ("Communication Services", 1.87),
    ])
    assert classify_etf_exposure(data) == ("Technology", None)


def test_broad_international_uses_objective_and_diversification_not_country_wrapper():
    data = exposure(
        objective="Tracks an index of large and mid-sized Japanese companies",
        weights=[
            ("Industrials", 25), ("Financials", 20), ("Technology", 19),
            ("Consumer Cyclical", 15), ("Communication Services", 8),
            ("Healthcare", 7), ("Utilities", 6),
        ],
    )
    data["country_weights"] = [{"country": "United States", "weight_percent": 99}]
    assert classify_etf_exposure(data) == ("International Equity", None)


def test_fixed_income_and_treasury_objectives():
    assert classify_etf_exposure(exposure(
        asset_class="Fixed Income", objective="Tracks U.S. Treasury bonds",
    )) == ("U.S. Treasuries", None)
    assert classify_etf_exposure(exposure(asset_class="Fixed Income")) == ("Fixed Income", None)


def test_mixed_or_malformed_equity_remains_unclassified_with_reason():
    mixed = exposure(weights=[
        ("Technology", 40), ("Industrials", 35), ("Financials", 25),
    ])
    assert classify_etf_exposure(mixed) == ("Unclassified", "mixed_or_unclear_exposure")
    malformed = exposure(weights=[("Technology", float("nan"))])
    assert classify_etf_exposure(malformed) == ("Unclassified", "sector_weights_incomplete")
    assert classify_etf_exposure({"status": "unavailable"}) == ("Unclassified", "provider_unavailable")
