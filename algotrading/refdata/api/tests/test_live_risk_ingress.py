from __future__ import annotations

from datetime import date, datetime, timezone
import json
from urllib.parse import parse_qs, urlparse

import pytest

from ..live_models import OptionChainRequest, OptionExpirationRequest
from ..option_symbols import build_occ_symbol, parse_occ_symbol
from ..reference_ingress import CmeReferenceDataService, FredRiskFreeCurveService, LiveOptionService


NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


class _Provider:
    def __init__(self):
        self.quote_calls = []
        self.chain_calls = []
        self.expiration_calls = []

    def get_options_quotes(self, symbols):
        self.quote_calls.append(symbols)
        return {
            "SPY   260918P00500000": {
                "bid": 9.0, "ask": 11.0, "mark": 10.0, "last": 9.5,
                "volatility": 24.5, "delta": -0.4, "gamma": 0.01,
                "theta": -0.05, "vega": 0.12, "rho": -0.03,
                "underlying_price": 510.0, "quote_time": NOW.isoformat(),
            }
        }

    def get_options_chain(self, symbol, **kwargs):
        self.chain_calls.append((symbol, kwargs))
        contract = {
            "symbol": "SPY   260918P00500000", "strikePrice": 500,
            "bid": 9, "ask": 11, "mark": 10, "volatility": 24.5,
            "exerciseType": "A", "settlementType": "P", "multiplier": 100,
        }
        return {"putExpDateMap": {"2026-09-18:32": {"500.0": [contract]}}}

    def get_option_expiration_chain(self, symbol):
        self.expiration_calls.append(symbol)
        return {"expirationList": [
            {"expirationDate": "2026-09-18", "daysToExpiration": 32, "expirationType": "S", "settlementType": "P", "optionRoots": "SPY", "standard": True},
            {"expirationDate": "2027-03-19", "daysToExpiration": 214, "expirationType": "S", "settlementType": "P", "optionRoots": "SPY", "standard": True},
        ]}


def test_occ_parser_accepts_compact_and_padded_roots_and_round_trips():
    for root in ("F", "SPY", "BRK.B", "ABC123"):
        built = build_occ_symbol(root, date(2026, 9, 18), "PUT", 500)
        assert parse_occ_symbol(built.canonical) == built
        assert parse_occ_symbol(built.provider) == built
        assert len(built.provider) == 21


def test_exact_option_quotes_batch_once_preserves_partial_results_and_units():
    provider = _Provider()
    service = LiveOptionService({}, lambda: provider, now_fn=lambda: NOW)
    result = service.option_quotes(["SPY260918P00500000", "QQQ260918C00500000"])
    again = service.option_quotes(["SPY260918P00500000", "QQQ260918C00500000"])

    assert provider.quote_calls == [["SPY   260918P00500000", "QQQ   260918C00500000"]]
    assert result.data[0].status == "ok"
    assert result.data[0].provider_iv_raw == 24.5
    assert result.data[0].provider_iv == pytest.approx(0.245)
    assert result.data[0].quote_time == NOW
    assert result.data[0].source_kind == "exchange_quote"
    assert result.data[0].source_units["provider_vega_per_1pct"] == "option_price_per_1_volatility_point"
    assert result.data[1].status == "missing"
    assert result.missing_symbols == ["QQQ260918C00500000"]
    assert again.meta.cache == "hit"


def test_exact_option_quotes_reject_non_positive_provider_iv():
    provider = _Provider()
    provider.get_options_quotes = lambda _symbols: {
        "SPY   260918P00500000": {"volatility": -9.99}
    }
    result = LiveOptionService({}, lambda: provider, now_fn=lambda: NOW).option_quotes(
        ["SPY260918P00500000"]
    )

    quote = result.data[0]
    assert quote.provider_iv_raw == -9.99
    assert quote.provider_iv is None
    assert "provider_iv_unusable" in quote.warnings


def test_premarket_option_quotes_use_previous_completed_session_close():
    provider = _Provider()
    provider.get_options_quotes = lambda _symbols: {
        "SPY   260918P00500000": {
            "bid": 9.0,
            "ask": 11.0,
            "mark": 10.0,
            "last": 9.5,
            "close": 8.25,
            "underlying_price": 512.0,
            "volatility": 24.5,
            "delta": -0.4,
            "quote_time": 0,
            "trade_time": 0,
        }
    }
    completed_close = datetime(2026, 8, 14, 20, 0, tzinfo=timezone.utc)
    service = LiveOptionService(
        {},
        lambda: provider,
        now_fn=lambda: NOW,
        market_open_fn=lambda: False,
        latest_completed_session_close_fn=lambda: completed_close,
    )

    quote = service.option_quotes(["SPY260918P00500000"]).data[0]

    assert quote.mark == 8.25
    assert quote.close == 8.25
    assert quote.bid is quote.ask is quote.last is None
    assert quote.quote_time == completed_close
    assert quote.trade_time is None
    assert quote.source_kind == "completed_session_mark"
    assert quote.underlying_price is None
    assert quote.provider_iv is None
    assert quote.provider_delta is None
    assert "previous_completed_session_close_used" in quote.warnings


def test_open_market_option_quotes_keep_live_fields_and_reject_zero_timestamp():
    provider = _Provider()
    provider.get_options_quotes = lambda _symbols: {
        "SPY   260918P00500000": {
            "bid": 9.0,
            "ask": 11.0,
            "mark": 10.0,
            "close": 8.25,
            "volatility": 24.5,
            "delta": -0.4,
            "quote_time": 0,
        }
    }
    service = LiveOptionService(
        {},
        lambda: provider,
        now_fn=lambda: NOW,
        market_open_fn=lambda: True,
        latest_completed_session_close_fn=lambda: datetime(
            2026, 8, 14, 20, 0, tzinfo=timezone.utc
        ),
    )

    quote = service.option_quotes(["SPY260918P00500000"]).data[0]

    assert quote.mark == 10.0
    assert quote.bid == 9.0
    assert quote.quote_time is None
    assert quote.source_kind == "exchange_quote"
    assert quote.provider_iv == pytest.approx(0.245)
    assert quote.provider_delta == -0.4
    assert "previous_completed_session_close_used" not in quote.warnings


def test_exact_option_terms_preserve_adjusted_multiplier_and_authoritative_expiry():
    provider = _Provider()
    provider.get_options_chain = lambda _symbol, **_kwargs: {
        "callExpDateMap": {
            "2026-09-18:32": {
                "500.0": [{"exerciseType": "A", "multiplier": 50}]
            }
        }
    }
    provider.get_options_quotes = lambda _symbols: {
        "SPY   260918P00500000": {
            "mark": 5.0,
            "multiplier": 50,
            "optionDeliverablesList": [{"symbol": "SPY", "deliverableUnits": 50}],
            "asset_subtype": "etf_option",
            "exercise_style": "american",
            "settlement_type": "physical",
            "aggregate_exercise_amount": 25000,
            "expiration_at": "2026-09-18T13:30:00-04:00",
            "quote_time": NOW.isoformat(),
        }
    }
    result = LiveOptionService({}, lambda: provider, now_fn=lambda: NOW).option_contract_terms(
        ["SPY260918P00500000"]
    )

    terms = result.data[0]
    assert terms.terms_status == "adjusted"
    assert terms.adjusted_contract is True
    assert terms.price_multiplier == 50
    assert terms.shares_per_contract == 50
    assert terms.expiration_at == datetime(2026, 9, 18, 17, 30, tzinfo=timezone.utc)
    assert "option_terms_assumed_standard_occ" not in terms.warnings


def test_exact_option_terms_use_one_cached_minimal_chain_sample_per_root():
    provider = _Provider()
    service = LiveOptionService({}, lambda: provider, now_fn=lambda: NOW)
    symbols = ["SPY260918P00500000", "SPY261016C00510000"]

    first = service.option_contract_terms(symbols)
    second = service.option_contract_terms(["SPY261218P00490000"])

    assert len(provider.chain_calls) == 1
    symbol, kwargs = provider.chain_calls[0]
    assert symbol == "SPY"
    assert kwargs == {
        "contract_type": "ALL",
        "strategy": "SINGLE",
        "strike_count": 1,
        "include_underlying_quote": False,
        "from_date": date(2026, 9, 18),
        "to_date": date(2026, 9, 18),
    }
    assert {item.price_multiplier for item in first.data} == {100}
    assert {item.exercise_style for item in first.data} == {"american"}
    assert {item.source_version for item in first.data} == {
        "refdata_exact_option_terms_v4"
    }
    assert {item.settlement_style for item in first.data} == {"PM"}
    assert {item.settlement_type for item in first.data} == {"cash"}
    assert all(
        "option_settlement_type_assumed_cash" in item.warnings
        for item in first.data
    )
    assert second.data[0].price_multiplier == 100
    assert second.data[0].exercise_style == "american"
    assert second.meta.cache == "hit"


def test_exact_option_terms_maps_european_chain_exercise_type():
    provider = _Provider()
    provider.get_options_chain = lambda _symbol, **_kwargs: {
        "putExpDateMap": {
            "2026-09-18:32": {
                "500.0": [{
                    "exerciseType": "E", "settlementType": "A", "multiplier": 100,
                }]
            }
        }
    }

    terms = LiveOptionService(
        {}, lambda: provider, now_fn=lambda: NOW
    ).option_contract_terms(["SPY260918P00500000"]).data[0]

    assert terms.exercise_style == "european"
    assert terms.price_multiplier == 100
    assert terms.settlement_style == "AM"
    assert terms.settlement_type == "cash"
    assert terms.expiration_at == datetime(2026, 9, 18, 13, 30, tzinfo=timezone.utc)
    assert terms.aggregate_exercise_amount == 50_000


def test_exact_option_terms_disclose_audited_standard_occ_fallback():
    provider = _Provider()
    provider.get_options_quotes = lambda _symbols: {}
    provider.get_options_chain = lambda _symbol, **_kwargs: None
    result = LiveOptionService({}, lambda: provider, now_fn=lambda: NOW).option_contract_terms(
        ["SPY260918P00500000"]
    )

    terms = result.data[0]
    assert terms.terms_status == "partial"
    assert terms.price_multiplier == 100
    assert terms.expiration_at == datetime(2026, 9, 18, 20, tzinfo=timezone.utc)
    assert "option_terms_assumed_standard_occ" in terms.warnings
    assert "option_expiration_derived_from_occ_and_settlement_session" in terms.warnings


def test_three_month_chain_forwards_wide_strike_bounds_once_and_flattens():
    provider = _Provider()
    service = LiveOptionService({}, lambda: provider, now_fn=lambda: NOW)
    request = OptionChainRequest(
        symbol="spy", contract_type="PUT", strike_count=100,
        from_date=date(2026, 8, 17), to_date=date(2026, 11, 17),
        include_underlying_quote=True,
    )
    first = service.option_chain(request)
    second = service.option_chain(request)

    assert len(provider.chain_calls) == 1
    symbol, kwargs = provider.chain_calls[0]
    assert symbol == "SPY"
    assert kwargs == {
        "contract_type": "PUT", "strategy": "SINGLE", "strike_count": 100,
        "include_underlying_quote": True,
        "from_date": date(2026, 8, 17), "to_date": date(2026, 11, 17),
    }
    assert first.data.contracts[0].canonical_occ_symbol == "SPY260918P00500000"
    assert second.meta.cache == "hit"


def test_expiration_calendar_uses_lightweight_provider_call_and_filters_horizon():
    provider = _Provider()
    service = LiveOptionService({}, lambda: provider, now_fn=lambda: NOW)
    request = OptionExpirationRequest(
        symbol="spy", from_date=date(2026, 8, 17), to_date=date(2027, 2, 17)
    )

    first = service.option_expirations(request)
    second = service.option_expirations(request)

    assert provider.expiration_calls == ["SPY"]
    assert [item.expiration for item in first.data.expirations] == [date(2026, 9, 18)]
    assert first.data.expirations[0].option_roots == ["SPY"]
    assert second.meta.cache == "hit"


class _Response:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def read(self): return json.dumps(self.payload).encode()


def test_fred_is_bounded_partial_and_restart_cached_without_exposing_key(tmp_path):
    calls = []
    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        series = parse_qs(urlparse(request.full_url).query)["series_id"][0]
        values = [{"date": "2026-08-14", "value": "."}] if series == "BAD" else [
            {"date": "2026-08-15", "value": "4.25"},
            {"date": "2026-08-14", "value": "4.20"},
        ]
        return _Response({"observations": values})
    context = {
        "refdata.fred.enabled": True, "refdata.fred.api_key": "DO_NOT_EXPOSE",
        "refdata.fred.series_tenors": "DGS1MO:30,BAD:90",
        "refdata.fred.observation_limit": 3,
        "refdata.api.cache.reference_path": str(tmp_path / "fred.sqlite"),
    }
    first = FredRiskFreeCurveService(context, now_fn=lambda: NOW, opener=opener).curve(date(2026, 8, 16))
    restarted = FredRiskFreeCurveService(context, now_fn=lambda: NOW, opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network called"))).curve(date(2026, 8, 16))

    assert len(calls) == 2
    assert all(parse_qs(urlparse(url).query)["limit"] == ["3"] for url, _ in calls)
    assert first.data.anchors[0].simple_annual_rate_decimal == pytest.approx(0.0425)
    assert first.data.anchors[1].status == "missing"
    assert restarted.meta.cache == "hit"
    assert "DO_NOT_EXPOSE" not in restarted.model_dump_json()


def test_fred_curve_history_loads_each_series_once_and_selects_causally(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        series = parse_qs(urlparse(request.full_url).query)["series_id"][0]
        values = {
            "DGS1MO": [
                {"date": "2026-08-13", "value": "4.10"},
                {"date": "2026-08-14", "value": "4.20"},
            ],
            "DGS3MO": [
                {"date": "2026-08-13", "value": "4.30"},
                {"date": "2026-08-14", "value": "4.40"},
            ],
        }[series]
        return _Response({"observations": values})

    service = FredRiskFreeCurveService(
        {
            "refdata.fred.enabled": True,
            "refdata.fred.api_key": "DO_NOT_EXPOSE",
            "refdata.fred.series_tenors": "DGS1MO:30,DGS3MO:90",
            "refdata.api.cache.reference_path": str(tmp_path / "fred-history.sqlite"),
        },
        now_fn=lambda: NOW,
        opener=opener,
    )

    result = service.curves([date(2026, 8, 13), date(2026, 8, 16)])
    restarted = FredRiskFreeCurveService(
        service.context,
        now_fn=lambda: NOW,
        opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network called")
        ),
    ).curves([date(2026, 8, 13), date(2026, 8, 16)])

    assert len(calls) == 2
    assert all("observation_start=" in url and "observation_end=2026-08-16" in url for url, _ in calls)
    assert result.data[0].anchors[0].observation_date == date(2026, 8, 13)
    assert result.data[1].anchors[0].observation_date == date(2026, 8, 14)
    assert result.data[1].anchors[0].simple_annual_rate_decimal == pytest.approx(0.042)
    assert restarted.meta.cache == "hit"


class _CmeOpener:
    def __init__(self): self.calls = []
    def open(self, request, timeout):
        self.calls.append((request.full_url, timeout, dict(request.header_items())))
        if request.get_method() == "POST":
            assert request.data == b"grant_type=client_credentials"
            return _Response({"access_token": "DO_NOT_EXPOSE_TOKEN", "token_type": "bearer", "expires_in": 1799})
        contract = parse_qs(urlparse(request.full_url).query)["globexSymbol"][0]
        return _Response({"_embedded": {"instruments": [{
            "globexSymbol": contract, "globexGroupCode": "ES", "instrumentName": "E-mini S&P 500",
            "guid": "INSTRUMENT_GUID", "productGuidInt": 12345,
            "unitOfMeasureQty": 50, "tradeTick": 0.25, "currency": "USD",
            "exchangeClearing": "CME", "exchangeGlobex": "XCME",
            "lastTradeDate": "2026-09-18", "lastUpdated": NOW.isoformat(),
        }]}})


def test_cme_normalizes_tick_value_and_survives_restart_without_secrets(tmp_path):
    context = {"refdata.cme_reference.enabled": True, "refdata.cme_reference.api_id": "SECRET_ID",
               "refdata.cme_reference.app_password": "SECRET_PASSWORD", "refdata.cme_reference.base_url": "https://example.invalid",
               "refdata.api.cache.reference_path": str(tmp_path / "cme.sqlite")}
    opener = _CmeOpener()
    first = CmeReferenceDataService(context, now_fn=lambda: NOW, opener=opener).specifications(["ESU6"])
    restarted = CmeReferenceDataService(context, now_fn=lambda: NOW, opener=object()).specifications(["ESU6"])
    assert len(opener.calls) == 2
    assert opener.calls[0][0] == "https://auth.cmegroup.com/as/token.oauth2"
    assert opener.calls[1][0].endswith("/v3/instruments?globexSymbol=ESU6")
    assert opener.calls[1][2]["Authorization"] == "Bearer DO_NOT_EXPOSE_TOKEN"
    assert first.data[0].tick_value == pytest.approx(12.5)
    assert first.data[0].tick_value_derivation == "minimum_tick * price_multiplier; cme_tick_derivation_v1"
    assert restarted.meta.cache == "hit"
    encoded = restarted.model_dump_json()
    assert "SECRET_ID" not in encoded and "SECRET_PASSWORD" not in encoded
    assert "DO_NOT_EXPOSE_TOKEN" not in encoded


def test_cme_translates_two_digit_broker_year_to_globex_symbol(tmp_path):
    context = {
        "refdata.cme_reference.enabled": True,
        "refdata.cme_reference.api_id": "SECRET_ID",
        "refdata.cme_reference.app_password": "SECRET_PASSWORD",
        "refdata.cme_reference.base_url": "https://example.invalid",
        "refdata.api.cache.reference_path": str(tmp_path / "cme-two-digit.sqlite"),
    }
    opener = _CmeOpener()

    result = CmeReferenceDataService(
        context,
        now_fn=lambda: NOW,
        opener=opener,
    ).specifications(["ESU26"])

    assert opener.calls[1][0].endswith("/v3/instruments?globexSymbol=ESU6")
    assert result.data[0].requested_contract == "ESU26"
    assert result.data[0].canonical_contract == "ESU26"
    assert result.data[0].status == "ok"
