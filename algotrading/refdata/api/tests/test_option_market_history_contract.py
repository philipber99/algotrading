from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import logging
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import Response
from pydantic import ValidationError

from ..app import create_app
from ..cache import PersistentOptionBarsCache
from ..live_models import OptionMarketHistoryRequest
from ..service import InvalidRequestError, RefDataService
from .fakes import FIXED_NOW, FakeMarketData


START = datetime(2026, 8, 1, tzinfo=timezone.utc)


class OptionHistoryProvider:
    def __init__(self) -> None:
        self.calls = []

    def get_bars(self, symbol, start, end, granularity):
        self.calls.append((symbol, start, end, granularity))
        dates = pd.date_range(start.date(), end.date(), freq="D")
        return pd.DataFrame(
            {
                "open": [10.0 + index for index in range(len(dates))],
                "high": [11.0 + index for index in range(len(dates))],
                "low": [9.0 + index for index in range(len(dates))],
                "close": [10.5 + index for index in range(len(dates))],
                "volume": [100 + index for index in range(len(dates))],
            },
            index=pd.DatetimeIndex(dates, name="timestamp"),
        )


def app_context(tmp_path, *, now_fn=lambda: FIXED_NOW, provider=None, **updates):
    context = {
        "logger": logging.getLogger("option-history-contract"),
        "local_timezone": "UTC",
        "refdata.api.timezone": "UTC",
        "refdata.api.option_history_cache.path": str(tmp_path / "option-bars.sqlite3"),
        "refdata.economic_calendar.auto_update.enabled": False,
        "refdata.earnings_calendar.auto_update.enabled": False,
    }
    context.update(updates)
    market_data = FakeMarketData()
    market_data.refdata_source = provider or OptionHistoryProvider()
    service = RefDataService(
        context,
        market_data_module=market_data,
        now_fn=now_fn,
    )
    return context, service, market_data.refdata_source


def call_option_history_route(app, payload):
    route = next(
        item
        for item in app.routes
        if getattr(item, "path", None) == "/api/v1/options/market-history/query"
    )
    response = Response()
    result = route.endpoint(
        payload=OptionMarketHistoryRequest.model_validate(payload),
        request=SimpleNamespace(app=app),
        response=response,
    )
    return result, response


def direct_test_app(context, service):
    app = create_app(context=context, service=service)
    app.state.refdata_context = context
    app.state.refdata_service = service
    return app


def test_option_history_request_normalizes_occ_and_rejects_duplicates_and_bad_intervals():
    request = OptionMarketHistoryRequest(
        symbols=["SPY   260918P00500000"],
        start_at=START,
        end_at=START + timedelta(days=1),
        interval="1d",
        include_greeks=True,
    )
    assert request.symbols == ["SPY260918P00500000"]

    with pytest.raises(ValidationError):
        OptionMarketHistoryRequest(
            symbols=["SPY260918P00500000", "SPY   260918P00500000"],
            start_at=START,
            end_at=START + timedelta(days=1),
        )
    with pytest.raises(ValidationError):
        OptionMarketHistoryRequest(
            symbols=["SPY260918P00500000"],
            start_at=START,
            end_at=START,
        )
    with pytest.raises(ValidationError):
        OptionMarketHistoryRequest(
            symbols=["not-an-occ-symbol"],
            start_at=START,
            end_at=START + timedelta(days=1),
        )


def test_option_history_route_returns_daily_ohlc_from_padded_occ_provider(tmp_path):
    context, service, provider = app_context(tmp_path)
    app = direct_test_app(context, service)
    schema = app.openapi()
    result, route_response = call_option_history_route(
        app,
        {
            "symbols": ["SPY260918P00500000"],
            "start_at": START.isoformat(),
            "end_at": (START + timedelta(days=3)).isoformat(),
            "interval": "1d",
            "include_greeks": True,
        },
    )
    operation = schema["paths"]["/api/v1/options/market-history/query"]["post"]
    assert "501" not in operation["responses"]
    assert route_response.headers["X-Cache"] == "MISS"
    assert len(provider.calls) == 1
    symbol, provider_start, provider_end, granularity = provider.calls[0]
    assert symbol == "SPY   260918P00500000"
    assert provider_start == datetime.combine(START.date(), time.min)
    assert provider_end.date() == (START + timedelta(days=2)).date()
    assert granularity == "1day"
    payload = result.model_dump(mode="json")
    assert len(payload["data"]) == 3
    assert payload["data"][0] == {
        "requested_symbol": "SPY260918P00500000",
        "canonical_occ_symbol": "SPY260918P00500000",
        "at": "2026-08-02T00:00:00Z",
        "observed_at": None,
        "session_date": "2026-08-01",
        "status": "ok",
        "bid": None,
        "ask": None,
        "mark": 10.5,
        "last": 10.5,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 100.0,
        "underlying_price": None,
        "provider_iv": None,
        "provider_delta": None,
        "provider_gamma": None,
        "provider_theta_per_day": None,
        "provider_vega_per_1pct": None,
            "provider_rho_per_1pct": None,
            "surface_state": None,
            "mark_basis": None,
        "source": "OptionHistoryProvider",
        "warnings": ["historical_option_greeks_unavailable"],
        "error": None,
    }


def test_option_history_cache_survives_restart_and_loads_only_missing_ranges(tmp_path):
    context, service, first_provider = app_context(tmp_path)
    request = {
        "symbols": ["SPY260918P00500000"],
        "start_at": START.isoformat(),
        "end_at": (START + timedelta(days=3)).isoformat(),
        "interval": "1d",
        "include_greeks": False,
    }
    call_option_history_route(direct_test_app(context, service), request)
    assert len(first_provider.calls) == 1

    second_provider = OptionHistoryProvider()
    context, service, _ = app_context(
        tmp_path, provider=second_provider
    )
    expanded = {
        **request,
        "start_at": (START - timedelta(days=2)).isoformat(),
        "end_at": (START + timedelta(days=5)).isoformat(),
    }
    result, _ = call_option_history_route(
        direct_test_app(context, service), expanded
    )
    assert result.meta.cache == "miss"
    assert [
        (call[1].date(), call[2].date(), call[3])
        for call in second_provider.calls
    ] == [
        (START.date() - timedelta(days=2), START.date() - timedelta(days=1), "1day"),
        (START.date() + timedelta(days=3), START.date() + timedelta(days=4), "1day"),
    ]

    third_provider = OptionHistoryProvider()
    context, service, _ = app_context(tmp_path, provider=third_provider)
    result, _ = call_option_history_route(
        direct_test_app(context, service), expanded
    )
    assert result.meta.cache == "hit"
    assert third_provider.calls == []


def test_option_history_contract_ttl_is_sliding_on_cache_hits(tmp_path):
    clock = [FIXED_NOW]
    context, service, provider = app_context(tmp_path, now_fn=lambda: clock[0])
    request = {
        "symbols": ["SPY260918P00500000"],
        "start_at": START.isoformat(),
        "end_at": (START + timedelta(days=1)).isoformat(),
        "interval": "1d",
        "include_greeks": False,
    }
    app = direct_test_app(context, service)
    first, _ = call_option_history_route(app, request)
    clock[0] += timedelta(days=6)
    second, _ = call_option_history_route(app, request)
    clock[0] += timedelta(days=6)
    third, _ = call_option_history_route(app, request)
    clock[0] += timedelta(days=8)
    fourth, _ = call_option_history_route(app, request)
    assert [
        response.meta.cache
        for response in (first, second, third, fourth)
    ] == ["miss", "hit", "hit", "miss"]
    assert len(provider.calls) == 2


def test_option_history_cache_detects_an_interior_coverage_gap(tmp_path):
    cache = PersistentOptionBarsCache(tmp_path / "interior-gap.sqlite3")
    ttl = timedelta(days=7)
    try:
        cache.store(
            "SPY260918P00500000",
            [],
            start=START.date(),
            end=START.date() + timedelta(days=1),
            now=FIXED_NOW,
            ttl=ttl,
        )
        cache.store(
            "SPY260918P00500000",
            [],
            start=START.date() + timedelta(days=4),
            end=START.date() + timedelta(days=5),
            now=FIXED_NOW,
            ttl=ttl,
        )
        lookup = cache.load(
            "SPY260918P00500000",
            start=START.date(),
            end=START.date() + timedelta(days=5),
            now=FIXED_NOW,
            ttl=ttl,
        )
    finally:
        cache.close()
    assert lookup.hit is True
    assert lookup.missing_ranges == [
        (START.date() + timedelta(days=2), START.date() + timedelta(days=3))
    ]


def test_option_history_configured_period_bound_is_validated(tmp_path):
    context, service, _ = app_context(tmp_path)
    context["refdata.api.option_market_history_max_days"] = 2
    with pytest.raises(InvalidRequestError) as captured:
        call_option_history_route(
            direct_test_app(context, service),
            {
                "symbols": ["SPY260918P00500000"],
                "start_at": START.isoformat(),
                "end_at": (START + timedelta(days=3)).isoformat(),
                "interval": "1d",
                "include_greeks": True,
            },
        )
    assert captured.value.code == "INVALID_REQUEST"
    assert captured.value.status_code == 422
    assert captured.value.details == {
        "endpoint": "/api/v1/options/market-history/query",
        "requested_start": START.isoformat(),
        "requested_end": (START + timedelta(days=3)).isoformat(),
        "maximum_days": 2,
        "exclusive_end": True,
    }


def test_normal_252_return_window_runs_through_actual_route_below_default_limit(tmp_path):
    context, service, provider = app_context(tmp_path)
    start = datetime(2025, 8, 25, tzinfo=timezone.utc)
    end = datetime(2026, 8, 27, tzinfo=timezone.utc)
    assert end - start == timedelta(days=367)

    app = direct_test_app(context, service)
    result, route_response = call_option_history_route(
        app,
        {
            "symbols": ["SPY260918P00500000"],
            "start_at": start.isoformat(),
            "end_at": end.isoformat(),
            "interval": "1d",
            "include_greeks": False,
        },
    )

    assert route_response.status_code == 200
    assert context.get("refdata.api.option_market_history_max_days", 376) == 376
    assert len(result.data) == 367
    assert len(provider.calls) == 1
    assert provider.calls[0][1].date() == start.date()
    assert provider.calls[0][2].date() == (end - timedelta(days=1)).date()
