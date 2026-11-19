from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
from threading import Barrier

import httpx

from algotrading.risk.market_inputs import client as client_module
from algotrading.risk.market_inputs.client import ReferenceDataClient


def test_security_classifications_uses_exact_symbol_filters():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/api/v1/universe/symbols"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"symbol": "TSM", "sector": "Technology"},
                    {"symbol": "WMT", "sector": "Consumer Defensive"},
                ]
            },
        )

    client = ReferenceDataClient(
        {"risk.refdata.maximum_classification_batch": 10},
        client=httpx.Client(
            base_url="http://refdata.test",
            transport=httpx.MockTransport(handler),
        ),
    )

    rows = client.security_classifications(("wmt", "TSM", "TSM"))

    assert rows == [
        {"symbol": "TSM", "sector": "Technology"},
        {"symbol": "WMT", "sector": "Consumer Defensive"},
    ]
    assert requests[0].url.params.get_list("symbols") == ["TSM", "WMT"]
    assert requests[0].url.params["active_only"] == "false"


def test_declared_sector_etf_classification_survives_missing_company_join():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/universe/symbols"
        return httpx.Response(200, json={"data": []})

    client = ReferenceDataClient(
        {}, client=httpx.Client(
            base_url="http://refdata.test", transport=httpx.MockTransport(handler)
        )
    )

    assert client.security_classifications(["XLP"]) == [{
        "symbol": "XLP",
        "security_type": "ETF",
        "taxonomy": "GICS",
        "sector": "Consumer Staples",
        "company_name": "State Street Consumer Staples Select Sector SPDR ETF",
    }]


def test_etf_exposure_lookup_is_cached_per_symbol():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": {
            "symbol": "CIBR", "status": "available", "asset_class": "Sector Equity",
            "sector_weights": [{"sector": "Technology", "weight_percent": 95}],
        }})

    client = ReferenceDataClient(
        {}, client=httpx.Client(
            base_url="http://refdata.test", transport=httpx.MockTransport(handler)
        ),
    )
    assert client.etf_exposure("cibr") == client.etf_exposure("CIBR")
    assert len(requests) == 1
    assert requests[0].url.path == "/api/v1/etfs/CIBR/exposure"


def test_finra_short_volume_uses_published_bounded_batch_route():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": {
            "sessions": ["2026-08-24T00:00:00Z"],
            "observations": [{
                "symbol": "AAPL", "timestamp": "2026-08-24T00:00:00Z",
                "short_volume_percent": 0.61,
            }],
        }})

    client = ReferenceDataClient(
        {}, client=httpx.Client(
            base_url="http://refdata.test", transport=httpx.MockTransport(handler)
        )
    )
    payload = client.finra_short_volume(
        ["aapl", "AAPL"], as_of=date(2026, 8, 25), session_limit=5
    )

    assert payload["observations"][0]["short_volume_percent"] == 0.61
    assert requests[0].url.path == "/api/v1/finra/short-volume"
    assert requests[0].url.params["symbols"] == "AAPL"
    assert requests[0].url.params["session_limit"] == "5"


def test_candidate_universe_and_current_price_use_published_bounded_routes():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/universe/symbols":
            return httpx.Response(200, json={"data": [{"symbol": "SPY"}]})
        return httpx.Response(
            200,
            json={
                "data": {"symbol": "SPY", "price": 510.25},
                "meta": {"as_of": "2026-08-25T20:00:00Z"},
            },
        )

    client = ReferenceDataClient(
        {},
        client=httpx.Client(
            base_url="http://refdata.test",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.universe_symbols(["spy", "SPY"]) == [{"symbol": "SPY"}]
    assert client.current_price("spy")["data"]["price"] == 510.25
    assert requests[0].url.params.get_list("symbols") == ["SPY"]
    assert requests[0].url.params["active_only"] == "true"
    assert requests[0].url.params["include_stale_daily_bars"] == "false"
    assert requests[1].url.path == "/api/v1/symbols/SPY/current-price"
    assert requests[1].url.params["view"] == "price"


def test_futures_provider_slash_is_not_sent_as_a_refdata_path_separator():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/current-price"):
            return httpx.Response(
                200,
                json={
                    "data": {"symbol": "ESU26", "price": 6500.0},
                    "meta": {"as_of": "2026-08-26T13:00:00Z"},
                },
            )
        return httpx.Response(200, json={"data": [], "meta": {}})

    client = ReferenceDataClient(
        {},
        client=httpx.Client(
            base_url="http://refdata.test",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.current_price("/esu26")["data"]["symbol"] == "ESU26"
    assert client.current_futures_price("/nqu26")["data"]["symbol"] == "ESU26"
    client.futures_specifications(["/ESU26", "nqu26"])

    assert requests[0].url.path == "/api/v1/symbols/ESU26/current-price"
    assert "//ESU26" not in requests[0].url.path
    assert requests[1].url.path == "/api/v1/futures/NQU26/current-price"
    assert json.loads(requests[2].content) == {"contracts": ["ESU26", "NQU26"]}


def test_bars_forwards_remote_query_as_the_hyphenated_refdata_parameter():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": []})

    client = ReferenceDataClient(
        {},
        client=httpx.Client(
            base_url="http://refdata.test",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.bars(
        ("MSFT",),
        start=date(2026, 7, 1),
        end=date(2026, 7, 2),
        include_current_quote=True,
        remote_query=True,
    ) == []
    assert requests[0].url.path == "/api/v1/bars"
    assert requests[0].url.params["include_current_quote"] == "true"
    assert requests[0].url.params["remote-query"] == "true"


def test_historical_minute_bars_requests_stored_completed_minute_window():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"close": 101.25}]})

    client = ReferenceDataClient(
        {},
        client=httpx.Client(
            base_url="http://refdata.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    start = datetime(2026, 8, 20, 13, 20, tzinfo=timezone.utc)
    end = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)

    assert client.historical_minute_bars(
        "wmt", start_at=start, end_at=end
    ) == [{"close": 101.25}]
    assert requests[0].url.path == "/api/v1/symbols/WMT/bars"
    assert requests[0].url.params["start"] == start.isoformat()
    assert requests[0].url.params["end"] == end.isoformat()
    assert requests[0].url.params["interval"] == "1m"
    assert requests[0].url.params["market_hours_only"] == "false"
    assert requests[0].url.params["remote-query"] == "false"


def test_earnings_calendar_uses_bulk_calendar_route():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"symbol": "SPY"}]})

    client = ReferenceDataClient(
        {},
        client=httpx.Client(
            base_url="http://refdata.test",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.earnings_calendar(
        start=date(2026, 8, 29), end=date(2026, 9, 30)
    ) == [{"symbol": "SPY"}]
    assert requests[0].url.path == "/api/v1/calendars/earnings"


def test_risk_free_curve_is_cached_by_date_for_thirty_minutes(monkeypatch):
    requests: list[httpx.Request] = []
    clock = [100.0]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": {
                    "curve_id": "usd-risk-free",
                    "anchors": [{"tenor_days": 30, "rate": len(requests) / 100}],
                }
            },
        )

    monkeypatch.setattr(client_module, "monotonic", lambda: clock[0])
    client = ReferenceDataClient(
        {},
        client=httpx.Client(
            base_url="http://refdata.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    session = date(2026, 8, 28)

    first = client.risk_free_curve(session)
    first["data"]["anchors"][0]["rate"] = 99
    clock[0] += 1799
    cached = client.risk_free_curve(session)

    assert cached["data"]["anchors"][0]["rate"] == 0.01
    assert len(requests) == 1
    assert requests[0].url.params["as_of"] == session.isoformat()

    clock[0] += 2
    refreshed = client.risk_free_curve(session)

    assert refreshed["data"]["anchors"][0]["rate"] == 0.02
    assert len(requests) == 2


def test_risk_free_curve_cache_keeps_dates_separate():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": {"as_of": request.url.params["as_of"]}})

    client = ReferenceDataClient(
        {},
        client=httpx.Client(
            base_url="http://refdata.test",
            transport=httpx.MockTransport(handler),
        ),
    )

    first_date = date(2026, 8, 27)
    second_date = date(2026, 8, 28)
    assert client.risk_free_curve(first_date)["data"]["as_of"] == first_date.isoformat()
    assert client.risk_free_curve(second_date)["data"]["as_of"] == second_date.isoformat()
    assert client.risk_free_curve(first_date)["data"]["as_of"] == first_date.isoformat()
    assert len(requests) == 2


def test_risk_free_curve_history_batches_misses_and_populates_local_cache():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/api/v1/rates/risk-free-curves/query"
        dates = json.loads(request.content)["as_of_dates"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"requested_as_of": item, "curve_id": "usd", "anchors": []}
                    for item in dates
                ],
                "meta": {"cache": "miss"},
            },
        )

    client = ReferenceDataClient(
        {},
        client=httpx.Client(
            base_url="http://refdata.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    dates = (date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28))

    first = client.risk_free_curves(dates)
    second = client.risk_free_curves(reversed(dates))

    assert set(first) == set(second) == set(dates)
    assert len(requests) == 1
    assert json.loads(requests[0].content)["as_of_dates"] == [
        item.isoformat() for item in dates
    ]


def test_distinct_risk_free_curve_dates_do_not_serialize_network_requests():
    entered = Barrier(2, timeout=2)

    def handler(request: httpx.Request) -> httpx.Response:
        entered.wait()
        return httpx.Response(
            200, json={"data": {"as_of": request.url.params["as_of"]}}
        )

    client = ReferenceDataClient(
        {},
        client=httpx.Client(
            base_url="http://refdata.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    sessions = (date(2026, 8, 27), date(2026, 8, 28))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(client.risk_free_curve, sessions))

    assert [item["data"]["as_of"] for item in results] == [
        session.isoformat() for session in sessions
    ]
