from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import requests
import pytest

from algotrading.risk.option_prices.backfill import (
    HistoricalOptionPriceBackfill,
    MassiveAggregateClient,
    MassiveAggregateRequestError,
    compile_holding_histories,
)
from algotrading.risk.transactions.models import (
    TransactionItemModel,
    TransactionModel,
)


CALL = "XYZ260117C00100000"
NOW = datetime(2026, 1, 20, 12, tzinfo=timezone.utc)


def _trade(identifier: int, at: str, instruction: str, quantity: int):
    timestamp = datetime.fromisoformat(at)
    return TransactionModel(
        id=identifier,
        broker_transaction_id=str(identifier),
        transaction_at=timestamp,
        transaction_type="TRADE",
        trade_date=timestamp.date(),
        ingested_at=NOW,
        items=[
            TransactionItemModel(
                item_index=0,
                symbol=CALL,
                asset_type="OPTION",
                instruction=instruction,
                quantity=Decimal(quantity),
                price=Decimal("2.50"),
            )
        ],
    )


def _bar(day: date, close: float):
    timestamp = datetime(
        day.year, day.month, day.day, 5, tzinfo=timezone.utc
    ).timestamp()
    return {
        "t": int(timestamp * 1000),
        "o": close - 0.25,
        "h": close + 0.5,
        "l": close - 0.5,
        "c": close,
        "v": 10,
    }


def test_compile_holding_histories_clips_windows_and_caps_open_contract_at_expiry():
    from zoneinfo import ZoneInfo

    transactions = [
        _trade(1, "2026-01-02T15:00:00+00:00", "BUY_TO_OPEN", 2),
        _trade(2, "2026-01-05T15:00:00+00:00", "SELL_TO_CLOSE", 1),
        _trade(3, "2026-01-07T15:00:00+00:00", "SELL_TO_CLOSE", 1),
        _trade(4, "2026-01-10T15:00:00+00:00", "SELL_TO_OPEN", 1),
    ]

    histories = compile_holding_histories(
        "ACCOUNT-1",
        transactions,
        through_date=date(2026, 1, 19),
        market_timezone=ZoneInfo("America/New_York"),
        from_date=date(2026, 1, 3),
    )

    assert len(histories) == 1
    history = histories[0]
    assert [(item.start_date, item.end_date) for item in history.windows] == [
        (date(2026, 1, 3), date(2026, 1, 7)),
        (date(2026, 1, 10), date(2026, 1, 17)),
    ]
    assert history.quantity_at_close(date(2026, 1, 6)) == 1
    assert history.quantity_at_close(date(2026, 1, 7)) == 0
    assert history.quantity_at_close(date(2026, 1, 11)) == -1


class TransactionRepository:
    def __init__(self, transactions):
        self.transactions = transactions
        self.calls = []

    def list_trade_accounts(self):
        return ["ACCOUNT-1"]

    def query_all_trade_history(self, account_id, *, through, max_rows):
        self.calls.append((account_id, through, max_rows))
        return list(self.transactions)


class PriceRepository:
    def __init__(self):
        self.saved = []
        self.existing = {}
        self.existing_calls = []

    def existing_session_dates(
        self, account_id, canonical_occ_symbol, *, start_date, end_date
    ):
        self.existing_calls.append(
            (account_id, canonical_occ_symbol, start_date, end_date)
        )
        return set(self.existing.get((account_id, canonical_occ_symbol), set()))

    def save_backfill_batch(self, summary, prices, *, final):
        self.saved.append((summary, list(prices), final))


class Provider:
    def __init__(self, bars):
        self.bars = bars
        self.calls = []

    def daily_bars(self, ticker, *, start_date, end_date, adjusted):
        self.calls.append((ticker, start_date, end_date, adjusted))
        return list(self.bars)


def test_backfill_reads_all_inventory_history_and_persists_ohlc_without_greeks():
    transactions = [
        _trade(1, "2026-01-02T15:00:00+00:00", "BUY_TO_OPEN", 2),
        _trade(2, "2026-01-06T15:00:00+00:00", "SELL_TO_CLOSE", 2),
    ]
    transaction_repository = TransactionRepository(transactions)
    price_repository = PriceRepository()
    provider = Provider(
        [
            _bar(date(2026, 1, 2), 2.0),
            _bar(date(2026, 1, 5), 2.5),
            _bar(date(2026, 1, 6), 3.0),
            _bar(date(2026, 1, 7), 99.0),
        ]
    )

    result = HistoricalOptionPriceBackfill(
        transaction_repository,
        price_repository,
        provider,
        {"risk.option_prices.market_timezone": "America/New_York"},
        now_fn=lambda: NOW,
    ).run(through_date=date(2026, 1, 6))

    assert provider.calls == [
        ("O:" + CALL, date(2026, 1, 2), date(2026, 1, 6), False)
    ]
    assert result.summary.quote_count == 3
    assert result.contracts_without_bars == ()
    saved_summary, prices, final = price_repository.saved[0]
    assert saved_summary.run_id == result.summary.run_id
    assert saved_summary.status == "running"
    assert saved_summary.quote_count == result.summary.quote_count
    assert final is False
    assert [item.session_date for item in prices] == [
        date(2026, 1, 2),
        date(2026, 1, 5),
        date(2026, 1, 6),
    ]
    assert [item.position.signed_quantity for item in prices] == [2, 2, 0]
    assert prices[-1].mark == 3.0
    assert prices[-1].close == 3.0
    assert prices[-1].provider_iv is None
    assert prices[-1].provider_delta is None
    assert prices[-1].source == "MassiveAggregates"
    final_summary, final_prices, final = price_repository.saved[-1]
    assert final_summary == result.summary
    assert final_prices == []
    assert final is True


def test_backfill_dry_run_queries_provider_without_writing():
    transaction_repository = TransactionRepository(
        [_trade(1, "2026-01-02T15:00:00+00:00", "BUY_TO_OPEN", 1)]
    )
    price_repository = PriceRepository()
    provider = Provider([_bar(date(2026, 1, 2), 2.0)])

    result = HistoricalOptionPriceBackfill(
        transaction_repository,
        price_repository,
        provider,
        {"risk.option_prices.market_timezone": "America/New_York"},
        now_fn=lambda: NOW,
    ).run(through_date=date(2026, 1, 3), dry_run=True)

    assert result.summary.quote_count == 1
    assert price_repository.saved == []


def test_backfill_skips_existing_sessions_and_commits_each_new_batch():
    transactions = [
        _trade(1, "2026-01-02T15:00:00+00:00", "BUY_TO_OPEN", 2),
        _trade(2, "2026-01-06T15:00:00+00:00", "SELL_TO_CLOSE", 2),
    ]
    transaction_repository = TransactionRepository(transactions)
    price_repository = PriceRepository()
    price_repository.existing[("ACCOUNT-1", CALL)] = {
        date(2026, 1, 2),
        date(2026, 1, 6),
    }
    provider = Provider([_bar(date(2026, 1, 5), 2.5)])

    result = HistoricalOptionPriceBackfill(
        transaction_repository,
        price_repository,
        provider,
        {"risk.option_prices.market_timezone": "America/New_York"},
        now_fn=lambda: NOW,
    ).run(through_date=date(2026, 1, 6))

    assert price_repository.existing_calls == [
        ("ACCOUNT-1", CALL, date(2026, 1, 2), date(2026, 1, 6))
    ]
    assert provider.calls == [("O:" + CALL, date(2026, 1, 5), date(2026, 1, 5), False)]
    assert result.summary.quote_count == 1
    batch_summary, batch_prices, final = price_repository.saved[0]
    assert batch_summary.status == "running"
    assert [item.session_date for item in batch_prices] == [date(2026, 1, 5)]
    assert final is False
    assert price_repository.saved[-1][2] is True


class Response:
    def __init__(
        self, status_code, payload=None, headers=None, *, text="", reason=""
    ):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text
        self.reason = reason

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


class Session:
    def __init__(self, responses):
        self.headers = {}
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return self.responses.pop(0)


def test_massive_client_retries_and_follows_same_origin_pagination():
    session = Session(
        [
            Response(429, headers={"Retry-After": "0"}),
            Response(
                200,
                {
                    "status": "OK",
                    "results": [{"t": 1, "c": 2}],
                    "next_url": "/v2/aggs/next?cursor=abc",
                },
            ),
            Response(200, {"status": "OK", "results": [{"t": 2, "c": 3}]}),
        ]
    )
    sleeps = []
    client = MassiveAggregateClient(
        "secret",
        session=session,
        sleep_fn=sleeps.append,
    )

    rows = client.daily_bars(
        "O:" + CALL,
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 3),
    )

    assert rows == [{"t": 1, "c": 2}, {"t": 2, "c": 3}]
    assert sleeps == [0.0]
    assert session.headers["Authorization"] == "Bearer secret"
    assert session.calls[0][1] == {
        "adjusted": "false",
        "sort": "asc",
        "limit": 50_000,
    }
    assert session.calls[-1][0] == "https://api.massive.com/v2/aggs/next?cursor=abc"
    assert session.calls[-1][1] is None


def test_massive_client_logs_http_status_without_provider_error_text(caplog):
    marker = "SYNTHETIC_PROVIDER_CREDENTIAL"
    session = Session(
        [
            Response(
                429,
                headers={"Retry-After": "0"},
                reason=f"Too Many Requests {marker}",
                text=f'{{"error":"api_key={marker}"}}',
            ),
            Response(
                429,
                reason=f"Too Many Requests {marker}",
                text=f'{{"error":"api_key={marker}"}}',
            ),
        ]
    )
    client = MassiveAggregateClient(
        "secret", session=session, sleep_fn=lambda _: None, max_attempts=2
    )

    with pytest.raises(MassiveAggregateRequestError, match="status=429") as error:
        client.daily_bars(
            "O:" + CALL,
            start_date=date(2026, 1, 2),
            end_date=date(2026, 1, 2),
        )

    assert "status=429" in caplog.text
    assert marker not in caplog.text
    assert marker not in str(error.value)


def test_massive_client_sanitizes_transport_failures(caplog):
    marker = "SYNTHETIC_TRANSPORT_CREDENTIAL"

    class FailingSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, *, params, timeout):
            raise requests.ConnectionError(f"https://example.invalid/?api_key={marker}")

    client = MassiveAggregateClient(
        "placeholder", session=FailingSession(), sleep_fn=lambda _: None, max_attempts=2
    )
    with pytest.raises(MassiveAggregateRequestError, match="ConnectionError") as error:
        client.daily_bars(
            "O:" + CALL, start_date=date(2026, 1, 2), end_date=date(2026, 1, 2)
        )

    assert marker not in str(error.value)
    assert marker not in caplog.text


class Clock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


def test_massive_client_throttles_retries_and_pagination_at_five_per_minute():
    responses = []
    for page in range(6):
        payload = {"status": "OK", "results": []}
        if page < 5:
            payload["next_url"] = f"/v2/aggs/page/{page + 1}"
        responses.append(Response(200, payload))
    clock = Clock()
    session = Session(responses)
    client = MassiveAggregateClient(
        "secret",
        session=session,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        max_requests_per_minute=5,
    )

    client.daily_bars(
        "O:" + CALL,
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 3),
    )

    assert len(session.calls) == 6
    assert clock.sleeps == [60.0]
