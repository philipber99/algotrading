from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from algotrading.refdata.api.live_models import OptionMarketHistoryRequest
from algotrading.risk.option_prices import capture as capture_module
from algotrading.risk.option_prices.capture import OptionPriceCapture
from algotrading.risk.option_prices.models import (
    CaptureSummary,
    ChainSnapshotStatus,
    MarkSource,
    SurfaceObservationSource,
)
from algotrading.risk.option_prices.repository import (
    OptionPriceRepository,
    OptionPriceRepositoryUnavailable,
)
from algotrading.risk.option_prices.service import RiskOptionHistoryService


NOW = datetime(2026, 8, 18, 23, 30, tzinfo=timezone.utc)
CALL = "XYZ260918C00100000"
PROVIDER_CALL = "XYZ   260918C00100000"
PUT = "XYZ260918P00090000"
PROVIDER_PUT = "XYZ   260918P00090000"


class Broker:
    def __init__(self) -> None:
        self.calls = 0
        self.transaction_calls = []

    def get_accounts(self):
        self.calls += 1
        return [
            {
                "securitiesAccount": {
                    "accountNumber": "ACCOUNT-0001",
                    "positions": [
                        {
                            "longQuantity": 2,
                            "shortQuantity": 0,
                            "instrument": {
                                "symbol": PROVIDER_CALL,
                                "type": "OPTION",
                            },
                        },
                        {
                            "longQuantity": 5,
                            "shortQuantity": 0,
                            "instrument": {"symbol": "XYZ", "type": "EQUITY"},
                        },
                    ],
                }
            }
        ]

    def get_transactions(self, account, start, end, transaction_type):
        self.transaction_calls.append((account, start, end, transaction_type))
        return []


class Provider:
    def __init__(self) -> None:
        self.calls = []

    def get_option_quotes(self, symbols):
        self.calls.append(symbols)
        return {
            PROVIDER_CALL: {
                "bid": 4.8,
                "ask": 5.2,
                "mark": 5.0,
                "last": 4.95,
                "quote_time": int(NOW.timestamp() * 1000),
                "volatility": 25.0,
                "delta": 0.45,
            }
        }

    def get_options_chain(self, symbol, **kwargs):
        assert symbol == "XYZ"
        assert kwargs["from_date"].isoformat() == "2026-09-18"
        assert kwargs["to_date"].isoformat() == "2026-09-18"
        return {
            "symbol": "XYZ",
            "underlyingPrice": 100.0,
            "callExpDateMap": {
                "2026-09-18:31": {
                    "100.0": [
                        {
                            "symbol": PROVIDER_CALL,
                            "bid": 4.8,
                            "ask": 5.2,
                            "mark": 5.0,
                            "last": 4.95,
                            "quoteTimeInLong": int(NOW.timestamp() * 1000),
                            "openInterest": 10,
                            "totalVolume": 3,
                            "securityStatus": "NORMAL",
                        }
                    ]
                }
            },
            "putExpDateMap": {
                "2026-09-18:31": {
                    "90.0": [
                        {
                            "symbol": PROVIDER_PUT,
                            "bid": 2.4,
                            "ask": 2.6,
                            "mark": 2.5,
                            "last": 2.45,
                            "quoteTimeInLong": int(NOW.timestamp() * 1000),
                        }
                    ]
                }
            },
        }


class Repository:
    def __init__(self) -> None:
        self.saved = []

    def save_capture(self, summary, prices, *, chain_snapshots=()):
        self.saved.append((summary, list(prices), list(chain_snapshots)))


class CliMonitor:
    MONITOR_OK = 0
    MONITOR_CRITICAL = 2
    instances = []

    def __init__(self, context):
        self.context = context
        self.events = []
        type(self).instances.append(self)

    def report_event(self, level, message):
        self.events.append((level, message))


def _run_cli(
    monkeypatch, *, missing=(), chain_problem=False, setup_error=None, error=None
):
    CliMonitor.instances.clear()
    base_context = {
        "nagios.service_id": "ALGO_REFDATA-Refdata Update",
        "risk.option_prices.market_timezone": "America/New_York",
    }
    provider_contexts = []
    summary = CaptureSummary(
        run_id="capture-run",
        captured_at=NOW,
        account_count=1,
        position_count=1,
        quote_count=0 if missing else 1,
        missing_count=len(missing),
        status="partial" if missing or chain_problem else "success",
        provider_class="test.Provider",
        chain_count=1 if chain_problem else 0,
    )

    class CliProvider:
        def __init__(self, context):
            provider_contexts.append(context)

    class CliCapture:
        def __init__(self, *args):
            pass

        def capture(self, **kwargs):
            assert CliMonitor.instances[0].events == []
            if error is not None:
                raise error
            return summary, list(missing)

    def repository_from_context(context):
        if setup_error is not None:
            raise setup_error
        return object()

    monkeypatch.setattr(
        capture_module,
        "build_context",
        lambda *args, **kwargs: base_context,
    )
    monkeypatch.setattr(capture_module, "NagiosSystemMonitor", CliMonitor)
    monkeypatch.setattr(
        capture_module.OptionPriceRepository,
        "from_context",
        repository_from_context,
    )
    monkeypatch.setattr(capture_module, "create_tda_broker", lambda context: object())
    monkeypatch.setattr(capture_module, "TDAReferenceData", CliProvider)
    monkeypatch.setattr(capture_module, "OptionPriceCapture", CliCapture)
    result = capture_module.main(["--dry-run"])
    return result, base_context, provider_contexts


def test_capture_uses_risk_account_normalization_batches_exact_options_and_writes_once():
    broker = Broker()
    provider = Provider()
    repository = Repository()
    summary, missing = OptionPriceCapture(
        broker,
        provider,
        repository,
        {"risk.option_prices.market_timezone": "America/New_York"},
        now_fn=lambda: NOW,
    ).capture()

    assert broker.calls == 1
    assert broker.transaction_calls == [
        (
                "ACCOUNT-0001",
                datetime(2026, 8, 18),
                datetime(2026, 8, 18, 19, 30),
                "TRADE",
        )
    ]
    assert provider.calls == [[PROVIDER_CALL]]
    assert missing == []
    assert (summary.account_count, summary.position_count, summary.quote_count) == (
        1,
        1,
        1,
    )
    saved_summary, prices, chains = repository.saved[0]
    assert saved_summary == summary
    assert prices[0].position.canonical_occ_symbol == CALL
    assert prices[0].position.signed_quantity == 2
    assert prices[0].session_date.isoformat() == "2026-08-18"
    assert prices[0].provider_iv == pytest.approx(0.25)
    assert summary.status == "success"
    assert len(chains) == 1


def test_capture_adds_todays_closed_option_transactions_and_persists_zero_quantity():
    class TransactionBroker(Broker):
        def get_transactions(self, account, start, end, transaction_type):
            self.transaction_calls.append((account, start, end, transaction_type))
            return [
                {
                    "transactionId": "current-put",
                    "transactionDate": "2026-08-18T18:00:00-04:00",
                    "tradeDate": "2026-08-18",
                    "type": "TRADE",
                    "transferItems": [
                        {
                            "instrument": {
                                "symbol": PROVIDER_PUT,
                                "assetType": "OPTION",
                            }
                        },
                        {
                            "instrument": {
                                "symbol": PROVIDER_CALL,
                                "assetType": "OPTION",
                            }
                        },
                        {
                            "instrument": {
                                "symbol": "XYZ",
                                "assetType": "EQUITY",
                            }
                        },
                    ],
                },
                {
                    "transactionId": "previous-call",
                    "transactionDate": "2026-08-17T18:00:00-04:00",
                    "tradeDate": "2026-08-17",
                    "type": "TRADE",
                    "instrument": {
                        "symbol": "XYZ260918C00110000",
                        "assetType": "OPTION",
                    },
                },
            ]

    class TransactionProvider(Provider):
        def get_option_quotes(self, symbols):
            self.calls.append(symbols)
            return {
                PROVIDER_CALL: {
                    "mark": 5.0,
                    "quote_time": int(NOW.timestamp() * 1000),
                },
                PROVIDER_PUT: {
                    "mark": 2.5,
                    "quote_time": int(NOW.timestamp() * 1000),
                },
            }

    broker = TransactionBroker()
    provider = TransactionProvider()
    repository = Repository()
    summary, missing = OptionPriceCapture(
        broker,
        provider,
        repository,
        {
            "LocalTZ": "US/Pacific",
            "risk.option_prices.market_timezone": "America/New_York",
        },
        now_fn=lambda: NOW,
    ).capture()

    assert broker.transaction_calls == [
        (
            "ACCOUNT-0001",
            datetime(2026, 8, 17, 21),
            datetime(2026, 8, 18, 16, 30),
            "TRADE",
        )
    ]
    assert provider.calls == [[PROVIDER_CALL, PROVIDER_PUT]]
    assert missing == []
    assert (summary.position_count, summary.quote_count) == (2, 2)
    prices = repository.saved[0][1]
    prices_by_symbol = {item.position.canonical_occ_symbol: item for item in prices}
    assert prices_by_symbol[CALL].position.signed_quantity == 2
    assert prices_by_symbol[PUT].position.signed_quantity == 0
    assert prices_by_symbol[PUT].mark == pytest.approx(2.5)


def test_cli_reports_one_terminal_ok_event_with_option_price_service_suffix(
    monkeypatch,
):
    result, base_context, provider_contexts = _run_cli(monkeypatch)

    assert result == 0
    assert len(CliMonitor.instances) == 1
    monitor = CliMonitor.instances[0]
    assert monitor.context["nagios.service_id"] == (
        "ALGO_REFDATA-Refdata Update - Option Position Prices"
    )
    assert monitor.events == [
        (
            CliMonitor.MONITOR_OK,
                "Option position price capture completed successfully: "
                "run=capture-run accounts=1 positions=1 quotes=1 missing=0 "
                "chains=0 surface_points=0 dry_run=true",
        )
    ]
    assert base_context["nagios.service_id"] == "ALGO_REFDATA-Refdata Update"
    assert provider_contexts == [base_context]


def test_cli_reports_one_terminal_critical_event_for_missing_quotes(monkeypatch):
    result, _, _ = _run_cli(monkeypatch, missing=[CALL])

    assert result == 1
    assert len(CliMonitor.instances[0].events) == 1
    level, message = CliMonitor.instances[0].events[0]
    assert level == CliMonitor.MONITOR_CRITICAL
    assert "capture incomplete" in message
    assert "missing=1" in message
    assert "user intervention required" in message


def test_cli_describes_chain_failure_without_claiming_quotes_are_missing(
    monkeypatch, capsys
):
    result, _, _ = _run_cli(monkeypatch, chain_problem=True)

    captured = capsys.readouterr()
    assert result == 1
    assert "option-chain snapshots were incomplete" in captured.err
    assert "Missing current quotes" not in captured.err


def test_cli_reports_one_terminal_critical_event_for_processing_failure(monkeypatch):
    result, _, _ = _run_cli(monkeypatch, error=RuntimeError("provider unavailable"))

    assert result == 1
    assert len(CliMonitor.instances[0].events) == 1
    level, message = CliMonitor.instances[0].events[0]
    assert level == CliMonitor.MONITOR_CRITICAL
    assert "capture failed" in message
    assert "error=RuntimeError" in message
    assert "user intervention required" in message


def test_cli_reports_one_terminal_critical_event_for_setup_failure(monkeypatch):
    result, _, _ = _run_cli(
        monkeypatch,
        setup_error=RuntimeError("database configuration unavailable"),
    )

    assert result == 1
    assert len(CliMonitor.instances[0].events) == 1
    level, message = CliMonitor.instances[0].events[0]
    assert level == CliMonitor.MONITOR_CRITICAL
    assert "capture failed" in message
    assert "error=RuntimeError" in message


class SqlCursor:
    def __init__(self) -> None:
        self.executed = []
        self.executed_many = []
        self.lastrowid = None
        self._next_chain_snapshot_id = 100

    def execute(self, sql, parameters):
        self.executed.append((sql, parameters))
        if "INSERT INTO risk_option_chain_snapshots" in sql:
            self._next_chain_snapshot_id += 1
            self.lastrowid = self._next_chain_snapshot_id

    def executemany(self, sql, parameters):
        self.executed_many.append((sql, list(parameters)))


class SqlFactory:
    def __init__(self) -> None:
        self.cursor_instance = SqlCursor()
        self.transactions = 0

    @contextmanager
    def transaction(self, *, dictionary=True):
        self.transactions += 1
        yield object(), self.cursor_instance, object()


def test_repository_writes_capture_run_and_prices_in_one_direct_transaction():
    capture_repository = Repository()
    summary, _ = OptionPriceCapture(
        Broker(),
        Provider(),
        capture_repository,
        {"risk.option_prices.market_timezone": "America/New_York"},
        now_fn=lambda: NOW,
    ).capture()
    prices = capture_repository.saved[0][1]
    factory = SqlFactory()

    OptionPriceRepository(factory).save_capture(summary, prices)

    assert factory.transactions == 1
    assert len(factory.cursor_instance.executed) == 1
    assert len(factory.cursor_instance.executed_many) == 1
    price_sql, parameters = factory.cursor_instance.executed_many[0]
    assert "ON DUPLICATE KEY UPDATE" in price_sql
    assert len(parameters) == 1
    assert len(parameters[0]) == 33


def test_repository_persists_run_chain_surface_and_prices_with_shared_chain_id():
    capture_repository = Repository()
    summary, _ = OptionPriceCapture(
        Broker(),
        Provider(),
        capture_repository,
        {"risk.option_prices.market_timezone": "America/New_York"},
        now_fn=lambda: NOW,
    ).capture()
    _, prices, snapshots = capture_repository.saved[0]
    factory = SqlFactory()

    OptionPriceRepository(factory).save_capture(
        summary,
        prices,
        chain_snapshots=snapshots,
    )

    assert factory.transactions == 1
    assert len(factory.cursor_instance.executed) == 2
    assert len(factory.cursor_instance.executed_many) == 2
    surface_sql, surface_rows = factory.cursor_instance.executed_many[0]
    price_sql, price_rows = factory.cursor_instance.executed_many[1]
    assert "risk_option_surface_points" in surface_sql
    assert surface_rows[0][0] == 101
    assert "chain_snapshot_id=VALUES(chain_snapshot_id)" in price_sql
    assert price_rows[0][-2:] == (101, int(MarkSource.PROVIDER_MARK))


def test_repository_failure_rolls_back_the_single_capture_transaction():
    class FailingCursor(SqlCursor):
        def executemany(self, sql, parameters):
            raise RuntimeError("database down")

    class FailingFactory:
        def __init__(self):
            self.cursor_instance = FailingCursor()
            self.rolled_back = False

        @contextmanager
        def transaction(self, *, dictionary=True):
            try:
                yield object(), self.cursor_instance, object()
            except Exception:
                self.rolled_back = True
                raise

    capture_repository = Repository()
    summary, _ = OptionPriceCapture(
        Broker(),
        Provider(),
        capture_repository,
        {"risk.option_prices.market_timezone": "America/New_York"},
        now_fn=lambda: NOW,
    ).capture()
    _, prices, snapshots = capture_repository.saved[0]
    factory = FailingFactory()

    with pytest.raises(OptionPriceRepositoryUnavailable):
        OptionPriceRepository(factory).save_capture(
            summary,
            prices,
            chain_snapshots=snapshots,
        )
    assert factory.rolled_back is True


def test_repository_backfill_mode_preserves_an_existing_richer_session_row():
    capture_repository = Repository()
    summary, _ = OptionPriceCapture(
        Broker(),
        Provider(),
        capture_repository,
        {"risk.option_prices.market_timezone": "America/New_York"},
        now_fn=lambda: NOW,
    ).capture()
    prices = capture_repository.saved[0][1]
    factory = SqlFactory()

    OptionPriceRepository(factory).save_capture(
        summary, prices, preserve_existing=True
    )

    price_sql, _ = factory.cursor_instance.executed_many[0]
    assert "ON DUPLICATE KEY UPDATE id=id" in price_sql
    assert "provider_iv=VALUES(provider_iv)" not in price_sql


def test_repository_backfill_batch_writes_a_resumable_running_run_and_prices():
    capture_repository = Repository()
    summary, _ = OptionPriceCapture(
        Broker(),
        Provider(),
        capture_repository,
        {"risk.option_prices.market_timezone": "America/New_York"},
        now_fn=lambda: NOW,
    ).capture()
    prices = capture_repository.saved[0][1]
    factory = SqlFactory()

    OptionPriceRepository(factory).save_backfill_batch(
        summary, prices, final=False
    )

    assert factory.transactions == 1
    run_sql, run_parameters = factory.cursor_instance.executed[0]
    assert "ON DUPLICATE KEY UPDATE" in run_sql
    assert run_parameters[2] == "running"
    assert run_parameters[-1] is None
    price_sql, _ = factory.cursor_instance.executed_many[0]
    assert "ON DUPLICATE KEY UPDATE id=id" in price_sql


class HistoryRepository:
    def __init__(self, rows, *, fail=False):
        self.rows = rows
        self.fail = fail
        self.calls = []

    def query_history(self, symbols, *, start_date, end_date):
        self.calls.append((symbols, start_date, end_date))
        if self.fail:
            raise OptionPriceRepositoryUnavailable("database unavailable")
        return list(self.rows)


class Reference:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def option_market_history(self, payload):
        self.calls.append(payload)
        return {
            "data": list(self.rows),
            "missing_symbols": [],
            "meta": {"warnings": []},
        }


def point(session, mark, source):
    return {
        "requested_symbol": CALL,
        "canonical_occ_symbol": CALL,
        "at": f"{session}T00:00:00+00:00",
        "session_date": session,
        "status": "ok",
        "mark": mark,
        "last": mark,
        "source": source,
        "warnings": [],
    }


def history_request():
    return OptionMarketHistoryRequest(
        symbols=[CALL],
        start_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
        end_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        interval="1d",
        include_greeks=False,
    )


def test_hybrid_history_uses_sql_first_and_remote_only_when_a_weekday_gap_exists():
    database = HistoryRepository(
        [point("2026-08-17", 7.0, "risk_option_price_capture")]
    )
    remote = Reference(
        [
            point("2026-08-17", 99.0, "RemoteProvider"),
            point("2026-08-18", 8.0, "RemoteProvider"),
        ]
    )
    result = RiskOptionHistoryService(database, remote, now_fn=lambda: NOW).query(
        history_request()
    )

    assert len(remote.calls) == 1
    assert [
        (item.session_date.isoformat(), item.mark, item.source) for item in result.data
    ] == [
        ("2026-08-17", 7.0, "risk_option_price_capture"),
        ("2026-08-18", 8.0, "RemoteProvider"),
    ]
    assert result.meta.cache == "miss"


def test_hybrid_history_skips_remote_when_sql_has_complete_weekday_coverage():
    database = HistoryRepository(
        [
            point("2026-08-17", 7.0, "risk_option_price_capture"),
            point("2026-08-18", 8.0, "risk_option_price_capture"),
        ]
    )
    remote = Reference([])
    result = RiskOptionHistoryService(database, remote, now_fn=lambda: NOW).query(
        history_request()
    )

    assert remote.calls == []
    assert result.meta.cache == "hit"
    assert result.missing_symbols == []


def test_sql_failure_degrades_to_remote_history():
    database = HistoryRepository([], fail=True)
    remote = Reference([point("2026-08-18", 8.0, "RemoteProvider")])
    result = RiskOptionHistoryService(database, remote, now_fn=lambda: NOW).query(
        history_request()
    )
    assert len(remote.calls) == 1
    assert result.data[0].mark == 8.0
    assert "risk_option_price_repository_unavailable" in result.meta.warnings


def test_option_price_schema_is_idempotent_and_session_unique():
    schema = (
        Path(__file__).resolve().parents[1]
        / "sql"
        / "mysql"
        / "002_create_risk_option_prices.sql"
    ).read_text(encoding="utf-8")
    assert schema.count("CREATE TABLE IF NOT EXISTS") == 2
    assert "uq_risk_option_position_price_session" in schema


EXPIRY = date(2026, 9, 18)
CALENDAR_CALL = "XYZ261016C00100000"
PROVIDER_CALENDAR_CALL = "XYZ   261016C00100000"
NEIGHBOR_CALL = "XYZ260918C00105000"
PROVIDER_NEIGHBOR_CALL = "XYZ   260918C00105000"
PROVIDER_LOWER_CALL = "XYZ   260918C00095000"
PROVIDER_FARTHER_CALL = "XYZ   260918C00110000"


def _account(account_id, *symbols):
    return {
        "securitiesAccount": {
            "accountNumber": account_id,
            "positions": [
                {
                    "longQuantity": 1,
                    "shortQuantity": 0,
                    "instrument": {"symbol": symbol, "type": "OPTION"},
                }
                for symbol in symbols
            ],
        }
    }


def _chain_contract(symbol, *, quote_time=None, **fields):
    return {
        "symbol": symbol,
        "bid": 4.8,
        "ask": 5.2,
        "mark": 5.0,
        "last": 4.95,
        "quoteTimeInLong": quote_time or int(NOW.timestamp() * 1000),
        **fields,
    }


def _chain_response(
    *contracts,
    expiration=EXPIRY,
    spot=100.0,
    **fields,
):
    calls = [item for item in contracts if "C" in str(item.get("symbol"))[-9:-8]]
    puts = [item for item in contracts if "P" in str(item.get("symbol"))[-9:-8]]
    response = {
        "symbol": "XYZ",
        "underlyingPrice": spot,
        "callExpDateMap": {
            f"{expiration.isoformat()}:31": {"100.0": calls}
        },
        "putExpDateMap": {
            f"{expiration.isoformat()}:31": {"90.0": puts}
        },
    }
    return {**response, **fields}


class ChainBroker:
    def __init__(self, accounts, transactions=None):
        self.accounts = accounts
        self.transactions = transactions or {}
        self.transaction_calls = []

    def get_accounts(self):
        return self.accounts

    def get_transactions(self, account_id, start, end, transaction_type):
        self.transaction_calls.append((account_id, start, end, transaction_type))
        return list(self.transactions.get(account_id, ()))


class ChainProvider:
    def __init__(self, quotes, responses):
        self.quotes = quotes
        self.responses = responses
        self.exact_calls = []
        self.chain_calls = []

    def get_option_quotes(self, symbols):
        self.exact_calls.append(list(symbols))
        return {symbol: self.quotes[symbol] for symbol in symbols if symbol in self.quotes}

    def get_options_chain(self, symbol, **kwargs):
        self.chain_calls.append((symbol, kwargs))
        key = (symbol, kwargs["from_date"], kwargs["strike_count"])
        response = self.responses[key] if key in self.responses else self.responses[(symbol, kwargs["from_date"])]
        if isinstance(response, list):
            return response.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _quotes(*symbols):
    return {
        symbol: {
            "bid": 4.8,
            "ask": 5.2,
            "mark": 5.0,
            "last": 4.95,
            "quote_time": int(NOW.timestamp() * 1000),
        }
        for symbol in symbols
    }


def _capture_chain_fixture(accounts, quotes, responses, *, context=None, repository=None):
    repository = repository or Repository()
    provider = ChainProvider(quotes, responses)
    summary, missing = OptionPriceCapture(
        ChainBroker(accounts),
        provider,
        repository,
        {"risk.option_prices.market_timezone": "America/New_York", **(context or {})},
        now_fn=lambda: NOW,
    ).capture()
    return summary, missing, provider, repository


def test_vertical_shares_one_exact_expiration_chain_snapshot():
    summary, missing, provider, repository = _capture_chain_fixture(
        [_account("ACCOUNT-1", PROVIDER_CALL, PROVIDER_PUT)],
        _quotes(PROVIDER_CALL, PROVIDER_PUT),
        {
            ("XYZ", EXPIRY): _chain_response(
                _chain_contract(PROVIDER_CALL), _chain_contract(PROVIDER_PUT)
            )
        },
    )

    assert missing == []
    assert summary.status == "success"
    assert summary.chain_count == 1
    assert summary.surface_point_count == 2
    assert [call[0] for call in provider.chain_calls] == ["XYZ"]
    assert provider.chain_calls[0][1]["from_date"] == EXPIRY
    _, prices, snapshots = repository.saved[0]
    assert len(prices) == 2
    assert len(snapshots) == 1
    assert {point.canonical_occ_symbol for point in snapshots[0].points} == {CALL, PUT}


def test_calendar_and_shared_accounts_group_chain_requests_by_underlying_and_expiration():
    summary, _, provider, repository = _capture_chain_fixture(
        [
            _account("ACCOUNT-1", PROVIDER_CALL, PROVIDER_CALENDAR_CALL),
            _account("ACCOUNT-2", PROVIDER_CALL),
        ],
        _quotes(PROVIDER_CALL, PROVIDER_CALENDAR_CALL),
        {
            ("XYZ", EXPIRY): _chain_response(_chain_contract(PROVIDER_CALL)),
            ("XYZ", date(2026, 10, 16)): _chain_response(
                _chain_contract(PROVIDER_CALENDAR_CALL), expiration=date(2026, 10, 16)
            ),
        },
    )

    assert summary.chain_count == 2
    assert len(provider.chain_calls) == 2
    assert {call[1]["from_date"] for call in provider.chain_calls} == {
        EXPIRY,
        date(2026, 10, 16),
    }
    assert len(repository.saved[0][2]) == 2


def test_chain_normalization_deduplicates_symbols_and_ignores_unrequested_expirations():
    newer = int((NOW + timedelta(seconds=1)).timestamp() * 1000)
    response = _chain_response(
        _chain_contract(PROVIDER_CALL, quote_time=int(NOW.timestamp() * 1000), bid=1.0),
        _chain_contract(PROVIDER_CALL, quote_time=newer, bid=2.0),
        _chain_contract(PROVIDER_PUT),
        _chain_contract(PROVIDER_CALENDAR_CALL),
    )
    _, _, _, repository = _capture_chain_fixture(
        [_account("ACCOUNT-1", PROVIDER_CALL, PROVIDER_PUT)],
        _quotes(PROVIDER_CALL, PROVIDER_PUT),
        {("XYZ", EXPIRY): response},
    )

    snapshot = repository.saved[0][2][0]
    points = {point.canonical_occ_symbol: point for point in snapshot.points}
    assert set(points) == {CALL, PUT}
    assert points[CALL].bid == 2.0


def test_chain_strike_count_retries_and_exact_quote_fallback_marks_partial(caplog):
    response_without_target = _chain_response(
        _chain_contract(PROVIDER_LOWER_CALL),
        _chain_contract(PROVIDER_NEIGHBOR_CALL),
    )
    summary, _, provider, repository = _capture_chain_fixture(
        [_account("ACCOUNT-1", PROVIDER_CALL)],
        _quotes(PROVIDER_CALL),
        {
            ("XYZ", EXPIRY, 20): response_without_target,
            ("XYZ", EXPIRY, 40): response_without_target,
        },
        context={
            "risk.option_prices.chain.strike_count": 20,
            "risk.option_prices.chain.max_strike_count": 40,
        },
    )

    assert [call[1]["strike_count"] for call in provider.chain_calls] == [20, 40]
    snapshot = repository.saved[0][2][0]
    fallback = next(item for item in snapshot.points if item.canonical_occ_symbol == CALL)
    assert fallback.observation_source == int(
        SurfaceObservationSource.EXACT_OPTION_QUOTE_FALLBACK
    )
    assert snapshot.snapshot_status == int(ChainSnapshotStatus.PARTIAL)
    assert snapshot.error_code == "target_quote_fallback"
    assert summary.status == "partial"
    assert (
        "Option-chain snapshot incomplete: underlying=XYZ expiration=2026-09-18 "
        "status=partial error_code=target_quote_fallback flags=none points=3 "
        "targets=1 affected_targets=XYZ260918C00100000"
    ) in caplog.messages


def test_chain_expands_past_normal_maximum_for_target_outside_strike_range():
    summary, missing, provider, repository = _capture_chain_fixture(
        [_account("ACCOUNT-1", PROVIDER_CALL)],
        _quotes(PROVIDER_CALL),
        {
            ("XYZ", EXPIRY, 20): _chain_response(
                _chain_contract(PROVIDER_FARTHER_CALL)
            ),
            ("XYZ", EXPIRY, 40): _chain_response(
                _chain_contract(PROVIDER_NEIGHBOR_CALL)
            ),
            ("XYZ", EXPIRY, 80): _chain_response(
                _chain_contract(PROVIDER_CALL),
                _chain_contract(PROVIDER_NEIGHBOR_CALL),
            ),
        },
        context={
            "risk.option_prices.chain.strike_count": 20,
            "risk.option_prices.chain.max_strike_count": 40,
        },
    )

    assert missing == []
    assert [call[1]["strike_count"] for call in provider.chain_calls] == [20, 40, 80]
    snapshot = repository.saved[0][2][0]
    assert snapshot.snapshot_status == int(ChainSnapshotStatus.SUCCESS)
    assert snapshot.error_code is None
    assert summary.status == "success"


def test_adaptive_chain_expansion_stops_when_provider_range_does_not_widen():
    unchanged_response = _chain_response(
        _chain_contract(PROVIDER_NEIGHBOR_CALL)
    )
    summary, _, provider, repository = _capture_chain_fixture(
        [_account("ACCOUNT-1", PROVIDER_CALL)],
        _quotes(PROVIDER_CALL),
        {
            ("XYZ", EXPIRY, 20): unchanged_response,
            ("XYZ", EXPIRY, 40): unchanged_response,
        },
        context={
            "risk.option_prices.chain.strike_count": 20,
            "risk.option_prices.chain.max_strike_count": 20,
        },
    )

    assert [call[1]["strike_count"] for call in provider.chain_calls] == [20, 40]
    snapshot = repository.saved[0][2][0]
    assert snapshot.snapshot_status == int(ChainSnapshotStatus.PARTIAL)
    assert snapshot.error_code == "target_quote_fallback"
    assert summary.status == "partial"


def test_failed_chain_and_missing_spot_are_partial_without_losing_exact_prices(caplog):
    marker = "SYNTHETIC_CHAIN_CREDENTIAL"
    failed_summary, missing, _, failed_repository = _capture_chain_fixture(
        [_account("ACCOUNT-1", PROVIDER_CALL)],
        _quotes(PROVIDER_CALL),
        {("XYZ", EXPIRY): RuntimeError(f"provider unavailable api_key={marker}")},
    )
    failed_snapshot = failed_repository.saved[0][2][0]
    assert missing == []
    assert failed_summary.quote_count == 1
    assert failed_summary.status == "partial"
    assert failed_snapshot.snapshot_status == int(ChainSnapshotStatus.FAILED)
    assert failed_snapshot.error_code == "provider_request_failed"
    assert (
        "Option-chain request failed: underlying=XYZ expiration=2026-09-18 "
        "strike_count=20 error=RuntimeError"
    ) in caplog.messages
    assert marker not in caplog.text
    assert (
        "Option-chain snapshot incomplete: underlying=XYZ expiration=2026-09-18 "
        "status=failed error_code=provider_request_failed flags=none points=0 "
        "targets=1 affected_targets=XYZ260918C00100000"
    ) in caplog.messages

    no_spot_summary, _, _, no_spot_repository = _capture_chain_fixture(
        [_account("ACCOUNT-1", PROVIDER_CALL)],
        _quotes(PROVIDER_CALL),
        {("XYZ", EXPIRY): _chain_response(_chain_contract(PROVIDER_CALL), spot=None)},
    )
    no_spot_snapshot = no_spot_repository.saved[0][2][0]
    assert no_spot_summary.status == "partial"
    assert no_spot_snapshot.spot is None
    assert no_spot_snapshot.error_code == "spot_missing"


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ({"mark": 1.0}, MarkSource.PROVIDER_MARK),
        ({"bid": 1.0, "ask": 3.0}, MarkSource.BID_ASK_MIDPOINT),
        ({"last": 2.0}, MarkSource.LAST_TRADE),
        ({"close": 2.0}, MarkSource.CLOSE),
    ],
)
def test_exact_mark_selection_persists_its_source(quote, expected):
    summary, _, _, repository = _capture_chain_fixture(
        [_account("ACCOUNT-1", PROVIDER_CALL)],
        {PROVIDER_CALL: quote},
        {("XYZ", EXPIRY): _chain_response(_chain_contract(PROVIDER_CALL))},
    )
    assert summary.status == "success"
    assert repository.saved[0][1][0].mark_source == int(expected)


def test_dry_run_and_transaction_watermark_do_not_write_and_filter_locally():
    class WatermarkRepository(Repository):
        def latest_successful_capture_at(self, **kwargs):
            self.watermark_arguments = kwargs
            return datetime(2026, 8, 18, 20, tzinfo=timezone.utc)

    transactions = {
        "ACCOUNT-1": [
            {
                "transactionId": "before",
                "type": "TRADE",
                "transactionDate": "2026-08-18T19:00:00+00:00",
                "instrument": {"symbol": PROVIDER_PUT, "assetType": "OPTION"},
            },
            {
                "transactionId": "after",
                "type": "TRADE",
                "transactionDate": "2026-08-18T21:00:00+00:00",
                "instrument": {"symbol": PROVIDER_PUT, "assetType": "OPTION"},
            },
            {
                "transactionId": "future",
                "type": "TRADE",
                "transactionDate": "2026-08-19T01:00:00+00:00",
                "instrument": {"symbol": PROVIDER_CALENDAR_CALL, "assetType": "OPTION"},
            },
        ]
    }
    repository = WatermarkRepository()
    provider = ChainProvider(
        _quotes(PROVIDER_CALL, PROVIDER_PUT),
        {("XYZ", EXPIRY): _chain_response(_chain_contract(PROVIDER_CALL), _chain_contract(PROVIDER_PUT))},
    )
    broker = ChainBroker([_account("ACCOUNT-1", PROVIDER_CALL)], transactions)
    summary, missing = OptionPriceCapture(
        broker,
        provider,
        repository,
        {"risk.option_prices.market_timezone": "America/New_York"},
        now_fn=lambda: NOW,
    ).capture(dry_run=True)

    assert summary.position_count == 2
    assert missing == []
    assert repository.saved == []
    assert repository.watermark_arguments["minimum_account_count"] == 1
    assert broker.transaction_calls[0][1] == datetime(2026, 8, 18, 16)
    assert broker.transaction_calls[0][2] == datetime(2026, 8, 18, 19, 30)


def test_expanded_schema_contains_only_compact_chain_storage():
    schema = (
        Path(__file__).resolve().parents[1]
        / "sql"
        / "mysql"
        / "003_expand_risk_option_daily_chain_capture.sql"
    ).read_text(encoding="utf-8")
    normalized = schema.lower()
    assert "create table risk_option_chain_snapshots" in normalized
    assert "create table risk_option_surface_points" in normalized
    assert "primary key (chain_snapshot_id, canonical_occ_symbol)" in normalized
    assert "ix_risk_option_surface_contract_history" in normalized
    assert "(run_id, underlying_symbol, expiration_date)" in normalized
    assert normalized.count("alter table") == 2
    assert "interest_rate" not in normalized
    assert "dividend_yield" not in normalized
    assert "provider_iv" not in normalized
    assert "json" not in normalized
    assert "contract_reference" not in normalized
