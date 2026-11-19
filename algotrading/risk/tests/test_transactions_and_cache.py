from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import inspect

import pytest

from algotrading.tda.TDABrokerInterface import TDABrokerInterface
from ..cache import PersistentRiskCache
from ..transactions.normalization import normalize_transaction
from ..transactions.repository import TransactionRepository
from ..transactions.sync import TransactionSynchronizer
from ..transactions import sync as sync_module


NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)


def test_transaction_method_signature():
    signature = inspect.signature(TDABrokerInterface.get_transactions)
    assert list(signature.parameters) == ["self", "account_number", "start_date", "end_date", "types", "symbol"]
    assert signature.parameters["symbol"].default is None


class _CapturingBroker:
    def __init__(self):
        self.calls = []

    def get_transactions(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return []


def test_sync_passes_naive_local_wallclock_timestamps_to_broker():
    broker = _CapturingBroker()
    sync = TransactionSynchronizer(
        broker,
        _NoWriteRepository(),
        {"LocalTZ": "US/Pacific", "risk.transactions.sync.chunk_days": 365},
    )
    sync.sync(
        accounts=["TEST"],
        transaction_types=["TRADE"],
        start_at=datetime(2026, 6, 15, 7, tzinfo=timezone.utc),
        through=datetime(2026, 8, 17, 7, tzinfo=timezone.utc),
        dry_run=True,
    )

    (account, start, end, transaction_type), kwargs = broker.calls[0]
    assert (account, transaction_type, kwargs) == ("TEST", "TRADE", {"symbol": None})
    assert start == datetime(2026, 6, 15)
    assert end == datetime(2026, 8, 17)
    assert start.tzinfo is None and end.tzinfo is None


def test_transaction_normalization_preserves_decimal_signs_items_and_broker_identity():
    raw = {"activityId": "test-transaction", "transactionDate": "2026-08-16T12:30:00-04:00", "type": "TRADE",
           "amount": "-100.25", "fees": {"commission": "-1.25", "other": "-0.25"},
           "transferItems": [{"amount": "2", "price": "50.125", "instrument": {"symbol": "spy", "assetType": "EQUITY"}},
                             {"amount": "-1", "price": "10.00", "instrument": {"symbol": "spy", "assetType": "EQUITY"}}]}
    first = normalize_transaction(raw, account_id="TEST", requested_type="TRADE")
    second = normalize_transaction(dict(raw), account_id="TEST", requested_type="TRADE")
    assert first.identifier_method == "broker_id"
    assert first.broker_transaction_id == second.broker_transaction_id
    assert first.amount == Decimal("-100.25")
    assert first.fees == Decimal("-1.50")
    assert first.net_amount == Decimal("-101.75")
    assert len(first.items) == 2 and first.items[0].quantity == Decimal("2")
    assert first.fees_by_type == ()
    assert first.transaction_at == datetime(2026, 8, 16, 16, 30, tzinfo=timezone.utc)


def test_transaction_normalization_extracts_transfer_item_fee_components():
    raw = {
        "activityId": "fee-components",
        "time": "2026-07-30T19:50:28+0000",
        "type": "TRADE",
        "netAmount": "7640.23",
        "transferItems": [
            {"feeType": "COMMISSION", "amount": "3.50", "cost": "-3.50"},
            {"feeType": "SEC_FEE", "amount": "0.16", "cost": "-0.16"},
            {"feeType": "OPT_REG_FEE", "amount": "0.09", "cost": "-0.09"},
            {"feeType": "TAF_FEE", "amount": "0.02", "cost": "-0.02"},
            {"feeType": "TAF_FEE", "amount": "0.00", "cost": "0.00"},
            {
                "amount": "-7",
                "price": "10.92",
                "instrument": {"symbol": "AAPL  260821P00335000", "assetType": "OPTION"},
            },
        ],
    }

    transaction = normalize_transaction(raw, account_id="TEST", requested_type="TRADE")

    assert transaction.fees == Decimal("-3.77")
    assert [(fee.fee_index, fee.fee_type, fee.amount) for fee in transaction.fees_by_type] == [
        (0, "COMMISSION", Decimal("3.50")),
        (1, "SEC_FEE", Decimal("0.16")),
        (2, "OPT_REG_FEE", Decimal("0.09")),
        (3, "TAF_FEE", Decimal("0.02")),
    ]
    assert [(item.item_index, item.symbol, item.asset_type) for item in transaction.items] == [
        (5, "AAPL  260821P00335000", "OPTION"),
    ]
    assert (transaction.symbol, transaction.asset_type) == ("AAPL  260821P00335000", "OPTION")


def test_transaction_normalization_requires_a_broker_identifier():
    with pytest.raises(ValueError, match="broker transaction ID is required"):
        normalize_transaction(
            {"time": "2026-07-30T19:50:28+0000", "type": "TRADE"},
            account_id="TEST",
            requested_type="TRADE",
        )


class _RecordingCursor:
    def __init__(self):
        self.executed = []
        self.executemany_calls = []
        self.lastrowid = 0

    def execute(self, operation, params=None):
        self.executed.append((operation, params))
        if "INSERT INTO risk_transactions" in operation:
            self.lastrowid = 42

    def executemany(self, operation, params):
        self.executemany_calls.append((operation, list(params)))

    def fetchone(self):
        return None


class _RecordingConnectionFactory:
    def __init__(self):
        self.cursor = _RecordingCursor()

    @contextmanager
    def transaction(self, **_kwargs):
        yield None, self.cursor, None


def test_upsert_persists_fee_components_in_child_table():
    transaction = normalize_transaction(
        {
            "activityId": "fee-upsert",
            "time": "2026-07-30T19:50:28+0000",
            "type": "TRADE",
            "transferItems": [
                {"feeType": "COMMISSION", "amount": "3.50"},
                {"feeType": "SEC_FEE", "amount": "0.16"},
                {"feeType": "TAF_FEE", "amount": "0.00"},
                {"amount": "-7", "instrument": {"symbol": "AAPL", "assetType": "OPTION"}},
            ],
        },
        account_id="TEST",
        requested_type="TRADE",
    )
    factory = _RecordingConnectionFactory()
    repository = TransactionRepository(factory, cursor_secret=b"test")

    inserted, updated = repository.upsert_chunk(
        [transaction],
    )

    assert (inserted, updated) == (1, 0)
    assert any(
        "DELETE FROM risk_transaction_fees" in operation and params == (42,)
        for operation, params in factory.cursor.executed
    )
    fee_insert = next(
        values
        for operation, values in factory.cursor.executemany_calls
        if "INSERT INTO risk_transaction_fees" in operation
    )
    assert fee_insert == [
        (42, 0, "COMMISSION", Decimal("3.50")),
        (42, 1, "SEC_FEE", Decimal("0.16")),
    ]


class _ReadCursor:
    def __init__(self):
        self.executed = []
        self.operation = ""

    def execute(self, operation, params=None):
        self.operation = operation
        self.executed.append((operation, params))

    def fetchall(self):
        if "FROM risk_transactions t WHERE" in self.operation:
            return [{
                "id": 42,
                "broker_transaction_id": "broker-fees",
                "transaction_at": NOW - timedelta(hours=1),
                "transaction_type": "TRADE",
                "transaction_subtype": None,
                "description": "test trade",
                "symbol": "AAPL",
                "asset_type": "EQUITY",
                "trade_date": None,
                "settlement_date": None,
                "amount": Decimal("-100"),
                "fees": Decimal("-3.66"),
                "net_amount": Decimal("-103.66"),
                "currency": "USD",
                "order_id": "order-1",
                "source_updated_at": None,
                "ingested_at": NOW,
            }]
        if "FROM risk_transaction_items" in self.operation:
            return [{
                "transaction_id": 42,
                "item_index": 4,
                "symbol": "AAPL",
                "asset_type": "EQUITY",
                "instruction": "BUY",
                "position_effect": "OPENING",
                "quantity": Decimal("1"),
                "price": Decimal("100"),
                "amount": Decimal("-100"),
                "cost": None,
                "currency": "USD",
            }]
        if "FROM risk_transaction_fees" in self.operation:
            return [
                {"transaction_id": 42, "fee_index": 0, "fee_type": "COMMISSION", "amount": Decimal("3.50")},
                {"transaction_id": 42, "fee_index": 1, "fee_type": "SEC_FEE", "amount": Decimal("0.16")},
            ]
        if "GROUP BY transaction_type" in self.operation:
            return [{
                "transaction_type": "TRADE",
                "first_transaction_at": NOW - timedelta(hours=1),
                "last_transaction_at": NOW - timedelta(hours=1),
                "last_ingested_at": NOW,
            }]
        return []


class _ReadConnectionFactory:
    def __init__(self):
        self.cursor_instance = _ReadCursor()

    @contextmanager
    def cursor(self, **_kwargs):
        yield None, self.cursor_instance, None


def test_transaction_query_returns_component_fee_records_from_the_new_table():
    factory = _ReadConnectionFactory()
    repository = TransactionRepository(factory, cursor_secret=b"test", now_fn=lambda: NOW)

    response = repository.query(
        "TEST",
        start_at=NOW - timedelta(days=1),
        end_at=NOW,
        types=["TRADE"],
    )

    assert [(fee.fee_type, fee.amount) for fee in response.data[0].fee_components] == [
        ("COMMISSION", Decimal("3.50")),
        ("SEC_FEE", Decimal("0.16")),
    ]
    assert any(
        "FROM risk_transaction_fees" in operation
        for operation, _params in factory.cursor_instance.executed
    )


class _NoWriteRepository:
    def __getattr__(self, name):
        if name in {"last_sync_through", "last_transaction_at"}: return lambda *_: None
        raise AssertionError(f"repository write attempted: {name}")


class _WatermarkRepository:
    def __init__(self, watermark):
        self.watermark = watermark
        self.lookups = []
        self.upserted = []
        self.recorded = []

    def last_sync_through(self, account, transaction_type):
        self.lookups.append((account, transaction_type))
        return self.watermark

    def last_transaction_at(self, *_args):
        raise AssertionError("stored transaction time is not a feed watermark")

    @contextmanager
    def advisory_lock(self, *_args):
        yield

    def upsert_chunk(self, rows):
        self.upserted.append(rows)
        return 0, 0

    def record_sync_success(self, account, transaction_type, start, through):
        self.recorded.append((account, transaction_type, start, through))


def test_sync_resumes_from_latest_stored_transaction_timestamp():
    broker = _CapturingBroker()
    watermark = NOW - timedelta(hours=1)
    repository = _WatermarkRepository(watermark)
    sync = TransactionSynchronizer(
        broker,
        repository,
        {"LocalTZ": "UTC", "risk.transactions.sync.overlap_minutes": 10},
        now_fn=lambda: NOW,
    )

    results, failures = sync.sync(
        accounts=["TEST"], transaction_types=["TRADE"], through=NOW
    )

    (account, start, end, transaction_type), kwargs = broker.calls[0]
    assert (account, transaction_type, kwargs) == ("TEST", "TRADE", {"symbol": None})
    assert start == (watermark - timedelta(minutes=10)).replace(tzinfo=None)
    assert end == NOW.replace(tzinfo=None)
    assert repository.lookups == [("TEST", "TRADE")]
    assert len(repository.upserted) == 1
    assert repository.recorded == [
        ("TEST", "TRADE", watermark - timedelta(minutes=10), NOW)
    ]
    assert results[("TEST", "TRADE")].rows_received == 0
    assert failures == []


def test_daily_feed_coverage_does_not_require_intraday_transactions():
    now = datetime(2026, 8, 21, 21, tzinfo=timezone.utc)  # Friday 2 PM Pacific
    repository = TransactionRepository(
        _ReadConnectionFactory(),
        cursor_secret=b"test",
        now_fn=lambda: now,
        feed_timezone="America/Los_Angeles",
        daily_feed_complete_hour=20,
    )
    coverage = repository._coverage(
        ("TRADE",),
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        now,
        [
            {
                "transaction_type": "TRADE",
                "first_synced_at": datetime(2026, 7, 1, tzinfo=timezone.utc),
                "synced_through_at": datetime(2026, 8, 21, 3, tzinfo=timezone.utc),
                "last_success_at": datetime(2026, 8, 21, 3, 5, tzinfo=timezone.utc),
            }
        ],
    )

    assert coverage.complete is True
    assert coverage.missing_intervals == []


def test_daily_feed_coverage_warns_only_when_completed_session_is_stale():
    now = datetime(2026, 8, 21, 21, tzinfo=timezone.utc)
    repository = TransactionRepository(
        _ReadConnectionFactory(),
        cursor_secret=b"test",
        now_fn=lambda: now,
        feed_timezone="America/Los_Angeles",
    )
    coverage = repository._coverage(
        ("TRADE",),
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        now,
        [
            {
                "transaction_type": "TRADE",
                "first_synced_at": datetime(2026, 7, 1, tzinfo=timezone.utc),
                "synced_through_at": datetime(2026, 8, 20, 3, tzinfo=timezone.utc),
                "last_success_at": datetime(2026, 8, 20, 3, 5, tzinfo=timezone.utc),
            }
        ],
    )

    assert coverage.complete is False
    assert coverage.missing_intervals == ["TRADE:daily_feed_stale"]


class _PlaceholderBroker:
    def get_transactions(self, *_args, **_kwargs): raise NotImplementedError("owner action")


def test_placeholder_failure_and_dry_run_leave_repository_untouched():
    sync = TransactionSynchronizer(_PlaceholderBroker(), _NoWriteRepository(), {"risk.transactions.sync.initial_lookback_days": 1}, now_fn=lambda: NOW)
    with pytest.raises(NotImplementedError):
        sync.sync(accounts=["TEST"], transaction_types=["TRADE"])


def test_cli_reports_owner_action_and_exits_two_without_writes(monkeypatch, capsys):
    repository = _NoWriteRepository()
    monkeypatch.setattr(sync_module, "build_context", lambda *_args, **_kwargs: {"risk.transactions.sync.initial_lookback_days": 1})
    monkeypatch.setattr(sync_module.TransactionRepository, "from_context", lambda _context: repository)
    monkeypatch.setattr(sync_module, "create_tda_broker", lambda _context: _PlaceholderBroker())
    code = sync_module.main(["--account", "TEST", "--type", "TRADE"])
    captured = capsys.readouterr()
    assert code == 2
    assert "must be implemented by the owner" in captured.err


class _DiscoveringBroker(_CapturingBroker):
    def __init__(self):
        super().__init__()
        self.discovery_calls = 0

    def get_accounts(self):
        self.discovery_calls += 1
        return [
            {"securitiesAccount": {"accountNumber": "FIRST-0001"}},
            {"securitiesAccount": {"accountNumber": "SECOND-0002"}},
            {"securitiesAccount": {"accountNumber": "FIRST-0001"}},
        ]


def test_cli_discovers_and_updates_all_accounts_when_account_is_omitted(monkeypatch, capsys):
    broker = _DiscoveringBroker()
    context = {"LocalTZ": "US/Pacific", "risk.transactions.sync.chunk_days": 365}
    monkeypatch.setattr(sync_module, "build_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(sync_module.TransactionRepository, "from_context", lambda _context: _NoWriteRepository())
    monkeypatch.setattr(sync_module, "create_tda_broker", lambda _context: broker)

    code = sync_module.main(
        ["--type", "TRADE", "--start-at", "2026-08-16T00:00:00-07:00", "--through", "2026-08-17T00:00:00-07:00", "--dry-run"]
    )

    assert code == 0
    assert broker.discovery_calls == 1
    assert [call[0][0] for call in broker.calls] == ["FIRST-0001", "SECOND-0002"]
    output = capsys.readouterr().out
    assert "FIRST-0001" not in output and "SECOND-0002" not in output
    assert "account=***0001" in output and "account=***0002" in output


def test_cli_explicit_account_skips_discovery(monkeypatch):
    broker = _DiscoveringBroker()
    context = {"LocalTZ": "US/Pacific", "risk.transactions.sync.chunk_days": 365}
    monkeypatch.setattr(sync_module, "build_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(sync_module.TransactionRepository, "from_context", lambda _context: _NoWriteRepository())
    monkeypatch.setattr(sync_module, "create_tda_broker", lambda _context: broker)

    code = sync_module.main(
        ["--account", "EXPLICIT-0003", "--type", "TRADE", "--start-at", "2026-08-16T00:00:00-07:00", "--through", "2026-08-17T00:00:00-07:00", "--dry-run"]
    )

    assert code == 0
    assert broker.discovery_calls == 0
    assert broker.calls[0][0][0] == "EXPLICIT-0003"


def test_persistent_cache_checksum_eviction_restart_and_failure_ledger(tmp_path):
    path = tmp_path / "risk.sqlite"
    cache = PersistentRiskCache(path, max_entries=1)
    args = dict(namespace="test", request_fingerprint="fp", source_as_of=NOW,
                fetched_at=NOW, soft_expires_at=NOW+timedelta(seconds=60), hard_expires_at=NOW+timedelta(minutes=5))
    cache.put("a", payload={"value": 1}, **args)
    cache.put("b", payload={"value": 2}, **args)
    assert cache.get("a") is None
    assert cache.get("b")["payload"] == {"value": 2}
    cache.record_attempt("provider:symbol", at=NOW, success=False, error="token\nsecret")
    restarted = PersistentRiskCache(path, max_entries=1)
    assert restarted.get("b")["payload"] == {"value": 2}
    state = restarted.attempt_state("provider:symbol")
    assert state["last_success_at"] is None
    assert "\n" not in state["last_error"]
