from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone

import pytest

from algotrading.risk.option_prices.repository import (
    OptionPriceRepository,
    OptionPriceRepositoryUnavailable,
    OptionSurfaceHistoryLimitExceeded,
)


CALL = "XYZ260918C00100000"
PROVIDER_CALL = "XYZ   260918C00100000"
CALENDAR = "XYZ261016C00100000"
START = datetime(2026, 8, 17, tzinfo=timezone.utc)
END = datetime(2026, 8, 19, tzinfo=timezone.utc)


class Cursor:
    def __init__(self, headers, points, *, fail=False):
        self.headers = headers
        self.points = points
        self.fail = fail
        self.executed = []
        self.current = []

    def execute(self, sql, parameters):
        if self.fail:
            raise RuntimeError("database unavailable")
        self.executed.append((sql, parameters))
        if "FROM risk_option_chain_snapshots" in sql:
            self.current = list(self.headers)
        else:
            ids = set(parameters[:-1])
            self.current = [
                row for row in self.points if row["chain_snapshot_id"] in ids
            ]

    def fetchall(self):
        return list(self.current)


class Factory:
    def __init__(self, cursor):
        self.cursor_instance = cursor

    @contextmanager
    def cursor(self, *, dictionary=True):
        yield object(), self.cursor_instance, object()


def header(snapshot_id, captured_at, expiration):
    return {
        "chain_snapshot_id": snapshot_id,
        "run_id": f"run-{snapshot_id}",
        "captured_at": captured_at,
        "underlying_symbol": "XYZ",
        "expiration_date": expiration,
        "spot": 100,
        "spot_source": 1,
        "underlying_quote_time": captured_at,
        "snapshot_status": 1,
        "snapshot_flags": 0,
        "error_code": None,
    }


def point(snapshot_id, symbol):
    return {
        "chain_snapshot_id": snapshot_id,
        "canonical_occ_symbol": symbol,
        "bid": 4.9,
        "ask": 5.1,
        "mark": 5,
        "last": 4.95,
        "quote_time": datetime(2026, 8, 17, 23),
        "trade_time": None,
        "observation_source": 1,
        "security_status": 1,
    }


def test_surface_history_query_is_parameterized_batched_deterministic_and_utc():
    cursor = Cursor(
        [
            header(10, datetime(2026, 8, 17, 23), date(2026, 9, 18)),
            header(11, datetime(2026, 8, 18, 23), date(2026, 10, 16)),
        ],
        [point(10, CALL), point(10, "XYZ260918C00105000"), point(11, CALENDAR)],
    )
    repository = OptionPriceRepository(
        Factory(cursor), surface_point_batch_size=1
    )

    result = repository.query_surface_history(
        [PROVIDER_CALL, CALL, CALENDAR], start_at=START, end_at=END
    )

    assert [item.header.snapshot_id for item in result] == [10, 11]
    assert result[0].header.captured_at.tzinfo == timezone.utc
    assert result[0].observations[0].quote_time.tzinfo == timezone.utc
    assert [item.canonical_occ_symbol for item in result[0].observations] == [
        CALL,
        "XYZ260918C00105000",
    ]
    header_sql, header_parameters = cursor.executed[0]
    assert "JOIN risk_option_price_capture_runs" in header_sql
    assert "run.captured_at >= %s" in header_sql
    assert "run.captured_at < %s" in header_sql
    assert "ORDER BY run.captured_at" in header_sql
    assert CALL not in header_sql and "XYZ" not in header_sql
    assert header_parameters[:4] == (
        "XYZ",
        date(2026, 9, 18),
        "XYZ",
        date(2026, 10, 16),
    )
    point_queries = cursor.executed[1:]
    assert len(point_queries) == 2
    assert all(" IN (%s)" in sql for sql, _ in point_queries)
    assert all(len(parameters) == 2 for _, parameters in point_queries)


def test_surface_history_limits_and_database_failures_are_typed():
    too_many = Cursor(
        [
            header(10, datetime(2026, 8, 17, 23), date(2026, 9, 18)),
            header(11, datetime(2026, 8, 18, 23), date(2026, 9, 18)),
        ],
        [],
    )
    repository = OptionPriceRepository(
        Factory(too_many), maximum_surface_snapshots=1
    )
    with pytest.raises(OptionSurfaceHistoryLimitExceeded):
        repository.query_surface_history([CALL], start_at=START, end_at=END)

    failing = OptionPriceRepository(Factory(Cursor([], [], fail=True)))
    with pytest.raises(OptionPriceRepositoryUnavailable):
        failing.query_surface_history([CALL], start_at=START, end_at=END)


def test_surface_point_limit_stops_a_large_slice_before_assembly():
    cursor = Cursor(
        [header(10, datetime(2026, 8, 17, 23), date(2026, 9, 18))],
        [point(10, CALL), point(10, "XYZ260918C00105000")],
    )
    repository = OptionPriceRepository(
        Factory(cursor), maximum_surface_points=1
    )

    with pytest.raises(OptionSurfaceHistoryLimitExceeded):
        repository.query_surface_history([CALL], start_at=START, end_at=END)
