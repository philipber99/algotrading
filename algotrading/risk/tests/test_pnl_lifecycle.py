from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from algotrading.risk.market_inputs.client import (
    ReferenceDataClient,
    ReferenceDataClientError,
)
from algotrading.risk.pnl.analyzer import analyze_selection
from algotrading.risk.pnl.models import (
    PnlAnalyzerRequest,
    PnlExitRule,
    PnlTradeQueryRequest,
)
from algotrading.risk.pnl.reconstruction import (
    allocate_parent_fees,
    lifecycle_status,
    normalize_execution_legs,
    recognize_structure,
    reconstruct_lifecycles,
)
from algotrading.risk.pnl.valuation import (
    MarketMarkBook,
    PnlMarkHistoryUnavailable,
    _sectors,
    query_trades,
)
from algotrading.risk.transactions.models import (
    TransactionCoverage,
    TransactionFeeModel,
    TransactionItemModel,
    TransactionModel,
)


UTC = timezone.utc
START = datetime(2026, 7, 1, tzinfo=UTC)
END = datetime(2026, 7, 4, tzinfo=UTC)
OPTION_100 = "XYZ260918C00100000"
OPTION_105 = "XYZ260918C00105000"
OPTION_110 = "XYZ261218C00110000"


def tx(
    row_id: int,
    at: datetime,
    legs: list[tuple[str, str, str, str, float, float]],
    *,
    fee: float = 0.0,
    order_id: str | None = None,
) -> TransactionModel:
    return TransactionModel(
        id=row_id,
        broker_transaction_id=f"broker-{row_id}",
        transaction_at=at,
        transaction_type="TRADE",
        description=f"transaction {row_id}",
        fees=Decimal(str(fee)),
        order_id=order_id,
        ingested_at=at + timedelta(minutes=1),
        items=[
            TransactionItemModel(
                item_index=index,
                symbol=symbol,
                asset_type=asset,
                instruction=instruction,
                position_effect=effect,
                quantity=Decimal(str(quantity)),
                price=Decimal(str(price)),
                currency="USD",
            )
            for index, (symbol, asset, instruction, effect, quantity, price) in enumerate(legs)
        ],
    )


def reconstruct(rows: list[TransactionModel], account: str = "A"):
    executions, warnings = normalize_execution_legs(account, rows)
    assert not warnings
    return reconstruct_lifecycles(executions)


def test_equity_round_trip_and_flip_through_zero_are_deterministic():
    round_trip = reconstruct(
        [
            tx(1, START + timedelta(hours=1), [("XYZ", "EQUITY", "BUY", "OPENING", 10, 100)]),
            tx(2, START + timedelta(days=1), [("XYZ", "EQUITY", "SELL", "CLOSING", 10, 110)]),
        ]
    )
    repeated = reconstruct(
        [
            tx(1, START + timedelta(hours=1), [("XYZ", "EQUITY", "BUY", "OPENING", 10, 100)]),
            tx(2, START + timedelta(days=1), [("XYZ", "EQUITY", "SELL", "CLOSING", 10, 110)]),
        ]
    )
    assert len(round_trip) == 1 and round_trip[0].is_closed
    assert recognize_structure(round_trip[0]) == "XYZ"
    assert round_trip[0].trade_id == repeated[0].trade_id

    flipped = reconstruct(
        [
            tx(10, START, [("XYZ", "EQUITY", "BUY", "OPENING", 10, 100)]),
            tx(11, START + timedelta(days=1), [("XYZ", "EQUITY", "SELL", "CLOSING", 15, 101)], fee=-3),
        ]
    )
    assert len(flipped) == 2
    assert flipped[0].is_closed
    assert flipped[1].inventory == {"XYZ": -5.0}
    assert sum(item.fees_usd for lifecycle in flipped for item in lifecycle.executions) == pytest.approx(-3)


def test_vertical_calendar_partial_roll_and_cross_account_isolation():
    vertical = reconstruct(
        [
            tx(20, START, [(OPTION_100, "OPTION", "BUY", "OPENING", 1, 5), (OPTION_105, "OPTION", "SELL", "OPENING", 1, 3)]),
            tx(21, START + timedelta(days=1), [(OPTION_100, "OPTION", "SELL", "CLOSING", 1, 7), (OPTION_105, "OPTION", "BUY", "CLOSING", 1, 4)]),
        ]
    )
    assert len(vertical) == 1 and vertical[0].is_closed
    assert recognize_structure(vertical[0]) == "XYZ Vertical"

    calendar = reconstruct(
        [
            tx(30, START, [(OPTION_100, "OPTION", "SELL", "OPENING", 1, 5), ("XYZ261218C00100000", "OPTION", "BUY", "OPENING", 1, 8)]),
            tx(31, START + timedelta(days=1), [(OPTION_100, "OPTION", "BUY", "CLOSING", 1, 2)]),
            tx(32, START + timedelta(days=2), [("XYZ261218C00100000", "OPTION", "SELL", "CLOSING", 1, 10)]),
        ]
    )
    assert len(calendar) == 1 and calendar[0].is_closed
    assert recognize_structure(calendar[0]) == "XYZ Calendar"

    four_leg = reconstruct(
        [
            tx(40, START, [
                ("XYZ260918P00090000", "OPTION", "BUY", "OPENING", 1, 1),
                ("XYZ260918P00095000", "OPTION", "SELL", "OPENING", 1, 2),
                (OPTION_105, "OPTION", "SELL", "OPENING", 1, 2),
                ("XYZ260918C00110000", "OPTION", "BUY", "OPENING", 1, 1),
            ]),
            tx(41, START + timedelta(days=1), [(OPTION_105, "OPTION", "BUY", "CLOSING", 1, 1)]),
        ]
    )
    assert len(four_leg) == 1
    assert lifecycle_status(four_leg[0], end_at=END, lookahead_end_at=END) == "partial"

    rolled = reconstruct(
        [
            tx(50, START, [(OPTION_100, "OPTION", "SELL", "OPENING", 1, 5)], order_id="open"),
            tx(51, START + timedelta(days=1), [(OPTION_100, "OPTION", "BUY", "CLOSING", 1, 2), (OPTION_110, "OPTION", "SELL", "OPENING", 1, 4)], order_id="roll"),
        ]
    )
    assert len(rolled) == 1
    assert rolled[0].inventory[OPTION_110] == -1

    left, _ = normalize_execution_legs("A", [tx(60, START, [("XYZ", "EQUITY", "BUY", "OPENING", 1, 1)])])
    right, _ = normalize_execution_legs("B", [tx(61, START, [("XYZ", "EQUITY", "SELL", "CLOSING", 1, 1)])])
    accounts = reconstruct_lifecycles([*left, *right])
    assert len(accounts) == 2
    assert {item.account_id for item in accounts} == {"A", "B"}


def test_parent_fee_allocation_conserves_total_and_parent_cash_is_not_doubled():
    parent = tx(
        70,
        START,
        [
            (OPTION_100, "OPTION", "BUY", "OPENING", 1, 5),
            (OPTION_105, "OPTION", "SELL", "OPENING", 1, 2),
        ],
        fee=-2.37,
    )
    allocations = allocate_parent_fees(parent, parent.items)
    executions, _ = normalize_execution_legs("A", [parent])
    assert sum(allocations) == pytest.approx(-2.37)
    assert sum(item.fees_usd for item in executions) == pytest.approx(-2.37)
    assert sum(item.cash_effect_usd or 0 for item in executions) == pytest.approx(-300)


def test_missing_instruction_infers_trade_side_from_cash_and_keeps_signed_quantity():
    opening_sell = TransactionModel(
        id=71,
        broker_transaction_id="broker-71",
        transaction_at=START + timedelta(hours=1),
        transaction_type="TRADE",
        amount=Decimal("4420"),
        ingested_at=START + timedelta(hours=1, minutes=1),
        items=[
            TransactionItemModel(
                item_index=0,
                symbol=OPTION_100,
                asset_type="OPTION",
                instruction=None,
                position_effect="OPENING",
                quantity=Decimal("10"),
                price=Decimal("4.42"),
                currency="USD",
            )
        ],
    )
    closing_buy = TransactionModel(
        id=72,
        broker_transaction_id="broker-72",
        transaction_at=START + timedelta(hours=2),
        transaction_type="TRADE",
        amount=Decimal("-150"),
        ingested_at=START + timedelta(hours=2, minutes=1),
        items=[
            TransactionItemModel(
                item_index=0,
                symbol=OPTION_100,
                asset_type="OPTION",
                instruction=None,
                position_effect="CLOSING",
                quantity=Decimal("10"),
                price=Decimal("0.15"),
                currency="USD",
            )
        ],
    )

    executions, warnings = normalize_execution_legs("A", [opening_sell, closing_buy])

    assert warnings == []
    assert [item.signed_quantity for item in executions] == [-10, 10]
    assert [item.action for item in executions] == ["Sell · Opening", "Buy · Closing"]

    response = query_trades(
        request(),
        repository=Repo({"A": [opening_sell, closing_buy]}),
        snapshot=snapshot(),
        reference_client=Ref(),
        context={},
    )
    assert [item.quantity for item in response.data.trades[0].transactions] == [-10, 10]


def test_fee_components_are_authoritative_for_pnl_and_returned_by_type():
    rows = [
        tx(
            75,
            START + timedelta(hours=14),
            [("XYZ", "EQUITY", "BUY", "OPENING", 10, 100)],
            fee=-99,
        ).model_copy(
            update={
                "fee_components": [
                    TransactionFeeModel(fee_index=0, fee_type="COMMISSION", amount=Decimal("1")),
                    TransactionFeeModel(fee_index=1, fee_type="SEC_FEE", amount=Decimal("0.12")),
                ]
            }
        ),
        tx(
            76,
            START + timedelta(days=1, hours=14),
            [("XYZ", "EQUITY", "SELL", "CLOSING", 10, 110)],
            fee=-99,
        ).model_copy(
            update={
                "fee_components": [
                    TransactionFeeModel(fee_index=0, fee_type="COMMISSION", amount=Decimal("1")),
                    TransactionFeeModel(fee_index=1, fee_type="SEC_FEE", amount=Decimal("0.12")),
                ]
            }
        ),
    ]

    response = query_trades(
        request(), repository=Repo({"A": rows}), snapshot=snapshot(), reference_client=Ref(), context={}
    )

    summary = response.data.summary
    assert summary.period_pnl_usd == pytest.approx(97.76)
    assert summary.fees_and_carry_usd == pytest.approx(-2.24)
    assert summary.fee_components_usd == {"COMMISSION": -2.0, "SEC_FEE": -0.24}
    assert response.data.trades[0].fee_components_usd == summary.fee_components_usd
    assert response.data.trades[0].transactions[0].fee_components_usd == {
        "COMMISSION": -1.0,
        "SEC_FEE": -0.12,
    }


def test_zero_quantity_currency_fee_rows_are_not_execution_warnings():
    parent = tx(
        71,
        START,
        [
            ("CURRENCY_USD", "CURRENCY", "", "", 0, 0),
            ("NVDA", "EQUITY", "BUY", "OPENING", 20, 208.545),
        ],
    )

    executions, warnings = normalize_execution_legs("A", [parent])

    assert warnings == []
    assert [(item.symbol, item.signed_quantity) for item in executions] == [
        ("NVDA", 20.0)
    ]


def test_non_equity_lifecycles_keep_their_instrument_types():
    rows = [
        tx(72, START, [("SCC", "COLLECTIVE_INVESTMENT", "BUY", "OPENING", 1, 10)]),
        tx(73, START, [("SWVXX", "MUTUAL_FUND", "BUY", "OPENING", 1, 1)]),
        tx(74, START, [("CURRENCY_USD", "CURRENCY", "BUY", "OPENING", 1, 1)]),
    ]

    lifecycles = reconstruct(rows)

    assert {item.underlying_symbol: item.instrument_type for item in lifecycles} == {
        "CURRENCY_USD": "Currency",
        "SCC": "ETF",
        "SWVXX": "Mutual Fund",
    }
    assert {item.underlying_symbol: recognize_structure(item) for item in lifecycles} == {
        "CURRENCY_USD": "CURRENCY_USD Currency",
        "SCC": "SCC Etf",
        "SWVXX": "SWVXX Mutual Fund",
    }


def test_sector_warnings_only_apply_to_equity_like_instruments():
    class ClassificationRef(Ref):
        def __init__(self):
            super().__init__()
            self.requested = []

        def security_classifications(self, symbols):
            self.requested.append(list(symbols))
            return []

    lifecycles = [
        SimpleNamespace(underlying_symbol="CURRENCY_USD", instrument_type="Currency"),
        SimpleNamespace(underlying_symbol="SCC", instrument_type="ETF"),
        SimpleNamespace(underlying_symbol="SWVXX", instrument_type="Mutual Fund"),
        SimpleNamespace(underlying_symbol="VOLT", instrument_type="ETF"),
        SimpleNamespace(underlying_symbol="XYZ", instrument_type="Equity"),
    ]
    reference = ClassificationRef()

    sectors, warnings = _sectors(lifecycles, snapshot(), reference)

    assert reference.requested == [["XYZ"]]
    assert all(sectors[symbol] == "Not applicable" for symbol in ("CURRENCY_USD", "SCC", "SWVXX", "VOLT"))
    assert sectors["XYZ"] == "Unclassified"
    assert warnings == []


class Repo:
    def __init__(self, rows_by_account, *, complete=True):
        self.rows_by_account = rows_by_account
        self.complete = complete

    def query_lifecycle_history(self, account_id, **kwargs):
        coverage = TransactionCoverage(
            requested_start_at=kwargs["requested_start_at"],
            requested_end_at=kwargs["requested_end_at"],
            sync_through_by_type={"TRADE": kwargs["requested_end_at"] if self.complete else None},
            complete=self.complete,
            missing_intervals=[] if self.complete else ["TRADE:not_synced_through_end"],
        )
        rows = list(self.rows_by_account.get(account_id, []))
        if len(rows) > kwargs["max_rows"]:
            raise AssertionError("test fixture exceeded limit")
        return rows, coverage, END


class Ref:
    def __init__(self, *, option_history=False):
        self.option_history = option_history

    def bars(self, symbols, *, start, end):
        prices = {
            date(2026, 6, 30): 99,
            date(2026, 7, 1): 102,
            date(2026, 7, 2): 110,
            date(2026, 7, 3): 111,
        }
        return [
            {
                "symbol": symbol,
                "bars": [
                    {"timestamp": day.isoformat(), "close": close}
                    for day, close in prices.items()
                ],
            }
            for symbol in symbols
        ]

    def security_classifications(self, symbols):
        return [{"symbol": symbol, "sector": "Technology"} for symbol in symbols]

    def option_market_history(self, payload):
        if not self.option_history:
            raise ReferenceDataClientError(
                "Reference Data request failed status=501 code=OPTION_MARK_HISTORY_NOT_IMPLEMENTED"
            )
        return {"data": [], "missing_symbols": payload["symbols"], "meta": {}}


def snapshot(*, positions=()):
    return SimpleNamespace(
        portfolio_snapshot_id="snapshot-1",
        configuration_version="configuration-1",
        as_of=datetime(2026, 7, 6, 20, tzinfo=UTC),
        data=SimpleNamespace(positions=list(positions)),
    )


def request(*, include=True):
    return PnlTradeQueryRequest(
        account_ids=["A"],
        portfolio_snapshot_id="snapshot-1",
        start_at=START,
        end_at=END,
        include_closing_after_end=include,
        lookahead_end_at=END + timedelta(days=2) if include else None,
        sectors=[],
        instrument_types=[],
    )


def test_equity_query_reconciles_daily_path_attribution_and_closed_after_policy():
    rows = [
        tx(80, START + timedelta(hours=14), [("XYZ", "EQUITY", "BUY", "OPENING", 10, 100)], fee=-1),
        tx(81, START + timedelta(days=1, hours=14), [("XYZ", "EQUITY", "SELL", "CLOSING", 10, 110)], fee=-1),
    ]
    response = query_trades(
        request(), repository=Repo({"A": rows}), snapshot=snapshot(), reference_client=Ref(), context={}
    )
    trade = response.data.trades[0]
    assert response.data.summary.period_pnl_usd == pytest.approx(98)
    assert trade.realized_pnl_usd + trade.unrealized_pnl_usd + trade.fees_and_carry_usd == pytest.approx(trade.period_pnl_usd)
    assert response.data.daily_pnl[-1].cumulative_pnl_usd == pytest.approx(response.data.summary.period_pnl_usd)
    assert sum(item.pnl_usd for item in response.data.sector_attribution) == pytest.approx(98)

    after = [
        tx(90, START + timedelta(hours=1), [("XYZ", "EQUITY", "BUY", "OPENING", 1, 100)]),
        tx(91, END + timedelta(days=1), [("XYZ", "EQUITY", "SELL", "CLOSING", 1, 120)]),
    ]
    included = query_trades(
        request(), repository=Repo({"A": after}), snapshot=snapshot(), reference_client=Ref(), context={}
    )
    excluded = query_trades(
        request(include=False), repository=Repo({"A": after}), snapshot=snapshot(), reference_client=Ref(), context={}
    )
    assert included.data.trades[0].status == "closed_after_period"
    assert excluded.data.trades == []
    assert excluded.data.summary.period_pnl_usd == 0


def test_open_current_mark_is_estimated_and_nested_open_row_is_unrealized():
    rows = [tx(100, START + timedelta(hours=1), [("XYZ", "EQUITY", "BUY", "OPENING", 2, 100)], fee=-1)]
    position = SimpleNamespace(
        account_id="A",
        canonical_occ_symbol=None,
        exact_symbol="XYZ",
        underlying_symbol="XYZ",
        sector="Technology",
        mark=115.0,
    )
    response = query_trades(
        request(), repository=Repo({"A": rows}), snapshot=snapshot(positions=[position]), reference_client=Ref(), context={}
    )
    trade = response.data.trades[0]
    assert trade.status == "open"
    assert trade.lifecycle_pnl_usd == pytest.approx(29)
    assert trade.transactions[0].is_unrealized is True


def test_trade_status_filter_recalculates_the_response_and_empty_means_all():
    rows = [
        tx(102, START + timedelta(hours=1), [("XYZ", "EQUITY", "BUY", "OPENING", 1, 100)]),
        tx(103, START + timedelta(days=1), [("XYZ", "EQUITY", "SELL", "CLOSING", 1, 110)]),
        tx(104, START + timedelta(hours=2), [("ABC", "EQUITY", "BUY", "OPENING", 1, 100)]),
    ]
    position = SimpleNamespace(
        account_id="A",
        canonical_occ_symbol=None,
        exact_symbol="ABC",
        underlying_symbol="ABC",
        sector="Technology",
        mark=111.0,
    )
    kwargs = {
        "repository": Repo({"A": rows}),
        "snapshot": snapshot(positions=[position]),
        "reference_client": Ref(),
        "context": {},
    }
    all_statuses = query_trades(request(), **kwargs)
    closed = query_trades(
        request().model_copy(update={"statuses": ["closed", "closed_after_period"]}),
        **kwargs,
    )

    assert {trade.status for trade in all_statuses.data.trades} == {"closed", "open"}
    assert [trade.status for trade in closed.data.trades] == ["closed"]
    assert closed.data.summary.period_pnl_usd == pytest.approx(
        closed.data.trades[0].period_pnl_usd
    )
    assert closed.data.daily_pnl[-1].cumulative_pnl_usd == pytest.approx(
        closed.data.summary.period_pnl_usd
    )


def test_equity_history_is_sql_first_and_remote_repair_is_allowlisted():
    class RoutingRef(Ref):
        def __init__(self):
            super().__init__()
            self.bar_requests = []

        def bars(self, symbols, *, start, end, remote_query=False):
            self.bar_requests.append(
                {
                    "symbols": tuple(symbols),
                    "start": start,
                    "end": end,
                    "remote_query": remote_query,
                }
            )
            return super().bars(symbols, start=start, end=end)

    rows = [
        tx(105, START + timedelta(hours=1), [("MSFT", "EQUITY", "BUY", "OPENING", 1, 100)]),
        tx(106, START + timedelta(days=1), [("MSFT", "EQUITY", "SELL", "CLOSING", 1, 110)]),
        tx(107, START + timedelta(hours=2), [("SCC", "EQUITY", "BUY", "OPENING", 1, 100)]),
        tx(108, START + timedelta(days=1, hours=1), [("SCC", "EQUITY", "SELL", "CLOSING", 1, 110)]),
    ]
    reference = RoutingRef()
    query_trades(
        request(),
        repository=Repo({"A": rows}),
        snapshot=snapshot(),
        reference_client=reference,
        context={
            "risk.pnl.equity_history_lookback_days": 4,
            "risk.pnl.remote_daily_bar_symbols": ["SCC", "TSCO", "VOLT"],
        },
    )

    assert reference.bar_requests == [
        {
            "symbols": ("MSFT", "SCC"),
            "start": START.date() - timedelta(days=4),
            "end": END.date(),
            "remote_query": False,
        },
        {
            "symbols": ("SCC",),
            "start": START.date() - timedelta(days=4),
            "end": END.date(),
            "remote_query": True,
        },
    ]


def test_missing_current_mark_warning_names_trade_and_exact_symbol():
    rows = [
        tx(101, START + timedelta(hours=1), [("XYZ", "EQUITY", "BUY", "OPENING", 2, 100)])
    ]

    response = query_trades(
        request(),
        repository=Repo({"A": rows}),
        snapshot=snapshot(),
        reference_client=Ref(),
        context={},
    )

    warning = next(
        item
        for item in response.data_quality.warnings
        if "current_exact_mark_unavailable" in item
    )
    assert warning == "XYZ trade: current_exact_mark_unavailable (XYZ)"
    assert "pnl:" not in warning


def test_transaction_coverage_stops_at_snapshot_time_not_future_midnight():
    class RecordingRepo(Repo):
        def __init__(self, rows_by_account):
            super().__init__(rows_by_account)
            self.requested_end_at = None

        def query_lifecycle_history(self, account_id, **kwargs):
            self.requested_end_at = kwargs["requested_end_at"]
            return super().query_lifecycle_history(account_id, **kwargs)

    rows = [
        tx(102, START, [("XYZ", "EQUITY", "BUY", "OPENING", 1, 100)]),
        tx(103, START + timedelta(days=1), [("XYZ", "EQUITY", "SELL", "CLOSING", 1, 101)]),
    ]
    repository = RecordingRepo({"A": rows})
    payload = request(include=False).model_copy(
        update={"end_at": datetime(2026, 7, 7, tzinfo=UTC)}
    )

    query_trades(
        payload,
        repository=repository,
        snapshot=snapshot(),
        reference_client=Ref(),
        context={},
    )

    assert repository.requested_end_at == snapshot().as_of


def test_missing_coverage_and_missing_historical_option_marks_are_explicit():
    equity = [
        tx(110, START, [("XYZ", "EQUITY", "BUY", "OPENING", 1, 100)]),
        tx(111, START + timedelta(days=1), [("XYZ", "EQUITY", "SELL", "CLOSING", 1, 101)]),
    ]
    response = query_trades(
        request(), repository=Repo({"A": equity}, complete=False), snapshot=snapshot(), reference_client=Ref(), context={}
    )
    assert response.data_quality.status == "insufficient_data"
    assert response.data_quality.transaction_coverage[0].complete is False

    option = [
        tx(120, START, [(OPTION_100, "OPTION", "BUY", "OPENING", 1, 5)]),
        tx(121, START + timedelta(days=2), [(OPTION_100, "OPTION", "SELL", "CLOSING", 1, 7)]),
    ]
    with pytest.raises(PnlMarkHistoryUnavailable):
        query_trades(
            request(), repository=Repo({"A": option}), snapshot=snapshot(), reference_client=Ref(), context={}
        )


def test_missing_price_excludes_only_affected_trade_from_query():
    rows = [
        tx(112, START, [("ABC", "EQUITY", "BUY", "OPENING", 1, 100)]),
        tx(113, START + timedelta(days=1), [("ABC", "EQUITY", "SELL", "CLOSING", 1, 101)]),
        tx(120, START, [(OPTION_100, "OPTION", "BUY", "OPENING", 1, 5)]),
        tx(121, START + timedelta(days=2), [(OPTION_100, "OPTION", "SELL", "CLOSING", 1, 7)]),
    ]

    response = query_trades(
        request(),
        repository=Repo({"A": rows}),
        snapshot=snapshot(),
        reference_client=Ref(option_history=True),
        context={},
    )

    assert [trade.underlying_symbol for trade in response.data.trades] == ["ABC"]
    assert response.data.summary.period_pnl_usd == pytest.approx(1)
    assert response.data.daily_pnl
    assert response.data_quality.status == "usable_with_warnings"
    assert response.data_quality.mark_coverage_percent < 100
    assert response.data_quality.warnings.count("trade_valuation_incomplete") == 1
    assert len(response.data.excluded_trades) == 1
    excluded = response.data.excluded_trades[0]
    assert excluded.underlying_symbol == "XYZ"
    assert excluded.instrument_type == "Options"
    assert excluded.missing_symbols == [OPTION_100]
    assert excluded.transaction_ids == ["120", "121"]


def test_expired_open_option_settles_intrinsic_and_moves_pnl_to_realized():
    expired_symbol = "XYZ260703C00100000"

    class ExpiredOptionRef(Ref):
        def __init__(self):
            super().__init__(option_history=True)
            self.current_quote_requests = []

        def option_market_history(self, payload):
            return {
                "data": [
                    {
                        "canonical_occ_symbol": expired_symbol,
                        "at": (START + timedelta(days=day, hours=20)).isoformat(),
                        "status": "ok",
                        "mark": mark,
                        "underlying_price": 101.0 + day,
                    }
                    for day, mark in ((0, 2.0), (1, 2.2), (2, 2.5))
                ],
                "missing_symbols": [],
                "meta": {},
            }

        def option_quotes(self, symbols):
            self.current_quote_requests.append(list(symbols))
            return {"data": [], "missing_symbols": list(symbols), "meta": {}}

    rows = [
        tx(124, START + timedelta(hours=1), [(expired_symbol, "OPTION", "BUY", "OPENING", 1, 2)]),
    ]
    reference = ExpiredOptionRef()
    response = query_trades(
        request(),
        repository=Repo({"A": rows}),
        snapshot=snapshot(),
        reference_client=reference,
        context={},
    )

    assert len(response.data.trades) == 1
    trade = response.data.trades[0]
    assert trade.status == "closed"
    assert trade.closed_at == datetime(2026, 7, 4, tzinfo=UTC)
    assert trade.lifecycle_pnl_usd == pytest.approx(100)
    assert trade.period_pnl_usd == pytest.approx(100)
    assert trade.realized_pnl_usd == pytest.approx(100)
    assert trade.unrealized_pnl_usd == 0
    assert reference.current_quote_requests == []
    assert not any(
        "current_exact_mark_unavailable" in warning
        for warning in response.data_quality.warnings
    )


def test_expired_option_without_history_is_excluded_without_missing_price_failure():
    expired_symbol = "KMB260717C00125000"

    class NoExpiredHistoryRef(Ref):
        def __init__(self):
            super().__init__(option_history=True)
            self.option_history_requests = []

        def option_market_history(self, payload):
            self.option_history_requests.append(payload)
            return {"data": [], "missing_symbols": payload["symbols"], "meta": {}}

    start = datetime(2026, 8, 1, tzinfo=UTC)
    end = datetime(2026, 8, 28, tzinfo=UTC)
    payload = PnlTradeQueryRequest(
        account_ids=["A"],
        portfolio_snapshot_id="snapshot-1",
        start_at=start,
        end_at=end,
        include_closing_after_end=True,
        lookahead_end_at=end + timedelta(hours=20),
        sectors=[],
        instrument_types=[],
    )
    current_snapshot = SimpleNamespace(
        portfolio_snapshot_id="snapshot-1",
        configuration_version="configuration-1",
        as_of=datetime(2026, 8, 28, 20, tzinfo=UTC),
        data=SimpleNamespace(positions=[]),
    )
    rows = [
        tx(
            125,
            datetime(2026, 7, 1, 14, tzinfo=UTC),
            [(expired_symbol, "OPTION", "BUY", "OPENING", 1, 2)],
        )
    ]
    reference = NoExpiredHistoryRef()

    response = query_trades(
        payload,
        repository=Repo({"A": rows}),
        snapshot=current_snapshot,
        reference_client=reference,
        context={},
    )

    assert response.data_quality.status == "usable_with_warnings"
    assert response.data_quality.mark_coverage_percent == 100
    assert reference.option_history_requests == []
    assert response.data.trades == []
    assert len(response.data.excluded_trades) == 1
    exclusion = response.data.excluded_trades[0]
    assert exclusion.missing_symbols == []
    assert exclusion.reason == (
        "expired_option_without_history_excluded:KMB260717C00125000"
    )
    assert f"{expired_symbol}:expired_option_without_history_excluded" in (
        response.data_quality.warnings
    )
    assert not any(
        "historical_mark_unavailable" in warning
        for warning in response.data_quality.warnings
    )


def test_analyzer_values_expired_calendar_leg_from_its_last_historical_mark():
    expired_symbol = "XYZ260703C00100000"

    class ExpiredCalendarRef(Ref):
        def __init__(self):
            super().__init__(option_history=True)
            self.current_quote_requests = []
            self.option_history_requests = []

        def option_market_history(self, payload):
            requested = set(payload["symbols"])
            self.option_history_requests.extend(requested)
            assert requested <= {expired_symbol, OPTION_100}
            return {
                "data": [
                    {
                        "canonical_occ_symbol": symbol,
                        "at": (START + timedelta(days=day, hours=20)).isoformat(),
                        "status": "ok",
                        "mark": mark,
                    }
                        for symbol, marks in (
                            (expired_symbol, ((0, 2.0), (1, 2.2), (2, 2.5))),
                            (OPTION_100, ((0, 5.0), (1, 5.1), (2, 5.0))),
                        )
                        if symbol in requested
                        for day, mark in marks
                ],
                "missing_symbols": [],
                "meta": {},
            }

        def option_quotes(self, symbols):
            self.current_quote_requests.append(list(symbols))
            return {"data": [], "missing_symbols": list(symbols), "meta": {}}

    rows = [
        tx(125, START + timedelta(hours=1), [
            (expired_symbol, "OPTION", "SELL", "OPENING", 1, 2),
            (OPTION_100, "OPTION", "BUY", "OPENING", 1, 5),
        ]),
    ]
    lifecycle = reconstruct(rows)[0]
    reference = ExpiredCalendarRef()
    response = analyze_selection(
        PnlAnalyzerRequest(
            account_ids=["A"],
            portfolio_snapshot_id="snapshot-1",
            trade_ids=[lifecycle.trade_id],
            transaction_ids=[],
            decision_at=None,
            fill_model="historical_touch_plus_tick",
            modifications=[],
            exit_rule=PnlExitRule(kind="actual", value=None),
        ),
        repository=Repo({"A": rows}),
        snapshot=snapshot(positions=[
            SimpleNamespace(
                account_id="A",
                canonical_occ_symbol=OPTION_100,
                exact_symbol=OPTION_100,
                mark=5.0,
            )
        ]),
        reference_client=reference,
        context={},
    )

    assert response.data.lifecycle_pnl_usd == pytest.approx(-50)
    assert response.data.actual_path
    assert reference.current_quote_requests == []
    assert set(reference.option_history_requests) == {expired_symbol, OPTION_100}


def test_trade_query_uses_historical_option_closes_with_boundary_lookback():
    class OptionHistoryRef(Ref):
        def __init__(self):
            super().__init__(option_history=True)
            self.payload = None

        def option_market_history(self, payload):
            self.payload = payload
            return {
                "data": [
                    {
                        "canonical_occ_symbol": OPTION_100,
                        "at": at.isoformat(),
                        "session_date": (at - timedelta(days=1)).date().isoformat(),
                        "status": "ok",
                        "mark": mark,
                        "close": mark,
                    }
                    for at, mark in (
                        (START, 4.0),
                        (START + timedelta(days=1), 6.0),
                        (START + timedelta(days=2), 7.0),
                        (END, 8.0),
                    )
                ],
                "missing_symbols": [],
                "meta": {"cache": "miss"},
            }

    option = [
        tx(122, START - timedelta(days=2), [(OPTION_100, "OPTION", "BUY", "OPENING", 1, 5)]),
        tx(123, START + timedelta(days=2, hours=1), [(OPTION_100, "OPTION", "SELL", "CLOSING", 1, 7)]),
    ]
    reference = OptionHistoryRef()
    response = query_trades(
        request(),
        repository=Repo({"A": option}),
        snapshot=snapshot(),
        reference_client=reference,
        context={},
    )
    assert response.data.summary.period_pnl_usd == pytest.approx(300)
    assert response.data.daily_pnl
    assert response.data_quality.mark_coverage_percent == pytest.approx(100)
    assert datetime.fromisoformat(reference.payload["start_at"]) == START - timedelta(days=4)


def test_analyzer_preserves_option_chronology_when_history_is_501():
    rows = [
        tx(130, START, [(OPTION_100, "OPTION", "BUY", "OPENING", 1, 5), (OPTION_105, "OPTION", "SELL", "OPENING", 1, 3)], fee=-1),
        tx(131, START + timedelta(days=2), [(OPTION_100, "OPTION", "SELL", "CLOSING", 1, 7), (OPTION_105, "OPTION", "BUY", "CLOSING", 1, 4)], fee=-1),
    ]
    lifecycle = reconstruct(rows)[0]
    payload = PnlAnalyzerRequest(
        account_ids=["A"],
        portfolio_snapshot_id="snapshot-1",
        trade_ids=[lifecycle.trade_id],
        transaction_ids=[],
        decision_at=None,
        fill_model="historical_touch_plus_tick",
        modifications=[],
        exit_rule=PnlExitRule(kind="actual", value=None),
    )
    response = analyze_selection(
        payload,
        repository=Repo({"A": rows}),
        snapshot=snapshot(),
        reference_client=Ref(),
        context={},
    )
    assert response.data_quality.historical_option_marks_available is False
    assert response.data.actual_path == [] and response.data.scenarios == []
    assert len(response.data.chronology) == 4
    assert any(
        OPTION_100 in warning for warning in response.data_quality.warnings
    )
    assert OPTION_105 in response.data_quality.warnings[-1]
    assert sum(item.pnl_usd for item in response.data.attribution) == pytest.approx(response.data.lifecycle_pnl_usd)


def test_option_analyzer_uses_opening_fills_before_first_session_close_mark():
    class SessionCloseOptionRef(Ref):
        def __init__(self):
            super().__init__(option_history=True)

        def option_market_history(self, payload):
            return {
                "data": [
                    {
                        "canonical_occ_symbol": symbol,
                        "at": (START + timedelta(days=day, hours=20)).isoformat(),
                        "status": "ok",
                        "mark": mark,
                    }
                    for symbol, opening in ((OPTION_100, 5.0), (OPTION_105, 3.0))
                    for day, mark in ((0, opening), (1, opening + 1), (2, opening + 2))
                ],
                "missing_symbols": [],
                "meta": {},
            }

    rows = [
        tx(132, START + timedelta(hours=1), [(OPTION_100, "OPTION", "BUY", "OPENING", 1, 5), (OPTION_105, "OPTION", "SELL", "OPENING", 1, 3)]),
        tx(133, START + timedelta(days=2, hours=1), [(OPTION_100, "OPTION", "SELL", "CLOSING", 1, 7), (OPTION_105, "OPTION", "BUY", "CLOSING", 1, 5)]),
    ]
    lifecycle = reconstruct(rows)[0]
    response = analyze_selection(
        PnlAnalyzerRequest(
            account_ids=["A"],
            portfolio_snapshot_id="snapshot-1",
            trade_ids=[lifecycle.trade_id],
            transaction_ids=[],
            decision_at=None,
            fill_model="historical_touch_plus_tick",
            modifications=[],
            exit_rule=PnlExitRule(kind="actual", value=None),
        ),
        repository=Repo({"A": rows}),
        snapshot=snapshot(),
        reference_client=SessionCloseOptionRef(),
        context={},
    )

    assert response.data_quality.historical_option_marks_available is True
    assert response.data.actual_path
    assert response.data.actual_path[0].pnl_usd == pytest.approx(0)


def test_option_analyzer_omits_partial_close_point_with_stale_remaining_leg_mark():
    near_option = "XYZ260703C00100000"

    class CalendarHistoryRef(Ref):
        def __init__(self):
            super().__init__(option_history=True)

        def option_market_history(self, payload):
            return {
                "data": [
                    {
                        "canonical_occ_symbol": symbol,
                        "at": at.isoformat(),
                        "status": "ok",
                        "mark": mark,
                    }
                    for symbol, observations in (
                        (
                            near_option,
                            (
                                (START + timedelta(hours=20), 4.5),
                                (START + timedelta(days=1, hours=20), 5.3),
                            ),
                        ),
                        (
                            OPTION_100,
                            (
                                (START + timedelta(hours=20), 4.9),
                                (START + timedelta(days=1, hours=20), 5.7),
                                (START + timedelta(days=2, hours=20), 0.1),
                            ),
                        ),
                    )
                    for at, mark in observations
                ],
                "missing_symbols": [],
                "meta": {},
            }

    opened_at = START + timedelta(hours=13)
    partial_close_at = START + timedelta(days=2, hours=13)
    rows = [
        tx(
            135,
            opened_at,
            [
                (OPTION_100, "OPTION", "BUY", "OPENING", 1, 4.82),
                (near_option, "OPTION", "SELL", "OPENING", 1, 4.42),
            ],
        ),
        tx(
            136,
            partial_close_at,
            [(near_option, "OPTION", "BUY", "CLOSING", 1, 0.15)],
        ),
    ]
    lifecycle = reconstruct(rows)[0]
    response = analyze_selection(
        PnlAnalyzerRequest(
            account_ids=["A"],
            portfolio_snapshot_id="snapshot-1",
            trade_ids=[lifecycle.trade_id],
            transaction_ids=[],
            decision_at=None,
            fill_model="historical_touch_plus_tick",
            modifications=[],
            exit_rule=PnlExitRule(kind="actual", value=None),
        ),
        repository=Repo({"A": rows}),
        snapshot=snapshot(
            positions=[
                SimpleNamespace(
                    account_id="A",
                    canonical_occ_symbol=OPTION_100,
                    exact_symbol=OPTION_100,
                    mark=0.1,
                )
            ]
        ),
        reference_client=CalendarHistoryRef(),
        context={},
    )

    partial_close_boundary = partial_close_at + timedelta(microseconds=1)
    assert all(point.at != partial_close_boundary for point in response.data.actual_path)
    assert max(point.pnl_usd for point in response.data.actual_path) == pytest.approx(0)
    assert response.data.peak_open_pnl_usd == pytest.approx(0)


def test_option_history_ignores_asynchronous_closes_that_break_vertical_bounds():
    class IncoherentVerticalRef(Ref):
        def option_market_history(self, payload):
            return {
                "data": [
                    {
                        "canonical_occ_symbol": symbol,
                        "at": at.isoformat(),
                        "status": "ok",
                        "mark": mark,
                    }
                    for symbol, at, mark in (
                        (OPTION_100, START + timedelta(days=1), 5.0),
                        (OPTION_105, START + timedelta(days=1), 2.0),
                        # A higher-strike call cannot be worth more than the
                        # lower-strike call at the same observation time.
                        (OPTION_100, START + timedelta(days=2), 4.0),
                        (OPTION_105, START + timedelta(days=2), 6.0),
                    )
                ],
                "missing_symbols": [],
                "meta": {},
            }

    lifecycle = reconstruct(
        [
            tx(
                134,
                START,
                [
                    (OPTION_100, "OPTION", "BUY", "OPENING", 1, 5),
                    (OPTION_105, "OPTION", "SELL", "OPENING", 1, 2),
                ],
            )
        ]
    )[0]
    marks = MarketMarkBook(snapshot(), IncoherentVerticalRef())
    marks.load_history(
        [lifecycle],
        start_at=START,
        end_at=END,
        maximum_option_symbols=10,
    )

    assert marks.mark("A", OPTION_100, START + timedelta(days=2)) == pytest.approx(5)
    assert marks.mark("A", OPTION_105, START + timedelta(days=2)) == pytest.approx(2)
    assert any("ignored 1 asynchronous close session" in item for item in marks.warnings)


def test_equity_analyzer_accepts_new_hedge_symbol_and_uses_bounded_fill():
    class RangeAwareRef(Ref):
        def __init__(self):
            super().__init__()
            self.bar_requests = []

        def bars(self, symbols, *, start, end):
            self.bar_requests.append((start, end))
            return [
                {
                    "symbol": group["symbol"],
                    "bars": [
                        {**bar, "low": bar["close"] - 1, "high": bar["close"] + 1}
                        for bar in group["bars"]
                        if start <= date.fromisoformat(bar["timestamp"]) <= end
                    ],
                }
                for group in super().bars(symbols, start=start, end=end)
            ]

    rows = [
        tx(140, START + timedelta(hours=1), [("XYZ", "EQUITY", "BUY", "OPENING", 2, 100)], fee=-1),
        tx(141, START + timedelta(days=2, hours=1), [("XYZ", "EQUITY", "SELL", "CLOSING", 2, 110)], fee=-1),
    ]
    lifecycle = reconstruct(rows)[0]
    payload = PnlAnalyzerRequest.model_validate(
        {
            "account_ids": ["A"],
            "portfolio_snapshot_id": "snapshot-1",
            "trade_ids": [lifecycle.trade_id],
            "transaction_ids": [],
            "decision_at": (START + timedelta(days=1)).isoformat(),
            "fill_model": "historical_touch_plus_tick",
            "modifications": [
                {
                    "action": "add",
                    "instrument_kind": "underlying",
                    "symbol": "QQQ",
                    "quantity": 1,
                }
            ],
            "exit_rule": {"kind": "actual", "value": None},
        }
    )
    reference = RangeAwareRef()
    response = analyze_selection(
        payload,
        repository=Repo({"A": rows}),
        snapshot=snapshot(),
        reference_client=reference,
        context={"risk.pnl.transaction_cost_per_unit": 0.10},
    )
    assert response.data_quality.historical_option_marks_available is True
    assert response.data.actual_path
    assert len(response.data.scenarios) == 1
    scenario = response.data.scenarios[0]
    assert scenario.path[-1].at <= response.data.actual_path[-1].at
    assert scenario.net_pnl_usd != response.data.lifecycle_pnl_usd
    assert (
        (START + timedelta(days=1)).date() - timedelta(days=4),
        scenario.path[-1].at.date(),
    ) in reference.bar_requests
    scenario_by_time = {point.at: point.pnl_usd for point in scenario.path}
    differences = [
        scenario_by_time[point.at] - point.pnl_usd
        for point in response.data.actual_path
        if point.at in scenario_by_time and point.at >= START + timedelta(days=1)
    ]
    assert len({round(value, 2) for value in differences}) > 1


def test_analyzer_accepts_new_option_on_an_unselected_underlying():
    added_option = "QQQ260918P00620000"

    class OptionAdditionRef(Ref):
        def option_market_history(self, payload):
            assert payload["symbols"] == [added_option]
            return {
                "data": [
                    {
                        "canonical_occ_symbol": added_option,
                        "at": (START + timedelta(days=day, hours=20)).isoformat(),
                        "status": "ok",
                        "bid": mark - 0.05,
                        "ask": mark + 0.05,
                        "mark": mark,
                    }
                    for day, mark in ((0, 6.0), (1, 6.5), (2, 7.0))
                ],
                "missing_symbols": [],
                "meta": {},
            }

    rows = [
        tx(142, START + timedelta(hours=1), [("XYZ", "EQUITY", "BUY", "OPENING", 2, 100)]),
        tx(143, START + timedelta(days=2, hours=1), [("XYZ", "EQUITY", "SELL", "CLOSING", 2, 110)]),
    ]
    lifecycle = reconstruct(rows)[0]
    payload = PnlAnalyzerRequest.model_validate(
        {
            "account_ids": ["A"],
            "portfolio_snapshot_id": "snapshot-1",
            "trade_ids": [lifecycle.trade_id],
            "transaction_ids": [],
            "decision_at": (START + timedelta(days=1, hours=21)).isoformat(),
            "fill_model": "historical_touch_plus_tick",
            "modifications": [
                {
                    "action": "add",
                    "instrument_kind": "option",
                    "symbol": added_option,
                    "quantity": 1,
                }
            ],
            "exit_rule": {"kind": "actual", "value": None},
        }
    )

    response = analyze_selection(
        payload,
        repository=Repo({"A": rows}),
        snapshot=snapshot(),
        reference_client=OptionAdditionRef(),
        context={},
    )

    assert response.data.actual_path
    assert len(response.data.scenarios) == 1
    assert response.data.scenarios[0].path


def test_analyzer_replays_mixed_equity_and_option_hypothetical_trades():
    added_option = "QQQ260918P00620000"

    class MixedAdditionRef(Ref):
        def option_market_history(self, payload):
            assert payload["symbols"] == [added_option]
            return {
                "data": [
                    {
                        "canonical_occ_symbol": added_option,
                        "at": (START + timedelta(days=day, hours=20)).isoformat(),
                        "status": "ok",
                        "bid": mark - 0.05,
                        "ask": mark + 0.05,
                        "mark": mark,
                    }
                    for day, mark in ((0, 6.0), (1, 6.5), (2, 7.0))
                ],
                "missing_symbols": [],
                "meta": {},
            }

    rows = [
        tx(144, START + timedelta(hours=1), [("XYZ", "EQUITY", "BUY", "OPENING", 2, 100)]),
        tx(145, START + timedelta(days=2, hours=1), [("XYZ", "EQUITY", "SELL", "CLOSING", 2, 110)]),
    ]
    lifecycle = reconstruct(rows)[0]
    response = analyze_selection(
        PnlAnalyzerRequest.model_validate(
            {
                "account_ids": ["A"],
                "portfolio_snapshot_id": "snapshot-1",
                "trade_ids": [lifecycle.trade_id],
                "transaction_ids": [],
                "decision_at": (START + timedelta(days=1)).isoformat(),
                "fill_model": "historical_touch_plus_tick",
                "modifications": [
                    {
                        "action": "add",
                        "instrument_kind": "underlying",
                        "symbol": "QQQ",
                        "quantity": 1,
                        "fill_price": 110,
                    },
                    {
                        "action": "add",
                        "instrument_kind": "option",
                        "symbol": added_option,
                        "quantity": 1,
                        "fill_price": 6.5,
                    },
                ],
                "exit_rule": {"kind": "actual", "value": None},
            }
        ),
        repository=Repo({"A": rows}),
        snapshot=snapshot(),
        reference_client=MixedAdditionRef(),
        context={},
    )

    assert len(response.data.scenarios) == 1
    scenario = response.data.scenarios[0]
    assert scenario.path
    actual_by_time = {point.at: point.pnl_usd for point in response.data.actual_path}
    assert any(
        point.pnl_usd != actual_by_time[point.at]
        for point in scenario.path
        if point.at in actual_by_time
    )


def test_option_market_history_client_posts_once_without_retry():
    class Response:
        status_code = 501

        @staticmethod
        def json():
            return {"error": {"code": "OPTION_MARK_HISTORY_NOT_IMPLEMENTED"}}

    class HttpClient:
        def __init__(self):
            self.calls = []

        def request(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return Response()

    http = HttpClient()
    client = ReferenceDataClient(
        {"risk.refdata.retry_count": 5}, client=http
    )
    with pytest.raises(ReferenceDataClientError):
        client.option_market_history(
            {
                "symbols": [OPTION_100],
                "start_at": START.isoformat(),
                "end_at": END.isoformat(),
                "interval": "1d",
                "include_greeks": True,
            }
        )
    assert len(http.calls) == 1
    assert http.calls[0][0][:2] == (
        "POST",
        "/api/v1/options/market-history/query",
    )


def test_analyzer_exact_history_without_surfaces_is_additive_fallback_and_loaded_once():
    class ExactHistoryRef(Ref):
        def __init__(self):
            super().__init__(option_history=True)
            self.calls = 0

        def option_market_history(self, payload):
            self.calls += 1
            return {
                "data": [
                    {
                        "canonical_occ_symbol": symbol,
                        "at": (START + timedelta(days=day, hours=20)).isoformat(),
                        "status": "ok",
                        "mark": opening
                        + day * (1.0 if symbol == OPTION_100 else 0.4),
                        "underlying_price": 100.0 + 2.0 * day,
                        "provider_iv": 0.25,
                        "provider_delta": None,
                        "provider_gamma": None,
                        "provider_theta_per_day": None,
                        "provider_vega_per_1pct": None,
                        "provider_rho_per_1pct": None,
                        "source": "fixture",
                    }
                    for symbol, opening in ((OPTION_100, 5.0), (OPTION_105, 3.0))
                    for day in (0, 1)
                ],
                "missing_symbols": [],
                "meta": {},
            }

    rows = [
        tx(180, START + timedelta(hours=1), [(OPTION_100, "OPTION", "BUY", "OPENING", 1, 5), (OPTION_105, "OPTION", "SELL", "OPENING", 1, 3)]),
        tx(181, START + timedelta(days=2, hours=1), [(OPTION_100, "OPTION", "SELL", "CLOSING", 1, 7), (OPTION_105, "OPTION", "BUY", "CLOSING", 1, 4)]),
    ]
    lifecycle = reconstruct(rows)[0]
    reference = ExactHistoryRef()
    response = analyze_selection(
        PnlAnalyzerRequest(
            account_ids=["A"],
            portfolio_snapshot_id="snapshot-1",
            trade_ids=[lifecycle.trade_id],
            transaction_ids=[],
            decision_at=None,
            fill_model="historical_touch_plus_tick",
            modifications=[],
            exit_rule=PnlExitRule(kind="actual", value=None),
        ),
        repository=Repo({"A": rows}),
        snapshot=snapshot(),
        reference_client=reference,
        context={
            "risk.rates.configured_flat_continuous_rate": 0.04,
            "risk.dividends.default_continuous_yield": 0.0,
        },
    )

    assert reference.calls == 1
    assert response.data_quality.historical_option_marks_available is True
    assert response.data_quality.attribution_method == "price_history_fallback"
    assert response.data_quality.surface_coverage_percent == 0.0
    assert [item.factor for item in response.data.attribution] == [
        "Underlying movement",
        "Volatility",
        "Skew",
        "Term structure",
        "Time decay",
        "Rates",
            "Dividends",
            "Commission & fees",
            "Option repricing residual",
            "Model residual",
    ]
    factors = {item.factor: item.pnl_usd for item in response.data.attribution}
    assert any(
        abs(factors[name]) > 0.01
        for name in ("Underlying movement", "Volatility", "Time decay")
    )
    assert factors["Skew"] == factors["Term structure"] == 0.0
    assert factors["Model residual"] != pytest.approx(
        response.data.lifecycle_pnl_usd
        - factors["Commission & fees"]
    )
    assert sum(item.pnl_usd for item in response.data.attribution) == pytest.approx(
        response.data.lifecycle_pnl_usd, abs=0.01
    )


def test_same_day_option_trade_has_zero_surface_coverage_without_insufficient_status():
    rows = [
        tx(182, START + timedelta(hours=1), [(OPTION_100, "OPTION", "BUY", "OPENING", 1, 5)]),
        tx(183, START + timedelta(hours=6), [(OPTION_100, "OPTION", "SELL", "CLOSING", 1, 6)]),
    ]
    lifecycle = reconstruct(rows)[0]
    response = analyze_selection(
        PnlAnalyzerRequest(
            account_ids=["A"],
            portfolio_snapshot_id="snapshot-1",
            trade_ids=[lifecycle.trade_id],
            transaction_ids=[],
            decision_at=None,
            fill_model="historical_touch_plus_tick",
            modifications=[],
            exit_rule=PnlExitRule(kind="actual", value=None),
        ),
        repository=Repo({"A": rows}),
        snapshot=snapshot(),
        reference_client=Ref(),
        context={},
    )

    assert response.data_quality.attribution_method == "price_history_fallback"
    assert response.data_quality.surface_coverage_percent == 0.0
    assert response.data_quality.historical_option_marks_available is False
    assert response.data_quality.status == "usable_with_warnings"
