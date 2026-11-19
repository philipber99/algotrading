from __future__ import annotations

import ast
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from algotrading.refdata.api.live_models import OptionMarketHistoryRequest
from algotrading.risk.history.planner import (
    HistoryWindowPlanner,
    chunk_session_window,
    clip_required_sessions,
)
from algotrading.risk.history.revaluation import (
    CurrentPortfolioHistoricalRevaluationEngine,
)
from algotrading.risk.market_inputs.client import (
    ReferenceDataClient,
    ReferenceDataClientError,
)
from algotrading.risk.option_prices.service import RiskOptionHistoryService
from algotrading.risk.api.hedge_models import HedgeCandidateQueryRequest
from algotrading.risk.api.hedge_service import _primary_objective_usable
from algotrading.risk.api.bootstrap import DEFAULT_CONTEXT


UTC = timezone.utc
AS_OF = datetime(2026, 8, 27, 16, tzinfo=UTC)
CALL = "AAA261218C00100000"
PUT = "BBB261218P00100000"


def _history_row(symbol: str, session: date, mark: float, **updates):
    row = {
        "requested_symbol": symbol,
        "canonical_occ_symbol": symbol,
        "session_date": session.isoformat(),
        "at": datetime.combine(session + timedelta(days=1), datetime.min.time(), UTC).isoformat(),
        "observed_at": datetime.combine(session + timedelta(days=1), datetime.min.time(), UTC).isoformat(),
        "status": "ok",
        "mark": mark,
        "underlying_price": 100.0,
        "source": "risk_mysql_option_prices",
        "warnings": [],
    }
    row.update(updates)
    return row


class _Repository:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls: list[tuple[tuple[str, ...], date, date]] = []

    def query_history(self, symbols, *, start_date, end_date):
        wanted = set(symbols)
        self.calls.append((tuple(symbols), start_date, end_date))
        return [
            dict(row)
            for row in self.rows
            if row["canonical_occ_symbol"] in wanted
            and start_date <= date.fromisoformat(row["session_date"]) <= end_date
        ]


class _Remote:
    def __init__(self, rows=(), *, maximum=376, fail_starts=()):
        self.context = {"risk.option_prices.max_history_days": 376}
        self.rows = list(rows)
        self.maximum = maximum
        self.fail_starts = {str(value) for value in fail_starts}
        self.calls: list[dict] = []

    def option_history_maximum_days(self):
        return self.maximum

    def option_market_history(self, payload):
        payload = dict(payload)
        self.calls.append(payload)
        start = datetime.fromisoformat(payload["start_at"])
        end = datetime.fromisoformat(payload["end_at"])
        if start.date().isoformat() in self.fail_starts:
            raise ReferenceDataClientError(
                "fixture range failed",
                http_status=422,
                server_error_code="INVALID_REQUEST",
                server_message="fixture limit violation",
                request_id="fixture-request",
                endpoint="/api/v1/options/market-history/query",
                requested_start=payload["start_at"],
                requested_end=payload["end_at"],
            )
        symbols = set(payload["symbols"])
        return {
            "data": [
                dict(row)
                for row in self.rows
                if row["canonical_occ_symbol"] in symbols
                and start.date() <= date.fromisoformat(row["session_date"]) < end.date()
            ],
            "meta": {"warnings": []},
        }


@pytest.mark.parametrize(
    ("returns", "first", "last", "span_days"),
    [
        (10, date(2026, 8, 12), date(2026, 8, 26), 15),
        (126, date(2026, 2, 25), date(2026, 8, 26), 183),
        (252, date(2025, 8, 25), date(2026, 8, 26), 367),
    ],
)
def test_planner_uses_exact_n_plus_one_completed_sessions(
    returns, first, last, span_days
):
    plan = HistoryWindowPlanner().plan(
        as_of=AS_OF,
        requested_return_count=returns,
        calendar_id="XNYS",
        history_kind="exact_option_mark",
    )

    assert len(plan.required_sessions) == returns + 1
    assert plan.first_required_session == first
    assert plan.last_required_session == last
    assert plan.start_at == datetime.combine(first, datetime.min.time(), UTC)
    assert plan.end_at == datetime.combine(last + timedelta(days=1), datetime.min.time(), UTC)
    assert (plan.end_at - plan.start_at).days == span_days
    assert sum(len(chunk.sessions) for chunk in plan.chunks) == returns + 1


def test_planner_is_causal_for_intraday_and_honors_early_close():
    planner = HistoryWindowPlanner()
    default_plan = planner.plan(
        as_of=datetime(2026, 8, 27, 23, 59, tzinfo=UTC),
        requested_return_count=1,
        calendar_id="XNYS",
        history_kind="daily_close",
    )
    before_early_close = planner.plan(
        as_of=datetime(2026, 11, 27, 17, 59, tzinfo=UTC),
        requested_return_count=1,
        calendar_id="XNYS",
        history_kind="daily_close",
        availability_convention="session_close",
    )
    at_early_close = planner.plan(
        as_of=datetime(2026, 11, 27, 18, 0, tzinfo=UTC),
        requested_return_count=1,
        calendar_id="XNYS",
        history_kind="daily_close",
        availability_convention="session_close",
    )

    assert default_plan.last_required_session == date(2026, 8, 26)
    assert before_early_close.last_required_session == date(2026, 11, 25)
    assert at_early_close.last_required_session == date(2026, 11, 27)


def test_planner_handles_holiday_weekends_and_leap_day_as_exchange_sessions():
    planner = HistoryWindowPlanner()
    independence_weekend = planner.plan(
        as_of=datetime(2026, 7, 5, 18, tzinfo=UTC),
        requested_return_count=1,
        calendar_id="XNYS",
        history_kind="daily_close",
    )
    leap_day = planner.plan(
        as_of=datetime(2024, 3, 1, 12, tzinfo=UTC),
        requested_return_count=1,
        calendar_id="XNYS",
        history_kind="daily_close",
    )

    assert independence_weekend.required_sessions == (
        date(2026, 7, 1),
        date(2026, 7, 2),
    )
    assert leap_day.required_sessions == (date(2024, 2, 28), date(2024, 2, 29))


def test_daily_history_rejects_an_observation_available_after_analysis_boundary():
    class BarReference:
        def bars(self, symbols, *, start, end, remote_query=False):
            assert symbols == ["AAA"]
            assert remote_query is False
            return [
                {
                    "symbol": "AAA",
                    "bars": [
                        {
                            "session_date": "2026-08-25",
                            "timestamp": "2026-08-25T20:00:00Z",
                            "observed_at": "2026-08-26T00:00:00Z",
                            "close": 100.0,
                        },
                        {
                            "session_date": "2026-08-26",
                            "timestamp": "2026-08-26T20:00:00Z",
                            "observed_at": "2026-08-28T00:00:00Z",
                            "close": 101.0,
                        },
                    ],
                }
            ]

    returns, diagnostics, plan = CurrentPortfolioHistoricalRevaluationEngine(
        BarReference(), {}
    ).load_returns(["AAA"], as_of=AS_OF, lookback_sessions=1)

    assert plan.required_sessions == (date(2026, 8, 25), date(2026, 8, 26))
    assert returns["AAA"] == {}
    assert diagnostics["AAA"]["available_mark_count"] == 1
    assert diagnostics["AAA"]["return_count"] == 0


def test_chunking_uses_effective_limit_exact_union_and_exclusive_end():
    plan = HistoryWindowPlanner().plan(
        as_of=AS_OF,
        requested_return_count=252,
        calendar_id="XNYS",
        history_kind="exact_option_mark",
        local_maximum_days=376,
        advertised_maximum_days=100,
    )
    flattened = tuple(session for chunk in plan.chunks for session in chunk.sessions)

    assert plan.effective_maximum_days == 100
    assert plan.server_limit_negotiated is True
    assert len(plan.chunks) > 1
    assert flattened == plan.required_sessions
    assert len(flattened) == len(set(flattened))
    for chunk in plan.chunks:
        assert chunk.end_at - chunk.start_at <= timedelta(days=100)
        assert chunk.end_at == datetime.combine(
            chunk.sessions[-1] + timedelta(days=1), datetime.min.time(), UTC
        )


def test_contract_lifecycle_clipping_never_broadens_a_window():
    sessions = HistoryWindowPlanner().plan(
        as_of=AS_OF,
        requested_return_count=10,
        calendar_id="XNYS",
        history_kind="exact_option_mark",
    ).required_sessions
    clipped = clip_required_sessions(
        sessions, listing_date=date(2026, 8, 20), expiration_date=date(2026, 8, 25)
    )
    assert clipped == (date(2026, 8, 20), date(2026, 8, 21), date(2026, 8, 24), date(2026, 8, 25))
    assert set(clipped) < set(sessions)


def test_complete_sql_marks_and_greek_gaps_make_no_remote_request():
    sessions = (date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19))
    repository = _Repository([_history_row(CALL, session, 4.0) for session in sessions])
    remote = _Remote([_history_row(CALL, sessions[0], 99.0, provider_delta=0.5)])
    service = RiskOptionHistoryService(repository, remote, now_fn=lambda: AS_OF)

    result = service.query(
        OptionMarketHistoryRequest(
            symbols=[CALL],
            start_at=datetime(2026, 8, 17, tzinfo=UTC),
            end_at=datetime(2026, 8, 20, tzinfo=UTC),
            include_greeks=True,
        ),
        required_sessions_by_symbol={CALL: sessions},
    )

    assert remote.calls == []
    assert [item.mark for item in result.data] == [4.0, 4.0, 4.0]
    assert result.diagnostics["greek_gaps_trigger_remote_requests"] is False


def test_remote_query_contains_only_the_exact_sql_missing_session_range():
    sessions = tuple(date(2026, 8, day) for day in (17, 18, 19, 20, 21))
    repository = _Repository(
        [_history_row(CALL, session, 4.0) for session in sessions if session != sessions[2]]
    )
    remote = _Remote([_history_row(CALL, sessions[2], 4.25, source="refdata_remote")])
    service = RiskOptionHistoryService(repository, remote, now_fn=lambda: AS_OF)

    result = service.query(
        OptionMarketHistoryRequest(
            symbols=[CALL],
            start_at=datetime(2026, 8, 17, tzinfo=UTC),
            end_at=datetime(2026, 8, 22, tzinfo=UTC),
        ),
        required_sessions_by_symbol={CALL: sessions},
    )

    assert len(remote.calls) == 1
    assert remote.calls[0]["symbols"] == [CALL]
    assert remote.calls[0]["start_at"] == "2026-08-19T00:00:00+00:00"
    assert remote.calls[0]["end_at"] == "2026-08-20T00:00:00+00:00"
    assert len(result.data) == 5
    assert result.diagnostics["sql_mark_count_by_symbol"][CALL] == 4
    assert result.diagnostics["remote_mark_count_by_symbol"][CALL] == 1


def test_remote_option_mark_available_after_analysis_boundary_is_not_used():
    session = date(2026, 8, 19)
    remote = _Remote([_history_row(CALL, session, 4.25, source="refdata_remote")])
    service = RiskOptionHistoryService(_Repository([]), remote, now_fn=lambda: AS_OF)

    result = service.query(
        OptionMarketHistoryRequest(
            symbols=[CALL],
            start_at=datetime(2026, 8, 19, tzinfo=UTC),
            end_at=datetime(2026, 8, 20, tzinfo=UTC),
        ),
        required_sessions_by_symbol={CALL: (session,)},
        causal_as_of=datetime(2026, 8, 19, 23, 59, tzinfo=UTC),
    )

    assert result.data == []
    assert result.missing_symbols == [CALL]
    assert result.diagnostics["remote_mark_count_by_symbol"][CALL] == 0


def test_contracts_with_different_required_windows_are_not_widened_for_batching():
    a_sessions = (date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19))
    b_sessions = (date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 20))
    repository = _Repository([])
    remote = _Remote()
    service = RiskOptionHistoryService(repository, remote, now_fn=lambda: AS_OF)

    service.query(
        OptionMarketHistoryRequest(
            symbols=[CALL, PUT],
            start_at=datetime(2026, 8, 17, tzinfo=UTC),
            end_at=datetime(2026, 8, 21, tzinfo=UTC),
        ),
        required_sessions_by_symbol={CALL: a_sessions, PUT: b_sessions},
    )

    assert repository.calls == [
        ((CALL,), date(2026, 8, 17), date(2026, 8, 19)),
        ((PUT,), date(2026, 8, 18), date(2026, 8, 20)),
    ]
    assert [(call["symbols"], call["start_at"], call["end_at"]) for call in remote.calls] == [
        ([CALL], "2026-08-17T00:00:00+00:00", "2026-08-20T00:00:00+00:00"),
        ([PUT], "2026-08-18T00:00:00+00:00", "2026-08-21T00:00:00+00:00"),
    ]


def test_failed_remote_range_is_isolated_and_preserves_other_evidence():
    sessions = tuple(date(2026, 8, day) for day in (17, 18, 19, 20, 21))
    repository = _Repository([_history_row(CALL, sessions[2], 4.0)])
    remote = _Remote(
        [
            _history_row(CALL, sessions[3], 4.1, source="refdata_remote"),
            _history_row(CALL, sessions[4], 4.2, source="refdata_remote"),
        ],
        fail_starts=(sessions[0],),
    )
    service = RiskOptionHistoryService(repository, remote, now_fn=lambda: AS_OF)

    result = service.query(
        OptionMarketHistoryRequest(
            symbols=[CALL],
            start_at=datetime(2026, 8, 17, tzinfo=UTC),
            end_at=datetime(2026, 8, 22, tzinfo=UTC),
        ),
        required_sessions_by_symbol={CALL: sessions},
    )

    assert len(remote.calls) == 2
    assert len(result.data) == 3
    assert len(result.diagnostics["failed_remote_ranges"]) == 1
    failure = result.diagnostics["failed_remote_ranges"][0]["error"]
    assert failure["http_status"] == 422
    assert failure["server_error_code"] == "INVALID_REQUEST"
    assert failure["request_id"] == "fixture-request"


def test_sql_non_null_mark_wins_if_a_concurrent_capture_overlaps_remote_fill():
    session = date(2026, 8, 19)

    class ConcurrentCapture(dict):
        def __init__(self):
            super().__init__(_history_row(CALL, session, 5.0))
            self.mark_reads = 0

        def get(self, key, default=None):
            if key == "mark":
                self.mark_reads += 1
                if self.mark_reads == 1:
                    return None
            return super().get(key, default)

    capture = ConcurrentCapture()

    class ConcurrentRepository:
        def query_history(self, *_args, **_kwargs):
            return [capture]

    remote = _Remote([_history_row(CALL, session, 99.0, source="refdata_remote")])
    result = RiskOptionHistoryService(
        ConcurrentRepository(), remote, now_fn=lambda: AS_OF
    ).query(
        OptionMarketHistoryRequest(
            symbols=[CALL],
            start_at=datetime(2026, 8, 19, tzinfo=UTC),
            end_at=datetime(2026, 8, 20, tzinfo=UTC),
        ),
        required_sessions_by_symbol={CALL: (session,)},
    )

    assert len(remote.calls) == 1
    assert result.data[0].mark == 5.0
    assert result.data[0].source == "risk_mysql_option_prices"


def test_default_tail_option_window_is_one_bounded_request_below_refdata_limit():
    plan = HistoryWindowPlanner().plan(
        as_of=AS_OF,
        requested_return_count=252,
        calendar_id="XNYS",
        history_kind="exact_option_mark",
        advertised_maximum_days=376,
    )
    remote = _Remote(maximum=376)
    service = RiskOptionHistoryService(_Repository([]), remote, now_fn=lambda: AS_OF)

    service.query(
        OptionMarketHistoryRequest(
            symbols=[CALL], start_at=plan.start_at, end_at=plan.end_at
        ),
        required_sessions_by_symbol={CALL: plan.required_sessions},
    )

    assert len(remote.calls) == 1
    start = datetime.fromisoformat(remote.calls[0]["start_at"])
    end = datetime.fromisoformat(remote.calls[0]["end_at"])
    assert end - start == timedelta(days=367)
    assert end - start <= timedelta(days=376)
    assert remote.calls[0]["start_at"] == plan.start_at.isoformat()
    assert remote.calls[0]["end_at"] == plan.end_at.isoformat()


def test_refdata_client_preserves_structured_422_diagnostics():
    def handler(request: httpx.Request):
        return httpx.Response(
            422,
            headers={"X-Request-ID": "server-request"},
            json={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "option history interval exceeds 376 days",
                    "request_id": "server-request",
                    "details": {
                        "maximum_days": 376,
                        "exclusive_end": True,
                    },
                }
            },
        )

    client = ReferenceDataClient(
        {},
        client=httpx.Client(
            base_url="http://refdata.test", transport=httpx.MockTransport(handler)
        ),
    )
    with pytest.raises(ReferenceDataClientError) as captured:
        client.option_market_history(
            {
                "symbols": [CALL],
                "start_at": "2025-01-01T00:00:00+00:00",
                "end_at": "2026-08-27T00:00:00+00:00",
                "interval": "1d",
                "include_greeks": False,
            }
        )

    diagnostic = captured.value.diagnostic()
    assert diagnostic == {
        "http_status": 422,
        "server_error_code": "INVALID_REQUEST",
        "server_message": "option history interval exceeds 376 days",
        "server_details": {"maximum_days": 376, "exclusive_end": True},
        "request_id": "server-request",
        "endpoint": "/api/v1/options/market-history/query",
        "requested_start": "2025-01-01T00:00:00+00:00",
        "requested_end": "2026-08-27T00:00:00+00:00",
        "retryable": False,
    }


def test_chunk_helper_deduplicates_sessions_deterministically():
    sessions = [date(2026, 8, 18), date(2026, 8, 17), date(2026, 8, 18)]
    chunks = chunk_session_window(sessions, 10)
    assert chunks[0].sessions == (date(2026, 8, 17), date(2026, 8, 18))


def test_every_production_revaluation_constructor_injects_shared_sql_history_dependency():
    risk_root = Path(__file__).parents[1]
    calls = []
    for path in risk_root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "CurrentPortfolioHistoricalRevaluationEngine"
            ):
                calls.append((path.relative_to(risk_root).as_posix(), node))

    calls.sort(key=lambda item: item[0])
    assert [path for path, _ in calls] == ["api/hedge_service.py", "api/routes.py"]
    assert all(
        "risk_history_service" in {keyword.arg for keyword in node.keywords}
        for _, node in calls
    )
    for relative in ("api/routes.py", "api/hedge_service.py"):
        consumer_source = (risk_root / relative).read_text()
        assert "revaluation.scenario_pnl" in consumer_source
        assert "revaluation.unit_pnl" not in consumer_source


def test_tail_dependent_hedge_objective_stays_unavailable_for_insufficient_tail():
    request = HedgeCandidateQueryRequest.model_validate(
        {
            "portfolio_snapshot_id": "snapshot",
            "target": {
                "kind": "Portfolio",
                "label": "Portfolio",
                "position_ids": ["position"],
            },
            "objective": "VaR / Tail Risk",
            "objective_parameters": {"hedge_region": "near_curve"},
        }
    )
    insufficient = {
        "status": "insufficient_data",
        "var_usd": None,
        "expected_shortfall_usd": None,
    }
    evaluation = {
        "target": {
            "objective": {
                "before": {"near": insufficient, "far": insufficient},
                "after": {"near": insufficient, "far": insufficient},
            }
        }
    }

    assert not _primary_objective_usable(request, evaluation)


def test_release_thresholds_remain_126_complete_scenarios_and_90_percent_coverage():
    assert DEFAULT_CONTEXT["risk.history.minimum_observations"] == 126
    assert DEFAULT_CONTEXT["risk.factors.minimum_observations"] == 126
    assert DEFAULT_CONTEXT["risk.option_history.minimum_contract_iv_observations"] == 126
    assert DEFAULT_CONTEXT["risk.tail_risk.minimum_gross_delta_coverage"] == pytest.approx(0.90)
