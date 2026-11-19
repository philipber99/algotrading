from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from ..diagnostics import DIAGNOSTICS_VERSION, MAX_EXAMPLES_PER_REASON
from ...app import create_app
from ..errors import CoverageUnavailableError, InvalidRequestError
from ..model import ProfileRequest
from ..query_engine import TimeOfDayQueryEngine


UTC = timezone.utc
AS_OF = datetime(2026, 7, 30, 20, 0, tzinfo=UTC)


class _Adapter:
    def __init__(self, *, defect: bool = False) -> None:
        self.calls = 0
        self.defect = defect

    def price_many(self, requests, volatilities, *, requested_dtype):
        self.calls += 1
        if self.defect:
            raise RuntimeError("synthetic solver defect")
        return np.asarray([1.0] * len(requests), dtype=np.float64)


class _Registry:
    def __init__(self, *, defect: bool = False) -> None:
        self.adapter = _Adapter(defect=defect)

    def get(self, method):
        return self.adapter


def _session_row(
    session_date: date,
    *,
    first: int = 0,
    last: int = 385,
    coverage: float = 1.0,
) -> dict:
    return {
        "underlying_symbol": "QQQ",
        "session_date": session_date,
        "day_of_week": session_date.weekday(),
        "first_sample_minute": first,
        "last_sample_minute": last,
        "coverage_fraction": coverage,
    }


def _surface_row(
    session_date: date,
    minute: int,
    *,
    dte_days: float = 30.0,
    expiration_at: datetime | None = None,
    minimum_moneyness: float = -0.20,
    maximum_moneyness: float = 0.20,
    quality_flags: int = 0,
    butterfly_flags: int = 0,
    sample_age_seconds: float = 0.0,
) -> dict:
    collected = datetime.combine(session_date, datetime.min.time(), tzinfo=UTC)
    collected += timedelta(hours=13, minutes=30 + minute)
    expiration_at = expiration_at or collected + timedelta(days=dte_days)
    return {
        "underlying_symbol": "QQQ",
        "session_date": session_date,
        "session_minute": minute,
        "snapshot_id": f"{session_date}:{minute}:{expiration_at.isoformat()}",
        "source_collected_at": collected,
        "sample_age_seconds": sample_age_seconds,
        "expiration_at": expiration_at,
        "price_basis": "mid",
        "pricing_method": "tian",
        "fit_method": "SVI",
        "svi_a": 0.04,
        "svi_b": 0.10,
        "svi_rho": -0.20,
        "svi_m": 0.0,
        "svi_sigma": 0.10,
        "pchip_k": [],
        "pchip_total_variance": [],
        "spot_price": 100.0,
        "forward_price": 100.0,
        "time_to_expiry_seconds": int(dte_days * 86400),
        "risk_free_rate": 0.04,
        "dividend_yield": 0.01,
        "quality_flags": quality_flags,
        "butterfly_arbitrage_flags": butterfly_flags,
        "calendar_arbitrage_flags": 0,
        "minimum_log_forward_moneyness": minimum_moneyness,
        "maximum_log_forward_moneyness": maximum_moneyness,
        "__file_rank": 0,
    }


def _request(
    *,
    mode: str = "comparable_entry",
    entry: int | None = None,
    dte_days: float = 30.0,
    tolerance: float = 2.0,
    moneyness: float = 0.0,
    quality_excluded: list[int | str] | None = None,
) -> ProfileRequest:
    return ProfileRequest(
        underlying_symbol="QQQ",
        as_of_utc=AS_OF,
        mode=mode,
        entry_session_minute=entry,
        value_component="total_value",
        price_basis="mid",
        legs=[
            {
                "side": "long",
                "put_call": "CALL",
                "quantity": 1,
                "strike": 100.0 * math.exp(moneyness),
                "expiration_at": AS_OF + timedelta(days=dte_days),
                "anchor_forward": 100.0,
            }
        ],
        cohort={
            "historical_window": "10d",
            "dte_tolerance_days": tolerance,
            "minimum_session_coverage": 0.9,
            "minimum_sample_count": 1,
            "quality_flags_excluded": quality_excluded or [],
        },
        aggregation={"interval_minutes": 5},
    )


def _engine(
    sessions: list[dict],
    surfaces: list[dict],
    *,
    registry: _Registry | None = None,
) -> TimeOfDayQueryEngine:
    context = {
        "options_chain.time_of_day_cache.output_directory": "/tmp/diagnostic-cache-test",
        "options_chain.time_of_day_cache.symbols": "QQQ",
        "options_chain.time_of_day_cache.windows": "10d:10,3m:63",
        "options_chain.time_of_day_cache.price_bases": "mid,bid",
    }
    engine = TimeOfDayQueryEngine(
        context,
        solver_registry=registry or _Registry(),
    )
    engine._manifest = {
        "generation_id": "generation-diagnostics",
        "data_through_session": "2026-07-30",
        "symbols": ["QQQ"],
        "windows": [
            {"name": "10d", "label": "10d", "sessions": 10},
            {"name": "3m", "label": "3m", "sessions": 63},
        ],
        "default_window": "10d",
        "sample_interval_minutes": 5,
        "max_snapshot_age_seconds": 120,
        "price_bases": ["mid", "bid"],
        "coverage_by_symbol": {
            "QQQ": {
                "minimum_dte_days": 10.0,
                "maximum_dte_days": 60.0,
                "minimum_moneyness": -0.20,
                "maximum_moneyness": 0.20,
            }
        },
        "files": [],
    }
    engine._session_rows = sessions
    engine._ready = True
    engine.ensure_current = lambda: None
    table = (
        pa.Table.from_pylist(surfaces)
        if surfaces
        else pa.table({"__file_rank": pa.array([], pa.int32())})
    )
    engine._load_surface_table = mock.Mock(return_value=table)
    engine.startup = lambda: None
    engine.shutdown = lambda: None
    return engine


def _details(error: CoverageUnavailableError) -> dict:
    json.dumps(error.details)
    assert error.details["diagnostics_version"] == DIAGNOSTICS_VERSION
    return error.details


def test_late_enter_and_hold_reports_entry_coverage_and_suggestion():
    dates = [date(2026, 7, 28), date(2026, 7, 29)]
    engine = _engine(
        [_session_row(item, last=385) for item in dates],
        [_surface_row(item, minute) for item in dates for minute in (380, 385)],
    )

    with pytest.raises(CoverageUnavailableError) as caught:
        engine.profile(_request(mode="enter_and_hold", entry=390))

    details = _details(caught.value)
    assert (
        details["rejections"]["dominant_reason"]
        == "NO_SESSION_MINUTE_AT_OR_AFTER_ENTRY"
    )
    assert (
        details["rejections"]["by_reason"]["NO_SESSION_MINUTE_AT_OR_AFTER_ENTRY"] == 2
    )
    assert details["funnel"]["entry_eligible_sessions"] == 0
    assert details["cache_coverage"]["latest_common_session_minute"] == 385
    assert details["suggestions"][0] == {
        "code": "USE_EARLIER_ENTRY_MINUTE",
        "message": "The requested entry minute is later than the common cached session range.",
        "parameters": {
            "requested_entry_session_minute": 390,
            "recommended_maximum_entry_session_minute": 385,
        },
    }


def test_mixed_any_and_common_minute_coverage_and_availability():
    sessions = [
        _session_row(date(2026, 7, 28), last=385),
        _session_row(date(2026, 7, 29), last=390),
    ]
    engine = _engine(sessions, [])

    coverage = engine.coverage("QQQ", "10d")

    assert coverage["latest_any_session_minute"] == 390
    assert coverage["latest_common_session_minute"] == 385
    availability = {
        item["session_minute"]: item["session_count"]
        for item in coverage["session_minute_availability"]
    }
    assert availability[385] == 2
    assert availability[390] == 1


def test_dte_failure_reports_failed_leg_and_nearest_dte():
    session = date(2026, 7, 29)
    engine = _engine(
        [_session_row(session)],
        [_surface_row(session, 5, dte_days=20)],
    )

    with pytest.raises(CoverageUnavailableError) as caught:
        engine.profile(_request(dte_days=30, tolerance=2))

    details = _details(caught.value)
    assert details["rejections"]["dominant_reason"] == "DTE_OUTSIDE_TOLERANCE"
    leg = details["per_leg"][0]
    assert leg["leg_index"] == 0
    assert leg["target_dte_days"] == 30
    assert leg["nearest_dte_days_min"] == 20
    assert leg["dte_match_failures"] == 1


@pytest.mark.parametrize(
    ("moneyness", "reason", "field"),
    [
        (-0.30, "MONEYNESS_BELOW_SURFACE_COVERAGE", "moneyness_below_failures"),
        (0.30, "MONEYNESS_ABOVE_SURFACE_COVERAGE", "moneyness_above_failures"),
    ],
)
def test_moneyness_bounds_are_distinguished(moneyness, reason, field):
    session = date(2026, 7, 29)
    engine = _engine(
        [_session_row(session)],
        [_surface_row(session, 5, minimum_moneyness=-0.2, maximum_moneyness=0.2)],
    )

    with pytest.raises(CoverageUnavailableError) as caught:
        engine.profile(_request(moneyness=moneyness))

    details = _details(caught.value)
    assert details["rejections"]["dominant_reason"] == reason
    assert details["per_leg"][0][field] == 1
    assert details["suggestions"][0]["code"] == "USE_STRIKE_WITHIN_SURFACE_COVERAGE"


def test_quality_exclusion_is_measured_and_only_then_suggested():
    session = date(2026, 7, 29)
    engine = _engine(
        [_session_row(session)],
        [_surface_row(session, 5, butterfly_flags=1)],
    )

    with pytest.raises(CoverageUnavailableError) as caught:
        engine.profile(_request(quality_excluded=["FIT_ARBITRAGE"]))

    details = _details(caught.value)
    assert details["rejections"]["dominant_reason"] == "FIT_ARBITRAGE_EXCLUDED"
    assert details["rejections"]["by_reason"]["FIT_ARBITRAGE_EXCLUDED"] == 1
    assert [item["code"] for item in details["suggestions"]] == [
        "REMOVE_OR_RELAX_QUALITY_EXCLUSIONS"
    ]


def test_missing_later_expiration_is_nonfatal_when_entry_point_is_complete():
    session = date(2026, 7, 29)
    entry_expiration = datetime(2026, 8, 28, 13, 30, tzinfo=UTC)
    other_expiration = entry_expiration + timedelta(days=7)
    engine = _engine(
        [_session_row(session, last=5)],
        [
            _surface_row(session, 0, expiration_at=entry_expiration),
            _surface_row(session, 5, expiration_at=other_expiration),
        ],
    )

    result, _ = engine.profile(_request(mode="enter_and_hold", entry=0))

    diagnostics = result["diagnostics"]
    assert diagnostics["rejections"]["by_reason"]["MISSING_LATER_EXPIRATION"] == 1
    assert diagnostics["entry"]["later_path_candidate_points"] == 2
    assert diagnostics["entry"]["later_path_complete_points"] == 1


def test_success_has_compact_diagnostics_and_cache_hit_is_identical():
    session = date(2026, 7, 29)
    registry = _Registry()
    engine = _engine(
        [_session_row(session, last=10)],
        [_surface_row(session, minute) for minute in (5, 10)],
        registry=registry,
    )
    request = _request()

    first, first_hit = engine.profile(request)
    second, second_hit = engine.profile(request)

    assert first_hit is False
    assert second_hit is True
    assert second["diagnostics"] == first["diagnostics"]
    funnel = first["diagnostics"]["funnel"]
    assert funnel["candidate_session_minutes"] == 2
    assert funnel["complete_observations"] == 2
    assert funnel["sessions_with_complete_observations"] == 1
    assert funnel["profile_bins"] == 2
    assert registry.adapter.calls == 1


def test_entry_slippage_is_reported_on_success():
    session = date(2026, 7, 29)
    engine = _engine(
        [_session_row(session, first=5, last=5)],
        [_surface_row(session, 5)],
    )

    result, _ = engine.profile(_request(mode="enter_and_hold", entry=1))

    entry = result["diagnostics"]["entry"]
    assert entry["sessions_resolved_after_entry"] == 1
    assert entry["minimum_resolved_entry_minute"] == 5
    assert entry["maximum_entry_slippage_minutes"] == 4


def test_multileg_failure_marks_only_the_rejected_leg():
    session = date(2026, 7, 29)
    engine = _engine(
        [_session_row(session)],
        [_surface_row(session, 5)],
    )
    request = _request()
    request = request.model_copy(
        update={
            "legs": [
                request.legs[0],
                request.legs[0].model_copy(
                    update={
                        "strike": 100.0 * math.exp(0.30),
                        "side": "short",
                    }
                ),
            ]
        }
    )

    with pytest.raises(CoverageUnavailableError) as caught:
        engine.profile(request)

    legs = _details(caught.value)["per_leg"]
    assert legs[0]["moneyness_above_failures"] == 0
    assert legs[1]["moneyness_above_failures"] == 1


def test_low_sample_count_remains_a_success_warning():
    session = date(2026, 7, 29)
    engine = _engine(
        [_session_row(session)],
        [_surface_row(session, 5)],
    )
    request = _request().model_copy(
        update={
            "cohort": _request().cohort.model_copy(update={"minimum_sample_count": 2})
        }
    )

    result, _ = engine.profile(request)

    assert "LOW_SAMPLE_COUNT" in {item["code"] for item in result["warnings"]}
    assert (
        result["diagnostics"]["rejections"]["by_reason"]["INSUFFICIENT_SAMPLE_COUNT"]
        == 1
    )


@pytest.mark.parametrize(
    ("update", "reason", "suggestion"),
    [
        ({"underlying_symbol": "SPY"}, "SYMBOL_NOT_CACHED", None),
        (
            {"price_basis": "ask"},
            "PRICE_BASIS_UNAVAILABLE",
            "USE_AVAILABLE_PRICE_BASIS",
        ),
    ],
)
def test_request_level_coverage_errors_use_diagnostic_shape(
    update,
    reason,
    suggestion,
):
    engine = _engine([_session_row(date(2026, 7, 29))], [])

    with pytest.raises(CoverageUnavailableError) as caught:
        engine.profile(_request().model_copy(update=update))

    details = _details(caught.value)
    assert details["rejections"]["dominant_reason"] == reason
    if suggestion is not None:
        assert suggestion in {item["code"] for item in details["suggestions"]}


def test_examples_are_bounded_and_json_native():
    dates = [date(2026, 7, day) for day in range(20, 30)]
    engine = _engine(
        [_session_row(item) for item in dates],
        [_surface_row(item, 5, dte_days=10) for item in dates],
    )

    with pytest.raises(CoverageUnavailableError) as caught:
        engine.profile(_request(dte_days=30, tolerance=1))

    details = _details(caught.value)
    examples = details["rejections"]["examples"]["DTE_OUTSIDE_TOLERANCE"]
    assert len(examples) == MAX_EXAMPLES_PER_REASON
    assert all(isinstance(item["session_date"], str) for item in examples)


def test_unknown_coverage_window_is_422():
    engine = _engine([_session_row(date(2026, 7, 29))], [])
    with pytest.raises(InvalidRequestError):
        engine.coverage("QQQ", "unknown")


def test_http_409_envelope_and_structured_log_include_request_id(caplog):
    session = date(2026, 7, 29)
    engine = _engine(
        [_session_row(session, last=385)],
        [_surface_row(session, 385)],
    )
    app = create_app(engine=engine)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        response = client.post(
            "/v1/time-of-day/profile",
            headers={"X-Request-ID": "diagnostic-test-001"},
            json=json.loads(
                _request(mode="enter_and_hold", entry=390).model_dump_json()
            ),
        )

    assert response.status_code == 409
    assert response.headers["X-Request-ID"] == "diagnostic-test-001"
    error = response.json()["error"]
    assert error["request_id"] == "diagnostic-test-001"
    assert (
        error["details"]["rejections"]["dominant_reason"]
        == "NO_SESSION_MINUTE_AT_OR_AFTER_ENTRY"
    )
    summary = next(
        record.message
        for record in caplog.records
        if "event=cache_profile_coverage_unavailable" in record.message
    )
    assert "request_id=diagnostic-test-001" in summary
    assert "dominant_reason=NO_SESSION_MINUTE_AT_OR_AFTER_ENTRY" in summary
    assert "current_strategy_value" not in summary


def test_coverage_and_profile_etags_remain_conditional():
    session = date(2026, 7, 29)
    engine = _engine(
        [_session_row(session, last=5)],
        [_surface_row(session, 5)],
    )
    app = create_app(engine=engine)

    with TestClient(app) as client:
        coverage = client.get("/v1/cache/coverage?symbol=QQQ&window=10d")
        assert coverage.status_code == 200
        coverage_etag = coverage.headers["ETag"]
        assert (
            client.get(
                "/v1/cache/coverage?symbol=QQQ&window=10d",
                headers={"If-None-Match": coverage_etag},
            ).status_code
            == 304
        )

        body = json.loads(_request().model_dump_json())
        profile = client.post("/v1/time-of-day/profile", json=body)
        assert profile.status_code == 200
        profile_etag = profile.headers["ETag"]
        assert (
            client.post(
                "/v1/time-of-day/profile",
                json=body,
                headers={"If-None-Match": profile_etag},
            ).status_code
            == 304
        )
    assert engine._load_surface_table.call_count == 1


def test_unexpected_solver_defect_is_not_reclassified_as_coverage():
    session = date(2026, 7, 29)
    engine = _engine(
        [_session_row(session)],
        [_surface_row(session, 5)],
        registry=_Registry(defect=True),
    )

    with pytest.raises(RuntimeError, match="synthetic solver defect"):
        engine.profile(_request())


def test_single_file_reads_ignore_hive_names_in_increment_parents(tmp_path):
    session = date(2026, 7, 31)
    context = {
        "options_chain.time_of_day_cache.output_directory": str(tmp_path),
        "options_chain.time_of_day_cache.symbols": "QQQ",
    }
    engine = TimeOfDayQueryEngine(context, solver_registry=_Registry())
    partition = tmp_path / "data/increments/session_date=2026-07-31"
    session_path = partition / "session_index/symbol=QQQ/part.parquet"
    surface_path = partition / "surface_observations/symbol=QQQ/part.parquet"
    session_path.parent.mkdir(parents=True)
    surface_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_session_row(session)]), session_path)
    pq.write_table(pa.Table.from_pylist([_surface_row(session, 5)]), surface_path)
    manifest = {
        "session_dates": [session.isoformat()],
        "files": [
            {
                "dataset": "session_index",
                "path": str(session_path.relative_to(tmp_path)),
            },
            {
                "dataset": "surface_observations",
                "path": str(surface_path.relative_to(tmp_path)),
                "symbol": "QQQ",
                "minimum_session": session.isoformat(),
                "maximum_session": session.isoformat(),
            },
        ],
    }

    session_rows = engine._load_session_rows(manifest)
    surface_table = engine._load_surface_table(
        manifest,
        symbol="QQQ",
        session_dates=[session],
        price_basis="mid",
    )

    assert len(session_rows) == 1
    assert session_rows[0]["session_date"] == session
    assert surface_table.num_rows == 1
    assert surface_table.schema.field("session_date").type == pa.date32()
