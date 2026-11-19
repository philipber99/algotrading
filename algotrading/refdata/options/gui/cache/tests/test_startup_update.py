from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from ... import app as endpoints_module
from .. import startup_update as update_module
from ...api import _parser
from ..builder import BuildResult
from ...app import create_app
from ..startup_update import (
    CacheUpdateDecision,
    choose_update_mode,
    update_cache_if_needed,
)


UTC = timezone.utc


class _Store:
    def __init__(self, manifest):
        self.config = SimpleNamespace(local_timezone="America/Los_Angeles")
        self.manifest = manifest
        self.manifests = SimpleNamespace()

    def load_active_manifest(self):
        return self.manifest


def _manifest(updated: str) -> dict:
    return {
        "generation_id": "generation-test",
        "built_at_utc": updated,
        "data_through_session": "2026-07-30",
    }


def test_api_parser_accepts_update_cache():
    options = _parser().parse_args(["-e", "options_chain", "--update-cache"])
    assert options.update_cache is True


def test_generation_updated_yesterday_gets_daily_increment():
    decision = choose_update_mode(
        _Store(_manifest("2026-07-30T19:00:00Z")),
        now=datetime(2026, 7, 31, 19, 0, tzinfo=UTC),
    )
    assert decision.mode == "daily"
    assert decision.reason == "cache_updated_yesterday"
    assert decision.last_updated_date == date(2026, 7, 30)


def test_local_timezone_controls_yesterday_boundary():
    decision = choose_update_mode(
        _Store(_manifest("2026-07-31T06:30:00Z")),
        now=datetime(2026, 7, 31, 19, 0, tzinfo=UTC),
    )
    assert decision.mode == "daily"
    assert decision.last_updated_date == date(2026, 7, 30)


def test_older_generation_crossing_weekend_gets_weekly_rebuild():
    decision = choose_update_mode(
        _Store(_manifest("2026-07-31T19:00:00Z")),
        now=datetime(2026, 8, 3, 19, 0, tzinfo=UTC),
    )
    assert decision.mode == "weekly"
    assert decision.reason == "cache_is_older_than_one_day_and_weekend_elapsed"


def test_older_generation_without_weekend_uses_existing_cache():
    decision = choose_update_mode(
        _Store(_manifest("2026-07-29T19:00:00Z")),
        now=datetime(2026, 7, 31, 19, 0, tzinfo=UTC),
    )
    assert decision.mode is None
    assert decision.reason == "cache_is_older_than_one_day_without_elapsed_weekend"


def test_generation_updated_today_uses_existing_cache():
    decision = choose_update_mode(
        _Store(_manifest("2026-07-31T18:00:00Z")),
        now=datetime(2026, 7, 31, 19, 0, tzinfo=UTC),
    )
    assert decision.mode is None
    assert decision.reason == "cache_updated_today_or_later"


def test_missing_generation_gets_weekly_bootstrap():
    decision = choose_update_mode(
        _Store(None),
        now=datetime(2026, 7, 31, 19, 0, tzinfo=UTC),
    )
    assert decision.mode == "weekly"
    assert decision.reason == "no_active_generation"


def test_update_calls_builder_directly_and_closes_client(tmp_path):
    context = {
        "options_chain.time_of_day_cache.output_directory": str(tmp_path),
        "options_chain.time_of_day_cache.symbols": "QQQ",
    }
    decision = CacheUpdateDecision(
        "daily",
        "cache_updated_yesterday",
        date(2026, 7, 30),
        date(2026, 7, 31),
    )
    client = mock.Mock()
    builder = mock.Mock()
    builder.build.return_value = BuildResult(
        status="ok",
        mode="daily",
        build_id="build-test",
        generation_id="generation-new",
        data_through_session="2026-07-31",
    )

    with (
        mock.patch.object(
            update_module,
            "choose_update_mode",
            return_value=decision,
        ),
        mock.patch.object(
            update_module,
            "CacheBuilder",
            return_value=builder,
        ) as builder_type,
    ):
        outcome = update_cache_if_needed(
            context,
            logger=logging.getLogger("startup-update-test"),
            client_factory=lambda _: client,
        )

    assert outcome.result is builder.build.return_value
    assert builder.build.call_args.args[0].mode == "daily"
    builder_type.assert_called_once()
    client.close.assert_called_once_with()


def test_update_check_does_not_create_client_when_cache_is_used(tmp_path):
    context = {
        "options_chain.time_of_day_cache.output_directory": str(tmp_path),
        "options_chain.time_of_day_cache.symbols": "QQQ",
    }
    decision = CacheUpdateDecision(
        None,
        "cache_updated_today_or_later",
        date(2026, 7, 31),
        date(2026, 7, 31),
    )
    client_factory = mock.Mock()

    with mock.patch.object(
        update_module,
        "choose_update_mode",
        return_value=decision,
    ):
        outcome = update_cache_if_needed(
            context,
            logger=logging.getLogger("startup-update-test"),
            client_factory=client_factory,
        )

    assert outcome.result is None
    client_factory.assert_not_called()


def test_app_runs_update_before_engine_startup():
    events = []
    engine = mock.Mock()
    engine.startup.side_effect = lambda: events.append("engine")
    context = {
        "options_chain.time_of_day_cache.output_directory": "/tmp/startup-update-test",
        "options_chain.time_of_day_cache.symbols": "QQQ",
    }

    with mock.patch.object(
        endpoints_module,
        "update_cache_if_needed",
        side_effect=lambda *args, **kwargs: events.append("update"),
    ):
        app = create_app(context=context, engine=engine, update_cache=True)
        with TestClient(app) as client:
            assert client.get("/").status_code == 200

    assert events == ["update", "engine"]
    engine.shutdown.assert_called_once_with()
