from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from .. import bootstrap
from ..config import RiskApiConfig


def test_build_context_uses_standard_loader_once_and_sets_required_fields(
    monkeypatch, tmp_path
):
    calls = []

    class Common:
        @staticmethod
        def load_configuration(context, home, environment, *, quiet):
            calls.append((context, home, environment, quiet))
            return context

    fields = SimpleNamespace(
        PARAM_ALGO_HOME="AlgoHome",
        PARAM_APP_NAME="AppName",
        PARAM_LOGGER="logger",
    )

    def fake_import(name):
        return Common if name == "algotrading.Common" else fields

    monkeypatch.setattr(bootstrap, "import_module", fake_import)
    monkeypatch.setenv("ALGO_HOME", str(tmp_path))

    context = bootstrap.build_context("risk-test", quiet_configuration=True)

    assert len(calls) == 1
    assert calls[0][1] == str(tmp_path.resolve()) + os.sep
    assert calls[0][2] == "risk-test"
    assert calls[0][3] is True
    assert context["AlgoHome"] == str(tmp_path.resolve())
    assert context["AppName"] == "risk-api"
    assert context["risk.api.environment"] == "risk-test"
    assert context["logger"].name == "algotrading.risk.api"


@pytest.mark.parametrize(
    "overrides",
    [
        {"risk.api.cors_origins": "*"},
        {"risk.api.account_cache_ttl_seconds": -1},
        {"risk.api.account_cache_ttl_seconds": "nan"},
        {"risk.api.max_concurrent_broker_calls": 2},
        {"risk.api.port": 70000},
    ],
)
def test_invalid_server_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        RiskApiConfig.from_context(overrides)


def test_explicit_cors_origins_are_parsed_without_wildcards():
    config = RiskApiConfig.from_context(
        {"risk.api.cors_origins": "http://127.0.0.1:3000, https://gui.test"}
    )
    assert config.cors_origins == (
        "http://127.0.0.1:3000",
        "https://gui.test",
    )
