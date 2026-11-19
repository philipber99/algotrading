"""Regression checks for sensitive values at external error boundaries."""

from __future__ import annotations

import logging
import traceback
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient

from algotrading.refdata.options.Calculations import FredCurve
from algotrading.refdata.options.gui.app import create_app
from algotrading.refdata.options.gui.cache.query_engine import TimeOfDayQueryEngine
from algotrading.refdata.options.gui.screener import scenarios


SENSITIVE_VALUE = "synthetic-sensitive-value-for-security-test"


class CacheEngine:
    ready = True
    generation_id = "test-generation"

    def startup(self):
        pass

    def shutdown(self):
        pass

    def ensure_current(self):
        pass

    def health(self):
        return {"ready": self.ready}


def context(caplog):
    logger = logging.getLogger("options-publication-security")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    return {"logger": logger}


def test_validation_response_does_not_echo_submitted_values(caplog):
    app = create_app(
        context=context(caplog), engine=CacheEngine(), screener_service=Mock()
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/screener/spreads",
            json={"underlying_symbol": "SPY", "result_count_per_strategy": SENSITIVE_VALUE},
        )
    assert response.status_code == 422
    assert response.json()["error"]["message"] == "Request validation failed"
    assert SENSITIVE_VALUE not in response.text + caplog.text


def test_unhandled_errors_do_not_log_sensitive_exception_text(caplog):
    app = create_app(
        context=context(caplog), engine=CacheEngine(), screener_service=Mock()
    )

    @app.get("/synthetic-failure")
    def failure():
        raise RuntimeError(SENSITIVE_VALUE)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/synthetic-failure")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "error_class=RuntimeError" in caplog.text
    assert SENSITIVE_VALUE not in response.text + caplog.text


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_screener_startup_failure_does_not_log_connection_details(
    caplog, monkeypatch, cleanup_failure
):
    resources = Mock()
    if cleanup_failure:
        resources.close.side_effect = RuntimeError(SENSITIVE_VALUE)
    monkeypatch.setattr(scenarios, "PricingResources", lambda *args, **kwargs: resources)

    def unavailable(*args, **kwargs):
        raise ConnectionError(SENSITIVE_VALUE)

    app = create_app(
        context=context(caplog), engine=CacheEngine(),
        clickhouse_client_factory=unavailable,
    )
    with TestClient(app) as client:
        response = client.get("/v1/screener/health")
    assert response.status_code == 503
    assert "error_class=ConnectionError" in caplog.text
    assert SENSITIVE_VALUE not in response.text + caplog.text


def test_cache_health_does_not_expose_startup_exception_text(caplog, monkeypatch, tmp_path):
    config = context(caplog)
    config.update({
        "options_chain.symbols": "SPY",
        "options_chain.time_of_day_cache.output_directory": str(tmp_path),
    })
    engine = TimeOfDayQueryEngine(config, solver_registry=object())

    def unavailable(*args, **kwargs):
        raise OSError(SENSITIVE_VALUE)

    monkeypatch.setattr(engine, "_reload", unavailable)
    app = create_app(context=config, engine=engine, screener_service=Mock())
    with TestClient(app) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["details"]["startup_error"] == "OSError"
    assert SENSITIVE_VALUE not in response.text + caplog.text


@pytest.mark.parametrize("stage", ["transport", "status", "json"])
def test_fred_failures_do_not_expose_api_keys_in_tracebacks(stage):
    leaked_url = f"https://example.test/rates?api_key={SENSITIVE_VALUE}"
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"observations": []}
    session = Mock()
    session.get.return_value = response
    if stage == "transport":
        session.get.side_effect = requests.ConnectionError(leaked_url)
    elif stage == "status":
        response.raise_for_status.side_effect = requests.HTTPError(leaked_url)
    else:
        response.json.side_effect = ValueError(SENSITIVE_VALUE)
    client = Mock()
    client.query.return_value = SimpleNamespace(result_rows=[])
    curve = FredCurve(
        {"options_chain.clickhouse.database": "test",
         "options_calcs.fred.api_key": SENSITIVE_VALUE},
        client, logging.getLogger("fred-publication-security"), http_session=session,
    )
    with pytest.raises(RuntimeError, match="FRED request failed") as caught:
        curve.refresh()
    rendered = "".join(traceback.format_exception(caught.value))
    assert SENSITIVE_VALUE not in rendered
    assert leaked_url not in rendered
