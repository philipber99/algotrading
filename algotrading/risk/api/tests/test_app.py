from __future__ import annotations

import io
import logging

from fastapi.testclient import TestClient

from ..app import create_app
from ..broker_source import BrokerAccountSource
from ..config import RiskApiConfig
from .fakes import FakeSource, account


def _client(source, *, logger: logging.Logger | None = None, ttl: float = 2):
    context = {"logger": logger or logging.getLogger("test-risk-api")}
    config = RiskApiConfig(account_cache_ttl_seconds=ttl)
    return TestClient(create_app(context=context, source=source, config=config))


def test_health_endpoints_do_not_call_account_source():
    source = FakeSource()
    with _client(source) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert source.all_calls == 0
    assert source.one_calls == []


def test_desktop_origin_snapshot_preflight_is_allowed():
    source = FakeSource()
    config = RiskApiConfig(cors_origins=("http://127.0.0.1:5173",))
    app = create_app(
        context={"logger": logging.getLogger("cors-test")},
        source=source,
        config=config,
    )
    client = TestClient(app)
    response = client.options(
        "/api/v1/portfolio/snapshot?account_id=all&refresh=true",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "GET" in response.headers["access-control-allow-methods"]
    assert source.all_calls == 0


def test_account_headers_metadata_request_id_and_include_positions():
    source = FakeSource()
    with _client(source) as client:
        response = client.get(
            "/api/v1/accounts",
            params={"include_positions": "false"},
            headers={"X-Request-ID": "request-123"},
        )

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Pragma"] == "no-cache"
    assert response.headers["X-Request-ID"] == "request-123"
    body = response.json()
    assert body["meta"] == {
        "count": 1,
        "as_of": "2026-08-15T18:30:00Z",
        "source": "TDABrokerInterface",
        "cache": "miss",
        "request_id": "request-123",
    }
    assert body["data"][0]["position_count"] == 1
    assert "positions" not in body["data"][0]


def test_specific_routes_use_only_specific_source_path():
    source = FakeSource()
    with _client(source, ttl=0) as client:
        account_response = client.get("/api/v1/accounts/FAKE-2222")
        positions_response = client.get(
            "/api/v1/accounts/FAKE-2222/positions",
            params={"refresh": "true"},
        )
        query_response = client.get(
            "/api/v1/positions",
            params={"account_id": "FAKE-2222"},
        )

    assert account_response.status_code == 200
    assert positions_response.status_code == 200
    assert query_response.status_code == 200
    assert source.one_calls == ["FAKE-2222", "FAKE-2222", "FAKE-2222"]
    assert source.all_calls == 0


def test_all_positions_calls_all_accounts_once_and_flattens_without_deduping():
    source = FakeSource(all_payload=[account("FAKE-A"), account("FAKE-B")])
    with _client(source) as client:
        response = client.get("/api/v1/positions")

    assert response.status_code == 200
    assert response.json()["meta"]["count"] == 2
    assert [item["account_id"] for item in response.json()["data"]] == [
        "FAKE-A",
        "FAKE-B",
    ]
    assert source.all_calls == 1
    assert source.one_calls == []


def test_empty_accounts_and_missing_positions_return_valid_empty_lists():
    empty_accounts = FakeSource(all_payload=[])
    with _client(empty_accounts) as client:
        response = client.get("/api/v1/accounts")
    assert response.status_code == 200
    assert response.json()["data"] == []
    assert response.json()["meta"]["count"] == 0

    empty_positions = FakeSource(one_payload=account(include_positions=False))
    with _client(empty_positions) as client:
        response = client.get("/api/v1/accounts/FAKE-0001/positions")
    assert response.status_code == 200
    assert response.json()["data"] == []


def test_not_found_schema_and_broker_errors_are_sanitized_and_no_store():
    not_found = FakeSource(one_payload=404)
    with _client(not_found) as client:
        response = client.get("/api/v1/accounts/FAKE-404")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ACCOUNT_NOT_FOUND"
    assert response.json()["error"]["details"] is None
    assert response.headers["Cache-Control"] == "no-store"

    malformed = FakeSource(all_payload={"secret_payload": "DO_NOT_LEAK"})
    with _client(malformed) as client:
        response = client.get("/api/v1/accounts")
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "UPSTREAM_SCHEMA_ERROR"
    assert "DO_NOT_LEAK" not in response.text

    class RaisingBroker:
        def get_accounts(self):
            raise RuntimeError("secret-token-and-account-FAKE-9999")

        def get_account(self, *, account):
            raise AssertionError("not called")

    logger = logging.getLogger("test-risk-upstream")
    source = BrokerAccountSource(
        {"logger": logger}, broker_factory=lambda _: RaisingBroker()
    )
    with _client(source, logger=logger) as client:
        response = client.get("/api/v1/accounts")
    assert response.status_code == 502
    assert response.json()["error"] == {
        "code": "BROKER_UPSTREAM_ERROR",
        "message": "Account information is temporarily unavailable",
        "request_id": response.headers["X-Request-ID"],
        "details": None,
    }
    assert "secret-token" not in response.text


def test_validation_error_is_sanitized():
    source = FakeSource()
    with _client(source) as client:
        response = client.get("/api/v1/accounts/%27full-account-number%27")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "full-account-number" not in response.text
    assert source.one_calls == []


def test_readiness_reports_missing_broker_method_without_fallback_call():
    class MissingMethodBroker:
        def get_accounts(self):
            raise AssertionError("health must not invoke this")

    source = BrokerAccountSource(
        {"logger": logging.getLogger("test-missing-method")},
        broker_factory=lambda _: MissingMethodBroker(),
    )
    with _client(source) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["required_methods_callable"] is False
    assert "get_account" in response.text


def test_logs_use_route_template_and_never_account_payloads():
    stream = io.StringIO()
    logger = logging.getLogger("risk-api-capture")
    logger.handlers = [logging.StreamHandler(stream)]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    source = FakeSource(one_payload=account("FULL-ACCOUNT-918273645"))

    with _client(source, logger=logger, ttl=0) as client:
        response = client.get("/api/v1/accounts/FULL-ACCOUNT-918273645")

    assert response.status_code == 200
    logs = stream.getvalue()
    assert "/api/v1/accounts/{account_id}" in logs
    assert "FULL-ACCOUNT-918273645" not in logs
    assert "FICTIONAL" not in logs
    assert "initialBalances" not in logs


def test_legacy_broker_response_debug_log_is_filtered():
    stream = io.StringIO()
    logger = logging.getLogger("risk-broker-payload-filter")
    logger.handlers = [logging.StreamHandler(stream)]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    class VerboseBroker:
        def get_accounts(self):
            return []

        def get_account(self, *, account):
            payload = globals()["account"](account)
            logger.debug("get_full_account_details resp: %r", payload)
            return payload

    source = BrokerAccountSource(
        {"logger": logger}, broker_factory=lambda _: VerboseBroker()
    )
    with _client(source, logger=logger, ttl=0) as client:
        response = client.get("/api/v1/accounts/FULL-ACCOUNT-123456")

    assert response.status_code == 200
    assert "get_full_account_details resp" not in stream.getvalue()
    assert "FULL-ACCOUNT-123456" not in stream.getvalue()


def test_startup_constructs_one_broker_with_streaming_false_and_never_closes_it(
    monkeypatch,
):
    from .. import broker_source

    instances = []

    class RecordingBroker:
        def __init__(self, context, *, streaming):
            self.context = context
            self.streaming = streaming
            self.close_calls = 0
            self.all_calls = 0
            instances.append(self)

        def get_accounts(self):
            self.all_calls += 1
            return []

        def get_account(self, *, account):
            return 404

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(broker_source, "TDABrokerInterface", RecordingBroker)
    context = {"logger": logging.getLogger("test-single-broker")}
    app = create_app(context=context)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/api/v1/accounts").status_code == 200

    assert len(instances) == 1
    assert instances[0].context is context
    assert instances[0].streaming is False
    assert instances[0].all_calls == 1
    assert instances[0].close_calls == 0
