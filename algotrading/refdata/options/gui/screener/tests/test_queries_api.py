from __future__ import annotations

from fastapi.testclient import TestClient

from algotrading.refdata.options.Queries import (
    build_effective_chain_query,
    build_expiration_inputs_query,
    build_latest_completed_snapshot_query,
    load_expiration_inputs,
    load_atm_term_structure,
    load_calculation_failures,
    load_forward_variance,
)
from ...app import create_app


def test_snapshot_and_expiration_queries_are_parameterized_bounded_and_authorized():
    latest = build_latest_completed_snapshot_query("algotrading", "qqq")
    assert "{underlying_symbol:String}" in latest.sql
    assert "LIMIT 1" in latest.sql
    assert "is_active" in latest.sql
    assert "complete_with_failures" in latest.sql
    assert "s.source_collected_at AS source_collected_at" in latest.sql
    assert "s.underlying_symbol AS underlying_symbol" in latest.sql
    assert latest.parameters == {"underlying_symbol": "QQQ"}
    surfaced = build_latest_completed_snapshot_query(
        "algotrading", "qqq", require_surface=True
    )
    assert "options_chain_surface_state FINAL" in surfaced.sql
    assert "ss.successful_expirations > 0" in surfaced.sql
    assert "ss.source_calculation_definition_id = cs.calculation_definition_id" in surfaced.sql
    assert "options_chain_surface_fits FINAL" not in surfaced.sql
    snapshot_id = "00000000-0000-0000-0000-000000000001"
    explicit = build_latest_completed_snapshot_query(
        "algotrading", "qqq", snapshot_id, require_surface=True
    )
    assert "options_chain_surface_state FINAL" not in explicit.sql
    assert explicit.parameters["snapshot_id"] == snapshot_id
    expiration = build_expiration_inputs_query("algotrading", snapshot_id)
    assert "options_chain_expiration_inputs AS i" in expiration.sql
    assert "options_chain_calculation_state FINAL" in expiration.sql
    assert "toString(i.snapshot_id) AS snapshot_id" in expiration.sql
    assert "ORDER BY i.expiration_at" in expiration.sql
    assert "options_chain_expiration_inputs FINAL" not in expiration.sql
    chain = build_effective_chain_query("algotrading", snapshot_id)
    assert "o.contract_id AS contract_id" in chain.sql
    assert "c.expiration_at AS expiration_at" in chain.sql


def test_screener_queries_can_use_named_rows_without_arrow_uuid_conversion():
    class Client:
        def __init__(self):
            self.sql = []
            self.parameters = []

        def query(self, sql, *, parameters):
            self.sql.append(sql)
            self.parameters.append(parameters)
            return "named-row-result"

        def query_arrow(self, sql, *, parameters):
            raise AssertionError("Arrow transport must not be selected")

    client = Client()
    snapshot_id = "00000000-0000-0000-0000-000000000001"
    result = load_expiration_inputs(
        client,
        "algotrading",
        snapshot_id,
        arrow=False,
    )
    assert result == "named-row-result"
    load_forward_variance(client, "algotrading", snapshot_id, arrow=False)
    assert "square(" not in client.sql[-1]
    assert "n.fitted_iv_mid * n.fitted_iv_mid" in client.sql[-1]
    load_atm_term_structure(
        client, "algotrading", snapshot_id, price_basis="mid", arrow=False
    )
    assert "price_basis = {price_basis:String}" in client.sql[-1]
    assert client.parameters[-1]["price_basis"] == "mid"
    load_calculation_failures(
        client, "algotrading", snapshot_id=snapshot_id, arrow=False
    )
    assert "options_chain_calculation_failures" in client.sql[-1]


class _CacheEngine:
    ready = True
    generation_id = "generation-test"

    def startup(self):
        pass

    def shutdown(self):
        pass

    def ensure_current(self):
        pass

    def health(self):
        return {"ready": True}

    def metadata(self):
        return {"generation_id": self.generation_id}

    def windows(self):
        return {"windows": []}

    def symbols(self):
        return {"symbols": []}

    def coverage(self, symbol, window):
        return {"symbol": symbol, "window": window}


class _Screener:
    ready = True

    def health(self):
        return {"status": "ok", "model_version": "test"}

    def capabilities(self):
        from ..models import CapabilitiesResponse

        return CapabilitiesResponse(
            strategies=[
                "vertical",
                "calendar",
                "diagonal",
                "butterfly",
                "iron_condor",
                "double_calendar",
            ],
            static_delta_hedging=True,
            dynamic_delta_hedging=False,
            price_bases=["mid", "bid", "ask"],
            directional_assumption_modes=["zero", "risk_neutral", "explicit"],
            minimum_dte=0,
            maximum_dte=365,
            maximum_results_per_strategy=50,
            maximum_simulation_paths=25000,
            maximum_holding_sessions=30,
            model_version="test",
        )

    def shutdown(self):
        pass


def test_screener_health_capabilities_and_existing_cache_route():
    app = create_app(context={}, engine=_CacheEngine(), screener_service=_Screener())
    with TestClient(app) as client:
        assert client.get("/v1/screener/health").status_code == 200
        capabilities = client.get("/v1/screener/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["dynamic_delta_hedging"] is False
        assert capabilities.json()["strategies"] == [
            "vertical",
            "calendar",
            "diagonal",
            "butterfly",
            "iron_condor",
            "double_calendar",
        ]
        cache = client.get("/v1/cache/metadata")
        assert cache.status_code == 200
        assert "etag" in cache.headers


def test_clickhouse_initialization_failure_does_not_disable_cache_routes():
    def unavailable(*args, **kwargs):
        raise ConnectionError("synthetic unavailable")

    app = create_app(
        context={},
        engine=_CacheEngine(),
        clickhouse_client_factory=unavailable,
    )
    with TestClient(app) as client:
        assert client.get("/v1/cache/metadata").status_code == 200
        health = client.get("/v1/screener/health")
        assert health.status_code == 503
        response = client.post(
            "/v1/screener/spreads", json={"underlying_symbol": "QQQ"}
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "SCREENER_NOT_READY"
