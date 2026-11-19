from __future__ import annotations

from concurrent.futures import Future
from datetime import timedelta

import numpy as np
import pytest

from ...cache.errors import ConfigurationError
from ...pricing_pool import PricingPool
from algotrading.refdata.options import Queries
from ..config import ScreenerConfig
from ..models import (
    CandidateDraft,
    CandidateLeg,
    ExpirationInput,
    ScreeningRequest,
    SnapshotRecord,
    Universe,
)
from ..repository import ScreenerRepository
from ..scenarios import PricingResources, ScenarioEngine
from .test_strategies_execution import NOW, contract


class _SynchronousPricing:
    def __init__(self):
        self.jobs = 0
        self.calls = 0

    def pricing_method(self, exercise_type, tenor_seconds):
        return "tian"

    def price_jobs(self, jobs, *, deadline):
        self.calls += 1
        self.jobs += len(jobs)
        return np.asarray(
            [
                max(0.0, job["spot"] - job["strike"])
                if job["put_call"] == "CALL"
                else max(0.0, job["strike"] - job["spot"])
                for job in jobs
            ],
            dtype=np.float64,
        )


class _Executor:
    def __init__(self):
        self.batches = []

    def submit(self, function, jobs):
        self.batches.append(jobs)
        future = Future()
        future.set_result(np.asarray([job["ordinal"] for job in jobs]))
        return future

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_args = (wait, cancel_futures)


def test_screener_pricing_configuration_defaults_and_forced_routes():
    config = ScreenerConfig.from_context({})
    assert config.pricing_method == "tian"
    assert 1 <= config.resolved_pricing_workers <= 8
    assert config.resolved_max_pending_pricing_batches == (
        config.resolved_pricing_workers * 2
    )
    signatures = set()
    for method in ("tian", "crr", "crank_nicolson"):
        resources = PricingResources(
            {"options_chain.gui.screener.pricing_method": method},
            registry=object(),
        )
        assert resources.pricing_method("AMERICAN", 60) == method
        assert resources.pricing_method("EUROPEAN", 60) == method
        signatures.add(resources.configuration_signature)
    assert len(signatures) == 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("pricing_method", ""),
        ("pricing_method", "none"),
        ("pricing_workers", 65),
        ("pricing_batch_size", 127),
        ("max_pending_pricing_batches", 513),
        ("pricing_worker_threads", 9),
        ("max_full_scenario_candidates_per_strategy", 49),
    ],
)
def test_invalid_screener_pricing_bounds_fail_fast(key, value):
    with pytest.raises(ConfigurationError):
        ScreenerConfig.from_context(
            {f"options_chain.gui.screener.{key}": value}
        )


def test_request_workspace_reuses_shocks_paths_grids_and_option_pnl():
    first = contract(1, 100, 30, bid=4.0, ask=4.2)
    second = contract(2, 105, 30, bid=1.5, ask=1.7)
    third = contract(3, 110, 30, bid=0.5, ask=0.7)
    candidate_one = CandidateDraft(
        strategy="vertical",
        variant="one",
        outlook="bullish",
        legs=(CandidateLeg("long", first), CandidateLeg("short", second)),
    )
    candidate_two = CandidateDraft(
        strategy="vertical",
        variant="two",
        outlook="bullish",
        legs=(CandidateLeg("long", first), CandidateLeg("short", third)),
    )
    snapshot = SnapshotRecord(
        snapshot_id="workspace",
        source_collected_at=NOW,
        underlying_symbol="QQQ",
        underlying_bid=101.9,
        underlying_ask=102.1,
        underlying_mark=102.0,
        calculation_definition_id=7,
    )
    expiration = ExpirationInput(
        expiration_at=first.expiration_at,
        time_to_expiry_seconds=30 * 86400,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        forward_price=102.25,
        calculation_definition_id=7,
    )
    universe = Universe(
        snapshot=snapshot,
        contracts=(first, second, third),
        expiration_inputs={first.expiration_at: expiration},
        surface_fits={},
        atm_volatility=0.20,
    )
    request = ScreeningRequest(
        underlying_symbol="QQQ",
        enabled_strategies=["vertical"],
        surface_relative_value=False,
        delta_hedge_mode="both",
        probability={"simulation_paths": 100, "drift_mode": "zero"},
    )
    config = ScreenerConfig(
        simulation_paths=100,
        maximum_holding_sessions=2,
        spot_grid_points=11,
        pricing_batch_size=128,
    )
    pricing = _SynchronousPricing()
    workspace = ScenarioEngine({}, config, pricing).prepare(
        [(candidate_one, 2), (candidate_two, 2)],
        universe,
        request,
        deadline=float("inf"),
    )
    expected_jobs = 3 * 2 * 11
    assert pricing.jobs == expected_jobs
    assert workspace.performance.shock_matrix_generations == 1
    assert workspace.performance.path_sets_generated == 1
    assert workspace.performance.unique_price_grids == 6

    before = pricing.jobs
    metrics = workspace.evaluate_variants(
        candidate_one,
        planned_holding_sessions=2,
        target_profit_per_share=0.2,
        hedge_shares_values=[0, -20],
    )
    assert len(metrics) == 2
    assert pricing.jobs == before


def test_pricing_pool_preserves_ordinals_and_outer_batch_bound():
    executor = _Executor()
    pool = PricingPool(
        {},
        workers=2,
        batch_size=2,
        max_pending_batches=2,
        executor=executor,
    )
    values = pool.price_jobs(
        [{"ordinal": index} for index in range(5)],
        deadline=float("inf"),
    )
    np.testing.assert_array_equal(values, np.arange(5))
    assert [len(batch) for batch in executor.batches] == [2, 2, 1]


def test_owned_pricing_pool_initializes_and_shuts_down_once():
    executor = _Executor()
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return executor

    pool = PricingPool(
        {},
        workers=2,
        batch_size=2,
        max_pending_batches=2,
        executor_factory=factory,
    )
    pool.start()
    pool.start()
    pool.shutdown()
    pool.shutdown()
    assert len(calls) == 1
    assert calls[0]["max_workers"] == 2
    assert executor.shutdown_args == (True, True)


def test_failure_mapping_preserves_contract_expiration_snapshot_precedence():
    expiration = NOW + timedelta(days=30)
    contracts = (
        contract(1, 100, 30),
        contract(2, 105, 30),
    )
    mapped = ScreenerRepository._apply_failures(
        contracts,
        [
            {"failure_scope": "snapshot", "reason_code": "snapshot"},
            {
                "failure_scope": "expiration",
                "expiration_at": expiration,
                "reason_code": "expiration",
            },
            {
                "failure_scope": "contract",
                "contract_id": 1,
                "reason_code": "contract",
            },
        ],
    )
    assert mapped[0].active_failure_reason == "contract"
    assert mapped[1].active_failure_reason == "expiration"


def test_repository_uses_narrow_bounded_query_set(monkeypatch):
    expiration_at = NOW + timedelta(days=30)
    snapshot = SnapshotRecord(
        snapshot_id="snapshot",
        source_collected_at=NOW,
        underlying_symbol="QQQ",
        underlying_bid=99.9,
        underlying_ask=100.1,
        underlying_mark=100.0,
        calculation_definition_id=7,
    )
    calls = []

    def expiration_inputs(*args, **kwargs):
        calls.append("expiration_inputs")
        return [
            {
                "expiration_at": expiration_at,
                "time_to_expiry_seconds": 30 * 86400,
                "risk_free_rate": 0.04,
                "dividend_yield": 0.01,
                "forward_price": 100.25,
                "calculation_definition_id": 7,
            }
        ]

    def effective_chain(*args, **kwargs):
        calls.append("effective_chain")
        assert "include_failures" not in kwargs
        return [
            {
                "contract_id": 1,
                "option_symbol": "QQQ-C",
                "put_call": "CALL",
                "strike_price": 100.0,
                "expiration_at": expiration_at,
                "exercise_type": "AMERICAN",
                "multiplier": 100.0,
            }
        ]

    def failures(*args, **kwargs):
        calls.append("failures")
        assert kwargs["arrow"] is False
        return []

    monkeypatch.setattr(Queries, "load_expiration_inputs", expiration_inputs)
    monkeypatch.setattr(Queries, "load_effective_chain", effective_chain)
    monkeypatch.setattr(Queries, "load_calculation_failures", failures)
    monkeypatch.setattr(
        Queries,
        "load_surface_nodes",
        lambda *args, **kwargs: pytest.fail("full surface nodes must not load"),
    )
    monkeypatch.setattr(
        Queries,
        "load_forward_variance",
        lambda *args, **kwargs: pytest.fail("forward variance must not load"),
    )
    repository = ScreenerRepository(object(), "algotrading")
    universe = repository.load_universe(
        "QQQ",
        snapshot.snapshot_id,
        price_basis="mid",
        require_surface=False,
        resolved_snapshot=snapshot,
    )
    assert len(universe.contracts) == 1
    assert calls == ["expiration_inputs", "effective_chain", "failures"]
    assert repository.last_query_count == 3


def test_surface_repository_skips_full_nodes_and_forward_variance(monkeypatch):
    expiration_at = NOW + timedelta(days=30)
    snapshot = SnapshotRecord(
        snapshot_id="snapshot",
        source_collected_at=NOW,
        underlying_symbol="QQQ",
        underlying_bid=99.9,
        underlying_ask=100.1,
        underlying_mark=100.0,
        calculation_definition_id=7,
    )
    calls = []
    monkeypatch.setattr(
        Queries,
        "load_expiration_inputs",
        lambda *args, **kwargs: calls.append("expiration_inputs")
        or [
            {
                "expiration_at": expiration_at,
                "time_to_expiry_seconds": 30 * 86400,
                "risk_free_rate": 0.04,
                "dividend_yield": 0.01,
                "forward_price": 100.25,
                "calculation_definition_id": 7,
            }
        ],
    )
    monkeypatch.setattr(
        Queries,
        "load_effective_chain",
        lambda *args, **kwargs: calls.append("effective_chain") or [],
    )
    monkeypatch.setattr(
        Queries,
        "load_calculation_failures",
        lambda *args, **kwargs: calls.append("failures") or [],
    )
    monkeypatch.setattr(
        Queries,
        "load_surface_state",
        lambda *args, **kwargs: calls.append("surface_state")
        or [{"status": "complete"}],
    )
    monkeypatch.setattr(
        Queries,
        "load_surface_fits",
        lambda *args, **kwargs: calls.append("surface_fits")
        or [
            {
                "expiration_at": expiration_at,
                "price_basis": "mid",
                "pricing_method": "tian",
                "fit_method": "SVI",
                "svi_a": 0.04,
                "svi_b": 0.1,
                "svi_rho": -0.2,
                "svi_m": 0.0,
                "svi_sigma": 0.1,
                "pchip_k": [],
                "pchip_total_variance": [],
                "minimum_log_forward_moneyness": -0.2,
                "maximum_log_forward_moneyness": 0.2,
                "surface_definition_id": 3,
                "source_calculation_definition_id": 7,
            }
        ],
    )

    def atm(*args, **kwargs):
        calls.append("atm")
        assert kwargs["price_basis"] == "mid"
        return [{"price_basis": "mid", "fitted_iv_mid": 0.25}]

    monkeypatch.setattr(Queries, "load_atm_term_structure", atm)
    monkeypatch.setattr(
        Queries,
        "load_surface_nodes",
        lambda *args, **kwargs: pytest.fail("full surface nodes must not load"),
    )
    monkeypatch.setattr(
        Queries,
        "load_forward_variance",
        lambda *args, **kwargs: pytest.fail("forward variance must not load"),
    )
    repository = ScreenerRepository(object(), "algotrading")
    universe = repository.load_universe(
        "QQQ",
        snapshot.snapshot_id,
        price_basis="mid",
        require_surface=True,
        resolved_snapshot=snapshot,
    )
    assert universe.atm_volatility == 0.25
    assert universe.forward_variance == ()
    assert calls == [
        "expiration_inputs",
        "effective_chain",
        "failures",
        "surface_state",
        "surface_fits",
        "atm",
    ]
    assert repository.last_query_count == 6
