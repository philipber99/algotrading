from __future__ import annotations

from ..config import ScreenerConfig
from ..models import ScreeningRequest, SnapshotRecord, Universe
from ..scenarios import ScenarioMetrics
from ..service import ScreenerService
from .test_strategies_execution import NOW, contract


class _Repository:
    def __init__(self):
        self.snapshot = SnapshotRecord(
            snapshot_id="snapshot-test",
            source_collected_at=NOW,
            underlying_symbol="QQQ",
            underlying_bid=101.9,
            underlying_ask=102.1,
            underlying_mark=102.0,
            calculation_definition_id=7,
        )
        self.closed = False
        self.require_surface = None

    def resolve_snapshot(self, symbol, snapshot_id, *, require_surface=False):
        self.require_surface = require_surface
        return self.snapshot

    def load_universe(self, *args, **kwargs):
        return Universe(
            snapshot=self.snapshot,
            contracts=(
                contract(1, 100, 30, bid=4.0, ask=4.1),
                contract(2, 105, 30, bid=1.8, ask=1.9),
            ),
            expiration_inputs={},
            surface_fits={},
            atm_volatility=0.25,
        )

    def close(self):
        self.closed = True


class _Pricing:
    registry = None

    def close(self):
        pass


class _Scenarios:
    def __init__(self):
        self.hedges = []

    def evaluate(self, candidate, universe, request, **kwargs):
        self.hedges.append(kwargs["hedge_shares"])
        return ScenarioMetrics(
            probability_of_profit=0.60,
            target_probability=0.40,
            expected_pnl=25.0,
            pnl_percentiles={
                "p05": -100,
                "p25": -20,
                "p50": 10,
                "p75": 50,
                "p95": 120,
            },
            modeled_pnl_range=(-150.0, 200.0),
            expected_holding_sessions=4.0,
            holding_basis="target_hit_median",
            robustness=1.0,
            warnings=(),
        )


def test_service_returns_ranked_candidates_diagnostics_and_cache_hits():
    repository = _Repository()
    scenarios = _Scenarios()
    service = ScreenerService(
        {},
        repository,
        config=ScreenerConfig(simulation_paths=100),
        pricing=_Pricing(),
        scenario_engine=scenarios,
    )
    request = ScreeningRequest(
        underlying_symbol="QQQ",
        enabled_strategies=["vertical"],
        option_types=["call"],
        surface_relative_value=False,
        delta_hedge_mode="both",
    )
    first = service.screen(request, request_id="request-1")
    assert first.cache_hit is False
    assert first.funnel.contracts_loaded == 2
    assert first.funnel.results_returned == 4
    assert {item.static_hedge_shares is None for item in first.candidates} == {
        True,
        False,
    }
    assert len({item.candidate_id for item in first.candidates}) == 4
    assert all("guarante" not in item.rationale.sentence.lower() for item in first.candidates)
    assert repository.require_surface is False
    assert first.applied_assumptions["scenario_pricing_method"] == "tian"
    assert first.applied_assumptions["static_hedge_rebalanced"] is False
    assert first.performance_diagnostics["scenario_pricing_method"] == "tian"
    assert service.health()["scenario_pricing_method"] == "tian"
    second = service.screen(request, request_id="request-2")
    assert second.cache_hit is True
    assert second.request_id == "request-2"


def test_full_scenario_bound_applies_to_base_candidates_not_hedge_variants():
    class ManyContractsRepository(_Repository):
        def load_universe(self, *args, **kwargs):
            return Universe(
                snapshot=self.snapshot,
                contracts=tuple(
                    contract(
                        index,
                        strike,
                        30,
                        bid=max(0.2, 5.0 - index * 0.3),
                        ask=max(0.3, 5.1 - index * 0.3),
                    )
                    for index, strike in enumerate(
                        (95, 100, 105, 110, 115), start=1
                    )
                ),
                expiration_inputs={},
                surface_fits={},
                atm_volatility=0.25,
            )

    repository = ManyContractsRepository()
    scenarios = _Scenarios()
    service = ScreenerService(
        {},
        repository,
        config=ScreenerConfig(
            max_results_per_strategy=1,
            max_prescore_candidates_per_strategy=10,
            max_generated_candidates_per_strategy=100,
            max_full_scenario_candidates_per_strategy=2,
            simulation_paths=100,
        ),
        pricing=_Pricing(),
        scenario_engine=scenarios,
    )
    request = ScreeningRequest(
        underlying_symbol="QQQ",
        enabled_strategies=["vertical"],
        option_types=["call"],
        surface_relative_value=False,
        delta_hedge_mode="both",
        result_count_per_strategy=1,
    )
    response = service.screen(request, request_id="bounded")
    assert response.funnel.candidates_fully_evaluated <= 2
    assert len(scenarios.hedges) == response.funnel.candidates_fully_evaluated * 2
