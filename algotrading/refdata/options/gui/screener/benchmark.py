"""Non-flaky structural benchmark for request-scoped screener scenario work."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from .config import ScreenerConfig
from .models import (
    CandidateDraft,
    CandidateLeg,
    ContractRecord,
    ExpirationInput,
    ScreeningRequest,
    SnapshotRecord,
    Universe,
)
from .scenarios import PricingResources, ScenarioEngine, geometric_paths


class _FakePricer:
    def __init__(self) -> None:
        self.jobs = 0
        self.batches = 0

    @staticmethod
    def pricing_method(exercise_type: str, tenor_seconds: float) -> str:
        return "tian"

    def price_jobs(
        self,
        jobs: Sequence[Mapping[str, Any]],
        *,
        deadline: float,
    ) -> np.ndarray:
        self.jobs += len(jobs)
        self.batches += 1
        spot = np.fromiter((float(job["spot"]) for job in jobs), dtype=float)
        strike = np.fromiter((float(job["strike"]) for job in jobs), dtype=float)
        calls = np.fromiter(
            (job["put_call"] == "CALL" for job in jobs), dtype=bool
        )
        return np.where(calls, np.maximum(spot - strike, 0), np.maximum(strike - spot, 0))


def _synthetic(
    *,
    candidate_count: int,
    contract_count: int,
    paths: int,
    holding: int,
) -> tuple[Universe, ScreeningRequest, list[tuple[CandidateDraft, int]]]:
    now = datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc)
    expiration_at = now + timedelta(days=60)
    contracts = []
    for index in range(contract_count):
        strike = 80.0 + 40.0 * index / max(1, contract_count - 1)
        contracts.append(
            ContractRecord(
                contract_id=index + 1,
                option_symbol=f"SYN{index + 1}",
                put_call="CALL",
                strike=strike,
                expiration_at=expiration_at,
                dte=60.0,
                exercise_type="AMERICAN",
                multiplier=100.0,
                is_mini=False,
                is_nonstandard=False,
                symbol_map_mismatch=False,
                bid=max(0.05, 8.0 - abs(strike - 100.0) * 0.2),
                ask=max(0.10, 8.1 - abs(strike - 100.0) * 0.2),
                bid_size=100,
                ask_size=100,
                volume=1000,
                open_interest=5000,
                quote_time=now,
                effective_iv=0.25,
                delta=0.50,
                gamma=0.02,
                theta=-0.02,
                vega=0.10,
                rho=0.05,
                iv_value_source="source",
                greeks_value_source="source",
                result_origin="source_only",
                quality_flags=0,
            )
        )
    snapshot = SnapshotRecord(
        snapshot_id="00000000-0000-0000-0000-000000000001",
        source_collected_at=now,
        underlying_symbol="SYN",
        underlying_bid=99.99,
        underlying_ask=100.01,
        underlying_mark=100.0,
        calculation_definition_id=1,
    )
    universe = Universe(
        snapshot=snapshot,
        contracts=tuple(contracts),
        expiration_inputs={
            expiration_at: ExpirationInput(
                expiration_at=expiration_at,
                time_to_expiry_seconds=60 * 86400,
                risk_free_rate=0.04,
                dividend_yield=0.01,
                forward_price=100.5,
                calculation_definition_id=1,
            )
        },
        surface_fits={},
        atm_volatility=0.25,
    )
    request = ScreeningRequest(
        underlying_symbol="SYN",
        enabled_strategies=["vertical", "calendar", "diagonal"],
        surface_relative_value=False,
        delta_hedge_mode="both",
        probability={"simulation_paths": paths, "drift_mode": "zero"},
    )
    names = ("vertical", "calendar", "diagonal")
    plans: list[tuple[CandidateDraft, int]] = []
    for index in range(candidate_count):
        first = contracts[index % contract_count]
        second = contracts[(index * 7 + 1) % contract_count]
        if second.contract_id == first.contract_id:
            second = contracts[(index + 1) % contract_count]
        plans.append(
            (
                CandidateDraft(
                    strategy=names[index % len(names)],
                    variant=f"synthetic_{index}",
                    outlook="neutral",
                    legs=(CandidateLeg("long", first), CandidateLeg("short", second)),
                ),
                holding,
            )
        )
    return universe, request, plans


def run_orchestration(args: argparse.Namespace) -> dict[str, Any]:
    universe, request, plans = _synthetic(
        candidate_count=args.candidates,
        contract_count=args.contracts,
        paths=args.paths,
        holding=args.holding,
    )
    legacy_pricer = _FakePricer()
    legacy_path_ms = 0.0
    legacy_grid_ms = 0.0
    legacy_started = time.perf_counter()
    for candidate, planned in plans:
        for _hedge_variant in range(2):
            path_started = time.perf_counter()
            paths = geometric_paths(
                spot=universe.snapshot.underlying_mark,
                sessions=planned,
                path_count=args.paths,
                annualized_drift=0.0,
                annualized_volatility=universe.atm_volatility,
                seed=12345,
            )
            legacy_path_ms += (time.perf_counter() - path_started) * 1000
            grid_started = time.perf_counter()
            combined = np.zeros_like(paths)
            for leg in candidate.legs:
                for horizon_index in range(planned):
                    observed = paths[:, horizon_index]
                    grid = np.linspace(
                        max(1.0e-6, float(np.min(observed)) * 0.995),
                        float(np.max(observed)) * 1.005,
                        args.grid_points,
                    )
                    jobs = [
                        {
                            "spot": float(spot),
                            "strike": leg.contract.strike,
                            "put_call": leg.contract.put_call,
                        }
                        for spot in grid
                    ]
                    prices = legacy_pricer.price_jobs(
                        jobs, deadline=float("inf")
                    )
                    combined[:, horizon_index] += leg.sign * np.interp(
                        observed, grid, prices
                    )
            # Force the same candidate-sized numerical reduction as evaluation.
            float(np.mean(combined[:, -1]))
            legacy_grid_ms += (time.perf_counter() - grid_started) * 1000
    legacy_total_ms = (time.perf_counter() - legacy_started) * 1000

    config = ScreenerConfig(
        simulation_paths=args.paths,
        maximum_simulation_paths=max(25000, args.paths),
        spot_grid_points=args.grid_points,
        maximum_spot_grid_points=max(161, args.grid_points),
        maximum_holding_sessions=args.holding,
        pricing_batch_size=args.batch_size,
    )
    pricer = _FakePricer()
    engine = ScenarioEngine({}, config, pricer)
    prepare_started = time.perf_counter()
    workspace = engine.prepare(
        plans,
        universe,
        request,
        deadline=time.perf_counter() + 120.0,
    )
    prepare_ms = (time.perf_counter() - prepare_started) * 1000
    evaluate_started = time.perf_counter()
    for candidate, planned in plans:
        workspace.evaluate_variants(
            candidate,
            planned_holding_sessions=planned,
            target_profit_per_share=0.25,
            hedge_shares_values=[0, -10],
        )
    evaluate_ms = (time.perf_counter() - evaluate_started) * 1000
    legacy_jobs = legacy_pricer.jobs
    optimized_jobs = workspace.performance.numerical_pricing_jobs
    workspace_bytes = sum(value.nbytes for value in workspace.paths_by_drift_key.values())
    workspace_bytes += sum(value.nbytes for value in workspace.spot_grids.values())
    workspace_bytes += sum(value.nbytes for value in workspace.price_grids.values())
    return {
        "candidate_count": args.candidates,
        "unique_contract_count": workspace.performance.unique_contracts_priced,
        "path_count": args.paths,
        "holding_sessions": args.holding,
        "grid_points": args.grid_points,
        "before": {
            "shock_path_generations": args.candidates * 2,
            "solver_jobs": legacy_jobs,
            "phase_ms": {
                "path_generation": round(legacy_path_ms, 3),
                "grid_price_interpolate": round(legacy_grid_ms, 3),
                "total": round(legacy_total_ms, 3),
            },
        },
        "after": {
            "shock_matrix_generations": workspace.performance.shock_matrix_generations,
            "path_sets_generated": workspace.performance.path_sets_generated,
            "unique_price_grids": workspace.performance.unique_price_grids,
            "solver_jobs": optimized_jobs,
            "pricing_batches": workspace.performance.pricing_batches_submitted,
            "surface_evaluators": workspace.performance.surface_evaluators_compiled,
            "hedge_both_solver_jobs": optimized_jobs,
            "none_solver_jobs": optimized_jobs,
        },
        "phase_ms": {
            "scenario_prepare": round(prepare_ms, 3),
            "scenario_evaluate": round(evaluate_ms, 3),
            "total": round(prepare_ms + evaluate_ms, 3),
        },
        "peak_request_workspace_bytes_approx": workspace_bytes,
    }


def run_real_tian(job_count: int, batch_size: int) -> dict[str, Any]:
    context = {
        "options_chain.gui.screener.pricing_method": "tian",
        "options_chain.gui.screener.pricing_workers": 1,
        "options_chain.gui.screener.pricing_batch_size": max(128, batch_size),
        "options_chain.gui.screener.max_pending_pricing_batches": 2,
        "options_calcs.greeks.tian.steps": 256,
        "options_calcs.greeks.tian.chunk_size": 50000,
        "options_calcs.greeks.tian.device": "cpu",
        "options_calcs.greeks.tian.vectorized_enabled": True,
        "options_calcs.greeks.tian.scalar_fallback_enabled": True,
    }
    resources = PricingResources(context)
    now = datetime.now(timezone.utc)
    jobs = [
        {
            "pricing_method": "tian",
            "spot": 80.0 + 40.0 * index / max(1, job_count - 1),
            "strike": 100.0,
            "tenor_seconds": 30 * 86400.0,
            "rate": 0.04,
            "dividend_yield": 0.01,
            "volatility": 0.25,
            "put_call": "CALL",
            "snapshot_id": "benchmark",
            "source_collected_at": now,
            "expiration_at": now + timedelta(days=30),
        }
        for index in range(job_count)
    ]
    started = time.perf_counter()
    try:
        values = resources.price_jobs(
            jobs, deadline=time.perf_counter() + 60.0
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "jobs": job_count,
            "batches": resources.pool.batches_submitted if resources.pool else 0,
            "workers": resources.config.resolved_pricing_workers,
            "finite_prices": int(np.count_nonzero(np.isfinite(values))),
            "elapsed_ms": round(elapsed_ms, 3),
        }
    finally:
        resources.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=75)
    parser.add_argument("--contracts", type=int, default=30)
    parser.add_argument("--paths", type=int, default=5000)
    parser.add_argument("--holding", type=int, default=30)
    parser.add_argument("--grid-points", type=int, default=81)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--real-tian-jobs", type=int, default=0)
    args = parser.parse_args()
    result = {"orchestration": run_orchestration(args)}
    if args.real_tian_jobs:
        result["real_tian"] = run_real_tian(
            args.real_tian_jobs, args.batch_size
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
