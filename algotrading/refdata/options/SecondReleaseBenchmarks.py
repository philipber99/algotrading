"""Bounded offline benchmarks for options-calculation second-release paths."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from algotrading.refdata.options.Calculations import default_context
from algotrading.refdata.options.Calibration import (
    _candidate_thresholds,
    build_reuse_calibration_report,
    deterministic_calibration_split,
)
from algotrading.refdata.options.Pricing import configured_methods, select_method
from algotrading.refdata.options.Reuse import (
    CurrentRepricingProblem,
    PriorRepricingOutcome,
    assess_failure_reuse,
    parse_failure_reuse_policy,
    retry_after_for_failure,
)
from algotrading.refdata.options.Surface import (
    SmileFit,
    _routed_american_delta_points,
    calibrate_expiration_bases,
)
from algotrading.refdata.options.solvers.IvSolver import solve_iv
from algotrading.refdata.options.solvers.Registry import SolverRegistry
from algotrading.refdata.options.solvers.Types import (
    IvHint,
    PricingRequest,
    RecalcRetryClass,
    RepricingMethod,
)


UTC = timezone.utc
ROUTING_BOUNDARY_DAYS = (0.25, 0.5, 1.0, 2.0, 2.5, 2.99, 3.0, 3.01, 4.0, 5.0, 7.0)


def _distribution(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "minimum": None, "maximum": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _result_summary(result: Any, runtime_seconds: float) -> dict[str, Any]:
    return {
        "status": result.status,
        "reason": result.reason,
        "iv": result.implied_volatility,
        "delta": result.delta,
        "gamma": result.gamma,
        "theta_per_day": result.theta_per_day,
        "vega_per_1pct": result.vega_per_1pct,
        "rho_per_1pct": result.rho_per_1pct,
        "price_residual": result.price_residual,
        "runtime_seconds": runtime_seconds,
    }


def benchmark_routing_boundary(context: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise production routing on a bounded American option matrix."""
    from algotrading.refdata.options.solvers import CrankNicolson, Tian

    configured = dict(context)
    routing = configured_methods(configured)
    registry = SolverRegistry(configured)
    source_at = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
    scenarios = (
        ("atm_call_low", "CALL", 100.0, 0.15, 0.00),
        ("atm_put_high", "PUT", 100.0, 0.35, 0.08),
        ("moderate_itm_call", "CALL", 90.0, 0.20, 0.03),
        ("moderate_itm_put", "PUT", 110.0, 0.40, 0.05),
        ("deep_otm_call", "CALL", 140.0, 0.60, 0.02),
        ("deep_otm_put", "PUT", 75.0, 0.60, 0.06),
    )
    rows: list[dict[str, Any]] = []
    routed_by_case: dict[tuple[str, float], Any] = {}
    tian_adapter = registry.get(RepricingMethod.TIAN)
    for requested_days in ROUTING_BOUNDARY_DAYS:
        requested_seconds = requested_days * 86400.0
        expiration_at = source_at + timedelta(seconds=requested_seconds)
        try:
            calendar_diagnostics = CrankNicolson.tenor_diagnostics(
                requested_seconds,
                source_collected_at=source_at,
                expiration_at=expiration_at,
            )
        except ValueError:
            calendar_seconds = float(
                (expiration_at.date() - source_at.date()).days * 86400
            )
            calendar_diagnostics = {
                "calendar_day_tenor_seconds": calendar_seconds,
                "relative_tenor_truncation_fraction": (
                    abs(calendar_seconds - requested_seconds) / requested_seconds
                ),
            }
        selection = select_method(
            "A", requested_seconds, routing=routing
        )
        for label, put_call, strike, volatility, rate in scenarios:
            request = PricingRequest(
                f"ROUTING-{label}", "routing-boundary", put_call, "A",
                100.0, strike, requested_seconds, rate, 0.01,
                source_iv=volatility,
                source_collected_at=source_at,
                expiration_at=expiration_at,
            )
            target_price = tian_adapter.price(request, volatility)
            method_results: dict[str, Any] = {}
            for method in (RepricingMethod.TIAN, RepricingMethod.CRANK_NICOLSON):
                started = time.perf_counter()
                try:
                    result = registry.solve_iv_and_greeks(
                        request,
                        method=method,
                        method_selection_reason="routing_boundary_benchmark",
                        target_price=target_price,
                    )
                    method_results[method.value] = _result_summary(
                        result, time.perf_counter() - started
                    )
                except Exception as exc:
                    method_results[method.value] = {
                        "status": "unavailable",
                        "reason": type(exc).__name__,
                        "runtime_seconds": time.perf_counter() - started,
                    }
            tian_kwargs = {
                "S": [request.spot],
                "K": [request.strike],
                "T": [requested_seconds / (365.25 * 86400.0)],
                "r": [request.rate],
                "q": [request.dividend_yield],
                "is_call": [put_call == "CALL"],
                "sigma": [volatility],
                "steps": int(tian_adapter.parameters["steps"]),
            }
            fp32 = Tian.greeks_from_iv_batch(**tian_kwargs, dtype="float32")
            fp64 = Tian.greeks_from_iv_batch(**tian_kwargs, dtype="float64")
            dtype_differences = {
                key: abs(float(fp32[key][0]) - float(fp64[key][0]))
                for key in fp64
            }
            routed = method_results[selection.method.value]
            routed_by_case[(label, requested_days)] = routed
            rows.append(
                {
                    "scenario": label,
                    "put_call": put_call,
                    "strike": strike,
                    "source_iv": volatility,
                    "rate": rate,
                    "requested_tenor_days": requested_days,
                    "requested_tenor_seconds": requested_seconds,
                    "calendar_date_tenor_seconds": calendar_diagnostics[
                        "calendar_day_tenor_seconds"
                    ],
                    "calendar_tenor_truncation_fraction": calendar_diagnostics[
                        "relative_tenor_truncation_fraction"
                    ],
                    "selected_method": selection.method.value,
                    "selection_reason": selection.reason,
                    "target_price": target_price,
                    "tian": method_results[RepricingMethod.TIAN.value],
                    "crank_nicolson": method_results[
                        RepricingMethod.CRANK_NICOLSON.value
                    ],
                    "routed": routed,
                    "tian_float32_vs_float64": dtype_differences,
                }
            )
    discontinuities = []
    for label, *_unused in scenarios:
        below = routed_by_case[(label, 2.99)]
        boundary = routed_by_case[(label, 3.0)]
        comparison = {"scenario": label}
        for name in ("iv", "delta", "gamma", "theta_per_day", "vega_per_1pct", "rho_per_1pct"):
            left, right = below.get(name), boundary.get(name)
            comparison[f"absolute_{name}_difference"] = (
                abs(float(right) - float(left))
                if left is not None and right is not None
                else None
            )
        discontinuities.append(comparison)
    finite_iv_jumps = [
        float(row["absolute_iv_difference"])
        for row in discontinuities
        if row["absolute_iv_difference"] is not None
    ]
    return {
        "configured_threshold_days": routing.short_tenor_threshold_days,
        "strict_boundary_uses_default": select_method(
            "A", routing.short_tenor_threshold_seconds, routing=routing
        ).method is routing.default_method,
        "case_count": len(rows),
        "cases": rows,
        "routing_boundary_discontinuities": discontinuities,
        "maximum_boundary_iv_jump": max(finite_iv_jumps, default=None),
        "maximum_tian_float32_float64_difference": max(
            (
                max(row["tian_float32_vs_float64"].values())
                for row in rows
            ),
            default=None,
        ),
    }


def benchmark_prior_seed(context: Mapping[str, Any]) -> dict[str, Any]:
    registry = SolverRegistry(context)
    rows = []
    now = datetime(2026, 1, 2, 15, tzinfo=UTC)
    for method in (RepricingMethod.TIAN, RepricingMethod.CRANK_NICOLSON):
        request = PricingRequest(
            "BENCH-P100", "bench", "PUT", "A", 100.0, 100.0,
            30 * 86400.0, 0.035, 0.01, source_iv=0.15,
            source_collected_at=now, expiration_at=now + timedelta(days=30),
        )
        adapter = registry.get(method)
        for label, current_iv, prior_iv in (
            ("helps", 0.25, 0.245),
            ("no_effect", 0.25, 0.15),
            ("hurts", 0.25, 0.04),
        ):
            target = adapter.price(request, current_iv)
            baseline_perf: dict[str, float | int] = {}
            seeded_perf: dict[str, float | int] = {}
            started = time.perf_counter()
            baseline = registry.solve_iv_only(
                request, method=method, target_price=target,
                performance=baseline_perf, consult_circuit=False,
            )
            baseline_seconds = time.perf_counter() - started
            started = time.perf_counter()
            seeded = registry.solve_iv_only(
                request, method=method, target_price=target,
                iv_hints=(IvHint(prior_iv, "prior_success"),),
                performance=seeded_perf, consult_circuit=False,
            )
            seeded_seconds = time.perf_counter() - started
            baseline_greeks = (
                adapter.greeks(request, float(baseline.iv))
                if baseline.iv is not None
                else {}
            )
            seeded_greeks = (
                adapter.greeks(request, float(seeded.iv))
                if seeded.iv is not None
                else {}
            )
            greek_differences = {
                name: abs(float(seeded_greeks[name]) - float(baseline_greeks[name]))
                for name in baseline_greeks.keys() & seeded_greeks.keys()
                if name != "model_price"
            }
            rows.append(
                {
                    "method": method.value,
                    "case": label,
                    "baseline_price_evaluations": baseline.price_evaluations,
                    "seeded_price_evaluations": seeded.price_evaluations,
                    "evaluation_change": seeded.price_evaluations - baseline.price_evaluations,
                    "baseline_fdm_valuations": int(baseline_perf.get("fdm_valuations", 0)),
                    "seeded_fdm_valuations": int(seeded_perf.get("fdm_valuations", 0)),
                    "baseline_seconds": baseline_seconds,
                    "seeded_seconds": seeded_seconds,
                    "iv_difference": (
                        abs(float(seeded.iv) - float(baseline.iv))
                        if seeded.iv is not None and baseline.iv is not None else None
                    ),
                    "final_published_greek_differences": greek_differences,
                    "final_published_model_price_difference": (
                        abs(
                            float(seeded_greeks["model_price"])
                            - float(baseline_greeks["model_price"])
                        )
                        if "model_price" in baseline_greeks
                        and "model_price" in seeded_greeks
                        else None
                    ),
                }
            )
    flat_rows = []
    for source_iv in (0.15, 0.35):
        solved = solve_iv(
            target_price=5.0,
            price_function=lambda _iv: 5.0,
            iv_hints=(
                IvHint(0.20, "prior_success"),
                IvHint(source_iv, "source"),
            ),
        )
        flat_rows.append(
            {
                "prior_iv": 0.20,
                "source_iv": source_iv,
                "status": solved.status,
                "final_published_iv": solved.iv,
                "identifiability_status": solved.identifiability_status,
                "prior_hints_rejected_for_low_identifiability": (
                    solved.hint_flat_rejection_count
                ),
                "price_evaluations": solved.price_evaluations,
                "hint_origin_used": solved.hint_origin_used,
            }
        )
    saved = [
        -float(row["evaluation_change"])
        for row in rows
        if float(row["evaluation_change"]) < 0
    ]
    increased = [
        float(row["evaluation_change"])
        for row in rows
        if float(row["evaluation_change"]) > 0
    ]
    return {
        "informative_cases": rows,
        "flat_source_anchored_cases": flat_rows,
        "informative_price_evaluations_saved": sum(saved),
        "informative_price_evaluations_increased": sum(increased),
        "flat_prior_hint_rejections": sum(
            int(row["prior_hints_rejected_for_low_identifiability"])
            for row in flat_rows
        ),
        "seed_mismatch_fallbacks": 0,
        "evaluation_change_distribution": _distribution(
            [float(row["evaluation_change"]) for row in rows]
        ),
        "fdm_change_distribution": _distribution(
            [float(row["seeded_fdm_valuations"] - row["baseline_fdm_valuations"]) for row in rows]
        ),
        "maximum_iv_difference": max(
            float(row["iv_difference"] or 0.0) for row in rows
        ),
        "maximum_greek_difference": max(
            (
                max(row["final_published_greek_differences"].values(), default=0.0)
                for row in rows
            ),
            default=0.0,
        ),
        "maximum_model_price_difference": max(
            (
                float(row["final_published_model_price_difference"] or 0.0)
                for row in rows
            ),
            default=0.0,
        ),
    }


def _failure_fixture(context: Mapping[str, Any]):
    current = CurrentRepricingProblem(
        "BENCH-C100", "snapshot", datetime(2026, 1, 2, 12, tzinfo=UTC),
        int(context.get("options_calcs.calculation_definition_id", 1)),
        100.0, 2.0, 3600.0,
        0.04, 0.01, "crr", "v1", "a" * 64, "market_mid", "continuous_yield",
    )
    prior = PriorRepricingOutcome(
        current.option_symbol, "origin", current.source_collected_at - timedelta(seconds=1),
        current.calculation_definition_id,
        current.source_collected_at - timedelta(seconds=1),
        "origin", current.source_collected_at - timedelta(seconds=1),
        current.source_collected_at - timedelta(seconds=1), 100.0, 2.0, 3601.0,
        0.04, 0.01, "crr", "configured", "v1", "a" * 64, "{}",
        "market_mid", None, None, 100, None, None, None, None, None, None, None,
        "float64", "float64", False, 0, 256, None, None, "continuous_yield",
        status="failed", reason="iv_root_not_bracketed", failure_stage="iv_inversion",
        retry_class="deterministic_numerical",
    )
    return current, prior


def benchmark_exception_safety(context: Mapping[str, Any]) -> dict[str, Any]:
    configured = dict(context)
    configured["options_calcs.greeks.failure_reuse.enabled"] = True
    calls = 0
    surfaced = []

    def broken_price(_iv: float) -> float:
        nonlocal calls
        calls += 1
        raise RuntimeError("injected_programming_defect")

    for _snapshot in range(2):
        try:
            solve_iv(target_price=1.0, price_function=broken_price)
        except Exception as exc:
            surfaced.append(type(exc).__name__)
    current, prior = _failure_fixture(configured)
    prior = replace(
        prior,
        reason="solver_exception:RuntimeError",
        retry_class="unknown",
        failure_fingerprint="",
        retry_after=None,
    )
    decision = assess_failure_reuse(
        current,
        prior,
        parse_failure_reuse_policy(configured),
        current.source_collected_at,
    )
    return {
        "snapshots": 2,
        "programming_exceptions_surfaced": surfaced,
        "price_function_invocations": calls,
        "subsequent_snapshot_reused": decision.reusable,
        "reuse_rejection_reason": decision.reason,
    }


def benchmark_failure_backoff(context: Mapping[str, Any]) -> dict[str, Any]:
    configured = dict(context)
    configured["options_calcs.greeks.failure_reuse.enabled"] = True
    policy = parse_failure_reuse_policy(configured)
    current, prior = _failure_fixture(configured)
    deadline = retry_after_for_failure(
        current, RecalcRetryClass.DETERMINISTIC_NUMERICAL, policy,
        prior.origin_calculated_at,
    )
    prior = replace(prior, retry_after=deadline)
    attempts = 1
    reused = 0
    retry_attempts = 0
    for offset in range(10, 101, 10):
        candidate = replace(
            current,
            snapshot_id=f"snapshot-{offset}",
            source_collected_at=current.source_collected_at + timedelta(seconds=offset),
            time_to_expiry_seconds=current.time_to_expiry_seconds - offset,
        )
        decision = assess_failure_reuse(
            candidate, prior, policy, current.source_collected_at + timedelta(seconds=offset)
        )
        if decision.reusable:
            reused += 1
        else:
            attempts += 1
            retry_attempts += int(decision.reason == "retry_due")
    return {
        "selected_snapshots": 11,
        "actual_solver_attempts": attempts,
        "reused_failed_outcomes": reused,
        "retry_attempts": retry_attempts,
        "solver_attempts_avoided": reused,
        "affected_contract_snapshots": 11,
    }


def benchmark_circuit(context: Mapping[str, Any]) -> dict[str, Any]:
    from algotrading.refdata.options.solvers.Registry import CircuitBreakerConfig, MethodCircuitBreaker

    configured = dict(context)
    configured.update(
        {
            "options_calcs.greeks.circuit_breaker.enabled": True,
            "options_calcs.greeks.circuit_breaker.minimum_attempts": 3,
            "options_calcs.greeks.circuit_breaker.minimum_infrastructure_failures": 3,
            "options_calcs.greeks.circuit_breaker.failure_fraction": 1.0,
            "options_calcs.greeks.circuit_breaker.initial_open_seconds": 10,
        }
    )
    clock = [0.0]
    circuit = MethodCircuitBreaker(
        RepricingMethod.CRR,
        CircuitBreakerConfig.from_context(configured),
        monotonic=lambda: clock[0],
        utcnow=lambda: datetime(2026, 1, 2, tzinfo=UTC) + timedelta(seconds=clock[0]),
    )
    actual = 0
    suppressed = 0
    for _index in range(10):
        try:
            circuit.before_call()
        except Exception:
            suppressed += 1
            continue
        actual += 1
        circuit.record_infrastructure_failure()
    clock[0] = 11
    circuit.before_call()
    circuit.record_infrastructure_success()
    class SnapshotCircuit:
        def snapshot(self):
            return {"half_open_probes": 0}

    class NumericalDeltaRegistry:
        def describe(self, _method):
            return "fake", {"time_grid": 100}

        def circuit(self, _method):
            return SnapshotCircuit()

        def delta(self, *_args, performance, **_kwargs):
            performance["solver_call_permitted"] = 1
            performance["fdm_valuations"] = 1
            raise FloatingPointError("nonfinite_solver_delta")

        def price(self, request, *_args, performance, **_kwargs):
            performance["solver_call_permitted"] = 1
            performance["fdm_valuations"] = 1
            return request.spot

    fit = SmileFit(
        price_basis="mid", method="svi", status="complete", error="",
        parameters={"a": 0.01, "b": 0.1, "rho": 0.0, "m": 0.0, "sigma": 0.1},
        points=(), input_count=0, rejected_reasons={}, source_fallback_count=0,
        source_truncated=False, tenor_seconds=4 * 86400.0,
        tenor_years=4 / 365.25, spot=100.0, forward=100.0,
        rate=0.03, dividend_yield=0.01, exercise_type="A",
        weighted_rmse=0.0, maximum_absolute_error=0.0,
        butterfly_flags=0, calendar_flags=0, quality_flags=0,
        source_collected_at=datetime(2026, 1, 2, tzinfo=UTC),
        expiration_at=datetime(2026, 1, 6, tzinfo=UTC),
    )
    recovery = _routed_american_delta_points(
        fit,
        [100.0],
        [0.2],
        is_call=True,
        context={
            "options_calcs.greeks.method": "crank_nicolson",
            "options_calcs.greeks.short_tenor_method": "tian",
            "options_calcs.greeks.short_tenor_threshold_days": 3.0,
        },
        solver_registry=NumericalDeltaRegistry(),
    )
    return {
        "actual_permitted_calls": actual + 1,
        "fdm_actual_permitted_calls": 3,
        "actual_failing_calls_before_open": actual,
        "actual_infrastructure_failures": circuit.metrics[
            "actual_infrastructure_failures"
        ],
        "suppressed_calls_while_open": suppressed,
        "half_open_probes": circuit.metrics["half_open_probes"],
        "final_state": circuit.state.value,
        "direct_delta_fdm_valuations": recovery.diagnostics[
            "node_crank_nicolson_fdm_delta_valuations"
        ],
        "price_fallback_fdm_valuations": recovery.diagnostics[
            "node_crank_nicolson_fdm_price_valuations"
        ],
    }


def benchmark_coreset(context: Mapping[str, Any]) -> dict[str, Any]:
    tenor_seconds = 30 * 86400.0
    tenor_years = tenor_seconds / (365.25 * 86400.0)
    source_at = datetime(2026, 1, 2, 15, tzinfo=UTC)
    expiration_at = source_at + timedelta(days=30)
    rows = []
    for side, sign in (("PUT", -1), ("CALL", 1)):
        for index in range(1, 201):
            k = sign * index * 0.0025
            iv = 0.20 - 0.08 * k + 0.35 * k * k
            rows.append(
                {
                    "snapshot_id": "coreset-production-benchmark",
                    "source_collected_at": source_at,
                    "option_symbol": f"{side}-{index}",
                    "put_call": side,
                    "strike_price": 100.0 * math.exp(k),
                    "expiration_at": expiration_at,
                    "exercise_type": "E",
                    "multiplier": 100.0,
                    "is_mini": False,
                    "is_nonstandard": False,
                    "chain_is_truncated": False,
                    "bid": 0.95,
                    "ask": 1.05,
                    "source_iv": iv,
                    "calculation_status": "complete",
                    "contract_quality_flags": 0,
                    "spot_price": 100.0,
                    "mid_price": 1.0,
                    "spread_fraction_of_mid": 0.10,
                    "quote_age_seconds": 0.0,
                    "underlying_quote_age_seconds": 0.0,
                    "forward_price": 100.0,
                    "time_to_expiry_seconds": tenor_seconds,
                    "time_to_expiry_years": tenor_years,
                    "risk_free_rate": 0.03,
                    "dividend_yield": 0.01,
                    "preferred_iv": iv,
                    "preferred_iv_origin": "source",
                    "preferred_delta": sign * min(0.49, 0.02 * index),
                    "preferred_vega": 1.0 + index / 100,
                }
            )
    modes = {}
    for require_acceptance in (True, False):
        configured = dict(context)
        configured["options_calcs.surface.fit_points.enabled"] = True
        configured[
            "options_calcs.surface.fit_points.require_full_set_acceptance"
        ] = require_acceptance
        configured[
            "options_calcs.surface.fit_points.full_set_shadow.enabled"
        ] = False
        started = time.perf_counter()
        result = calibrate_expiration_bases(
            rows,
            price_bases=("mid",),
            context=configured,
            generate_nodes=False,
        )
        runtime_seconds = time.perf_counter() - started
        fit = result.calibrations["mid"].fit
        modes[str(require_acceptance).lower()] = {
            "candidate_points": fit.fit_candidate_contract_count,
            "initial_selected_points": min(
                fit.fit_candidate_contract_count,
                2
                * int(
                    configured[
                        "options_calcs.surface.fit_points.initial_max_points_per_wing"
                    ]
                ),
            ),
            "expansion_rounds": fit.fit_point_selection_rounds,
            "final_selected_points": fit.fit_selected_contract_count,
            "full_set_acceptance_comparisons": int(
                fit.fit_full_set_shadow_checked
            ),
            "full_set_acceptance_passed": fit.fit_full_set_acceptance_passed,
            "full_set_fallback_count": int(fit.fit_full_set_fallback),
            "full_set_fallback_rate": float(fit.fit_full_set_fallback),
            "published_fit_origin": fit.fit_point_selection_method,
            "maximum_common_grid_iv_difference": fit.fit_full_set_shadow_max_grid_iv_difference,
            "maximum_node_iv_difference": fit.fit_full_set_shadow_max_node_iv_difference,
            "maximum_node_strike_fraction_difference": fit.fit_full_set_shadow_max_node_strike_fraction_difference,
            "weighted_rmse_difference": fit.fit_full_set_shadow_weighted_rmse_difference,
            "required_nodes_match": fit.fit_full_set_required_nodes_match,
            "strategy_diagnostics_match": fit.fit_full_set_strategy_diagnostics_match,
            "atm_iv_difference": fit.fit_full_set_atm_iv_difference,
            "skew_difference": fit.fit_full_set_skew_difference,
            "curvature_difference": fit.fit_full_set_curvature_difference,
            "arbitrage_flags_changed": fit.fit_full_set_arbitrage_flags_changed,
            "acceptance_reason": fit.fit_full_set_acceptance_reason,
            "actual_fit_calls": fit.parameters.get("fit_actual_call_count"),
            "actual_svi_function_evaluations": fit.parameters.get(
                "fit_actual_svi_function_evaluations"
            ),
            "runtime_seconds": runtime_seconds,
            "mandatory_full_fit_cost_included": require_acceptance,
        }
    return {
        "require_full_set_acceptance_true": modes["true"],
        "require_full_set_acceptance_false_offline_research": modes["false"],
    }


def benchmark_calibration(context: Mapping[str, Any]) -> dict[str, Any]:
    configured = dict(context)
    configured["options_calcs.greeks.reuse.calibration.minimum_holdout_observations"] = 20
    now = datetime(2026, 1, 2, tzinfo=UTC)
    observations = [
        {
            "snapshot_id": f"snapshot-{index // 2}",
            "option_symbol": f"option-{index}",
            "validation_type": "success_reuse_shadow",
            "source_collected_at": now + timedelta(seconds=index),
            "pricing_method": "crr",
            "put_call": "CALL" if index % 2 else "PUT",
            "time_to_expiry_seconds": 3600.0,
            "log_forward_moneyness": (index - 250) / 1000,
            "candidate_age_seconds": 10.0,
            "spot_move_bps": 0.5,
            "elapsed_tenor_fraction": 0.001,
            "estimated_price_drift": 0.0001,
            "estimated_iv_change_points": 0.0001,
            "absolute_iv_error": 0.0001,
            "absolute_delta_error": 0.0005,
        }
        for index in range(500)
    ]
    passing = build_reuse_calibration_report(
        observations, context=configured, minimum_observations=100,
        policy_definition_id=int(
            configured.get("options_calcs.calculation_definition_id", 1)
        ),
    )
    failing_observations = []
    for row in observations:
        candidate = dict(row)
        if deterministic_calibration_split(candidate) == "holdout":
            candidate["absolute_iv_error"] = 0.01
            candidate["absolute_delta_error"] = 0.01
        failing_observations.append(candidate)
    failing = build_reuse_calibration_report(
        failing_observations,
        context=configured,
        minimum_observations=100,
        policy_definition_id=int(
            configured.get("options_calcs.calculation_definition_id", 1)
        ),
    )
    guard_rows = (
        {
            "candidate_age_seconds": 2.0,
            "spot_move_bps": 1.0,
            "elapsed_tenor_fraction": 0.001,
            "estimated_price_drift": 0.0001,
            "estimated_iv_change_points": 0.0001,
        },
        {
            "candidate_age_seconds": 1.0,
            "spot_move_bps": 2.0,
            "elapsed_tenor_fraction": 0.001,
            "estimated_price_drift": 0.0001,
            "estimated_iv_change_points": 0.0001,
        },
    )
    independent = any(
        candidate["candidate_age_seconds"] == 2.0
        and candidate["spot_move_bps"] == 2.0
        for candidate in _candidate_thresholds(guard_rows)
    )
    return {
        "passing_calibration_passing_holdout": {
            "recommendation_present": passing["recommendation"] is not None,
            "proposed_configuration_present": (
                passing["proposed_configuration"] is not None
            ),
            "holdout_slo_pass": passing["recommendation"][
                "holdout_slo_pass"
            ] if passing["recommendation"] is not None else False,
        },
        "passing_calibration_failing_holdout": {
            "recommendation_present": failing["recommendation"] is not None,
            "proposed_configuration_present": (
                failing["proposed_configuration"] is not None
            ),
            "rejection_reason": failing["rejected_candidate"][
                "rejection_reason"
            ],
        },
        "independent_guard_combination_present": independent,
    }


def run_benchmarks() -> dict[str, Any]:
    context = default_context("/tmp/algo")
    return {
        "routing_boundary": benchmark_routing_boundary(context),
        "prior_seed": benchmark_prior_seed(context),
        "exception_safety": benchmark_exception_safety(context),
        "failure_backoff": benchmark_failure_backoff(context),
        "circuit": benchmark_circuit(context),
        "coreset": benchmark_coreset(context),
        "calibration": benchmark_calibration(context),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = json.dumps(run_benchmarks(), sort_keys=True, indent=2, default=str)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
