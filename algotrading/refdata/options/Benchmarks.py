"""Bounded synthetic and ClickHouse-sample solver comparison CLI."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import multiprocessing
import os
import resource
import statistics
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from algotrading.refdata.options.solvers.Registry import SolverRegistry
from algotrading.refdata.options.solvers.Types import PricingRequest, RepricingMethod
from algotrading.refdata.options.solvers.Types import SolverBatchJob, SolverOperation


@dataclass(frozen=True)
class BenchmarkCase:
    symbol: str
    option_symbol: str
    put_call: str
    tenor_seconds: float
    spot: float
    strike: float
    rate: float
    dividend_yield: float
    volatility: float
    sample_source: str = "synthetic"
    source_collected_at: Any = None
    expiration_at: Any = None
    market_target: Optional[float] = None


def synthetic_cases() -> tuple[BenchmarkCase, ...]:
    cases = []
    for put_call in ("CALL", "PUT"):
        for strike in (80.0, 100.0, 120.0):
            for days in (0.25, 2.0, 30.0, 120.0):
                cases.append(
                    BenchmarkCase(
                        "SYNTH",
                        f"SYNTH-{put_call}-{strike:g}-{days:g}",
                        put_call,
                        days * 86400.0,
                        100.0,
                        strike,
                        0.035 if strike != 120 else -0.005,
                        0.01,
                        0.18 if strike == 100 else 0.40,
                    )
                )
    return tuple(cases)


def _request(case: BenchmarkCase) -> PricingRequest:
    return PricingRequest(
        case.option_symbol,
        "synthetic",
        case.put_call,
        "A",
        case.spot,
        case.strike,
        case.tenor_seconds,
        case.rate,
        case.dividend_yield,
        source_iv=case.volatility,
        source_collected_at=case.source_collected_at,
        expiration_at=case.expiration_at,
    )


def load_clickhouse_cases(
    client: Any,
    database: str,
    *,
    sample_limit: int,
    snapshot_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    symbols: Iterable[str] = (),
) -> tuple[BenchmarkCase, ...]:
    """Read a bounded, read-only American-option sample from canonical facts."""
    from algotrading.refdata.options.Common import validate_database_identifier

    validate_database_identifier(database)
    if sample_limit <= 0 or sample_limit > 1000:
        raise ValueError("sample-limit must be between 1 and 1000")
    conditions = [
        "exercise_type = 'A'",
        "expiration_at > collected_at",
        "chain_underlying_price > 0",
        "strike_price > 0",
        "source_iv > 0",
        "bid >= 0",
        "ask >= bid",
        "(bid + ask) > 0",
    ]
    parameters: dict[str, Any] = {"sample_limit": int(sample_limit)}
    if snapshot_id:
        conditions.append("s.snapshot_id = {snapshot_id:UUID}")
        parameters["snapshot_id"] = str(snapshot_id)
    if from_date:
        conditions.append("toDate(s.source_collected_at) >= {from_date:Date}")
        parameters["from_date"] = str(from_date)
    if to_date:
        conditions.append("toDate(s.source_collected_at) <= {to_date:Date}")
        parameters["to_date"] = str(to_date)
    selected_symbols = tuple(
        sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    )
    if selected_symbols:
        conditions.append("s.underlying_symbol IN {symbols:Array(String)}")
        parameters["symbols"] = list(selected_symbols)
    from algotrading.refdata.options.Queries import (
        authoritative_calculation_fact_subquery,
    )
    sql = f"""
        SELECT
            s.underlying_symbol,
            c.option_symbol,
            toString(c.put_call) AS put_call,
            s.source_collected_at AS collected_at,
            c.expiration_at,
            dateDiff('second', s.source_collected_at, c.expiration_at) AS tenor_seconds,
            toFloat64(s.chain_underlying_price) AS spot,
            toFloat64(c.strike_price) AS strike,
            toFloat64(i.risk_free_rate) AS rate,
            toFloat64(i.dividend_yield) AS dividend_yield,
            toFloat64(o.source_iv) AS source_iv,
            (toFloat64(o.bid) + toFloat64(o.ask)) / 2.0 AS market_target
        FROM {database}.options_chain_snapshots AS s
        INNER JOIN {database}.options_chain_observations AS o
          ON o.snapshot_id = s.snapshot_id
         AND o.source_collected_at = s.source_collected_at
        INNER JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c
          ON c.contract_id = o.contract_id
        INNER JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_expiration_inputs', fact_alias='i', state_alias='calculation_state')}
          ON i.snapshot_id = s.snapshot_id
         AND i.source_collected_at = s.source_collected_at
         AND i.expiration_at = c.expiration_at
        WHERE {' AND '.join(conditions)}
        ORDER BY s.source_collected_at DESC, s.underlying_symbol, c.option_symbol
        LIMIT {{sample_limit:UInt32}}
    """
    result = client.query(sql, parameters=parameters)
    names = tuple(result.column_names)
    cases = []
    for values in result.result_rows:
        row = dict(zip(names, values))
        cases.append(
            BenchmarkCase(
                symbol=str(row["underlying_symbol"]),
                option_symbol=str(row["option_symbol"]),
                put_call=str(row["put_call"]),
                tenor_seconds=float(row["tenor_seconds"]),
                spot=float(row["spot"]),
                strike=float(row["strike"]),
                rate=float(row["rate"]),
                dividend_yield=float(row["dividend_yield"]),
                volatility=float(row["source_iv"]),
                sample_source="clickhouse",
                source_collected_at=row["collected_at"],
                expiration_at=row["expiration_at"],
                market_target=float(row["market_target"]),
            )
        )
    return tuple(cases)


def load_clickhouse_surface_fits(
    client: Any,
    database: str,
    *,
    sample_limit: int,
    snapshot_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    symbols: Iterable[str] = (),
) -> tuple[Any, ...]:
    """Load a deterministic, bounded, read-only sample of completed fits."""
    from algotrading.refdata.options.Common import validate_database_identifier
    from algotrading.refdata.options.Queries import (
        authoritative_calculation_fact_subquery,
        authoritative_surface_fact_subquery,
    )
    from algotrading.refdata.options.Surface import SmileFit, SurfacePoint

    validate_database_identifier(database)
    if sample_limit <= 0 or sample_limit > 1000:
        raise ValueError("sample-limit must be between 1 and 1000")
    conditions = [
        "f.pricing_method != 'none'",
        "f.price_basis IN ('mid', 'source')",
        "f.expiration_at > f.source_collected_at",
        "s.chain_underlying_price > 0",
        "i.forward_price > 0",
        "f.maximum_log_forward_moneyness > f.minimum_log_forward_moneyness",
    ]
    parameters: dict[str, Any] = {"sample_limit": int(sample_limit)}
    if snapshot_id:
        conditions.append("f.snapshot_id = {snapshot_id:UUID}")
        parameters["snapshot_id"] = str(snapshot_id)
    if from_date:
        conditions.append("toDate(f.source_collected_at) >= {from_date:Date}")
        parameters["from_date"] = str(from_date)
    if to_date:
        conditions.append("toDate(f.source_collected_at) <= {to_date:Date}")
        parameters["to_date"] = str(to_date)
    selected_symbols = tuple(
        sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    )
    if selected_symbols:
        conditions.append("f.underlying_symbol IN {symbols:Array(String)}")
        parameters["symbols"] = list(selected_symbols)
    sql = f"""
        SELECT
            toString(f.snapshot_id) AS snapshot_id,
            f.source_collected_at AS source_collected_at,
            f.underlying_symbol AS underlying_symbol,
            f.expiration_at AS expiration_at,
            toString(f.price_basis) AS price_basis,
            toString(f.fit_method) AS fit_method,
            f.svi_a, f.svi_b, f.svi_rho, f.svi_m, f.svi_sigma,
            f.pchip_k, f.pchip_total_variance,
            f.input_contract_count,
            f.minimum_log_forward_moneyness,
            f.maximum_log_forward_moneyness,
            f.weighted_rmse,
            f.maximum_absolute_error,
            f.butterfly_arbitrage_flags,
            f.calendar_arbitrage_flags,
            f.quality_flags AS quality_flags,
            toString(f.actual_dtype) AS actual_dtype,
            toFloat64(s.chain_underlying_price) AS spot,
            toFloat64(i.forward_price) AS forward,
            toFloat64(i.risk_free_rate) AS rate,
            toFloat64(i.dividend_yield) AS dividend_yield,
            toFloat64(i.time_to_expiry_seconds) AS tenor_seconds
        FROM {authoritative_surface_fact_subquery(database, 'options_chain_surface_fits', fact_alias='f', state_alias='surface_state')}
        INNER JOIN (
            SELECT * FROM {database}.options_chain_calculation_state FINAL
        ) AS calculation_state
          ON calculation_state.snapshot_id = f.snapshot_id
         AND calculation_state.source_collected_at = f.source_collected_at
         AND calculation_state.calculation_definition_id = f.source_calculation_definition_id
         AND calculation_state.is_active
         AND calculation_state.status IN ('complete', 'complete_with_failures')
        INNER JOIN {database}.options_chain_snapshots AS s
          ON s.snapshot_id = f.snapshot_id
         AND s.source_collected_at = f.source_collected_at
        INNER JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_expiration_inputs', fact_alias='i', state_alias='expiration_state')}
          ON i.snapshot_id = f.snapshot_id
         AND i.source_collected_at = f.source_collected_at
         AND i.calculation_definition_id = calculation_state.calculation_definition_id
         AND i.expiration_at = f.expiration_at
        WHERE {' AND '.join(conditions)}
        ORDER BY cityHash64(
            concat(toString(f.snapshot_id), '|', toString(f.expiration_at), '|', f.price_basis)
        ), f.underlying_symbol, f.expiration_at
        LIMIT {{sample_limit:UInt32}}
    """
    result = client.query(sql, parameters=parameters)
    names = tuple(result.column_names)
    fits = []
    for values in result.result_rows:
        row = dict(zip(names, values))
        method = str(row["fit_method"]).strip().lower()
        if method == "svi":
            parameters_value = {
                name: float(row[f"svi_{name}"])
                for name in ("a", "b", "rho", "m", "sigma")
            }
        elif method == "pchip":
            parameters_value = {
                "k": [float(value) for value in row["pchip_k"]],
                "total_variance": [
                    float(value) for value in row["pchip_total_variance"]
                ],
            }
        else:
            continue
        minimum_k = float(row["minimum_log_forward_moneyness"])
        maximum_k = float(row["maximum_log_forward_moneyness"])
        tenor_seconds = float(row["tenor_seconds"])
        tenor_years = tenor_seconds / (365.25 * 86400.0)

        def total_variance(k: float) -> float:
            if method == "svi":
                a, b, rho, m, sigma = (
                    parameters_value[name]
                    for name in ("a", "b", "rho", "m", "sigma")
                )
                shifted = k - m
                return float(
                    a + b * (rho * shifted + math.sqrt(shifted * shifted + sigma * sigma))
                )
            return float(
                np.interp(
                    k,
                    parameters_value["k"],
                    parameters_value["total_variance"],
                )
            )

        coverage_points = tuple(
            SurfacePoint(
                option_symbol=(
                    f"{row['underlying_symbol']}-benchmark-coverage-{index}"
                ),
                strike=float(row["forward"]) * math.exp(k),
                log_forward_moneyness=k,
                iv=math.sqrt(max(total_variance(k), 0.0) / tenor_years),
                total_variance=total_variance(k),
                weight=1.0,
                delta=None,
                put_call="PUT" if k < 0.0 else "CALL",
                price=1.0,
                price_residual=0.0,
            )
            for index, k in enumerate((minimum_k, maximum_k))
        )
        fits.append(
            SmileFit(
                price_basis=str(row["price_basis"]),
                method=method,
                status="complete",
                error="",
                parameters=parameters_value,
                points=coverage_points,
                input_count=int(row["input_contract_count"]),
                rejected_reasons={},
                source_fallback_count=0,
                source_truncated=False,
                tenor_seconds=tenor_seconds,
                tenor_years=tenor_years,
                spot=float(row["spot"]),
                forward=float(row["forward"]),
                rate=float(row["rate"]),
                dividend_yield=float(row["dividend_yield"]),
                exercise_type="A",
                weighted_rmse=float(row["weighted_rmse"]),
                maximum_absolute_error=float(row["maximum_absolute_error"]),
                butterfly_flags=int(row["butterfly_arbitrage_flags"]),
                calendar_flags=int(row["calendar_arbitrage_flags"]),
                quality_flags=int(row["quality_flags"]),
                requested_dtype=str(row["actual_dtype"]),
                actual_dtype=str(row["actual_dtype"]),
                source_collected_at=row["source_collected_at"],
                expiration_at=row["expiration_at"],
                snapshot_id=str(row["snapshot_id"]),
            )
        )
    return tuple(fits)


def run_surface_node_ab_benchmark(
    context: Mapping[str, Any], fits: Iterable[Any]
) -> list[dict[str, Any]]:
    """Compare legacy and hybrid node generation without publishing rows."""
    from algotrading.refdata.options.Surface import generate_surface_nodes

    fit_values = tuple(fits)
    if not fit_values:
        raise ValueError("surface node A/B benchmark requires at least one fit")
    legacy_context = {
        **context,
        "options_calcs.surface.node_hybrid_enabled": False,
        "options_calcs.surface.node_search_method": "observed_then_analytic",
        "options_calcs.surface.node_exact_max_evaluations": 8,
        "options_calcs.surface.node_dense_exact_fallback_enabled": True,
        "options_calcs.surface.node_dense_exact_fallback_grid_size": 81,
    }
    hybrid_context = {
        **context,
        "options_calcs.surface.node_hybrid_enabled": True,
        "options_calcs.surface.node_search_method": "tian",
        "options_calcs.surface.node_validation_method": context.get(
            "options_calcs.surface.node_validation_method", "configured"
        ),
        "options_calcs.surface.node_tian_grid_size": int(
            context.get("options_calcs.surface.node_tian_grid_size", 81)
        ),
        "options_calcs.surface.node_validation_max_evaluations": int(
            context.get("options_calcs.surface.node_validation_max_evaluations", 4)
        ),
        "options_calcs.surface.node_validation_initial_candidates": int(
            context.get("options_calcs.surface.node_validation_initial_candidates", 1)
        ),
        "options_calcs.surface.node_legacy_exact_fallback_enabled": context.get(
            "options_calcs.surface.node_legacy_exact_fallback_enabled", True
        ),
    }
    # Exclude one-time adapter/JIT initialization from both timed paths.  The
    # production worker keeps one registry alive across many surfaces.
    legacy_registry = SolverRegistry(legacy_context)
    hybrid_registry = SolverRegistry(hybrid_context)
    generate_surface_nodes(
        fit_values[0],
        context=legacy_context,
        dtype=np.dtype("float64"),
        solver_registry=legacy_registry,
    )
    generate_surface_nodes(
        fit_values[0],
        context=hybrid_context,
        dtype=np.dtype("float64"),
        solver_registry=hybrid_registry,
    )
    # Timed runs start with fresh caches/circuit state after process-wide code
    # paths have been initialized.
    legacy_registry = SolverRegistry(legacy_context)
    hybrid_registry = SolverRegistry(hybrid_context)
    rows: list[dict[str, Any]] = []
    residuals_by_snapshot: dict[str, list[float]] = {}
    strike_differences_by_snapshot: dict[str, list[float]] = {}
    for fit in fit_values:
        legacy_diagnostics: dict[str, int] = {}
        hybrid_diagnostics: dict[str, int] = {}
        started = time.perf_counter()
        legacy_nodes = generate_surface_nodes(
            fit,
            context=legacy_context,
            dtype=np.dtype("float64"),
            diagnostics=legacy_diagnostics,
            solver_registry=legacy_registry,
        )
        legacy_seconds = time.perf_counter() - started
        started = time.perf_counter()
        hybrid_nodes = generate_surface_nodes(
            fit,
            context=hybrid_context,
            dtype=np.dtype("float64"),
            diagnostics=hybrid_diagnostics,
            solver_registry=hybrid_registry,
        )
        hybrid_seconds = time.perf_counter() - started
        legacy_by_key = {
            (node.node_value, node.put_call): node
            for node in legacy_nodes
            if node.node_type == "delta"
        }
        hybrid_by_key = {
            (node.node_value, node.put_call): node
            for node in hybrid_nodes
            if node.node_type == "delta"
        }
        residuals = sorted(
            abs(
                float(node.fitted_delta)
                - (key[0] if key[1] == "CALL" else -key[0])
            )
            for key, node in hybrid_by_key.items()
            if node.status == "complete" and node.fitted_delta is not None
        )
        strike_differences = sorted(
            abs(float(node.fitted_strike) - float(legacy_by_key[key].fitted_strike))
            for key, node in hybrid_by_key.items()
            if node.status == "complete"
            and node.fitted_strike is not None
            and key in legacy_by_key
            and legacy_by_key[key].status == "complete"
            and legacy_by_key[key].fitted_strike is not None
        )
        residuals_by_snapshot.setdefault(str(fit.snapshot_id), []).extend(residuals)
        strike_differences_by_snapshot.setdefault(
            str(fit.snapshot_id), []
        ).extend(strike_differences)

        def percentile(values: list[float], fraction: float) -> float:
            if not values:
                return math.nan
            return values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]

        legacy_evaluations = int(
            legacy_diagnostics.get("node_exact_delta_evaluations", 0)
        )
        hybrid_evaluations = int(
            hybrid_diagnostics.get("surface.node_validation_evaluations", 0)
        )
        rows.append(
            {
                "comparison": "surface_node_ab_expiration",
                "sample_source": "clickhouse_fit_shadow",
                "snapshot_id": fit.snapshot_id,
                "symbol": fit.points[0].option_symbol.split("-")[0]
                if fit.points else "",
                "expiration_at": str(fit.expiration_at),
                "tenor_seconds": fit.tenor_seconds,
                "legacy_wall_seconds": legacy_seconds,
                "hybrid_wall_seconds": hybrid_seconds,
                "speedup_ratio": legacy_seconds / max(hybrid_seconds, 1.0e-15),
                "legacy_authoritative_evaluations": legacy_evaluations,
                "hybrid_authoritative_evaluations": hybrid_evaluations,
                "authoritative_evaluation_reduction_fraction": (
                    (legacy_evaluations - hybrid_evaluations) / legacy_evaluations
                    if legacy_evaluations else math.nan
                ),
                "tian_search_evaluations": hybrid_diagnostics.get(
                    "surface.node_tian_search_evaluations", 0
                ),
                "legacy_dense_exact_evaluations": legacy_diagnostics.get(
                    "node_dense_exact_fallback_evaluations", 0
                ),
                "hybrid_dense_exact_evaluations": hybrid_diagnostics.get(
                    "surface.node_dense_exact_fallback_evaluations", 0
                ),
                "legacy_successful_nodes": sum(
                    node.status == "complete" for node in legacy_by_key.values()
                ),
                "hybrid_successful_nodes": sum(
                    node.status == "complete" for node in hybrid_by_key.values()
                ),
                "hybrid_unavailable_nodes": sum(
                    node.status != "complete" for node in hybrid_by_key.values()
                ),
                "maximum_final_delta_residual": max(residuals, default=math.nan),
                "p95_final_delta_residual": percentile(residuals, 0.95),
                "maximum_strike_difference": max(
                    strike_differences, default=math.nan
                ),
                "p95_strike_difference": percentile(strike_differences, 0.95),
            }
        )
    expiration_rows = list(rows)
    by_snapshot: dict[str, list[dict[str, Any]]] = {}
    for row in expiration_rows:
        by_snapshot.setdefault(str(row["snapshot_id"]), []).append(row)
    full_surface = max(
        by_snapshot.values(), key=lambda values: (len(values), values[0]["snapshot_id"])
    )

    def aggregate_row(
        values: list[dict[str, Any]], comparison: str
    ) -> dict[str, Any]:
        legacy_seconds = sum(float(value["legacy_wall_seconds"]) for value in values)
        hybrid_seconds = sum(float(value["hybrid_wall_seconds"]) for value in values)
        legacy_evaluations = sum(
            int(value["legacy_authoritative_evaluations"]) for value in values
        )
        hybrid_evaluations = sum(
            int(value["hybrid_authoritative_evaluations"]) for value in values
        )
        return {
            "comparison": comparison,
            "sample_source": "clickhouse_fit_shadow",
            "snapshot_id": values[0]["snapshot_id"],
            "symbol": values[0]["symbol"],
            "expiration_at": f"{len(values)} expirations",
            "tenor_seconds": sum(float(value["tenor_seconds"]) for value in values),
            "legacy_wall_seconds": legacy_seconds,
            "hybrid_wall_seconds": hybrid_seconds,
            "speedup_ratio": legacy_seconds / max(hybrid_seconds, 1.0e-15),
            "legacy_authoritative_evaluations": legacy_evaluations,
            "hybrid_authoritative_evaluations": hybrid_evaluations,
            "authoritative_evaluation_reduction_fraction": (
                (legacy_evaluations - hybrid_evaluations) / legacy_evaluations
                if legacy_evaluations else math.nan
            ),
            "tian_search_evaluations": sum(
                int(value["tian_search_evaluations"]) for value in values
            ),
            "legacy_dense_exact_evaluations": sum(
                int(value["legacy_dense_exact_evaluations"]) for value in values
            ),
            "hybrid_dense_exact_evaluations": sum(
                int(value["hybrid_dense_exact_evaluations"]) for value in values
            ),
            "legacy_successful_nodes": sum(
                int(value["legacy_successful_nodes"]) for value in values
            ),
            "hybrid_successful_nodes": sum(
                int(value["hybrid_successful_nodes"]) for value in values
            ),
            "hybrid_unavailable_nodes": sum(
                int(value["hybrid_unavailable_nodes"]) for value in values
            ),
            "maximum_final_delta_residual": max(
                float(value["maximum_final_delta_residual"]) for value in values
            ),
            "p95_final_delta_residual": max(
                float(value["p95_final_delta_residual"]) for value in values
            ),
            "maximum_strike_difference": max(
                float(value["maximum_strike_difference"]) for value in values
            ),
            "p95_strike_difference": max(
                float(value["p95_strike_difference"]) for value in values
            ),
        }

    full_surface_row = aggregate_row(
        full_surface, "surface_node_ab_full_surface"
    )
    full_snapshot_id = str(full_surface_row["snapshot_id"])
    full_residuals = sorted(residuals_by_snapshot.get(full_snapshot_id, []))
    full_strike_differences = sorted(
        strike_differences_by_snapshot.get(full_snapshot_id, [])
    )
    full_surface_row["maximum_final_delta_residual"] = max(
        full_residuals, default=math.nan
    )
    full_surface_row["p95_final_delta_residual"] = percentile(
        full_residuals, 0.95
    )
    full_surface_row["maximum_strike_difference"] = max(
        full_strike_differences, default=math.nan
    )
    full_surface_row["p95_strike_difference"] = percentile(
        full_strike_differences, 0.95
    )
    rows.append(full_surface_row)

    cross_owner_fits = fit_values[: min(4, len(fit_values))]
    cross_owner_diagnostics: list[dict[str, int]] = [
        {} for _ in cross_owner_fits
    ]
    cross_owner_registry = SolverRegistry(hybrid_context)
    from algotrading.refdata.options.Surface import (
        ExpirationCalibration,
        generate_calibration_nodes_many,
    )

    started = time.perf_counter()
    cross_owner_results = generate_calibration_nodes_many(
        [ExpirationCalibration(fit) for fit in cross_owner_fits],
        context=hybrid_context,
        solver_registry=cross_owner_registry,
        diagnostics=cross_owner_diagnostics,
    )
    cross_owner_seconds = time.perf_counter() - started
    comparable_rows = expiration_rows[: len(cross_owner_fits)]
    cross_owner_row = aggregate_row(
        comparable_rows, "surface_node_ab_cross_owner"
    )
    cross_owner_row["hybrid_wall_seconds"] = cross_owner_seconds
    cross_owner_row["speedup_ratio"] = float(
        cross_owner_row["legacy_wall_seconds"]
    ) / max(cross_owner_seconds, 1.0e-15)
    cross_owner_row["hybrid_authoritative_evaluations"] = sum(
        values.get("surface.node_validation_evaluations", 0)
        for values in cross_owner_diagnostics
    )
    cross_owner_row["tian_search_evaluations"] = sum(
        values.get("surface.node_tian_search_evaluations", 0)
        for values in cross_owner_diagnostics
    )
    cross_owner_row["hybrid_successful_nodes"] = sum(
        node.node_type == "delta" and node.status == "complete"
        for result in cross_owner_results
        for node in result.nodes
    )
    cross_owner_row["hybrid_unavailable_nodes"] = sum(
        node.node_type == "delta" and node.status != "complete"
        for result in cross_owner_results
        for node in result.nodes
    )
    cross_owner_residuals = sorted(
        abs(
            float(node.fitted_delta)
            - (
                float(node.node_value)
                if node.put_call == "CALL"
                else -float(node.node_value)
            )
        )
        for result in cross_owner_results
        for node in result.nodes
        if node.node_type == "delta"
        and node.status == "complete"
        and node.fitted_delta is not None
    )
    cross_owner_row["maximum_final_delta_residual"] = max(
        cross_owner_residuals, default=math.nan
    )
    cross_owner_row["p95_final_delta_residual"] = percentile(
        cross_owner_residuals, 0.95
    )
    legacy_evaluations = int(cross_owner_row["legacy_authoritative_evaluations"])
    hybrid_evaluations = int(cross_owner_row["hybrid_authoritative_evaluations"])
    cross_owner_row["authoritative_evaluation_reduction_fraction"] = (
        (legacy_evaluations - hybrid_evaluations) / legacy_evaluations
        if legacy_evaluations else math.nan
    )
    rows.append(cross_owner_row)
    return rows


def write_surface_node_ab_report(
    rows: list[dict[str, Any]], output_dir: Path
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "surface_node_ab.csv"
    fields = sorted({name for row in rows for name in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "surface_node_ab.json"
    summary_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    expiration_rows = [
        row
        for row in rows
        if row.get("comparison") == "surface_node_ab_expiration"
    ]
    speedups = [float(row["speedup_ratio"]) for row in expiration_rows]
    reductions = [
        float(row["authoritative_evaluation_reduction_fraction"])
        for row in expiration_rows
        if math.isfinite(float(row["authoritative_evaluation_reduction_fraction"]))
    ]
    markdown_path = output_dir / "surface_node_ab.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Surface node legacy/hybrid A/B",
                "",
                "Read-only shadow calculation; no rows were published.",
                "",
                f"Expirations: {len(expiration_rows)}",
                f"Median speedup: {statistics.median(speedups):.3f}x",
                f"Median authoritative-evaluation reduction: {statistics.median(reductions):.1%}",
                f"Hybrid successful nodes: {sum(int(row['hybrid_successful_nodes']) for row in expiration_rows)}",
                f"Legacy successful nodes: {sum(int(row['legacy_successful_nodes']) for row in expiration_rows)}",
                f"Hybrid dense exact evaluations: {sum(int(row['hybrid_dense_exact_evaluations']) for row in expiration_rows)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return csv_path, markdown_path, summary_path


def run_synthetic(
    context: dict[str, Any], cases: Optional[Iterable[BenchmarkCase]] = None
) -> list[dict[str, Any]]:
    registry = SolverRegistry(context)
    registry.preload()
    rows: list[dict[str, Any]] = []
    for case in cases if cases is not None else synthetic_cases():
        request = _request(case)
        methods = [RepricingMethod.CRR, RepricingMethod.TIAN]
        if case.tenor_seconds >= 86400:
            methods.append(RepricingMethod.CRANK_NICOLSON)
        target = (
            float(case.market_target)
            if case.market_target is not None
            else registry.get(RepricingMethod.TIAN).price(request, case.volatility)
        )
        for method in methods:
            adapter = registry.get(method)
            full_session = adapter.create_pricing_session(request)
            started = time.perf_counter()
            supplied = adapter.greeks(
                request, case.volatility, pricing_session=full_session
            )
            supplied_runtime = time.perf_counter() - started
            full_fdm_valuations = int(
                getattr(full_session, "fdm_valuations", 0)
            )
            rows.append(
                {
                    "comparison": "same_supplied_iv",
                    "sample_source": case.sample_source,
                    "symbol": case.symbol,
                    "option_symbol": case.option_symbol,
                    "put_call": case.put_call,
                    "tenor_seconds": case.tenor_seconds,
                    "moneyness": case.strike / case.spot,
                    "method": method.value,
                    "status": "complete",
                    "failure_reason": "",
                    "solver_parameters": json.dumps(adapter.parameters, sort_keys=True),
                    "runtime_seconds": supplied_runtime,
                    "iv": case.volatility,
                    "target_price": "",
                    "price_residual": "",
                    **supplied,
                }
            )
            delta_session = adapter.create_pricing_session(request)
            started = time.perf_counter()
            delta_only = adapter.delta(
                request, case.volatility, pricing_session=delta_session
            )
            delta_runtime = time.perf_counter() - started
            rows.append(
                {
                    "comparison": "delta_only",
                    "sample_source": case.sample_source,
                    "symbol": case.symbol,
                    "option_symbol": case.option_symbol,
                    "put_call": case.put_call,
                    "tenor_seconds": case.tenor_seconds,
                    "moneyness": case.strike / case.spot,
                    "method": method.value,
                    "status": "complete",
                    "failure_reason": "",
                    "solver_parameters": json.dumps(adapter.parameters, sort_keys=True),
                    "runtime_seconds": delta_runtime,
                    "full_greeks_runtime_seconds": supplied_runtime,
                    "delta_only_runtime_seconds": delta_runtime,
                    "full_greeks_fdm_valuations": full_fdm_valuations,
                    "delta_only_fdm_valuations": int(
                        getattr(delta_session, "fdm_valuations", 0)
                    ),
                    "delta_difference": abs(delta_only - float(supplied["delta"])),
                    "iv": case.volatility,
                    "target_price": "",
                    "price_residual": "",
                    "model_price": "",
                    "delta": delta_only,
                }
            )
            started = time.perf_counter()
            performance: dict[str, float | int] = {}
            solved = registry.solve_iv_and_greeks(
                request,
                method=method,
                method_selection_reason="benchmark",
                target_price=target,
                performance=performance,
            )
            solved_runtime = time.perf_counter() - started
            rows.append(
                {
                    "comparison": "same_market_target",
                    "sample_source": case.sample_source,
                    "symbol": case.symbol,
                    "option_symbol": case.option_symbol,
                    "put_call": case.put_call,
                    "tenor_seconds": case.tenor_seconds,
                    "moneyness": case.strike / case.spot,
                    "method": method.value,
                    "status": solved.status,
                    "failure_reason": solved.reason if solved.status != "complete" else "",
                    "solver_parameters": json.dumps(adapter.parameters, sort_keys=True),
                    "runtime_seconds": solved_runtime,
                    "solver_total_seconds": performance.get(
                        "total_seconds", solved_runtime
                    ),
                    "session_init_seconds": performance.get(
                        "session_init_seconds", 0.0
                    ),
                    "iv_seconds": performance.get("iv_seconds", 0.0),
                    "greeks_seconds": performance.get("greeks_seconds", 0.0),
                    "iv_price_evaluations": int(
                        performance.get("iv_price_evaluations", 0)
                    ),
                    "fdm_valuations_total": int(
                        performance.get("fdm_valuations_total", 0)
                    ),
                    "fdm_valuations_iv": int(
                        performance.get("fdm_valuations_iv", 0)
                    ),
                    "fdm_valuations_greeks": int(
                        performance.get("fdm_valuations_greeks", 0)
                    ),
                    "iv": solved.implied_volatility,
                    "target_price": target,
                    "price_residual": solved.price_residual,
                    "model_price": solved.model_price,
                    "delta": solved.delta,
                    "gamma": solved.gamma,
                    "theta_per_day": solved.theta_per_day,
                    "vega_per_1pct": solved.vega_per_1pct,
                    "rho_per_1pct": solved.rho_per_1pct,
                }
            )
    return rows


def run_tian_kernel_benchmark() -> list[dict[str, Any]]:
    """Reproducible scalar-reference versus aligned Tian release matrix."""
    from algotrading.refdata.options.solvers import Tian
    import torch

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260717)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def maximum_difference(left: Any, right: Any) -> float:
        if isinstance(left, Mapping):
            return max(
                float(
                    np.nanmax(
                        np.abs(
                            np.asarray(left[name], dtype=np.float64)
                            - np.asarray(right[name], dtype=np.float64)
                        )
                    )
                )
                for name in left
            )
        return float(
            np.nanmax(
                np.abs(
                    np.asarray(left, dtype=np.float64)
                    - np.asarray(right, dtype=np.float64)
                )
            )
        )

    try:
        for count in (1, 10, 20, 40, 80, 160, 320):
            spot = rng.uniform(80.0, 140.0, count)
            strike = spot * rng.uniform(0.75, 1.25, count)
            tenor = rng.uniform(60.0, 3.0 * 86400.0, count) / (365.25 * 86400.0)
            rate = rng.uniform(-0.01, 0.08, count)
            dividend = rng.uniform(0.0, 0.05, count)
            sigma = rng.uniform(0.08, 1.2, count)
            calls = rng.integers(0, 2, count).astype(bool)
            kwargs = dict(
                S=spot, K=strike, T=tenor, r=rate, q=dividend,
                sigma=sigma, is_call=calls, steps=256, dtype=np.float64,
                chunk_size=max(1, count),
            )
            operations = (
                ("price", Tian.price_from_iv_batch),
                ("price_delta", Tian.price_delta_from_iv_batch),
                ("full_greeks", Tian.greeks_from_iv_batch),
            )
            for operation, function in operations:
                function(**kwargs)
                started = time.perf_counter()
                scalar = function(
                    **kwargs,
                    vectorized_enabled=False,
                    scalar_fallback_enabled=True,
                )
                scalar_seconds = time.perf_counter() - started
                started = time.perf_counter()
                vector = function(**kwargs)
                vector_seconds = time.perf_counter() - started
                error = maximum_difference(vector, scalar)
                rows.append(
                    {
                        "comparison": f"tian_kernel_{operation}",
                        "method": "tian",
                        "route": "aligned_vector",
                        "contract_count": count,
                        "logical_job_count": count,
                        "baseline_seconds": scalar_seconds,
                        "runtime_seconds": vector_seconds,
                        "throughput_per_second": count / vector_seconds,
                        "speedup_vs_scalar": scalar_seconds / vector_seconds,
                        "physical_call_count": 1,
                        "baseline_physical_call_count": count,
                        "mean_batch_size": count,
                        "p95_batch_size": count,
                        "peak_memory_estimate_mib": (
                            count * 257 * 8 * (12 if operation == "full_greeks" else 5)
                            / (1024.0 * 1024.0)
                        ),
                        "maximum_absolute_difference": error,
                        "status_reason_mismatch_count": 0,
                        "status": "complete",
                        "price_residual": "",
                        "sample_source": "synthetic_seed_20260717",
                    }
                )

            probe_kwargs = {**kwargs, "sigma": sigma * 1.05}
            Tian.price_from_iv_batch(**probe_kwargs)
            started = time.perf_counter()
            scalar_probe = Tian.price_from_iv_batch(
                **probe_kwargs, vectorized_enabled=False
            )
            scalar_seconds = time.perf_counter() - started
            started = time.perf_counter()
            vector_probe = Tian.price_from_iv_batch(**probe_kwargs)
            vector_seconds = time.perf_counter() - started
            rows.append(
                {
                    "comparison": "tian_iv_probe_wave",
                    "method": "tian",
                    "route": "aligned_vector",
                    "contract_count": count,
                    "logical_job_count": count,
                    "baseline_seconds": scalar_seconds,
                    "runtime_seconds": vector_seconds,
                    "throughput_per_second": count / vector_seconds,
                    "speedup_vs_scalar": scalar_seconds / vector_seconds,
                    "physical_call_count": 1,
                    "baseline_physical_call_count": count,
                    "mean_batch_size": count,
                    "p95_batch_size": count,
                    "peak_memory_estimate_mib": (
                        count * 257 * 8 * 4 / (1024.0 * 1024.0)
                    ),
                    "maximum_absolute_difference": maximum_difference(
                        vector_probe, scalar_probe
                    ),
                    "status_reason_mismatch_count": 0,
                    "status": "complete",
                    "price_residual": "",
                    "sample_source": "synthetic_seed_20260717",
                }
            )

            target_prices = np.asarray(Tian.price_from_iv_batch(**kwargs))
            jobs = []
            for index in range(count):
                request = PricingRequest(
                    option_symbol=f"TIAN-IV-{count}-{index}",
                    snapshot_id=f"tian-kernel-{index % 8}",
                    put_call="CALL" if calls[index] else "PUT",
                    exercise_type="A",
                    spot=float(spot[index]),
                    strike=float(strike[index]),
                    tenor_seconds=float(tenor[index] * 365.25 * 86400.0),
                    rate=float(rate[index]),
                    dividend_yield=float(dividend[index]),
                    source_iv=float(sigma[index]),
                    requested_dtype="float64",
                )
                jobs.append(
                    SolverBatchJob(
                        job_id=f"tian-iv-{count}-{index}",
                        owner_run_id=f"owner-{index % 8}",
                        owner_snapshot_id=request.snapshot_id,
                        output_index=index,
                        operation=SolverOperation.SOLVE_IV_AND_GREEKS,
                        request=request,
                        method=RepricingMethod.TIAN,
                        method_selection_reason="benchmark",
                        target_price=float(target_prices[index]),
                        requested_dtype="float64",
                    )
                )
            scalar_registry = SolverRegistry(
                {
                    "options_calcs.greeks.tian.steps": 256,
                    "options_calcs.greeks.batch.enabled": False,
                }
            )
            vector_registry = SolverRegistry(
                {"options_calcs.greeks.tian.steps": 256}
            )
            started = time.perf_counter()
            scalar_results = scalar_registry.solve_iv_and_greeks_many(jobs)
            scalar_seconds = time.perf_counter() - started
            started = time.perf_counter()
            vector_results = vector_registry.solve_iv_and_greeks_many(jobs)
            vector_seconds = time.perf_counter() - started
            mismatches = sum(
                (left.status, left.reason) != (right.status, right.reason)
                for left, right in zip(scalar_results, vector_results)
            )
            numerical_names = (
                "implied_volatility", "model_price", "delta", "gamma",
                "theta_per_day", "vega_per_1pct", "rho_per_1pct",
            )
            errors = [
                abs(float(getattr(left, name)) - float(getattr(right, name)))
                for left, right in zip(scalar_results, vector_results)
                for name in numerical_names
                if getattr(left, name) is not None and getattr(right, name) is not None
            ]
            sizes = sorted(vector_registry.batch_sizes)
            rows.append(
                {
                    "comparison": "tian_iv_and_final_greeks",
                    "method": "tian",
                    "route": "aligned_vector",
                    "contract_count": count,
                    "logical_job_count": count,
                    "baseline_seconds": scalar_seconds,
                    "runtime_seconds": vector_seconds,
                    "throughput_per_second": count / vector_seconds,
                    "speedup_vs_scalar": scalar_seconds / vector_seconds,
                    "physical_call_count": vector_registry.batch_metrics["physical_calls"],
                    "baseline_physical_call_count": count,
                    "mean_batch_size": statistics.mean(sizes) if sizes else 0.0,
                    "p95_batch_size": (
                        sizes[math.ceil(0.95 * len(sizes)) - 1] if sizes else 0
                    ),
                    "peak_memory_estimate_mib": (
                        count * 257 * 8 * 12 / (1024.0 * 1024.0)
                    ),
                    "maximum_absolute_difference": max(errors, default=0.0),
                    "status_reason_mismatch_count": mismatches,
                    "status": "complete" if not mismatches else "failed",
                    "price_residual": "",
                    "sample_source": "synthetic_seed_20260717",
                }
            )
    finally:
        torch.set_num_threads(previous_threads)
    return rows


def _solver_jobs_by_owner(
    context: dict[str, Any], *, owners: int, jobs_per_owner: int, mixed: bool
) -> tuple[SolverRegistry, list[list[SolverBatchJob]]]:
    registry = SolverRegistry(context)
    grouped: list[list[SolverBatchJob]] = []
    for owner in range(owners):
        owner_jobs: list[SolverBatchJob] = []
        for position in range(jobs_per_owner):
            method = (
                RepricingMethod.CRANK_NICOLSON
                if mixed and position % 8 == 7
                else RepricingMethod.TIAN
            )
            tenor = 30.0 * 86400.0 if method is RepricingMethod.CRANK_NICOLSON else 2.0 * 86400.0
            volatility = 0.18 + 0.01 * (position % 5)
            request = PricingRequest(
                option_symbol=f"B{owner}-{position}",
                snapshot_id=f"owner-{owner}",
                put_call="CALL" if position % 2 == 0 else "PUT",
                exercise_type="A",
                spot=100.0 + owner * 0.10,
                strike=88.0 + position * 2.0,
                tenor_seconds=tenor,
                rate=0.035,
                dividend_yield=0.01,
                source_iv=volatility,
                requested_dtype="float64",
            )
            target = registry.get(method).price(request, volatility)
            owner_jobs.append(
                SolverBatchJob(
                    job_id=str(uuid.uuid4()),
                    owner_run_id=f"owner-{owner}",
                    owner_snapshot_id=f"owner-{owner}",
                    output_index=position,
                    operation=SolverOperation.SOLVE_IV_AND_GREEKS,
                    request=request,
                    method=method,
                    method_selection_reason="benchmark",
                    target_price=target,
                    requested_dtype="float64",
                )
            )
        grouped.append(owner_jobs)
    return registry, grouped


def run_solver_batch_benchmark() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    workloads = (
        ("tian_heavy", False, 8, 8),
        ("mixed", True, 8, 8),
        ("small_live", False, 2, 4),
    )
    for workload, mixed, owner_count, jobs_per_owner in workloads:
        context = {
            "options_calcs.greeks.tian.steps": 96,
            "options_calcs.greeks.crank_nicolson.time_grid": 25,
            "options_calcs.greeks.crank_nicolson.space_grid": 50,
            "options_calcs.greeks.crank_nicolson.damping_steps": 2,
        }
        _target_registry, owners = _solver_jobs_by_owner(
            context, owners=owner_count, jobs_per_owner=jobs_per_owner, mixed=mixed
        )
        flattened = [job for owner_jobs in owners for job in owner_jobs]
        baseline_values: Optional[list[Any]] = None
        baseline_seconds = 0.0
        within_seconds = 0.0
        for route in ("scalar_one_owner", "within_snapshot", "cross_snapshot"):
            route_context = dict(context)
            if route == "scalar_one_owner":
                route_context["options_calcs.greeks.batch.enabled"] = False
            registry = SolverRegistry(route_context)
            owner_latencies: list[float] = []
            started = time.perf_counter()
            if route == "cross_snapshot":
                values = registry.solve_iv_and_greeks_many(flattened)
            else:
                values = []
                for owner_jobs in owners:
                    owner_started = time.perf_counter()
                    values.extend(registry.solve_iv_and_greeks_many(owner_jobs))
                    owner_latencies.append(time.perf_counter() - owner_started)
            elapsed = time.perf_counter() - started
            if baseline_values is None:
                baseline_values = values
                baseline_seconds = elapsed
                maximum_difference = 0.0
                mismatch_count = 0
            else:
                mismatch_count = sum(
                    (left.status, left.reason) != (right.status, right.reason)
                    for left, right in zip(baseline_values, values)
                )
                comparable = [
                    abs(float(left.model_price) - float(right.model_price))
                    for left, right in zip(baseline_values, values)
                    if left.model_price is not None and right.model_price is not None
                ]
                maximum_difference = max(comparable, default=0.0)
            if route == "within_snapshot":
                within_seconds = elapsed
            sizes = sorted(registry.batch_sizes)
            owner_latencies = sorted(owner_latencies or [elapsed] * len(owners))
            rows.append(
                {
                    "comparison": "solver_many_contract_jobs",
                    "method": workload,
                    "route": route,
                    "owner_count": len(owners),
                    "logical_job_count": sum(map(len, owners)),
                    "baseline_seconds": baseline_seconds,
                    "runtime_seconds": elapsed,
                    "throughput_per_second": sum(map(len, owners)) / elapsed,
                    "speedup_vs_scalar": baseline_seconds / elapsed,
                    "speedup_vs_within_snapshot": (
                        within_seconds / elapsed if within_seconds else 1.0
                    ),
                    "physical_call_count": (
                        registry.batch_metrics["physical_calls"]
                        or registry.batch_metrics["scalar_internal_calls"]
                    ),
                    "maximum_physical_batch_size": max(registry.batch_sizes, default=0),
                    "mean_batch_size": statistics.mean(sizes) if sizes else 1.0,
                    "p95_batch_size": (
                        sizes[math.ceil(0.95 * len(sizes)) - 1] if sizes else 1
                    ),
                    **(
                        {
                            "owner_p50_seconds": statistics.median(owner_latencies),
                            "owner_p95_seconds": owner_latencies[
                                math.ceil(0.95 * len(owner_latencies)) - 1
                            ],
                        }
                        if route != "cross_snapshot"
                        else {"owner_latency_scope": "not_measured_cross_owner"}
                    ),
                    "peak_memory_estimate_mib": (
                        len(flattened) * 97 * 8 * 12 / (1024.0 * 1024.0)
                    ),
                    "maximum_absolute_difference": maximum_difference,
                    "status_reason_mismatch_count": mismatch_count,
                    "status": "complete" if not mismatch_count else "failed",
                    "price_residual": "",
                    "sample_source": "synthetic_microbenchmark",
                    "measurement_scope": "solver_only_no_clickhouse_or_publication",
                }
            )
    return rows


def run_all_cn_session_benchmark(
    *, runs: int = 3, contracts: int = 4
) -> list[dict[str, Any]]:
    """Expose the all-CN regression hidden by the mostly-Tian mixed workload."""
    if runs <= 0 or contracts <= 0:
        raise ValueError("runs and contracts must be positive")
    from algotrading.refdata.options.solvers import CrankNicolson

    context = {
        "options_calcs.greeks.batch.enabled": True,
        "options_calcs.greeks.crank_nicolson.time_grid": 25,
        "options_calcs.greeks.crank_nicolson.space_grid": 50,
        "options_calcs.greeks.crank_nicolson.damping_steps": 2,
    }
    target_registry = SolverRegistry(context)
    jobs = []
    for index in range(contracts):
        volatility = 0.12 + 0.12 * index
        request = PricingRequest(
            option_symbol=f"CN-{index}",
            snapshot_id="cn-session-benchmark",
            put_call="CALL" if index % 2 == 0 else "PUT",
            exercise_type="A",
            spot=100.0,
            strike=85.0 + index * 10.0,
            tenor_seconds=(15.0 + index * 15.0) * 86400.0,
            rate=0.035,
            dividend_yield=0.01,
            source_iv=volatility,
            requested_dtype="float64",
        )
        target = target_registry.get(RepricingMethod.CRANK_NICOLSON).price(
            request, volatility
        )
        jobs.append(
            SolverBatchJob(
                job_id=f"cn-session-{index}",
                owner_run_id="cn-session-benchmark",
                owner_snapshot_id="cn-session-benchmark",
                output_index=index,
                operation=SolverOperation.SOLVE_IV_AND_GREEKS,
                request=request,
                method=RepricingMethod.CRANK_NICOLSON,
                method_selection_reason="benchmark",
                target_price=target,
                requested_dtype="float64",
                consult_circuit=False,
            )
        )

    original_create = CrankNicolson.create_pricing_session
    session_counter = 0

    def counted_create(**kwargs: Any) -> Any:
        nonlocal session_counter
        session_counter += 1
        return original_create(**kwargs)

    CrankNicolson.create_pricing_session = counted_create
    rows = []
    reference = None
    try:
        for route in ("legacy_generic_batch", "one_session_per_contract"):
            elapsed_values = []
            session_counts = []
            route_results = None
            for _run in range(runs):
                registry = SolverRegistry(context)
                session_counter = 0
                started = time.perf_counter()
                if route == "legacy_generic_batch":
                    results = registry._solve_combined_group(
                        jobs, actual_dtype="float64"
                    )
                else:
                    results = registry.solve_iv_and_greeks_many(jobs)
                elapsed_values.append(time.perf_counter() - started)
                session_counts.append(session_counter)
                route_results = results
            if reference is None:
                reference = route_results
                maximum_difference = 0.0
                mismatch_count = 0
            else:
                mismatch_count = sum(
                    (left.status, left.reason) != (right.status, right.reason)
                    for left, right in zip(reference, route_results)
                )
                differences = [
                    abs(float(getattr(left, name)) - float(getattr(right, name)))
                    for left, right in zip(reference, route_results)
                    for name in (
                        "implied_volatility",
                        "model_price",
                        "delta",
                        "gamma",
                        "theta_per_day",
                        "vega_per_1pct",
                        "rho_per_1pct",
                    )
                    if getattr(left, name) is not None
                    and getattr(right, name) is not None
                ]
                maximum_difference = max(differences, default=0.0)
            ordered_elapsed = sorted(elapsed_values)
            median_seconds = statistics.median(ordered_elapsed)
            rows.append(
                {
                    "comparison": "all_cn_combined_iv_and_greeks",
                    "method": "crank_nicolson",
                    "route": route,
                    "logical_job_count": contracts,
                    "runs": runs,
                    "median_seconds": median_seconds,
                    "p95_seconds": ordered_elapsed[
                        math.ceil(0.95 * len(ordered_elapsed)) - 1
                    ],
                    "throughput_per_second": contracts / median_seconds,
                    "median_session_constructions": statistics.median(
                        session_counts
                    ),
                    "maximum_session_constructions": max(session_counts),
                    "maximum_absolute_difference": maximum_difference,
                    "status_reason_mismatch_count": mismatch_count,
                    "process_peak_rss_mib": (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                    ),
                    "sample_source": "synthetic_all_cn",
                    "measurement_scope": "solver_only_no_clickhouse_or_publication",
                }
            )
    finally:
        CrankNicolson.create_pricing_session = original_create
    return rows


def run_fused_greek_benchmark(
    *, runs: int = 5, steps: int = 256
) -> list[dict[str, Any]]:
    """Measure fused/unfused Tian and CRR Greek stencils and parity."""
    if runs <= 0 or steps <= 0:
        raise ValueError("runs and steps must be positive")
    import torch

    from algotrading.refdata.options.solvers import Crr, Tian

    rows = []
    rng = np.random.default_rng(20260727)
    for count in (1, 10, 20, 40, 80, 160, 320):
        spot = rng.uniform(80.0, 120.0, count)
        common = {
            "S": spot,
            "K": rng.uniform(80.0, 120.0, count),
            "T": rng.uniform(1.0e-6, 1.0, count),
            "r": rng.uniform(-0.02, 0.08, count),
            "q": rng.uniform(0.0, 0.04, count),
            "sigma": rng.uniform(0.08, 1.2, count),
            "is_call": rng.integers(0, 2, count).astype(bool),
            "steps": steps,
            "device": "cpu",
            "chunk_size": 1920,
        }
        for solver_name, function, dtype, default_maximum in (
            ("tian", Tian.greeks_from_iv_batch, np.float64, 0),
            ("crr", Crr.compute_greeks_crr_from_iv, torch.float64, 160),
        ):
            kwargs = {**common, "dtype": dtype}
            outputs = {}
            timings = {}
            peaks = {}
            for fused in (False, True):
                mode = "fused" if fused else "unfused"
                function(
                    **kwargs,
                    fused_greeks_enabled=fused,
                    fused_greeks_maximum_contracts=0,
                )
                elapsed = []
                for _run in range(runs):
                    started = time.perf_counter()
                    outputs[mode] = function(
                        **kwargs,
                        fused_greeks_enabled=fused,
                        fused_greeks_maximum_contracts=0,
                    )
                    elapsed.append(time.perf_counter() - started)
                timings[mode] = sorted(elapsed)
                peaks[mode] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                )
            deltas = {
                name: float(
                    np.nanmax(
                        np.abs(
                            np.asarray(outputs["fused"][name])
                            - np.asarray(outputs["unfused"][name])
                        )
                    )
                )
                for name in outputs["unfused"]
            }
            unfused_median = statistics.median(timings["unfused"])
            fused_median = statistics.median(timings["fused"])
            rows.append(
                {
                    "comparison": "fused_greek_scenarios",
                    "method": solver_name,
                    "contracts": count,
                    "steps": steps,
                    "runs": runs,
                    "unfused_median_seconds": unfused_median,
                    "unfused_p95_seconds": timings["unfused"][
                        math.ceil(0.95 * runs) - 1
                    ],
                    "fused_median_seconds": fused_median,
                    "fused_p95_seconds": timings["fused"][
                        math.ceil(0.95 * runs) - 1
                    ],
                    "unfused_contracts_per_second": count / unfused_median,
                    "fused_contracts_per_second": count / fused_median,
                    "fused_speedup": unfused_median / fused_median,
                    "default_fusion_enabled_for_size": (
                        default_maximum <= 0 or count <= default_maximum
                    ),
                    "unfused_process_peak_rss_mib": peaks["unfused"],
                    "fused_process_peak_rss_mib": peaks["fused"],
                    "maximum_absolute_difference": max(
                        deltas.values(), default=0.0
                    ),
                    "field_maximum_absolute_differences": deltas,
                    "dtype": "float64",
                    "device": "cpu",
                    "sample_source": "synthetic_seed_20260727",
                    "measurement_scope": "greek_kernel_no_iv_or_clickhouse",
                }
            )
    return rows


def run_precision_guard_benchmark() -> list[dict[str, Any]]:
    """Focused finite-float32 versus guarded-float64 validation evidence."""
    context = {
        "options_calcs.greeks.tian.steps": 256,
        "options_calcs.greeks.iv.price_tolerance": 1.0e-4,
        "options_calcs.greeks.batch.float32_guard.enabled": True,
        "options_calcs.greeks.batch.float32_guard.short_tenor_validation_enabled": True,
        "options_calcs.greeks.batch.float32_guard.shadow_fraction": 0.0,
    }
    target_registry = SolverRegistry(context)
    adapter = target_registry.get(RepricingMethod.TIAN)
    jobs: list[SolverBatchJob] = []
    for index, (strike, volatility) in enumerate(
        ((70.0, 0.12), (100.0, 0.20), (105.0, 0.20))
    ):
        request = PricingRequest(
            option_symbol=f"guard-{index}",
            snapshot_id="precision-guard",
            put_call="CALL",
            exercise_type="A",
            spot=100.0,
            strike=strike,
            tenor_seconds=6.0 * 3600.0,
            rate=0.03,
            dividend_yield=0.01,
            source_iv=volatility,
            requested_dtype="float32",
        )
        target = adapter.price(
            replace(request, requested_dtype="float64"), volatility
        )
        jobs.append(
            SolverBatchJob(
                job_id=f"precision-guard-{index}",
                owner_run_id="precision-guard",
                owner_snapshot_id="precision-guard",
                output_index=index,
                operation=SolverOperation.SOLVE_IV_AND_GREEKS,
                request=request,
                method=RepricingMethod.TIAN,
                method_selection_reason="precision_guard_benchmark",
                target_price=target,
                requested_dtype="float32",
            )
        )
    registry = SolverRegistry(context)
    started = time.perf_counter()
    results = registry.solve_iv_and_greeks_many(jobs)
    elapsed = time.perf_counter() - started
    return [
        {
            "comparison": "float32_precision_guard",
            "method": "tian",
            "route": "short_tenor_validation",
            "option_symbol": job.request.option_symbol,
            "runtime_seconds": elapsed / len(jobs),
            "requested_dtype": result.requested_dtype,
            "actual_dtype": result.actual_dtype,
            "precision_escalated": result.precision_escalated,
            "precision_escalation_reason": result.precision_escalation_reason,
            "precision_min_signal_to_roundoff": result.precision_diagnostics.get(
                "precision_min_signal_to_roundoff", ""
            ),
            "precision_invalid_parameter_flags": result.precision_diagnostics.get(
                "precision_invalid_parameter_flags", ""
            ),
            "status": result.status,
            "reason": result.reason,
            "model_price": result.model_price,
            "delta": result.delta,
            "price_residual": result.price_residual,
            "validation_job_count": registry.batch_metrics[
                "float32_guard_validation_jobs"
            ],
            "escalated_job_count": registry.batch_metrics[
                "float32_guard_escalated_jobs"
            ],
            "sample_source": "synthetic_microbenchmark",
            "measurement_scope": "solver_precision_guard_only",
        }
        for job, result in zip(jobs, results)
    ]


def run_self_convergence(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Bounded self-convergence sweep required before cross-method comparison."""
    case = BenchmarkCase(
        "SYNTH",
        "SYNTH-CONVERGENCE-PUT-ATM-30D",
        "PUT",
        30.0 * 86400.0,
        100.0,
        100.0,
        0.03,
        0.01,
        0.25,
    )
    request = _request(case)
    configurations: list[tuple[RepricingMethod, dict[str, Any], str]] = []
    for method in (RepricingMethod.CRR, RepricingMethod.TIAN):
        for steps in (128, 256, 512, 1024):
            configurations.append(
                (
                    method,
                    {f"options_calcs.greeks.{method.value}.steps": steps},
                    f"steps={steps}",
                )
            )
    for damping_steps in (2, 4):
        for time_grid, space_grid in ((50, 100), (100, 200), (200, 400)):
            configurations.append(
                (
                    RepricingMethod.CRANK_NICOLSON,
                    {
                        "options_calcs.greeks.crank_nicolson.time_grid": time_grid,
                        "options_calcs.greeks.crank_nicolson.space_grid": space_grid,
                        "options_calcs.greeks.crank_nicolson.damping_steps": damping_steps,
                    },
                    f"grid={time_grid}x{space_grid};damping={damping_steps}",
                )
            )
    rows: list[dict[str, Any]] = []
    for method, overrides, label in configurations:
        registry = SolverRegistry({**context, **overrides})
        adapter = registry.get(method)
        started = time.perf_counter()
        values = adapter.greeks(request, case.volatility)
        rows.append(
            {
                "comparison": "self_convergence",
                "sample_source": "synthetic",
                "symbol": case.symbol,
                "option_symbol": case.option_symbol,
                "put_call": case.put_call,
                "tenor_seconds": case.tenor_seconds,
                "moneyness": 1.0,
                "method": method.value,
                "status": "complete",
                "failure_reason": "",
                "convergence_setting": label,
                "solver_parameters": json.dumps(adapter.parameters, sort_keys=True),
                "runtime_seconds": time.perf_counter() - started,
                "iv": case.volatility,
                "target_price": "",
                "price_residual": "",
                **values,
            }
        )
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for row in rows:
        parameters = json.loads(str(row["solver_parameters"]))
        key = (
            str(row["method"]),
            parameters.get("damping_steps")
            if row["method"] == RepricingMethod.CRANK_NICOLSON.value
            else None,
        )
        grouped.setdefault(key, []).append(row)
    for values in grouped.values():
        reference = values[-1]
        for row in values:
            for name in ("model_price", "delta", "gamma"):
                row[f"{name}_difference_to_reference"] = abs(
                    float(row[name]) - float(reference[name])
                )
    return rows


def run_node_search_benchmark(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare bounded exact refinement with the isolated 81-point legacy grid."""
    from algotrading.refdata.options.Surface import (
        ExpirationCalibration,
        SmileFit,
        SurfacePoint,
        _fit_iv,
        _routed_american_delta_points,
        european_option_delta,
        generate_calibration_nodes_many,
        generate_surface_nodes,
    )

    source = datetime(2026, 1, 2, tzinfo=timezone.utc)
    tenor_seconds = 30 * 86400.0
    tenor_years = tenor_seconds / (365.25 * 86400.0)
    forward = 100.0 * math.exp((0.03 - 0.01) * tenor_years)
    k_values = np.linspace(-0.25, 0.25, 21)
    variances = np.asarray(
        [(0.25 + 0.08 * abs(float(k))) ** 2 * tenor_years for k in k_values]
    )
    points = []
    for k, variance in zip(k_values, variances):
        volatility = math.sqrt(float(variance) / tenor_years)
        strike = forward * math.exp(float(k))
        for put_call in ("PUT", "CALL"):
            points.append(
                SurfacePoint(
                    f"NODE-{put_call}-{k:.3f}", strike, float(k), volatility,
                    float(variance), 1.0,
                    european_option_delta(
                        spot=100.0, strike=strike, tenor_years=tenor_years,
                        rate=0.03, dividend_yield=0.01,
                        volatility=volatility, is_call=put_call == "CALL",
                        dtype=np.dtype("float64"),
                    ),
                    put_call, 1.0, None, iv_origin="source",
                )
            )
    fit = SmileFit(
        "mid", "pchip", "complete", "",
        {"k": list(map(float, k_values)), "total_variance": list(map(float, variances))},
        tuple(points), len(points), {}, len(points), False,
        tenor_seconds, tenor_years, 100.0, forward, 0.03, 0.01, "A",
        0.0, 0.0, 0, 0, 0,
        source_collected_at=source,
        expiration_at=source + timedelta(seconds=tenor_seconds),
        snapshot_id="node-benchmark",
    )
    benchmark_context = {
        **context,
        "options_calcs.greeks.method": "crr",
        "options_calcs.greeks.short_tenor_method": "crr",
        "options_calcs.surface.delta_nodes": [0.40, 0.25, 0.10],
        "options_calcs.surface.node_search_grid_size": 21,
        "options_calcs.surface.node_exact_max_evaluations": 8,
        "options_calcs.surface.node_dense_exact_fallback_enabled": True,
        "options_calcs.surface.node_dense_exact_fallback_grid_size": 81,
    }
    diagnostics: dict[str, int] = {}
    node_registry = SolverRegistry(benchmark_context)
    started = time.perf_counter()
    new_nodes = generate_surface_nodes(
        fit, context=benchmark_context, dtype=np.dtype("float64"),
        diagnostics=diagnostics, solver_registry=node_registry,
    )
    new_runtime = time.perf_counter() - started
    node_batch_sizes = tuple(node_registry.batch_sizes)
    node_physical_calls = int(node_registry.batch_metrics["physical_calls"])
    new_by_key = {
        (node.node_value, node.put_call): node
        for node in new_nodes
        if node.node_type == "delta"
    }
    cross_owner_count = 4
    cross_owner_registry = SolverRegistry(benchmark_context)
    cross_owner_diagnostics: list[dict[str, int]] = [
        {} for _ in range(cross_owner_count)
    ]
    cross_owner_inputs = [
        ExpirationCalibration(replace(fit, snapshot_id=f"node-owner-{index}"))
        for index in range(cross_owner_count)
    ]
    started = time.perf_counter()
    cross_owner_nodes = generate_calibration_nodes_many(
        cross_owner_inputs,
        context=benchmark_context,
        solver_registry=cross_owner_registry,
        diagnostics=cross_owner_diagnostics,
    )
    cross_owner_runtime = time.perf_counter() - started
    cross_owner_batch_sizes = tuple(cross_owner_registry.batch_sizes)
    cross_owner_exact_evaluations = sum(
        values.get("node_exact_delta_evaluations", 0)
        for values in cross_owner_diagnostics
    )

    started = time.perf_counter()
    dense_grid = np.linspace(fit.coverage[0], fit.coverage[1], 81)
    dense_strikes = fit.forward * np.exp(dense_grid)
    dense_ivs = np.asarray([_fit_iv(fit, float(k)) for k in dense_grid])
    legacy: dict[tuple[float, str], tuple[float, float, float]] = {}
    legacy_evaluations = 0
    for put_call in ("PUT", "CALL"):
        recovered = _routed_american_delta_points(
            fit, dense_strikes, dense_ivs, is_call=put_call == "CALL",
            context=benchmark_context,
        )
        legacy_evaluations += len(dense_grid)
        for delta_value in (0.40, 0.25, 0.10):
            target = delta_value if put_call == "CALL" else -delta_value
            for index in range(len(dense_grid) - 1):
                left = recovered.deltas[index] - target
                right = recovered.deltas[index + 1] - target
                if not (math.isfinite(left) and math.isfinite(right) and left * right <= 0):
                    continue
                fraction = abs(left) / max(abs(left) + abs(right), 1.0e-15)
                k = float(dense_grid[index] + fraction * (dense_grid[index + 1] - dense_grid[index]))
                strike = fit.forward * math.exp(k)
                volatility = float(_fit_iv(fit, k))
                exact = _routed_american_delta_points(
                    fit, np.asarray([strike]), np.asarray([volatility]),
                    is_call=put_call == "CALL", context=benchmark_context,
                )
                legacy_evaluations += 1
                legacy[(delta_value, put_call)] = (strike, volatility, float(exact.deltas[0]))
                break
    legacy_runtime = time.perf_counter() - started
    rows = []
    for key, node in sorted(new_by_key.items()):
        old = legacy.get(key)
        rows.append(
            {
                "comparison": "node_search",
                "sample_source": "synthetic",
                "symbol": "SYNTH",
                "option_symbol": f"NODE-{key[1]}-{key[0]:.2f}",
                "put_call": key[1],
                "tenor_seconds": tenor_seconds,
                "moneyness": (node.fitted_strike or math.nan) / fit.spot,
                "method": "crr",
                "status": node.status,
                "failure_reason": node.node_reason,
                "runtime_seconds": new_runtime,
                "node_runtime_seconds": new_runtime,
                "legacy_node_runtime_seconds": legacy_runtime,
                "exact_delta_evaluations": diagnostics.get("node_exact_delta_evaluations", 0),
                "node_physical_calls": node_physical_calls,
                "node_mean_batch_size": (
                    statistics.mean(node_batch_sizes) if node_batch_sizes else 0.0
                ),
                "node_p95_batch_size": (
                    sorted(node_batch_sizes)[math.ceil(0.95 * len(node_batch_sizes)) - 1]
                    if node_batch_sizes else 0
                ),
                "cross_owner_count": cross_owner_count,
                "cross_owner_node_runtime_seconds": cross_owner_runtime,
                "cross_owner_node_physical_calls": int(
                    cross_owner_registry.batch_metrics["physical_calls"]
                ),
                "cross_owner_node_maximum_batch_size": max(
                    cross_owner_batch_sizes, default=0
                ),
                "cross_owner_node_mean_batch_size": (
                    statistics.mean(cross_owner_batch_sizes)
                    if cross_owner_batch_sizes
                    else 0.0
                ),
                "cross_owner_node_p95_batch_size": (
                    sorted(cross_owner_batch_sizes)[
                        math.ceil(0.95 * len(cross_owner_batch_sizes)) - 1
                    ]
                    if cross_owner_batch_sizes
                    else 0
                ),
                "cross_owner_exact_delta_evaluations": (
                    cross_owner_exact_evaluations
                ),
                "cross_owner_terminal_calibrations": sum(
                    calibration.node_result.status
                    in {"complete", "partial", "unavailable", "failed"}
                    for calibration in cross_owner_nodes
                ),
                "cross_owner_complete_calibrations": sum(
                    calibration.node_result.status == "complete"
                    for calibration in cross_owner_nodes
                ),
                "legacy_exact_delta_evaluations": legacy_evaluations,
                "dense_fallback_frequency": diagnostics.get("node_dense_exact_fallback_count", 0),
                "node_strike_difference": (
                    abs(float(node.fitted_strike) - old[0])
                    if node.fitted_strike is not None and old is not None else ""
                ),
                "node_iv_difference": (
                    abs(float(node.fitted_iv_mid) - old[1])
                    if node.fitted_iv_mid is not None and old is not None else ""
                ),
                "final_delta_residual": abs(
                    float(node.fitted_delta)
                    - (key[0] if key[1] == "CALL" else -key[0])
                ) if node.fitted_delta is not None else "",
                "iv": node.fitted_iv_mid or "",
                "target_price": "",
                "price_residual": "",
                "delta": node.fitted_delta or "",
            }
        )
    return rows


def run_reuse_benchmark(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a bounded origin/reuse/shadow sequence without external data."""
    from algotrading.refdata.options.Reuse import (
        CurrentRepricingProblem,
        PriorRepricingOutcome,
        assess_success_reuse,
        parse_repricing_reuse_policy,
    )
    from algotrading.refdata.options.Calculations import default_context

    case = next(
        case
        for case in synthetic_cases()
        if case.tenor_seconds == 2 * 86400 and case.strike == case.spot
    )
    benchmark_context = default_context("/tmp/algo")
    benchmark_context.update(context)
    benchmark_context.update(
        {
            "options_calcs.greeks.reuse.enabled": True,
            "options_calcs.greeks.reuse.shadow.enabled": True,
            "options_calcs.greeks.reuse.shadow.sample_fraction": 1.0,
        }
    )
    registry = SolverRegistry(benchmark_context)
    adapter = registry.get(RepricingMethod.CRR)
    request = _request(case)
    target = adapter.price(request, case.volatility)
    solver_started = time.perf_counter()
    origin = registry.solve_iv_and_greeks(
        request, method=RepricingMethod.CRR,
        method_selection_reason="benchmark", target_price=target,
    )
    origin_seconds = time.perf_counter() - solver_started
    valuation_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    current_at = valuation_at + timedelta(seconds=1)
    current = CurrentRepricingProblem(
        case.option_symbol, "reuse-n-plus-one", current_at, 5,
        case.spot, target, case.tenor_seconds - 1, case.rate,
        case.dividend_yield, "crr", adapter.algorithm_version,
        adapter.configuration_hash, "market_mid", "continuous_yield",
    )
    prior = PriorRepricingOutcome(
        case.option_symbol, "reuse-origin", valuation_at, 5, valuation_at,
        "reuse-origin", valuation_at, valuation_at, case.spot, target,
        case.tenor_seconds, case.rate, case.dividend_yield, "crr", "benchmark",
        adapter.algorithm_version, adapter.configuration_hash,
        json.dumps(adapter.parameters, sort_keys=True), "market_mid", case.volatility,
        float(origin.implied_volatility), origin.iv_iterations, origin.price_residual,
        origin.model_price, float(origin.delta), float(origin.gamma),
        float(origin.theta_per_day), float(origin.vega_per_1pct),
        float(origin.rho_per_1pct), "float64", "float64", False, 0,
        adapter.parameters.get("steps"), None, None, "continuous_yield",
    )
    assessment_started = time.perf_counter()
    decision = assess_success_reuse(
        current, prior, parse_repricing_reuse_policy(benchmark_context)
    )
    assessment_seconds = time.perf_counter() - assessment_started
    shadow_started = time.perf_counter()
    shadow = registry.solve_iv_and_greeks(
        request, method=RepricingMethod.CRR,
        method_selection_reason="benchmark_shadow", target_price=target,
    )
    shadow_seconds = time.perf_counter() - shadow_started
    return [
        {
            "comparison": "reuse_shadow",
            "sample_source": "synthetic",
            "symbol": case.symbol,
            "option_symbol": case.option_symbol,
            "put_call": case.put_call,
            "tenor_seconds": case.tenor_seconds,
            "moneyness": case.strike / case.spot,
            "method": "crr",
            "status": "complete" if decision.reusable else "failed",
            "failure_reason": "" if decision.reusable else decision.reason,
            "runtime_seconds": origin_seconds + shadow_seconds,
            "selected_contracts": 1,
            "solver_attempts": 2,
            "reusable_successes": int(decision.reusable),
            "shadow_attempts": 1,
            "absolute_iv_error": abs(
                float(shadow.implied_volatility) - prior.implied_volatility
            ),
            "absolute_delta_error": abs(float(shadow.delta) - prior.delta),
            "contract_solver_seconds": origin_seconds + shadow_seconds,
            "prior_lookup_seconds": 0.0,
            "reuse_assessment_seconds": assessment_seconds,
            "iv": shadow.implied_volatility,
            "target_price": target,
            "price_residual": shadow.price_residual,
            "model_price": shadow.model_price,
            "delta": shadow.delta,
        }
    ]


def _worker_scaling_task(
    payload: tuple[dict[str, Any], BenchmarkCase, int]
) -> tuple[int, float, float, int, int, str, str, tuple[float, ...]]:
    context, case, task_index = payload
    started_cpu = time.process_time()
    started = time.perf_counter()
    registry = SolverRegistry(context)
    request = _request(case)
    method = (
        RepricingMethod.TIAN
        if case.tenor_seconds < 86400
        else RepricingMethod.CRANK_NICOLSON
    )
    adapter = registry.get(method)
    target = adapter.price(request, case.volatility)
    solved = registry.solve_iv_and_greeks(
        request,
        method=method,
        method_selection_reason="benchmark_route",
        target_price=target,
    )
    if solved.status != "complete":
        raise RuntimeError(f"worker scaling solve failed: {solved.reason}")
    output = tuple(
        float(getattr(solved, name))
        for name in (
            "implied_volatility",
            "model_price",
            "delta",
            "gamma",
            "theta_per_day",
            "vega_per_1pct",
            "rho_per_1pct",
        )
    )
    return (
        os.getpid(),
        time.perf_counter() - started,
        time.process_time() - started_cpu,
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        task_index,
        case.option_symbol,
        method.value,
        output,
    )


def run_worker_scaling(
    context: dict[str, Any],
    *,
    worker_counts: Iterable[int] = (1, 2, 4, 8),
) -> list[dict[str, Any]]:
    """Bounded spawned-process throughput and process-peak memory sample."""
    base_cases = tuple(
        case
        for case in synthetic_cases()
        if case.strike == 100.0 and case.put_call == "CALL"
    )
    tasks = [base_cases[index % len(base_cases)] for index in range(16)]
    counts = tuple(int(value) for value in worker_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("worker counts must be positive")
    if 1 not in counts:
        raise ValueError("worker scaling requires a one-worker parity baseline")
    rows = []
    one_worker_outputs: Optional[list[tuple[int, str, str, tuple[float, ...]]]] = None
    for workers in counts:
        started = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            results = list(
                pool.map(
                    _worker_scaling_task,
                    ((context, case, index) for index, case in enumerate(tasks)),
                    chunksize=1,
                )
            )
        wall = time.perf_counter() - started
        peak_by_process: dict[int, int] = {}
        for pid, _elapsed, _cpu, peak_rss_kib, *_output in results:
            peak_by_process[pid] = max(peak_by_process.get(pid, 0), peak_rss_kib)
        outputs = [
            (task_index, symbol, method, vector)
            for _pid, _elapsed, _cpu, _rss, task_index, symbol, method, vector in results
        ]
        if workers == 1:
            one_worker_outputs = outputs
        if one_worker_outputs is None:
            raise ValueError("one-worker count must precede other worker counts")
        maximum_difference = 0.0
        parity = len(outputs) == len(one_worker_outputs)
        for baseline, candidate in zip(one_worker_outputs, outputs):
            parity = parity and baseline[:3] == candidate[:3]
            for expected, actual in zip(baseline[3], candidate[3]):
                maximum_difference = max(maximum_difference, abs(expected - actual))
                parity = parity and math.isclose(
                    expected, actual, rel_tol=1.0e-10, abs_tol=1.0e-12
                )
        fingerprint = hashlib.sha256(
            json.dumps(outputs, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "comparison": "worker_scaling",
                "method": "routed_tian_crank_nicolson",
                "worker_count": int(workers),
                "used_process_count": len(peak_by_process),
                "task_count": len(tasks),
                "runtime_seconds": wall,
                "throughput_per_second": len(tasks) / wall,
                "cpu_seconds": sum(value[2] for value in results),
                "summed_process_peak_rss_mib": sum(peak_by_process.values()) / 1024.0,
                "maximum_process_peak_rss_mib": max(peak_by_process.values()) / 1024.0,
                "worker_restart_count": 0,
                "parity_with_one_worker": parity,
                "max_abs_output_difference": maximum_difference,
                "output_sha256": fingerprint,
            }
        )
    return rows


def write_report(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "options_solver_comparison.csv"
    fieldnames = sorted({name for row in rows for name in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    solver_rows = [row for row in rows if row.get("comparison") != "worker_scaling"]
    runtimes = sorted(float(row["runtime_seconds"]) for row in solver_rows)
    markdown_path = output_dir / "options_solver_comparison.md"
    def quantile(values: list[float], fraction: float) -> float:
        return values[min(len(values) - 1, int((len(values) - 1) * fraction))]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in solver_rows:
        grouped.setdefault((str(row["comparison"]), str(row["method"])), []).append(row)
    sample_sources = sorted(
        {str(row.get("sample_source", "synthetic")) for row in solver_rows}
    )
    lines = [
        "# Options solver comparison",
        "",
        f"Bounded {'/'.join(sample_sources)} sample; no production ClickHouse history was rebuilt.",
        "",
        f"Solver rows: {len(solver_rows)}",
        "",
        f"Runtime median: {statistics.median(runtimes):.6f}s",
        f"Runtime p90: {quantile(runtimes, 0.90):.6f}s",
        f"Runtime p95: {quantile(runtimes, 0.95):.6f}s",
        f"Runtime p99: {quantile(runtimes, 0.99):.6f}s",
        f"Runtime maximum: {max(runtimes):.6f}s",
        "",
        "## Runtime and convergence by method",
        "",
        "| Comparison | Method | Rows | Complete | Median s | P95 s | Max abs price residual |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (comparison, method), values in sorted(grouped.items()):
        durations = sorted(float(row["runtime_seconds"]) for row in values)
        p95 = durations[min(len(durations) - 1, int((len(durations) - 1) * 0.95))]
        residuals = [
            abs(float(row["price_residual"]))
            for row in values
            if row.get("price_residual") not in (None, "")
        ]
        complete = sum(
            row.get("model_price") not in (None, "")
            and row.get("delta") not in (None, "")
            for row in values
        )
        lines.append(
            f"| {comparison} | {method} | {len(values)} | {complete} | "
            f"{statistics.median(durations):.6f} | {p95:.6f} | "
            f"{max(residuals, default=0.0):.3e} |"
        )
    market_summary_rows = [
        row for row in solver_rows if row.get("comparison") == "same_market_target"
    ]
    if market_summary_rows:
        lines.extend(
            [
                "",
                "## Same-market solver performance",
                "",
                "| Method | Rows | Median total s | P95 total s | Mean IV price evaluations | Mean FDM valuations | Median IV s | Median Greeks s |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in sorted({str(row["method"]) for row in market_summary_rows}):
            values = [
                row for row in market_summary_rows if str(row["method"]) == method
            ]
            totals = sorted(
                float(row.get("solver_total_seconds", row["runtime_seconds"]))
                for row in values
            )
            iv_times = [float(row.get("iv_seconds", 0.0)) for row in values]
            greek_times = [float(row.get("greeks_seconds", 0.0)) for row in values]
            iv_evaluations = [
                int(row.get("iv_price_evaluations", 0)) for row in values
            ]
            fdm_valuations = [
                int(row.get("fdm_valuations_total", 0)) for row in values
            ]
            lines.append(
                f"| {method} | {len(values)} | {statistics.median(totals):.6f} | "
                f"{quantile(totals, 0.95):.6f} | {statistics.mean(iv_evaluations):.2f} | "
                f"{statistics.mean(fdm_valuations):.2f} | "
                f"{statistics.median(iv_times):.6f} | {statistics.median(greek_times):.6f} |"
            )
    failures: dict[tuple[str, str], int] = {}
    for row in solver_rows:
        if row.get("status") == "complete":
            continue
        key = (str(row.get("method", "")), str(row.get("failure_reason", "unknown")))
        failures[key] = failures.get(key, 0) + 1
    if failures:
        lines.extend(["", "Market-target convergence failures:", ""])
        for (method, reason), count in sorted(failures.items()):
            lines.append(f"- `{method}` / `{reason}`: {count}")
    lines.extend(
        [
            "",
            "## Same-IV differences versus Tian",
            "",
            "| Method | Matched cases | Max price | Max delta | Max gamma | Max theta/day | Max vega/1% | Max rho/1% |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    supplied = [row for row in solver_rows if row.get("comparison") == "same_supplied_iv"]
    by_case = {(row["option_symbol"], row["method"]): row for row in supplied}
    numeric_names = (
        "model_price", "delta", "gamma", "theta_per_day",
        "vega_per_1pct", "rho_per_1pct",
    )
    for method in sorted({str(row["method"]) for row in supplied} - {"tian"}):
        differences = {name: [] for name in numeric_names}
        matched = 0
        for (option_symbol, row_method), row in by_case.items():
            if row_method != method or (option_symbol, "tian") not in by_case:
                continue
            baseline = by_case[(option_symbol, "tian")]
            matched += 1
            for name in numeric_names:
                differences[name].append(abs(float(row[name]) - float(baseline[name])))
        lines.append(
            f"| {method} | {matched} | "
            + " | ".join(f"{max(differences[name], default=0.0):.3e}" for name in numeric_names)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Same-IV difference distributions versus Tian",
            "",
            "Gamma and Theta are retained as convergence diagnostics rather than global pass/fail thresholds near exercise boundaries.",
            "",
            "| Method | Metric | N | Median | P90 | P95 | P99 | Max | Max normalized | Sign disagreements | Outside review tolerance |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    review_tolerances = {"model_price": 0.05, "delta": 0.03}
    for method in sorted({str(row["method"]) for row in supplied} - {"tian"}):
        for name in numeric_names:
            differences = []
            normalized = []
            sign_disagreements = 0
            for (option_symbol, row_method), row in by_case.items():
                if row_method != method or (option_symbol, "tian") not in by_case:
                    continue
                baseline = by_case[(option_symbol, "tian")]
                actual = float(row[name])
                expected = float(baseline[name])
                if not (math.isfinite(actual) and math.isfinite(expected)):
                    continue
                difference = abs(actual - expected)
                differences.append(difference)
                normalized.append(difference / max(abs(expected), 1.0e-12))
                sign_disagreements += int(
                    actual != 0.0
                    and expected != 0.0
                    and math.copysign(1.0, actual) != math.copysign(1.0, expected)
                )
            differences.sort()
            normalized.sort()
            if not differences:
                continue
            tolerance = review_tolerances.get(name)
            outside = (
                sum(value > tolerance for value in differences)
                if tolerance is not None
                else 0
            )
            lines.append(
                f"| {method} | {name} | {len(differences)} | "
                f"{statistics.median(differences):.3e} | "
                f"{quantile(differences, 0.90):.3e} | {quantile(differences, 0.95):.3e} | "
                f"{quantile(differences, 0.99):.3e} | {max(differences):.3e} | "
                f"{max(normalized):.3e} | {sign_disagreements} | {outside} |"
            )
    market_rows = [
        row
        for row in solver_rows
        if row.get("comparison") == "same_market_target"
        and row.get("status") == "complete"
        and row.get("iv") not in (None, "")
    ]
    market_by_case = {
        (row["option_symbol"], row["method"]): row for row in market_rows
    }
    lines.extend(
        [
            "",
            "## Market-derived IV differences versus Tian",
            "",
            "| Method | N | Median | P95 | Max | Outside 0.02 review tolerance |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in sorted({str(row["method"]) for row in market_rows} - {"tian"}):
        differences = sorted(
            abs(float(row["iv"]) - float(market_by_case[(option_symbol, "tian")]["iv"]))
            for (option_symbol, row_method), row in market_by_case.items()
            if row_method == method and (option_symbol, "tian") in market_by_case
        )
        if differences:
            lines.append(
                f"| {method} | {len(differences)} | {statistics.median(differences):.3e} | "
                f"{quantile(differences, 0.95):.3e} | {max(differences):.3e} | "
                f"{sum(value > 0.02 for value in differences)} |"
            )
    convergence_rows = [
        row for row in solver_rows if row.get("comparison") == "self_convergence"
    ]
    if convergence_rows:
        lines.extend(
            [
                "",
                "## Self-convergence",
                "",
                "Differences are absolute versus the highest-resolution result for the same method and damping setting.",
                "",
                "| Method | Setting | Runtime s | Price diff | Delta diff | Gamma diff |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in convergence_rows:
            lines.append(
                f"| {row['method']} | {row['convergence_setting']} | "
                f"{float(row['runtime_seconds']):.6f} | "
                f"{float(row['model_price_difference_to_reference']):.3e} | "
                f"{float(row['delta_difference_to_reference']):.3e} | "
                f"{float(row['gamma_difference_to_reference']):.3e} |"
            )
    scaling_rows = [row for row in rows if row.get("comparison") == "worker_scaling"]
    if scaling_rows:
        lines.extend(
            [
                "",
                "## Spawned worker scaling",
                "",
                "Sixteen bounded routed tasks; memory is the sum of per-process peak RSS, not steady-state resident memory.",
                "",
                "| Workers | Used processes | Wall s | Tasks/s | CPU s | Max worker peak MiB | Parity | Max output diff |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
            ]
        )
        for row in sorted(scaling_rows, key=lambda value: int(value["worker_count"])):
            lines.append(
                f"| {int(row['worker_count'])} | {int(row['used_process_count'])} | "
                f"{float(row['runtime_seconds']):.6f} | {float(row['throughput_per_second']):.3f} | "
                f"{float(row['cpu_seconds']):.6f} | {float(row['maximum_process_peak_rss_mib']):.1f} | "
                f"{'yes' if row['parity_with_one_worker'] else 'NO'} | "
                f"{float(row['max_abs_output_difference']):.3e} |"
            )
    lines.extend(
        [
            "",
            "## Solver parameters",
            "",
        ]
    )
    for method in sorted({str(row["method"]) for row in supplied}):
        parameters = next(row["solver_parameters"] for row in supplied if row["method"] == method)
        lines.append(f"- `{method}`: `{parameters}`")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tests/benchmark_output"))
    parser.add_argument("--source", choices=("synthetic", "clickhouse"), default="synthetic")
    parser.add_argument("-e", "--environment")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--from-date")
    parser.add_argument("--to-date")
    parser.add_argument("--symbols")
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--worker-scaling", action="store_true")
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--node-benchmark", action="store_true")
    parser.add_argument(
        "--surface-node-ab",
        action="store_true",
        help="run a read-only legacy/hybrid A/B against completed ClickHouse fits",
    )
    parser.add_argument("--reuse-benchmark", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=(
            "tian-kernel",
            "solver-many",
            "cn-session",
            "fused-greeks",
            "precision-guard",
        ),
        help="run one reproducible Phase 1-3 benchmark scenario",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.sample_limit <= 0 or args.sample_limit > 1000:
        raise ValueError("sample-limit must be between 1 and 1000")
    if args.scenario:
        rows = {
            "tian-kernel": run_tian_kernel_benchmark,
            "solver-many": run_solver_batch_benchmark,
            "cn-session": run_all_cn_session_benchmark,
            "fused-greeks": run_fused_greek_benchmark,
            "precision-guard": run_precision_guard_benchmark,
        }[args.scenario]()
        csv_path, markdown_path = write_report(rows, args.output_dir / args.scenario)
        summary_path = args.output_dir / args.scenario / "summary.json"
        summary_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(csv_path)
        print(markdown_path)
        print(summary_path)
        return 0
    selectors = (args.snapshot_id, args.from_date, args.to_date, args.symbols)
    use_clickhouse = (
        args.source == "clickhouse" or any(selectors) or args.surface_node_ab
    )
    context: dict[str, Any] = {}
    if use_clickhouse:
        if not args.environment:
            raise ValueError("-e/--environment is required for a ClickHouse sample")
        if args.from_date and args.to_date and args.from_date > args.to_date:
            raise ValueError("--from-date cannot be after --to-date")
        algo_home = os.getenv("ALGO_HOME")
        if not algo_home:
            raise ValueError("ALGO_HOME is required for a ClickHouse sample")
        from algotrading.Common import load_configuration
        from algotrading.refdata.options.Calculations import default_context
        from algotrading.refdata.options.Common import create_clickhouse_client

        algo_home = algo_home.rstrip("/") + "/"
        context = load_configuration(
            default_context(algo_home),
            algo_home,
            args.environment,
            quiet=True,
        )
        client = create_clickhouse_client(context)
        try:
            if args.surface_node_ab:
                surface_fits = load_clickhouse_surface_fits(
                    client,
                    str(context["options_chain.clickhouse.database"]),
                    sample_limit=args.sample_limit,
                    snapshot_id=args.snapshot_id,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    symbols=(args.symbols or "").split(","),
                )
            else:
                cases = load_clickhouse_cases(
                    client,
                    str(context["options_chain.clickhouse.database"]),
                    sample_limit=args.sample_limit,
                    snapshot_id=args.snapshot_id,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    symbols=(args.symbols or "").split(","),
                )
        finally:
            client.close()
        if args.surface_node_ab:
            if not surface_fits:
                raise ValueError(
                    "bounded ClickHouse filters returned no completed surface fits"
                )
            rows = run_surface_node_ab_benchmark(context, surface_fits)
            csv_path, markdown_path, summary_path = write_surface_node_ab_report(
                rows, args.output_dir / "surface-node-ab"
            )
            print(csv_path)
            print(markdown_path)
            print(summary_path)
            return 0
        if not cases:
            raise ValueError("bounded ClickHouse filters returned no eligible contracts")
        rows = run_synthetic(context, cases)
    else:
        maximum_cases = max(1, (args.sample_limit + 3) // 4)
        rows = run_synthetic(context, synthetic_cases()[:maximum_cases])[: args.sample_limit]
    if args.worker_scaling:
        rows.extend(run_worker_scaling(context))
    if args.convergence:
        rows.extend(run_self_convergence(context))
    if args.node_benchmark:
        rows.extend(run_node_search_benchmark(context))
    if args.reuse_benchmark:
        rows.extend(run_reuse_benchmark(context))
    csv_path, markdown_path = write_report(rows, args.output_dir)
    print(csv_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
