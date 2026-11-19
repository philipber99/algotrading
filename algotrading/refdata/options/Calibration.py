"""Bounded, advisory reuse-policy calibration with deterministic holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Any, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class CalibrationSlo:
    maximum_p99_iv_error: float
    maximum_p99_delta_error: float
    maximum_max_iv_error: float
    minimum_holdout_observations: int

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "CalibrationSlo":
        prefix = "options_calcs.greeks.reuse.calibration."
        value = cls(
            float(context.get(prefix + "maximum_p99_iv_error", 0.0005)),
            float(context.get(prefix + "maximum_p99_delta_error", 0.0025)),
            float(context.get(prefix + "maximum_max_iv_error", 0.0020)),
            int(context.get(prefix + "minimum_holdout_observations", 100)),
        )
        if (
            any(
                not math.isfinite(item) or item < 0
                for item in (
                    value.maximum_p99_iv_error,
                    value.maximum_p99_delta_error,
                    value.maximum_max_iv_error,
                )
            )
            or value.minimum_holdout_observations <= 0
        ):
            raise ValueError("invalid reuse calibration SLO configuration")
        return value


def deterministic_calibration_split(
    row: Mapping[str, Any], *, holdout_fraction: float = 0.20
) -> str:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be strictly between zero and one")
    payload = (
        f"{row.get('snapshot_id', '')}\0{row.get('option_symbol', '')}\0"
        f"{row.get('validation_type', '')}"
    ).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return "holdout" if value < int(holdout_fraction * (1 << 64)) else "calibration"


def _finite(row: Mapping[str, Any], key: str) -> Optional[float]:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _quantile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1)]


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    iv = [value for row in rows if (value := _finite(row, "absolute_iv_error")) is not None]
    delta = [value for row in rows if (value := _finite(row, "absolute_delta_error")) is not None]
    methods: dict[str, int] = {}
    put_call: dict[str, int] = {}
    moneyness = []
    for row in rows:
        method = str(row.get("pricing_method") or "none")
        methods[method] = methods.get(method, 0) + 1
        side = str(row.get("put_call") or "none")
        put_call[side] = put_call.get(side, 0) + 1
        value = _finite(row, "log_forward_moneyness")
        if value is not None:
            moneyness.append(value)
    return {
        "sample_count": len(rows),
        "iv_error": {
            "p50": _quantile(iv, 0.50), "p95": _quantile(iv, 0.95),
            "p99": _quantile(iv, 0.99), "max": max(iv) if iv else None,
        },
        "delta_error": {
            "p50": _quantile(delta, 0.50), "p95": _quantile(delta, 0.95),
            "p99": _quantile(delta, 0.99), "max": max(delta) if delta else None,
        },
        "method_composition": dict(sorted(methods.items())),
        "put_call_composition": dict(sorted(put_call.items())),
        "moneyness_coverage": (
            {"minimum": min(moneyness), "maximum": max(moneyness)}
            if moneyness else None
        ),
    }


def _passes(metrics: Mapping[str, Any], slo: CalibrationSlo) -> bool:
    iv = metrics["iv_error"]
    delta = metrics["delta_error"]
    return (
        iv["p99"] is not None
        and delta["p99"] is not None
        and iv["max"] is not None
        and iv["p99"] <= slo.maximum_p99_iv_error
        and delta["p99"] <= slo.maximum_p99_delta_error
        and iv["max"] <= slo.maximum_max_iv_error
    )


_GUARDS = (
    "candidate_age_seconds",
    "spot_move_bps",
    "elapsed_tenor_fraction",
    "estimated_price_drift",
    "estimated_iv_change_points",
)


def _eligible(row: Mapping[str, Any], thresholds: Mapping[str, float]) -> bool:
    for key, maximum in thresholds.items():
        value = _finite(row, key)
        if value is None or abs(value) > maximum:
            return False
    return True


def _candidate_thresholds(
    rows: Sequence[Mapping[str, Any]], *, maximum_combinations: int = 4096
) -> Iterable[dict[str, float]]:
    """Build a bounded independent grid rather than row-coincident vectors."""
    if maximum_combinations <= 0:
        raise ValueError("maximum candidate combinations must be positive")
    maximum_values_per_guard = max(
        1, int(maximum_combinations ** (1.0 / len(_GUARDS)))
    )
    values_by_guard: list[list[float]] = []
    for key in _GUARDS:
        values = sorted(
            {
                abs(value)
                for row in rows
                if (value := _finite(row, key)) is not None
            }
        )
        if not values:
            return ()
        if len(values) > maximum_values_per_guard:
            last = len(values) - 1
            indices = sorted(
                {
                    round(index * last / (maximum_values_per_guard - 1))
                    if maximum_values_per_guard > 1
                    else last
                    for index in range(maximum_values_per_guard)
                }
            )
            values = [values[index] for index in indices]
        values_by_guard.append(values)
    candidates = [
        dict(zip(_GUARDS, combination))
        for combination in product(*values_by_guard)
    ]
    if len(candidates) > maximum_combinations:
        candidates = candidates[:maximum_combinations]
    return sorted(
        candidates,
        key=lambda item: (
            -sum(_eligible(row, item) for row in rows),
            tuple(item[key] for key in _GUARDS),
        ),
    )


def _calibration_candidate(
    rows: Sequence[Mapping[str, Any]],
    slo: CalibrationSlo,
    *,
    maximum_combinations: int,
) -> Optional[dict[str, Any]]:
    for thresholds in _candidate_thresholds(
        rows, maximum_combinations=maximum_combinations
    ):
        eligible = [row for row in rows if _eligible(row, thresholds)]
        metrics = _metrics(eligible)
        if _passes(metrics, slo):
            return {
                "thresholds": thresholds,
                "calibration": metrics,
                "eligible_count": len(eligible),
                "estimated_reuse_coverage": (
                    len(eligible) / len(rows) if rows else 0.0
                ),
            }
    return None


def build_reuse_calibration_report(
    observations: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    minimum_observations: int = 200,
    holdout_fraction: float = 0.20,
    policy_definition_id: int = 0,
) -> dict[str, Any]:
    """Propose broad guards on calibration rows and independently score holdout."""
    if minimum_observations <= 0:
        raise ValueError("minimum_observations must be positive")
    ordered = sorted(
        (dict(row) for row in observations),
        key=lambda row: (
            str(row.get("source_collected_at", "")),
            str(row.get("snapshot_id", "")),
            str(row.get("option_symbol", "")),
        ),
    )
    slo = CalibrationSlo.from_context(context)
    maximum_combinations = int(
        context.get(
            "options_calcs.greeks.reuse.calibration.maximum_candidate_combinations",
            4096,
        )
    )
    if maximum_combinations <= 0 or maximum_combinations > 100_000:
        raise ValueError(
            "reuse calibration maximum_candidate_combinations must be in [1, 100000]"
        )
    calibration = [row for row in ordered if deterministic_calibration_split(row, holdout_fraction=holdout_fraction) == "calibration"]
    holdout = [row for row in ordered if deterministic_calibration_split(row, holdout_fraction=holdout_fraction) == "holdout"]
    recommendation = None
    rejected_candidate = None
    calibration_metrics = _metrics(calibration)
    holdout_metrics = _metrics(holdout)
    if len(ordered) < minimum_observations:
        rejected_candidate = {
            "thresholds": None,
            "calibration": calibration_metrics,
            "holdout": holdout_metrics,
            "rejection_reason": "insufficient_total_observations",
        }
    else:
        candidate = _calibration_candidate(
            calibration, slo, maximum_combinations=maximum_combinations
        )
        if candidate is None:
            rejected_candidate = {
                "thresholds": None,
                "calibration": calibration_metrics,
                "holdout": holdout_metrics,
                "rejection_reason": "no_candidate_satisfied_slo",
            }
        else:
            thresholds = candidate["thresholds"]
            holdout_eligible = [
                row for row in holdout if _eligible(row, thresholds)
            ]
            scored = _metrics(holdout_eligible)
            sufficient = len(holdout_eligible) >= slo.minimum_holdout_observations
            holdout_pass = sufficient and _passes(scored, slo)
            assessed = {
                **candidate,
                "holdout": scored,
                "holdout_sufficient": sufficient,
                "holdout_slo_pass": holdout_pass,
            }
            if holdout_pass:
                recommendation = assessed
            else:
                rejected_candidate = {
                    **assessed,
                    "rejection_reason": (
                        "insufficient_holdout_observations"
                        if not sufficient
                        else "holdout_slo_failed"
                    ),
                }
    from algotrading.refdata.options.Reuse import parse_repricing_reuse_policy

    reuse_policy = parse_repricing_reuse_policy(context)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in ordered:
        tenor = _finite(row, "time_to_expiry_seconds")
        if tenor is None:
            continue
        band = reuse_policy.band_for(tenor)
        band_name = (
            "catch_all"
            if band.max_tenor_seconds is None
            else f"le_{int(band.max_tenor_seconds)}s"
        )
        key = f"{str(row.get('pricing_method') or 'none')}|{band_name}"
        grouped.setdefault(key, []).append(row)
    group_reports = {}
    for key, group_rows in sorted(grouped.items()):
        group_calibration = [row for row in group_rows if deterministic_calibration_split(row, holdout_fraction=holdout_fraction) == "calibration"]
        group_holdout = [row for row in group_rows if deterministic_calibration_split(row, holdout_fraction=holdout_fraction) == "holdout"]
        group_recommendation = None
        group_rejected = None
        if len(group_rows) >= minimum_observations:
            candidate = _calibration_candidate(
                group_calibration,
                slo,
                maximum_combinations=maximum_combinations,
            )
            if candidate is not None:
                thresholds = candidate["thresholds"]
                holdout_eligible = [row for row in group_holdout if _eligible(row, thresholds)]
                holdout_scored = _metrics(holdout_eligible)
                sufficient = len(holdout_eligible) >= slo.minimum_holdout_observations
                holdout_pass = sufficient and _passes(holdout_scored, slo)
                if holdout_pass:
                    group_recommendation = {
                        **candidate,
                        "holdout": holdout_scored,
                        "holdout_sufficient": sufficient,
                        "holdout_slo_pass": True,
                    }
                else:
                    group_rejected = {
                        **candidate,
                        "holdout": holdout_scored,
                        "holdout_sufficient": sufficient,
                        "holdout_slo_pass": False,
                        "rejection_reason": (
                            "insufficient_holdout_observations"
                            if not sufficient
                            else "holdout_slo_failed"
                        ),
                    }
            else:
                group_rejected = {
                    "rejection_reason": "no_candidate_satisfied_slo"
                }
        else:
            group_rejected = {
                "rejection_reason": "insufficient_total_observations"
            }
        group_reports[key] = {
            "sample_count": len(group_rows),
            "calibration_count": len(group_calibration),
            "holdout_count": len(group_holdout),
            "recommendation": group_recommendation,
            "rejected_candidate": group_rejected,
        }
    dates = [str(row.get("source_collected_at")) for row in ordered if row.get("source_collected_at") is not None]
    return {
        "policy_definition_id": int(policy_definition_id),
        "observation_count": len(ordered),
        "observation_date_range": {
            "from": min(dates) if dates else None,
            "to": max(dates) if dates else None,
        },
        "split": {
            "holdout_fraction": holdout_fraction,
            "calibration_count": len(calibration),
            "holdout_count": len(holdout),
        },
        "slo": {
            "maximum_p99_iv_error": slo.maximum_p99_iv_error,
            "maximum_p99_delta_error": slo.maximum_p99_delta_error,
            "maximum_max_iv_error": slo.maximum_max_iv_error,
            "minimum_holdout_observations": slo.minimum_holdout_observations,
        },
        "all_calibration_metrics": calibration_metrics,
        "all_holdout_metrics": holdout_metrics,
        "recommendation": recommendation,
        "rejected_candidate": rejected_candidate,
        "groups": group_reports,
        "proposed_configuration": (
            {
                "policy_definition_id": int(policy_definition_id),
                "observation_date_range": {
                    "from": min(dates) if dates else None,
                    "to": max(dates) if dates else None,
                },
                "calibration_count": len(calibration),
                "holdout_count": recommendation["holdout"]["sample_count"],
                "tenor_method_group": "all",
                "guards": dict(recommendation["thresholds"]),
                "holdout_iv_error": recommendation["holdout"]["iv_error"],
                "holdout_delta_error": recommendation["holdout"]["delta_error"],
                "holdout_slo_pass": True,
            }
            if recommendation is not None else None
        ),
        "configuration_mutated": False,
    }


def load_bounded_observations(
    client: Any,
    database: str,
    *,
    row_limit: int,
    from_time: Optional[datetime] = None,
    to_time: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    if row_limit <= 0 or row_limit > 100_000:
        raise ValueError("row_limit must be between 1 and 100000")
    conditions = ["validation_type = 'success_reuse_shadow'"]
    parameters: dict[str, Any] = {"limit": row_limit}
    if from_time is not None:
        conditions.append("source_collected_at >= {from_time:DateTime64(6, 'UTC')}")
        parameters["from_time"] = from_time
    if to_time is not None:
        conditions.append("source_collected_at < {to_time:DateTime64(6, 'UTC')}")
        parameters["to_time"] = to_time
    table = client.query_arrow(
        f"SELECT * FROM {database}.options_chain_reuse_validation "
        f"WHERE {' AND '.join(conditions)} "
        "ORDER BY source_collected_at DESC, snapshot_id, option_symbol "
        "LIMIT {limit:UInt32}",
        parameters=parameters,
    )
    return table.to_pylist()


def report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2, default=str)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--environment", required=True)
    parser.add_argument("--row-limit", type=int, default=10_000)
    parser.add_argument("--minimum-observations", type=int, default=200)
    parser.add_argument("--from-time", type=datetime.fromisoformat)
    parser.add_argument("--to-time", type=datetime.fromisoformat)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    algo_home = os.getenv("ALGO_HOME")
    if not algo_home:
        raise ValueError("ALGO_HOME is required")
    from algotrading.Common import load_configuration
    from algotrading.refdata.options.Calculations import default_context
    from algotrading.refdata.options.Common import create_clickhouse_client

    home = algo_home.rstrip("/") + "/"
    context = load_configuration(default_context(home), home, args.environment)
    client = create_clickhouse_client(context)
    try:
        observations = load_bounded_observations(
            client,
            str(context["options_chain.clickhouse.database"]),
            row_limit=args.row_limit,
            from_time=args.from_time,
            to_time=args.to_time,
        )
    finally:
        client.close()
    report = build_reuse_calibration_report(
        observations,
        context=context,
        minimum_observations=args.minimum_observations,
        holdout_fraction=args.holdout_fraction,
        policy_definition_id=int(
            context.get("options_calcs.calculation_definition_id", 0)
        ),
    )
    print(report_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
