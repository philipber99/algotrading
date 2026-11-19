"""Average-linkage clustering over signed scenario P&L vectors."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


def cluster_risk_units(
    scenario_pnl_by_unit: Mapping[str, Sequence[float]],
    *,
    exposures_usd: Mapping[str, float] | None = None,
    scenario_names: Sequence[str] | None = None,
    minimum_correlation: float = 0.60,
    minimum_members: int = 2,
    minimum_cluster_loss_usd: float = 10_000,
    methodology_version: str = "correlated_clusters_v2",
    historical_pnl_by_unit: Mapping[str, Mapping[str, float]] | None = None,
    factor_vectors_by_unit: Mapping[str, Sequence[float]] | None = None,
    source_weights: Mapping[str, float] | None = None,
    minimum_pairwise_overlap: int = 60,
) -> list[dict[str, Any]]:
    # Scenario loss is required to name and quantify every returned cluster;
    # optional evidence may enrich those units but never create a zero-filled
    # synthetic scenario vector.
    ids = sorted(scenario_pnl_by_unit)
    vectors = {key: np.asarray(scenario_pnl_by_unit[key], dtype=np.float64) for key in ids}
    shapes = {value.shape for value in vectors.values()}
    if len(shapes) > 1 or any(value.ndim != 1 for value in vectors.values()):
        raise ValueError("scenario P&L vectors must be aligned and one-dimensional")
    configured_weights = {
        "scenario": 1.0,
        "historical": 0.0,
        "factor": 0.0,
        **{key: float(value) for key, value in (source_weights or {}).items()},
    }
    if any(not math.isfinite(value) or value < 0 for value in configured_weights.values()):
        raise ValueError("cluster source weights must be finite and non-negative")
    correlation: dict[tuple[str, str], float] = {}
    pair_diagnostics: dict[tuple[str, str], dict[str, Any]] = {}
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1 :]:
            source_values: dict[str, float] = {}
            if left in vectors and right in vectors and len(vectors[left]) >= 2:
                scenario_rho = _correlation(vectors[left], vectors[right])
                if scenario_rho is not None:
                    source_values["scenario"] = scenario_rho
            overlap = 0
            if historical_pnl_by_unit and left in historical_pnl_by_unit and right in historical_pnl_by_unit:
                dates = sorted(
                    set(historical_pnl_by_unit[left])
                    & set(historical_pnl_by_unit[right])
                )
                overlap = len(dates)
                if overlap >= minimum_pairwise_overlap:
                    history_rho = _correlation(
                        np.asarray([historical_pnl_by_unit[left][day] for day in dates]),
                        np.asarray([historical_pnl_by_unit[right][day] for day in dates]),
                    )
                    if history_rho is not None:
                        source_values["historical"] = history_rho
            if factor_vectors_by_unit and left in factor_vectors_by_unit and right in factor_vectors_by_unit:
                left_factor = np.asarray(factor_vectors_by_unit[left], dtype=np.float64)
                right_factor = np.asarray(factor_vectors_by_unit[right], dtype=np.float64)
                if left_factor.shape == right_factor.shape and left_factor.size >= 2:
                    factor_rho = _correlation(left_factor, right_factor)
                    if factor_rho is not None:
                        source_values["factor"] = factor_rho
            # Renormalize the configured mix over evidence available for this pair.
            used_weights = {
                source: configured_weights.get(source, 0.0)
                for source in source_values
                if configured_weights.get(source, 0.0) > 0
            }
            weight_total = sum(used_weights.values())
            normalized_weights = {
                source: value / weight_total for source, value in used_weights.items()
            } if weight_total else {}
            rho = sum(
                normalized_weights[source] * source_values[source]
                for source in normalized_weights
            ) if normalized_weights else 0.0
            correlation[(left, right)] = max(-1.0, min(1.0, rho))
            pair_diagnostics[(left, right)] = {
                "historical_overlap": overlap,
                "sources_available": sorted(source_values),
                "source_weights_used": normalized_weights,
            }
    correlation = _project_pairwise_psd(ids, correlation)
    # Merge the most correlated groups until average linkage falls below the cutoff.
    clusters: list[tuple[str, ...]] = [(item,) for item in ids]
    while True:
        candidates = []
        for left_index, left in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                values = [_rho(correlation, a, b) for a in left for b in right]
                average = sum(values) / len(values)
                candidates.append((average, left, right, left_index, right_index))
        if not candidates:
            break
        best = max(candidates, key=lambda item: (item[0], tuple(item[1]), tuple(item[2])))
        if best[0] < minimum_correlation:
            break
        _, left, right, left_index, right_index = best
        merged = tuple(sorted((*left, *right)))
        clusters = [item for index, item in enumerate(clusters) if index not in {left_index, right_index}]
        clusters.append(merged)
        clusters.sort()
    results = []
    for members in sorted(clusters):
        if len(members) < minimum_members:
            continue
        matrix = np.vstack([vectors[item] for item in members])
        total_pnl = matrix.sum(axis=0)
        # Measure cluster loss on one shared adverse scenario, preserving offsets.
        worst_index = int(np.argmin(total_pnl))
        loss = max(0.0, -float(total_pnl[worst_index]))
        if loss < minimum_cluster_loss_usd:
            continue
        weights = {item: abs(float((exposures_usd or {}).get(item, 1.0))) for item in members}
        numerator = denominator = 0.0
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                weight = weights[left] * weights[right]
                numerator += weight * _rho(correlation, left, right)
                denominator += weight
        effective = numerator / denominator if denominator else 0.0
        cluster_pairs = [
            pair_diagnostics.get((left, right) if left < right else (right, left), {})
            for left_index, left in enumerate(members)
            for right in members[left_index + 1 :]
        ]
        mix_totals = {"scenario": 0.0, "historical": 0.0, "factor": 0.0}
        for diagnostic in cluster_pairs:
            for source, value in diagnostic.get("source_weights_used", {}).items():
                mix_totals[source] += float(value)
        mix_denominator = sum(mix_totals.values())
        actual_mix = {
            source: value / mix_denominator if mix_denominator else 0.0
            for source, value in mix_totals.items()
        }
        stable = hashlib.sha256(
            (methodology_version + "|" + "|".join(members)).encode("utf-8")
        ).hexdigest()
        results.append(
            {
                "cluster_id": f"sha256:{stable}",
                "name": f"Correlated group {len(results) + 1}",
                "members": list(members),
                "estimated_loss_usd": loss,
                "adverse_scenario": (
                    scenario_names[worst_index]
                    if scenario_names and worst_index < len(scenario_names)
                    else f"scenario-{worst_index}"
                ),
                "effective_correlation": effective,
                "correlation_source_mix": actual_mix,
                "pairwise_source_coverage": cluster_pairs,
                "methodology_version": (
                    "hybrid_clusters_v3"
                    if actual_mix["historical"] or actual_mix["factor"]
                    else methodology_version
                ),
            }
        )
    return results


def _rho(values: Mapping[tuple[str, str], float], left: str, right: str) -> float:
    if left == right:
        return 1.0
    key = (left, right) if left < right else (right, left)
    return values.get(key, 0.0)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        return None
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return None
    if np.std(left) <= np.finfo(np.float64).eps or np.std(right) <= np.finfo(np.float64).eps:
        return None
    value = float(np.corrcoef(left - left.mean(), right - right.mean())[0, 1])
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else None


def _project_pairwise_psd(
    ids: Sequence[str], values: Mapping[tuple[str, str], float]
) -> dict[tuple[str, str], float]:
    if len(ids) < 2:
        return dict(values)
    matrix = np.eye(len(ids), dtype=np.float64)
    for left_index, left in enumerate(ids):
        for right_index in range(left_index + 1, len(ids)):
            right = ids[right_index]
            matrix[left_index, right_index] = matrix[right_index, left_index] = _rho(values, left, right)
    # Clip negative eigenvalues and restore a unit diagonal for valid correlations.
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    projected = eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(projected), np.finfo(np.float64).eps))
    projected = projected / np.outer(diagonal, diagonal)
    output: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(ids):
        for right_index in range(left_index + 1, len(ids)):
            output[(left, ids[right_index])] = max(
                -1.0, min(1.0, float(projected[left_index, right_index]))
            )
    return output


__all__ = ["cluster_risk_units"]
