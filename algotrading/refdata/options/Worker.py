"""Canonical option calculation and volatility-surface worker.

This module coordinates the numerical code in :mod:`Calculations` and
:mod:`Surface` with the generation-3 repositories.  It has no raw-payload or
pre-redesign storage dependency.
"""

from __future__ import annotations

import hashlib
import logging
import math
import signal
import threading
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import date, datetime, time as date_time, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

from algotrading.refdata.options.Common import (
    OPTIONS_CALCS_CONFIG_PREFIX,
    OPTIONS_CALCS_GREEKS_PREFIX,
    OPTIONS_CALCS_SURFACE_PREFIX,
    _as_bool,
    _finite_float,
    _parse_timestamp,
    _utc_now,
    create_clickhouse_client,
    is_operational_options_calcs_key,
    validate_database_identifier,
    validate_options_calcs_configuration,
)
from algotrading.refdata.options.Definitions import (
    DefinitionRepository,
    DefinitionType,
    build_definition,
)
from algotrading.refdata.options.Runtime import (
    CalculationRepository,
    CalculationWork,
    SurfaceRepository,
    SurfaceWork,
    discover_interrupted_calculation_work,
    discover_interrupted_surface_work,
    discover_calculation_work,
    discover_surface_work,
    maintenance_lease_active,
    query_expired_surface_work,
    query_surface_cadence_health,
    typed_fit_columns,
)
from algotrading.refdata.options.Surface import (
    SVIWarmStartCache,
    SurfaceNumericalRuntime,
    is_required_surface_node,
)
from algotrading.refdata.options.Workers import (
    LocalWorkerPool,
    ScheduledCalculationTask,
    ScheduledSurfaceTask,
    WorkerTaskResult,
)
from algotrading.util.NagiosSystemMonitor import NagiosSystemMonitor


UTC = timezone.utc


def _digest(value: Any) -> bytes:
    if isinstance(value, bytes) and len(value) == 32:
        return value
    text = str(value or "")
    try:
        decoded = bytes.fromhex(text)
    except ValueError:
        decoded = b""
    return decoded if len(decoded) == 32 else hashlib.sha256(text.encode()).digest()


def _definition_configuration(
    context: Mapping[str, Any], *, surface: bool
) -> dict[str, Any]:
    prefix = OPTIONS_CALCS_SURFACE_PREFIX if surface else OPTIONS_CALCS_CONFIG_PREFIX
    return {
        str(key): context[key]
        for key in sorted(context)
        if str(key).startswith(prefix)
        and not is_operational_options_calcs_key(str(key))
        and (
            surface
            or (
                not str(key).startswith(OPTIONS_CALCS_SURFACE_PREFIX)
                and not str(key).startswith(OPTIONS_CALCS_GREEKS_PREFIX)
            )
        )
    }


def _latest_processing_source_at(
    client: Any, database: str, definition_type: DefinitionType
) -> Optional[datetime]:
    """Return the authoritative high-water mark owned by one worker stage."""
    database = validate_database_identifier(database)
    table = {
        DefinitionType.CALCULATION: "options_chain_calculation_state",
        DefinitionType.SURFACE: "options_chain_surface_state",
    }.get(definition_type)
    if table is None:
        raise ValueError(
            f"Unsupported processing definition type: {definition_type.value}"
        )
    rows = client.query(
        f"SELECT maxOrNull(source_collected_at) "
        f"FROM {database}.{table} FINAL WHERE is_active"
    ).result_rows
    return _parse_timestamp(rows[0][0]) if rows and rows[0][0] is not None else None


def _processing_activation_boundary(
    context: Mapping[str, Any],
    key: str,
    *,
    latest_published_at: Optional[datetime],
    now: datetime,
) -> tuple[datetime, bool]:
    """Resolve an explicit boundary or derive a strictly prospective one."""
    configured = context.get(key)
    if configured is not None and str(configured).strip():
        parsed = _parse_timestamp(configured)
        if parsed is None:
            raise ValueError(f"{key} is invalid")
        return parsed, True
    boundary = _parse_timestamp(now)
    if boundary is None:
        raise ValueError("Current worker activation time is invalid")
    if latest_published_at is not None:
        boundary = max(
            boundary,
            latest_published_at + timedelta(microseconds=1),
        )
    return boundary, False


def _reject_backdated_new_definition(
    repository: DefinitionRepository,
    definition: Any,
    latest_published_at: Optional[datetime],
    *,
    configured_key: str,
) -> None:
    if any(
        stored.definition_id == definition.definition_id
        for stored in repository.load(definition.definition_type)
    ):
        return
    if (
        latest_published_at is not None
        and definition.activated_at <= latest_published_at
    ):
        raise ValueError(
            f"{configured_key}={definition.activated_at.isoformat()} must be after "
            f"the latest published {definition.definition_type.value} source "
            f"timestamp {latest_published_at.isoformat()}"
        )


def ensure_processing_definitions(
    context: Mapping[str, Any], client: Any, database: str
) -> tuple[int, int]:
    """Ensure deterministic prospective calculation/surface definitions."""
    repository = DefinitionRepository(client, database)
    code_version_key = OPTIONS_CALCS_CONFIG_PREFIX + "code_version"
    code_version = str(context.get(code_version_key) or "").strip()
    if not code_version or code_version.lower() == "unknown":
        raise ValueError(f"{code_version_key} must identify the deployed calculation code")
    activation_now = _utc_now()
    calculation_key = OPTIONS_CALCS_CONFIG_PREFIX + "definition.activated_at"
    calculation_latest = _latest_processing_source_at(
        client, database, DefinitionType.CALCULATION
    )
    calculation_activation, _ = _processing_activation_boundary(
        context,
        calculation_key,
        latest_published_at=calculation_latest,
        now=activation_now,
    )
    calculation = build_definition(
        DefinitionType.CALCULATION,
        code_version=code_version,
        configuration=_definition_configuration(context, surface=False),
        solver_configuration={
            str(key): context[key]
            for key in sorted(context)
            if str(key).startswith(OPTIONS_CALCS_GREEKS_PREFIX)
            and not is_operational_options_calcs_key(str(key))
        },
        activated_at=calculation_activation,
    )
    _reject_backdated_new_definition(
        repository,
        calculation,
        calculation_latest,
        configured_key=calculation_key,
    )
    calculation = repository.activate_prospective(calculation)
    surface_key = OPTIONS_CALCS_SURFACE_PREFIX + "definition.activated_at"
    surface_latest = _latest_processing_source_at(
        client, database, DefinitionType.SURFACE
    )
    surface_activation, surface_explicit = _processing_activation_boundary(
        context,
        surface_key,
        latest_published_at=surface_latest,
        now=activation_now,
    )
    if surface_activation < calculation.activated_at:
        if surface_explicit:
            raise ValueError(
                f"{surface_key} cannot precede its calculation definition activation "
                f"at {calculation.activated_at.isoformat()}"
            )
        surface_activation = calculation.activated_at
    surface_definition = build_definition(
        DefinitionType.SURFACE,
        code_version=code_version,
        configuration=_definition_configuration(context, surface=True),
        parent_definition_id=calculation.definition_id,
        activated_at=surface_activation,
    )
    _reject_backdated_new_definition(
        repository,
        surface_definition,
        surface_latest,
        configured_key=surface_key,
    )
    surface_definition = repository.activate_prospective(surface_definition)
    return calculation.definition_id, surface_definition.definition_id


def _snapshot_spot(row: Mapping[str, Any]) -> Optional[float]:
    bid = _finite_float(row.get("underlying_bid"))
    ask = _finite_float(row.get("underlying_ask"))
    if bid is not None and ask is not None and 0 <= bid <= ask and ask > 0:
        return (bid + ask) / 2.0
    for name in ("underlying_mark", "chain_underlying_price", "underlying_last"):
        value = _finite_float(row.get(name))
        if value is not None and value > 0:
            return value
    return None


def _calculation_failure(
    work: CalculationWork,
    row: Mapping[str, Any],
    *,
    reason: str,
    stage: str,
) -> dict[str, Any]:
    now = _parse_timestamp(row.get("calculated_at")) or _utc_now()
    base = row.get("_base") or {}
    contract_id = int(base["contract_id"])
    fingerprint = _digest(
        row.get("recalc_failure_fingerprint")
        or f"{work.snapshot_id}|{contract_id}|{stage}|{reason}"
    )
    tenor = row.get("time_to_expiry_seconds")
    return {
        "snapshot_id": str(work.snapshot_id),
        "source_collected_at": work.source_collected_at,
        "underlying_symbol": work.underlying_symbol,
        "calculation_definition_id": work.calculation_definition_id,
        "failure_scope": "contract",
        "expiration_at": _parse_timestamp(base.get("expiration_at")),
        "contract_id": contract_id,
        "failure_stage": stage or "validation",
        "reason_code": reason or "unknown_failure",
        "error_class": "NumericalFailure",
        "error_detail": str(row.get("calculation_error") or reason),
        "retry_class": str(row.get("recalc_retry_class") or "deterministic"),
        "retry_after": _parse_timestamp(row.get("recalc_retry_after")),
        "failure_fingerprint": fingerprint,
        "first_failed_at": now,
        "last_failed_at": now,
        "failure_count": 1,
        "updated_at": now,
        "is_active": True,
        "valuation_at": work.source_collected_at,
        "spot": _finite_float(row.get("spot_price")),
        "target_price": _finite_float(row.get("recalc_target_price")),
        "time_to_expiry_seconds": int(tenor) if tenor is not None and tenor >= 0 else None,
        "risk_free_rate": _finite_float(row.get("risk_free_rate")),
        "dividend_yield": _finite_float(row.get("dividend_yield")),
        "method": str(row.get("recalc_method") or "none"),
        "input_basis": str(row.get("recalc_input_basis") or "market_mid"),
        "dividend_model": str(row.get("recalc_dividend_model") or "continuous_yield"),
        "partial_iv": _finite_float(row.get("recalc_iv")),
        "partial_model_price": _finite_float(row.get("recalc_model_price")),
        "partial_price_residual": _finite_float(row.get("recalc_price_residual")),
        "seed_iv": _finite_float(row.get("recalc_iv_seed_value")),
        "seed_origin_snapshot_id": row.get("recalc_iv_seed_snapshot_id"),
        "quality_flags": int(row.get("recalc_quality_flags") or 0),
    }


def compose_calculation_publication(
    work: CalculationWork,
    source_rows: Sequence[Mapping[str, Any]],
    numerical_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    list[int],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Separate numerical successes, source-only outcomes, and failures."""
    if len(source_rows) != len(numerical_rows):
        raise ValueError("Numerical output cardinality differs from normalized source")
    observations: dict[int, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    source_only: list[int] = []
    failures: list[dict[str, Any]] = []
    expiration_inputs: dict[datetime, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    for source, numerical in zip(source_rows, numerical_rows):
        base = numerical.get("_base") or source
        contract_id = int(base["contract_id"])
        observations[contract_id] = {
            **dict(source),
            "underlying_symbol": work.underlying_symbol,
        }
        calculation_status = str(numerical.get("calculation_status") or "")
        recalc_status = str(numerical.get("recalc_status") or "")
        if calculation_status != "complete":
            failures.append(
                _calculation_failure(
                    work,
                    numerical,
                    reason=str(numerical.get("calculation_error") or "invalid_input"),
                    stage="input_validation",
                )
            )
            continue
        expiration = _parse_timestamp(base.get("expiration_at"))
        tenor = numerical.get("time_to_expiry_seconds")
        if (
            expiration is not None
            and tenor is not None
            and 0 < int(tenor) <= 2**32 - 1
            and all(
                _finite_float(numerical.get(name)) is not None
                for name in ("risk_free_rate", "dividend_yield", "forward_price")
            )
            and numerical.get("fred_observation_date") is not None
        ):
            candidate = {
                "snapshot_id": str(work.snapshot_id),
                "source_collected_at": work.source_collected_at,
                "underlying_symbol": work.underlying_symbol,
                "expiration_at": expiration,
                "calculation_definition_id": work.calculation_definition_id,
                "time_to_expiry_seconds": int(tenor),
                "risk_free_rate": float(numerical["risk_free_rate"]),
                "dividend_yield": float(numerical["dividend_yield"]),
                "forward_price": float(numerical["forward_price"]),
                "fred_observation_date": numerical["fred_observation_date"],
                "fred_lower_series": str(numerical.get("fred_lower_series") or ""),
                "fred_upper_series": str(numerical.get("fred_upper_series") or ""),
                "quality_flags": 0,
                "calculated_at": numerical["calculated_at"],
            }
            previous = expiration_inputs.setdefault(expiration, candidate)
            if previous != candidate:
                raise ValueError("Expiration inputs differ within one snapshot expiration")
        if recalc_status in {"disabled", "skipped", "not_selected", "source_only"}:
            source_only.append(contract_id)
            continue
        if recalc_status != "complete":
            failures.append(
                _calculation_failure(
                    work,
                    numerical,
                    reason=str(numerical.get("recalc_reason") or "numerical_failure"),
                    stage=str(numerical.get("recalc_failure_stage") or "solver"),
                )
            )
            continue
        source_anchor = (
            str(numerical.get("recalc_iv_identifiability_status"))
            == "source_anchor_flat"
        )
        origin = (
            "reused"
            if str(numerical.get("recalc_result_origin")) == "reused"
            else "fresh"
        )
        actual_dtype = str(numerical.get("recalc_actual_dtype") or "float64").lower()
        if actual_dtype not in {"float32", "float64"}:
            actual_dtype = "float64"
        required_outputs = (
            "recalc_target_price", "recalc_iv", "recalc_price_residual",
            "recalc_model_price", "recalc_delta", "recalc_gamma",
            "recalc_theta_per_day", "recalc_vega_per_1pct",
            "recalc_rho_per_1pct",
        )
        try:
            iterations = int(numerical.get("recalc_iv_iterations") or 0)
        except (TypeError, ValueError, OverflowError):
            iterations = -1
        if (
            any(_finite_float(numerical.get(name)) is None for name in required_outputs)
            or not 0 <= iterations <= 2**16 - 1
            or (source_anchor and _finite_float(source.get("source_iv")) is None)
            or _parse_timestamp(numerical.get("calculated_at")) is None
        ):
            failures.append(
                _calculation_failure(
                    work,
                    numerical,
                    reason="nonfinite_or_invalid_success_output",
                    stage="solver_output_validation",
                )
            )
            continue
        result = {
            "snapshot_id": str(work.snapshot_id),
            "source_collected_at": work.source_collected_at,
            "underlying_symbol": work.underlying_symbol,
            "contract_id": contract_id,
            "calculation_definition_id": work.calculation_definition_id,
            "calculated_at": numerical["calculated_at"],
            "recalc_method": str(numerical.get("recalc_method") or "none"),
            "result_origin": origin,
            "origin_snapshot_id": str(
                numerical.get("recalc_origin_snapshot_id") or work.snapshot_id
            ),
            "input_basis": str(numerical.get("recalc_input_basis") or "market_mid"),
            "dividend_model": str(
                numerical.get("recalc_dividend_model") or "continuous_yield"
            ),
            "iv_selection": "source" if source_anchor else "result",
            "greeks_selection": "result",
            "target_price": float(numerical["recalc_target_price"]),
            "iv": float(numerical["recalc_iv"]),
            "iv_iterations": iterations,
            "price_residual": float(numerical["recalc_price_residual"]),
            "model_price": float(numerical["recalc_model_price"]),
            "delta": float(numerical["recalc_delta"]),
            "gamma": float(numerical["recalc_gamma"]),
            "theta_per_day": float(numerical["recalc_theta_per_day"]),
            "vega_per_1pct": float(numerical["recalc_vega_per_1pct"]),
            "rho_per_1pct": float(numerical["recalc_rho_per_1pct"]),
            "actual_dtype": actual_dtype,
            "precision_escalated": bool(numerical.get("recalc_precision_escalated")),
            "quality_flags": int(numerical.get("recalc_quality_flags") or 0),
        }
        if origin == "reused":
            required_reuse_values = (
                "recalc_origin_snapshot_id", "recalc_origin_calculated_at",
                "recalc_valuation_at", "recalc_valuation_spot",
                "recalc_valuation_target_price",
                "recalc_valuation_time_to_expiry_seconds",
                "recalc_valuation_rate", "recalc_valuation_dividend_yield",
            )
            origin_tenor = numerical.get(
                "recalc_valuation_time_to_expiry_seconds"
            )
            try:
                origin_tenor_int = int(origin_tenor or 0)
            except (TypeError, ValueError, OverflowError):
                origin_tenor_int = 0
            if (
                any(numerical.get(name) is None for name in required_reuse_values)
                or any(
                    _finite_float(numerical.get(name)) is None
                    for name in (
                        "recalc_valuation_spot", "recalc_valuation_target_price",
                        "recalc_valuation_rate",
                        "recalc_valuation_dividend_yield", "risk_free_rate",
                        "dividend_yield",
                    )
                )
                or not 0 < origin_tenor_int <= 2**32 - 1
                or _parse_timestamp(numerical.get("recalc_origin_calculated_at")) is None
                or _parse_timestamp(numerical.get("recalc_valuation_at")) is None
            ):
                failures.append(
                    _calculation_failure(
                        work,
                        numerical,
                        reason="incomplete_reuse_provenance",
                        stage="reuse_validation",
                    )
                )
                continue
            origin_target = float(numerical["recalc_valuation_target_price"])
            current_target = float(numerical["recalc_target_price"])
            try:
                uuid.UUID(str(numerical["recalc_origin_snapshot_id"]))
                reuse_age_seconds = int(
                    numerical.get("recalc_reuse_age_seconds") or 0
                )
            except (TypeError, ValueError, OverflowError):
                reuse_age_seconds = -1
            provenance_row = {
                    "snapshot_id": str(work.snapshot_id),
                    "source_collected_at": work.source_collected_at,
                    "underlying_symbol": work.underlying_symbol,
                    "contract_id": contract_id,
                    "calculation_definition_id": work.calculation_definition_id,
                    "origin_snapshot_id": str(numerical["recalc_origin_snapshot_id"]),
                    "origin_calculated_at": numerical["recalc_origin_calculated_at"],
                    "origin_valuation_at": numerical["recalc_valuation_at"],
                    "origin_spot": float(numerical["recalc_valuation_spot"]),
                    "origin_target_price": origin_target,
                    "origin_time_to_expiry_seconds": origin_tenor_int,
                    "origin_risk_free_rate": float(numerical["recalc_valuation_rate"]),
                    "origin_dividend_yield": float(
                        numerical["recalc_valuation_dividend_yield"]
                    ),
                    "origin_input_basis": str(numerical["recalc_input_basis"]),
                    "origin_method": str(numerical["recalc_method"]),
                    "origin_dividend_model": str(numerical["recalc_dividend_model"]),
                    "reuse_age_seconds": reuse_age_seconds,
                    "spot_movement_bps": (
                        0.0
                        if numerical.get("recalc_reuse_spot_move_bps") is None
                        else _finite_float(numerical.get("recalc_reuse_spot_move_bps"))
                    ),
                    "target_price_absolute_change": abs(current_target - origin_target),
                    "target_price_fractional_change": (
                        abs(current_target - origin_target) / abs(origin_target)
                        if origin_target
                        else 0.0
                    ),
                    "elapsed_tenor_fraction": (
                        0.0
                        if numerical.get("recalc_reuse_elapsed_tenor_fraction") is None
                        else _finite_float(
                            numerical.get("recalc_reuse_elapsed_tenor_fraction")
                        )
                    ),
                    "rate_change": abs(
                        float(numerical["risk_free_rate"])
                        - float(numerical["recalc_valuation_rate"])
                    ),
                    "dividend_yield_change": abs(
                        float(numerical["dividend_yield"])
                        - float(numerical["recalc_valuation_dividend_yield"])
                    ),
                    "estimated_iv_drift": (
                        0.0
                        if numerical.get("recalc_reuse_estimated_iv_change_points") is None
                        else _finite_float(
                            numerical.get("recalc_reuse_estimated_iv_change_points")
                        )
                    ),
                    "estimated_delta_drift": (
                        0.0
                        if numerical.get("recalc_shadow_delta_error") is None
                        else _finite_float(numerical.get("recalc_shadow_delta_error"))
                    ),
                    "reuse_mode": str(numerical.get("recalc_reuse_mode") or "bounded"),
                    "policy_reason": str(numerical.get("recalc_reason") or "reused"),
                    "created_at": numerical["calculated_at"],
                }
            measurement_names = (
                "origin_spot", "origin_target_price", "origin_risk_free_rate",
                "origin_dividend_yield", "spot_movement_bps",
                "target_price_absolute_change", "target_price_fractional_change",
                "elapsed_tenor_fraction", "rate_change", "dividend_yield_change",
                "estimated_iv_drift", "estimated_delta_drift",
            )
            if reuse_age_seconds < 0 or any(
                _finite_float(provenance_row.get(name)) is None
                for name in measurement_names
            ) or any(
                abs(float(provenance_row[name])) > 3.4028235e38
                for name in measurement_names[4:]
                if provenance_row.get(name) is not None
            ):
                failures.append(
                    _calculation_failure(
                        work,
                        numerical,
                        reason="invalid_reuse_measurements",
                        stage="reuse_validation",
                    )
                )
                continue
            provenance.append(provenance_row)
        results.append(result)
    return (
        results,
        observations,
        source_only,
        failures,
        list(expiration_inputs.values()),
        provenance,
    )


def load_prior_outcomes(
    client: Any,
    database: str,
    selected_rows: Sequence[Mapping[str, Any]],
    policy: Any,
) -> Mapping[str, Any]:
    """Load causal success/failure reuse candidates from canonical facts.

    Fresh origins derive their valuation inputs from the prior snapshot and
    expiration-input fact.  Reused origins prefer the self-contained sparse
    provenance row, preserving the first origin across repeated reuse.
    """
    from algotrading.refdata.options.Reuse import (
        PriorOutcomeCandidates,
        PriorRepricingOutcome,
    )
    from algotrading.refdata.options.Queries import (
        authoritative_calculation_fact_subquery,
    )

    if not selected_rows:
        return {}
    need_success_reuse = bool(getattr(policy, "need_success_reuse", True))
    need_success_seed = bool(getattr(policy, "need_success_seed", True))
    need_failure_reuse = bool(getattr(policy, "need_failure_reuse", True))
    need_failure_seed = bool(getattr(policy, "need_failure_seed", True))
    need_success = need_success_reuse or need_success_seed
    need_failure = need_failure_reuse or need_failure_seed
    if not need_success and not need_failure:
        return {}
    definition_ids = {
        int(row["calculation_definition_id"]) for row in selected_rows
    }
    if len(definition_ids) != 1:
        raise ValueError("Prior lookup spans multiple calculation definitions")
    definition_id = next(iter(definition_ids))
    cutoff = min(_parse_timestamp(row["source_collected_at"]) for row in selected_rows)
    lower = cutoff.timestamp() - (
        float(policy.maximum_age_seconds) + float(policy.lookup_allowance_seconds)
    )
    symbols = sorted({str(row["option_symbol"]) for row in selected_rows})
    current_by_symbol = {
        str(row["option_symbol"]): row for row in selected_rows
    }
    successes = []
    if need_success:
        success_seed_filter = (
            """
          AND isFinite(r.iv)
          AND r.iv >= {minimum_seed_iv:Float64}
          AND r.iv <= {maximum_seed_iv:Float64}
            """
            if need_success_seed and not need_success_reuse
            else ""
        )
        successes = client.query_arrow(
            f"""
        SELECT c.option_symbol AS option_symbol,
               toString(r.snapshot_id) AS source_snapshot_id,
               r.source_collected_at AS source_collected_at,
               r.calculated_at AS calculated_at,
               toString(r.result_origin) AS result_origin,
               toString(r.origin_snapshot_id) AS result_origin_snapshot_id,
               r.recalc_method, r.input_basis, r.dividend_model, r.target_price,
               r.iv, r.iv_iterations, r.price_residual, r.model_price, r.delta,
               r.gamma, r.theta_per_day, r.vega_per_1pct, r.rho_per_1pct,
               toString(r.actual_dtype) AS actual_dtype, r.precision_escalated,
               r.quality_flags AS quality_flags,
               e.time_to_expiry_seconds, e.risk_free_rate,
               e.dividend_yield, s.underlying_bid, s.underlying_ask,
               s.underlying_mark, s.chain_underlying_price, s.underlying_last,
               toString(p.origin_snapshot_id) AS provenance_origin_snapshot_id,
               p.origin_calculated_at, p.origin_valuation_at, p.origin_spot,
               p.origin_target_price, p.origin_time_to_expiry_seconds,
               p.origin_risk_free_rate, p.origin_dividend_yield,
               p.origin_input_basis, p.origin_method, p.origin_dividend_model
        FROM {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_results', fact_alias='r', state_alias='result_state')}
        INNER JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c
          ON c.contract_id = r.contract_id
        INNER JOIN {database}.options_chain_snapshots AS s
          ON s.snapshot_id = r.snapshot_id
         AND s.source_collected_at = r.source_collected_at
        INNER JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_expiration_inputs', fact_alias='e', state_alias='expiration_state')}
          ON e.snapshot_id = r.snapshot_id
         AND e.source_collected_at = r.source_collected_at
         AND e.calculation_definition_id = r.calculation_definition_id
         AND e.expiration_at = c.expiration_at
        LEFT JOIN {database}.options_chain_calculation_reuse_provenance AS p
          ON p.snapshot_id = r.snapshot_id
         AND p.source_collected_at = r.source_collected_at
         AND p.calculation_definition_id = r.calculation_definition_id
         AND p.contract_id = r.contract_id
        WHERE r.source_collected_at >= fromUnixTimestamp64Micro({{lower_us:Int64}})
          AND r.source_collected_at < fromUnixTimestamp64Micro({{cutoff_us:Int64}})
          AND r.calculation_definition_id = {{definition_id:UInt32}}
          AND c.option_symbol IN {{symbols:Array(String)}}
          AND (
            (
              r.result_origin = 'reused'
              AND isNotNull(p.origin_snapshot_id)
              AND isNotNull(p.origin_calculated_at)
              AND isNotNull(p.origin_valuation_at)
              AND isNotNull(p.origin_spot)
              AND isNotNull(p.origin_target_price)
              AND isNotNull(p.origin_time_to_expiry_seconds)
              AND isNotNull(p.origin_risk_free_rate)
              AND isNotNull(p.origin_dividend_yield)
              AND isNotNull(p.origin_input_basis)
              AND isNotNull(p.origin_method)
              AND isNotNull(p.origin_dividend_model)
            )
            OR
            (
              r.result_origin != 'reused'
              AND (
                (
                  isFinite(s.underlying_bid)
                  AND isFinite(s.underlying_ask)
                  AND s.underlying_bid >= 0
                  AND s.underlying_bid <= s.underlying_ask
                  AND s.underlying_ask > 0
                )
                OR (isFinite(s.underlying_mark) AND s.underlying_mark > 0)
                OR (
                  isFinite(s.chain_underlying_price)
                  AND s.chain_underlying_price > 0
                )
                OR (isFinite(s.underlying_last) AND s.underlying_last > 0)
              )
            )
          )
          {success_seed_filter}
        ORDER BY c.option_symbol, r.source_collected_at DESC,
                 r.calculated_at DESC, r.snapshot_id DESC, r.contract_id DESC
        LIMIT 1 BY c.option_symbol
        """,
            parameters={
                "lower_us": int(lower * 1_000_000),
                "cutoff_us": int(cutoff.timestamp() * 1_000_000),
                "definition_id": definition_id,
                "symbols": symbols,
                "minimum_seed_iv": float(getattr(policy, "minimum_seed_iv", 0.0)),
                "maximum_seed_iv": float(
                    getattr(policy, "maximum_seed_iv", 1.0e9)
                ),
            },
        ).to_pylist()
    success_by_symbol: dict[str, Any] = {}
    for row in successes:
        symbol = str(row["option_symbol"])
        if symbol in success_by_symbol:
            continue
        if need_success_seed and not need_success_reuse:
            seed_iv = _finite_float(row.get("iv"))
            if (
                seed_iv is None
                or seed_iv < float(getattr(policy, "minimum_seed_iv", 0.0))
                or seed_iv > float(getattr(policy, "maximum_seed_iv", 1.0e9))
            ):
                continue
        current = current_by_symbol[symbol]
        reused = str(row["result_origin"]) == "reused"
        if reused:
            required = (
                "provenance_origin_snapshot_id",
                "origin_calculated_at",
                "origin_valuation_at",
                "origin_spot",
                "origin_target_price",
                "origin_time_to_expiry_seconds",
                "origin_risk_free_rate",
                "origin_dividend_yield",
                "origin_input_basis",
                "origin_method",
                "origin_dividend_model",
            )
            if any(row.get(name) is None for name in required):
                continue
            origin_snapshot = row["provenance_origin_snapshot_id"]
            origin_calculated = row["origin_calculated_at"]
            valuation_at = row["origin_valuation_at"]
            valuation_spot = row["origin_spot"]
            valuation_target = row["origin_target_price"]
            valuation_tenor = row["origin_time_to_expiry_seconds"]
            valuation_rate = row["origin_risk_free_rate"]
            valuation_dividend = row["origin_dividend_yield"]
            input_basis = row["origin_input_basis"]
            method = row["origin_method"]
            dividend_model = row["origin_dividend_model"]
        else:
            valuation_spot = _snapshot_spot(row)
            if valuation_spot is None:
                continue
            origin_snapshot = row["source_snapshot_id"]
            origin_calculated = row["calculated_at"]
            valuation_at = row["source_collected_at"]
            valuation_target = row["target_price"]
            valuation_tenor = row["time_to_expiry_seconds"]
            valuation_rate = row["risk_free_rate"]
            valuation_dividend = row["dividend_yield"]
            input_basis = row["input_basis"]
            method = row["recalc_method"]
            dividend_model = row["dividend_model"]
        success_by_symbol[symbol] = PriorRepricingOutcome(
            option_symbol=symbol,
            source_snapshot_id=str(row["source_snapshot_id"]),
            source_row_collected_at=_parse_timestamp(row["source_collected_at"]),
            calculation_definition_id=definition_id,
            source_calculated_at=_parse_timestamp(row["calculated_at"]),
            origin_snapshot_id=str(origin_snapshot),
            origin_calculated_at=_parse_timestamp(origin_calculated),
            valuation_at=_parse_timestamp(valuation_at),
            valuation_spot=float(valuation_spot),
            valuation_target_price=float(valuation_target),
            valuation_time_to_expiry_seconds=float(valuation_tenor),
            valuation_rate=float(valuation_rate),
            valuation_dividend_yield=float(valuation_dividend),
            method=str(method),
            method_selection_reason="assigned_definition",
            algorithm_version=str(current.get("recalc_algorithm_version") or ""),
            solver_config_hash=str(current.get("recalc_solver_config_hash") or ""),
            solver_parameters_json=str(
                current.get("recalc_solver_parameters_json") or "{}"
            ),
            input_basis=str(input_basis),
            input_iv=float(row["iv"]),
            implied_volatility=float(row["iv"]),
            iv_iterations=int(row["iv_iterations"]),
            price_residual=float(row["price_residual"]),
            model_price=float(row["model_price"]),
            delta=float(row["delta"]),
            gamma=float(row["gamma"]),
            theta_per_day=float(row["theta_per_day"]),
            vega_per_1pct=float(row["vega_per_1pct"]),
            rho_per_1pct=float(row["rho_per_1pct"]),
            requested_dtype=str(row["actual_dtype"]),
            actual_dtype=str(row["actual_dtype"]),
            precision_escalated=bool(row["precision_escalated"]),
            quality_flags=int(row["quality_flags"]),
            time_steps=None,
            space_steps=None,
            damping_steps=None,
            dividend_model=str(dividend_model),
            result_origin=str(row["result_origin"]),
        )
    failed = []
    if need_failure:
        partial_seed_expression = """
          isNotNull(f.partial_iv)
          AND isFinite(f.partial_iv)
          AND f.partial_iv >= {minimum_seed_iv:Float64}
          AND f.partial_iv <= {maximum_seed_iv:Float64}
        """
        failure_seed_filter = (
            f"AND ({partial_seed_expression})"
            if need_failure_seed and not need_failure_reuse
            else ""
        )
        failure_limit = (
            "LIMIT 1 BY c.option_symbol, partial_seed_eligible"
            if need_failure_seed and need_failure_reuse
            else "LIMIT 1 BY c.option_symbol"
        )
        failed = client.query_arrow(
            f"""
        SELECT c.option_symbol, toString(f.snapshot_id) AS source_snapshot_id,
               f.source_collected_at, f.last_failed_at, f.valuation_at, f.spot,
               f.target_price, f.time_to_expiry_seconds, f.risk_free_rate,
               f.dividend_yield, f.method, f.input_basis, f.dividend_model,
               f.partial_iv, f.partial_model_price, f.partial_price_residual,
               f.retry_class, f.retry_after, f.failure_fingerprint,
               f.reason_code, f.failure_stage, f.quality_flags,
               toString(f.seed_origin_snapshot_id) AS seed_origin_snapshot_id,
               ({partial_seed_expression}) AS partial_seed_eligible
        FROM {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_failures', fact_alias='f', state_alias='failure_state', require_active_fact=True)}
        INNER JOIN (SELECT * FROM {database}.options_contracts FINAL) AS c
          ON c.contract_id = f.contract_id
        WHERE f.is_active AND f.failure_scope = 'contract'
          AND f.calculation_definition_id = {{definition_id:UInt32}}
          AND f.source_collected_at >= fromUnixTimestamp64Micro({{lower_us:Int64}})
          AND f.source_collected_at < fromUnixTimestamp64Micro({{cutoff_us:Int64}})
          AND c.option_symbol IN {{symbols:Array(String)}}
          AND isNotNull(f.valuation_at) AND isNotNull(f.spot)
          AND isNotNull(f.target_price) AND isNotNull(f.time_to_expiry_seconds)
          AND isNotNull(f.risk_free_rate) AND isNotNull(f.dividend_yield)
          {failure_seed_filter}
        ORDER BY c.option_symbol, f.source_collected_at DESC,
                 f.last_failed_at DESC, f.snapshot_id DESC, f.contract_id DESC
        {failure_limit}
        """,
            parameters={
                "lower_us": int(lower * 1_000_000),
                "cutoff_us": int(cutoff.timestamp() * 1_000_000),
                "definition_id": definition_id,
                "symbols": symbols,
                "minimum_seed_iv": float(getattr(policy, "minimum_seed_iv", 0.0)),
                "maximum_seed_iv": float(
                    getattr(policy, "maximum_seed_iv", 1.0e9)
                ),
            },
        ).to_pylist()
    failure_by_symbol: dict[str, Any] = {}
    failure_seed_by_symbol: dict[str, Any] = {}
    for row in failed:
        symbol = str(row["option_symbol"])
        seed_iv = _finite_float(row.get("partial_iv"))
        seed_eligible = bool(row.get("partial_seed_eligible")) or (
            seed_iv is not None
            and seed_iv >= float(getattr(policy, "minimum_seed_iv", 0.0))
            and seed_iv <= float(getattr(policy, "maximum_seed_iv", 1.0e9))
        )
        if (
            symbol in failure_by_symbol
            and (not need_failure_seed or symbol in failure_seed_by_symbol)
        ):
            continue
        if need_failure_seed and not need_failure_reuse:
            if not seed_eligible:
                continue
        current = current_by_symbol[symbol]
        outcome = PriorRepricingOutcome(
            option_symbol=symbol,
            source_snapshot_id=str(row["source_snapshot_id"]),
            source_row_collected_at=_parse_timestamp(row["source_collected_at"]),
            calculation_definition_id=definition_id,
            source_calculated_at=_parse_timestamp(row["last_failed_at"]),
            origin_snapshot_id=str(
                row.get("seed_origin_snapshot_id") or row["source_snapshot_id"]
            ),
            origin_calculated_at=_parse_timestamp(row["last_failed_at"]),
            valuation_at=_parse_timestamp(row["valuation_at"]),
            valuation_spot=float(row["spot"]),
            valuation_target_price=float(row["target_price"]),
            valuation_time_to_expiry_seconds=float(row["time_to_expiry_seconds"]),
            valuation_rate=float(row["risk_free_rate"]),
            valuation_dividend_yield=float(row["dividend_yield"]),
            method=str(row.get("method") or current.get("recalc_method") or "none"),
            method_selection_reason="assigned_definition",
            algorithm_version=str(current.get("recalc_algorithm_version") or ""),
            solver_config_hash=str(current.get("recalc_solver_config_hash") or ""),
            solver_parameters_json=str(
                current.get("recalc_solver_parameters_json") or "{}"
            ),
            input_basis=str(row.get("input_basis") or "market_mid"),
            input_iv=_finite_float(row.get("partial_iv")),
            implied_volatility=_finite_float(row.get("partial_iv")),
            iv_iterations=0,
            price_residual=_finite_float(row.get("partial_price_residual")),
            model_price=_finite_float(row.get("partial_model_price")),
            delta=None,
            gamma=None,
            theta_per_day=None,
            vega_per_1pct=None,
            rho_per_1pct=None,
            requested_dtype="float64",
            actual_dtype="float64",
            precision_escalated=False,
            quality_flags=int(row.get("quality_flags") or 0),
            time_steps=None,
            space_steps=None,
            damping_steps=None,
            dividend_model=str(row.get("dividend_model") or "continuous_yield"),
            status="failed",
            reason=str(row["reason_code"]),
            failure_stage=str(row["failure_stage"]),
            retry_class=str(row["retry_class"]),
            retry_after=_parse_timestamp(row.get("retry_after")),
            failure_fingerprint=bytes(row["failure_fingerprint"]).hex(),
            result_origin="failed",
        )
        failure_by_symbol.setdefault(symbol, outcome)
        if seed_eligible:
            failure_seed_by_symbol.setdefault(symbol, outcome)
    return {
        symbol: PriorOutcomeCandidates(
            success=success_by_symbol.get(symbol),
            failure=failure_by_symbol.get(symbol),
            failure_seed=failure_seed_by_symbol.get(symbol),
        )
        for symbol in symbols
        if (
            symbol in success_by_symbol
            or symbol in failure_by_symbol
            or symbol in failure_seed_by_symbol
        )
    }


def _surface_failure(
    work: SurfaceWork,
    *,
    expiration_at: datetime,
    price_basis: str,
    scope: str,
    reason: str,
    stage: str,
    calculated_at: datetime,
    node: Any = None,
    input_count: Optional[int] = None,
    accepted_count: Optional[int] = None,
    rejected_count: Optional[int] = None,
    quality_flags: int = 0,
) -> dict[str, Any]:
    identity = (
        f"{work.snapshot_id}|{expiration_at}|{price_basis}|{scope}|"
        f"{getattr(node, 'node_type', '')}|{getattr(node, 'node_value', '')}|{reason}"
    )
    return {
        "snapshot_id": str(work.snapshot_id),
        "source_collected_at": work.source_collected_at,
        "underlying_symbol": work.underlying_symbol,
        "surface_definition_id": work.surface_definition_id,
        "failure_scope": scope,
        "expiration_at": expiration_at,
        "price_basis": price_basis,
        "node_type": getattr(node, "node_type", None),
        "node_value": getattr(node, "node_value", None),
        "put_call": getattr(node, "put_call", None),
        "failure_stage": stage,
        "reason_code": reason,
        "error_class": "SurfaceNumericalFailure",
        "error_detail": reason,
        "retry_class": "deterministic_numerical",
        "retry_after": None,
        "failure_fingerprint": hashlib.sha256(identity.encode()).digest(),
        "input_contract_count": input_count,
        "accepted_contract_count": accepted_count,
        "rejected_contract_count": rejected_count,
        "first_failed_at": calculated_at,
        "last_failed_at": calculated_at,
        "failure_count": 1,
        "updated_at": calculated_at,
        "quality_flags": int(quality_flags),
        "is_active": True,
        "is_required": bool(
            node is not None
            and is_required_surface_node(node.node_type, node.node_value)
        ),
    }


def calculate_surface_publication(
    context: Mapping[str, Any],
    work: SurfaceWork,
    source_rows: list[dict[str, Any]],
    logger: logging.Logger,
    *,
    warm_start_cache: SVIWarmStartCache,
    numerical_runtime: Optional[SurfaceNumericalRuntime] = None,
    solver_registry: Optional[Any] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """Run the existing numerical surface engine and return canonical rows."""
    from algotrading.refdata.options.Calculations import (
        generate_surface_owner_nodes,
        prepare_surface_owner,
        _surface_solver_provenance,
    )

    now = _utc_now()
    projected = []
    for row in source_rows:
        spot = _snapshot_spot(row)
        bid = _finite_float(row.get("bid"))
        ask = _finite_float(row.get("ask"))
        mid = (bid + ask) / 2.0 if bid is not None and ask is not None and ask >= bid else None
        spread = ask - bid if bid is not None and ask is not None and ask >= bid else None
        result_present = row.get("result_iv") is not None
        origin = str(row.get("result_origin") or "source_only")
        projected.append(
            {
                "snapshot_id": str(work.snapshot_id),
                "source_collected_at": work.source_collected_at,
                "option_symbol": str(row["option_symbol"]),
                "underlying_symbol": work.underlying_symbol,
                "option_deliverables_json": row.get("option_deliverables_json", "[]"),
                "put_call": str(row["put_call"]),
                "strike_price": row["strike_price"],
                "expiration_at": row["expiration_at"],
                "exercise_type": row["exercise_type"],
                "multiplier": row["multiplier"],
                "is_mini": row["is_mini"],
                "is_nonstandard": row["is_nonstandard"],
                "chain_is_truncated": row.get("chain_is_truncated", False),
                "bid": bid,
                "ask": ask,
                "source_iv": row.get("source_iv"),
                "calculation_status": "complete",
                "contract_quality_flags": int(row.get("result_quality_flags") or 0),
                "spot_price": spot,
                "mid_price": mid,
                "bid_ask_spread": spread,
                "spread_fraction_of_mid": spread / mid if spread is not None and mid else None,
                "quote_age_seconds": None,
                "underlying_quote_age_seconds": None,
                "forward_price": row["expiration_forward_price"],
                "log_forward_moneyness": (
                    math.log(float(row["strike_price"]) / float(row["expiration_forward_price"]))
                    if row["expiration_forward_price"] and row["strike_price"]
                    else None
                ),
                "time_to_expiry_seconds": row["expiration_time_to_expiry_seconds"],
                "time_to_expiry_years": row["expiration_time_to_expiry_seconds"] / (365.25 * 86400.0),
                "risk_free_rate": row["expiration_risk_free_rate"],
                "dividend_yield": row["expiration_dividend_yield"],
                "recalc_status": "complete" if result_present else "source_only",
                "recalc_result_origin": (
                    "reused" if origin == "reused" else "calculated"
                ),
                "recalc_iv_identifiability_status": (
                    "source_anchor_flat" if row.get("iv_selection") == "source" and result_present else "identified"
                ),
                "recalc_iv": row.get("result_iv"),
                "recalc_price_residual": None,
                "preferred_iv": row.get("effective_iv"),
                "preferred_iv_origin": (
                    "source"
                    if row.get("iv_selection") == "source" or not result_present
                    else "recalculated_reused" if origin == "reused" else "recalculated_current"
                ),
                "preferred_delta": row.get("effective_delta"),
                "preferred_vega": row.get("effective_vega"),
            }
        )
    runtime_context = dict(context)
    runtime_context[OPTIONS_CALCS_CONFIG_PREFIX + "calculation_definition_id"] = (
        work.calculation_definition_id
    )
    runtime_context[OPTIONS_CALCS_CONFIG_PREFIX + "surface_definition_id"] = (
        work.surface_definition_id
    )
    if numerical_runtime is None or solver_registry is None:
        raise ValueError(
            "canonical surface calculation requires process-local numerical state"
        )
    prepared = prepare_surface_owner(
        runtime_context,
        work,
        projected,
        logger,
        warm_start_cache=warm_start_cache,
        numerical_runtime=numerical_runtime,
        solver_registry=solver_registry,
        preloaded_american_inversions={},
    )
    generate_surface_owner_nodes(
        runtime_context, prepared, logger, warm_start_cache=warm_start_cache
    )
    fits: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped_expirations: set[datetime] = set()
    primary_basis = "source" if prepared.price_bases == ("source",) else "mid"
    for expiration, basis, calibration in prepared.entries:
        fit = calibration.fit
        if fit.status.startswith("skipped"):
            if basis == primary_basis:
                skipped_expirations.add(expiration)
            continue
        if not fit.successful:
            failures.append(
                _surface_failure(
                    work,
                    expiration_at=expiration,
                    price_basis=basis,
                    scope="fit",
                    reason=str(fit.error or fit.status),
                    stage=str(fit.failure_stage or "smile_fit"),
                    calculated_at=now,
                    input_count=fit.input_count,
                    accepted_count=len(fit.points),
                    rejected_count=max(0, fit.input_count - len(fit.points)),
                    quality_flags=fit.quality_flags,
                )
            )
            continue
        fit.assert_finite()
        pricing_method, _pricing_version, _solver_hash = _surface_solver_provenance(
            context,
            exercise_type=fit.exercise_type,
            tenor_seconds=fit.tenor_seconds,
            solver_registry=solver_registry,
        )
        typed = typed_fit_columns(
            {"fit_method": fit.method, "parameters": fit.parameters}
        )
        minimum_k, maximum_k = fit.coverage
        total_weight = sum(point.weight for point in fit.points)
        fits.append(
            {
                "snapshot_id": str(work.snapshot_id),
                "source_collected_at": work.source_collected_at,
                "underlying_symbol": work.underlying_symbol,
                "expiration_at": expiration,
                "price_basis": basis,
                "surface_definition_id": work.surface_definition_id,
                "calculated_at": now,
                "pricing_method": pricing_method,
                **typed,
                "input_contract_count": fit.input_count,
                "accepted_contract_count": fit.fit_candidate_contract_count or len(fit.points),
                "rejected_contract_count": max(0, fit.input_count - (fit.fit_candidate_contract_count or len(fit.points))),
                "selected_contract_count": fit.fit_selected_contract_count or len(fit.points),
                "source_current_iv_point_count": sum(point.iv_origin == "recalculated_current" for point in fit.points),
                "source_reused_iv_point_count": sum(point.iv_origin == "recalculated_reused" for point in fit.points),
                "source_vendor_iv_point_count": sum(point.iv_origin == "source" for point in fit.points),
                "no_iv_excluded_count": int(fit.rejected_reasons.get("no_valid_preferred_iv", 0)),
                "source_iv_weight_fraction": (
                    sum(point.weight for point in fit.points if point.iv_origin == "source") / total_weight
                    if total_weight else 0.0
                ),
                "minimum_input_delta": min((point.delta for point in fit.points if point.delta is not None), default=None),
                "maximum_input_delta": max((point.delta for point in fit.points if point.delta is not None), default=None),
                "minimum_log_forward_moneyness": float(minimum_k),
                "maximum_log_forward_moneyness": float(maximum_k),
                "weighted_rmse": float(fit.weighted_rmse),
                "maximum_absolute_error": float(fit.maximum_absolute_error),
                "butterfly_arbitrage_flags": int(fit.butterfly_flags),
                "calendar_arbitrage_flags": int(fit.calendar_flags),
                "quality_flags": int(fit.quality_flags),
                "actual_dtype": str(fit.actual_dtype),
                "precision_escalated": bool(fit.precision_escalated),
                "required_node_status": str(calibration.node_result.required_status),
                "optional_node_status": str(calibration.node_result.optional_status),
            }
        )
        if basis != primary_basis:
            continue
        for node in calibration.node_result.nodes:
            required = is_required_surface_node(node.node_type, node.node_value)
            core = (
                node.fitted_strike,
                node.fitted_iv_mid,
                node.fit_confidence,
            )
            delta_is_valid = (
                node.fitted_delta is not None
                and math.isfinite(float(node.fitted_delta))
            )
            if (
                node.status == "complete"
                and all(
                    value is not None and math.isfinite(float(value))
                    for value in core
                )
                and (node.node_type != "delta" or delta_is_valid)
            ):
                nodes.append(
                    {
                        "snapshot_id": str(work.snapshot_id),
                        "source_collected_at": work.source_collected_at,
                        "underlying_symbol": work.underlying_symbol,
                        "expiration_at": expiration,
                        "price_basis": primary_basis,
                        "surface_definition_id": work.surface_definition_id,
                        "node_type": node.node_type,
                        "node_value": float(node.node_value),
                        "put_call": node.put_call,
                        "fitted_strike": float(node.fitted_strike),
                        "fitted_delta": (
                            float(node.fitted_delta)
                            if node.fitted_delta is not None
                            else None
                        ),
                        "fitted_iv_bid": node.fitted_iv_bid,
                        "fitted_iv_mid": float(node.fitted_iv_mid),
                        "fitted_iv_ask": node.fitted_iv_ask,
                        "fit_confidence": float(node.fit_confidence),
                        "quality_flags": int(node.quality_flags),
                        "calculated_at": now,
                        "is_required": required,
                    }
                )
            else:
                failures.append(
                    _surface_failure(
                        work,
                        expiration_at=expiration,
                        price_basis=primary_basis,
                        scope="node",
                        reason=str(node.node_reason or node.status),
                        stage="node_generation",
                        calculated_at=now,
                        node=node,
                        quality_flags=node.quality_flags,
                    )
                )
    for failed in prepared.failed_entries:
        expiration = _parse_timestamp(failed["expiration_at"])
        failures.append(
            _surface_failure(
                work,
                expiration_at=expiration,
                price_basis=str(failed["price_basis"]),
                scope="fit",
                reason=str(failed.get("fit_error") or "calibration_failure"),
                stage=str(failed.get("failure_stage") or "calibration"),
                calculated_at=now,
                input_count=int(failed.get("input_contract_count") or 0),
                accepted_count=int(failed.get("accepted_contract_count") or 0),
                rejected_count=int(failed.get("rejected_contract_count") or 0),
                quality_flags=int(failed.get("quality_flags") or 0),
            )
        )
    expected_required_nodes = sum(
        bool(row.get("is_required")) for row in (*nodes, *failures)
        if row.get("failure_scope", "node") == "node"
    )
    prepared.timings.values["surface.total_seconds"] = (
        time.perf_counter() - prepared.total_started
    )
    logger.info(
        "event=surface_numerical_summary snapshot_id=%s search_method=%s "
        "validation_method=%s timings=%s node_metrics=%s",
        work.snapshot_id,
        str(
            context.get(
                OPTIONS_CALCS_SURFACE_PREFIX + "node_search_method", "tian"
            )
        ),
        str(
            context.get(
                OPTIONS_CALCS_SURFACE_PREFIX + "node_validation_method",
                "configured",
            )
        ),
        dict(sorted(prepared.timings.values.items())),
        dict(sorted(prepared.american_delta_diagnostics.items())),
    )
    return fits, nodes, failures, len(skipped_expirations), expected_required_nodes


@dataclass
class WorkerHealth:
    worker_alive: bool = True
    terminal_failure: bool = False
    last_error: str = ""
    requested_workers: int = 0
    resolved_capacity: int = 0
    surface_capacity: int = 0
    active_calculations: int = 0
    active_surfaces: int = 0
    queued_calculations: int = 0
    queued_surfaces: int = 0
    completed_calculations: int = 0
    completed_surfaces: int = 0
    failed_calculations: int = 0
    failed_surfaces: int = 0
    claim_conflicts: int = 0
    pool_restarts: int = 0
    observed_worker_pids: set[int] = field(default_factory=set)

    def update(self, **values: Any) -> None:
        for name, value in values.items():
            if hasattr(self, name):
                setattr(self, name, value)


def resolve_worker_capacity(context: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return configured workers, resolved total capacity, and surface cap."""
    prefix = OPTIONS_CALCS_CONFIG_PREFIX
    def integer(key: str, default: int) -> int:
        raw = context.get(key, default)
        if isinstance(raw, bool) or isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"{key} must be an integer")
        try:
            return int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be an integer") from exc

    workers = integer(prefix + "workers", 1)
    global_cap = integer(prefix + "max_inflight", 0)
    configured_surface = integer(prefix + "surface.max_inflight", 0)
    if workers <= 0 or global_cap < 0 or configured_surface < 0:
        raise ValueError("worker counts and in-flight caps must be nonnegative; workers must be positive")
    capacity = min(workers, global_cap if global_cap > 0 else workers)
    surface_capacity = min(
        capacity, configured_surface if configured_surface > 0 else capacity
    )
    if capacity < 1:
        raise ValueError("resolved worker capacity must be at least one")
    return workers, capacity, surface_capacity


class OptionsChainWorker:
    """Fair spawn-based coordinator for generation-3 calculation work."""

    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        logger: Optional[logging.Logger] = None,
        client: Any = None,
    ):
        self.context = dict(context)
        validate_options_calcs_configuration(self.context)
        self.surface_cadence_multiplier = int(
            self.context.get(
                OPTIONS_CALCS_SURFACE_PREFIX + "cadence_multiplier", 1
            )
        )
        self.surface_max_minutes_for_calculation = int(
            self.context.get(
                OPTIONS_CALCS_SURFACE_PREFIX + "max_minutes_for_calculation",
                120,
            )
        )
        self.logger = logger or logging.getLogger(__name__)
        self.client = client or create_clickhouse_client(context)
        self.database = validate_database_identifier(
            str(context["options_chain.clickhouse.database"])
        )
        self.stop_event = threading.Event()
        self.health = WorkerHealth()
        self._monitor: Optional[Any] = None
        self._last_health_report_monotonic = 0.0
        self._pool: Optional[LocalWorkerPool] = None
        self._active: dict[Any, ScheduledCalculationTask | ScheduledSurfaceTask] = {}
        self._calculation_queue: deque[CalculationWork] = deque()
        self._surface_queue: deque[SurfaceWork] = deque()
        self._scheduled_calculations: set[uuid.UUID] = set()
        self._scheduled_surfaces: set[uuid.UUID] = set()
        self._next_stage = "calculation"
        self._pool_factory = LocalWorkerPool
        self._pool_restarts = 0
        self._stop_requested_at: Optional[float] = None
        self._stop_signal: Optional[str] = None
        self._shutdown_logged = False
        self._shutdown_grace_logged = False
        self._shutdown_forced = False
        self._shutdown_completed = 0
        self._shutdown_complete_logged = False
        self._abandoned_calculations = 0
        self._abandoned_surfaces = 0
        self._closed = False
        self._claim_conflict_only = False
        self._surface_age_warning_checked = False
        # The complete read/write schema contract is checked before creating
        # repositories, resolving definitions, starting monitoring, or
        # discovering work.  This remains mandatory when create_schema=false.
        from algotrading.refdata.options.Calculations import calculations_preflight

        calculations_preflight(self.context, self.client, logger=self.logger)
        workers, capacity, surface_capacity = resolve_worker_capacity(self.context)
        self.health.update(
            requested_workers=workers, resolved_capacity=capacity,
            surface_capacity=surface_capacity,
        )
        worker_id = str(self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "worker_id", "worker"))
        stale = int(self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "running_stale_seconds", 900))
        self.calculations = CalculationRepository(
            self.client, self.database, worker_id=worker_id, stale_after_seconds=stale
        )
        self.surfaces = SurfaceRepository(
            self.client, self.database, worker_id=worker_id, stale_after_seconds=stale
        )
        ensure_processing_definitions(self.context, self.client, self.database)
        self.logger.info(
            "event=scheduler_configuration workers=%s max_inflight=%s resolved_capacity=%s "
            "surface_max_inflight=%s resolved_surface_capacity=%s contract_pending_batch_size=%s "
            "surface_pending_batch_size=%s surface_cadence_multiplier=%s "
            "surface_max_minutes_for_calculation=%s "
            "cpu_threads_per_process=%s max_tasks_per_child=%s shutdown_timeout_seconds=%s",
            workers,
            self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "max_inflight", 0),
            capacity,
            self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "surface.max_inflight", 0),
            surface_capacity,
            self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "contract.pending_batch_size", 100),
            self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "surface.pending_batch_size", 16),
            self.surface_cadence_multiplier,
            self.surface_max_minutes_for_calculation,
            self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "worker.cpu_threads_per_process", 1),
            self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "worker.max_tasks_per_child", 0),
            self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "worker.shutdown_timeout_seconds", 60),
        )

    def request_stop(self, *_args: Any) -> None:
        """Record a stop request without doing ClickHouse or pool work.

        This method is registered directly as the SIGINT/SIGTERM handler, so
        the coordinator loop performs all writes, joins, and logging later.
        Repeated signals intentionally collapse to the first request.
        """
        if self._stop_requested_at is None:
            self._stop_requested_at = time.monotonic()
            signum = _args[0] if _args else None
            if signum is not None:
                try:
                    self._stop_signal = signal.Signals(signum).name
                except (TypeError, ValueError):
                    self._stop_signal = f"SIGNAL_{signum}"
        self.stop_event.set()

    def _report_health(self, *, final: bool = False) -> None:
        if self._pool is not None:
            for process in getattr(
                getattr(self._pool, "_executor", None), "_processes", {}
            ).values():
                pid = getattr(process, "pid", None)
                if pid:
                    self.health.observed_worker_pids.add(int(pid))
        self.health.update(
            active_calculations=sum(
                isinstance(task, ScheduledCalculationTask)
                for task in self._active.values()
            ),
            active_surfaces=sum(
                isinstance(task, ScheduledSurfaceTask) for task in self._active.values()
            ),
            queued_calculations=len(self._calculation_queue),
            queued_surfaces=len(self._surface_queue),
        )
        calculation_health = (0, None, 0, [])
        surface_health = None
        try:
            surface_enabled = _as_bool(
                self.context.get(OPTIONS_CALCS_SURFACE_PREFIX + "enabled", True)
            )
            rows = self.client.query(
                f"""SELECT countIf(isNull(c.snapshot_id) OR c.status NOT IN
                ('complete', 'complete_with_failures')), minIf(s.source_collected_at,
                isNull(c.snapshot_id) OR c.status NOT IN ('complete', 'complete_with_failures')),
                countIf(c.status = 'failed'), groupUniqArrayIf(c.calculation_definition_id,
                NOT isNull(c.snapshot_id))
                FROM {self.database}.options_chain_snapshots AS s
                LEFT JOIN (SELECT * FROM {self.database}.options_chain_calculation_state FINAL) AS c
                  ON c.snapshot_id = s.snapshot_id
                 AND c.source_collected_at = s.source_collected_at
                 AND c.is_active""",
            ).result_rows
            if rows:
                calculation_health = rows[0]
            surface_health = query_surface_cadence_health(
                self.client,
                self.database,
                cadence_multiplier=self.surface_cadence_multiplier,
                surface_enabled=surface_enabled,
            )
        except Exception:
            self.logger.warning("event=health_query_failed", exc_info=True)
        calc_pending, oldest, calc_failed, calc_defs = calculation_health
        surf_pending = surface_health.pending if surface_health else 0
        surf_failed = surface_health.failed if surface_health else 0
        surf_defs = surface_health.definition_ids if surface_health else ()
        oldest_at = _parse_timestamp(oldest)
        oldest_age = max(0.0, (_utc_now() - oldest_at).total_seconds()) if oldest_at else 0.0
        lag_limit = float(self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "max_calculation_lag_seconds", 600))
        warning = bool(calc_failed or surf_failed) or bool(calc_pending and oldest_age > lag_limit)
        if self.health.terminal_failure:
            level, monitor_level = logging.ERROR, NagiosSystemMonitor.MONITOR_CRITICAL
        elif warning:
            level, monitor_level = logging.WARNING, NagiosSystemMonitor.MONITOR_WARNING
        else:
            level, monitor_level = logging.INFO, NagiosSystemMonitor.MONITOR_OK
        message = (
            f"final={final} calculation_pending={int(calc_pending or 0)} "
            f"surface_pending={int(surf_pending or 0)} "
            f"surface_cadence_multiplier={self.surface_cadence_multiplier} "
            f"active_calculations={self.health.active_calculations} "
            f"active_surfaces={self.health.active_surfaces} queued_calculations={self.health.queued_calculations} "
            f"queued_surfaces={self.health.queued_surfaces} requested_workers={self.health.requested_workers} "
            f"resolved_capacity={self.health.resolved_capacity} surface_capacity={self.health.surface_capacity} "
            f"observed_worker_pids={sorted(self.health.observed_worker_pids)} "
            f"completed_calculations={self.health.completed_calculations} completed_surfaces={self.health.completed_surfaces} "
            f"failed_calculations={self.health.failed_calculations} failed_surfaces={self.health.failed_surfaces} "
            f"claim_conflicts={self.health.claim_conflicts} pool_restarts={self.health.pool_restarts} "
            f"error={self.health.last_error}"
        )
        self.logger.log(level, "event=worker_pool_status %s", message)
        try:
            if self._monitor is None:
                self._monitor = NagiosSystemMonitor(self.context)
            self._monitor.report_event(monitor_level, message)
        except Exception:
            self.logger.warning("canonical_worker_monitor_failure", exc_info=True)

    def _discover(
        self,
        *,
        stage: str,
        snapshot_id: Optional[uuid.UUID],
        start_at: Optional[datetime],
        end_at: Optional[datetime],
        symbols: Sequence[str],
        retry_failed: bool,
    ) -> list[CalculationWork] | list[SurfaceWork]:
        stale = int(self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "running_stale_seconds", 900))
        common = dict(
            client=self.client, database=self.database, stale_after_seconds=stale,
            snapshot_id=snapshot_id, start_at=start_at, end_at=end_at,
            symbols=symbols, retry_failed=retry_failed,
        )
        if stage == "calculation":
            common["limit"] = int(self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "contract.pending_batch_size", 100))
            return discover_calculation_work(**common)
        common["limit"] = int(self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "surface.pending_batch_size", 16))
        common["cadence_multiplier"] = self.surface_cadence_multiplier
        # An explicit snapshot is an operator-directed request and retains its
        # existing ability to process retained history.  The age limit protects
        # the automatic backlog that can appear after reducing cadence.
        if snapshot_id is None:
            common["max_minutes_for_calculation"] = getattr(
                self,
                "surface_max_minutes_for_calculation",
                int(
                    self.context.get(
                        OPTIONS_CALCS_SURFACE_PREFIX
                        + "max_minutes_for_calculation",
                        120,
                    )
                ),
            )
        return discover_surface_work(**common)

    def _warn_expired_surface_work(
        self,
        *,
        start_at: Optional[datetime],
        end_at: Optional[datetime],
        symbols: Sequence[str],
        retry_failed: bool,
    ) -> None:
        """Log one startup warning for the automatic backlog age guard."""
        if getattr(self, "_surface_age_warning_checked", False):
            return
        self._surface_age_warning_checked = True
        max_minutes = getattr(
            self,
            "surface_max_minutes_for_calculation",
            int(
                self.context.get(
                    OPTIONS_CALCS_SURFACE_PREFIX + "max_minutes_for_calculation",
                    120,
                )
            ),
        )
        try:
            report = query_expired_surface_work(
                self.client,
                self.database,
                max_minutes_for_calculation=max_minutes,
                start_at=start_at,
                end_at=end_at,
                symbols=symbols,
                retry_failed=retry_failed,
                cadence_multiplier=self.surface_cadence_multiplier,
            )
        except Exception:
            self.logger.warning(
                "event=surface_age_warning_query_failed", exc_info=True
            )
            return
        if report is None:
            return
        self.logger.warning(
            "event=surface_work_skipped_max_age skipped_count=%s "
            "max_minutes_for_calculation=%s newest_snapshot_id=%s "
            "newest_symbol=%s newest_source_collected_at=%s",
            report.skipped_count,
            max_minutes,
            report.newest_snapshot_id,
            report.newest_underlying_symbol,
            report.newest_source_collected_at.isoformat(),
        )

    def _enqueue(self, items: Sequence[CalculationWork] | Sequence[SurfaceWork], *, stage: str) -> int:
        queue = self._calculation_queue if stage == "calculation" else self._surface_queue
        scheduled = self._scheduled_calculations if stage == "calculation" else self._scheduled_surfaces
        added = 0
        for work in items:
            if work.snapshot_id in scheduled:
                continue
            scheduled.add(work.snapshot_id)
            queue.append(work)
            added += 1
        return added

    def _submit_one(self, stage: str) -> bool:
        if getattr(self, "stop_event", None) is not None and self.stop_event.is_set():
            return False
        if self._pool is None:
            raise RuntimeError("worker pool is not initialized")
        if stage == "surface" and sum(isinstance(t, ScheduledSurfaceTask) for t in self._active.values()) >= self.health.surface_capacity:
            return False
        if stage == "calculation":
            if not self._calculation_queue:
                return False
            work = self._calculation_queue.popleft()
            task: ScheduledCalculationTask | ScheduledSurfaceTask = ScheduledCalculationTask(work, uuid.uuid4(), _utc_now())
            future = self._pool.submit_calculation(task)
        else:
            if not self._surface_queue:
                return False
            work = self._surface_queue.popleft()
            task = ScheduledSurfaceTask(work, uuid.uuid4(), _utc_now())
            future = self._pool.submit_surface(task)
        self._active[future] = task
        active_surfaces = sum(isinstance(t, ScheduledSurfaceTask) for t in self._active.values())
        self.logger.info(
            "event=work_submitted process_type=%s snapshot_id=%s symbol=%s attempt_id=%s active_total=%s active_surfaces=%s queued_surfaces=%s",
            stage, work.snapshot_id, work.underlying_symbol, task.attempt_id,
            len(self._active), active_surfaces, len(self._surface_queue),
        )
        self._next_stage = "surface" if stage == "calculation" else "calculation"
        return True

    def _submit_available(self) -> None:
        while (
            len(self._active) < self.health.resolved_capacity
            and not (
                getattr(self, "stop_event", None) is not None
                and self.stop_event.is_set()
            )
        ):
            calc_ready = bool(self._calculation_queue)
            surface_ready = bool(self._surface_queue) and (
                sum(isinstance(t, ScheduledSurfaceTask) for t in self._active.values()) < self.health.surface_capacity
            )
            if not calc_ready and not surface_ready:
                return
            active_calc = sum(isinstance(t, ScheduledCalculationTask) for t in self._active.values())
            active_surface = sum(isinstance(t, ScheduledSurfaceTask) for t in self._active.values())
            if calc_ready and surface_ready and self.health.resolved_capacity >= 2:
                if active_calc == 0:
                    stage = "calculation"
                elif active_surface == 0:
                    stage = "surface"
                else:
                    stage = self._next_stage
            elif surface_ready:
                stage = "surface"
            else:
                stage = "calculation"
            if not self._submit_one(stage):
                return

    def _startup_recovery(
        self,
        *,
        snapshot_id: Optional[uuid.UUID],
        start_at: Optional[datetime],
        end_at: Optional[datetime],
        symbols: Sequence[str],
        contract_only: bool,
        surface_only: bool,
        surface_enabled: bool,
    ) -> None:
        """Reclaim only interrupted owners before the first pool submission.

        The recovery queries are deliberately unbounded by the normal
        reconcile window.  They select only running/abandoned/stale-running
        state, so old genuine failed work and the unrelated historical
        backlog remain governed by normal discovery and retry policy.
        """
        if contract_only and surface_only:
            return
        if maintenance_lease_active(self.client, self.database):
            raise RuntimeError(
                "Startup recovery is blocked by an active schema-migration lease"
            )
        if self.stop_event.is_set():
            return
        calculation_recovered = 0
        calculation_abandoned = 0
        surface_recovered = 0
        surface_abandoned = 0
        if not self.stop_event.is_set() and not surface_only:
            calculations = discover_interrupted_calculation_work(
                self.client,
                self.database,
                snapshot_id=snapshot_id,
                start_at=start_at,
                end_at=end_at,
                symbols=symbols,
            )
            for recovered in calculations:
                if self.stop_event.is_set():
                    break
                if recovered.status == "running":
                    if recovered.attempt_id is None:
                        raise RuntimeError(
                            f"Running calculation {recovered.work.snapshot_id} has no active attempt"
                        )
                    changed = self.calculations.abandon_attempt_if_running(
                        snapshot_id=recovered.work.snapshot_id,
                        attempt_id=recovered.attempt_id,
                        reason="startup_recovery",
                    )
                    if not changed:
                        # A terminal child publication won a race between the
                        # recovery read and ownership write.  Do not enqueue
                        # an owner that is no longer retryable.
                        continue
                    calculation_abandoned += 1
                calculation_recovered += 1
                self._enqueue([recovered.work], stage="calculation")
        if not self.stop_event.is_set() and not contract_only and surface_enabled:
            surfaces = discover_interrupted_surface_work(
                self.client,
                self.database,
                snapshot_id=snapshot_id,
                start_at=start_at,
                end_at=end_at,
                symbols=symbols,
            )
            for recovered in surfaces:
                if self.stop_event.is_set():
                    break
                if recovered.status == "running":
                    if recovered.attempt_id is None:
                        raise RuntimeError(
                            f"Running surface {recovered.work.snapshot_id} has no active attempt"
                        )
                    changed = self.surfaces.abandon_attempt_if_running(
                        snapshot_id=recovered.work.snapshot_id,
                        attempt_id=recovered.attempt_id,
                        reason="startup_recovery",
                    )
                    if not changed:
                        continue
                    surface_abandoned += 1
                surface_recovered += 1
                self._enqueue([recovered.work], stage="surface")
        self.logger.info(
            "event=startup_recovery calculation_running_abandoned=%s "
            "calculation_retryable=%s surface_running_abandoned=%s "
            "surface_retryable=%s",
            calculation_abandoned,
            calculation_recovered,
            surface_abandoned,
            surface_recovered,
        )

    def _abandon_task(
        self,
        task: ScheduledCalculationTask | ScheduledSurfaceTask,
        reason: str,
    ) -> bool:
        """Reconcile coordinator-side interruption as retryable abandonment."""
        if isinstance(task, ScheduledCalculationTask):
            changed = self.calculations.abandon_attempt_if_running(
                snapshot_id=task.work.snapshot_id,
                attempt_id=task.attempt_id,
                reason=reason,
            )
            process_type = "calculation"
            if changed:
                self._abandoned_calculations = getattr(
                    self, "_abandoned_calculations", 0
                ) + 1
        else:
            changed = self.surfaces.abandon_attempt_if_running(
                snapshot_id=task.work.snapshot_id,
                attempt_id=task.attempt_id,
                reason=reason,
            )
            process_type = "surface"
            if changed:
                self._abandoned_surfaces = getattr(
                    self, "_abandoned_surfaces", 0
                ) + 1
        if changed:
            self.logger.info(
                "event=attempt_abandoned process_type=%s snapshot_id=%s "
                "attempt_id=%s reason=%s",
                process_type, task.work.snapshot_id, task.attempt_id, reason,
            )
        return changed

    def _abandon_tasks(
        self,
        tasks: Sequence[ScheduledCalculationTask | ScheduledSurfaceTask],
        reason: str,
    ) -> None:
        for task in tasks:
            try:
                self._abandon_task(task, reason)
            except BaseException as exc:
                self.health.update(last_error=str(exc))
                self.logger.error(
                    "event=attempt_abandonment_failed process_type=%s "
                    "snapshot_id=%s attempt_id=%s",
                    "calculation" if isinstance(task, ScheduledCalculationTask) else "surface",
                    task.work.snapshot_id,
                    task.attempt_id,
                    exc_info=True,
                )

    def _reconcile_task(self, task: ScheduledCalculationTask | ScheduledSurfaceTask, error: BaseException | str) -> None:
        """Compatibility wrapper: future failures are infrastructure loss."""
        try:
            reason = "worker_process_lost"
            if getattr(self, "_stop_signal", None):
                reason = f"signal_{self._stop_signal}"
            elif str(error) == "coordinator shutdown":
                reason = "coordinator_shutdown"
            self._abandon_task(task, reason)
        except BaseException:
            self.logger.error(
                "event=attempt_abandonment_failed process_type=%s snapshot_id=%s attempt_id=%s",
                "calculation" if isinstance(task, ScheduledCalculationTask) else "surface",
                task.work.snapshot_id, task.attempt_id, exc_info=True,
            )

    def _consume_future(self, future: Any) -> None:
        task = self._active.pop(future)
        scheduled = (
            self._scheduled_calculations
            if isinstance(task, ScheduledCalculationTask)
            else self._scheduled_surfaces
        )
        scheduled.discard(task.work.snapshot_id)
        try:
            value = future.result()
        except BaseException as exc:
            if isinstance(exc, BrokenProcessPool):
                # Keep this task visible until the pool has been terminated;
                # abandonment must never race a surviving child publisher.
                self._active[future] = task
                raise
            self._reconcile_task(task, exc)
            self.health.update(last_error=str(exc))
            if getattr(self, "stop_event", threading.Event()).is_set() and not getattr(self, "_shutdown_forced", False):
                self._shutdown_completed += 1
            return
        if not isinstance(value, WorkerTaskResult):
            self._abandon_task(task, "worker_process_lost")
            self.health.update(last_error=f"Invalid worker result: {value!r}")
            return
        self.health.observed_worker_pids.add(int(value.worker_pid))
        stage = "calculation" if value.process_type == "calculation" else "surface"
        if value.status == "claim_conflict":
            self.health.claim_conflicts += 1
            self._claim_conflict_only = True
            self.logger.info(
                "event=claim_conflict process_type=%s snapshot_id=%s attempt_id=%s",
                stage, value.snapshot_id, value.attempt_id,
            )
        elif value.status == "complete":
            self._claim_conflict_only = False
            self.health.update(**{f"completed_{stage}s": getattr(self.health, f"completed_{stage}s") + 1})
        else:
            self.health.update(
                **{f"failed_{stage}s": getattr(self.health, f"failed_{stage}s") + 1,
                   "last_error": value.error_message}
            )
        self.logger.info(
            "event=work_completed process_type=%s snapshot_id=%s attempt_id=%s pid=%s process_name=%s status=%s execution_seconds=%.6f queue_seconds=%.6f",
            stage, value.snapshot_id, value.attempt_id, value.worker_pid,
            value.worker_process_name, value.status,
            max(
                0.0,
                (value.finished_at - value.started_at).total_seconds(),
            )
            if value.started_at
            else 0.0,
            max(0.0, (value.started_at - value.submitted_at).total_seconds()) if value.started_at else 0.0,
        )
        if getattr(self, "stop_event", threading.Event()).is_set() and not getattr(self, "_shutdown_forced", False):
            self._shutdown_completed += 1

    def _new_pool(self) -> None:
        self._pool = self._pool_factory(self.context, self.logger)
        self.logger.info(
            "event=worker_pool_started requested_workers=%s total_capacity=%s surface_capacity=%s start_method=spawn",
            self.health.requested_workers, self.health.resolved_capacity,
            self.health.surface_capacity,
        )

    def _log_shutdown_requested(self) -> None:
        if self._shutdown_logged:
            return
        self._shutdown_logged = True
        self.logger.info(
            "event=shutdown_requested signal=%s active_tasks=%s "
            "queued_calculations=%s queued_surfaces=%s",
            self._stop_signal or "REQUESTED",
            len(self._active),
            len(self._calculation_queue),
            len(self._surface_queue),
        )

    def _log_shutdown_grace_complete(self) -> None:
        if self._shutdown_grace_logged:
            return
        self._shutdown_grace_logged = True
        self.logger.info(
            "event=shutdown_grace_complete completed_during_shutdown=%s "
            "remaining_active=%s",
            self._shutdown_completed,
            len(self._active),
        )

    def _log_shutdown_complete(self) -> None:
        if self._shutdown_complete_logged:
            return
        self._shutdown_complete_logged = True
        self.logger.info(
            "event=shutdown_complete abandoned_calculations=%s "
            "abandoned_surfaces=%s",
            self._abandoned_calculations,
            self._abandoned_surfaces,
        )

    def _drain_completed_futures(self) -> None:
        """Consume futures already terminal before forced pool termination."""
        done = [
            future for future in tuple(self._active)
            if bool(getattr(future, "done", lambda: False)())
        ]
        for future in done:
            try:
                self._consume_future(future)
            except BrokenProcessPool:
                # The caller will stop the pool before abandoning any owner.
                return

    def _force_shutdown_active(
        self,
        *,
        reason: str,
        timeout: float,
    ) -> None:
        """Terminate children before abandoning the remaining owned attempts."""
        self._drain_completed_futures()
        remaining = list(self._active.values())
        self.logger.warning(
            "event=shutdown_timeout timeout_seconds=%s remaining_active=%s",
            timeout,
            len(remaining),
        )
        self._terminate_pool_before_abandon(remaining, reason)
        self._shutdown_forced = True
        self._active.clear()
        self._scheduled_calculations.clear()
        self._scheduled_surfaces.clear()
        self._calculation_queue.clear()
        self._surface_queue.clear()
        self._log_shutdown_complete()

    def _terminate_pool_before_abandon(
        self,
        tasks: Sequence[ScheduledCalculationTask | ScheduledSurfaceTask],
        reason: str,
    ) -> None:
        """Stop all child publishers before changing their ownership state."""
        # This ordering is intentional: a child must not be able to publish
        # after the state has been changed to abandoned.
        pool = self._pool
        self._pool = None
        if pool is not None:
            pool.close(wait=False, cancel_futures=True)
        self._abandon_tasks(tasks, reason)

    def _continue_after_pool_recovery(
        self,
        *,
        once: bool,
        snapshot_id: Optional[uuid.UUID],
        symbols: Sequence[str],
        start_at: Optional[datetime],
        end_at: Optional[datetime],
        retry_failed: bool,
        contract_only: bool,
        surface_only: bool,
        surface_enabled: bool,
    ) -> None:
        """Continue draining/replenishing work after replacing a broken pool."""
        poll_interval = float(
            self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "poll_interval_seconds", 10)
        )
        while True:
            if self.stop_event.is_set() and not self._active:
                self._log_shutdown_grace_complete()
                return
            if self.stop_event.is_set() and self._stop_requested_at is not None:
                timeout = float(
                    self.context.get(
                        OPTIONS_CALCS_CONFIG_PREFIX
                        + "worker.shutdown_timeout_seconds",
                        60,
                    )
                )
                if time.monotonic() - self._stop_requested_at >= timeout:
                    reason = f"signal_{self._stop_signal}" if self._stop_signal else "coordinator_shutdown"
                    self._force_shutdown_active(
                        reason=reason,
                        timeout=timeout,
                    )
                    return
            if not self.stop_event.is_set():
                self._submit_available()
            if self._active:
                done, _ = wait(
                    tuple(self._active), timeout=0.5, return_when=FIRST_COMPLETED
                )
                for future in done:
                    self._consume_future(future)
                now = time.monotonic()
                monitor_interval = float(
                    self.context.get(
                        OPTIONS_CALCS_CONFIG_PREFIX + "monitor_interval", 300
                    )
                )
                if now - self._last_health_report_monotonic >= monitor_interval:
                    self._report_health()
                    self._last_health_report_monotonic = now
                continue
            if once:
                return
            if self._claim_conflict_only:
                self._claim_conflict_only = False
                self.stop_event.wait(poll_interval)
                continue
            added = 0
            if not surface_only:
                added += self._enqueue(
                    self._discover(
                        stage="calculation", snapshot_id=snapshot_id,
                        start_at=start_at, end_at=end_at, symbols=symbols,
                        retry_failed=retry_failed,
                    ),
                    stage="calculation",
                )
            if not contract_only and surface_enabled:
                added += self._enqueue(
                    self._discover(
                        stage="surface", snapshot_id=snapshot_id,
                        start_at=start_at, end_at=end_at, symbols=symbols,
                        retry_failed=retry_failed,
                    ),
                    stage="surface",
                )
            if not added:
                self.stop_event.wait(poll_interval)

    def run(
        self,
        *,
        once: bool = False,
        snapshot_id: Optional[uuid.UUID] = None,
        symbols: Optional[Sequence[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        retry_failed: Optional[bool] = None,
        contract_only: bool = False,
        surface_only: bool = False,
    ) -> None:
        retry = bool(retry_failed) if retry_failed is not None else _as_bool(
            self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "retry_failed", False)
        )
        selected_symbols = tuple(sorted({value.upper() for value in symbols or ()}))
        if snapshot_id is not None:
            start_at = end_at = None
        else:
            lower_date = from_date or (
                _utc_now().date()
                - timedelta(days=int(self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "reconcile_days", 7)))
            )
            start_at = datetime.combine(lower_date, date_time.min, tzinfo=UTC)
            end_at = datetime.combine(to_date + timedelta(days=1), date_time.min, tzinfo=UTC) if to_date else None
        surface_enabled = _as_bool(self.context.get(OPTIONS_CALCS_SURFACE_PREFIX + "enabled", True))
        recovery_start_at = (
            datetime.combine(from_date, date_time.min, tzinfo=UTC)
            if from_date is not None else None
        )
        recovery_end_at = (
            datetime.combine(to_date + timedelta(days=1), date_time.min, tzinfo=UTC)
            if to_date is not None else None
        )
        if not contract_only and surface_enabled and snapshot_id is None:
            self._warn_expired_surface_work(
                start_at=start_at,
                end_at=end_at,
                symbols=selected_symbols,
                retry_failed=retry,
            )
        self._startup_recovery(
            snapshot_id=snapshot_id,
            start_at=recovery_start_at if snapshot_id is None else None,
            end_at=recovery_end_at if snapshot_id is None else None,
            symbols=selected_symbols,
            contract_only=contract_only,
            surface_only=surface_only,
            surface_enabled=surface_enabled,
        )
        self._new_pool()
        initial_discovered = False
        try:
            while True:
                if self.stop_event.is_set():
                    self._log_shutdown_requested()
                if not initial_discovered:
                    if not self.stop_event.is_set() and not surface_only:
                        self._enqueue(self._discover(stage="calculation", snapshot_id=snapshot_id, start_at=start_at, end_at=end_at, symbols=selected_symbols, retry_failed=retry), stage="calculation")
                    if not self.stop_event.is_set() and not contract_only and surface_enabled:
                        self._enqueue(self._discover(stage="surface", snapshot_id=snapshot_id, start_at=start_at, end_at=end_at, symbols=selected_symbols, retry_failed=retry), stage="surface")
                    initial_discovered = True
                if not self.stop_event.is_set():
                    self._submit_available()
                if self.stop_event.is_set() and not self._active:
                    self._log_shutdown_grace_complete()
                    break
                if self.stop_event.is_set() and self._stop_requested_at is not None:
                    timeout = float(
                        self.context.get(
                            OPTIONS_CALCS_CONFIG_PREFIX
                            + "worker.shutdown_timeout_seconds",
                            60,
                        )
                    )
                    if time.monotonic() - self._stop_requested_at >= timeout:
                        reason = f"signal_{self._stop_signal}" if self._stop_signal else "coordinator_shutdown"
                        self._force_shutdown_active(
                            reason=reason,
                            timeout=timeout,
                        )
                        break
                if not self._active:
                    if once:
                        break
                    if self._claim_conflict_only:
                        self._claim_conflict_only = False
                        self.stop_event.wait(
                            float(
                                self.context.get(
                                    OPTIONS_CALCS_CONFIG_PREFIX
                                    + "poll_interval_seconds",
                                    10,
                                )
                            )
                        )
                        continue
                    # Refill only after the current bounded queue is drained.
                    added = 0
                    if not surface_only:
                        added += self._enqueue(self._discover(stage="calculation", snapshot_id=snapshot_id, start_at=start_at, end_at=end_at, symbols=selected_symbols, retry_failed=retry), stage="calculation")
                    if not contract_only and surface_enabled:
                        added += self._enqueue(self._discover(stage="surface", snapshot_id=snapshot_id, start_at=start_at, end_at=end_at, symbols=selected_symbols, retry_failed=retry), stage="surface")
                    if added:
                        continue
                    self.stop_event.wait(float(self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "poll_interval_seconds", 10)))
                    continue
                done, _ = wait(tuple(self._active), timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    self._consume_future(future)
                now = time.monotonic()
                if now - self._last_health_report_monotonic >= float(self.context.get(OPTIONS_CALCS_CONFIG_PREFIX + "monitor_interval", 300)):
                    self._report_health()
                    self._last_health_report_monotonic = now
                if not once and not self.stop_event.is_set():
                    if len(self._calculation_queue) < self.health.resolved_capacity and not surface_only:
                        self._enqueue(self._discover(stage="calculation", snapshot_id=snapshot_id, start_at=start_at, end_at=end_at, symbols=selected_symbols, retry_failed=retry), stage="calculation")
                    if len(self._surface_queue) < self.health.resolved_capacity and not contract_only and surface_enabled:
                        self._enqueue(self._discover(stage="surface", snapshot_id=snapshot_id, start_at=start_at, end_at=end_at, symbols=selected_symbols, retry_failed=retry), stage="surface")
        except BrokenProcessPool as exc:
            self._pool_restarts += 1
            self.health.update(pool_restarts=self._pool_restarts, last_error=str(exc))
            log = self.logger.warning if self.stop_event.is_set() else self.logger.error
            log("event=worker_pool_broken error=%s", exc, exc_info=True)
            broken_tasks = list(self._active.values())
            self._terminate_pool_before_abandon(broken_tasks, "worker_process_lost")
            for task in broken_tasks:
                if isinstance(task, ScheduledCalculationTask):
                    self._scheduled_calculations.discard(task.work.snapshot_id)
                else:
                    self._scheduled_surfaces.discard(task.work.snapshot_id)
            self._active.clear()
            if self.stop_event.is_set():
                self._log_shutdown_requested()
                self._log_shutdown_complete()
                return
            if self._pool_restarts <= 1:
                try:
                    self._new_pool()
                except BaseException as recovery_error:
                    self.health.update(
                        terminal_failure=True, last_error=str(recovery_error)
                    )
                    self.logger.critical(
                        "event=worker_pool_recovery_failed", exc_info=True
                    )
                    raise
                for task in broken_tasks:
                    self._enqueue(
                        [task.work],
                        stage="calculation"
                        if isinstance(task, ScheduledCalculationTask)
                        else "surface",
                    )
                if not once:
                    if not surface_only:
                        self._enqueue(self._discover(stage="calculation", snapshot_id=snapshot_id, start_at=start_at, end_at=end_at, symbols=selected_symbols, retry_failed=retry), stage="calculation")
                    if not contract_only and surface_enabled:
                        self._enqueue(self._discover(stage="surface", snapshot_id=snapshot_id, start_at=start_at, end_at=end_at, symbols=selected_symbols, retry_failed=retry), stage="surface")
                self._continue_after_pool_recovery(
                    once=once,
                    snapshot_id=snapshot_id,
                    symbols=selected_symbols,
                    start_at=start_at,
                    end_at=end_at,
                    retry_failed=retry,
                    contract_only=contract_only,
                    surface_only=surface_only,
                    surface_enabled=surface_enabled,
                )
            else:
                self.health.update(terminal_failure=True)
                raise
        finally:
            remaining = list(self._active.values())
            if remaining:
                reason = (
                    f"signal_{self._stop_signal}"
                    if self._stop_signal
                    else "coordinator_shutdown"
                )
                self._terminate_pool_before_abandon(remaining, reason)
                self._active.clear()
            elif self._pool is not None:
                pool = self._pool
                self._pool = None
                pool.close(
                    wait=not self.stop_event.is_set(),
                    cancel_futures=self.stop_event.is_set(),
                )
            self._scheduled_calculations.clear()
            self._scheduled_surfaces.clear()
            self._calculation_queue.clear()
            self._surface_queue.clear()
            self.health.update(worker_alive=False)
            if self.stop_event.is_set() and not self._shutdown_forced:
                self._log_shutdown_grace_complete()
                self._log_shutdown_complete()
            self._report_health(final=True)

    def close(self) -> None:
        if self._closed:
            return
        self.request_stop()
        remaining = list(self._active.values())
        if remaining:
            self._terminate_pool_before_abandon(
                remaining,
                f"signal_{self._stop_signal}" if self._stop_signal else "coordinator_shutdown",
            )
            self._active.clear()
            self._scheduled_calculations.clear()
            self._scheduled_surfaces.clear()
        elif self._pool is not None:
            pool = self._pool
            self._pool = None
            pool.close(wait=False, cancel_futures=True)
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        self._closed = True


__all__ = [
    "OptionsChainWorker",
    "WorkerHealth",
    "resolve_worker_capacity",
    "calculate_surface_publication",
    "compose_calculation_publication",
    "ensure_processing_definitions",
    "load_prior_outcomes",
]
