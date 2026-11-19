"""Supported direct-table query contract for normalized option-chain data.

This module owns the effective-value rules used by the GUI, Surface,
reporting, migration validation, and operator tooling.  Callers must not copy
the CASE expressions or implement their own result/source fallback policy.
"""

from __future__ import annotations

import logging
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from algotrading.refdata.options.Common import (
    _datetime64_microseconds,
    _finite_float,
    validate_database_identifier,
)


TERMINAL_STATES = ("complete", "complete_with_failures")


# This is the single source projection contract shared by the numerical
# calculator and the persistence repository.  The left-hand names are the
# canonical normalized-table names; callers may request the numerical aliases
# for the two deliberately internal names below.  Keeping the expressions in
# one place prevents qualified ClickHouse expressions from leaking driver-
# dependent column names into application code.
CALCULATION_SOURCE_FIELD_EXPRESSIONS = (
    ("snapshot_id", "toString(s.snapshot_id)"),
    ("source_collected_at", "s.source_collected_at"),
    ("underlying_symbol", "s.underlying_symbol"),
    ("chain_is_delayed", "s.chain_is_delayed"),
    ("chain_is_truncated", "s.chain_is_truncated"),
    ("underlying_is_delayed", "s.underlying_is_delayed"),
    ("symbol_map_mismatch", "o.symbol_map_mismatch"),
    ("chain_dividend_yield", "s.chain_dividend_yield"),
    ("chain_underlying_price", "s.chain_underlying_price"),
    ("underlying_bid", "s.underlying_bid"),
    ("underlying_ask", "s.underlying_ask"),
    ("underlying_last", "s.underlying_last"),
    ("underlying_mark", "s.underlying_mark"),
    ("underlying_quote_time", "s.underlying_quote_time"),
    ("contract_id", "o.contract_id"),
    ("contract_ordinal", "o.contract_ordinal"),
    ("option_symbol", "c.option_symbol"),
    ("put_call", "toString(c.put_call)"),
    ("strike_price", "c.strike_price"),
    ("expiration_at", "c.expiration_at"),
    ("exercise_type", "c.exercise_type"),
    ("multiplier", "c.multiplier"),
    ("is_mini", "c.is_mini"),
    ("is_nonstandard", "c.is_nonstandard"),
    ("bid", "o.bid"),
    ("ask", "o.ask"),
    ("quote_time", "o.quote_time"),
    ("source_iv", "o.source_iv"),
    ("source_delta", "o.source_delta"),
    ("source_gamma", "o.source_gamma"),
    ("source_theta", "o.source_theta"),
    ("source_vega", "o.source_vega"),
    ("source_rho", "o.source_rho"),
    ("stored_observation_count", "s.stored_observation_count"),
)


def calculation_source_projection_columns(*, numerical: bool = False) -> list[str]:
    """Render the canonical source projection with explicit stable aliases."""
    numerical_aliases = {
        "source_collected_at": "collected_at",
        "stored_observation_count": "stored_contract_count",
    }
    return [
        f"{expression} AS {numerical_aliases.get(name, name) if numerical else name}"
        for name, expression in CALCULATION_SOURCE_FIELD_EXPRESSIONS
    ]


def calculation_fact_authorization_predicate(
    fact_alias: str = "fact",
    state_alias: str = "state",
    *,
    include_lineage: bool = True,
) -> str:
    """Return the terminal active-state policy for one calculation fact."""
    predicate = (
        f"{state_alias}.is_active "
        f"AND {state_alias}.status IN ('complete', 'complete_with_failures') "
        f"AND {fact_alias}.snapshot_id = {state_alias}.snapshot_id "
        f"AND {fact_alias}.source_collected_at = {state_alias}.source_collected_at "
        f"AND {fact_alias}.calculation_definition_id = "
        f"{state_alias}.calculation_definition_id "
    )
    if include_lineage:
        predicate += (
            f"AND ((isNull({state_alias}.source_calculation_run_id) "
            f"AND isNull({fact_alias}.source_calculation_run_id)) "
            f"OR {fact_alias}.source_calculation_run_id = "
            f"{state_alias}.source_calculation_run_id)"
        )
    return predicate


def authoritative_calculation_fact_subquery(
    database: str,
    table: str,
    *,
    fact_alias: str = "fact",
    state_alias: str = "state",
    require_active_fact: bool = False,
    snapshot_scoped: bool = False,
) -> str:
    """Return a state-authorized calculation fact relation.

    A snapshot-scoped relation requires the enclosing query to bind snapshot_id.
    """
    database = validate_database_identifier(database)
    allowed_tables = {
        "options_chain_calculation_results",
        "options_chain_calculation_failures",
        "options_chain_expiration_inputs",
    }
    if table not in allowed_tables:
        raise ValueError(f"Unsupported calculation fact table: {table!r}")
    active = f" AND {fact_alias}.is_active" if require_active_fact else ""
    include_lineage = table != "options_chain_expiration_inputs"
    authorization = calculation_fact_authorization_predicate(
        fact_alias, state_alias, include_lineage=include_lineage
    )
    state_relation = f"SELECT * FROM {database}.options_chain_calculation_state FINAL"
    scope = ""
    if snapshot_scoped:
        # Fact tables are ordered by symbol, timestamp, then ID. Restrict all
        # three keys before the authorization join to prune result granules.
        snapshot_symbol = (
            f"(SELECT underlying_symbol FROM {database}.options_chain_snapshots "
            "WHERE snapshot_id = {snapshot_id:UUID} LIMIT 1)"
        )
        snapshot_time = (
            f"(SELECT source_collected_at FROM {database}.options_chain_snapshots "
            "WHERE snapshot_id = {snapshot_id:UUID} LIMIT 1)"
        )
        state_relation += (
            f" WHERE snapshot_id = {{snapshot_id:UUID}} "
            f"AND underlying_symbol = {snapshot_symbol} "
            f"AND source_collected_at = {snapshot_time}"
        )
        scope = (
            f" AND {fact_alias}.snapshot_id = {{snapshot_id:UUID}}"
            f" AND {fact_alias}.underlying_symbol = {snapshot_symbol}"
            f" AND {fact_alias}.source_collected_at = {snapshot_time}"
        )
    return (
        f"(SELECT {fact_alias}.* FROM {database}.{table} AS {fact_alias} "
        f"INNER JOIN ({state_relation}) "
        f"AS {state_alias} ON {authorization} "
        f"WHERE {state_alias}.is_active "
        f"AND {state_alias}.status IN ('complete', 'complete_with_failures')"
        f"{active}{scope}) AS {fact_alias}"
    )


def surface_fact_authorization_predicate(
    fact_alias: str = "fact", state_alias: str = "state"
) -> str:
    """Return the terminal active-state policy for one surface fact."""
    return (
        f"{state_alias}.is_active "
        f"AND {state_alias}.status IN ('complete', 'complete_with_failures') "
        f"AND {fact_alias}.snapshot_id = {state_alias}.snapshot_id "
        f"AND {fact_alias}.source_collected_at = {state_alias}.source_collected_at "
        f"AND {fact_alias}.surface_definition_id = {state_alias}.surface_definition_id "
        f"AND ((isNull({state_alias}.source_surface_run_id) "
        f"AND isNull({fact_alias}.source_surface_run_id)) "
        f"OR {fact_alias}.source_surface_run_id = "
        f"{state_alias}.source_surface_run_id)"
    )


def authoritative_surface_fact_subquery(
    database: str,
    table: str,
    *,
    fact_alias: str = "fact",
    state_alias: str = "state",
    require_active_fact: bool = False,
) -> str:
    """Return a state-authorized surface fact relation."""
    database = validate_database_identifier(database)
    allowed_tables = {
        "options_chain_surface_fits",
        "options_chain_surface_nodes",
        "options_chain_surface_failures",
    }
    if table not in allowed_tables:
        raise ValueError(f"Unsupported surface fact table: {table!r}")
    active = f" AND {fact_alias}.is_active" if require_active_fact else ""
    selected = (
        f"{fact_alias}.*, {state_alias}.source_calculation_definition_id "
        "AS source_calculation_definition_id"
    )
    return (
        f"(SELECT {selected} FROM {database}.{table} AS {fact_alias} "
        f"INNER JOIN (SELECT * FROM {database}.options_chain_surface_state FINAL) "
        f"AS {state_alias} ON {surface_fact_authorization_predicate(fact_alias, state_alias)} "
        f"WHERE {state_alias}.is_active "
        f"AND {state_alias}.status IN ('complete', 'complete_with_failures')"
        f"{active}) AS {fact_alias}"
    )


@dataclass(frozen=True)
class Query:
    sql: str
    parameters: Mapping[str, Any]

    def execute(self, client: Any, *, arrow: bool = True) -> Any:
        method = client.query_arrow if arrow and hasattr(client, "query_arrow") else client.query
        return method(self.sql, parameters=dict(self.parameters))


def current_contracts_sql(database: str, alias: str = "c") -> str:
    """Return the one authoritative current-contract relation.

    ``options_contracts`` is a small slowly-changing dimension.  ``FINAL`` is
    intentionally confined to it; large source/result fact tables never use
    ``FINAL`` in supported queries.
    """
    database = validate_database_identifier(database)
    return f"(SELECT * FROM {database}.options_contracts FINAL) AS {alias}"


def _effective_expression(result_column: str, source_column: str, selector: str) -> str:
    return (
        f"if(r.{selector} = 'result', r.{result_column}, "
        f"o.{source_column})"
    )


EFFECTIVE_IV_SQL = _effective_expression("iv", "source_iv", "iv_selection")
EFFECTIVE_GREEK_SQL = {
    "delta": _effective_expression("delta", "source_delta", "greeks_selection"),
    "gamma": _effective_expression("gamma", "source_gamma", "greeks_selection"),
    "theta": _effective_expression(
        "theta_per_day", "source_theta", "greeks_selection"
    ),
    "vega": _effective_expression(
        "vega_per_1pct", "source_vega", "greeks_selection"
    ),
    "rho": _effective_expression(
        "rho_per_1pct", "source_rho", "greeks_selection"
    ),
}


def effective_values(row: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the canonical selection rules to one joined row in Python."""
    iv_result = row.get("result_iv", row.get("iv"))
    values = {
        "effective_iv": (
            iv_result
            if str(row.get("iv_selection") or "source").lower() == "result"
            else row.get("source_iv")
        ),
        "iv_value_source": (
            "result"
            if str(row.get("iv_selection") or "source").lower() == "result"
            else "source"
        ),
    }
    result_names = {
        "delta": "result_delta",
        "gamma": "result_gamma",
        "theta": "result_theta_per_day",
        "vega": "result_vega_per_1pct",
        "rho": "result_rho_per_1pct",
    }
    use_result = str(row.get("greeks_selection") or "source").lower() == "result"
    for name, result_name in result_names.items():
        result_value = row.get(result_name, row.get(result_name.removeprefix("result_")))
        values[f"effective_{name}"] = (
            result_value if use_result else row.get(f"source_{name}")
        )
    values["greeks_value_source"] = "result" if use_result else "source"
    return values


def validate_effective_selection(row: Mapping[str, Any]) -> None:
    """Fail closed when a success row selects unavailable source/result data."""
    selected = effective_values(row)
    required = ["effective_iv"]
    required.extend(f"effective_{name}" for name in ("delta", "gamma", "theta", "vega", "rho"))
    missing = [name for name in required if _finite_float(selected[name]) is None]
    if missing:
        raise ValueError(
            "Effective selection produced missing/nonfinite values: " + ", ".join(missing)
        )


def build_effective_chain_query(
    database: str,
    snapshot_id: uuid.UUID | str,
    *,
    include_details: bool = False,
    include_failures: bool = False,
) -> Query:
    database = validate_database_identifier(database)
    joins = [
        f"INNER JOIN {current_contracts_sql(database)} ON c.contract_id = o.contract_id",
        f"LEFT JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_results', fact_alias='r', state_alias='result_state', snapshot_scoped=True)} "
        "ON r.snapshot_id = o.snapshot_id AND r.contract_id = o.contract_id "
        "AND r.source_collected_at = o.source_collected_at",
    ]
    select = [
        "toString(s.snapshot_id) AS snapshot_id",
        "s.source_collected_at AS source_collected_at",
        "s.underlying_symbol AS underlying_symbol",
        "s.chain_underlying_price",
        "s.underlying_bid",
        "s.underlying_ask",
        "s.underlying_last",
        "s.underlying_mark",
        "o.contract_id AS contract_id",
        "o.contract_ordinal",
        "c.option_symbol",
        "c.option_root",
        "toString(c.put_call) AS put_call",
        "c.strike_price",
        "c.expiration_at AS expiration_at",
        "c.expiration_type",
        "c.settlement_type",
        "c.exercise_type",
        "c.multiplier",
        "c.is_mini",
        "c.is_nonstandard",
        "c.option_deliverables_json",
        "o.symbol_map_mismatch",
        "o.bid",
        "o.ask",
        "o.last",
        "o.mark",
        "o.bid_size",
        "o.ask_size",
        "o.last_size",
        "o.total_volume",
        "o.open_interest",
        "o.quote_time",
        "o.trade_time",
        "o.source_iv",
        "o.source_delta",
        "o.source_gamma",
        "o.source_theta",
        "o.source_vega",
        "o.source_rho",
        "r.iv AS result_iv",
        "r.delta AS result_delta",
        "r.gamma AS result_gamma",
        "r.theta_per_day AS result_theta_per_day",
        "r.vega_per_1pct AS result_vega_per_1pct",
        "r.rho_per_1pct AS result_rho_per_1pct",
        "if(isNull(r.contract_id), 'source', toString(r.iv_selection)) AS iv_selection",
        "if(isNull(r.contract_id), 'source', toString(r.greeks_selection)) AS greeks_selection",
        "if(isNull(r.contract_id), 'source_only', toString(r.result_origin)) AS result_origin",
        "toString(r.origin_snapshot_id) AS origin_snapshot_id",
        "r.quality_flags AS result_quality_flags",
        f"{EFFECTIVE_IV_SQL} AS effective_iv",
        *(f"{sql} AS effective_{name}" for name, sql in EFFECTIVE_GREEK_SQL.items()),
    ]
    if include_details:
        joins.append(
            f"LEFT JOIN {database}.options_chain_observation_details AS d "
            "ON d.snapshot_id = o.snapshot_id AND d.contract_id = o.contract_id "
            "AND d.source_collected_at = o.source_collected_at"
        )
        select.extend(
            f"d.{name} AS detail_{name}"
            for name in (
                "high_52_week",
                "low_52_week",
                "net_change",
                "percent_change",
                "mark_change",
                "mark_percent_change",
                "intrinsic_value",
                "extrinsic_value",
                "time_value",
                "break_even",
                "theoretical_option_value",
                "theoretical_volatility",
            )
        )
    if include_failures:
        joins.append(
            f"LEFT JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_failures', require_active_fact=True, fact_alias='fc', state_alias='fc_state', snapshot_scoped=True)} "
            "ON fc.snapshot_id = o.snapshot_id AND fc.contract_id = o.contract_id "
            "AND fc.source_collected_at = o.source_collected_at AND fc.is_active "
            "AND fc.failure_scope = 'contract'"
        )
        joins.append(
            f"LEFT JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_failures', require_active_fact=True, fact_alias='fe', state_alias='fe_state', snapshot_scoped=True)} "
            "ON fe.snapshot_id = o.snapshot_id AND fe.expiration_at = c.expiration_at "
            "AND fe.source_collected_at = o.source_collected_at AND fe.is_active "
            "AND fe.failure_scope = 'expiration'"
        )
        joins.append(
            f"LEFT JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_failures', require_active_fact=True, fact_alias='fs', state_alias='fs_state', snapshot_scoped=True)} "
            "ON fs.snapshot_id = o.snapshot_id "
            "AND fs.source_collected_at = o.source_collected_at AND fs.is_active "
            "AND fs.failure_scope = 'snapshot'"
        )
        select.extend(
            (
                "coalesce(toString(fc.failure_scope), toString(fe.failure_scope), "
                "toString(fs.failure_scope)) AS active_failure_scope",
                "coalesce(fc.failure_stage, fe.failure_stage, fs.failure_stage) "
                "AS active_failure_stage",
                "coalesce(fc.reason_code, fe.reason_code, fs.reason_code) "
                "AS active_failure_reason",
                "coalesce(fc.retry_class, fe.retry_class, fs.retry_class) "
                "AS active_failure_retry_class",
                "coalesce(fc.retry_after, fe.retry_after, fs.retry_after) "
                "AS active_failure_retry_after",
            )
        )
    sql = f"""
SELECT
    {', '.join(select)}
FROM {database}.options_chain_snapshots AS s
INNER JOIN {database}.options_chain_observations AS o
    ON o.snapshot_id = s.snapshot_id
   AND o.source_collected_at = s.source_collected_at
{' '.join(joins)}
PREWHERE o.source_collected_at =
    (SELECT source_collected_at
     FROM {database}.options_chain_snapshots
     WHERE snapshot_id = {{snapshot_id:UUID}}
     LIMIT 1)
WHERE s.snapshot_id = {{snapshot_id:UUID}}
ORDER BY o.contract_ordinal, o.contract_id
"""
    return Query(sql, {"snapshot_id": str(snapshot_id)})


def load_effective_chain(
    client: Any,
    database: str,
    snapshot_id: uuid.UUID | str,
    *,
    include_details: bool = False,
    include_failures: bool = False,
    arrow: bool = True,
) -> Any:
    return build_effective_chain_query(
        database,
        snapshot_id,
        include_details=include_details,
        include_failures=include_failures,
    ).execute(client, arrow=arrow)


def build_latest_completed_snapshot_query(
    database: str,
    underlying_symbol: str,
    snapshot_id: uuid.UUID | str | None = None,
    *,
    require_surface: bool = False,
) -> Query:
    """Build the bounded active-calculation snapshot resolution query.

    Calculation and surface state are small replacing state tables, so
    ``FINAL`` is intentional here. The observation, snapshot and surface-fit
    fact tables are not read with ``FINAL``. An explicit ID is subject to the
    same active terminal calculation-state authorization as implicit
    latest-snapshot resolution. Surface compatibility is applied only while
    selecting an implicit latest snapshot; an explicitly requested snapshot is
    returned here and receives the screener's existing, more specific 409
    ``SURFACE_NOT_AVAILABLE`` validation later in the request.
    """

    database = validate_database_identifier(database)
    clauses = [
        "cs.underlying_symbol = {underlying_symbol:String}",
        "cs.is_active",
        "cs.status IN ('complete', 'complete_with_failures')",
    ]
    parameters: dict[str, Any] = {
        "underlying_symbol": str(underlying_symbol).strip().upper(),
    }
    if snapshot_id is not None:
        clauses.append("cs.snapshot_id = {snapshot_id:UUID}")
        parameters["snapshot_id"] = str(snapshot_id)
    surface_join = ""
    if require_surface and snapshot_id is None:
        surface_join = f"""
INNER JOIN (SELECT * FROM {database}.options_chain_surface_state FINAL) AS ss
  ON ss.snapshot_id = cs.snapshot_id
 AND ss.source_collected_at = cs.source_collected_at
 AND ss.underlying_symbol = cs.underlying_symbol
 AND ss.source_calculation_definition_id = cs.calculation_definition_id
"""
        clauses.extend(
            [
                "ss.is_active",
                "ss.status IN ('complete', 'complete_with_failures')",
                "ss.successful_expirations > 0",
            ]
        )
    return Query(
        f"""
SELECT toString(s.snapshot_id) AS snapshot_id,
       s.source_collected_at AS source_collected_at,
       s.underlying_symbol AS underlying_symbol,
       s.chain_underlying_price AS chain_underlying_price,
       s.underlying_bid AS underlying_bid,
       s.underlying_ask AS underlying_ask,
       s.underlying_last AS underlying_last,
       s.underlying_mark AS underlying_mark,
       s.underlying_quote_time AS underlying_quote_time,
       cs.calculation_definition_id AS calculation_definition_id,
       toString(cs.source_calculation_run_id) AS source_calculation_run_id,
       toString(cs.active_attempt_id) AS active_attempt_id,
       toString(cs.status) AS calculation_status,
       cs.finished_at AS calculation_finished_at
FROM (SELECT * FROM {database}.options_chain_calculation_state FINAL) AS cs
{surface_join}
INNER JOIN {database}.options_chain_snapshots AS s
  ON s.snapshot_id = cs.snapshot_id
 AND s.source_collected_at = cs.source_collected_at
 AND s.underlying_symbol = cs.underlying_symbol
WHERE {' AND '.join(clauses)}
ORDER BY cs.source_collected_at DESC, cs.snapshot_id DESC
LIMIT 1
""",
        parameters,
    )


def load_latest_completed_snapshot(
    client: Any,
    database: str,
    underlying_symbol: str,
    snapshot_id: uuid.UUID | str | None = None,
    *,
    require_surface: bool = False,
    arrow: bool = True,
) -> Any:
    """Return the newest authorized compatible snapshot, or validate an ID."""

    return build_latest_completed_snapshot_query(
        database,
        underlying_symbol,
        snapshot_id,
        require_surface=require_surface,
    ).execute(client, arrow=arrow)


def build_expiration_inputs_query(
    database: str,
    snapshot_id: uuid.UUID | str,
) -> Query:
    """Build one bounded query for active calculation expiration inputs."""

    database = validate_database_identifier(database)
    authorization = calculation_fact_authorization_predicate(
        "i", "cs", include_lineage=False
    )
    return Query(
        f"""
SELECT toString(i.snapshot_id) AS snapshot_id,
       i.source_collected_at AS source_collected_at,
       i.underlying_symbol AS underlying_symbol,
       i.expiration_at AS expiration_at,
       i.calculation_definition_id AS calculation_definition_id,
       i.time_to_expiry_seconds AS time_to_expiry_seconds,
       i.risk_free_rate AS risk_free_rate,
       i.dividend_yield AS dividend_yield,
       i.forward_price AS forward_price,
       i.quality_flags AS quality_flags,
       i.calculated_at AS calculated_at,
       toString(cs.source_calculation_run_id) AS source_calculation_run_id,
       toString(cs.active_attempt_id) AS active_attempt_id,
       toString(cs.status) AS calculation_status
FROM {database}.options_chain_expiration_inputs AS i
INNER JOIN (SELECT * FROM {database}.options_chain_calculation_state FINAL) AS cs
  ON {authorization}
PREWHERE i.source_collected_at =
  (SELECT source_collected_at FROM {database}.options_chain_snapshots
   WHERE snapshot_id = {{snapshot_id:UUID}} LIMIT 1)
WHERE i.snapshot_id = {{snapshot_id:UUID}}
  AND cs.is_active
  AND cs.status IN ('complete', 'complete_with_failures')
ORDER BY i.expiration_at
""",
        {"snapshot_id": str(snapshot_id)},
    )


def load_expiration_inputs(
    client: Any,
    database: str,
    snapshot_id: uuid.UUID | str,
    *,
    arrow: bool = True,
) -> Any:
    """Load all state-authorized pricing inputs for one snapshot in one query."""

    return build_expiration_inputs_query(database, snapshot_id).execute(
        client, arrow=arrow
    )


def build_contract_history_query(
    database: str,
    contract_id: int,
    start_at: datetime,
    end_at: datetime,
) -> Query:
    database = validate_database_identifier(database)
    sql = f"""
SELECT o.*, r.iv AS result_iv, r.delta AS result_delta,
       r.gamma AS result_gamma, r.theta_per_day AS result_theta_per_day,
       r.vega_per_1pct AS result_vega_per_1pct,
       r.rho_per_1pct AS result_rho_per_1pct,
       if(isNull(r.contract_id), 'source', toString(r.iv_selection)) AS iv_selection,
       if(isNull(r.contract_id), 'source', toString(r.greeks_selection)) AS greeks_selection,
       {EFFECTIVE_IV_SQL} AS effective_iv,
       {EFFECTIVE_GREEK_SQL['delta']} AS effective_delta,
       {EFFECTIVE_GREEK_SQL['gamma']} AS effective_gamma,
       {EFFECTIVE_GREEK_SQL['theta']} AS effective_theta,
       {EFFECTIVE_GREEK_SQL['vega']} AS effective_vega,
       {EFFECTIVE_GREEK_SQL['rho']} AS effective_rho
FROM {database}.options_chain_observations AS o
LEFT JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_results', fact_alias='r', state_alias='result_state')}
  ON r.snapshot_id = o.snapshot_id AND r.contract_id = o.contract_id
 AND r.source_collected_at = o.source_collected_at
PREWHERE o.source_collected_at >= fromUnixTimestamp64Micro({{start_us:Int64}})
     AND o.source_collected_at < fromUnixTimestamp64Micro({{end_us:Int64}})
WHERE o.contract_id = {{contract_id:UInt64}}
ORDER BY o.source_collected_at, o.snapshot_id
"""
    return Query(
        sql,
        {
            "contract_id": int(contract_id),
            "start_us": _datetime64_microseconds(start_at),
            "end_us": _datetime64_microseconds(end_at),
        },
    )


def load_contract_history(
    client: Any,
    database: str,
    contract_id: int,
    start_at: datetime,
    end_at: datetime,
) -> Any:
    return build_contract_history_query(database, contract_id, start_at, end_at).execute(client)


def _snapshot_filter(
    *,
    database: str,
    snapshot_ids: Optional[Sequence[uuid.UUID | str]],
    start_at: Optional[datetime],
    end_at: Optional[datetime],
    alias: str,
) -> tuple[list[str], dict[str, Any]]:
    clauses: list[str] = []
    parameters: dict[str, Any] = {}
    if snapshot_ids:
        clauses.append(f"{alias}.snapshot_id IN {{snapshot_ids:Array(UUID)}}")
        clauses.append(
            f"{alias}.source_collected_at IN "
            f"(SELECT source_collected_at FROM {database}.options_chain_snapshots "
            "WHERE snapshot_id IN {snapshot_ids:Array(UUID)})"
        )
        parameters["snapshot_ids"] = [str(value) for value in snapshot_ids]
    if start_at is not None:
        clauses.append(
            f"toUnixTimestamp64Micro({alias}.source_collected_at) >= {{start_us:Int64}}"
        )
        parameters["start_us"] = _datetime64_microseconds(start_at)
    if end_at is not None:
        clauses.append(
            f"toUnixTimestamp64Micro({alias}.source_collected_at) < {{end_us:Int64}}"
        )
        parameters["end_us"] = _datetime64_microseconds(end_at)
    if not clauses:
        raise ValueError("A snapshot ID or bounded source time range is required")
    return clauses, parameters


def _load_state(
    client: Any,
    database: str,
    table: str,
    *,
    snapshot_ids: Optional[Sequence[uuid.UUID | str]] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    arrow: bool = True,
) -> Any:
    database = validate_database_identifier(database)
    clauses, parameters = _snapshot_filter(
        database=database,
        snapshot_ids=snapshot_ids,
        start_at=start_at,
        end_at=end_at,
        alias="s",
    )
    query = Query(
        f"SELECT * FROM (SELECT * FROM {database}.{table} FINAL) AS s "
        f"WHERE s.is_active AND {' AND '.join(clauses)} "
        "ORDER BY s.source_collected_at, s.snapshot_id",
        parameters,
    )
    return query.execute(client, arrow=arrow)


def load_calculation_state(client: Any, database: str, **filters: Any) -> Any:
    return _load_state(client, database, "options_chain_calculation_state", **filters)


def load_surface_state(client: Any, database: str, **filters: Any) -> Any:
    return _load_state(client, database, "options_chain_surface_state", **filters)


def _load_failures(
    client: Any,
    database: str,
    table: str,
    *,
    snapshot_id: Optional[uuid.UUID | str] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    failure_scope: Optional[str] = None,
    reason_code: Optional[str] = None,
    arrow: bool = True,
) -> Any:
    database = validate_database_identifier(database)
    ids = [snapshot_id] if snapshot_id is not None else None
    clauses, parameters = _snapshot_filter(
        database=database,
        snapshot_ids=ids,
        start_at=start_at,
        end_at=end_at,
        alias="f",
    )
    clauses.append("f.is_active")
    if failure_scope:
        clauses.append("f.failure_scope = {failure_scope:String}")
        parameters["failure_scope"] = failure_scope
    if reason_code:
        clauses.append("f.reason_code = {reason_code:String}")
        parameters["reason_code"] = reason_code
    if table == "options_chain_calculation_failures":
        relation = authoritative_calculation_fact_subquery(
            database, table, fact_alias="f", state_alias="failure_state",
            require_active_fact=True,
        )
    elif table == "options_chain_surface_failures":
        relation = authoritative_surface_fact_subquery(
            database, table, fact_alias="f", state_alias="failure_state",
            require_active_fact=True,
        )
    else:
        raise ValueError(f"Unsupported failure table: {table!r}")
    return Query(
        f"SELECT * FROM {relation} "
        f"WHERE {' AND '.join(clauses)} ORDER BY f.source_collected_at",
        parameters,
    ).execute(client, arrow=arrow)


def load_calculation_failures(client: Any, database: str, **filters: Any) -> Any:
    return _load_failures(
        client, database, "options_chain_calculation_failures", **filters
    )


def load_surface_failures(client: Any, database: str, **filters: Any) -> Any:
    return _load_failures(client, database, "options_chain_surface_failures", **filters)


def load_expiration_summary(client: Any, database: str, snapshot_id: uuid.UUID | str) -> Any:
    database = validate_database_identifier(database)
    query = Query(
        f"""
SELECT c.expiration_at,
       count() AS observation_count,
       countIf(c.put_call = 'CALL') AS call_count,
       countIf(c.put_call = 'PUT') AS put_count,
       min(c.strike_price) AS minimum_strike,
       max(c.strike_price) AS maximum_strike,
       countIf(NOT isNull(r.contract_id)) AS result_count,
       countIf(f.is_active) AS active_contract_failure_count
FROM {database}.options_chain_observations AS o
INNER JOIN {current_contracts_sql(database)} ON c.contract_id = o.contract_id
LEFT JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_results', fact_alias='r', state_alias='result_state')}
  ON r.snapshot_id = o.snapshot_id AND r.contract_id = o.contract_id
 AND r.source_collected_at = o.source_collected_at
LEFT JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_failures', fact_alias='f', state_alias='failure_state', require_active_fact=True)}
  ON f.snapshot_id = o.snapshot_id AND f.contract_id = o.contract_id
 AND f.source_collected_at = o.source_collected_at AND f.is_active
PREWHERE o.source_collected_at =
  (SELECT source_collected_at FROM {database}.options_chain_snapshots
   WHERE snapshot_id = {{snapshot_id:UUID}} LIMIT 1)
WHERE o.snapshot_id = {{snapshot_id:UUID}}
GROUP BY c.expiration_at
ORDER BY c.expiration_at
""",
        {"snapshot_id": str(snapshot_id)},
    )
    return query.execute(client)


def _load_surface_success(
    client: Any,
    database: str,
    table: str,
    snapshot_id: uuid.UUID | str,
    price_basis: str,
    *,
    arrow: bool = True,
) -> Any:
    database = validate_database_identifier(database)
    order = (
        "expiration_at, price_basis"
        if table == "options_chain_surface_fits"
        else "expiration_at, price_basis, node_type, node_value, put_call"
    )
    relation = authoritative_surface_fact_subquery(
        database, table, fact_alias="f", state_alias="surface_state"
    )
    return Query(
        f"SELECT * FROM {relation} WHERE source_collected_at = "
        f"(SELECT source_collected_at FROM {database}.options_chain_snapshots "
        "WHERE snapshot_id = {snapshot_id:UUID} LIMIT 1) "
        "AND snapshot_id = {snapshot_id:UUID} AND price_basis = {price_basis:String} "
        f"ORDER BY {order}",
        {"snapshot_id": str(snapshot_id), "price_basis": price_basis},
    ).execute(client, arrow=arrow)


def load_surface_fits(
    client: Any,
    database: str,
    snapshot_id: uuid.UUID | str,
    price_basis: str,
    *,
    arrow: bool = True,
) -> Any:
    return _load_surface_success(
        client,
        database,
        "options_chain_surface_fits",
        snapshot_id,
        price_basis,
        arrow=arrow,
    )


def load_surface_nodes(
    client: Any,
    database: str,
    snapshot_id: uuid.UUID | str,
    price_basis: str,
    *,
    arrow: bool = True,
) -> Any:
    return _load_surface_success(
        client,
        database,
        "options_chain_surface_nodes",
        snapshot_id,
        price_basis,
        arrow=arrow,
    )


def load_atm_term_structure(
    client: Any,
    database: str,
    snapshot_id: uuid.UUID | str,
    *,
    price_basis: str | None = None,
    arrow: bool = True,
) -> Any:
    database = validate_database_identifier(database)
    basis_clause = ""
    parameters = {"snapshot_id": str(snapshot_id)}
    if price_basis is not None:
        basis_clause = " AND price_basis = {price_basis:String}"
        parameters["price_basis"] = str(price_basis)
    return Query(
        f"""
SELECT expiration_at, price_basis, fitted_iv_mid, fitted_strike,
       fitted_delta, fit_confidence, quality_flags
FROM {authoritative_surface_fact_subquery(database, 'options_chain_surface_nodes', fact_alias='n', state_alias='surface_state')}
WHERE source_collected_at =
  (SELECT source_collected_at FROM {database}.options_chain_snapshots
   WHERE snapshot_id = {{snapshot_id:UUID}} LIMIT 1)
  AND snapshot_id = {{snapshot_id:UUID}}
  AND node_type = 'atm_forward' AND node_value = 0 AND put_call = 'NONE'
  {basis_clause}
ORDER BY expiration_at, price_basis
""",
        parameters,
    ).execute(client, arrow=arrow)


def load_skew(
    client: Any,
    database: str,
    snapshot_id: uuid.UUID | str,
    expiration_at: datetime,
    price_basis: str,
) -> Any:
    database = validate_database_identifier(database)
    return Query(
        f"SELECT * FROM {authoritative_surface_fact_subquery(database, 'options_chain_surface_nodes', fact_alias='n', state_alias='surface_state')} "
        f"WHERE source_collected_at = (SELECT source_collected_at FROM "
        f"{database}.options_chain_snapshots WHERE snapshot_id = "
        "{snapshot_id:UUID} LIMIT 1) "
        "AND snapshot_id = {snapshot_id:UUID} "
        "AND toUnixTimestamp64Milli(expiration_at) = {expiration_ms:Int64} "
        "AND price_basis = {price_basis:String} ORDER BY node_type, node_value, put_call",
        {
            "snapshot_id": str(snapshot_id),
            "expiration_ms": _datetime64_microseconds(expiration_at) // 1000,
            "price_basis": price_basis,
        },
    ).execute(client)


def load_forward_variance(
    client: Any,
    database: str,
    snapshot_id: uuid.UUID | str,
    *,
    arrow: bool = True,
) -> Any:
    database = validate_database_identifier(database)
    return Query(
        f"""
WITH atm AS
(
    SELECT n.snapshot_id, n.expiration_at, n.price_basis, n.fitted_iv_mid,
           i.time_to_expiry_seconds,
           (n.fitted_iv_mid * n.fitted_iv_mid)
             * i.time_to_expiry_seconds / 31557600.0 AS total_variance
    FROM {authoritative_surface_fact_subquery(database, 'options_chain_surface_nodes', fact_alias='n', state_alias='surface_state')}
    INNER JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_expiration_inputs', fact_alias='i', state_alias='calculation_state')}
     ON i.snapshot_id = n.snapshot_id
     AND i.source_collected_at = n.source_collected_at
     AND i.calculation_definition_id = n.source_calculation_definition_id
     AND i.expiration_at = n.expiration_at
    WHERE n.source_collected_at =
      (SELECT source_collected_at FROM {database}.options_chain_snapshots
       WHERE snapshot_id = {{snapshot_id:UUID}} LIMIT 1)
      AND n.snapshot_id = {{snapshot_id:UUID}}
      AND n.node_type = 'atm_forward' AND n.node_value = 0
      AND n.put_call = 'NONE'
)
SELECT *,
       if(isNull(lagInFrame(total_variance) OVER w), NULL,
          (total_variance - lagInFrame(total_variance) OVER w) /
          greatest(1, time_to_expiry_seconds - lagInFrame(time_to_expiry_seconds) OVER w)
          * 31557600.0) AS forward_variance
FROM atm
WINDOW w AS (PARTITION BY price_basis ORDER BY expiration_at
             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)
ORDER BY price_basis, expiration_at
""",
        {"snapshot_id": str(snapshot_id)},
    ).execute(client, arrow=arrow)


def load_intraday_node_changes(
    client: Any,
    database: str,
    symbol: str,
    node_type: str,
    node_value: float,
    put_call: str,
    price_basis: str,
    start_at: datetime,
    end_at: datetime,
) -> Any:
    database = validate_database_identifier(database)
    return Query(
        f"""
SELECT *, fitted_iv_mid - lagInFrame(fitted_iv_mid) OVER
       (PARTITION BY expiration_at, price_basis, node_type, node_value, put_call
        ORDER BY source_collected_at ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS iv_change
FROM {authoritative_surface_fact_subquery(database, 'options_chain_surface_nodes', fact_alias='n', state_alias='surface_state')}
WHERE source_collected_at >= fromUnixTimestamp64Micro({{start_us:Int64}})
  AND source_collected_at < fromUnixTimestamp64Micro({{end_us:Int64}})
  AND underlying_symbol = {{symbol:String}}
  AND node_type = {{node_type:String}} AND node_value = {{node_value:Float64}}
  AND put_call = {{put_call:String}} AND price_basis = {{price_basis:String}}
ORDER BY source_collected_at, expiration_at
""",
        {
            "symbol": symbol.upper(),
            "node_type": node_type,
            "node_value": float(node_value),
            "put_call": put_call.upper(),
            "price_basis": price_basis,
            "start_us": _datetime64_microseconds(start_at),
            "end_us": _datetime64_microseconds(end_at),
        },
    ).execute(client)


def load_strike_coverage(
    client: Any,
    database: str,
    snapshot_id: uuid.UUID | str,
    expiration_at: datetime,
) -> Any:
    database = validate_database_identifier(database)
    return Query(
        f"""
SELECT c.put_call, count() AS contract_count,
       min(c.strike_price) AS minimum_strike,
       max(c.strike_price) AS maximum_strike,
       countIf(NOT isNull(o.bid) AND NOT isNull(o.ask)) AS quoted_count,
       countIf(NOT isNull(r.contract_id)) AS calculated_count,
       min({EFFECTIVE_IV_SQL}) AS minimum_effective_iv,
       max({EFFECTIVE_IV_SQL}) AS maximum_effective_iv
FROM {database}.options_chain_observations AS o
INNER JOIN {current_contracts_sql(database)} ON c.contract_id = o.contract_id
LEFT JOIN {authoritative_calculation_fact_subquery(database, 'options_chain_calculation_results', fact_alias='r', state_alias='result_state')}
  ON r.snapshot_id = o.snapshot_id AND r.contract_id = o.contract_id
 AND r.source_collected_at = o.source_collected_at
PREWHERE o.source_collected_at =
  (SELECT source_collected_at FROM {database}.options_chain_snapshots
   WHERE snapshot_id = {{snapshot_id:UUID}} LIMIT 1)
WHERE o.snapshot_id = {{snapshot_id:UUID}}
  AND toUnixTimestamp64Milli(c.expiration_at) = {{expiration_ms:Int64}}
GROUP BY c.put_call ORDER BY c.put_call
""",
        {
            "snapshot_id": str(snapshot_id),
            "expiration_ms": _datetime64_microseconds(expiration_at) // 1000,
        },
    ).execute(client)


def benchmark_query(
    client: Any,
    query: Query,
    *,
    repetitions: int = 3,
    warm_target_seconds: float = 0.250,
    cold_target_seconds: float = 1.0,
) -> Mapping[str, Any]:
    """Run a bounded supported query and return operator-facing plan metrics."""
    if repetitions < 2 or repetitions > 20:
        raise ValueError("GUI query benchmark repetitions must be between 2 and 20")
    plan_result = client.query(
        "EXPLAIN indexes = 1 " + query.sql,
        parameters=dict(query.parameters),
    )
    plan = "\n".join(str(row[0]) for row in plan_result.result_rows)
    elapsed: list[float] = []
    rows_read = 0
    bytes_read = 0
    memory_usage = 0
    result_rows = 0
    for _ in range(repetitions):
        started = time.perf_counter()
        result = client.query(query.sql, parameters=dict(query.parameters))
        elapsed.append(time.perf_counter() - started)
        result_rows = len(result.result_rows)
        summary = dict(getattr(result, "summary", {}) or {})
        rows_read = max(rows_read, int(summary.get("read_rows", 0) or 0))
        bytes_read = max(bytes_read, int(summary.get("read_bytes", 0) or 0))
        memory_usage = max(
            memory_usage, int(summary.get("memory_usage", 0) or 0)
        )
    warm = elapsed[1:]
    warm_p95 = (
        statistics.quantiles(warm, n=20, method="inclusive")[18]
        if len(warm) > 1
        else warm[0]
    )
    report = {
        "result_rows": result_rows,
        "rows_read": rows_read,
        "bytes_read": bytes_read,
        "memory_bytes": memory_usage,
        "cold_seconds": elapsed[0],
        "warm_seconds": warm,
        "warm_p95_seconds": warm_p95,
        "cold_target_seconds": float(cold_target_seconds),
        "warm_target_seconds": float(warm_target_seconds),
        "cold_target_met": elapsed[0] <= cold_target_seconds,
        "warm_target_met": warm_p95 <= warm_target_seconds,
        "query_plan": plan,
    }
    logging.getLogger(__name__).info(
        "gui_query_benchmark result_rows=%s rows_read=%s bytes_read=%s "
        "memory_bytes=%s cold_seconds=%.6f warm_p95_seconds=%.6f "
        "cold_target_met=%s warm_target_met=%s",
        result_rows,
        rows_read,
        bytes_read,
        memory_usage,
        elapsed[0],
        warm_p95,
        report["cold_target_met"],
        report["warm_target_met"],
    )
    return report


def benchmark_effective_chain(
    client: Any,
    database: str,
    snapshot_id: uuid.UUID | str,
    *,
    repetitions: int = 3,
) -> Mapping[str, Any]:
    return benchmark_query(
        client,
        build_effective_chain_query(database, snapshot_id),
        repetitions=repetitions,
    )


__all__ = [
    "CALCULATION_SOURCE_FIELD_EXPRESSIONS",
    "EFFECTIVE_GREEK_SQL",
    "EFFECTIVE_IV_SQL",
    "Query",
    "authoritative_calculation_fact_subquery",
    "authoritative_surface_fact_subquery",
    "benchmark_effective_chain",
    "benchmark_query",
    "build_contract_history_query",
    "build_effective_chain_query",
    "build_expiration_inputs_query",
    "build_latest_completed_snapshot_query",
    "current_contracts_sql",
    "calculation_fact_authorization_predicate",
    "calculation_source_projection_columns",
    "effective_values",
    "load_atm_term_structure",
    "load_calculation_failures",
    "load_calculation_state",
    "load_contract_history",
    "load_effective_chain",
    "load_expiration_inputs",
    "load_expiration_summary",
    "load_forward_variance",
    "load_intraday_node_changes",
    "load_latest_completed_snapshot",
    "load_skew",
    "load_strike_coverage",
    "load_surface_failures",
    "load_surface_fits",
    "load_surface_nodes",
    "load_surface_state",
    "surface_fact_authorization_predicate",
    "validate_effective_selection",
]
