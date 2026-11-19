"""Observed option evidence and current-position historical revaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import math
from statistics import median
from typing import Any, Mapping, Sequence

from pydantic import Field

from algotrading.refdata.api.market_calendars import calendar_for_futures_root
from algotrading.refdata.api.option_symbols import parse_occ_symbol
from algotrading.risk.history.planner import (
    HistoryWindowPlan,
    HistoryWindowPlanner,
    advertised_option_history_limit,
    chunk_session_window,
    clip_required_sessions,
)
from algotrading.risk.market_inputs.rates import interpolate_risk_free_rate
from algotrading.risk.market_inputs.dividends import escrowed_dividend_adjustment
from algotrading.risk.models import ImmutableModel, InstrumentTerms, NormalizedPosition
from algotrading.risk.option_history.resolver import (
    HistoricalOptionSeries,
    HistoricalOptionStateResolver,
)
from algotrading.risk.pricing.tian_service import (
    IvGreeksContract,
    IvGreeksRequest,
    TianRepriceContract,
    TianRepriceRequest,
)


class PositionHistoryCoverage(ImmutableModel):
    position_id: str
    exact_symbol: str
    asset_type: str
    economic_position_id: str
    required_first_session: date | None = None
    required_last_session: date | None = None
    requested_return_count: int
    required_mark_count: int
    sql_mark_count: int = 0
    remote_mark_count: int = 0
    exact_interval_count: int = 0
    current_state_scenario_count: int = 0
    first_usable_exact_interval: date | None = None
    last_usable_exact_interval: date | None = None
    first_usable_current_state_scenario: date | None = None
    last_usable_current_state_scenario: date | None = None
    interval_method_counts: dict[str, int] = Field(default_factory=dict)
    scenario_method_counts: dict[str, int] = Field(default_factory=dict)
    surface_usage_count: int = 0
    contract_iv_usage_count: int = 0
    greek_bridge_usage_count: int = 0
    marks_only_usage_count: int = 0
    proxy_usage_count: int = 0
    requested_remote_ranges: tuple[dict[str, Any], ...] = ()
    failed_remote_ranges: tuple[dict[str, Any], ...] = ()
    missing_fields: tuple[str, ...] = ()
    materiality: str
    fail_closed: bool
    calendar_id: str
    calendar_version: str
    methodology_version: str = "current_portfolio_filtered_historical_revaluation_v5"


class HistoricalRevaluationResult(ImmutableModel):
    methodology_version: str = "current_portfolio_filtered_historical_revaluation_v5"
    # Immutable exact mark-to-mark evidence retained for attribution. Risk
    # consumers use the separate current-state ``scenario_pnl`` matrix.
    unit_pnl: dict[str, dict[str, float]]
    scenario_pnl: dict[str, dict[str, float]] = Field(default_factory=dict)
    unit_methods: dict[str, str]
    scenario_methods: dict[str, str] = Field(default_factory=dict)
    unit_symbols: dict[str, str]
    unit_interval_methods: dict[str, dict[str, str]] = Field(default_factory=dict)
    scenario_provenance: dict[str, dict[str, dict[str, Any]]] = Field(
        default_factory=dict
    )
    position_coverage: tuple[PositionHistoryCoverage, ...] = ()
    missing_position_ids: tuple[str, ...]
    proxy_mappings: tuple[dict[str, Any], ...] = ()
    return_diagnostics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def historical_risk_position_eligible(position: NormalizedPosition) -> bool:
    """Apply the established cash/cash-equivalent exclusion consistently."""
    broker_type = str(
        position.raw_broker_fields.get("instrument_type") or ""
    ).upper()
    return (
        position.asset_type != "CASH"
        and "NO_LOAD" not in broker_type
        and "MONEY_MARKET" not in broker_type
    )


class CurrentPortfolioHistoricalRevaluationEngine:
    """Apply identified one-session historical shocks to today's positions."""

    def __init__(
        self,
        reference_client: Any,
        context: Mapping[str, Any],
        *,
        risk_history_service: Any | None = None,
        planner: HistoryWindowPlanner | None = None,
    ) -> None:
        self.reference_client = reference_client
        self.context = context
        self.planner = planner or HistoryWindowPlanner(context)
        self.option_resolver = HistoricalOptionStateResolver(
            reference_client,
            context,
            risk_history_service=risk_history_service,
        )
        # A current-state historical run shocks the same option hundreds of
        # times.  The unshocked Tian price is identical for every scenario and
        # is retained only for diagnostics, so solve it once per input state.
        self._option_base_reprice_cache: dict[tuple[Any, ...], Any] = {}

    def revalue(
        self,
        positions: Sequence[NormalizedPosition],
        *,
        as_of: datetime,
        lookback_sessions: int | None = None,
    ) -> HistoricalRevaluationResult:
        lookback = lookback_sessions or int(
            self.context.get("risk.history.lookback_sessions", 252)
        )
        advertised = advertised_option_history_limit(self.reference_client)
        option_plan = self.planner.plan(
            as_of=as_of,
            requested_return_count=lookback,
            calendar_id="XNYS",
            history_kind="exact_option_mark",
            advertised_maximum_days=advertised,
        )
        plans: dict[str, HistoryWindowPlan] = {"XNYS": option_plan}
        # Keep observed contract P&L separate from shocks applied to today's holdings.
        observed_pnl: dict[str, dict[str, float]] = {}
        scenario_pnl: dict[str, dict[str, float]] = {}
        observed_methods: dict[str, str] = {}
        scenario_methods: dict[str, str] = {}
        symbols: dict[str, str] = {}
        interval_methods: dict[str, dict[str, str]] = {}
        provenance: dict[str, dict[str, dict[str, Any]]] = {}
        missing: list[str] = []
        warnings: list[str] = []
        proxy_rows: list[dict[str, Any]] = []
        coverage_rows: dict[str, PositionHistoryCoverage] = {}

        eligible = [item for item in positions if historical_risk_position_eligible(item)]
        linear = [
            item
            for item in eligible
            if item.asset_type in {"EQUITY", "ETF", "MUTUAL_FUND", "FUTURE"}
            and _field_valid(item, "mark")
        ]
        invalid_linear = [
            item
            for item in eligible
            if item.asset_type in {"EQUITY", "ETF", "MUTUAL_FUND", "FUTURE"}
            and not _field_valid(item, "mark")
        ]
        for position in invalid_linear:
            missing.append(position.position_id)
            warnings.append(
                f"{position.position_id}:current_mark_not_valid_for_historical_risk"
            )

        price_symbols: dict[str, str] = {}
        history_methods: dict[str, str] = {}
        sessions_by_history_symbol: dict[str, tuple[date, ...]] = {}
        futures_map = self.context.get("risk.history.futures_series_map", {})
        proxy_map = self.context.get("risk.history.economic_proxies", {})
        allow_proxy = _as_bool(
            self.context.get("risk.history.allow_configured_economic_proxies", False)
        )
        # Plan futures history on each product's exchange session calendar.
        for position in linear:
            if position.asset_type == "FUTURE":
                calendar_id = (
                    position.instrument_terms.exchange_calendar
                    if position.instrument_terms is not None
                    and position.instrument_terms.exchange_calendar
                    else calendar_for_futures_root(position.futures_root)
                )
                if calendar_id not in plans:
                    plans[calendar_id] = self.planner.plan(
                        as_of=as_of,
                        requested_return_count=lookback,
                        calendar_id=calendar_id,
                        history_kind="daily_close",
                    )
                mapping = (
                    futures_map.get(position.exact_symbol)
                    or futures_map.get((position.futures_root or "").removeprefix("/"))
                    if isinstance(futures_map, Mapping)
                    else None
                )
                if isinstance(mapping, Mapping):
                    history_symbol = str(mapping.get("symbol") or "").upper()
                    roll_method = str(mapping.get("roll_adjustment_method") or "").strip()
                    if not roll_method:
                        warnings.append(
                            f"{position.position_id}:futures_roll_adjustment_unavailable"
                        )
                        missing.append(position.position_id)
                        continue
                    history_methods[position.position_id] = (
                        f"configured_roll_adjusted_futures_history:{roll_method}"
                    )
                else:
                    history_symbol = ""
                if not history_symbol:
                    missing.append(position.position_id)
                    warnings.append(
                        f"{position.position_id}:futures_history_mapping_unavailable"
                    )
                    continue
                sessions_by_history_symbol[history_symbol] = plans[
                    calendar_id
                ].required_sessions
            else:
                history_symbol = position.exact_symbol
                sessions_by_history_symbol[history_symbol] = option_plan.required_sessions
            price_symbols[position.position_id] = history_symbol

        if allow_proxy and isinstance(proxy_map, Mapping):
            for position in linear:
                mapping = proxy_map.get(position.exact_symbol)
                if isinstance(mapping, Mapping) and mapping.get("proxy_symbol"):
                    proxy_symbol = str(mapping["proxy_symbol"]).upper()
                    sessions_by_history_symbol[proxy_symbol] = option_plan.required_sessions

        option_positions = [
            item
            for item in eligible
            if item.asset_type == "OPTION"
            and _field_valid(item, "mark")
            and _field_valid(item, "historical_mark_revaluation_terms")
        ]
        option_ids = {item.position_id for item in option_positions}
        invalid_options = [
            item
            for item in eligible
            if item.asset_type == "OPTION" and item.position_id not in option_ids
        ]
        option_roots = {
            (item.underlying_symbol or parse_occ_symbol(
                item.canonical_occ_symbol or item.exact_symbol
            ).root).upper()
            for item in option_positions
        }
        for root in option_roots:
            sessions_by_history_symbol.setdefault(root, option_plan.required_sessions)

        returns, mark_rows, return_diagnostics = self._returns(
            sessions_by_history_symbol,
            causal_as_of=as_of,
        )

        for position in linear:
            history_symbol = price_symbols.get(position.position_id)
            if not history_symbol:
                continue
            series = returns.get(history_symbol)
            method = history_methods.get(
                position.position_id, "exact_underlying_history"
            )
            if not series and allow_proxy and isinstance(proxy_map, Mapping):
                mapping = proxy_map.get(position.exact_symbol)
                if isinstance(mapping, Mapping):
                    proxy_symbol = str(mapping.get("proxy_symbol") or "").upper()
                    beta = _finite(mapping.get("beta"))
                    if proxy_symbol in returns and beta is not None:
                        series = {
                            day: value * beta for day, value in returns[proxy_symbol].items()
                        }
                        method = "configured_calibrated_economic_proxy"
                        proxy_rows.append(
                            {
                                "position_id": position.position_id,
                                "source_symbol": position.exact_symbol,
                                "proxy_symbol": proxy_symbol,
                                "beta": beta,
                                "calibration_window": mapping.get("calibration_window"),
                                "residual_error": mapping.get("residual_error"),
                            }
                        )
            # Scale linear shocks by futures notional or current security liquidation value.
            exposure = position.signed_notional_exposure_usd
            if position.asset_type != "FUTURE":
                exposure = position.signed_liquidation_value_usd
            calendar_id = (
                calendar_for_futures_root(position.futures_root)
                if position.asset_type == "FUTURE"
                else "XNYS"
            )
            plan = plans.get(calendar_id, option_plan)
            if not series or exposure is None:
                missing.append(position.position_id)
                warnings.append(f"{position.position_id}:linear_history_unavailable")
                coverage_rows[position.position_id] = _coverage(
                    position,
                    plan,
                    current_dates=(),
                    missing_fields=("daily_close_history",),
                    fail_closed=_material(position),
                )
                continue
            requested_days = {item.isoformat() for item in plan.required_sessions[1:]}
            pnl = {
                day: float(exposure) * value
                for day, value in sorted(series.items())
                if day in requested_days
            }
            observed_pnl[position.position_id] = pnl
            scenario_pnl[position.position_id] = pnl
            observed_methods[position.position_id] = method
            scenario_methods[position.position_id] = method
            symbols[position.position_id] = position.exact_symbol
            coverage_rows[position.position_id] = _coverage(
                position,
                plan,
                current_dates=tuple(date.fromisoformat(day) for day in pnl),
                scenario_method_counts={method: len(pnl)},
                requested_remote_ranges=tuple(
                    return_diagnostics.get(history_symbol, {}).get(
                        "requested_remote_ranges", ()
                    )
                ),
                failed_remote_ranges=tuple(
                    return_diagnostics.get(history_symbol, {}).get(
                        "failed_remote_ranges", ()
                    )
                ),
            )

        for position in invalid_options:
            missing.append(position.position_id)
            warnings.append(
                f"{position.position_id}:current_option_state_not_valid_for_historical_risk"
            )
            coverage_rows[position.position_id] = _coverage(
                position,
                option_plan,
                current_dates=(),
                missing_fields=("current_option_state",),
                fail_closed=True,
            )

        histories: dict[str, HistoricalOptionSeries] = {}
        if option_positions:
            terms = {
                item.canonical_occ_symbol or item.exact_symbol: item.instrument_terms
                for item in option_positions
                if item.instrument_terms is not None
            }
            sessions_by_option: dict[str, tuple[date, ...]] = {}
            listing_dates = self.context.get("risk.option_history.listing_dates", {})
            for item in option_positions:
                exact = item.canonical_occ_symbol or item.exact_symbol
                configured_listing = (
                    _date_value(listing_dates.get(exact))
                    if isinstance(listing_dates, Mapping)
                    else None
                )
                authoritative_listing = (
                    item.instrument_terms.listing_date
                    if item.instrument_terms is not None
                    else None
                )
                # Clip contract history to its known listing and expiration lifetime.
                sessions_by_option[exact] = clip_required_sessions(
                    option_plan.required_sessions,
                    listing_date=authoritative_listing or configured_listing,
                    expiration_date=(
                        item.expiration_at.date() if item.expiration_at else None
                    ),
                )
            try:
                histories = self.option_resolver.resolve(
                    [item.canonical_occ_symbol or item.exact_symbol for item in option_positions],
                    start_at=option_plan.start_at,
                    end_at=option_plan.end_at,
                    causal_as_of=as_of,
                    terms_by_symbol=terms,
                    required_sessions_by_symbol=sessions_by_option,
                    underlying_closes_by_symbol=mark_rows,
                )
            except Exception as exc:
                warnings.append(f"option_history_resolution_failed:{type(exc).__name__}")

        option_underlying_returns = _option_observation_returns(
            histories, option_plan.required_sessions
        )
        for root, series in option_underlying_returns.items():
            current = returns.setdefault(root, {})
            for day, value in series.items():
                current.setdefault(day, value)
        volatility_shocks, shock_provenance = self._volatility_shocks(
            histories, option_plan
        )
        option_proxy_reported: set[str] = set()
        fixed_iv_fallback_reported: set[str] = set()

        for position in option_positions:
            exact = position.canonical_occ_symbol or position.exact_symbol
            history = histories.get(exact)
            exact_dates: tuple[date, ...] = ()
            interval_count: Counter[str] = Counter()
            history_diagnostics: Mapping[str, Any] = {}
            if history is not None:
                history_diagnostics = history.history_diagnostics
                # Observed interval P&L uses exact marks available by the causal boundary.
                exact_pnl = {
                    interval.end_session.isoformat(): (
                        position.signed_quantity * interval.exact_pnl_per_contract
                    )
                    for interval in history.intervals
                    if interval.end_observed_at <= as_of
                }
                if exact_pnl:
                    observed_pnl[position.position_id] = dict(sorted(exact_pnl.items()))
                    used = {
                        interval.end_session.isoformat(): interval.method
                        for interval in history.intervals
                        if interval.end_session.isoformat() in exact_pnl
                    }
                    interval_methods[position.position_id] = used
                    interval_count.update(used.values())
                    unique = sorted(set(used.values()))
                    observed_methods[position.position_id] = (
                        unique[0]
                        if len(unique) == 1
                        else "mixed_exact_option_interval_methods"
                    )
                    exact_dates = tuple(
                        date.fromisoformat(day) for day in sorted(exact_pnl)
                    )
                warnings.extend(
                    f"{position.position_id}:{warning}" for warning in history.warnings
                )
                for interval in history.intervals:
                    warnings.extend(
                        f"{position.position_id}:{warning}"
                        for warning in interval.warnings
                    )

            root = (
                position.underlying_symbol or parse_occ_symbol(exact).root
            ).upper()
            current, current_provenance, failure = self._current_option_state(
                position, as_of
            )
            if failure is not None:
                missing.append(position.position_id)
                warnings.append(f"{position.position_id}:{failure}")
                coverage_rows[position.position_id] = _coverage(
                    position,
                    option_plan,
                    exact_dates=exact_dates,
                    interval_method_counts=dict(interval_count),
                    history_diagnostics=history_diagnostics,
                    current_dates=(),
                    missing_fields=(failure,),
                    fail_closed=True,
                )
                continue

            position_scenarios: dict[str, float] = {}
            position_provenance: dict[str, dict[str, Any]] = {}
            scenario_count: Counter[str] = Counter()
            # Reapply historical spot, volatility, and rate shocks to the current option state.
            for end_session in option_plan.required_sessions[1:]:
                day = end_session.isoformat()
                underlying_return = returns.get(root, {}).get(day)
                if underlying_return is None:
                    continue
                selected_shock = self._select_volatility_shock(
                    position, end_session, as_of.date(), volatility_shocks
                )
                if selected_shock is None:
                    if (
                        _as_bool(
                            self.context.get(
                                "risk.history.allow_fixed_current_iv_fallback", True
                            )
                        )
                        and _positive(current.get("iv")) is not None
                    ):
                        selected_shock = (
                            0.0,
                            "fixed_current_iv_no_historical_volatility_shock",
                        )
                        if position.position_id not in fixed_iv_fallback_reported:
                            warnings.append(
                                f"{position.position_id}:historical_volatility_shock_unavailable_fixed_current_iv_greek_bridge_used"
                            )
                            fixed_iv_fallback_reported.add(position.position_id)
                    else:
                        continue
                iv_points, vol_method = selected_shock
                shock_key = (root, vol_method, day)
                if (
                    vol_method.startswith("configured_option_volatility_proxy:")
                    and position.position_id not in option_proxy_reported
                ):
                    factor_provenance = shock_provenance.get(shock_key, {})
                    proxy_rows.append(
                        {
                            "position_id": position.position_id,
                            "source_symbol": exact,
                            "proxy_symbol": root,
                            "proxy_kind": "same_underlying_option_volatility_factor",
                            "method": vol_method,
                            "bucket": factor_provenance.get("bucket"),
                            "source_kind": factor_provenance.get("source_kind"),
                            "calibration_observations": factor_provenance.get(
                                "calibration_observations"
                            ),
                            "calibration_error": factor_provenance.get(
                                "calibration_error"
                            ),
                            "economic_relationship": factor_provenance.get(
                                "economic_relationship"
                            ),
                        }
                    )
                    option_proxy_reported.add(position.position_id)
                rate_points, rate_source = self._rate_shock(position, day)
                if rate_points is None:
                    continue
                pnl, pricing_method, pricing_diagnostics = self._reprice_option(
                    position,
                    as_of=as_of,
                    current=current,
                    underlying_return=underlying_return,
                    iv_points=iv_points,
                    rate_points=rate_points,
                    prefer_fixed_iv_greek_bridge=(
                        vol_method
                        == "fixed_current_iv_no_historical_volatility_shock"
                    ),
                )
                if pnl is None:
                    continue
                position_scenarios[day] = pnl
                scenario_count[pricing_method] += 1
                position_provenance[day] = {
                    "historical_session": day,
                    "underlying_shock": {
                        "return": underlying_return,
                        "symbol": root,
                        "source": return_diagnostics.get(root, {}).get(
                            "source", "exact_option_observation_underlying_close"
                        ),
                    },
                    "volatility_shock": {
                        "iv_points": iv_points,
                        "method": vol_method,
                        **shock_provenance.get(shock_key, {}),
                    },
                    "rate_shock": {
                        "basis_points": rate_points,
                        "source": rate_source,
                    },
                    "pricing": {
                        "method": pricing_method,
                        "current_anchor": current_provenance,
                        "price_multiplier": position.multiplier,
                        "horizon_calendar_days": 1,
                        "terms_version": position.instrument_terms.source_version
                        if position.instrument_terms
                        else None,
                        "diagnostics": pricing_diagnostics,
                    },
                }
            if position_scenarios:
                scenario_pnl[position.position_id] = position_scenarios
                scenario_methods[position.position_id] = (
                    next(iter(scenario_count))
                    if len(scenario_count) == 1
                    else "mixed_current_state_option_revaluation"
                )
                provenance[position.position_id] = position_provenance
                symbols[position.position_id] = exact
            else:
                missing.append(position.position_id)
                warnings.append(
                    f"{position.position_id}:current_state_option_scenarios_unavailable"
                )
            current_dates = tuple(
                date.fromisoformat(day) for day in sorted(position_scenarios)
            )
            coverage_rows[position.position_id] = _coverage(
                position,
                option_plan,
                exact_dates=exact_dates,
                interval_method_counts=dict(interval_count),
                history_diagnostics=history_diagnostics,
                current_dates=current_dates,
                scenario_method_counts=dict(scenario_count),
                missing_fields=(
                    ()
                    if len(current_dates) >= lookback
                    else ("complete_current_state_option_scenarios",)
                ),
                fail_closed=not position_scenarios,
                surface_usage_count=sum(
                    1
                    for row in position_provenance.values()
                    if str(row["volatility_shock"].get("method", "")).startswith(
                        "surface:"
                    )
                    or row["pricing"]["current_anchor"].get("method")
                    == "current_surface"
                ),
                contract_iv_usage_count=sum(
                    1
                    for row in position_provenance.values()
                    if str(row["volatility_shock"].get("method", "")).startswith(
                        "contract_iv:"
                    )
                ),
                proxy_usage_count=sum(
                    1
                    for row in position_provenance.values()
                    if "proxy" in str(row["volatility_shock"].get("method"))
                    or "bucket" in str(row["volatility_shock"].get("method"))
                ),
            )

        unknown = [
            item
            for item in eligible
            if item.asset_type not in {
                "EQUITY", "ETF", "MUTUAL_FUND", "FUTURE", "OPTION"
            }
        ]
        for position in unknown:
            missing.append(position.position_id)
            warnings.append(f"{position.position_id}:unsupported_asset_type")
            coverage_rows[position.position_id] = _coverage(
                position,
                option_plan,
                current_dates=(),
                missing_fields=("unsupported_asset_type",),
                fail_closed=True,
            )

        return HistoricalRevaluationResult(
            unit_pnl=observed_pnl,
            scenario_pnl=scenario_pnl,
            unit_methods=observed_methods,
            scenario_methods=scenario_methods,
            unit_symbols=symbols,
            unit_interval_methods=interval_methods,
            scenario_provenance=provenance,
            position_coverage=tuple(
                coverage_rows[item.position_id]
                for item in eligible
                if item.position_id in coverage_rows
            ),
            missing_position_ids=tuple(sorted(set(missing))),
            proxy_mappings=tuple(proxy_rows),
            return_diagnostics=return_diagnostics,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def load_returns(
        self,
        symbols: Sequence[str],
        *,
        as_of: datetime,
        lookback_sessions: int | None = None,
        calendar_id: str = "XNYS",
        history_kind: str = "factor_proxy",
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]], HistoryWindowPlan]:
        """Load the same exact causal return window used by revaluation."""
        lookback = lookback_sessions or int(
            self.context.get("risk.history.lookback_sessions", 252)
        )
        plan = self.planner.plan(
            as_of=as_of,
            requested_return_count=lookback,
            calendar_id=calendar_id,
            history_kind=history_kind,
            advertised_maximum_days=(
                advertised_option_history_limit(self.reference_client)
                if history_kind == "exact_option_mark"
                else None
            ),
        )
        returns, _marks, diagnostics = self._returns(
            {
                str(symbol).strip().upper(): plan.required_sessions
                for symbol in symbols
                if str(symbol).strip()
            },
            causal_as_of=as_of,
        )
        return returns, diagnostics, plan

    def _returns(
        self,
        sessions_by_symbol: Mapping[str, Sequence[date]],
        *,
        causal_as_of: datetime,
    ) -> tuple[
        dict[str, dict[str, float]],
        dict[str, dict[date, float]],
        dict[str, dict[str, Any]],
    ]:
        marks: dict[str, dict[date, float]] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        allowlist = {
            str(item).strip().upper()
            for item in (
                *self.context.get("risk.history.remote_daily_bar_symbols", ("SHY",)),
                *self.context.get(
                    "risk.pnl.remote_daily_bar_symbols", ("SCC", "TSCO", "VOLT")
                ),
            )
            if str(item).strip()
        }
        required_by_symbol = {
            symbol: tuple(sorted(dict.fromkeys(sessions)))
            for symbol, sessions in sessions_by_symbol.items()
            if sessions
        }
        local_rows_by_symbol: dict[str, list[dict[str, Any]]] = {
            symbol: [] for symbol in required_by_symbol
        }
        local_failures: dict[str, list[dict[str, Any]]] = {
            symbol: [] for symbol in required_by_symbol
        }
        # Batch local history reads for symbols with identical session windows.
        request_groups: dict[tuple[date, date], list[str]] = defaultdict(list)
        for symbol, required in required_by_symbol.items():
            request_groups[(required[0], required[-1])].append(symbol)

        # RefData's bars contract is already vectorized. Most portfolio symbols
        # share one history window, so load that SQL window in one request
        # instead of issuing a serial HTTP/database round trip per symbol.
        for (start, end), symbols in sorted(request_groups.items()):
            ordered_symbols = sorted(symbols)
            try:
                local_groups = self.reference_client.bars(
                    ordered_symbols, start=start, end=end, remote_query=False
                )
            except TypeError:
                try:
                    local_groups = self.reference_client.bars(
                        ordered_symbols, start=start, end=end
                    )
                except Exception as exc:
                    local_groups = []
                    for symbol in ordered_symbols:
                        local_failures[symbol].append(
                            {"stage": "local_sql", **_history_error(exc)}
                        )
            except Exception as exc:
                local_groups = []
                for symbol in ordered_symbols:
                    local_failures[symbol].append(
                        {"stage": "local_sql", **_history_error(exc)}
                    )
            for group in local_groups:
                returned_symbol = str(group.get("symbol") or "").strip().upper()
                if returned_symbol in local_rows_by_symbol:
                    local_rows_by_symbol[returned_symbol].extend(
                        dict(row) for row in group.get("bars") or ()
                    )

        for symbol in sorted(required_by_symbol):
            required = required_by_symbol[symbol]
            local_rows = local_rows_by_symbol[symbol]
            failed = list(local_failures[symbol])
            requested: list[dict[str, Any]] = []
            local_marks = _daily_marks(
                local_rows, set(required), causal_as_of=causal_as_of
            )
            selected = dict(local_marks)
            missing_sessions = set(required) - set(selected)
            # Fill history gaps remotely only for explicitly allowed symbols.
            if symbol in allowlist and missing_sessions:
                for missing_range in _coalesced(required, missing_sessions):
                    for chunk in chunk_session_window(
                        missing_range,
                        int(self.context.get("risk.history.daily_bar_maximum_days", 376)),
                    ):
                        row = {
                            "symbol": symbol,
                            "start": chunk.sessions[0].isoformat(),
                            "end": chunk.sessions[-1].isoformat(),
                            "sessions": [item.isoformat() for item in chunk.sessions],
                        }
                        requested.append(row)
                        try:
                            remote_groups = self.reference_client.bars(
                                [symbol],
                                start=chunk.sessions[0],
                                end=chunk.sessions[-1],
                                remote_query=True,
                            )
                        except Exception as exc:
                            failed.append(
                                {**row, "stage": "remote_repair", **_history_error(exc)}
                            )
                            continue
                        remote_rows = [
                            dict(bar)
                            for group in remote_groups
                            if str(group.get("symbol") or "").strip().upper() == symbol
                            for bar in group.get("bars") or ()
                        ]
                        remote_marks = _daily_marks(
                            remote_rows,
                            set(chunk.sessions),
                            causal_as_of=causal_as_of,
                        )
                        selected = {**remote_marks, **selected}
            marks[symbol] = selected
            diagnostics[symbol] = {
                "required_mark_count": len(required),
                "local_mark_count": len(local_marks),
                "available_mark_count": len(selected),
                "requested_remote_ranges": requested,
                "failed_remote_ranges": failed,
                "remote_repair_allowed": symbol in allowlist,
                "source": (
                    "refdata_sql_first_with_controlled_remote_repair"
                    if requested
                    else "refdata_sql_daily_close"
                ),
            }
        returns: dict[str, dict[str, float]] = {}
        for symbol, values in marks.items():
            required = tuple(sorted(dict.fromkeys(sessions_by_symbol[symbol])))
            # Require adjacent planned sessions so missing marks never create multi-day returns.
            series = {
                right.isoformat(): values[right] / values[left] - 1.0
                for left, right in zip(required, required[1:])
                if left in values and right in values and values[left] > 0
            }
            returns[symbol] = series
            diagnostics[symbol]["return_count"] = len(series)
        return returns, marks, diagnostics

    def _volatility_shocks(
        self,
        histories: Mapping[str, HistoricalOptionSeries],
        plan: HistoryWindowPlan,
    ) -> tuple[
        dict[tuple[str, str, str], float],
        dict[tuple[str, str, str], dict[str, Any]],
    ]:
        required_index = {
            session: index for index, session in enumerate(plan.required_sessions)
        }
        samples: dict[tuple[str, str, str], list[tuple[float, str]]] = defaultdict(list)
        surface_samples: dict[tuple[str, str, str], list[tuple[float, str]]] = defaultdict(list)
        shocks: dict[tuple[str, str, str], float] = {}
        provenance: dict[tuple[str, str, str], dict[str, Any]] = {}
        for symbol, history in sorted(histories.items()):
            root = parse_occ_symbol(symbol).root
            for left, right in zip(history.observations, history.observations[1:]):
                if (
                    left.session_date not in required_index
                    or required_index.get(right.session_date)
                    != required_index[left.session_date] + 1
                ):
                    continue
                bucket = _option_bucket(
                    history.instrument_terms,
                    right.underlying_spot,
                    right.session_date,
                )
                if bucket is None:
                    continue
                left_surface_iv = _surface_iv(left.surface_state)
                right_surface_iv = _surface_iv(right.surface_state)
                if left_surface_iv is not None and right_surface_iv is not None:
                    surface_samples[
                        (root, bucket, right.session_date.isoformat())
                    ].append(
                        (100.0 * (right_surface_iv - left_surface_iv), symbol)
                    )
                if (
                    left.locally_inverted_iv is None
                    or right.locally_inverted_iv is None
                ):
                    continue
                iv_change = 100.0 * (
                    right.locally_inverted_iv - left.locally_inverted_iv
                )
                day = right.session_date.isoformat()
                samples[(root, bucket, day)].append((iv_change, symbol))
                contract_key = (root, f"contract_iv:{symbol}:{bucket}", day)
                shocks[contract_key] = iv_change
                provenance[contract_key] = {
                    "bucket": bucket,
                    "sample_contracts": [symbol],
                    "sample_count": 1,
                    "source_kind": "exact_contract_daily_option_price_derived",
                }
        for (root, bucket, day), values in sorted(surface_samples.items()):
            key = (root, f"surface:{bucket}", day)
            shocks[key] = median(value for value, _ in values)
            provenance[key] = {
                "bucket": bucket,
                "estimator": "median",
                "sample_contracts": sorted({symbol for _, symbol in values}),
                "sample_count": len(values),
                "source_kind": "historical_eod_surface",
            }
        minimum_siblings = max(
            2,
            int(
                self.context.get(
                    "risk.history.minimum_same_underlying_volatility_contracts", 2
                )
            ),
        )
        maximum_bucket_dispersion = max(
            0.0,
            float(
                self.context.get(
                    "risk.history.maximum_same_underlying_volatility_dispersion_points",
                    3.0,
                )
            ),
        )
        # Accept sibling-contract volatility factors only with enough consistent evidence.
        for (root, bucket, day), values in sorted(samples.items()):
            distinct_contracts = sorted({symbol for _, symbol in values})
            if len(distinct_contracts) < minimum_siblings:
                continue
            estimate = median(value for value, _ in values)
            dispersion = median(abs(value - estimate) for value, _ in values)
            if dispersion > maximum_bucket_dispersion:
                continue
            method = "same_underlying_bucket_iv_factor"
            key = (root, f"{method}:{bucket}", day)
            shocks[key] = estimate
            provenance[key] = {
                "bucket": bucket,
                "estimator": "median",
                "sample_contracts": distinct_contracts,
                "sample_count": len(values),
                "cross_contract_mad_points": dispersion,
                "maximum_cross_contract_mad_points": maximum_bucket_dispersion,
                "calibration_quality": "passed_cross_contract_dispersion_gate",
                "source_kind": "daily_option_price_derived",
            }

        configured = self.context.get("risk.history.option_volatility_factors", {})
        maximum_error = float(
            self.context.get("risk.history.maximum_option_volatility_factor_error", 0.10)
        )
        minimum_overlap = int(
            self.context.get("risk.history.minimum_option_volatility_factor_overlap", 126)
        )
        if isinstance(configured, Mapping):
            for root, row in configured.items():
                if not isinstance(row, Mapping):
                    continue
                overlap = int(row.get("calibration_observations") or 0)
                error = _finite(row.get("calibration_error"))
                source_kind = str(row.get("source_kind") or "")
                if (
                    source_kind != "daily_option_price_derived"
                    or overlap < minimum_overlap
                    or error is None
                    or error > maximum_error
                    or not str(row.get("economic_relationship") or "").strip()
                ):
                    continue
                for day, value in (row.get("shocks") or {}).items():
                    shock = _finite(value)
                    if shock is None:
                        continue
                    bucket = str(row.get("bucket") or "all")
                    method = f"configured_option_volatility_proxy:{bucket}"
                    key = (str(root).upper(), method, str(day))
                    shocks[key] = shock
                    provenance[key] = {
                        "bucket": bucket,
                        "source_kind": source_kind,
                        "calibration_observations": overlap,
                        "calibration_error": error,
                        "economic_relationship": row.get("economic_relationship"),
                    }
        return shocks, provenance

    def _select_volatility_shock(
        self,
        position: NormalizedPosition,
        session: date,
        valuation_date: date,
        shocks: Mapping[tuple[str, str, str], float],
    ) -> tuple[float, str] | None:
        terms = position.instrument_terms
        spot = _positive(position.raw_broker_fields.get("underlying_price"))
        if terms is None:
            return None
        root = (position.underlying_symbol or terms.underlying_identifier or "").upper()
        exact = position.canonical_occ_symbol or position.exact_symbol
        bucket = _option_bucket(terms, spot, valuation_date)
        day = session.isoformat()
        candidates = [
            (method, value)
            for (candidate_root, method, candidate_day), value in shocks.items()
            if candidate_root == root
            and candidate_day == day
            and (
                method == f"contract_iv:{exact}:{bucket}"
                or (
                    not method.startswith("contract_iv:")
                    and (method.endswith(f":{bucket}") or method.endswith(":all"))
                )
            )
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                0 if item[0].startswith("contract_iv") else
                -1 if item[0].startswith("surface") else
                1 if item[0].startswith("same_underlying_bucket") else 2,
                item[0],
            )
        )
        method, value = candidates[0]
        return value, method

    def _current_option_state(
        self, position: NormalizedPosition, as_of: datetime
    ) -> tuple[dict[str, float], dict[str, Any], str | None]:
        terms = position.instrument_terms
        if terms is None:
            return {}, {}, "authoritative_option_terms_unavailable"
        if terms.contract_subtype == "futures_option":
            return {}, {}, "unsupported_option_subtype:futures_option"
        if terms.adjusted_contract:
            return {}, {}, "unsupported_option_subtype:adjusted_deliverable"
        if terms.deliverable is None or not terms.deliverable.fully_supported:
            return {}, {}, "authoritative_option_deliverable_unavailable"
        if terms.exercise_style == "unknown" or terms.exercise_style is None:
            return {}, {}, "unsupported_exercise_style:unknown"
        if terms.exercise_style not in {"american", "european"}:
            return {}, {}, f"unsupported_exercise_style:{terms.exercise_style}"
        spot = _positive(position.raw_broker_fields.get("underlying_price"))
        rate = _finite(position.raw_broker_fields.get("risk_free_rate_continuous"))
        if rate is None:
            rate = _finite(self.context.get("risk.rates.configured_flat_continuous_rate"))
        if rate is None and position.expiration_at is not None:
            try:
                payload = self.reference_client.risk_free_curve(as_of.date())
                curve = payload.get("data")
                if isinstance(curve, Mapping):
                    rate = interpolate_risk_free_rate(
                        curve,
                        valuation_date=as_of.date(),
                        tenor_days=max(
                            0.0,
                            (position.expiration_at - as_of).total_seconds() / 86400,
                        ),
                        max_age_days=int(self.context.get("risk.rates.max_age_days", 5)),
                        allow_flat_extrapolation=_as_bool(
                            self.context.get("risk.rates.allow_flat_extrapolation", True)
                        ),
                    ).continuous_annual_rate_decimal
            except Exception:
                rate = None
        dividend = _finite(
            position.raw_broker_fields.get("dividend_yield_continuous")
        )
        if dividend is None:
            dividend = _finite(
                self.context.get("risk.dividends.default_continuous_yield", 0.0)
            )
        if spot is None:
            return {}, {}, "current_underlying_spot_unavailable"
        if rate is None:
            return {}, {}, "current_risk_free_rate_unavailable"
        if dividend is None:
            return {}, {}, "current_dividend_treatment_unavailable"
        if position.mark is None or position.strike is None or position.expiration_at is None:
            return {}, {}, "current_option_anchor_terms_incomplete"

        pricing_spot = spot
        escrowed_diagnostics: dict[str, Any] = {}
        dividend_inputs = position.raw_broker_fields.get("dividend_inputs") or ()
        if dividend_inputs and terms.exercise_style == "american":
            if not _as_bool(
                self.context.get(
                    "risk.options.tian.escrowed_dividend_mode_enabled", True
                )
            ):
                return {}, {}, "discrete_dividend_early_exercise_solver_unavailable"
            try:
                adjustment = escrowed_dividend_adjustment(
                    dividend_inputs,
                    spot=spot,
                    valuation_at=as_of,
                    expiration_at=position.expiration_at,
                    continuous_rate=rate,
                )
            except (TypeError, ValueError):
                return {}, {}, "escrowed_dividend_inputs_unusable"
            if adjustment.event_count:
                pricing_spot = adjustment.adjusted_spot
                dividend = 0.0
                escrowed_diagnostics = {
                    "escrowed_dividend": {
                        "model": adjustment.algorithm_version,
                        "dividend_present_value": adjustment.present_value,
                        "adjusted_spot": adjustment.adjusted_spot,
                        "event_count": adjustment.event_count,
                        "event_dates": [
                            item.isoformat() for item in adjustment.event_dates
                        ],
                    }
                }

        surface = position.raw_broker_fields.get("surface_state") or position.raw_broker_fields.get("current_surface_state")
        surface_iv = _surface_iv(surface)
        # Prefer current surface IV before attempting inversion from the exact mark.
        if surface_iv is not None:
            return (
                {
                    "spot": spot,
                    "pricing_spot": pricing_spot,
                    "rate": rate,
                    "dividend": dividend,
                    "iv": surface_iv,
                },
                {
                    "method": "current_surface",
                    "surface_state": surface,
                    **escrowed_diagnostics,
                },
                None,
            )
        try:
            solved = self.option_resolver.tian_service.solve(
                IvGreeksRequest(
                    valuation_at=as_of,
                    contracts=[
                        IvGreeksContract(
                            request_id=f"current-anchor:{position.position_id}",
                            canonical_occ_symbol=position.canonical_occ_symbol or position.exact_symbol,
                            underlying_symbol=position.underlying_symbol or terms.underlying_identifier or "",
                            put_call=position.put_call or terms.put_call or "CALL",
                            exercise_type=(
                                "EUROPEAN"
                                if terms.exercise_style == "european"
                                else "AMERICAN"
                            ),
                            strike=float(position.strike),
                            expiration_at=position.expiration_at,
                            spot=pricing_spot,
                            mark=position.mark,
                            quote_time=position.quote_as_of or as_of,
                            risk_free_rate=rate,
                            dividend_yield=dividend,
                        )
                    ],
                )
            ).data[0]
            inverted_iv = _positive(getattr(solved, "implied_volatility", None))
        except Exception:
            inverted_iv = None
        if inverted_iv is not None:
            return (
                {
                    "spot": spot,
                    "pricing_spot": pricing_spot,
                    "rate": rate,
                    "dividend": dividend,
                    "iv": inverted_iv,
                },
                {
                    "method": "current_exact_mark_inverted_iv",
                    "mark": position.mark,
                    **escrowed_diagnostics,
                },
                None,
            )
        complete_greeks = all(
            _finite(value) is not None
            for value in (
                position.delta,
                position.gamma,
                position.theta_per_day,
                position.vega_per_1pct,
                position.rho_per_1pct,
            )
        )
        if complete_greeks:
            return (
                {
                    "spot": spot,
                    "pricing_spot": pricing_spot,
                    "rate": rate,
                    "dividend": dividend,
                },
                {
                    "method": "bounded_current_provider_greek_bridge",
                    "mark": position.mark,
                    **escrowed_diagnostics,
                },
                None,
            )
        return {}, {}, "current_mark_without_nonlinear_sensitivity"

    def _rate_shock(
        self, position: NormalizedPosition, day: str
    ) -> tuple[float | None, str]:
        rho = abs(
            float(position.rho_per_1pct or 0.0)
            * float(position.signed_quantity)
            * float(position.multiplier or 0.0)
        )
        material = float(
            self.context.get("risk.history.material_rate_rho_usd_per_1pct", 10_000)
        )
        # Skip rate-history requirements only when configured rho exposure is immaterial.
        if rho < material:
            return 0.0, "rate_shock_not_material_under_configured_rho_gate"
        shocks = self.context.get("risk.history.rate_shocks_basis_points", {})
        value = _finite(shocks.get(day)) if isinstance(shocks, Mapping) else None
        return (
            (value, "observed_rate_factor")
            if value is not None
            else (None, "material_rate_shock_unavailable")
        )

    def _reprice_option(
        self,
        position: NormalizedPosition,
        *,
        as_of: datetime,
        current: Mapping[str, float],
        underlying_return: float,
        iv_points: float,
        rate_points: float,
        prefer_fixed_iv_greek_bridge: bool = False,
    ) -> tuple[float | None, str, dict[str, Any]]:
        spot = float(current["spot"])
        scenario_spot = spot * (1.0 + underlying_return)
        if scenario_spot <= 0:
            return None, "invalid_scenario_spot", {}
        multiplier = position.multiplier
        terms = position.instrument_terms
        if multiplier is None or terms is None or position.expiration_at is None:
            return None, "authoritative_option_terms_unavailable", {}
        shocked_at = as_of + timedelta(days=1)
        if shocked_at >= position.expiration_at:
            if terms.deliverable is None or not terms.deliverable.fully_supported:
                return None, "unsupported_expiration_deliverable", {}
            intrinsic = (
                max(0.0, scenario_spot - float(position.strike or 0.0))
                if position.put_call == "CALL"
                else max(0.0, float(position.strike or 0.0) - scenario_spot)
            )
            pnl = position.signed_quantity * float(multiplier) * (
                intrinsic - float(position.mark or 0.0)
            )
            return pnl, "current_state_authoritative_expiration_revaluation", {
                "expiration_at": position.expiration_at.isoformat()
            }
        complete_greeks = all(
            _finite(value) is not None
            for value in (
                position.delta,
                position.gamma,
                position.theta_per_day,
                position.vega_per_1pct,
                position.rho_per_1pct,
            )
        )
        if (
            prefer_fixed_iv_greek_bridge
            and complete_greeks
            and abs(underlying_return)
            <= float(
                self.context.get(
                    "risk.history.fixed_current_iv_greek_bridge_maximum_spot_return",
                    0.20,
                )
            )
        ):
            move = scenario_spot - spot
            quote_change = (
                float(position.delta) * move
                + 0.5 * float(position.gamma) * move * move
                + float(position.theta_per_day)
                + float(position.vega_per_1pct) * iv_points
                + float(position.rho_per_1pct) * rate_points / 100.0
            )
            return (
                position.signed_quantity * float(multiplier) * quote_change,
                "current_mark_anchored_fixed_iv_delta_gamma_bridge",
                {
                    "domain": "fixed_current_iv_one_day_delta_gamma_theta_v1",
                    "authoritative_current_mark": float(position.mark or 0.0),
                    "underlying_return": underlying_return,
                    "maximum_absolute_underlying_return": float(
                        self.context.get(
                            "risk.history.fixed_current_iv_greek_bridge_maximum_spot_return",
                            0.20,
                        )
                    ),
                },
            )
        iv = _positive(current.get("iv"))
        if iv is not None:
            base_pricing_spot = float(current.get("pricing_spot", spot))
            shocked_pricing_spot = scenario_spot
            pricing_dividend = float(current["dividend"])
            escrowed_diagnostics: dict[str, Any] | None = None
            dividend_inputs = position.raw_broker_fields.get("dividend_inputs") or ()
            shocked_rate = float(current["rate"]) + rate_points / 10_000
            if dividend_inputs and terms.exercise_style == "american":
                try:
                    adjustment = escrowed_dividend_adjustment(
                        dividend_inputs,
                        spot=scenario_spot,
                        valuation_at=shocked_at,
                        expiration_at=position.expiration_at,
                        continuous_rate=shocked_rate,
                    )
                except (TypeError, ValueError):
                    return None, "escrowed_dividend_inputs_unusable", {}
                shocked_pricing_spot = adjustment.adjusted_spot
                pricing_dividend = 0.0
                escrowed_diagnostics = {
                    "model": adjustment.algorithm_version,
                    "base_adjusted_spot": base_pricing_spot,
                    "shocked_adjusted_spot": adjustment.adjusted_spot,
                    "shocked_dividend_present_value": adjustment.present_value,
                    "event_count": adjustment.event_count,
                }
            common = {
                "canonical_occ_symbol": position.canonical_occ_symbol or position.exact_symbol,
                "underlying_symbol": position.underlying_symbol or terms.underlying_identifier or "",
                "put_call": position.put_call or terms.put_call or "CALL",
                "exercise_type": (
                    "EUROPEAN"
                    if terms.exercise_style == "european"
                    else "AMERICAN"
                ),
                "strike": float(position.strike or terms.strike or 0.0),
                "expiration_at": position.expiration_at,
                "source_valuation_at": as_of,
                "source_spot": base_pricing_spot,
                "source_mark": position.mark,
                "implied_volatility": iv,
                "dividend_yield": pricing_dividend,
            }
            try:
                # Reuse the unchanged base valuation across historical one-day shocks.
                base_key = (
                    position.position_id,
                    as_of,
                    common["canonical_occ_symbol"],
                    common["underlying_symbol"],
                    common["put_call"],
                    common["exercise_type"],
                    common["strike"],
                    common["expiration_at"],
                    base_pricing_spot,
                    float(current["rate"]),
                    iv,
                    pricing_dividend,
                )
                base = self._option_base_reprice_cache.get(base_key)
                if base is None:
                    base = self.option_resolver.tian_service.reprice(
                        TianRepriceRequest(
                            valuation_at=as_of,
                            contracts=[
                                TianRepriceContract(
                                    request_id=f"history-base:{position.position_id}",
                                    scenario_spot=base_pricing_spot,
                                    risk_free_rate=float(current["rate"]),
                                    **common,
                                )
                            ],
                        )
                    ).data[0]
                    self._option_base_reprice_cache[base_key] = base
                shocked = self.option_resolver.tian_service.reprice(
                    TianRepriceRequest(
                        valuation_at=shocked_at,
                        contracts=[
                            TianRepriceContract(
                                request_id=f"history-shock:{position.position_id}",
                                scenario_spot=shocked_pricing_spot,
                                iv_shift_points=iv_points,
                                risk_free_rate=shocked_rate,
                                **common,
                            )
                        ],
                    )
                ).data[0]
                if base.status == "ok" and shocked.status == "ok":
                    authoritative_mark = float(position.mark or 0.0)
                    pnl = (
                        position.signed_quantity
                        * float(multiplier)
                        * (float(shocked.price) - authoritative_mark)
                    )
                    method = (
                        "current_exact_mark_anchored_quantlib_european_repricing"
                        if terms.exercise_style == "european"
                        else "current_exact_mark_anchored_full_tian_repricing"
                    )
                    diagnostics = {
                        "base": base.solver_diagnostics,
                        "shocked": shocked.solver_diagnostics,
                        "authoritative_current_mark": authoritative_mark,
                        "base_model_price": float(base.price),
                        "base_model_mark_difference": (
                            float(base.price) - authoritative_mark
                        ),
                    }
                    if escrowed_diagnostics is not None:
                        diagnostics["escrowed_dividend"] = escrowed_diagnostics
                    return pnl, method, diagnostics
            except Exception:
                pass
        # A Greek approximation is allowed only inside its calibrated fallback domain.
        if _bridge_domain(
            position, underlying_return, iv_points, as_of, self.context
        ) and complete_greeks:
            move = scenario_spot - spot
            quote_change = (
                float(position.delta) * move
                + 0.5 * float(position.gamma) * move * move
                + float(position.theta_per_day)
                + float(position.vega_per_1pct) * iv_points
                + float(position.rho_per_1pct) * rate_points / 100.0
            )
            return (
                position.signed_quantity * float(multiplier) * quote_change,
                "bounded_current_greek_bridge",
                {"domain": "calibrated_bridge_v1"},
            )
        return None, "full_repricing_failed_and_bridge_outside_approved_domain", {}


def _coverage(
    position: NormalizedPosition,
    plan: HistoryWindowPlan,
    *,
    current_dates: Sequence[date],
    exact_dates: Sequence[date] = (),
    interval_method_counts: Mapping[str, int] | None = None,
    scenario_method_counts: Mapping[str, int] | None = None,
    history_diagnostics: Mapping[str, Any] | None = None,
    requested_remote_ranges: tuple[dict[str, Any], ...] = (),
    failed_remote_ranges: tuple[dict[str, Any], ...] = (),
    missing_fields: tuple[str, ...] = (),
    fail_closed: bool = False,
    proxy_usage_count: int = 0,
    surface_usage_count: int = 0,
    contract_iv_usage_count: int = 0,
) -> PositionHistoryCoverage:
    diagnostics = history_diagnostics or {}
    interval_counts = dict(interval_method_counts or {})
    scenario_counts = dict(scenario_method_counts or {})
    requested = requested_remote_ranges or tuple(
        diagnostics.get("requested_remote_ranges") or ()
    )
    failed = failed_remote_ranges or tuple(
        diagnostics.get("failed_remote_ranges") or ()
    )
    return PositionHistoryCoverage(
        position_id=position.position_id,
        exact_symbol=position.canonical_occ_symbol or position.exact_symbol,
        asset_type=position.asset_type,
        economic_position_id=position.economic_position_id,
        required_first_session=plan.first_required_session,
        required_last_session=plan.last_required_session,
        requested_return_count=plan.requested_return_count,
        required_mark_count=plan.required_mark_count,
        sql_mark_count=int(diagnostics.get("sql_mark_count") or 0),
        remote_mark_count=int(diagnostics.get("remote_mark_count") or 0),
        exact_interval_count=len(exact_dates),
        current_state_scenario_count=len(current_dates),
        first_usable_exact_interval=min(exact_dates) if exact_dates else None,
        last_usable_exact_interval=max(exact_dates) if exact_dates else None,
        first_usable_current_state_scenario=min(current_dates) if current_dates else None,
        last_usable_current_state_scenario=max(current_dates) if current_dates else None,
        interval_method_counts=interval_counts,
        scenario_method_counts=scenario_counts,
        surface_usage_count=max(
            surface_usage_count, interval_counts.get("surface", 0)
        ),
        contract_iv_usage_count=max(
            contract_iv_usage_count, interval_counts.get("contract_iv", 0)
        ),
        greek_bridge_usage_count=(
            interval_counts.get("bounded_greek_bridge", 0)
            + scenario_counts.get("bounded_current_greek_bridge", 0)
        ),
        marks_only_usage_count=interval_counts.get("observed_price_only", 0),
        proxy_usage_count=proxy_usage_count,
        requested_remote_ranges=requested,
        failed_remote_ranges=failed,
        missing_fields=missing_fields,
        materiality="material_fail_closed" if _material(position) else "count_coverage",
        fail_closed=fail_closed,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
    )


def _daily_marks(
    rows: Sequence[Mapping[str, Any]],
    required: set[date],
    *,
    causal_as_of: datetime,
) -> dict[date, float]:
    boundary = causal_as_of.astimezone(timezone.utc)
    output: dict[date, float] = {}
    for row in sorted(rows, key=lambda item: str(item.get("timestamp") or item.get("session_date") or "")):
        observed = _datetime_value(
            row.get("observed_at")
            or row.get("available_at")
            or row.get("timestamp")
        )
        # Exclude observations published after the requested valuation boundary.
        if observed is not None and observed > boundary:
            continue
        session = _date_value(
            row.get("session_date") or str(row.get("timestamp") or "")[:10]
        )
        close = _positive(row.get("close"))
        if session in required and close is not None:
            output[session] = close
    return output


def _datetime_value(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _history_error(exc: Exception) -> dict[str, Any]:
    diagnostic = getattr(exc, "diagnostic", None)
    structured = diagnostic() if callable(diagnostic) else None
    if isinstance(structured, Mapping):
        return {
            "error_code": str(
                structured.get("server_error_code") or type(exc).__name__
            ),
            "error": dict(structured),
        }
    return {"error_code": type(exc).__name__}


def _coalesced(
    expected: Sequence[date], missing: set[date]
) -> tuple[tuple[date, ...], ...]:
    result: list[list[date]] = []
    current: list[date] = []
    for session in expected:
        if session in missing:
            current.append(session)
        elif current:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return tuple(tuple(row) for row in result)


def _option_observation_returns(
    histories: Mapping[str, HistoricalOptionSeries], required: Sequence[date]
) -> dict[str, dict[str, float]]:
    index = {session: offset for offset, session in enumerate(required)}
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for symbol, history in histories.items():
        root = parse_occ_symbol(symbol).root
        for left, right in zip(history.observations, history.observations[1:]):
            if (
                left.underlying_spot is None
                or right.underlying_spot is None
                or left.underlying_spot <= 0
                or index.get(right.session_date) != index.get(left.session_date, -2) + 1
            ):
                continue
            samples[(root, right.session_date.isoformat())].append(
                right.underlying_spot / left.underlying_spot - 1.0
            )
    output: dict[str, dict[str, float]] = defaultdict(dict)
    for (root, day), values in samples.items():
        output[root][day] = median(values)
    return dict(output)


def _option_bucket(
    terms: InstrumentTerms, spot: float | None, session: date
) -> str | None:
    if terms.strike is None or spot is None or spot <= 0 or terms.expiration_at is None:
        return None
    days = max(0, (terms.expiration_at.date() - session).days)
    maturity = "0_30d" if days <= 30 else "31_90d" if days <= 90 else "91_180d" if days <= 180 else "181d_plus"
    distance = abs(math.log(float(spot) / float(terms.strike)))
    moneyness = "atm" if distance <= 0.05 else "near" if distance <= 0.15 else "far"
    return f"{maturity}:{moneyness}"


def _surface_iv(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("implied_volatility", "iv", "atm_iv"):
        converted = _positive(value.get(key))
        if converted is not None:
            return converted
    return None


def _bridge_domain(
    position: NormalizedPosition,
    underlying_return: float,
    iv_points: float,
    as_of: datetime,
    context: Mapping[str, Any],
) -> bool:
    calibration = context.get("risk.option_history.greek_bridge_calibration", {})
    if not isinstance(calibration, Mapping):
        return False
    median_error = _finite(calibration.get("median_absolute_error_fraction"))
    p95_error = _finite(calibration.get("p95_absolute_error_fraction"))
    maximum_error = _finite(calibration.get("maximum_absolute_error_fraction"))
    if (
        median_error is None
        or p95_error is None
        or maximum_error is None
        or median_error > 0.05
        or p95_error > 0.10
        or maximum_error > 0.15
        or not bool(calibration.get("loss_sign_preserved"))
        or not bool(calibration.get("tail_ordering_preserved"))
    ):
        return False
    if abs(underlying_return) > float(
        context.get("risk.option_history.greek_bridge_maximum_spot_return", 0.10)
    ):
        return False
    if abs(iv_points) > float(
        context.get("risk.option_history.greek_bridge_maximum_iv_points", 15.0)
    ):
        return False
    gamma = _finite(position.gamma)
    vega = _finite(position.vega_per_1pct)
    if (
        gamma is None
        or abs(gamma)
        > float(
            context.get(
                "risk.option_history.greek_bridge_maximum_absolute_gamma", 0.20
            )
        )
        or vega is None
        or abs(vega)
        > float(
            context.get(
                "risk.option_history.greek_bridge_maximum_absolute_vega_per_1pct",
                5.0,
            )
        )
    ):
        return False
    if position.expiration_at is None or position.strike is None:
        return False
    days = (position.expiration_at - as_of).total_seconds() / 86400.0
    if not (
        float(context.get("risk.option_history.greek_bridge_minimum_dte", 7))
        <= days
        <= float(context.get("risk.option_history.greek_bridge_maximum_dte", 730))
    ):
        return False
    spot = _positive(position.raw_broker_fields.get("underlying_price"))
    if spot is None:
        return False
    dividend = _finite(
        position.raw_broker_fields.get("dividend_yield_continuous")
    )
    if dividend is None:
        dividend = _finite(
            context.get("risk.dividends.default_continuous_yield", 0.0)
        )
    if (
        dividend is None
        or dividend < 0
        or dividend
        > float(
            context.get(
                "risk.option_history.greek_bridge_maximum_dividend_yield", 0.08
            )
        )
    ):
        return False
    log_moneyness = abs(math.log(spot / float(position.strike)))
    if log_moneyness > float(
        context.get("risk.option_history.greek_bridge_maximum_log_moneyness", 0.25)
    ):
        return False
    terms = position.instrument_terms
    early_exercise_review_days = float(
        context.get(
            "risk.option_history.greek_bridge_early_exercise_review_days", 30
        )
    )
    early_exercise_dividend = float(
        context.get(
            "risk.option_history.greek_bridge_early_exercise_dividend_yield", 0.01
        )
    )
    if (
        position.put_call == "CALL"
        and dividend >= early_exercise_dividend
        and days <= early_exercise_review_days
        and spot > float(position.strike)
    ):
        return False
    return bool(
        terms is not None
        and terms.exercise_style == "american"
        and not terms.adjusted_contract
    )


def _material(position: NormalizedPosition) -> bool:
    if position.asset_type in {"OPTION", "FUTURE", "UNKNOWN"}:
        return True
    return position.signed_quantity < 0


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _positive(value: Any) -> float | None:
    converted = _finite(value)
    return converted if converted is not None and converted > 0 else None


def _date_value(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _field_valid(position: NormalizedPosition, field: str) -> bool:
    validity = position.field_validity.get(field)
    if validity is not None:
        return bool(validity.valid_for_risk)
    if field == "historical_mark_revaluation_terms":
        return (
            position.instrument_terms is not None
            and position.instrument_terms.price_multiplier is not None
        )
    return getattr(position, field, None) is not None


__all__ = [
    "CurrentPortfolioHistoricalRevaluationEngine",
    "HistoricalRevaluationResult",
    "PositionHistoryCoverage",
]
