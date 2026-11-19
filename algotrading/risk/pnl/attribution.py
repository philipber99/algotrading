"""Causal option-surface and exact-price P&L attribution.

Exact marks remain the accounting source of truth. Supported daily and
execution-fill intervals are revalued; only model/mark differences and
intervals with unavailable causal inputs remain in ``Model residual``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import math
from typing import Any, Literal
import uuid

from algotrading.refdata.api.option_symbols import OccSymbol, parse_occ_symbol
from algotrading.refdata.options.solvers.Registry import SolverRegistry
from algotrading.refdata.options.solvers.Types import (
    IvHint,
    PricingRequest,
    RepricingMethod,
    SolverBatchJob,
    SolverError,
    SolverOperation,
)
from algotrading.risk.market_inputs.client import (
    ReferenceDataClient,
    ReferenceDataClientError,
)
from algotrading.risk.market_inputs.dividends import derive_trailing_cash_yield
from algotrading.risk.market_inputs.rates import interpolate_risk_free_rate
from algotrading.risk.instruments.terms import build_instrument_terms
from algotrading.risk.option_history.resolver import (
    HistoricalOptionObservation,
    resolve_historical_option_interval,
)
from algotrading.risk.option_prices.models import (
    ChainSnapshotFlag,
    ChainSnapshotStatus,
    OptionSurfaceObservation,
    OptionSurfaceSnapshot,
    SecurityStatus,
    SurfaceObservationSource,
)
from algotrading.risk.option_prices.repository import (
    OptionPriceRepository,
    OptionPriceRepositoryUnavailable,
    OptionSurfaceHistoryLimitExceeded,
)

from .models import PnlFactorAttribution
from .reconstruction import EPSILON, Lifecycle, inventory_at
from .valuation import (
    ExactOptionObservation,
    MarketMarkBook,
    _money,
    _multipliers,
    _required_cash,
)


AttributionMethod = Literal[
    "surface_revaluation", "price_history_fallback", "mark_revaluation"
]

FACTOR_ORDER = (
    "Underlying movement",
    "Volatility",
    "Skew",
    "Term structure",
    "Time decay",
    "Rates",
    "Dividends",
    "Commission & fees",
    "Option repricing residual",
    "Model residual",
)

_WATERFALL_FACTORS = (
    "time",
    "underlying",
    "rates",
    "dividends",
    "volatility",
    "term_structure",
    "skew",
)

_FALLBACK_WATERFALL_FACTORS = (
    "time",
    "underlying",
    "rates",
    "dividends",
    "volatility",
)


@dataclass(frozen=True)
class AttributionResult:
    factors: list[PnlFactorAttribution]
    attribution_method: AttributionMethod
    surface_coverage_percent: float | None
    warnings: list[str]
    calculation_method: str = "mark_revaluation"
    input_mode: str = "exact_contract_price_history"
    methodology_version: str = "pnl_option_attribution_v2"


@dataclass(frozen=True)
class OptionMarketState:
    at: datetime
    spot: float
    rate: float
    dividend_yield: float
    volatility: float
    term_structure: float = 0.0
    skew: float = 0.0

    @property
    def total_iv(self) -> float:
        return self.volatility + self.term_structure + self.skew


@dataclass(frozen=True)
class _OptionInterval:
    lifecycle: Lifecycle
    symbol: str
    quantity: float
    multiplier: float
    expiration_at: datetime | None
    start: ExactOptionObservation
    end: ExactOptionObservation


@dataclass(frozen=True)
class _SmileCurve:
    atm_iv: float
    points: tuple[tuple[float, float], ...]
    forward: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _ResolvedMarketInput:
    value: float | None
    observed: bool


class _SurfaceUnusable(RuntimeError):
    pass


class _AttributionSolverFailure(RuntimeError):
    pass


class TianBatchRepricingAdapter:
    """Small fixed-IV adapter around the repository's installed Tian registry."""

    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        registry: SolverRegistry | Any | None = None,
    ) -> None:
        self.context = context
        self.registry = registry or SolverRegistry(context)
        requested = str(context.get("risk.options.requested_dtype", "float64")).lower()
        self.requested_dtype = requested if requested in {"float32", "float64"} else "float64"

    def solve_ivs(
        self,
        requests: Sequence[PricingRequest],
        target_prices: Sequence[float],
        hints: Sequence[float | None],
    ) -> list[float | None]:
        jobs = []
        for index, (request, target, hint) in enumerate(
            zip(requests, target_prices, hints)
        ):
            iv_hints = (
                (IvHint(float(hint), "provider_quote"),)
                if _valid_iv(hint)
                else ()
            )
            jobs.append(
                SolverBatchJob(
                    job_id=str(uuid.uuid4()),
                    owner_run_id=request.snapshot_id,
                    owner_snapshot_id=request.snapshot_id,
                    output_index=index,
                    operation=SolverOperation.SOLVE_IV_ONLY,
                    request=request,
                    method=RepricingMethod.TIAN,
                    method_selection_reason="risk_pnl_attribution_tian",
                    target_price=float(target),
                    iv_hints=iv_hints,
                    requested_dtype=self.requested_dtype,
                )
            )
        try:
            results = self.registry.solve_iv_only_many(jobs)
        except (SolverError, ArithmeticError, FloatingPointError, ValueError) as exc:
            raise _AttributionSolverFailure("tian_iv_batch_failed") from exc
        output: list[float | None] = []
        for result in results:
            iv = getattr(result, "iv", None)
            output.append(float(iv) if _valid_iv(iv) else None)
        return output

    def prices(
        self,
        occ: OccSymbol,
        states: Sequence[OptionMarketState],
        *,
        expiration_at: datetime,
    ) -> list[float]:
        converted: list[float | None] = [None] * len(states)
        live_indices: list[int] = []
        requests: list[PricingRequest] = []
        volatilities: list[float] = []
        for index, state in enumerate(states):
            if state.at >= expiration_at:
                converted[index] = _intrinsic_value(occ, state.spot)
                continue
            live_indices.append(index)
            requests.append(
                _pricing_request(
                    occ, state, self.requested_dtype, expiration_at=expiration_at
                )
            )
            volatilities.append(state.total_iv)
        if any(not _valid_iv(value) for value in volatilities):
            raise _AttributionSolverFailure("invalid_hybrid_iv")
        if requests:
            try:
                values = self.registry.price_many(
                    requests,
                    volatilities,
                    RepricingMethod.TIAN,
                    [self.requested_dtype] * len(requests),
                )
            except (SolverError, ArithmeticError, FloatingPointError, ValueError) as exc:
                raise _AttributionSolverFailure("tian_price_batch_failed") from exc
            if len(values) != len(requests):
                raise _AttributionSolverFailure("tian_price_batch_nonfinite")
            for index, value in zip(live_indices, values):
                converted[index] = float(value)
        if any(value is None or not math.isfinite(value) for value in converted):
            raise _AttributionSolverFailure("tian_price_batch_nonfinite")
        return [float(value) for value in converted if value is not None]


class _MarketInputCache:
    def __init__(
        self,
        reference_client: ReferenceDataClient,
        context: Mapping[str, Any],
        *,
        analysis_start: datetime,
        analysis_end: datetime,
    ) -> None:
        self.reference_client = reference_client
        self.context = context
        self.analysis_start = analysis_start
        self.analysis_end = analysis_end
        self.curves: dict[date, Mapping[str, Any] | None] = {}
        self.rates: dict[tuple[date, date], _ResolvedMarketInput] = {}
        self.dividend_events: dict[str, list[dict[str, Any]] | None] = {}
        self.dividends: dict[tuple[str, date, float], _ResolvedMarketInput] = {}
        self.underlying_closes: dict[tuple[str, date], float | None] = {}
        self.underlying_execution_prices: dict[tuple[str, datetime], float | None] = {}
        self.warnings: list[str] = []

    def rate(self, at: datetime, expiration_at: datetime) -> _ResolvedMarketInput:
        key = (at.date(), expiration_at.date())
        if key in self.rates:
            return self.rates[key]
        curve = self.curves.get(at.date(), _MISSING)
        if curve is _MISSING:
            try:
                payload = self.reference_client.risk_free_curve(as_of=at.date())
                candidate = payload.get("data") if isinstance(payload, Mapping) else None
                curve = candidate if isinstance(candidate, Mapping) else None
            except (ReferenceDataClientError, AttributeError, TypeError, ValueError):
                curve = None
            self.curves[at.date()] = curve
        value = None
        if curve is not None:
            tenor_days = max(
                0.0,
                (expiration_at - at).total_seconds() / 86400.0,
            )
            try:
                value = interpolate_risk_free_rate(
                    curve,
                    valuation_date=at.date(),
                    tenor_days=tenor_days,
                    max_age_days=int(self.context.get("risk.rates.max_age_days", 5)),
                    allow_flat_extrapolation=_as_bool(
                        self.context.get("risk.rates.allow_flat_extrapolation", True)
                    ),
                ).continuous_annual_rate_decimal
            except (TypeError, ValueError):
                value = None
        observed = value is not None
        if not observed:
            configured = self.context.get("risk.rates.configured_flat_continuous_rate")
            if configured not in (None, ""):
                try:
                    candidate = float(configured)
                    value = candidate if math.isfinite(candidate) else None
                except (TypeError, ValueError):
                    value = None
        resolved = _ResolvedMarketInput(value=value, observed=observed)
        self.rates[key] = resolved
        return resolved

    def dividend(
        self, root: str, at: datetime, spot: float
    ) -> _ResolvedMarketInput:
        key = (root, at.date(), round(float(spot), 8))
        if key in self.dividends:
            return self.dividends[key]
        if root not in self.dividend_events:
            try:
                events = self.reference_client.dividends(
                    root,
                    start=self.analysis_start.date() - timedelta(days=365),
                    end=self.analysis_end.date(),
                )
                self.dividend_events[root] = [dict(item) for item in events]
            except (ReferenceDataClientError, AttributeError, TypeError, ValueError):
                self.dividend_events[root] = None
        events = self.dividend_events[root]
        value = None
        if events is not None:
            try:
                derived = derive_trailing_cash_yield(
                    events,
                    spot=spot,
                    valuation_date=at.date(),
                )
                value = derived.continuous_yield
                self.warnings.extend(
                    f"{root}:{warning}" for warning in derived.warnings
                )
            except (TypeError, ValueError):
                value = None
        observed = value is not None
        if not observed:
            configured = self.context.get(
                "risk.dividends.default_continuous_yield", 0.0
            )
            if configured not in (None, ""):
                try:
                    candidate = float(configured)
                    value = candidate if math.isfinite(candidate) else None
                except (TypeError, ValueError):
                    value = None
        resolved = _ResolvedMarketInput(value=value, observed=observed)
        self.dividends[key] = resolved
        return resolved

    def underlying_close(
        self, root: str, session_date: date, available_at: datetime
    ) -> float | None:
        key = (root, session_date)
        if key in self.underlying_closes:
            return self.underlying_closes[key]
        rows: Sequence[Mapping[str, Any]] = ()
        try:
            rows = self.reference_client.bars(
                [root],
                start=session_date,
                end=session_date,
                remote_query=False,
            )
        except TypeError:
            try:
                rows = self.reference_client.bars(
                    [root], start=session_date, end=session_date
                )
            except (ReferenceDataClientError, AttributeError, TypeError, ValueError):
                rows = ()
        except (ReferenceDataClientError, AttributeError, ValueError):
            rows = ()
        value = None
        for row in rows:
            if str(row.get("symbol") or "").strip().upper() != root:
                continue
            for bar in row.get("bars") or ():
                bar_session = _date_value(bar.get("session_date"))
                if bar_session is None:
                    bar_session = _bar_session_date(bar.get("timestamp"))
                if bar_session != session_date:
                    continue
                published_at = _aware_value(bar.get("timestamp"))
                if published_at is None or published_at > available_at:
                    continue
                close = _positive_float(bar.get("close"))
                if close is not None:
                    value = close
        self.underlying_closes[key] = value
        return value

    def underlying_at_execution(self, root: str, at: datetime) -> float | None:
        """Resolve the last completed one-minute close before an execution."""
        boundary = at.astimezone(timezone.utc)
        key = (root, boundary)
        if key in self.underlying_execution_prices:
            return self.underlying_execution_prices[key]
        rows: Sequence[Mapping[str, Any]] = ()
        try:
            rows = self.reference_client.historical_minute_bars(
                root,
                start_at=boundary - timedelta(minutes=10),
                end_at=boundary,
            )
        except (ReferenceDataClientError, AttributeError, TypeError, ValueError):
            rows = ()
        eligible: list[tuple[datetime, float]] = []
        for bar in rows:
            timestamp = _aware_value(bar.get("timestamp"))
            close = _positive_float(bar.get("close"))
            if timestamp is None or close is None:
                continue
            if timestamp + timedelta(minutes=1) <= boundary:
                eligible.append((timestamp, close))
        value = max(eligible, default=(boundary, None), key=lambda item: item[0])[1]
        self.underlying_execution_prices[key] = value
        return value


_MISSING = object()


def attribute_selection(
    *,
    lifecycles: Sequence[Lifecycle],
    marks: MarketMarkBook,
    option_price_repository: OptionPriceRepository | None,
    reference_client: ReferenceDataClient,
    context: Mapping[str, Any],
    analysis_start: datetime,
    analysis_end: datetime,
    lifecycle_pnl: float,
    fees: float,
    solver_registry: SolverRegistry | Any | None = None,
) -> AttributionResult:
    """Attribute supported market moves and reconcile exactly to lifecycle P&L."""
    warnings: list[str] = []
    values = {factor: 0.0 for factor in FACTOR_ORDER}
    values["Commission & fees"] = fees
    has_options = any(
        execution.asset_type == "OPTION"
        for lifecycle in lifecycles
        for execution in lifecycle.executions
    )
    # Separate directly valued linear instruments before explaining option drivers.
    values["Underlying movement"] += _direct_instrument_pnl(
        lifecycles, marks, analysis_end
    )
    if not has_options:
        return _finish(
            values,
            lifecycle_pnl=lifecycle_pnl,
            method="mark_revaluation",
            coverage=None,
            warnings=warnings,
        )

    intervals = _eligible_option_intervals(
        lifecycles,
        marks,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        warnings=warnings,
    )
    if not intervals:
        return _finish(
            values,
            lifecycle_pnl=lifecycle_pnl,
            method="price_history_fallback",
            coverage=0.0,
            warnings=warnings,
        )

    option_symbols = sorted({interval.symbol for interval in intervals})
    snapshots: list[OptionSurfaceSnapshot] = []
    if option_price_repository is None:
        warnings.append("surface_repository_unavailable")
    else:
        try:
            snapshots = option_price_repository.query_surface_history(
                option_symbols,
                start_at=analysis_start
                - timedelta(
                    days=max(
                        0,
                        int(context.get("risk.pnl.surface_history_lookback_days", 7)),
                    )
                ),
                end_at=analysis_end + timedelta(microseconds=1),
            )
        except OptionSurfaceHistoryLimitExceeded:
            warnings.append("surface_history_limit_exceeded")
        except OptionPriceRepositoryUnavailable:
            warnings.append("surface_repository_unavailable")
    if not snapshots:
        warnings.append("surface_history_unavailable")

    # Index surface snapshots by underlying and expiration for repeated interval lookups.
    by_pair: dict[tuple[str, date], list[OptionSurfaceSnapshot]] = {}
    for snapshot in snapshots:
        if _aware_datetime(snapshot.header.captured_at) is None:
            warnings.append(f"surface_malformed_snapshot:{snapshot.header.snapshot_id}")
            continue
        key = (
            snapshot.header.underlying_symbol,
            snapshot.header.expiration_date,
        )
        by_pair.setdefault(key, []).append(snapshot)
    for rows in by_pair.values():
        rows.sort(
            key=lambda item: (
                item.header.captured_at.timestamp(),
                item.header.snapshot_id,
            )
        )
    expiration_terms: dict[tuple[str, date], datetime] = {}
    for interval in intervals:
        occ = parse_occ_symbol(interval.symbol)
        if interval.expiration_at is None:
            warnings.append(f"option_expiration_terms_unavailable:{interval.symbol}")
            continue
        key = (occ.root, occ.expiration)
        existing = expiration_terms.get(key)
        if existing is not None and existing != interval.expiration_at:
            warnings.append(f"option_expiration_terms_inconsistent:{interval.symbol}")
            expiration_terms.pop(key, None)
            continue
        expiration_terms[key] = interval.expiration_at

    pricer: TianBatchRepricingAdapter | None
    try:
        pricer = TianBatchRepricingAdapter(context, registry=solver_registry)
    except (SolverError, ImportError, ModuleNotFoundError, TypeError, ValueError):
        warnings.append("attribution_solver_unavailable")
        pricer = None
    inputs = _MarketInputCache(
        reference_client,
        context,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
    )
    # Reuse fitted smiles across intervals with identical snapshot and market inputs.
    fit_cache: dict[tuple[int, datetime, float, float], _SmileCurve] = {}
    surfaced = 0
    surface_eligible = 0
    for interval in intervals:
        daily_surface_interval = not (
            interval.start.is_execution_fill or interval.end.is_execution_fill
        )
        if daily_surface_interval:
            surface_eligible += 1
        occ = parse_occ_symbol(interval.symbol)
        start_snapshot = _causal_snapshot(by_pair, occ, interval.start.at)
        end_snapshot = _causal_snapshot(by_pair, occ, interval.end.at)
        start_spot = _endpoint_spot(interval.start, start_snapshot)
        end_spot = _endpoint_spot(interval.end, end_snapshot)
        if start_spot is None and interval.start.is_execution_fill:
            start_spot = inputs.underlying_at_execution(occ.root, interval.start.at)
        if end_spot is None and interval.end.is_execution_fill:
            end_spot = inputs.underlying_at_execution(occ.root, interval.end.at)
        if start_spot is None:
            start_spot = inputs.underlying_close(
                occ.root,
                _observation_session(interval.start),
                interval.start.at,
            )
        if end_spot is None:
            end_spot = inputs.underlying_close(
                occ.root,
                _observation_session(interval.end),
                interval.end.at,
            )
        if start_spot is None:
            warnings.append(
                "fallback_underlying_history_unavailable:"
                f"{occ.root}:{_observation_session(interval.start).isoformat()}"
            )
        if end_spot is None:
            warnings.append(
                "fallback_underlying_history_unavailable:"
                f"{occ.root}:{_observation_session(interval.end).isoformat()}"
            )
        expiration_at = interval.expiration_at
        if expiration_at is not None:
            start_rate, end_rate, rates_supported = _held_constant_inputs(
                inputs.rate(interval.start.at, expiration_at),
                inputs.rate(interval.end.at, expiration_at),
            )
        else:
            start_rate = end_rate = None
            rates_supported = False
        start_dividend, end_dividend, dividends_supported = _held_constant_inputs(
            inputs.dividend(occ.root, interval.start.at, start_spot)
            if start_spot is not None
            else _ResolvedMarketInput(None, False),
            inputs.dividend(occ.root, interval.end.at, end_spot)
            if end_spot is not None
            else _ResolvedMarketInput(None, False),
        )
        if not rates_supported:
            warnings.append(
                "fallback_rate_held_constant:"
                f"{_observation_session(interval.start).isoformat()}"
            )
        if not dividends_supported:
            warnings.append(
                "fallback_dividend_held_constant:"
                f"{occ.root}:{_observation_session(interval.start).isoformat()}"
            )

        contributions: dict[str, float] | None = None
        interval_label = (
            f"{interval.symbol}:{interval.start.at.date().isoformat()}"
        )
        if start_snapshot is None or end_snapshot is None:
            warnings.append(f"surface_endpoint_unavailable:{interval_label}")
        if (
            start_snapshot is not None
            and end_snapshot is not None
            and pricer is not None
            and expiration_at is not None
            and start_rate is not None
            and end_rate is not None
            and start_dividend is not None
            and end_dividend is not None
        ):
            try:
                start_state, start_warnings = _surface_state(
                    occ,
                    interval.start.at,
                    by_pair,
                    expiration_terms,
                    inputs,
                    pricer,
                    fit_cache,
                    target_rate=start_rate,
                    target_dividend=start_dividend,
                )
                end_state, end_warnings = _surface_state(
                    occ,
                    interval.end.at,
                    by_pair,
                    expiration_terms,
                    inputs,
                    pricer,
                    fit_cache,
                    target_rate=end_rate,
                    target_dividend=end_dividend,
                )
                contributions = _symmetric_waterfall(
                    occ,
                    start_state,
                    end_state,
                    pricer,
                    expiration_at=expiration_at,
                )
                warnings.extend(start_warnings)
                warnings.extend(end_warnings)
            except (_SurfaceUnusable, _AttributionSolverFailure, SolverError, ValueError) as exc:
                contributions = None
                reason = str(exc).strip().replace(" ", "_") or type(exc).__name__
                warnings.append(f"surface_fit_unusable:{interval_label}:{reason}")
        elif start_snapshot is not None and end_snapshot is not None:
            warnings.append(f"surface_inputs_unavailable:{interval_label}")
        if contributions is not None and daily_surface_interval:
            surfaced += 1
            warnings.append(f"historical_option_interval_method:{interval_label}:surface")
            contributions = _shared_interval_contributions(
                occ,
                interval.start,
                interval.end,
                context=context,
                start_spot=start_spot,
                end_spot=end_spot,
                start_rate=start_rate,
                end_rate=end_rate,
                start_dividend=start_dividend,
                end_dividend=end_dividend,
                expiration_at=expiration_at,
                quantity=interval.quantity,
                premium_multiplier=interval.multiplier,
                surface_components=contributions,
            )
        else:
            contributions = _price_history_fallback(
                occ,
                interval.start,
                interval.end,
                start_spot=start_spot,
                end_spot=end_spot,
                start_rate=start_rate,
                end_rate=end_rate,
                start_dividend=start_dividend,
                end_dividend=end_dividend,
                pricer=pricer,
                expiration_at=expiration_at,
                context=context,
                quantity=interval.quantity,
                premium_multiplier=interval.multiplier,
                warnings=warnings,
            )
        scale = interval.quantity * interval.multiplier
        for factor, value in contributions.items():
            values[_display_factor(factor)] += scale * value

    warnings.extend(inputs.warnings)
    # Report how much eligible history received full surface attribution.
    coverage = 100.0 * surfaced / surface_eligible if surface_eligible else 0.0
    method: AttributionMethod = (
        "surface_revaluation"
        if surface_eligible and surfaced == surface_eligible
        else "price_history_fallback"
    )
    return _finish(
        values,
        lifecycle_pnl=lifecycle_pnl,
        method=method,
        coverage=coverage,
        warnings=warnings,
    )


def _eligible_option_intervals(
    lifecycles: Sequence[Lifecycle],
    marks: MarketMarkBook,
    *,
    analysis_start: datetime,
    analysis_end: datetime,
    warnings: list[str] | None = None,
) -> list[_OptionInterval]:
    result: list[_OptionInterval] = []
    end_exclusive = analysis_end + timedelta(microseconds=1)
    for lifecycle in lifecycles:
        asset_by_symbol = {
            execution.symbol: execution.asset_type for execution in lifecycle.executions
        }
        multipliers = _multipliers(lifecycle)
        for symbol in sorted(asset_by_symbol):
            if asset_by_symbol[symbol] != "OPTION":
                continue
            expirations = {
                execution.expiration_at
                for execution in lifecycle.executions
                if execution.symbol == symbol and execution.expiration_at is not None
            }
            expiration_at = next(iter(expirations)) if len(expirations) == 1 else None
            if len(expirations) > 1 and warnings is not None:
                warnings.append(f"option_expiration_terms_inconsistent:{symbol}")
            all_observations = list(
                marks.exact_option_history(
                    symbol,
                    start_at=analysis_start,
                    end_at=end_exclusive,
                    include_execution_fills=True,
                )
            )
            by_session: dict[date, ExactOptionObservation] = {}
            execution_observations: list[ExactOptionObservation] = []
            for observation in all_observations:
                if observation.is_execution_fill:
                    execution_observations.append(observation)
                    continue
                session = _observation_session(observation)
                previous = by_session.get(session)
                if previous is None or observation.at > previous.at:
                    by_session[session] = observation
            observations = [by_session[session] for session in sorted(by_session)]
            observations.extend(execution_observations)
            if abs(
                inventory_at(lifecycle, end_exclusive).get(symbol, 0.0)
            ) > EPSILON:
                current = marks.option_observation(
                    lifecycle.account_id,
                    symbol,
                    analysis_end,
                )
                if current is not None:
                    observations.append(current)
            by_timestamp: dict[datetime, ExactOptionObservation] = {}
            for observation in sorted(
                observations,
                key=lambda item: (item.at, item.is_execution_fill),
            ):
                by_timestamp[observation.at] = observation
            ordered = [by_timestamp[at] for at in sorted(by_timestamp)]
            for start, end in zip(ordered, ordered[1:]):
                quantity = inventory_at(
                    lifecycle, start.at + timedelta(microseconds=1)
                ).get(symbol, 0.0)
                if abs(quantity) <= EPSILON:
                    continue
                if any(
                    start.at < execution.timestamp < end.at
                    and execution.symbol == symbol
                    for execution in lifecycle.executions
                ):
                    if warnings is not None:
                        warnings.append(
                            "fallback_exact_price_residual:"
                            f"{symbol}:{_observation_session(start).isoformat()}:"
                            "inventory_changed_inside_interval"
                        )
                    continue
                result.append(
                    _OptionInterval(
                        lifecycle=lifecycle,
                        symbol=symbol,
                        quantity=quantity,
                        multiplier=multipliers[symbol],
                        expiration_at=expiration_at,
                        start=start,
                        end=end,
                    )
                )
    return sorted(
        result,
        key=lambda item: (
            item.start.at,
            item.end.at,
            item.lifecycle.account_id,
            item.symbol,
        ),
    )


def _direct_instrument_pnl(
    lifecycles: Sequence[Lifecycle], marks: MarketMarkBook, analysis_end: datetime
) -> float:
    total = 0.0
    for lifecycle in lifecycles:
        direct_symbols = {
            execution.symbol
            for execution in lifecycle.executions
            if execution.asset_type in {"EQUITY", "ETF", "FUTURE"}
        }
        for execution in lifecycle.executions:
            if execution.symbol in direct_symbols:
                total += _required_cash(execution)
        inventory = inventory_at(lifecycle, analysis_end + timedelta(microseconds=1))
        multipliers = _multipliers(lifecycle)
        for symbol in direct_symbols:
            quantity = inventory.get(symbol, 0.0)
            if abs(quantity) <= EPSILON:
                continue
            mark = marks.current_mark(lifecycle.account_id, symbol)
            if mark is not None:
                total += quantity * multipliers[symbol] * mark
    return total


def _causal_snapshot(
    by_pair: Mapping[tuple[str, date], Sequence[OptionSurfaceSnapshot]],
    occ: OccSymbol,
    boundary: datetime,
) -> OptionSurfaceSnapshot | None:
    eligible = [
        item
        for item in by_pair.get((occ.root, occ.expiration), ())
        if _aware_datetime(item.header.captured_at) is not None
        and item.header.captured_at <= boundary
    ]
    return eligible[-1] if eligible else None


def _endpoint_spot(
    observation: ExactOptionObservation,
    snapshot: OptionSurfaceSnapshot | None,
) -> float | None:
    if snapshot is not None and _positive_finite(snapshot.header.spot):
        return float(snapshot.header.spot)
    return (
        float(observation.underlying_price)
        if _positive_finite(observation.underlying_price)
        else None
    )


def _surface_state(
    target: OccSymbol,
    boundary: datetime,
    by_pair: Mapping[tuple[str, date], Sequence[OptionSurfaceSnapshot]],
    expiration_terms: Mapping[tuple[str, date], datetime],
    inputs: _MarketInputCache,
    pricer: TianBatchRepricingAdapter,
    fit_cache: dict[tuple[int, datetime, float, float], _SmileCurve],
    *,
    target_rate: float,
    target_dividend: float,
) -> tuple[OptionMarketState, list[str]]:
    fits: dict[
        date, tuple[_SmileCurve, float, OptionSurfaceSnapshot, float, float]
    ] = {}
    warnings: list[str] = []
    expiries = sorted(
        expiration
        for root, expiration in expiration_terms
        if root == target.root
    )
    for expiration in expiries:
        expiration_at = expiration_terms.get((target.root, expiration))
        if expiration_at is None:
            continue
        occ_for_pair = replace(target, expiration=expiration) if expiration != target.expiration else target
        snapshot = _causal_snapshot(by_pair, occ_for_pair, boundary)
        if snapshot is None:
            continue
        if expiration == target.expiration:
            rate, dividend = target_rate, target_dividend
        else:
            rate_input = inputs.rate(boundary, expiration_at)
            dividend_input = (
                inputs.dividend(target.root, boundary, float(snapshot.header.spot))
                if _positive_finite(snapshot.header.spot)
                else _ResolvedMarketInput(None, False)
            )
            rate = rate_input.value
            dividend = dividend_input.value
        if rate is None or dividend is None:
            continue
        try:
            cache_key = (snapshot.header.snapshot_id, boundary, rate, dividend)
            fit = fit_cache.get(cache_key)
            if fit is None:
                fit = _fit_snapshot(
                    snapshot,
                    boundary,
                    expiration_at=expiration_at,
                    rate=rate,
                    dividend=dividend,
                    pricer=pricer,
                )
                fit_cache[cache_key] = fit
            target_iv = fit.atm_iv
            if expiration == target.expiration:
                target_x = math.log(float(target.strike) / fit.forward)
                if not fit.points[0][0] <= target_x <= fit.points[-1][0]:
                    raise _SurfaceUnusable("surface_target_outside_range")
                target_iv = _linear_interpolate(fit.points, target_x)
        except (_SurfaceUnusable, _AttributionSolverFailure):
            if expiration == target.expiration:
                raise
            continue
        fits[expiration] = (fit, target_iv, snapshot, rate, dividend)
        warnings.extend(fit.warnings)
    if target.expiration not in fits:
        raise _SurfaceUnusable("target_surface_unusable")
    # Split fitted IV into parallel volatility, maturity, and strike-specific components.
    target_fit, target_iv, snapshot, _, _ = fits[target.expiration]
    parallel = sum(item[0].atm_iv for item in fits.values()) / len(fits)
    term = target_fit.atm_iv - parallel if len(fits) > 1 else 0.0
    skew = target_iv - target_fit.atm_iv
    return (
        OptionMarketState(
            at=boundary,
            spot=float(snapshot.header.spot),
            rate=target_rate,
            dividend_yield=target_dividend,
            volatility=parallel,
            term_structure=term,
            skew=skew,
        ),
        warnings,
    )


def _fit_snapshot(
    snapshot: OptionSurfaceSnapshot,
    boundary: datetime,
    *,
    expiration_at: datetime,
    rate: float,
    dividend: float,
    pricer: TianBatchRepricingAdapter,
) -> _SmileCurve:
    header = snapshot.header
    if header.snapshot_status not in {
        int(ChainSnapshotStatus.SUCCESS),
        int(ChainSnapshotStatus.PARTIAL),
    }:
        raise _SurfaceUnusable("surface_snapshot_failed")
    if not _positive_finite(header.spot):
        raise _SurfaceUnusable("surface_spot_invalid")
    if (
        header.underlying_quote_time is not None
        and (
            _aware_datetime(header.underlying_quote_time) is None
            or header.underlying_quote_time > boundary
        )
    ):
        raise _SurfaceUnusable("surface_spot_lookahead")
    spot = float(header.spot)
    warnings: list[str] = []
    if header.snapshot_status == int(ChainSnapshotStatus.PARTIAL):
        warnings.append(f"surface_partial_used:{header.snapshot_id}")
    if header.snapshot_flags & int(ChainSnapshotFlag.DELAYED):
        warnings.append(f"surface_delayed_used:{header.snapshot_id}")
    if header.snapshot_flags & int(ChainSnapshotFlag.TRUNCATED):
        warnings.append(f"surface_truncated_used:{header.snapshot_id}")
    selected: list[tuple[OccSymbol, float, OptionSurfaceObservation]] = []
    for point in snapshot.observations:
        if point.quote_time is not None and (
            _aware_datetime(point.quote_time) is None or point.quote_time > boundary
        ):
            continue
        if point.security_status == int(SecurityStatus.HALTED):
            continue
        try:
            occ = parse_occ_symbol(point.canonical_occ_symbol)
        except ValueError:
            continue
        if occ.root != header.underlying_symbol or occ.expiration != header.expiration_date:
            continue
        price, basis = _surface_price(point)
        if price is None:
            continue
        intrinsic = (
            max(0.0, spot - float(occ.strike))
            if occ.put_call == "CALL"
            else max(0.0, float(occ.strike) - spot)
        )
        upper = spot if occ.put_call == "CALL" else float(occ.strike)
        if price + 1e-7 < intrinsic or price > upper + 1e-7:
            continue
        if basis != "midpoint":
            warnings.append(f"surface_{basis}_used:{point.canonical_occ_symbol}")
        if point.observation_source == int(
            SurfaceObservationSource.EXACT_OPTION_QUOTE_FALLBACK
        ):
            warnings.append(f"surface_exact_quote_point_used:{point.canonical_occ_symbol}")
        selected.append((occ, price, point))
    # Filter inconsistent chain prices before fitting a local volatility smile.
    _reject_static_arbitrage(selected)
    if len(selected) < 2:
        raise _SurfaceUnusable("surface_points_insufficient")
    tenor_seconds = (expiration_at - boundary).total_seconds()
    if tenor_seconds <= 0:
        raise _SurfaceUnusable("surface_expired")
    requests = [
        PricingRequest(
            option_symbol=occ.canonical,
            snapshot_id=f"surface:{header.snapshot_id}:{index}",
            put_call=occ.put_call,
            exercise_type="A",
            spot=spot,
            strike=float(occ.strike),
            tenor_seconds=tenor_seconds,
            rate=rate,
            dividend_yield=dividend,
            mark=price,
            source_collected_at=point.quote_time,
            expiration_at=expiration_at,
            requested_dtype=pricer.requested_dtype,
        )
        for index, (occ, price, point) in enumerate(selected)
    ]
    ivs = pricer.solve_ivs(
        requests,
        [price for _, price, _ in selected],
        [None] * len(selected),
    )
    tenor_years = tenor_seconds / (365.25 * 86400.0)
    forward = spot * math.exp((rate - dividend) * tenor_years)
    fitted: dict[float, list[float]] = {}
    for (occ, _, _), iv in zip(selected, ivs):
        if iv is None:
            continue
        x = math.log(float(occ.strike) / forward)
        fitted.setdefault(x, []).append(iv)
    curve = sorted((x, sum(values) / len(values)) for x, values in fitted.items())
    # Require observations on both sides of the forward before interpolating ATM IV.
    if len(curve) < 2 or not curve[0][0] <= 0.0 <= curve[-1][0]:
        raise _SurfaceUnusable("surface_atm_not_interpolable")
    atm = _linear_interpolate(curve, 0.0)
    return _SmileCurve(
        atm_iv=atm,
        points=tuple(curve),
        forward=forward,
        warnings=tuple(warnings),
    )


def _surface_price(
    point: OptionSurfaceObservation,
) -> tuple[float | None, str | None]:
    bid = _nonnegative_finite(point.bid)
    ask = _nonnegative_finite(point.ask)
    if bid is not None and ask is not None:
        if bid > ask:
            return None, None
        return (bid + ask) / 2.0, "midpoint"
    mark = _nonnegative_finite(point.mark)
    if mark is not None:
        return mark, "provider_mark"
    last = _nonnegative_finite(point.last)
    return (last, "last") if last is not None else (None, None)


def _reject_static_arbitrage(
    selected: Sequence[tuple[OccSymbol, float, OptionSurfaceObservation]],
) -> None:
    by_side: dict[str, list[tuple[float, float]]] = {"CALL": [], "PUT": []}
    for occ, price, _ in selected:
        by_side[occ.put_call].append((float(occ.strike), price))
    for side, rows in by_side.items():
        rows.sort()
        for (left_strike, left_price), (right_strike, right_price) in zip(
            rows, rows[1:]
        ):
            width = right_strike - left_strike
            spread = (
                left_price - right_price
                if side == "CALL"
                else right_price - left_price
            )
            if spread < -1e-7 or spread > width + 1e-7:
                raise _SurfaceUnusable("surface_static_arbitrage")


def _linear_interpolate(points: Sequence[tuple[float, float]], target: float) -> float:
    if target < points[0][0] or target > points[-1][0]:
        raise _SurfaceUnusable("surface_extrapolation_forbidden")
    for left, right in zip(points, points[1:]):
        if left[0] <= target <= right[0]:
            if right[0] == left[0]:
                return left[1]
            weight = (target - left[0]) / (right[0] - left[0])
            return left[1] + weight * (right[1] - left[1])
    return points[-1][1]


def _price_history_fallback(
    occ: OccSymbol,
    start: ExactOptionObservation,
    end: ExactOptionObservation,
    *,
    start_spot: float | None,
    end_spot: float | None,
    start_rate: float | None,
    end_rate: float | None,
    start_dividend: float | None,
    end_dividend: float | None,
    pricer: TianBatchRepricingAdapter | None,
    warnings: list[str],
    expiration_at: datetime | None = None,
    context: Mapping[str, Any] | None = None,
    quantity: float = 1.0,
    premium_multiplier: float = 1.0,
) -> dict[str, float]:
    session = _observation_session(start).isoformat()
    label = f"{occ.canonical}:{session}"
    tier_one_failure = "contract_iv_inputs_unavailable"
    invalid_exact_state = any(
        not _mark_within_no_arbitrage_bounds(occ, observation.mark, spot)
        for observation, spot in ((start, start_spot), (end, end_spot))
        if spot is not None
    )
    if pricer is None:
        tier_one_failure = "solver_unavailable"
    elif invalid_exact_state:
        tier_one_failure = "exact_mark_outside_no_arbitrage_bounds"
    elif expiration_at is None:
        tier_one_failure = "authoritative_expiration_unavailable"
    elif (
        start_spot is not None
        and end_spot is not None
        and start_rate is not None
        and end_rate is not None
        and start_dividend is not None
        and end_dividend is not None
    ):
        start_base = OptionMarketState(
            at=start.at,
            spot=start_spot,
            rate=start_rate,
            dividend_yield=start_dividend,
            volatility=0.20,
        )
        end_base = OptionMarketState(
            at=end.at,
            spot=end_spot,
            rate=end_rate,
            dividend_yield=end_dividend,
            volatility=0.20,
        )
        try:
            endpoint_rows = (
                (start, start_base),
                (end, end_base),
            )
            live_rows = [
                (index, observation, state)
                for index, (observation, state) in enumerate(endpoint_rows)
                if state.at < expiration_at
            ]
            live_ivs = (
                pricer.solve_ivs(
                    [
                        _pricing_request(
                            occ,
                            state,
                            pricer.requested_dtype,
                            expiration_at=expiration_at,
                        )
                        for _, _, state in live_rows
                    ],
                    [observation.mark for _, observation, _ in live_rows],
                    [observation.provider_iv for _, observation, _ in live_rows],
                )
                if live_rows
                else []
            )
            resolved_ivs: list[float | None] = [None, None]
            for (index, _, _), iv in zip(live_rows, live_ivs):
                resolved_ivs[index] = iv
            terminal_iv = next(
                (float(iv) for iv in resolved_ivs if _valid_iv(iv)),
                0.20,
            )
            for index, (_, state) in enumerate(endpoint_rows):
                if state.at >= expiration_at:
                    resolved_ivs[index] = terminal_iv
            if all(_valid_iv(iv) for iv in resolved_ivs):
                start_state = replace(start_base, volatility=float(resolved_ivs[0]))
                end_state = replace(end_base, volatility=float(resolved_ivs[1]))
                contributions = _symmetric_waterfall(
                    occ,
                    start_state,
                    end_state,
                    pricer,
                    expiration_at=expiration_at,
                    factors=_FALLBACK_WATERFALL_FACTORS,
                )
                warnings.append("fallback_contract_iv_revaluation_used")
                warnings.append(
                    f"historical_option_interval_method:{label}:contract_iv"
                )
                if end.at >= expiration_at:
                    warnings.append(f"fallback_terminal_revaluation_used:{label}")
                return _shared_interval_contributions(
                    occ,
                    start,
                    end,
                    context=context or {},
                    start_spot=start_spot,
                    end_spot=end_spot,
                    start_rate=start_rate,
                    end_rate=end_rate,
                    start_dividend=start_dividend,
                    end_dividend=end_dividend,
                    expiration_at=expiration_at,
                    quantity=quantity,
                    premium_multiplier=premium_multiplier,
                    local_ivs=(float(resolved_ivs[0]), float(resolved_ivs[1])),
                    contract_iv_components=contributions,
                )
            tier_one_failure = "iv_inversion_failed"
        except (_AttributionSolverFailure, SolverError, ValueError) as exc:
            tier_one_failure = _reason_code(exc, "contract_iv_revaluation_failed")
    warnings.append(
        f"fallback_contract_iv_revaluation_failed:{label}:{tier_one_failure}"
    )
    # Delegate fallback attribution to the shared exact-mark interval resolver.
    resolved = _shared_interval_result(
        occ,
        start,
        end,
        context=context or {},
        start_spot=start_spot,
        end_spot=end_spot,
        start_rate=start_rate,
        end_rate=end_rate,
        start_dividend=start_dividend,
        end_dividend=end_dividend,
        expiration_at=expiration_at,
        quantity=quantity,
        premium_multiplier=premium_multiplier,
    )
    if resolved is not None and resolved.method == "bounded_greek_bridge":
        warnings.append(f"fallback_greek_bridge_used:{label}")
        warnings.append(
            f"historical_option_interval_method:{label}:bounded_greek_bridge"
        )
        return _interval_display_components(resolved, quantity * premium_multiplier)
    residual_reason = (
        "underlying_history_unavailable"
        if start_spot is None or end_spot is None
        else (
            "exact_mark_outside_no_arbitrage_bounds"
            if invalid_exact_state
            else "captured_greeks_unusable"
        )
    )
    warnings.append(
        f"fallback_exact_price_driver_detail_unavailable:{label}:{residual_reason}"
    )
    if resolved is None:
        return {"observed_repricing": float(end.mark) - float(start.mark)}
    warnings.append(
        f"historical_option_interval_method:{label}:{resolved.method}"
    )
    return _interval_display_components(resolved, quantity * premium_multiplier)


def _shared_interval_contributions(
    occ: OccSymbol,
    start: ExactOptionObservation,
    end: ExactOptionObservation,
    *,
    context: Mapping[str, Any],
    start_spot: float | None,
    end_spot: float | None,
    start_rate: float | None,
    end_rate: float | None,
    start_dividend: float | None,
    end_dividend: float | None,
    expiration_at: datetime | None,
    quantity: float,
    premium_multiplier: float,
    local_ivs: tuple[float, float] | None = None,
    surface_components: Mapping[str, float] | None = None,
    contract_iv_components: Mapping[str, float] | None = None,
) -> dict[str, float]:
    interval = _shared_interval_result(
        occ,
        start,
        end,
        context=context,
        start_spot=start_spot,
        end_spot=end_spot,
        start_rate=start_rate,
        end_rate=end_rate,
        start_dividend=start_dividend,
        end_dividend=end_dividend,
        expiration_at=expiration_at,
        quantity=quantity,
        premium_multiplier=premium_multiplier,
        local_ivs=local_ivs,
        surface_components=surface_components,
        contract_iv_components=contract_iv_components,
    )
    if interval is None:
        return {"observed_repricing": float(end.mark) - float(start.mark)}
    return _interval_display_components(interval, quantity * premium_multiplier)


def _shared_interval_result(
    occ: OccSymbol,
    start: ExactOptionObservation,
    end: ExactOptionObservation,
    *,
    context: Mapping[str, Any],
    start_spot: float | None,
    end_spot: float | None,
    start_rate: float | None,
    end_rate: float | None,
    start_dividend: float | None,
    end_dividend: float | None,
    expiration_at: datetime | None,
    quantity: float,
    premium_multiplier: float,
    local_ivs: tuple[float, float] | None = None,
    surface_components: Mapping[str, float] | None = None,
    contract_iv_components: Mapping[str, float] | None = None,
):
    terms = build_instrument_terms(
        occ.canonical,
        broker_type="OPTION",
        context=context,
    ).model_copy(
        update={
            "price_multiplier": premium_multiplier,
            "expiration_at": expiration_at,
        }
    )

    def observation(
        source: ExactOptionObservation,
        *,
        spot: float | None,
        rate: float | None,
        dividend: float | None,
        local_iv: float | None,
        surface: bool,
    ) -> HistoricalOptionObservation:
        return HistoricalOptionObservation(
            canonical_occ_symbol=occ.canonical,
            instrument_terms=terms,
            session_date=_observation_session(source),
            at=source.at,
            observed_at=source.observed_at or source.at,
            bid=source.bid,
            ask=source.ask,
            mark=source.mark,
            mark_basis=source.mark_basis or (
                "execution_fill" if source.is_execution_fill else "exact_contract_mark"
            ),
            source="risk_pnl_exact_option_observation",
            underlying_spot=spot,
            underlying_source="risk_pnl_causal_endpoint" if spot is not None else None,
            risk_free_rate=rate,
            risk_free_rate_source="risk_pnl_causal_endpoint" if rate is not None else None,
            dividend_yield=dividend,
            dividend_source="risk_pnl_causal_endpoint" if dividend is not None else None,
            provider_iv=source.provider_iv,
            provider_delta=source.provider_delta,
            provider_gamma=source.provider_gamma,
            provider_theta_per_day=source.provider_theta_per_day,
            provider_vega_per_1pct=source.provider_vega_per_1pct,
            provider_rho_per_1pct=source.provider_rho_per_1pct,
            locally_inverted_iv=local_iv,
            surface_state={"source": "risk_pnl_surface"} if surface else None,
            quality_status="usable",
        )

    left = observation(
        start,
        spot=start_spot,
        rate=start_rate,
        dividend=start_dividend,
        local_iv=local_ivs[0] if local_ivs else None,
        surface=surface_components is not None,
    )
    right = observation(
        end,
        spot=end_spot,
        rate=end_rate,
        dividend=end_dividend,
        local_iv=local_ivs[1] if local_ivs else None,
        surface=surface_components is not None,
    )
    return resolve_historical_option_interval(
        left,
        right,
        terms,
        quantity=quantity,
        context=context,
        surface_components=surface_components,
        contract_iv_components=contract_iv_components,
    )


def _interval_display_components(interval: Any, scale: float) -> dict[str, float]:
    if not math.isfinite(scale) or abs(scale) <= EPSILON:
        return {}
    values = {
        "underlying": interval.spot_pnl_usd,
        "volatility": interval.volatility_pnl_usd,
        "skew": interval.skew_pnl_usd,
        "term_structure": interval.term_structure_pnl_usd,
        "time": interval.time_pnl_usd,
        "rates": interval.rate_pnl_usd,
        "dividends": interval.dividend_pnl_usd,
        "observed_repricing": interval.observed_repricing_pnl_usd,
        "model_residual": interval.residual_pnl_usd,
    }
    return {
        key: float(value) / scale
        for key, value in values.items()
        if value is not None and abs(float(value)) > EPSILON
    }


def _greek_approximation(
    start: ExactOptionObservation,
    end: ExactOptionObservation,
    *,
    start_spot: float | None,
    end_spot: float | None,
    start_rate: float | None,
    end_rate: float | None,
) -> dict[str, float]:
    result: dict[str, float] = {}
    if start_spot is None or end_spot is None:
        return result
    start_delta = _finite(start.provider_delta)
    end_delta = _finite(end.provider_delta)
    move = end_spot - start_spot
    if start_delta is not None and end_delta is not None:
        result["underlying"] = 0.5 * (start_delta + end_delta) * move
    elif start_delta is not None:
        result["underlying"] = start_delta * move
        start_gamma = _finite(start.provider_gamma)
        if start_gamma is not None:
            result["underlying"] += 0.5 * start_gamma * move * move
    theta = _endpoint_average_or_start(
        start.provider_theta_per_day, end.provider_theta_per_day
    )
    if theta is not None:
        result["time"] = (
            theta * (end.at - start.at).total_seconds() / 86400.0
        )
    if _valid_iv(start.provider_iv) and _valid_iv(end.provider_iv):
        vega = _endpoint_average_or_start(
            start.provider_vega_per_1pct,
            end.provider_vega_per_1pct,
        )
        if vega is not None:
            result["volatility"] = (
                vega
                * (float(end.provider_iv) - float(start.provider_iv))
                * 100.0
            )
    if start_rate is not None and end_rate is not None:
        rho = _endpoint_average_or_start(
            start.provider_rho_per_1pct,
            end.provider_rho_per_1pct,
        )
        if rho is not None:
            result["rates"] = (
                rho * (end_rate - start_rate) * 100.0
            )
    return result


def _symmetric_waterfall(
    occ: OccSymbol,
    start: OptionMarketState,
    end: OptionMarketState,
    pricer: TianBatchRepricingAdapter,
    *,
    expiration_at: datetime,
    factors: Sequence[str] = _WATERFALL_FACTORS,
) -> dict[str, float]:
    orders = (tuple(factors), tuple(reversed(factors)))
    paths: list[list[OptionMarketState]] = []
    for order in orders:
        state = start
        path = [state]
        for factor in order:
            state = _move_factor(state, end, factor)
            path.append(state)
        paths.append(path)
    flattened = [state for path in paths for state in path]
    prices = pricer.prices(occ, flattened, expiration_at=expiration_at)
    contributions = {factor: 0.0 for factor in factors}
    offset = 0
    for order, path in zip(orders, paths):
        path_prices = prices[offset : offset + len(path)]
        offset += len(path)
        for index, factor in enumerate(order):
            contributions[factor] += 0.5 * (
                path_prices[index + 1] - path_prices[index]
            )
    return contributions


def _move_factor(
    state: OptionMarketState, target: OptionMarketState, factor: str
) -> OptionMarketState:
    field = {
        "time": "at",
        "underlying": "spot",
        "rates": "rate",
        "dividends": "dividend_yield",
        "volatility": "volatility",
        "term_structure": "term_structure",
        "skew": "skew",
    }[factor]
    return replace(state, **{field: getattr(target, field)})


def _pricing_request(
    occ: OccSymbol,
    state: OptionMarketState,
    requested_dtype: str,
    *,
    expiration_at: datetime,
) -> PricingRequest:
    tenor_seconds = (expiration_at - state.at).total_seconds()
    if tenor_seconds <= 0:
        raise _AttributionSolverFailure("expired_hybrid_state")
    return PricingRequest(
        option_symbol=occ.canonical,
        snapshot_id=f"pnl-attribution:{occ.canonical}:{state.at.isoformat()}:{uuid.uuid4().hex}",
        put_call=occ.put_call,
        exercise_type="A",
        spot=state.spot,
        strike=float(occ.strike),
        tenor_seconds=tenor_seconds,
        rate=state.rate,
        dividend_yield=state.dividend_yield,
        source_iv=state.total_iv,
        source_collected_at=state.at,
        expiration_at=expiration_at,
        requested_dtype=requested_dtype,
    )


def _held_constant_inputs(
    start: _ResolvedMarketInput, end: _ResolvedMarketInput
) -> tuple[float | None, float | None, bool]:
    if start.observed and end.observed:
        return start.value, end.value, True
    if start.observed and start.value is not None:
        return start.value, start.value, False
    if end.observed and end.value is not None:
        return end.value, end.value, False
    configured = start.value if start.value is not None else end.value
    return configured, configured, False


def _endpoint_average_or_start(start: Any, end: Any) -> float | None:
    start_value = _finite(start)
    end_value = _finite(end)
    if start_value is None:
        return None
    if end_value is None:
        return start_value
    return 0.5 * (start_value + end_value)


def _observation_session(observation: ExactOptionObservation) -> date:
    if observation.session_date is not None:
        return observation.session_date
    return observation.at.date()


def _reason_code(exc: Exception, default: str) -> str:
    value = str(exc).strip().lower().replace(" ", "_")
    return value if value and value.replace("_", "").isalnum() else default


def _display_factor(factor: str) -> str:
    return {
        "underlying": "Underlying movement",
        "volatility": "Volatility",
        "skew": "Skew",
        "term_structure": "Term structure",
        "time": "Time decay",
        "rates": "Rates",
        "dividends": "Dividends",
        "observed_repricing": "Option repricing residual",
        "model_residual": "Model residual",
    }[factor]


def _finish(
    values: dict[str, float],
    *,
    lifecycle_pnl: float,
    method: AttributionMethod,
    coverage: float | None,
    warnings: Sequence[str],
) -> AttributionResult:
    values["Model residual"] = lifecycle_pnl - sum(
        value for factor, value in values.items() if factor != "Model residual"
    )
    rounded = [
        PnlFactorAttribution(factor=factor, pnl_usd=_money(values[factor]))
        for factor in FACTOR_ORDER
    ]
    difference = _money(lifecycle_pnl - sum(item.pnl_usd for item in rounded))
    if difference:
        rounded[-1] = rounded[-1].model_copy(
            update={"pnl_usd": _money(rounded[-1].pnl_usd + difference)}
        )
    bounded_coverage = (
        None if coverage is None else min(100.0, max(0.0, float(coverage)))
    )
    return AttributionResult(
        factors=rounded,
        attribution_method=method,
        surface_coverage_percent=bounded_coverage,
        warnings=list(dict.fromkeys(warnings)),
        calculation_method=_calculation_method(method, warnings),
        input_mode=(
            "surface"
            if method == "surface_revaluation"
            else "exact_contract_price_history"
            if method == "price_history_fallback"
            else "exact_marks"
        ),
    )


def _calculation_method(method: AttributionMethod, warnings: Sequence[str]) -> str:
    if method == "surface_revaluation":
        return "surface_revaluation_anchored_to_exact_marks"
    if any("fallback_contract_iv_revaluation_used" in item for item in warnings):
        return "exact_contract_price_history_contract_iv"
    if any("fallback_greek_bridge_used" in item for item in warnings):
        return "exact_contract_price_history_greek_bridge"
    if method == "price_history_fallback":
        return "exact_contract_price_history_marks_only"
    return "mark_revaluation"


def _intrinsic_value(occ: OccSymbol, spot: float) -> float:
    strike = float(occ.strike)
    return (
        max(0.0, float(spot) - strike)
        if occ.put_call == "CALL"
        else max(0.0, strike - float(spot))
    )


def _mark_within_no_arbitrage_bounds(
    occ: OccSymbol, mark: float, spot: float
) -> bool:
    lower = _intrinsic_value(occ, spot)
    upper = float(spot) if occ.put_call == "CALL" else float(occ.strike)
    tolerance = max(0.01, upper * 1e-7)
    return lower - tolerance <= float(mark) <= upper + tolerance


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _positive_finite(value: Any) -> bool:
    converted = _finite(value)
    return converted is not None and converted > 0.0


def _positive_float(value: Any) -> float | None:
    converted = _finite(value)
    return converted if converted is not None and converted > 0.0 else None


def _valid_iv(value: Any) -> bool:
    converted = _finite(value)
    return converted is not None and 0.0 < converted <= 10.0


def _nonnegative_finite(value: Any) -> float | None:
    converted = _finite(value)
    return converted if converted is not None and converted >= 0.0 else None


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _aware_datetime(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _aware_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value not in (None, ""):
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _bar_session_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _date_value(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


__all__ = [
    "AttributionResult",
    "FACTOR_ORDER",
    "OptionMarketState",
    "TianBatchRepricingAdapter",
    "attribute_selection",
]
