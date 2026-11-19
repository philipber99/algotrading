"""Historical lifecycle replay and bounded counterfactual analysis."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from algotrading.refdata.api.option_symbols import parse_occ_symbol
from algotrading.risk.market_inputs.client import ReferenceDataClient, ReferenceDataClientError
from algotrading.risk.transactions.repository import TransactionRepository
from algotrading.risk.option_prices.repository import OptionPriceRepository

from .attribution import attribute_selection
from .models import (
    PnlActiveLeg,
    PnlAnalyzerData,
    PnlAnalyzerDataQuality,
    PnlAnalyzerRequest,
    PnlAnalyzerResponse,
    PnlChronologyEvent,
    PnlReplayPoint,
    PnlScenarioOutcome,
)
from .reconstruction import EPSILON, Lifecycle, inventory_at, recognize_structure
from .valuation import (
    METHODOLOGY_VERSION,
    MarketMarkBook,
    PnlMarkHistoryUnavailable,
    PnlReconstructionLimitExceeded,
    _daily_boundaries,
    _multipliers,
    _option_expired,
    _required_cash,
    _setting_symbols,
    load_lifecycles,
)


class PnlSelectionNotFound(RuntimeError):
    pass


class InvalidPnlAnalysis(RuntimeError):
    pass


def analyze_selection(
    payload: PnlAnalyzerRequest,
    *,
    repository: TransactionRepository,
    snapshot: Any,
    reference_client: ReferenceDataClient,
    context: Mapping[str, Any],
    option_price_repository: OptionPriceRepository | None = None,
) -> PnlAnalyzerResponse:
    maximum_accounts = _setting_int(context, "risk.pnl.max_accounts", 20, 1)
    if len(payload.account_ids) > maximum_accounts:
        raise PnlReconstructionLimitExceeded(
            f"account count exceeds configured maximum of {maximum_accounts}"
        )
    load_end = snapshot.as_of.astimezone(timezone.utc) + timedelta(microseconds=1)
    loaded = load_lifecycles(
        repository,
        account_ids=payload.account_ids,
        requested_start_at=snapshot.as_of - timedelta(microseconds=1),
        requested_end_at=snapshot.as_of,
        load_end_at=load_end,
        context=context,
        reference_client=reference_client,
        reuse_cached=True,
    )
    # Resolve selected IDs against reconstructed lifecycles before performing valuation.
    by_trade_id = {item.trade_id: item for item in loaded.lifecycles}
    selected: dict[str, Lifecycle] = {}
    missing_trade_ids = []
    for trade_id in payload.trade_ids:
        lifecycle = by_trade_id.get(trade_id)
        if lifecycle is None:
            missing_trade_ids.append(trade_id)
        else:
            selected[trade_id] = lifecycle
    transaction_ids = set(payload.transaction_ids)
    matched_transaction_ids: set[str] = set()
    for lifecycle in loaded.lifecycles:
        evidence = {
            str(item.repository_transaction_id) for item in lifecycle.executions
        }
        overlap = evidence & transaction_ids
        if overlap:
            selected[lifecycle.trade_id] = lifecycle
            matched_transaction_ids.update(overlap)
    if missing_trade_ids or matched_transaction_ids != transaction_ids or not selected:
        raise PnlSelectionNotFound(
            "One or more selected trades or transactions were not found in the requested account and snapshot scope."
        )
    lifecycles = sorted(
        selected.values(), key=lambda item: (item.opened_at, item.trade_id)
    )
    analysis_end = max(
        (
            lifecycle.closed_at
            if lifecycle.closed_at is not None
            else snapshot.as_of.astimezone(timezone.utc)
        )
        for lifecycle in lifecycles
    )
    # Keep every replay and counterfactual inside the current snapshot boundary.
    analysis_end = min(analysis_end, snapshot.as_of.astimezone(timezone.utc))
    analysis_start = min(item.opened_at for item in lifecycles)
    _validate_analysis_request(
        payload,
        lifecycles=lifecycles,
        snapshot=snapshot,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
    )

    warnings = list(loaded.warnings)
    has_options = any(
        execution.asset_type == "OPTION"
        for lifecycle in lifecycles
        for execution in lifecycle.executions
    )
    historical_options_available = not has_options
    actual_path: list[PnlReplayPoint] = []
    marks = MarketMarkBook(snapshot, reference_client)
    fills_loaded = False
    try:
        marks.load_history(
            lifecycles,
            start_at=analysis_start,
            end_at=analysis_end + timedelta(microseconds=1),
            maximum_option_symbols=_setting_int(
                context, "risk.pnl.max_option_symbols", 500, 1
            ),
            history_lookback_days=0,
            remote_equity_symbols=_setting_symbols(
                context,
                "risk.pnl.remote_daily_bar_symbols",
                ("SCC", "TSCO", "VOLT"),
            ),
        )
        if has_options:
            option_symbols = {
                execution.symbol
                for lifecycle in lifecycles
                for execution in lifecycle.executions
                if execution.asset_type == "OPTION"
            }
            historical_options_available = all(
                marks.has_historical_option_mark(symbol)
                for symbol in option_symbols
            )
        # A daily historical mark may be timestamped at session close, after an
        # intraday opening execution.  The broker fill is the exact observable
        # value at that boundary and prevents a false "history unavailable"
        # result for otherwise complete option histories.
        marks.load_execution_fills(lifecycles)
        fills_loaded = True
        if not has_options or historical_options_available:
            actual_path = _actual_path(
                lifecycles,
                marks=marks,
                start_at=analysis_start,
                end_at=analysis_end + timedelta(microseconds=1),
                maximum_points=_setting_int(
                    context, "risk.pnl.max_analyzer_path_points", 1_000, 1
                ),
            )
    except PnlMarkHistoryUnavailable as exc:
        if has_options:
            historical_options_available = False
            missing = ", ".join(exc.symbols)
            detail = str(exc).rstrip(".")
            warnings.append(
                f"{detail}. Missing price history: {missing}. Transaction chronology and fill-derived lifecycle P&L remain available."
                if missing
                else f"{detail}. The option-history service returned no contract-level reason. Transaction chronology and fill-derived lifecycle P&L remain available."
            )
            actual_path = []
        else:
            raise
    finally:
        if not fills_loaded:
            marks.load_execution_fills(lifecycles)

    marks.load_current_fallback(lifecycles)
    warnings.extend(marks.warnings)

    lifecycle_pnl = sum(
        _lifecycle_estimate(item, marks, snapshot.as_of) for item in lifecycles
    )
    fees = sum(
        execution.fees_usd
        for lifecycle in lifecycles
        for execution in lifecycle.executions
    )
    chronology = _chronology(lifecycles)
    peak, drawdown = _path_statistics(actual_path)
    if actual_path:
        chronology.extend(_extrema_events(actual_path))
        chronology.sort(key=lambda item: (item.at, item.kind, item.title))
    # Explain exact lifecycle P&L using the available historical market evidence.
    attribution_result = attribute_selection(
        lifecycles=lifecycles,
        marks=marks,
        option_price_repository=option_price_repository,
        reference_client=reference_client,
        context=context,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        lifecycle_pnl=lifecycle_pnl,
        fees=fees,
    )
    attribution = attribution_result.factors
    warnings.extend(attribution_result.warnings)
    if (
        has_options
        and not historical_options_available
        and not _requires_historical_option_marks(
            lifecycles, analysis_start=analysis_start, analysis_end=analysis_end
        )
    ):
        warnings.append("option_no_eligible_daily_intervals")
    scenarios: list[PnlScenarioOutcome] = []
    if actual_path and (payload.modifications or payload.exit_rule.kind != "actual"):
        try:
            scenario, scenario_warnings = _scenario(
                payload,
                lifecycles=lifecycles,
                actual_path=actual_path,
                reference_client=reference_client,
                context=context,
            )
        except PnlMarkHistoryUnavailable:
            if any(
                item.instrument_kind == "option" for item in payload.modifications
            ):
                historical_options_available = False
            warnings.append(
                "The requested counterfactual cannot be replayed because required historical executable marks are unavailable."
            )
        else:
            scenarios.append(scenario)
            warnings.extend(scenario_warnings)
    if payload.fill_model == "historical_mid" and payload.modifications:
        warnings.append("historical_mid is a non-executable research fill model")
    warnings = list(dict.fromkeys(warnings))
    status = "usable"
    if warnings:
        status = "usable_with_warnings"
    if (
        has_options
        and not historical_options_available
        and _requires_historical_option_marks(
            lifecycles, analysis_start=analysis_start, analysis_end=analysis_end
        )
    ):
        status = "insufficient_data"
    return PnlAnalyzerResponse(
        portfolio_snapshot_id=snapshot.portfolio_snapshot_id,
        as_of=snapshot.as_of,
        methodology_version=METHODOLOGY_VERSION,
        data_quality=PnlAnalyzerDataQuality(
            status=status,
            historical_option_marks_available=historical_options_available,
            attribution_method=attribution_result.attribution_method,
            surface_coverage_percent=attribution_result.surface_coverage_percent,
            option_price_history_coverage_percent=(
                100.0 if historical_options_available else 0.0 if has_options else None
            ),
            calculation_method=attribution_result.calculation_method,
            input_mode=attribution_result.input_mode,
            warnings=warnings,
        ),
        data=PnlAnalyzerData(
            selection_name=(
                recognize_structure(lifecycles[0])
                if len(lifecycles) == 1
                else f"{len(lifecycles)} selected trade lifecycles"
            ),
            match_confidence_percent=min(
                item.confidence_percent for item in lifecycles
            ),
            lifecycle_pnl_usd=_money(lifecycle_pnl),
            peak_open_pnl_usd=_money(peak),
            maximum_drawdown_usd=_money(drawdown),
            fees_and_slippage_usd=_money(fees),
            actual_path=actual_path,
            chronology=chronology,
            attribution=attribution,
            active_legs=_active_legs(lifecycles, snapshot.as_of),
            scenarios=scenarios,
        ),
    )


def _validate_analysis_request(
    payload: PnlAnalyzerRequest,
    *,
    lifecycles: Sequence[Lifecycle],
    snapshot: Any,
    analysis_start: datetime,
    analysis_end: datetime,
) -> None:
    if payload.decision_at is not None and not (
        analysis_start <= payload.decision_at <= analysis_end
    ):
        raise InvalidPnlAnalysis(
            "decision_at must fall within the selected lifecycle history"
        )
    allowed_underlyings = {item.underlying_symbol for item in lifecycles}
    allowed_symbols = {
        execution.symbol
        for lifecycle in lifecycles
        for execution in lifecycle.executions
    }
    allowed_symbols.update(
        position.canonical_occ_symbol or position.exact_symbol
        for position in snapshot.data.positions
    )
    modification_symbols: set[tuple[str, datetime]] = set()
    for modification in payload.modifications:
        modification_at = modification.at or payload.decision_at
        assert modification_at is not None
        if not (analysis_start <= modification_at <= analysis_end):
            raise InvalidPnlAnalysis(
                "transaction date must fall within the selected lifecycle history"
            )
        modification_key = (modification.symbol, modification_at)
        if modification_key in modification_symbols:
            raise InvalidPnlAnalysis(
                "modifications must not contain duplicate transactions"
            )
        modification_symbols.add(modification_key)
        if modification.instrument_kind == "option":
            option = parse_occ_symbol(modification.symbol)
            root = option.root
            # Additions are explicit counterfactual positions and may introduce a
            # new underlying.  Scope restrictions only apply when removing an
            # existing selected position.
            if modification.action == "remove" and root not in allowed_underlyings:
                raise InvalidPnlAnalysis(
                    "option modification must use an exact contract for a selected underlying"
                )
            if option.expiration < modification_at.date():
                raise InvalidPnlAnalysis(
                    "option modification cannot use a contract expired before decision_at"
                )
        elif (
            modification.action == "remove"
            and modification.symbol not in allowed_underlyings
            and modification.symbol not in allowed_symbols
        ):
            raise InvalidPnlAnalysis(
                "modification symbol is outside the selected snapshot scope"
            )
        if modification.action == "remove":
            at = modification_at
            active = sum(
                inventory_at(item, at + timedelta(microseconds=1)).get(
                    modification.symbol, 0.0
                )
                for item in lifecycles
            )
            if abs(active) <= EPSILON or abs(modification.quantity) > abs(active) + EPSILON:
                raise InvalidPnlAnalysis(
                    "remove modification exceeds the active selected leg quantity"
                )
    if payload.exit_rule.kind == "timestamp":
        assert isinstance(payload.exit_rule.value, str)
        exit_at = datetime.fromisoformat(
            payload.exit_rule.value.replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        lower = payload.decision_at or analysis_start
        if exit_at < lower or exit_at > analysis_end:
            raise InvalidPnlAnalysis(
                "timestamp exit must be on or after decision_at and within the selected history"
            )
    if payload.exit_rule.kind == "days_before_short_expiry":
        short_expiries = []
        for lifecycle in lifecycles:
            opening = inventory_at(
                lifecycle, lifecycle.opened_at + timedelta(microseconds=1)
            )
            for symbol, quantity in opening.items():
                if quantity < 0:
                    try:
                        short_expiries.append(parse_occ_symbol(symbol).expiration)
                    except ValueError:
                        pass
        if not short_expiries:
            raise InvalidPnlAnalysis(
                "days_before_short_expiry requires an active short option leg"
            )
        cutoff = datetime.combine(
            min(short_expiries) - timedelta(days=int(float(payload.exit_rule.value))),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        if cutoff < (payload.decision_at or analysis_start) or cutoff > analysis_end:
            raise InvalidPnlAnalysis(
                "days_before_short_expiry resolves outside the selected lifecycle history"
            )


def _actual_path(
    lifecycles: Sequence[Lifecycle],
    *,
    marks: MarketMarkBook,
    start_at: datetime,
    end_at: datetime,
    maximum_points: int,
) -> list[PnlReplayPoint]:
    # Combine daily boundaries with execution times that have coherent mark coverage.
    times = {boundary for _, boundary in _daily_boundaries(start_at, end_at)}
    times.update(_coherent_execution_boundaries(lifecycles, start_at, end_at))
    boundaries = sorted(times)
    if len(boundaries) > maximum_points:
        raise PnlReconstructionLimitExceeded(
            "analyzer path exceeds the configured point limit"
        )
    points: list[PnlReplayPoint] = []
    prior_boundary = start_at
    for boundary in boundaries:
        pnl = 0.0
        underlying_price = None
        delta = 0.0
        event_names: list[str] = []
        for lifecycle in lifecycles:
            inventory = inventory_at(lifecycle, boundary)
            value = 0.0
            for symbol, quantity in inventory.items():
                multiplier = _multipliers(lifecycle)[symbol]
                value += quantity * multiplier * marks.mark(
                    lifecycle.account_id, symbol, boundary
                )
                if symbol == lifecycle.underlying_symbol:
                    underlying_price = marks.mark(
                        lifecycle.account_id, symbol, boundary
                    )
                    delta += quantity
            if underlying_price is None and any(
                item.symbol == lifecycle.underlying_symbol
                and item.asset_type in {"EQUITY", "ETF"}
                for item in lifecycle.executions
            ):
                underlying_price = marks.mark(
                    lifecycle.account_id,
                    lifecycle.underlying_symbol,
                    boundary,
                )
            # Add execution cash and fees to marked inventory to reconstruct cumulative P&L.
            flows = sum(
                _required_cash(item) + item.fees_usd
                for item in lifecycle.executions
                if item.timestamp < boundary
            )
            pnl += value + flows
            event_names.extend(
                item.action
                for item in lifecycle.executions
                if prior_boundary <= item.timestamp < boundary
            )
        points.append(
            PnlReplayPoint(
                at=boundary,
                pnl_usd=_money(pnl),
                underlying_price=_money(underlying_price),
                delta=_money(delta) if delta else None,
                gamma=None,
                vega_per_1pct_usd=None,
                theta_per_day_usd=None,
                event="; ".join(dict.fromkeys(event_names)) or None,
            )
        )
        prior_boundary = boundary
    return points


def _coherent_execution_boundaries(
    lifecycles: Sequence[Lifecycle],
    start_at: datetime,
    end_at: datetime,
) -> set[datetime]:
    """Return execution instants that can be valued without mixing mark times.

    A fill is an exact observation for its own contract only.  After one leg of
    a multi-leg position closes, carrying the prior daily marks of the remaining
    legs into that intraday instant can create a large, short-lived P&L jump.
    Keep the point when the selection is flat or every still-active contract has
    a fill at that instant; otherwise the next daily boundary records the trade
    with a coherent set of marks.
    """
    execution_times = {
        item.timestamp
        for lifecycle in lifecycles
        for item in lifecycle.executions
        if start_at <= item.timestamp < end_at
    }
    boundaries: set[datetime] = set()
    for execution_at in execution_times:
        boundary = execution_at + timedelta(microseconds=1)
        active_symbols = {
            (lifecycle.account_id, symbol)
            for lifecycle in lifecycles
            for symbol, quantity in inventory_at(lifecycle, boundary).items()
            if abs(quantity) > EPSILON
        }
        filled_symbols = {
            (lifecycle.account_id, item.symbol)
            for lifecycle in lifecycles
            for item in lifecycle.executions
            if item.timestamp == execution_at and item.fill_price is not None
        }
        # Add an execution boundary only when every active leg has a contemporaneous fill.
        if active_symbols <= filled_symbols:
            boundaries.add(boundary)
    return boundaries


def _lifecycle_estimate(
    lifecycle: Lifecycle, marks: MarketMarkBook, as_of: datetime
) -> float:
    cash = sum(
        _required_cash(item) + item.fees_usd for item in lifecycle.executions
    )
    current = inventory_at(lifecycle, as_of + timedelta(microseconds=1))
    asset_by_symbol = {
        execution.symbol: execution.asset_type for execution in lifecycle.executions
    }
    for symbol, quantity in current.items():
        if (
            asset_by_symbol.get(symbol) == "OPTION"
            and _option_expired(symbol, as_of)
        ):
            try:
                mark = marks.option_settlement_mark(symbol, as_of)
            except PnlMarkHistoryUnavailable:
                mark = None
        else:
            mark = marks.current_mark(lifecycle.account_id, symbol)
        if mark is None:
            raise PnlMarkHistoryUnavailable(
                f"Current exact mark is unavailable for {symbol}."
            )
        cash += quantity * _multipliers(lifecycle)[symbol] * mark
    return cash


def _chronology(lifecycles: Sequence[Lifecycle]) -> list[PnlChronologyEvent]:
    rows = []
    for lifecycle in lifecycles:
        for item in lifecycle.executions:
            fill = (
                f" at {item.fill_price:g}"
                if item.fill_price is not None
                else " at an unavailable fill"
            )
            rows.append(
                PnlChronologyEvent(
                    at=item.timestamp,
                    title=f"{item.action}: {item.symbol}",
                    detail=(
                        f"{abs(item.signed_quantity):g} unit(s){fill}; "
                        f"repository transaction {item.repository_transaction_id}"
                    ),
                    kind="transaction",
                )
            )
    return sorted(rows, key=lambda item: (item.at, item.title))


def _path_statistics(
    path: Sequence[PnlReplayPoint],
) -> tuple[float | None, float | None]:
    if not path:
        return None, None
    peak = path[0].pnl_usd
    maximum_drawdown = 0.0
    running_peak = peak
    for point in path:
        running_peak = max(running_peak, point.pnl_usd)
        maximum_drawdown = min(maximum_drawdown, point.pnl_usd - running_peak)
        peak = max(peak, point.pnl_usd)
    return peak, maximum_drawdown


def _extrema_events(path: Sequence[PnlReplayPoint]) -> list[PnlChronologyEvent]:
    if not path:
        return []
    peak = max(path, key=lambda item: item.pnl_usd)
    running_peak = path[0].pnl_usd
    drawdown_point = path[0]
    drawdown = 0.0
    for point in path:
        running_peak = max(running_peak, point.pnl_usd)
        current = point.pnl_usd - running_peak
        if current < drawdown:
            drawdown = current
            drawdown_point = point
    events = [
        PnlChronologyEvent(
            at=peak.at,
            title="Peak open P&L",
            detail=f"Marked open P&L reached {_money(peak.pnl_usd):g} USD.",
            kind="risk",
        )
    ]
    if drawdown < -0.005:
        events.append(
            PnlChronologyEvent(
                at=drawdown_point.at,
                title="Maximum drawdown",
                detail=f"Drawdown from the prior marked peak reached {_money(drawdown):g} USD.",
                kind="risk",
            )
        )
    return events


def _active_legs(
    lifecycles: Sequence[Lifecycle], as_of: datetime
) -> list[PnlActiveLeg]:
    rows: list[PnlActiveLeg] = []
    for lifecycle in lifecycles:
        for symbol, quantity in sorted(
            inventory_at(lifecycle, as_of + timedelta(microseconds=1)).items()
        ):
            if abs(quantity) <= EPSILON or _option_expired(symbol, as_of):
                continue
            rows.append(
                PnlActiveLeg(
                    leg_id=hashlib.sha256(
                        f"{lifecycle.trade_id}\x1f{symbol}".encode("utf-8")
                    ).hexdigest(),
                    label=("Long " if quantity > 0 else "Short ") + symbol,
                    symbol=symbol,
                    quantity=quantity,
                )
            )
    return rows


def _requires_historical_option_marks(
    lifecycles: Sequence[Lifecycle],
    *,
    analysis_start: datetime,
    analysis_end: datetime,
) -> bool:
    option_symbols = {
        execution.symbol
        for lifecycle in lifecycles
        for execution in lifecycle.executions
        if execution.asset_type == "OPTION"
    }
    for _, boundary in _daily_boundaries(
        analysis_start, analysis_end + timedelta(microseconds=1)
    ):
        if any(
            abs(inventory_at(lifecycle, boundary).get(symbol, 0.0)) > EPSILON
            for lifecycle in lifecycles
            for symbol in option_symbols
        ):
            return True
    return False


def _scenario(
    payload: PnlAnalyzerRequest,
    *,
    lifecycles: Sequence[Lifecycle],
    actual_path: Sequence[PnlReplayPoint],
    reference_client: ReferenceDataClient,
    context: Mapping[str, Any],
) -> tuple[PnlScenarioOutcome, list[str]]:
    warnings: list[str] = []
    path = list(actual_path)
    decision_at = payload.decision_at or path[0].at
    actual_units = sum(
        abs(item.signed_quantity)
        for lifecycle in lifecycles
        for item in lifecycle.executions
    )
    actual_fee_rate = (
        sum(
            abs(item.fees_usd)
            for lifecycle in lifecycles
            for item in lifecycle.executions
        )
        / actual_units
        if actual_units > EPSILON
        else 0.0
    )
    unit_cost = float(
        context.get("risk.pnl.transaction_cost_per_unit", actual_fee_rate)
    )
    if not math.isfinite(unit_cost) or unit_cost < 0:
        raise InvalidPnlAnalysis(
            "risk.pnl.transaction_cost_per_unit must be finite and non-negative"
        )
    if payload.exit_rule.kind == "timestamp":
        assert isinstance(payload.exit_rule.value, str)
        exit_at = datetime.fromisoformat(
            payload.exit_rule.value.replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        path = [item for item in path if item.at <= exit_at]
    elif payload.exit_rule.kind in {"profit_target", "drawdown_from_peak"}:
        threshold = float(payload.exit_rule.value)
        running_peak = -math.inf
        truncated: list[PnlReplayPoint] = []
        for point in path:
            truncated.append(point)
            running_peak = max(running_peak, point.pnl_usd)
            if payload.exit_rule.kind == "profit_target" and point.pnl_usd >= threshold:
                break
            if payload.exit_rule.kind == "drawdown_from_peak" and running_peak - point.pnl_usd >= threshold:
                break
        path = truncated
    elif payload.exit_rule.kind == "days_before_short_expiry":
        days = int(float(payload.exit_rule.value))
        expiries = []
        for lifecycle in lifecycles:
            for execution in lifecycle.executions:
                if execution.asset_type == "OPTION" and execution.signed_quantity < 0:
                    expiries.append(parse_occ_symbol(execution.symbol).expiration)
        cutoff = datetime.combine(
            min(expiries) - timedelta(days=days),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        path = [item for item in path if item.at <= cutoff]

    # Resolve counterfactual fills once before applying changes to the replay path.
    prepared_modifications: list[
        tuple[Any, datetime, float, float, float, list[tuple[datetime, float]]]
    ] = []
    for modification in payload.modifications:
        modification_at = modification.at or decision_at
        signed_quantity = modification.quantity
        if modification.action == "remove":
            active = sum(
                inventory_at(
                    lifecycle, modification_at + timedelta(microseconds=1)
                ).get(modification.symbol, 0.0)
                for lifecycle in lifecycles
            )
            signed_quantity = -math.copysign(abs(modification.quantity), active)
        fill, fill_warnings, observations = _historical_fill(
            modification,
            signed_quantity=signed_quantity,
            decision_at=modification_at,
            history_end_at=path[-1].at if path else modification_at,
            fill_model=payload.fill_model,
            reference_client=reference_client,
            actual_path=path,
            context=context,
        )
        if modification.fill_price is not None:
            fill = modification.fill_price
        warnings.extend(fill_warnings)
        prepared_modifications.append(
            (
                modification,
                modification_at,
                signed_quantity,
                fill,
                abs(signed_quantity) * unit_cost,
                observations,
            )
        )

    adjusted: list[PnlReplayPoint] = []
    for point in path:
        pnl = point.pnl_usd
        if point.at >= decision_at:
            for modification, modification_at, signed_quantity, fill, transaction_cost, observations in prepared_modifications:
                if point.at < modification_at:
                    continue
                observable_prices = [
                    price for at, price in observations if at <= point.at
                ]
                point_price = observable_prices[-1] if observable_prices else None
                if point_price is None and modification.symbol in {
                    lifecycle.underlying_symbol for lifecycle in lifecycles
                }:
                    point_price = point.underlying_price
                if point_price is None:
                    raise PnlMarkHistoryUnavailable(
                        f"A sequential scenario mark is unavailable for {modification.symbol}."
                    )
                multiplier = 100.0 if modification.instrument_kind == "option" else 1.0
                pnl += signed_quantity * multiplier * (point_price - fill)
                pnl -= transaction_cost
        adjusted.append(point.model_copy(update={"pnl_usd": _money(pnl)}))
    # Charge closing costs when the selected exit ends the counterfactual early.
    if adjusted and payload.exit_rule.kind != "actual":
        exit_at = adjusted[-1].at
        closing_units = sum(
            abs(quantity)
            for lifecycle in lifecycles
            for quantity in inventory_at(
                lifecycle, exit_at + timedelta(microseconds=1)
            ).values()
        )
        if closing_units > EPSILON and unit_cost > 0:
            adjusted[-1] = adjusted[-1].model_copy(
                update={
                    "pnl_usd": _money(
                        adjusted[-1].pnl_usd - closing_units * unit_cost
                    )
                }
            )
    if payload.modifications and not adjusted:
        warnings.append("scenario_modification_could_not_be_marked")
    _, drawdown = _path_statistics(adjusted)
    net = adjusted[-1].pnl_usd if adjusted else 0.0
    fingerprint = payload.model_dump(mode="json")
    scenario_id = "scenario:" + hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return (
        PnlScenarioOutcome(
            scenario_id=scenario_id,
            name="Selected counterfactual",
            net_pnl_usd=_money(net),
            max_drawdown_usd=_money(drawdown or 0.0),
            capital_at_risk_usd=None,
            interpretation=(
                "Sequential replay using only marks available at or before each path timestamp."
            ),
            path=adjusted,
        ),
        warnings,
    )


def _historical_fill(
    modification: Any,
    *,
    signed_quantity: float,
    decision_at: datetime,
    history_end_at: datetime,
    fill_model: str,
    reference_client: ReferenceDataClient,
    actual_path: Sequence[PnlReplayPoint],
    context: Mapping[str, Any],
) -> tuple[float, list[str], list[tuple[datetime, float]]]:
    warnings: list[str] = []
    observations: list[tuple[datetime, float]] = []
    bid = ask = mid = None
    history_lookback = _setting_int(
        context, "risk.pnl.scenario_history_lookback_days", 4, 0
    )
    if modification.instrument_kind == "option":
        try:
            payload = reference_client.option_market_history(
                {
                    "symbols": [modification.symbol],
                    "start_at": (decision_at - timedelta(days=history_lookback)).isoformat(),
                    "end_at": (history_end_at + timedelta(microseconds=1)).isoformat(),
                    "interval": "1d",
                    "include_greeks": True,
                }
            )
        except ReferenceDataClientError as exc:
            raise PnlMarkHistoryUnavailable(
                "Historical exact-option fill data is unavailable."
            ) from exc
        eligible = sorted(
            (
                item
                for item in payload.get("data") or []
                if str(item.get("canonical_occ_symbol") or "").upper()
                == modification.symbol
                and _as_datetime(item.get("at")) is not None
                and _as_datetime(item.get("at")) <= decision_at
            ),
            key=lambda item: _as_datetime(item.get("at")),
        )
        if eligible:
            bid = _number(eligible[-1].get("bid"))
            ask = _number(eligible[-1].get("ask"))
            mid = _number(eligible[-1].get("mark"))
        observations = sorted(
            (
                (at, mark)
                for item in payload.get("data") or []
                if str(item.get("canonical_occ_symbol") or "").upper()
                == modification.symbol
                and (at := _as_datetime(item.get("at"))) is not None
                and (mark := _number(item.get("mark"))) is not None
            ),
            key=lambda item: item[0],
        )
        tick = float(context.get("risk.pnl.default_option_tick", 0.01))
    else:
        try:
            groups = reference_client.bars(
                [modification.symbol],
                start=(decision_at - timedelta(days=history_lookback)).date(),
                end=history_end_at.date(),
            )
        except ReferenceDataClientError as exc:
            raise PnlMarkHistoryUnavailable(
                "Historical underlying/ETF fill data is unavailable."
            ) from exc
        eligible_bars = []
        for group in groups:
            if str(group.get("symbol") or "").upper() != modification.symbol:
                continue
            for bar in group.get("bars") or []:
                available_at = _daily_bar_available_at(bar.get("timestamp"))
                if available_at is not None and available_at <= decision_at:
                    eligible_bars.append((available_at, bar))
        if eligible_bars:
            bar = max(eligible_bars, key=lambda item: item[0])[1]
            low = _number(bar.get("low"))
            high = _number(bar.get("high"))
            close = _number(bar.get("close"))
            bid = low or close
            ask = high or close
            if low is not None and high is not None:
                mid = (low + high) / 2.0
            else:
                mid = close
        observations = sorted(
            (
                (available_at, close)
                for group in groups
                if str(group.get("symbol") or "").upper() == modification.symbol
                for bar in group.get("bars") or []
                if (available_at := _daily_bar_available_at(bar.get("timestamp")))
                is not None
                and (close := _number(bar.get("close"))) is not None
            ),
            key=lambda item: item[0],
        )
        tick = float(context.get("risk.pnl.default_equity_tick", 0.01))
    if not math.isfinite(tick) or tick <= 0:
        raise InvalidPnlAnalysis("configured minimum tick must be finite and positive")
    # Choose the historical bid or ask according to the proposed trade direction.
    buying = signed_quantity > 0
    if fill_model == "historical_mid":
        fill = mid
        warnings.append("historical_mid is a non-executable research fill model")
    elif fill_model == "historical_touch_plus_tick":
        touch = ask if buying else bid
        fill = touch + tick if buying and touch is not None else touch - tick if touch is not None else None
    else:
        fill = ask if buying else bid
    if fill is None:
        decision_point = next(
            (
                item
                for item in actual_path
                if item.at >= decision_at and item.underlying_price is not None
            ),
            None,
        )
        if decision_point is None or decision_point.underlying_price is None:
            raise PnlMarkHistoryUnavailable(
                f"A historical executable fill is unavailable for {modification.symbol}."
            )
        fill = decision_point.underlying_price
        warnings.append(
            f"{modification.symbol}: executable touch unavailable; prior observable mark used"
        )
    return fill, warnings, observations


def _daily_bar_available_at(value: Any) -> datetime | None:
    parsed = _as_datetime(value)
    if parsed is None:
        return None
    text = str(value)
    if "T" not in text and " " not in text:
        return datetime.combine(
            parsed.date() + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
    return parsed


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value not in (None, ""):
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _setting_int(
    context: Mapping[str, Any], key: str, default: int, minimum: int
) -> int:
    value = int(context.get(key, default))
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def _money(value: float | None) -> float | None:
    if value is None:
        return None
    if abs(value) < 0.0000005:
        return 0.0
    return round(float(value), 6)


__all__ = [
    "InvalidPnlAnalysis",
    "PnlSelectionNotFound",
    "analyze_selection",
]
