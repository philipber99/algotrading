"""Causal, SQL-first exact-contract historical option state resolver."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import math
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field

from algotrading.refdata.api.live_models import OptionMarketHistoryRequest
from algotrading.refdata.api.option_symbols import parse_occ_symbol
from algotrading.risk.instruments.terms import build_instrument_terms
from algotrading.risk.market_inputs.dividends import derive_trailing_cash_yield
from algotrading.risk.market_inputs.rates import interpolate_risk_free_rate
from algotrading.risk.models import ImmutableModel, InstrumentTerms
from algotrading.risk.pricing.tian_service import (
    IvGreeksContract,
    IvGreeksRequest,
    TianIvGreeksService,
)


class HistoricalOptionObservation(ImmutableModel):
    canonical_occ_symbol: str
    instrument_terms: InstrumentTerms
    session_date: date
    at: datetime
    observed_at: datetime | None = None
    bid: float | None = Field(default=None, ge=0)
    ask: float | None = Field(default=None, ge=0)
    mark: float = Field(ge=0)
    mark_basis: str
    source: str
    underlying_spot: float | None = Field(default=None, gt=0)
    underlying_source: str | None = None
    risk_free_rate: float | None = None
    risk_free_rate_source: str | None = None
    dividend_yield: float | None = Field(default=None, ge=0)
    dividend_inputs: tuple[dict[str, Any], ...] = ()
    dividend_source: str | None = None
    provider_iv: float | None = Field(default=None, gt=0)
    provider_delta: float | None = None
    provider_gamma: float | None = None
    provider_theta_per_day: float | None = None
    provider_vega_per_1pct: float | None = None
    provider_rho_per_1pct: float | None = None
    locally_inverted_iv: float | None = Field(default=None, gt=0)
    local_delta: float | None = None
    local_gamma: float | None = None
    local_theta_per_day: float | None = None
    local_vega_per_1pct: float | None = None
    local_rho_per_1pct: float | None = None
    solver_diagnostics: dict[str, Any] = Field(default_factory=dict)
    surface_state: dict[str, Any] | None = None
    quality_status: str
    missing_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class HistoricalOptionSeries(ImmutableModel):
    canonical_occ_symbol: str
    instrument_terms: InstrumentTerms
    observations: tuple[HistoricalOptionObservation, ...]
    intervals: tuple["HistoricalOptionInterval", ...] = ()
    calculation_method: str
    input_mode: str
    methodology_version: str = "historical_option_interval_resolver_v4"
    quality_status: str
    contract_iv_observations: int = 0
    minimum_contract_iv_observations: int = 2
    time_series_usable: bool = False
    history_diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class HistoricalOptionInterval(ImmutableModel):
    canonical_occ_symbol: str
    start_session: date
    end_session: date
    start_at: datetime
    end_at: datetime
    start_observed_at: datetime
    end_observed_at: datetime
    start_mark: float = Field(ge=0)
    end_mark: float = Field(ge=0)
    start_mark_basis: str
    end_mark_basis: str
    exact_pnl_per_contract: float
    signed_position_quantity: float = 1.0
    exact_position_pnl_usd: float
    method: Literal[
        "surface",
        "contract_iv",
        "bounded_greek_bridge",
        "observed_price_only",
        "single_mark_only",
        "insufficient",
    ]
    spot_pnl_usd: float | None = None
    volatility_pnl_usd: float | None = None
    skew_pnl_usd: float | None = None
    term_structure_pnl_usd: float | None = None
    time_pnl_usd: float | None = None
    rate_pnl_usd: float | None = None
    dividend_pnl_usd: float | None = None
    residual_pnl_usd: float | None = None
    observed_repricing_pnl_usd: float | None = None
    lifecycle_pnl_usd: float = 0.0
    premium_multiplier: float = Field(gt=0)
    terms_version: str
    start_source: str
    end_source: str
    quality_status: str
    missing_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    methodology_version: str = "historical_option_interval_resolver_v4"
    reconciliation_difference_usd: float = 0.0


class HistoricalOptionStateResolver:
    def __init__(
        self,
        reference_client: Any,
        context: Mapping[str, Any],
        *,
        risk_history_service: Any | None = None,
        tian_service: TianIvGreeksService | None = None,
    ) -> None:
        self.reference_client = reference_client
        self.context = context
        self.risk_history_service = risk_history_service
        if tian_service is not None:
            self.tian_service = tian_service
        else:
            # Historical IVs support daily shock estimation rather than the
            # authoritative current valuation. Bound the adaptive tree here so
            # thousands of old marks cannot each escalate to the 2,048-step
            # current-pricing maximum. Non-convergent rows transparently fall
            # through to the disclosed fixed-current-IV bridge.
            historical_context = dict(context)
            historical_context["risk.pricing.adaptive_base_steps"] = int(
                context.get("risk.option_history.adaptive_base_steps", 128)
            )
            historical_context["risk.pricing.adaptive_max_steps"] = int(
                context.get("risk.option_history.adaptive_max_steps", 256)
            )
            self.tian_service = TianIvGreeksService(historical_context)

    def resolve(
        self,
        symbols: Sequence[str],
        *,
        start_at: datetime,
        end_at: datetime,
        causal_as_of: datetime | None = None,
        terms_by_symbol: Mapping[str, InstrumentTerms] | None = None,
        quantity_by_symbol: Mapping[str, float] | None = None,
        required_sessions_by_symbol: Mapping[str, Sequence[date]] | None = None,
        underlying_closes_by_symbol: Mapping[
            str, Mapping[date, float]
        ] | None = None,
    ) -> dict[str, HistoricalOptionSeries]:
        started = perf_counter()
        start = _aware(start_at)
        end = _aware(end_at)
        causal_boundary = _aware(causal_as_of) if causal_as_of is not None else end
        if end <= start:
            raise ValueError("end_at must be after start_at")
        canonical = tuple(sorted({parse_occ_symbol(item).canonical for item in symbols}))
        if not canonical:
            return {}
        request = OptionMarketHistoryRequest(
            symbols=list(canonical), start_at=start, end_at=end, include_greeks=True
        )
        # Prefer durable Risk history when the repository-backed service is available.
        if self.risk_history_service is not None:
            payload_model = self.risk_history_service.query(
                request,
                required_sessions_by_symbol=required_sessions_by_symbol,
                causal_as_of=causal_boundary,
            )
            payload = payload_model.model_dump(mode="python")
        else:
            payload = self.reference_client.option_market_history(
                request.model_dump(mode="json")
            )
        query_finished = perf_counter()
        raw_rows = [
            dict(row)
            for row in payload.get("data") or []
            if str(row.get("status") or "") == "ok"
            and _finite(row.get("mark")) is not None
        ]
        roots = {parse_occ_symbol(symbol).root for symbol in canonical}
        bars = (
            {
                (str(root).upper(), session): float(close)
                for root, values in underlying_closes_by_symbol.items()
                if str(root).upper() in roots
                for session, close in values.items()
            }
            if underlying_closes_by_symbol is not None
            else self._underlying_closes(roots, start.date(), end.date())
        )
        dividends = self._dividends(start.date(), end.date(), roots)
        prepared: list[dict[str, Any]] = []
        for row in sorted(
            raw_rows,
            key=lambda item: (
                str(item.get("canonical_occ_symbol") or ""),
                str(item.get("at") or ""),
                str(item.get("source") or ""),
            ),
        ):
            symbol = str(row.get("canonical_occ_symbol") or "").upper()
            if symbol not in canonical:
                continue
            at = _timestamp(row.get("at"))
            if at is None:
                continue
            session = _session(row, at)
            allowed_sessions = (
                set((required_sessions_by_symbol or {}).get(symbol, ()))
                if required_sessions_by_symbol is not None
                else None
            )
            if allowed_sessions is not None:
                if session not in allowed_sessions:
                    continue
            elif not (start.date() <= session <= (end - timedelta(microseconds=1)).date()):
                continue
            observed_at = _timestamp(row.get("observed_at")) or at
            # Publication time, not just session date, enforces causal history selection.
            if observed_at > causal_boundary:
                continue
            occ = parse_occ_symbol(symbol)
            terms = (terms_by_symbol or {}).get(symbol) or build_instrument_terms(
                symbol, broker_type="OPTION", context=self.context
            )
            spot = _positive(row.get("underlying_price"))
            underlying_source = str(row.get("source") or "exact_option_history") if spot else None
            if spot is None:
                spot = bars.get((occ.root, session))
                underlying_source = "refdata_daily_close" if spot is not None else None
            event_rows = tuple(dividends.get(occ.root, ()))
            dividend_yield = None
            dividend_source = None
            if spot is not None and event_rows:
                try:
                    dividend_yield = derive_trailing_cash_yield(
                        event_rows,
                        spot=spot,
                        valuation_date=at.date(),
                    ).continuous_yield
                    dividend_source = "refdata_known_dividends_trailing_yield"
                except ValueError:
                    dividend_yield = None
            if dividend_yield is None:
                configured = _finite(
                    self.context.get("risk.dividends.default_continuous_yield")
                )
                if configured is not None and configured >= 0:
                    dividend_yield = configured
                    dividend_source = "configured_continuous_yield_fallback"
            prepared.append(
                {
                    "row": row,
                    "symbol": symbol,
                    "at": at,
                    "observed_at": observed_at,
                    "session": session,
                    "terms": terms,
                    "spot": spot,
                    "underlying_source": underlying_source,
                    "rate": None,
                    "rate_source": None,
                    "dividend_yield": dividend_yield,
                    "dividend_source": dividend_source,
                    "dividend_inputs": event_rows,
                }
            )
        prepared_finished = perf_counter()
        # Prefetch distinct session curves before solving option IV in batches.
        curves = self._risk_free_curves(
            {item["session"] for item in prepared}
        )
        curves_finished = perf_counter()
        for item in prepared:
            item["rate"], item["rate_source"] = self._rate(
                item["session"], item["terms"], curves
            )
        solved = self._solve_local_iv(prepared)
        solved_finished = perf_counter()
        grouped: dict[str, list[HistoricalOptionObservation]] = {
            symbol: [] for symbol in canonical
        }
        for index, item in enumerate(prepared):
            row = item["row"]
            diagnostics = solved.get(index, {})
            missing = [
                field
                for field, value in (
                    ("underlying_spot", item["spot"]),
                    ("risk_free_rate", item["rate"]),
                    ("dividend_yield", item["dividend_yield"]),
                )
                if value is None
            ]
            warnings = [str(value) for value in row.get("warnings") or ()]
            if diagnostics.get("reason") not in {None, "complete"}:
                warnings.append(
                    f"contract_iv_unavailable:{diagnostics.get('reason')}"
                )
            grouped[item["symbol"]].append(
                HistoricalOptionObservation(
                    canonical_occ_symbol=item["symbol"],
                    instrument_terms=item["terms"],
                    session_date=item["session"],
                    at=item["at"],
                    observed_at=item["observed_at"],
                    bid=_nonnegative(row.get("bid")),
                    ask=_nonnegative(row.get("ask")),
                    mark=float(row["mark"]),
                    mark_basis=str(row.get("mark_basis") or "provider_mark"),
                    source=str(row.get("source") or "unknown"),
                    underlying_spot=item["spot"],
                    underlying_source=item["underlying_source"],
                    risk_free_rate=item["rate"],
                    risk_free_rate_source=item["rate_source"],
                    dividend_yield=item["dividend_yield"],
                    dividend_inputs=item["dividend_inputs"],
                    dividend_source=item["dividend_source"],
                    provider_iv=_positive(row.get("provider_iv")),
                    provider_delta=_finite(row.get("provider_delta")),
                    provider_gamma=_finite(row.get("provider_gamma")),
                    provider_theta_per_day=_finite(row.get("provider_theta_per_day")),
                    provider_vega_per_1pct=_finite(row.get("provider_vega_per_1pct")),
                    provider_rho_per_1pct=_finite(row.get("provider_rho_per_1pct")),
                    locally_inverted_iv=_positive(diagnostics.get("implied_volatility")),
                    local_delta=_finite(diagnostics.get("delta")),
                    local_gamma=_finite(diagnostics.get("gamma")),
                    local_theta_per_day=_finite(diagnostics.get("theta_per_day")),
                    local_vega_per_1pct=_finite(diagnostics.get("vega_per_1pct")),
                    local_rho_per_1pct=_finite(diagnostics.get("rho_per_1pct")),
                    solver_diagnostics=diagnostics,
                    surface_state=(
                        dict(row["surface_state"])
                        if isinstance(row.get("surface_state"), Mapping)
                        else None
                    ),
                    quality_status="usable" if not missing and diagnostics.get("implied_volatility") else "usable_with_warnings",
                    missing_fields=tuple(missing),
                    warnings=tuple(dict.fromkeys(warnings)),
                )
            )
        output: dict[str, HistoricalOptionSeries] = {}
        for symbol in canonical:
            observations = _deduplicated_observations(grouped[symbol])
            iv_count = sum(item.locally_inverted_iv is not None for item in observations)
            minimum_iv = max(
                2,
                int(self.context.get("risk.option_history.minimum_contract_iv_observations", 126)),
            )
            if len(observations) < 1:
                status, mode = "insufficient_data", "insufficient"
                warnings = ("exact_option_price_history_unavailable",)
            elif iv_count >= 2:
                status, mode = "usable_with_warnings", "exact_contract_price_history"
                warnings = ("surface_unavailable_contract_iv_history_used",)
            else:
                status, mode = "usable_with_warnings", "exact_contract_price_history"
                warnings = ("contract_iv_history_partial",)
            terms = (terms_by_symbol or {}).get(symbol) or build_instrument_terms(
                symbol, broker_type="OPTION", context=self.context
            )
            # Two exact observations define each interval, even when sensitivities are missing.
            intervals = _build_intervals(
                observations,
                terms,
                quantity=float((quantity_by_symbol or {}).get(symbol, 1.0)),
                context=self.context,
            )
            if not intervals and len(observations) == 1:
                mode = "single_mark_only"
                warnings = (*warnings, "historical_interval_requires_two_exact_marks")
            time_series_usable = len(intervals) >= 1 and (
                iv_count >= minimum_iv
                or any(item.method in {"observed_price_only", "bounded_greek_bridge"} for item in intervals)
            )
            if iv_count >= 2 and iv_count < minimum_iv:
                warnings = (*warnings, "contract_iv_history_minimum_observations_not_met")
            output[symbol] = HistoricalOptionSeries(
                canonical_occ_symbol=symbol,
                instrument_terms=terms,
                observations=observations,
                intervals=intervals,
                calculation_method=(
                    "exact_contract_price_history_contract_iv"
                    if iv_count >= 2
                    else "exact_contract_price_history_marks_only"
                ),
                input_mode=mode,
                quality_status=status,
                contract_iv_observations=iv_count,
                minimum_contract_iv_observations=minimum_iv,
                time_series_usable=time_series_usable,
                history_diagnostics=_symbol_history_diagnostics(
                    symbol, payload.get("diagnostics")
                ),
                warnings=tuple(dict.fromkeys(warnings)),
            )
        logger = self.context.get("logger")
        if logger is not None:
            logger.info(
                "Historical option resolution symbols=%d rows=%d query_ms=%.2f "
                "prepare_ms=%.2f curves_ms=%.2f solve_ms=%.2f finalize_ms=%.2f",
                len(canonical),
                len(raw_rows),
                (query_finished - started) * 1000,
                (prepared_finished - query_finished) * 1000,
                (curves_finished - prepared_finished) * 1000,
                (solved_finished - curves_finished) * 1000,
                (perf_counter() - solved_finished) * 1000,
            )
        return output

    def _underlying_closes(
        self, roots: set[str], start: date, end: date
    ) -> dict[tuple[str, date], float]:
        try:
            groups = self.reference_client.bars(
                sorted(roots), start=start, end=end, remote_query=False
            )
        except TypeError:
            groups = self.reference_client.bars(sorted(roots), start=start, end=end)
        except Exception:
            return {}
        result: dict[tuple[str, date], float] = {}
        for group in groups:
            root = str(group.get("symbol") or "").upper()
            for bar in group.get("bars") or ():
                session = _date(bar.get("session_date") or str(bar.get("timestamp") or "")[:10])
                close = _positive(bar.get("close"))
                if root and session and close is not None:
                    result[(root, session)] = close
        return result

    def _dividends(
        self, start: date, end: date, roots: set[str]
    ) -> dict[str, list[dict[str, Any]]]:
        try:
            rows = self.reference_client.dividend_calendar(
                start=start - timedelta(days=365), end=end
            )
        except Exception:
            return {}
        result: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            root = str(row.get("symbol") or "").upper()
            if root in roots:
                result.setdefault(root, []).append(dict(row))
        return result

    def _rate(
        self,
        session: date,
        terms: InstrumentTerms,
        curves: dict[date, Mapping[str, Any] | None],
    ) -> tuple[float | None, str | None]:
        if session not in curves:
            try:
                payload = self.reference_client.risk_free_curve(session)
                curves[session] = payload.get("data")
            except Exception:
                curves[session] = None
        curve = curves[session]
        if curve is not None and terms.expiration_at is not None:
            try:
                rate = interpolate_risk_free_rate(
                    curve,
                    valuation_date=session,
                    tenor_days=max(0, (terms.expiration_at.date() - session).days),
                    max_age_days=int(self.context.get("risk.rates.max_age_days", 5)),
                    allow_flat_extrapolation=True,
                ).continuous_annual_rate_decimal
                return rate, "refdata_risk_free_curve"
            except ValueError:
                pass
        configured = _finite(
            self.context.get("risk.rates.configured_flat_continuous_rate")
        )
        return configured, "configured_flat_rate" if configured is not None else None

    def _risk_free_curves(
        self, sessions: set[date]
    ) -> dict[date, Mapping[str, Any] | None]:
        """Load point-in-time curves in one batch, with legacy fallback."""
        ordered = tuple(sorted(sessions))
        if not ordered:
            return {}
        batch = getattr(self.reference_client, "risk_free_curves", None)
        if callable(batch):
            try:
                payloads = batch(ordered)
                return {
                    session: (
                        payloads[session].get("data")
                        if session in payloads
                        else None
                    )
                    for session in ordered
                }
            except Exception:
                # Older RefData deployments do not publish the batch contract.
                # Retain the bounded concurrent path during rolling upgrades.
                pass
        workers = min(
            len(ordered),
            max(
                1,
                int(
                    self.context.get(
                        "risk.history.risk_free_curve_prefetch_workers", 8
                    )
                ),
            ),
        )

        def load(session: date) -> tuple[date, Mapping[str, Any] | None]:
            try:
                payload = self.reference_client.risk_free_curve(session)
                return session, payload.get("data")
            except Exception:
                return session, None

        if workers == 1:
            return dict(load(session) for session in ordered)
        curves: dict[date, Mapping[str, Any] | None] = {}
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="risk-curve-prefetch",
        ) as executor:
            pending = {executor.submit(load, session): session for session in ordered}
            for future in as_completed(pending):
                session, curve = future.result()
                curves[session] = curve
        return curves

    def _solve_local_iv(self, prepared: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
        # Group IV solves by valuation timestamp so every batch shares one market clock.
        grouped: dict[datetime, list[tuple[int, IvGreeksContract]]] = {}
        for index, item in enumerate(prepared):
            terms = item["terms"]
            if (
                item["spot"] is None
                or item["rate"] is None
                or item["dividend_yield"] is None
                or terms.expiration_at is None
                or terms.exercise_style not in {"american", "european"}
                or item["at"] >= terms.expiration_at
            ):
                continue
            contract = (
                IvGreeksContract(
                    request_id=f"history:{item['symbol']}:{item['at'].isoformat()}",
                    canonical_occ_symbol=item["symbol"],
                    underlying_symbol=terms.underlying_identifier or parse_occ_symbol(item["symbol"]).root,
                    put_call=terms.put_call or parse_occ_symbol(item["symbol"]).put_call,
                    exercise_type={
                        "european": "EUROPEAN",
                        "american": "AMERICAN",
                    }[terms.exercise_style],
                    strike=terms.strike or float(parse_occ_symbol(item["symbol"]).strike),
                    expiration_at=terms.expiration_at,
                    spot=item["spot"],
                    mark=float(item["row"]["mark"]),
                    quote_time=item["at"],
                    source_iv=_positive(item["row"].get("provider_iv")),
                    risk_free_rate=item["rate"],
                    dividend_yield=item["dividend_yield"],
                )
            )
            grouped.setdefault(item["at"], []).append((index, contract))
        output: dict[int, dict[str, Any]] = {}
        for valuation_at, rows in sorted(grouped.items()):
            try:
                response = self.tian_service.solve(
                    IvGreeksRequest(
                        valuation_at=valuation_at,
                        contracts=[contract for _, contract in rows],
                    )
                )
            except Exception:
                continue
            output.update(
                {
                    index: item.model_dump(mode="python")
                    for (index, _), item in zip(rows, response.data)
                }
            )
        return output


def _deduplicated_observations(
    rows: Sequence[HistoricalOptionObservation],
) -> tuple[HistoricalOptionObservation, ...]:
    selected: dict[date, HistoricalOptionObservation] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            item.session_date,
            _source_priority(item.source),
            item.observed_at or item.at,
            item.at,
        ),
    ):
        # Resolve duplicate sessions by source priority before constructing intervals.
        existing = selected.get(row.session_date)
        if existing is None or _source_priority(row.source) >= _source_priority(existing.source):
            selected[row.session_date] = row
    return tuple(selected[session] for session in sorted(selected))


def _source_priority(source: str) -> int:
    normalized = source.lower()
    if any(token in normalized for token in ("risk_mysql", "risk_option", "sql", "database")):
        return 3
    if any(token in normalized for token in ("refdata", "remote", "provider")):
        return 2
    return 1


def _build_intervals(
    observations: Sequence[HistoricalOptionObservation],
    terms: InstrumentTerms,
    *,
    quantity: float,
    context: Mapping[str, Any],
) -> tuple[HistoricalOptionInterval, ...]:
    intervals = [
        resolve_historical_option_interval(
            left,
            right,
            terms,
            quantity=quantity,
            context=context,
        )
        for left, right in zip(observations, observations[1:])
    ]
    return tuple(item for item in intervals if item is not None)


def resolve_historical_option_interval(
    left: HistoricalOptionObservation,
    right: HistoricalOptionObservation,
    terms: InstrumentTerms,
    *,
    quantity: float,
    context: Mapping[str, Any],
    surface_components: Mapping[str, float] | None = None,
    contract_iv_components: Mapping[str, float] | None = None,
) -> HistoricalOptionInterval | None:
    """Resolve the single canonical exact-mark interval used by every consumer.

    Optional model components are quote-price changes per unit of premium. The
    resolver applies signed quantity and premium multiplier exactly once and
    keeps the exact observed mark change authoritative.
    """
    multiplier = terms.price_multiplier
    if multiplier is None or not math.isfinite(multiplier) or multiplier <= 0:
        return None
    # Exact mark changes define total P&L independently of its modeled explanation.
    per_contract = (right.mark - left.mark) * multiplier
    exact = quantity * per_contract
    scale = quantity * multiplier
    missing: list[str] = []
    warnings: list[str] = []
    components: dict[str, float | None] = {
        "spot": None,
        "volatility": None,
        "skew": None,
        "term_structure": None,
        "time": None,
        "rate": None,
        "dividend": None,
        "residual": None,
        "observed": None,
    }

    def apply_model(values: Mapping[str, float]) -> None:
        aliases = {
            "underlying": "spot",
            "rates": "rate",
            "dividends": "dividend",
            "model_residual": "residual",
            "observed_repricing": "observed",
        }
        for key, value in values.items():
            target = aliases.get(key, key)
            if target in components and _finite(value) is not None:
                components[target] = scale * float(value)

    # Choose the strongest available attribution evidence before considering a Greek bridge.
    if surface_components is not None or (
        left.surface_state is not None and right.surface_state is not None
    ):
        method = "surface"
        if surface_components is not None:
            apply_model(surface_components)
        else:
            components["residual"] = exact
            warnings.append(
                "surface_interval_decomposition_not_available_in_exact_mark_resolver"
            )
    elif contract_iv_components is not None or all(
        value is not None
        for value in (
            left.locally_inverted_iv,
            right.locally_inverted_iv,
            left.underlying_spot,
            right.underlying_spot,
        )
    ):
        method = "contract_iv"
        if contract_iv_components is not None:
            apply_model(contract_iv_components)
        else:
            spot_change = float(right.underlying_spot) - float(left.underlying_spot)
            average_delta = _average(left.local_delta, right.local_delta)
            average_gamma = _average(left.local_gamma, right.local_gamma)
            components["spot"] = scale * (
                (average_delta or 0.0) * spot_change
                + 0.5 * (average_gamma or 0.0) * spot_change * spot_change
            )
            iv_points = 100.0 * (
                float(right.locally_inverted_iv) - float(left.locally_inverted_iv)
            )
            components["volatility"] = scale * (
                (_average(left.local_vega_per_1pct, right.local_vega_per_1pct) or 0.0)
                * iv_points
            )
            elapsed_days = (right.at - left.at).total_seconds() / 86400.0
            components["time"] = scale * (
                (_average(left.local_theta_per_day, right.local_theta_per_day) or 0.0)
                * elapsed_days
            )
            rate_points = 100.0 * (
                float(right.risk_free_rate or 0.0) - float(left.risk_free_rate or 0.0)
            )
            components["rate"] = scale * (
                (_average(left.local_rho_per_1pct, right.local_rho_per_1pct) or 0.0)
                * rate_points
            )
            components["dividend"] = 0.0
    else:
        left_delta = left.provider_delta if left.provider_delta is not None else left.local_delta
        right_delta = right.provider_delta if right.provider_delta is not None else right.local_delta
        left_gamma = left.provider_gamma if left.provider_gamma is not None else left.local_gamma
        right_gamma = right.provider_gamma if right.provider_gamma is not None else right.local_gamma
        maximum_spot_move = float(
            context.get("risk.option_history.greek_bridge_maximum_spot_return", 0.10)
        )
        bounded = (
            left.underlying_spot is not None
            and right.underlying_spot is not None
            and left_delta is not None
            and abs(float(right.underlying_spot) / float(left.underlying_spot) - 1.0)
            <= maximum_spot_move
        )
        if bounded:
            method = "bounded_greek_bridge"
            move = float(right.underlying_spot) - float(left.underlying_spot)
            average_delta = _average(left_delta, right_delta)
            average_gamma = _average(left_gamma, right_gamma)
            components["spot"] = scale * (
                float(average_delta or 0.0) * move
                + 0.5 * float(average_gamma or 0.0) * move * move
            )
            average_theta = _average(
                left.provider_theta_per_day, right.provider_theta_per_day
            )
            if average_theta is not None:
                components["time"] = (
                    scale
                    * average_theta
                    * (right.at - left.at).total_seconds()
                    / 86400.0
                )
            average_vega = _average(
                left.provider_vega_per_1pct, right.provider_vega_per_1pct
            )
            if (
                average_vega is not None
                and left.provider_iv is not None
                and right.provider_iv is not None
            ):
                components["volatility"] = (
                    scale
                    * average_vega
                    * 100.0
                    * (right.provider_iv - left.provider_iv)
                )
            average_rho = _average(
                left.provider_rho_per_1pct, right.provider_rho_per_1pct
            )
            if (
                left.risk_free_rate is not None
                and right.risk_free_rate is not None
                and average_rho is not None
            ):
                components["rate"] = (
                    scale
                    * average_rho
                    * 100.0
                    * (right.risk_free_rate - left.risk_free_rate)
                )
            explained = sum(
                float(components[key] or 0.0)
                for key in ("spot", "volatility", "time", "rate", "dividend")
            )
            components["observed"] = exact - explained
            components["residual"] = 0.0
            warnings.append("bounded_greek_bridge_driver_detail_partial")
        else:
            method = "observed_price_only"
            components["observed"] = exact
            components["residual"] = 0.0
            missing.extend(
                [
                    "spot_driver",
                    "volatility_driver",
                    "time_driver",
                    "rate_driver",
                    "dividend_driver",
                ]
            )
            warnings.append("observed_option_repricing_driver_detail_unavailable")

    # Put unexplained P&L into the residual so attribution reconciles to exact marks.
    explained = sum(
        float(value or 0.0)
        for key, value in components.items()
        if key != "residual"
    )
    components["residual"] = float(components["residual"] or 0.0) + exact - (
        explained + float(components["residual"] or 0.0)
    )
    lifecycle = 0.0
    # Disclose expiry-crossing intervals as lifecycle events.
    if terms.expiration_at is not None and left.at < terms.expiration_at <= right.at:
        lifecycle = exact
        warnings.append("interval_crosses_authoritative_expiration")
    reconciliation = exact - sum(float(value or 0.0) for value in components.values())
    if abs(reconciliation) <= 0.01:
        reconciliation = 0.0
    else:
        components["residual"] = float(components["residual"] or 0.0) + reconciliation
        reconciliation = 0.0
    return HistoricalOptionInterval(
        canonical_occ_symbol=terms.canonical_instrument_id,
        start_session=left.session_date,
        end_session=right.session_date,
        start_at=left.at,
        end_at=right.at,
        start_observed_at=left.observed_at or left.at,
        end_observed_at=right.observed_at or right.at,
        start_mark=left.mark,
        end_mark=right.mark,
        start_mark_basis=left.mark_basis,
        end_mark_basis=right.mark_basis,
        exact_pnl_per_contract=per_contract,
        signed_position_quantity=quantity,
        exact_position_pnl_usd=exact,
        method=method,
        spot_pnl_usd=components["spot"],
        volatility_pnl_usd=components["volatility"],
        skew_pnl_usd=components["skew"],
        term_structure_pnl_usd=components["term_structure"],
        time_pnl_usd=components["time"],
        rate_pnl_usd=components["rate"],
        dividend_pnl_usd=components["dividend"],
        residual_pnl_usd=components["residual"],
        observed_repricing_pnl_usd=components["observed"],
        lifecycle_pnl_usd=lifecycle,
        premium_multiplier=multiplier,
        terms_version=terms.source_version,
        start_source=left.source,
        end_source=right.source,
        quality_status=(
            "usable" if method in {"surface", "contract_iv"} else "usable_with_warnings"
        ),
        missing_fields=tuple(missing),
        warnings=tuple(warnings),
        reconciliation_difference_usd=reconciliation,
    )


def _average(left: float | None, right: float | None) -> float | None:
    values = [float(value) for value in (left, right) if value is not None]
    return sum(values) / len(values) if values else None


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone aware")
    return value.astimezone(timezone.utc)


def _timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _session(row: Mapping[str, Any], at: datetime) -> date:
    return _date(row.get("session_date")) or at.date()


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


def _nonnegative(value: Any) -> float | None:
    converted = _finite(value)
    return converted if converted is not None and converted >= 0 else None


def _symbol_history_diagnostics(
    symbol: str, value: Any
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = {
        "sql_mark_count": int(
            (value.get("sql_mark_count_by_symbol") or {}).get(symbol, 0)
        ),
        "remote_mark_count": int(
            (value.get("remote_mark_count_by_symbol") or {}).get(symbol, 0)
        ),
        "required_sessions": list(
            (value.get("required_sessions_by_symbol") or {}).get(symbol, ())
        ),
        "local_maximum_days": value.get("local_maximum_days"),
        "advertised_maximum_days": value.get("advertised_maximum_days"),
        "effective_maximum_days": value.get("effective_maximum_days"),
        "server_limit_negotiated": bool(value.get("server_limit_negotiated")),
    }
    result["requested_remote_ranges"] = [
        row
        for row in value.get("requested_remote_ranges") or ()
        if symbol in (row.get("symbols") or ())
    ]
    result["failed_remote_ranges"] = [
        row
        for row in value.get("failed_remote_ranges") or ()
        if symbol in (row.get("symbols") or ())
    ]
    return result
