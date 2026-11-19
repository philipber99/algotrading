"""Two-stage bounded screening orchestration, caching, and diagnostics."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from .. import SCREENER_MODEL_VERSION
from ..cache.model import canonical_json
from ..errors import (
    PricingUnavailableError,
    ScreenerBusyError,
    ScreenerInvalidRequestError,
    ScreenerNotReadyError,
    ScreenerTimeoutError,
)
from .config import ScreenerConfig
from .eligibility import (
    CAPITAL_AT_RISK_LIMIT_EXCEEDED,
    DUPLICATE_CANDIDATE,
    INVALID_MARKET,
    NONPOSITIVE_EXPECTED_VALUE,
    PRICING_FAILURE,
    RejectionTracker,
    eligible_contracts,
)
from .execution import (
    SpreadPrices,
    execution_quality,
    quality_at_least,
    spread_prices,
)
from .models import (
    CandidateDraft,
    CandidateResult,
    CapabilitiesResponse,
    ExecutionResult,
    LegResult,
    ScreeningFunnel,
    ScreeningRequest,
    ScreeningResponse,
)
from .rationale import build_rationale
from .repository import ScreenerRepository
from .scenarios import PricingResources, ScenarioEngine, static_hedge_shares
from .scoring import fast_prescore, score_candidate, surface_relative_factors
from .strategies import StrategyRegistry, default_registry
from .strategies.base import PayoffMetrics


@dataclass(frozen=True)
class _Prescored:
    candidate: CandidateDraft
    prices: SpreadPrices
    execution: ExecutionResult
    entry_value: float
    payoff: PayoffMetrics
    prescore: float


class _ByteBoundedLru:
    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self._size = 0
        self._values: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> ScreeningResponse | None:
        with self._lock:
            payload = self._values.pop(key, None)
            if payload is None:
                return None
            self._values[key] = payload
        return ScreeningResponse.model_validate_json(payload)

    def put(self, key: str, value: ScreeningResponse) -> None:
        payload = value.model_dump_json().encode("utf-8")
        if len(payload) > self.maximum_bytes:
            return
        with self._lock:
            previous = self._values.pop(key, None)
            if previous is not None:
                self._size -= len(previous)
            self._values[key] = payload
            self._size += len(payload)
            while self._size > self.maximum_bytes and self._values:
                _, removed = self._values.popitem(last=False)
                self._size -= len(removed)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._size = 0


class ScreenerService:
    def __init__(
        self,
        context: Mapping[str, Any],
        repository: ScreenerRepository,
        *,
        config: ScreenerConfig | None = None,
        pricing: PricingResources | None = None,
        scenario_engine: ScenarioEngine | None = None,
        strategy_registry: StrategyRegistry | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.context = context
        self.config = config or ScreenerConfig.from_context(context)
        self.repository = repository
        self.pricing = pricing or PricingResources(context, config=self.config)
        self.scenarios = scenario_engine or ScenarioEngine(
            context, self.config, self.pricing
        )
        self.strategies = strategy_registry or default_registry()
        self.logger = logger or logging.getLogger(__name__)
        self._capacity = threading.BoundedSemaphore(self.config.max_concurrent_screens)
        self._cache = _ByteBoundedLru(self.config.result_cache_mb * 1024 * 1024)
        self._inflight: dict[str, Future[ScreeningResponse]] = {}
        self._inflight_lock = threading.Lock()
        self.ready = True

    def shutdown(self) -> None:
        self.ready = False
        self._cache.clear()
        self.repository.close()
        self.pricing.close()

    def health(self) -> dict[str, Any]:
        pricing_health = getattr(self.pricing, "health", None)
        return {
            "status": "ok" if self.ready else "not_ready",
            "model_version": SCREENER_MODEL_VERSION,
            "strategies": [
                "vertical",
                "calendar",
                "diagonal",
                "butterfly",
                "iron_condor",
                "double_calendar",
            ],
            "scenario_pricing_method": self.config.pricing_method,
            "pricing_pool": pricing_health() if callable(pricing_health) else None,
        }

    def capabilities(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            strategies=[
                "vertical",
                "calendar",
                "diagonal",
                "butterfly",
                "iron_condor",
                "double_calendar",
            ],
            static_delta_hedging=True,
            dynamic_delta_hedging=False,
            price_bases=["mid", "bid", "ask"],
            directional_assumption_modes=["zero", "risk_neutral", "explicit"],
            minimum_dte=0,
            maximum_dte=self.config.maximum_dte,
            maximum_results_per_strategy=self.config.max_results_per_strategy,
            maximum_simulation_paths=self.config.maximum_simulation_paths,
            maximum_holding_sessions=self.config.maximum_holding_sessions,
            model_version=SCREENER_MODEL_VERSION,
        )

    def screen(self, request: ScreeningRequest, *, request_id: str) -> ScreeningResponse:
        if not self.ready:
            raise ScreenerNotReadyError("The spread screener is not initialized")
        self._validate_request_bounds(request)
        started = time.perf_counter()
        require_surface = request.surface_relative_value or any(
            name in {"calendar", "diagonal", "double_calendar"}
            for name in request.enabled_strategies
        )
        snapshot = self.repository.resolve_snapshot(
            request.underlying_symbol,
            request.snapshot_id,
            require_surface=require_surface,
        )
        deadline = started + self.config.screen_timeout_seconds
        self._check_deadline(deadline)
        key = self._cache_key(snapshot.snapshot_id, request)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.model_copy(
                update={
                    "request_id": request_id,
                    "cache_hit": True,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                }
            )

        leader, future = self._inflight_future(key)
        if not leader:
            try:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise FutureTimeout()
                shared = future.result(timeout=remaining)
            except FutureTimeout as exc:
                raise ScreenerTimeoutError(
                    "Timed out waiting for an identical in-flight screen"
                ) from exc
            return shared.model_copy(
                update={
                    "request_id": request_id,
                    "cache_hit": True,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                }
            )

        acquired = self._capacity.acquire(blocking=False)
        if not acquired:
            error = ScreenerBusyError(
                "All configured spread-screening slots are in use"
            )
            self._finish_inflight(key, future, error=error)
            raise error
        try:
            response = self._screen_uncached(
                request,
                request_id=request_id,
                snapshot=snapshot,
                started=started,
            )
            self._cache.put(key, response)
            self._finish_inflight(key, future, response=response)
            return response
        except ScreenerTimeoutError as exc:
            pricing_health = getattr(self.pricing, "health", None)
            pool = pricing_health() if callable(pricing_health) else {}
            self.logger.warning(
                "event=spread_screen_timeout request_id=%s symbol=%s "
                "snapshot_id=%s elapsed_ms=%.2f scenario_pricing_method=%s "
                "pricing_pool=%s details=%s",
                request_id,
                request.underlying_symbol,
                snapshot.snapshot_id,
                (time.perf_counter() - started) * 1000,
                self.config.pricing_method,
                pool,
                exc.details,
            )
            self._finish_inflight(key, future, error=exc)
            raise
        except Exception as exc:
            self._finish_inflight(key, future, error=exc)
            raise
        finally:
            self._capacity.release()

    def _screen_uncached(
        self,
        request: ScreeningRequest,
        *,
        request_id: str,
        snapshot: Any,
        started: float,
    ) -> ScreeningResponse:
        deadline = started + self.config.screen_timeout_seconds
        tracker = RejectionTracker()
        funnel = ScreeningFunnel()
        require_surface = request.surface_relative_value or any(
            name in {"calendar", "diagonal", "double_calendar"}
            for name in request.enabled_strategies
        )
        self._check_deadline(deadline)
        database_started = time.perf_counter()
        universe = self.repository.load_universe(
            request.underlying_symbol,
            snapshot.snapshot_id,
            price_basis=request.price_basis,
            require_surface=require_surface,
            resolved_snapshot=snapshot,
        )
        database_elapsed_ms = (time.perf_counter() - database_started) * 1000
        self._check_deadline(deadline)
        funnel.contracts_loaded = len(universe.contracts)
        eligible = eligible_contracts(
            universe,
            request,
            tracker,
            require_surface=require_surface,
            deadline=deadline,
        )
        funnel.contracts_eligible = len(eligible)
        funnel.contracts_rejected = funnel.contracts_loaded - len(eligible)

        prescored: dict[str, list[_Prescored]] = defaultdict(list)
        seen: set[str] = set()
        for definition in self.strategies.enabled(request.enabled_strategies):
            self._check_deadline(deadline)
            generated = definition.generate(
                eligible,
                universe.snapshot.underlying_mark,
                request,
                self.config.max_generated_candidates_per_strategy,
            )
            funnel.structural_combinations_generated += len(generated)
            for candidate_index, candidate in enumerate(generated):
                if candidate_index % 64 == 0:
                    self._check_deadline(deadline)
                if not definition.structurally_valid(candidate):
                    tracker.add("STRUCTURAL_VALIDATION_FAILED", candidate.variant)
                    continue
                funnel.candidates_passing_structural_validation += 1
                identity = self._candidate_id(universe.snapshot.snapshot_id, candidate)
                if identity in seen:
                    tracker.add(DUPLICATE_CANDIDATE, identity)
                    continue
                seen.add(identity)
                prices = spread_prices(candidate)
                entry = prices.entry(request.entry_basis)
                if abs(entry) < 1.0e-9:
                    tracker.add(INVALID_MARKET, identity)
                    continue
                if self._has_deep_itm_short(
                    candidate, universe.snapshot.underlying_mark
                ):
                    tracker.add("EARLY_ASSIGNMENT_RISK", identity)
                    continue
                execution = execution_quality(
                    candidate,
                    prices,
                    self.config,
                    as_of=universe.snapshot.source_collected_at,
                )
                if (
                    execution.rating == "poor"
                    and not request.include_poor_execution
                    and self.config.poor_execution_excluded
                ) or not quality_at_least(
                    execution.rating, request.minimum_execution_quality
                ):
                    tracker.add("EXECUTION_QUALITY_BELOW_MINIMUM", identity)
                    continue
                funnel.candidates_passing_liquidity_filters += 1
                payoff = definition.payoff_metrics(candidate, entry)
                multiplier = min(leg.contract.multiplier for leg in candidate.legs)
                if request.maximum_debit is not None and entry > 0 and entry * multiplier > request.maximum_debit:
                    tracker.add(CAPITAL_AT_RISK_LIMIT_EXCEEDED, identity)
                    continue
                if (
                    request.maximum_capital_at_risk is not None
                    and payoff.capital_at_risk is not None
                    and payoff.capital_at_risk > request.maximum_capital_at_risk
                ):
                    tracker.add(CAPITAL_AT_RISK_LIMIT_EXCEEDED, identity)
                    continue
                reward_hint = self._reward_risk_hint(payoff, entry, multiplier)
                score = fast_prescore(candidate, execution.score, reward_hint)
                prescored[definition.name].append(
                    _Prescored(candidate, prices, execution, entry, payoff, score)
                )

        for name in prescored:
            prescored[name].sort(
                key=lambda item: (
                    -item.prescore,
                    self._candidate_id(universe.snapshot.snapshot_id, item.candidate),
                )
            )
            prescored[name] = prescored[name][
                : self.config.max_prescore_candidates_per_strategy
            ]
        funnel.candidates_pre_scored = sum(len(items) for items in prescored.values())

        selected: list[tuple[Any, _Prescored, int, float]] = []
        for name in request.enabled_strategies:
            definition = self.strategies.get(name)
            full_limit = min(
                self.config.max_full_scenario_candidates_per_strategy,
                max(
                    request.result_count_per_strategy,
                    request.result_count_per_strategy * 2,
                ),
            )
            for item in prescored.get(name, [])[:full_limit]:
                target_fraction = self._target_fraction(item, request)
                target_profit = target_fraction * abs(item.entry_value)
                planned = definition.planned_holding_sessions(
                    item.candidate,
                    request,
                    self.config.maximum_holding_sessions,
                )
                selected.append((definition, item, planned, target_profit))

        scenario_prepare_started = time.perf_counter()
        prepare = getattr(self.scenarios, "prepare", None)
        workspace = None
        if callable(prepare) and selected:
            workspace = prepare(
                [(item.candidate, planned) for _, item, planned, _ in selected],
                universe,
                request,
                deadline=deadline,
            )
        scenario_prepare_ms = (
            time.perf_counter() - scenario_prepare_started
        ) * 1000
        self._check_deadline(deadline)

        results: list[CandidateResult] = []
        pricing_failures: list[PricingUnavailableError] = []
        attempted_variants = 0
        failed_variants = 0
        scenario_evaluate_started = time.perf_counter()
        for _, base_item, planned, target_profit in selected:
            self._check_deadline(deadline)
            funnel.candidates_fully_evaluated += 1
            hedge_specs: list[tuple[bool, int]] = []
            if request.delta_hedge_mode in {"none", "both"}:
                hedge_specs.append((False, 0))
            if request.delta_hedge_mode in {"static", "both"}:
                hedge_specs.append(
                    (True, static_hedge_shares(base_item.candidate))
                )
            hedge_values = [shares for _, shares in hedge_specs]
            scenario_variants: list[tuple[bool, int, Any]] = []
            try:
                if workspace is not None:
                    metrics = workspace.evaluate_variants(
                        base_item.candidate,
                        planned_holding_sessions=planned,
                        target_profit_per_share=target_profit,
                        hedge_shares_values=hedge_values,
                    )
                    scenario_variants = [
                        (is_static, returned_shares, scenario)
                        for (is_static, _), (returned_shares, scenario) in zip(
                            hedge_specs, metrics
                        )
                    ]
                else:
                    for is_static, hedge_shares in hedge_specs:
                        scenario_variants.append(
                            (
                                is_static,
                                hedge_shares,
                                self.scenarios.evaluate(
                                    base_item.candidate,
                                    universe,
                                    request,
                                    planned_holding_sessions=planned,
                                    target_profit_per_share=target_profit,
                                    hedge_shares=hedge_shares,
                                    deadline=deadline,
                                ),
                            )
                        )
            except PricingUnavailableError as exc:
                tracker.add(
                    PRICING_FAILURE,
                    self._candidate_id(
                        universe.snapshot.snapshot_id, base_item.candidate
                    ),
                )
                pricing_failures.append(exc)
                attempted_variants += len(hedge_values)
                failed_variants += len(hedge_values)
                funnel.candidates_rejected_after_scenario_evaluation += len(
                    hedge_values
                )
                continue
            for is_static, hedge_shares, scenario in scenario_variants:
                self._check_deadline(deadline)
                attempted_variants += 1
                item = base_item
                if is_static:
                    item = replace(
                        base_item,
                        candidate=replace(
                            base_item.candidate,
                            hedge_mode="static",
                            hedge_shares=hedge_shares,
                        ),
                    )
                if scenario.expected_pnl <= 0 and not request.include_nonpositive_ev:
                    tracker.add(
                        NONPOSITIVE_EXPECTED_VALUE,
                        self._candidate_id(universe.snapshot.snapshot_id, item.candidate),
                    )
                    funnel.candidates_rejected_after_scenario_evaluation += 1
                    continue
                modeled_risk = max(0.0, -scenario.modeled_pnl_range[0])
                capital = item.payoff.capital_at_risk or modeled_risk
                if capital <= 0:
                    tracker.add("MISSING_CAPITAL_AT_RISK", item.candidate.variant)
                    funnel.candidates_rejected_after_scenario_evaluation += 1
                    continue
                if request.maximum_capital_at_risk is not None and capital > request.maximum_capital_at_risk:
                    tracker.add(CAPITAL_AT_RISK_LIMIT_EXCEEDED, item.candidate.variant)
                    funnel.candidates_rejected_after_scenario_evaluation += 1
                    continue
                result = self._candidate_result(
                    item,
                    universe,
                    scenario,
                    capital,
                    target_profit,
                    planned,
                )
                results.append(result)

        scenario_evaluate_ms = (
            time.perf_counter() - scenario_evaluate_started
        ) * 1000
        if pricing_failures and not results and attempted_variants == failed_variants:
            raise pricing_failures[0]
        results.sort(
            key=lambda item: (
                item.strategy,
                -item.scores.overall,
                -item.expected_pnl_after_costs,
                item.candidate_id,
            )
        )
        limited: list[CandidateResult] = []
        returned_by_strategy: dict[str, int] = defaultdict(int)
        for result in results:
            if returned_by_strategy[result.strategy] >= request.result_count_per_strategy:
                continue
            returned_by_strategy[result.strategy] += 1
            limited.append(result)
        limited.sort(
            key=lambda item: (
                -item.scores.overall,
                -item.expected_pnl_after_costs,
                item.strategy,
                item.candidate_id,
            )
        )
        funnel.results_returned = len(limited)
        diagnostics = tracker.diagnostics()
        elapsed_ms = (time.perf_counter() - started) * 1000
        performance = (
            asdict(workspace.performance)
            if workspace is not None
            else {
                "pricing_workers": self.config.resolved_pricing_workers,
                "scenario_pricing_method": self.config.pricing_method,
            }
        )
        performance.update(
            {
                "database_query_count": getattr(
                    self.repository, "last_query_count", None
                )
                + 1
                if isinstance(getattr(self.repository, "last_query_count", None), int)
                else None,
                "database_elapsed_ms": database_elapsed_ms,
                "scenario_prepare_ms": scenario_prepare_ms,
                "scenario_evaluate_ms": scenario_evaluate_ms,
                "total_elapsed_ms": elapsed_ms,
            }
        )
        self.logger.info(
            "event=spread_screen request_id=%s symbol=%s snapshot_id=%s "
            "strategies=%s funnel=%s cache_hit=false elapsed_ms=%.2f "
            "dominant_rejection=%s scenario_pricing_method=%s "
            "pricing_signature=%s performance=%s",
            request_id,
            request.underlying_symbol,
            universe.snapshot.snapshot_id,
            request.enabled_strategies,
            funnel.model_dump(),
            elapsed_ms,
            diagnostics.dominant_reason,
            self.config.pricing_method,
            getattr(self.pricing, "configuration_signature", "unavailable"),
            performance,
        )
        return ScreeningResponse(
            request_id=request_id,
            model_version=SCREENER_MODEL_VERSION,
            snapshot_id=universe.snapshot.snapshot_id,
            snapshot_timestamp=universe.snapshot.source_collected_at,
            underlying_bid=universe.snapshot.underlying_bid,
            underlying_ask=universe.snapshot.underlying_ask,
            underlying_mark=universe.snapshot.underlying_mark,
            applied_assumptions={
                "entry_basis": request.entry_basis,
                "price_basis": request.price_basis,
                "delta_hedge_mode": request.delta_hedge_mode,
                "drift_mode": request.probability.drift_mode,
                "annualized_drift": request.probability.annualized_drift,
                "scenario_pricing_method": self.config.pricing_method,
                "pricing_configuration_signature": getattr(
                    self.pricing, "configuration_signature", "unavailable"
                ),
                "simulation_paths": min(
                    request.probability.simulation_paths or self.config.simulation_paths,
                    self.config.maximum_simulation_paths,
                ),
                "iv_residual_half_life_sessions": (
                    request.probability.iv_residual_half_life_sessions
                    or self.config.iv_residual_half_life_sessions
                ),
                "spot_grid_points": self.config.spot_grid_points,
                "maximum_holding_sessions": self.config.maximum_holding_sessions,
                "max_full_scenario_candidates_per_strategy": (
                    self.config.max_full_scenario_candidates_per_strategy
                ),
                "static_hedge_rebalanced": False,
                "probability_outputs": "conditional_model_estimates",
            },
            candidates=limited,
            funnel=funnel,
            rejections=diagnostics,
            cache_hit=False,
            elapsed_ms=elapsed_ms,
            warnings=(
                []
                if limited
                else ["NO_QUALIFYING_CANDIDATES; see rejection diagnostics"]
            ),
            performance_diagnostics=performance,
        )

    def _candidate_result(
        self,
        item: _Prescored,
        universe: Any,
        scenario: Any,
        capital: float,
        target_profit: float,
        planned: int,
    ) -> CandidateResult:
        candidate = item.candidate
        factors = surface_relative_factors(candidate)
        scores = score_candidate(
            expected_pnl=scenario.expected_pnl,
            capital_at_risk=capital,
            probability_of_profit=scenario.probability_of_profit,
            target_probability=scenario.target_probability,
            execution_score=item.execution.score,
            robustness=scenario.robustness,
            target_profit=target_profit
            * min(leg.contract.multiplier for leg in candidate.legs),
        )
        expected_return = scenario.expected_pnl / capital
        rationale = build_rationale(
            candidate,
            surface_factors=factors,
            execution_rating=item.execution.rating,
            expected_return=expected_return,
        )
        net = {
            greek: sum(
                leg.sign
                * leg.quantity
                * float(getattr(leg.contract, greek) or 0.0)
                * leg.contract.multiplier
                for leg in candidate.legs
            )
            for greek in ("delta", "gamma", "theta", "vega")
        }
        legs = [
            LegResult(
                side=leg.side,
                quantity=leg.quantity,
                contract_id=leg.contract.contract_id,
                option_symbol=leg.contract.option_symbol,
                put_call=leg.contract.put_call,
                strike=leg.contract.strike,
                expiration_at=leg.contract.expiration_at,
                dte=leg.contract.dte,
                exercise_type=leg.contract.exercise_type,
                multiplier=leg.contract.multiplier,
                bid=float(leg.contract.bid),
                midpoint=leg.contract.midpoint,
                ask=float(leg.contract.ask),
                bid_size=leg.contract.bid_size,
                ask_size=leg.contract.ask_size,
                open_interest=leg.contract.open_interest,
                volume=leg.contract.volume,
                quote_time=leg.contract.quote_time,
                effective_iv=float(leg.contract.effective_iv),
                fitted_surface_iv=leg.contract.fitted_iv,
                iv_residual=leg.contract.iv_residual,
                delta=float(leg.contract.delta),
                gamma=float(leg.contract.gamma),
                theta=float(leg.contract.theta),
                vega=float(leg.contract.vega),
                iv_value_source=leg.contract.iv_value_source,
                greeks_value_source=leg.contract.greeks_value_source,
                execution_reasons=item.execution.per_leg_reasons.get(
                    leg.contract.option_symbol, []
                ),
            )
            for leg in candidate.legs
        ]
        return CandidateResult(
            candidate_id=self._candidate_id(universe.snapshot.snapshot_id, candidate),
            strategy=candidate.strategy,
            variant=candidate.variant,
            outlook=candidate.outlook,
            legs=legs,
            synthetic_bid=item.prices.synthetic_bid,
            synthetic_midpoint=item.prices.synthetic_midpoint,
            synthetic_ask=item.prices.synthetic_ask,
            entry_value=item.entry_value,
            opening_classification="debit" if item.entry_value > 0 else "credit",
            target_exit_value=item.entry_value + target_profit,
            target_profit=target_profit,
            expected_holding_sessions=scenario.expected_holding_sessions,
            expected_holding_time_basis=scenario.holding_basis,
            planned_maximum_holding_sessions=planned,
            model_probability_of_profit=scenario.probability_of_profit,
            probability_of_reaching_target=scenario.target_probability,
            expected_pnl_after_costs=scenario.expected_pnl,
            return_on_capital_at_risk=expected_return,
            exact_maximum_profit=item.payoff.exact_maximum_profit,
            exact_maximum_loss=item.payoff.exact_maximum_loss,
            maximum_profit_loss_basis=item.payoff.basis,
            breakeven=item.payoff.breakeven,
            breakevens=list(item.payoff.breakevens),
            modeled_pnl_range=scenario.modeled_pnl_range,
            pnl_percentiles=scenario.pnl_percentiles,
            net_delta=net["delta"],
            net_gamma=net["gamma"],
            net_theta=net["theta"],
            net_vega=net["vega"],
            static_hedge_shares=(
                candidate.hedge_shares if candidate.hedge_mode == "static" else None
            ),
            execution_quality=item.execution,
            surface_relative_factors=factors,
            scores=scores,
            rationale=rationale,
            warnings=list(dict.fromkeys((*candidate.warnings, *scenario.warnings))),
        )

    def _target_fraction(self, item: _Prescored, request: ScreeningRequest) -> float:
        debit = item.entry_value > 0
        if item.candidate.strategy in {"vertical", "butterfly", "iron_condor"}:
            if debit:
                return (
                    request.exit_policy.debit_vertical_target_fraction
                    or self.config.vertical_target_debit_fraction
                )
            return (
                request.exit_policy.credit_vertical_target_fraction
                or self.config.vertical_target_credit_fraction
            )
        if debit:
            return (
                request.exit_policy.debit_calendar_diagonal_target_fraction
                or self.config.calendar_diagonal_target_debit_fraction
            )
        return (
            request.exit_policy.credit_calendar_diagonal_target_fraction
            or self.config.calendar_diagonal_target_credit_fraction
        )

    @staticmethod
    def _reward_risk_hint(
        payoff: PayoffMetrics,
        entry_value: float,
        multiplier: float,
    ) -> float:
        if payoff.exact_maximum_profit is not None and payoff.capital_at_risk:
            return max(0.0, min(1.0, payoff.exact_maximum_profit / payoff.capital_at_risk))
        return 0.5 if entry_value != 0 else 0.0

    @staticmethod
    def _candidate_id(snapshot_id: str, candidate: CandidateDraft) -> str:
        material = {
            "snapshot_id": snapshot_id,
            "strategy": candidate.strategy,
            "variant": candidate.variant,
            "hedge_mode": candidate.hedge_mode,
            "hedge_shares": candidate.hedge_shares,
            "legs": sorted(
                (
                    leg.contract.contract_id,
                    leg.side,
                    leg.quantity,
                    leg.contract.multiplier,
                )
                for leg in candidate.legs
            ),
        }
        return hashlib.sha256(canonical_json(material)).hexdigest()[:24]

    def _cache_key(self, snapshot_id: str, request: ScreeningRequest) -> str:
        return hashlib.sha256(
            canonical_json(
                {
                    "snapshot_id": snapshot_id,
                    "request": request.model_dump(mode="json", exclude_none=True),
                    "model_version": SCREENER_MODEL_VERSION,
                    "scenario_pricing_method": self.config.pricing_method,
                    "pricing_configuration_signature": getattr(
                        self.pricing,
                        "configuration_signature",
                        canonical_json(asdict(self.config)).hex(),
                    ),
                }
            )
        ).hexdigest()

    def _inflight_future(self, key: str) -> tuple[bool, Future[ScreeningResponse]]:
        with self._inflight_lock:
            existing = self._inflight.get(key)
            if existing is not None:
                return False, existing
            future: Future[ScreeningResponse] = Future()
            self._inflight[key] = future
            return True, future

    def _finish_inflight(
        self,
        key: str,
        future: Future[ScreeningResponse],
        *,
        response: ScreeningResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        with self._inflight_lock:
            self._inflight.pop(key, None)
        if error is not None:
            future.set_exception(error)
        elif response is not None:
            future.set_result(response)

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        if time.perf_counter() > deadline:
            raise ScreenerTimeoutError("The bounded spread screen exceeded its deadline")

    def _validate_request_bounds(self, request: ScreeningRequest) -> None:
        violations: list[str] = []
        if request.dte.maximum > self.config.maximum_dte:
            violations.append("dte.maximum")
        if request.dte.back_maximum > self.config.maximum_dte:
            violations.append("dte.back_maximum")
        if request.result_count_per_strategy > self.config.max_results_per_strategy:
            violations.append("result_count_per_strategy")
        if (
            request.probability.simulation_paths is not None
            and request.probability.simulation_paths
            > self.config.maximum_simulation_paths
        ):
            violations.append("probability.simulation_paths")
        if (
            request.exit_policy.maximum_holding_sessions is not None
            and request.exit_policy.maximum_holding_sessions
            > self.config.maximum_holding_sessions
        ):
            violations.append("exit_policy.maximum_holding_sessions")
        if violations:
            raise ScreenerInvalidRequestError(
                "Screening request exceeds configured bounds",
                details={"fields": violations},
            )

    def _has_deep_itm_short(
        self,
        candidate: CandidateDraft,
        spot: float,
    ) -> bool:
        threshold = self.config.maximum_short_itm_fraction
        for leg in candidate.legs:
            if leg.side != "short":
                continue
            if (
                leg.contract.put_call == "CALL"
                and leg.contract.strike < spot * (1.0 - threshold)
            ) or (
                leg.contract.put_call == "PUT"
                and leg.contract.strike > spot * (1.0 + threshold)
            ):
                return True
        return False


__all__ = ["ScreenerService"]
