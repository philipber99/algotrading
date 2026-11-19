"""Snapshot-coherent, research-only Hedge Lab calculations.

The service deliberately owns no broker, transaction, strategy-group, or order
dependency.  Proposed positions exist only as immutable in-memory models for the
duration of one request.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from threading import RLock
from typing import Any, Iterable, Literal, Mapping, Sequence
from zoneinfo import ZoneInfo

from algotrading.refdata.api.option_symbols import parse_occ_symbol
from algotrading.risk.exposures.factors import FACTOR_PROXY_SYMBOLS, ridge_factor_exposure
from algotrading.risk.exposures.portfolio import delta_equivalent_usd
from algotrading.risk.market_inputs.rates import interpolate_risk_free_rate
from algotrading.risk.market_inputs.dividends import (
    derive_trailing_cash_yield,
    escrowed_dividend_adjustment,
)
from algotrading.risk.market_inputs.freshness import evaluate_quote_freshness
from algotrading.risk.materiality import assess_omission_materiality
from algotrading.risk.history.revaluation import (
    CurrentPortfolioHistoricalRevaluationEngine,
    HistoricalRevaluationResult,
    historical_risk_position_eligible,
)
from algotrading.risk.instruments.terms import build_instrument_terms
from algotrading.risk.payoff.engine import certify_maximum_loss
from algotrading.risk.models import (
    Coverage,
    DataQuality,
    InstrumentTerms,
    NormalizedPosition,
    PortfolioSnapshot,
    SourceState,
)
from algotrading.risk.pricing.tian_service import IvGreeksContract, IvGreeksRequest

from .historical_cache import historical_revaluation_cache_key
from algotrading.risk.scenarios.engine import (
    DEFAULT_SCENARIOS,
    ScenarioDefinition,
    ScenarioEngine,
    omission_requires_fail_closed,
)
from algotrading.risk.scenarios.sector_rotation import (
    SECTOR_PROXY_SYMBOLS,
    build_sector_rotation_shocks,
)
from algotrading.risk.scenarios.etf_response import (
    ETF_RESPONSE_PROXY_SYMBOLS,
    build_etf_history_shocks,
)
from algotrading.risk.tail.historical import historical_var_es

from .hedge_models import (
    FactorsParameters,
    GreeksParameters,
    HedgeAssessmentRequest,
    HedgeCandidateQueryRequest,
    HedgeLeg,
    HedgeProposalRequest,
    MaximumLossParameters,
    ScenarioLossParameters,
    TailRiskParameters,
)
from .portfolio_models import PortfolioAnalysisResponse, PortfolioSnapshotResponse
from .portfolio_service import PortfolioApplicationService


_INDEX_ETF_SYMBOLS = ("SPY", "QQQ", "IWM", "DIA")
_FACTOR_ETF_SYMBOLS = tuple(
    symbol for symbol in FACTOR_PROXY_SYMBOLS
    if symbol not in {"SPY", "QQQ"}
)
_SECTOR_PROXIES = {
    "Communication Services": "XLC", "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP", "Energy": "XLE", "Financials": "XLF",
    "Health Care": "XLV", "Industrials": "XLI", "Information Technology": "XLK",
    "Technology": "XLK", "Materials": "XLB", "Real Estate": "XLRE",
    "Utilities": "XLU",
}
_SECTOR_ETF_SYMBOLS = tuple(dict.fromkeys(_SECTOR_PROXIES.values()))
_EQUITY_PROXY_SYMBOLS = tuple(dict.fromkeys((
    *_INDEX_ETF_SYMBOLS,
    *_FACTOR_ETF_SYMBOLS,
    *_SECTOR_ETF_SYMBOLS,
)))
_FACTOR_NAME_TO_ETF = {
    "Market": "SPY",
    "Technology/growth tilt": "QQQ",
    "Size": "SIZE",
    "Value": "VLUE",
    "Momentum": "MTUM",
    "Quality": "QUAL",
    "Low volatility": "USMV",
    "Long duration": "TLT",
    "Intermediate duration": "IEF",
    "Gold": "GLD",
    "Silver": "SLV",
    "Oil": "USO",
    "Natural gas": "UNG",
    "U.S. dollar": "UUP",
}
_PORTFOLIO_UNIVERSE_PRIORITY = {
    "index_etf": 0,
    "index_future": 1,
    "index_etf_option": 2,
    "volatility_future": 3,
    "volatility_option": 4,
    "factor_future": 5,
    "factor_etf": 6,
    "sector_etf": 7,
    "company_equity": 8,
    "company_option": 9,
    "other": 10,
}
_RANKING_WEIGHTS = {
    "primary_objective_improvement": 0.40,
    "non_target_residuals": 0.15,
    "scenario_robustness": 0.15,
    "entry_cost": 0.10,
    "carry_30d": 0.05,
    "liquidity": 0.10,
    "basis_risk": 0.05,
}


class HedgeLabService:
    """Read-only application service for the three Hedge Lab contracts."""

    def __init__(
        self,
        portfolio_service: PortfolioApplicationService,
        *,
        risk_history_service: Any,
        historical_revaluation_cache: dict[tuple[Any, ...], HistoricalRevaluationResult] | None = None,
        historical_revaluation_lock: RLock | None = None,
    ) -> None:
        self.portfolio_service = portfolio_service
        if risk_history_service is None:
            raise ValueError("Hedge Lab requires the configured Risk SQL option-history service")
        self.risk_history_service = risk_history_service
        self._history_cache_lock = historical_revaluation_lock or RLock()
        self._history_cache = (
            historical_revaluation_cache
            if historical_revaluation_cache is not None else {}
        )
        self._factor_returns_cache: dict[tuple[int, str], dict[str, dict[str, float]]] = {}
        self._returns_cache: dict[
            tuple[int, str, tuple[str, ...]], dict[str, dict[str, float]]
        ] = {}
        self._scenario_engine = ScenarioEngine(self.context)

    def _historical_revaluation_engine(self) -> CurrentPortfolioHistoricalRevaluationEngine:
        return CurrentPortfolioHistoricalRevaluationEngine(
            self.reference_client,
            self.context,
            risk_history_service=self.risk_history_service,
        )

    def _cached_revaluation(
        self,
        positions: Sequence[NormalizedPosition],
        *,
        as_of: datetime,
    ) -> HistoricalRevaluationResult:
        lookback = int(self.context.get("risk.history.lookback_sessions", 252))
        eligible = tuple(
            item for item in positions if historical_risk_position_eligible(item)
        )
        # Share historical results only when every normalized pricing input matches.
        key = historical_revaluation_cache_key(
            self.reference_client,
            eligible,
            as_of=as_of,
            lookback_sessions=lookback,
        )
        with self._history_cache_lock:
            cached = self._history_cache.get(key)
            if cached is not None:
                return cached
            result = self._historical_revaluation_engine().revalue(
                eligible, as_of=as_of, lookback_sessions=lookback
            )
            self._history_cache[key] = result
            while len(self._history_cache) > 24:
                self._history_cache.pop(next(iter(self._history_cache)))
        return result

    def _cached_factor_returns(self, as_of: datetime) -> dict[str, dict[str, float]]:
        key = (id(self.reference_client), as_of.isoformat())
        with self._history_cache_lock:
            cached = self._factor_returns_cache.get(key)
        if cached is not None:
            return cached
        result = self._load_returns(set(FACTOR_PROXY_SYMBOLS), as_of)
        with self._history_cache_lock:
            self._factor_returns_cache[key] = result
            while len(self._factor_returns_cache) > 4:
                self._factor_returns_cache.pop(next(iter(self._factor_returns_cache)))
        return result

    @property
    def context(self) -> Mapping[str, Any]:
        return self.portfolio_service.context

    @property
    def reference_client(self):
        # Tests and application extensions may replace this client after startup.
        return self.portfolio_service.reference_client

    def _quote_fresh(
        self,
        source_as_of: datetime,
        valuation_at: datetime,
        _legacy_maximum_age_seconds: float | None = None,
        *,
        exchange_timezone: str = "America/New_York",
        source_kind: str = "intraday_quote",
    ) -> bool:
        return evaluate_quote_freshness(
            source_as_of,
            as_of=valuation_at,
            context=self.context,
            exchange_timezone=exchange_timezone,
            source_kind=source_kind,
        ).valid_for_risk

    def assess(self, payload: HedgeAssessmentRequest) -> PortfolioAnalysisResponse:
        response, selected_snapshot, target = self._resolve_target(payload)
        data, warnings = self._assessment_data(payload, response, selected_snapshot, target)
        return self._analysis_response(
            response,
            "hedge_target_assessment_v1",
            data,
            warnings=warnings,
            insufficient=_nested_insufficient(data),
        )

    def candidates(self, payload: HedgeCandidateQueryRequest) -> PortfolioAnalysisResponse:
        response, selected_snapshot, target = self._resolve_target(payload)
        assessment, assessment_warnings = self._assessment_data(
            payload, response, selected_snapshot, target
        )
        warnings = list(assessment_warnings)
        candidates: list[dict[str, Any]] = []
        # Preserve missing target exposure instead of treating it as a measured zero.
        target_delta_value = _finite(
            assessment["current"].get("directional_delta_exposure_usd")
        )
        target_delta = target_delta_value if target_delta_value is not None else 0.0
        if payload.objective == "Greeks" and payload.evaluation_mode == "effects":
            preview, preview_warnings = self._greek_effect_preview_candidates(
                payload, response, selected_snapshot, assessment, target_delta
            )
            warnings.extend(preview_warnings)
            status = "usable" if preview and not warnings else (
                "usable_with_warnings" if preview else "insufficient_data"
            )
            data = {
                "analysis_only": True,
                "recommendation": False,
                "target": target,
                "objective": payload.objective,
                "items": preview[: payload.limit],
                "result_limit": payload.limit,
                "ranking": {
                    "methodology_version": "spy_cross_greek_effect_preview_v1",
                    "full_revaluation_applied": False,
                    "tie_breaker": "requested_dte_then_contract_template",
                },
                "data_quality": {
                    "status": status,
                    "reason": None if preview else "no_fully_priced_spy_option_available",
                },
                "warnings": _dedupe(warnings),
            }
            return self._analysis_response(
                response,
                "spy_cross_greek_effect_preview_v1",
                data,
                warnings=warnings,
                insufficient=not preview,
            )
        bounded_portfolio_scenario_scan = (
            payload.target.kind == "Portfolio" and payload.objective == "Scenario Loss"
        )
        if bounded_portfolio_scenario_scan:
            maximum = min(
                len(_INDEX_ETF_SYMBOLS),
                max(
                    1,
                    int(self.context.get("risk.hedge_lab.maximum_equity_symbols", 4)),
                ),
            )
            relevant = list(_INDEX_ETF_SYMBOLS[:maximum])
            universe_rows = [{"symbol": symbol} for symbol in relevant]
            prices, price_warnings = self._current_prices(relevant)
            warnings.extend(price_warnings)
        else:
            relevant = self._relevant_symbols(
                selected_snapshot,
                target_kind=payload.target.kind,
                objective=payload.objective,
                objective_parameters=payload.objective_parameters,
            )
            universe_rows, universe_warnings = self._universe_rows(relevant)
            warnings.extend(universe_warnings)
            prices, price_warnings = self._current_prices(relevant)
            warnings.extend(price_warnings)

        universe_symbols = {
            str(item.get("symbol") or "").strip().upper() for item in universe_rows
        }
        linear_candidates_requested = _linear_candidates_applicable(payload)
        linear_candidates = linear_candidates_requested and target_delta_value is not None
        if linear_candidates_requested and target_delta_value is None:
            warnings.append(
                "delta_dependent_candidate_discovery_unavailable:current_directional_delta_incomplete"
            )
        for symbol in relevant if linear_candidates else ():
            price = prices.get(symbol)
            if price is None:
                continue
            price_as_of = _datetime(price.get("as_of"))
            if price_as_of is None or not self._quote_fresh(
                price_as_of,
                response.as_of,
                float(self.context.get("risk.hedge_lab.maximum_quote_age_seconds", 900)),
                source_kind=str(price.get("source_kind") or "intraday_quote"),
            ):
                warnings.append(f"{symbol}:current_price_stale_or_undated")
                continue
            if universe_symbols and symbol not in universe_symbols:
                continue
            candidates.extend(
                self._underlying_candidates(
                    payload, assessment, selected_snapshot, symbol, price, target_delta
                )
            )

        if payload.objective == "Factors":
            candidates.extend(
                self._factor_spread_candidates(
                    payload, assessment, selected_snapshot, prices
                )
            )

        if payload.objective != "Factors" and not bounded_portfolio_scenario_scan:
            option_candidates, option_warnings = self._option_candidates(
                payload, response, selected_snapshot, assessment, target_delta
            )
            candidates.extend(option_candidates)
            warnings.extend(option_warnings)
        elif bounded_portfolio_scenario_scan:
            warnings.append(
                "broad_portfolio_option_chain_scan_deferred_use_scoped_target_or_manual_option"
            )

        if linear_candidates and not bounded_portfolio_scenario_scan:
            future_candidates, future_warnings = self._future_candidates(
                payload, assessment, target_delta, response.as_of
            )
            candidates.extend(future_candidates)
            warnings.extend(future_warnings)
        elif bounded_portfolio_scenario_scan:
            warnings.append(
                "broad_portfolio_futures_scan_deferred_use_scoped_target_or_manual_future"
            )

        # Deduplicate candidate identities before applying objective-specific eligibility.
        unique: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            unique.setdefault(str(candidate["candidate_id"]), candidate)
        eligible = [
            item for item in unique.values()
            if _candidate_applicable(payload, item)
        ]
        # The caller's result limit is also the bounded full-revaluation
        # budget. Preliminary fit is only used to choose which candidates earn
        # the expensive exact evaluation; returned fit is replaced below by
        # the full model result.
        if len(eligible) > payload.limit:
            eligible = sorted(
                eligible, key=lambda item: _candidate_sort_key(payload, item)
            )[:payload.limit]

        if payload.evaluation_mode == "effects":
            ordered = _diverse_candidates(payload, eligible, payload.limit)
            status = "usable" if ordered and not warnings else (
                "usable_with_warnings" if ordered else "insufficient_data"
            )
            data = {
                "analysis_only": True,
                "recommendation": False,
                "target": target,
                "objective": payload.objective,
                "items": ordered,
                "result_limit": payload.limit,
                "ranking": {
                    "methodology_version": "hedge_candidate_effect_preview_v1",
                    "full_revaluation_applied": False,
                    "tie_breaker": "preliminary_fit_then_candidate_id",
                },
                "data_quality": {
                    "status": status,
                    "reason": None if ordered else "no_fully_priced_candidate_available",
                },
                "warnings": _dedupe(warnings),
            }
            return self._analysis_response(
                response,
                "hedge_candidate_effect_preview_v1",
                data,
                warnings=warnings,
                insufficient=not ordered,
            )

        # Rank candidates from complete proposal revaluation and its disclosed audit facts.
        fully_evaluated: list[dict[str, Any]] = []

        def full_revaluation(
            candidate: dict[str, Any],
        ) -> tuple[dict[str, Any] | None, str | None]:
            try:
                proposal_payload = HedgeProposalRequest.model_validate(
                    {
                        **payload.model_dump(
                            mode="python", exclude={"limit", "evaluation_mode"}
                        ),
                        "legs": [
                            {
                                key: value
                                for key, value in leg.items()
                                if key in HedgeLeg.model_fields
                            }
                            for leg in candidate.get("legs") or []
                        ],
                    }
                )
                evaluation = self.evaluate(proposal_payload)
            except Exception as exc:
                return None, (
                    f"{candidate.get('candidate_id')}:full_revaluation_failed:"
                    f"{type(exc).__name__}"
                )
            evaluation_data = dict(evaluation.data)
            if not _primary_objective_usable(payload, evaluation_data):
                return None, (
                    f"{candidate.get('candidate_id')}:primary_objective_unavailable"
                )
            primary_improvement = _full_objective_improvement(
                payload, evaluation_data
            )
            if not math.isfinite(primary_improvement) or primary_improvement <= 1e-12:
                return None, (
                    f"{candidate.get('candidate_id')}:"
                    "primary_objective_not_strictly_improved"
                )
            candidate = dict(candidate)
            candidate["primary_objective_improvement"] = primary_improvement
            candidate["fit_score"] = round(
                _full_evaluation_fit(payload, candidate, evaluation_data), 2
            )
            candidate["target_after"] = _evaluated_target_after(
                payload, evaluation_data
            )
            candidate["scenario_pnl_after_usd"] = _worst_scenario_after(
                evaluation_data
            )
            candidate["full_evaluation"] = {
                "methodology_version": evaluation.methodology_version,
                "data_quality_status": evaluation.data_quality.status,
                "target": evaluation_data.get("target"),
                "portfolio": evaluation_data.get("portfolio"),
                "tail_risk": _compact_tail_risk(evaluation_data.get("tail_risk")),
                "factors": evaluation_data.get("factors"),
                "maximum_loss": evaluation_data.get("maximum_loss"),
                "scenarios": _compact_scenario_rows(evaluation_data.get("scenarios")),
                "margin": evaluation_data.get("margin"),
                "liquidity": evaluation_data.get("liquidity"),
                "residuals": evaluation_data.get("residuals"),
            }
            candidate["warnings"] = _dedupe(
                [
                    *(candidate.get("warnings") or []),
                    *evaluation.data_quality.warnings,
                ]
            )
            return candidate, None

        workers = min(
            len(eligible),
            max(
                1,
                int(
                    self.context.get(
                        "risk.hedge_lab.candidate_evaluation_workers", 3
                    )
                ),
            ),
        )
        if workers:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for candidate, warning in executor.map(full_revaluation, eligible):
                    if warning:
                        warnings.append(warning)
                    if candidate is not None:
                        fully_evaluated.append(candidate)
        # Remove dominated choices before reserving space for distinct hedge classes.
        pareto = _pareto_candidates(fully_evaluated)
        ordered = _diverse_candidates(payload, pareto, payload.limit)
        status = "usable" if ordered and not warnings else (
            "usable_with_warnings" if ordered else "insufficient_data"
        )
        data = {
            "analysis_only": True,
            "recommendation": False,
            "target": target,
            "objective": payload.objective,
            "items": ordered,
            "result_limit": payload.limit,
            "ranking": {
                "methodology_version": "hedge_candidate_full_revaluation_rank_v5",
                "weights": _RANKING_WEIGHTS,
                "pareto_filter_applied": True,
                "missing_liquidity_policy": "conservative_zero_score",
                "tie_breaker": "measured_fit_then_candidate_id",
            },
            "data_quality": {
                "status": status,
                "reason": None if ordered else "no_fully_priced_candidate_available",
            },
            "warnings": _dedupe(warnings),
        }
        return self._analysis_response(
            response,
            "hedge_candidate_full_revaluation_rank_v5",
            data,
            warnings=warnings,
            insufficient=not ordered,
        )

    def evaluate(self, payload: HedgeProposalRequest) -> PortfolioAnalysisResponse:
        response, selected_snapshot, target = self._resolve_target(payload)
        before_fingerprint = response.model_dump(mode="json")
        # Keep hidden legs in the audit while excluding them from proposed exposure.
        hidden = [item.model_dump(mode="json") for item in payload.legs if item.hidden]
        active = [item for item in payload.legs if not item.hidden]
        resolved_positions: list[NormalizedPosition] = []
        resolved_legs: list[dict[str, Any]] = []
        pricing_audit: list[dict[str, Any]] = []
        warnings: list[str] = []
        for leg in active:
            position, public_leg, audit, leg_warnings = self._resolve_leg(
                leg, response, payload.portfolio_snapshot_id
            )
            resolved_positions.append(position)
            resolved_legs.append(public_leg)
            pricing_audit.append(audit)
            warnings.extend(leg_warnings)
        audit_times = sorted(
            parsed
            for item in pricing_audit
            if (parsed := _datetime(item.get("quote_as_of"))) is not None
        )
        if len(audit_times) > 1 and (
            audit_times[-1] - audit_times[0]
        ).total_seconds() > float(
            self.context.get("risk.hedge_lab.maximum_mixed_quote_skew_seconds", 60)
        ):
            warnings.append("proposal_quote_timestamps_mixed")

        # Build proposed snapshots immutably so the loaded portfolio remains reusable.
        proposed_target = selected_snapshot.model_copy(
            update={"positions": tuple((*selected_snapshot.positions, *resolved_positions))}
        )
        _, full_snapshot = self.portfolio_service.require_snapshot(
            payload.account_ids, payload.portfolio_snapshot_id
        )
        proposed_portfolio = full_snapshot.model_copy(
            update={"positions": tuple((*full_snapshot.positions, *resolved_positions))}
        )

        target_before = _current_metrics(selected_snapshot.positions, selected_snapshot.portfolio_value_usd, self.context)
        target_after = _current_metrics(proposed_target.positions, proposed_target.portfolio_value_usd, self.context)
        portfolio_before = _current_metrics(full_snapshot.positions, full_snapshot.portfolio_value_usd, self.context)
        portfolio_after = _current_metrics(proposed_portfolio.positions, proposed_portfolio.portfolio_value_usd, self.context)
        greeks_before = _greek_metrics(full_snapshot.positions, full_snapshot.portfolio_value_usd, self.context)
        greeks_after = _greek_metrics(proposed_portfolio.positions, proposed_portfolio.portfolio_value_usd, self.context)

        near, far = _tail_confidences(payload)
        try:
            before_revaluation = self._cached_revaluation(full_snapshot.positions, as_of=response.as_of)
            target_before_revaluation = self._cached_revaluation(
                selected_snapshot.positions, as_of=response.as_of
            )
            if resolved_positions:
                additions_revaluation = self._cached_revaluation(
                    resolved_positions, as_of=response.as_of
                )
                after_revaluation = _merge_historical_revaluations(
                    before_revaluation, additions_revaluation
                )
                target_after_revaluation = _merge_historical_revaluations(
                    target_before_revaluation, additions_revaluation
                )
            else:
                after_revaluation = before_revaluation
                target_after_revaluation = target_before_revaluation
        except Exception:
            before_revaluation = None
            after_revaluation = None
            target_before_revaluation = None
            target_after_revaluation = None
        if (
            payload.evaluation_mode == "build"
            and isinstance(payload.parsed_objective_parameters(), FactorsParameters)
        ):
            return self._evaluate_factor_build(
                payload=payload,
                response=response,
                target=target,
                selected_snapshot=selected_snapshot,
                proposed_target=proposed_target,
                full_snapshot=full_snapshot,
                proposed_portfolio=proposed_portfolio,
                target_before=target_before,
                target_after=target_after,
                portfolio_before=portfolio_before,
                portfolio_after=portfolio_after,
                greeks_before=greeks_before,
                greeks_after=greeks_after,
                before_revaluation=before_revaluation,
                after_revaluation=after_revaluation,
                target_before_revaluation=target_before_revaluation,
                target_after_revaluation=target_after_revaluation,
                resolved_positions=resolved_positions,
                resolved_legs=resolved_legs,
                hidden=hidden,
                pricing_audit=pricing_audit,
                warnings=warnings,
                before_fingerprint=before_fingerprint,
            )
        if (
            payload.evaluation_mode == "build"
            and isinstance(payload.parsed_objective_parameters(), TailRiskParameters)
        ):
            return self._evaluate_tail_build(
                payload=payload,
                response=response,
                target=target,
                selected_snapshot=selected_snapshot,
                proposed_target=proposed_target,
                full_snapshot=full_snapshot,
                proposed_portfolio=proposed_portfolio,
                target_before=target_before,
                target_after=target_after,
                portfolio_before=portfolio_before,
                portfolio_after=portfolio_after,
                greeks_before=greeks_before,
                greeks_after=greeks_after,
                before_revaluation=before_revaluation,
                after_revaluation=after_revaluation,
                target_before_revaluation=target_before_revaluation,
                target_after_revaluation=target_after_revaluation,
                resolved_positions=resolved_positions,
                resolved_legs=resolved_legs,
                hidden=hidden,
                pricing_audit=pricing_audit,
                warnings=warnings,
                before_fingerprint=before_fingerprint,
            )
        # Compare before and after tail metrics over shared historical revaluation inputs.
        tail_before = self._tail_analysis(full_snapshot.positions, response.as_of, near, full_snapshot.portfolio_value_usd, revaluation=before_revaluation)
        tail_after = self._tail_analysis(proposed_portfolio.positions, response.as_of, near, proposed_portfolio.portfolio_value_usd, revaluation=after_revaluation)
        far_before = self._tail_analysis(full_snapshot.positions, response.as_of, far, full_snapshot.portfolio_value_usd, revaluation=before_revaluation)
        far_after = self._tail_analysis(proposed_portfolio.positions, response.as_of, far, proposed_portfolio.portfolio_value_usd, revaluation=after_revaluation)
        target_tail_before = self._tail_analysis(
            selected_snapshot.positions, response.as_of, near,
            selected_snapshot.portfolio_value_usd, revaluation=target_before_revaluation,
        )
        target_tail_after = self._tail_analysis(
            proposed_target.positions, response.as_of, near,
            proposed_target.portfolio_value_usd, revaluation=target_after_revaluation,
        )
        target_far_before = self._tail_analysis(
            selected_snapshot.positions, response.as_of, far,
            selected_snapshot.portfolio_value_usd, revaluation=target_before_revaluation,
        )
        target_far_after = self._tail_analysis(
            proposed_target.positions, response.as_of, far,
            proposed_target.portfolio_value_usd, revaluation=target_after_revaluation,
        )
        warnings.extend(tail_before.get("warnings") or [])
        warnings.extend(tail_after.get("warnings") or [])
        warnings.extend(target_tail_before.get("warnings") or [])
        warnings.extend(target_tail_after.get("warnings") or [])

        factor_returns: dict[str, dict[str, float]] | None = None
        if isinstance(payload.parsed_objective_parameters(), ScenarioLossParameters):
            factors_before = {"status": "not_applicable", "items": [], "warnings": []}
            factors_after = {"status": "not_applicable", "items": [], "warnings": []}
        else:
            try:
                factor_returns = self._cached_factor_returns(response.as_of)
            except Exception:
                factor_returns = None
            factors_before = self._factor_analysis(full_snapshot.positions, response.as_of, full_snapshot.portfolio_value_usd, revaluation=before_revaluation, factor_returns=factor_returns)
            factors_after = self._factor_analysis(proposed_portfolio.positions, response.as_of, proposed_portfolio.portfolio_value_usd, revaluation=after_revaluation, factor_returns=factor_returns)
        factor_change = _factor_changes(factors_before, factors_after)
        warnings.extend(factors_before.get("warnings") or [])
        warnings.extend(factors_after.get("warnings") or [])

        scenarios = []
        definitions = (
            *DEFAULT_SCENARIOS,
            ScenarioDefinition(
                scenario_id="validation_equity_flat",
                name="Broad market unchanged",
                broad_equity_shock=0.0,
            ),
            ScenarioDefinition(
                scenario_id="validation_equity_up_10",
                name="Broad market +10%",
                broad_equity_shock=0.10,
            ),
            ScenarioDefinition(
                scenario_id="joint_market_down_8_iv_up_12_rates_up_50",
                name="Joint market / volatility / rates stress",
                default_equity_shock=-0.08,
                iv_points=12.0,
                rate_basis_points=50.0,
                horizon_calendar_days=1.0,
            ),
        )
        # Apply the same validation scenarios to the unchanged and proposed portfolios.
        engine = self._scenario_engine
        for definition in definitions:
            before = engine.evaluate(full_snapshot, definition)
            after = engine.evaluate(proposed_portfolio, definition)
            target_scenario_before = engine.evaluate(selected_snapshot, definition)
            target_scenario_after = engine.evaluate(proposed_target, definition)
            scenarios.append({
                "scenario_id": definition.scenario_id,
                "name": definition.name,
                "definition": definition.model_dump(mode="json"),
                "before_pnl_usd": before.get("scenario_pnl_usd"),
                "after_pnl_usd": after.get("scenario_pnl_usd"),
                "change_usd": _difference(after.get("scenario_pnl_usd"), before.get("scenario_pnl_usd")),
                "before": before,
                "after": after,
                "target_before": target_scenario_before,
                "target_after": target_scenario_after,
            })

        # The validation chart is a one-dimensional response curve.  Do not
        # connect the heterogeneous stress-library scenarios (for example an
        # -8% equity / IV / rates joint shock and an -8% one-month shock) as if
        # they were adjacent points on the same curve.
        scenario_response = []
        for move_percent in (-10.0, -5.0, 0.0, 5.0, 10.0):
            definition = ScenarioDefinition(
                scenario_id=f"validation_equity_{move_percent:+g}",
                name=f"Broad market {move_percent:+g}%",
                broad_equity_shock=move_percent / 100.0,
            )
            before = engine.evaluate(full_snapshot, definition)
            after = engine.evaluate(proposed_portfolio, definition)
            target_scenario_before = engine.evaluate(selected_snapshot, definition)
            target_scenario_after = engine.evaluate(proposed_target, definition)
            scenario_response.append({
                "scenario_id": definition.scenario_id,
                "name": definition.name,
                "definition": definition.model_dump(mode="json"),
                "before_pnl_usd": before.get("scenario_pnl_usd"),
                "after_pnl_usd": after.get("scenario_pnl_usd"),
                "change_usd": _difference(
                    after.get("scenario_pnl_usd"), before.get("scenario_pnl_usd")
                ),
                "before": before,
                "after": after,
                "target_before": target_scenario_before,
                "target_after": target_scenario_after,
            })

        primary_objective = self._primary_objective(
            payload,
            response,
            selected_snapshot,
            proposed_target,
            target_before,
            target_after,
            warnings,
            before_revaluation=target_before_revaluation,
            after_revaluation=target_after_revaluation,
            factor_returns=factor_returns,
        )

        # Certify target and full-portfolio expiry payoff separately.
        maximum_loss = _terminal_payoff(proposed_target.positions)
        portfolio_maximum_loss = _terminal_payoff(proposed_portfolio.positions)
        maximum_loss["portfolio"] = portfolio_maximum_loss
        warnings.extend(maximum_loss.get("warnings") or [])
        warnings.extend(portfolio_maximum_loss.get("warnings") or [])
        var_after = tail_after.get("var_usd")
        entry_cost_usd = sum(
            float(item.signed_liquidation_value_usd)
            for item in resolved_positions
            if item.asset_type != "FUTURE"
            and item.signed_liquidation_value_usd is not None
        )
        proposal_tail_risk = {
            "near_confidence": near,
            "far_confidence": far,
            "var_before_usd": tail_before.get("var_usd"),
            "var_after_usd": var_after,
            "expected_shortfall_before_usd": far_before.get("expected_shortfall_usd"),
            "expected_shortfall_after_usd": far_after.get("expected_shortfall_usd"),
            "distribution_before": tail_before.get("distribution", []),
            "distribution_after": tail_after.get("distribution", []),
            "near_before_status": tail_before.get("status"),
            "near_after_status": tail_after.get("status"),
            "far_before_status": far_before.get("status"),
            "far_after_status": far_after.get("status"),
            "near_before": tail_before,
            "near_after": tail_after,
            "far_before": far_before,
            "far_after": far_after,
            "target_near_before": target_tail_before,
            "target_near_after": target_tail_after,
            "target_far_before": target_far_before,
            "target_far_after": target_far_after,
        }
        data = {
            "analysis_only": True,
            "var_after_usd": var_after,
            "entry_cost_usd": entry_cost_usd,
            "snapshot_as_of": response.as_of.isoformat(),
            "timestamps": {
                "before_snapshot_as_of": response.as_of.isoformat(),
                "after_snapshot_as_of": response.as_of.isoformat(),
                "quote_as_of_by_leg": {
                    str(item["client_leg_id"]): item.get("quote_as_of")
                    for item in pricing_audit
                },
            },
            "target": {
                "definition": target,
                "before": target_before,
                "after": target_after,
                "change": _numeric_changes(target_before, target_after),
                "objective": _compact_primary_objective(primary_objective),
            },
            "portfolio": {
                "before": portfolio_before,
                "after": portfolio_after,
                "change": _numeric_changes(portfolio_before, portfolio_after),
            },
            "greeks": {
                "before": greeks_before,
                "after": greeks_after,
                "change": _numeric_changes(greeks_before, greeks_after),
            },
            "factors": {
                "before": factors_before.get("items", []),
                "after": factors_after.get("items", []),
                "change": factor_change,
                "before_status": factors_before.get("status"),
                "after_status": factors_after.get("status"),
            },
            "tail_risk": _compact_tail_risk(
                proposal_tail_risk, include_distributions=True
            ),
            "scenarios": _compact_scenario_rows(scenarios),
            "scenario_response": _compact_scenario_rows(scenario_response),
            "maximum_loss": maximum_loss,
            "margin": {
                "status": "insufficient_data",
                "reason": "broker_and_strategy_specific_incremental_margin_model_unavailable",
            },
            "liquidity": {
                "status": "insufficient_data",
                "reason": "portfolio_liquidation_capacity_model_unavailable",
            },
            "residuals": _residuals(portfolio_before, portfolio_after, factors_before, factors_after),
            "active_legs": resolved_legs,
            "hidden_legs": hidden,
            "pricing_audit": pricing_audit,
            "warnings": _dedupe(warnings),
        }
        # Guard the read-only service contract against accidental snapshot mutation.
        if response.model_dump(mode="json") != before_fingerprint:
            raise RuntimeError("portfolio snapshot mutation detected")
        insufficient = not _primary_objective_usable(payload, data)
        return self._analysis_response(
            response,
            "hedge_proposal_evaluation_v1",
            data,
            warnings=warnings,
            insufficient=insufficient,
        )

    def _evaluate_factor_build(
        self,
        *,
        payload: HedgeProposalRequest,
        response: PortfolioSnapshotResponse,
        target: Mapping[str, Any],
        selected_snapshot: PortfolioSnapshot,
        proposed_target: PortfolioSnapshot,
        full_snapshot: PortfolioSnapshot,
        proposed_portfolio: PortfolioSnapshot,
        target_before: Mapping[str, Any],
        target_after: Mapping[str, Any],
        portfolio_before: Mapping[str, Any],
        portfolio_after: Mapping[str, Any],
        greeks_before: Mapping[str, Any],
        greeks_after: Mapping[str, Any],
        before_revaluation: HistoricalRevaluationResult | None,
        after_revaluation: HistoricalRevaluationResult | None,
        target_before_revaluation: HistoricalRevaluationResult | None,
        target_after_revaluation: HistoricalRevaluationResult | None,
        resolved_positions: Sequence[NormalizedPosition],
        resolved_legs: Sequence[Mapping[str, Any]],
        hidden: Sequence[Mapping[str, Any]],
        pricing_audit: Sequence[Mapping[str, Any]],
        warnings: list[str],
        before_fingerprint: Mapping[str, Any],
    ) -> PortfolioAnalysisResponse:
        """Return the exact Factor effect without precomputing Validate data."""

        try:
            factor_returns = self._cached_factor_returns(response.as_of)
        except Exception:
            factor_returns = None
        factors_before = self._factor_analysis(
            full_snapshot.positions,
            response.as_of,
            full_snapshot.portfolio_value_usd,
            revaluation=before_revaluation,
            factor_returns=factor_returns,
        )
        factors_after = self._factor_analysis(
            proposed_portfolio.positions,
            response.as_of,
            proposed_portfolio.portfolio_value_usd,
            revaluation=after_revaluation,
            factor_returns=factor_returns,
        )
        same_target = tuple(item.position_id for item in selected_snapshot.positions) == tuple(
            item.position_id for item in full_snapshot.positions
        )
        target_factors_before = factors_before if same_target else self._factor_analysis(
            selected_snapshot.positions,
            response.as_of,
            selected_snapshot.portfolio_value_usd,
            revaluation=target_before_revaluation,
            factor_returns=factor_returns,
        )
        target_factors_after = factors_after if same_target else self._factor_analysis(
            proposed_target.positions,
            response.as_of,
            proposed_target.portfolio_value_usd,
            revaluation=target_after_revaluation,
            factor_returns=factor_returns,
        )
        params = payload.parsed_objective_parameters()
        assert isinstance(params, FactorsParameters)
        primary_objective = {
            "objective": payload.objective,
            "factor": params.factor,
            "target_exposure": params.target_exposure,
            "before": target_factors_before,
            "after": target_factors_after,
            "change": _factor_changes(target_factors_before, target_factors_after),
        }
        warnings.extend(factors_before.get("warnings") or [])
        warnings.extend(factors_after.get("warnings") or [])
        entry_cost_usd = sum(
            float(item.signed_liquidation_value_usd)
            for item in resolved_positions
            if item.asset_type != "FUTURE"
            and item.signed_liquidation_value_usd is not None
        )
        # Build mode explicitly defers unrelated validation metrics.
        deferred = {
            "status": "not_evaluated",
            "reason": "deferred_until_full_validation",
        }
        data = {
            "analysis_only": True,
            "evaluation_mode": "build",
            "entry_cost_usd": entry_cost_usd,
            "snapshot_as_of": response.as_of.isoformat(),
            "target": {
                "definition": target,
                "before": target_before,
                "after": target_after,
                "change": _numeric_changes(target_before, target_after),
                "objective": _compact_primary_objective(primary_objective),
            },
            "portfolio": {
                "before": portfolio_before,
                "after": portfolio_after,
                "change": _numeric_changes(portfolio_before, portfolio_after),
            },
            "greeks": {
                "before": greeks_before,
                "after": greeks_after,
                "change": _numeric_changes(greeks_before, greeks_after),
            },
            "factors": {
                "before": factors_before.get("items", []),
                "after": factors_after.get("items", []),
                "change": _factor_changes(factors_before, factors_after),
                "before_status": factors_before.get("status"),
                "after_status": factors_after.get("status"),
            },
            "tail_risk": deferred,
            "scenarios": [],
            "scenario_response": [],
            "maximum_loss": deferred,
            "margin": deferred,
            "liquidity": deferred,
            "residuals": _residuals(
                portfolio_before, portfolio_after, factors_before, factors_after
            ),
            "active_legs": list(resolved_legs),
            "hidden_legs": list(hidden),
            "pricing_audit": list(pricing_audit),
            "warnings": _dedupe(warnings),
        }
        if response.model_dump(mode="json") != before_fingerprint:
            raise RuntimeError("portfolio snapshot mutation detected")
        insufficient = not _primary_objective_usable(payload, data)
        return self._analysis_response(
            response,
            "hedge_proposal_build_evaluation_v1",
            data,
            warnings=warnings,
            insufficient=insufficient,
        )

    def _evaluate_tail_build(
        self,
        *,
        payload: HedgeProposalRequest,
        response: PortfolioSnapshotResponse,
        target: Mapping[str, Any],
        selected_snapshot: PortfolioSnapshot,
        proposed_target: PortfolioSnapshot,
        full_snapshot: PortfolioSnapshot,
        proposed_portfolio: PortfolioSnapshot,
        target_before: Mapping[str, Any],
        target_after: Mapping[str, Any],
        portfolio_before: Mapping[str, Any],
        portfolio_after: Mapping[str, Any],
        greeks_before: Mapping[str, Any],
        greeks_after: Mapping[str, Any],
        before_revaluation: HistoricalRevaluationResult | None,
        after_revaluation: HistoricalRevaluationResult | None,
        target_before_revaluation: HistoricalRevaluationResult | None,
        target_after_revaluation: HistoricalRevaluationResult | None,
        resolved_positions: Sequence[NormalizedPosition],
        resolved_legs: Sequence[Mapping[str, Any]],
        hidden: Sequence[Mapping[str, Any]],
        pricing_audit: Sequence[Mapping[str, Any]],
        warnings: list[str],
        before_fingerprint: Mapping[str, Any],
    ) -> PortfolioAnalysisResponse:
        """Evaluate the selected tail objective without precomputing Validate data."""

        params = payload.parsed_objective_parameters()
        assert isinstance(params, TailRiskParameters)
        near, far = params.near_confidence, params.far_confidence
        portfolio_near_before = self._tail_analysis(
            full_snapshot.positions,
            response.as_of,
            near,
            full_snapshot.portfolio_value_usd,
            revaluation=before_revaluation,
        )
        portfolio_near_after = self._tail_analysis(
            proposed_portfolio.positions,
            response.as_of,
            near,
            proposed_portfolio.portfolio_value_usd,
            revaluation=after_revaluation,
        )
        portfolio_far_before = (
            portfolio_near_before
            if far == near
            else self._tail_analysis(
                full_snapshot.positions,
                response.as_of,
                far,
                full_snapshot.portfolio_value_usd,
                revaluation=before_revaluation,
            )
        )
        portfolio_far_after = (
            portfolio_near_after
            if far == near
            else self._tail_analysis(
                proposed_portfolio.positions,
                response.as_of,
                far,
                proposed_portfolio.portfolio_value_usd,
                revaluation=after_revaluation,
            )
        )
        same_target = {
            item.position_id for item in selected_snapshot.positions
        } == {item.position_id for item in full_snapshot.positions}
        if same_target:
            target_near_before = portfolio_near_before
            target_near_after = portfolio_near_after
            target_far_before = portfolio_far_before
            target_far_after = portfolio_far_after
        else:
            target_near_before = self._tail_analysis(
                selected_snapshot.positions,
                response.as_of,
                near,
                selected_snapshot.portfolio_value_usd,
                revaluation=target_before_revaluation,
            )
            target_near_after = self._tail_analysis(
                proposed_target.positions,
                response.as_of,
                near,
                proposed_target.portfolio_value_usd,
                revaluation=target_after_revaluation,
            )
            target_far_before = (
                target_near_before
                if far == near
                else self._tail_analysis(
                    selected_snapshot.positions,
                    response.as_of,
                    far,
                    selected_snapshot.portfolio_value_usd,
                    revaluation=target_before_revaluation,
                )
            )
            target_far_after = (
                target_near_after
                if far == near
                else self._tail_analysis(
                    proposed_target.positions,
                    response.as_of,
                    far,
                    proposed_target.portfolio_value_usd,
                    revaluation=target_after_revaluation,
                )
            )
        tail_results = (
            portfolio_near_before,
            portfolio_near_after,
            portfolio_far_before,
            portfolio_far_after,
            target_near_before,
            target_near_after,
            target_far_before,
            target_far_after,
        )
        for result in tail_results:
            warnings.extend(result.get("warnings") or [])
        primary_objective = {
            "objective": payload.objective,
            "hedge_region": params.hedge_region,
            "before": {"near": target_near_before, "far": target_far_before},
            "after": {"near": target_near_after, "far": target_far_after},
        }
        proposal_tail_risk = {
            "near_confidence": near,
            "far_confidence": far,
            "var_before_usd": portfolio_near_before.get("var_usd"),
            "var_after_usd": portfolio_near_after.get("var_usd"),
            "expected_shortfall_before_usd": portfolio_far_before.get(
                "expected_shortfall_usd"
            ),
            "expected_shortfall_after_usd": portfolio_far_after.get(
                "expected_shortfall_usd"
            ),
            "near_before": portfolio_near_before,
            "near_after": portfolio_near_after,
            "far_before": portfolio_far_before,
            "far_after": portfolio_far_after,
            "target_near_before": target_near_before,
            "target_near_after": target_near_after,
            "target_far_before": target_far_before,
            "target_far_after": target_far_after,
        }
        entry_cost_usd = sum(
            float(item.signed_liquidation_value_usd)
            for item in resolved_positions
            if item.asset_type != "FUTURE"
            and item.signed_liquidation_value_usd is not None
        )
        deferred = {
            "status": "not_evaluated",
            "reason": "deferred_until_full_validation",
        }
        empty_factors = {"status": "not_evaluated", "items": []}
        data = {
            "analysis_only": True,
            "evaluation_mode": "build",
            "var_after_usd": portfolio_near_after.get("var_usd"),
            "entry_cost_usd": entry_cost_usd,
            "snapshot_as_of": response.as_of.isoformat(),
            "target": {
                "definition": target,
                "before": target_before,
                "after": target_after,
                "change": _numeric_changes(target_before, target_after),
                "objective": _compact_primary_objective(primary_objective),
            },
            "portfolio": {
                "before": portfolio_before,
                "after": portfolio_after,
                "change": _numeric_changes(portfolio_before, portfolio_after),
            },
            "greeks": {
                "before": greeks_before,
                "after": greeks_after,
                "change": _numeric_changes(greeks_before, greeks_after),
            },
            "factors": deferred,
            "tail_risk": _compact_tail_risk(proposal_tail_risk),
            "scenarios": [],
            "scenario_response": [],
            "maximum_loss": deferred,
            "margin": deferred,
            "liquidity": deferred,
            "residuals": _residuals(
                portfolio_before, portfolio_after, empty_factors, empty_factors
            ),
            "active_legs": list(resolved_legs),
            "hidden_legs": list(hidden),
            "pricing_audit": list(pricing_audit),
            "warnings": _dedupe(warnings),
        }
        if response.model_dump(mode="json") != before_fingerprint:
            raise RuntimeError("portfolio snapshot mutation detected")
        insufficient = not _primary_objective_usable(payload, data)
        return self._analysis_response(
            response,
            "hedge_proposal_build_evaluation_v1",
            data,
            warnings=warnings,
            insufficient=insufficient,
        )

    def _primary_objective(
        self,
        payload: HedgeProposalRequest,
        response: PortfolioSnapshotResponse,
        before_snapshot: PortfolioSnapshot,
        after_snapshot: PortfolioSnapshot,
        before_metrics: Mapping[str, Any],
        after_metrics: Mapping[str, Any],
        warnings: list[str],
        *,
        before_revaluation: HistoricalRevaluationResult | None = None,
        after_revaluation: HistoricalRevaluationResult | None = None,
        factor_returns: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        params = payload.parsed_objective_parameters()
        if isinstance(params, ScenarioLossParameters):
            definition = self._scenario_definition(
                params, before_snapshot, response.as_of, warnings
            )
            engine = self._scenario_engine
            before = engine.evaluate(before_snapshot, definition)
            after = engine.evaluate(after_snapshot, definition)
            return {
                "objective": payload.objective,
                "before": before,
                "after": after,
                "change_usd": _difference(
                    after.get("scenario_pnl_usd"), before.get("scenario_pnl_usd")
                ),
            }
        if isinstance(params, TailRiskParameters):
            near_before = self._tail_analysis(
                before_snapshot.positions, response.as_of, params.near_confidence,
                before_snapshot.portfolio_value_usd,
                revaluation=before_revaluation,
            )
            near_after = self._tail_analysis(
                after_snapshot.positions, response.as_of, params.near_confidence,
                after_snapshot.portfolio_value_usd,
                revaluation=after_revaluation,
            )
            far_before = self._tail_analysis(
                before_snapshot.positions, response.as_of, params.far_confidence,
                before_snapshot.portfolio_value_usd,
                revaluation=before_revaluation,
            )
            far_after = self._tail_analysis(
                after_snapshot.positions, response.as_of, params.far_confidence,
                after_snapshot.portfolio_value_usd,
                revaluation=after_revaluation,
            )
            return {
                "objective": payload.objective,
                "hedge_region": params.hedge_region,
                "before": {"near": near_before, "far": far_before},
                "after": {"near": near_after, "far": far_after},
            }
        if isinstance(params, GreeksParameters):
            return {
                "objective": payload.objective,
                "selected_greeks": params.selected_greeks,
                "targets": params.targets.model_dump(mode="json"),
                "before": dict(before_metrics),
                "after": dict(after_metrics),
                "change": _numeric_changes(before_metrics, after_metrics),
            }
        if isinstance(params, FactorsParameters):
            before = self._factor_analysis(
                before_snapshot.positions,
                response.as_of,
                before_snapshot.portfolio_value_usd,
                revaluation=before_revaluation,
                factor_returns=factor_returns,
            )
            after = self._factor_analysis(
                after_snapshot.positions,
                response.as_of,
                after_snapshot.portfolio_value_usd,
                revaluation=after_revaluation,
                factor_returns=factor_returns,
            )
            return {
                "objective": payload.objective,
                "factor": params.factor,
                "target_exposure": params.target_exposure,
                "before": before,
                "after": after,
                "change": _factor_changes(before, after),
            }
        return {
            "objective": payload.objective,
            "desired_loss_floor_usd": (
                params.desired_loss_floor_usd
                if isinstance(params, MaximumLossParameters)
                else None
            ),
            "before": _terminal_payoff(before_snapshot.positions),
            "after": _terminal_payoff(after_snapshot.positions),
        }

    def _resolve_target(
        self, payload: HedgeAssessmentRequest
    ) -> tuple[PortfolioSnapshotResponse, PortfolioSnapshot, dict[str, Any]]:
        response, snapshot = self.portfolio_service.require_snapshot(
            payload.account_ids, payload.portfolio_snapshot_id
        )
        # Resolve target membership only within the requested coherent snapshot.
        positions_by_id = {item.position_id: item for item in snapshot.positions}
        unknown = sorted(set(payload.target.position_ids) - set(positions_by_id))
        if unknown:
            raise ValueError("unknown or out-of-scope position_ids: " + ", ".join(unknown))
        selected = tuple(
            positions_by_id[position_id] for position_id in sorted(payload.target.position_ids)
        )
        if not selected:
            raise ValueError("target resolves to no positions")
        selected_economic = {item.economic_position_id for item in selected}
        unknown_economic = sorted(
            set(payload.target.economic_position_ids) - selected_economic
        )
        if unknown_economic:
            raise ValueError(
                "unknown or out-of-target economic_position_ids: "
                + ", ".join(unknown_economic)
            )
        value = sum(
            item.signed_market_value_usd or 0.0
            for item in selected
            if item.asset_type != "FUTURE"
        )
        scoped = snapshot.model_copy(
            update={"positions": selected, "portfolio_value_usd": value}
        )
        target = payload.target.model_dump(mode="json")
        target["position_ids"] = [item.position_id for item in selected]
        target["economic_position_ids"] = sorted(selected_economic)
        return response, scoped, target

    def _assessment_data(
        self,
        payload: HedgeAssessmentRequest,
        response: PortfolioSnapshotResponse,
        snapshot: PortfolioSnapshot,
        target: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        warnings: list[str] = []
        current = _current_metrics(
            snapshot.positions, snapshot.portfolio_value_usd, self.context
        )
        warnings.extend(current.pop("warnings", []))
        params = payload.parsed_objective_parameters()
        scenario_definition = self._scenario_definition(params, snapshot, response.as_of, warnings)
        scenario = self._scenario_engine.evaluate(snapshot, scenario_definition)
        warnings.extend(scenario.get("warnings") or [])
        near, far = _tail_confidences(payload)
        try:
            shared_revaluation = self._cached_revaluation(snapshot.positions, as_of=response.as_of)
        except Exception:
            shared_revaluation = None
        near_tail = self._tail_analysis(
            snapshot.positions,
            response.as_of,
            near,
            snapshot.portfolio_value_usd,
            revaluation=shared_revaluation,
        )
        far_tail = (
            self._tail_analysis(
                snapshot.positions,
                response.as_of,
                far,
                snapshot.portfolio_value_usd,
                revaluation=shared_revaluation,
            )
            if far != near else near_tail
        )
        warnings.extend(near_tail.get("warnings") or [])
        warnings.extend(far_tail.get("warnings") or [])
        if isinstance(params, ScenarioLossParameters):
            factors = {"status": "not_applicable", "items": [], "warnings": []}
        else:
            try:
                factor_returns = self._cached_factor_returns(response.as_of)
            except Exception:
                factor_returns = None
            factors = self._factor_analysis(
                snapshot.positions,
                response.as_of,
                snapshot.portfolio_value_usd,
                revaluation=shared_revaluation,
                factor_returns=factor_returns,
            )
        warnings.extend(factors.get("warnings") or [])
        payoff = _terminal_payoff(snapshot.positions)
        warnings.extend(payoff.get("warnings") or [])
        data = {
            "analysis_only": True,
            "target": target,
            "objective": payload.objective,
            "objective_parameters": payload.objective_parameters,
            "current": current,
            "scenario": scenario,
            "var_usd": near_tail.get("var_usd"),
            "expected_shortfall_usd": far_tail.get("expected_shortfall_usd"),
            "distribution": near_tail.get("distribution", []),
            # Preserve the established single-confidence fields as the near-tail
            # view while exposing both independently evaluated confidence levels.
            "tail_risk": {**near_tail, "near": near_tail, "far": far_tail},
            "factors": factors,
            "diversification": _diversification(snapshot.positions, factors),
            "maximum_loss": payoff,
            "warnings": _dedupe(warnings),
        }
        return data, warnings

    def _scenario_definition(
        self,
        params: object,
        snapshot: PortfolioSnapshot,
        as_of: datetime,
        warnings: list[str],
    ) -> ScenarioDefinition:
        if not isinstance(params, ScenarioLossParameters):
            return ScenarioDefinition(
                scenario_id="hedge_lab_default",
                name="Hedge Lab current-state scenario",
                default_equity_shock=0.0,
            )
        shocks: dict[str, float] = {}
        diagnostics: dict[str, Any] = {}
        etf_diagnostics: dict[str, Any] = {}
        etf_symbols = {
            item.exact_symbol for item in snapshot.positions if item.asset_type == "ETF"
        }
        if (
            params.sector_rotation_sector
            and params.sector_rotation_move is not None
        ) or etf_symbols:
            symbols = {
                item.underlying_symbol or item.exact_symbol for item in snapshot.positions
                if item.underlying_symbol or item.exact_symbol
            }
            try:
                returns = self._load_returns(
                    symbols
                    | set(SECTOR_PROXY_SYMBOLS.values())
                    | set(ETF_RESPONSE_PROXY_SYMBOLS),
                    as_of,
                )
                if (
                    params.sector_rotation_sector
                    and params.sector_rotation_move is not None
                ):
                    shocks, diagnostics = build_sector_rotation_shocks(
                        snapshot.positions,
                        returns,
                        selected_sector=params.sector_rotation_sector,
                        sector_move=params.sector_rotation_move,
                        default_equity_shock=params.default_equity_shock,
                        explicit_shocks={},
                        minimum_observations=int(
                            self.context.get("risk.scenarios.sector_rotation_minimum_observations", 126)
                        ),
                        maximum_observations=252,
                    )
                shocks, etf_diagnostics = build_etf_history_shocks(
                    snapshot.positions,
                    returns,
                    broad_equity_shock=params.default_equity_shock,
                    sector_shocks={},
                    explicit_shocks=shocks,
                    minimum_observations=int(
                        self.context.get("risk.scenarios.etf_response_minimum_observations", 126)
                    ),
                    maximum_observations=252,
                    ridge_alpha=float(
                        self.context.get("risk.scenarios.etf_response_ridge_alpha", 1e-3)
                    ),
                    minimum_r_squared=float(
                        self.context.get("risk.scenarios.etf_response_minimum_r_squared", 0.20)
                    ),
                    maximum_absolute_shock=float(
                        self.context.get("risk.scenarios.maximum_absolute_equity_shock", 0.50)
                    ),
                )
                if etf_diagnostics.get("unmodeled_symbols"):
                    warnings.append("partial_etf_response_history")
            except Exception as exc:
                warnings.append(f"scenario_response_history_unavailable:{type(exc).__name__}")
        return ScenarioDefinition(
            scenario_id="hedge_lab_custom",
            name="Hedge Lab target scenario",
            underlying_shocks=shocks,
            default_equity_shock=params.default_equity_shock,
            iv_points=params.iv_points,
            rate_basis_points=params.rate_basis_points,
            horizon_calendar_days=params.horizon_calendar_days,
            sector_rotation_sector=params.sector_rotation_sector,
            sector_rotation_move=params.sector_rotation_move,
            sector_rotation_diagnostics=diagnostics,
            etf_response_diagnostics=etf_diagnostics,
        )

    def _load_returns(
        self, symbols: set[str], as_of: datetime
    ) -> dict[str, dict[str, float]]:
        key = (id(self.reference_client), as_of.isoformat(), tuple(sorted(symbols)))
        with self._history_cache_lock:
            cached = self._returns_cache.get(key)
        if cached is not None:
            return cached
        lookback = int(self.context.get("risk.history.lookback_sessions", 252))
        returns, _diagnostics, _plan = self._historical_revaluation_engine().load_returns(
            sorted(symbols), as_of=as_of, lookback_sessions=lookback
        )
        with self._history_cache_lock:
            self._returns_cache[key] = returns
            while len(self._returns_cache) > 16:
                self._returns_cache.pop(next(iter(self._returns_cache)))
        return returns

    def _tail_analysis(
        self,
        positions: Sequence[NormalizedPosition],
        as_of: datetime,
        confidence: float,
        portfolio_value_usd: float | None = None,
        *,
        revaluation: HistoricalRevaluationResult | None = None,
    ) -> dict[str, Any]:
        minimum = int(self.context.get("risk.history.minimum_observations", 126))
        lookback = int(self.context.get("risk.history.lookback_sessions", 252))
        try:
            revaluation = revaluation or self._historical_revaluation_engine().revalue(
                positions, as_of=as_of, lookback_sessions=lookback
            )
        except Exception as exc:
            result = _insufficient_tail(
                confidence, f"daily_history_request_failed:{type(exc).__name__}"
            )
            result["failure_causes"] = [
                {
                    "code": "daily_history_request_failed",
                    "error_type": type(exc).__name__,
                }
            ]
            return result
        eligible = [item for item in positions if historical_risk_position_eligible(item)]
        by_id = {item.position_id: item for item in eligible}
        modeled_ids = sorted(set(revaluation.scenario_pnl) & set(by_id))
        material_missing = sorted(
            position_id
            for position_id in revaluation.missing_position_ids
            if position_id in by_id
            and omission_requires_fail_closed(by_id[position_id])
        )
        omission = assess_omission_materiality(
            revaluation.missing_position_ids,
            positions,
            portfolio_value_usd=portfolio_value_usd,
            context=self.context,
        )
        if omission.may_exclude:
            material_missing = []
        coverage = 100.0 * len(modeled_ids) / len(eligible) if eligible else 100.0
        # Align every modeled position on common sessions before calculating portfolio tails.
        dates = (
            sorted(
                set.intersection(
                    *(set(revaluation.scenario_pnl[item_id]) for item_id in modeled_ids)
                )
            )
            if modeled_ids
            else []
        )
        dates = dates[-lookback:]
        required_coverage = 100.0 * float(
            self.context.get("risk.tail_risk.minimum_gross_delta_coverage", 0.90)
        )
        failure_causes: list[dict[str, Any]] = []
        if len(dates) < minimum:
            failure_causes.append({
                "code": "complete_scenarios_below_minimum",
                "actual": len(dates), "required": minimum,
            })
        if coverage < required_coverage and not omission.may_exclude:
            failure_causes.append({
                "code": "modeled_coverage_below_minimum",
                "actual_percent": coverage,
                "required_percent": required_coverage,
            })
        if material_missing:
            failure_causes.append({
                "code": "material_positions_unmodeled",
                "position_ids": material_missing,
            })
        position_coverage = [
            item.model_dump(mode="json") for item in revaluation.position_coverage
        ]
        failed_request_positions = sorted(
            item.position_id
            for item in revaluation.position_coverage
            if item.failed_remote_ranges
        )
        if failed_request_positions and failure_causes:
            failure_causes.append({
                "code": "upstream_history_range_failed",
                "position_ids": failed_request_positions,
                "failed_ranges": [
                    row
                    for item in position_coverage
                    if item["position_id"] in failed_request_positions
                    for row in item.get("failed_remote_ranges") or ()
                ],
            })
        missing_positions = [
            {
                "position_id": position_id,
                "exact_symbol": by_id[position_id].canonical_occ_symbol
                or by_id[position_id].exact_symbol,
            }
            for position_id in revaluation.missing_position_ids
            if position_id in by_id
        ]
        if failure_causes:
            result = _insufficient_tail(
                confidence,
                "material_nonlinear_or_unknown_exposure_unmodeled"
                if material_missing
                else "complete_portfolio_scenarios_insufficient"
                if len(dates) < minimum
                else "modeled_coverage_insufficient",
            )
            result.update({
                "failure_causes": failure_causes,
                "observations": len(dates),
                "minimum_observations": minimum,
                "observations_label": "Complete portfolio P&L scenarios",
                "coverage": {
                    "eligible_positions": len(eligible),
                    "modeled_positions": len(modeled_ids),
                    "percent": coverage,
                    "count_percent": coverage,
                    "weight_basis": "full_revaluation_eligible_economic_unit_count",
                    "denominator_usd": None,
                    "modeled_usd": None,
                    "missing_position_ids": list(revaluation.missing_position_ids),
                },
                "missing_position_ids": list(revaluation.missing_position_ids),
                "missing_positions": missing_positions,
                "material_missing_position_ids": material_missing,
                "proxy_mappings": list(revaluation.proxy_mappings),
                "position_coverage": position_coverage,
                "scenario_provenance": revaluation.scenario_provenance,
                "warnings": [
                    "tail_history_insufficient",
                    *([omission.warning] if omission.warning else []),
                    *revaluation.warnings,
                ],
            })
            return result
        unit_ids = modeled_ids
        matrix = [[
            revaluation.scenario_pnl[unit_id][day]
            for unit_id in unit_ids
        ] for day in dates]
        result = historical_var_es(
            matrix,
            confidence=confidence,
            half_life_sessions=float(self.context.get("risk.tail_risk.half_life_sessions", 63)),
            unit_ids=unit_ids,
            dates=dates,
        )
        for contribution in result.get("contributions", []):
            unit_id = str(contribution.get("unit_id") or "")
            position = by_id.get(unit_id)
            if position is None:
                continue
            contribution.update(
                {
                    "position_ids": [position.position_id],
                    "economic_position_id": position.economic_position_id,
                    "symbol": position.exact_symbol,
                }
            )
        result.update({
            "status": (
                "usable_with_warnings"
                if revaluation.warnings or omission.warning
                else "usable"
            ),
            "input_method": "current_portfolio_filtered_historical_revaluation_v5",
            "calculation_method": "current_state_filtered_historical_shock_revaluation",
            "unit_methods": revaluation.scenario_methods,
            "failure_causes": [],
            "minimum_observations": minimum,
            "observations_label": "Complete portfolio P&L scenarios",
            "coverage": {
                "eligible_positions": len(eligible),
                "modeled_positions": len(modeled_ids),
                "percent": coverage,
                "count_percent": coverage,
                "weight_basis": "full_revaluation_eligible_economic_unit_count",
                "denominator_usd": None,
                "modeled_usd": None,
                "missing_position_ids": list(revaluation.missing_position_ids),
            },
            "missing_position_ids": list(revaluation.missing_position_ids),
            "missing_positions": missing_positions,
            "proxy_mappings": list(revaluation.proxy_mappings),
            "position_coverage": position_coverage,
            "scenario_provenance": revaluation.scenario_provenance,
            "distribution": [
                {
                    "date": day,
                    "pnl_usd": sum(row),
                    "loss_usd": max(0.0, -sum(row)),
                    **next(
                        (
                            {
                                "weight": weight["weight"],
                                "tail_weight": weight["tail_weight"],
                            }
                            for weight in result.get("observation_weights", [])
                            if weight.get("date") == day
                        ),
                        {},
                    ),
                }
                for day, row in zip(dates, matrix)
            ],
            "warnings": [
                *([omission.warning] if omission.warning else []),
                *revaluation.warnings,
            ],
        })
        return result

    def _factor_analysis(
        self,
        positions: Sequence[NormalizedPosition],
        as_of: datetime,
        portfolio_value_usd: float | None = None,
        *,
        revaluation: HistoricalRevaluationResult | None = None,
        factor_returns: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        try:
            returns = factor_returns or self._load_returns(set(FACTOR_PROXY_SYMBOLS), as_of)
            missing = [symbol for symbol in FACTOR_PROXY_SYMBOLS if symbol not in returns]
            if missing:
                raise ValueError("factor_proxy_history_unavailable:" + ",".join(missing))
            revaluation = revaluation or self._historical_revaluation_engine().revalue(
                positions, as_of=as_of
            )
        except Exception as exc:
            return {
                "status": "insufficient_data",
                "reason": f"factor_history_request_failed:{type(exc).__name__}",
                "items": [],
                "warnings": ["factor_history_unavailable"],
            }
        modeled = revaluation.scenario_pnl
        by_id = {
            item.position_id: item
            for item in positions
            if historical_risk_position_eligible(item)
        }
        material_missing = sorted(
            position_id
            for position_id in revaluation.missing_position_ids
            if position_id in by_id
            and omission_requires_fail_closed(by_id[position_id])
        )
        omission = assess_omission_materiality(
            revaluation.missing_position_ids,
            positions,
            portfolio_value_usd=portfolio_value_usd,
            context=self.context,
        )
        if omission.may_exclude:
            material_missing = []
        if material_missing:
            return {
                "status": "insufficient_data",
                "reason": "material_nonlinear_or_unknown_exposure_unmodeled",
                "failure_causes": [{
                    "code": "material_positions_unmodeled",
                    "position_ids": material_missing,
                }],
                "items": [],
                "coverage": {
                    "eligible_positions": len(by_id),
                    "modeled_positions": len(modeled),
                    "count_percent": 100.0 * len(modeled) / len(by_id) if by_id else 100.0,
                    "percent": None,
                    "weight_basis": "complete_required_economic_unit_coverage",
                    "missing_position_ids": list(revaluation.missing_position_ids),
                },
                "warnings": [
                    "material_nonlinear_or_unknown_exposure_unmodeled",
                    *revaluation.warnings,
                ],
                "position_coverage": [
                    item.model_dump(mode="json")
                    for item in revaluation.position_coverage
                ],
                "scenario_provenance": revaluation.scenario_provenance,
            }
        proxy = {symbol: returns[symbol] for symbol in FACTOR_PROXY_SYMBOLS}
        aligned = [*proxy.values(), *modeled.values()]
        # Align factor proxies and dollar P&L before fitting portfolio exposures.
        dates = sorted(set.intersection(*(set(series) for series in aligned))) if aligned else []
        dates = dates[-int(self.context.get("risk.history.lookback_sessions", 252)):]
        target = [
            sum(series[day] for series in modeled.values()) for day in dates
        ] if modeled else []
        proxy_values = {
            symbol: [proxy[symbol][day] for day in dates] for symbol in FACTOR_PROXY_SYMBOLS
        }
        unit_values = {
            unit_id: [series[day] for day in dates]
            for unit_id, series in modeled.items()
        }
        try:
            current_delta = _finite(
                _current_metrics(
                    positions, portfolio_value_usd, self.context
                ).get("directional_delta_exposure_usd")
            )
            result = ridge_factor_exposure(
                target,
                proxy_values,
                unit_pnl_by_id=unit_values,
                ridge_alpha=float(self.context.get("risk.factors.ridge_alpha", 1e-3)),
                minimum_observations=int(self.context.get("risk.history.minimum_observations", 126)),
            )
            if current_delta not in (None, 0):
                for factor in result.get("factors") or []:
                    coefficient = _finite(
                        factor.get("coefficient_usd_per_unit_factor_return")
                    )
                    factor["beta"] = (
                        coefficient / current_delta
                        if coefficient is not None else None
                    )
        except (ValueError, KeyError) as exc:
            return {
                "status": "insufficient_data",
                "reason": f"factor_history_unusable:{type(exc).__name__}",
                "items": [],
                "warnings": ["factor_history_unusable"],
            }
        result["items"] = result.pop("factors", [])
        result["proxy_symbols"] = list(FACTOR_PROXY_SYMBOLS)
        result["underlying_proxy_symbols"] = []
        result["unit_methods"] = revaluation.scenario_methods
        result["position_coverage"] = [
            item.model_dump(mode="json") for item in revaluation.position_coverage
        ]
        result["scenario_provenance"] = revaluation.scenario_provenance
        result["failure_causes"] = []
        result["coverage"] = {
            "eligible_positions": len(by_id),
            "modeled_positions": len(modeled),
            "percent": (
                100.0 * len(modeled) / len(by_id) if by_id else 100.0
            ),
            "count_percent": (
                100.0 * len(modeled) / len(by_id) if by_id else 100.0
            ),
            "weight_basis": "full_revaluation_eligible_economic_unit_count",
            "denominator_usd": None,
            "modeled_usd": None,
            "missing_position_ids": list(revaluation.missing_position_ids),
            "unavailable_denominator_position_ids": list(revaluation.missing_position_ids),
        }
        result["warnings"] = [
            *(result.get("warnings") or []),
            *([omission.warning] if omission.warning else []),
            *revaluation.warnings,
        ]
        if omission.warning and result.get("status") == "usable":
            result["status"] = "usable_with_warnings"
        return result

    def _relevant_symbols(
        self,
        snapshot: PortfolioSnapshot,
        *,
        target_kind: str = "Portfolio",
        objective: str = "Scenario Loss",
        objective_parameters: Mapping[str, Any] | None = None,
    ) -> list[str]:
        target_weights: dict[str, float] = {}
        for item in snapshot.positions:
            symbol = item.underlying_symbol or item.exact_symbol
            if item.asset_type == "CASH" or not symbol:
                continue
            target_weights[symbol] = target_weights.get(symbol, 0.0) + abs(
                float(item.signed_market_value_usd or 0.0)
            )
        target_symbols = sorted(
            target_weights,
            key=lambda symbol: (-target_weights[symbol], symbol),
        )
        sectors = sorted({item.sector for item in snapshot.positions if item.sector})
        sector_symbols = [_SECTOR_PROXIES[item] for item in sectors if item in _SECTOR_PROXIES]
        configured = [
            str(item).strip().upper()
            for item in self.context.get("risk.hedge_lab.equity_proxy_symbols", _EQUITY_PROXY_SYMBOLS)
            if str(item).strip()
        ]
        maximum = min(25, max(1, int(self.context.get("risk.hedge_lab.maximum_equity_symbols", 12))))
        index_symbols = [symbol for symbol in configured if symbol in _INDEX_ETF_SYMBOLS]
        configured_factors = [symbol for symbol in configured if symbol in _FACTOR_ETF_SYMBOLS]
        configured_sectors = [symbol for symbol in configured if symbol in _SECTOR_ETF_SYMBOLS]
        other_configured = [
            symbol for symbol in configured
            if symbol not in {*_INDEX_ETF_SYMBOLS, *_FACTOR_ETF_SYMBOLS, *_SECTOR_ETF_SYMBOLS}
        ]
        factor_symbols: list[str] = []
        if objective == "Factors":
            factor_name = str((objective_parameters or {}).get("factor") or "Market")
            factor_symbol = _FACTOR_NAME_TO_ETF.get(factor_name)
            if factor_symbol and factor_symbol in configured_factors:
                factor_symbols.append(factor_symbol)
        # Portfolio scans reserve capacity by hedge class. Security and other
        # scoped scans still start with the target name before using proxies.
        ordered = (
            [*index_symbols, *factor_symbols, *sector_symbols, *target_symbols, *other_configured]
            if target_kind == "Portfolio"
            else [*target_symbols, *index_symbols, *factor_symbols, *sector_symbols, *other_configured]
        )
        return list(dict.fromkeys(ordered))[:maximum]

    def _universe_rows(self, symbols: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
        try:
            method = getattr(self.reference_client, "universe_symbols", None)
            if not callable(method):
                method = self.reference_client.security_classifications
            return list(method(symbols)), []
        except Exception as exc:
            return [], [f"reference_universe_unavailable:{type(exc).__name__}"]

    def _current_prices(
        self,
        symbols: Sequence[str],
        *,
        instrument_type: Literal["EQUITY", "FUTURE"] = "EQUITY",
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        prices: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        if not symbols:
            return prices, warnings
        quotes = getattr(self.reference_client, "quotes", None)
        if instrument_type == "EQUITY" and callable(quotes):
            try:
                for row in quotes(symbols):
                    symbol = str(row.get("symbol") or "").strip().upper()
                    mark = _positive(row.get("mark"))
                    if symbol and mark is not None:
                        prices[symbol] = {
                            "mark": mark,
                            "as_of": row.get("quote_time") or row.get("as_of"),
                            "source": str(row.get("source") or "RefData quotes"),
                            "source_kind": str(
                                row.get("source_kind") or "intraday_quote"
                            ),
                        }
                missing = [symbol for symbol in symbols if symbol not in prices]
                if not missing:
                    return prices, warnings
                symbols = missing
            except Exception as exc:
                warnings.append(f"bulk_current_prices_unavailable:{type(exc).__name__}")
        current_price = (
            getattr(self.reference_client, "current_futures_price", None)
            if instrument_type == "FUTURE"
            else getattr(self.reference_client, "current_price", None)
        )
        if not callable(current_price) and instrument_type == "FUTURE":
            current_price = getattr(self.reference_client, "current_price", None)
        if callable(current_price):
            workers = min(
                4,
                max(1, int(self.context.get("risk.hedge_lab.price_query_workers", 4))),
                len(symbols),
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(current_price, symbol): symbol for symbol in symbols}
                completed: dict[str, Any] = {}
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        completed[symbol] = future.result()
                    except Exception as exc:
                        completed[symbol] = exc
            for symbol in symbols:
                try:
                    payload = completed[symbol]
                    if isinstance(payload, Exception):
                        raise payload
                    data = payload.get("data") or {}
                    price = _positive(data.get("price"))
                    if price is None:
                        warnings.append(f"{symbol}:current_price_missing")
                        continue
                    prices[symbol] = {
                        "mark": price,
                        "as_of": (payload.get("meta") or {}).get("as_of"),
                        "source": "RefData current-price",
                        "source_kind": str(
                            (payload.get("meta") or {}).get("source_kind")
                            or data.get("source_kind")
                            or "intraday_quote"
                        ),
                    }
                except Exception as exc:
                    warnings.append(f"{symbol}:current_price_unavailable:{type(exc).__name__}")
            return prices, warnings
        try:
            for row in self.reference_client.quotes(symbols):
                symbol = str(row.get("symbol") or "").strip().upper()
                mark = _positive(row.get("mark"))
                if symbol and mark is not None:
                    prices[symbol] = {
                        "mark": mark,
                        "as_of": row.get("quote_time") or row.get("as_of"),
                        "source": str(row.get("source") or "RefData quotes"),
                        "source_kind": str(
                            row.get("source_kind") or "intraday_quote"
                        ),
                    }
        except Exception as exc:
            warnings.append(f"current_prices_unavailable:{type(exc).__name__}")
        return prices, warnings

    def _underlying_candidates(
        self,
        payload: HedgeAssessmentRequest,
        assessment: Mapping[str, Any],
        snapshot: PortfolioSnapshot,
        symbol: str,
        price_row: Mapping[str, Any],
        target_delta: float,
    ) -> list[dict[str, Any]]:
        mark = _positive(price_row.get("mark"))
        if mark is None or target_delta == 0:
            return []
        side = "SELL" if target_delta > 0 else "BUY"
        quantities = _bounded_hedge_quantities(
            abs(target_delta),
            mark,
            maximum_quantity=int(
                self.context.get("risk.hedge_lab.maximum_candidate_quantity", 100)
            ),
            maximum_overhedge_fraction=float(
                self.context.get("risk.hedge_lab.maximum_overhedge_fraction", 0.25)
            ),
        )
        basis = _basis_risk(symbol, snapshot)
        output: list[dict[str, Any]] = []
        for quantity in quantities:
            signed = quantity if side == "BUY" else -quantity
            leg = {
                "client_leg_id": _stable_hash(payload.portfolio_snapshot_id, symbol, side, quantity)[:32],
                "instrument_type": "EQUITY", "side": side, "quantity": quantity,
                "symbol": symbol, "canonical_occ_symbol": None, "expiration_at": None,
                "put_call": None, "strike": None, "mark": mark, "multiplier": 1.0,
                "spot": mark, "delta": 1.0, "gamma": 0.0,
                "vega_per_1pct": 0.0, "theta_per_day": 0.0, "rho_per_1pct": 0.0,
                "source": str(price_row.get("source") or "RefData current-price"),
            }
            effect = signed * mark
            improvement = _primary_improvement(
                payload, assessment, target_delta, effect, [leg]
            )
            fit = _fit_score(improvement, basis, liquidity_score=0.5, cost_score=0.5, carry_score=0.0)
            output.append(_candidate_item(
                payload,
                name=f"{symbol} {'short' if side == 'SELL' else 'long'} × {int(quantity)}",
                description=f"Research-only {symbol} directional offset using a current RefData mark.",
                legs=[leg], fit=fit, liquidity="Insufficient data",
                target_after=_target_after(payload, assessment, delta_effect=effect),
                scenario_after=_scenario_after(assessment, effect), entry_cost=signed * mark,
                carry=None, basis=basis,
                rationale=[
                    "Quantity is one member of the bounded integer objective search.",
                    "No order is created or recommended.",
                ],
                residuals={
                    "directional_delta_effect_usd": effect,
                    **_candidate_greek_effects([leg]),
                },
                data_quality={
                    "status": "usable_with_warnings", "price_source": price_row.get("source"),
                    "price_as_of": price_row.get("as_of"), "liquidity_status": "insufficient_data",
                    "carry_status": "insufficient_data",
                },
                warnings=["liquidity_metrics_unavailable", "carry_inputs_unavailable"],
            ))
        return output

    def _factor_spread_candidates(
        self,
        payload: HedgeCandidateQueryRequest,
        assessment: Mapping[str, Any],
        snapshot: PortfolioSnapshot,
        prices: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        params = payload.parsed_objective_parameters()
        if not isinstance(params, FactorsParameters):
            return []
        spread = {
            "Technology/growth tilt": ("QQQ", "SPY"),
            "Size": ("SIZE", "SPY"),
            "Value": ("VLUE", "SPY"),
            "Momentum": ("MTUM", "SPY"),
            "Quality": ("QUAL", "SPY"),
            "Low volatility": ("USMV", "SPY"),
            "Long duration": ("TLT", "SHY"),
            "Intermediate duration": ("IEF", "SHY"),
        }.get(params.factor)
        if spread is None:
            return []
        long_symbol, short_symbol = spread
        long_mark = _positive((prices.get(long_symbol) or {}).get("mark"))
        short_mark = _positive((prices.get(short_symbol) or {}).get("mark"))
        if long_mark is None or short_mark is None:
            return []
        current_row = next(
            (
                item
                for item in (assessment.get("factors") or {}).get("items") or []
                if item.get("name") == params.factor
            ),
            {},
        )
        coefficient = _finite(
            current_row.get("coefficient_usd_per_unit_factor_return")
        )
        desired = float(params.target_exposure)
        direction = -1.0 if (coefficient or 0.0) > desired else 1.0
        maximum_quantity = int(
            self.context.get("risk.hedge_lab.maximum_candidate_quantity", 100)
        )
        output: list[dict[str, Any]] = []
        for long_quantity in range(1, maximum_quantity + 1):
            short_quantity = max(1, round(long_quantity * long_mark / short_mark))
            if short_quantity > maximum_quantity:
                continue
            legs = []
            for symbol, mark, quantity, factor_sign in (
                (long_symbol, long_mark, long_quantity, 1.0),
                (short_symbol, short_mark, short_quantity, -1.0),
            ):
                signed_factor = direction * factor_sign
                side = "BUY" if signed_factor > 0 else "SELL"
                legs.append(
                    {
                        "client_leg_id": _stable_hash(
                            payload.portfolio_snapshot_id, params.factor, symbol, side, quantity
                        )[:32],
                        "instrument_type": "EQUITY", "side": side,
                        "quantity": float(quantity), "symbol": symbol,
                        "canonical_occ_symbol": None, "expiration_at": None,
                        "put_call": None, "strike": None, "mark": mark,
                        "multiplier": 1.0, "spot": mark, "delta": 1.0,
                        "gamma": 0.0, "vega_per_1pct": 0.0,
                        "theta_per_day": 0.0, "rho_per_1pct": 0.0,
                        "source": "RefData current-price",
                    }
                )
            signed_notionals = [
                (1.0 if leg["side"] == "BUY" else -1.0)
                * float(leg["quantity"]) * float(leg["mark"])
                for leg in legs
            ]
            residual = sum(signed_notionals)
            entry = sum(signed_notionals)
            output.append(_candidate_item(
                payload,
                name=f"{params.factor} factor spread × {long_quantity}",
                description=(
                    f"Paired {long_symbol} minus {short_symbol} hedge with explicit "
                    "rounding residual."
                ),
                legs=legs,
                fit=50.0,
                liquidity="Insufficient data",
                target_after="Requires full dollar-factor regression",
                scenario_after=None,
                entry_cost=entry,
                carry=None,
                basis=abs(residual) / max(sum(abs(value) for value in signed_notionals), 1.0),
                rationale=[
                    "Matches the published spread factor definition.",
                    "Both integer legs are one member of the full objective search.",
                    "Research output only; not a recommendation.",
                ],
                residuals={
                    "market_factor_residual_usd": residual,
                    "factor_spread": f"{long_symbol} - {short_symbol}",
                },
                data_quality={
                    "status": "usable_with_warnings",
                    "liquidity_status": "insufficient_data",
                    "margin_status": "not_applicable_equities",
                },
                warnings=["liquidity_metrics_unavailable"],
            ))
        return output

    def _option_candidates(
        self,
        payload: HedgeCandidateQueryRequest,
        response: PortfolioSnapshotResponse,
        snapshot: PortfolioSnapshot,
        assessment: Mapping[str, Any],
        target_delta: float,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not response.data.capabilities.get("available", {}).get("options_chain", False):
            return [], ["option_chain_capability_unavailable"]
        direct = [
            item.underlying_symbol or item.exact_symbol
            for item in snapshot.positions
            if item.asset_type != "CASH"
            and (item.underlying_symbol or item.exact_symbol)
        ]
        relevant = self._relevant_symbols(
            snapshot,
            target_kind=payload.target.kind,
            objective=payload.objective,
            objective_parameters=payload.objective_parameters,
        )
        # A portfolio scan should start with broad/sector proxies selected by
        # _relevant_symbols, not the first six held names in snapshot order.
        underlyings = relevant if payload.target.kind == "Portfolio" else [*direct, *relevant]
        dominant = _dominant_symbol(snapshot.positions)
        if (
            payload.target.kind != "Portfolio"
            and dominant
            and dominant not in underlyings
            and len(dominant) <= 16
        ):
            underlyings.insert(0, dominant)
        maximum_symbols = int(
            self.context.get(
                "risk.hedge_lab.maximum_preview_option_symbols"
                if payload.evaluation_mode == "effects"
                else "risk.hedge_lab.maximum_option_symbols",
                2 if payload.evaluation_mode == "effects" else 4,
            )
        )
        underlyings = list(dict.fromkeys(underlyings))[:max(1, maximum_symbols)]
        output: list[dict[str, Any]] = []
        warnings: list[str] = []
        if not underlyings:
            return output, warnings
        def load_symbol(symbol: str) -> tuple[list[dict[str, Any]], list[str]]:
            try:
                chain = self.portfolio_service.hedge_chain(
                    payload.account_ids,
                    portfolio_snapshot_id=payload.portfolio_snapshot_id,
                    symbol=symbol,
                    contract_type="ALL",
                    strike_count=20 if payload.evaluation_mode == "effects" else 40,
                    from_date=response.as_of.date().isoformat(),
                    to_date=(response.as_of.date() + timedelta(days=35)).isoformat(),
                    include_underlying_quote=True,
                )
                rows = self._normalized_chain_rows(chain, symbol, response.as_of)
                if not rows:
                    return [], [f"{symbol}:no_usable_option_contracts"]
                return (
                    self._structures_from_chain(
                        payload, assessment, snapshot, symbol, rows, target_delta
                    ),
                    [],
                )
            except Exception as exc:
                return [], [f"{symbol}:option_chain_unavailable:{type(exc).__name__}"]

        workers = min(
            len(underlyings),
            max(1, int(self.context.get("risk.hedge_lab.option_chain_workers", 4))),
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for candidates, symbol_warnings in executor.map(load_symbol, underlyings):
                output.extend(candidates)
                warnings.extend(symbol_warnings)
        return output, warnings

    def _greek_effect_preview_candidates(
        self,
        payload: HedgeCandidateQueryRequest,
        response: PortfolioSnapshotResponse,
        snapshot: PortfolioSnapshot,
        assessment: Mapping[str, Any],
        target_delta: float,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Return the twelve fixed SPY contracts used by the Cross Greek table."""

        if not response.data.capabilities.get("available", {}).get("options_chain", False):
            return [], ["option_chain_capability_unavailable"]
        try:
            calendar = self.portfolio_service.hedge_expirations(
                payload.account_ids,
                portfolio_snapshot_id=payload.portfolio_snapshot_id,
                symbol="SPY",
                from_date=response.as_of.date().isoformat(),
                to_date=(response.as_of.date() + timedelta(days=95)).isoformat(),
                include_underlying_quote=False,
            )
            available = {
                expiration
                for item in (calendar.get("data") or {}).get("expirations") or []
                if isinstance(item, Mapping)
                and (expiration := _date_value(item.get("expiration"))) is not None
            }
            selected_dates: list[date] = []
            for requested_dte in (7, 30, 90):
                if not available:
                    break
                expiry = min(
                    available,
                    key=lambda value: (
                        abs((value - response.as_of.date()).days - requested_dte),
                        value,
                    ),
                )
                selected_dates.append(expiry)
                available.remove(expiry)
            if not selected_dates:
                return [], ["SPY:no_option_expirations"]

            def load_expiration(expiration: date) -> tuple[dict[str, Any] | None, str | None]:
                try:
                    return self.portfolio_service.hedge_chain(
                        payload.account_ids,
                        portfolio_snapshot_id=payload.portfolio_snapshot_id,
                        symbol="SPY",
                        contract_type="ALL",
                        strike_count=12,
                        from_date=expiration.isoformat(),
                        to_date=expiration.isoformat(),
                        include_underlying_quote=True,
                    ), None
                except Exception as exc:
                    return None, f"SPY:{expiration.isoformat()}:option_chain_unavailable:{type(exc).__name__}"

            chains: list[dict[str, Any]] = []
            warnings: list[str] = []
            with ThreadPoolExecutor(max_workers=len(selected_dates)) as executor:
                for chain, warning in executor.map(load_expiration, selected_dates):
                    if chain is not None:
                        chains.append(chain)
                    if warning is not None:
                        warnings.append(warning)
            combined_contracts = [
                dict(item)
                for chain in chains
                for item in (chain.get("data") or {}).get("contracts") or []
                if isinstance(item, Mapping)
            ]
            underlying = next(
                (
                    (chain.get("data") or {}).get("underlying_quote")
                    for chain in chains
                    if (chain.get("data") or {}).get("underlying_quote")
                ),
                None,
            )
            chain = {"data": {
                "symbol": "SPY",
                "underlying_quote": underlying,
                "contracts": combined_contracts,
            }}
            rows = self._normalized_chain_rows(chain, "SPY", response.as_of)
        except Exception as exc:
            return [], [f"SPY:option_chain_unavailable:{type(exc).__name__}"]
        if not rows:
            return [], [*warnings, "SPY:no_usable_option_contracts"]

        by_expiry: dict[date, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_expiry[row["expiration_at"].date()].append(row)
        available_expiries = set(by_expiry)
        selected_expiries: dict[int, date] = {}
        for requested_dte in (7, 30, 90):
            if not available_expiries:
                break
            expiry = min(
                available_expiries,
                key=lambda value: (
                    abs((value - response.as_of.date()).days - requested_dte),
                    value,
                ),
            )
            selected_expiries[requested_dte] = expiry
            available_expiries.remove(expiry)

        output: list[dict[str, Any]] = []
        templates = (
            ("CALL ATM", "CALL", "atm"),
            ("PUT ATM", "PUT", "atm"),
            ("PUT OTM 25 delta", "PUT", "otm25"),
            ("CALL OTM 25 delta", "CALL", "otm25"),
        )
        for requested_dte, expiry in selected_expiries.items():
            expiry_rows = by_expiry[expiry]
            spot = float(expiry_rows[0]["spot"])
            for template_name, put_call, selection in templates:
                side = [row for row in expiry_rows if row["put_call"] == put_call]
                if not side:
                    continue
                if selection == "atm":
                    contract = min(
                        side,
                        key=lambda row: (
                            abs(float(row["strike"]) - spot),
                            row["canonical_occ_symbol"],
                        ),
                    )
                else:
                    otm = [
                        row for row in side
                        if (put_call == "PUT" and float(row["strike"]) < spot)
                        or (put_call == "CALL" and float(row["strike"]) > spot)
                    ]
                    pool = otm or side
                    contract = min(
                        pool,
                        key=lambda row: (
                            abs(abs(float(row["delta"])) - 0.25),
                            abs(float(row["strike"]) - spot),
                            row["canonical_occ_symbol"],
                        ),
                    )
                leg = self._candidate_option_leg(
                    payload, "BUY", 1.0, "SPY", contract, len(output)
                )
                effects = _candidate_greek_effects([leg])
                delta_effect = effects["option_delta_exposure_effect_usd"]
                candidate = _candidate_item(
                    payload,
                    name=f"SPY {template_name}",
                    description=(
                        f"One quoted SPY {put_call.lower()} nearest {requested_dte} DTE; "
                        + ("strike nearest spot." if selection == "atm" else "OTM contract nearest 25 delta.")
                    ),
                    legs=[leg],
                    fit=0.0,
                    liquidity=_option_liquidity([("BUY", contract)])[0],
                    target_after=_target_after(
                        payload, assessment, delta_effect=delta_effect, legs=[leg]
                    ),
                    scenario_after=_scenario_after(assessment, delta_effect),
                    entry_cost=float(leg["mark"]) * float(leg["multiplier"]),
                    carry=None,
                    basis=_basis_risk("SPY", snapshot),
                    rationale=[
                        "Fixed Cross Greek comparison contract.",
                        "One-contract effect; no order is created or recommended.",
                    ],
                    residuals={
                        "directional_delta_effect_usd": delta_effect,
                        **effects,
                    },
                    data_quality={
                        "status": "usable_with_warnings",
                        "price_source": "RefData option chain",
                        "greek_source": "Risk Tian",
                    },
                    warnings=[],
                )
                candidate["preview_dte"] = requested_dte
                candidate["preview_actual_dte"] = (
                    expiry - response.as_of.date()
                ).days
                candidate["preview_contract"] = template_name
                output.append(candidate)
        return output, warnings

    def _normalized_chain_rows(
        self, chain: Mapping[str, Any], symbol: str, as_of: datetime
    ) -> list[dict[str, Any]]:
        # The portfolio snapshot remains the comparison anchor, but a hedge
        # chain is fetched after that snapshot. Price the hypothetical leg at
        # the later chain/request time so a perfectly current quote is not
        # rejected merely for being newer than the immutable snapshot.
        valuation_at = max(
            as_of,
            _datetime(self.portfolio_service._now()) or as_of,
        )
        data = chain.get("data") or {}
        chain_source_as_of = _datetime(chain.get("source_as_of"))
        underlying = data.get("underlying_quote") or {}
        spot = next(
            (
                value for key in ("mark", "last", "close", "delayedPrice")
                if (value := _positive(underlying.get(key))) is not None
            ),
            None,
        )
        if spot is None:
            price_rows, _ = self._current_prices([symbol])
            price_row = price_rows.get(symbol) or {}
            price_as_of = _datetime(price_row.get("as_of"))
            if price_as_of is not None and self._quote_fresh(
                price_as_of,
                valuation_at,
                float(self.context.get("risk.hedge_lab.maximum_quote_age_seconds", 900)),
                source_kind=str(price_row.get("source_kind") or "intraday_quote"),
            ):
                spot = _positive(price_row.get("mark"))
        if spot is None:
            return []
        raw_contracts = [dict(item) for item in data.get("contracts") or []]
        exact_symbols = [
            str(item.get("canonical_occ_symbol") or "").strip().upper()
            for item in raw_contracts
            if item.get("canonical_occ_symbol")
        ]
        # Reuse attached authoritative terms before requesting missing contract facts.
        term_rows = {
            str(item.get("canonical_occ_symbol") or "").strip().upper(): dict(item["instrument_terms"])
            for item in raw_contracts
            if item.get("canonical_occ_symbol") and isinstance(item.get("instrument_terms"), Mapping)
        }
        missing_term_symbols = [symbol for symbol in exact_symbols if symbol not in term_rows]
        if missing_term_symbols:
            try:
                term_payload = self.reference_client.option_contract_terms(missing_term_symbols)
                term_rows.update({
                    str(item.get("canonical_occ_symbol") or "").strip().upper(): dict(item)
                    for item in term_payload.get("data") or []
                    if item.get("canonical_occ_symbol")
                })
            except Exception:
                pass
        maximum_age = float(self.context.get("risk.hedge_lab.maximum_quote_age_seconds", 900))
        candidates: list[tuple[dict[str, Any], IvGreeksContract]] = []
        for index, raw in enumerate(raw_contracts):
            canonical = str(raw.get("canonical_occ_symbol") or "").strip().upper()
            if not canonical:
                continue
            bid, ask, mark = _finite(raw.get("bid")), _finite(raw.get("ask")), _positive(raw.get("mark"))
            if bid is not None and ask is not None:
                if bid < 0 or ask <= 0 or bid > ask:
                    continue
                mark = (bid + ask) / 2.0
            if mark is None:
                continue
            terms = build_instrument_terms(
                canonical,
                broker_type="OPTION",
                option_terms=term_rows.get(canonical),
                context=self.context,
            )
            multiplier = terms.price_multiplier
            if (
                multiplier is None
                or terms.expiration_at is None
                or terms.exercise_style not in {"american", "european"}
            ):
                continue
            quote_time = _datetime(raw.get("quote_time")) or chain_source_as_of
            if quote_time is None or not self._quote_fresh(
                quote_time,
                valuation_at,
                maximum_age,
                source_kind=str(raw.get("source_kind") or "intraday_quote"),
            ):
                continue
            expiration_at = terms.expiration_at
            if expiration_at <= valuation_at:
                continue
            rate = self._risk_free_rate(valuation_at, expiration_at)
            if rate is None:
                continue
            source_iv = _positive(raw.get("provider_iv_raw"))
            if source_iv and source_iv > 5:
                source_iv /= 100.0
            try:
                contract = IvGreeksContract(
                    request_id=f"candidate:{canonical}:{index}",
                    canonical_occ_symbol=canonical,
                    underlying_symbol=symbol,
                    put_call=str(raw.get("put_call")),
                    exercise_type={
                        "european": "EUROPEAN",
                        "american": "AMERICAN",
                    }[terms.exercise_style],
                    strike=float(raw.get("strike")),
                    expiration_at=expiration_at,
                    spot=spot,
                    bid=bid,
                    ask=ask,
                    mark=mark,
                    quote_time=quote_time,
                    source_iv=source_iv,
                    risk_free_rate=rate,
                    dividend_yield=float(self.context.get("risk.dividends.default_continuous_yield", 0.0)),
                )
            except Exception:
                continue
            candidates.append(({
                **dict(raw),
                "resolved_multiplier": multiplier,
                "instrument_terms": terms.model_dump(mode="json"),
            }, contract))
        if not candidates:
            return []
        solved_items: list[Any] = []
        try:
            # IvGreeksRequest deliberately caps a request at 500 contracts. A
            # normal three-month, all-strikes chain can be substantially larger.
            # Preserve chain order while pricing it in accepted request sizes so
            # the Cross Greek preview and option picker do not silently empty.
            for offset in range(0, len(candidates), 500):
                batch = candidates[offset : offset + 500]
                solved = self.portfolio_service.tian_service.solve(
                    IvGreeksRequest(
                        valuation_at=valuation_at,
                        contracts=[item[1] for item in batch],
                    )
                )
                solved_items.extend(solved.data)
        except Exception:
            return []
        output = []
        for (raw, contract), greek in zip(candidates, solved_items):
            values = greek.model_dump(mode="json") if hasattr(greek, "model_dump") else vars(greek)
            if values.get("status") not in {"ok", "usable_with_warnings"}:
                continue
            if any(_finite(values.get(name)) is None for name in ("delta", "gamma", "vega_per_1pct", "theta_per_day", "rho_per_1pct")):
                continue
            mark = _positive(values.get("target_price")) or _positive(contract.mark)
            if mark is None:
                continue
            output.append({
                **raw,
                "canonical_occ_symbol": contract.canonical_occ_symbol,
                "put_call": contract.put_call,
                "strike": contract.strike,
                "expiration_at": contract.expiration_at,
                "mark": mark,
                "multiplier": float(raw["resolved_multiplier"]),
                "instrument_terms": raw.get("instrument_terms"),
                "spot": contract.spot,
                "quote_time": contract.quote_time,
                "delta": float(values["delta"]),
                "gamma": float(values["gamma"]),
                "implied_volatility": _finite(values.get("implied_volatility")),
                "vega_per_1pct": float(values["vega_per_1pct"]),
                "theta_per_day": float(values["theta_per_day"]),
                "rho_per_1pct": float(values["rho_per_1pct"]),
                "open_interest": _finite(raw.get("open_interest")),
                "volume": _finite(raw.get("volume")),
                "bid": _finite(raw.get("bid")),
                "ask": _finite(raw.get("ask")),
                "source": "RefData option chain + Risk Tian Greeks",
            })
        return sorted(output, key=lambda item: (
            item["expiration_at"], item["put_call"], item["strike"], item["canonical_occ_symbol"]
        ))

    def _structures_from_chain(
        self,
        payload: HedgeCandidateQueryRequest,
        assessment: Mapping[str, Any],
        snapshot: PortfolioSnapshot,
        symbol: str,
        rows: list[dict[str, Any]],
        target_delta: float,
    ) -> list[dict[str, Any]]:
        # Delta-neutral tail/maximum-loss targets still need downside convexity.
        # Choose the protective option side from the target's directional exposure.
        protect_with_put = target_delta >= 0
        put_call = "PUT" if protect_with_put else "CALL"
        side_rows = [item for item in rows if item["put_call"] == put_call]
        if not side_rows:
            return []
        expiry_groups: dict[date, list[dict[str, Any]]] = defaultdict(list)
        for item in side_rows:
            expiry_groups[item["expiration_at"].date()].append(item)
        expiries = sorted(expiry_groups)
        spot = float(side_rows[0]["spot"])
        structures: list[tuple[str, str, list[tuple[str, dict[str, Any]]], list[str]]] = []
        nearest = min(side_rows, key=lambda item: (
            abs((item["expiration_at"].date() - payload_date(payload, snapshot)).days - 30),
            abs(item["strike"] - spot), item["canonical_occ_symbol"],
        ))
        structures.append((
            f"{symbol} {put_call.lower()}",
            f"Single {put_call.lower()} with Tian-normalized Greeks.",
            [("BUY", nearest)],
            ["Direct convex protection for the target's directional exposure."],
        ))
        same_expiry = sorted(expiry_groups[nearest["expiration_at"].date()], key=lambda item: item["strike"])
        if put_call == "PUT":
            lower = [item for item in same_expiry if item["strike"] < nearest["strike"]]
            if lower:
                structures.append((
                    f"{symbol} put spread",
                    "Long put spread with a defined entry price and terminal width.",
                    [("BUY", nearest), ("SELL", lower[-1])],
                    ["Reduces debit and carry relative to a single long put."],
                ))
        else:
            upper = [item for item in same_expiry if item["strike"] > nearest["strike"]]
            if upper:
                structures.append((
                    f"{symbol} call spread",
                    "Long call spread with a defined entry price and terminal width.",
                    [("BUY", nearest), ("SELL", upper[0])],
                    ["Offsets short directional exposure with bounded convexity."],
                ))
        long_shares = sum(
            max(0.0, item.signed_quantity)
            for item in snapshot.positions
            if item.asset_type in {"EQUITY", "ETF"}
            and item.exact_symbol == symbol
        )
        if long_shares > 0:
            all_expiry_groups: dict[date, list[dict[str, Any]]] = defaultdict(list)
            for item in rows:
                all_expiry_groups[item["expiration_at"].date()].append(item)
            for expiry in sorted(all_expiry_groups):
                expiry_rows = all_expiry_groups[expiry]
                puts = [item for item in expiry_rows if item["put_call"] == "PUT" and item["strike"] < spot]
                calls = [item for item in expiry_rows if item["put_call"] == "CALL" and item["strike"] > spot]
                if not puts or not calls:
                    continue
                protective_put = max(puts, key=lambda item: (item["strike"], item["canonical_occ_symbol"]))
                covered_call = min(calls, key=lambda item: (item["strike"], item["canonical_occ_symbol"]))
                put_shares = _supported_deliverable_shares(protective_put)
                call_shares = _supported_deliverable_shares(covered_call)
                if (
                    put_shares is None
                    or call_shares is None
                    or not math.isclose(put_shares, call_shares)
                ):
                    continue
                structures.append((
                    f"{symbol} protective collar",
                    "Long put plus covered short call against existing shares.",
                    [("BUY", protective_put), ("SELL", covered_call)],
                    ["Bounds downside while financing part of the put with a covered call."],
                ))
                break
        for horizon, label in ((7, "one week"), (30, "one month")):
            puts = [item for item in rows if item["put_call"] == "PUT"]
            if puts:
                ten_delta = min(puts, key=lambda item: (
                    abs((item["expiration_at"].date() - payload_date(payload, snapshot)).days - horizon),
                    abs(item["delta"] + 0.10), item["canonical_occ_symbol"],
                ))
                structures.append((
                    f"{symbol} approximately 10-delta put ({label})",
                    "Existing contract nearest the requested tenor and -0.10 Tian delta.",
                    [("BUY", ten_delta)],
                    ["Far-tail convexity from an actually quoted contract."],
                ))
        output: list[dict[str, Any]] = []
        for name, description, structure, rationale in structures:
            per_structure_delta = abs(sum(
                (1.0 if side == "BUY" else -1.0)
                * row["delta"]
                * row["spot"]
                * row["multiplier"]
                for side, row in structure
            ))
            sizing_exposure = _objective_sizing_exposure(
                payload, assessment, target_delta, max(per_structure_delta, 1.0)
            )
            maximum_quantity = int(
                self.context.get("risk.hedge_lab.maximum_candidate_quantity", 100)
            )
            if "collar" in name.lower():
                shares_per_contract = _supported_deliverable_shares(structure[0][1])
                if shares_per_contract is None:
                    continue
                maximum_quantity = min(
                    maximum_quantity, int(long_shares // shares_per_contract)
                )
            quantities = _bounded_hedge_quantities(
                sizing_exposure,
                max(per_structure_delta, 1.0),
                maximum_quantity=maximum_quantity,
                maximum_overhedge_fraction=float(
                    self.context.get("risk.hedge_lab.maximum_overhedge_fraction", 0.25)
                ),
            )
            if not quantities:
                continue
            basis = _basis_risk(symbol, snapshot)
            liquidity_value, liquidity_score, liquidity_warnings = _option_liquidity(structure)
            for quantity in quantities:
                legs = [self._candidate_option_leg(payload, side, quantity, symbol, row, index) for index, (side, row) in enumerate(structure)]
                entry = sum((1 if leg["side"] == "BUY" else -1) * leg["quantity"] * leg["multiplier"] * leg["mark"] for leg in legs)
                carry = None
                delta_effect = sum((1 if leg["side"] == "BUY" else -1) * leg["quantity"] * leg["multiplier"] * leg["delta"] * leg["spot"] for leg in legs)
                improvement = _primary_improvement(
                    payload, assessment, target_delta, delta_effect, legs
                )
                fit = _fit_score(
                    improvement, basis, liquidity_score=liquidity_score,
                    cost_score=1.0 / (1.0 + abs(entry) / max(sizing_exposure, 1.0)),
                    carry_score=0.0,
                )
                output.append(_candidate_item(
                    payload, name=f"{name} × {int(quantity)}", description=description,
                    legs=legs, fit=fit, liquidity=liquidity_value,
                    target_after=_target_after(payload, assessment, delta_effect=delta_effect, legs=legs),
                    scenario_after=_scenario_after(assessment, delta_effect), entry_cost=entry,
                    carry=carry, basis=basis,
                    rationale=[
                        *rationale,
                        "Quantity is one member of the bounded integer objective search.",
                        "Research output only; not a recommendation.",
                    ],
                    residuals={"directional_delta_effect_usd": delta_effect, **_candidate_greek_effects(legs)},
                    data_quality={
                        "status": "usable_with_warnings", "price_source": "RefData option chain",
                        "greek_source": "Risk Tian", "carry_status": "insufficient_data_until_dated_repricing",
                        "quote_as_of": max((str(leg.get("quote_time") or "") for leg in legs), default=None),
                    },
                    warnings=[*liquidity_warnings, "carry_requires_dated_lifecycle_repricing"],
                ))
        return output

    def _candidate_option_leg(
        self,
        payload: HedgeCandidateQueryRequest,
        side: str,
        quantity: float,
        symbol: str,
        row: Mapping[str, Any],
        index: int,
    ) -> dict[str, Any]:
        canonical = str(row["canonical_occ_symbol"])
        expiration = row["expiration_at"]
        return {
            "client_leg_id": _stable_hash(payload.portfolio_snapshot_id, canonical, side, index)[:32],
            "instrument_type": "OPTION",
            "side": side,
            "quantity": quantity,
            "symbol": symbol,
            "display_name": f"{symbol} {expiration.date().isoformat()} {row['strike']:g} {str(row['put_call']).title()}",
            "canonical_occ_symbol": canonical,
            "expiration_at": expiration.isoformat(),
            "put_call": row["put_call"],
            "strike": row["strike"],
            "mark": row["mark"],
            "multiplier": row["multiplier"],
            "instrument_terms": row.get("instrument_terms"),
            "spot": row["spot"],
            "delta": row["delta"],
            "gamma": row["gamma"],
            "vega_per_1pct": row["vega_per_1pct"],
            "theta_per_day": row["theta_per_day"],
            "rho_per_1pct": row["rho_per_1pct"],
            "quote_time": row["quote_time"].isoformat() if row.get("quote_time") else None,
            "source": "RefData option chain + Risk Tian Greeks",
        }

    def _future_candidates(
        self,
        payload: HedgeCandidateQueryRequest,
        assessment: Mapping[str, Any],
        target_delta: float,
        as_of: datetime,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        configured_key = "risk.hedge_lab.futures_contracts"
        automatic = configured_key not in self.context
        allowed_roots = {
            str(root).strip().upper().removeprefix("/")
            for root in self.context.get("risk.hedge_lab.allowed_futures_roots", ())
            if str(root).strip()
        }
        objective_roots = _objective_future_roots(payload)
        eligible_roots = allowed_roots & objective_roots
        ladders = {
            root: contracts
            for root, contracts in _future_contract_ladders(as_of.date()).items()
            if root in eligible_roots
        } if automatic else {}
        requested = (
            [contract for contracts in ladders.values() for contract in contracts]
            if automatic
            else [
                _canonical_future_contract(item)
                for item in self.context.get(configured_key, ())
                if str(item).strip()
                and _future_root(item) in eligible_roots
            ][:8]
        )
        requested = list(dict.fromkeys(contract for contract in requested if contract))
        if not requested:
            return [], []
        try:
            specification_payload = self.reference_client.futures_specifications(requested)
        except Exception as exc:
            return [], [f"futures_specifications_unavailable:{type(exc).__name__}"]
        specification_rows = [
            dict(row) for row in specification_payload.get("data") or []
        ]
        selection_warnings: list[str] = []
        if automatic:
            selected, selection_warnings = _select_active_future_contracts(
                ladders,
                specification_rows,
                as_of=as_of,
                roll_buffer_days=max(
                    0,
                    int(self.context.get("risk.hedge_lab.futures_roll_buffer_days", 2)),
                ),
            )
        else:
            selected = requested
        rows_by_contract = {
            _canonical_future_contract(
                row.get("canonical_contract") or row.get("requested_contract")
            ): row
            for row in specification_rows
            if _canonical_future_contract(
                row.get("canonical_contract") or row.get("requested_contract")
            )
        }
        prices, price_warnings = self._current_prices(
            selected, instrument_type="FUTURE"
        )
        output = []
        for selected_contract in selected:
            row = rows_by_contract.get(selected_contract) or {}
            requested_contract = _canonical_future_contract(
                row.get("requested_contract") or selected_contract
            )
            contract = _canonical_future_contract(
                row.get("canonical_contract") or requested_contract
            )
            multiplier = _positive(row.get("price_multiplier"))
            price_row = prices.get(contract) or prices.get(requested_contract) or {}
            mark = _positive(price_row.get("mark"))
            price_as_of = _datetime(price_row.get("as_of"))
            if (
                row.get("status") not in {"ok", "partial"}
                or not contract
                or multiplier is None
                or mark is None
            ):
                continue
            if price_as_of is None or not self._quote_fresh(
                price_as_of,
                as_of,
                float(self.context.get("risk.hedge_lab.maximum_quote_age_seconds", 900)),
                source_kind=str(price_row.get("source_kind") or "intraday_quote"),
            ):
                price_warnings.append(f"{contract}:current_price_stale_or_undated")
                continue
            side = "SELL" if target_delta > 0 else "BUY"
            signed = -1.0 if side == "SELL" else 1.0
            quantities = _bounded_hedge_quantities(
                abs(target_delta),
                multiplier * mark,
                maximum_quantity=int(
                    self.context.get("risk.hedge_lab.maximum_candidate_quantity", 100)
                ),
                maximum_overhedge_fraction=float(
                    self.context.get("risk.hedge_lab.maximum_overhedge_fraction", 0.25)
                ),
            )
            if not quantities:
                continue
            expiration_at = _datetime(row.get("last_trade_at"))
            display_contract = f"/{contract}"
            for quantity in quantities:
                effect = signed * quantity * multiplier * mark
                leg = {
                    "client_leg_id": _stable_hash(payload.portfolio_snapshot_id, contract, side, quantity)[:32],
                    "instrument_type": "FUTURE", "side": side, "quantity": quantity,
                    "symbol": display_contract, "canonical_occ_symbol": None,
                    "expiration_at": expiration_at.isoformat() if expiration_at else None,
                    "put_call": None, "strike": None, "mark": mark, "multiplier": multiplier,
                    "spot": mark, "delta": 1.0, "gamma": 0.0, "vega_per_1pct": 0.0,
                    "theta_per_day": 0.0, "rho_per_1pct": 0.0,
                    "source": "RefData futures specification + current-price",
                }
                output.append(_candidate_item(
                    payload, name=f"{display_contract} future × {int(quantity)}",
                    description="Priced futures delta offset using a published contract multiplier.",
                    legs=[leg], fit=_fit_score(
                        _primary_improvement(payload, assessment, target_delta, effect, [leg]),
                        0.25, 0.5, 0.75, 0.0,
                    ),
                    liquidity="Insufficient data",
                    target_after=_target_after(payload, assessment, delta_effect=effect),
                    scenario_after=_scenario_after(assessment, effect), entry_cost=0.0,
                    carry=None, basis=0.25,
                    rationale=[
                        "Capital-efficient directional offset; margin is not modeled.",
                        "Quantity is one member of the bounded integer objective search.",
                    ],
                    residuals={
                        "directional_delta_effect_usd": effect,
                        **_candidate_greek_effects([leg]),
                    },
                    data_quality={"status": "usable_with_warnings", "margin_status": "insufficient_data"},
                    warnings=["futures_margin_and_liquidity_unavailable", "futures_carry_unavailable"],
                ))
        return output, [*selection_warnings, *price_warnings]

    def _resolve_leg(
        self,
        leg: HedgeLeg,
        response: PortfolioSnapshotResponse,
        snapshot_id: str,
    ) -> tuple[NormalizedPosition, dict[str, Any], dict[str, Any], list[str]]:
        if leg.instrument_type == "OPTION":
            return self._resolve_option_leg(leg, response, snapshot_id)
        if leg.instrument_type == "FUTURE":
            return self._resolve_future_leg(leg, response, snapshot_id)
        return self._resolve_equity_leg(leg, response, snapshot_id)

    def _resolve_equity_leg(self, leg, response, snapshot_id):
        warnings: list[str] = []
        mark = _positive(leg.mark)
        source = "client_supplied_mark" if mark is not None else None
        quote_as_of = None
        if mark is None:
            prices, price_warnings = self._current_prices([leg.symbol])
            warnings.extend(price_warnings)
            row = prices.get(leg.symbol) or {}
            mark = _positive(row.get("mark"))
            source = row.get("source")
            quote_as_of = _datetime(row.get("as_of"))
            if quote_as_of is None or not self._quote_fresh(
                quote_as_of,
                response.as_of,
                float(self.context.get("risk.hedge_lab.maximum_quote_age_seconds", 900)),
                source_kind=str(row.get("source_kind") or "intraday_quote"),
            ):
                raise ValueError(f"{leg.client_leg_id}: equity price is stale or undated")
        else:
            warnings.append(f"{leg.client_leg_id}:client_mark_has_no_timestamp")
        if mark is None:
            raise ValueError(f"{leg.client_leg_id}: required equity price is unavailable")
        if leg.multiplier is not None and not math.isclose(leg.multiplier, 1.0):
            raise ValueError(f"{leg.client_leg_id}: equity multiplier must be 1")
        signed_quantity = leg.quantity if leg.side == "BUY" else -leg.quantity
        position = _proposal_position(
            snapshot_id=snapshot_id, leg=leg, asset_type="EQUITY", exact_symbol=leg.symbol,
            underlying_symbol=leg.symbol, signed_quantity=signed_quantity, multiplier=1.0,
            mark=mark, quote_as_of=quote_as_of, quote_source=str(source or "unknown"),
        )
        public = _public_resolved_leg(leg, position, source=str(source), spot=mark)
        audit = _pricing_audit(leg, mark, source, quote_as_of, None)
        return position, public, audit, warnings

    def _resolve_option_leg(self, leg, response, snapshot_id):
        warnings: list[str] = []
        assert leg.expiration_at and leg.put_call and leg.strike
        option_symbol = leg.canonical_occ_symbol or leg.symbol
        try:
            occ = parse_occ_symbol(option_symbol)
        except ValueError as exc:
            raise ValueError(
                f"{leg.client_leg_id}: canonical_occ_symbol is required for an option root symbol"
            ) from exc
        if leg.canonical_occ_symbol and occ.root != leg.symbol:
            raise ValueError(f"{leg.client_leg_id}: option root does not match canonical OCC symbol")
        if occ.put_call != leg.put_call or not math.isclose(float(occ.strike), leg.strike):
            raise ValueError(f"{leg.client_leg_id}: option fields do not match canonical OCC symbol")
        if occ.expiration != leg.expiration_at.date():
            raise ValueError(f"{leg.client_leg_id}: expiration does not match canonical OCC symbol")
        term_row: dict[str, Any] | None = None
        if occ.root in {"SPY", "QQQ"} and (
            leg.multiplier is None or math.isclose(leg.multiplier, 100.0)
        ):
            term_row = {
                "canonical_occ_symbol": occ.canonical,
                "asset_subtype": "etf_option",
                "terms_status": "standard",
                "price_multiplier": 100.0,
                "shares_per_contract": 100.0,
                "adjusted_contract": False,
                "exercise_style": "american",
                "settlement_type": "physical",
                "expiration_at": leg.expiration_at.isoformat(),
                "source": "standard_listed_etf_option_terms",
            }
        else:
            try:
                terms_payload = self.reference_client.option_contract_terms([occ.canonical])
                term_row = next(
                    (
                        dict(item)
                        for item in terms_payload.get("data") or []
                        if item.get("canonical_occ_symbol") == occ.canonical
                    ),
                    None,
                )
            except Exception as exc:
                warnings.append(
                    f"{leg.client_leg_id}:option_contract_terms_unavailable:{type(exc).__name__}"
                )
        terms = build_instrument_terms(
            occ.canonical,
            broker_type="OPTION",
            option_terms=term_row,
            context=self.context,
        )
        multiplier = terms.price_multiplier
        if multiplier is None or terms.expiration_at is None:
            raise ValueError(f"{leg.client_leg_id}: option contract terms are insufficient")
        if terms.expiration_at != leg.expiration_at:
            raise ValueError(
                f"{leg.client_leg_id}: supplied expiration timestamp conflicts with contract terms"
            )
        if leg.multiplier is not None and not math.isclose(leg.multiplier, multiplier):
            raise ValueError(f"{leg.client_leg_id}: supplied option multiplier conflicts with contract terms")
        quote: dict[str, Any] = {}
        option_quote_as_of: datetime | None = None
        try:
            payload = self.reference_client.option_quotes([occ.canonical])
            quote = next((dict(item) for item in payload.get("data") or [] if item.get("status") == "ok"), {})
            option_quote_as_of = _datetime((payload.get("meta") or {}).get("as_of"))
        except Exception as exc:
            warnings.append(f"{leg.client_leg_id}:option_quote_unavailable:{type(exc).__name__}")
        quote_as_of = _datetime(quote.get("quote_time")) or option_quote_as_of
        service_now = _datetime(self.portfolio_service._now()) or response.as_of
        freshness_valuation_at = max(
            response.as_of,
            service_now,
        )
        if quote and (
            quote_as_of is None
            or not self._quote_fresh(
                quote_as_of,
                freshness_valuation_at,
                float(self.context.get("risk.hedge_lab.maximum_quote_age_seconds", 900)),
                source_kind=str(quote.get("source_kind") or "intraday_quote"),
            )
        ):
            warnings.append(f"{leg.client_leg_id}:option_quote_stale_or_undated")
            quote = {}
            quote_as_of = None
        mark: float | None = None
        source = "RefData exact option quote"
        bid, ask = _finite(quote.get("bid")), _finite(quote.get("ask"))
        if bid is not None and ask is not None:
            if bid < 0 or ask <= 0 or bid > ask:
                raise ValueError(f"{leg.client_leg_id}: option quote is crossed or invalid")
            mark = (bid + ask) / 2.0
        else:
            mark = _positive(quote.get("mark"))
        client_mark = _positive(leg.mark)
        if mark is not None and client_mark is not None and not math.isclose(
            mark, client_mark, rel_tol=0.005, abs_tol=0.01
        ):
            warnings.append(f"{leg.client_leg_id}:client_mark_replaced_by_exact_quote")
        if mark is None:
            mark = client_mark
            source = "client_supplied_mark"
        if source == "client_supplied_mark":
            warnings.append(f"{leg.client_leg_id}:client_mark_has_no_timestamp")
        if mark is None:
            raise ValueError(f"{leg.client_leg_id}: required option mark is unavailable")
        spot = _positive(quote.get("underlying_price"))
        if spot is None:
            held_spots = [
                _positive(item.raw_broker_fields.get("underlying_price"))
                for item in response.data.positions if item.underlying_symbol == occ.root
            ]
            spot = next((item for item in held_spots if item is not None), None)
        if spot is None:
            prices, price_warnings = self._current_prices([occ.root])
            warnings.extend(price_warnings)
            spot = _positive((prices.get(occ.root) or {}).get("mark"))
        if spot is None:
            raise ValueError(f"{leg.client_leg_id}: underlying spot is unavailable")
        pricing_valuation_at = max(response.as_of, quote_as_of or response.as_of)
        rate = self._risk_free_rate(pricing_valuation_at, leg.expiration_at)
        if rate is None:
            raise ValueError(f"{leg.client_leg_id}: risk-free rate is unavailable")
        dividend_inputs: list[dict[str, Any]] = []
        try:
            dividend_inputs = list(
                self.reference_client.dividends(
                    occ.root,
                    start=response.as_of.date() - timedelta(days=365),
                    end=leg.expiration_at.date(),
                )
            )
        except Exception as exc:
            warnings.append(
                f"{leg.client_leg_id}:dividend_history_unavailable:{type(exc).__name__}"
            )
        dividend_yield = float(
            self.context.get("risk.dividends.default_continuous_yield", 0.0)
        )
        if dividend_inputs:
            try:
                dividend_yield = derive_trailing_cash_yield(
                    dividend_inputs,
                    spot=spot,
                    valuation_date=pricing_valuation_at.date(),
                ).continuous_yield
            except ValueError:
                warnings.append(f"{leg.client_leg_id}:dividend_inputs_unusable")
        future_cash_dividends = [
            item
            for item in dividend_inputs
            if (
                (event_date := _dividend_event_date(item)) is not None
                and pricing_valuation_at.date() <= event_date <= leg.expiration_at.date()
                and _is_ordinary_cash_dividend(item)
            )
        ]
        pricing_spot = spot
        if terms.exercise_style == "american" and future_cash_dividends:
            if not _as_bool(
                self.context.get(
                    "risk.options.tian.escrowed_dividend_mode_enabled", True
                )
            ):
                raise ValueError(
                    f"{leg.client_leg_id}: discrete-dividend early-exercise pricing is unavailable"
                )
            try:
                adjustment = escrowed_dividend_adjustment(
                    future_cash_dividends,
                    spot=spot,
                    valuation_at=pricing_valuation_at,
                    expiration_at=leg.expiration_at,
                    continuous_rate=rate,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{leg.client_leg_id}: escrowed-dividend inputs are unusable"
                ) from exc
            pricing_spot = adjustment.adjusted_spot
            dividend_yield = 0.0
            warnings.append(f"{leg.client_leg_id}:escrowed_dividend_model_applied")
        if terms.exercise_style not in {"american", "european"}:
            raise ValueError(
                f"{leg.client_leg_id}: option exercise style is required for proposal pricing"
            )
        contract = IvGreeksContract(
            request_id=f"proposal:{leg.client_leg_id}", canonical_occ_symbol=occ.canonical,
            underlying_symbol=occ.root, put_call=occ.put_call, strike=float(occ.strike),
            exercise_type={
                "european": "EUROPEAN",
                "american": "AMERICAN",
            }[terms.exercise_style],
            expiration_at=leg.expiration_at, spot=pricing_spot, mark=mark,
            risk_free_rate=rate,
            dividend_yield=dividend_yield,
            quote_time=quote_as_of,
        )
        try:
            solved = self.portfolio_service.tian_service.solve(
                IvGreeksRequest(valuation_at=pricing_valuation_at, contracts=[contract])
            ).data[0]
            greek = solved.model_dump(mode="python") if hasattr(solved, "model_dump") else vars(solved)
        except Exception as exc:
            raise ValueError(f"{leg.client_leg_id}: Tian pricing is unavailable") from exc
        if greek.get("status") not in {"ok", "usable_with_warnings"}:
            reason = str(greek.get("reason") or "unknown Tian normalization failure")
            raise ValueError(
                f"{leg.client_leg_id}: option Greeks could not be normalized ({reason})"
            )
        required = {name: _finite(greek.get(name)) for name in ("delta", "gamma", "vega_per_1pct", "theta_per_day", "rho_per_1pct", "implied_volatility")}
        if any(value is None for value in required.values()):
            raise ValueError(f"{leg.client_leg_id}: normalized option Greeks are incomplete")
        signed_quantity = leg.quantity if leg.side == "BUY" else -leg.quantity
        position = _proposal_position(
            snapshot_id=snapshot_id, leg=leg, asset_type="OPTION", exact_symbol=occ.canonical,
            underlying_symbol=occ.root, signed_quantity=signed_quantity, multiplier=multiplier,
            mark=mark, quote_as_of=quote_as_of, quote_source=str(source),
            canonical_occ_symbol=occ.canonical, expiration_at=leg.expiration_at,
            put_call=occ.put_call, strike=float(occ.strike), spot=spot, rate=rate,
            delta=required["delta"], gamma=required["gamma"], vega=required["vega_per_1pct"],
            theta=required["theta_per_day"], rho=required["rho_per_1pct"],
            implied_volatility=required["implied_volatility"],
            instrument_terms=terms,
            dividend_yield=dividend_yield,
            dividend_inputs=dividend_inputs,
        )
        public = _public_resolved_leg(leg, position, source=f"{source} + Risk Tian", spot=spot)
        audit = _pricing_audit(leg, mark, source, quote_as_of, "Risk Tian Greeks")
        return position, public, audit, warnings

    def _resolve_future_leg(self, leg, response, snapshot_id):
        warnings: list[str] = []
        try:
            payload = self.reference_client.futures_specifications([leg.symbol])
            specification = next((dict(item) for item in payload.get("data") or [] if item.get("status") == "ok"), None)
        except Exception as exc:
            raise ValueError(f"{leg.client_leg_id}: futures specification is unavailable") from exc
        if not specification:
            raise ValueError(f"{leg.client_leg_id}: futures specification is unavailable")
        multiplier = _positive(specification.get("price_multiplier"))
        if multiplier is None:
            raise ValueError(f"{leg.client_leg_id}: futures multiplier is unavailable")
        if leg.multiplier is not None and not math.isclose(leg.multiplier, multiplier):
            raise ValueError(f"{leg.client_leg_id}: supplied futures multiplier conflicts with RefData")
        specification_expiration = _datetime(specification.get("last_trade_at"))
        if (
            leg.expiration_at is not None
            and specification_expiration is not None
            and leg.expiration_at.date() != specification_expiration.date()
        ):
            raise ValueError(
                f"{leg.client_leg_id}: supplied futures expiration conflicts with RefData"
            )
        mark = _positive(leg.mark)
        source = "client_supplied_mark" if mark is not None else "RefData current-price"
        quote_as_of = None
        if mark is None:
            prices, price_warnings = self._current_prices(
                [leg.symbol], instrument_type="FUTURE"
            )
            warnings.extend(price_warnings)
            row = prices.get(leg.symbol) or {}
            mark = _positive(row.get("mark"))
            quote_as_of = _datetime(row.get("as_of"))
            if quote_as_of is None or not self._quote_fresh(
                quote_as_of,
                response.as_of,
                float(self.context.get("risk.hedge_lab.maximum_quote_age_seconds", 900)),
                source_kind=str(row.get("source_kind") or "intraday_quote"),
            ):
                raise ValueError(f"{leg.client_leg_id}: futures price is stale or undated")
        else:
            warnings.append(f"{leg.client_leg_id}:client_mark_has_no_timestamp")
        if mark is None:
            raise ValueError(f"{leg.client_leg_id}: required futures mark is unavailable")
        signed_quantity = leg.quantity if leg.side == "BUY" else -leg.quantity
        position = _proposal_position(
            snapshot_id=snapshot_id, leg=leg, asset_type="FUTURE", exact_symbol=leg.symbol,
            underlying_symbol=None, signed_quantity=signed_quantity, multiplier=multiplier,
            mark=mark, quote_as_of=quote_as_of, quote_source=source,
            futures_root=str(specification.get("root") or "") or None,
            expiration_at=specification_expiration or leg.expiration_at,
            instrument_terms=build_instrument_terms(
                leg.symbol,
                broker_type="FUTURE",
                futures_spec=specification,
                context=self.context,
            ),
        )
        public = _public_resolved_leg(leg, position, source=source, spot=mark)
        audit = _pricing_audit(leg, mark, source, quote_as_of, "RefData futures specification")
        return position, public, audit, warnings

    def _risk_free_rate(self, valuation_at: datetime, expiration_at: datetime) -> float | None:
        configured = self.context.get("risk.rates.configured_flat_continuous_rate")
        if configured not in (None, ""):
            return _finite(configured)
        try:
            payload = self.reference_client.risk_free_curve(valuation_at.date())
            curve = payload.get("data")
            tenor = max(0.0, (expiration_at - valuation_at).total_seconds() / 86400.0)
            return interpolate_risk_free_rate(
                curve,
                valuation_date=valuation_at.date(),
                tenor_days=tenor,
                max_age_days=int(self.context.get("risk.rates.max_age_days", 5)),
                allow_flat_extrapolation=_as_bool(self.context.get("risk.rates.allow_flat_extrapolation", True)),
            ).continuous_annual_rate_decimal
        except Exception:
            return None

    def _analysis_response(
        self,
        response: PortfolioSnapshotResponse,
        methodology_version: str,
        data: dict[str, Any],
        *,
        warnings: Sequence[str],
        insufficient: bool,
    ) -> PortfolioAnalysisResponse:
        quality = response.data_quality
        combined = tuple(
            _dedupe([*quality.warnings, *warnings, *_collect_nested_warnings(data)])
        )
        status = quality.status
        if insufficient:
            status = "insufficient_data"
        elif combined and status == "usable":
            status = "usable_with_warnings"
        quality = quality.model_copy(
            update={"status": status, "warnings": combined}
        )
        return PortfolioAnalysisResponse(
            portfolio_snapshot_id=response.portfolio_snapshot_id,
            configuration_version=response.configuration_version,
            as_of=response.as_of,
            display_timezone=response.display_timezone,
            data_quality=quality,
            methodology_version=methodology_version,
            data=data,
        )


def _proposal_position(
    *, snapshot_id: str, leg: HedgeLeg, asset_type: str, exact_symbol: str,
    underlying_symbol: str | None, signed_quantity: float, multiplier: float,
    mark: float, quote_as_of: datetime | None, quote_source: str,
    canonical_occ_symbol: str | None = None, expiration_at: datetime | None = None,
    put_call: str | None = None, strike: float | None = None, spot: float | None = None,
    rate: float | None = None, delta: float | None = None, gamma: float | None = None,
    vega: float | None = None, theta: float | None = None, rho: float | None = None,
    implied_volatility: float | None = None, futures_root: str | None = None,
    instrument_terms: InstrumentTerms | None = None,
    dividend_yield: float = 0.0,
    dividend_inputs: Sequence[Mapping[str, Any]] = (),
) -> NormalizedPosition:
    position_id = "proposal:" + _stable_hash(snapshot_id, leg.client_leg_id)
    economic = "proposal:" + _stable_hash(snapshot_id, underlying_symbol or futures_root or exact_symbol)
    value = signed_quantity * multiplier * mark
    # Futures proposal notional is exposure rather than a paid liquidation value.
    liquidation_value = None if asset_type == "FUTURE" else value
    notional_value = value if asset_type in {"EQUITY", "ETF", "FUTURE"} else None
    delta_exposure = (
        value
        if asset_type in {"EQUITY", "ETF", "FUTURE"}
        else signed_quantity * multiplier * float(delta) * float(spot)
        if asset_type == "OPTION" and delta is not None and spot is not None
        else None
    )
    return NormalizedPosition(
        position_id=position_id,
        economic_position_id=economic,
        account_id="hedge_lab_ephemeral",
        instrument_id=canonical_occ_symbol or exact_symbol,
        asset_type=asset_type,
        exact_symbol=exact_symbol,
        canonical_occ_symbol=canonical_occ_symbol,
        underlying_symbol=underlying_symbol,
        futures_root=futures_root,
        put_call=put_call,
        strike=strike,
        expiration_at=expiration_at,
        signed_quantity=signed_quantity,
        multiplier=multiplier,
        multiplier_source="hedge_lab_validated_reference_v1",
        instrument_terms=instrument_terms,
        raw_average_price=mark,
        normalized_average_price=mark,
        normalization_method="current_mark_ephemeral_analysis",
        signed_cost_basis_usd=None if asset_type == "FUTURE" else value,
        mark=mark,
        signed_market_value_usd=liquidation_value,
        signed_liquidation_value_usd=liquidation_value,
        signed_notional_exposure_usd=notional_value,
        delta_exposure_usd=delta_exposure,
        unrealized_pnl_usd=0.0,
        sector="Proposed hedge",
        delta=delta,
        gamma=gamma,
        vega_per_1pct=vega,
        theta_per_day=theta,
        rho_per_1pct=rho,
        implied_volatility=implied_volatility,
        greek_source="tian" if asset_type == "OPTION" else None,
        quote_as_of=quote_as_of,
        quote_source=quote_source,
        data_quality=DataQuality(
            status="usable" if quote_as_of else "usable_with_warnings",
            sources=(SourceState(name=quote_source, as_of=quote_as_of, state="request_scoped"),),
            coverage=Coverage(positions_total=1, positions_modeled=1),
            warnings=() if quote_as_of else ("quote_timestamp_unavailable",),
        ),
        raw_broker_fields={
            "analysis_only": True,
            "client_leg_id": leg.client_leg_id,
            "underlying_price": spot,
            "risk_free_rate_continuous": rate,
            "dividend_yield_continuous": dividend_yield,
            "dividend_inputs": [dict(item) for item in dividend_inputs],
        },
    )


def _current_metrics(
    positions: Sequence[NormalizedPosition],
    portfolio_value_usd: float | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    non_future = [item for item in positions if item.asset_type != "FUTURE"]
    directional_positions = [
        item for item in positions if item.asset_type != "CASH"
    ]
    market_values = [
        float(item.signed_market_value_usd)
        for item in non_future
        if item.signed_market_value_usd is not None and _risk_field_valid(item, "mark")
    ]
    deltas = [
        value
        for item in positions
        if _risk_field_valid(item, "delta")
        and (value := delta_equivalent_usd(item)) is not None
    ]
    warnings = []
    market_missing = [
        item.position_id
        for item in non_future
        if item.signed_market_value_usd is None or not _risk_field_valid(item, "mark")
    ]
    delta_missing = [
        item.position_id
        for item in directional_positions
        if not _risk_field_valid(item, "delta") or delta_equivalent_usd(item) is None
    ]
    market_omission = assess_omission_materiality(
        market_missing,
        positions,
        portfolio_value_usd=portfolio_value_usd,
        context=context,
    )
    delta_omission = assess_omission_materiality(
        delta_missing,
        positions,
        portfolio_value_usd=portfolio_value_usd,
        context=context,
    )
    if market_missing and not market_omission.may_exclude:
        warnings.append("partial_market_value_coverage")
    if delta_missing and not delta_omission.may_exclude:
        warnings.append("partial_directional_delta_coverage")
    warnings.extend(
        warning
        for decision in (market_omission, delta_omission)
        if (warning := decision.warning) is not None
    )
    greeks = _greek_metrics(positions, portfolio_value_usd, context)
    market_complete = not market_missing or market_omission.may_exclude
    directional_complete = not delta_missing or delta_omission.may_exclude
    market_total = sum(market_values) if market_complete else None
    gross_market = (
        sum(abs(value) for value in market_values)
        if market_complete else None
    )
    directional_total = (
        sum(deltas) if directional_complete else None
    )
    return {
        "signed_market_value_usd": market_total,
        "gross_market_value_usd": gross_market,
        "long_exposure_usd": (
            sum(value for value in market_values if value > 0)
            if market_total is not None else None
        ),
        "short_exposure_usd": (
            sum(-value for value in market_values if value < 0)
            if market_total is not None else None
        ),
        "directional_delta_exposure_usd": directional_total,
        **greeks,
        "warnings": list(dict.fromkeys(
            warnings + list(greeks.pop("warnings", []))
            if "warnings" in greeks else warnings
        )),
    }


def _greek_metrics(
    positions: Sequence[NormalizedPosition],
    portfolio_value_usd: float | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    options = [item for item in positions if item.asset_type == "OPTION"]
    warnings: list[str] = []
    definitions = {
        "option_delta_exposure_usd": "delta",
        "gamma_pnl_for_squared_1pct_move_usd": "gamma",
        "vega_per_1pct_usd": "vega",
        "theta_per_day_usd": "theta",
        "rho_per_1pct_usd": "rho",
    }
    values: dict[str, list[float]] = {key: [] for key in definitions}
    missing: dict[str, list[str]] = {key: [] for key in definitions}
    contributions: list[dict[str, Any]] = []
    for item in options:
        spot = _positive(item.raw_broker_fields.get("underlying_price"))
        scale = item.signed_quantity * item.multiplier if item.multiplier is not None else None
        if (
            spot is None
            and scale not in (None, 0)
            and item.delta not in (None, 0)
            and item.delta_exposure_usd is not None
        ):
            spot = _positive(item.delta_exposure_usd / (scale * item.delta))
        row: dict[str, Any] = {
            "position_id": item.position_id,
            "economic_position_id": item.economic_position_id,
            "expiration_at": item.expiration_at,
        }
        calculated = {
            "option_delta_exposure_usd": (
                float(item.delta_exposure_usd)
                if item.delta_exposure_usd is not None and _risk_field_valid(item, "delta")
                else scale * item.delta * spot
                if scale is not None and spot is not None and item.delta is not None
                and _risk_field_valid(item, "delta") else None
            ),
            "gamma_pnl_for_squared_1pct_move_usd": (
                0.5 * scale * item.gamma * (0.01 * spot) ** 2
                if scale is not None and spot is not None and item.gamma is not None
                and _risk_field_valid(item, "gamma") else None
            ),
            "vega_per_1pct_usd": (
                scale * item.vega_per_1pct
                if scale is not None and item.vega_per_1pct is not None
                and _risk_field_valid(item, "vega") else None
            ),
            "theta_per_day_usd": (
                scale * item.theta_per_day
                if scale is not None and item.theta_per_day is not None
                and _risk_field_valid(item, "theta") else None
            ),
            "rho_per_1pct_usd": (
                scale * item.rho_per_1pct
                if scale is not None and item.rho_per_1pct is not None
                and _risk_field_valid(item, "rho") else None
            ),
        }
        for key, value in calculated.items():
            row[key] = value
            if value is None:
                missing[key].append(item.position_id)
            else:
                values[key].append(float(value))
        contributions.append(row)
    # Assess missing sensitivities separately for each aggregate Greek.
    omissions = {
        key: assess_omission_materiality(
            position_ids,
            positions,
            portfolio_value_usd=portfolio_value_usd,
            context=context,
        )
        for key, position_ids in missing.items()
    }
    totals: dict[str, Any] = {
        key: (
            sum(values[key])
            if not missing[key] or omissions[key].may_exclude
            else None
        )
        for key in definitions
    }
    for key, position_ids in missing.items():
        if position_ids and not omissions[key].may_exclude:
            warnings.extend(
                f"{position_id}:{definitions[key]}_unavailable"
                for position_id in position_ids
            )
    warnings.extend(
        warning
        for decision in omissions.values()
        if (warning := decision.warning) is not None
    )
    totals["greek_coverage"] = {
        key: {
            "eligible_positions": len(options),
            "modeled_positions": len(options) - len(position_ids),
            "percent": (100.0 * (len(options) - len(position_ids)) / len(options)) if options else 100.0,
            "missing_position_ids": position_ids,
        }
        for key, position_ids in missing.items()
    }
    totals["greek_contributions"] = contributions
    totals["warnings"] = list(dict.fromkeys(warnings))
    return totals


def _risk_field_valid(item: NormalizedPosition, field: str) -> bool:
    validity = item.field_validity.get(field)
    if validity is not None:
        return validity.valid_for_risk
    return item.data_quality.status in {"usable", "usable_with_warnings"}


def _position_exposures(positions: Sequence[NormalizedPosition]) -> dict[str, float]:
    exposures: dict[str, float] = {}
    for item in positions:
        broker_type = str(item.raw_broker_fields.get("instrument_type") or "")
        if item.asset_type == "CASH" or "NO_LOAD" in broker_type or "MONEY_MARKET" in broker_type:
            continue
        symbol = item.underlying_symbol or item.exact_symbol
        if not _risk_field_valid(item, "delta"):
            continue
        value = delta_equivalent_usd(item)
        if symbol and value is not None:
            exposures[symbol] = exposures.get(symbol, 0.0) + float(value)
    return exposures


def _economic_exposure_units(
    positions: Sequence[NormalizedPosition],
) -> dict[str, dict[str, Any]]:
    units: dict[str, dict[str, Any]] = {}
    for item in positions:
        broker_type = str(item.raw_broker_fields.get("instrument_type") or "")
        if item.asset_type == "CASH" or "NO_LOAD" in broker_type or "MONEY_MARKET" in broker_type:
            continue
        symbol = item.underlying_symbol or item.exact_symbol
        exposure = delta_equivalent_usd(item)
        if not symbol or exposure is None:
            continue
        unit_id = item.economic_position_id
        existing = units.get(unit_id)
        if existing is not None and existing["symbol"] != symbol:
            # Defensive fallback for malformed grouping metadata. The original
            # economic identifier remains present in the contribution details.
            unit_id = f"{item.economic_position_id}:{item.position_id}"
            existing = None
        if existing is None:
            units[unit_id] = {
                "economic_position_id": item.economic_position_id,
                "symbol": symbol,
                "exposure_usd": float(exposure),
                "position_ids": [item.position_id],
            }
        else:
            existing["exposure_usd"] = float(existing["exposure_usd"]) + float(exposure)
            existing["position_ids"].append(item.position_id)
    for unit in units.values():
        unit["position_ids"] = sorted(set(unit["position_ids"]))
    return units


def _risk_denominator(item: NormalizedPosition) -> float:
    value = (
        item.signed_notional_exposure_usd
        if item.asset_type == "FUTURE"
        else item.signed_liquidation_value_usd
    )
    if value is None:
        value = item.delta_exposure_usd
    return float(value or 0.0)


def _terminal_payoff(positions: Sequence[NormalizedPosition]) -> dict[str, Any]:
    return certify_maximum_loss(positions)


def _position_terminal_pnl(item: NormalizedPosition, spot: float) -> float:
    if item.asset_type in {"EQUITY", "ETF", "MUTUAL_FUND"}:
        return item.signed_quantity * (spot - float(item.mark))
    intrinsic = (
        max(0.0, spot - float(item.strike))
        if item.put_call == "CALL"
        else max(0.0, float(item.strike) - spot)
    )
    return item.signed_quantity * float(item.multiplier) * intrinsic - float(item.signed_market_value_usd)


def _insufficient_payoff(reason: str) -> dict[str, Any]:
    return {
        "status": "insufficient_data", "reason": reason, "terminal_payoff": [],
        "minimum_pnl_usd": None, "maximum_loss_usd": None,
        "position_contributions": [], "warnings": ["maximum_loss_unavailable"],
    }


def _diversification(
    positions: Sequence[NormalizedPosition], factors: Mapping[str, Any]
) -> dict[str, Any]:
    expirations: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"values": [], "position_ids": [], "missing_position_ids": []}
    )
    sectors: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"values": [], "position_ids": [], "missing_position_ids": []}
    )
    for item in positions:
        expiration = item.expiration_at.date().isoformat() if item.expiration_at else "No expiration"
        for group in (expirations[expiration], sectors[item.sector]):
            group["position_ids"].append(item.position_id)
            if item.signed_market_value_usd is not None and _risk_field_valid(item, "mark"):
                group["values"].append(abs(float(item.signed_market_value_usd)))
            else:
                group["missing_position_ids"].append(item.position_id)

    def rows(groups: Mapping[str, Mapping[str, Any]], label: str) -> list[dict[str, Any]]:
        return [
            {
                label: key,
                "gross_market_value_usd": (
                    sum(value["values"])
                    if not value["missing_position_ids"]
                    else None
                ),
                "position_ids": sorted(value["position_ids"]),
                "coverage": {
                    "eligible_positions": len(value["position_ids"]),
                    "modeled_positions": len(value["values"]),
                    "percent": (
                        100.0 * len(value["values"]) / len(value["position_ids"])
                        if value["position_ids"] else 100.0
                    ),
                    "missing_position_ids": sorted(value["missing_position_ids"]),
                    "weight_basis": "field_valid_position_count",
                },
            }
            for key, value in sorted(groups.items())
        ]
    return {
        "factors": list(factors.get("items") or []),
        "expirations": rows(expirations, "expiration"),
        "sectors": rows(sectors, "sector"),
    }


def _linear_candidates_applicable(payload: HedgeCandidateQueryRequest) -> bool:
    if payload.objective == "Maximum Loss":
        return False
    if payload.objective != "Greeks":
        return True
    parameters = payload.parsed_objective_parameters()
    return isinstance(parameters, GreeksParameters) and "delta" in parameters.selected_greeks


def _candidate_universe_category(legs: Sequence[Mapping[str, Any]]) -> str:
    if not legs:
        return "other"
    instrument_type = str(legs[0].get("instrument_type") or "").upper()
    symbol = str(legs[0].get("symbol") or "").strip().upper()
    if instrument_type == "FUTURE":
        if symbol.startswith(("/ES", "/MES", "/NQ", "/MNQ", "/RTY", "/M2K")):
            return "index_future"
        if symbol.startswith("/VX"):
            return "volatility_future"
        if symbol.startswith(("/ZN", "/ZB", "/CL", "/GC", "/6E", "/DX")):
            return "factor_future"
        return "other"
    if instrument_type == "OPTION":
        if symbol == "VIX":
            return "volatility_option"
        if symbol in {*_INDEX_ETF_SYMBOLS, *_FACTOR_ETF_SYMBOLS, *_SECTOR_ETF_SYMBOLS}:
            return "index_etf_option"
        return "company_option"
    if instrument_type == "EQUITY":
        if symbol in _INDEX_ETF_SYMBOLS:
            return "index_etf"
        if symbol in _FACTOR_ETF_SYMBOLS:
            return "factor_etf"
        if symbol in _SECTOR_ETF_SYMBOLS:
            return "sector_etf"
        return "company_equity"
    return "other"


def _candidate_applicable(
    payload: HedgeCandidateQueryRequest, candidate: Mapping[str, Any]
) -> bool:
    category = str(candidate.get("universe_category") or "other")
    if payload.objective == "Maximum Loss":
        return category in {"index_etf_option", "volatility_option", "company_option"}
    if payload.objective == "Factors":
        return category in {
            "index_etf", "index_future", "factor_future", "factor_etf", "sector_etf"
        }
    if payload.objective == "Greeks" and not _linear_candidates_applicable(payload):
        return category in {"index_etf_option", "volatility_option", "company_option"}
    return True


def _full_evaluation_fit(
    payload: HedgeCandidateQueryRequest,
    candidate: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> float:
    """Rank only from the same full proposal revaluation exposed to the user."""

    # Combine objective improvement with deterioration, basis, cost, carry, and liquidity.
    primary = _full_objective_improvement(payload, evaluation)
    scenario_changes: list[float] = []
    for row in evaluation.get("scenarios") or []:
        before = _finite(row.get("before_pnl_usd"))
        after = _finite(row.get("after_pnl_usd"))
        if before is None or after is None:
            continue
        before_loss, after_loss = max(0.0, -before), max(0.0, -after)
        scenario_changes.append(
            (before_loss - after_loss) / max(before_loss, abs(before), 1.0)
        )
    deterioration = max(
        [0.0, *(-value for value in scenario_changes if value < 0.0)]
    )
    basis = (_finite(candidate.get("basis_risk_score")) or 100.0) / 100.0
    cost = abs(_finite(candidate.get("entry_cost_usd")) or 0.0)
    gross = _finite(
        ((evaluation.get("portfolio") or {}).get("before") or {}).get(
            "gross_market_value_usd"
        )
    )
    cost_score = 1.0 / (1.0 + cost / max(gross or 0.0, 1.0))
    carry = _finite(candidate.get("carry_30d_usd"))
    carry_score = (
        1.0 / (1.0 + abs(carry) / max(gross or 0.0, 1.0))
        if carry is not None
        else 0.0
    )
    liquidity_status = str(
        (evaluation.get("liquidity") or {}).get("status") or "insufficient_data"
    )
    liquidity_score = 0.5 if liquidity_status == "usable" else 0.0
    return _fit_score(
        primary,
        max(0.0, min(1.0, max(basis, deterioration))),
        liquidity_score,
        cost_score,
        carry_score,
    )


def _primary_objective_usable(
    payload: HedgeCandidateQueryRequest, evaluation: Mapping[str, Any]
) -> bool:
    objective = ((evaluation.get("target") or {}).get("objective") or {})
    before = objective.get("before") or {}
    after = objective.get("after") or {}
    params = payload.parsed_objective_parameters()
    if isinstance(params, ScenarioLossParameters):
        return all(
            _finite(row.get("scenario_pnl_usd")) is not None
            and str(row.get("quality_state") or row.get("status"))
            in {"usable", "usable_with_warnings"}
            and (
                not row.get("unmodeled_position_ids")
                or set(row.get("unmodeled_position_ids") or [])
                <= set(row.get("excluded_position_ids") or [])
            )
            and not row.get("material_unmodeled_position_ids")
            for row in (before, after)
        )
    if isinstance(params, TailRiskParameters):
        regions = ("far",) if params.hedge_region == "far_tail" else ("near",)
        return all(
            str((side.get(region) or {}).get("status"))
            in {"usable", "usable_with_warnings"}
            for side in (before, after)
            for region in regions
        )
    if isinstance(params, GreeksParameters):
        definitions = {
            "delta": "option_delta_exposure_usd",
            "gamma": "gamma_pnl_for_squared_1pct_move_usd",
            "vega": "vega_per_1pct_usd",
            "theta": "theta_per_day_usd",
            "rho": "rho_per_1pct_usd",
        }
        return all(
            _finite(side.get(definitions[name])) is not None
            for side in (before, after)
            for name in params.selected_greeks
        )
    if isinstance(params, FactorsParameters):
        return all(
            str(side.get("status")) == "usable"
            and not ((side.get("coverage") or {}).get("missing_position_ids") or [])
            and any(item.get("name") == params.factor for item in side.get("items") or [])
            for side in (before, after)
        )
    return all(
        str(side.get("status")) in {"exact_bounded", "unbounded"}
        for side in (before, after)
    )


def _full_objective_improvement(
    payload: HedgeCandidateQueryRequest, evaluation: Mapping[str, Any]
) -> float:
    objective = ((evaluation.get("target") or {}).get("objective") or {})
    before = objective.get("before") or {}
    after = objective.get("after") or {}
    params = payload.parsed_objective_parameters()
    if isinstance(params, ScenarioLossParameters):
        return _loss_improvement(
            _finite(before.get("scenario_pnl_usd")),
            _finite(after.get("scenario_pnl_usd")),
        )
    if isinstance(params, TailRiskParameters):
        region = params.hedge_region
        if region == "far_tail":
            before_value = _finite((before.get("far") or {}).get("expected_shortfall_usd"))
            after_value = _finite((after.get("far") or {}).get("expected_shortfall_usd"))
        else:
            before_value = _finite((before.get("near") or {}).get("var_usd"))
            after_value = _finite((after.get("near") or {}).get("var_usd"))
        return _positive_loss_metric_improvement(before_value, after_value)
    if isinstance(params, GreeksParameters):
        definitions = {
            "delta": "option_delta_exposure_usd",
            "gamma": "gamma_pnl_for_squared_1pct_move_usd",
            "vega": "vega_per_1pct_usd",
            "theta": "theta_per_day_usd",
            "rho": "rho_per_1pct_usd",
        }
        targets = params.targets.model_dump(mode="python")
        scores = [
            _distance_improvement(
                _finite(before.get(definitions[name])),
                _finite(after.get(definitions[name])),
                _finite(targets.get(name)),
            )
            for name in params.selected_greeks
        ]
        usable = [value for value in scores if value is not None]
        return sum(usable) / len(usable) if usable else 0.0
    if isinstance(params, FactorsParameters):
        def exposure(rows: Mapping[str, Any]) -> float | None:
            return next(
                (
                    _finite(item.get("coefficient_usd_per_unit_factor_return"))
                    for item in rows.get("items") or []
                    if item.get("name") == params.factor
                ),
                None,
            )
        return _distance_improvement(
            exposure(before), exposure(after), float(params.target_exposure)
        ) or 0.0
    before_status, after_status = str(before.get("status")), str(after.get("status"))
    if before_status == "unbounded" and after_status == "exact_bounded":
        return 1.0
    if after_status == "unbounded":
        return 0.0
    return _positive_loss_metric_improvement(
        _finite(before.get("maximum_loss_usd")),
        _finite(after.get("maximum_loss_usd")),
    )


def _loss_improvement(before: float | None, after: float | None) -> float:
    if before is None or after is None:
        return 0.0
    before_loss, after_loss = max(0.0, -before), max(0.0, -after)
    return max(0.0, min(1.0, (before_loss - after_loss) / max(before_loss, 1.0)))


def _positive_loss_metric_improvement(
    before: float | None, after: float | None
) -> float:
    if before is None or after is None:
        return 0.0
    return max(0.0, min(1.0, (before - after) / max(abs(before), 1.0)))


def _distance_improvement(
    before: float | None, after: float | None, target: float | None
) -> float | None:
    if before is None or after is None or target is None:
        return None
    distance = abs(before - target)
    return max(0.0, min(1.0, (distance - abs(after - target)) / max(distance, 1.0)))


def _evaluated_target_after(
    payload: HedgeCandidateQueryRequest, evaluation: Mapping[str, Any]
) -> str:
    objective = ((evaluation.get("target") or {}).get("objective") or {})
    after = objective.get("after") or {}
    params = payload.parsed_objective_parameters()
    if isinstance(params, ScenarioLossParameters):
        value = _finite(after.get("scenario_pnl_usd"))
        return f"${value:,.2f} scenario P&L" if value is not None else "Unavailable"
    if isinstance(params, TailRiskParameters):
        value = _finite((after.get("near") or {}).get("var_usd"))
        return f"${value:,.2f} VaR" if value is not None else "Unavailable"
    if isinstance(params, GreeksParameters):
        name = params.selected_greeks[0]
        key = {
            "delta": "option_delta_exposure_usd",
            "gamma": "gamma_pnl_for_squared_1pct_move_usd",
            "vega": "vega_per_1pct_usd",
            "theta": "theta_per_day_usd",
            "rho": "rho_per_1pct_usd",
        }[name]
        value = _finite(after.get(key))
        return f"${value:,.2f} {name}" if value is not None else "Unavailable"
    if isinstance(params, FactorsParameters):
        value = next(
            (
                _finite(item.get("coefficient_usd_per_unit_factor_return"))
                for item in after.get("items") or []
                if item.get("name") == params.factor
            ),
            None,
        )
        return f"${value:,.2f} factor coefficient" if value is not None else "Unavailable"
    status = str(after.get("status") or "insufficient_data")
    value = _finite(after.get("maximum_loss_usd"))
    return f"{status}: ${value:,.2f}" if value is not None else status


def _worst_scenario_after(evaluation: Mapping[str, Any]) -> float | None:
    values = [
        value
        for row in evaluation.get("scenarios") or []
        if (value := _finite(row.get("after_pnl_usd"))) is not None
    ]
    return min(values) if values else None


def _candidate_sort_key(
    payload: HedgeCandidateQueryRequest, candidate: Mapping[str, Any]
) -> tuple[Any, ...]:
    fit = -float(candidate.get("fit_score") or 0.0)
    candidate_id = str(candidate.get("candidate_id") or "")
    return (fit, candidate_id)


def _diverse_candidates(
    payload: HedgeCandidateQueryRequest,
    candidates: Sequence[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda item: _candidate_sort_key(payload, item))
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    # Reserve leading slots for distinct instrument or structure classes.
    seen_classes: set[str] = set()
    for candidate in ranked:
        diversity_class = _candidate_diversity_class(candidate)
        candidate["diversity_class"] = diversity_class
        if diversity_class in seen_classes:
            continue
        selected.append(candidate)
        seen_classes.add(diversity_class)
        if len(selected) == limit:
            return selected
    selected_ids = {str(item.get("candidate_id")) for item in selected}
    selected.extend(
        item for item in ranked
        if str(item.get("candidate_id")) not in selected_ids
    )
    return selected[:limit]


def _candidate_diversity_class(candidate: Mapping[str, Any]) -> str:
    legs = candidate.get("legs") or []
    if not legs:
        return "other"
    kinds = {str(leg.get("instrument_type") or "").upper() for leg in legs}
    if kinds == {"OPTION"}:
        return "convex_option"
    if kinds == {"FUTURE"}:
        root = _future_root((legs[0] or {}).get("symbol"))
        return "micro_future" if root in {"MES", "MNQ", "M2K", "MCL", "MGC"} else "standard_future"
    if len(legs) > 1:
        return "factor_spread"
    return "direct" if float(candidate.get("basis_risk_score") or 0.0) == 0.0 else "proxy"


def _pareto_candidates(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep candidates not dominated on fit, cost, carry, and basis risk."""
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        left = _candidate_dimensions(candidate)
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            right = _candidate_dimensions(other)
            no_worse = all(r >= l for r, l in zip(right, left))
            strictly_better = any(r > l + 1e-12 for r, l in zip(right, left))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            output.append(candidate)
    return output


def _candidate_dimensions(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    fit = _finite(candidate.get("fit_score")) or 0.0
    cost = abs(_finite(candidate.get("entry_cost_usd")) or 0.0)
    carry_raw = _finite(candidate.get("carry_30d_usd"))
    carry = abs(carry_raw) if carry_raw is not None else 1e18
    basis = _finite(candidate.get("basis_risk_score"))
    basis_penalty = basis if basis is not None else 100.0
    return (fit, -cost, -carry, -basis_penalty)


def _candidate_item(
    payload, *, name, description, legs, fit, liquidity, target_after,
    scenario_after, entry_cost, carry, basis, rationale, residuals,
    data_quality, warnings,
) -> dict[str, Any]:
    leg_identity = [
        {key: leg.get(key) for key in ("instrument_type", "side", "quantity", "symbol", "canonical_occ_symbol", "expiration_at", "strike", "mark", "multiplier")}
        for leg in legs
    ]
    candidate_id = "sha256:" + _stable_hash(
        payload.portfolio_snapshot_id,
        json.dumps(payload.target.model_dump(mode="json"), sort_keys=True),
        payload.objective,
        json.dumps(payload.objective_parameters, sort_keys=True, default=str),
        json.dumps(leg_identity, sort_keys=True, default=str),
    )
    universe_category = _candidate_universe_category(legs)
    return {
        "candidate_id": candidate_id,
        "name": name,
        "description": description,
        "fit_score": round(max(0.0, min(100.0, fit)), 2),
        "universe_category": universe_category,
        "universe_priority": _PORTFOLIO_UNIVERSE_PRIORITY[universe_category],
        "liquidity": liquidity,
        "target_after": str(target_after),
        "scenario_pnl_after_usd": scenario_after,
        "entry_cost_usd": entry_cost,
        "carry_30d_usd": carry,
        "basis_risk_score": round(100.0 * basis, 2),
        "rationale": rationale,
        "residuals": residuals,
        "legs": legs,
        "data_quality": data_quality,
        "warnings": warnings,
    }


def _target_after(payload, assessment, *, delta_effect: float, legs: Sequence[Mapping[str, Any]] = ()):
    current = assessment.get("current") or {}
    if payload.objective == "Greeks":
        params = payload.parsed_objective_parameters()
        selected = params.selected_greeks[0] if isinstance(params, GreeksParameters) else "delta"
        key = {
            "delta": "option_delta_exposure_usd", "gamma": "gamma_pnl_for_squared_1pct_move_usd",
            "vega": "vega_per_1pct_usd", "theta": "theta_per_day_usd", "rho": "rho_per_1pct_usd",
        }[selected]
        effects = _candidate_greek_effects(legs)
        effect = effects[{
            "delta": "option_delta_exposure_effect_usd",
            "gamma": "gamma_pnl_for_squared_1pct_move_effect_usd",
            "vega": "vega_per_1pct_effect_usd",
            "theta": "theta_per_day_effect_usd",
            "rho": "rho_per_1pct_effect_usd",
        }[selected]]
        current_value = _finite(current.get(key))
        return (
            f"{current_value + effect:,.2f} {selected} units"
            if current_value is not None
            else f"{selected} target unavailable"
        )
    if payload.objective == "Scenario Loss":
        return f"${_scenario_after(assessment, delta_effect):,.2f} scenario P&L"
    if payload.objective == "Factors":
        params = payload.parsed_objective_parameters()
        target = params.target_exposure if isinstance(params, FactorsParameters) else 0.0
        return f"target beta {target:.3f} (requires full evaluation)"
    if payload.objective == "Maximum Loss":
        params = payload.parsed_objective_parameters()
        target = params.desired_loss_floor_usd if isinstance(params, MaximumLossParameters) else 0.0
        return f"desired floor ${target:,.2f}"
    current_var = assessment.get("var_usd")
    return f"${float(current_var):,.2f} VaR before full evaluation" if _finite(current_var) is not None else "VaR requires full evaluation"


def _scenario_after(assessment: Mapping[str, Any], delta_effect: float) -> float | None:
    scenario = assessment.get("scenario") or {}
    before = _finite(scenario.get("scenario_pnl_usd"))
    shock = _finite((scenario.get("assumptions") or {}).get("default_equity_shock"))
    return before + delta_effect * (shock or 0.0) if before is not None else None


def _candidate_greek_effects(
    legs: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    effects = {
        "option_delta_exposure_effect_usd": 0.0,
        "gamma_pnl_for_squared_1pct_move_effect_usd": 0.0,
        "vega_per_1pct_effect_usd": 0.0,
        "theta_per_day_effect_usd": 0.0,
        "rho_per_1pct_effect_usd": 0.0,
    }
    for leg in legs:
        if leg.get("instrument_type") != "OPTION":
            continue
        sign = 1.0 if leg.get("side") == "BUY" else -1.0
        scale = sign * float(leg.get("quantity") or 0.0) * float(
            leg.get("multiplier") or 0.0
        )
        spot = float(leg.get("spot") or 0.0)
        effects["option_delta_exposure_effect_usd"] += (
            scale * float(leg.get("delta") or 0.0) * spot
        )
        effects["gamma_pnl_for_squared_1pct_move_effect_usd"] += (
            0.5 * scale * float(leg.get("gamma") or 0.0) * (0.01 * spot) ** 2
        )
        effects["vega_per_1pct_effect_usd"] += scale * float(
            leg.get("vega_per_1pct") or 0.0
        )
        effects["theta_per_day_effect_usd"] += scale * float(
            leg.get("theta_per_day") or 0.0
        )
        effects["rho_per_1pct_effect_usd"] += scale * float(
            leg.get("rho_per_1pct") or 0.0
        )
    return effects


def _primary_improvement(
    payload: HedgeAssessmentRequest,
    assessment: Mapping[str, Any],
    target_delta: float,
    delta_effect: float,
    legs: Sequence[Mapping[str, Any]],
) -> float:
    if payload.objective == "Scenario Loss":
        before = _finite((assessment.get("scenario") or {}).get("scenario_pnl_usd"))
        after = _scenario_after(assessment, delta_effect)
        if before is None or after is None:
            return 0.0
        before_loss = max(0.0, -before)
        after_loss = max(0.0, -after)
        return max(0.0, min(1.0, (before_loss - after_loss) / max(before_loss, 1.0)))
    if payload.objective == "Greeks":
        params = payload.parsed_objective_parameters()
        if not isinstance(params, GreeksParameters):
            return 0.0
        effects = _candidate_greek_effects(legs)
        current = assessment.get("current") or {}
        definitions = {
            "delta": ("option_delta_exposure_usd", "option_delta_exposure_effect_usd"),
            "gamma": ("gamma_pnl_for_squared_1pct_move_usd", "gamma_pnl_for_squared_1pct_move_effect_usd"),
            "vega": ("vega_per_1pct_usd", "vega_per_1pct_effect_usd"),
            "theta": ("theta_per_day_usd", "theta_per_day_effect_usd"),
            "rho": ("rho_per_1pct_usd", "rho_per_1pct_effect_usd"),
        }
        improvements = []
        targets = params.targets.model_dump(mode="python")
        for greek in params.selected_greeks:
            current_key, effect_key = definitions[greek]
            before = _finite(current.get(current_key))
            target = _finite(targets.get(greek))
            if before is None or target is None:
                continue
            before_distance = abs(before - target)
            after_distance = abs(before + effects[effect_key] - target)
            improvements.append(
                max(0.0, min(1.0, (before_distance - after_distance) / max(before_distance, 1.0)))
            )
        return sum(improvements) / len(improvements) if improvements else 0.0
    if payload.objective == "Maximum Loss":
        # Discovery cannot certify aggregate payoff improvement. Ranking uses
        # the before/after payoff states from full proposal evaluation below.
        return 0.0
    # Historical-tail and factor candidates use the bounded directional change
    # as a transparent first-pass fit; full proposal evaluation recomputes the
    # empirical distribution and ridge residuals.
    before_distance = abs(target_delta)
    after_distance = abs(target_delta + delta_effect)
    return max(
        0.0,
        min(1.0, (before_distance - after_distance) / max(before_distance, 1.0)),
    )


def _fit_score(improvement, basis, liquidity_score, cost_score, carry_score) -> float:
    non_target = max(0.0, 1.0 - improvement * 0.25)
    robustness = max(0.0, 1.0 - basis)
    score = (
        _RANKING_WEIGHTS["primary_objective_improvement"] * improvement
        + _RANKING_WEIGHTS["non_target_residuals"] * non_target
        + _RANKING_WEIGHTS["scenario_robustness"] * robustness
        + _RANKING_WEIGHTS["entry_cost"] * cost_score
        + _RANKING_WEIGHTS["carry_30d"] * carry_score
        + _RANKING_WEIGHTS["liquidity"] * liquidity_score
        + _RANKING_WEIGHTS["basis_risk"] * (1.0 - basis)
    )
    return 100.0 * score


def _basis_risk(symbol: str, snapshot: PortfolioSnapshot) -> float:
    target_symbols = {
        item.underlying_symbol or item.exact_symbol for item in snapshot.positions
    }
    if symbol in target_symbols:
        return 0.0
    sector_symbols = {
        _SECTOR_PROXIES.get(item.sector) for item in snapshot.positions
    }
    if symbol in sector_symbols:
        return 0.20
    if symbol in {"SPY", "QQQ"}:
        return 0.35
    return 0.60


def _option_liquidity(structure):
    scores = []
    warnings = []
    for _, row in structure:
        bid, ask, mark = _finite(row.get("bid")), _finite(row.get("ask")), _positive(row.get("mark"))
        oi = _finite(row.get("open_interest"))
        if bid is None or ask is None or mark is None or oi is None:
            warnings.append("option_liquidity_inputs_incomplete")
            continue
        spread = (ask - bid) / mark if mark else math.inf
        scores.append(1.0 if oi >= 1000 and spread <= 0.10 else 0.65 if oi >= 100 and spread <= 0.25 else 0.3)
    if not scores:
        return "Insufficient data", 0.0, _dedupe(warnings or ["option_liquidity_inputs_incomplete"])
    score = min(scores)
    return ("High" if score >= 0.8 else "Medium" if score >= 0.5 else "Low"), score, _dedupe(warnings)


def _public_resolved_leg(leg, position, *, source, spot):
    result = leg.model_dump(mode="json")
    result.update({
        "mark": position.mark, "multiplier": position.multiplier, "spot": spot,
        "expiration_at": (
            position.expiration_at.isoformat() if position.expiration_at else None
        ),
        "delta": position.delta if position.asset_type == "OPTION" else 1.0,
        "gamma": position.gamma if position.asset_type == "OPTION" else 0.0,
        "vega_per_1pct": position.vega_per_1pct if position.asset_type == "OPTION" else 0.0,
        "theta_per_day": position.theta_per_day if position.asset_type == "OPTION" else 0.0,
        "rho_per_1pct": position.rho_per_1pct if position.asset_type == "OPTION" else 0.0,
        "implied_volatility": position.implied_volatility if position.asset_type == "OPTION" else None,
        "quote_as_of": position.quote_as_of.isoformat() if position.quote_as_of else None,
        "source": source,
    })
    return result


def _pricing_audit(leg, mark, source, quote_as_of, enrichment):
    return {
        "client_leg_id": leg.client_leg_id,
        "mark": mark,
        "mark_source": source,
        "client_supplied": source == "client_supplied_mark",
        "quote_as_of": quote_as_of.isoformat() if quote_as_of else None,
        "enrichment_source": enrichment,
        "analysis_only": True,
    }


def _factor_changes(before, after):
    left = {item["name"]: item for item in before.get("items", [])}
    right = {item["name"]: item for item in after.get("items", [])}
    return [{
        "name": name,
        "beta_before": (left.get(name) or {}).get("beta"),
        "beta_after": (right.get(name) or {}).get("beta"),
        "beta_change": _difference((right.get(name) or {}).get("beta"), (left.get(name) or {}).get("beta")),
        "estimated_pnl_change_for_plus_1pct_factor_move_usd": _difference(
            (right.get(name) or {}).get("estimated_pnl_for_plus_1pct_factor_move_usd"),
            (left.get(name) or {}).get("estimated_pnl_for_plus_1pct_factor_move_usd"),
        ),
    } for name in sorted(set(left) | set(right))]


def _residuals(before_metrics, after_metrics, before_factors, after_factors):
    rows = []
    for key in (
        "directional_delta_exposure_usd", "option_delta_exposure_usd",
        "gamma_pnl_for_squared_1pct_move_usd", "vega_per_1pct_usd",
        "theta_per_day_usd", "rho_per_1pct_usd",
    ):
        before, after = _finite(before_metrics.get(key)), _finite(after_metrics.get(key))
        if before is not None and after is not None:
            rows.append({
                "name": key, "before": before, "after": after, "change": after - before,
                "absolute_risk_change": abs(after) - abs(before),
            })
    for item in _factor_changes(before_factors, after_factors):
        before, after = _finite(item.get("beta_before")), _finite(item.get("beta_after"))
        if before is not None and after is not None:
            rows.append({
                "name": f"factor:{item['name']}", "before": before, "after": after,
                "change": after - before, "absolute_risk_change": abs(after) - abs(before),
            })
    return sorted(rows, key=lambda item: (-item["absolute_risk_change"], item["name"]))


def _numeric_changes(before, after):
    return {
        key: _difference(after.get(key), before.get(key))
        for key in sorted(set(before) & set(after))
        if _finite(before.get(key)) is not None and _finite(after.get(key)) is not None
    }


def _difference(after, before):
    left, right = _finite(after), _finite(before)
    return left - right if left is not None and right is not None else None


def _tail_confidences(payload):
    params = payload.parsed_objective_parameters()
    return (params.near_confidence, params.far_confidence) if isinstance(params, TailRiskParameters) else (0.95, 0.975)


def _insufficient_tail(confidence, reason):
    return {
        "status": "insufficient_data", "confidence": confidence, "reason": reason,
        "var_usd": None, "expected_shortfall_usd": None, "distribution": [],
        "warnings": ["tail_history_insufficient"],
    }


def _dominant_symbol(positions):
    exposures = _position_exposures(positions)
    return max(exposures, key=lambda symbol: (abs(exposures[symbol]), symbol)) if exposures else None


def payload_date(payload, snapshot):
    return snapshot.as_of.date()


def _nested_insufficient(data):
    return all(
        (data.get(name) or {}).get("status") == "insufficient_data"
        for name in ("tail_risk", "factors", "maximum_loss")
    )


def _stable_hash(*values: Any) -> str:
    return hashlib.sha256("\x1f".join(str(item) for item in values).encode("utf-8")).hexdigest()


def _merge_historical_revaluations(
    base: HistoricalRevaluationResult,
    additions: HistoricalRevaluationResult,
) -> HistoricalRevaluationResult:
    """Append proposed-leg history without repricing the unchanged portfolio."""

    mapping_fields = (
        "unit_pnl",
        "scenario_pnl",
        "unit_methods",
        "scenario_methods",
        "unit_symbols",
        "unit_interval_methods",
        "scenario_provenance",
        "return_diagnostics",
    )
    # Append proposed-leg history while retaining the base portfolio's cached evidence.
    update: dict[str, Any] = {
        field: {**getattr(base, field), **getattr(additions, field)}
        for field in mapping_fields
    }
    update.update({
        "position_coverage": tuple((*base.position_coverage, *additions.position_coverage)),
        "missing_position_ids": tuple(sorted(set(base.missing_position_ids) | set(additions.missing_position_ids))),
        "proxy_mappings": tuple((*base.proxy_mappings, *additions.proxy_mappings)),
        "warnings": tuple(_dedupe((*base.warnings, *additions.warnings))),
    })
    return base.model_copy(update=update)


def _dedupe(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value not in (None, "")))


def _collect_nested_warnings(value: Any) -> list[str]:
    output: list[str] = []
    visited: set[int] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            for key, child in current.items():
                if key == "warnings" and isinstance(child, (list, tuple)):
                    output.extend(
                        str(item) for item in child if item not in (None, "")
                    )
                elif isinstance(child, (Mapping, list, tuple)):
                    stack.append(child)
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            stack.extend(
                child
                for child in current
                if isinstance(child, (Mapping, list, tuple))
            )
    return _dedupe(output)


def _compact_tail_risk(
    value: Any, *, include_distributions: bool = False
) -> dict[str, Any]:
    """Keep candidate audit facts without repeating historical distributions."""

    source = value if isinstance(value, Mapping) else {}
    compact: dict[str, Any] = {}
    for key, item in source.items():
        if isinstance(item, Mapping):
            compact[key] = {
                field: item.get(field)
                for field in (
                    "status",
                    "confidence",
                    "var_usd",
                    "expected_shortfall_usd",
                    "observations",
                    "minimum_observations",
                    "coverage",
                    "warnings",
                    "failure_causes",
                )
                if field in item
            }
        elif include_distributions or key not in {"distribution_before", "distribution_after"}:
            compact[key] = item
    return compact


def _compact_primary_objective(value: Any) -> dict[str, Any]:
    """Preserve objective facts while excluding repeated audit-sized payloads."""

    source = dict(value) if isinstance(value, Mapping) else {}
    objective = str(source.get("objective") or "")
    if objective == "VaR / Tail Risk":
        for side_name in ("before", "after"):
            side = source.get(side_name)
            if not isinstance(side, Mapping):
                continue
            source[side_name] = _compact_tail_risk(side)
    elif objective == "Scenario Loss":
        compacted = _compact_scenario_rows(
            [{"before": source.get("before"), "after": source.get("after")}]
        )
        if compacted:
            source["before"] = compacted[0].get("before", {})
            source["after"] = compacted[0].get("after", {})
    elif objective == "Factors":
        # Full factor provenance remains available once in the assessment
        # response.  Proposal effects need only the selected-target factor
        # values and compact quality facts; repeating unit histories here can
        # inflate one response by tens of megabytes.
        for side_name in ("before", "after"):
            side = source.get(side_name)
            if not isinstance(side, Mapping):
                continue
            source[side_name] = {
                field: side.get(field)
                for field in (
                    "status",
                    "reason",
                    "items",
                    "coverage",
                    "warnings",
                    "failure_causes",
                )
                if field in side
            }
    return source


def _compact_scenario_rows(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []

    def compact_side(side: Any) -> dict[str, Any]:
        item = side if isinstance(side, Mapping) else {}
        return {
            field: item.get(field)
            for field in (
                "scenario_pnl_usd",
                "scenario_loss_usd",
                "status",
                "quality_state",
                "warnings",
                "excluded_position_ids",
                "unmodeled_position_ids",
                "material_unmodeled_position_ids",
            )
            if field in item
        }

    return [
        {
            key: (
                compact_side(row.get(key))
                if key in {"before", "after", "target_before", "target_after"}
                else row.get(key)
            )
            for key in (
                "scenario_id",
                "name",
                "definition",
                "before_pnl_usd",
                "after_pnl_usd",
                "change_usd",
                "before",
                "after",
                "target_before",
                "target_after",
            )
            if key in row
        }
        for row in rows
        if isinstance(row, Mapping)
    ]


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


def _supported_deliverable_shares(row: Mapping[str, Any]) -> float | None:
    terms = row.get("instrument_terms")
    deliverable = terms.get("deliverable") if isinstance(terms, Mapping) else None
    if not isinstance(deliverable, Mapping) or not deliverable.get("fully_supported"):
        return None
    return _positive(deliverable.get("shares_per_contract"))


def _dividend_event_date(value: Mapping[str, Any]) -> date | None:
    try:
        return date.fromisoformat(
            str(value.get("ex_dividend_date") or value.get("date") or "")[:10]
        )
    except ValueError:
        return None


def _is_ordinary_cash_dividend(value: Mapping[str, Any]) -> bool:
    classification = str(
        value.get("distribution_type") or value.get("type") or "cash"
    ).lower()
    return not any(
        marker in classification
        for marker in ("special", "return of capital", "stock dividend")
    )


def _datetime(value: Any) -> datetime | None:
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


def _date_value(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[0:10])
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}


def _fresh(source_as_of: datetime, valuation_at: datetime, maximum_age_seconds: float) -> bool:
    age = (valuation_at - source_as_of).total_seconds()
    return -60.0 <= age <= maximum_age_seconds


_FUTURE_MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}
_FUTURE_PRODUCT_MONTHS = {
    "ES": (3, 6, 9, 12),
    "MES": (3, 6, 9, 12),
    "NQ": (3, 6, 9, 12),
    "MNQ": (3, 6, 9, 12),
    "RTY": (3, 6, 9, 12),
    "M2K": (3, 6, 9, 12),
    "ZN": (3, 6, 9, 12),
    "CL": tuple(range(1, 13)),
    "MCL": tuple(range(1, 13)),
    "GC": (2, 4, 6, 8, 10, 12),
    "MGC": (2, 4, 6, 8, 10, 12),
    "6E": (3, 6, 9, 12),
}


def _canonical_future_contract(value: Any) -> str:
    return str(value or "").strip().upper().removeprefix("/")


def _future_root(value: Any) -> str:
    contract = _canonical_future_contract(value)
    return next(
        (
            root
            for root in sorted(_FUTURE_PRODUCT_MONTHS, key=len, reverse=True)
            if contract.startswith(root)
        ),
        "",
    )


def _objective_future_roots(payload: HedgeCandidateQueryRequest) -> set[str]:
    params = payload.parsed_objective_parameters()
    if not isinstance(params, FactorsParameters):
        return {"ES", "MES", "NQ", "MNQ", "RTY", "M2K"}
    return {
        "Market": {"ES", "MES"},
        "Technology/growth tilt": {"NQ", "MNQ"},
        "Size": {"RTY", "M2K"},
        "Long duration": {"ZN"},
        "Intermediate duration": {"ZN"},
        "Gold": {"GC", "MGC"},
        "Oil": {"CL", "MCL"},
        "U.S. dollar": {"6E"},
    }.get(params.factor, set())


def _bounded_hedge_quantity(
    target_exposure: float,
    per_unit_exposure: float,
    *,
    maximum_quantity: int,
    maximum_overhedge_fraction: float,
) -> float | None:
    """Compatibility helper returning the closest feasible full-target integer."""

    quantities = _bounded_hedge_quantities(
        target_exposure,
        per_unit_exposure,
        maximum_quantity=maximum_quantity,
        maximum_overhedge_fraction=maximum_overhedge_fraction,
    )
    if not quantities:
        return None
    target = abs(float(target_exposure))
    unit = abs(float(per_unit_exposure))
    return min(quantities, key=lambda quantity: (abs(quantity * unit - target), quantity))


def _bounded_hedge_quantities(
    target_exposure: float,
    per_unit_exposure: float,
    *,
    maximum_quantity: int,
    maximum_overhedge_fraction: float,
) -> tuple[float, ...]:
    """Enumerate every admissible integer size for full objective revaluation."""

    target = abs(float(target_exposure))
    unit = abs(float(per_unit_exposure))
    if target <= 0 or unit <= 0 or maximum_quantity < 1:
        return ()
    maximum_effect = target * (1.0 + maximum_overhedge_fraction)
    return tuple(
        float(quantity)
        for quantity in range(1, maximum_quantity + 1)
        if quantity * unit <= maximum_effect + 1e-9
    )


def _objective_sizing_exposure(
    payload: HedgeCandidateQueryRequest,
    assessment: Mapping[str, Any],
    target_delta: float,
    per_unit_exposure: float,
) -> float:
    if abs(target_delta) > 0:
        return abs(target_delta)
    params = payload.parsed_objective_parameters()
    current = assessment.get("current") or {}
    if isinstance(params, GreeksParameters):
        keys = {
            "delta": "option_delta_exposure_usd",
            "gamma": "gamma_pnl_for_squared_1pct_move_usd",
            "vega": "vega_per_1pct_usd",
            "theta": "theta_per_day_usd",
            "rho": "rho_per_1pct_usd",
        }
        distances = []
        for greek in params.selected_greeks:
            before = _finite(current.get(keys[greek]))
            target = _finite(getattr(params.targets, greek)) or 0.0
            if before is not None:
                distances.append(abs(before - target))
        if distances:
            return max(max(distances), per_unit_exposure)
    if isinstance(params, ScenarioLossParameters):
        loss = _finite((assessment.get("scenario") or {}).get("scenario_loss_usd"))
        if loss is not None and loss > 0:
            return loss
    if isinstance(params, TailRiskParameters):
        tail = assessment.get("tail_risk") or {}
        region = tail.get("far") if params.hedge_region == "far_tail" else tail.get("near")
        value = _finite((region or {}).get(
            "expected_shortfall_usd" if params.hedge_region == "far_tail" else "var_usd"
        ))
        if value is not None and value > 0:
            return value
    if isinstance(params, MaximumLossParameters):
        maximum_loss = _finite((assessment.get("maximum_loss") or {}).get("maximum_loss_usd"))
        if maximum_loss is not None and maximum_loss > 0:
            return maximum_loss
    gross = _finite(current.get("gross_market_value_usd"))
    return max(gross or 0.0, per_unit_exposure * 4.0, 1.0)


def _future_contract_ladders(
    as_of: date, *, depth: int | None = None
) -> dict[str, list[str]]:
    """Return bounded product-specific contract ladders for CME validation.

    The ladder intentionally does not guess which contract is active.  It only
    models each product's listing cycle; CME ``last_trade_at`` metadata selects
    the first still-eligible contract below.  This matters for monthly energy
    contracts such as CL, whose named delivery month is not simply the next
    calendar month.
    """
    ladders: dict[str, list[str]] = {}
    for root, listed_months in _FUTURE_PRODUCT_MONTHS.items():
        # Two adjacent quarterly contracts cover a roll.  CL needs three
        # adjacent monthly delivery contracts because its nearby contract can
        # stop trading before the named delivery month begins.
        product_depth = depth if depth is not None else (3 if root == "CL" else 2)
        bounded_depth = min(8, max(2, int(product_depth)))
        contracts: list[str] = []
        for offset in range(0, 30):
            absolute_month = as_of.year * 12 + (as_of.month - 1) + offset
            year, zero_based_month = divmod(absolute_month, 12)
            month = zero_based_month + 1
            if month not in listed_months:
                continue
            contracts.append(
                f"{root}{_FUTURE_MONTH_CODES[month]}{year % 100:02d}"
            )
            if len(contracts) == bounded_depth:
                break
        ladders[root] = contracts
    return ladders


def _select_active_future_contracts(
    ladders: Mapping[str, Sequence[str]],
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of: datetime,
    roll_buffer_days: int,
) -> tuple[list[str], list[str]]:
    """Select one eligible contract per root from exact CME specifications."""
    by_contract = {
        _canonical_future_contract(
            row.get("canonical_contract") or row.get("requested_contract")
        ): row
        for row in rows
        if _canonical_future_contract(
            row.get("canonical_contract") or row.get("requested_contract")
        )
    }
    # Avoid contracts too close to their last trading date under the roll buffer.
    cutoff = as_of.astimezone(timezone.utc) + timedelta(
        days=max(0, int(roll_buffer_days))
    )
    selected: list[str] = []
    warnings: list[str] = []
    for root, contracts in ladders.items():
        active = None
        for raw_contract in contracts:
            contract = _canonical_future_contract(raw_contract)
            row = by_contract.get(contract) or {}
            last_trade_at = _datetime(row.get("last_trade_at"))
            if (
                row.get("status") in {"ok", "partial"}
                and _positive(row.get("price_multiplier")) is not None
                and last_trade_at is not None
                and last_trade_at > cutoff
            ):
                active = contract
                break
        if active is None:
            warnings.append(f"{root}:active_futures_contract_unavailable")
        else:
            selected.append(active)
    return selected, warnings


__all__ = ["HedgeLabService"]
