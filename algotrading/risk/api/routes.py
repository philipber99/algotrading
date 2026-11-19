"""Versioned portfolio analytics, Tian, health, and transaction routes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from threading import RLock
from typing import Any, Literal, Mapping

from fastapi import APIRouter, Path, Query, Request, Response

from algotrading.refdata.api.live_models import (
    OptionContractTermsRequest,
    OptionContractTermsResponse,
    OptionMarketHistoryRequest,
    OptionMarketHistoryResponse,
)

from algotrading.risk.clusters.hierarchical import cluster_risk_units
from algotrading.risk.exposures.factors import FACTOR_PROXY_SYMBOLS, ridge_factor_exposure
from algotrading.risk.exposures.portfolio import delta_equivalent_usd, portfolio_kpis, sector_exposures
from algotrading.risk.history.revaluation import (
    CurrentPortfolioHistoricalRevaluationEngine,
    historical_risk_position_eligible,
)
from algotrading.risk.materiality import assess_omission_materiality
from algotrading.risk.market_inputs.rates import interpolate_risk_free_rate
from algotrading.risk.pricing.tian_service import (
    IvGreeksRequest,
    IvGreeksResponse,
    TianRepriceRequest,
    TianRepriceResponse,
)
from algotrading.risk.scenarios.engine import (
    DEFAULT_SCENARIOS,
    ScenarioDefinition,
    ScenarioEngine,
    omission_requires_fail_closed,
)
from algotrading.risk.scenarios.etf_response import (
    ETF_RESPONSE_PROXY_SYMBOLS,
    build_etf_history_shocks,
)
from algotrading.risk.scenarios.sector_rotation import (
    SECTOR_PROXY_SYMBOLS,
    build_sector_rotation_shocks,
)
from algotrading.risk.transactions.models import (
    SUPPORTED_TRANSACTION_TYPES,
    TransactionStatusResponse,
    TransactionsResponse,
)
from algotrading.risk.transactions.repository import (
    InvalidCursor,
    TransactionRepository,
    TransactionRepositoryUnavailable,
)
from algotrading.risk.pnl.analyzer import (
    InvalidPnlAnalysis,
    PnlSelectionNotFound,
    analyze_selection,
)
from algotrading.risk.pnl.models import (
    PnlAnalyzerRequest,
    PnlAnalyzerResponse,
    PnlMarketHistoryPoint,
    PnlMarketHistoryRequest,
    PnlMarketHistoryResponse,
    PnlMarketHistorySeries,
    PnlTradeQueryRequest,
    PnlTradeQueryResponse,
)
from algotrading.risk.market_inputs.client import ReferenceDataClientError
from algotrading.risk.pnl.valuation import (
    PnlMarkHistoryUnavailable,
    PnlReconstructionLimitExceeded,
    query_trades,
)
from algotrading.risk.option_prices.service import (
    HybridOptionHistoryClient,
    RiskOptionHistoryService,
)
from algotrading.risk.payoff.engine import (
    certify_maximum_loss,
    option_intrinsic_per_contract,
)
from algotrading.risk.tail.historical import historical_var_es
from algotrading.risk.alerts.models import (
    AlertEpisode,
    AlertEvaluationRequest,
    AlertPatchRequest,
    AlertsResponse,
)
from algotrading.risk.alerts.persistence import (
    AlertNotFound,
    AlertVersionConflict,
)
from algotrading.risk.positions.strategies import (
    StrategyGroup,
    StrategyGroupMutation,
)

from .errors import (
    AlertNotFoundError,
    AlertVersionConflictError,
    InvalidRiskRequestError,
    PnlMarkHistoryUnavailableError,
    PnlReconstructionLimitError,
    PnlSelectionNotFoundError,
    ReferenceDataUnavailableError,
    SnapshotMismatchError,
    SnapshotUnavailableError,
    TransactionRepositoryError,
)
from .portfolio_models import (
    HedgeChainRequest,
    HedgeExpirationRequest,
    PortfolioAnalysisResponse,
    PortfolioSnapshotResponse,
    ScenarioEvaluateRequest,
    RiskConfigurationPatch,
)
from .hedge_models import (
    HedgeAssessmentRequest,
    HedgeCandidateQueryRequest,
    HedgeProposalRequest,
)
from .historical_cache import historical_revaluation_cache_key
from .portfolio_service import NoSnapshotLoaded, SnapshotMismatch


router = APIRouter(prefix="/api/v1")
_HISTORICAL_CACHE_INIT_LOCK = RLock()


@router.get("/portfolio/snapshot", response_model=PortfolioSnapshotResponse)
def portfolio_snapshot(
    request: Request,
    account_id: list[str] = Query(default=["all"]),
    refresh: bool = False,
) -> PortfolioSnapshotResponse:
    try:
        return request.app.state.portfolio_service.snapshot(
            _split_values(account_id), refresh=refresh
        )
    except NoSnapshotLoaded as exc:
        raise SnapshotUnavailableError(str(exc)) from exc
    except ValueError as exc:
        raise InvalidRiskRequestError(str(exc)) from exc


@router.get("/portfolio/dashboard", response_model=PortfolioAnalysisResponse)
def portfolio_dashboard(
    request: Request,
    account_id: list[str] = Query(default=["all"]),
) -> PortfolioAnalysisResponse:
    response, _ = _required_snapshot(request, _split_values(account_id))
    return _analysis(
        response,
        "risk_dashboard_bundle_v1",
        {
            "kpis": response.data.kpis.model_dump(mode="json"),
            "risk_calendar": _risk_calendar(response),
            "attention_queue": {
                "status": "unavailable",
                "reason": "alerts_have_not_been_evaluated_for_this_snapshot",
                "items": [],
            },
            "sector_risk": sector_exposures(
                response.data.positions,
                portfolio_value_usd=response.data.kpis.portfolio_value_usd,
                context=request.app.state.portfolio_service.context,
            ),
        },
    )


@router.get("/portfolio/exposures", response_model=PortfolioAnalysisResponse)
def portfolio_exposures(
    request: Request,
    account_id: list[str] = Query(default=["all"]),
    asset_scope: Literal["all", "equities", "options", "futures"] = "all",
    include_factors: bool = True,
) -> PortfolioAnalysisResponse:
    response, _ = _required_snapshot(request, _split_values(account_id))
    response = _response_for_asset_scope(
        response, asset_scope, request.app.state.portfolio_service.context
    )
    factor_result = (
        _factor_analysis(request, response)
        if include_factors
        else {
            "status": "not_requested",
            "reason": "factor_history_excluded_from_lightweight_request",
            "items": [],
        }
    )
    return _analysis(
        response,
        "directional_exposure_v1",
        {
            "kpis": response.data.kpis.model_dump(mode="json"),
            "sectors": sector_exposures(
                response.data.positions,
                portfolio_value_usd=response.data.kpis.portfolio_value_usd,
                context=request.app.state.portfolio_service.context,
            ),
            "factors": factor_result,
        },
    )


@router.get("/reference/companies/{symbol}", response_model=dict[str, Any])
def reference_company_profile(
    request: Request,
    symbol: str = Path(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9.-]+$"),
) -> dict[str, Any]:
    """Expose RefData profiles, retaining missing symbols as Unclassified."""
    try:
        profile = request.app.state.portfolio_service.reference_client.company_profile(symbol)
    except ValueError as exc:
        raise InvalidRiskRequestError(str(exc)) from exc
    except ReferenceDataClientError:
        canonical = symbol.strip().upper()
        profile = {
            "symbol": canonical,
            "display_name": canonical,
            "sector": "Unclassified",
            "industry": None,
            "profile_status": "unavailable_classified_unclassified",
        }
    if not str(profile.get("sector") or "").strip():
        profile = {
            **profile,
            "sector": "Unclassified",
            "profile_status": "missing_sector_classified_unclassified",
        }
    return {"data": profile}


@router.get("/portfolio/scenarios", response_model=PortfolioAnalysisResponse)
def scenario_library(
    request: Request,
    account_id: list[str] = Query(default=["all"]),
) -> PortfolioAnalysisResponse:
    response, _ = _required_snapshot(request, _split_values(account_id))
    return _analysis(
        response,
        "risk_scenario_library_v1",
        {"items": [item.model_dump(mode="json") for item in DEFAULT_SCENARIOS]},
    )


@router.post(
    "/portfolio/scenarios/evaluate", response_model=PortfolioAnalysisResponse
)
def evaluate_scenario(
    payload: ScenarioEvaluateRequest,
    request: Request,
    account_id: list[str] = Query(default=["all"]),
) -> PortfolioAnalysisResponse:
    # Resolve the requested snapshot before adding any custom scenario shocks.
    response, snapshot = _required_snapshot(
        request, _split_values(account_id), payload.portfolio_snapshot_id
    )
    selected = next(
        (item for item in DEFAULT_SCENARIOS if item.scenario_id == payload.scenario_id),
        None,
    )
    underlying_shocks = dict(payload.underlying_shocks)
    rotation_diagnostics: dict[str, Any] = {}
    if payload.sector_rotation_sector and payload.sector_rotation_move is not None:
        symbols = {
            position.underlying_symbol or position.exact_symbol
            for position in snapshot.positions
            if position.asset_type
            in {"EQUITY", "ETF", "MUTUAL_FUND", "OPTION", "FUTURE"}
        }
        try:
            returns = _load_returns(
                request,
                symbols | set(SECTOR_PROXY_SYMBOLS.values()),
                response.as_of,
            )
            underlying_shocks, rotation_diagnostics = build_sector_rotation_shocks(
                snapshot.positions,
                returns,
                selected_sector=payload.sector_rotation_sector,
                sector_move=payload.sector_rotation_move,
                default_equity_shock=payload.default_equity_shock,
                explicit_shocks=underlying_shocks,
                minimum_observations=int(
                    request.app.state.portfolio_service.context.get(
                        "risk.scenarios.sector_rotation_minimum_observations", 126
                    )
                ),
                maximum_observations=252,
            )
        except ValueError as exc:
            raise InvalidRiskRequestError(str(exc)) from exc
    if selected is None and payload.path_rule == "explicit_dated_points":
        horizon_at = response.as_of + timedelta(days=payload.horizon_calendar_days)
        path_times = sorted(payload.dated_path_points)
        if any(at <= response.as_of or at > horizon_at for at in path_times):
            raise InvalidRiskRequestError(
                "explicit scenario path points must be after the snapshot and no later than the horizon"
            )
        crossing_expiries = [
            item.expiration_at
            for item in snapshot.positions
            if item.asset_type == "OPTION"
            and item.expiration_at is not None
            and response.as_of < item.expiration_at <= horizon_at
        ]
        if any(not any(at <= expiry for at in path_times) for expiry in crossing_expiries):
            raise InvalidRiskRequestError(
                "explicit scenario path must include a causal point at or before every crossed expiry"
            )
    definition = selected or ScenarioDefinition(
        scenario_id=payload.scenario_id or "custom",
        name=payload.name or "Custom scenario",
        underlying_shocks=underlying_shocks,
        exact_underlying_shocks=payload.exact_underlying_shocks,
        broad_equity_shock=payload.broad_equity_shock,
        sector_shocks=payload.sector_shocks,
        futures_root_shocks=payload.futures_root_shocks,
        commodity_root_shocks=payload.commodity_root_shocks,
        fx_pair_shocks=payload.fx_pair_shocks,
        yield_curve_basis_points=payload.yield_curve_basis_points,
        default_equity_shock=payload.default_equity_shock,
        iv_points=payload.iv_points,
        rate_basis_points=payload.rate_basis_points,
        horizon_calendar_days=payload.horizon_calendar_days,
        path_rule=payload.path_rule,
        dated_path_points=payload.dated_path_points,
        sector_rotation_sector=payload.sector_rotation_sector,
        sector_rotation_move=payload.sector_rotation_move,
        sector_rotation_diagnostics=rotation_diagnostics,
    )
    # Translate economic shocks through held ETFs' observed proxy responses.
    etf_symbols = {
        position.exact_symbol
        for position in snapshot.positions
        if position.asset_type == "ETF"
    }
    if etf_symbols:
        explicit = {
            **definition.underlying_shocks,
            **definition.exact_underlying_shocks,
        }
        try:
            returns = _load_returns(
                request,
                etf_symbols | set(ETF_RESPONSE_PROXY_SYMBOLS),
                response.as_of,
            )
            modeled_shocks, etf_diagnostics = build_etf_history_shocks(
                snapshot.positions,
                returns,
                broad_equity_shock=(
                    definition.broad_equity_shock
                    if definition.broad_equity_shock is not None
                    else definition.default_equity_shock
                ),
                sector_shocks=definition.sector_shocks,
                explicit_shocks=explicit,
                minimum_observations=int(
                    request.app.state.portfolio_service.context.get(
                        "risk.scenarios.etf_response_minimum_observations", 126
                    )
                ),
                maximum_observations=252,
                ridge_alpha=float(
                    request.app.state.portfolio_service.context.get(
                        "risk.scenarios.etf_response_ridge_alpha", 1e-3
                    )
                ),
                minimum_r_squared=float(
                    request.app.state.portfolio_service.context.get(
                        "risk.scenarios.etf_response_minimum_r_squared", 0.20
                    )
                ),
                maximum_absolute_shock=float(
                    request.app.state.portfolio_service.context.get(
                        "risk.scenarios.maximum_absolute_equity_shock", 0.50
                    )
                ),
            )
            exact_symbols = set(definition.exact_underlying_shocks)
            updated_shocks = dict(definition.underlying_shocks)
            updated_shocks.update(
                {
                    symbol: shock
                    for symbol, shock in modeled_shocks.items()
                    if symbol not in exact_symbols
                }
            )
        except Exception as exc:
            updated_shocks = dict(definition.underlying_shocks)
            etf_diagnostics = {
                "methodology_version": "etf_market_sector_ridge_response_v1",
                "positions": [],
                "unmodeled": [
                    {
                        "symbol": symbol,
                        "reason": "response_history_request_failed",
                        "error_type": type(exc).__name__,
                    }
                    for symbol in sorted(etf_symbols - set(explicit))
                ],
                "unmodeled_symbols": sorted(etf_symbols - set(explicit)),
            }
        definition = definition.model_copy(
            update={
                "underlying_shocks": updated_shocks,
                "etf_response_diagnostics": etf_diagnostics,
            }
        )
    result = ScenarioEngine(request.app.state.portfolio_service.context).evaluate(
        snapshot, definition
    )
    return _analysis(response, definition.methodology_version, result)


@router.get("/portfolio/options-risk", response_model=PortfolioAnalysisResponse)
def options_risk(
    request: Request,
    account_id: list[str] = Query(default=["all"]),
) -> PortfolioAnalysisResponse:
    response, _ = _required_snapshot(request, _split_values(account_id))
    return _analysis(
        response,
        "held_options_risk_v3",
        _options_risk(response, request.app.state.portfolio_service.context),
    )


@router.get("/portfolio/risk-calendar", response_model=PortfolioAnalysisResponse)
def risk_calendar(
    request: Request,
    account_id: list[str] = Query(default=["all"]),
) -> PortfolioAnalysisResponse:
    response, _ = _required_snapshot(request, _split_values(account_id))
    calendar = _enriched_risk_calendar(request, response)
    return _analysis(
        response,
        "current_exposure_event_calendar_v1",
        {
            "events": calendar["events"],
            "label": "Current exposure mapped to upcoming dates",
            "omissions": calendar["omissions"],
        },
    )


@router.get("/portfolio/tail-risk", response_model=PortfolioAnalysisResponse)
def tail_risk(
    request: Request,
    account_id: list[str] = Query(default=["all"]),
    confidence: float = Query(default=0.95, ge=0.5, lt=1.0),
) -> PortfolioAnalysisResponse:
    response, _ = _required_snapshot(request, _split_values(account_id))
    result = _tail_analysis(request, response, confidence)
    return _analysis(
        response,
        str(result.get("method") or "current_portfolio_weighted_historical_simulation_v1"),
        result,
        insufficient=result.get("status") not in {"usable", "usable_with_warnings"},
    )


@router.get("/portfolio/clusters", response_model=PortfolioAnalysisResponse)
def clusters(
    request: Request,
    account_id: list[str] = Query(default=["all"]),
    asset_scope: Literal["all", "equities", "options", "futures"] = "all",
) -> PortfolioAnalysisResponse:
    response, snapshot = _required_snapshot(request, _split_values(account_id))
    response = _response_for_asset_scope(
        response, asset_scope, request.app.state.portfolio_service.context
    )
    selected_types = _asset_types_for_scope(asset_scope)
    if selected_types is not None:
        selected_positions = tuple(item for item in snapshot.positions if item.asset_type in selected_types)
        selected_marked = [item for item in selected_positions if item.asset_type != "FUTURE"]
        selected_values = [
            item.signed_market_value_usd
            for item in selected_marked
            if _position_field_valid(item, "mark")
            and item.signed_market_value_usd is not None
        ]
        selected_value = (
            sum(selected_values)
            if len(selected_values) == len(selected_marked)
            else None
        )
        snapshot = snapshot.model_copy(update={"positions": selected_positions, "portfolio_value_usd": selected_value})
    result = _cluster_analysis(request, snapshot)
    methodology = (
        "hybrid_clusters_v3"
        if result.get("mode") == "hybrid"
        else "correlated_clusters_v2"
    )
    return _analysis(response, methodology, result)


@router.get("/portfolio/pnl-explain", response_model=PortfolioAnalysisResponse)
def pnl_explain(
    request: Request,
    account_id: list[str] = Query(default=["all"]),
) -> PortfolioAnalysisResponse:
    accounts = _split_values(account_id)
    response, _ = _required_snapshot(request, accounts)
    previous = request.app.state.portfolio_service.previous(accounts)
    return _analysis(response, "refresh_anchor_pnl_explain_v2", _pnl_explain(response, previous))


@router.post(
    "/risk-gui/options/market-history/query",
    response_model=OptionMarketHistoryResponse,
    description=(
        "Risk-GUI-only exact option history. Captured broker-position marks are "
        "loaded from Risk MySQL first and Reference Data fills remaining gaps."
    ),
)
def risk_gui_option_market_history(
    payload: OptionMarketHistoryRequest,
    request: Request,
    response: Response,
) -> OptionMarketHistoryResponse:
    # This aggregate Risk endpoint may span more than one upstream request.
    # RiskOptionHistoryService plans exact SQL gaps and chunks only those gaps
    # against the negotiated effective RefData limit.
    result = _option_history_service(request).query(payload)
    response.headers["X-Cache"] = result.meta.cache.upper()
    response.headers["X-Data-As-Of"] = result.meta.as_of.isoformat()
    return result


@router.post(
    "/risk-gui/options/contract-terms/query",
    response_model=OptionContractTermsResponse,
    description=(
        "Risk-GUI proxy for authoritative exact-contract option terms. The GUI "
        "must use these timestamps and multipliers instead of deriving them from OCC text."
    ),
)
def risk_gui_option_contract_terms(
    payload: OptionContractTermsRequest,
    request: Request,
) -> OptionContractTermsResponse:
    result = request.app.state.portfolio_service.reference_client.option_contract_terms(
        payload.symbols
    )
    return OptionContractTermsResponse.model_validate(result)


@router.post(
    "/portfolio/pnl/market-history/query",
    response_model=PnlMarketHistoryResponse,
    description=(
        "Bounded daily comparison history for P&L charts. SQL is queried first; "
        "remote repair is limited to explicitly configured non-universe symbols."
    ),
)
def pnl_market_history(
    payload: PnlMarketHistoryRequest,
    request: Request,
) -> PnlMarketHistoryResponse:
    client = request.app.state.portfolio_service.reference_client
    context = request.app.state.portfolio_service.context
    start = payload.start_at.date()
    end = (payload.end_at - timedelta(microseconds=1)).date()
    warnings: list[str] = []
    try:
        groups = client.bars(
            payload.symbols,
            start=start,
            end=end,
            remote_query=False,
        )
    except TypeError:
        try:
            groups = client.bars(payload.symbols, start=start, end=end)
        except ReferenceDataClientError as exc:
            raise InvalidRiskRequestError(
                "Daily comparison history is unavailable from the SQL-backed reference-data service."
            ) from exc
    except ReferenceDataClientError as exc:
        raise InvalidRiskRequestError(
            "Daily comparison history is unavailable from the SQL-backed reference-data service."
        ) from exc

    by_symbol = {
        str(group.get("symbol") or "").strip().upper(): dict(group)
        for group in groups
    }
    # Limit remote P&L history repair to the explicitly configured symbol set.
    repair_allowlist = {
        str(symbol).strip().upper()
        for symbol in context.get(
            "risk.pnl.remote_daily_bar_symbols", ["SCC", "TSCO", "VOLT"]
        )
        if str(symbol).strip()
    }
    repair_symbols = sorted(
        symbol
        for symbol in set(payload.symbols) & repair_allowlist
        if not (by_symbol.get(symbol) or {}).get("bars")
    )
    if repair_symbols:
        try:
            repaired = client.bars(
                repair_symbols,
                start=start,
                end=end,
                remote_query=True,
            )
        except (ReferenceDataClientError, TypeError):
            warnings.extend(
                f"{symbol}: remote daily-bar repair unavailable"
                for symbol in repair_symbols
            )
        else:
            for group in repaired:
                symbol = str(group.get("symbol") or "").strip().upper()
                if symbol:
                    by_symbol[symbol] = dict(group)

    result: list[PnlMarketHistorySeries] = []
    for symbol in payload.symbols:
        points: list[PnlMarketHistoryPoint] = []
        for bar in (by_symbol.get(symbol) or {}).get("bars") or []:
            try:
                close = float(bar.get("close"))
                at = datetime.fromisoformat(
                    str(bar.get("timestamp") or "").replace("Z", "+00:00")
                )
            except (TypeError, ValueError):
                continue
            if not math.isfinite(close):
                continue
            if at.tzinfo is None:
                at = at.replace(tzinfo=timezone.utc)
            if start <= at.date() <= end:
                points.append(PnlMarketHistoryPoint(
                    at=at,
                    open=_finite(bar.get("open")),
                    high=_finite(bar.get("high")),
                    low=_finite(bar.get("low")),
                    close=close,
                    volume=_finite(bar.get("volume")),
                ))
        points.sort(key=lambda point: point.at)
        if not points:
            warnings.append(f"{symbol}: no daily closing prices in the selected interval")
        result.append(PnlMarketHistorySeries(symbol=symbol, points=points))
    return PnlMarketHistoryResponse(data=result, warnings=warnings)


@router.post(
    "/portfolio/pnl/trades/query",
    response_model=PnlTradeQueryResponse,
)
def pnl_trade_query(
    payload: PnlTradeQueryRequest,
    request: Request,
) -> PnlTradeQueryResponse:
    snapshot, _ = _required_snapshot(
        request, payload.account_ids, payload.portfolio_snapshot_id
    )
    try:
        return query_trades(
            payload,
            repository=_transaction_repository(request),
            snapshot=snapshot,
            reference_client=_pnl_reference_client(request),
            context=request.app.state.portfolio_service.context,
        )
    except PnlMarkHistoryUnavailable as exc:
        raise PnlMarkHistoryUnavailableError(str(exc)) from exc
    except PnlReconstructionLimitExceeded as exc:
        raise PnlReconstructionLimitError(str(exc)) from exc
    except TransactionRepositoryUnavailable as exc:
        raise TransactionRepositoryError() from exc
    except ValueError as exc:
        raise InvalidRiskRequestError(str(exc)) from exc


@router.post(
    "/portfolio/pnl/trades/analyze",
    response_model=PnlAnalyzerResponse,
)
def pnl_trade_analyze(
    payload: PnlAnalyzerRequest,
    request: Request,
) -> PnlAnalyzerResponse:
    snapshot, _ = _required_snapshot(
        request, payload.account_ids, payload.portfolio_snapshot_id
    )
    try:
        return analyze_selection(
            payload,
            repository=_transaction_repository(request),
            snapshot=snapshot,
            reference_client=_pnl_reference_client(request),
            context=request.app.state.portfolio_service.context,
            option_price_repository=getattr(
                request.app.state, "option_price_repository", None
            ),
        )
    except PnlSelectionNotFound as exc:
        raise PnlSelectionNotFoundError(str(exc)) from exc
    except InvalidPnlAnalysis as exc:
        raise InvalidRiskRequestError(str(exc)) from exc
    except PnlMarkHistoryUnavailable as exc:
        raise PnlMarkHistoryUnavailableError(str(exc)) from exc
    except PnlReconstructionLimitExceeded as exc:
        raise PnlReconstructionLimitError(str(exc)) from exc
    except TransactionRepositoryUnavailable as exc:
        raise TransactionRepositoryError() from exc
    except ValueError as exc:
        raise InvalidRiskRequestError(str(exc)) from exc


@router.get("/data-health", response_model=PortfolioAnalysisResponse)
def data_health(
    request: Request,
    account_id: list[str] = Query(default=["all"]),
) -> PortfolioAnalysisResponse:
    response, _ = _required_snapshot(request, _split_values(account_id))
    transaction_ready = request.app.state.transaction_repository is not None
    return _analysis(
        response,
        "risk_data_health_v1",
        {
            "portfolio": response.data_quality.model_dump(mode="json"),
            "capabilities": response.data.capabilities,
            "cache": request.app.state.portfolio_service.cache.stats(),
            "transactions": {
                "repository_ready": transaction_ready,
                "status": "available" if transaction_ready else "unavailable",
            },
            "forbidden_dependencies": {
                "clickhouse": False,
                "options_gui_api": False,
            },
        },
    )


@router.post("/options/iv-greeks", response_model=IvGreeksResponse)
def iv_greeks(payload: IvGreeksRequest, request: Request) -> IvGreeksResponse:
    context = request.app.state.portfolio_service.context
    # Load shared curve evidence only when contracts lack an explicit or configured rate.
    missing_rate = any(contract.risk_free_rate is None for contract in payload.contracts)
    configured = context.get("risk.rates.configured_flat_continuous_rate")
    configured_rate = float(configured) if configured not in (None, "") else None
    curve: dict[str, Any] | None = None
    if missing_rate and configured_rate is None:
        try:
            curve_payload = request.app.state.portfolio_service.reference_client.risk_free_curve(
                payload.valuation_at.date()
            )
            candidate = curve_payload.get("data")
            curve = candidate if isinstance(candidate, dict) else None
        except Exception:
            curve = None
    valuation_at = payload.valuation_at
    if valuation_at.tzinfo is None:
        valuation_at = valuation_at.replace(tzinfo=timezone.utc)
    allow_flat_value = context.get("risk.rates.allow_flat_extrapolation", True)
    allow_flat_extrapolation = (
        allow_flat_value
        if isinstance(allow_flat_value, bool)
        else str(allow_flat_value).strip().lower() in {"1", "true", "yes", "on"}
    )
    contracts = []
    for contract in payload.contracts:
        rate = contract.risk_free_rate if contract.risk_free_rate is not None else configured_rate
        if rate is None and curve is not None:
            expiration_at = contract.expiration_at
            if expiration_at.tzinfo is None:
                expiration_at = expiration_at.replace(tzinfo=timezone.utc)
            tenor_days = max(
                0.0,
                (expiration_at.astimezone(timezone.utc) - valuation_at.astimezone(timezone.utc)).total_seconds() / 86400,
            )
            try:
                rate = interpolate_risk_free_rate(
                    curve,
                    valuation_date=valuation_at.date(),
                    tenor_days=tenor_days,
                    max_age_days=int(context.get("risk.rates.max_age_days", 5)),
                    allow_flat_extrapolation=allow_flat_extrapolation,
                ).continuous_annual_rate_decimal
            except ValueError:
                rate = None
        contracts.append(
            contract.model_copy(
                update={
                    "risk_free_rate": rate,
                    "dividend_yield": (
                        contract.dividend_yield
                        if contract.dividend_yield is not None
                        else float(context.get("risk.dividends.default_continuous_yield", 0.0))
                    ),
                }
            )
        )
    return request.app.state.tian_service.solve(
        payload.model_copy(update={"contracts": contracts})
    )


@router.post("/options/reprice", response_model=TianRepriceResponse)
def reprice_options(payload: TianRepriceRequest, request: Request) -> TianRepriceResponse:
    """Fixed-IV Tian repricing for dated Hedge Analysis response curves."""
    context = request.app.state.portfolio_service.context
    valuation_at = payload.valuation_at
    if valuation_at.tzinfo is None:
        valuation_at = valuation_at.replace(tzinfo=timezone.utc)
    configured = context.get("risk.rates.configured_flat_continuous_rate")
    configured_rate = float(configured) if configured not in (None, "") else None
    curves: dict[Any, dict[str, Any] | None] = {}
    if configured_rate is None and any(item.risk_free_rate is None for item in payload.contracts):
        source_dates = set()
        for item in payload.contracts:
            source_valuation_at = item.source_valuation_at
            if source_valuation_at.tzinfo is None:
                source_valuation_at = source_valuation_at.replace(tzinfo=timezone.utc)
            source_dates.add(source_valuation_at.astimezone(timezone.utc).date())
        for source_date in source_dates:
            try:
                curve_payload = request.app.state.portfolio_service.reference_client.risk_free_curve(
                    source_date
                )
                candidate = curve_payload.get("data")
                curves[source_date] = candidate if isinstance(candidate, dict) else None
            except Exception:
                curves[source_date] = None
    allow_flat_value = context.get("risk.rates.allow_flat_extrapolation", True)
    allow_flat = allow_flat_value if isinstance(allow_flat_value, bool) else str(allow_flat_value).strip().lower() in {"1", "true", "yes", "on"}
    contracts = []
    for contract in payload.contracts:
        rate = contract.risk_free_rate if contract.risk_free_rate is not None else configured_rate
        source_valuation_at = contract.source_valuation_at
        if source_valuation_at.tzinfo is None:
            source_valuation_at = source_valuation_at.replace(tzinfo=timezone.utc)
        source_date = source_valuation_at.astimezone(timezone.utc).date()
        curve = curves.get(source_date)
        if rate is None and curve is not None:
            expiration_at = contract.expiration_at
            if expiration_at.tzinfo is None:
                expiration_at = expiration_at.replace(tzinfo=timezone.utc)
            tenor_days = max(0.0, (expiration_at.astimezone(timezone.utc) - valuation_at.astimezone(timezone.utc)).total_seconds() / 86400)
            try:
                rate = interpolate_risk_free_rate(
                    curve,
                    # Curves are published observations, not forecasts. Use
                    # the contract's source valuation date for curve
                    # freshness while pricing the remaining future tenor.
                    valuation_date=source_date,
                    tenor_days=tenor_days,
                    max_age_days=int(context.get("risk.rates.max_age_days", 5)),
                    allow_flat_extrapolation=allow_flat,
                ).continuous_annual_rate_decimal
            except ValueError:
                rate = None
        contracts.append(contract.model_copy(update={"risk_free_rate": rate}))
    return request.app.state.tian_service.reprice(
        payload.model_copy(update={"contracts": contracts})
    )


@router.post("/hedge-lab/options/chain", response_model=dict[str, Any])
def hedge_lab_chain(payload: HedgeChainRequest, request: Request) -> dict[str, Any]:
    try:
        return request.app.state.portfolio_service.hedge_chain(
            payload.account_ids,
            portfolio_snapshot_id=payload.portfolio_snapshot_id,
            symbol=payload.symbol,
            contract_type=payload.contract_type,
            strike_count=payload.strike_count,
            from_date=payload.from_date,
            to_date=payload.to_date,
            include_underlying_quote=payload.include_underlying_quote,
        )
    except NoSnapshotLoaded as exc:
        raise SnapshotUnavailableError(str(exc)) from exc
    except SnapshotMismatch as exc:
        raise SnapshotMismatchError(str(exc)) from exc
    except ReferenceDataClientError as exc:
        raise ReferenceDataUnavailableError() from exc
    except ValueError as exc:
        raise InvalidRiskRequestError(str(exc)) from exc


@router.post("/hedge-lab/options/expirations", response_model=dict[str, Any])
def hedge_lab_expirations(
    payload: HedgeExpirationRequest, request: Request
) -> dict[str, Any]:
    """Return a compact expiration calendar without shipping a multi-expiry chain."""

    try:
        return request.app.state.portfolio_service.hedge_expirations(
            payload.account_ids,
            portfolio_snapshot_id=payload.portfolio_snapshot_id,
            symbol=payload.symbol,
            from_date=payload.from_date,
            to_date=payload.to_date,
            include_underlying_quote=payload.include_underlying_quote,
        )
    except NoSnapshotLoaded as exc:
        raise SnapshotUnavailableError(str(exc)) from exc
    except SnapshotMismatch as exc:
        raise SnapshotMismatchError(str(exc)) from exc
    except ReferenceDataClientError as exc:
        raise ReferenceDataUnavailableError() from exc
    except ValueError as exc:
        raise InvalidRiskRequestError(str(exc)) from exc


@router.post(
    "/hedge-lab/targets/assess",
    response_model=PortfolioAnalysisResponse,
    description="Snapshot-bound, analysis-only risk assessment for a selected position set.",
)
def hedge_target_assessment(
    payload: HedgeAssessmentRequest,
    request: Request,
) -> PortfolioAnalysisResponse:
    try:
        return request.app.state.hedge_lab_service.assess(payload)
    except NoSnapshotLoaded as exc:
        raise SnapshotUnavailableError(str(exc)) from exc
    except SnapshotMismatch as exc:
        raise SnapshotMismatchError(str(exc)) from exc
    except ValueError as exc:
        raise InvalidRiskRequestError(str(exc)) from exc


@router.post(
    "/hedge-lab/candidates/query",
    response_model=PortfolioAnalysisResponse,
    description="Bounded research-only hedge candidate generation and ranking.",
)
def hedge_candidate_query(
    payload: HedgeCandidateQueryRequest,
    request: Request,
) -> PortfolioAnalysisResponse:
    try:
        return request.app.state.hedge_lab_service.candidates(payload)
    except NoSnapshotLoaded as exc:
        raise SnapshotUnavailableError(str(exc)) from exc
    except SnapshotMismatch as exc:
        raise SnapshotMismatchError(str(exc)) from exc
    except ValueError as exc:
        raise InvalidRiskRequestError(str(exc)) from exc


@router.post(
    "/hedge-lab/proposals/evaluate",
    response_model=PortfolioAnalysisResponse,
    description="Ephemeral proposal revaluation; no order or portfolio mutation is performed.",
)
def hedge_proposal_evaluation(
    payload: HedgeProposalRequest,
    request: Request,
) -> PortfolioAnalysisResponse:
    try:
        return request.app.state.hedge_lab_service.evaluate(payload)
    except NoSnapshotLoaded as exc:
        raise SnapshotUnavailableError(str(exc)) from exc
    except SnapshotMismatch as exc:
        raise SnapshotMismatchError(str(exc)) from exc
    except ValueError as exc:
        raise InvalidRiskRequestError(str(exc)) from exc


@router.get("/alerts", response_model=AlertsResponse)
def alerts(
    request: Request,
    view: str = Query(default="active", pattern=r"^(new|active|history|suppressed)$"),
) -> AlertsResponse:
    return request.app.state.alert_store.list(view)


@router.get("/strategy-groups", response_model=list[StrategyGroup])
def strategy_groups(
    request: Request,
    account_id: list[str] = Query(default=["all"]),
) -> list[StrategyGroup]:
    current_ids: set[str] | None = None
    try:
        response, _ = _required_snapshot(request, _split_values(account_id))
        current_ids = {item.position_id for item in response.data.positions}
    except SnapshotUnavailableError:
        pass
    return request.app.state.strategy_group_store.list(current_ids)


@router.put("/strategy-groups/{group_id}", response_model=StrategyGroup)
def put_strategy_group(
    payload: StrategyGroupMutation,
    request: Request,
    group_id: str = Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
) -> StrategyGroup:
    try:
        return request.app.state.strategy_group_store.put(group_id, payload)
    except ValueError as exc:
        raise InvalidRiskRequestError(str(exc)) from exc


@router.delete("/strategy-groups/{group_id}", response_model=dict[str, bool])
def delete_strategy_group(
    request: Request,
    group_id: str = Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
    expected_version: int = Query(ge=1),
) -> dict[str, bool]:
    deleted = request.app.state.strategy_group_store.delete(group_id, expected_version)
    if not deleted:
        raise InvalidRiskRequestError("strategy group not found or version changed")
    return {"deleted": True}


@router.get("/risk-configuration", response_model=dict[str, Any])
def risk_configuration(request: Request) -> dict[str, Any]:
    context = request.app.state.portfolio_service.context
    return {
        "schema_version": 1,
        "configuration_version": _configuration_version(context),
        "data": _editable_configuration(context),
        "provider_readiness": {
            "fred": "redacted_in_reference_capabilities",
            "cme": "redacted_in_reference_capabilities",
            "transactions": "ready" if request.app.state.transaction_repository else "unavailable",
        },
    }


@router.patch("/risk-configuration", response_model=dict[str, Any])
def patch_risk_configuration(
    payload: RiskConfigurationPatch,
    request: Request,
) -> dict[str, Any]:
    context = request.app.state.portfolio_service.context
    if not isinstance(context, dict):
        raise InvalidRiskRequestError("configuration is not mutable in this process")
    if payload.position_alert_conditions is not None:
        condition_names = [item.name for item in payload.position_alert_conditions]
        if len({name.casefold() for name in condition_names}) != len(condition_names):
            raise InvalidRiskRequestError("position alert condition names must be unique")
    else:
        condition_names = [
            str(item.get("name") or "")
            for item in context.get("risk.alerts.position.conditions", [])
            if isinstance(item, dict)
        ]
    if payload.position_alert_rules is not None:
        rule_ids = [item.rule_id for item in payload.position_alert_rules]
        rule_conditions = [item.condition_name for item in payload.position_alert_rules]
    else:
        existing_rules = [item for item in context.get("risk.alerts.position.rules", []) if isinstance(item, dict)]
        rule_ids = [str(item.get("rule_id") or "") for item in existing_rules]
        rule_conditions = [str(item.get("condition_name") or "") for item in existing_rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise InvalidRiskRequestError("position alert rule ids must be unique")
    # Validate rule references before applying editable alert configuration.
    known_conditions = {name.casefold() for name in condition_names}
    missing = sorted({name for name in rule_conditions if name.casefold() not in known_conditions})
    if missing:
        raise InvalidRiskRequestError("unknown position alert condition: " + ", ".join(missing))
    updates = {
        "risk.options.greek_source": payload.greek_source,
        "risk.options.fallback_to_provider": payload.fallback_to_provider,
        "risk.tail_risk.confidence": payload.tail_confidence,
        "risk.history.lookback_sessions": payload.tail_lookback_sessions,
        "risk.clusters.minimum_cluster_correlation": payload.cluster_minimum_correlation,
        "risk.clusters.apply_stressed_correlations": payload.apply_stressed_correlations,
        "risk.alerts.margin.warning_utilization": payload.margin_warning_utilization,
        "risk.alerts.margin.critical_utilization": payload.margin_critical_utilization,
        "risk.alerts.position.conditions": [item.model_dump(mode="json") for item in payload.position_alert_conditions] if payload.position_alert_conditions is not None else None,
        "risk.alerts.position.rules": [item.model_dump(mode="json") for item in payload.position_alert_rules] if payload.position_alert_rules is not None else None,
    }
    for key, value in updates.items():
        if value is not None:
            context[key] = value
    if float(context.get("risk.alerts.margin.critical_utilization", 0.70)) <= float(
        context.get("risk.alerts.margin.warning_utilization", 0.50)
    ):
        raise InvalidRiskRequestError("critical margin threshold must exceed warning threshold")
    if payload.position_alert_conditions is not None or payload.position_alert_rules is not None:
        request.app.state.alert_store.save_position_configuration(
            list(context.get("risk.alerts.position.conditions", [])),
            list(context.get("risk.alerts.position.rules", [])),
        )
    return risk_configuration(request)


@router.post("/alerts/evaluate", response_model=AlertsResponse)
def evaluate_alerts(
    payload: AlertEvaluationRequest,
    request: Request,
) -> AlertsResponse:
    response, _ = _required_snapshot(
        request, payload.account_ids, payload.portfolio_snapshot_id
    )
    candidates, evaluated_rules = request.app.state.alert_engine.evaluate(
        response, evaluated_at=datetime.now(timezone.utc)
    )
    return request.app.state.alert_store.apply_evaluation(
        candidates,
        evaluated_rule_ids=evaluated_rules,
        evaluated_at=datetime.now(timezone.utc),
        portfolio_snapshot_id=response.portfolio_snapshot_id,
    )


@router.patch("/alerts/{alert_id}", response_model=AlertEpisode)
def patch_alert(
    payload: AlertPatchRequest,
    request: Request,
    alert_id: str = Path(min_length=1, max_length=64),
) -> AlertEpisode:
    try:
        return request.app.state.alert_store.patch(alert_id, payload)
    except AlertNotFound as exc:
        raise AlertNotFoundError() from exc
    except AlertVersionConflict as exc:
        raise AlertVersionConflictError() from exc
    except ValueError as exc:
        raise InvalidRiskRequestError(str(exc)) from exc


@router.get(
    "/accounts/{account_id}/transactions", response_model=TransactionsResponse
)
def transactions(
    request: Request,
    account_id: str = Path(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    ),
    start_at: datetime = Query(),
    end_at: datetime = Query(),
    types: list[str] | None = Query(default=None),
    symbol: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=250, ge=1, le=1000),
    cursor: str | None = Query(default=None, max_length=1024),
) -> TransactionsResponse:
    try:
        return _transaction_repository(request).query(
            account_id,
            start_at=start_at,
            end_at=end_at,
            types=_transaction_types(types),
            symbol=symbol,
            limit=limit,
            cursor=cursor,
        )
    except (ValueError, InvalidCursor) as exc:
        raise InvalidRiskRequestError(str(exc)) from exc
    except TransactionRepositoryUnavailable as exc:
        raise TransactionRepositoryError() from exc


@router.get(
    "/accounts/{account_id}/transactions/status",
    response_model=TransactionStatusResponse,
)
def transaction_status(
    request: Request,
    account_id: str = Path(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    ),
) -> TransactionStatusResponse:
    try:
        return _transaction_repository(request).status(account_id)
    except TransactionRepositoryUnavailable as exc:
        raise TransactionRepositoryError() from exc


def _required_snapshot(
    request: Request,
    account_ids: list[str],
    snapshot_id: str | None = None,
):
    try:
        return request.app.state.portfolio_service.require_snapshot(
            account_ids, snapshot_id
        )
    except NoSnapshotLoaded as exc:
        raise SnapshotUnavailableError(str(exc)) from exc
    except SnapshotMismatch as exc:
        raise SnapshotMismatchError(str(exc)) from exc


def _asset_types_for_scope(asset_scope: str) -> set[str] | None:
    if asset_scope == "equities":
        return {"EQUITY", "ETF", "MUTUAL_FUND"}
    if asset_scope == "options":
        return {"OPTION"}
    if asset_scope == "futures":
        return {"FUTURE"}
    return None


def _response_for_asset_scope(
    response: PortfolioSnapshotResponse,
    asset_scope: str,
    context: Mapping[str, Any] | None = None,
) -> PortfolioSnapshotResponse:
    selected_types = _asset_types_for_scope(asset_scope)
    if selected_types is None:
        return response
    positions = [item for item in response.data.positions if item.asset_type in selected_types]
    # Scoped numerators remain scoped, but coverage, materiality, leverage and
    # "% Portfolio" all require the full portfolio NAV as their denominator.
    portfolio_value = response.data.kpis.portfolio_value_usd
    kpis = portfolio_kpis(
        positions, portfolio_value_usd=portfolio_value, context=context
    )
    data = response.data.model_copy(update={"positions": positions, "kpis": kpis})
    return response.model_copy(update={"data": data})


def _analysis(
    response: PortfolioSnapshotResponse,
    methodology_version: str,
    data: dict[str, Any],
    *,
    insufficient: bool = False,
) -> PortfolioAnalysisResponse:
    quality = response.data_quality
    nested_insufficient = str(
        data.get("status") or data.get("quality_state") or ""
    ) == "insufficient_data"
    # Carry nested calculation warnings into the common response quality envelope.
    nested_warnings = _nested_warnings(data)
    warnings = tuple(
        dict.fromkeys(
            (
                *quality.warnings,
                *nested_warnings,
                *(("calculation_insufficient_data",) if insufficient or nested_insufficient else ()),
            )
        )
    )
    status = "insufficient_data" if insufficient or nested_insufficient else quality.status
    if warnings and status == "usable":
        status = "usable_with_warnings"
    quality = quality.model_copy(update={"status": status, "warnings": warnings})
    return PortfolioAnalysisResponse(
        portfolio_snapshot_id=response.portfolio_snapshot_id,
        configuration_version=response.configuration_version,
        as_of=response.as_of,
        display_timezone=response.display_timezone,
        data_quality=quality,
        methodology_version=methodology_version,
        data=data,
    )


def _nested_warnings(value: Any) -> tuple[str, ...]:
    output: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "warnings" and isinstance(child, (list, tuple)):
                    output.extend(str(row) for row in child if row not in (None, ""))
                else:
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(dict.fromkeys(output))


def _position_exposures(response: PortfolioSnapshotResponse) -> dict[str, float]:
    exposures: dict[str, float] = {}
    for position in response.data.positions:
        # Broker money-market mutual funds behave as cash equivalents in the
        # one-day market model; they do not need an invented zero return row.
        broker_type = str(position.raw_broker_fields.get("instrument_type") or "")
        if position.asset_type == "CASH" or "NO_LOAD" in broker_type or "MONEY_MARKET" in broker_type:
            continue
        symbol = position.underlying_symbol or position.exact_symbol
        if not _position_field_valid(position, "delta"):
            continue
        exposure = delta_equivalent_usd(position)
        if symbol and exposure is not None:
            exposures[symbol] = exposures.get(symbol, 0.0) + float(exposure)
    return exposures


def _load_returns(request: Request, symbols: set[str], as_of: datetime) -> dict[str, dict[str, float]]:
    lookback = int(request.app.state.portfolio_service.context.get("risk.history.lookback_sessions", 252))
    key = (as_of.isoformat(), lookback, tuple(sorted(symbols)))
    state = request.app.state
    now = datetime.now(timezone.utc)
    with _HISTORICAL_CACHE_INIT_LOCK:
        if not hasattr(state, "scenario_returns_lock"):
            state.scenario_returns_lock = RLock()
            state.scenario_returns_cache = {}
    # Scenario sweeps repeatedly need the same ETF/proxy history.  Keep the
    # immutable return matrix locally for 30 minutes and hold the lock through
    # a miss so concurrent chart requests collapse into one RefData query.
    with state.scenario_returns_lock:
        cached = state.scenario_returns_cache.get(key)
        if cached is not None and (now - cached[0]).total_seconds() < 1800:
            return cached[1]
        returns, _diagnostics, _plan = _historical_revaluation_engine(
            request
        ).load_returns(
            sorted(symbols), as_of=as_of, lookback_sessions=lookback
        )
        state.scenario_returns_cache[key] = (now, returns)
        while len(state.scenario_returns_cache) > 16:
            state.scenario_returns_cache.pop(next(iter(state.scenario_returns_cache)))
        return returns


def _tail_analysis(
    request: Request,
    response: PortfolioSnapshotResponse,
    confidence: float,
) -> dict[str, Any]:
    positions = [
        item
        for item in response.data.positions
        if historical_risk_position_eligible(item)
    ]
    minimum = int(request.app.state.portfolio_service.context.get("risk.history.minimum_observations", 126))
    lookback = int(request.app.state.portfolio_service.context.get("risk.history.lookback_sessions", 252))
    try:
        revaluation = _historical_revalue(
            request,
            positions,
            as_of=response.as_of,
            lookback_sessions=lookback,
        )
    except Exception as exc:
        return {
            "status": "insufficient_data", "confidence": confidence,
            "reason": f"daily_history_request_failed:{type(exc).__name__}",
            "failure_causes": [
                {
                    "code": "daily_history_request_failed",
                    "error_type": type(exc).__name__,
                }
            ],
            "observations": 0,
            "minimum_observations": minimum,
            "observations_label": "Complete portfolio P&L scenarios",
            "var_usd": None, "expected_shortfall_usd": None,
            "missing_position_ids": [],
            "missing_positions": [],
            "position_coverage": [],
            "warnings": ["tail_history_unavailable"],
        }
    modeled_ids = sorted(revaluation.scenario_pnl)
    by_id = {item.position_id: item for item in positions}
    material_missing = sorted(
        position_id
        for position_id in revaluation.missing_position_ids
        if position_id in by_id and omission_requires_fail_closed(by_id[position_id])
    )
    omission = assess_omission_materiality(
        revaluation.missing_position_ids,
        positions,
        portfolio_value_usd=getattr(
            getattr(response.data, "kpis", None), "portfolio_value_usd", None
        ),
        context=request.app.state.portfolio_service.context,
    )
    if omission.may_exclude:
        material_missing = []
    coverage = 100 * len(modeled_ids) / len(positions) if positions else 100.0
    # Tail calculations use only sessions shared by every modeled position.
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
    minimum_coverage = float(request.app.state.portfolio_service.context.get("risk.tail_risk.minimum_gross_delta_coverage", 0.90)) * 100
    position_coverage = [
        item.model_dump(mode="json") for item in revaluation.position_coverage
    ]
    failed_request_positions = sorted(
        item.position_id
        for item in revaluation.position_coverage
        if item.failed_remote_ranges
    )
    failure_causes: list[dict[str, Any]] = []
    if len(dates) < minimum:
        failure_causes.append(
            {
                "code": "complete_scenarios_below_minimum",
                "actual": len(dates),
                "required": minimum,
            }
        )
    if coverage < minimum_coverage and not omission.may_exclude:
        failure_causes.append(
            {
                "code": "modeled_coverage_below_minimum",
                "actual_percent": coverage,
                "required_percent": minimum_coverage,
            }
        )
    if material_missing:
        failure_causes.append(
            {
                "code": "material_positions_unmodeled",
                "position_ids": material_missing,
            }
        )
    if failed_request_positions and failure_causes:
        failure_causes.append(
            {
                "code": "upstream_history_range_failed",
                "position_ids": failed_request_positions,
                "failed_ranges": [
                    row
                    for item in position_coverage
                    if item["position_id"] in failed_request_positions
                    for row in item.get("failed_remote_ranges") or ()
                ],
            }
        )
    missing_symbols = [
        {
            "position_id": position_id,
            "exact_symbol": by_id[position_id].canonical_occ_symbol
            or by_id[position_id].exact_symbol,
        }
        for position_id in revaluation.missing_position_ids
        if position_id in by_id
    ]
    if failure_causes:
        primary_reason = (
            "material_nonlinear_or_unknown_exposure_unmodeled"
            if material_missing
            else "complete_portfolio_scenarios_insufficient"
            if len(dates) < minimum
            else "modeled_coverage_insufficient"
            if coverage < minimum_coverage
            else "upstream_history_range_failed"
        )
        return {
            "status": "insufficient_data", "confidence": confidence,
            "reason": primary_reason,
            "failure_causes": failure_causes,
            "observations": len(dates), "minimum_observations": minimum,
            "observations_label": "Complete portfolio P&L scenarios",
            "coverage": {
                "eligible_positions": len(positions),
                "modeled_positions": len(modeled_ids),
                "percent": coverage,
                "count_percent": coverage,
                "weight_basis": "full_revaluation_eligible_economic_unit_count",
                "denominator_usd": None,
                "modeled_usd": None,
                "missing_position_ids": list(revaluation.missing_position_ids),
            },
            "var_usd": None, "expected_shortfall_usd": None,
            "missing_position_ids": list(revaluation.missing_position_ids),
            "missing_positions": missing_symbols,
            "material_missing_position_ids": material_missing,
            "proxy_mappings": list(revaluation.proxy_mappings),
            "position_coverage": position_coverage,
            "scenario_provenance": revaluation.scenario_provenance,
            "warnings": [
                "tail_history_insufficient",
                *([omission.warning] if omission.warning else []),
                *revaluation.warnings,
            ],
        }
    unit_ids = modeled_ids
    # Keep a stable unit order when constructing the historical dollar P&L matrix.
    matrix = [
        [revaluation.scenario_pnl[item_id][day] for item_id in unit_ids]
        for day in dates
    ]
    result = historical_var_es(
        matrix,
        confidence=confidence,
        half_life_sessions=float(request.app.state.portfolio_service.context.get("risk.tail_risk.half_life_sessions", 63)),
        unit_ids=unit_ids,
        dates=dates,
    )
    result.update({
        "status": "usable_with_warnings" if revaluation.warnings else "usable",
        "input_method": "current_portfolio_filtered_historical_revaluation_v5",
        "calculation_method": "current_state_filtered_historical_shock_revaluation",
        "input_mode": "mixed_by_position",
        "failure_causes": [],
        "minimum_observations": minimum,
        "observations_label": "Complete portfolio P&L scenarios",
        "unit_methods": revaluation.scenario_methods,
        "coverage": {
            "eligible_positions": len(positions),
            "modeled_positions": len(modeled_ids),
            "percent": coverage,
            "count_percent": coverage,
            "weight_basis": "full_revaluation_eligible_economic_unit_count",
            "denominator_usd": None,
            "modeled_usd": None,
            "missing_position_ids": list(revaluation.missing_position_ids),
        },
        "missing_position_ids": list(revaluation.missing_position_ids),
        "missing_positions": missing_symbols,
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


def _factor_analysis(request: Request, response: PortfolioSnapshotResponse) -> dict[str, Any]:
    positions = [
        item
        for item in response.data.positions
        if historical_risk_position_eligible(item)
    ]
    try:
        returns = _load_returns(
            request, set(FACTOR_PROXY_SYMBOLS), response.as_of
        )
        missing_factor_proxies = [symbol for symbol in FACTOR_PROXY_SYMBOLS if symbol not in returns]
        if missing_factor_proxies:
            return {
                "status": "insufficient_data",
                "reason": "factor_proxy_history_unavailable",
                "missing_proxy_symbols": missing_factor_proxies,
                "items": [],
            }
        proxy = {symbol: returns[symbol] for symbol in FACTOR_PROXY_SYMBOLS}
        revaluation = _historical_revalue(
            request, positions, as_of=response.as_of
        )
    except Exception as exc:
        return {"status": "insufficient_data", "reason": f"factor_history_request_failed:{type(exc).__name__}", "items": []}
    modeled = revaluation.scenario_pnl
    by_id = {item.position_id: item for item in positions}
    material_missing = sorted(
        position_id
        for position_id in revaluation.missing_position_ids
        if position_id in by_id and omission_requires_fail_closed(by_id[position_id])
    )
    omission = assess_omission_materiality(
        revaluation.missing_position_ids,
        positions,
        portfolio_value_usd=getattr(
            getattr(response.data, "kpis", None), "portfolio_value_usd", None
        ),
        context=request.app.state.portfolio_service.context,
    )
    if omission.may_exclude:
        material_missing = []
    if material_missing:
        return {
            "status": "insufficient_data",
            "reason": "material_nonlinear_or_unknown_exposure_unmodeled",
            "failure_causes": [
                {
                    "code": "material_positions_unmodeled",
                    "position_ids": material_missing,
                }
            ],
            "items": [],
            "coverage": {
                "eligible_positions": len(positions),
                "modeled_positions": len(modeled),
                "count_percent": 100.0 * len(modeled) / len(positions) if positions else 100.0,
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
    # Fit factor exposures only on sessions shared by proxies and portfolio P&L.
    aligned_series = [*proxy.values(), *modeled.values()]
    dates = sorted(set.intersection(*(set(series) for series in aligned_series))) if aligned_series else []
    lookback = int(request.app.state.portfolio_service.context.get("risk.history.lookback_sessions", 252))
    dates = dates[-lookback:]
    target = [
        sum(series[day] for series in modeled.values())
        for day in dates
    ] if modeled else []
    proxy_values = {symbol: [proxy[symbol][day] for day in dates] for symbol in FACTOR_PROXY_SYMBOLS}
    unit_values = {
        unit_id: [series[day] for day in dates]
        for unit_id, series in modeled.items()
    }
    try:
        result = ridge_factor_exposure(
            target,
            proxy_values,
            unit_pnl_by_id=unit_values,
            ridge_alpha=float(request.app.state.portfolio_service.context.get("risk.factors.ridge_alpha", 1e-3)),
            minimum_observations=int(request.app.state.portfolio_service.context.get("risk.factors.minimum_observations", request.app.state.portfolio_service.context.get("risk.history.minimum_observations", 126))),
        )
    except (ValueError, KeyError) as exc:
        return {"status": "insufficient_data", "reason": f"factor_history_unusable:{type(exc).__name__}", "items": []}
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
        "eligible_positions": len(positions),
        "modeled_positions": len(modeled),
        "percent": 100.0 * len(modeled) / len(positions) if positions else 100.0,
        "count_percent": 100.0 * len(modeled) / len(positions) if positions else 100.0,
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
    if revaluation.missing_position_ids and result.get("status") == "usable":
        result["status"] = "usable_with_warnings"
    return result


def _cluster_analysis(request: Request, snapshot: Any) -> dict[str, Any]:
    context = request.app.state.portfolio_service.context
    engine = ScenarioEngine(context)
    scenario_names: list[str] = []
    # Collect signed per-position P&L before grouping correlated economic units.
    scenario_values: dict[str, dict[int, float]] = {}
    exposures: dict[str, float] = {}
    warnings: list[str] = []
    scenario_material_missing: set[str] = set()
    etf_symbols = {
        position.exact_symbol
        for position in snapshot.positions
        if position.asset_type == "ETF"
    }
    etf_returns: dict[str, dict[str, float]] | None = None
    if etf_symbols:
        try:
            etf_returns = _load_returns(
                request,
                etf_symbols | set(ETF_RESPONSE_PROXY_SYMBOLS),
                snapshot.as_of,
            )
        except Exception as exc:
            warnings.append(
                f"cluster_etf_response_history_unavailable:{type(exc).__name__}"
            )
    for scenario_index, scenario in enumerate(DEFAULT_SCENARIOS):
        evaluated_scenario = scenario
        if etf_returns is not None:
            explicit = {
                **scenario.underlying_shocks,
                **scenario.exact_underlying_shocks,
            }
            modeled_shocks, diagnostics = build_etf_history_shocks(
                snapshot.positions,
                etf_returns,
                broad_equity_shock=(
                    scenario.broad_equity_shock
                    if scenario.broad_equity_shock is not None
                    else scenario.default_equity_shock
                ),
                sector_shocks=scenario.sector_shocks,
                explicit_shocks=explicit,
                minimum_observations=int(
                    context.get("risk.scenarios.etf_response_minimum_observations", 126)
                ),
                maximum_observations=252,
                ridge_alpha=float(
                    context.get("risk.scenarios.etf_response_ridge_alpha", 1e-3)
                ),
                minimum_r_squared=float(
                    context.get("risk.scenarios.etf_response_minimum_r_squared", 0.20)
                ),
                maximum_absolute_shock=float(
                    context.get("risk.scenarios.maximum_absolute_equity_shock", 0.50)
                ),
            )
            exact_symbols = set(scenario.exact_underlying_shocks)
            updated_shocks = dict(scenario.underlying_shocks)
            updated_shocks.update(
                {
                    symbol: shock
                    for symbol, shock in modeled_shocks.items()
                    if symbol not in exact_symbols
                }
            )
            evaluated_scenario = scenario.model_copy(
                update={
                    "underlying_shocks": updated_shocks,
                    "etf_response_diagnostics": diagnostics,
                }
            )
        evaluated = engine.evaluate(snapshot, evaluated_scenario)
        scenario_material_missing.update(
            str(item)
            for item in evaluated.get("material_unmodeled_position_ids") or ()
        )
        scenario_names.append(scenario.name)
        for item in evaluated.get("contributions") or []:
            unit = str(item["economic_position_id"])
            bucket = scenario_values.setdefault(unit, {})
            bucket[scenario_index] = bucket.get(scenario_index, 0.0) + float(item["pnl_usd"])
    vectors = {
        unit: [values[index] for index in range(len(DEFAULT_SCENARIOS))]
        for unit, values in scenario_values.items()
        if len(values) == len(DEFAULT_SCENARIOS)
    }
    omitted_scenario_units = sorted(set(scenario_values) - set(vectors))
    if omitted_scenario_units:
        warnings.append("partial_scenario_unit_coverage")
    for position in snapshot.positions:
        value = delta_equivalent_usd(position)
        if value is not None:
            exposures[position.economic_position_id] = exposures.get(position.economic_position_id, 0.0) + value
    historical: dict[str, dict[str, float]] = {}
    factor_vectors: dict[str, list[float]] = {}
    history_window: dict[str, Any] | None = None
    factor_methodology: str | None = None
    position_coverage: list[dict[str, Any]] = []
    scenario_provenance: dict[str, Any] = {}
    try:
        revaluation = _historical_revalue(
            request, snapshot.positions, as_of=snapshot.as_of
        )
        warnings.extend(revaluation.warnings)
        position_coverage = [
            item.model_dump(mode="json") for item in revaluation.position_coverage
        ]
        scenario_provenance = revaluation.scenario_provenance
        by_position_id = {item.position_id: item for item in snapshot.positions}
        historical_material_missing = {
            position_id
            for position_id in revaluation.missing_position_ids
            if position_id in by_position_id
            and omission_requires_fail_closed(by_position_id[position_id])
        }
        all_material_missing = sorted(
            scenario_material_missing | historical_material_missing
        )
        omission = assess_omission_materiality(
            all_material_missing,
            snapshot.positions,
            portfolio_value_usd=snapshot.portfolio_value_usd,
            context=context,
        )
        if omission.may_exclude:
            all_material_missing = []
            if omission.warning:
                warnings.append(omission.warning)
        if all_material_missing:
            return {
                "status": "insufficient_data",
                "mode": "unavailable",
                "reason": "material_nonlinear_or_unknown_exposure_unmodeled",
                "failure_causes": [
                    {
                        "code": "material_positions_unmodeled",
                        "position_ids": all_material_missing,
                    }
                ],
                "items": [],
                "position_coverage": position_coverage,
                "scenario_provenance": scenario_provenance,
                "missing_position_ids": all_material_missing,
                "warnings": [
                    "material_nonlinear_or_unknown_exposure_unmodeled",
                    *warnings,
                ],
            }
        position_to_economic = {
            item.position_id: item.economic_position_id for item in snapshot.positions
        }
        grouped_series: dict[str, list[dict[str, float]]] = {}
        for position_id, series in revaluation.scenario_pnl.items():
            economic_id = position_to_economic.get(position_id)
            if economic_id in vectors:
                grouped_series.setdefault(economic_id, []).append(series)
        for economic_id, series_rows in grouped_series.items():
            common = sorted(set.intersection(*(set(row) for row in series_rows)))
            historical[economic_id] = {
                day: sum(row[day] for row in series_rows) for day in common
            }
        all_history_dates = sorted(
            {day for series in historical.values() for day in series}
        )
        history_window = {
            "start": all_history_dates[0] if all_history_dates else None,
            "end": all_history_dates[-1] if all_history_dates else None,
            "units": len(historical),
        }
        proxy_returns = _load_returns(request, set(FACTOR_PROXY_SYMBOLS), snapshot.as_of)
        if all(symbol in proxy_returns for symbol in FACTOR_PROXY_SYMBOLS) and historical:
            factor_dates = sorted(
                set.intersection(
                    *(set(proxy_returns[symbol]) for symbol in FACTOR_PROXY_SYMBOLS),
                    *(set(series) for series in historical.values()),
                )
            )
            proxy_values = {
                symbol: [proxy_returns[symbol][day] for day in factor_dates]
                for symbol in FACTOR_PROXY_SYMBOLS
            }
            unit_values = {
                unit: [series[day] for day in factor_dates]
                for unit, series in historical.items()
            }
            factor_result = ridge_factor_exposure(
                [sum(series[day] for series in historical.values()) for day in factor_dates],
                proxy_values,
                unit_pnl_by_id=unit_values,
                ridge_alpha=float(context.get("risk.factors.ridge_alpha", 1e-3)),
                minimum_observations=int(context.get("risk.factors.minimum_observations", 126)),
            )
            if factor_result.get("status") == "usable":
                for contribution in factor_result.get("unit_contributions") or []:
                    factor_vectors[str(contribution["unit_id"])] = [
                        float(item["coefficient_usd_per_unit_factor_return"])
                        for item in contribution.get("factors") or []
                    ]
                factor_methodology = str(factor_result.get("methodology_version"))
    except Exception as exc:
        warnings.append(f"hybrid_cluster_evidence_unavailable:{type(exc).__name__}")
        history_required = sorted(
            item.position_id
            for item in snapshot.positions
            if historical_risk_position_eligible(item)
            and omission_requires_fail_closed(item)
        )
        if history_required:
            return {
                "status": "insufficient_data",
                "mode": "unavailable",
                "reason": "material_nonlinear_or_unknown_exposure_unmodeled",
                "failure_causes": [
                    {
                        "code": "historical_revaluation_failed",
                        "error_type": type(exc).__name__,
                    },
                    {
                        "code": "material_positions_unmodeled",
                        "position_ids": history_required,
                    },
                ],
                "items": [],
                "position_coverage": position_coverage,
                "scenario_provenance": scenario_provenance,
                "missing_position_ids": history_required,
                "warnings": [
                    "material_historical_cluster_evidence_unavailable",
                    *warnings,
                ],
            }
    if scenario_material_missing:
        return {
            "status": "insufficient_data",
            "mode": "unavailable",
            "reason": "material_nonlinear_or_unknown_exposure_unmodeled",
            "failure_causes": [
                {
                    "code": "material_positions_unmodeled",
                    "position_ids": sorted(scenario_material_missing),
                }
            ],
            "items": [],
            "position_coverage": position_coverage,
            "scenario_provenance": scenario_provenance,
            "missing_position_ids": sorted(scenario_material_missing),
            "warnings": [
                "material_nonlinear_or_unknown_exposure_unmodeled",
                *warnings,
            ],
        }
    configured_weights = context.get("risk.clusters.source_weights", {})
    source_weights = {
        "scenario": float(configured_weights.get("scenario", 1.0)),
        "historical": float(
            configured_weights.get("historical", configured_weights.get("history", 0.0))
        ),
        "factor": float(configured_weights.get("factor", 0.0)),
    } if isinstance(configured_weights, dict) else {"scenario": 1.0}
    items = cluster_risk_units(
        vectors,
        exposures_usd=exposures,
        scenario_names=scenario_names,
        minimum_correlation=float(context.get("risk.clusters.minimum_cluster_correlation", 0.60)),
        minimum_cluster_loss_usd=float(context.get("risk.clusters.minimum_cluster_risk_usd", 10_000)),
        methodology_version="correlated_clusters_v2",
        historical_pnl_by_unit=historical,
        factor_vectors_by_unit=factor_vectors,
        source_weights=source_weights,
        minimum_pairwise_overlap=int(context.get("risk.clusters.minimum_pairwise_overlap", 60)),
    )
    aggregate_mix = {"scenario": 0.0, "historical": 0.0, "factor": 0.0}
    for item in items:
        for source, value in (item.get("correlation_source_mix") or {}).items():
            if source in aggregate_mix:
                aggregate_mix[source] += float(value)
    mix_total = sum(aggregate_mix.values())
    actual_mix = {
        source: value / mix_total if mix_total else 0.0
        for source, value in aggregate_mix.items()
    }
    used_hybrid = actual_mix["historical"] > 0 or actual_mix["factor"] > 0
    return {
        "status": (
            "usable_with_warnings" if warnings or (items and not used_hybrid)
            else "usable"
        ),
        "mode": (
            "evaluated_no_clusters"
            if not items
            else "hybrid"
            if used_hybrid
            else "scenario_only"
        ),
        "items": items,
        "scenario_count": len(scenario_names), "unit_count": len(vectors),
        "history_window": history_window,
        "factor_methodology": factor_methodology,
        "configured_source_weights": source_weights,
        "actual_correlation_source_mix": actual_mix,
        "unit_exposures_usd": exposures,
        "stress_transform": context.get("risk.clusters.stressed_correlation_transform"),
        "missing_scenario_unit_ids": omitted_scenario_units,
        "position_coverage": position_coverage,
        "scenario_provenance": scenario_provenance,
        "failure_causes": [],
        "warnings": [
            *(warnings or (["scenario_only_cluster_evidence"] if not used_hybrid else []))
        ],
        "reason": None if items else "no_groups_met_the_configured_correlation_and_loss_thresholds",
    }


def _risk_calendar(response: PortfolioSnapshotResponse) -> list[dict[str, Any]]:
    events = []
    for position in response.data.positions:
        if position.asset_type != "OPTION" or position.expiration_at is None:
            continue
        spot = position.raw_broker_fields.get("underlying_price")
        intrinsic = None
        if isinstance(spot, (int, float)):
            try:
                intrinsic = option_intrinsic_per_contract(position, float(spot))
            except ValueError:
                intrinsic = None
        extrinsic = (
            max(
                0.0,
                position.mark * position.multiplier - intrinsic,
            ) * abs(position.signed_quantity)
            if position.mark is not None
            and intrinsic is not None
            and position.multiplier is not None
            and _position_field_valid(position, "mark")
            else None
        )
        risk_tone = (
            "short_itm"
            if position.signed_quantity < 0 and intrinsic is not None and intrinsic > 0
            else "standard"
        )
        expiration_warning = (
            "option_expiration_assumed_configured_standard_session"
            in position.data_quality.warnings
        )
        events.append(
            {
                "event_id": "expiry:" + position.position_id,
                "type": "option_expiration",
                "title": f"{position.canonical_occ_symbol} expiration",
                "confirmed_status": (
                    "estimated_time"
                    if expiration_warning
                    else "authoritative_contract_terms"
                    if position.instrument_terms is not None
                    and position.instrument_terms.quality_status in {"standard", "adjusted"}
                    else "partial_contract_terms"
                ),
                "event_at": position.expiration_at.isoformat(),
                "affected_position_ids": [position.position_id],
                "premium_expiring_usd": (
                    abs(position.signed_market_value_usd)
                    if position.signed_market_value_usd is not None
                    and _position_field_valid(position, "mark")
                    else None
                ),
                "side": "long" if position.signed_quantity > 0 else "short",
                "quantity": abs(position.signed_quantity),
                "cost_basis_usd": position.signed_cost_basis_usd,
                "current_value_usd": (
                    position.signed_market_value_usd
                    if _position_field_valid(position, "mark") else None
                ),
                "current_pnl_usd": (
                    position.unrealized_pnl_usd
                    if _position_field_valid(position, "mark") else None
                ),
                "extrinsic_value_usd": extrinsic,
                "delta": position.delta if _position_field_valid(position, "delta") else None,
                "theta_per_day": position.theta_per_day if _position_field_valid(position, "theta") else None,
                "vega_per_1pct": position.vega_per_1pct if _position_field_valid(position, "vega") else None,
                "implied_volatility": position.implied_volatility,
                "underlying_price": spot,
                "risk_tone": risk_tone,
                "warnings": ["expiration_time_assumed_configured_standard_session"] if expiration_warning else [],
            }
        )
    return sorted(events, key=lambda item: item["event_at"])


def _enriched_risk_calendar(request: Request, response: PortfolioSnapshotResponse) -> dict[str, Any]:
    events = _risk_calendar(response)
    omissions: list[str] = []
    start = response.as_of.date()
    end = start + timedelta(days=45)
    client = request.app.state.portfolio_service.reference_client
    symbols = sorted({position.underlying_symbol for position in response.data.positions if position.underlying_symbol})
    tasks: dict[Any, tuple[str, str | None]] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        if symbols:
            tasks[executor.submit(client.earnings_calendar, start=start, end=end)] = ("earnings", None)
            tasks[executor.submit(client.dividend_calendar, start=start, end=end)] = ("dividend", None)
        tasks[executor.submit(client.fomc_calendar, start=start, end=end)] = ("fomc", None)
        tasks[executor.submit(client.economic_calendar, start=start, end=end)] = ("economic", None)
        for future in as_completed(tasks):
            kind, symbol = tasks[future]
            try:
                rows = future.result()
            except Exception as exc:
                omissions.append(f"{kind}_events_unavailable:{type(exc).__name__}")
                continue
            for index, row in enumerate(rows):
                if kind == "economic" and str(row.get("impact") or "").strip().casefold() != "high":
                    continue
                event_symbol = symbol or (
                    str(row.get("symbol") or "").strip().upper()
                    if kind in {"earnings", "dividend"}
                    else None
                )
                if kind in {"earnings", "dividend"} and event_symbol not in symbols:
                    continue
                raw_at = next((row.get(key) for key in ("timestamp", "event_at", "date", "scheduled_date", "earnings_date", "ex_dividend_date") if row.get(key)), None)
                if raw_at is None:
                    continue
                event_at = str(raw_at)
                if len(event_at) == 10:
                    event_at += "T12:00:00-04:00"
                title = str(row.get("report") or row.get("title") or row.get("event_name") or (f"{event_symbol} earnings" if kind == "earnings" else f"{event_symbol} dividend" if kind == "dividend" else "FOMC meeting" if kind == "fomc" else "Economic release"))
                events.append({
                    "event_id": f"{kind}:{event_symbol or 'market'}:{index}:{event_at}",
                    "type": kind, "title": title, "event_at": event_at,
                    "affected_position_ids": [position.position_id for position in response.data.positions if event_symbol and position.underlying_symbol == event_symbol],
                    "details": row, "warnings": [],
                })
    fomc_days = {item["event_at"][:10] for item in events if item["type"] == "fomc"}
    events = [
        item for item in events
        if not (item["type"] == "economic"
                and item["details"].get("internal_name") == "fed_interest_rates"
                and item["event_at"][:10] in fomc_days)
    ]
    return {"events": sorted(events, key=lambda item: item["event_at"]), "omissions": list(dict.fromkeys(omissions))}


def _pnl_explain(
    response: PortfolioSnapshotResponse,
    previous: PortfolioSnapshotResponse | None,
) -> dict[str, Any]:
    """Explain refresh-to-refresh marked value change with an explicit broker-day fallback."""
    current_positions = list(response.data.positions)
    prior_by_id = {
        position.position_id: position
        for position in (previous.data.positions if previous is not None else [])
    }
    current_ids = {position.position_id for position in current_positions}
    contributions: list[dict[str, Any]] = []
    price_effect = 0.0
    holdings_effect = 0.0

    if previous is not None:
        for position in current_positions:
            prior = prior_by_id.get(position.position_id)
            if position.signed_market_value_usd is None:
                continue
            prior_value = prior.signed_market_value_usd if prior is not None else 0.0
            if prior_value is None:
                continue
            change = position.signed_market_value_usd - prior_value
            marked_move = 0.0
            if (
                prior is not None
                and position.mark is not None
                and prior.mark is not None
                and position.signed_quantity == prior.signed_quantity
                and position.multiplier == prior.multiplier
                and position.multiplier is not None
            ):
                marked_move = (
                    (position.mark - prior.mark)
                    * position.signed_quantity
                    * position.multiplier
                )
            price_effect += marked_move
            holdings_effect += change - marked_move
            contributions.append({
                "position_id": position.position_id,
                "symbol": position.underlying_symbol or position.exact_symbol,
                "asset_type": position.asset_type,
                "modeled_value_change_usd": change,
                "mark_move_usd": marked_move,
                "quantity_or_other_change_usd": change - marked_move,
            })
        for position in previous.data.positions:
            if position.position_id in current_ids or position.signed_market_value_usd is None:
                continue
            change = -position.signed_market_value_usd
            holdings_effect += change
            contributions.append({
                "position_id": position.position_id,
                "symbol": position.underlying_symbol or position.exact_symbol,
                "asset_type": position.asset_type,
                "modeled_value_change_usd": change,
                "mark_move_usd": 0.0,
                "quantity_or_other_change_usd": change,
                "closed_since_prior_anchor": True,
            })
        anchor_mode = "refresh_to_refresh"
        warnings: list[str] = []
    else:
        contributions = [
            {
                "position_id": position.position_id,
                "symbol": position.underlying_symbol or position.exact_symbol,
                "asset_type": position.asset_type,
                "modeled_value_change_usd": position.current_day_pnl_usd,
                "mark_move_usd": position.current_day_pnl_usd,
                "quantity_or_other_change_usd": None,
            }
            for position in current_positions
            if position.current_day_pnl_usd is not None
        ]
        price_effect = sum(
            position.current_day_pnl_usd or 0.0 for position in current_positions
        )
        anchor_mode = "broker_current_day_fallback"
        warnings = [
            "no_preceding_refresh_anchor_broker_current_day_pnl_used"
        ]

    contributions.sort(
        key=lambda item: abs(float(item.get("modeled_value_change_usd") or 0.0)),
        reverse=True,
    )
    broker_day_values = [
        position.current_day_pnl_usd
        for position in current_positions
        if position.current_day_pnl_usd is not None
    ]
    broker_day_pnl = sum(broker_day_values) if broker_day_values else None
    broker_day_coverage = (
        100.0 * len(broker_day_values) / len(current_positions)
        if current_positions else 100.0
    )
    modeled_value_change = sum(
        float(item.get("modeled_value_change_usd") or 0.0)
        for item in contributions
    ) if contributions else None
    residual = (
        broker_day_pnl - modeled_value_change
        if broker_day_pnl is not None and modeled_value_change is not None
        else None
    )
    if broker_day_coverage < 100:
        warnings.append("broker_current_day_pnl_incomplete_for_some_positions")
    return {
        "status": "usable" if not warnings else "usable_with_warnings",
        "label": "Refresh-to-refresh marked value change",
        "anchor_mode": anchor_mode,
        "current_anchor_at": response.as_of.isoformat(),
        "prior_anchor_at": previous.as_of.isoformat() if previous is not None else None,
        "anchor_interval_seconds": (
            max(0.0, (response.as_of - previous.as_of).total_seconds())
            if previous is not None else None
        ),
        "actual_pnl": broker_day_pnl,
        "actual_pnl_label": "broker_reported_current_day_pnl",
        "modeled_value_change": modeled_value_change,
        "residual": residual,
        "transaction_coverage": "not_required_for_broker_reported_day_pnl",
        "broker_day_pnl_coverage_percent": broker_day_coverage,
        "factor_attribution": [
            {"factor": "Mark movement", "pnl_usd": price_effect},
            {"factor": "Position quantity and other changes", "pnl_usd": holdings_effect},
        ],
        "contributions": contributions,
        "warnings": warnings,
        "reason": None,
    }


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _options_risk(
    response: PortfolioSnapshotResponse,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    positions = [item for item in response.data.positions if item.asset_type == "OPTION"]
    position_risk: list[dict[str, Any]] = []
    for item in positions:
        scale = item.signed_quantity * item.multiplier if item.multiplier is not None else None
        spot = _finite(item.raw_broker_fields.get("underlying_price"))
        position_risk.append(
            {
                "position_id": item.position_id,
                "economic_position_id": item.economic_position_id,
                "expiration_at": item.expiration_at,
                "signed_market_value_usd": (
                    item.signed_market_value_usd
                    if item.signed_market_value_usd is not None and _position_field_valid(item, "mark")
                    else None
                ),
                "delta_equivalent_usd": (
                    scale * item.delta * spot
                    if scale is not None and item.delta is not None and spot is not None
                    and _position_field_valid(item, "delta") else None
                ),
                "gamma_pnl_1pct_usd": (
                    0.5 * scale * item.gamma * (0.01 * spot) ** 2
                    if scale is not None and item.gamma is not None and spot is not None
                    and _position_field_valid(item, "gamma") else None
                ),
                "vega_per_1pct_usd": (
                    scale * item.vega_per_1pct
                    if scale is not None and item.vega_per_1pct is not None
                    and _position_field_valid(item, "vega") else None
                ),
                "theta_per_day_usd": (
                    scale * item.theta_per_day
                    if scale is not None and item.theta_per_day is not None
                    and _position_field_valid(item, "theta") else None
                ),
                "rho_per_1pct_usd": (
                    scale * item.rho_per_1pct
                    if scale is not None and item.rho_per_1pct is not None
                    and _position_field_valid(item, "rho") else None
                ),
            }
        )
    risk_by_id = {str(item["position_id"]): item for item in position_risk}
    omission_warnings: list[str] = []

    def material_sum(
        rows: list[dict[str, Any]], field: str, *, absolute: bool = False,
        positive: bool | None = None,
    ) -> float | None:
        missing = [str(item["position_id"]) for item in rows if _finite(item.get(field)) is None]
        decision = assess_omission_materiality(
            missing,
            response.data.positions,
            portfolio_value_usd=response.data.kpis.portfolio_value_usd,
            context=context,
        )
        if missing and not decision.may_exclude:
            return None
        if decision.warning and decision.warning not in omission_warnings:
            omission_warnings.append(decision.warning)
        values = [float(item[field]) for item in rows if _finite(item.get(field)) is not None]
        if positive is True:
            return sum(max(0.0, value) for value in values)
        if positive is False:
            return sum(abs(min(0.0, value)) for value in values)
        return sum(abs(value) if absolute else value for value in values)

    signed_value = material_sum(position_risk, "signed_market_value_usd")
    gross_value = material_sum(position_risk, "signed_market_value_usd", absolute=True)
    greek_fields = {
        "delta": "delta_equivalent_usd",
        "gamma": "gamma_pnl_1pct_usd",
        "vega": "vega_per_1pct_usd",
        "theta": "theta_per_day_usd",
        "rho": "rho_per_1pct_usd",
    }
    by_expiry: dict[str, dict[str, Any]] = {}
    for item in positions:
        expiry = item.expiration_at.date().isoformat() if item.expiration_at else "unknown"
        row = by_expiry.setdefault(
            expiry,
            {
                "expiry": expiry,
                "strategy_count": 0,
                "underlyings": set(),
                "warnings": [],
            },
        )
        row["strategy_count"] += 1
        row["underlyings"].add(item.underlying_symbol)
        row["warnings"].extend(item.data_quality.warnings)
    expiry_rows = []
    for row in by_expiry.values():
        row["underlyings"] = sorted(item for item in row["underlyings"] if item)
        row["warnings"] = list(dict.fromkeys(row["warnings"]))
        expiry_positions = [
            item for item in positions
            if (item.expiration_at.date().isoformat() if item.expiration_at else "unknown") == row["expiry"]
        ]
        # Expiration Risk describes the option contracts expiring on this row.
        # Pulling every stock/ETF holding in the same underlying into this
        # payoff made the displayed loss mostly a stock-to-zero calculation
        # and materially overstated the expiry-specific option risk.
        payoff = certify_maximum_loss(expiry_positions)
        row["maximum_loss_state"] = payoff.get("status")
        row["maximum_loss_usd"] = payoff.get("maximum_loss_usd")
        row["unbounded_loss_symbols"] = [
            item.get("underlying_symbol")
            for item in payoff.get("underlying_results") or []
            if item.get("status") == "unbounded"
        ]
        row["payoff"] = payoff
        expiry_risk = [risk_by_id[item.position_id] for item in expiry_positions]
        row["signed_value_usd"] = material_sum(expiry_risk, "signed_market_value_usd")
        row["gross_value_usd"] = material_sum(expiry_risk, "signed_market_value_usd", absolute=True)
        row["long_exposure_usd"] = material_sum(expiry_risk, "signed_market_value_usd", positive=True)
        row["short_exposure_usd"] = material_sum(expiry_risk, "signed_market_value_usd", positive=False)
        for key in greek_fields.values():
            row[key] = material_sum(expiry_risk, key)
        row["field_coverage"] = {
            name: _option_count_coverage(expiry_risk, field)
            for name, field in greek_fields.items()
        }
        row["sector_risk"] = []
        for sector in sorted({item.sector or "Unclassified" for item in expiry_positions}):
            sector_positions = [
                item for item in expiry_positions if (item.sector or "Unclassified") == sector
            ]
            sector_risk = [risk_by_id[item.position_id] for item in sector_positions]
            sector_pnl = [
                item.unrealized_pnl_usd if _position_field_valid(item, "mark") else None
                for item in sector_positions
            ]
            row["sector_risk"].append(
                {
                    "sector": sector,
                    "net_exposure_usd": material_sum(sector_risk, "signed_market_value_usd"),
                    "pnl_usd": _complete_sum(sector_pnl),
                    "delta_equivalent_usd": material_sum(sector_risk, "delta_equivalent_usd"),
                    "theta_per_day_usd": material_sum(sector_risk, "theta_per_day_usd"),
                    "position_ids": [item.position_id for item in sector_positions],
                }
            )
        expiry_rows.append(row)
    expiry_rows.sort(key=lambda item: item["expiry"])
    for index, row in enumerate(expiry_rows):
        row["remaining_signed_value_usd"] = _complete_sum(
            [later.get("signed_value_usd") for later in expiry_rows[index + 1 :]]
        )
    return {
        "signed_option_market_value_usd": signed_value,
        "gross_option_market_value_usd": gross_value,
        "delta_equivalent_usd": material_sum(position_risk, "delta_equivalent_usd"),
        "gamma_pnl_1pct_usd": material_sum(position_risk, "gamma_pnl_1pct_usd"),
        "vega_per_1pct_usd": material_sum(position_risk, "vega_per_1pct_usd"),
        "theta_per_day_usd": material_sum(position_risk, "theta_per_day_usd"),
        "rho_per_1pct_usd": material_sum(position_risk, "rho_per_1pct_usd"),
        "long_contracts": sum(item.signed_quantity for item in positions if item.signed_quantity > 0),
        "short_contracts": abs(sum(item.signed_quantity for item in positions if item.signed_quantity < 0)),
        "held_contracts": sum(abs(item.signed_quantity) for item in positions),
        "greek_coverage": {
            name: _option_count_coverage(position_risk, field)
            for name, field in greek_fields.items()
        },
        "position_risk": position_risk,
        "expiries": expiry_rows,
        "warnings": omission_warnings,
    }


def _position_field_valid(position: Any, field: str) -> bool:
    state = position.field_validity.get(field)
    if state is not None:
        return bool(state.valid_for_risk)
    attribute = {"vega": "vega_per_1pct", "theta": "theta_per_day", "rho": "rho_per_1pct"}.get(field, field)
    return getattr(position, attribute, None) is not None


def _complete_sum(values: list[Any], *, absolute: bool = False) -> float | None:
    if any(_finite(value) is None for value in values):
        return None
    return sum(abs(float(value)) if absolute else float(value) for value in values)


def _complete_directional_sum(values: list[Any], *, positive: bool) -> float | None:
    if any(_finite(value) is None for value in values):
        return None
    if positive:
        return sum(max(0.0, float(value)) for value in values)
    return sum(abs(min(0.0, float(value))) for value in values)


def _option_count_coverage(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    missing = [str(item["position_id"]) for item in rows if _finite(item.get(field)) is None]
    modeled = len(rows) - len(missing)
    return {
        "eligible_positions": len(rows),
        "modeled_positions": modeled,
        "percent": None,
        "count_percent": 100.0 * modeled / len(rows) if rows else 100.0,
        "weight_basis": "eligible_option_count",
        "missing_position_ids": missing,
    }


def _expiry_maximum_loss(
    positions: list[Any], portfolio_positions: list[Any]
) -> tuple[float | None, list[str], str]:
    """Estimate current-mark maximum loss from the piecewise-linear expiry payoff."""
    if not positions or any(
        item.strike is None
        or item.multiplier is None
        or item.signed_market_value_usd is None
        for item in positions
    ):
        return None, [], "unavailable_incomplete_instrument_parameters"
    by_underlying: dict[str, list[Any]] = {}
    for item in positions:
        symbol = item.underlying_symbol or item.exact_symbol
        by_underlying.setdefault(symbol, []).append(item)
    unbounded: list[str] = []
    maximum_loss = 0.0
    for symbol, legs in by_underlying.items():
        call_obligation = max(
            0.0,
            -sum(
                item.signed_quantity * item.multiplier
                for item in legs if item.put_call == "CALL"
            ),
        )
        covering_shares = sum(
            item.signed_quantity
            for item in portfolio_positions
            if item.asset_type in {"EQUITY", "ETF"}
            and item.exact_symbol == symbol
            and item.signed_quantity > 0
        )
        if call_obligation > covering_shares:
            unbounded.append(symbol)
            continue
        strikes = sorted({float(item.strike) for item in legs})
        current_value = sum(float(item.signed_market_value_usd) for item in legs)
        outcomes = []
        for spot in [0.0, *strikes]:
            payoff = 0.0
            for item in legs:
                intrinsic = (
                    max(0.0, spot - float(item.strike))
                    if item.put_call == "CALL"
                    else max(0.0, float(item.strike) - spot)
                )
                payoff += item.signed_quantity * item.multiplier * intrinsic
            outcomes.append(payoff - current_value)
        maximum_loss += max(0.0, -min(outcomes))
    if unbounded:
        return None, sorted(unbounded), "unbounded_uncovered_short_call"
    return maximum_loss, [], "estimated_from_current_mark_and_expiry_payoff"


def _split_values(values: list[str]) -> list[str]:
    return [
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    ] or ["all"]


def _transaction_types(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    parsed = [item.strip().upper() for value in values for item in value.split(",") if item.strip()]
    unsupported = set(parsed) - set(SUPPORTED_TRANSACTION_TYPES)
    if unsupported:
        raise InvalidRiskRequestError("unsupported transaction type")
    return parsed


def _transaction_repository(request: Request) -> TransactionRepository:
    repository = request.app.state.transaction_repository
    if repository is None:
        raise TransactionRepositoryError()
    return repository


def _option_history_service(request: Request) -> RiskOptionHistoryService:
    service = request.app.state.option_history_service
    # Tests and application extensions may replace the portfolio RefData client
    # after startup. Keep the hybrid path on that same current client.
    service.reference_client = request.app.state.portfolio_service.reference_client
    return service


def _historical_revaluation_engine(
    request: Request,
) -> CurrentPortfolioHistoricalRevaluationEngine:
    """The single dependency-complete construction path for Risk consumers."""
    return CurrentPortfolioHistoricalRevaluationEngine(
        request.app.state.portfolio_service.reference_client,
        request.app.state.portfolio_service.context,
        risk_history_service=_option_history_service(request),
    )


def _historical_revalue(
    request: Request,
    positions: Any,
    *,
    as_of: datetime,
    lookback_sessions: int | None = None,
):
    """Share one immutable snapshot revaluation across tail/factor/cluster calls."""
    eligible = tuple(
        item for item in positions if historical_risk_position_eligible(item)
    )
    effective_lookback = lookback_sessions or int(
        request.app.state.portfolio_service.context.get(
            "risk.history.lookback_sessions", 252
        )
    )
    key = historical_revaluation_cache_key(
        request.app.state.portfolio_service.reference_client,
        eligible,
        as_of=as_of,
        lookback_sessions=effective_lookback,
    )
    state = request.app.state
    with _HISTORICAL_CACHE_INIT_LOCK:
        if not hasattr(state, "historical_revaluation_lock"):
            state.historical_revaluation_lock = RLock()
            state.historical_revaluation_cache = {}
    with state.historical_revaluation_lock:
        cached = state.historical_revaluation_cache.get(key)
        if cached is not None:
            return cached
        result = _historical_revaluation_engine(request).revalue(
            eligible,
            as_of=as_of,
            lookback_sessions=effective_lookback,
        )
        state.historical_revaluation_cache[key] = result
        while len(state.historical_revaluation_cache) > 24:
            state.historical_revaluation_cache.pop(
                next(iter(state.historical_revaluation_cache))
            )
        return result


def _pnl_reference_client(request: Request) -> HybridOptionHistoryClient:
    reference_client = request.app.state.portfolio_service.reference_client
    return HybridOptionHistoryClient(
        reference_client,
        _option_history_service(request),
    )


def _configuration_version(context: dict[str, Any]) -> str:
    editable = _editable_configuration(context)
    return "sha256:" + hashlib.sha256(
        json.dumps(editable, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _editable_configuration(context: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "risk.options.greek_source",
        "risk.options.fallback_to_provider",
        "risk.tail_risk.confidence",
        "risk.history.lookback_sessions",
        "risk.clusters.minimum_cluster_correlation",
        "risk.clusters.apply_stressed_correlations",
        "risk.alerts.margin.warning_utilization",
        "risk.alerts.margin.critical_utilization",
        "risk.alerts.position.conditions",
        "risk.alerts.position.rules",
        "risk.refresh.remote_attempt_floor_seconds",
    )
    return {key: context.get(key) for key in keys}


__all__ = ["router"]
