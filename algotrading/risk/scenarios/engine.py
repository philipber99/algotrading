"""Current-portfolio scenarios with anchored Tian option model changes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from threading import RLock
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from algotrading.refdata.options.solvers.Registry import SolverRegistry
from algotrading.refdata.options.solvers.Types import PricingRequest, RepricingMethod

from ..models import NormalizedPosition, PortfolioSnapshot
from ..materiality import assess_omission_materiality
from ..market_inputs.dividends import escrowed_dividend_adjustment
from ..payoff.engine import option_intrinsic_per_contract
from ..pricing.quantlib_european import (
    EuropeanPricingUnavailable,
    price_european,
)
from ..pricing.tian_service import (
    TianIvGreeksService,
    TianRepriceContract,
    TianRepriceItem,
    TianRepriceRequest,
)


class ScenarioDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    scenario_id: str
    name: str
    methodology_version: str = "economic_scenario_engine_v5"
    exact_underlying_shocks: dict[str, float] = Field(default_factory=dict)
    underlying_shocks: dict[str, float] = Field(default_factory=dict)
    broad_equity_shock: float | None = None
    sector_shocks: dict[str, float] = Field(default_factory=dict)
    futures_root_shocks: dict[str, float] = Field(default_factory=dict)
    commodity_root_shocks: dict[str, float] = Field(default_factory=dict)
    fx_pair_shocks: dict[str, float] = Field(default_factory=dict)
    yield_curve_basis_points: dict[str, float] = Field(default_factory=dict)
    default_equity_shock: float | None = None
    iv_points: float = 0.0
    rate_basis_points: float = 0.0
    horizon_calendar_days: float = 0.0
    sector_rotation_sector: str | None = None
    sector_rotation_move: float | None = None
    sector_rotation_diagnostics: dict[str, Any] = Field(default_factory=dict)
    etf_response_diagnostics: dict[str, Any] = Field(default_factory=dict)
    path_rule: Literal[
        "instantaneous_and_held", "log_linear_to_horizon", "explicit_dated_points"
    ] | None = "instantaneous_and_held"
    dated_path_points: dict[datetime, float] = Field(default_factory=dict)


class ScenarioEngine:
    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        registry: SolverRegistry | None = None,
    ) -> None:
        self.context = context
        self._registry = registry
        self._registry_was_injected = registry is not None
        self._tian_service = TianIvGreeksService(context) if registry is None else None
        self._lock = RLock()
        self._reprice_cache: dict[tuple[Any, ...], TianRepriceItem] = {}
        self._reprice_cache_max_entries = max(
            500,
            int(context.get("risk.scenarios.reprice_cache_max_entries", 20_000)),
        )

    def _cached_tian_reprice(
        self,
        valuation_at: datetime,
        contract: TianRepriceContract,
    ) -> TianRepriceItem:
        """Reuse identical fixed-IV states across scenarios and candidates."""

        # Include every pricing input so cached reprices cannot cross scenario states.
        key = (
            valuation_at.isoformat(),
            contract.canonical_occ_symbol,
            contract.underlying_symbol,
            contract.put_call,
            contract.exercise_type,
            contract.strike,
            contract.expiration_at.isoformat(),
            contract.source_valuation_at.isoformat(),
            contract.source_spot,
            contract.source_bid,
            contract.source_ask,
            contract.source_mark,
            contract.implied_volatility,
            contract.scenario_spot,
            contract.iv_shift_points,
            contract.risk_free_rate,
            contract.dividend_yield,
        )
        with self._lock:
            cached = self._reprice_cache.get(key)
        if cached is not None:
            return cached
        assert self._tian_service is not None
        item = self._tian_service.reprice(
            TianRepriceRequest(valuation_at=valuation_at, contracts=[contract])
        ).data[0]
        with self._lock:
            self._reprice_cache[key] = item
            while len(self._reprice_cache) > self._reprice_cache_max_entries:
                self._reprice_cache.pop(next(iter(self._reprice_cache)))
        return item

    def evaluate(
        self,
        snapshot: PortfolioSnapshot,
        scenario: ScenarioDefinition,
    ) -> dict[str, Any]:
        contributions = []
        unmodeled = []
        material_unmodeled = []
        warnings: list[str] = []
        if scenario.default_equity_shock is not None:
            warnings.append("legacy_default_equity_shock_translated_to_broad_equity")
        for position in snapshot.positions:
            result = self._position_pnl(position, scenario, snapshot.as_of)
            if result is None:
                unmodeled.append(position.position_id)
                if omission_requires_fail_closed(position):
                    material_unmodeled.append(position.position_id)
                continue
            contributions.append(
                {
                    "position_id": position.position_id,
                    "economic_position_id": position.economic_position_id,
                    "symbol": position.exact_symbol,
                    "sector": position.sector,
                    "pnl_usd": result["pnl_usd"],
                    "calculation_method": result["calculation_method"],
                    "lifecycle_event": result.get("lifecycle_event"),
                    "solver_diagnostics": result.get("solver_diagnostics"),
                }
            )
        # Evaluate missing positions before deciding whether a portfolio total is usable.
        omission = assess_omission_materiality(
            unmodeled,
            snapshot.positions,
            portfolio_value_usd=snapshot.portfolio_value_usd,
            context=self.context,
        )
        if omission.may_exclude:
            material_unmodeled = []
            if omission.warning:
                warnings.append(omission.warning)
        pnl = sum(item["pnl_usd"] for item in contributions)
        eligible = [
            item for item in snapshot.positions if _scenario_position_eligible(item)
        ]
        modeled_ids = {item["position_id"] for item in contributions}
        eligible_ids = {item.position_id for item in eligible}
        eligible_count = len(eligible)
        modeled_count = len(modeled_ids & eligible_ids)
        no_modeled_failure = bool(
            eligible_count and modeled_count == 0 and not omission.may_exclude
        )
        reported_pnl = (
            None
            if material_unmodeled or no_modeled_failure
            else pnl
        )
        coverage = 100 * modeled_count / eligible_count if eligible_count else 100.0
        rotation_unmodeled = list(
            scenario.sector_rotation_diagnostics.get("unmodeled_symbols") or []
        )
        etf_unmodeled = list(
            scenario.etf_response_diagnostics.get("unmodeled_symbols") or []
        )
        warnings.extend(["partial_scenario_coverage"] if unmodeled else [])
        warnings.extend(
            ["material_nonlinear_or_unknown_exposure_unmodeled"]
            if material_unmodeled
            else []
        )
        warnings.extend(
            ["partial_sector_rotation_history"] if rotation_unmodeled else []
        )
        warnings.extend(["partial_etf_response_history"] if etf_unmodeled else [])
        return {
            "scenario_id": scenario.scenario_id,
            "name": scenario.name,
            "methodology_version": scenario.methodology_version,
            "pricing_method": "typed_economic_shocks_and_anchored_option_revaluation",
            "scenario_pnl_usd": reported_pnl,
            "scenario_loss_usd": (
                max(0.0, -reported_pnl) if reported_pnl is not None else None
            ),
            "percent_nav": (
                100 * reported_pnl / snapshot.portfolio_value_usd
                if reported_pnl is not None
                and snapshot.portfolio_value_usd
                and snapshot.portfolio_value_usd > 0
                else None
            ),
            "new_modeled_nav_usd": (
                snapshot.portfolio_value_usd + reported_pnl
                if snapshot.portfolio_value_usd is not None
                and reported_pnl is not None
                else None
            ),
            "contributions": contributions,
            "coverage": {
                "eligible_positions": eligible_count,
                "modeled_positions": modeled_count,
                "percent": coverage,
                "count_percent": coverage,
                "weight_basis": "full_revaluation_eligible_economic_unit_count",
                "denominator_usd": None,
                "modeled_usd": None,
                "missing_position_ids": sorted(unmodeled),
                "unavailable_denominator_position_ids": sorted(unmodeled),
            },
            "unmodeled_position_ids": unmodeled,
            "material_unmodeled_position_ids": material_unmodeled,
            "excluded_position_ids": (
                list(omission.position_ids) if omission.may_exclude else []
            ),
            "excluded_gross_marked_value_usd": (
                omission.gross_marked_value_usd if omission.may_exclude else None
            ),
            "excluded_portfolio_percent": (
                100.0 * float(omission.portfolio_fraction)
                if omission.may_exclude and omission.portfolio_fraction is not None
                else None
            ),
            "quality_state": (
                "insufficient_data"
                if material_unmodeled or no_modeled_failure
                else "usable"
                if not warnings
                else "usable_with_warnings"
            ),
            "warnings": warnings,
            "assumptions": scenario.model_dump(mode="json"),
        }

    def _position_pnl(
        self,
        position: NormalizedPosition,
        scenario: ScenarioDefinition,
        valuation_at: datetime,
    ) -> dict[str, Any] | None:
        if not _scenario_position_eligible(position):
            return {
                "pnl_usd": 0.0,
                "calculation_method": "cash_equivalent_no_direct_channel",
            }
        symbol = position.underlying_symbol or position.exact_symbol
        shock = self._economic_price_shock(position, scenario)
        if position.asset_type != "CASH" and not _field_valid(position, "mark"):
            return None
        if shock is None and position.asset_type != "FUTURE":
            economic_type = _economic_type(position, self.context)
            option_nonprice_channel = (
                position.asset_type == "OPTION"
                and (scenario.iv_points != 0 or scenario.rate_basis_points != 0)
            )
            european_option_zero_price_channel = bool(
                position.asset_type == "OPTION"
                and position.instrument_terms is not None
                and position.instrument_terms.exercise_style == "european"
            )
            if (
                _scenario_has_price_channel(scenario)
                and economic_type
                in {"rates", "commodity", "currency", "mixed", "unknown"}
                and not option_nonprice_channel
                and not european_option_zero_price_channel
            ):
                return None
            shock = 0.0
        maximum = float(self.context.get("risk.scenarios.maximum_absolute_equity_shock", 0.50))
        if shock is not None:
            shock = max(-maximum, min(maximum, float(shock)))
        if position.asset_type in {"EQUITY", "ETF", "MUTUAL_FUND"}:
            if position.mark is None:
                return None
            return {
                "pnl_usd": position.signed_quantity * position.mark * float(shock),
                "calculation_method": "exact_or_classified_equity_return_shock",
            }
        if position.asset_type == "FUTURE":
            if position.mark is None or position.multiplier is None:
                return None
            root = (position.futures_root or "").removeprefix("/")
            # Convert validated Treasury DV01 into P&L using yield changes in basis points.
            if root in {"ZN", "ZF", "ZB", "ZT", "UB"} and (
                scenario.rate_basis_points
                or scenario.yield_curve_basis_points
            ):
                dv01 = _finite(position.raw_broker_fields.get("dv01_usd_per_bp"))
                if dv01 is None:
                    return None
                basis_points = scenario.rate_basis_points or float(
                    scenario.yield_curve_basis_points.get("parallel", 0.0)
                )
                return {
                    "pnl_usd": -position.signed_quantity * dv01 * basis_points,
                    "calculation_method": "validated_futures_dv01",
                }
            if shock is None:
                return {
                    "pnl_usd": 0.0,
                    "calculation_method": "no_direct_channel_for_typed_scenario",
                }
            return {
                "pnl_usd": position.signed_quantity * position.multiplier * position.mark * shock,
                "calculation_method": "futures_root_price_shock_with_resolved_multiplier",
            }
        if position.asset_type == "CASH":
            return {"pnl_usd": 0.0, "calculation_method": "cash_no_direct_channel"}
        if position.asset_type != "OPTION":
            return None
        if any(
            value is None
            for value in (
                position.multiplier,
                position.strike,
                position.expiration_at,
                position.mark,
            )
        ):
            return None
        terms = position.instrument_terms
        if terms is None or terms.contract_subtype == "futures_option":
            return None
        if terms.exercise_style not in {"american", "european"}:
            return None
        spot = position.raw_broker_fields.get("underlying_price")
        rate = position.raw_broker_fields.get("risk_free_rate_continuous")
        dividend = position.raw_broker_fields.get("dividend_yield_continuous", 0.0)
        if spot is None or rate is None:
            return None
        tenor_seconds = (
            position.expiration_at.astimezone(timezone.utc)
            - valuation_at.astimezone(timezone.utc)
        ).total_seconds()
        shocked_at = valuation_at + timedelta(days=scenario.horizon_calendar_days)
        shocked_spot = float(spot) * (1 + float(shock))
        shocked_rate = float(rate) + scenario.rate_basis_points / 10_000
        # Crossing expiry requires a dated path and authoritative delivery terms.
        if shocked_at >= position.expiration_at:
            if scenario.path_rule is None:
                return None
            expiry_spot = _expiry_spot(
                float(spot), shocked_spot, valuation_at, position.expiration_at,
                shocked_at, scenario,
            )
            if expiry_spot is None:
                return None
            try:
                intrinsic = option_intrinsic_per_contract(position, expiry_spot)
            except ValueError:
                return None
            deliverable = terms.deliverable
            if deliverable is None or not deliverable.fully_supported:
                return None
            delivered_shares = sum(
                float(component.quantity or 0.0)
                for component in deliverable.components
                if component.component_type == "security"
                and component.security_identifier == terms.underlying_identifier
            )
            resulting_quantity = (
                0.0
                if terms.settlement_type == "cash" or intrinsic <= 0
                else position.signed_quantity * delivered_shares
                * (1.0 if position.put_call == "CALL" else -1.0)
            )
            # Physical exercise leaves underlying exposure through the remaining horizon.
            post_expiry_pnl = resulting_quantity * (shocked_spot - expiry_spot)
            premium_value = float(position.mark) * float(position.multiplier)
            return {
                "pnl_usd": position.signed_quantity * (intrinsic - premium_value) + post_expiry_pnl,
                "calculation_method": "authoritative_expiry_event_path_and_resulting_exposure_v2",
                "lifecycle_event": {
                    "event": "cash_settlement"
                    if terms is not None and terms.settlement_type == "cash"
                    else "exercise_or_assignment",
                    "expiration_at": position.expiration_at.isoformat(),
                    "resulting_underlying_quantity": (
                        None
                        if terms is not None and terms.settlement_type == "cash"
                        else resulting_quantity
                    ),
                    "continued_through_horizon": True,
                    "path_rule": scenario.path_rule,
                    "expiry_underlying_price": expiry_spot,
                    "horizon_underlying_price": shocked_spot,
                    "expiry_to_horizon_pnl_usd": post_expiry_pnl,
                },
            }
        if position.implied_volatility is None:
            return None
        shocked_iv = max(
            float(self.context.get("risk.scenarios.minimum_iv", 0.0001)),
            float(position.implied_volatility) + scenario.iv_points / 100,
        )
        shocked_tenor = max(
            1.0,
            (position.expiration_at.astimezone(timezone.utc) - shocked_at).total_seconds(),
        )
        dividend_inputs = position.raw_broker_fields.get("dividend_inputs") or ()
        pricing_spot = float(spot)
        shocked_pricing_spot = shocked_spot
        pricing_dividend = float(dividend)
        dividend_diagnostics: dict[str, Any] | None = None
        # Escrow cash dividends consistently in both base and shocked valuations.
        if dividend_inputs and terms.exercise_style == "american":
            if not _as_bool(
                self.context.get(
                    "risk.options.tian.escrowed_dividend_mode_enabled", True
                )
            ):
                return None
            try:
                base_adjustment = escrowed_dividend_adjustment(
                    dividend_inputs,
                    spot=float(spot),
                    valuation_at=valuation_at,
                    expiration_at=position.expiration_at,
                    continuous_rate=float(rate),
                )
                shocked_adjustment = escrowed_dividend_adjustment(
                    dividend_inputs,
                    spot=shocked_spot,
                    valuation_at=shocked_at,
                    expiration_at=position.expiration_at,
                    continuous_rate=shocked_rate,
                )
            except (TypeError, ValueError):
                return None
            if base_adjustment.event_count or shocked_adjustment.event_count:
                pricing_spot = base_adjustment.adjusted_spot
                shocked_pricing_spot = shocked_adjustment.adjusted_spot
                pricing_dividend = 0.0
                dividend_diagnostics = {
                    "model": base_adjustment.algorithm_version,
                    "base_dividend_present_value": base_adjustment.present_value,
                    "shocked_dividend_present_value": shocked_adjustment.present_value,
                    "base_adjusted_spot": base_adjustment.adjusted_spot,
                    "shocked_adjusted_spot": shocked_adjustment.adjusted_spot,
                    "event_count": base_adjustment.event_count,
                    "event_dates": [
                        item.isoformat() for item in base_adjustment.event_dates
                    ],
                }
        # Route European exercise to the analytic pricer and American exercise to Tian.
        if terms.exercise_style == "european":
            try:
                base_valuation = price_european(
                    put_call=position.put_call or "CALL",
                    spot=pricing_spot,
                    strike=float(position.strike),
                    valuation_at=valuation_at,
                    expiration_at=position.expiration_at,
                    rate=float(rate),
                    dividend_yield=pricing_dividend,
                    volatility=float(position.implied_volatility),
                )
                shocked_valuation = price_european(
                    put_call=position.put_call or "CALL",
                    spot=shocked_pricing_spot,
                    strike=float(position.strike),
                    valuation_at=shocked_at,
                    expiration_at=position.expiration_at,
                    rate=shocked_rate,
                    dividend_yield=pricing_dividend,
                    volatility=shocked_iv,
                )
            except EuropeanPricingUnavailable:
                return None
            return {
                "pnl_usd": position.signed_quantity
                * float(position.multiplier)
                * (shocked_valuation.price - base_valuation.price),
                "calculation_method": "quantlib_analytic_european_change",
                "solver_diagnostics": {
                    "engine": "AnalyticEuropeanEngine",
                    "algorithm_version": base_valuation.algorithm_version,
                },
            }
        base_request = PricingRequest(
            option_symbol=position.canonical_occ_symbol or position.exact_symbol,
            snapshot_id="scenario-base",
            put_call=position.put_call or "CALL",
            exercise_type="A",
            spot=pricing_spot,
            strike=float(position.strike),
            tenor_seconds=max(1.0, tenor_seconds),
            rate=float(rate),
            dividend_yield=pricing_dividend,
            requested_dtype="float64",
        )
        shocked_request = PricingRequest(
            option_symbol=base_request.option_symbol,
            snapshot_id="scenario-shocked",
            put_call=base_request.put_call,
            exercise_type="A",
            spot=shocked_pricing_spot,
            strike=base_request.strike,
            tenor_seconds=shocked_tenor,
            rate=shocked_rate,
            dividend_yield=base_request.dividend_yield,
            requested_dtype="float64",
        )
        try:
            if self._registry_was_injected:
                adapter = self._get_registry().get(RepricingMethod.TIAN)
                base_value = adapter.price(base_request, float(position.implied_volatility))
                shocked_value = adapter.price(shocked_request, shocked_iv)
                diagnostics: dict[str, Any] = {"external_registry": True}
            else:
                assert self._tian_service is not None
                common = {
                    "canonical_occ_symbol": position.canonical_occ_symbol or position.exact_symbol,
                    "underlying_symbol": position.underlying_symbol or position.exact_symbol,
                    "put_call": position.put_call or "CALL",
                    "exercise_type": "AMERICAN",
                    "strike": float(position.strike),
                    "expiration_at": position.expiration_at,
                    "source_valuation_at": valuation_at,
                    "source_spot": pricing_spot,
                    "source_mark": position.mark,
                    "implied_volatility": float(position.implied_volatility),
                    "risk_free_rate": float(rate),
                    "dividend_yield": pricing_dividend,
                }
                base_item = self._cached_tian_reprice(
                    valuation_at,
                    TianRepriceContract(
                        request_id=f"scenario-base:{position.position_id}",
                        scenario_spot=pricing_spot,
                        **common,
                    ),
                )
                shocked_item = self._cached_tian_reprice(
                    shocked_at,
                    TianRepriceContract(
                        request_id=f"scenario-shocked:{position.position_id}",
                        scenario_spot=shocked_pricing_spot,
                        iv_shift_points=scenario.iv_points,
                        **common,
                    ),
                )
                if base_item.status != "ok" or shocked_item.status != "ok":
                    return None
                base_value = float(base_item.price)
                shocked_value = float(shocked_item.price)
                diagnostics = {
                    "base": base_item.solver_diagnostics,
                    "shocked": shocked_item.solver_diagnostics,
                }
            if dividend_diagnostics is not None:
                diagnostics["escrowed_dividend"] = dividend_diagnostics
        except Exception:
            return None
        # Reject failed reprices before aggregating their dollar changes.
        if not math.isfinite(base_value) or not math.isfinite(shocked_value):
            return None
        return {
            "pnl_usd": position.signed_quantity
            * float(position.multiplier)
            * (shocked_value - base_value),
            "calculation_method": (
                "current_exact_mark_anchored_tian_escrowed_dividend_change"
                if dividend_diagnostics is not None
                else "current_exact_mark_anchored_tian_change"
            ),
            "solver_diagnostics": diagnostics,
        }

    def _economic_price_shock(
        self, position: NormalizedPosition, scenario: ScenarioDefinition
    ) -> float | None:
        symbol = position.underlying_symbol or position.exact_symbol
        exact = {
            **scenario.underlying_shocks,
            **scenario.exact_underlying_shocks,
        }
        # Exact symbol shocks take precedence over sector and asset-class mappings.
        if symbol in exact:
            return float(exact[symbol])
        if position.asset_type in {"EQUITY", "ETF", "MUTUAL_FUND", "OPTION"}:
            economic_type = _economic_type(position, self.context)
            mapping = _economic_mapping(position, self.context)
            if economic_type == "rates":
                duration = _finite(
                    mapping.get("effective_duration")
                    or position.raw_broker_fields.get("effective_duration")
                )
                basis_points = scenario.rate_basis_points or float(
                    scenario.yield_curve_basis_points.get("parallel", 0.0)
                )
                return (
                    -duration * basis_points / 10_000
                    if duration is not None and basis_points
                    else None
                )
            if economic_type == "commodity":
                factor = str(mapping.get("factor") or symbol)
                return _finite(scenario.commodity_root_shocks.get(factor))
            if economic_type == "currency":
                factor = str(mapping.get("factor") or symbol)
                return _finite(scenario.fx_pair_shocks.get(factor))
            if economic_type != "equity":
                return None
            if position.sector in scenario.sector_shocks:
                return float(scenario.sector_shocks[position.sector])
            broad = (
                scenario.broad_equity_shock
                if scenario.broad_equity_shock is not None
                else scenario.default_equity_shock
            )
            return float(broad) if broad is not None else None
        if position.asset_type != "FUTURE":
            return None
        root = (position.futures_root or "").removeprefix("/")
        if root in scenario.futures_root_shocks:
            return float(scenario.futures_root_shocks[root])
        if root in scenario.commodity_root_shocks:
            return float(scenario.commodity_root_shocks[root])
        if root in scenario.fx_pair_shocks:
            return float(scenario.fx_pair_shocks[root])
        if root in {"ES", "MES", "NQ", "MNQ", "RTY", "M2K", "YM"}:
            broad = (
                scenario.broad_equity_shock
                if scenario.broad_equity_shock is not None
                else scenario.default_equity_shock
            )
            return float(broad) if broad is not None else None
        return None

    def _get_registry(self) -> SolverRegistry:
        if self._registry is None:
            with self._lock:
                if self._registry is None:
                    self._registry = SolverRegistry(dict(self.context))
        return self._registry


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _field_valid(position: NormalizedPosition, field: str) -> bool:
    state = position.field_validity.get(field)
    return state.valid_for_risk if state is not None else getattr(position, field, None) is not None


def _scenario_position_eligible(position: NormalizedPosition) -> bool:
    """Exclude cash equivalents consistently with the historical risk engines."""
    broker_type = str(
        position.raw_broker_fields.get("instrument_type") or ""
    ).upper()
    return (
        position.asset_type != "CASH"
        and "NO_LOAD" not in broker_type
        and "MONEY_MARKET" not in broker_type
    )


def _economic_mapping(
    position: NormalizedPosition, context: Mapping[str, Any]
) -> Mapping[str, Any]:
    configured = context.get("risk.scenarios.economic_instrument_mappings", {})
    if not isinstance(configured, Mapping):
        return {}
    symbol = position.underlying_symbol or position.exact_symbol
    row = configured.get(position.exact_symbol) or configured.get(symbol)
    return row if isinstance(row, Mapping) else {}


def omission_requires_fail_closed(position: NormalizedPosition) -> bool:
    """Identify omitted exposure whose magnitude cannot be bounded by position count."""

    if position.asset_type in {"OPTION", "UNKNOWN"}:
        return True
    if position.asset_type == "FUTURE":
        return True
    if (
        position.asset_type in {"EQUITY", "ETF", "MUTUAL_FUND"}
        and position.signed_quantity < 0
    ):
        return True
    raw_type = str(
        position.raw_broker_fields.get("economic_exposure_type") or ""
    ).lower()
    terms_type = str(
        getattr(position.instrument_terms, "economic_exposure_type", "") or ""
    ).lower()
    return raw_type in {
        "mixed",
        "unknown",
    } or terms_type in {
        "mixed",
        "unknown",
    }


def _economic_type(
    position: NormalizedPosition, context: Mapping[str, Any]
) -> str:
    mapping = _economic_mapping(position, context)
    mapped = str(mapping.get("economic_exposure_type") or "").lower()
    if mapped in {"equity", "rates", "commodity", "currency", "mixed", "unknown"}:
        return mapped
    raw = str(position.raw_broker_fields.get("economic_exposure_type") or "").lower()
    if raw in {"equity", "rates", "commodity", "currency", "mixed", "unknown"}:
        return raw
    if position.asset_type == "EQUITY":
        return "equity"
    if position.asset_type == "OPTION" and position.instrument_terms is not None:
        return position.instrument_terms.economic_exposure_type
    if position.asset_type == "FUTURE" and position.instrument_terms is not None:
        return position.instrument_terms.economic_exposure_type
    return "unknown"


def _scenario_has_price_channel(scenario: ScenarioDefinition) -> bool:
    return any(
        (
            scenario.exact_underlying_shocks,
            scenario.underlying_shocks,
            scenario.sector_shocks,
            scenario.futures_root_shocks,
            scenario.commodity_root_shocks,
            scenario.fx_pair_shocks,
            scenario.broad_equity_shock is not None,
            scenario.default_equity_shock is not None,
        )
    )


def _expiry_spot(
    start_spot: float,
    horizon_spot: float,
    valuation_at: datetime,
    expiration_at: datetime,
    horizon_at: datetime,
    scenario: ScenarioDefinition,
) -> float | None:
    if scenario.path_rule == "instantaneous_and_held":
        return horizon_spot
    if scenario.path_rule == "log_linear_to_horizon":
        total = (horizon_at - valuation_at).total_seconds()
        elapsed = (expiration_at - valuation_at).total_seconds()
        if total <= 0 or elapsed < 0 or horizon_spot <= 0:
            return None
        # Interpolate the expiry price along the declared log-linear horizon path.
        fraction = min(1.0, elapsed / total)
        return start_spot * math.exp(math.log(horizon_spot / start_spot) * fraction)
    if scenario.path_rule == "explicit_dated_points":
        points = sorted(
            (at.astimezone(timezone.utc), float(value))
            for at, value in scenario.dated_path_points.items()
            if at.tzinfo is not None
        )
        eligible = [value for at, value in points if at <= expiration_at]
        return eligible[-1] if eligible else None
    return None


DEFAULT_SCENARIOS = (
    ScenarioDefinition(scenario_id="market_down_5_iv_up_10", name="Broad market -5% / IV +10", broad_equity_shock=-0.05, iv_points=10),
    ScenarioDefinition(scenario_id="market_down_10_iv_up_20", name="Broad market -10% / IV +20", broad_equity_shock=-0.10, iv_points=20),
    ScenarioDefinition(scenario_id="market_up_5_iv_down_5", name="Broad market +5% / IV -5", broad_equity_shock=0.05, iv_points=-5),
    ScenarioDefinition(scenario_id="technology_down_8", name="Technology sector -8%", sector_shocks={"Technology": -0.08}),
    ScenarioDefinition(scenario_id="financials_down_8", name="Financials sector -8%", sector_shocks={"Financials": -0.08}),
    ScenarioDefinition(scenario_id="energy_down_10", name="Energy sector -10%", sector_shocks={"Energy": -0.10}),
    ScenarioDefinition(scenario_id="rates_up_100bp", name="Yield curve +100 bp", rate_basis_points=100, yield_curve_basis_points={"parallel": 100}),
    ScenarioDefinition(scenario_id="rates_down_100bp", name="Yield curve -100 bp", rate_basis_points=-100, yield_curve_basis_points={"parallel": -100}),
    ScenarioDefinition(scenario_id="crude_down_15", name="Crude oil roots -15%", commodity_root_shocks={"CL": -0.15, "MCL": -0.15}),
    ScenarioDefinition(scenario_id="gold_up_10", name="Gold roots +10%", commodity_root_shocks={"GC": 0.10, "MGC": 0.10}),
    ScenarioDefinition(scenario_id="eurusd_down_5", name="EUR/USD futures -5%", fx_pair_shocks={"6E": -0.05, "M6E": -0.05}),
    ScenarioDefinition(scenario_id="iv_up_10", name="Implied volatility +10 points", iv_points=10),
    ScenarioDefinition(scenario_id="iv_down_10", name="Implied volatility -10 points", iv_points=-10),
    ScenarioDefinition(scenario_id="one_month_market_down_8", name="One-month broad market -8% / IV +12", broad_equity_shock=-0.08, iv_points=12, horizon_calendar_days=30),
)


__all__ = [
    "DEFAULT_SCENARIOS",
    "ScenarioDefinition",
    "ScenarioEngine",
    "omission_requires_fail_closed",
]
