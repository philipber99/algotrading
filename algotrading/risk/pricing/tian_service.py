"""Risk-facing vectorized Tian IV and Greek service."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from threading import RLock
from time import monotonic
from typing import Any, Callable, Literal, Mapping
import uuid

from pydantic import BaseModel, ConfigDict, Field

from algotrading.refdata.options.solvers.Registry import SolverRegistry
from algotrading.refdata.options.solvers.Types import (
    IvHint,
    PricingRequest,
    RepricingMethod,
    SolverError,
    SolverBatchJob,
    SolverOperation,
)

from .quantlib_european import (
    EuropeanPricingUnavailable,
    price_european,
    solve_european_iv_and_greeks,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class IvGreeksContract(StrictModel):
    request_id: str = Field(min_length=1, max_length=256)
    canonical_occ_symbol: str
    underlying_symbol: str
    put_call: Literal["CALL", "PUT"]
    exercise_type: Literal["AMERICAN", "A", "EUROPEAN", "E"] = "AMERICAN"
    strike: float = Field(gt=0)
    expiration_at: datetime
    spot: float = Field(gt=0)
    bid: float | None = None
    ask: float | None = None
    mark: float | None = None
    last: float | None = None
    last_trade_at: datetime | None = None
    quote_time: datetime | None = None
    source_iv: float | None = Field(default=None, gt=0)
    risk_free_rate: float | None = None
    dividend_yield: float | None = None


class IvGreeksRequest(StrictModel):
    valuation_at: datetime
    contracts: list[IvGreeksContract] = Field(min_length=1, max_length=500)


class IvGreeksItem(StrictModel):
    request_id: str
    canonical_occ_symbol: str
    status: Literal["ok", "usable_with_warnings", "unable_to_evaluate"]
    reason: str
    failure_stage: str | None = None
    input_basis: Literal["market_mid", "provider_mark", "last_trade"] | None = None
    target_price: float | None = None
    implied_volatility: float | None = None
    model_price: float | None = None
    price_residual: float | None = None
    iv_iterations: int | None = None
    delta: float | None = None
    gamma: float | None = None
    theta_per_day: float | None = None
    vega_per_1pct: float | None = None
    rho_per_1pct: float | None = None
    spot: float
    tenor_years: float | None = None
    risk_free_rate_continuous: float | None = None
    risk_free_rate_source: str | None = None
    dividend_yield_continuous: float | None = None
    dividend_yield_source: str | None = None
    method: Literal["tian", "quantlib_analytic_european"] = "tian"
    method_selection_reason: str = "risk_api_configured_tian"
    algorithm_version: str | None = None
    solver_parameters: dict[str, Any] = Field(default_factory=dict)
    solver_configuration_hash: str | None = None
    requested_dtype: Literal["float32", "float64"]
    actual_dtype: str | None = None
    precision_escalated: bool = False
    quote_time: datetime | None = None
    calculation_time: datetime
    quality_state: Literal["usable", "usable_with_warnings", "insufficient_data"]
    warnings: list[str] = Field(default_factory=list)


class IvGreeksResponse(StrictModel):
    schema_version: int = 1
    method_version: str = "risk_tian_quantlib_iv_greeks_v3"
    as_of: datetime
    data: list[IvGreeksItem]


class TianRepriceContract(StrictModel):
    """One fixed-IV Tian valuation state for a listed option."""

    request_id: str = Field(min_length=1, max_length=256)
    canonical_occ_symbol: str
    underlying_symbol: str
    put_call: Literal["CALL", "PUT"]
    exercise_type: Literal["AMERICAN", "A", "EUROPEAN", "E"] = "AMERICAN"
    strike: float = Field(gt=0)
    expiration_at: datetime
    source_valuation_at: datetime
    source_spot: float = Field(gt=0)
    source_bid: float | None = None
    source_ask: float | None = None
    source_mark: float | None = None
    implied_volatility: float | None = Field(default=None, gt=0)
    scenario_spot: float = Field(gt=0)
    iv_shift_points: float = Field(default=0.0, ge=-500.0, le=500.0)
    risk_free_rate: float | None = None
    dividend_yield: float | None = None


class TianRepriceRequest(StrictModel):
    valuation_at: datetime
    contracts: list[TianRepriceContract] = Field(min_length=1, max_length=500)


class TianRepriceItem(StrictModel):
    request_id: str
    canonical_occ_symbol: str
    status: Literal["ok", "unable_to_evaluate"]
    reason: str
    price: float | None = None
    implied_volatility: float | None = None
    scenario_volatility: float | None = None
    method: Literal["tian", "quantlib_analytic_european"] = "tian"
    solver_diagnostics: dict[str, Any] = Field(default_factory=dict)


class TianRepriceResponse(StrictModel):
    schema_version: int = 1
    method_version: str = "risk_tian_quantlib_fixed_iv_reprice_v3"
    as_of: datetime
    valuation_at: datetime
    data: list[TianRepriceItem]


class TianIvGreeksService:
    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        registry: SolverRegistry | None = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.context = context
        self._registry = registry
        self._registry_was_injected = registry is not None
        self._adaptive_registries: dict[int, SolverRegistry] = {}
        self._registry_lock = RLock()
        self._now = now_fn
        self._clock = clock
        self._cache: dict[str, tuple[float, IvGreeksItem]] = {}
        self._cache_lock = RLock()
        # Startup analyses arrive concurrently. Serializing the cache-miss phase
        # prevents the same option state from entering the expensive solver in
        # several requests before the first request has populated the cache.
        self._solve_lock = RLock()

    def preload(self) -> None:
        registry = self._get_registry()
        preload = getattr(registry, "preload", None)
        if callable(preload):
            preload()

    def solve(self, request: IvGreeksRequest) -> IvGreeksResponse:
        with self._solve_lock:
            return self._solve(request)

    def _solve(self, request: IvGreeksRequest) -> IvGreeksResponse:
        valuation_at = _aware_utc(request.valuation_at)
        calculation_time = _aware_utc(self._now())
        requested_dtype = str(self.context.get("risk.options.requested_dtype", "float64")).lower()
        if requested_dtype not in {"float32", "float64"}:
            requested_dtype = "float64"
        # Track batch indices so failures and cached results preserve request order.
        output: list[IvGreeksItem | None] = [None] * len(request.contracts)
        jobs: list[SolverBatchJob] = []
        job_metadata: list[tuple[int, IvGreeksContract, str, float, list[str], str]] = []
        for index, contract in enumerate(request.contracts):
            basis = self._target_basis(contract, valuation_at)
            if basis is None:
                output[index] = self._failure(
                    contract,
                    requested_dtype,
                    calculation_time,
                    reason="no_usable_market_price",
                    stage="market_price_selection",
                )
                continue
            input_basis, target_price, warnings = basis
            tenor_seconds = (_aware_utc(contract.expiration_at) - valuation_at).total_seconds()
            if tenor_seconds <= 0:
                output[index] = self._failure(
                    contract,
                    requested_dtype,
                    calculation_time,
                    reason="expired_or_invalid_tenor",
                    stage="input_validation",
                    basis=input_basis,
                    target=target_price,
                )
                continue
            rate = contract.risk_free_rate
            if rate is None:
                configured = self.context.get("risk.rates.configured_flat_continuous_rate")
                rate = float(configured) if configured not in (None, "") else None
                rate_source = "configured_flat_rate" if rate is not None else None
            else:
                rate_source = "request"
            if rate is None or not math.isfinite(rate):
                output[index] = self._failure(
                    contract,
                    requested_dtype,
                    calculation_time,
                    reason="risk_free_rate_unavailable",
                    stage="market_inputs",
                    basis=input_basis,
                    target=target_price,
                )
                continue
            dividend = contract.dividend_yield
            dividend_source = "request"
            if dividend is None:
                dividend = float(self.context.get("risk.dividends.default_continuous_yield", 0.0))
                dividend_source = "configured_default"
                warnings.append("dividend_yield_defaulted")
            cache_key = self._cache_key(
                contract,
                valuation_at,
                target_price,
                rate,
                dividend,
                requested_dtype,
            )
            cached = self._cache_get(cache_key)
            if cached is not None:
                output[index] = cached.model_copy(
                    update={"request_id": contract.request_id}
                )
                continue
            # European contracts use the analytic engine rather than an American tree.
            if contract.exercise_type in {"EUROPEAN", "E"}:
                try:
                    valuation = solve_european_iv_and_greeks(
                        put_call=contract.put_call,
                        spot=contract.spot,
                        strike=contract.strike,
                        valuation_at=valuation_at,
                        expiration_at=contract.expiration_at,
                        rate=float(rate),
                        dividend_yield=float(dividend),
                        target_price=target_price,
                        minimum_iv=float(
                            self.context.get("risk.pricing.european.minimum_iv", 0.0001)
                        ),
                        maximum_iv=float(
                            self.context.get("risk.pricing.european.maximum_iv", 5.0)
                        ),
                    )
                    status = "usable_with_warnings" if warnings else "ok"
                    item = IvGreeksItem(
                        request_id=contract.request_id,
                        canonical_occ_symbol=contract.canonical_occ_symbol,
                        status=status,
                        reason="complete",
                        input_basis=input_basis,
                        target_price=target_price,
                        implied_volatility=valuation.implied_volatility,
                        model_price=valuation.price,
                        price_residual=valuation.price - target_price,
                        delta=valuation.delta,
                        gamma=valuation.gamma,
                        theta_per_day=valuation.theta_per_day,
                        vega_per_1pct=valuation.vega_per_1pct,
                        rho_per_1pct=valuation.rho_per_1pct,
                        spot=contract.spot,
                        tenor_years=tenor_seconds / (365.25 * 86400),
                        risk_free_rate_continuous=float(rate),
                        risk_free_rate_source=rate_source,
                        dividend_yield_continuous=float(dividend),
                        dividend_yield_source=dividend_source,
                        method="quantlib_analytic_european",
                        method_selection_reason="european_fallback_quantlib_analytic",
                        algorithm_version=valuation.algorithm_version,
                        solver_parameters={"engine": "AnalyticEuropeanEngine"},
                        requested_dtype="float64",
                        actual_dtype="float64",
                        quote_time=contract.quote_time,
                        calculation_time=calculation_time,
                        quality_state=(
                            "usable_with_warnings" if warnings else "usable"
                        ),
                        warnings=warnings,
                    )
                except EuropeanPricingUnavailable:
                    item = self._failure(
                        contract,
                        requested_dtype,
                        calculation_time,
                        reason="quantlib_european_pricing_failed",
                        stage="quantlib_analytic_european",
                        basis=input_basis,
                        target=target_price,
                        method="quantlib_analytic_european",
                    )
                output[index] = item
                self._cache_put(cache_key, item)
                continue
            pricing = PricingRequest(
                option_symbol=contract.canonical_occ_symbol,
                snapshot_id=request.contracts[index].request_id,
                put_call=contract.put_call,
                exercise_type="A" if contract.exercise_type in {"AMERICAN", "A"} else "E",
                spot=contract.spot,
                strike=contract.strike,
                tenor_seconds=tenor_seconds,
                rate=float(rate),
                dividend_yield=float(dividend),
                bid=contract.bid,
                ask=contract.ask,
                mark=contract.mark,
                source_iv=contract.source_iv,
                source_collected_at=contract.quote_time,
                expiration_at=contract.expiration_at,
                requested_dtype=requested_dtype,
            )
            # Provider IV is a solver hint; the selected market price remains the target.
            hints = (
                (IvHint(contract.source_iv, "provider_quote"),)
                if contract.source_iv is not None
                else ()
            )
            jobs.append(
                SolverBatchJob(
                    job_id=str(uuid.uuid4()),
                    owner_run_id=contract.request_id,
                    owner_snapshot_id=contract.request_id,
                    output_index=index,
                    operation=SolverOperation.SOLVE_IV_AND_GREEKS,
                    request=pricing,
                    method=RepricingMethod.TIAN,
                    method_selection_reason="risk_api_configured_tian",
                    target_price=target_price,
                    iv_hints=hints,
                    requested_dtype=requested_dtype,
                )
            )
            job_metadata.append(
                (index, contract, input_basis, target_price, warnings, cache_key)
            )
        if jobs:
            results, adaptive_diagnostics = self._adaptive_solve(jobs)
            for metadata, result, job, adaptive in zip(
                job_metadata, results, jobs, adaptive_diagnostics
            ):
                index, contract, input_basis, target_price, warnings, cache_key = metadata
                converged = bool(adaptive.get("converged"))
                status = "ok" if result.status == "complete" and converged and not warnings else (
                    "usable_with_warnings" if result.status == "complete" and converged else "unable_to_evaluate"
                )
                quality = "usable" if status == "ok" else (
                    "usable_with_warnings" if status == "usable_with_warnings" else "insufficient_data"
                )
                solver_parameters = dict(result.solver_parameters or {})
                solver_parameters["adaptive_convergence"] = adaptive
                configuration_hash = hashlib.sha256(
                    json.dumps(solver_parameters, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
                item = IvGreeksItem(
                    request_id=contract.request_id,
                    canonical_occ_symbol=contract.canonical_occ_symbol,
                    status=status,
                    reason=(
                        result.reason if converged else "adaptive_pricing_nonconvergent"
                    ),
                    failure_stage=(
                        result.failure_stage or None
                        if converged
                        else "adaptive_convergence"
                    ),
                    input_basis=input_basis,
                    target_price=target_price,
                    implied_volatility=result.implied_volatility,
                    model_price=result.model_price,
                    price_residual=result.price_residual,
                    iv_iterations=result.iv_iterations,
                    delta=result.delta,
                    gamma=result.gamma,
                    theta_per_day=result.theta_per_day,
                    vega_per_1pct=result.vega_per_1pct,
                    rho_per_1pct=result.rho_per_1pct,
                    spot=contract.spot,
                    tenor_years=(_aware_utc(contract.expiration_at) - valuation_at).total_seconds() / (365.25 * 86400),
                    risk_free_rate_continuous=job.request.rate,
                    risk_free_rate_source="request" if contract.risk_free_rate is not None else "configured_flat_rate",
                    dividend_yield_continuous=job.request.dividend_yield,
                    dividend_yield_source="request" if contract.dividend_yield is not None else "configured_default",
                    algorithm_version=result.algorithm_version,
                    solver_parameters=solver_parameters,
                    solver_configuration_hash=configuration_hash,
                    requested_dtype=requested_dtype,
                    actual_dtype=result.actual_dtype,
                    precision_escalated=result.precision_escalated,
                    quote_time=contract.quote_time,
                    calculation_time=calculation_time,
                    quality_state=quality,
                    warnings=warnings + (
                        []
                        if result.status == "complete" and converged
                        else [
                            result.reason
                            if result.status != "complete"
                            else "adaptive_pricing_nonconvergent"
                        ]
                    ),
                )
                output[index] = item
                self._cache_put(cache_key, item)
        assert all(item is not None for item in output)
        return IvGreeksResponse(
            as_of=calculation_time,
            data=[item for item in output if item is not None],
        )

    def _adaptive_solve(
        self, jobs: list[SolverBatchJob]
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        if self._registry_was_injected:
            results = self._get_registry().solve_iv_and_greeks_many(jobs)
            return results, [
                {
                    "converged": result.status == "complete",
                    "steps_attempted": [
                        int((result.solver_parameters or {}).get("steps", 0))
                    ],
                    "external_registry": True,
                }
                for result in results
            ]
        base = int(self.context.get("risk.pricing.adaptive_base_steps", 256))
        maximum = int(self.context.get("risk.pricing.adaptive_max_steps", 2048))
        steps = base
        previous = self._registry_for_steps(steps).solve_iv_and_greeks_many(jobs)
        attempts = [[steps] for _ in jobs]
        diagnostics = [
            {"converged": False, "steps_attempted": attempts[index]}
            for index in range(len(jobs))
        ]
        # Double the tree depth until price and Greek changes meet convergence limits.
        while steps < maximum:
            steps = min(maximum, steps * 2)
            current = self._registry_for_steps(steps).solve_iv_and_greeks_many(jobs)
            complete = True
            for index, (left, right) in enumerate(zip(previous, current)):
                attempts[index].append(steps)
                errors = self._convergence_errors(left, right)
                converged = right.status == "complete" and not errors
                diagnostics[index] = {
                    "converged": converged,
                    "steps_attempted": list(attempts[index]),
                    "selected_steps": steps,
                    "errors": errors,
                }
                complete = complete and converged
            previous = current
            if complete:
                break
        return previous, diagnostics

    def _registry_for_steps(self, steps: int) -> SolverRegistry:
        registry = self._adaptive_registries.get(steps)
        if registry is None:
            configured = dict(self.context)
            configured["options_calcs.greeks.tian.steps"] = steps
            registry = SolverRegistry(configured)
            self._adaptive_registries[steps] = registry
        return registry

    def _convergence_errors(self, left: Any, right: Any) -> dict[str, float]:
        if left.status != "complete" or right.status != "complete":
            return {"solver_status": -1.0}
        errors: dict[str, float] = {}
        left_price, right_price = _finite(left.model_price), _finite(right.model_price)
        if left_price is None or right_price is None:
            errors["price"] = -1.0
        else:
            error = abs(right_price - left_price)
            tolerance = max(
                float(self.context.get("risk.pricing.price_absolute_tolerance", 0.01)),
                float(self.context.get("risk.pricing.price_relative_tolerance", 0.005))
                * max(abs(right_price), abs(left_price)),
            )
            if error > tolerance:
                errors["price"] = error
        delta_error = _absolute_error(left.delta, right.delta)
        if delta_error is None or delta_error > float(
            self.context.get("risk.pricing.delta_absolute_tolerance", 0.01)
        ):
            errors["delta"] = delta_error if delta_error is not None else -1.0
        floor = float(self.context.get("risk.pricing.greek_absolute_floor", 1e-6))
        relative_tolerance = float(
            self.context.get("risk.pricing.greek_relative_tolerance", 0.05)
        )
        for name in ("gamma", "vega_per_1pct"):
            first, second = _finite(getattr(left, name)), _finite(getattr(right, name))
            if first is None or second is None:
                errors[name] = -1.0
                continue
            scale = max(abs(first), abs(second))
            if scale > floor and abs(second - first) / scale > relative_tolerance:
                errors[name] = abs(second - first) / scale
        return errors

    def reprice(self, request: TianRepriceRequest) -> TianRepriceResponse:
        """Reprice American options at a future valuation state using Tian.

        A current market mark is inverted when the caller does not already
        have an IV. That IV is then held constant apart from the explicit
        scenario shift, so advancing the valuation date actually decays the
        option instead of re-solving a new IV to the old market price.
        """
        valuation_at = _aware_utc(request.valuation_at)
        calculation_time = _aware_utc(self._now())
        requested_dtype = str(self.context.get("risk.options.requested_dtype", "float64")).lower()
        if requested_dtype not in {"float32", "float64"}:
            requested_dtype = "float64"
        output: list[TianRepriceItem | None] = [None] * len(request.contracts)
        price_indices: list[int] = []
        pricing_requests: list[PricingRequest] = []
        volatilities: list[float] = []
        base_volatilities: list[float] = []

        for index, contract in enumerate(request.contracts):
            expiration_at = _aware_utc(contract.expiration_at)
            if valuation_at >= expiration_at:
                output[index] = TianRepriceItem(
                    request_id=contract.request_id,
                    canonical_occ_symbol=contract.canonical_occ_symbol,
                    status="unable_to_evaluate",
                    reason="valuation_at_or_after_expiration",
                )
                continue
            rate = contract.risk_free_rate
            if rate is None:
                configured = self.context.get("risk.rates.configured_flat_continuous_rate")
                rate = float(configured) if configured not in (None, "") else None
            if rate is None or not math.isfinite(rate):
                output[index] = TianRepriceItem(
                    request_id=contract.request_id,
                    canonical_occ_symbol=contract.canonical_occ_symbol,
                    status="unable_to_evaluate",
                    reason="risk_free_rate_unavailable",
                )
                continue
            dividend = contract.dividend_yield
            if dividend is None:
                dividend = float(self.context.get("risk.dividends.default_continuous_yield", 0.0))
            base_iv = contract.implied_volatility
            # Infer a missing base IV from the source valuation before shifting scenarios.
            if base_iv is None:
                solved = self.solve(
                    IvGreeksRequest(
                        valuation_at=contract.source_valuation_at,
                        contracts=[
                            IvGreeksContract(
                                request_id=f"reprice-iv:{contract.request_id}",
                                canonical_occ_symbol=contract.canonical_occ_symbol,
                                underlying_symbol=contract.underlying_symbol,
                                put_call=contract.put_call,
                                exercise_type=contract.exercise_type,
                                strike=contract.strike,
                                expiration_at=contract.expiration_at,
                                spot=contract.source_spot,
                                bid=contract.source_bid,
                                ask=contract.source_ask,
                                mark=contract.source_mark,
                                risk_free_rate=rate,
                                dividend_yield=dividend,
                            )
                        ],
                    )
                ).data[0]
                base_iv = solved.implied_volatility
            if base_iv is None or not math.isfinite(base_iv) or base_iv <= 0:
                output[index] = TianRepriceItem(
                    request_id=contract.request_id,
                    canonical_occ_symbol=contract.canonical_occ_symbol,
                    status="unable_to_evaluate",
                    reason="implied_volatility_unavailable",
                )
                continue
            # Convert volatility percentage points to decimal IV before repricing.
            scenario_iv = max(0.0001, float(base_iv) + contract.iv_shift_points / 100.0)
            if contract.exercise_type in {"EUROPEAN", "E"}:
                try:
                    valuation = price_european(
                        put_call=contract.put_call,
                        spot=contract.scenario_spot,
                        strike=contract.strike,
                        valuation_at=valuation_at,
                        expiration_at=expiration_at,
                        rate=float(rate),
                        dividend_yield=float(dividend),
                        volatility=scenario_iv,
                    )
                    output[index] = TianRepriceItem(
                        request_id=contract.request_id,
                        canonical_occ_symbol=contract.canonical_occ_symbol,
                        status="ok",
                        reason="complete",
                        price=valuation.price,
                        implied_volatility=float(base_iv),
                        scenario_volatility=scenario_iv,
                        method="quantlib_analytic_european",
                        solver_diagnostics={
                            "engine": "AnalyticEuropeanEngine",
                            "algorithm_version": valuation.algorithm_version,
                        },
                    )
                except EuropeanPricingUnavailable:
                    output[index] = TianRepriceItem(
                        request_id=contract.request_id,
                        canonical_occ_symbol=contract.canonical_occ_symbol,
                        status="unable_to_evaluate",
                        reason="quantlib_european_pricing_failed",
                        method="quantlib_analytic_european",
                    )
                continue
            price_indices.append(index)
            base_volatilities.append(float(base_iv))
            volatilities.append(scenario_iv)
            pricing_requests.append(
                PricingRequest(
                    option_symbol=contract.canonical_occ_symbol,
                    snapshot_id=contract.request_id,
                    put_call=contract.put_call,
                    exercise_type="A" if contract.exercise_type in {"AMERICAN", "A"} else "E",
                    spot=contract.scenario_spot,
                    strike=contract.strike,
                    tenor_seconds=(expiration_at - valuation_at).total_seconds(),
                    rate=float(rate),
                    dividend_yield=float(dividend),
                    expiration_at=contract.expiration_at,
                    requested_dtype=requested_dtype,
                )
            )

        if pricing_requests:
            try:
                values, price_diagnostics = self._adaptive_price_many(
                    pricing_requests,
                    volatilities,
                    requested_dtype,
                )
                if len(values) != len(pricing_requests):
                    raise ValueError("Tian returned an incomplete price batch")
                for index, value, base_iv, scenario_iv, diagnostics in zip(
                    price_indices, values, base_volatilities, volatilities,
                    price_diagnostics,
                ):
                    finite_price = _finite(value)
                    contract = request.contracts[index]
                    converged = bool(diagnostics.get("converged"))
                    output[index] = TianRepriceItem(
                        request_id=contract.request_id,
                        canonical_occ_symbol=contract.canonical_occ_symbol,
                        status="ok" if finite_price is not None and converged else "unable_to_evaluate",
                        reason=(
                            "complete"
                            if finite_price is not None and converged
                            else "adaptive_pricing_nonconvergent"
                            if finite_price is not None
                            else "nonfinite_model_price"
                        ),
                        price=finite_price if converged else None,
                        implied_volatility=base_iv,
                        scenario_volatility=scenario_iv,
                        solver_diagnostics=diagnostics,
                    )
            except (SolverError, ArithmeticError, FloatingPointError, ValueError):
                for index in price_indices:
                    contract = request.contracts[index]
                    output[index] = TianRepriceItem(
                        request_id=contract.request_id,
                        canonical_occ_symbol=contract.canonical_occ_symbol,
                        status="unable_to_evaluate",
                        reason="tian_price_batch_failed",
                    )

        assert all(item is not None for item in output)
        return TianRepriceResponse(
            as_of=calculation_time,
            valuation_at=valuation_at,
            data=[item for item in output if item is not None],
        )

    def _adaptive_price_many(
        self,
        requests: list[PricingRequest],
        volatilities: list[float],
        requested_dtype: str,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        dtypes = [requested_dtype] * len(requests)
        if self._registry_was_injected:
            values = self._get_registry().price_many(
                requests, volatilities, RepricingMethod.TIAN, dtypes
            )
            return values, [
                {
                    "converged": _finite(value) is not None,
                    "steps_attempted": [],
                    "external_registry": True,
                }
                for value in values
            ]
        base = int(self.context.get("risk.pricing.adaptive_base_steps", 256))
        maximum = int(self.context.get("risk.pricing.adaptive_max_steps", 2048))
        steps = base
        previous = self._registry_for_steps(steps).price_many(
            requests, volatilities, RepricingMethod.TIAN, dtypes
        )
        attempts = [[steps] for _ in requests]
        diagnostics = [
            {"converged": False, "steps_attempted": list(item)}
            for item in attempts
        ]
        while steps < maximum:
            steps = min(maximum, steps * 2)
            current = self._registry_for_steps(steps).price_many(
                requests, volatilities, RepricingMethod.TIAN, dtypes
            )
            complete = True
            for index, (pricing, left, right) in enumerate(
                zip(requests, previous, current)
            ):
                attempts[index].append(steps)
                first, second = _finite(left), _finite(right)
                tolerance = (
                    max(
                        float(self.context.get("risk.pricing.price_absolute_tolerance", 0.01)),
                        float(self.context.get("risk.pricing.price_relative_tolerance", 0.005))
                        * max(abs(first or 0.0), abs(second or 0.0)),
                    )
                )
                error = abs(second - first) if first is not None and second is not None else None
                converged = (
                    error is not None
                    and error <= tolerance
                    and _price_is_within_bounds(pricing, second)
                )
                diagnostics[index] = {
                    "converged": converged,
                    "steps_attempted": list(attempts[index]),
                    "selected_steps": steps,
                    "price_absolute_error": error,
                    "price_tolerance": tolerance,
                    "no_arbitrage_bounds_satisfied": (
                        _price_is_within_bounds(pricing, second)
                        if second is not None
                        else False
                    ),
                }
                complete = complete and converged
            previous = current
            if complete:
                break
        return previous, diagnostics

    def _target_basis(
        self, contract: IvGreeksContract, valuation_at: datetime
    ) -> tuple[str, float, list[str]] | None:
        warnings: list[str] = []
        bid, ask = _finite(contract.bid), _finite(contract.ask)
        # Prefer a valid market midpoint, with explicit wide-spread diagnostics.
        if bid is not None and ask is not None and 0 <= bid <= ask and ask > 0:
            midpoint = (bid + ask) / 2
            maximum = float(self.context.get("risk.options.maximum_spread_fraction_of_mid", 0.50))
            if midpoint > 0 and (ask - bid) / midpoint > maximum:
                if not _as_bool(self.context.get("risk.options.allow_wide_spread_inversion", True)):
                    return None
                warnings.append("wide_bid_ask_spread")
            return "market_mid", midpoint, warnings
        mark = _finite(contract.mark)
        if mark is not None and mark > 0:
            return "provider_mark", mark, warnings
        last = _finite(contract.last)
        # Use last-trade prices only within the configured recency window.
        if last is not None and last > 0 and contract.last_trade_at is not None:
            age = (valuation_at - _aware_utc(contract.last_trade_at)).total_seconds()
            if 0 <= age <= float(self.context.get("risk.options.maximum_last_trade_age_seconds", 900)):
                return "last_trade", last, warnings
        return None

    def _failure(
        self,
        contract: IvGreeksContract,
        requested_dtype: str,
        calculation_time: datetime,
        *,
        reason: str,
        stage: str,
        basis: str | None = None,
        target: float | None = None,
        method: Literal["tian", "quantlib_analytic_european"] = "tian",
    ) -> IvGreeksItem:
        return IvGreeksItem(
            request_id=contract.request_id,
            canonical_occ_symbol=contract.canonical_occ_symbol,
            status="unable_to_evaluate",
            reason=reason,
            failure_stage=stage,
            input_basis=basis,
            target_price=target,
            spot=contract.spot,
            requested_dtype=requested_dtype,
            calculation_time=calculation_time,
            quality_state="insufficient_data",
            warnings=[reason],
            method=method,
        )

    def _get_registry(self) -> SolverRegistry:
        if self._registry is None:
            with self._registry_lock:
                if self._registry is None:
                    self._registry = SolverRegistry(dict(self.context))
        return self._registry

    def _cache_key(
        self,
        contract: IvGreeksContract,
        valuation_at: datetime,
        target_price: float,
        rate: float,
        dividend: float,
        requested_dtype: str,
    ) -> str:
        bucket = int(valuation_at.timestamp() // 60)
        raw = (
            contract.canonical_occ_symbol,
            contract.exercise_type,
            bucket,
            target_price,
            contract.spot,
            _aware_utc(contract.expiration_at).isoformat(),
            rate,
            dividend,
            requested_dtype,
            "risk_tian_quantlib_iv_greeks_v3",
        )
        return hashlib.sha256(repr(raw).encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> IvGreeksItem | None:
        now = self._clock()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expires, value = entry
            if expires <= now:
                del self._cache[key]
                return None
            return value.model_copy(deep=True)

    def _cache_put(self, key: str, value: IvGreeksItem) -> None:
        ttl = float(self.context.get("risk.options.cache_ttl_seconds", 60))
        with self._cache_lock:
            self._cache[key] = (self._clock() + ttl, value.model_copy(deep=True))
            if len(self._cache) > 2048:
                oldest = min(self._cache, key=lambda item: self._cache[item][0])
                del self._cache[oldest]


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone aware")
    return value.astimezone(timezone.utc)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _absolute_error(left: Any, right: Any) -> float | None:
    first, second = _finite(left), _finite(right)
    return abs(second - first) if first is not None and second is not None else None


def _price_is_within_bounds(request: PricingRequest, value: float) -> bool:
    if not math.isfinite(value):
        return False
    if request.put_call == "CALL":
        lower = max(0.0, float(request.spot) - float(request.strike))
        upper = float(request.spot)
    else:
        lower = max(0.0, float(request.strike) - float(request.spot))
        upper = float(request.strike)
    tolerance = 1e-8 * max(1.0, upper)
    return lower - tolerance <= value <= upper + tolerance


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "IvGreeksContract",
    "IvGreeksItem",
    "IvGreeksRequest",
    "IvGreeksResponse",
    "TianRepriceContract",
    "TianRepriceItem",
    "TianRepriceRequest",
    "TianRepriceResponse",
    "TianIvGreeksService",
]
