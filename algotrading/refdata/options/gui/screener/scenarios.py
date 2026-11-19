"""Request-scoped deterministic paths, shared price grids, exits, and hedges."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from algotrading.refdata.options.Pricing import configured_methods, select_method

from .. import SCREENER_MODEL_VERSION
from ..cache.model import canonical_json
from ..cache.pricing import CompiledSurfaceIvEvaluator, price_iv_only_jobs
from ..errors import PricingUnavailableError, ScreenerTimeoutError
from ..pricing_pool import (
    PricingPool,
    PricingPoolBusy,
    PricingPoolFailure,
    PricingPoolTimeout,
)
from .config import ScreenerConfig
from .models import CandidateDraft, ScreeningRequest, Universe


@dataclass(frozen=True)
class ScenarioMetrics:
    probability_of_profit: float
    target_probability: float
    expected_pnl: float
    pnl_percentiles: dict[str, float]
    modeled_pnl_range: tuple[float, float]
    expected_holding_sessions: float | None
    holding_basis: str
    robustness: float
    warnings: tuple[str, ...]


@dataclass
class ScenarioPerformance:
    path_sets_generated: int = 0
    shock_matrix_generations: int = 0
    unique_contracts_priced: int = 0
    unique_price_grids: int = 0
    price_grid_cache_hits: int = 0
    price_grid_cache_misses: int = 0
    numerical_pricing_jobs: int = 0
    pricing_batches_submitted: int = 0
    pricing_workers: int = 0
    scenario_pricing_method: str = "tian"
    surface_evaluators_compiled: int = 0
    tian_scalar_fallback_jobs: int = 0


class PricingResources:
    """Screener-only routing and ownership of one persistent pricing pool."""

    def __init__(
        self,
        context: Mapping[str, Any],
        registry: Any | None = None,
        *,
        config: ScreenerConfig | None = None,
        pool: PricingPool | None = None,
    ) -> None:
        self.context = context
        self.config = config or ScreenerConfig.from_context(context)
        self.registry = registry
        self.routing = configured_methods(context)
        self.pool = pool
        if self.registry is None and self.pool is None:
            self.pool = PricingPool(
                context,
                workers=self.config.resolved_pricing_workers,
                batch_size=self.config.pricing_batch_size,
                max_pending_batches=(
                    self.config.resolved_max_pending_pricing_batches
                ),
                worker_threads=self.config.pricing_worker_threads,
                clock=time.perf_counter,
            )
        numerical_context: dict[str, Any] = {}
        for key, value in context.items():
            if not str(key).startswith("options_calcs.greeks."):
                continue
            try:
                canonical_json(value)
            except (TypeError, ValueError, OverflowError):
                continue
            numerical_context[str(key)] = value
        dependency_versions: dict[str, str] = {"numpy": np.__version__}
        for package in ("scipy", "torch"):
            try:
                dependency_versions[package] = version(package)
            except PackageNotFoundError:
                dependency_versions[package] = "unavailable"
        self.configuration_signature = hashlib.sha256(
            canonical_json(
                {
                    "screener": asdict(self.config),
                    "numerical": numerical_context,
                    "dependencies": dependency_versions,
                }
            )
        ).hexdigest()[:24]

    def price_jobs(
        self,
        jobs: Sequence[Mapping[str, Any]],
        *,
        deadline: float | None = None,
    ) -> np.ndarray:
        if self.registry is not None:
            return price_iv_only_jobs(
                jobs,
                context=self.context,
                registry=self.registry,
            )
        if self.pool is None:
            raise PricingPoolFailure("Screener pricing pool is unavailable")
        return self.pool.price_jobs(
            jobs,
            deadline=deadline if deadline is not None else math.inf,
        )

    def pricing_method(self, exercise_type: str, tenor_seconds: float) -> str:
        if tenor_seconds <= 0:
            raise PricingUnavailableError("Scenario pricing tenor must be positive")
        if self.config.pricing_method != "configured":
            return self.config.pricing_method
        return select_method(
            exercise_type,
            tenor_seconds,
            routing=self.routing,
        ).method.value

    def health(self) -> dict[str, Any]:
        pool_health = self.pool.health() if self.pool is not None else {
            "workers": 0,
            "batch_size": self.config.pricing_batch_size,
            "maximum_pending_batches": 0,
            "pending_batches": 0,
            "started": False,
        }
        return {
            "scenario_pricing_method": self.config.pricing_method,
            "configuration_signature": self.configuration_signature,
            **pool_health,
        }

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
        close = getattr(self.registry, "close", None)
        if callable(close):
            close()


def deterministic_seed(
    snapshot_id: str,
    request: ScreeningRequest | Mapping[str, Any],
    model_version: str = SCREENER_MODEL_VERSION,
) -> int:
    request_value = (
        request.model_dump(mode="json", exclude_none=True)
        if isinstance(request, ScreeningRequest)
        else request
    )
    digest = hashlib.sha256(
        canonical_json(
            {
                "snapshot_id": snapshot_id,
                "request": request_value,
                "model_version": model_version,
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def geometric_paths(
    *,
    spot: float,
    sessions: int,
    path_count: int,
    annualized_drift: float,
    annualized_volatility: float,
    seed: int,
) -> np.ndarray:
    if sessions < 1 or path_count < 1:
        raise ValueError("sessions and path_count must be positive")
    shocks = np.random.default_rng(seed).standard_normal((path_count, sessions))
    return _paths_from_shocks(
        spot=spot,
        annualized_drift=annualized_drift,
        annualized_volatility=annualized_volatility,
        shocks=shocks,
    )


def _paths_from_shocks(
    *,
    spot: float,
    annualized_drift: float,
    annualized_volatility: float,
    shocks: np.ndarray,
) -> np.ndarray:
    dt = 1.0 / 252.0
    increments = (
        (annualized_drift - 0.5 * annualized_volatility**2) * dt
        + annualized_volatility * math.sqrt(dt) * shocks
    )
    return spot * np.exp(np.cumsum(increments, axis=1))


def residual_decay(
    residual: float,
    sessions: float | np.ndarray,
    half_life_sessions: float,
) -> float | np.ndarray:
    return residual * np.power(2.0, -np.asarray(sessions) / half_life_sessions)


def interpolate_price_grid(
    spot_grid: np.ndarray,
    price_grid: np.ndarray,
    target_spots: np.ndarray,
) -> np.ndarray:
    """Interpolate only inside a finite, strictly increasing bounded grid."""

    spots = np.asarray(spot_grid, dtype=float)
    prices = np.asarray(price_grid, dtype=float)
    targets = np.asarray(target_spots, dtype=float)
    if (
        spots.ndim != 1
        or prices.ndim != 1
        or len(spots) < 2
        or len(spots) != len(prices)
        or not np.all(np.isfinite(spots))
        or not np.all(np.isfinite(prices))
        or np.any(np.diff(spots) <= 0)
    ):
        raise ValueError("Option price interpolation grid is invalid")
    if not np.all(np.isfinite(targets)) or np.any(targets < spots[0]) or np.any(
        targets > spots[-1]
    ):
        raise ValueError("Option price interpolation would extrapolate")
    return np.interp(targets, spots, prices)


def static_hedge_shares(candidate: CandidateDraft) -> int:
    option_delta_in_shares = sum(
        leg.sign
        * leg.quantity
        * float(leg.contract.delta or 0.0)
        * leg.contract.multiplier
        for leg in candidate.legs
    )
    return round(-option_delta_in_shares)


def distribution_metrics(
    pnl: np.ndarray,
    target_hit: np.ndarray,
    exit_sessions: np.ndarray,
    minimum_hits: int,
) -> tuple[float, float, float, dict[str, float], float | None, str]:
    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1 or not len(pnl) or not np.all(np.isfinite(pnl)):
        raise ValueError("P&L distribution must be finite and nonempty")
    hits = np.asarray(target_hit, dtype=bool)
    sessions = np.asarray(exit_sessions, dtype=float)
    holding = (
        float(np.median(sessions[hits]))
        if int(hits.sum()) >= minimum_hits
        else None
    )
    basis = "target_hit_median" if holding is not None else "planned_maximum"
    quantiles = np.quantile(pnl, [0.05, 0.25, 0.50, 0.75, 0.95])
    return (
        float(np.mean(pnl > 0)),
        float(np.mean(hits)),
        float(np.mean(pnl)),
        {
            "p05": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p75": float(quantiles[3]),
            "p95": float(quantiles[4]),
        },
        holding,
        basis,
    )


def _structure_key(candidate: CandidateDraft) -> tuple[tuple[int, str, int], ...]:
    return tuple(
        (leg.contract.contract_id, leg.side, leg.quantity)
        for leg in candidate.legs
    )


class ScenarioWorkspace:
    """All deterministic scenario work shared by one bounded screen request."""

    def __init__(
        self,
        *,
        engine: "ScenarioEngine",
        candidate_plans: Sequence[tuple[CandidateDraft, int]],
        universe: Universe,
        request: ScreeningRequest,
        deadline: float,
    ) -> None:
        self.engine = engine
        self.config = engine.config
        self.pricing = engine.pricing
        self.universe = universe
        self.request = request
        self.deadline = deadline
        self.performance = ScenarioPerformance(
            pricing_workers=self.config.resolved_pricing_workers,
            scenario_pricing_method=self.config.pricing_method,
        )
        self.path_count = min(
            request.probability.simulation_paths or self.config.simulation_paths,
            self.config.maximum_simulation_paths,
        )
        self.maximum_holding_sessions = max(
            (
                min(int(planned), self.config.maximum_holding_sessions)
                for _, planned in candidate_plans
            ),
            default=1,
        )
        self._candidate_drift_keys: dict[
            tuple[tuple[int, str, int], ...], str
        ] = {}
        self._drifts: dict[str, float] = {}
        self.paths_by_drift_key: dict[str, np.ndarray] = {}
        self.spot_grids: dict[tuple[str, int], np.ndarray] = {}
        self.compiled_surfaces: dict[
            tuple[object, str], CompiledSurfaceIvEvaluator
        ] = dict(universe.compiled_surface_evaluators)
        self.price_grids: dict[tuple[int, str, int], np.ndarray] = {}
        self._contracts: dict[int, Any] = {}
        self._prepare(candidate_plans)

    def _check_deadline(self) -> None:
        if self.engine.clock() > self.deadline:
            raise ScreenerTimeoutError(
                "The bounded spread screen exceeded its deadline",
                details={
                    **asdict(self.performance),
                    "unique_contracts": len(self._contracts),
                    "unique_price_grids": len(self.price_grids),
                    "queued_or_running_batches": (
                        self.pricing.pool.pending_batches
                        if getattr(self.pricing, "pool", None) is not None
                        else 0
                    ),
                },
            )

    def _prepare(
        self, candidate_plans: Sequence[tuple[CandidateDraft, int]]
    ) -> None:
        self._check_deadline()
        seed = deterministic_seed(
            self.universe.snapshot.snapshot_id,
            self.request,
        )
        shocks = np.random.default_rng(seed).standard_normal(
            (self.path_count, self.maximum_holding_sessions)
        )
        self.performance.shock_matrix_generations = 1

        required_grids: set[tuple[int, str, int]] = set()
        for candidate_index, (candidate, planned) in enumerate(candidate_plans):
            if candidate_index % 32 == 0:
                self._check_deadline()
            drift = self.engine._drift(candidate, self.universe, self.request)
            drift_key = float(drift).hex()
            self._drifts[drift_key] = drift
            self._candidate_drift_keys[_structure_key(candidate)] = drift_key
            holding = min(int(planned), self.config.maximum_holding_sessions)
            for leg in candidate.legs:
                self._contracts[leg.contract.contract_id] = leg.contract
                for session in range(1, holding + 1):
                    required_grids.add(
                        (leg.contract.contract_id, drift_key, session)
                    )

        for drift_key in sorted(self._drifts):
            self._check_deadline()
            self.paths_by_drift_key[drift_key] = _paths_from_shocks(
                spot=self.universe.snapshot.underlying_mark,
                annualized_drift=self._drifts[drift_key],
                annualized_volatility=self.universe.atm_volatility,
                shocks=shocks,
            )
        self.performance.path_sets_generated = len(self.paths_by_drift_key)

        for _, drift_key, session in sorted(
            required_grids, key=lambda value: (value[1], value[2], value[0])
        ):
            key = (drift_key, session)
            if key in self.spot_grids:
                continue
            observed = self.paths_by_drift_key[drift_key][:, session - 1]
            lower = max(1.0e-6, float(np.min(observed)) * 0.995)
            upper = float(np.max(observed)) * 1.005
            if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
                raise PricingUnavailableError("Scenario spot grid is invalid")
            self.spot_grids[key] = np.linspace(
                lower,
                upper,
                self.config.spot_grid_points,
                dtype=np.float64,
            )

        for contract in self._contracts.values():
            fit_key = (contract.expiration_at, self.request.price_basis)
            if fit_key in self.compiled_surfaces:
                continue
            fit = self.universe.surface_fits.get(fit_key)
            expiration = self.universe.expiration_inputs.get(contract.expiration_at)
            if fit is not None and expiration is not None:
                self.compiled_surfaces[fit_key] = (
                    CompiledSurfaceIvEvaluator.from_fit(
                        fit, expiration.time_to_expiry_seconds
                    )
                )
        self.performance.surface_evaluators_compiled = len(
            self.compiled_surfaces
        )
        self._build_price_grids(required_grids)
        self.performance.unique_contracts_priced = len(self._contracts)
        self.performance.unique_price_grids = len(self.price_grids)

    def _build_price_grids(
        self, required_grids: set[tuple[int, str, int]]
    ) -> None:
        jobs: list[Mapping[str, Any]] = []
        placements: list[tuple[tuple[int, str, int], int]] = []
        # Retain only a bounded submission window while allowing the persistent
        # pool to keep several workers busy. The pool still enforces the outer
        # batch bound for every individual submitted future.
        submission_window_jobs = min(
            100_000,
            self.config.pricing_batch_size
            * min(
                self.config.resolved_max_pending_pricing_batches,
                max(1, self.config.resolved_pricing_workers * 2),
            ),
        )
        submission_window_jobs = max(
            self.config.pricing_batch_size, submission_window_jobs
        )

        def flush() -> None:
            if not jobs:
                return
            self._check_deadline()
            try:
                values = np.asarray(
                    self.pricing.price_jobs(jobs, deadline=self.deadline),
                    dtype=np.float64,
                ).reshape(-1)
            except PricingPoolTimeout as exc:
                raise ScreenerTimeoutError(
                    "Scenario pricing exceeded the screen deadline",
                    details={
                        **asdict(self.performance),
                        "unique_contracts": len(self._contracts),
                        "unique_price_grids": len(self.price_grids),
                        "queued_or_running_batches": (
                            self.pricing.pool.pending_batches
                            if getattr(self.pricing, "pool", None) is not None
                            else 0
                        ),
                    },
                ) from exc
            except (PricingPoolBusy, PricingPoolFailure) as exc:
                raise PricingUnavailableError(
                    "Scenario pricing capacity is unavailable"
                ) from exc
            except PricingUnavailableError:
                raise
            except Exception as exc:
                raise PricingUnavailableError("Canonical option pricing failed") from exc
            if len(values) != len(placements) or not np.all(np.isfinite(values)):
                raise PricingUnavailableError(
                    "Canonical option pricing returned an invalid aligned batch"
                )
            for value, (grid_key, grid_index) in zip(values, placements):
                self.price_grids[grid_key][grid_index] = value
            self.performance.numerical_pricing_jobs += len(jobs)
            self.performance.pricing_batches_submitted += math.ceil(
                len(jobs) / self.config.pricing_batch_size
            )
            jobs.clear()
            placements.clear()

        for grid_index, key in enumerate(sorted(required_grids)):
            if grid_index % 32 == 0:
                self._check_deadline()
            contract_id, drift_key, session = key
            contract = self._contracts[contract_id]
            expiration = self.universe.expiration_inputs.get(contract.expiration_at)
            if expiration is None:
                raise PricingUnavailableError(
                    "State-authorized expiration pricing inputs are unavailable",
                    details={"option_symbol": contract.option_symbol},
                )
            remaining = max(
                1.0,
                expiration.time_to_expiry_seconds
                - session * 86400.0 * 365.25 / 252.0,
            )
            grid = self.spot_grids[(drift_key, session)]
            fit_key = (contract.expiration_at, self.request.price_basis)
            fit = self.universe.surface_fits.get(fit_key)
            evaluator = self.compiled_surfaces.get(fit_key)
            if fit is not None and evaluator is not None:
                carry = math.exp(
                    (expiration.risk_free_rate - expiration.dividend_yield)
                    * remaining
                    / (365.25 * 86400.0)
                )
                moneyness = np.log(contract.strike / (grid * carry))
                bounded = np.clip(
                    moneyness,
                    fit.minimum_log_forward_moneyness,
                    fit.maximum_log_forward_moneyness,
                )
                fitted = evaluator.evaluate(bounded)
                residual = float(contract.iv_residual or 0.0)
                volatility = fitted + residual_decay(
                    residual,
                    session,
                    self.request.probability.iv_residual_half_life_sessions
                    or self.config.iv_residual_half_life_sessions,
                )
            else:
                volatility = np.full(
                    len(grid), float(contract.effective_iv), dtype=np.float64
                )
            volatility = np.clip(
                volatility, self.config.minimum_iv, self.config.maximum_iv
            )
            if not np.all(np.isfinite(volatility)):
                raise PricingUnavailableError(
                    "Scenario volatility grid is invalid",
                    details={"option_symbol": contract.option_symbol},
                )
            method = self.pricing.pricing_method(contract.exercise_type, remaining)
            vectorized_setting = self.engine.context.get(
                "options_calcs.greeks.tian.vectorized_enabled", True
            )
            vectorized_enabled = (
                str(vectorized_setting).strip().lower()
                not in {"0", "false", "no", "off", ""}
            )
            if method == "tian" and not vectorized_enabled:
                self.performance.tian_scalar_fallback_jobs += len(grid)
            self.price_grids[key] = np.full(
                len(grid), np.nan, dtype=np.float64
            )
            self.performance.price_grid_cache_misses += 1
            for spot_index, spot in enumerate(grid):
                jobs.append(
                    {
                        "pricing_method": method,
                        "spot": float(spot),
                        "strike": contract.strike,
                        "tenor_seconds": remaining,
                        "rate": expiration.risk_free_rate,
                        "dividend_yield": expiration.dividend_yield,
                        "volatility": float(volatility[spot_index]),
                        "put_call": contract.put_call,
                        "snapshot_id": self.universe.snapshot.snapshot_id,
                        "source_collected_at": (
                            self.universe.snapshot.source_collected_at
                        ),
                        "expiration_at": contract.expiration_at,
                    }
                )
                placements.append((key, spot_index))
                if len(jobs) >= submission_window_jobs:
                    flush()
        flush()
        for key, values in self.price_grids.items():
            if not np.all(np.isfinite(values)):
                raise PricingUnavailableError(
                    "Canonical option pricing returned an incomplete price grid",
                    details={"contract_id": key[0]},
                )

    def evaluate_variants(
        self,
        candidate: CandidateDraft,
        *,
        planned_holding_sessions: int,
        target_profit_per_share: float,
        hedge_shares_values: Sequence[int],
    ) -> list[tuple[int, ScenarioMetrics]]:
        self._check_deadline()
        holding = min(
            planned_holding_sessions, self.config.maximum_holding_sessions
        )
        drift_key = self._candidate_drift_keys[_structure_key(candidate)]
        paths = self.paths_by_drift_key[drift_key][:, :holding]
        base_pnl, outside_count, evaluated_count = self._base_option_pnl(
            candidate, drift_key, paths
        )
        output: list[tuple[int, ScenarioMetrics]] = []
        for hedge_shares in hedge_shares_values:
            self._check_deadline()
            pnl_by_horizon = base_pnl
            if hedge_shares:
                pnl_by_horizon = base_pnl + self.engine._hedge_pnl(
                    hedge_shares, self.universe, paths
                )
            target_multiplier = min(
                leg.contract.multiplier for leg in candidate.legs
            )
            reached = pnl_by_horizon >= target_profit_per_share * target_multiplier
            any_hit = np.any(reached, axis=1)
            first_hit = np.argmax(reached, axis=1)
            selected_index = np.where(any_hit, first_hit, holding - 1)
            selected_pnl = pnl_by_horizon[
                np.arange(self.path_count), selected_index
            ]
            exit_sessions = selected_index + 1
            (
                pop,
                target_probability,
                expected_pnl,
                percentiles,
                expected_holding,
                holding_basis,
            ) = distribution_metrics(
                selected_pnl,
                any_hit,
                exit_sessions,
                self.config.minimum_target_hits_for_estimate,
            )
            outside_fraction = (
                outside_count / evaluated_count if evaluated_count else 0.0
            )
            warnings: list[str] = []
            if outside_fraction > 0:
                warnings.append(
                    "SURFACE_BOUNDARY_TREATMENT: fitted moneyness was clamped "
                    "to the nearest covered boundary; no surface extrapolation "
                    "was used"
                )
            if outside_fraction > self.config.surface_boundary_probability_limit:
                warnings.append("MATERIAL_SURFACE_TAIL_PROBABILITY")
            if any(
                leg.contract.exercise_type == "AMERICAN" and leg.side == "short"
                for leg in candidate.legs
            ):
                warnings.append("EARLY_EXERCISE_OR_ASSIGNMENT_RISK")
            if expected_holding is None:
                warnings.append("TARGET_HIT_HOLDING_ESTIMATE_LOW_CONFIDENCE")
            output.append(
                (
                    hedge_shares,
                    ScenarioMetrics(
                        probability_of_profit=pop,
                        target_probability=target_probability,
                        expected_pnl=expected_pnl,
                        pnl_percentiles=percentiles,
                        modeled_pnl_range=(
                            float(np.min(selected_pnl)),
                            float(np.max(selected_pnl)),
                        ),
                        expected_holding_sessions=expected_holding,
                        holding_basis=holding_basis,
                        robustness=max(0.0, 1.0 - outside_fraction),
                        warnings=tuple(warnings),
                    ),
                )
            )
        return output

    def _base_option_pnl(
        self,
        candidate: CandidateDraft,
        drift_key: str,
        paths: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        exit_values = np.zeros_like(paths)
        opening_dollars = 0.0
        outside_count = 0
        evaluated_count = 0
        for leg in candidate.legs:
            contract = leg.contract
            if contract.bid is None or contract.ask is None:
                raise PricingUnavailableError(
                    "A scenario leg has no executable market"
                )
            opening_price = (
                contract.midpoint
                if self.request.entry_basis == "midpoint"
                else contract.ask if leg.side == "long" else contract.bid
            )
            opening_dollars += (
                leg.sign * leg.quantity * opening_price * contract.multiplier
            )
            values, outside, evaluated = self._leg_path_values(
                contract, drift_key, paths
            )
            outside_count += outside
            evaluated_count += evaluated
            exit_values += (
                leg.sign * leg.quantity * contract.multiplier * values
            )
        pnl = exit_values - opening_dollars
        pnl -= self.engine._closing_cost(candidate)
        return pnl, outside_count, evaluated_count

    def _leg_path_values(
        self,
        contract: Any,
        drift_key: str,
        paths: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        expiration = self.universe.expiration_inputs.get(contract.expiration_at)
        if expiration is None:
            raise PricingUnavailableError(
                "State-authorized expiration pricing inputs are unavailable",
                details={"option_symbol": contract.option_symbol},
            )
        fit = self.universe.surface_fits.get(
            (contract.expiration_at, self.request.price_basis)
        )
        result = np.empty_like(paths)
        outside_count = 0
        evaluated_count = 0
        for horizon_index in range(paths.shape[1]):
            self._check_deadline()
            session = horizon_index + 1
            key = (contract.contract_id, drift_key, session)
            grid_values = self.price_grids.get(key)
            if grid_values is None:
                raise PricingUnavailableError("Scenario price grid is unavailable")
            self.performance.price_grid_cache_hits += 1
            observed = paths[:, horizon_index]
            spot_grid = self.spot_grids[(drift_key, session)]
            result[:, horizon_index] = interpolate_price_grid(
                spot_grid, grid_values, observed
            )
            if fit is not None:
                remaining = max(
                    1.0,
                    expiration.time_to_expiry_seconds
                    - session * 86400.0 * 365.25 / 252.0,
                )
                carry = math.exp(
                    (expiration.risk_free_rate - expiration.dividend_yield)
                    * remaining
                    / (365.25 * 86400.0)
                )
                observed_moneyness = np.log(
                    contract.strike / (observed * carry)
                )
                outside_count += int(
                    np.count_nonzero(
                        (observed_moneyness < fit.minimum_log_forward_moneyness)
                        | (
                            observed_moneyness
                            > fit.maximum_log_forward_moneyness
                        )
                    )
                )
                evaluated_count += len(observed)
        return result, outside_count, evaluated_count


class ScenarioEngine:
    def __init__(
        self,
        context: Mapping[str, Any],
        config: ScreenerConfig,
        pricing: PricingResources,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.context = context
        self.config = config
        self.pricing = pricing
        self.clock = clock

    def prepare(
        self,
        candidate_plans: Sequence[tuple[CandidateDraft, int]],
        universe: Universe,
        request: ScreeningRequest,
        *,
        deadline: float,
    ) -> ScenarioWorkspace:
        return ScenarioWorkspace(
            engine=self,
            candidate_plans=candidate_plans,
            universe=universe,
            request=request,
            deadline=deadline,
        )

    def evaluate(
        self,
        candidate: CandidateDraft,
        universe: Universe,
        request: ScreeningRequest,
        *,
        planned_holding_sessions: int,
        target_profit_per_share: float,
        hedge_shares: int,
        deadline: float | None = None,
    ) -> ScenarioMetrics:
        workspace = self.prepare(
            [(candidate, planned_holding_sessions)],
            universe,
            request,
            deadline=(deadline if deadline is not None else math.inf),
        )
        return workspace.evaluate_variants(
            candidate,
            planned_holding_sessions=planned_holding_sessions,
            target_profit_per_share=target_profit_per_share,
            hedge_shares_values=[hedge_shares],
        )[0][1]

    def _drift(
        self,
        candidate: CandidateDraft,
        universe: Universe,
        request: ScreeningRequest,
    ) -> float:
        if request.probability.drift_mode == "zero":
            return 0.0
        if request.probability.drift_mode == "explicit":
            return float(request.probability.annualized_drift or 0.0)
        inputs = [
            universe.expiration_inputs[leg.contract.expiration_at]
            for leg in candidate.legs
            if leg.contract.expiration_at in universe.expiration_inputs
        ]
        if not inputs:
            return 0.0
        return float(
            np.mean(
                [item.risk_free_rate - item.dividend_yield for item in inputs]
            )
        )

    @staticmethod
    def _hedge_pnl(
        hedge_shares: int, universe: Universe, paths: np.ndarray
    ) -> np.ndarray:
        snapshot = universe.snapshot
        if hedge_shares > 0:
            entry = snapshot.underlying_ask or snapshot.underlying_mark
        else:
            entry = snapshot.underlying_bid or snapshot.underlying_mark
        half_spread = (
            (snapshot.underlying_ask - snapshot.underlying_bid) / 2.0
            if snapshot.underlying_ask is not None
            and snapshot.underlying_bid is not None
            else 0.0
        )
        return hedge_shares * (paths - entry) - abs(hedge_shares) * half_spread

    def _closing_cost(self, candidate: CandidateDraft) -> float:
        slippage = 0.0
        contracts = 0
        for leg in candidate.legs:
            width = float(leg.contract.ask or 0) - float(leg.contract.bid or 0)
            slippage += (
                max(0.0, width)
                * self.config.modeled_closing_slippage_fraction
                * leg.quantity
                * leg.contract.multiplier
            )
            contracts += leg.quantity
        return (
            slippage
            + 2.0 * contracts * self.config.commission_per_contract
        )


__all__ = [
    "PricingResources",
    "ScenarioEngine",
    "ScenarioMetrics",
    "ScenarioPerformance",
    "ScenarioWorkspace",
    "deterministic_seed",
    "distribution_metrics",
    "geometric_paths",
    "interpolate_price_grid",
    "residual_decay",
    "static_hedge_shares",
]
