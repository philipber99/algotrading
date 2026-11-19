"""Solver-neutral bounded implied-volatility inversion."""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

from algotrading.refdata.options.solvers.Types import (
    IvHint,
    SolverError,
    SolverNumericalEvaluationError,
    SolverResourceError,
)


_BRENTQ: Optional[Callable[..., Any]] = None


def _load_brentq() -> Callable[..., Any]:
    global _BRENTQ
    if _BRENTQ is None:
        from scipy.optimize import brentq

        _BRENTQ = brentq
    return _BRENTQ


def preload_iv_solver_dependencies() -> None:
    """Load the configured IV root solver before timed contract work."""
    _load_brentq()


@dataclass(frozen=True)
class IvSolveResult:
    status: str
    reason: str
    iv: Optional[float]
    iterations: int
    model_price: Optional[float]
    price_residual: Optional[float]
    price_evaluations: int = 0
    hint_origin_used: str = "none"
    hint_value_used: Optional[float] = None
    hint_exact_match: bool = False
    hint_local_bracket_found: bool = False
    hint_evaluations: int = 0
    fallback_global_probes_used: bool = False
    identifiability_status: str = "not_checked"
    identifiability_price_span: Optional[float] = None
    identifiability_slope: Optional[float] = None
    hint_flat_rejection_count: int = 0
    requested_dtype: str = "float64"
    actual_dtype: str = "float64"
    precision_escalated: bool = False


@dataclass(frozen=True)
class IvBatchProblem:
    """One independent invocation of the mature scalar IV state machine."""

    target_price: float
    source_iv: Optional[float] = None
    iv_hints: tuple[IvHint, ...] = ()
    minimum: float = 1.0e-4
    maximum: float = 10.0
    price_tolerance: float = 1.0e-7
    max_iterations: int = 100
    source_bracket_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
    hint_bracket_multipliers: Optional[tuple[float, ...]] = None
    hint_identifiability_enabled: bool = True
    hint_identifiability_probe_iv_points: float = 1.0
    hint_identifiability_minimum_price_span_absolute: float = 1.0e-6
    hint_identifiability_price_tolerance_multiplier: float = 10.0


@dataclass
class _PendingPrice:
    volatility: float
    event: threading.Event
    value: Optional[float] = None
    exception: Optional[BaseException] = None


class _BatchPriceBroker:
    """Barrier between independent IV state machines and aligned price waves."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.pending: dict[int, _PendingPrice] = {}

    def evaluate(self, index: int, volatility: float) -> float:
        pending = _PendingPrice(float(volatility), threading.Event())
        with self.condition:
            if index in self.pending:
                raise RuntimeError("one IV job issued overlapping price evaluations")
            self.pending[index] = pending
            self.condition.notify_all()
        pending.event.wait()
        if pending.exception is not None:
            raise pending.exception
        return float(pending.value)

    def notify_completion(self, _future: Future[Any]) -> None:
        with self.condition:
            self.condition.notify_all()


def solve_iv_many(
    problems: Sequence[IvBatchProblem],
    price_many: Callable[[Sequence[int], Any], Any],
    *,
    maximum_state_workers: int = 64,
    coalesce_seconds: float = 0.0002,
) -> list[IvSolveResult]:
    """Preserve :func:`solve_iv` exactly while batching its valuation callbacks.

    Each logical solve still executes the established hint, identifiability,
    local/global bracket, cache, and Brent code.  Its price callback blocks at a
    small broker, which combines the next requested evaluation from every active
    state machine into one aligned numerical call.  The contract-axis work is
    therefore performed by CRR/Tian tensors without reimplementing IV semantics.
    """
    count = len(problems)
    if count == 0:
        return []
    if maximum_state_workers <= 0:
        raise ValueError("maximum_state_workers must be positive")
    if not math.isfinite(float(coalesce_seconds)) or coalesce_seconds < 0.0:
        raise ValueError("coalesce_seconds must be finite and nonnegative")
    broker = _BatchPriceBroker()

    def run(index: int, problem: IvBatchProblem) -> IvSolveResult:
        return solve_iv(
            target_price=problem.target_price,
            price_function=lambda volatility: broker.evaluate(index, volatility),
            source_iv=problem.source_iv,
            iv_hints=problem.iv_hints,
            minimum=problem.minimum,
            maximum=problem.maximum,
            price_tolerance=problem.price_tolerance,
            max_iterations=problem.max_iterations,
            source_bracket_multipliers=problem.source_bracket_multipliers,
            hint_bracket_multipliers=problem.hint_bracket_multipliers,
            hint_identifiability_enabled=problem.hint_identifiability_enabled,
            hint_identifiability_probe_iv_points=(
                problem.hint_identifiability_probe_iv_points
            ),
            hint_identifiability_minimum_price_span_absolute=(
                problem.hint_identifiability_minimum_price_span_absolute
            ),
            hint_identifiability_price_tolerance_multiplier=(
                problem.hint_identifiability_price_tolerance_multiplier
            ),
        )

    executor = ThreadPoolExecutor(
        max_workers=min(count, int(maximum_state_workers)),
        thread_name_prefix="iv-state",
    )
    futures = [executor.submit(run, index, problem) for index, problem in enumerate(problems)]
    for future in futures:
        future.add_done_callback(broker.notify_completion)
    try:
        while not all(future.done() for future in futures):
            with broker.condition:
                if not broker.pending:
                    broker.condition.wait(timeout=0.05)
                    continue
                if coalesce_seconds:
                    deadline = time.monotonic() + float(coalesce_seconds)
                    previous = -1
                    while previous != len(broker.pending):
                        previous = len(broker.pending)
                        remaining = deadline - time.monotonic()
                        if remaining <= 0.0:
                            break
                        broker.condition.wait(timeout=remaining)
                indices = sorted(broker.pending)
                wave = [broker.pending.pop(index) for index in indices]
            try:
                import numpy as np

                values = np.asarray(
                    price_many(
                        indices,
                        np.ascontiguousarray(
                            [pending.volatility for pending in wave], dtype=np.float64
                        ),
                    )
                ).reshape(-1)
                if len(values) != len(wave):
                    raise ValueError("price_many returned a misaligned result")
                for pending, value in zip(wave, values):
                    pending.value = float(value)
            except BaseException as exc:
                for pending in wave:
                    pending.exception = exc
            finally:
                for pending in wave:
                    pending.event.set()
        return [future.result() for future in futures]
    finally:
        # Release any waiters if an unexpected coordinator exception occurred.
        with broker.condition:
            stranded = list(broker.pending.values())
            broker.pending.clear()
        for pending in stranded:
            pending.exception = RuntimeError("IV batch coordinator terminated")
            pending.event.set()
        executor.shutdown(wait=True, cancel_futures=False)


def _finite_price(function: Callable[[float], float], volatility: float) -> Optional[float]:
    try:
        price = float(function(float(volatility)))
    except MemoryError as exc:
        raise SolverResourceError("solver valuation exhausted memory") from exc
    except SolverError:
        raise
    except FloatingPointError as exc:
        raise SolverNumericalEvaluationError(
            "solver price evaluation failed numerically"
        ) from exc
    except Exception:
        raise
    return price if math.isfinite(price) else None


def solve_iv(
    *,
    target_price: float,
    price_function: Callable[[float], float],
    source_iv: Optional[float] = None,
    iv_hints: Sequence[IvHint] = (),
    minimum: float = 1.0e-4,
    maximum: float = 10.0,
    price_tolerance: float = 1.0e-7,
    max_iterations: int = 100,
    source_bracket_multipliers: Iterable[float] = (0.5, 0.75, 1.0, 1.25, 1.5),
    hint_bracket_multipliers: Optional[Iterable[float]] = None,
    hint_identifiability_enabled: bool = True,
    hint_identifiability_probe_iv_points: float = 1.0,
    hint_identifiability_minimum_price_span_absolute: float = 1.0e-6,
    hint_identifiability_price_tolerance_multiplier: float = 10.0,
) -> IvSolveResult:
    """Use ordered hints, then deterministic global probes, in one cached solve."""
    try:
        target = float(target_price)
        low_bound = float(minimum)
        high_bound = float(maximum)
        tolerance = float(price_tolerance)
        limit = int(max_iterations)
        probe_width = float(hint_identifiability_probe_iv_points) / 100.0
        minimum_span_absolute = float(
            hint_identifiability_minimum_price_span_absolute
        )
        tolerance_multiplier = float(
            hint_identifiability_price_tolerance_multiplier
        )
    except (TypeError, ValueError, OverflowError):
        return IvSolveResult("failed", "invalid_iv_configuration", None, 0, None, None)
    if not (
        math.isfinite(target)
        and target > 0.0
        and math.isfinite(low_bound)
        and math.isfinite(high_bound)
        and 0.0 < low_bound < high_bound
        and math.isfinite(tolerance)
        and tolerance >= 0.0
        and limit > 0
        and math.isfinite(probe_width)
        and probe_width > 0.0
        and math.isfinite(minimum_span_absolute)
        and minimum_span_absolute >= 0.0
        and math.isfinite(tolerance_multiplier)
        and tolerance_multiplier > 0.0
    ):
        return IvSolveResult("failed", "invalid_iv_configuration", None, 0, None, None)

    price_cache: dict[float, Optional[float]] = {}
    evaluated: dict[float, float] = {}
    evaluated_origin: dict[float, str] = {}
    hint_evaluations = 0
    global_used = False
    local_bracket_found = False
    selected_hint_origin = "none"
    selected_hint_value: Optional[float] = None
    identifiability_status = "not_checked"
    identifiability_price_span: Optional[float] = None
    identifiability_slope: Optional[float] = None
    flat_rejection_count = 0
    flat_region_seen = False

    def result(
        status: str,
        reason: str,
        iv: Optional[float],
        iterations: int,
        model_price: Optional[float],
        price_residual: Optional[float],
        *,
        hint_origin: Optional[str] = None,
        hint_value: Optional[float] = None,
        exact: bool = False,
    ) -> IvSolveResult:
        return IvSolveResult(
            status,
            reason,
            iv,
            iterations,
            model_price,
            price_residual,
            len(price_cache),
            hint_origin if hint_origin is not None else selected_hint_origin,
            hint_value if hint_value is not None else selected_hint_value,
            exact,
            local_bracket_found,
            hint_evaluations,
            global_used,
            identifiability_status,
            identifiability_price_span,
            identifiability_slope,
            flat_rejection_count,
        )

    def evaluated_price(volatility: float) -> Optional[float]:
        key = float(volatility)
        if key not in price_cache:
            price_cache[key] = _finite_price(price_function, key)
        return price_cache[key]

    normalized_hints: list[IvHint] = []
    seen_hints: set[tuple[float, str]] = set()
    candidates = list(iv_hints)
    if source_iv is not None and not any(hint.origin == "source" for hint in candidates):
        candidates.append(IvHint(source_iv, "source"))
    for candidate in candidates:
        try:
            value = float(candidate.value)
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            not math.isfinite(value)
            or value < low_bound
            or value > high_bound
            or (value, str(candidate.origin or "none")) in seen_hints
        ):
            continue
        seen_hints.add((value, str(candidate.origin or "none")))
        normalized_hints.append(
            IvHint(
                value,
                str(candidate.origin or "none"),
                candidate.source_snapshot_id,
                candidate.source_calculated_at,
            )
        )

    multipliers = tuple(
        hint_bracket_multipliers
        if hint_bracket_multipliers is not None
        else source_bracket_multipliers
    )

    def evaluate(
        volatility: float, origin: str, *, hint_phase: bool, global_phase: bool = False
    ) -> Optional[tuple[float, float]]:
        nonlocal hint_evaluations, global_used
        value = float(volatility)
        fresh = value not in price_cache
        price = evaluated_price(value)
        if fresh and hint_phase:
            hint_evaluations += 1
        if global_phase:
            global_used = True
        if price is None:
            return None
        residual = price - target
        evaluated[value] = residual
        evaluated_origin.setdefault(value, origin)
        return price, residual

    def current_bracket() -> Optional[tuple[float, float, float, float]]:
        ordered = sorted(evaluated.items())
        for (left, f_left), (right, f_right) in zip(ordered, ordered[1:]):
            if f_left * f_right < 0.0:
                return left, right, f_left, f_right
        return None

    def identifiable_at(volatility: float, origin: str) -> bool:
        nonlocal identifiability_status
        nonlocal identifiability_price_span, identifiability_slope
        if not hint_identifiability_enabled:
            identifiability_status = "not_checked"
            return True
        half_width = probe_width / 2.0
        left = max(low_bound, float(volatility) - half_width)
        right = min(high_bound, float(volatility) + half_width)
        if right <= left:
            identifiability_price_span = 0.0
            identifiability_slope = 0.0
            return False
        prices = []
        for candidate in (left, float(volatility), right):
            checked = evaluate(candidate, origin, hint_phase=True)
            if checked is not None:
                prices.append(checked[0])
        if len(prices) < 2:
            identifiability_price_span = 0.0
            identifiability_slope = 0.0
            return False
        span = max(prices) - min(prices)
        identifiability_price_span = float(span)
        identifiability_slope = float(span / (right - left))
        required = max(minimum_span_absolute, tolerance * tolerance_multiplier)
        return span >= required

    bracket: Optional[tuple[float, float, float, float]] = None
    for hint in normalized_hints:
        checked = evaluate(hint.value, hint.origin, hint_phase=True)
        if checked is not None and abs(checked[1]) <= tolerance:
            if not identifiable_at(hint.value, hint.origin):
                flat_region_seen = True
                if hint.origin == "source":
                    identifiability_status = "source_anchor_flat"
                    return result(
                        "complete",
                        "source_candidate_validated",
                        hint.value,
                        1,
                        checked[0],
                        checked[1],
                        hint_origin=hint.origin,
                        hint_value=hint.value,
                        exact=True,
                    )
                identifiability_status = "prior_hint_rejected_flat"
                flat_rejection_count += 1
                continue
            identifiability_status = "identifiable"
            return result(
                "complete",
                "source_candidate_validated"
                if hint.origin == "source"
                else "hint_candidate_validated",
                hint.value,
                1,
                checked[0],
                checked[1],
                hint_origin=hint.origin,
                hint_value=hint.value,
                exact=True,
            )
        local_values: list[float] = []
        for multiplier in multipliers:
            try:
                value = hint.value * float(multiplier)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                bounded = min(high_bound, max(low_bound, value))
                if bounded != hint.value and bounded not in local_values:
                    local_values.append(bounded)
        hint_residual = checked[1] if checked is not None else None
        local_values.sort(
            key=lambda value: (
                0
                if hint_residual is not None
                and ((hint_residual < 0 and value > hint.value)
                     or (hint_residual > 0 and value < hint.value))
                else 1,
                abs(math.log(max(value, 1.0e-300) / hint.value)),
                value,
            )
        )
        for value in local_values:
            checked = evaluate(value, hint.origin, hint_phase=True)
            if checked is not None and abs(checked[1]) <= tolerance:
                if not identifiable_at(value, hint.origin):
                    flat_region_seen = True
                    if hint.origin == "source":
                        identifiability_status = "source_anchor_flat"
                        return result(
                            "complete", "price_residual_converged", value, 1,
                            checked[0], checked[1],
                            hint_origin=hint.origin, hint_value=hint.value,
                        )
                    identifiability_status = "prior_hint_rejected_flat"
                    flat_rejection_count += 1
                    break
                identifiability_status = "identifiable"
                return result(
                    "complete", "price_residual_converged", value, 1,
                    checked[0], checked[1],
                    hint_origin=hint.origin, hint_value=hint.value,
                )
            bracket = current_bracket()
            if bracket is not None:
                local_bracket_found = True
                selected_hint_origin = hint.origin
                selected_hint_value = hint.value
                break
        if bracket is not None:
            break

    if bracket is None:
        center = (
            normalized_hints[0].value
            if normalized_hints
            else min(high_bound, max(low_bound, 0.20))
        )
        global_candidates = {low_bound, high_bound}
        for candidate in (
            0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0
        ):
            if low_bound < candidate < high_bound:
                global_candidates.add(candidate)
        center_residual = evaluated.get(center)
        lower = sorted((value for value in global_candidates if value < center), reverse=True)
        upper = sorted(value for value in global_candidates if value > center)
        if center_residual is not None and center_residual < 0:
            ordered_candidates = upper + lower
        elif center_residual is not None and center_residual > 0:
            ordered_candidates = lower + upper
        else:
            ordered_candidates = sorted(
                global_candidates,
                key=lambda value: (
                    abs(math.log(max(value, 1.0e-300) / max(center, 1.0e-300))),
                    value,
                ),
            )
        if normalized_hints:
            selected_hint_origin = normalized_hints[0].origin
            selected_hint_value = normalized_hints[0].value
        for volatility in ordered_candidates:
            checked = evaluate(
                volatility, "global", hint_phase=False, global_phase=True
            )
            if checked is None:
                continue
            if abs(checked[1]) <= tolerance:
                if not identifiable_at(float(volatility), "global"):
                    flat_region_seen = True
                    identifiability_status = "not_identifiable"
                    continue
                identifiability_status = "identifiable"
                return result(
                    "complete", "price_residual_converged", float(volatility), 1,
                    checked[0], checked[1],
                )
            bracket = current_bracket()
            if bracket is not None:
                break
    if bracket is None:
        if flat_region_seen:
            identifiability_status = "not_identifiable"
            selected_hint_origin = "none"
            selected_hint_value = None
            return result(
                "failed", "iv_price_not_identifiable", None, 0, None, None
            )
        if len(evaluated) < 2:
            return result(
                "failed", "nonfinite_model_price", None, 0, None, None
            )
        return result("failed", "iv_root_not_bracketed", None, 0, None, None)
    left, right, _f_left, _f_right = bracket
    if hint_identifiability_enabled:
        identifiability_status = "identifiable"
    if selected_hint_origin == "none":
        endpoint_origins = (evaluated_origin.get(left), evaluated_origin.get(right))
        for hint in normalized_hints:
            if hint.origin in endpoint_origins:
                selected_hint_origin = hint.origin
                selected_hint_value = hint.value
                break

    class _PriceToleranceReached(Exception):
        def __init__(self, volatility: float, price: float, residual: float):
            self.volatility = volatility
            self.price = price
            self.residual = residual

    class _NonfiniteObjective(Exception):
        pass

    brent_calls = 0

    def objective(volatility: float) -> float:
        nonlocal brent_calls
        brent_calls += 1
        price = evaluated_price(volatility)
        if price is None:
            raise _NonfiniteObjective
        residual = price - target
        if abs(residual) <= tolerance:
            raise _PriceToleranceReached(float(volatility), price, residual)
        return residual

    try:
        root, details = _load_brentq()(
            objective,
            left,
            right,
            xtol=1.0e-12,
            rtol=4.0 * math.ulp(1.0),
            maxiter=limit,
            full_output=True,
            disp=False,
        )
    except _PriceToleranceReached as solved:
        return result(
            "complete",
            "price_residual_converged",
            solved.volatility,
            max(1, brent_calls),
            solved.price,
            solved.residual,
        )
    except _NonfiniteObjective:
        return result(
            "failed", "nonfinite_model_price", None, brent_calls, None, None
        )
    final_price = evaluated_price(float(root))
    final_residual = final_price - target if final_price is not None else None
    iterations = max(1, int(getattr(details, "iterations", brent_calls)))
    if (
        bool(getattr(details, "converged", False))
        and final_residual is not None
        and abs(final_residual) <= tolerance
    ):
        return result(
            "complete",
            "price_residual_converged",
            float(root),
            iterations,
            final_price,
            final_residual,
        )
    return result(
        "failed",
        "iv_price_tolerance_not_reached",
        None,
        iterations,
        final_price,
        final_residual,
    )
