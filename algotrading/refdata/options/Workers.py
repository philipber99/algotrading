"""Spawn-based generation-3 process tasks for option calculations."""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import os
import socket
import time
import uuid
import atexit
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from algotrading.ContextFields import PARAM_LOGGER
from algotrading.refdata.options.Common import OPTIONS_CALCS_CONFIG_PREFIX
from algotrading.refdata.options.Runtime import (
    CalculationRepository,
    CalculationWork,
    SurfaceRepository,
    SurfaceWork,
)


UTC = timezone.utc
_WORKER_CONTEXT: Optional[dict[str, Any]] = None
_WORKER_CLIENT: Any = None
_WORKER_LOGGER: Optional[logging.Logger] = None
_WORKER_CALCULATIONS: Optional[CalculationRepository] = None
_WORKER_SURFACES: Optional[SurfaceRepository] = None
_WORKER_SURFACE_CACHE: Any = None
_WORKER_SURFACE_RUNTIME: Any = None
_WORKER_SOLVER_REGISTRY: Any = None
_WORKER_FRED_CURVE: Any = None


@dataclass(frozen=True)
class ScheduledCalculationTask:
    work: CalculationWork
    attempt_id: uuid.UUID
    submitted_at: datetime


@dataclass(frozen=True)
class ScheduledSurfaceTask:
    work: SurfaceWork
    attempt_id: uuid.UUID
    submitted_at: datetime


@dataclass(frozen=True)
class WorkerTaskResult:
    process_type: str
    snapshot_id: uuid.UUID
    attempt_id: uuid.UUID
    status: str
    worker_pid: int
    worker_process_name: str
    submitted_at: datetime
    started_at: Optional[datetime]
    finished_at: datetime
    claim_acquired: bool
    error_type: str = ""
    error_message: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(UTC)


def _device_type(value: object) -> str:
    name = str(value or "cpu").strip().lower()
    if name.startswith("cuda") or name == "rocm":
        return "cuda"
    return name.split(":", 1)[0]


def validate_cpu_only(context: Mapping[str, Any]) -> None:
    from algotrading.refdata.options.Common import validate_options_calcs_configuration

    validate_options_calcs_configuration(context)
    prefix = OPTIONS_CALCS_CONFIG_PREFIX
    workers = int(context.get(prefix + "workers", 1))
    if workers <= 1:
        return
    possible_devices = {
        _device_type(context.get(prefix + "greeks.crr.device", "cpu")),
        _device_type(context.get(prefix + "greeks.tian.device", "cpu")),
    }
    if possible_devices != {"cpu"}:
        raise ValueError(
            "options_calcs.workers > 1 requires CPU-only pricing engines"
        )


def configure_cpu_thread_limits(context: Mapping[str, Any]) -> int:
    prefix = OPTIONS_CALCS_CONFIG_PREFIX
    threads = int(context.get(prefix + "worker.cpu_threads_per_process", 1))
    if threads <= 0:
        raise ValueError(
            "options_calcs.worker.cpu_threads_per_process must be positive"
        )
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    return threads


def _configure_dependency_log_levels() -> None:
    logging.getLogger("clickhouse_connect").setLevel(logging.WARNING)


def _close_worker_resources() -> None:
    close_curve = getattr(_WORKER_FRED_CURVE, "close", None)
    if callable(close_curve):
        try:
            close_curve()
        except Exception:
            pass
    close = getattr(_WORKER_CLIENT, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _create_worker_fred_curve(
    context: Mapping[str, Any], client: Any, logger: logging.Logger
) -> Any:
    """Small factory kept separate so process-local curve reuse is testable."""
    from algotrading.refdata.options.Calculations import FredCurve

    return FredCurve(context, client, logger)


def _get_worker_fred_curve() -> Any:
    if _WORKER_FRED_CURVE is None:
        raise RuntimeError("Worker-local FRED curve has not been initialized")
    return _WORKER_FRED_CURVE


def _worker_initializer(context: dict[str, Any], log_queue: Any) -> None:
    """Initialize all mutable numerical and persistence state once per child."""
    global _WORKER_CONTEXT, _WORKER_CLIENT, _WORKER_LOGGER
    global _WORKER_CALCULATIONS, _WORKER_SURFACES
    global _WORKER_SURFACE_CACHE, _WORKER_SURFACE_RUNTIME, _WORKER_SOLVER_REGISTRY
    global _WORKER_FRED_CURVE

    # This is deliberately repeated in children: imports of NumPy/Torch may
    # happen during repository/client initialization on some deployments.
    configure_cpu_thread_limits(context)
    from algotrading.refdata.options.Common import create_clickhouse_client
    from algotrading.refdata.options.Surface import (
        SVIWarmStartCache,
        SurfaceNumericalRuntime,
    )
    from algotrading.refdata.options.solvers.Registry import SolverRegistry

    _WORKER_CONTEXT = context
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    _configure_dependency_log_levels()
    _WORKER_LOGGER = logging.getLogger("options.worker")
    _WORKER_CLIENT = create_clickhouse_client(context)
    atexit.register(_close_worker_resources)
    database = str(context["options_chain.clickhouse.database"])
    base_worker_id = str(context.get(OPTIONS_CALCS_CONFIG_PREFIX + "worker_id", socket.gethostname()))
    child_worker_id = f"{base_worker_id}:{os.getpid()}"
    stale = int(context.get(OPTIONS_CALCS_CONFIG_PREFIX + "running_stale_seconds", 900))
    _WORKER_CALCULATIONS = CalculationRepository(
        _WORKER_CLIENT, database, worker_id=child_worker_id, stale_after_seconds=stale
    )
    _WORKER_SURFACES = SurfaceRepository(
        _WORKER_CLIENT, database, worker_id=child_worker_id, stale_after_seconds=stale
    )
    _WORKER_SURFACE_CACHE = SVIWarmStartCache(
        int(context.get(OPTIONS_CALCS_CONFIG_PREFIX + "surface.svi_warm_start_cache_size", 2048))
    )
    _WORKER_FRED_CURVE = _create_worker_fred_curve(
        context, _WORKER_CLIENT, _WORKER_LOGGER
    )
    _WORKER_SOLVER_REGISTRY = SolverRegistry(context)
    _WORKER_SOLVER_REGISTRY.preload()
    _WORKER_SURFACE_RUNTIME = SurfaceNumericalRuntime.from_context(context)
    _WORKER_LOGGER.info(
        "event=worker_process_started pid=%s process_name=%s worker_id=%s",
        os.getpid(), multiprocessing.current_process().name, child_worker_id,
    )
    _WORKER_LOGGER.info(
        "event=surface_node_methods search_method=%s validation_method=%s "
        "configured_default_method=%s configured_short_tenor_method=%s "
        "short_tenor_threshold_days=%s hybrid_enabled=%s tian_grid_size=%s "
        "validation_budget=%s",
        context.get(OPTIONS_CALCS_CONFIG_PREFIX + "surface.node_search_method", "tian"),
        context.get(
            OPTIONS_CALCS_CONFIG_PREFIX + "surface.node_validation_method",
            "configured",
        ),
        context.get(OPTIONS_CALCS_CONFIG_PREFIX + "greeks.method", "crank_nicolson"),
        context.get(OPTIONS_CALCS_CONFIG_PREFIX + "greeks.short_tenor_method", "tian"),
        context.get(
            OPTIONS_CALCS_CONFIG_PREFIX + "greeks.short_tenor_threshold_days",
            3.0,
        ),
        context.get(OPTIONS_CALCS_CONFIG_PREFIX + "surface.node_hybrid_enabled", True),
        context.get(OPTIONS_CALCS_CONFIG_PREFIX + "surface.node_tian_grid_size", 81),
        context.get(
            OPTIONS_CALCS_CONFIG_PREFIX
            + "surface.node_validation_max_evaluations",
            4,
        ),
    )


def _result(
    task: ScheduledCalculationTask | ScheduledSurfaceTask,
    *,
    process_type: str,
    status: str,
    started_at: Optional[datetime],
    claim_acquired: bool,
    error: Optional[BaseException] = None,
    metrics: Optional[Mapping[str, Any]] = None,
) -> WorkerTaskResult:
    return WorkerTaskResult(
        process_type=process_type,
        snapshot_id=task.work.snapshot_id,
        attempt_id=task.attempt_id,
        status=status,
        worker_pid=os.getpid(),
        worker_process_name=multiprocessing.current_process().name,
        submitted_at=task.submitted_at,
        started_at=started_at,
        finished_at=_now(),
        claim_acquired=claim_acquired,
        error_type=type(error).__name__ if error else "",
        error_message=str(error) if error else "",
        metrics=dict(metrics or {}),
    )


def _claim_conflict(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "already has a live running claim", "lost concurrent ownership",
            "terminal historical", "active migration lease",
            "requires explicit stale/abandoned reconciliation",
        )
    )


def execute_calculation_task(task: ScheduledCalculationTask) -> WorkerTaskResult:
    """Claim, calculate, and publish exactly one generation-3 snapshot."""
    assert _WORKER_CONTEXT is not None and _WORKER_CALCULATIONS is not None
    assert _WORKER_CLIENT is not None and _WORKER_LOGGER is not None
    work = task.work
    _WORKER_LOGGER.info(
        "event=work_started process_type=calculation snapshot_id=%s attempt_id=%s pid=%s process_name=%s",
        work.snapshot_id, task.attempt_id, os.getpid(), multiprocessing.current_process().name,
    )
    try:
        claim = _WORKER_CALCULATIONS.claim(work, attempt_id=task.attempt_id)
    except Exception as exc:
        status = "claim_conflict" if _claim_conflict(exc) else "failed"
        _WORKER_LOGGER.warning(
            "event=claim_conflict process_type=calculation snapshot_id=%s attempt_id=%s error=%s"
            if status == "claim_conflict"
            else "event=work_failed process_type=calculation snapshot_id=%s attempt_id=%s error=%s",
            work.snapshot_id, task.attempt_id, exc,
        )
        return _result(
            task, process_type="calculation", status=status,
            started_at=None, claim_acquired=False, error=exc,
        )
    try:
        from algotrading.refdata.options.Calculations import (
            _normalized_source_diagnostics, calculate_rows,
            load_normalized_diagnostics, project_canonical_calculation_rows,
        )
        from algotrading.refdata.options.Worker import (
            compose_calculation_publication, load_prior_outcomes,
        )

        source_rows = _WORKER_CALCULATIONS.load_source_rows(work.snapshot_id)
        raw_diagnostics = _normalized_source_diagnostics(source_rows)
        if raw_diagnostics is None:
            _WORKER_LOGGER.warning(
                "event=normalized_diagnostics_compatibility_fallback "
                "snapshot_id=%s reason=missing_required_source_fields",
                work.snapshot_id,
            )
            raw_diagnostics = load_normalized_diagnostics(
                _WORKER_CLIENT,
                str(_WORKER_CALCULATIONS.database),
                work.snapshot_id,
            )
        numerical_source = project_canonical_calculation_rows(work, source_rows)
        runtime_context = dict(_WORKER_CONTEXT)
        runtime_context[OPTIONS_CALCS_CONFIG_PREFIX + "calculation_definition_id"] = work.calculation_definition_id
        numerical = calculate_rows(
            runtime_context,
            numerical_source,
            _get_worker_fred_curve(),
            raw_diagnostics=raw_diagnostics,
            prior_outcome_loader=lambda rows, policy: load_prior_outcomes(
                _WORKER_CLIENT, str(_WORKER_CALCULATIONS.database), rows, policy
            ),
            solver_registry=_WORKER_SOLVER_REGISTRY,
        )
        publication = compose_calculation_publication(work, source_rows, numerical)
        metrics = _WORKER_CALCULATIONS.publish(
            claim, results=publication[0], observations=publication[1],
            source_only_contract_ids=publication[2], failures=publication[3],
            expiration_inputs=publication[4], reuse_provenance=publication[5],
        )
        return _result(
            task, process_type="calculation", status="complete",
            started_at=claim.started_at, claim_acquired=True, metrics=metrics,
        )
    except Exception as exc:
        try:
            _WORKER_CALCULATIONS.fail(
                claim, underlying_symbol=work.underlying_symbol,
                expected_observations=work.expected_observations, error=exc,
            )
        except Exception:
            raise
        _WORKER_LOGGER.exception(
            "event=work_failed process_type=calculation snapshot_id=%s attempt_id=%s",
            work.snapshot_id, task.attempt_id,
        )
        return _result(
            task, process_type="calculation", status="failed",
            started_at=claim.started_at, claim_acquired=True, error=exc,
        )


def execute_surface_task(task: ScheduledSurfaceTask) -> WorkerTaskResult:
    """Claim, fit, and publish exactly one surface snapshot."""
    assert _WORKER_CONTEXT is not None and _WORKER_SURFACES is not None
    assert _WORKER_LOGGER is not None and _WORKER_SURFACE_CACHE is not None
    work = task.work
    task_started = time.perf_counter()
    _WORKER_LOGGER.info(
        "event=work_started process_type=surface snapshot_id=%s attempt_id=%s pid=%s process_name=%s",
        work.snapshot_id, task.attempt_id, os.getpid(), multiprocessing.current_process().name,
    )
    try:
        source = _WORKER_SURFACES.load_source_rows(work.snapshot_id)
        expected_expirations = len({row["expiration_at"] for row in source})
        claim = _WORKER_SURFACES.claim(
            work, expected_expirations=expected_expirations,
            expected_required_nodes=0, attempt_id=task.attempt_id,
        )
    except Exception as exc:
        status = "claim_conflict" if _claim_conflict(exc) else "failed"
        _WORKER_LOGGER.warning(
            "event=%s process_type=surface snapshot_id=%s attempt_id=%s error=%s",
            "claim_conflict" if status == "claim_conflict" else "work_failed",
            work.snapshot_id, task.attempt_id, exc,
        )
        return _result(
            task, process_type="surface", status=status,
            started_at=None, claim_acquired=False, error=exc,
        )
    try:
        from algotrading.refdata.options.Worker import calculate_surface_publication

        fits, nodes, failures, skipped, required = calculate_surface_publication(
            _WORKER_CONTEXT, work, source, _WORKER_LOGGER,
            warm_start_cache=_WORKER_SURFACE_CACHE,
            numerical_runtime=_WORKER_SURFACE_RUNTIME,
            solver_registry=_WORKER_SOLVER_REGISTRY,
        )
        publish_metrics = _WORKER_SURFACES.publish(
            claim,
            source_calculation_definition_id=work.calculation_definition_id,
            fits=fits, nodes=nodes, failures=failures,
            expected_expirations=expected_expirations,
            skipped_expirations=skipped,
            expected_required_nodes=required,
            underlying_symbol=work.underlying_symbol,
        )
        metrics = dict(publish_metrics)
        metrics["surface.total_seconds"] = time.perf_counter() - task_started
        _WORKER_LOGGER.info(
            "event=surface_work_completed snapshot_id=%s metrics=%s",
            work.snapshot_id,
            dict(sorted(metrics.items())),
        )
        return _result(
            task, process_type="surface", status="complete",
            started_at=claim.started_at, claim_acquired=True, metrics=metrics,
        )
    except Exception as exc:
        try:
            _WORKER_SURFACES.fail(
                claim, underlying_symbol=work.underlying_symbol,
                source_calculation_definition_id=work.calculation_definition_id,
                expected_expirations=expected_expirations,
                expected_required_nodes=0, error=exc,
            )
        except Exception:
            raise
        _WORKER_LOGGER.exception(
            "event=work_failed process_type=surface snapshot_id=%s attempt_id=%s",
            work.snapshot_id, task.attempt_id,
        )
        return _result(
            task, process_type="surface", status="failed",
            started_at=claim.started_at, claim_acquired=True, error=exc,
        )


@dataclass
class LocalWorkerPool:
    """Small, explicitly spawn-only process-pool facade."""

    context: Mapping[str, Any]
    logger: logging.Logger

    def __post_init__(self) -> None:
        self._closed = False
        prefix = OPTIONS_CALCS_CONFIG_PREFIX
        validate_cpu_only(self.context)
        configure_cpu_thread_limits(self.context)
        workers = int(self.context.get(prefix + "workers", 1))
        if workers <= 0:
            raise ValueError("LocalWorkerPool requires a positive worker count")
        mp_context = multiprocessing.get_context("spawn")
        self._queue = mp_context.Queue()
        handlers = tuple(self.logger.handlers) or (logging.StreamHandler(),)
        self._listener = logging.handlers.QueueListener(
            self._queue, *handlers, respect_handler_level=True
        )
        self._listener.start()
        sanitized = dict(self.context)
        sanitized.pop(PARAM_LOGGER, None)
        max_tasks = int(self.context.get(prefix + "worker.max_tasks_per_child", 0))
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context,
            initializer=_worker_initializer,
            initargs=(sanitized, self._queue),
            max_tasks_per_child=max_tasks if max_tasks > 0 else None,
        )

    def submit_calculation(self, task: ScheduledCalculationTask) -> Future[Any]:
        return self._executor.submit(execute_calculation_task, task)

    def submit_surface(self, task: ScheduledSurfaceTask) -> Future[Any]:
        return self._executor.submit(execute_surface_task, task)

    def close(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        processes = tuple(getattr(self._executor, "_processes", {}).values())
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        if not wait:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=1.0)
                if process.is_alive():
                    kill = getattr(process, "kill", None)
                    if callable(kill):
                        kill()
        self._listener.stop()
        self._queue.close()

    def __enter__(self) -> "LocalWorkerPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(wait=True)


__all__ = [
    "LocalWorkerPool", "ScheduledCalculationTask", "ScheduledSurfaceTask",
    "WorkerTaskResult", "configure_cpu_thread_limits", "execute_calculation_task",
    "execute_surface_task", "validate_cpu_only",
]
