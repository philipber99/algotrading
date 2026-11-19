"""Persistent, bounded process-pool execution for GUI option pricing."""

from __future__ import annotations

import math
import multiprocessing
import pickle
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .cache.pricing import create_solver_registry, price_iv_only_jobs


class PricingPoolTimeout(RuntimeError):
    """The request deadline elapsed while pricing work was pending."""


class PricingPoolBusy(RuntimeError):
    """The configured submitted-work capacity could not be acquired."""


class PricingPoolFailure(RuntimeError):
    """A worker failed or returned an invalid aligned result."""


_WORKER_CONTEXT: Mapping[str, Any] | None = None
_WORKER_REGISTRY: Any | None = None


def sanitize_worker_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only stable string keys and values safe for spawn pickling."""

    output: dict[str, Any] = {}
    for key, value in context.items():
        try:
            pickle.dumps(value)
        except Exception:
            continue
        output[str(key)] = value
    return output


def _initialize_worker(context: Mapping[str, Any], worker_threads: int) -> None:
    global _WORKER_CONTEXT, _WORKER_REGISTRY
    try:
        import torch

        torch.set_num_threads(max(1, int(worker_threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch permits setting inter-op threads only before parallel work.
            pass
    except ImportError:
        pass
    _WORKER_CONTEXT = dict(context)
    _WORKER_REGISTRY = create_solver_registry(_WORKER_CONTEXT)


def _price_worker_batch(jobs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if _WORKER_CONTEXT is None or _WORKER_REGISTRY is None:
        raise RuntimeError("Pricing worker is not initialized")
    return price_iv_only_jobs(
        jobs,
        context=_WORKER_CONTEXT,
        registry=_WORKER_REGISTRY,
    )


class PricingPool:
    """One lazily-created spawn pool with deterministic bounded batching."""

    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        workers: int,
        batch_size: int,
        max_pending_batches: int,
        worker_threads: int = 1,
        executor: Any | None = None,
        executor_factory: Callable[..., Any] = ProcessPoolExecutor,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if workers < 1 or batch_size < 1 or max_pending_batches < 1:
            raise ValueError("Pricing pool bounds must be positive")
        self.context = sanitize_worker_context(context)
        self.workers = int(workers)
        self.batch_size = int(batch_size)
        self.max_pending_batches = int(max_pending_batches)
        self.worker_threads = int(worker_threads)
        self.clock = clock
        self._executor = executor
        self._executor_factory = executor_factory
        self._owns_executor = executor is None
        self._pending = threading.BoundedSemaphore(self.max_pending_batches)
        self._pending_count = 0
        self._pending_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self.batches_submitted = 0
        self.jobs_submitted = 0

    @property
    def pending_batches(self) -> int:
        with self._pending_lock:
            return self._pending_count

    @property
    def started(self) -> bool:
        return self._executor is not None

    def _ensure_executor(self) -> Any:
        with self._lifecycle_lock:
            if self._closed:
                raise PricingPoolFailure("Pricing pool is closed")
            if self._executor is None:
                self._executor = self._executor_factory(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker,
                    initargs=(self.context, self.worker_threads),
                )
            return self._executor

    def start(self) -> None:
        """Create the persistent executor once without submitting work."""

        self._ensure_executor()

    def price_jobs(
        self,
        jobs: Sequence[Mapping[str, Any]],
        *,
        deadline: float,
    ) -> np.ndarray:
        if not jobs:
            return np.asarray([], dtype=np.float64)
        executor = self._ensure_executor()
        output = np.full(len(jobs), np.nan, dtype=np.float64)
        futures: list[tuple[Future[Any], int, int]] = []
        try:
            for start in range(0, len(jobs), self.batch_size):
                end = min(len(jobs), start + self.batch_size)
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise PricingPoolTimeout("Pricing deadline elapsed before submit")
                acquired = (
                    self._pending.acquire()
                    if math.isinf(remaining)
                    else self._pending.acquire(timeout=remaining)
                )
                if not acquired:
                    if self.clock() >= deadline:
                        raise PricingPoolTimeout(
                            "Pricing deadline elapsed waiting for capacity"
                        )
                    raise PricingPoolBusy("Pricing pending-work capacity is full")
                with self._pending_lock:
                    self._pending_count += 1
                try:
                    future = executor.submit(
                        _price_worker_batch,
                        tuple(jobs[start:end]),
                    )
                except BaseException as exc:
                    self._batch_finished()
                    raise PricingPoolFailure("Unable to submit a pricing batch") from exc
                future.add_done_callback(lambda _future: self._batch_finished())
                futures.append((future, start, end))
                self.batches_submitted += 1
                self.jobs_submitted += end - start

            # Placement uses submission ordinals, never completion order.
            for future, start, end in futures:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise PricingPoolTimeout("Pricing deadline elapsed while waiting")
                try:
                    values = np.asarray(
                        future.result(
                            timeout=None if math.isinf(remaining) else remaining
                        ),
                        dtype=np.float64,
                    ).reshape(-1)
                except FutureTimeout as exc:
                    raise PricingPoolTimeout(
                        "Pricing deadline elapsed while waiting"
                    ) from exc
                except BaseException as exc:
                    raise PricingPoolFailure("A pricing worker failed") from exc
                if len(values) != end - start or not np.all(np.isfinite(values)):
                    raise PricingPoolFailure(
                        "A pricing worker returned an invalid aligned batch"
                    )
                output[start:end] = values
        except BaseException:
            for future, _, _ in futures:
                future.cancel()
            raise
        if not np.all(np.isfinite(output)):
            raise PricingPoolFailure("Pricing output contains unfilled values")
        return output

    def _batch_finished(self) -> None:
        with self._pending_lock:
            self._pending_count = max(0, self._pending_count - 1)
        self._pending.release()

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None and self._owns_executor:
            executor.shutdown(wait=True, cancel_futures=True)

    def health(self) -> dict[str, int | bool]:
        return {
            "workers": self.workers,
            "batch_size": self.batch_size,
            "maximum_pending_batches": self.max_pending_batches,
            "pending_batches": self.pending_batches,
            "started": self.started,
        }


__all__ = [
    "PricingPool",
    "PricingPoolBusy",
    "PricingPoolFailure",
    "PricingPoolTimeout",
    "sanitize_worker_context",
]
