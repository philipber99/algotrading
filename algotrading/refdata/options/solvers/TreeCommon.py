"""Small numerical utilities shared by independently exposed tree engines."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


SECONDS_PER_YEAR = 365.25 * 86400.0


def numpy_dtype(value: Any, tenors: Iterable[float] = ()) -> np.dtype:
    name = str(value).lower().replace("torch.", "")
    requested = np.dtype("float32" if "32" in name else "float64")
    # Sub-day trees are always evaluated in float64.
    if any(0.0 < float(tenor) < 1.0 / 365.25 for tenor in tenors):
        return np.dtype("float64")
    return requested


def finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def aligned_arrays(*values: Any) -> list[np.ndarray]:
    arrays = [np.asarray(value).reshape(-1) for value in values]
    if arrays and any(len(value) != len(arrays[0]) for value in arrays[1:]):
        raise ValueError("All tree inputs must have the same length")
    return arrays

