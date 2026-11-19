"""Stable generic health field names shared by coordinator reporting."""

GENERIC_RECALC_METRICS = (
    "recalc_attempted_contracts",
    "recalc_completed_contracts",
    "recalc_recoverable_failures",
    "recalc_unrecoverable_failures",
    "recalc_batch_exceptions",
    "recalc_infrastructure_failures",
)

METHOD_RECALC_METRICS = tuple(
    f"recalc_{method}_{status}"
    for method in ("crr", "tian", "crank_nicolson")
    for status in ("complete", "failed")
)

