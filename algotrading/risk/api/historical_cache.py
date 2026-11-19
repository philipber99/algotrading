"""Shared keys for immutable snapshot historical-revaluation results."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from algotrading.risk.history.revaluation import historical_risk_position_eligible
from algotrading.risk.models import NormalizedPosition


def historical_revaluation_cache_key(
    reference_client: Any,
    positions: Sequence[NormalizedPosition],
    *,
    as_of: datetime,
    lookback_sessions: int,
) -> tuple[Any, ...]:
    """Key by every position input that can alter historical P&L.

    Snapshot position IDs alone are stable, but proposal legs retain their ID
    while quantity or price can be edited.  Including the normalized pricing
    inputs lets portfolio routes and Hedge Analysis safely share one cache.
    """

    eligible = tuple(item for item in positions if historical_risk_position_eligible(item))
    position_key = tuple(sorted(
        (
            item.position_id,
            item.signed_quantity,
            item.mark,
            item.delta,
            item.gamma,
            item.vega_per_1pct,
            item.theta_per_day,
            item.rho_per_1pct,
        )
        for item in eligible
    ))
    return (
        id(reference_client),
        as_of.isoformat(),
        int(lookback_sessions),
        position_key,
    )


__all__ = ["historical_revaluation_cache_key"]
