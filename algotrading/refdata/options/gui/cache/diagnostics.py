"""Bounded, versioned diagnostics for Time-of-Day profile queries."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


DIAGNOSTICS_VERSION = 1
MAX_EXAMPLES_PER_REASON = 3

REJECTION_REASONS = (
    "SESSION_COVERAGE_BELOW_MINIMUM",
    "DAY_OF_WEEK_EXCLUDED",
    "NO_SURFACE_FILE_FOR_WINDOW",
    "QUALITY_FLAG_EXCLUDED",
    "FIT_ARBITRAGE_EXCLUDED",
    "STALE_SAMPLE_EXCLUDED",
    "INTERVAL_EXCLUDED",
    "NO_SESSION_MINUTE_AT_OR_AFTER_ENTRY",
    "ENTRY_AFTER_PATH_END",
    "NO_EXPIRATION_CANDIDATE",
    "DTE_OUTSIDE_TOLERANCE",
    "MISSING_LATER_EXPIRATION",
    "INVALID_FORWARD_PRICE",
    "MONEYNESS_BELOW_SURFACE_COVERAGE",
    "MONEYNESS_ABOVE_SURFACE_COVERAGE",
    "SURFACE_VALUE_UNAVAILABLE",
    "OPTION_PRICING_UNAVAILABLE",
    "ZERO_ENTRY_VALUE_FOR_NORMALIZATION",
    "NO_COMPLETE_OBSERVATIONS",
    "INSUFFICIENT_SAMPLE_COUNT",
    "PRICE_BASIS_UNAVAILABLE",
    "SYMBOL_NOT_CACHED",
)

SUGGESTION_CODES = (
    "USE_EARLIER_ENTRY_MINUTE",
    "INCREASE_DTE_TOLERANCE",
    "USE_SUPPORTED_DTE",
    "USE_STRIKE_WITHIN_SURFACE_COVERAGE",
    "LOWER_MINIMUM_SESSION_COVERAGE",
    "REMOVE_OR_RELAX_QUALITY_EXCLUSIONS",
    "USE_AVAILABLE_PRICE_BASIS",
    "USE_SHORTER_HISTORICAL_WINDOW",
    "LOWER_MINIMUM_SAMPLE_COUNT",
)

# This explicit order makes dominant-reason ties and output deterministic.
_REASON_PRIORITY = {reason: index for index, reason in enumerate(REJECTION_REASONS)}

FUNNEL_FIELDS = (
    "symbol_sessions",
    "sessions_after_coverage_filter",
    "sessions_after_day_of_week_filter",
    "sessions_selected_by_window",
    "surface_rows_loaded",
    "surface_rows_after_overlay_deduplication",
    "surface_rows_after_quality_filter",
    "surface_rows_after_interval_filter",
    "candidate_session_minutes",
    "entry_eligible_sessions",
    "dte_matched_session_minutes",
    "complete_observations",
    "sessions_with_complete_observations",
    "profile_bins",
)

ENTRY_FIELDS = (
    "requested_entry_session_minute",
    "sessions_with_sample_at_or_after_entry",
    "sessions_resolved_exactly_at_entry",
    "sessions_resolved_after_entry",
    "minimum_resolved_entry_minute",
    "maximum_resolved_entry_minute",
    "maximum_entry_slippage_minutes",
    "later_path_candidate_points",
    "later_path_complete_points",
)


def _native(value: Any) -> Any:
    """Convert diagnostics to JSON-native, finite, deterministic values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        text = value.isoformat()
        if isinstance(value, datetime) and value.tzinfo is not None:
            text = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return text
    if isinstance(value, Enum):
        return _native(value.value)
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_native(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _native(item())
    return str(value)


@dataclass
class LegDiagnostics:
    leg_index: int
    put_call: str
    side: str
    target_dte_days: float
    dte_tolerance_days: float
    target_log_forward_moneyness: float
    expiration_group: str
    dte_match_failures: int = 0
    moneyness_below_failures: int = 0
    moneyness_above_failures: int = 0
    surface_value_failures: int = 0
    nearest_dte_days_min: float | None = None
    nearest_dte_days_max: float | None = None
    observed_moneyness_min: float | None = None
    observed_moneyness_max: float | None = None

    def observe_dte(self, value: float | None, *, failed: bool = False) -> None:
        if failed:
            self.dte_match_failures += 1
        if value is None or not math.isfinite(value):
            return
        self.nearest_dte_days_min = (
            value
            if self.nearest_dte_days_min is None
            else min(self.nearest_dte_days_min, value)
        )
        self.nearest_dte_days_max = (
            value
            if self.nearest_dte_days_max is None
            else max(self.nearest_dte_days_max, value)
        )

    def observe_moneyness(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.observed_moneyness_min = (
            value
            if self.observed_moneyness_min is None
            else min(self.observed_moneyness_min, value)
        )
        self.observed_moneyness_max = (
            value
            if self.observed_moneyness_max is None
            else max(self.observed_moneyness_max, value)
        )

    def as_dict(self) -> dict[str, Any]:
        return _native(vars(self))


@dataclass
class ProfileDiagnostics:
    request_summary: dict[str, Any]
    cache_coverage: dict[str, Any]
    funnel: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in FUNNEL_FIELDS}
    )
    entry: dict[str, Any] = field(
        default_factory=lambda: {
            name: None
            if name
            in {
                "requested_entry_session_minute",
                "minimum_resolved_entry_minute",
                "maximum_resolved_entry_minute",
                "maximum_entry_slippage_minutes",
            }
            else 0
            for name in ENTRY_FIELDS
        }
    )
    legs: list[LegDiagnostics] = field(default_factory=list)
    rejection_counts: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    selected_dates: list[date] = field(default_factory=list, repr=False)

    def increment_stage(self, name: str, amount: int = 1) -> None:
        if name not in self.funnel:
            raise KeyError(f"Unknown diagnostic funnel stage {name!r}")
        self.funnel[name] += int(amount)

    def set_stage(self, name: str, value: int) -> None:
        if name not in self.funnel:
            raise KeyError(f"Unknown diagnostic funnel stage {name!r}")
        self.funnel[name] = int(value)

    def reject(
        self,
        reason: str,
        *,
        stage: str,
        leg_index: int | None = None,
        example: Mapping[str, Any] | None = None,
        amount: int = 1,
    ) -> None:
        if reason not in _REASON_PRIORITY:
            raise ValueError(f"Unknown diagnostic rejection reason {reason!r}")
        self.rejection_counts[reason] += int(amount)
        if example is not None:
            values = self.examples.setdefault(reason, [])
            if len(values) < MAX_EXAMPLES_PER_REASON:
                item = {"stage": stage, **dict(example)}
                if leg_index is not None:
                    item["leg_index"] = int(leg_index)
                values.append(_native(item))

    def record_entry_resolution(self, requested: int, resolved: int) -> None:
        self.entry["requested_entry_session_minute"] = int(requested)
        self.entry["sessions_with_sample_at_or_after_entry"] += 1
        if resolved == requested:
            self.entry["sessions_resolved_exactly_at_entry"] += 1
        else:
            self.entry["sessions_resolved_after_entry"] += 1
        self.entry["minimum_resolved_entry_minute"] = _minimum(
            self.entry["minimum_resolved_entry_minute"], resolved
        )
        self.entry["maximum_resolved_entry_minute"] = _maximum(
            self.entry["maximum_resolved_entry_minute"], resolved
        )
        slippage = resolved - requested
        self.entry["maximum_entry_slippage_minutes"] = _maximum(
            self.entry["maximum_entry_slippage_minutes"], slippage
        )

    def leg(self, index: int) -> LegDiagnostics:
        return self.legs[index]

    @property
    def dominant_reason(self) -> str | None:
        if not self.rejection_counts:
            return None
        return min(
            self.rejection_counts,
            key=lambda reason: (
                -self.rejection_counts[reason],
                _REASON_PRIORITY[reason],
            ),
        )

    def add_suggestion(
        self,
        code: str,
        message: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if code not in SUGGESTION_CODES:
            raise ValueError(f"Unknown diagnostic suggestion code {code!r}")
        if any(item["code"] == code for item in self.suggestions):
            return
        self.suggestions.append(
            _native(
                {
                    "code": code,
                    "message": message,
                    "parameters": dict(parameters or {}),
                }
            )
        )

    def as_dict(self) -> dict[str, Any]:
        counts = {
            reason: int(self.rejection_counts[reason])
            for reason in sorted(
                self.rejection_counts,
                key=lambda item: _REASON_PRIORITY[item],
            )
        }
        examples = {
            reason: list(self.examples[reason])
            for reason in sorted(self.examples, key=lambda item: _REASON_PRIORITY[item])
        }
        return _native(
            {
                "diagnostics_version": DIAGNOSTICS_VERSION,
                "request_summary": self.request_summary,
                "cache_coverage": self.cache_coverage,
                "funnel": self.funnel,
                "rejections": {
                    "dominant_reason": self.dominant_reason,
                    "by_reason": counts,
                    "examples": examples,
                },
                "per_leg": [item.as_dict() for item in self.legs],
                "entry": self.entry,
                "suggestions": self.suggestions,
            }
        )


def _minimum(current: int | None, value: int) -> int:
    return value if current is None else min(current, value)


def _maximum(current: int | None, value: int) -> int:
    return value if current is None else max(current, value)
