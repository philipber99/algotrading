"""Exact, causal exchange-session windows shared by Risk history consumers."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from threading import RLock
from typing import Any, Literal, Mapping, Sequence
from zoneinfo import ZoneInfo

from pydantic import Field

from algotrading.refdata.api.market_calendars import (
    CALENDAR_VERSION,
    MARKET_CALENDARS,
    SUPPORTED_START,
    exchange_sessions,
)
from algotrading.risk.models import ImmutableModel


HistoryKind = Literal["daily_close", "exact_option_mark", "factor_proxy"]
AvailabilityConvention = Literal["next_utc_midnight", "session_close"]


class HistoryPlanningError(ValueError):
    """A requested causal session window cannot be planned exactly."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class HistoryRequestChunk(ImmutableModel):
    start_at: datetime
    end_at: datetime
    sessions: tuple[date, ...]


class HistoryWindowPlan(ImmutableModel):
    as_of: datetime
    history_kind: HistoryKind
    availability_convention: AvailabilityConvention
    calendar_id: str
    calendar_version: str = CALENDAR_VERSION
    first_required_session: date
    last_required_session: date
    requested_return_count: int = Field(gt=0)
    required_mark_count: int = Field(gt=1)
    required_sessions: tuple[date, ...]
    start_at: datetime
    end_at: datetime
    local_maximum_days: int = Field(gt=0)
    advertised_maximum_days: int | None = Field(default=None, gt=0)
    effective_maximum_days: int = Field(gt=0)
    server_limit_negotiated: bool
    chunks: tuple[HistoryRequestChunk, ...]


class HistoryWindowPlanner:
    """Plan exactly ``N + 1`` causally available exchange sessions.

    The RefData daily endpoint interprets a half-open UTC interval by taking
    ``start_at.date()`` through ``(end_at - 1 microsecond).date()``.  Therefore
    each request starts at UTC midnight on its first session date and ends at
    UTC midnight after its final session date.  Chunk construction measures
    those exact boundaries, avoiding an inclusive-end microsecond overrun.
    """

    def __init__(self, context: Mapping[str, Any] | None = None) -> None:
        self.context = context or {}

    def plan(
        self,
        *,
        as_of: datetime,
        requested_return_count: int,
        calendar_id: str,
        history_kind: HistoryKind,
        availability_convention: AvailabilityConvention | None = None,
        local_maximum_days: int | None = None,
        advertised_maximum_days: int | None = None,
    ) -> HistoryWindowPlan:
        boundary = _aware_utc(as_of)
        if requested_return_count < 1:
            raise HistoryPlanningError(
                "invalid_return_count", "requested_return_count must be positive"
            )
        if calendar_id not in MARKET_CALENDARS:
            raise HistoryPlanningError(
                "unknown_exchange_calendar", f"unknown market calendar: {calendar_id}"
            )
        convention = availability_convention or str(
            self.context.get(
                "risk.history.daily_availability_convention",
                "next_utc_midnight",
            )
        )
        if convention not in {"next_utc_midnight", "session_close"}:
            raise HistoryPlanningError(
                "unknown_availability_convention",
                f"unknown daily availability convention: {convention}",
            )
        last = self.latest_available_session(
            as_of=boundary,
            calendar_id=calendar_id,
            availability_convention=convention,
        )
        available = exchange_sessions(calendar_id, SUPPORTED_START, last)
        # N one-session returns require N + 1 causally available closing marks.
        mark_count = requested_return_count + 1
        if len(available) < mark_count:
            raise HistoryPlanningError(
                "calendar_sessions_unavailable",
                f"calendar {calendar_id} cannot supply {mark_count} completed sessions",
            )
        sessions = tuple(available[-mark_count:])
        default_limit_key = (
            "risk.option_prices.max_history_days"
            if history_kind == "exact_option_mark"
            else "risk.history.daily_bar_maximum_days"
        )
        local_limit = max(
            1,
            int(
                local_maximum_days
                if local_maximum_days is not None
                else self.context.get(default_limit_key, 376)
            ),
        )
        advertised = (
            max(1, int(advertised_maximum_days))
            if advertised_maximum_days is not None
            else None
        )
        # Respect both the local request bound and the provider-advertised limit.
        effective = min(local_limit, advertised) if advertised else local_limit
        chunks = chunk_session_window(sessions, effective)
        return HistoryWindowPlan(
            as_of=boundary,
            history_kind=history_kind,
            availability_convention=convention,
            calendar_id=calendar_id,
            first_required_session=sessions[0],
            last_required_session=sessions[-1],
            requested_return_count=requested_return_count,
            required_mark_count=mark_count,
            required_sessions=sessions,
            start_at=_request_start(sessions[0]),
            end_at=_request_end(sessions[-1]),
            local_maximum_days=local_limit,
            advertised_maximum_days=advertised,
            effective_maximum_days=effective,
            server_limit_negotiated=advertised is not None,
            chunks=chunks,
        )

    def latest_available_session(
        self,
        *,
        as_of: datetime,
        calendar_id: str,
        availability_convention: AvailabilityConvention,
    ) -> date:
        boundary = _aware_utc(as_of)
        row = MARKET_CALENDARS.get(calendar_id)
        if row is None:
            raise HistoryPlanningError(
                "unknown_exchange_calendar", f"unknown market calendar: {calendar_id}"
            )
        local_day = boundary.astimezone(ZoneInfo(str(row["timezone"]))).date()
        candidates = exchange_sessions(
            calendar_id,
            max(SUPPORTED_START, local_day - timedelta(days=14)),
            local_day,
        )
        lag = max(
            0.0,
            float(self.context.get("risk.history.daily_availability_lag_seconds", 0)),
        )
        # Select the latest close observable at the requested as-of timestamp.
        for session in reversed(candidates):
            available_at = session_available_at(
                session,
                calendar_id=calendar_id,
                convention=availability_convention,
            ) + timedelta(seconds=lag)
            if available_at <= boundary:
                return session
        raise HistoryPlanningError(
            "completed_session_unavailable",
            f"no completed {calendar_id} session is available at {boundary.isoformat()}",
        )


def session_available_at(
    session: date,
    *,
    calendar_id: str,
    convention: AvailabilityConvention,
) -> datetime:
    row = MARKET_CALENDARS.get(calendar_id)
    if row is None:
        raise HistoryPlanningError(
            "unknown_exchange_calendar", f"unknown market calendar: {calendar_id}"
        )
    if convention == "next_utc_midnight":
        return datetime.combine(session + timedelta(days=1), time.min, timezone.utc)
    close_text = str((row.get("early_closes") or {}).get(session.isoformat()) or row["close"])
    hour, minute = (int(value) for value in close_text.split(":", 1))
    return datetime.combine(
        session,
        time(hour, minute),
        ZoneInfo(str(row["timezone"])),
    ).astimezone(timezone.utc)


def clip_required_sessions(
    sessions: Sequence[date],
    *,
    listing_date: date | None = None,
    expiration_date: date | None = None,
) -> tuple[date, ...]:
    """Clip, but never broaden, one contract's consumer session set."""
    return tuple(
        session
        for session in sessions
        if (listing_date is None or session >= listing_date)
        and (expiration_date is None or session <= expiration_date)
    )


def chunk_session_window(
    sessions: Sequence[date], maximum_days: int
) -> tuple[HistoryRequestChunk, ...]:
    ordered = tuple(sorted(dict.fromkeys(sessions)))
    if not ordered:
        return ()
    if maximum_days < 1:
        raise HistoryPlanningError(
            "invalid_history_limit", "history request limit must be positive"
        )
    chunks: list[HistoryRequestChunk] = []
    index = 0
    # Pack whole sessions into bounded, non-overlapping request intervals.
    while index < len(ordered):
        end_index = index
        while end_index + 1 < len(ordered):
            proposed_end = _request_end(ordered[end_index + 1])
            if proposed_end - _request_start(ordered[index]) > timedelta(
                days=maximum_days
            ):
                break
            end_index += 1
        selected = ordered[index : end_index + 1]
        start_at = _request_start(selected[0])
        end_at = _request_end(selected[-1])
        if end_at - start_at > timedelta(days=maximum_days):
            raise HistoryPlanningError(
                "history_limit_below_single_session_span",
                "effective history limit cannot represent one daily session",
            )
        chunks.append(
            HistoryRequestChunk(
                start_at=start_at,
                end_at=end_at,
                sessions=selected,
            )
        )
        index = end_index + 1
    return tuple(chunks)


_CAPABILITY_LOCK = RLock()


def advertised_option_history_limit(reference_client: Any) -> int | None:
    """Read the cached RefData capability when the client exposes it safely."""
    method = getattr(reference_client, "option_history_maximum_days", None)
    if callable(method):
        try:
            value = method()
            return max(1, int(value)) if value is not None else None
        except Exception:
            return None
    method = getattr(reference_client, "capabilities", None)
    if not callable(method):
        return None
    try:
        with _CAPABILITY_LOCK:
            payload = method()
        supported = payload.get("supported") if isinstance(payload, Mapping) else None
        value = (
            supported.get("option_market_history_max_days")
            if isinstance(supported, Mapping)
            else None
        )
        return max(1, int(value)) if value is not None else None
    except Exception:
        return None


def _request_start(session: date) -> datetime:
    return datetime.combine(session, time.min, timezone.utc)


def _request_end(session: date) -> datetime:
    return datetime.combine(session + timedelta(days=1), time.min, timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise HistoryPlanningError(
            "naive_analysis_boundary", "as_of must be timezone aware"
        )
    return value.astimezone(timezone.utc)


__all__ = [
    "AvailabilityConvention",
    "HistoryKind",
    "HistoryPlanningError",
    "HistoryRequestChunk",
    "HistoryWindowPlan",
    "HistoryWindowPlanner",
    "advertised_option_history_limit",
    "chunk_session_window",
    "clip_required_sessions",
    "session_available_at",
]
