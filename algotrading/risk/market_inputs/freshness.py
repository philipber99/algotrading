"""Shared session-aware quote freshness policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import math
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from algotrading.refdata.api.market_calendars import (
    CALENDAR_VERSION,
    MARKET_CALENDARS as DEFAULT_MARKET_CALENDARS,
)


@dataclass(frozen=True)
class QuoteFreshness:
    valid_for_risk: bool
    state: str
    source_at: datetime | None
    age_seconds: float | None
    market_session_state: str
    session_date: date | None
    reason: str
    calendar_id: str
    calendar_version: str
    session_close_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid_for_risk": self.valid_for_risk,
            "state": self.state,
            "source_at": self.source_at.isoformat() if self.source_at else None,
            "age_seconds": self.age_seconds,
            "market_session_state": self.market_session_state,
            "session_date": self.session_date.isoformat() if self.session_date else None,
            "reason": self.reason,
            "calendar_id": self.calendar_id,
            "calendar_version": self.calendar_version,
            "session_close_at": self.session_close_at.isoformat() if self.session_close_at else None,
        }


def evaluate_quote_freshness(
    quote_at: datetime | None,
    *,
    as_of: datetime,
    context: Mapping[str, Any],
    exchange_timezone: str = "America/New_York",
    source_kind: str = "intraday_quote",
    calendar_id: str | None = None,
    option_delta: float | None = None,
) -> QuoteFreshness:
    """Classify a quote without treating overnight age as intraday staleness."""

    now = _aware(as_of)
    selected_id, calendar = _calendar(
        context, calendar_id=calendar_id, exchange_timezone=exchange_timezone
    )
    calendar_version = str(
        context.get("risk.freshness.calendar_version", CALENDAR_VERSION)
    )
    timezone_name = str(calendar.get("timezone") or exchange_timezone)
    local_date = now.astimezone(ZoneInfo(timezone_name)).date()
    supported_start = _date(calendar.get("supported_start"))
    supported_end = _date(calendar.get("supported_end"))
    if (
        supported_start is not None
        and supported_end is not None
        and not supported_start <= local_date <= supported_end
    ):
        raise ValueError(
            f"freshness calendar {selected_id} does not support {local_date}"
        )
    if quote_at is None:
        return QuoteFreshness(
            False, "undated", None, None, _session_state(now, timezone_name, context, calendar)[0],
            None, "quote_timestamp_missing", selected_id, calendar_version,
        )
    observed = _aware(quote_at)
    age = (now - observed).total_seconds()
    # A coherent snapshot is assembled from several bounded upstream calls.
    # Quotes returned later in that assembly can legitimately be a few seconds
    # newer than its initial broker anchor, and provider clocks can differ by a
    # similarly small amount. Reject meaningful future timestamps while
    # allowing that acquisition/clock-skew window.
    future_tolerance = max(
        0.0,
        float(context.get("risk.freshness.future_timestamp_tolerance_seconds", 5)),
    )
    if age < -future_tolerance:
        return QuoteFreshness(
            False, "invalid", observed, age,
            _session_state(now, timezone_name, context, calendar)[0], None,
            "quote_timestamp_in_future", selected_id, calendar_version,
        )
    session_state, current_session = _session_state(now, timezone_name, context, calendar)
    observed_local = observed.astimezone(ZoneInfo(timezone_name))
    observed_session = _session_label(observed_local, calendar)
    # Mutual fund NAV freshness is measured in completed sessions, not seconds.
    if source_kind == "mutual_fund_nav":
        completed = _latest_completed_session(
            now, timezone_name, context, calendar
        )
        maximum_lag = max(
            0,
            int(context.get("risk.freshness.mutual_fund_nav_max_session_lag", 1)),
        )
        session_lag = _completed_session_lag(
            observed_session, completed, calendar
        )
        valid_nav = session_lag is not None and session_lag <= maximum_lag
        return QuoteFreshness(
            valid_nav,
            "valid_latest_available_nav" if valid_nav else "stale",
            observed,
            max(0.0, age),
            session_state,
            observed_session,
            (
                "latest_available_mutual_fund_nav"
                if valid_nav
                else "mutual_fund_nav_too_old"
            ),
            selected_id,
            calendar_version,
            _session_close(
                observed_session, timezone_name, context, calendar
            ),
        )
    maximum_age = float(context.get("risk.freshness.intraday_max_age_seconds", 900))
    if option_delta is not None:
        delta = abs(float(option_delta))
        low_delta_threshold = float(
            context.get("risk.freshness.low_delta_option_threshold", 0.10)
        )
        if math.isfinite(delta) and delta <= low_delta_threshold:
            maximum_age = float(
                context.get(
                    "risk.freshness.low_delta_option_intraday_max_age_seconds",
                    4 * 60 * 60,
                )
            )
    # An open-session quote must be recent and belong to that same session.
    if session_state == "open":
        valid = age <= maximum_age and observed_session == current_session
        return QuoteFreshness(
            valid,
            "fresh_intraday" if valid else "stale",
            observed,
            max(0.0, age),
            session_state,
            observed_session,
            "within_intraday_age_policy" if valid else "intraday_quote_too_old",
            selected_id,
            calendar_version,
            _session_close(observed_session, timezone_name, context, calendar),
        )
    permit_eod = _as_bool(
        context.get("risk.freshness.completed_session_eod_allowed", True)
    )
    completed = _latest_completed_session(now, timezone_name, context, calendar)
    eod_source = source_kind in {
        "eod", "daily_close", "completed_session_mark", "exchange_quote",
    }
    # Providers often label their latest quote rather than its storage mode.
    # A timestamp at/after the configured close is sufficient EOD provenance.
    close_local = _session_close(observed_session, timezone_name, context, calendar)
    eod_timestamp = observed_local >= close_local
    # Closed-market marks must refer to the latest completed exchange session.
    valid_eod = (
        permit_eod
        and observed_session == completed
        and (eod_source or eod_timestamp)
    )
    return QuoteFreshness(
        valid_eod,
        "valid_completed_session_eod" if valid_eod else "stale",
        observed,
        max(0.0, age),
        session_state,
        observed_session,
        "latest_completed_session_eod" if valid_eod else "not_latest_completed_session_eod",
        selected_id,
        calendar_version,
        close_local,
    )


def _session_state(
    value: datetime,
    timezone_name: str,
    context: Mapping[str, Any],
    calendar: Mapping[str, Any],
) -> tuple[str, date]:
    local = value.astimezone(ZoneInfo(timezone_name))
    session = _session_label(local, calendar)
    if not _is_session(session, calendar):
        return "closed_calendar", session
    open_time = _configured_clock(
        calendar.get("open") or context.get("risk.freshness.regular_session_open_local_time", "09:30"),
        fallback=time(9, 30),
    )
    close_at = _session_close(session, timezone_name, context, calendar)
    open_day = session - timedelta(days=1) if calendar.get("overnight") else session
    open_at = datetime.combine(open_day, open_time, ZoneInfo(timezone_name))
    if open_at <= local < close_at:
        return "open", session
    return ("pre_open" if local < open_at else "closed"), session


def _latest_completed_session(
    value: datetime,
    timezone_name: str,
    context: Mapping[str, Any],
    calendar: Mapping[str, Any],
) -> date:
    local = value.astimezone(ZoneInfo(timezone_name))
    candidate = _session_label(local, calendar)
    if _is_session(candidate, calendar) and local >= _session_close(
        candidate, timezone_name, context, calendar
    ):
        return candidate
    candidate -= timedelta(days=1)
    while not _is_session(candidate, calendar):
        candidate -= timedelta(days=1)
    return candidate


def _completed_session_lag(
    observed: date,
    completed: date,
    calendar: Mapping[str, Any],
) -> int | None:
    """Count completed exchange sessions between an observation and latest close."""

    if observed > completed or not _is_session(observed, calendar):
        return None
    lag = 0
    cursor = completed
    while cursor > observed:
        cursor -= timedelta(days=1)
        if _is_session(cursor, calendar):
            lag += 1
    return lag if cursor == observed else None


def _session_close(
    session: date,
    timezone_name: str,
    context: Mapping[str, Any],
    calendar: Mapping[str, Any],
) -> datetime:
    early = (calendar.get("early_closes") or {}).get(session.isoformat())
    close_time = _configured_clock(
        early or calendar.get("close") or context.get("risk.freshness.regular_session_close_local_time", "16:00"),
        fallback=time(16, 0),
    )
    return datetime.combine(session, close_time, ZoneInfo(timezone_name))


def _calendar(
    context: Mapping[str, Any], *, calendar_id: str | None, exchange_timezone: str
) -> tuple[str, Mapping[str, Any]]:
    configured = context.get("risk.freshness.calendars", DEFAULT_MARKET_CALENDARS)
    calendars = configured if isinstance(configured, Mapping) else DEFAULT_MARKET_CALENDARS
    selected = calendar_id or (
        "CME_GLOBEX" if exchange_timezone == "America/Chicago" else "XNYS"
    )
    row = calendars.get(selected)
    if not isinstance(row, Mapping):
        raise ValueError(f"freshness calendar unavailable: {selected}")
    return selected, row


def _session_label(local: datetime, calendar: Mapping[str, Any]) -> date:
    # Overnight trading after the open belongs to the next session date.
    if calendar.get("overnight"):
        open_time = _configured_clock(calendar.get("open", "17:00"), fallback=time(17))
        if local.timetz().replace(tzinfo=None) >= open_time:
            return local.date() + timedelta(days=1)
    return local.date()


def _is_session(session: date, calendar: Mapping[str, Any]) -> bool:
    holidays = {str(item) for item in calendar.get("holidays") or ()}
    return session.weekday() < 5 and session.isoformat() not in holidays


def _configured_clock(value: Any, *, fallback: time) -> time:
    try:
        hour, minute = (int(item) for item in str(value).split(":", 1))
        return time(hour, minute)
    except (TypeError, ValueError):
        return fallback


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("freshness timestamps must be timezone aware")
    return value.astimezone(timezone.utc)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "CALENDAR_VERSION",
    "DEFAULT_MARKET_CALENDARS",
    "QuoteFreshness",
    "evaluate_quote_freshness",
]
