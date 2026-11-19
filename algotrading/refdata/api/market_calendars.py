"""Versioned, bounded exchange and futures-product session facts."""

from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
from typing import Any


CALENDAR_VERSION = "refdata_exchange_calendars_product_2018_2035_v3"
SUPPORTED_START = date(2018, 1, 1)
SUPPORTED_END = date(2035, 12, 31)


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month, monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    weekday_correction = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * weekday_correction) // 451
    month = (h + weekday_correction - 7 * m + 114) // 31
    return date(
        year,
        month,
        (h + weekday_correction - 7 * m + 114) % 31 + 1,
    )


def _xnys_holidays(year: int) -> set[date]:
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))
    # One-off national days of mourning when NYSE remained closed.
    if year == 2018:
        holidays.add(date(2018, 12, 5))
    if year == 2025:
        holidays.add(date(2025, 1, 9))
    return holidays


def _xnys_early_closes(year: int, holidays: set[date]) -> dict[date, str]:
    candidates = {
        _nth_weekday(year, 11, 3, 4) + timedelta(days=1),
        date(year, 12, 24),
        date(year, 7, 3),
    }
    return {
        day: "13:00"
        for day in candidates
        if day.weekday() < 5 and day not in holidays
    }


def _cme_full_holidays(year: int) -> set[date]:
    return {
        _observed(date(year, 1, 1)),
        _easter_sunday(year) - timedelta(days=2),
        _observed(date(year, 12, 25)),
    }


def _cme_early_closes(year: int, close: str) -> dict[date, str]:
    days = {
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _nth_weekday(year, 11, 3, 4) + timedelta(days=1),
        date(year, 12, 24),
    }
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))
    return {day: close for day in days if day.weekday() < 5}


def _facts(
    *, timezone: str, open_time: str, close_time: str,
    holidays: set[date], early_closes: dict[date, str],
    overnight: bool = False,
    maintenance_break: tuple[str, str] | None = None,
    product_roots: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "timezone": timezone, "open": open_time, "close": close_time,
        "overnight": overnight,
        "maintenance_break": list(maintenance_break) if maintenance_break else None,
        "product_roots": list(product_roots),
        "holidays": sorted(day.isoformat() for day in holidays),
        "early_closes": {
            day.isoformat(): value for day, value in sorted(early_closes.items())
        },
        "supported_start": SUPPORTED_START.isoformat(),
        "supported_end": SUPPORTED_END.isoformat(),
        "source_version": CALENDAR_VERSION,
        "fallback_policy": "bounded_rule_generated_exchange_facts",
    }


_YEARS = range(SUPPORTED_START.year, SUPPORTED_END.year + 1)
_XNYS_HOLIDAYS = set().union(*(_xnys_holidays(year) for year in _YEARS))
_XNYS_EARLY = {
    day: close
    for year in _YEARS
    for day, close in _xnys_early_closes(year, _XNYS_HOLIDAYS).items()
}
_CME_HOLIDAYS = set().union(*(_cme_full_holidays(year) for year in _YEARS))


def _cme_profile(close: str, roots: tuple[str, ...]) -> dict[str, Any]:
    early = {
        day: value
        for year in _YEARS
        for day, value in _cme_early_closes(year, close).items()
    }
    return _facts(
        timezone="America/Chicago", open_time="17:00", close_time="16:00",
        holidays=_CME_HOLIDAYS, early_closes=early, overnight=True,
        maintenance_break=("16:00", "17:00"), product_roots=roots,
    )


def _cfe_profile(roots: tuple[str, ...]) -> dict[str, Any]:
    # CFE has no session on the U.S. market holidays on which CME merely has
    # an early close. Using the CME holiday set here creates false VX gaps.
    early = {
        day: value
        for year in _YEARS
        for day, value in _cme_early_closes(year, "12:15").items()
        if day not in _XNYS_HOLIDAYS
    }
    return _facts(
        timezone="America/Chicago", open_time="17:00", close_time="15:00",
        holidays=_XNYS_HOLIDAYS, early_closes=early, overnight=True,
        maintenance_break=("15:00", "17:00"), product_roots=roots,
    )


MARKET_CALENDARS: dict[str, dict[str, Any]] = {
    "XNYS": _facts(
        timezone="America/New_York", open_time="09:30", close_time="16:00",
        holidays=_XNYS_HOLIDAYS, early_closes=_XNYS_EARLY,
    ),
    "CME_EQUITY_INDEX": _cme_profile("12:00", ("ES", "MES", "NQ", "MNQ", "RTY", "M2K", "YM")),
    "CME_RATES": _cme_profile("12:00", ("ZT", "ZF", "ZN", "TN", "ZB", "UB", "ZQ", "SR1", "SR3")),
    "CME_ENERGY": _cme_profile("12:30", ("CL", "MCL")),
    "CME_METALS": _cme_profile("12:30", ("GC", "MGC", "HG", "SI", "SIL")),
    "CME_FX": _cme_profile("12:00", ("6E", "M6E")),
    "CFE_VOLATILITY": _cfe_profile(("VX",)),
}
MARKET_CALENDARS["CME_GLOBEX"] = dict(MARKET_CALENDARS["CME_EQUITY_INDEX"])


def calendar_for_futures_root(root: str | None) -> str:
    normalized = str(root or "").strip().upper().removeprefix("/")
    # Use product-specific sessions before falling back to the generic Globex calendar.
    for calendar_id, row in MARKET_CALENDARS.items():
        if normalized in row.get("product_roots", ()):
            return calendar_id
    return "CME_GLOBEX"


def exchange_sessions(calendar_id: str, start: date, end: date) -> list[date]:
    row = MARKET_CALENDARS.get(calendar_id)
    if row is None:
        raise ValueError(f"unknown market calendar: {calendar_id}")
    # Reject dates outside the versioned calendar facts instead of extrapolating holidays.
    if start < SUPPORTED_START or end > SUPPORTED_END:
        raise ValueError(
            f"market calendar {calendar_id} supports {SUPPORTED_START} through {SUPPORTED_END}"
        )
    holidays = set(row.get("holidays") or ())
    sessions: list[date] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5 and cursor.isoformat() not in holidays:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    return sessions


__all__ = [
    "CALENDAR_VERSION", "MARKET_CALENDARS", "SUPPORTED_END", "SUPPORTED_START",
    "calendar_for_futures_root", "exchange_sessions",
]
