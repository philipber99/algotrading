"""Trading-session normalization and no-look-ahead snapshot sampling."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo


UTC = timezone.utc


def _aware_utc(value: Any, *, local_timezone: str) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        value = datetime.fromisoformat(str(value))
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo(local_timezone))
    return value.astimezone(UTC)


@dataclass(frozen=True)
class TradingSession:
    session_date: date
    open_utc: datetime
    close_utc: datetime
    is_early_close: bool

    @property
    def duration_minutes(self) -> int:
        return max(0, int((self.close_utc - self.open_utc).total_seconds() // 60))

    def sample_targets(self, interval_minutes: int) -> tuple[tuple[int, datetime], ...]:
        # Start at the first completed interval after the open. This excludes
        # the often-empty opening instant and is consistent with the design
        # examples where the first chart bin is session minute 5.
        return tuple(
            (
                minute,
                self.open_utc + timedelta(minutes=minute),
            )
            for minute in range(
                interval_minutes,
                self.duration_minutes + 1,
                interval_minutes,
            )
        )


def sessions_from_dataframe(
    dataframe: Any,
    *,
    local_timezone: str,
) -> tuple[TradingSession, ...]:
    if dataframe is None or len(dataframe) == 0:
        return ()
    rows = dataframe.to_dict(orient="records")
    sessions: list[TradingSession] = []
    for row in rows:
        raw_date = row.get("date")
        if isinstance(raw_date, datetime):
            raw_date = raw_date.date()
        elif hasattr(raw_date, "date") and not isinstance(raw_date, date):
            raw_date = raw_date.date()
        session_date = (
            raw_date
            if isinstance(raw_date, date)
            else date.fromisoformat(str(raw_date)[:10])
        )
        open_utc = _aware_utc(row["open"], local_timezone=local_timezone)
        close_utc = _aware_utc(row["close"], local_timezone=local_timezone)
        early = bool(
            row.get("half_day", row.get("is_early_close", False))
            or (close_utc - open_utc) < timedelta(hours=6)
        )
        if close_utc <= open_utc:
            continue
        sessions.append(TradingSession(session_date, open_utc, close_utc, early))
    return tuple(sorted(sessions, key=lambda item: item.session_date))


@dataclass(frozen=True)
class SurfaceStateSnapshot:
    snapshot_id: str
    underlying_symbol: str
    source_collected_at: datetime
    surface_definition_id: int
    calculation_definition_id: int
    source_surface_run_id: str | None
    status: str
    successful_expirations: int = 0


@dataclass(frozen=True)
class SampledSnapshot:
    state: SurfaceStateSnapshot
    session: TradingSession
    session_minute: int
    target_utc: datetime
    age_seconds: float


def sample_surface_states(
    states: Iterable[SurfaceStateSnapshot],
    sessions: Sequence[TradingSession],
    *,
    interval_minutes: int,
    max_age_seconds: int,
) -> tuple[SampledSnapshot, ...]:
    """Choose the latest source snapshot at or before each target.

    Sampling is performed independently by symbol and session. A snapshot
    after the target can never satisfy a bin.
    """

    sessions_by_date = {item.session_date: item for item in sessions}
    grouped: dict[tuple[str, date], list[SurfaceStateSnapshot]] = {}
    for state in states:
        timestamp = state.source_collected_at
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
            state = SurfaceStateSnapshot(
                state.snapshot_id,
                state.underlying_symbol,
                timestamp,
                state.surface_definition_id,
                state.calculation_definition_id,
                state.source_surface_run_id,
                state.status,
                state.successful_expirations,
            )
        for session in sessions:
            if session.open_utc <= timestamp <= session.close_utc:
                grouped.setdefault(
                    (state.underlying_symbol, session.session_date), []
                ).append(state)
                break

    sampled: list[SampledSnapshot] = []
    for (symbol, session_date), values in grouped.items():
        session = sessions_by_date[session_date]
        ordered = sorted(values, key=lambda item: item.source_collected_at)
        timestamps = [item.source_collected_at for item in ordered]
        for minute, target in session.sample_targets(interval_minutes):
            index = bisect.bisect_right(timestamps, target) - 1
            if index < 0:
                continue
            selected = ordered[index]
            age = (target - selected.source_collected_at).total_seconds()
            if age < 0 or age > max_age_seconds:
                continue
            sampled.append(
                SampledSnapshot(
                    state=selected,
                    session=session,
                    session_minute=minute,
                    target_utc=target,
                    age_seconds=age,
                )
            )
    return tuple(
        sorted(
            sampled,
            key=lambda item: (
                item.state.underlying_symbol,
                item.session.session_date,
                item.session_minute,
            ),
        )
    )
