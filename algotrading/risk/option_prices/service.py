"""Database-first option history reserved for Risk-GUI workflows."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Mapping, Sequence

from algotrading.refdata.api.live_models import (
    LiveResponseMeta,
    OptionMarketHistoryItem,
    OptionMarketHistoryRequest,
    OptionMarketHistoryResponse,
)
from algotrading.refdata.api.market_calendars import exchange_sessions
from algotrading.risk.market_inputs.client import (
    ReferenceDataClient,
    ReferenceDataClientError,
)
from algotrading.risk.history.planner import (
    advertised_option_history_limit,
    chunk_session_window,
)

from .repository import OptionPriceRepository, OptionPriceRepositoryUnavailable


class RiskOptionHistoryService:
    """Merge durable captured marks with remote RefData gap fills.

    Captured SQL marks always win for an exact OCC contract/session. RefData is
    contacted only for exact exchange-session ranges whose authoritative mark
    is absent. Missing Greeks never cause a repeated mark request.
    """

    def __init__(
        self,
        repository: OptionPriceRepository | None,
        reference_client: ReferenceDataClient,
        *,
        now_fn=lambda: datetime.now(timezone.utc),
    ) -> None:
        self.repository = repository
        self.reference_client = reference_client
        self._now = now_fn

    def query(
        self,
        payload: OptionMarketHistoryRequest | dict[str, Any],
        *,
        required_sessions_by_symbol: Mapping[str, Sequence[date]] | None = None,
        causal_as_of: datetime | None = None,
    ) -> OptionMarketHistoryResponse:
        request = (
            payload
            if isinstance(payload, OptionMarketHistoryRequest)
            else OptionMarketHistoryRequest.model_validate(payload)
        )
        start_date = request.start_at.date()
        end_date = (request.end_at - timedelta(microseconds=1)).date()
        expected_by_symbol = {
            symbol: tuple(
                sorted(
                    dict.fromkeys(
                        (required_sessions_by_symbol or {}).get(
                            symbol,
                            exchange_sessions("XNYS", start_date, end_date),
                        )
                    )
                )
            )
            for symbol in request.symbols
        }
        warnings: list[str] = []
        database_rows: list[dict[str, Any]] = []
        if self.repository is None:
            warnings.append("risk_option_price_repository_unavailable")
        else:
            # Group exact-contract SQL reads by the required session window.
            sql_windows: dict[tuple[date, date], list[str]] = {}
            for symbol, sessions in expected_by_symbol.items():
                if sessions:
                    sql_windows.setdefault((sessions[0], sessions[-1]), []).append(symbol)
            for (window_start, window_end), symbols in sorted(sql_windows.items()):
                try:
                    database_rows.extend(
                        self.repository.query_history(
                            sorted(symbols),
                            start_date=window_start,
                            end_date=window_end,
                        )
                    )
                except OptionPriceRepositoryUnavailable:
                    warnings.append("risk_option_price_repository_unavailable")

        causal_boundary = (
            _aware_utc(causal_as_of) if causal_as_of is not None else request.end_at
        )
        database_rows = [
            row
            for row in database_rows
            if (observed := _datetime_value(row.get("observed_at") or row.get("at")))
            is None
            or observed <= causal_boundary
        ]
        database_by_key = {
            key: item
            for item in database_rows
            if (key := _history_key(item)) is not None
        }
        # Request remote history only for gaps in durable exact-contract marks.
        missing_ranges: dict[str, tuple[tuple[date, ...], ...]] = {}
        for symbol, sessions in expected_by_symbol.items():
            missing_sessions = {
                session
                for session in sessions
                if (row := database_by_key.get((symbol, session))) is None
                or row.get("mark") is None
            }
            missing_ranges[symbol] = _coalesced_missing_sessions(
                sessions, missing_sessions
            )

        local_maximum = max(
            1,
            int(
                getattr(self.reference_client, "context", {}).get(
                    "risk.option_prices.max_history_days", 376
                )
            ),
        )
        advertised_maximum = advertised_option_history_limit(self.reference_client)
        # Honor the tighter local or advertised provider history limit.
        effective_maximum = min(local_maximum, advertised_maximum) if advertised_maximum else local_maximum
        grouped_requests: dict[
            tuple[datetime, datetime, tuple[date, ...]], list[str]
        ] = {}
        for symbol in request.symbols:
            for missing in missing_ranges[symbol]:
                for chunk in chunk_session_window(missing, effective_maximum):
                    key = (chunk.start_at, chunk.end_at, chunk.sessions)
                    grouped_requests.setdefault(key, []).append(symbol)

        requested_ranges: list[dict[str, Any]] = []
        failed_ranges: list[dict[str, Any]] = []
        remote_rows: list[dict[str, Any]] = []
        for (chunk_start, chunk_end, chunk_sessions), symbols in sorted(
            grouped_requests.items(),
            key=lambda item: (item[0][0], item[0][1], tuple(sorted(item[1]))),
        ):
            ordered_symbols = sorted(symbols)
            diagnostic = {
                "symbols": ordered_symbols,
                "start_at": chunk_start.isoformat(),
                "end_at": chunk_end.isoformat(),
                "sessions": [item.isoformat() for item in chunk_sessions],
                "span_days": (chunk_end - chunk_start).total_seconds() / 86400.0,
            }
            requested_ranges.append(diagnostic)
            try:
                remote = self.reference_client.option_market_history(
                    {
                        "symbols": ordered_symbols,
                        "start_at": chunk_start.isoformat(),
                        "end_at": chunk_end.isoformat(),
                        "interval": request.interval,
                        "include_greeks": request.include_greeks,
                    }
                )
            except ReferenceDataClientError as exc:
                failure = {**diagnostic, "error": exc.diagnostic()}
                failed_ranges.append(failure)
                warnings.append("remote_option_history_range_failed")
            else:
                wanted_sessions = set(chunk_sessions)
                wanted_symbols = set(ordered_symbols)
                remote_rows.extend(
                    dict(row)
                    for row in remote.get("data") or []
                    if str(row.get("canonical_occ_symbol") or "").upper()
                    in wanted_symbols
                    and _date_value(row.get("session_date")) in wanted_sessions
                    and _causally_available(row, causal_boundary)
                )
                meta = remote.get("meta") or {}
                warnings.extend(str(item) for item in meta.get("warnings") or [])

        merged: dict[tuple[str, date], dict[str, Any]] = {}
        for row in remote_rows:
            key = _history_key(row)
            if (
                key is not None
                and row.get("status") == "ok"
                and row.get("mark") is not None
            ):
                merged[key] = dict(row)
        # SQL is authoritative for any session captured while the position was
        # actually held, including contracts the remote provider now rejects.
        # Overlay authoritative SQL values while retaining remote fields that fill gaps.
        for row in database_rows:
            key = _history_key(row)
            if (
                key is not None
                and row.get("status") == "ok"
                and row.get("mark") is not None
            ):
                remote_row = merged.get(key, {})
                sql_row = dict(row)
                combined = {
                    **remote_row,
                    **{
                        name: value
                        for name, value in sql_row.items()
                        if value is not None
                    },
                }
                combined["source"] = str(sql_row.get("source") or "risk_mysql_option_prices")
                combined["warnings"] = list(
                    dict.fromkeys(
                        [
                            *list(remote_row.get("warnings") or ()),
                            *list(sql_row.get("warnings") or ()),
                            *( ["remote_fields_gap_filled_around_sql_authoritative_mark"] if remote_row else [] ),
                        ]
                    )
                )
                merged[key] = combined

        items = [
            OptionMarketHistoryItem.model_validate(row)
            for _, row in sorted(
                merged.items(), key=lambda item: (item[0][0], item[0][1])
            )
        ]
        # Report missing coverage against each contract's required session set.
        present = {
            (item.canonical_occ_symbol, item.session_date)
            for item in items
            if item.mark is not None
        }
        missing_symbols = [
            symbol
            for symbol in request.symbols
            if any(
                (symbol, session) not in present
                for session in expected_by_symbol[symbol]
            )
        ]
        if missing_symbols:
            warnings.append("historical_option_marks_incomplete")
        return OptionMarketHistoryResponse(
            data=items,
            missing_symbols=missing_symbols,
            meta=LiveResponseMeta(
                as_of=_aware_utc(self._now()),
                source="risk_option_history_hybrid",
                cache="miss" if requested_ranges else "hit",
                warnings=list(dict.fromkeys(warnings)),
            ),
            diagnostics={
                "calendar_id": "XNYS",
                "sql_mark_count_by_symbol": {
                    symbol: sum(
                        (symbol, session) in database_by_key
                        and database_by_key[(symbol, session)].get("mark") is not None
                        for session in expected_by_symbol[symbol]
                    )
                    for symbol in request.symbols
                },
                "remote_mark_count_by_symbol": {
                    symbol: sum(
                        _history_key(row) is not None
                        and _history_key(row)[0] == symbol
                        and row.get("mark") is not None
                        for row in remote_rows
                    )
                    for symbol in request.symbols
                },
                "required_sessions_by_symbol": {
                    symbol: [session.isoformat() for session in sessions]
                    for symbol, sessions in expected_by_symbol.items()
                },
                "requested_remote_ranges": requested_ranges,
                "failed_remote_ranges": failed_ranges,
                "local_maximum_days": local_maximum,
                "advertised_maximum_days": advertised_maximum,
                "effective_maximum_days": effective_maximum,
                "server_limit_negotiated": advertised_maximum is not None,
                "greek_gaps_trigger_remote_requests": False,
            },
        )


class HybridOptionHistoryClient:
    """ReferenceDataClient facade overriding only exact option history."""

    def __init__(
        self,
        reference_client: ReferenceDataClient,
        option_history: RiskOptionHistoryService,
    ) -> None:
        self._reference_client = reference_client
        self._option_history = option_history

    def option_market_history(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._option_history.query(payload).model_dump(mode="json")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reference_client, name)


def _history_key(row: dict[str, Any]) -> tuple[str, date] | None:
    symbol = str(row.get("canonical_occ_symbol") or "").strip().upper()
    session = _date_value(row.get("session_date"))
    return (symbol, session) if symbol and session is not None else None


def _date_value(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _datetime_value(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _causally_available(row: Mapping[str, Any], boundary: datetime) -> bool:
    observed = _datetime_value(row.get("observed_at") or row.get("at"))
    return observed is None or observed <= boundary


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _coalesced_missing_sessions(
    expected: Sequence[date], missing: set[date]
) -> tuple[tuple[date, ...], ...]:
    ranges: list[list[date]] = []
    active: list[date] = []
    for session in expected:
        if session in missing:
            active.append(session)
        elif active:
            ranges.append(active)
            active = []
    if active:
        ranges.append(active)
    return tuple(tuple(item) for item in ranges)


__all__ = ["HybridOptionHistoryClient", "RiskOptionHistoryService"]
