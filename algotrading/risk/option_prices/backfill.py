"""One-time historical option-position price backfill from Massive aggregates."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
import logging
import math
import os
import sys
import time as time_module
from typing import Any
from urllib.parse import quote, urljoin, urlparse
import uuid
from zoneinfo import ZoneInfo

import requests

from algotrading.refdata.api.option_symbols import OccSymbol, parse_occ_symbol
from algotrading.risk.api.bootstrap import build_context
from algotrading.risk.pnl.reconstruction import normalize_execution_legs
from algotrading.risk.transactions.models import TransactionModel
from algotrading.risk.transactions.repository import TransactionRepository

from .models import CapturedOptionPrice, CaptureSummary, OptionPosition
from .repository import OptionPriceRepository


LOGGER = logging.getLogger("algotrading.risk.option_prices.backfill")
EPSILON = 1e-9


class MassiveAggregateRequestError(RuntimeError):
    """An aggregate request failed with a provider-safe diagnostic."""


@dataclass(frozen=True)
class HoldingWindow:
    start_date: date
    end_date: date


@dataclass(frozen=True)
class ContractHoldingHistory:
    account_id: str
    occ: OccSymbol
    events: tuple[tuple[date, float], ...]
    windows: tuple[HoldingWindow, ...]

    def includes(self, session_date: date) -> bool:
        return any(
            window.start_date <= session_date <= window.end_date
            for window in self.windows
        )

    def quantity_at_close(self, session_date: date) -> float:
        return sum(
            delta for event_date, delta in self.events if event_date <= session_date
        )


@dataclass(frozen=True)
class BackfillResult:
    summary: CaptureSummary
    contracts_without_bars: tuple[str, ...]


def compile_holding_histories(
    account_id: str,
    transactions: Sequence[TransactionModel],
    *,
    through_date: date,
    market_timezone: ZoneInfo,
    from_date: date | None = None,
    symbols: Sequence[str] | None = None,
) -> list[ContractHoldingHistory]:
    """Reconstruct each exact option contract's inclusive holding windows."""
    executions, _ = normalize_execution_legs(account_id, transactions)
    trade_dates = {item.id: item.trade_date for item in transactions}
    selected = {
        parse_occ_symbol(item).canonical for item in symbols or ()
    }
    grouped: dict[str, list[tuple[date, float]]] = defaultdict(list)
    parsed: dict[str, OccSymbol] = {}
    for execution in executions:
        if execution.asset_type != "OPTION":
            continue
        try:
            occ = parse_occ_symbol(execution.symbol)
        except ValueError:
            continue
        if selected and occ.canonical not in selected:
            continue
        session_date = trade_dates.get(execution.repository_transaction_id) or (
            execution.timestamp.astimezone(market_timezone).date()
        )
        if session_date > through_date:
            continue
        parsed[occ.canonical] = occ
        grouped[occ.canonical].append((session_date, execution.signed_quantity))

    output: list[ContractHoldingHistory] = []
    for canonical in sorted(grouped):
        occ = parsed[canonical]
        events = tuple(sorted(grouped[canonical], key=lambda item: item[0]))
        # Reconstruct only actual holding periods and stop at contract expiration.
        windows = _holding_windows(
            events,
            through_date=min(through_date, occ.expiration),
            from_date=from_date,
        )
        if windows:
            output.append(
                ContractHoldingHistory(
                    account_id=account_id,
                    occ=occ,
                    events=events,
                    windows=windows,
                )
            )
    return output


def _holding_windows(
    events: Sequence[tuple[date, float]],
    *,
    through_date: date,
    from_date: date | None,
) -> tuple[HoldingWindow, ...]:
    balance = 0.0
    opened: date | None = None
    windows: list[HoldingWindow] = []
    for event_date, delta in events:
        if event_date > through_date:
            break
        before = balance
        balance += delta
        if abs(balance) <= EPSILON:
            balance = 0.0
        # Start and end holding windows when the balance enters or leaves zero.
        if abs(before) <= EPSILON and abs(balance) > EPSILON:
            opened = event_date
        elif abs(before) > EPSILON and abs(balance) <= EPSILON and opened is not None:
            windows.append(HoldingWindow(opened, event_date))
            opened = None
    if opened is not None and opened <= through_date:
        windows.append(HoldingWindow(opened, through_date))

    clipped: list[HoldingWindow] = []
    for window in windows:
        start = max(window.start_date, from_date) if from_date else window.start_date
        if start <= window.end_date:
            clipped.append(HoldingWindow(start, window.end_date))
    return tuple(clipped)


def _missing_weekday_windows(
    history: ContractHoldingHistory, existing_sessions: set[date]
) -> tuple[HoldingWindow, ...]:
    missing: list[HoldingWindow] = []
    for window in history.windows:
        session_date = window.start_date
        while session_date <= window.end_date:
            if session_date.weekday() < 5 and session_date not in existing_sessions:
                missing.append(HoldingWindow(session_date, session_date))
            session_date += timedelta(days=1)
    return _merge_holding_windows(missing)


def _merge_holding_windows(
    windows: Iterable[HoldingWindow],
) -> tuple[HoldingWindow, ...]:
    ordered = sorted(windows, key=lambda item: (item.start_date, item.end_date))
    if not ordered:
        return ()
    merged: list[HoldingWindow] = [ordered[0]]
    for window in ordered[1:]:
        current = merged[-1]
        if window.start_date <= _next_weekday(current.end_date):
            merged[-1] = HoldingWindow(
                current.start_date, max(current.end_date, window.end_date)
            )
        else:
            merged.append(window)
    return tuple(merged)


def _next_weekday(value: date) -> date:
    candidate = value + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _in_holding_windows(value: date, windows: Sequence[HoldingWindow]) -> bool:
    return any(window.start_date <= value <= window.end_date for window in windows)


class MassiveAggregateClient:
    """Small authenticated client for Massive's paginated daily aggregates."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.massive.com",
        timeout_seconds: float = 25.0,
        max_attempts: int = 5,
        max_requests_per_minute: int = 5,
        session: requests.Session | None = None,
        sleep_fn: Callable[[float], None] = time_module.sleep,
        monotonic_fn: Callable[[], float] = time_module.monotonic,
    ) -> None:
        if not str(api_key).strip():
            raise ValueError("Massive API key is required")
        self.base_url = str(base_url).rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Massive base URL must be an absolute HTTP(S) URL")
        self._origin = (parsed.scheme, parsed.netloc)
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = max(1, int(max_attempts))
        self.rate_limiter = _RequestRateLimiter(
            max_requests_per_minute,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
        )
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "RiskOptionPriceBackfill/1.0",
            }
        )
        self._sleep = sleep_fn

    def daily_bars(
        self,
        options_ticker: str,
        *,
        start_date: date,
        end_date: date,
        adjusted: bool = False,
    ) -> list[Mapping[str, Any]]:
        if end_date < start_date:
            return []
        ticker = str(options_ticker).strip().upper()
        url = (
            f"{self.base_url}/v2/aggs/ticker/{quote(ticker, safe=':')}/range/1/day/"
            f"{start_date.isoformat()}/{end_date.isoformat()}"
        )
        params: Mapping[str, Any] | None = {
            "adjusted": str(bool(adjusted)).lower(),
            "sort": "asc",
            "limit": 50_000,
        }
        output: list[Mapping[str, Any]] = []
        while url:
            payload = self._request_json(url, params=params)
            rows = payload.get("results") or []
            if not isinstance(rows, list) or any(
                not isinstance(item, Mapping) for item in rows
            ):
                raise ValueError("Massive aggregate response contains invalid results")
            output.extend(rows)
            next_url = payload.get("next_url")
            if not next_url:
                break
            url = urljoin(self.base_url + "/", str(next_url))
            parsed = urlparse(url)
            # Keep authenticated pagination on the configured provider origin.
            if (parsed.scheme, parsed.netloc) != self._origin:
                raise ValueError("Massive pagination URL changed origin")
            params = None
        return output

    def _request_json(
        self, url: str, *, params: Mapping[str, Any] | None
    ) -> Mapping[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            response = None
            try:
                self.rate_limiter.acquire()
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise ValueError("Massive aggregate response must be an object")
                status = str(payload.get("status") or "OK").upper()
                if status not in {"OK", "DELAYED"}:
                    raise ValueError(f"Massive aggregate request returned status {status}")
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                status_code = getattr(response, "status_code", None)
                if (
                    isinstance(exc, requests.HTTPError)
                    and isinstance(status_code, int)
                    and 400 <= status_code < 500
                    and status_code != 429
                ):
                    raise MassiveAggregateRequestError(
                        _http_error_detail(response)
                    ) from exc
                if attempt >= self.max_attempts:
                    break
                delay = _retry_delay(response, attempt)
                LOGGER.warning(
                    "Massive aggregate request failed attempt=%d/%d error=%s%s; retrying in %.2fs",
                    attempt,
                    self.max_attempts,
                    type(exc).__name__,
                    _http_error_log_suffix(response),
                    delay,
                )
                self._sleep(delay)
        assert last_error is not None
        if isinstance(last_error, requests.HTTPError):
            raise MassiveAggregateRequestError(_http_error_detail(response)) from last_error
        raise MassiveAggregateRequestError(
            f"Massive aggregate request failed: {type(last_error).__name__}"
        ) from last_error


class _RequestRateLimiter:
    """A rolling one-minute request limiter shared by retries and pagination."""

    def __init__(
        self,
        maximum_requests: int,
        *,
        sleep_fn: Callable[[float], None],
        monotonic_fn: Callable[[], float],
    ) -> None:
        self.maximum_requests = int(maximum_requests)
        if self.maximum_requests < 1:
            raise ValueError("max_requests_per_minute must be at least 1")
        self._timestamps: deque[float] = deque()
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn

    def acquire(self) -> None:
        # Enforce the provider request allowance over a rolling sixty-second window.
        while True:
            now = self._monotonic()
            cutoff = now - 60.0
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) < self.maximum_requests:
                self._timestamps.append(now)
                return
            wait_seconds = max(0.0, self._timestamps[0] + 60.0 - now)
            LOGGER.warning(
                "Massive request limit reached (%d requests per 60 seconds); waiting %.2fs",
                self.maximum_requests,
                wait_seconds,
            )
            self._sleep(wait_seconds)


class HistoricalOptionPriceBackfill:
    def __init__(
        self,
        transaction_repository: TransactionRepository,
        option_price_repository: OptionPriceRepository,
        provider: MassiveAggregateClient,
        context: Mapping[str, Any],
        *,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.transaction_repository = transaction_repository
        self.option_price_repository = option_price_repository
        self.provider = provider
        self.context = context
        self._now = now_fn

    def run(
        self,
        *,
        account_ids: Sequence[str] | None = None,
        from_date: date | None = None,
        through_date: date,
        symbols: Sequence[str] | None = None,
        adjusted: bool = False,
        dry_run: bool = False,
        max_transactions: int = 1_000_000,
    ) -> BackfillResult:
        captured_at = _aware_utc(self._now())
        accounts = list(
            dict.fromkeys(
                str(item).strip()
                for item in (
                    account_ids or self.transaction_repository.list_trade_accounts()
                )
                if str(item).strip()
            )
        )
        if not accounts:
            raise ValueError("no accounts with stored trade transactions were found")
        market_timezone = ZoneInfo(
            str(self.context.get("risk.option_prices.market_timezone", "America/New_York"))
        )
        through_exclusive = datetime.combine(
            through_date + timedelta(days=1), time.min, tzinfo=market_timezone
        ).astimezone(timezone.utc)
        histories: list[ContractHoldingHistory] = []
        for account_id in accounts:
            transactions = self.transaction_repository.query_all_trade_history(
                account_id,
                through=through_exclusive,
                max_rows=max_transactions,
            )
            histories.extend(
                compile_holding_histories(
                    account_id,
                    transactions,
                    through_date=through_date,
                    market_timezone=market_timezone,
                    from_date=from_date,
                    symbols=symbols,
                )
            )
        if not histories:
            raise ValueError("no option holding periods were found in stored transactions")

        by_symbol: dict[str, list[ContractHoldingHistory]] = defaultdict(list)
        for history in histories:
            by_symbol[history.occ.canonical].append(history)

        summary = CaptureSummary(
            run_id=str(uuid.uuid4()),
            captured_at=captured_at,
            account_count=len(accounts),
            position_count=len(histories),
            quote_count=0,
            missing_count=0,
            status="running",
            provider_class=(
                f"{type(self.provider).__module__}."
                f"{type(self.provider).__qualname__}"
            ),
        )
        existing_by_history: dict[tuple[str, str], set[date]] = {}
        missing_by_history: dict[tuple[str, str], tuple[HoldingWindow, ...]] = {}
        for history in histories:
            key = (history.account_id, history.occ.canonical)
            first = min(window.start_date for window in history.windows)
            last = max(window.end_date for window in history.windows)
            existing = self.option_price_repository.existing_session_dates(
                history.account_id,
                history.occ.canonical,
                start_date=first,
                end_date=last,
            )
            existing_by_history[key] = existing
            missing_by_history[key] = _missing_weekday_windows(history, existing)

        price_count = 0
        without_bars: list[str] = []
        for canonical in sorted(by_symbol):
            symbol_histories = by_symbol[canonical]
            provider_ticker = f"O:{canonical}"
            # Merge missing holding windows across accounts to reuse provider history.
            request_windows = _merge_holding_windows(
                window
                for history in symbol_histories
                for window in missing_by_history[
                    (history.account_id, history.occ.canonical)
                ]
            )
            if not request_windows:
                LOGGER.info("Skipping %s; all weekday sessions are already stored", provider_ticker)
                continue
            converted = 0
            for window in request_windows:
                LOGGER.info(
                    "Fetching %s from %s through %s",
                    provider_ticker,
                    window.start_date,
                    window.end_date,
                )
                raw_bars = self.provider.daily_bars(
                    provider_ticker,
                    start_date=window.start_date,
                    end_date=window.end_date,
                    adjusted=adjusted,
                )
                batch: list[CapturedOptionPrice] = []
                for raw_bar in raw_bars:
                    session_date = _bar_session_date(raw_bar, market_timezone)
                    close = _finite(raw_bar.get("c"))
                    if session_date is None or close is None:
                        continue
                    for history in symbol_histories:
                        key = (history.account_id, history.occ.canonical)
                        if (
                            session_date in existing_by_history[key]
                            or not _in_holding_windows(
                                session_date, missing_by_history[key]
                            )
                        ):
                            continue
                        batch.append(
                            _captured_aggregate_price(
                                history,
                                raw_bar,
                                session_date=session_date,
                                captured_at=captured_at,
                                close=close,
                            )
                        )
                if batch:
                    converted += len(batch)
                    price_count += len(batch)
                    if not dry_run:
                        self.option_price_repository.save_backfill_batch(
                            replace(summary, quote_count=price_count),
                            batch,
                            final=False,
                        )
            if converted == 0 and not any(
                existing_by_history[(history.account_id, history.occ.canonical)]
                for history in symbol_histories
            ):
                without_bars.append(canonical)

        summary = replace(
            summary,
            quote_count=price_count,
            missing_count=len(without_bars),
            status="partial" if without_bars else "success",
        )
        if not dry_run:
            self.option_price_repository.save_backfill_batch(summary, (), final=True)
        return BackfillResult(summary, tuple(without_bars))


def _captured_aggregate_price(
    history: ContractHoldingHistory,
    raw_bar: Mapping[str, Any],
    *,
    session_date: date,
    captured_at: datetime,
    close: float,
) -> CapturedOptionPrice:
    occ = history.occ
    return CapturedOptionPrice(
        position=OptionPosition(
            account_id=history.account_id,
            canonical_occ_symbol=occ.canonical,
            provider_option_symbol=f"O:{occ.canonical}",
            underlying_symbol=occ.root,
            expiration_date=occ.expiration,
            put_call=occ.put_call,
            strike=float(occ.strike),
            signed_quantity=history.quantity_at_close(session_date),
        ),
        captured_at=captured_at,
        session_date=session_date,
        quote_time=None,
        trade_time=None,
        bid=None,
        ask=None,
        mark=close,
        last=close,
        open=_finite(raw_bar.get("o")),
        high=_finite(raw_bar.get("h")),
        low=_finite(raw_bar.get("l")),
        close=close,
        underlying_price=None,
        provider_iv=None,
        provider_delta=None,
        provider_gamma=None,
        provider_theta_per_day=None,
        provider_vega_per_1pct=None,
        provider_rho_per_1pct=None,
        open_interest=None,
        volume=_finite(raw_bar.get("v")),
        source="MassiveAggregates",
    )


def _bar_session_date(
    raw_bar: Mapping[str, Any], market_timezone: ZoneInfo
) -> date | None:
    timestamp = _finite(raw_bar.get("t"))
    if timestamp is None or timestamp <= 0:
        return None
    try:
        return datetime.fromtimestamp(
            timestamp / 1000.0, tz=timezone.utc
        ).astimezone(market_timezone).date()
    except (OverflowError, OSError, ValueError):
        return None


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _retry_delay(response: Any, attempt: int) -> float:
    retry_after = getattr(response, "headers", {}).get("Retry-After")
    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        return min(20.0, float(2 ** (attempt - 1)))


def _http_error_log_suffix(response: Any) -> str:
    if not isinstance(getattr(response, "status_code", None), int):
        return ""
    return " " + _http_error_detail(response)


def _http_error_detail(response: Any) -> str:
    """Return only the HTTP status; provider text can echo authenticated requests."""
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return f"status={status_code}"
    return "HTTP request failed without a response"


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("backfill clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _api_key(context: Mapping[str, Any]) -> str:
    # Load provider credentials from runtime configuration or the environment.
    value = (
        context.get("massive_key")
        or context.get("polygon_key")
        or os.getenv("MASSIVE_API_KEY")
        or os.getenv("POLYGON_API_KEY")
    )
    if not str(value or "").strip():
        raise RuntimeError("Massive API key is not configured")
    return str(value).strip()


def _mask_account(value: str) -> str:
    text = str(value)
    return "***" + text[-4:] if len(text) > 4 else "***"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill historical option-position daily prices from stored Risk "
            "transactions and Massive aggregates"
        )
    )
    parser.add_argument("-e", "--environment", default="refdata")
    parser.add_argument("--account", action="append", help="account ID (repeatable)")
    parser.add_argument(
        "--from-date",
        type=_parse_date,
        help="earliest price session to write; prior transactions are still read",
    )
    parser.add_argument(
        "--through-date",
        type=_parse_date,
        help="latest price session to write (default: yesterday in the market timezone)",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        help="exact OCC option symbol to include (repeatable)",
    )
    parser.add_argument(
        "--adjusted",
        action="store_true",
        help="request split-adjusted Massive aggregates (default: unadjusted)",
    )
    parser.add_argument("--max-transactions", type=int, default=1_000_000)
    parser.add_argument(
        "--max-requests-per-minute",
        type=int,
        help="Massive request limit, including retries and pagination (default: 5)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        context = build_context(args.environment, quiet_configuration=True)
        market_timezone = ZoneInfo(
            str(context.get("risk.option_prices.market_timezone", "America/New_York"))
        )
        today = datetime.now(timezone.utc).astimezone(market_timezone).date()
        through_date = args.through_date or today - timedelta(days=1)
        if through_date >= today:
            raise ValueError("through-date must be earlier than today")
        if args.from_date is not None and args.from_date > through_date:
            raise ValueError("from-date must not be after through-date")
        max_requests_per_minute = args.max_requests_per_minute
        if max_requests_per_minute is None:
            max_requests_per_minute = int(
                context.get(
                    "risk.option_prices.backfill.max_requests_per_minute", 5
                )
            )
        transaction_repository = TransactionRepository.from_context(context)
        option_price_repository = OptionPriceRepository.from_context(context)
        provider = MassiveAggregateClient(
            _api_key(context),
            base_url=str(context.get("massive.base_url", "https://api.massive.com")),
            timeout_seconds=float(
                context.get("risk.option_prices.backfill.timeout_seconds", 25.0)
            ),
            max_attempts=int(
                context.get("risk.option_prices.backfill.max_attempts", 5)
            ),
            max_requests_per_minute=max_requests_per_minute,
        )
        result = HistoricalOptionPriceBackfill(
            transaction_repository,
            option_price_repository,
            provider,
            context,
        ).run(
            account_ids=args.account,
            from_date=args.from_date,
            through_date=through_date,
            symbols=args.symbol,
            adjusted=args.adjusted,
            dry_run=args.dry_run,
            max_transactions=args.max_transactions,
        )
    except Exception as exc:
        print(
            f"Historical option-price backfill failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    summary = result.summary
    print(
        f"run={summary.run_id} accounts={summary.account_count} "
        f"contracts={summary.position_count} prices={summary.quote_count} "
        f"contracts_without_bars={summary.missing_count} "
        f"dry_run={str(args.dry_run).lower()}"
    )
    if args.account:
        print("accounts=" + ",".join(_mask_account(item) for item in args.account))
    if result.contracts_without_bars:
        print(
            "Massive returned no eligible aggregate bars for "
            + ", ".join(result.contracts_without_bars),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BackfillResult",
    "ContractHoldingHistory",
    "HistoricalOptionPriceBackfill",
    "HoldingWindow",
    "MassiveAggregateClient",
    "compile_holding_histories",
    "main",
]
