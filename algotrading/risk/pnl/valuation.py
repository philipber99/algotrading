"""P&L valuation, boundary marking, decomposition, and query orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import math
from threading import Lock
from typing import Any, Mapping, Sequence

from algotrading.risk.market_inputs.client import ReferenceDataClient, ReferenceDataClientError
from algotrading.risk.instruments.terms import resolve_instrument_terms
from algotrading.risk.positions.normalization import canonical_sector
from algotrading.refdata.api.option_symbols import parse_occ_symbol
from algotrading.risk.transactions.models import TransactionCoverage, TransactionModel
from algotrading.risk.transactions.repository import (
    TransactionRepository,
    TransactionResultLimitExceeded,
)

from .models import (
    PnlAttributionRow,
    PnlAvailableFilters,
    PnlDailyPoint,
    PnlDataQuality,
    PnlExcludedTrade,
    PnlSummary,
    PnlTrade,
    PnlTradeQueryData,
    PnlTradeQueryRequest,
    PnlTradeQueryResponse,
    PnlTransaction,
)
from .reconstruction import (
    EPSILON,
    ExecutionLeg,
    Lifecycle,
    expiration_closed_at,
    inventory_at,
    lifecycle_status,
    normalize_execution_legs,
    recognize_structure,
    reconstruct_lifecycles,
)


METHODOLOGY_VERSION = "pnl_lifecycle_instrument_terms_v2"


class PnlMarkHistoryUnavailable(RuntimeError):
    def __init__(self, message: str, *, symbols: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.symbols = tuple(
            dict.fromkeys(
                symbol.strip().upper() for symbol in symbols if symbol.strip()
            )
        )


class PnlReconstructionLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedLifecycles:
    lifecycles: list[Lifecycle]
    coverage: list[TransactionCoverage]
    warnings: list[str]


@dataclass(frozen=True)
class _LifecycleCacheEntry:
    repository: TransactionRepository
    reference_client: ReferenceDataClient | None
    history_start: datetime
    load_end_at: datetime
    loaded: LoadedLifecycles


_LIFECYCLE_CACHE: dict[tuple[int, tuple[str, ...], int], _LifecycleCacheEntry] = {}
_LIFECYCLE_CACHE_LOCK = Lock()


@dataclass(frozen=True)
class ValuedTrade:
    model: PnlTrade
    lifecycle: Lifecycle
    cumulative_by_day: dict[date, float]


@dataclass
class _Lot:
    quantity: float
    price: float
    multiplier: float


@dataclass(frozen=True)
class ExactOptionObservation:
    """The rich exact-contract observation selected at a causal boundary."""

    canonical_occ_symbol: str
    at: datetime
    mark: float
    session_date: date | None = None
    observed_at: datetime | None = None
    mark_basis: str | None = None
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    underlying_price: float | None = None
    provider_iv: float | None = None
    provider_delta: float | None = None
    provider_gamma: float | None = None
    provider_theta_per_day: float | None = None
    provider_vega_per_1pct: float | None = None
    provider_rho_per_1pct: float | None = None
    source: str = "unknown"
    is_execution_fill: bool = False


class MarketMarkBook:
    """Exact, non-fabricating boundary marks from snapshot and RefData."""

    def __init__(self, snapshot: Any, reference_client: ReferenceDataClient) -> None:
        self.snapshot = snapshot
        self.reference_client = reference_client
        self.as_of = snapshot.as_of.astimezone(timezone.utc)
        self._history: dict[str, list[tuple[datetime, float]]] = {}
        self._option_history: dict[str, list[ExactOptionObservation]] = {}
        self._current: dict[tuple[str, str], float] = {}
        self._current_by_symbol: dict[str, float] = {}
        self._current_option: dict[tuple[str, str], ExactOptionObservation] = {}
        self._current_option_by_symbol: dict[str, ExactOptionObservation] = {}
        self.requested_marks = 0
        self.resolved_marks = 0
        self.warnings: list[str] = []
        for position in snapshot.data.positions:
            symbol = position.canonical_occ_symbol or position.exact_symbol
            mark = _finite(position.mark)
            if symbol and mark is not None:
                normalized = symbol.strip().upper()
                self._current[(position.account_id, normalized)] = mark
                self._current_by_symbol.setdefault(normalized, mark)
                if position.canonical_occ_symbol:
                    raw_fields = getattr(position, "raw_broker_fields", None)
                    raw_underlying = (
                        raw_fields.get("underlying_price")
                        if isinstance(raw_fields, Mapping)
                        else None
                    )
                    observation = ExactOptionObservation(
                        canonical_occ_symbol=normalized,
                        at=self.as_of,
                        mark=mark,
                        session_date=self.as_of.date(),
                        observed_at=self.as_of,
                        mark_basis="portfolio_snapshot_mark",
                        underlying_price=_finite(raw_underlying),
                        provider_iv=_finite(
                            getattr(position, "implied_volatility", None)
                        ),
                        provider_delta=_finite(getattr(position, "delta", None)),
                        provider_gamma=_finite(getattr(position, "gamma", None)),
                        provider_theta_per_day=_finite(
                            getattr(position, "theta_per_day", None)
                        ),
                        provider_vega_per_1pct=_finite(
                            getattr(position, "vega_per_1pct", None)
                        ),
                        provider_rho_per_1pct=_finite(
                            getattr(position, "rho_per_1pct", None)
                        ),
                        source=str(getattr(position, "quote_source", None) or "portfolio_snapshot"),
                    )
                    self._current_option[(position.account_id, normalized)] = observation
                    self._current_option_by_symbol.setdefault(normalized, observation)

    def load_history(
        self,
        lifecycles: Sequence[Lifecycle],
        *,
        start_at: datetime,
        end_at: datetime,
        maximum_option_symbols: int,
        history_lookback_days: int = 4,
        remote_equity_symbols: Sequence[str] = (),
    ) -> None:
        equity_symbols: set[str] = set()
        option_symbols: set[str] = set()
        for lifecycle in lifecycles:
            for execution in lifecycle.executions:
                if not self._needs_historical_mark(lifecycle, execution.symbol, start_at, end_at):
                    continue
                if execution.asset_type == "OPTION":
                    option_symbols.add(execution.symbol)
                elif execution.asset_type in {"EQUITY", "ETF"}:
                    equity_symbols.add(execution.symbol)
        if equity_symbols:
            self._load_equity_history(
                equity_symbols,
                start_at,
                end_at,
                lookback_days=history_lookback_days,
                remote_equity_symbols=remote_equity_symbols,
            )
        if option_symbols:
            if len(option_symbols) > maximum_option_symbols:
                raise PnlReconstructionLimitExceeded(
                    "historical option symbol count exceeds the configured limit"
                )
            self._load_option_history(
                option_symbols,
                start_at,
                min(end_at, self.as_of),
                lookback_days=history_lookback_days,
            )

    def load_execution_fills(self, lifecycles: Sequence[Lifecycle]) -> None:
        """Use actual fills as the exact mark at execution boundaries."""
        for lifecycle in lifecycles:
            for execution in lifecycle.executions:
                fill = _finite(execution.fill_price)
                if fill is None:
                    continue
                self._history.setdefault(execution.symbol, []).append(
                    (execution.timestamp, fill)
                )
                if execution.asset_type == "OPTION":
                    self._option_history.setdefault(execution.symbol, []).append(
                        ExactOptionObservation(
                            canonical_occ_symbol=execution.symbol,
                            at=execution.timestamp,
                            mark=fill,
                            session_date=execution.timestamp.date(),
                            observed_at=execution.timestamp,
                            mark_basis="execution_fill",
                            source="execution_fill",
                            is_execution_fill=True,
                        )
                    )
        for symbol in self._history:
            self._history[symbol].sort()
        for symbol in self._option_history:
            self._option_history[symbol].sort(key=lambda item: item.at)

    def mark(
        self,
        account_id: str,
        symbol: str,
        at: datetime,
        *,
        current_allowed: bool = True,
    ) -> float:
        self.requested_marks += 1
        normalized = symbol.strip().upper()
        at_utc = at.astimezone(timezone.utc)
        # Current marks are allowed only at or beyond the snapshot timestamp.
        if current_allowed and at_utc >= self.as_of:
            current = self._current.get((account_id, normalized))
            if current is None:
                current = self._current_by_symbol.get(normalized)
            if current is not None:
                self.resolved_marks += 1
                return current
        # Historical valuation uses the latest mark available at the requested boundary.
        eligible = [price for timestamp, price in self._history.get(normalized, []) if timestamp <= at_utc]
        if eligible:
            self.resolved_marks += 1
            return eligible[-1]
        raise PnlMarkHistoryUnavailable(
            f"A required historical mark is unavailable for {normalized}.",
            symbols=(normalized,),
        )

    def current_mark(self, account_id: str, symbol: str) -> float | None:
        self.requested_marks += 1
        normalized = symbol.strip().upper()
        value = self._current.get((account_id, normalized))
        if value is None:
            value = self._current_by_symbol.get(normalized)
        if value is not None:
            self.resolved_marks += 1
        return value

    def option_settlement_mark(self, symbol: str, at: datetime) -> float:
        """Value an expired standard OCC option at intrinsic settlement value."""

        normalized = symbol.strip().upper()
        try:
            option = parse_occ_symbol(normalized)
        except ValueError as exc:
            raise PnlMarkHistoryUnavailable(
                f"Authoritative expiration terms are unavailable for {normalized}.",
                symbols=(normalized,),
            ) from exc
        observations = [
            item for item in self._option_history.get(normalized, ())
            if (item.session_date or item.at.date()) <= option.expiration
            and item.underlying_price is not None
        ]
        if observations:
            spot = float(observations[-1].underlying_price)
            strike = float(option.strike)
            self.requested_marks += 1
            self.resolved_marks += 1
            return max(0.0, spot - strike if option.put_call == "CALL" else strike - spot)
        self.warnings.append(f"{normalized}: expiration_settlement_used_last_exact_option_mark")
        return self.mark("", normalized, at, current_allowed=False)

    def option_observation(
        self,
        account_id: str,
        symbol: str,
        at: datetime,
        *,
        current_allowed: bool = True,
    ) -> ExactOptionObservation | None:
        """Return the same causal exact-option observation used by ``mark``."""
        normalized = symbol.strip().upper()
        at_utc = at.astimezone(timezone.utc)
        if current_allowed and at_utc >= self.as_of:
            current = self._current_option.get((account_id, normalized))
            if current is None:
                current = self._current_option_by_symbol.get(normalized)
            if current is not None:
                return current
        eligible = [
            item
            for item in self._option_history.get(normalized, ())
            if item.at <= at_utc
        ]
        return eligible[-1] if eligible else None

    def exact_option_history(
        self,
        symbol: str,
        *,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        include_execution_fills: bool = False,
    ) -> tuple[ExactOptionObservation, ...]:
        normalized = symbol.strip().upper()
        rows = []
        for item in self._option_history.get(normalized, ()):
            if item.is_execution_fill and not include_execution_fills:
                continue
            if start_at is not None and item.at < start_at.astimezone(timezone.utc):
                continue
            if end_at is not None and item.at >= end_at.astimezone(timezone.utc):
                continue
            rows.append(item)
        return tuple(rows)

    def has_historical_option_mark(self, symbol: str) -> bool:
        return any(
            not item.is_execution_fill
            for item in self._option_history.get(symbol.strip().upper(), ())
        )

    def requires_historical_mark(
        self,
        lifecycle: Lifecycle,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
    ) -> bool:
        return self._needs_historical_mark(lifecycle, symbol, start_at, end_at)

    def load_current_fallback(self, lifecycles: Sequence[Lifecycle]) -> None:
        """Fill only missing current exact marks, with explicit provenance warnings."""
        equity_symbols: set[str] = set()
        option_symbols: set[str] = set()
        for lifecycle in lifecycles:
            current = inventory_at(
                lifecycle, self.as_of + timedelta(microseconds=1)
            )
            asset_by_symbol = {
                item.symbol: item.asset_type for item in lifecycle.executions
            }
            for symbol, quantity in current.items():
                if abs(quantity) <= EPSILON or symbol in self._current_by_symbol:
                    continue
                if asset_by_symbol.get(symbol) == "OPTION":
                    if _option_expired(symbol, self.as_of):
                        continue
                    option_symbols.add(symbol)
                elif asset_by_symbol.get(symbol) in {"EQUITY", "ETF"}:
                    equity_symbols.add(symbol)
        quotes_method = getattr(self.reference_client, "quotes", None)
        if equity_symbols and callable(quotes_method):
            try:
                rows = quotes_method(sorted(equity_symbols))
            except ReferenceDataClientError:
                rows = []
            for row in rows:
                symbol = str(row.get("symbol") or "").strip().upper()
                mark = _finite(row.get("mark"))
                if symbol and mark is not None:
                    self._current_by_symbol[symbol] = mark
                    self.warnings.append(f"{symbol}: current_mark_refdata_fallback")
        option_method = getattr(self.reference_client, "option_quotes", None)
        if option_symbols and callable(option_method):
            try:
                payload = option_method(sorted(option_symbols))
            except ReferenceDataClientError:
                payload = {"data": []}
            for row in payload.get("data") or []:
                symbol = str(row.get("canonical_occ_symbol") or "").strip().upper()
                mark = _finite(row.get("mark"))
                if symbol and mark is not None and row.get("status") == "ok":
                    self._current_by_symbol[symbol] = mark
                    self._current_option_by_symbol[symbol] = ExactOptionObservation(
                        canonical_occ_symbol=symbol,
                        at=_parse_datetime(row.get("quote_time")) or self.as_of,
                        mark=mark,
                        session_date=(_parse_datetime(row.get("quote_time")) or self.as_of).date(),
                        observed_at=_parse_datetime(row.get("quote_time")),
                        mark_basis=str(row.get("mark_basis") or "current_mark"),
                        bid=_finite(row.get("bid")),
                        ask=_finite(row.get("ask")),
                        last=_finite(row.get("last")),
                        underlying_price=_finite(row.get("underlying_price")),
                        provider_iv=_finite(row.get("provider_iv")),
                        provider_delta=_finite(row.get("provider_delta")),
                        provider_gamma=_finite(row.get("provider_gamma")),
                        provider_theta_per_day=_finite(
                            row.get("provider_theta_per_day")
                        ),
                        provider_vega_per_1pct=_finite(
                            row.get("provider_vega_per_1pct")
                        ),
                        provider_rho_per_1pct=_finite(
                            row.get("provider_rho_per_1pct")
                        ),
                        source=str(row.get("source") or "current_mark_refdata_fallback"),
                    )
                    self.warnings.append(f"{symbol}: current_mark_refdata_fallback")

    @property
    def coverage_percent(self) -> float | None:
        if self.requested_marks == 0:
            return 100.0
        return 100.0 * self.resolved_marks / self.requested_marks

    def _needs_historical_mark(
        self,
        lifecycle: Lifecycle,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
    ) -> bool:
        if abs(inventory_at(lifecycle, start_at).get(symbol, 0.0)) > EPSILON and start_at < self.as_of:
            return True
        boundary = _next_utc_midnight(start_at)
        while boundary < min(end_at, self.as_of):
            if abs(inventory_at(lifecycle, boundary).get(symbol, 0.0)) > EPSILON:
                return True
            boundary += timedelta(days=1)
        if end_at < self.as_of and abs(inventory_at(lifecycle, end_at).get(symbol, 0.0)) > EPSILON:
            return True
        return False

    def _load_equity_history(
        self,
        symbols: set[str],
        start_at: datetime,
        end_at: datetime,
        *,
        lookback_days: int,
        remote_equity_symbols: Sequence[str],
    ) -> None:
        start = start_at.date() - timedelta(days=lookback_days)
        end = min(end_at, self.as_of).date()
        # P&L is SQL-first.  Remote repair is a separate, explicitly bounded
        # request for the small configured set of symbols not guaranteed to be
        # in the daily-bar database.
        try:
            rows = self.reference_client.bars(
                sorted(symbols), start=start, end=end, remote_query=False
            )
        except (ReferenceDataClientError, TypeError):
            try:
                rows = self.reference_client.bars(sorted(symbols), start=start, end=end)
            except ReferenceDataClientError as exc:
                raise PnlMarkHistoryUnavailable(
                    "Historical equity/ETF marks are unavailable."
                ) from exc
        # Repair SQL history remotely only for explicitly configured equity symbols.
        repair_symbols = sorted(
            symbols & {symbol.strip().upper() for symbol in remote_equity_symbols}
        )
        if repair_symbols:
            try:
                repaired = self.reference_client.bars(
                    repair_symbols,
                    start=start,
                    end=end,
                    remote_query=True,
                )
            except (ReferenceDataClientError, TypeError):
                self.warnings.extend(
                    f"{symbol}: remote_daily_mark_repair_unavailable"
                    for symbol in repair_symbols
                )
            else:
                repaired_by_symbol = {
                    str(item.get("symbol") or "").strip().upper(): item
                    for item in repaired
                }
                original_symbols = {
                    str(item.get("symbol") or "").strip().upper()
                    for item in rows
                }
                rows = [
                    repaired_by_symbol.get(
                        str(item.get("symbol") or "").strip().upper(), item
                    )
                    for item in rows
                ]
                rows.extend(
                    item
                    for symbol, item in repaired_by_symbol.items()
                    if symbol not in original_symbols
                )
        for item in rows:
            symbol = str(item.get("symbol") or "").strip().upper()
            history: list[tuple[datetime, float]] = []
            for bar in item.get("bars") or []:
                close = _finite(bar.get("close"))
                timestamp = _bar_timestamp(bar.get("timestamp"))
                if close is not None and timestamp is not None:
                    history.append((timestamp, close))
            self._history[symbol] = sorted(history)
        missing = sorted(symbol for symbol in symbols if not self._history.get(symbol))
        self.warnings.extend(
            f"{symbol}: historical_mark_unavailable" for symbol in missing
        )

    def _load_option_history(
        self,
        symbols: set[str],
        start_at: datetime,
        end_at: datetime,
        *,
        lookback_days: int,
    ) -> None:
        history_start = start_at - timedelta(days=lookback_days)
        grouped: dict[datetime, set[str]] = {}
        for symbol in symbols:
            effective_end = _option_history_end(symbol, end_at)
            if effective_end <= start_at:
                continue
            grouped.setdefault(effective_end, set()).add(symbol)
        for effective_end, batch_symbols in grouped.items():
            try:
                payload = self.reference_client.option_market_history(
                    {
                        "symbols": sorted(batch_symbols),
                        "start_at": history_start.astimezone(timezone.utc).isoformat(),
                        "end_at": effective_end.astimezone(timezone.utc).isoformat(),
                        "interval": "1d",
                        "include_greeks": True,
                    }
                )
            except ReferenceDataClientError as exc:
                raise PnlMarkHistoryUnavailable(
                    "The option-history service could not return historical exact-option marks",
                    symbols=sorted(batch_symbols),
                ) from exc
            for item in payload.get("data") or []:
                symbol = str(item.get("canonical_occ_symbol") or "").strip().upper()
                at = _parse_datetime(item.get("at"))
                mark = _finite(item.get("mark"))
                if symbol and at is not None and mark is not None and item.get("status") == "ok":
                    self._history.setdefault(symbol, []).append((at, mark))
                    self._option_history.setdefault(symbol, []).append(
                        ExactOptionObservation(
                            canonical_occ_symbol=symbol,
                            at=at,
                            mark=mark,
                            session_date=_parse_date(item.get("session_date")),
                            observed_at=_parse_datetime(item.get("observed_at")),
                            mark_basis=(
                                str(item.get("mark_basis"))
                                if item.get("mark_basis")
                                else None
                            ),
                            bid=_finite(item.get("bid")),
                            ask=_finite(item.get("ask")),
                            last=_finite(item.get("last")),
                            underlying_price=_finite(item.get("underlying_price")),
                            provider_iv=_finite(item.get("provider_iv")),
                            provider_delta=_finite(item.get("provider_delta")),
                            provider_gamma=_finite(item.get("provider_gamma")),
                            provider_theta_per_day=_finite(
                                item.get("provider_theta_per_day")
                            ),
                            provider_vega_per_1pct=_finite(
                                item.get("provider_vega_per_1pct")
                            ),
                            provider_rho_per_1pct=_finite(
                                item.get("provider_rho_per_1pct")
                            ),
                            source=str(item.get("source") or "unknown"),
                        )
                    )
        for symbol in self._history:
            self._history[symbol].sort()
        for symbol in self._option_history:
            self._option_history[symbol].sort(key=lambda item: item.at)
        # Reject incoherent option sessions before exposing them to lifecycle valuation.
        self._reject_incoherent_option_sessions(symbols)
        missing = sorted(
            symbol
            for symbol in symbols
            if not self._history.get(symbol)
            and not _option_expired_before(symbol, end_at)
        )
        self.warnings.extend(
            f"{symbol}: historical_mark_unavailable" for symbol in missing
        )

    def _reject_incoherent_option_sessions(self, symbols: set[str]) -> None:
        """Discard asynchronous closes that imply an impossible vertical value.

        Exact-option daily bars contain each contract's last trade, which need
        not be simultaneous across a spread.  When those closes violate the
        zero-to-strike-width bounds of a same-expiry vertical, treating them as
        one portfolio mark creates artificial P&L jumps.  Removing the entire
        inconsistent session makes normal as-of marking carry forward the most
        recent coherent observation without fabricating a replacement price.
        """
        chains: dict[tuple[str, date, str], list[tuple[float, str]]] = {}
        for symbol in symbols:
            try:
                option = parse_occ_symbol(symbol)
            except ValueError:
                continue
            chains.setdefault(
                (option.root, option.expiration, option.put_call), []
            ).append((float(option.strike), symbol))

        for (root, expiration, put_call), contracts in chains.items():
            if len(contracts) < 2:
                continue
            contracts.sort()
            by_time: dict[datetime, list[tuple[float, str, float]]] = {}
            for strike, symbol in contracts:
                for at, mark in self._history.get(symbol, []):
                    by_time.setdefault(at, []).append((strike, symbol, mark))
            rejected: set[datetime] = set()
            for at, observations in by_time.items():
                if len(observations) < 2:
                    continue
                observations.sort()
                for left, right in zip(observations, observations[1:]):
                    left_strike, _, left_mark = left
                    right_strike, _, right_mark = right
                    vertical = (
                        left_mark - right_mark
                        if put_call == "CALL"
                        else right_mark - left_mark
                    )
                    width = right_strike - left_strike
                    if vertical < -1e-9 or vertical > width + 1e-9:
                        rejected.add(at)
                        break
            if not rejected:
                continue
            contract_symbols = {symbol for _, symbol in contracts}
            for symbol in contract_symbols:
                self._history[symbol] = [
                    item for item in self._history.get(symbol, []) if item[0] not in rejected
                ]
                self._option_history[symbol] = [
                    item
                    for item in self._option_history.get(symbol, [])
                    if item.at not in rejected
                ]
            side = "call" if put_call == "CALL" else "put"
            self.warnings.append(
                f"{root} {expiration.isoformat()} {side} history: "
                f"ignored {len(rejected)} asynchronous close session(s) whose leg prices "
                "violated vertical-spread bounds"
            )


def load_lifecycles(
    repository: TransactionRepository,
    *,
    account_ids: Sequence[str],
    requested_start_at: datetime,
    requested_end_at: datetime,
    load_end_at: datetime,
    context: Mapping[str, Any],
    reference_client: ReferenceDataClient | None = None,
    reuse_cached: bool = False,
) -> LoadedLifecycles:
    maximum_rows = _setting_int(context, "risk.pnl.max_transaction_rows", 50_000, minimum=1)
    history_days = _setting_int(context, "risk.pnl.max_history_lookback_days", 7_305, minimum=1)
    cluster_seconds = _setting_float(context, "risk.pnl.execution_cluster_seconds", 2.0, minimum=0.0)
    history_start = requested_start_at - timedelta(days=history_days)
    cache_key = (id(repository), tuple(sorted(account_ids)), id(reference_client))
    if reuse_cached:
        with _LIFECYCLE_CACHE_LOCK:
            cached = _LIFECYCLE_CACHE.get(cache_key)
        if (
            cached is not None
            and cached.repository is repository
            and cached.reference_client is reference_client
            and cached.history_start <= history_start
            and cached.load_end_at >= load_end_at
        ):
            return cached.loaded
    all_executions: list[ExecutionLeg] = []
    transaction_batches: list[tuple[str, list[TransactionModel]]] = []
    coverage: list[TransactionCoverage] = []
    warnings: list[str] = []
    total_rows = 0
    for account_id in sorted(account_ids):
        remaining_rows = max(1, maximum_rows - total_rows)
        try:
            if hasattr(repository, "query_lifecycle_history"):
                rows, account_coverage, _ = repository.query_lifecycle_history(
                    account_id,
                    history_start_at=history_start,
                    end_at=load_end_at,
                    requested_start_at=requested_start_at,
                    requested_end_at=requested_end_at,
                    max_rows=remaining_rows,
                )
            else:
                rows, account_coverage = _fallback_repository_read(
                    repository,
                    account_id,
                    history_start,
                    load_end_at,
                    requested_start_at,
                    requested_end_at,
                    remaining_rows,
                )
        except TransactionResultLimitExceeded as exc:
            raise PnlReconstructionLimitExceeded(str(exc)) from exc
        total_rows += len(rows)
        if total_rows > maximum_rows:
            raise PnlReconstructionLimitExceeded(
                "transaction history exceeds the configured reconstruction limit"
            )
        transaction_batches.append((account_id, rows))
        coverage.append(account_coverage)
        if not account_coverage.complete:
            warnings.append(f"{account_id}: transaction_sync_coverage_incomplete")
    # Resolve historical instrument terms before reconstructing signed execution legs.
    resolved_terms, term_warnings = _resolve_historical_execution_terms(
        transaction_batches,
        reference_client=reference_client,
        context=context,
    )
    warnings.extend(term_warnings)
    for account_id, rows in transaction_batches:
        normalized, account_warnings = normalize_execution_legs(
            account_id,
            rows,
            instrument_terms=resolved_terms,
            context=context,
        )
        all_executions.extend(normalized)
        warnings.extend(f"{account_id}: {warning}" for warning in account_warnings)
    lifecycles = reconstruct_lifecycles(
        all_executions, cluster_seconds=cluster_seconds
    )
    loaded = LoadedLifecycles(lifecycles, coverage, list(dict.fromkeys(warnings)))
    with _LIFECYCLE_CACHE_LOCK:
        _LIFECYCLE_CACHE[cache_key] = _LifecycleCacheEntry(
            repository, reference_client, history_start, load_end_at, loaded
        )
        while len(_LIFECYCLE_CACHE) > 8:
            _LIFECYCLE_CACHE.pop(next(iter(_LIFECYCLE_CACHE)))
    return loaded


def _resolve_historical_execution_terms(
    batches: Sequence[tuple[str, Sequence[TransactionModel]]],
    *,
    reference_client: ReferenceDataClient | None,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve derivative transaction terms once for the complete query scope."""
    symbol_types: set[tuple[str, str]] = set()
    for _account_id, transactions in batches:
        for parent in transactions:
            for item in parent.items:
                symbol = str(item.symbol or parent.symbol or "").strip().upper()
                asset_type = str(item.asset_type or parent.asset_type or "").strip().upper()
                if symbol:
                    symbol_types.add((symbol, asset_type))
    option_symbols: list[str] = []
    futures_contracts: list[str] = []
    for symbol, asset_type in sorted(symbol_types):
        try:
            option_symbols.append(parse_occ_symbol(symbol).canonical)
        except ValueError:
            if "FUTURE" in asset_type:
                futures_contracts.append(symbol.removeprefix("/").split(":", 1)[0])
    option_rows: dict[str, dict[str, Any]] = {}
    futures_rows: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    if reference_client is not None and option_symbols:
        try:
            payload = reference_client.option_contract_terms(option_symbols)
            option_rows = {
                str(row.get("canonical_occ_symbol") or "").upper(): dict(row)
                for row in payload.get("data") or ()
                if row.get("canonical_occ_symbol")
            }
        except Exception as exc:
            warnings.append(
                f"historical_option_terms_reference_unavailable:{type(exc).__name__}"
            )
    if reference_client is not None and futures_contracts:
        try:
            payload = reference_client.futures_specifications(futures_contracts)
            futures_rows = {
                str(row.get("canonical_contract") or "").upper(): dict(row)
                for row in payload.get("data") or ()
                if row.get("canonical_contract")
            }
        except Exception as exc:
            warnings.append(
                f"historical_futures_terms_reference_unavailable:{type(exc).__name__}"
            )
    resolved = resolve_instrument_terms(
        sorted(symbol_types),
        option_terms=option_rows,
        futures_specs=futures_rows,
        context=context,
    )
    for symbol, terms in resolved.items():
        if terms.asset_type in {"OPTION", "FUTURE"} and terms.price_multiplier is None:
            warnings.append(f"{symbol}:historical_instrument_terms_insufficient")
    return resolved, warnings


def query_trades(
    payload: PnlTradeQueryRequest,
    *,
    repository: TransactionRepository,
    snapshot: Any,
    reference_client: ReferenceDataClient,
    context: Mapping[str, Any],
) -> PnlTradeQueryResponse:
    _enforce_query_bounds(payload, context)
    snapshot_as_of = snapshot.as_of.astimezone(timezone.utc)
    if payload.start_at > snapshot_as_of:
        raise ValueError("start_at cannot be after the requested snapshot as_of")
    if payload.end_at > _next_utc_midnight(snapshot_as_of):
        raise ValueError(
            "end_at cannot extend beyond the snapshot's current UTC session"
        )
    if (
        payload.lookahead_end_at is not None
        and payload.lookahead_end_at > _next_utc_midnight(snapshot_as_of)
    ):
        raise ValueError(
            "lookahead_end_at cannot extend beyond the snapshot's current UTC session"
        )
    coverage_end = min(payload.end_at, snapshot_as_of)
    load_end = min(payload.lookahead_end_at or payload.end_at, snapshot_as_of)
    loaded = load_lifecycles(
        repository,
        account_ids=payload.account_ids,
        requested_start_at=payload.start_at,
        requested_end_at=coverage_end,
        load_end_at=load_end + timedelta(microseconds=1),
        context=context,
        reference_client=reference_client,
    )
    contributing = [
        lifecycle
        for lifecycle in loaded.lifecycles
        if _contributes(lifecycle, payload.start_at, payload.end_at)
        # Money-market mutual funds and currency sweep movements are cash
        # administration, not market-price P&L evidence.  Keeping them out of
        # the valuation set prevents irrelevant missing-mark warnings.
        and lifecycle.instrument_type not in {"Mutual Fund", "Currency", "Cash Equivalent"}
    ]
    # Compute lifecycle status before applying the requested trade filters.
    status_by_trade_id = {
        lifecycle.trade_id: lifecycle_status(
            lifecycle,
            end_at=payload.end_at,
            lookahead_end_at=payload.lookahead_end_at,
        )
        for lifecycle in contributing
    }
    wanted_statuses = set(payload.statuses)
    contributing = [
        lifecycle
        for lifecycle in contributing
        if (
            payload.include_closing_after_end
            or status_by_trade_id[lifecycle.trade_id] == "closed"
        )
        and (
            not wanted_statuses
            or status_by_trade_id[lifecycle.trade_id] in wanted_statuses
        )
    ]
    marks = MarketMarkBook(snapshot, reference_client)
    marks.load_history(
        contributing,
        start_at=payload.start_at,
        end_at=payload.end_at,
        maximum_option_symbols=_setting_int(
            context, "risk.pnl.max_option_symbols", 500, minimum=1
        ),
        history_lookback_days=_setting_int(
            context, "risk.pnl.equity_history_lookback_days", 4, minimum=0
        ),
        remote_equity_symbols=_setting_symbols(
            context,
            "risk.pnl.remote_daily_bar_symbols",
            ("SCC", "TSCO", "VOLT"),
        ),
    )
    marks.load_current_fallback(contributing)
    sectors, sector_warnings = _sectors(contributing, snapshot, reference_client)
    warnings = [*loaded.warnings, *sector_warnings]
    valued: list[ValuedTrade] = []
    excluded: list[PnlExcludedTrade] = []
    for lifecycle in contributing:
        status = status_by_trade_id[lifecycle.trade_id]
        sector = sectors.get(lifecycle.underlying_symbol, "Unclassified")
        expired_missing = _expired_options_without_history(
            lifecycle,
            marks,
            start_at=payload.start_at,
            end_at=payload.end_at,
        )
        if expired_missing:
            excluded.append(
                PnlExcludedTrade(
                    trade_id=lifecycle.trade_id,
                    name=recognize_structure(lifecycle),
                    underlying_symbol=lifecycle.underlying_symbol,
                    sector=sector,
                    instrument_type=lifecycle.instrument_type,
                    missing_symbols=[],
                    transaction_ids=list(
                        dict.fromkeys(
                            str(item.repository_transaction_id)
                            for item in lifecycle.executions
                        )
                    ),
                    reason=(
                        "expired_option_without_history_excluded:"
                        + ",".join(expired_missing)
                    ),
                )
            )
            warnings.extend(
                f"{symbol}:expired_option_without_history_excluded"
                for symbol in expired_missing
            )
            continue
        try:
            valued_trade, trade_warnings = _value_lifecycle(
                lifecycle,
                status=status,
                sector=sector,
                start_at=payload.start_at,
                end_at=payload.end_at,
                marks=marks,
            )
        except PnlMarkHistoryUnavailable as exc:
            excluded.append(
                PnlExcludedTrade(
                    trade_id=lifecycle.trade_id,
                    name=recognize_structure(lifecycle),
                    underlying_symbol=lifecycle.underlying_symbol,
                    sector=sector,
                    instrument_type=lifecycle.instrument_type,
                    missing_symbols=list(exc.symbols) or sorted(lifecycle.symbols),
                    transaction_ids=list(
                        dict.fromkeys(
                            str(item.repository_transaction_id)
                            for item in lifecycle.executions
                        )
                    ),
                    reason=str(exc),
                )
            )
            continue
        valued.append(valued_trade)
        warnings.extend(trade_warnings)

    available_filters = PnlAvailableFilters(
        sectors=sorted({item.model.sector for item in valued}, key=str.casefold),
        instrument_types=sorted(
            {item.model.instrument_type for item in valued}, key=str.casefold
        ),
    )
    wanted_sectors = {item.casefold() for item in payload.sectors}
    wanted_instruments = {item.casefold() for item in payload.instrument_types}
    filtered = [
        item
        for item in valued
        if (not wanted_sectors or item.model.sector.casefold() in wanted_sectors)
        and (
            not wanted_instruments
            or item.model.instrument_type.casefold() in wanted_instruments
        )
        and (not wanted_statuses or item.model.status in wanted_statuses)
    ]
    filtered_excluded = [
        item
        for item in excluded
        if (not wanted_sectors or item.sector.casefold() in wanted_sectors)
        and (
            not wanted_instruments
            or item.instrument_type.casefold() in wanted_instruments
        )
    ]
    filtered.sort(
        key=lambda item: (
            item.model.opened_at,
            item.model.trade_id,
        )
    )
    trades = [item.model for item in filtered]
    summary = _summary(trades)
    daily = _daily_path(filtered, payload.start_at, payload.end_at, summary.period_pnl_usd)
    warnings.extend(marks.warnings)
    if filtered_excluded:
        warnings.append("trade_valuation_incomplete")
    if any(item.model.match_confidence_percent < 80 for item in filtered):
        warnings.append("low_confidence_lifecycle_match_present")
    warnings = list(dict.fromkeys(warnings))
    complete = all(item.complete for item in loaded.coverage)
    status = "usable"
    if warnings or filtered_excluded or not complete or (marks.coverage_percent or 0.0) < 100.0:
        status = "usable_with_warnings"
    only_expired_history_exclusions = bool(filtered_excluded) and all(
        item.reason.startswith("expired_option_without_history_excluded:")
        for item in filtered_excluded
    )
    if (
        filtered_excluded
        and not filtered
        and contributing
        and not only_expired_history_exclusions
    ):
        status = "insufficient_data"
    if loaded.coverage and not any(item.complete for item in loaded.coverage):
        status = "insufficient_data"
    return PnlTradeQueryResponse(
        portfolio_snapshot_id=snapshot.portfolio_snapshot_id,
        configuration_version=snapshot.configuration_version,
        as_of=snapshot.as_of,
        methodology_version=METHODOLOGY_VERSION,
        data_quality=PnlDataQuality(
            status=status,
            transaction_coverage=loaded.coverage,
            mark_coverage_percent=_money(marks.coverage_percent),
            warnings=warnings,
        ),
        data=PnlTradeQueryData(
            summary=summary,
            daily_pnl=daily,
            sector_attribution=_composition(trades, "sector"),
            instrument_attribution=_composition(trades, "instrument_type"),
            available_filters=available_filters,
            trades=trades,
            excluded_trades=filtered_excluded,
        ),
    )


def _value_lifecycle(
    lifecycle: Lifecycle,
    *,
    status: str,
    sector: str,
    start_at: datetime,
    end_at: datetime,
    marks: MarketMarkBook,
) -> tuple[ValuedTrade, list[str]]:
    warnings: list[str] = []
    start_inventory = inventory_at(lifecycle, start_at)
    end_inventory = inventory_at(lifecycle, end_at)
    start_value = _inventory_value(lifecycle, start_inventory, start_at, marks)
    end_value = _inventory_value(lifecycle, end_inventory, end_at, marks)
    period_executions = [
        item for item in lifecycle.executions if start_at <= item.timestamp < end_at
    ]
    cash = sum(
        _required_cash(item) for item in period_executions
    )
    fees = sum(item.fees_usd for item in period_executions)
    fee_components = _fee_component_totals(period_executions)
    # Period P&L reconciles inventory value changes with execution cash and fees.
    period_pnl = end_value - start_value + cash + fees
    fifo_realized, unrealized = _period_decomposition(
        lifecycle,
        start_inventory=start_inventory,
        end_inventory=end_inventory,
        start_at=start_at,
        end_at=end_at,
        marks=marks,
    )
    realized = period_pnl - fees - unrealized
    if abs(realized - fifo_realized) > 0.01:
        warnings.append(f"{lifecycle.trade_id}: execution_cash_fifo_basis_adjustment")
    lifecycle_pnl, missing_current_marks = _lifecycle_pnl(lifecycle, marks)
    if missing_current_marks:
        warnings.append(
            f"{lifecycle.underlying_symbol} trade: current_exact_mark_unavailable "
            f"({', '.join(missing_current_marks)})"
        )
    capital = _capital_at_risk(lifecycle)
    return_percent = (
        period_pnl / capital * 100.0
        if capital is not None and capital > EPSILON
        else None
    )
    model = PnlTrade(
        trade_id=lifecycle.trade_id,
        name=recognize_structure(lifecycle),
        description=(
            f"{lifecycle.evidence.replace('_', ' ')} grouping; FIFO lots; "
            f"{len(lifecycle.symbols)} exact security leg(s)"
        ),
        underlying_symbol=lifecycle.underlying_symbol,
        sector=sector,
        instrument_type=lifecycle.instrument_type,
        opened_at=lifecycle.opened_at,
        closed_at=lifecycle.closed_at or expiration_closed_at(lifecycle, end_at),
        status=status,
        leg_count=max(1, len(lifecycle.symbols)),
        period_pnl_usd=_money(period_pnl),
        lifecycle_pnl_usd=_money(lifecycle_pnl),
        realized_pnl_usd=_money(realized),
        unrealized_pnl_usd=_money(unrealized),
        fees_and_carry_usd=_money(fees),
        fee_components_usd=fee_components,
        capital_at_risk_usd=_money(capital),
        return_percent=_money(return_percent),
        match_confidence_percent=lifecycle.confidence_percent,
        primary_driver=_primary_driver(realized, unrealized, fees),
        transaction_ids=list(
            dict.fromkeys(str(item.repository_transaction_id) for item in lifecycle.executions)
        ),
        transactions=[],
    )
    transactions = _public_transactions(
        lifecycle,
        status=status,
        start_at=start_at,
        end_at=end_at,
        period_pnl=model.period_pnl_usd,
    )
    model = model.model_copy(update={"transactions": transactions})
    cumulative = _lifecycle_daily_cumulative(
        lifecycle,
        start_at=start_at,
        end_at=end_at,
        start_value=start_value,
        marks=marks,
    )
    return ValuedTrade(model=model, lifecycle=lifecycle, cumulative_by_day=cumulative), warnings


def _period_decomposition(
    lifecycle: Lifecycle,
    *,
    start_inventory: dict[str, float],
    end_inventory: dict[str, float],
    start_at: datetime,
    end_at: datetime,
    marks: MarketMarkBook,
) -> tuple[float, float]:
    lots: dict[str, list[_Lot]] = {}
    multipliers = _multipliers(lifecycle)
    # Seed FIFO lots at opening marks to isolate the requested period's result.
    for symbol, quantity in start_inventory.items():
        lots[symbol] = [
            _Lot(
                quantity=quantity,
                price=marks.mark(lifecycle.account_id, symbol, start_at),
                multiplier=multipliers[symbol],
            )
        ]
    realized = 0.0
    for execution in lifecycle.executions:
        if not start_at <= execution.timestamp < end_at:
            continue
        price = execution.fill_price
        if price is None and execution.cash_effect_usd is not None:
            price = -execution.cash_effect_usd / (
                execution.signed_quantity * execution.multiplier
            )
        if price is None:
            raise PnlMarkHistoryUnavailable(
                f"Execution price is unavailable for transaction {execution.repository_transaction_id}."
            )
        remaining = execution.signed_quantity
        queue = lots.setdefault(execution.symbol, [])
        while queue and remaining * queue[0].quantity < 0:
            lot = queue[0]
            matched = min(abs(remaining), abs(lot.quantity))
            if lot.quantity > 0:
                realized += matched * (price - lot.price) * lot.multiplier
            else:
                realized += matched * (lot.price - price) * lot.multiplier
            lot.quantity += math.copysign(matched, remaining)
            remaining -= math.copysign(matched, remaining)
            if abs(lot.quantity) <= EPSILON:
                queue.pop(0)
        if abs(remaining) > EPSILON:
            queue.append(_Lot(remaining, price, execution.multiplier))
    unrealized = 0.0
    for symbol, queue in lots.items():
        if not queue:
            continue
        if _option_expired(symbol, end_at):
            mark = marks.option_settlement_mark(symbol, end_at)
            realized += sum(
                lot.quantity * (mark - lot.price) * lot.multiplier for lot in queue
            )
        else:
            mark = marks.mark(lifecycle.account_id, symbol, end_at)
            unrealized += sum(
                lot.quantity * (mark - lot.price) * lot.multiplier for lot in queue
            )
    return realized, unrealized


def _lifecycle_pnl(
    lifecycle: Lifecycle, marks: MarketMarkBook
) -> tuple[float | None, list[str]]:
    total = sum(_required_cash(item) + item.fees_usd for item in lifecycle.executions)
    if lifecycle.is_closed:
        return total, []
    current_value = 0.0
    missing_symbols: list[str] = []
    asset_by_symbol = {
        execution.symbol: execution.asset_type
        for execution in lifecycle.executions
    }
    for symbol, quantity in lifecycle.inventory.items():
        if abs(quantity) <= EPSILON:
            continue
        if (
            asset_by_symbol.get(symbol) == "OPTION"
            and _option_expired(symbol, marks.as_of)
        ):
            try:
                mark = marks.option_settlement_mark(symbol, marks.as_of)
            except PnlMarkHistoryUnavailable:
                mark = None
        else:
            mark = marks.current_mark(lifecycle.account_id, symbol)
        if mark is None:
            missing_symbols.append(symbol)
            continue
        current_value += quantity * _multipliers(lifecycle)[symbol] * mark
    if missing_symbols:
        return None, missing_symbols
    return total + current_value, []


def _option_expired(symbol: str, as_of: datetime) -> bool:
    try:
        return parse_occ_symbol(symbol).expiration < as_of.date()
    except ValueError:
        return False


def _lifecycle_daily_cumulative(
    lifecycle: Lifecycle,
    *,
    start_at: datetime,
    end_at: datetime,
    start_value: float,
    marks: MarketMarkBook,
) -> dict[date, float]:
    output: dict[date, float] = {}
    for day, boundary in _daily_boundaries(start_at, end_at):
        inventory = inventory_at(lifecycle, boundary)
        value = _inventory_value(lifecycle, inventory, boundary, marks)
        flows = sum(
            _required_cash(item) + item.fees_usd
            for item in lifecycle.executions
            if start_at <= item.timestamp < boundary
        )
        output[day] = value - start_value + flows
    return output


def _public_transactions(
    lifecycle: Lifecycle,
    *,
    status: str,
    start_at: datetime,
    end_at: datetime,
    period_pnl: float,
) -> list[PnlTransaction]:
    end_inventory = inventory_at(lifecycle, end_at)
    rows: list[PnlTransaction] = []
    period_indexes: list[int] = []
    period_fees = 0.0
    for execution in lifecycle.executions:
        in_period = start_at <= execution.timestamp < end_at
        open_at_end = abs(end_inventory.get(execution.symbol, 0.0)) > EPSILON
        if in_period:
            period_indexes.append(len(rows))
            period_fees += execution.fees_usd
        leg_status = (
            "open at period end"
            if open_at_end
            else ("closed after period" if execution.timestamp >= end_at else "closed")
        )
        rows.append(
            PnlTransaction(
                transaction_id=str(execution.repository_transaction_id),
                broker_transaction_id=execution.broker_transaction_id,
                account_id=execution.account_id,
                timestamp=execution.timestamp,
                role=execution.role,
                instrument=execution.symbol,
                symbol=execution.symbol,
                asset_type=execution.asset_type,
                action=execution.action,
                quantity=execution.signed_quantity,
                fill_price=execution.fill_price,
                fees_usd=_money(execution.fees_usd),
                fee_components_usd={
                    fee_type: _money(value)
                    for fee_type, value in sorted(execution.fee_components_usd.items())
                },
                cash_effect_usd=_money(execution.cash_effect_usd),
                period_pnl_usd=_money(execution.fees_usd) if in_period else None,
                leg_status=leg_status,
                position_effect=execution.position_effect,
                is_unrealized=open_at_end and execution.role == "opening",
            )
        )
    if period_indexes:
        index = period_indexes[-1]
        rows[index] = rows[index].model_copy(
            update={
                "period_pnl_usd": _money(
                    (rows[index].period_pnl_usd or 0.0)
                    + period_pnl
                    - period_fees
                )
            }
        )
    return rows


def _inventory_value(
    lifecycle: Lifecycle,
    inventory: Mapping[str, float],
    at: datetime,
    marks: MarketMarkBook,
) -> float:
    multipliers = _multipliers(lifecycle)
    asset_types = {item.symbol: item.asset_type for item in lifecycle.executions}
    return sum(
        quantity
        * multipliers[symbol]
        * (
            marks.option_settlement_mark(symbol, at)
            if asset_types.get(symbol) == "OPTION" and _option_expired(symbol, at)
            else marks.mark(lifecycle.account_id, symbol, at)
        )
        for symbol, quantity in inventory.items()
        if abs(quantity) > EPSILON
    )


def _multipliers(lifecycle: Lifecycle) -> dict[str, float]:
    output: dict[str, float] = {}
    for item in lifecycle.executions:
        output[item.symbol] = item.multiplier
    return output


def _capital_at_risk(lifecycle: Lifecycle) -> float | None:
    opening = [item for item in lifecycle.executions if item.role == "opening"]
    if not opening:
        return None
    if lifecycle.instrument_type in {"Equity", "ETF"}:
        if any(item.signed_quantity < 0 for item in opening):
            return None
        return max(
            0.0,
            -sum(_required_cash(item) + item.fees_usd for item in opening),
        )
    opening_cash = sum(_required_cash(item) + item.fees_usd for item in opening)
    if opening_cash < 0:
        return -opening_cash
    # A same-expiry two-leg vertical has a bounded credit risk denominator.
    option_opening = [item for item in opening if item.asset_type == "OPTION"]
    if len({item.symbol for item in option_opening}) == 2:
        try:
            from algotrading.refdata.api.option_symbols import parse_occ_symbol

            parsed = [parse_occ_symbol(item.symbol) for item in option_opening]
        except ValueError:
            return None
        if len({item.expiration for item in parsed}) == 1:
            width = abs(float(parsed[0].strike - parsed[1].strike)) * 100.0
            contracts = min(abs(item.signed_quantity) for item in option_opening)
            return max(0.0, width * contracts - opening_cash)
    return None


def _summary(trades: Sequence[PnlTrade]) -> PnlSummary:
    period = sum(item.period_pnl_usd for item in trades)
    return PnlSummary(
        period_pnl_usd=_money(period),
        realized_pnl_usd=_money(sum(item.realized_pnl_usd for item in trades)),
        unrealized_pnl_usd=_money(sum(item.unrealized_pnl_usd for item in trades)),
        fees_and_carry_usd=_money(sum(item.fees_and_carry_usd for item in trades)),
        fee_components_usd=_fee_component_totals(trades),
        gross_profit_usd=_money(sum(max(0.0, item.period_pnl_usd) for item in trades)),
        gross_loss_usd=_money(sum(min(0.0, item.period_pnl_usd) for item in trades)),
        closed_trade_count=sum(
            item.status in {"closed", "closed_after_period"} for item in trades
        ),
        open_trade_count=sum(item.status in {"open", "partial"} for item in trades),
    )


def _fee_component_totals(items: Sequence[Any]) -> dict[str, float]:
    """Aggregate signed P&L fee effects from executions or public trade models."""
    totals: dict[str, float] = {}
    for item in items:
        for fee_type, amount in getattr(item, "fee_components_usd", {}).items():
            value = _finite(amount)
            if value is not None:
                totals[fee_type] = totals.get(fee_type, 0.0) + value
    return {
        fee_type: _money(amount)
        for fee_type, amount in sorted(totals.items())
        if _money(amount) != 0.0
    }


def _daily_path(
    trades: Sequence[ValuedTrade],
    start_at: datetime,
    end_at: datetime,
    expected_total: float,
) -> list[PnlDailyPoint]:
    if not trades:
        return []
    rows: list[PnlDailyPoint] = []
    prior = 0.0
    # Aggregate trade paths over a shared sequence of daily period boundaries.
    boundaries = _daily_boundaries(start_at, end_at)
    for index, (day, boundary) in enumerate(boundaries):
        cumulative = sum(item.cumulative_by_day.get(day, 0.0) for item in trades)
        if index == len(boundaries) - 1:
            cumulative = expected_total
        rows.append(
            PnlDailyPoint(
                at=boundary,
                daily_pnl_usd=_money(cumulative - prior),
                cumulative_pnl_usd=_money(cumulative),
            )
        )
        prior = cumulative
    return rows


def _composition(trades: Sequence[PnlTrade], field: str) -> list[PnlAttributionRow]:
    grouped: dict[str, list[PnlTrade]] = {}
    for trade in trades:
        grouped.setdefault(str(getattr(trade, field)), []).append(trade)
    return [
        PnlAttributionRow(
            name=name,
            pnl_usd=_money(sum(item.period_pnl_usd for item in rows)),
            realized_pnl_usd=_money(sum(item.realized_pnl_usd for item in rows)),
            unrealized_pnl_usd=_money(sum(item.unrealized_pnl_usd for item in rows)),
            trade_count=len(rows),
        )
        for name, rows in sorted(grouped.items(), key=lambda item: item[0].casefold())
    ]


def _sectors(
    lifecycles: Sequence[Lifecycle],
    snapshot: Any,
    reference_client: ReferenceDataClient,
) -> tuple[dict[str, str], list[str]]:
    symbols = sorted({item.underlying_symbol for item in lifecycles})
    classifiable = {
        item.underlying_symbol
        for item in lifecycles
        if item.instrument_type in {"Equity", "Options"}
    }
    sectors: dict[str, str] = {}
    warnings: list[str] = []
    for position in snapshot.data.positions:
        symbol = position.underlying_symbol or position.exact_symbol
        if symbol and position.sector and position.sector != "Unclassified":
            sectors.setdefault(symbol.upper(), position.sector)
    missing = [symbol for symbol in symbols if symbol in classifiable and symbol not in sectors]
    if missing:
        try:
            rows = reference_client.security_classifications(missing)
        except ReferenceDataClientError:
            rows = []
            warnings.append("security_classification_service_unavailable")
        for row in rows:
            symbol = str(row.get("symbol") or "").strip().upper()
            sector = canonical_sector(row.get("sector"))
            if symbol and sector != "Unclassified":
                sectors[symbol] = sector
    for symbol in missing:
        if symbol not in sectors:
            sectors[symbol] = "Unclassified"
    for symbol in symbols:
        if symbol not in classifiable and symbol not in sectors:
            sectors[symbol] = "Not applicable"
    return sectors, list(dict.fromkeys(warnings))


def _fallback_repository_read(
    repository: Any,
    account_id: str,
    history_start: datetime,
    load_end: datetime,
    requested_start: datetime,
    requested_end: datetime,
    maximum_rows: int,
) -> tuple[list[TransactionModel], TransactionCoverage]:
    rows: list[TransactionModel] = []
    cursor: str | None = None
    coverage: TransactionCoverage | None = None
    while True:
        page = repository.query(
            account_id,
            start_at=history_start,
            end_at=load_end,
            types=["TRADE"],
            limit=min(1000, max(1, maximum_rows - len(rows))),
            cursor=cursor,
        )
        rows.extend(page.data)
        coverage = page.meta.coverage
        if len(rows) > maximum_rows:
            raise PnlReconstructionLimitExceeded(
                "transaction history exceeds the configured reconstruction limit"
            )
        cursor = page.meta.next_cursor
        if not cursor:
            break
    if coverage is None:
        coverage = TransactionCoverage(
            requested_start_at=requested_start,
            requested_end_at=requested_end,
            sync_through_by_type={"TRADE": None},
            complete=False,
            missing_intervals=["TRADE:not_synced_through_end"],
        )
    else:
        coverage = coverage.model_copy(
            update={
                "requested_start_at": requested_start,
                "requested_end_at": requested_end,
            }
        )
    rows.sort(key=lambda item: (item.transaction_at, item.id))
    return rows, coverage


def _enforce_query_bounds(payload: PnlTradeQueryRequest, context: Mapping[str, Any]) -> None:
    max_accounts = _setting_int(context, "risk.pnl.max_accounts", 20, minimum=1)
    if len(payload.account_ids) > max_accounts:
        raise PnlReconstructionLimitExceeded(
            f"account count exceeds configured maximum of {max_accounts}"
        )
    max_filters = _setting_int(context, "risk.pnl.max_filters", 100, minimum=0)
    if len(payload.sectors) + len(payload.instrument_types) + len(payload.statuses) > max_filters:
        raise PnlReconstructionLimitExceeded(
            f"filter count exceeds configured maximum of {max_filters}"
        )
    max_period = _setting_int(context, "risk.pnl.max_period_days", 366, minimum=1)
    if payload.end_at - payload.start_at > timedelta(days=max_period):
        raise PnlReconstructionLimitExceeded(
            f"requested period exceeds configured maximum of {max_period} days"
        )
    if payload.lookahead_end_at is not None:
        maximum = _setting_int(context, "risk.pnl.max_lookahead_days", 366, minimum=0)
        if payload.lookahead_end_at - payload.end_at > timedelta(days=maximum):
            raise PnlReconstructionLimitExceeded(
                f"lookahead exceeds configured maximum of {maximum} days"
            )


def _contributes(lifecycle: Lifecycle, start_at: datetime, end_at: datetime) -> bool:
    if inventory_at(lifecycle, start_at):
        return True
    if inventory_at(lifecycle, end_at):
        return True
    return any(start_at <= item.timestamp < end_at for item in lifecycle.executions)


def _daily_boundaries(start_at: datetime, end_at: datetime) -> list[tuple[date, datetime]]:
    output: list[tuple[date, datetime]] = []
    cursor = start_at
    while cursor < end_at:
        boundary = min(end_at, _next_utc_midnight(cursor))
        output.append((cursor.date(), boundary))
        cursor = boundary
    return output


def _expired_options_without_history(
    lifecycle: Lifecycle,
    marks: MarketMarkBook,
    *,
    start_at: datetime,
    end_at: datetime,
) -> list[str]:
    asset_types = {item.symbol: item.asset_type for item in lifecycle.executions}
    output: list[str] = []
    for symbol in sorted(lifecycle.symbols):
        if asset_types.get(symbol) != "OPTION":
            continue
        if not _option_expired_before(symbol, end_at):
            continue
        if not marks.requires_historical_mark(lifecycle, symbol, start_at, end_at):
            continue
        if not marks.has_historical_option_mark(symbol):
            output.append(symbol)
    return output


def _option_history_end(symbol: str, requested_end: datetime) -> datetime:
    try:
        expiration = parse_occ_symbol(symbol).expiration
    except ValueError:
        return requested_end
    after_expiration = datetime.combine(
        expiration + timedelta(days=1), time.min, tzinfo=timezone.utc
    )
    return min(requested_end, after_expiration)


def _option_expired_before(symbol: str, at: datetime) -> bool:
    try:
        expiration = parse_occ_symbol(symbol).expiration
    except ValueError:
        return False
    return expiration < at.astimezone(timezone.utc).date()


def _next_utc_midnight(value: datetime) -> datetime:
    return datetime.combine(value.date() + timedelta(days=1), time.min, tzinfo=timezone.utc)


def _required_cash(execution: ExecutionLeg) -> float:
    if execution.cash_effect_usd is not None:
        return execution.cash_effect_usd
    if execution.fill_price is None:
        raise PnlMarkHistoryUnavailable(
            f"Execution cash and fill are unavailable for transaction {execution.repository_transaction_id}."
        )
    return -execution.signed_quantity * execution.fill_price * execution.multiplier


def _primary_driver(realized: float, unrealized: float, fees: float) -> str | None:
    values = {
        "realized price movement": abs(realized),
        "unrealized price movement": abs(unrealized),
        "fees and carry": abs(fees),
    }
    name, value = max(values.items(), key=lambda item: item[1])
    return name if value > 0 else None


def _bar_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        # A daily bar is usable only after that session date has ended.
        return datetime.combine(parsed.date() + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value not in (None, ""):
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _setting_int(
    context: Mapping[str, Any], key: str, default: int, *, minimum: int
) -> int:
    value = int(context.get(key, default))
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def _setting_float(
    context: Mapping[str, Any], key: str, default: float, *, minimum: float
) -> float:
    value = float(context.get(key, default))
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{key} must be finite and at least {minimum}")
    return value


def _setting_symbols(
    context: Mapping[str, Any], key: str, default: Sequence[str]
) -> tuple[str, ...]:
    raw = context.get(key, default)
    values = raw.split(",") if isinstance(raw, str) else raw
    if not isinstance(values, Sequence):
        raise ValueError(f"{key} must be a sequence or comma-separated string")
    return tuple(
        dict.fromkeys(
            str(value).strip().upper()
            for value in values
            if str(value).strip()
        )
    )


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _money(value: float | None) -> float | None:
    if value is None:
        return None
    if abs(value) < 0.0000005:
        return 0.0
    return round(float(value), 6)


__all__ = [
    "ExactOptionObservation",
    "LoadedLifecycles",
    "METHODOLOGY_VERSION",
    "MarketMarkBook",
    "PnlMarkHistoryUnavailable",
    "PnlReconstructionLimitExceeded",
    "load_lifecycles",
    "query_trades",
]
