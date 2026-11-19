"""Standalone nightly capture of current broker option-position prices."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timezone
import logging
import math
import sys
import uuid
from typing import Any, Callable
from zoneinfo import ZoneInfo

from algotrading.ContextFields import PARAM_LOCAL_TIMEZONE
from algotrading.refdata.api.option_symbols import parse_occ_symbol
from algotrading.risk.api.bootstrap import build_context
from algotrading.risk.api.broker_source import create_tda_broker
from algotrading.risk.api.normalization import normalize_accounts
from algotrading.risk.transactions.normalization import normalize_transaction
from algotrading.tda.TDAReferenceData import TDAReferenceData
from algotrading.util.NagiosSystemMonitor import NagiosSystemMonitor

from .models import (
    CapturedChainSnapshot,
    CapturedOptionPrice,
    CapturedSurfacePoint,
    CaptureSummary,
    ChainSnapshotFlag,
    ChainSnapshotStatus,
    MarkSource,
    OptionPosition,
    SecurityStatus,
    SpotSource,
    SurfaceObservationSource,
)
from .repository import OptionPriceRepository


LOGGER = logging.getLogger("algotrading.risk.option_prices.capture")


class OptionPriceCapture:
    def __init__(
        self,
        broker: Any,
        quote_provider: Any,
        repository: OptionPriceRepository,
        context: Mapping[str, Any],
        *,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.broker = broker
        self.quote_provider = quote_provider
        self.repository = repository
        self.context = context
        self._now = now_fn

    def capture(
        self,
        *,
        account_ids: Sequence[str] | None = None,
        dry_run: bool = False,
    ) -> tuple[CaptureSummary, list[str]]:
        captured_at = _aware_utc(self._now())
        raw_accounts = self.broker.get_accounts()
        accounts = normalize_accounts(raw_accounts)
        selected = {
            str(item).strip() for item in account_ids or () if str(item).strip()
        }
        if selected:
            accounts = [item for item in accounts if item.account_id in selected]
            missing_accounts = sorted(selected - {item.account_id for item in accounts})
            if missing_accounts:
                raise ValueError(
                    "one or more requested accounts were not returned by the broker"
                )
        market_timezone = ZoneInfo(
            str(
                self.context.get(
                    "risk.option_prices.market_timezone", "America/New_York"
                )
            )
        )
        broker_timezone = ZoneInfo(
            str(self.context.get(PARAM_LOCAL_TIMEZONE) or market_timezone.key)
        )
        provider_class = _provider_class(self.quote_provider)
        strike_count, max_strike_count = _chain_strike_counts(self.context)
        positions_by_key = {
            (item.account_id, item.canonical_occ_symbol): item
            for item in _option_positions(accounts)
        }
        transaction_start_at = self._transaction_watermark(
            provider_class=provider_class,
            account_count=len(accounts),
            captured_at=captured_at,
            market_timezone=market_timezone,
        )
        # Include recently traded contracts that may already be absent from positions.
        transaction_positions = _today_transaction_option_positions(
            self.broker,
            accounts,
            captured_at=captured_at,
            start_at=transaction_start_at,
            broker_timezone=broker_timezone,
        )
        for item in transaction_positions:
            # A live position is authoritative for quantity. A contract found
            # only in today's trades is closed at snapshot time and remains 0.
            positions_by_key.setdefault(
                (item.account_id, item.canonical_occ_symbol), item
            )
        positions = list(positions_by_key.values())
        provider_symbols = list(
            dict.fromkeys(item.provider_option_symbol for item in positions)
        )
        raw_quotes = self._option_quotes(provider_symbols) if provider_symbols else {}
        if not isinstance(raw_quotes, Mapping):
            raise ValueError("option quote provider returned an invalid response")
        source = type(self.quote_provider).__name__
        prices: list[CapturedOptionPrice] = []
        missing: list[str] = []
        for position in positions:
            quote = raw_quotes.get(position.provider_option_symbol)
            if not isinstance(quote, Mapping):
                missing.append(position.canonical_occ_symbol)
                continue
            price = _captured_price(
                position,
                quote,
                captured_at=captured_at,
                market_timezone=market_timezone,
                source=source,
            )
            if price is None:
                missing.append(position.canonical_occ_symbol)
            else:
                prices.append(price)
        # Capture surrounding chain evidence alongside exact held-contract quotes.
        chains = self._capture_chains(
            positions,
            raw_quotes,
            strike_count=strike_count,
            max_strike_count=max_strike_count,
        )
        chain_count = len(chains)
        surface_point_count = sum(len(item.points) for item in chains)
        has_chain_problem = any(
            item.snapshot_status != int(ChainSnapshotStatus.SUCCESS)
            for item in chains
        )
        summary = CaptureSummary(
            run_id=str(uuid.uuid4()),
            captured_at=captured_at,
            account_count=len(accounts),
            position_count=len(positions),
            quote_count=len(prices),
            missing_count=len(missing),
            status="partial" if missing or has_chain_problem else "success",
            provider_class=provider_class,
            chain_count=chain_count,
            surface_point_count=surface_point_count,
        )
        if not dry_run:
            self.repository.save_capture(summary, prices, chain_snapshots=chains)
        return summary, sorted(dict.fromkeys(missing))

    def _transaction_watermark(
        self,
        *,
        provider_class: str,
        account_count: int,
        captured_at: datetime,
        market_timezone: ZoneInfo,
    ) -> datetime:
        method = getattr(self.repository, "latest_successful_capture_at", None)
        if callable(method):
            prior = method(
                provider_class=provider_class,
                minimum_account_count=account_count,
            )
            if isinstance(prior, datetime):
                try:
                    return _aware_utc(prior)
                except ValueError:
                    # MySQL DATETIME values are UTC at this repository boundary.
                    return prior.replace(tzinfo=timezone.utc)
        session_date = captured_at.astimezone(market_timezone).date()
        return datetime.combine(session_date, time.min, tzinfo=market_timezone).astimezone(
            timezone.utc
        )

    def _capture_chains(
        self,
        positions: Sequence[OptionPosition],
        raw_quotes: Mapping[str, Any],
        *,
        strike_count: int,
        max_strike_count: int,
    ) -> list[CapturedChainSnapshot]:
        # Share one chain capture across accounts holding the same expiry and underlying.
        targets_by_group: dict[tuple[str, date], set[str]] = {}
        quote_by_canonical: dict[str, Mapping[str, Any]] = {}
        for position in positions:
            key = (position.underlying_symbol, position.expiration_date)
            targets_by_group.setdefault(key, set()).add(position.canonical_occ_symbol)
            quote = raw_quotes.get(position.provider_option_symbol)
            if isinstance(quote, Mapping):
                quote_by_canonical.setdefault(position.canonical_occ_symbol, quote)
        chains: list[CapturedChainSnapshot] = []
        for (underlying_symbol, expiration_date), target_symbols in sorted(
            targets_by_group.items()
        ):
            snapshot = self._capture_chain(
                underlying_symbol,
                expiration_date,
                target_symbols,
                quote_by_canonical,
                strike_count=strike_count,
                max_strike_count=max_strike_count,
            )
            chains.append(snapshot)
            if snapshot.snapshot_status != int(ChainSnapshotStatus.SUCCESS):
                _log_incomplete_chain_snapshot(snapshot, target_symbols)
        return chains

    def _capture_chain(
        self,
        underlying_symbol: str,
        expiration_date: date,
        target_symbols: set[str],
        quote_by_canonical: Mapping[str, Mapping[str, Any]],
        *,
        strike_count: int,
        max_strike_count: int,
    ) -> CapturedChainSnapshot:
        current_strike_count = strike_count
        last_valid: tuple[Mapping[str, Any], tuple[CapturedSurfacePoint, ...]] | None = None
        previous_strike_ranges: tuple[tuple[str, float, float], ...] | None = None
        request_failed = False
        while True:
            try:
                response = self.quote_provider.get_options_chain(
                    underlying_symbol,
                    contract_type="ALL",
                    strategy="SINGLE",
                    strike_count=current_strike_count,
                    include_underlying_quote=True,
                    from_date=expiration_date,
                    to_date=expiration_date,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Option-chain request failed: underlying=%s expiration=%s "
                    "strike_count=%d error=%s",
                    underlying_symbol,
                    expiration_date.isoformat(),
                    current_strike_count,
                    type(exc).__name__,
                )
                request_failed = True
                break
            if not _valid_chain_response(response):
                break
            assert isinstance(response, Mapping)
            points = _normalize_chain_points(response, expiration_date)
            last_valid = (response, points)
            covered = {item.canonical_occ_symbol for item in points}
            # Widen the chain request until held strikes are covered or expansion stops.
            if target_symbols <= covered:
                break
            strike_ranges = _chain_strike_ranges(points)
            outside_targets = _targets_outside_strike_ranges(
                target_symbols,
                strike_ranges,
            )
            if current_strike_count < max_strike_count:
                next_strike_count = min(
                    max_strike_count, current_strike_count * 2
                )
            elif outside_targets and strike_ranges != previous_strike_ranges:
                next_strike_count = current_strike_count * 2
                LOGGER.debug(
                    "Expanding option-chain range beyond configured maximum: "
                    "underlying=%s expiration=%s strike_count=%d "
                    "outside_targets=%s",
                    underlying_symbol,
                    expiration_date.isoformat(),
                    next_strike_count,
                    ",".join(sorted(outside_targets)),
                )
            else:
                break
            previous_strike_ranges = strike_ranges
            current_strike_count = next_strike_count

        if last_valid is None:
            return CapturedChainSnapshot(
                underlying_symbol=underlying_symbol,
                expiration_date=expiration_date,
                spot=None,
                spot_source=None,
                underlying_quote_time=None,
                snapshot_status=int(ChainSnapshotStatus.FAILED),
                snapshot_flags=0,
                error_code=(
                    "provider_request_failed"
                    if request_failed
                    else "invalid_chain_response"
                ),
                points=(),
            )

        response, chain_points = last_valid
        flags = _chain_flags(response, underlying_symbol)
        spot, spot_source, underlying_quote_time = _select_chain_spot(response)
        if not chain_points:
            return CapturedChainSnapshot(
                underlying_symbol=underlying_symbol,
                expiration_date=expiration_date,
                spot=spot,
                spot_source=spot_source,
                underlying_quote_time=underlying_quote_time,
                snapshot_status=int(ChainSnapshotStatus.FAILED),
                snapshot_flags=flags,
                error_code="no_normalized_points",
                points=(),
            )

        chain_symbols = {item.canonical_occ_symbol for item in chain_points}
        missing_targets = target_symbols - chain_symbols
        # Retain exact target quotes when the chain response omits held contracts.
        fallback_points = tuple(
            _surface_point_from_exact_quote(symbol, quote_by_canonical[symbol])
            for symbol in sorted(missing_targets)
            if symbol in quote_by_canonical
        )
        represented = chain_symbols | {
            item.canonical_occ_symbol for item in fallback_points
        }
        unresolved_targets = target_symbols - represented
        points = tuple(sorted((*chain_points, *fallback_points), key=lambda item: item.canonical_occ_symbol))
        truncated = bool(flags & int(ChainSnapshotFlag.TRUNCATED))
        if fallback_points:
            snapshot_status = int(ChainSnapshotStatus.PARTIAL)
            error_code = "target_quote_fallback"
        elif unresolved_targets:
            snapshot_status = int(ChainSnapshotStatus.PARTIAL)
            error_code = "target_missing"
        elif spot is None:
            snapshot_status = int(ChainSnapshotStatus.PARTIAL)
            error_code = "spot_missing"
        elif truncated:
            snapshot_status = int(ChainSnapshotStatus.PARTIAL)
            error_code = "chain_truncated"
        else:
            snapshot_status = int(ChainSnapshotStatus.SUCCESS)
            error_code = None
        return CapturedChainSnapshot(
            underlying_symbol=underlying_symbol,
            expiration_date=expiration_date,
            spot=spot,
            spot_source=spot_source,
            underlying_quote_time=underlying_quote_time,
            snapshot_status=snapshot_status,
            snapshot_flags=flags,
            error_code=error_code,
            points=points,
        )

    def _option_quotes(self, provider_symbols: list[str]) -> Any:
        method = getattr(self.quote_provider, "get_option_quotes", None)
        if not callable(method):
            method = getattr(self.quote_provider, "get_options_quotes", None)
        if not callable(method):
            raise ValueError(
                "option quote provider does not implement get_option_quotes"
            )
        return method(provider_symbols)


def _option_positions(accounts: Sequence[Any]) -> list[OptionPosition]:
    result: list[OptionPosition] = []
    for account in accounts:
        for position in account.positions:
            symbol = str(position.instrument.symbol or "").strip().upper()
            try:
                occ = parse_occ_symbol(symbol)
            except ValueError:
                continue
            signed_quantity = float(position.long_quantity or 0) - float(
                position.short_quantity or 0
            )
            if abs(signed_quantity) <= 1e-12:
                continue
            result.append(
                OptionPosition(
                    account_id=account.account_id,
                    canonical_occ_symbol=occ.canonical,
                    provider_option_symbol=occ.provider,
                    underlying_symbol=occ.root,
                    expiration_date=occ.expiration,
                    put_call=occ.put_call,
                    strike=float(occ.strike),
                    signed_quantity=signed_quantity,
                )
            )
    return result


def _today_transaction_option_positions(
    broker: Any,
    accounts: Sequence[Any],
    *,
    captured_at: datetime,
    start_at: datetime,
    broker_timezone: ZoneInfo,
) -> list[OptionPosition]:
    broker_start = _aware_utc(start_at).astimezone(broker_timezone).replace(tzinfo=None)
    broker_end = captured_at.astimezone(broker_timezone).replace(tzinfo=None)
    result: list[OptionPosition] = []
    seen: set[tuple[str, str]] = set()
    for account in accounts:
        raw_transactions = broker.get_transactions(
            account.account_id,
            broker_start,
            broker_end,
            "TRADE",
        )
        if not isinstance(raw_transactions, list) or any(
            not isinstance(item, Mapping) for item in raw_transactions
        ):
            raise ValueError("broker transaction response must be a list of objects")
        for raw_transaction in raw_transactions:
            transaction = normalize_transaction(
                raw_transaction,
                account_id=account.account_id,
                requested_type="TRADE",
            )
            if not start_at < transaction.transaction_at <= captured_at:
                continue
            symbols = [transaction.symbol]
            symbols.extend(item.symbol for item in transaction.items)
            for symbol in symbols:
                if not symbol:
                    continue
                try:
                    occ = parse_occ_symbol(symbol)
                except ValueError:
                    continue
                key = (account.account_id, occ.canonical)
                if key in seen:
                    continue
                seen.add(key)
                result.append(
                    OptionPosition(
                        account_id=account.account_id,
                        canonical_occ_symbol=occ.canonical,
                        provider_option_symbol=occ.provider,
                        underlying_symbol=occ.root,
                        expiration_date=occ.expiration,
                        put_call=occ.put_call,
                        strike=float(occ.strike),
                        signed_quantity=0.0,
                    )
                )
    return result


def _captured_price(
    position: OptionPosition,
    quote: Mapping[str, Any],
    *,
    captured_at: datetime,
    market_timezone: ZoneInfo,
    source: str,
) -> CapturedOptionPrice | None:
    bid = _finite(quote.get("bid"))
    ask = _finite(quote.get("ask"))
    last = _finite(quote.get("last"))
    close = _finite(quote.get("close"))
    mark, mark_source = _select_exact_mark(quote)
    if mark is None:
        return None
    quote_time = _provider_timestamp(quote.get("quote_time"))
    trade_time = _provider_timestamp(quote.get("trade_time"))
    # Assign the stored session from provider time before using capture time.
    session_basis = quote_time or trade_time or captured_at
    return CapturedOptionPrice(
        position=position,
        captured_at=captured_at,
        session_date=session_basis.astimezone(market_timezone).date(),
        quote_time=quote_time,
        trade_time=trade_time,
        bid=bid,
        ask=ask,
        mark=mark,
        last=last,
        open=_finite(quote.get("open")),
        high=_finite(quote.get("high")),
        low=_finite(quote.get("low")),
        close=close,
        underlying_price=_finite(quote.get("underlying_price")),
        provider_iv=_percentage_to_decimal(quote.get("volatility")),
        provider_delta=_finite(quote.get("delta")),
        provider_gamma=_finite(quote.get("gamma")),
        provider_theta_per_day=_finite(quote.get("theta")),
        provider_vega_per_1pct=_finite(quote.get("vega")),
        provider_rho_per_1pct=_finite(quote.get("rho")),
        open_interest=_finite(quote.get("open_interest")),
        volume=_finite(quote.get("total_volume")),
        source=source,
        mark_source=mark_source,
    )


def _provider_class(provider: Any) -> str:
    return f"{type(provider).__module__}.{type(provider).__qualname__}"


def _chain_strike_counts(context: Mapping[str, Any]) -> tuple[int, int]:
    initial = _positive_int(
        context.get("risk.option_prices.chain.strike_count", 20),
        "risk.option_prices.chain.strike_count",
    )
    maximum = _positive_int(
        context.get("risk.option_prices.chain.max_strike_count", 80),
        "risk.option_prices.chain.max_strike_count",
    )
    if maximum < initial:
        raise ValueError(
            "risk.option_prices.chain.max_strike_count must not be smaller than "
            "risk.option_prices.chain.strike_count"
        )
    return initial, maximum


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if converted < 1 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _select_exact_mark(quote: Mapping[str, Any]) -> tuple[float | None, int | None]:
    # Prefer provider mark, then a valid midpoint, trade, or prior close.
    mark = _finite(quote.get("mark"))
    if mark is not None:
        return mark, int(MarkSource.PROVIDER_MARK)
    bid = _finite(quote.get("bid"))
    ask = _finite(quote.get("ask"))
    if bid is not None and ask is not None and bid >= 0 and ask >= bid:
        return (bid + ask) / 2.0, int(MarkSource.BID_ASK_MIDPOINT)
    last = _finite(quote.get("last"))
    if last is not None:
        return last, int(MarkSource.LAST_TRADE)
    close = _finite(quote.get("close"))
    if close is not None:
        return close, int(MarkSource.CLOSE)
    return None, None


def _valid_chain_response(value: Any) -> bool:
    return isinstance(value, Mapping) and (
        isinstance(value.get("callExpDateMap"), Mapping)
        or isinstance(value.get("putExpDateMap"), Mapping)
    )


def _normalize_chain_points(
    response: Mapping[str, Any], expiration_date: date
) -> tuple[CapturedSurfacePoint, ...]:
    selected: dict[str, CapturedSurfacePoint] = {}
    for map_name in ("callExpDateMap", "putExpDateMap"):
        expiration_map = response.get(map_name)
        if not isinstance(expiration_map, Mapping):
            continue
        for _, strike_map in sorted(expiration_map.items(), key=lambda item: str(item[0])):
            if not isinstance(strike_map, Mapping):
                continue
            for _, nodes in sorted(strike_map.items(), key=lambda item: str(item[0])):
                if isinstance(nodes, Mapping):
                    values = (nodes,)
                elif isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes)):
                    values = nodes
                else:
                    continue
                for node in values:
                    if not isinstance(node, Mapping):
                        continue
                    try:
                        occ = parse_occ_symbol(str(node.get("symbol") or ""))
                    except ValueError:
                        continue
                    if occ.expiration != expiration_date:
                        continue
                    point = _surface_point_from_quote(
                        occ.canonical,
                        node,
                        observation_source=int(SurfaceObservationSource.OPTION_CHAIN),
                    )
                    existing = selected.get(point.canonical_occ_symbol)
                    if _prefer_surface_point(point, existing):
                        selected[point.canonical_occ_symbol] = point
    return tuple(selected[key] for key in sorted(selected))


def _chain_strike_ranges(
    points: Sequence[CapturedSurfacePoint],
) -> tuple[tuple[str, float, float], ...]:
    strikes_by_side: dict[str, list[float]] = {}
    for point in points:
        try:
            option = parse_occ_symbol(point.canonical_occ_symbol)
        except ValueError:
            continue
        strikes_by_side.setdefault(option.put_call, []).append(float(option.strike))
    return tuple(
        (put_call, min(strikes), max(strikes))
        for put_call, strikes in sorted(strikes_by_side.items())
        if strikes
    )


def _targets_outside_strike_ranges(
    target_symbols: set[str],
    strike_ranges: tuple[tuple[str, float, float], ...],
) -> set[str]:
    ranges_by_side = {
        put_call: (minimum, maximum)
        for put_call, minimum, maximum in strike_ranges
    }
    outside: set[str] = set()
    for symbol in target_symbols:
        try:
            option = parse_occ_symbol(symbol)
        except ValueError:
            continue
        side_range = ranges_by_side.get(option.put_call)
        if side_range is None:
            continue
        strike = float(option.strike)
        if strike < side_range[0] or strike > side_range[1]:
            outside.add(symbol)
    return outside


def _log_incomplete_chain_snapshot(
    snapshot: CapturedChainSnapshot,
    target_symbols: set[str],
) -> None:
    point_symbols = {item.canonical_occ_symbol for item in snapshot.points}
    if snapshot.error_code == "target_quote_fallback":
        affected_targets = sorted(
            item.canonical_occ_symbol
            for item in snapshot.points
            if item.canonical_occ_symbol in target_symbols
            and item.observation_source
            == int(SurfaceObservationSource.EXACT_OPTION_QUOTE_FALLBACK)
        )
    elif snapshot.error_code == "target_missing":
        affected_targets = sorted(target_symbols - point_symbols)
    else:
        affected_targets = sorted(target_symbols)

    try:
        status = ChainSnapshotStatus(snapshot.snapshot_status).name.lower()
    except ValueError:
        status = str(snapshot.snapshot_status)
    flag_names = [
        flag.name.lower()
        for flag in ChainSnapshotFlag
        if snapshot.snapshot_flags & int(flag)
    ]
    LOGGER.warning(
        "Option-chain snapshot incomplete: underlying=%s expiration=%s "
        "status=%s error_code=%s flags=%s points=%d targets=%d "
        "affected_targets=%s",
        snapshot.underlying_symbol,
        snapshot.expiration_date.isoformat(),
        status,
        snapshot.error_code or "none",
        ",".join(flag_names) or "none",
        len(snapshot.points),
        len(target_symbols),
        ",".join(affected_targets) or "none",
    )


def _prefer_surface_point(
    candidate: CapturedSurfacePoint,
    existing: CapturedSurfacePoint | None,
) -> bool:
    if existing is None:
        return True
    if candidate.quote_time is None:
        return False
    return existing.quote_time is None or candidate.quote_time > existing.quote_time


def _surface_point_from_exact_quote(
    canonical_occ_symbol: str, quote: Mapping[str, Any]
) -> CapturedSurfacePoint:
    return _surface_point_from_quote(
        canonical_occ_symbol,
        quote,
        observation_source=int(SurfaceObservationSource.EXACT_OPTION_QUOTE_FALLBACK),
    )


def _surface_point_from_quote(
    canonical_occ_symbol: str,
    quote: Mapping[str, Any],
    *,
    observation_source: int,
) -> CapturedSurfacePoint:
    return CapturedSurfacePoint(
        canonical_occ_symbol=canonical_occ_symbol,
        quote_time=_provider_timestamp(
            _first_present(quote, "quoteTimeInLong", "quote_time", "quoteTime")
        ),
        trade_time=_provider_timestamp(
            _first_present(quote, "tradeTimeInLong", "trade_time", "tradeTime")
        ),
        bid=_finite(_first_present(quote, "bid", "bidPrice")),
        ask=_finite(_first_present(quote, "ask", "askPrice")),
        mark=_finite(_first_present(quote, "mark", "markPrice")),
        last=_finite(_first_present(quote, "last", "lastPrice")),
        open_interest=_nonnegative_int(
            _first_present(quote, "openInterest", "open_interest")
        ),
        volume=_nonnegative_int(_first_present(quote, "totalVolume", "volume")),
        observation_source=observation_source,
        security_status=_security_status(
            _first_present(quote, "securityStatus", "security_status")
        ),
    )


def _chain_flags(response: Mapping[str, Any], requested_symbol: str) -> int:
    flags = 0
    if _as_bool(response.get("isDelayed")):
        flags |= int(ChainSnapshotFlag.DELAYED)
    if _as_bool(response.get("isChainTruncated")):
        flags |= int(ChainSnapshotFlag.TRUNCATED)
    if _as_bool(response.get("isIndex")):
        flags |= int(ChainSnapshotFlag.UNDERLYING_IS_INDEX)
    returned_symbol = str(response.get("symbol") or "").strip().upper()
    if returned_symbol != requested_symbol.strip().upper():
        flags |= int(ChainSnapshotFlag.UNDERLYING_SYMBOL_MISMATCH)
    return flags


def _select_chain_spot(
    response: Mapping[str, Any],
) -> tuple[float | None, int | None, datetime | None]:
    underlying = response.get("underlying")
    underlying = underlying if isinstance(underlying, Mapping) else {}
    quote_time = _provider_timestamp(
        _first_present(underlying, "quoteTimeInLong", "quote_time", "quoteTime")
    )
    top_level = _positive(_finite(response.get("underlyingPrice")))
    if top_level is not None:
        return top_level, int(SpotSource.TOP_LEVEL_UNDERLYING_PRICE), quote_time
    nested_mark = _positive(
        _finite(_first_present(underlying, "mark", "markPrice"))
    )
    if nested_mark is not None:
        return nested_mark, int(SpotSource.NESTED_UNDERLYING_MARK), quote_time
    bid = _finite(_first_present(underlying, "bid", "bidPrice"))
    ask = _finite(_first_present(underlying, "ask", "askPrice"))
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        return (
            (bid + ask) / 2.0,
            int(SpotSource.NESTED_BID_ASK_MIDPOINT),
            quote_time,
        )
    nested_last = _positive(
        _finite(_first_present(underlying, "last", "lastPrice"))
    )
    if nested_last is not None:
        return nested_last, int(SpotSource.NESTED_UNDERLYING_LAST), quote_time
    return None, None, quote_time


def _first_present(value: Mapping[str, Any], *keys: str) -> Any:
    return next((value[key] for key in keys if value.get(key) is not None), None)


def _positive(value: float | None) -> float | None:
    return value if value is not None and value > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    number = _finite(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _security_status(value: Any) -> int | None:
    if value is None:
        return None
    status = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if status in {"NORMAL", "ACTIVE", "OPEN"}:
        return int(SecurityStatus.NORMAL)
    if status in {"HALTED", "HALT", "SUSPENDED"}:
        return int(SecurityStatus.HALTED)
    if status in {"CLOSED", "INACTIVE", "EXPIRED", "DELISTED"}:
        return int(SecurityStatus.CLOSED)
    return int(SecurityStatus.UNKNOWN)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    return str(value).strip().lower() in {"true", "t", "yes", "y", "1"}


def _provider_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _aware_utc(value)
    number = _finite(value)
    if number is None or number <= 0:
        return None
    # Schwab/TDA quote timestamps are Unix milliseconds.
    if number > 10_000_000_000:
        number /= 1000.0
    try:
        return datetime.fromtimestamp(number, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _percentage_to_decimal(value: Any) -> float | None:
    number = _finite(value)
    return number / 100.0 if number is not None else None


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("capture clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _mask_account(value: str) -> str:
    text = str(value)
    return "***" + text[-4:] if len(text) > 4 else "***"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture current broker option-position prices into Risk MySQL"
    )
    parser.add_argument("-e", "--environment", default="refdata")
    parser.add_argument(
        "--account",
        action="append",
        help="limit capture to an account returned by get_accounts (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    monitor = None
    exit_code = 1
    monitor_level = NagiosSystemMonitor.MONITOR_CRITICAL
    monitor_message = (
        "Option position price capture failed: startup did not complete; "
        "user intervention required"
    )
    try:
        context = build_context(args.environment, quiet_configuration=True)
        context["tda.oauthlib_logging"] = False
        context["tda.websockets_logging"] = False
        monitor_context = dict(context)
        monitor_context["nagios.service_id"] = (
            str(monitor_context["nagios.service_id"]) + " - Option Position Prices"
        )
        monitor = NagiosSystemMonitor(monitor_context)
        repository = OptionPriceRepository.from_context(context)
        broker = create_tda_broker(context)
        provider = TDAReferenceData(context)
        summary, missing = OptionPriceCapture(
            broker, provider, repository, context
        ).capture(account_ids=args.account, dry_run=args.dry_run)
    except Exception as exc:
        print(f"Option-price capture failed: {type(exc).__name__}", file=sys.stderr)
        monitor_message = (
            "Option position price capture failed: "
            f"error={type(exc).__name__}; user intervention required"
        )
    else:
        summary_message = (
            f"run={summary.run_id} accounts={summary.account_count} "
            f"positions={summary.position_count} quotes={summary.quote_count} "
            f"missing={summary.missing_count} chains={summary.chain_count} "
            f"surface_points={summary.surface_point_count} "
            f"dry_run={str(args.dry_run).lower()}"
        )
        print(summary_message)
        if summary.status != "success":
            if missing:
                print(
                    "Missing current quotes for " + ", ".join(missing),
                    file=sys.stderr,
                )
            else:
                print(
                    "One or more option-chain snapshots were incomplete; "
                    "see preceding option-chain warnings",
                    file=sys.stderr,
                )
            monitor_message = (
                "Option position price capture incomplete: "
                f"{summary_message}; user intervention required"
            )
        else:
            exit_code = 0
            monitor_level = NagiosSystemMonitor.MONITOR_OK
            monitor_message = (
                "Option position price capture completed successfully: "
                + summary_message
            )
        if args.account:
            print("accounts=" + ",".join(_mask_account(item) for item in args.account))
    finally:
        if monitor is not None:
            try:
                monitor.report_event(monitor_level, monitor_message)
            except Exception as exc:
                print(
                    "Option-price capture Nagios reporting failed: "
                    f"{type(exc).__name__}",
                    file=sys.stderr,
                )
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
