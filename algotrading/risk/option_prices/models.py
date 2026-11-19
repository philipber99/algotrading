"""Internal immutable records for captured option-position prices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import IntEnum


# Persist source codes so historical marks retain their selection provenance.
class MarkSource(IntEnum):
    PROVIDER_MARK = 1
    BID_ASK_MIDPOINT = 2
    LAST_TRADE = 3
    CLOSE = 4


class SpotSource(IntEnum):
    TOP_LEVEL_UNDERLYING_PRICE = 1
    NESTED_UNDERLYING_MARK = 2
    NESTED_BID_ASK_MIDPOINT = 3
    NESTED_UNDERLYING_LAST = 4


class ChainSnapshotStatus(IntEnum):
    SUCCESS = 1
    PARTIAL = 2
    FAILED = 3


# Flags are powers of two because one snapshot can carry several conditions.
class ChainSnapshotFlag(IntEnum):
    DELAYED = 1
    TRUNCATED = 2
    UNDERLYING_IS_INDEX = 4
    UNDERLYING_SYMBOL_MISMATCH = 8


class SurfaceObservationSource(IntEnum):
    OPTION_CHAIN = 1
    EXACT_OPTION_QUOTE_FALLBACK = 2


class SecurityStatus(IntEnum):
    UNKNOWN = 0
    NORMAL = 1
    HALTED = 2
    CLOSED = 3


@dataclass(frozen=True)
class OptionPosition:
    account_id: str
    canonical_occ_symbol: str
    provider_option_symbol: str
    underlying_symbol: str
    expiration_date: date
    put_call: str
    strike: float
    signed_quantity: float


@dataclass(frozen=True)
class CapturedOptionPrice:
    position: OptionPosition
    captured_at: datetime
    session_date: date
    quote_time: datetime | None
    trade_time: datetime | None
    bid: float | None
    ask: float | None
    mark: float
    last: float | None
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    underlying_price: float | None
    provider_iv: float | None
    provider_delta: float | None
    provider_gamma: float | None
    provider_theta_per_day: float | None
    provider_vega_per_1pct: float | None
    provider_rho_per_1pct: float | None
    open_interest: float | None
    volume: float | None
    source: str
    mark_source: int | None = None


@dataclass(frozen=True)
class CapturedSurfacePoint:
    canonical_occ_symbol: str
    quote_time: datetime | None
    trade_time: datetime | None
    bid: float | None
    ask: float | None
    mark: float | None
    last: float | None
    open_interest: int | None
    volume: int | None
    observation_source: int
    security_status: int | None


@dataclass(frozen=True)
class CapturedChainSnapshot:
    underlying_symbol: str
    expiration_date: date
    spot: float | None
    spot_source: int | None
    underlying_quote_time: datetime | None
    snapshot_status: int
    snapshot_flags: int
    error_code: str | None
    points: tuple[CapturedSurfacePoint, ...]


@dataclass(frozen=True)
class OptionSurfaceSnapshotHeader:
    """Compact immutable header used by historical surface consumers."""

    snapshot_id: int
    run_id: str
    captured_at: datetime
    underlying_symbol: str
    expiration_date: date
    spot: float | None
    spot_source: int | None
    underlying_quote_time: datetime | None
    snapshot_status: int
    snapshot_flags: int
    error_code: str | None


@dataclass(frozen=True)
class OptionSurfaceObservation:
    """Stored chain observation; contract reference fields stay in the OCC symbol."""

    snapshot_id: int
    canonical_occ_symbol: str
    bid: float | None
    ask: float | None
    mark: float | None
    last: float | None
    quote_time: datetime | None
    trade_time: datetime | None
    observation_source: int
    security_status: int | None


@dataclass(frozen=True)
class OptionSurfaceSnapshot:
    header: OptionSurfaceSnapshotHeader
    observations: tuple[OptionSurfaceObservation, ...]


@dataclass(frozen=True)
class CaptureSummary:
    run_id: str
    captured_at: datetime
    account_count: int
    position_count: int
    quote_count: int
    missing_count: int
    status: str
    provider_class: str
    chain_count: int = 0
    surface_point_count: int = 0
