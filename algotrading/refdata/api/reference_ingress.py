"""Bounded live-option, FRED, and optional CME ingress services.

These adapters intentionally return normalized current/reference values only.
They neither import nor query the options GUI/Calculation service or ClickHouse.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from threading import RLock
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import (
    Request,
    build_opener,
    urlopen,
)
from zoneinfo import ZoneInfo

from algotrading.refdata.providers.cme_auth import CmeOAuthClient

from .cache import SingleFlightTTLCache
from .live_models import (
    FuturesSpecificationItem,
    FuturesSpecificationResponse,
    LiveResponseMeta,
    OptionChainContract,
    OptionChainData,
    OptionChainRequest,
    OptionChainResponse,
    OptionExpirationData,
    OptionExpirationItem,
    OptionExpirationRequest,
    OptionExpirationResponse,
    OptionQuoteItem,
    OptionContractTermsItem,
    OptionContractTermsResponse,
    OptionsQuoteResponse,
    RiskFreeAnchor,
    RiskFreeCurveData,
    RiskFreeCurveHistoryResponse,
    RiskFreeCurveResponse,
)
from .option_symbols import build_occ_symbol, parse_occ_symbol
from .service import InvalidRequestError, UpstreamUnavailableError


class FeatureDisabledError(UpstreamUnavailableError):
    code = "FEATURE_DISABLED"


class FredDisabledError(FeatureDisabledError):
    code = "FRED_DISABLED"


class CmeReferenceDisabledError(FeatureDisabledError):
    code = "CME_REFERENCE_DISABLED"


class OptionChainDisabledError(FeatureDisabledError):
    code = "OPTIONS_CHAIN_DISABLED"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_LISTED_FUTURE_CONTRACT = re.compile(
    r"^(?P<root>[A-Z0-9]{1,4})(?P<month>[FGHJKMNQUVXZ])(?P<year>[0-9]{1,2})$"
)


def _cme_globex_contract(contract: str) -> str:
    """Translate the broker's two-digit year to CME Globex notation."""
    match = _LISTED_FUTURE_CONTRACT.fullmatch(contract)
    if match is None or len(match.group("year")) == 1:
        return contract
    return f"{match.group('root')}{match.group('month')}{match.group('year')[-1]}"


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) <= 0:
            return None
        seconds = float(value) / 1000.0 if abs(float(value)) > 10_000_000_000 else float(value)
        try:
            return datetime.fromtimestamp(seconds, timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class LatestResponseCache:
    """Small SQLite latest-value cache with checksummed JSON payloads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_latest_cache (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    source_as_of TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, cache_key)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_attempt_ledger (
                    namespace TEXT NOT NULL,
                    resource_key TEXT NOT NULL,
                    last_attempt_at TEXT NOT NULL,
                    last_success_at TEXT NULL,
                    last_error TEXT NULL,
                    PRIMARY KEY (namespace, resource_key)
                )
                """
            )

    def load(self, namespace: str, cache_key: str) -> tuple[dict[str, Any], datetime] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload, retrieved_at FROM reference_latest_cache "
                "WHERE namespace = ? AND cache_key = ? AND schema_version = 1",
                (namespace, cache_key),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(str(row["payload"]))
            retrieved = _timestamp(row["retrieved_at"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or retrieved is None:
            return None
        return payload, retrieved

    def store(
        self,
        namespace: str,
        cache_key: str,
        payload: Mapping[str, Any],
        *,
        source_as_of: datetime,
        retrieved_at: datetime,
    ) -> None:
        # Serialize deterministically so stored payload checksums are reproducible.
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO reference_latest_cache
                    (namespace, cache_key, schema_version, payload, source_as_of, retrieved_at)
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    payload = excluded.payload,
                    source_as_of = excluded.source_as_of,
                    retrieved_at = excluded.retrieved_at
                """,
                (
                    namespace,
                    cache_key,
                    encoded,
                    source_as_of.astimezone(timezone.utc).isoformat(),
                    retrieved_at.astimezone(timezone.utc).isoformat(),
                ),
            )

    def attempt_state(self, namespace: str, resource_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM reference_attempt_ledger WHERE namespace = ? AND resource_key = ?",
                (namespace, resource_key),
            ).fetchone()
        return dict(row) if row is not None else None

    def record_attempt(
        self,
        namespace: str,
        resource_key: str,
        *,
        at: datetime,
        success: bool,
        error: str | None = None,
    ) -> None:
        safe_error = str(error).replace("\n", " ")[:256] if error else None
        value = at.astimezone(timezone.utc).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO reference_attempt_ledger
                    (namespace, resource_key, last_attempt_at, last_success_at, last_error)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, resource_key) DO UPDATE SET
                    last_attempt_at=excluded.last_attempt_at,
                    last_success_at=CASE WHEN excluded.last_success_at IS NOT NULL
                        THEN excluded.last_success_at ELSE reference_attempt_ledger.last_success_at END,
                    last_error=excluded.last_error
                """,
                (namespace, resource_key, value, value if success else None, safe_error),
            )


def _cache_path(context: Mapping[str, Any]) -> Path:
    configured = str(context.get("refdata.api.cache.reference_path", "") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute() and context.get("AlgoHome"):
            path = Path(str(context["AlgoHome"])) / path
        return path
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "algotrading" / "refdata-api-reference.sqlite3"


class LiveOptionService:
    def __init__(
        self,
        context: Mapping[str, Any],
        provider_getter: Callable[[], Any],
        *,
        now_fn: Callable[[], datetime] = _utc_now,
        market_open_fn: Callable[[], bool] | None = None,
        latest_completed_session_close_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.context = context
        self._provider_getter = provider_getter
        self._now = now_fn
        self._market_open = market_open_fn
        self._latest_completed_session_close = latest_completed_session_close_fn
        self._cache = SingleFlightTTLCache[dict[str, Any]](
            max_entries=int(context.get("refdata.api.cache.max_entries", 1024))
        )

    @property
    def option_chain_ready(self) -> bool:
        if not _as_bool(self.context.get("refdata.api.options_chain.enabled", True)):
            return False
        try:
            return callable(getattr(self._provider_getter(), "get_options_chain", None))
        except Exception:
            return False

    @property
    def option_expirations_ready(self) -> bool:
        if not _as_bool(self.context.get("refdata.api.options_chain.enabled", True)):
            return False
        try:
            provider = self._provider_getter()
            return (
                callable(getattr(provider, "get_option_expiration_chain", None))
                and getattr(provider, "option_expiration_chain_uri", True) is not None
            )
        except Exception:
            return False

    @property
    def option_quotes_ready(self) -> bool:
        try:
            return callable(getattr(self._provider_getter(), "get_options_quotes", None))
        except Exception:
            return False

    def option_quotes(self, symbols: Sequence[str]) -> OptionsQuoteResponse:
        maximum = int(self.context.get("refdata.api.max_option_quote_symbols", 500))
        if len(symbols) > maximum:
            raise InvalidRequestError(
                f"option quote request contains {len(symbols)} contracts; maximum is {maximum}"
            )
        provider = self._provider_getter()
        provider_name = type(provider).__name__
        parsed: list[tuple[str, Any | None, str | None]] = []
        for requested in symbols:
            try:
                parsed.append((requested, parse_occ_symbol(requested), None))
            except ValueError as exc:
                parsed.append((requested, None, str(exc)))
        # Deduplicate provider symbols while preserving per-request validation results.
        provider_symbols = tuple(
            dict.fromkeys(item.provider for _, item, _ in parsed if item is not None)
        )
        loaded_at = self._now().astimezone(timezone.utc)
        completed_session_close = self._completed_session_close()
        cache_state = "bypass"
        raw: Mapping[str, Any] = {}
        provider_error: str | None = None
        if provider_symbols:
            ttl = float(self.context.get("refdata.api.cache.option_quotes_ttl_seconds", 2))

            def load() -> dict[str, Any]:
                method = getattr(provider, "get_options_quotes", None)
                if not callable(method):
                    raise UpstreamUnavailableError(
                        "The configured reference-data provider does not support exact option quotes"
                    )
                result = method(list(provider_symbols))
                if not isinstance(result, Mapping):
                    raise UpstreamUnavailableError("The option quote provider returned an invalid response")
                return dict(result)

            try:
                lookup = self._cache.get_or_load(("option_quotes", provider_symbols), ttl, load)
                raw = lookup.value
                cache_state = "hit" if lookup.hit else "miss"
            except UpstreamUnavailableError:
                raise
            except Exception as exc:
                provider_error = type(exc).__name__

        items: list[OptionQuoteItem] = []
        missing: list[str] = []
        for requested, occ, validation_error in parsed:
            if occ is None:
                items.append(
                    OptionQuoteItem(
                        requested_symbol=requested,
                        status="invalid",
                        source=provider_name,
                        error=validation_error,
                        warnings=["invalid_occ_symbol"],
                    )
                )
                missing.append(requested)
                continue
            source = raw.get(occ.provider)
            if not isinstance(source, Mapping):
                status = "provider_error" if provider_error else "missing"
                items.append(
                    OptionQuoteItem(
                        requested_symbol=requested,
                        canonical_occ_symbol=occ.canonical,
                        provider_option_symbol=occ.provider,
                        status=status,
                        source=provider_name,
                        error=provider_error,
                        warnings=["provider_omitted_contract" if not provider_error else "provider_error"],
                    )
                )
                missing.append(occ.canonical)
                continue
            close = _finite(source.get("close"))
            # Outside regular trading, prefer the completed-session close as the risk mark.
            use_completed_close = (
                completed_session_close is not None
                and close is not None
                and close > 0
            )
            quote_time = (
                completed_session_close
                if use_completed_close
                else _timestamp(source.get("quote_time"))
            )
            trade_time = (
                None
                if use_completed_close
                else _timestamp(source.get("trade_time"))
            )
            freshness_basis = max(
                (item for item in (quote_time, trade_time) if item is not None),
                default=loaded_at,
            )
            raw_iv = _finite(source.get("volatility"))
            warnings: list[str] = []
            if raw_iv is None:
                warnings.append("provider_iv_missing")
            elif raw_iv <= 0:
                warnings.append("provider_iv_unusable")
            if use_completed_close:
                warnings.append("previous_completed_session_close_used")
            # Convert provider IV percent to decimal units only for current quote evidence.
            provider_iv = (
                raw_iv / 100.0
                if not use_completed_close and raw_iv is not None and raw_iv > 0
                else None
            )
            items.append(
                OptionQuoteItem(
                    requested_symbol=requested,
                    canonical_occ_symbol=occ.canonical,
                    provider_option_symbol=occ.provider,
                    status="ok",
                    source=provider_name,
                    bid=None if use_completed_close else _finite(source.get("bid")),
                    ask=None if use_completed_close else _finite(source.get("ask")),
                    mark=close if use_completed_close else _finite(source.get("mark")),
                    last=None if use_completed_close else _finite(source.get("last")),
                    bid_size=_finite(source.get("bid_size")),
                    ask_size=_finite(source.get("ask_size")),
                    last_size=_finite(source.get("last_size")),
                    open=_finite(source.get("open")),
                    high=_finite(source.get("high")),
                    low=_finite(source.get("low")),
                    close=close,
                    quote_time=quote_time,
                    trade_time=trade_time,
                    source_kind=(
                        "completed_session_mark"
                        if use_completed_close
                        else "exchange_quote"
                    ),
                    underlying_price=(
                        None
                        if use_completed_close
                        else _finite(source.get("underlying_price"))
                    ),
                    intrinsic_value=(
                        None
                        if use_completed_close
                        else _finite(source.get("intrinsic_value"))
                    ),
                    time_value=(
                        None
                        if use_completed_close
                        else _finite(source.get("time_value"))
                    ),
                    provider_iv=provider_iv,
                    provider_iv_raw=raw_iv,
                    provider_delta=(
                        None if use_completed_close else _finite(source.get("delta"))
                    ),
                    provider_gamma=(
                        None if use_completed_close else _finite(source.get("gamma"))
                    ),
                    provider_theta_per_day=(
                        None if use_completed_close else _finite(source.get("theta"))
                    ),
                    provider_vega_per_1pct=(
                        None if use_completed_close else _finite(source.get("vega"))
                    ),
                    provider_rho_per_1pct=(
                        None if use_completed_close else _finite(source.get("rho"))
                    ),
                    theoretical_option_value=(
                        None
                        if use_completed_close
                        else _finite(source.get("theoretical_option_value"))
                    ),
                    implied_yield_raw=_finite(source.get("implied_yield")),
                    open_interest=_finite(source.get("open_interest")),
                    volume=_finite(source.get("total_volume")),
                    security_status=(str(source.get("security_status")) if source.get("security_status") is not None else None),
                    net_change=_finite(source.get("net_change")),
                    mark_change=_finite(source.get("mark_change")),
                    mark_percent_change=_finite(source.get("mark_percent_change")),
                    percent_change=_finite(source.get("percent_change")),
                    multiplier=_finite(
                        source.get("multiplier")
                        or source.get("contract_multiplier")
                    ),
                    deliverable=(
                        str(
                            source.get("deliverable")
                            or source.get("optionDeliverablesList")
                        )
                        if source.get("deliverable") is not None
                        or source.get("optionDeliverablesList") is not None
                        else None
                    ),
                    deliverable_components=_option_deliverable_components(source),
                    aggregate_exercise_amount=_finite(
                        source.get("aggregate_exercise_amount")
                        or source.get("aggregateExerciseAmount")
                    ),
                    asset_subtype=_text(
                        source.get("asset_subtype") or source.get("optionSubType")
                    ),
                    adjusted_contract=source.get("adjusted_contract"),
                    shares_per_contract=(
                        _finite(source.get("shares_per_contract"))
                        or _delivered_security_quantity(source)
                    ),
                    cash_per_contract=(
                        _finite(source.get("cash_per_contract"))
                        if source.get("cash_per_contract") is not None
                        else _delivered_cash_amount(source)
                    ),
                    exercise_cutoff_at=_timestamp(source.get("exercise_cutoff_at")),
                    last_trade_at=_timestamp(
                        source.get("last_trade_at") or source.get("lastTradingDay")
                    ),
                    settlement_at=_timestamp(
                        source.get("settlement_at") or source.get("settlementDateTime")
                    ),
                    exchange_calendar=_text(source.get("exchange_calendar")),
                    exchange_timezone=_text(source.get("exchange_timezone")),
                    calendar_version=_text(source.get("calendar_version")),
                    exercise_style=_text(
                        source.get("exercise_style") or source.get("exerciseType")
                    ),
                    settlement_style=_text(
                        source.get("settlement_style") or source.get("settlementType")
                    ),
                    settlement_type=_text(source.get("settlement_type")),
                    expiration_at=_timestamp(
                        source.get("expiration_at")
                        or source.get("expirationDateTime")
                    ),
                    source_units={
                        "provider_iv_raw": "percentage_points",
                        "provider_iv": "decimal",
                        "provider_theta_per_day": "option_price_per_calendar_day",
                        "provider_vega_per_1pct": "option_price_per_1_volatility_point",
                        "provider_rho_per_1pct": "option_price_per_1_rate_point",
                    },
                    stale_after=freshness_basis + timedelta(seconds=60),
                    warnings=warnings,
                )
            )
        return OptionsQuoteResponse(
            data=items,
            missing_symbols=missing,
            meta=LiveResponseMeta(
                as_of=loaded_at,
                source=provider_name,
                cache=cache_state,
                warnings=([f"provider_error:{provider_error}"] if provider_error else []),
            ),
        )

    def _completed_session_close(self) -> datetime | None:
        """Return an EOD anchor only while the regular market is closed."""

        if self._market_open is None or self._latest_completed_session_close is None:
            return None
        try:
            if self._market_open():
                return None
            value = self._latest_completed_session_close()
        except Exception:
            return None
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(
            tzinfo=timezone.utc
        )

    def _option_root_metadata(
        self,
        provider: Any,
        occ: Any,
    ) -> tuple[dict[str, Any], bool]:
        """Load one minimal chain sample and cache terms shared by a root."""

        ttl = float(
            self.context.get(
                "refdata.api.cache.option_contract_metadata_ttl_seconds",
                86400,
            )
        )

        def load() -> dict[str, Any]:
            method = getattr(provider, "get_options_chain", None)
            if not callable(method):
                raise UpstreamUnavailableError(
                    "The configured reference-data provider does not support option metadata"
                )
            # Use a minimal chain sample to discover metadata shared by the option root.
            result = method(
                occ.root,
                contract_type="ALL",
                strategy="SINGLE",
                strike_count=1,
                include_underlying_quote=False,
                from_date=occ.expiration,
                to_date=occ.expiration,
            )
            if not isinstance(result, Mapping):
                raise UpstreamUnavailableError(
                    "The option-chain provider returned no metadata snapshot"
                )
            contract: Mapping[str, Any] | None = None
            for map_name in ("callExpDateMap", "putExpDateMap"):
                expiration_map = result.get(map_name)
                if not isinstance(expiration_map, Mapping):
                    continue
                for strike_map in expiration_map.values():
                    if not isinstance(strike_map, Mapping):
                        continue
                    for values in strike_map.values():
                        if not isinstance(values, Sequence) or isinstance(
                            values, (str, bytes)
                        ):
                            continue
                        contract = next(
                            (item for item in values if isinstance(item, Mapping)),
                            None,
                        )
                        if contract is not None:
                            break
                    if contract is not None:
                        break
                if contract is not None:
                    break
            if contract is None:
                raise UpstreamUnavailableError(
                    "The option-chain provider returned no contract metadata"
                )
            multiplier = _finite(contract.get("multiplier"))
            exercise_style = _text(
                contract.get("exerciseType") or contract.get("exercise_style")
            )
            settlement_style = _option_settlement_style(
                contract.get("settlementType")
                or contract.get("settlement_style")
            )
            if (
                multiplier is None
                and exercise_style is None
                and settlement_style is None
            ):
                raise UpstreamUnavailableError(
                    "The option-chain provider omitted exercise, multiplier, and settlement metadata"
                )
            return {
                "multiplier": multiplier,
                "exercise_style": exercise_style,
                "settlement_style": settlement_style,
                "retrieved_at": self._now().astimezone(timezone.utc),
            }

        lookup = self._cache.get_or_load(
            ("option_contract_metadata", occ.root),
            ttl,
            load,
        )
        return lookup.value, lookup.hit

    def option_contract_terms(
        self, symbols: Sequence[str]
    ) -> OptionContractTermsResponse:
        """Resolve exact terms from one cached, minimal chain sample per root."""

        now = self._now().astimezone(timezone.utc)
        provider = self._provider_getter()
        provider_name = type(provider).__name__
        parsed: dict[str, Any] = {}
        first_by_root: dict[str, Any] = {}
        for requested in symbols:
            try:
                occ = parse_occ_symbol(requested)
            except ValueError:
                continue
            parsed[requested] = occ
            first_by_root.setdefault(occ.root, occ)
        metadata_by_root: dict[str, dict[str, Any]] = {}
        metadata_errors: dict[str, str] = {}
        metadata_hits: list[bool] = []
        # Load root metadata once for all requested contracts on that underlying.
        for root, occ in first_by_root.items():
            try:
                metadata, hit = self._option_root_metadata(provider, occ)
                metadata_by_root[root] = metadata
                metadata_hits.append(hit)
            except Exception as exc:
                metadata_errors[root] = type(exc).__name__

        quote_error: str | None = None
        try:
            quotes = self.option_quotes(symbols)
            by_symbol = {
                item.canonical_occ_symbol: item
                for item in quotes.data
                if item.canonical_occ_symbol
            }
            quote_meta = quotes.meta
        except Exception as exc:
            # Exact term resolution remains available in explicitly audited OCC
            # fallback mode even when the live quote category is disabled.
            by_symbol = {}
            quote_error = type(exc).__name__
            quote_meta = LiveResponseMeta(
                as_of=now,
                source="audited_occ_standard_assumption",
                cache="bypass",
                warnings=[f"option_quote_terms_source_unavailable:{quote_error}"],
            )
        allow_assumption = _as_bool(
            self.context.get(
                "refdata.api.option_terms.allow_standard_occ_assumption", True
            )
        )
        timezone_name = str(
            self.context.get(
                "refdata.api.option_terms.standard_exchange_timezone",
                "America/New_York",
            )
        )
        pm_settlement_time = _local_time(
            self.context.get(
                "refdata.api.option_terms.standard_expiration_local_time", "16:00"
            )
        )
        am_settlement_time = _local_time(
            self.context.get(
                "refdata.api.option_terms.standard_am_settlement_local_time", "09:30"
            )
        )
        output: list[OptionContractTermsItem] = []
        for requested in symbols:
            occ = parsed.get(requested)
            if occ is None:
                output.append(
                    OptionContractTermsItem(
                        requested_symbol=requested,
                        terms_status="unsupported",
                        asset_subtype="unknown",
                        source="occ_parser",
                        warnings=["invalid_occ_symbol"],
                    )
                )
                continue
            quote_item = by_symbol.get(occ.canonical)
            root_metadata = metadata_by_root.get(occ.root, {})
            chain_multiplier = _finite(root_metadata.get("multiplier"))
            multiplier = chain_multiplier or _finite(
                getattr(quote_item, "multiplier", None)
                if quote_item is not None
                else None
            )
            deliverable = (
                getattr(quote_item, "deliverable", None)
                if quote_item is not None
                else None
            )
            components = list(
                getattr(quote_item, "deliverable_components", ()) or ()
            ) if quote_item is not None else []
            explicit_adjusted = (
                getattr(quote_item, "adjusted_contract", None)
                if quote_item is not None else None
            )
            shares_from_terms = _finite(
                getattr(quote_item, "shares_per_contract", None)
                if quote_item is not None else None
            )
            cash_from_terms = _finite(
                getattr(quote_item, "cash_per_contract", None)
                if quote_item is not None else None
            )
            # Explicit adjusted-delivery evidence prevents a standard 100-share assumption.
            evidence_adjusted = bool(explicit_adjusted) or (
                multiplier is not None and not math.isclose(multiplier, 100.0)
            ) or bool(cash_from_terms) or any(
                isinstance(item, Mapping)
                and str(item.get("component_type") or "").lower() == "other"
                for item in components
            )
            warnings = list(getattr(quote_item, "warnings", ()) or ())
            if quote_error:
                warnings.append(f"option_quote_terms_source_unavailable:{quote_error}")
            metadata_error = metadata_errors.get(occ.root)
            if metadata_error:
                warnings.append(
                    f"option_chain_metadata_unavailable:{metadata_error}"
                )
            settlement_style = _option_settlement_style(
                root_metadata.get("settlement_style")
                or (
                    getattr(quote_item, "settlement_style", None)
                    if quote_item is not None
                    else None
                )
            )
            if settlement_style is None:
                warnings.append("option_settlement_style_unavailable")
            # Combine OCC expiry dates with the resolved settlement session and timezone.
            expiration = (
                getattr(quote_item, "expiration_at", None)
                if quote_item is not None
                else None
            )
            if expiration is None:
                expiration = datetime.combine(
                    occ.expiration,
                    am_settlement_time if settlement_style == "AM" else pm_settlement_time,
                    ZoneInfo(timezone_name),
                ).astimezone(timezone.utc)
                warnings.append(
                    "option_expiration_derived_from_occ_and_settlement_session"
                )
            if chain_multiplier is not None:
                adjusted = not math.isclose(chain_multiplier, 100.0)
                terms_status = "adjusted" if adjusted else "standard"
                shares = shares_from_terms or chain_multiplier
            elif evidence_adjusted:
                terms_status = "adjusted"
                adjusted = True
                shares = shares_from_terms
            elif multiplier is not None and explicit_adjusted is False:
                terms_status = "standard"
                adjusted = False
                shares = shares_from_terms
            elif allow_assumption:
                terms_status = "partial"
                multiplier = shares = 100.0
                adjusted = False
                warnings.append("option_terms_assumed_standard_occ")
            else:
                terms_status = "partial" if quote_item is not None else "unsupported"
                adjusted = None
                shares = None
                warnings.append("option_multiplier_unavailable")
            exercise = str(
                root_metadata.get("exercise_style")
                or getattr(quote_item, "exercise_style", None)
                or "unknown"
            ).lower()
            if exercise in {"a", "american"}:
                exercise = "american"
            elif exercise in {"e", "european"}:
                exercise = "european"
            else:
                exercise = "unknown"
                warnings.append("option_exercise_style_unavailable")
            settlement_raw = str(
                getattr(quote_item, "settlement_type", None) or "unknown"
            ).lower()
            settlement = (
                "cash" if "cash" in settlement_raw else
                "physical" if "physical" in settlement_raw else "unknown"
            )
            if settlement == "unknown":
                settlement = "cash"
                warnings.append("option_settlement_type_assumed_cash")
            subtype_raw = str(
                getattr(quote_item, "asset_subtype", None) or "unknown"
            ).lower()
            subtype = subtype_raw if subtype_raw in {
                "equity_option", "etf_option", "index_option", "futures_option"
            } else "unknown"
            if subtype == "unknown":
                warnings.append("option_contract_subtype_unavailable")
            if exercise == "unknown":
                if terms_status in {"standard", "adjusted"}:
                    terms_status = "partial"
            if settlement == "cash":
                shares = None
                components = []
            elif not components and shares is not None:
                components = [{
                    "component_type": "security",
                    "security_identifier": occ.root,
                    "quantity": shares,
                    "currency": "USD",
                    "description": f"{shares:g} {occ.root} shares",
                }]
            output.append(
                OptionContractTermsItem(
                    requested_symbol=requested,
                    canonical_occ_symbol=occ.canonical,
                    terms_status=terms_status,
                    asset_subtype=subtype,
                    underlying_identifier=occ.root,
                    price_multiplier=multiplier,
                    quote_convention="usd_per_share",
                    strike=float(occ.strike),
                    put_call=occ.put_call,
                    exercise_style=exercise,
                    deliverable=(
                        str(deliverable)
                        if deliverable
                        else (
                            f"cash-settled intrinsic value x {multiplier:g}"
                            if settlement == "cash" and multiplier is not None
                            else f"{shares:g} shares" if shares is not None else None
                        )
                    ),
                    shares_per_contract=shares,
                    cash_per_contract=cash_from_terms,
                    deliverable_components=components,
                    aggregate_exercise_amount=(
                        float(occ.strike) * multiplier
                        if settlement == "cash" and multiplier is not None
                        else _finite(getattr(quote_item, "aggregate_exercise_amount", None))
                        if quote_item is not None
                        else None
                    ),
                    adjusted_contract=adjusted,
                    listing_date=(
                        getattr(quote_item, "listing_date", None)
                        if quote_item is not None
                        else None
                    ),
                    expiration_at=expiration,
                    last_trade_at=(getattr(quote_item, "last_trade_at", None) if quote_item is not None else None),
                    settlement_at=(getattr(quote_item, "settlement_at", None) if quote_item is not None else None),
                    settlement_style=settlement_style,
                    exchange_timezone=(
                        str(getattr(quote_item, "exchange_timezone", None))
                        if quote_item is not None and getattr(quote_item, "exchange_timezone", None)
                        else timezone_name
                    ),
                    settlement_type=settlement,
                    exercise_cutoff_at=(getattr(quote_item, "exercise_cutoff_at", None) if quote_item is not None else None),
                    exchange_calendar=(getattr(quote_item, "exchange_calendar", None) if quote_item is not None else None),
                    calendar_version=(getattr(quote_item, "calendar_version", None) if quote_item is not None else None),
                    source=(
                        provider_name
                        if root_metadata
                        else getattr(quote_item, "source", None)
                        if quote_item is not None
                        else "audited_occ_standard_assumption"
                    ) or "audited_occ_standard_assumption",
                    source_as_of=(
                        _timestamp(root_metadata.get("retrieved_at"))
                        or quote_meta.as_of
                    ),
                    source_version="refdata_exact_option_terms_v4",
                    warnings=list(dict.fromkeys(warnings)),
                )
            )
        return OptionContractTermsResponse(
            data=output,
            meta=LiveResponseMeta(
                as_of=now,
                source=provider_name if metadata_by_root else quote_meta.source,
                cache=(
                    "hit"
                    if metadata_hits and all(metadata_hits)
                    else "miss"
                    if metadata_hits
                    else quote_meta.cache
                ),
                warnings=list(quote_meta.warnings) + [
                    f"{root}:option_chain_metadata_unavailable:{error}"
                    for root, error in metadata_errors.items()
                ],
            ),
        )

    def option_expirations(
        self, request: OptionExpirationRequest
    ) -> OptionExpirationResponse:
        if not self.option_expirations_ready:
            raise OptionChainDisabledError(
                "Live option-expiration capability is disabled or unavailable"
            )
        provider = self._provider_getter()
        provider_name = type(provider).__name__
        now = self._now().astimezone(timezone.utc)
        if request.from_date < now.date():
            raise InvalidRequestError("from_date may not be before the current provider date")
        max_span = int(
            self.context.get("refdata.api.option_expirations.max_date_span_days", 550)
        )
        if (request.to_date - request.from_date).days > max_span:
            raise InvalidRequestError(
                f"option-expiration date span exceeds {max_span} days"
            )
        key = (
            "option_expirations",
            request.symbol,
            request.from_date.isoformat(),
            request.to_date.isoformat(),
        )
        ttl = float(
            self.context.get("refdata.api.cache.option_expirations_ttl_seconds", 300)
        )

        def load() -> dict[str, Any]:
            result = provider.get_option_expiration_chain(request.symbol)
            expiration_list = result.get("expirationList") if isinstance(result, Mapping) else None
            if not isinstance(expiration_list, Sequence) or isinstance(
                expiration_list, (str, bytes)
            ):
                raise UpstreamUnavailableError(
                    "The option-expiration provider returned no usable calendar"
                )
            return dict(result)

        lookup = self._cache.get_or_load(key, ttl, load)
        expirations: dict[date, OptionExpirationItem] = {}
        for source in lookup.value.get("expirationList") or []:
            # Ignore malformed expiration records before constructing the public calendar.
            if not isinstance(source, Mapping):
                continue
            try:
                expiration = date.fromisoformat(str(source.get("expirationDate")))
            except ValueError:
                continue
            if not request.from_date <= expiration <= request.to_date:
                continue
            raw_days = _finite(source.get("daysToExpiration"))
            raw_roots = source.get("optionRoots")
            if isinstance(raw_roots, str):
                roots = [item.strip() for item in raw_roots.split(",") if item.strip()]
            elif isinstance(raw_roots, Sequence):
                roots = [str(item).strip() for item in raw_roots if str(item).strip()]
            else:
                roots = []
            expirations.setdefault(
                expiration,
                OptionExpirationItem(
                    expiration=expiration,
                    days_to_expiration=(
                        max(0, int(raw_days)) if raw_days is not None else None
                    ),
                    expiration_type=_text(source.get("expirationType")),
                    settlement_type=_text(source.get("settlementType")),
                    option_roots=roots,
                    standard=(
                        _as_bool(source.get("standard"))
                        if source.get("standard") is not None
                        else None
                    ),
                ),
            )
        return OptionExpirationResponse(
            data=OptionExpirationData(
                symbol=request.symbol,
                from_date=request.from_date,
                to_date=request.to_date,
                expirations=[expirations[value] for value in sorted(expirations)],
                retrieved_at=now,
            ),
            meta=LiveResponseMeta(
                as_of=now,
                source=provider_name,
                cache="hit" if lookup.hit else "miss",
                warnings=[],
            ),
        )

    def option_chain(self, request: OptionChainRequest) -> OptionChainResponse:
        if not self.option_chain_ready:
            raise OptionChainDisabledError("Live option-chain capability is disabled or unavailable")
        provider = self._provider_getter()
        provider_name = type(provider).__name__
        now = self._now().astimezone(timezone.utc)
        if request.from_date < now.date():
            raise InvalidRequestError("from_date may not be before the current provider date")
        max_span = int(self.context.get("refdata.api.options_chain.max_date_span_days", 95))
        max_strikes = int(self.context.get("refdata.api.options_chain.max_strike_count", 120))
        if (request.to_date - request.from_date).days > max_span:
            raise InvalidRequestError(f"option-chain date span exceeds {max_span} days")
        if request.strike_count > max_strikes:
            raise InvalidRequestError(f"strike_count exceeds {max_strikes}")
        key = (
            "option_chain",
            request.symbol,
            request.contract_type,
            request.strike_count,
            request.from_date.isoformat(),
            request.to_date.isoformat(),
            request.include_underlying_quote,
        )
        ttl = float(self.context.get("refdata.api.cache.option_chain_ttl_seconds", 2))

        def load() -> dict[str, Any]:
            result = provider.get_options_chain(
                request.symbol,
                contract_type=request.contract_type,
                strategy="SINGLE",
                strike_count=request.strike_count,
                include_underlying_quote=request.include_underlying_quote,
                from_date=request.from_date,
                to_date=request.to_date,
            )
            if not isinstance(result, Mapping):
                raise UpstreamUnavailableError("The option-chain provider returned no usable snapshot")
            encoded_size = len(json.dumps(result, default=str).encode("utf-8"))
            maximum_bytes = int(self.context.get("refdata.api.options_chain.max_response_bytes", 5_242_880))
            if encoded_size > maximum_bytes:
                raise UpstreamUnavailableError("The option-chain provider response exceeded the configured byte limit")
            return dict(result)

        lookup = self._cache.get_or_load(key, ttl, load)
        raw = lookup.value
        contracts = self._flatten_chain(raw, request)
        # Bound flattened chain size before constructing the public response.
        maximum_contracts = int(self.context.get("refdata.api.options_chain.max_contracts_in_response", 2000))
        if len(contracts) > maximum_contracts:
            raise UpstreamUnavailableError(
                f"The option-chain provider returned {len(contracts)} contracts; maximum is {maximum_contracts}"
            )
        provider_snapshot = _timestamp(raw.get("quoteTimeInLong") or raw.get("quoteTime"))
        return OptionChainResponse(
            data=OptionChainData(
                symbol=request.symbol,
                contract_type=request.contract_type,
                strike_count=request.strike_count,
                from_date=request.from_date,
                to_date=request.to_date,
                underlying_quote=(dict(raw["underlying"]) if isinstance(raw.get("underlying"), Mapping) else None),
                contracts=contracts,
                provider_snapshot_time=provider_snapshot,
                retrieved_at=now,
            ),
            meta=LiveResponseMeta(
                as_of=provider_snapshot or now,
                source=provider_name,
                cache="hit" if lookup.hit else "miss",
                warnings=[],
            ),
        )

    def _flatten_chain(
        self, raw: Mapping[str, Any], request: OptionChainRequest
    ) -> list[OptionChainContract]:
        rows: list[OptionChainContract] = []
        sides = (("CALL", "callExpDateMap"), ("PUT", "putExpDateMap"))
        for put_call, field in sides:
            expiration_map = raw.get(field)
            if not isinstance(expiration_map, Mapping):
                continue
            for expiry_key, strike_map in expiration_map.items():
                if not isinstance(strike_map, Mapping):
                    continue
                try:
                    expiration = date.fromisoformat(str(expiry_key).split(":", 1)[0])
                except ValueError:
                    continue
                if not request.from_date <= expiration <= request.to_date:
                    continue
                for strike_key, values in strike_map.items():
                    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                        continue
                    for source in values:
                        if not isinstance(source, Mapping):
                            continue
                        strike = _finite(source.get("strikePrice"))
                        if strike is None:
                            strike = _finite(strike_key)
                        if strike is None:
                            continue
                        canonical = provider_symbol = None
                        symbol = source.get("symbol")
                        try:
                            occ = parse_occ_symbol(str(symbol)) if symbol else build_occ_symbol(request.symbol, expiration, put_call, strike)
                            canonical, provider_symbol = occ.canonical, occ.provider
                        except ValueError:
                            pass
                        rows.append(
                            OptionChainContract(
                                canonical_occ_symbol=canonical,
                                provider_option_symbol=provider_symbol,
                                put_call=put_call,
                                expiration=expiration,
                                strike=strike,
                                multiplier=_finite(source.get("multiplier")),
                                deliverable=(str(source.get("optionDeliverablesList")) if source.get("optionDeliverablesList") is not None else None),
                                bid=_finite(source.get("bid")),
                                ask=_finite(source.get("ask")),
                                mark=_finite(source.get("mark")),
                                last=_finite(source.get("last")),
                                quote_time=_timestamp(source.get("quoteTimeInLong")),
                                trade_time=_timestamp(source.get("tradeTimeInLong")),
                                volume=_finite(source.get("totalVolume")),
                                open_interest=_finite(source.get("openInterest")),
                                provider_iv_raw=_finite(source.get("volatility")),
                                provider_delta=_finite(source.get("delta")),
                                provider_gamma=_finite(source.get("gamma")),
                                provider_theta_raw=_finite(source.get("theta")),
                                provider_vega_raw=_finite(source.get("vega")),
                                provider_rho_raw=_finite(source.get("rho")),
                                warnings=["provider_fields_omitted"] if canonical is None else [],
                            )
                        )
        rows.sort(key=lambda item: (item.expiration, item.strike, item.put_call))
        return rows


class FredRiskFreeCurveService:
    CURVE_ID = "fred_usd_front_curve_v1"
    CURVE_VERSION = "fred_simple_annual_anchors_v1"

    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        now_fn: Callable[[], datetime] = _utc_now,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.context = context
        self._now = now_fn
        self._opener = opener or urlopen
        self._cache = SingleFlightTTLCache[dict[str, Any]](max_entries=64)
        self._persistent = LatestResponseCache(_cache_path(context)) if self.ready else None

    @property
    def ready(self) -> bool:
        return bool(
            _as_bool(self.context.get("refdata.fred.enabled", True))
            and str(self.context.get("refdata.fred.api_key", "") or "").strip()
        )

    def series_tenors(self) -> tuple[tuple[str, int], ...]:
        value = self.context.get(
            "refdata.fred.series_tenors",
            "SOFR:0,DGS1MO:30,DGS3MO:90,DGS6MO:180,DGS1:365",
        )
        if isinstance(value, Mapping):
            pairs = [(str(key).strip().upper(), int(days)) for key, days in value.items()]
        else:
            pairs = []
            for item in str(value).split(","):
                series, separator, days = item.strip().partition(":")
                if not separator:
                    raise ValueError("refdata.fred.series_tenors must use SERIES:DAYS entries")
                pairs.append((series.strip().upper(), int(days)))
        if not pairs or any(not series or days < 0 for series, days in pairs):
            raise ValueError("refdata.fred.series_tenors is invalid")
        return tuple(pairs)

    def curve(self, requested_as_of: date | None = None) -> RiskFreeCurveResponse:
        if not self.ready:
            raise FredDisabledError("FRED risk-free curve is disabled or its API key is missing")
        today = self._now().astimezone(timezone.utc).date()
        as_of = requested_as_of or today
        if as_of > today:
            raise InvalidRequestError("as_of may not be in the future")
        mapping = self.series_tenors()
        cache_key = as_of.isoformat() + "|" + ",".join(f"{series}:{days}" for series, days in mapping)
        ttl = float(self.context.get("refdata.api.cache.risk_free_curve_ttl_seconds", 86400))
        # Reuse durable curve payloads before making another FRED request.
        persistent = self._persistent.load("fred_curve", cache_key) if self._persistent else None
        if persistent is not None:
            payload, retrieved_at = persistent
            if (self._now().astimezone(timezone.utc) - retrieved_at).total_seconds() < ttl:
                return self._curve_response(payload, cache="hit")

        def load() -> dict[str, Any]:
            anchors: list[dict[str, Any]] = []
            warnings: list[str] = []
            for series, tenor_days in mapping:
                try:
                    observation = self._observation(series, as_of)
                except Exception as exc:
                    anchors.append(
                        RiskFreeAnchor(
                            series_id=series,
                            tenor_days=tenor_days,
                            status="provider_error",
                            warning=type(exc).__name__,
                        ).model_dump(mode="json")
                    )
                    warnings.append(f"{series}:provider_error")
                    continue
                if observation is None:
                    anchors.append(
                        RiskFreeAnchor(
                            series_id=series,
                            tenor_days=tenor_days,
                            status="missing",
                            warning="no_numeric_observation_on_or_before_as_of",
                        ).model_dump(mode="json")
                    )
                    warnings.append(f"{series}:missing")
                    continue
                observation_date, quoted_percent = observation
                anchors.append(
                    RiskFreeAnchor(
                        series_id=series,
                        tenor_days=tenor_days,
                        observation_date=observation_date,
                        quoted_percent=quoted_percent,
                        simple_annual_rate_decimal=quoted_percent / 100.0,
                        status="ok",
                    ).model_dump(mode="json")
                )
            if not any(item["status"] == "ok" for item in anchors):
                if persistent is not None:
                    old, _ = persistent
                    old_warnings = list(old.get("warnings") or [])
                    old["warnings"] = old_warnings + ["fred_refresh_failed_using_last_good"]
                    return old
                raise UpstreamUnavailableError(
                    "FRED returned no usable curve anchors",
                    details={"failed_series": [series for series, _ in mapping]},
                )
            now = self._now().astimezone(timezone.utc)
            payload = {
                "requested_as_of": as_of.isoformat(),
                "curve_id": self.CURVE_ID,
                "curve_version": self.CURVE_VERSION,
                "source": "fred",
                "anchors": anchors,
                "retrieved_at": now.isoformat(),
                "warnings": warnings,
            }
            if self._persistent is not None:
                self._persistent.store(
                    "fred_curve", cache_key, payload, source_as_of=now, retrieved_at=now
                )
            return payload

        lookup = self._cache.get_or_load(("fred_curve", cache_key), ttl, load)
        return self._curve_response(lookup.value, cache="hit" if lookup.hit else "miss")

    def curves(self, requested_dates: Sequence[date]) -> RiskFreeCurveHistoryResponse:
        """Load a point-in-time curve history with one FRED query per series.

        A historical risk run commonly needs one curve for every trading
        session. Calling ``curve`` for every date multiplies both HTTP traffic
        and FRED work by the number of observations. This contract keeps each
        curve causal while retrieving the bounded series window only once.
        """
        if not self.ready:
            raise FredDisabledError("FRED risk-free curve is disabled or its API key is missing")
        dates = tuple(sorted(set(requested_dates)))
        if not dates:
            raise InvalidRequestError("at least one as_of date is required")
        if len(dates) > 400:
            raise InvalidRequestError("at most 400 as_of dates may be requested")
        today = self._now().astimezone(timezone.utc).date()
        if dates[-1] > today:
            raise InvalidRequestError("as_of dates may not be in the future")
        mapping = self.series_tenors()
        mapping_key = ",".join(f"{series}:{days}" for series, days in mapping)
        cache_key = ",".join(item.isoformat() for item in dates) + "|" + mapping_key
        ttl = float(self.context.get("refdata.api.cache.risk_free_curve_ttl_seconds", 86400))

        def load() -> dict[str, Any]:
            persistent_curves: list[dict[str, Any]] = []
            if self._persistent is not None:
                for as_of in dates:
                    stored = self._persistent.load(
                        "fred_curve", as_of.isoformat() + "|" + mapping_key
                    )
                    if stored is None:
                        persistent_curves = []
                        break
                    curve, retrieved_at = stored
                    if (
                        self._now().astimezone(timezone.utc) - retrieved_at
                    ).total_seconds() >= ttl:
                        persistent_curves = []
                        break
                    persistent_curves.append(curve)
            if len(persistent_curves) == len(dates):
                retrieved_values = [
                    value
                    for item in persistent_curves
                    if (value := _timestamp(item.get("retrieved_at"))) is not None
                ]
                retrieved = max(retrieved_values, default=self._now().astimezone(timezone.utc))
                return {
                    "curves": persistent_curves,
                    "retrieved_at": retrieved.isoformat(),
                    "persistent_hit": True,
                }

            lookback_days = max(
                1,
                int(
                    self.context.get(
                        "refdata.fred.curve_history_observation_lookback_days", 10
                    )
                ),
            )
            observations: dict[str, list[tuple[date, float]]] = {}
            errors: dict[str, str] = {}

            # Fetch each FRED series once for the entire requested curve-date range.
            def history(series: str) -> tuple[str, list[tuple[date, float]]]:
                return series, self._observation_history(
                    series,
                    dates[0] - timedelta(days=lookback_days),
                    dates[-1],
                )

            workers = min(
                len(mapping),
                max(
                    1,
                    int(self.context.get("refdata.fred.curve_history_workers", 5)),
                ),
            )
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="fred-curve-history",
            ) as executor:
                pending = {
                    executor.submit(history, series): series
                    for series, _tenor_days in mapping
                }
                for future in as_completed(pending):
                    series = pending[future]
                    try:
                        loaded_series, rows = future.result()
                        observations[loaded_series] = rows
                    except Exception as exc:
                        observations[series] = []
                        errors[series] = type(exc).__name__

            now = self._now().astimezone(timezone.utc)
            curves: list[dict[str, Any]] = []
            for as_of in dates:
                anchors: list[dict[str, Any]] = []
                warnings: list[str] = []
                for series, tenor_days in mapping:
                    # Select the latest observation available on or before each curve date.
                    observation = next(
                        (
                            item
                            for item in reversed(observations[series])
                            if item[0] <= as_of
                        ),
                        None,
                    )
                    if observation is None:
                        error = errors.get(series)
                        anchors.append(
                            RiskFreeAnchor(
                                series_id=series,
                                tenor_days=tenor_days,
                                status="provider_error" if error else "missing",
                                warning=error or "no_numeric_observation_on_or_before_as_of",
                            ).model_dump(mode="json")
                        )
                        warnings.append(
                            f"{series}:provider_error" if error else f"{series}:missing"
                        )
                        continue
                    observation_date, quoted_percent = observation
                    anchors.append(
                        RiskFreeAnchor(
                            series_id=series,
                            tenor_days=tenor_days,
                            observation_date=observation_date,
                            quoted_percent=quoted_percent,
                            simple_annual_rate_decimal=quoted_percent / 100.0,
                            status="ok",
                        ).model_dump(mode="json")
                    )
                curves.append(
                    {
                        "requested_as_of": as_of.isoformat(),
                        "curve_id": self.CURVE_ID,
                        "curve_version": self.CURVE_VERSION,
                        "source": "fred",
                        "anchors": anchors,
                        "retrieved_at": now.isoformat(),
                        "warnings": warnings,
                    }
                )
            if not any(
                anchor["status"] == "ok"
                for curve in curves
                for anchor in curve["anchors"]
            ):
                raise UpstreamUnavailableError(
                    "FRED returned no usable historical curve anchors",
                    details={"failed_series": [series for series, _ in mapping]},
                )
            if self._persistent is not None:
                for curve in curves:
                    individual_key = curve["requested_as_of"] + "|" + mapping_key
                    self._persistent.store(
                        "fred_curve",
                        individual_key,
                        curve,
                        source_as_of=now,
                        retrieved_at=now,
                    )
            return {"curves": curves, "retrieved_at": now.isoformat()}

        lookup = self._cache.get_or_load(("fred_curve_history", cache_key), ttl, load)
        retrieved = _timestamp(lookup.value.get("retrieved_at")) or self._now().astimezone(timezone.utc)
        warnings = list(
            dict.fromkeys(
                warning
                for curve in lookup.value.get("curves", [])
                for warning in curve.get("warnings", [])
            )
        )
        return RiskFreeCurveHistoryResponse(
            data=[
                RiskFreeCurveData.model_validate(
                    {
                        "requested_as_of": curve["requested_as_of"],
                        "curve_id": curve["curve_id"],
                        "curve_version": curve["curve_version"],
                        "source": curve["source"],
                        "anchors": curve["anchors"],
                    }
                )
                for curve in lookup.value.get("curves", [])
            ],
            meta=LiveResponseMeta(
                as_of=retrieved,
                source="fred",
                cache=(
                    "hit"
                    if lookup.hit or lookup.value.get("persistent_hit")
                    else "miss"
                ),
                warnings=warnings,
            ),
        )

    def _curve_response(self, payload: Mapping[str, Any], *, cache: str) -> RiskFreeCurveResponse:
        anchors = [RiskFreeAnchor.model_validate(item) for item in payload.get("anchors", [])]
        as_of = date.fromisoformat(str(payload["requested_as_of"]))
        retrieved = _timestamp(payload.get("retrieved_at")) or self._now().astimezone(timezone.utc)
        return RiskFreeCurveResponse(
            data=RiskFreeCurveData(
                requested_as_of=as_of,
                curve_id=str(payload.get("curve_id") or self.CURVE_ID),
                curve_version=str(payload.get("curve_version") or self.CURVE_VERSION),
                anchors=anchors,
            ),
            meta=LiveResponseMeta(
                as_of=retrieved,
                source="fred",
                cache=cache,
                warnings=list(payload.get("warnings") or []),
            ),
        )

    def _observation(self, series_id: str, as_of: date) -> tuple[date, float] | None:
        api_key = str(self.context.get("refdata.fred.api_key", "")).strip()
        base_url = str(self.context.get("refdata.fred.base_url", "https://api.stlouisfed.org")).rstrip("/")
        parameters = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_end": as_of.isoformat(),
            "sort_order": "desc",
            "limit": str(int(self.context.get("refdata.fred.observation_limit", 10))),
        }
        # Build the authenticated FRED URL only for the outbound provider request.
        url = f"{base_url}/fred/series/observations?{urlencode(parameters)}"
        timeout = float(self.context.get("refdata.fred.timeout_seconds", 10))
        attempts = max(1, int(self.context.get("refdata.fred.retry_count", 2)) + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                with self._opener(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                observations = payload.get("observations") if isinstance(payload, Mapping) else None
                if not isinstance(observations, list):
                    raise ValueError("invalid FRED response")
                for item in observations:
                    if not isinstance(item, Mapping) or item.get("value") in (None, "."):
                        continue
                    observation_date = date.fromisoformat(str(item.get("date")))
                    if observation_date > as_of:
                        continue
                    value = _finite(item.get("value"))
                    if value is not None:
                        return observation_date, value
                return None
            except HTTPError as exc:
                last_error = exc
                if not (500 <= exc.code <= 599) or attempt + 1 >= attempts:
                    break
            except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
            sleep(min(0.25 * (2**attempt), 1.0))
        raise UpstreamUnavailableError(
            f"FRED request failed for series {series_id}",
            details={"series_id": series_id, "error": type(last_error).__name__},
        )

    def _observation_history(
        self, series_id: str, start: date, end: date
    ) -> list[tuple[date, float]]:
        api_key = str(self.context.get("refdata.fred.api_key", "")).strip()
        base_url = str(
            self.context.get("refdata.fred.base_url", "https://api.stlouisfed.org")
        ).rstrip("/")
        parameters = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
            "sort_order": "asc",
            "limit": "100000",
        }
        url = f"{base_url}/fred/series/observations?{urlencode(parameters)}"
        timeout = float(self.context.get("refdata.fred.timeout_seconds", 10))
        attempts = max(1, int(self.context.get("refdata.fred.retry_count", 2)) + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                with self._opener(
                    Request(url, headers={"Accept": "application/json"}), timeout=timeout
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                raw = payload.get("observations") if isinstance(payload, Mapping) else None
                if not isinstance(raw, list):
                    raise ValueError("invalid FRED response")
                rows: list[tuple[date, float]] = []
                for item in raw:
                    if not isinstance(item, Mapping) or item.get("value") in (None, "."):
                        continue
                    observation_date = date.fromisoformat(str(item.get("date")))
                    value = _finite(item.get("value"))
                    if start <= observation_date <= end and value is not None:
                        rows.append((observation_date, value))
                return sorted(rows)
            except HTTPError as exc:
                last_error = exc
                if not (500 <= exc.code <= 599) or attempt + 1 >= attempts:
                    break
            except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
            sleep(min(0.25 * (2**attempt), 1.0))
        raise UpstreamUnavailableError(
            f"FRED history request failed for series {series_id}",
            details={"series_id": series_id, "error": type(last_error).__name__},
        )


class CmeReferenceDataService:
    """Entitlement-gated CME adapter with all provider paths isolated here."""

    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        now_fn: Callable[[], datetime] = _utc_now,
        opener: Any | None = None,
    ) -> None:
        self.context = context
        self._now = now_fn
        self._cache = SingleFlightTTLCache[dict[str, Any]](max_entries=128)
        self._persistent = LatestResponseCache(_cache_path(context)) if self.ready else None
        self._opener = opener or build_opener()
        self._oauth = CmeOAuthClient(context, now_fn=now_fn, opener=self._opener)

    @property
    def ready(self) -> bool:
        return bool(
            _as_bool(self.context.get("refdata.cme_reference.enabled", False))
            and str(self.context.get("refdata.cme_reference.api_id", "") or "").strip()
            and str(self.context.get("refdata.cme_reference.app_password", "") or "").strip()
            and str(self.context.get("refdata.cme_reference.base_url", "") or "").strip()
        )

    def _access_token(self) -> str:
        return self._oauth.access_token()

    def specifications(self, contracts: Sequence[str]) -> FuturesSpecificationResponse:
        if not self.ready:
            raise CmeReferenceDisabledError(
                "CME Reference Data is disabled or Basic Auth credentials/base URL are incomplete"
            )
        maximum = int(self.context.get("refdata.cme_reference.max_contracts_per_request", 100))
        normalized = [str(item).strip().upper() for item in contracts]
        if not normalized or len(normalized) > maximum:
            raise InvalidRequestError(f"futures specification request must contain 1-{maximum} contracts")
        now = self._now().astimezone(timezone.utc)
        items: list[FuturesSpecificationItem] = []
        cache_states: list[str] = []
        for requested in normalized:
            item, cache_state = self._specification(requested, now)
            items.append(item)
            cache_states.append(cache_state)
        return FuturesSpecificationResponse(
            data=items,
            meta=LiveResponseMeta(
                as_of=now,
                source="cme_reference",
                cache="hit" if cache_states and all(state == "hit" for state in cache_states) else "miss",
                warnings=["some_contracts_served_from_cache"] if "hit" in cache_states and "miss" in cache_states else [],
            ),
        )

    def _specification(self, contract: str, now: datetime) -> tuple[FuturesSpecificationItem, str]:
        cache_key = f"{contract}|cme_specification_v2"
        attempt_key = f"{contract}|cme_globex_v2"
        provider_contract = _cme_globex_contract(contract)
        cached = self._persistent.load("cme_spec", cache_key) if self._persistent else None
        soft_ttl = float(self.context.get("refdata.api.cache.futures_specs_ttl_seconds", 86400))
        hard_age = float(self.context.get("refdata.cme_reference.hard_age_seconds", 604800))
        if cached is not None:
            payload, retrieved = cached
            age = max(0.0, (now - retrieved).total_seconds())
            if age <= soft_ttl:
                return FuturesSpecificationItem.model_validate(payload), "hit"
        # Apply a durable retry floor independently of specification-cache age.
        attempt = self._persistent.attempt_state("cme_spec", attempt_key) if self._persistent else None
        floor = max(60, int(self.context.get("refdata.api.remote_attempt_floor_seconds", 60)))
        if attempt:
            last_attempt = _timestamp(attempt.get("last_attempt_at"))
            if last_attempt is not None and (now - last_attempt).total_seconds() < floor:
                if cached is not None and (now - cached[1]).total_seconds() <= hard_age:
                    stale = FuturesSpecificationItem.model_validate(cached[0])
                    return stale.model_copy(update={"warnings": list(dict.fromkeys([*stale.warnings, "refresh_suppressed_using_stale_cache"]))}), "hit"
                return FuturesSpecificationItem(
                    requested_contract=contract,
                    canonical_contract=contract,
                    status="provider_error",
                    retrieved_at=now,
                    warnings=["remote_attempt_suppressed_no_cache"],
                ), "miss"
        template = str(
            self.context.get(
                "refdata.cme_reference.instrument_path_template",
                "/v3/instruments?globexSymbol={contract}",
            )
        )
        base_url = str(self.context["refdata.cme_reference.base_url"]).rstrip("/")
        url = base_url + template.format(contract=quote(provider_contract, safe=""))
        timeout = float(self.context.get("refdata.cme_reference.timeout_seconds", 10))
        try:
            token = self._access_token()
            with self._opener.open(
                Request(
                    url,
                    headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
                ),
                timeout=timeout,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            if self._persistent:
                self._persistent.record_attempt("cme_spec", attempt_key, at=now, success=False, error=type(exc).__name__)
            if cached is not None and (now - cached[1]).total_seconds() <= hard_age:
                stale = FuturesSpecificationItem.model_validate(cached[0])
                return stale.model_copy(update={"warnings": list(dict.fromkeys([*stale.warnings, "provider_error_using_stale_cache"]))}), "hit"
            return FuturesSpecificationItem(
                requested_contract=contract,
                canonical_contract=contract,
                status="provider_error",
                retrieved_at=now,
                warnings=[f"provider_error:{type(exc).__name__}"],
            ), "miss"
        candidates: Any = payload
        if isinstance(payload, Mapping):
            embedded = payload.get("_embedded")
            if isinstance(embedded, Mapping):
                candidates = embedded.get("instruments", [])
            else:
                candidates = payload.get("data", payload)
        if isinstance(candidates, Mapping):
            candidates = [candidates]
        if not isinstance(candidates, list) or not candidates:
            result = FuturesSpecificationItem(
                requested_contract=contract,
                canonical_contract=contract,
                status="not_found",
                retrieved_at=now,
                warnings=["provider_returned_no_exact_instrument"],
            )
            if self._persistent:
                self._persistent.record_attempt("cme_spec", attempt_key, at=now, success=True)
            return result, "miss"
        # Accept only an exact provider contract match before normalizing its economics.
        exact = [
            item
            for item in candidates
            if isinstance(item, Mapping)
            and str(
                item.get("globexSymbol")
                or item.get("symbol")
                or item.get("contract")
                or ""
            ).upper()
            == provider_contract
        ]
        if len(exact) != 1:
            result = FuturesSpecificationItem(
                requested_contract=contract,
                canonical_contract=contract,
                status="ambiguous" if len(exact) > 1 else "not_found",
                retrieved_at=now,
                warnings=["provider_mapping_not_unique"],
            )
            if self._persistent:
                self._persistent.record_attempt("cme_spec", attempt_key, at=now, success=True)
            return result, "miss"
        source = exact[0]
        multiplier = _finite(
            source.get("priceMultiplier")
            or source.get("contractUnit")
            or source.get("unitOfMeasureQty")
        )
        tick = _finite(
            source.get("tradeTick")
            or source.get("minimumTick")
            or source.get("minPriceIncrement")
        )
        tick_value = _finite(source.get("tickValue"))
        derivation = None
        # Derive a missing tick value only from the resolved tick and price multiplier.
        if tick_value is None and tick is not None and multiplier is not None:
            tick_value = tick * multiplier
            derivation = "minimum_tick * price_multiplier; cme_tick_derivation_v1"
        status = "ok" if multiplier is not None and tick is not None else "partial"
        result = FuturesSpecificationItem(
            requested_contract=contract,
            canonical_contract=contract,
            status=status,
            provider_instrument_id=_text(source.get("guid") or source.get("instrumentId")),
            provider_product_id=_text(source.get("productGuidInt") or source.get("productId")),
            root=_text(source.get("globexGroupCode") or source.get("root") or source.get("productCode")),
            product_code=_text(source.get("productCode") or source.get("globexGroupCode")),
            product_name=_text(source.get("instrumentName") or source.get("productName")),
            exchange=_text(source.get("exchangeClearing") or source.get("exchange")),
            mic=_text(source.get("exchangeGlobex") or source.get("mic")),
            asset_class=_text(source.get("instrumentType") or source.get("assetClass")),
            currency=_text(source.get("currency")),
            contract_unit=_finite(source.get("contractUnit")),
            price_multiplier=multiplier,
            quote_convention=_text(source.get("quoteConvention")),
            minimum_tick=tick,
            tick_value=tick_value,
            tick_value_derivation=derivation,
            trading_timezone=_text(source.get("tradingTimezone")),
            trading_schedule=_text(source.get("tradingSchedule")),
            listing_date=_date_value(source.get("listingDate")),
            first_trade_date=_date_value(source.get("firstTradeDate")),
            last_trade_at=_timestamp(source.get("lastTradeTime") or source.get("lastTradeDate")),
            first_notice_date=_date_value(source.get("firstNoticeDate")),
            delivery_type=_text(source.get("deliveryType")),
            settlement_type=_text(source.get("settlementType")),
            delivery_start_date=_date_value(source.get("deliveryStartDate")),
            delivery_end_date=_date_value(source.get("deliveryEndDate")),
            settlement_date=_date_value(source.get("settlementDate")),
            provider_as_of=_timestamp(source.get("lastUpdated") or source.get("asOf")),
            retrieved_at=now,
            warnings=[] if status == "ok" else ["provider_specification_partial"],
        )
        if self._persistent:
            self._persistent.record_attempt("cme_spec", attempt_key, at=now, success=True)
            if result.status in {"ok", "partial"}:
                self._persistent.store(
                    "cme_spec",
                    cache_key,
                    result.model_dump(mode="json"),
                    source_as_of=result.provider_as_of or now,
                    retrieved_at=now,
                )
        return result, "miss"


def _option_deliverable_components(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_components = (
        source.get("deliverable_components")
        or source.get("optionDeliverableComponents")
        or source.get("optionDeliverablesList")
        or []
    )
    if not isinstance(raw_components, list):
        return []
    output: list[dict[str, Any]] = []
    for raw in raw_components:
        if not isinstance(raw, Mapping):
            continue
        explicit_type = str(raw.get("component_type") or "").strip().lower()
        identifier = _text(
            raw.get("security_identifier")
            or raw.get("symbol")
            or raw.get("deliverableSymbol")
        )
        quantity = _finite(
            raw.get("quantity")
            if raw.get("quantity") is not None
            else raw.get("deliverableUnits")
        )
        cash = _finite(
            raw.get("cash_amount")
            if raw.get("cash_amount") is not None
            else raw.get("cashAmount")
        )
        component_type = explicit_type if explicit_type in {"security", "cash", "other"} else (
            "security" if identifier and quantity is not None else
            "cash" if cash is not None else "other"
        )
        output.append({
            "component_type": component_type,
            "security_identifier": identifier,
            "quantity": quantity,
            "cash_amount": cash,
            "currency": str(raw.get("currency") or "USD"),
            "description": _text(raw.get("description") or raw.get("deliverableDescription")),
        })
    return output


def _delivered_security_quantity(source: Mapping[str, Any]) -> float | None:
    quantities = [
        _finite(component.get("quantity"))
        for component in _option_deliverable_components(source)
        if component.get("component_type") == "security"
    ]
    usable = [value for value in quantities if value is not None]
    return sum(usable) if usable else None


def _delivered_cash_amount(source: Mapping[str, Any]) -> float | None:
    amounts = [
        _finite(component.get("cash_amount"))
        for component in _option_deliverable_components(source)
        if component.get("component_type") == "cash"
    ]
    usable = [value for value in amounts if value is not None]
    return sum(usable) if usable else None


def _text(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _date_value(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value)[0:10])
    except ValueError:
        return None


def _local_time(value: Any) -> time:
    try:
        hour, minute = str(value).split(":", 1)
        return time(int(hour), int(minute))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "refdata option-terms expiration time must use HH:MM"
        ) from exc


def _option_settlement_style(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    if normalized in {"A", "AM", "A.M."}:
        return "AM"
    if normalized in {"P", "PM", "P.M."}:
        return "PM"
    return None


__all__ = [
    "CmeReferenceDataService",
    "CmeReferenceDisabledError",
    "FredDisabledError",
    "FredRiskFreeCurveService",
    "LatestResponseCache",
    "LiveOptionService",
    "OptionChainDisabledError",
]
