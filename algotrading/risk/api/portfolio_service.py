"""Coherent user-triggered portfolio snapshot orchestration."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import re
from threading import RLock
from typing import Any, Callable, Mapping, Sequence

from pydantic import ValidationError

from algotrading.refdata.api.option_symbols import parse_occ_symbol
from algotrading.refdata.stock_analysis.etf_universe import ETF_UNIVERSE
from algotrading.risk.cache import PersistentRiskCache, default_risk_cache_path
from algotrading.risk.exposures.portfolio import portfolio_kpis, portfolio_risk_coverage
from algotrading.risk.instruments.terms import (
    build_instrument_terms,
    resolve_instrument_terms,
)
from algotrading.risk.market_inputs.client import ReferenceDataClient
from algotrading.risk.market_inputs.etf_exposure import (
    CURATED_ETF_SECTORS, classify_etf_exposure,
)
from algotrading.risk.market_inputs.dividends import (
    derive_trailing_cash_yield,
    escrowed_dividend_adjustment,
)
from algotrading.risk.market_inputs.rates import interpolate_risk_free_rate
from algotrading.risk.models import Coverage, DataQuality, PortfolioSnapshot, SourceState
from algotrading.risk.positions.normalization import normalize_broker_position
from algotrading.risk.pricing.tian_service import (
    IvGreeksContract,
    IvGreeksRequest,
    TianIvGreeksService,
)

from .portfolio_models import (
    CacheMeta,
    PortfolioSnapshotData,
    PortfolioSnapshotResponse,
)
from .service import RiskService


_LISTED_FUTURE = re.compile(
    r"^/?[A-Z0-9]{1,6}[FGHJKMNQUVXZ][0-9]{1,2}$"
)


def _standard_share_option_deliverable(value: Any, root: str) -> bool:
    """Recognize an ordinary 100-share deliverable without guessing."""

    if value in (None, "", (), []):
        return True
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return False
    if isinstance(parsed, Mapping):
        parsed = [parsed]
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        return False
    components = [item for item in parsed if isinstance(item, Mapping)]
    if len(components) != 1:
        return False
    component = components[0]
    component_type = str(component.get("component_type") or "security").lower()
    identifier = str(
        component.get("security_identifier")
        or component.get("symbol")
        or component.get("deliverableSymbol")
        or ""
    ).upper()
    quantity = component.get("quantity")
    if quantity is None:
        quantity = component.get("deliverableUnits")
    cash = component.get("cash_amount")
    if cash is None:
        cash = component.get("cashAmount")
    try:
        standard_quantity = math.isclose(float(quantity), 100.0)
        no_cash = cash is None or math.isclose(float(cash), 0.0)
    except (TypeError, ValueError):
        return False
    return (
        component_type == "security"
        and identifier == root
        and standard_quantity
        and no_cash
    )


class NoSnapshotLoaded(RuntimeError):
    pass


class SnapshotMismatch(RuntimeError):
    pass


class PortfolioApplicationService:
    def __init__(
        self,
        account_service: RiskService,
        context: Mapping[str, Any],
        *,
        reference_client: ReferenceDataClient | None = None,
        tian_service: TianIvGreeksService | None = None,
        cache: PersistentRiskCache | None = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.account_service = account_service
        self.context = context
        self.reference_client = reference_client or ReferenceDataClient(context)
        self.tian_service = tian_service or TianIvGreeksService(context, now_fn=now_fn)
        self.cache = cache or PersistentRiskCache(
            default_risk_cache_path(context),
            max_entries=int(context.get("risk.cache.max_entries", 5000)),
        )
        self._now = now_fn
        self._lock = RLock()
        self._latest_by_scope: dict[str, PortfolioSnapshotResponse] = {}
        self._previous_by_scope: dict[str, PortfolioSnapshotResponse] = {}

    def close(self) -> None:
        close = getattr(self.reference_client, "close", None)
        if callable(close):
            close()

    def snapshot(
        self,
        account_ids: Sequence[str],
        *,
        refresh: bool,
    ) -> PortfolioSnapshotResponse:
        scope = _normalize_scope(account_ids)
        # Bind cached snapshots to both account scope and calculation configuration.
        configuration_version = _configuration_version(self.context)
        cache_key = _cache_key(scope, configuration_version)
        now = _aware_utc(self._now())
        with self._lock:
            cached = self._load_cached(cache_key, now)
            if not refresh:
                if cached is None:
                    raise NoSnapshotLoaded("No snapshot loaded; use an explicit refresh")
                return cached
            floor = max(60, int(self.context.get("risk.refresh.remote_attempt_floor_seconds", 60)))
            resource_key = "broker:" + ",".join(scope)
            attempt = self.cache.attempt_state(resource_key)
            if attempt:
                last_attempt = _parse_datetime(str(attempt["last_attempt_at"]))
                allowed_at = last_attempt + timedelta(seconds=floor)
                if allowed_at > now:
                    if cached is None:
                        raise NoSnapshotLoaded(
                            f"Refresh suppressed until {allowed_at.isoformat()} and no cached snapshot exists"
                        )
                    return cached.model_copy(
                        update={
                            "cache": CacheMeta(
                                state="suppressed",
                                age_seconds=max(0.0, (now - cached.as_of).total_seconds()),
                                source_as_of=cached.as_of,
                                refresh_suppressed_until=allowed_at,
                            )
                        }
                    )
            try:
                response = self._build_snapshot(scope, configuration_version, now)
            except Exception as exc:
                self.cache.record_attempt(resource_key, at=now, success=False, error=type(exc).__name__)
                raise
            self.cache.record_attempt(resource_key, at=now, success=True)
            scope_key = ",".join(scope)
            if cached is not None and cached.as_of < response.as_of:
                self._previous_by_scope[scope_key] = cached
            ttl = float(self.context.get("risk.snapshot.ttl_seconds", 60))
            hard = float(self.context.get("risk.snapshot.hard_max_age_seconds", 300))
            self.cache.put(
                cache_key,
                namespace="portfolio_snapshot",
                request_fingerprint=cache_key,
                payload=response.model_dump(mode="json"),
                source_as_of=response.as_of,
                fetched_at=now,
                soft_expires_at=now + timedelta(seconds=ttl),
                hard_expires_at=now + timedelta(seconds=hard),
            )
            self._latest_by_scope[scope_key] = response
            return response

    def latest(self, account_ids: Sequence[str]) -> PortfolioSnapshotResponse:
        return self.snapshot(account_ids, refresh=False)

    def previous(self, account_ids: Sequence[str]) -> PortfolioSnapshotResponse | None:
        """Return the snapshot immediately preceding the current user refresh."""
        return self._previous_by_scope.get(",".join(_normalize_scope(account_ids)))

    def require_snapshot(
        self, account_ids: Sequence[str], snapshot_id: str | None = None
    ) -> tuple[PortfolioSnapshotResponse, PortfolioSnapshot]:
        response = self.latest(account_ids)
        if snapshot_id and snapshot_id != response.portfolio_snapshot_id:
            raise SnapshotMismatch("portfolio_snapshot_id is stale or belongs to another account scope")
        snapshot = PortfolioSnapshot(
            portfolio_snapshot_id=response.portfolio_snapshot_id,
            configuration_version=response.configuration_version,
            as_of=response.as_of,
            display_timezone=response.display_timezone,
            account_ids=tuple(response.data.account_ids),
            portfolio_value_usd=response.data.kpis.portfolio_value_usd,
            positions=tuple(response.data.positions),
            data_quality=response.data_quality,
        )
        return response, snapshot

    def hedge_chain(
        self,
        account_ids: Sequence[str],
        *,
        portfolio_snapshot_id: str,
        symbol: str,
        contract_type: str,
        strike_count: int,
        from_date: str,
        to_date: str,
        include_underlying_quote: bool,
    ) -> dict[str, Any]:
        response, _ = self.require_snapshot(account_ids, portfolio_snapshot_id)
        if not response.data.capabilities.get("available", {}).get("options_chain", False):
            raise ValueError("live option-chain capability is disabled or unavailable")
        normalized_symbol = symbol.strip().upper()
        start = date.fromisoformat(from_date)
        end = date.fromisoformat(to_date)
        if end < start:
            raise ValueError("to_date must be on or after from_date")
        bounds = {
            "symbol": normalized_symbol,
            "contract_type": contract_type,
            "strike_count": strike_count,
            "from_date": start.isoformat(),
            "to_date": end.isoformat(),
            "include_underlying_quote": include_underlying_quote,
        }
        request_fingerprint = hashlib.sha256(
            json.dumps(bounds, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_key = "hedge-chain:" + request_fingerprint
        now = _aware_utc(self._now())
        cached = self.cache.get(cache_key)
        # Refresh suppression is request-bound. A short scanner chain must not
        # suppress the distinct three-month chain required by the picker.
        resource_key = "option-chain:" + request_fingerprint
        # Throttle repeated chain fetches independently of cached payload freshness.
        floor = max(60, int(self.context.get("risk.refresh.remote_attempt_floor_seconds", 60)))
        attempt = self.cache.attempt_state(resource_key)
        allowed_at = None
        if attempt:
            allowed_at = _parse_datetime(str(attempt["last_attempt_at"])) + timedelta(seconds=floor)
        cache_hard_expiration = (
            _parse_datetime(str(cached["hard_expires_at"]))
            if cached is not None
            else None
        )
        if cached is not None and cache_hard_expiration is not None and now <= cache_hard_expiration:
            payload = dict(cached["payload"])
            payload["data"] = self._enrich_chain_contract_terms(
                dict(payload.get("data") or {})
            )
            # The market chain cache is request-bound by its own fingerprint,
            # while the response envelope must be bound to the currently
            # validated portfolio snapshot.
            payload["portfolio_snapshot_id"] = portfolio_snapshot_id
            payload["analysis_only"] = True
            age = max(0.0, (now - _parse_datetime(cached["source_as_of"])).total_seconds())
            payload["cache"] = {
                "state": "hit" if allowed_at is None or allowed_at <= now else "suppressed",
                "age_seconds": age,
                "refresh_suppressed_until": allowed_at.isoformat() if allowed_at and allowed_at > now else None,
            }
            return payload
        if allowed_at is not None and allowed_at > now:
            raise NoSnapshotLoaded(
                f"Option-chain refresh suppressed until {allowed_at.isoformat()} for {normalized_symbol}; "
                "the cached chain is beyond its hard expiry"
            )
        try:
            upstream = self.reference_client.option_chain(bounds)
        except Exception as exc:
            self.cache.record_attempt(resource_key, at=now, success=False, error=type(exc).__name__)
            raise
        self.cache.record_attempt(resource_key, at=now, success=True)
        source_as_of = _parse_datetime(
            str((upstream.get("meta") or {}).get("as_of") or now.isoformat())
        )
        # Identify the exact chain payload used to evaluate a hedge proposal.
        chain_snapshot_id = "sha256:" + hashlib.sha256(
            json.dumps(upstream.get("data") or {}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        chain_data = self._enrich_chain_contract_terms(
            dict(upstream.get("data") or {})
        )
        payload = {
            "schema_version": 1,
            "portfolio_snapshot_id": portfolio_snapshot_id,
            "chain_snapshot_id": chain_snapshot_id,
            "requested_at": now.isoformat(),
            "source_as_of": source_as_of.isoformat(),
            "bounds": bounds,
            "data": chain_data,
            "cache": {
                "state": "miss",
                "age_seconds": max(0.0, (now - source_as_of).total_seconds()),
                "refresh_suppressed_until": None,
            },
            "analysis_only": True,
        }
        self.cache.put(
            cache_key,
            namespace="live_option_chain",
            request_fingerprint=request_fingerprint,
            payload=payload,
            source_as_of=source_as_of,
            fetched_at=now,
            soft_expires_at=now + timedelta(seconds=60),
            hard_expires_at=now + timedelta(minutes=5),
        )
        return payload

    def hedge_expirations(
        self,
        account_ids: Sequence[str],
        *,
        portfolio_snapshot_id: str,
        symbol: str,
        from_date: str,
        to_date: str,
        include_underlying_quote: bool,
    ) -> dict[str, Any]:
        """Discover expirations through RefData's calendar-only provider route."""

        response, _ = self.require_snapshot(account_ids, portfolio_snapshot_id)
        if not response.data.capabilities.get("available", {}).get("options_chain", False):
            raise ValueError("live option-chain capability is disabled or unavailable")
        normalized_symbol = symbol.strip().upper()
        start = date.fromisoformat(from_date)
        end = date.fromisoformat(to_date)
        if end < start:
            raise ValueError("to_date must be on or after from_date")
        upstream = self.reference_client.option_expirations({
            "symbol": normalized_symbol,
            "from_date": start.isoformat(),
            "to_date": end.isoformat(),
        })
        upstream_data = dict(upstream.get("data") or {})
        upstream_meta = dict(upstream.get("meta") or {})
        now = _aware_utc(self._now())
        source_as_of = _parse_datetime(str(upstream_meta.get("as_of") or now.isoformat()))
        expiration_rows = [
            dict(item) for item in upstream_data.get("expirations") or []
            if isinstance(item, Mapping)
            and str(item.get("expiration") or "").strip()
        ]
        expiration_rows.sort(key=lambda item: str(item["expiration"]))
        return {
            "schema_version": 1,
            "portfolio_snapshot_id": portfolio_snapshot_id,
            "requested_at": now.isoformat(),
            "source_as_of": source_as_of.isoformat(),
            "bounds": {
                "symbol": normalized_symbol,
                "from_date": start.isoformat(),
                "to_date": end.isoformat(),
            },
            "data": {
                "symbol": upstream_data.get("symbol") or normalized_symbol,
                "from_date": start.isoformat(),
                "to_date": end.isoformat(),
                # Expiration discovery deliberately avoids quote and contract payloads.
                "underlying_quote": None,
                "expirations": expiration_rows,
            },
            "cache": {
                "state": str(upstream_meta.get("cache") or "miss"),
                "age_seconds": max(0.0, (now - source_as_of).total_seconds()),
                "refresh_suppressed_until": None,
            },
            "analysis_only": True,
        }

    def _enrich_chain_contract_terms(self, data: dict[str, Any]) -> dict[str, Any]:
        """Attach exact contract terms so no GUI has to invent derivative facts."""

        contracts = [
            dict(item) for item in data.get("contracts") or []
            if isinstance(item, Mapping)
        ]
        if contracts and all(item.get("instrument_terms") for item in contracts):
            return {**data, "contracts": contracts}
        by_symbol: dict[str, dict[str, Any]] = {}
        symbols: list[str] = []
        for item in contracts:
            symbol = str(item.get("canonical_occ_symbol") or "")
            if not symbol:
                continue
            try:
                root = parse_occ_symbol(symbol).root
            except ValueError:
                symbols.append(symbol)
                continue
            supplied_multiplier = item.get("multiplier") or item.get("price_multiplier")
            adjusted_evidence = bool(
                item.get("adjusted_contract")
                or item.get("is_non_standard")
                or not _standard_share_option_deliverable(
                    item.get("deliverable"), root
                )
                or not _standard_share_option_deliverable(
                    item.get("deliverable_components"), root
                )
                or (
                    supplied_multiplier is not None
                    and not math.isclose(float(supplied_multiplier), 100.0)
                )
            )
            # SPY and QQQ are the supported GUI option roots. Their standard
            # OCC contracts are American-style ETF options, so ordinary chain
            # rows do not need a second broker metadata round-trip. Adjusted
            # contracts still require the exact terms service.
            if root in {"SPY", "QQQ"} and not adjusted_evidence:
                by_symbol[symbol] = {
                    "canonical_occ_symbol": symbol,
                    "asset_subtype": "etf_option",
                    "terms_status": "standard",
                    "price_multiplier": 100.0,
                    "shares_per_contract": 100.0,
                    "adjusted_contract": False,
                    "exercise_style": "american",
                    "settlement_type": "physical",
                    "source": "standard_listed_etf_option_terms",
                }
            else:
                symbols.append(symbol)
        if not symbols:
            payload = {"data": []}
        else:
            try:
                payload = self.reference_client.option_contract_terms(symbols)
            except Exception:
                payload = {"data": []}
        by_symbol.update({
            str(item.get("canonical_occ_symbol") or ""): dict(item)
            for item in payload.get("data") or []
            if isinstance(item, Mapping) and item.get("canonical_occ_symbol")
        })
        enriched = []
        for item in contracts:
            symbol = str(item.get("canonical_occ_symbol") or "")
            raw_terms = by_symbol.get(symbol)
            terms = build_instrument_terms(
                symbol,
                broker_type="OPTION",
                option_terms=raw_terms,
                context=self.context,
            ) if symbol else None
            enriched.append(
                {
                    **item,
                    "expiration_at": (
                        terms.expiration_at.isoformat()
                        if terms is not None and terms.expiration_at is not None
                        else None
                    ),
                    "multiplier": (
                        terms.price_multiplier if terms is not None else None
                    ),
                    "instrument_terms": (
                        terms.model_dump(mode="json") if terms is not None else None
                    ),
                }
            )
        return {**data, "contracts": enriched}

    def _load_cached(self, cache_key: str, now: datetime) -> PortfolioSnapshotResponse | None:
        in_memory = next(
            (
                value
                for value in self._latest_by_scope.values()
                if _cache_key(tuple(value.data.account_ids), value.configuration_version) == cache_key
            ),
            None,
        )
        record = self.cache.get(cache_key)
        if record is None and in_memory is None:
            return None
        response = in_memory or PortfolioSnapshotResponse.model_validate(record["payload"])
        soft = _parse_datetime(record["soft_expires_at"]) if record else response.as_of + timedelta(seconds=60)
        hard = _parse_datetime(record["hard_expires_at"]) if record else response.as_of + timedelta(seconds=300)
        # Distinguish fresh, soft-stale, and hard-stale snapshots without hiding their age.
        state = "hit" if now <= soft else "stale" if now <= hard else "hard_stale"
        quality = response.data_quality
        if state == "hard_stale":
            quality = DataQuality(
                status="insufficient_data",
                sources=quality.sources,
                coverage=quality.coverage,
                risk_coverage=quality.risk_coverage,
                warnings=tuple(dict.fromkeys((*quality.warnings, "portfolio_snapshot_hard_stale"))),
            )
        return response.model_copy(
            update={
                "data_quality": quality,
                "cache": CacheMeta(
                    state=state,
                    age_seconds=max(0.0, (now - response.as_of).total_seconds()),
                    source_as_of=response.as_of,
                )
            }
        )

    def _build_snapshot(
        self,
        scope: tuple[str, ...],
        configuration_version: str,
        now: datetime,
    ) -> PortfolioSnapshotResponse:
        broker_result = self.account_service.accounts(include_positions=True, refresh=True)
        sources: list[SourceState] = [
            SourceState(
                name="broker_accounts_positions",
                as_of=broker_result.as_of,
                age_seconds=max(0.0, (now - broker_result.as_of).total_seconds()),
                state="loaded_for_snapshot",
                method_version="broker_normalization_v1",
            )
        ]
        accounts = list(broker_result.data)
        if scope != ("all",):
            wanted = set(scope)
            accounts = [account for account in accounts if account.account_id in wanted]
            missing = wanted - {account.account_id for account in accounts}
            if missing:
                raise ValueError("unknown account scope")
        account_ids = sorted(account.account_id for account in accounts)
        broker_positions = [
            position.model_dump(mode="python")
            for account in accounts
            for position in account.positions
        ]
        warnings: list[str] = []
        capabilities: dict[str, Any] = {"available": {}, "supported": {}}
        try:
            capabilities = self.reference_client.capabilities()
            sources.append(SourceState(name="reference_capabilities", as_of=now, age_seconds=0.0, state="loaded_for_snapshot", method_version="capability_schema_1.0"))
        except Exception as exc:
            warnings.append(f"reference_capabilities_unavailable:{type(exc).__name__}")
            sources.append(SourceState(name="reference_capabilities", state="unavailable", method_version="capability_schema_1.0"))
        # Build provider request universes once from the broker positions.
        general_symbols: list[str] = []
        option_symbols: list[str] = []
        futures_symbols: list[str] = []
        symbol_types: list[tuple[str, str]] = []
        for raw in broker_positions:
            instrument = raw.get("instrument") or {}
            symbol = str(instrument.get("symbol") or "").strip().upper()
            broker_type = str(
                instrument.get("type") or instrument.get("assetType") or ""
            ).upper()
            symbol_types.append((symbol, broker_type))
            try:
                occ = parse_occ_symbol(symbol)
            except ValueError:
                if symbol:
                    general_symbols.append(symbol)
                    if _future_symbol(symbol) or "FUTURE" in broker_type:
                        futures_symbols.append(symbol.removeprefix("/"))
            else:
                option_symbols.append(occ.canonical)
                general_symbols.append(occ.root)
        general_quotes: dict[str, dict[str, Any]] = {}
        try:
            rows = self.reference_client.quotes(tuple(dict.fromkeys(general_symbols)))
            general_quotes = {str(row.get("symbol") or "").upper(): dict(row) for row in rows}
            sources.append(SourceState(name="current_general_quotes", as_of=now, age_seconds=0.0, state="loaded_for_snapshot", method_version="refdata_current_quotes_v1"))
        except Exception as exc:
            warnings.append(f"general_quotes_unavailable:{type(exc).__name__}")
            sources.append(SourceState(name="current_general_quotes", state="unavailable", method_version="refdata_current_quotes_v1"))
        classifications: dict[str, dict[str, Any]] = {}
        try:
            classification_rows = self.reference_client.security_classifications(
                tuple(dict.fromkeys(general_symbols))
            )
            classifications = {
                str(row.get("symbol") or "").strip().upper(): dict(row)
                for row in classification_rows
                if row.get("symbol")
            }
            sources.append(SourceState(
                name="security_classifications",
                as_of=now,
                age_seconds=0.0,
                state="loaded_for_snapshot",
                method_version="refdata_universe_classification_v1",
            ))
        except Exception as exc:
            warnings.append(f"security_classifications_unavailable:{type(exc).__name__}")
            sources.append(SourceState(
                name="security_classifications",
                state="unavailable",
                method_version="refdata_universe_classification_v1",
            ))
        # ETF issuer sectors describe asset managers, not the fund exposure.
        # The verified mapping and declared sector funds take precedence; every
        # other held ETF receives one bounded provider attempt per symbol.
        etf_symbols = sorted({
            symbol for symbol, broker_type in symbol_types
            if symbol and (
                broker_type in {"ETF", "EXCHANGE_TRADED_FUND"}
                or str(classifications.get(symbol, {}).get("security_type") or "").upper()
                in {"ETF", "EXCHANGE_TRADED_FUND"}
            )
        })
        dynamic_symbols = []
        for symbol in etf_symbols:
            base = classifications.setdefault(symbol, {"symbol": symbol, "security_type": "ETF"})
            definition = ETF_UNIVERSE.get(symbol)
            if symbol in CURATED_ETF_SECTORS:
                base.update(sector=CURATED_ETF_SECTORS[symbol], industry=None,
                            taxonomy="ETF_EXPOSURE", classification_reason=None)
            elif (definition is not None and definition.category == "GICS_SECTOR"
                  and definition.exposure_method == "DECLARED_SECTOR"):
                base.update(sector=definition.exposure_name, industry=None,
                            taxonomy="ETF_EXPOSURE", classification_reason=None)
            else:
                dynamic_symbols.append(symbol)
        if dynamic_symbols:
            def resolve(symbol: str) -> tuple[str, str, str | None]:
                try:
                    exposure = self.reference_client.etf_exposure(symbol)
                except Exception:
                    exposure = {"status": "unavailable", "reason": "provider_unavailable"}
                sector, reason = classify_etf_exposure(exposure)
                return symbol, sector, reason

            workers = min(4, len(dynamic_symbols))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for symbol, sector, reason in executor.map(resolve, dynamic_symbols):
                    classifications[symbol].update(
                        sector=sector, industry=None, taxonomy="ETF_EXPOSURE",
                        classification_reason=reason,
                    )
        if etf_symbols:
            sources.append(SourceState(
                name="etf_exposure", as_of=now, age_seconds=0.0,
                state="loaded_for_snapshot", method_version="etf_exposure_classification_v1",
            ))
        option_quote_payload: dict[str, Any] = {"data": []}
        try:
            option_quote_payload = self.reference_client.option_quotes(option_symbols)
            option_as_of = _optional_datetime((option_quote_payload.get("meta") or {}).get("as_of"))
            sources.append(SourceState(name="exact_option_quotes", as_of=option_as_of, age_seconds=(max(0.0, (now-option_as_of).total_seconds()) if option_as_of else None), state="loaded_for_snapshot" if option_symbols else "not_required", method_version="refdata_exact_option_quotes_v1"))
        except Exception as exc:
            warnings.append(f"option_quotes_unavailable:{type(exc).__name__}")
            sources.append(SourceState(name="exact_option_quotes", state="unavailable", method_version="refdata_exact_option_quotes_v1"))
        option_quotes = {
            str(row.get("canonical_occ_symbol") or ""): dict(row)
            for row in option_quote_payload.get("data") or []
            if row.get("canonical_occ_symbol")
        }
        option_term_rows: dict[str, dict[str, Any]] = {}
        if option_symbols:
            try:
                option_terms_payload = self.reference_client.option_contract_terms(
                    tuple(dict.fromkeys(option_symbols))
                )
                option_term_rows = {
                    str(row.get("canonical_occ_symbol") or ""): dict(row)
                    for row in option_terms_payload.get("data") or []
                    if row.get("canonical_occ_symbol")
                }
                for canonical, row in option_term_rows.items():
                    try:
                        root = parse_occ_symbol(canonical).root
                    except ValueError:
                        continue
                    subtype = _option_subtype_from_classification(
                        classifications.get(root)
                    )
                    if (
                        subtype is not None
                        and str(row.get("asset_subtype") or "").lower()
                        in {"", "unknown"}
                    ):
                        row["asset_subtype"] = subtype
                        row["warnings"] = [
                            warning
                            for warning in row.get("warnings") or ()
                            if warning != "option_contract_subtype_unavailable"
                        ]
                terms_as_of = _optional_datetime(
                    (option_terms_payload.get("meta") or {}).get("as_of")
                )
                sources.append(
                    SourceState(
                        name="exact_option_contract_terms",
                        as_of=terms_as_of,
                        age_seconds=(
                            max(0.0, (now - terms_as_of).total_seconds())
                            if terms_as_of
                            else None
                        ),
                        state="loaded_for_snapshot",
                        method_version="refdata_exact_option_terms_v4",
                    )
                )
            except Exception as exc:
                warnings.append(
                    f"option_contract_terms_unavailable:{type(exc).__name__}"
                )
                sources.append(
                    SourceState(
                        name="exact_option_contract_terms",
                        state="unavailable",
                        method_version="refdata_exact_option_terms_v4",
                    )
                )
        futures_spec_rows: dict[str, dict[str, Any]] = {}
        if futures_symbols:
            if capabilities.get("available", {}).get(
                "futures_contract_specifications", False
            ):
                try:
                    spec_payload = self.reference_client.futures_specifications(
                        tuple(dict.fromkeys(futures_symbols))
                    )
                    futures_spec_rows = {
                        str(row.get("canonical_contract") or "").upper(): dict(row)
                        for row in spec_payload.get("data") or []
                        if row.get("canonical_contract")
                    }
                    spec_as_of = _optional_datetime(
                        (spec_payload.get("meta") or {}).get("as_of")
                    )
                    sources.append(
                        SourceState(
                            name="futures_contract_specifications",
                            as_of=spec_as_of,
                            age_seconds=(
                                max(0.0, (now - spec_as_of).total_seconds())
                                if spec_as_of
                                else None
                            ),
                            state="loaded_for_snapshot",
                            method_version="canonical_instrument_terms_v4",
                        )
                    )
                except Exception as exc:
                    warnings.append(
                        f"futures_specifications_unavailable:{type(exc).__name__}"
                    )
            if not futures_spec_rows:
                sources.append(
                    SourceState(
                        name="futures_contract_specifications",
                        state="configured_fallback_or_unavailable",
                        method_version="risk_static_futures_terms_2026_08_v1",
                    )
                )
        resolved_symbol_types = []
        for symbol, broker_type in symbol_types:
            try:
                parsed_occ = parse_occ_symbol(symbol)
            except ValueError:
                classification_symbol = symbol
            else:
                classification_symbol = parsed_occ.root
            resolved_type = broker_type or str(
                classifications.get(classification_symbol, {}).get("security_type")
                or ""
            )
            resolved_symbol_types.append((symbol, resolved_type))
        # Resolve canonical contract economics before pricing or aggregating derivatives.
        terms_by_symbol = resolve_instrument_terms(
            resolved_symbol_types,
            option_terms=option_term_rows,
            futures_specs=futures_spec_rows,
            context=self.context,
        )
        curve = None
        if option_symbols and capabilities.get("available", {}).get("risk_free_curve"):
            try:
                curve_payload = self.reference_client.risk_free_curve(now.date())
                curve = curve_payload.get("data")
                curve_as_of = _optional_datetime((curve_payload.get("meta") or {}).get("as_of"))
                sources.append(SourceState(name="risk_free_curve", as_of=curve_as_of, age_seconds=(max(0.0, (now-curve_as_of).total_seconds()) if curve_as_of else None), state="loaded_for_snapshot", method_version=str((curve or {}).get("curve_version") or "fred_simple_annual_anchors_v1")))
            except Exception as exc:
                warnings.append(f"risk_free_curve_unavailable:{type(exc).__name__}")
                sources.append(SourceState(name="risk_free_curve", state="unavailable", method_version="fred_simple_annual_anchors_v1"))
        dividend_events: dict[str, list[dict[str, Any]]] = {}
        if option_symbols:
            try:
                calendar_rows = self.reference_client.dividend_calendar(
                    start=now.date() - timedelta(days=365),
                    end=max(
                        (
                            terms.expiration_at.date()
                            for terms in terms_by_symbol.values()
                            if terms.expiration_at is not None
                        ),
                        default=now.date(),
                    ),
                )
                for row in calendar_rows:
                    root = str(row.get("symbol") or "").strip().upper()
                    if root:
                        dividend_events.setdefault(root, []).append(dict(row))
                sources.append(
                    SourceState(
                        name="dividend_calendar",
                        as_of=now,
                        age_seconds=0.0,
                        state="loaded_for_snapshot",
                        method_version="known_discrete_dividends_v1",
                    )
                )
            except Exception as exc:
                warnings.append(f"dividend_calendar_unavailable:{type(exc).__name__}")
        # Prepare one pricing batch using resolved terms, rates, and dividend inputs.
        tian_contracts = []
        escrowed_dividend_diagnostics: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(broker_positions):
            instrument = raw.get("instrument") or {}
            symbol = str(instrument.get("symbol") or "").strip().upper()
            try:
                occ = parse_occ_symbol(symbol)
            except ValueError:
                continue
            quote = option_quotes.get(occ.canonical, {})
            underlying_quote = general_quotes.get(occ.root, {})
            spot = quote.get("underlying_price") or underlying_quote.get("mark")
            if not isinstance(spot, (int, float)) or spot <= 0:
                continue
            terms = terms_by_symbol.get(symbol)
            expiration_at = terms.expiration_at if terms is not None else None
            if expiration_at is None:
                warnings.append(f"{occ.canonical}:expiration_terms_unavailable")
                continue
            tenor_days = max(0.0, (expiration_at - now).total_seconds() / 86400)
            rate = None
            if curve:
                try:
                    rate = interpolate_risk_free_rate(
                        curve,
                        valuation_date=now.date(),
                        tenor_days=tenor_days,
                        max_age_days=int(self.context.get("risk.rates.max_age_days", 5)),
                        allow_flat_extrapolation=_as_bool(self.context.get("risk.rates.allow_flat_extrapolation", True)),
                    ).continuous_annual_rate_decimal
                except ValueError as exc:
                    warnings.append(f"{occ.canonical}:curve_unusable:{type(exc).__name__}")
            if rate is None:
                configured = self.context.get("risk.rates.configured_flat_continuous_rate")
                rate = float(configured) if configured not in (None, "") else None
            source_iv = _optional_positive_finite(quote.get("provider_iv"))
            if quote.get("provider_iv") is not None and source_iv is None:
                warnings.append(f"{occ.canonical}:provider_iv_unusable")
            dividend_yield = None
            events = dividend_events.get(occ.root, [])
            if events:
                try:
                    dividend_yield = derive_trailing_cash_yield(
                        events,
                        spot=float(spot),
                        valuation_date=now.date(),
                    ).continuous_yield
                except ValueError:
                    warnings.append(f"{occ.canonical}:dividend_inputs_unusable")
            if dividend_yield is None:
                dividend_yield = float(
                    self.context.get("risk.dividends.default_continuous_yield", 0.0)
                )
            future_cash_dividends = [
                event
                for event in events
                if (
                    (event_date := _event_date(event)) is not None
                    and now.date() <= event_date <= expiration_at.date()
                    and _ordinary_cash_dividend(event)
                )
            ]
            if terms is None or terms.exercise_style not in {"american", "european"}:
                warnings.append(f"{occ.canonical}:exercise_style_unknown_current_greeks_unavailable")
                continue
            pricing_spot = float(spot)
            pricing_dividend_yield = dividend_yield
            if future_cash_dividends and terms.exercise_style == "american":
                if not _as_bool(
                    self.context.get(
                        "risk.options.tian.escrowed_dividend_mode_enabled", True
                    )
                ):
                    warnings.append(
                        f"{occ.canonical}:discrete_dividend_early_exercise_solver_unavailable"
                    )
                    continue
                try:
                    adjustment = escrowed_dividend_adjustment(
                        future_cash_dividends,
                        spot=float(spot),
                        valuation_at=now,
                        expiration_at=expiration_at,
                        continuous_rate=float(rate),
                    )
                except (TypeError, ValueError):
                    warnings.append(f"{occ.canonical}:escrowed_dividend_inputs_unusable")
                    continue
                pricing_spot = adjustment.adjusted_spot
                pricing_dividend_yield = 0.0
                escrowed_dividend_diagnostics[occ.canonical] = {
                    "model": adjustment.algorithm_version,
                    "original_spot": adjustment.original_spot,
                    "adjusted_spot": adjustment.adjusted_spot,
                    "dividend_present_value": adjustment.present_value,
                    "event_count": adjustment.event_count,
                    "event_dates": [item.isoformat() for item in adjustment.event_dates],
                }
                warnings.append(f"{occ.canonical}:escrowed_dividend_model_applied")
            try:
                contract = IvGreeksContract(
                    request_id=f"{raw.get('account_id')}:{occ.canonical}:{index}",
                    canonical_occ_symbol=occ.canonical,
                    underlying_symbol=occ.root,
                    put_call=occ.put_call,
                    exercise_type={
                        "european": "EUROPEAN",
                        "american": "AMERICAN",
                    }[terms.exercise_style],
                    strike=float(occ.strike),
                    expiration_at=expiration_at,
                    spot=pricing_spot,
                    bid=quote.get("bid"),
                    ask=quote.get("ask"),
                    mark=quote.get("mark"),
                    last=quote.get("last"),
                    last_trade_at=quote.get("trade_time"),
                    quote_time=quote.get("quote_time"),
                    source_iv=source_iv,
                    risk_free_rate=rate,
                    dividend_yield=pricing_dividend_yield,
                )
            except ValidationError:
                warnings.append(f"{occ.canonical}:tian_contract_unusable")
                continue
            tian_contracts.append(contract)
        greeks_by_symbol: dict[str, dict[str, Any]] = {}
        if tian_contracts:
            tian = self.tian_service.solve(IvGreeksRequest(valuation_at=now, contracts=tian_contracts))
            sources.append(SourceState(name="tian_iv_greeks", as_of=tian.as_of, age_seconds=max(0.0, (now-tian.as_of).total_seconds()), state="calculated", method_version=tian.method_version))
            greeks_by_symbol = {
                item.canonical_occ_symbol: item.model_dump(mode="python") for item in tian.data
            }
        for canonical, greek in greeks_by_symbol.items():
            try:
                root = parse_occ_symbol(canonical).root
            except ValueError:
                continue
            greek["dividend_inputs"] = list(dividend_events.get(root, ()))
            if canonical in escrowed_dividend_diagnostics:
                diagnostics = dict(greek.get("solver_diagnostics") or {})
                diagnostics["escrowed_dividend"] = escrowed_dividend_diagnostics[canonical]
                greek["solver_diagnostics"] = diagnostics
        # Combine broker holdings with reference evidence into immutable risk positions.
        normalized = []
        for raw in broker_positions:
            instrument = raw.get("instrument") or {}
            symbol = str(instrument.get("symbol") or "").strip().upper()
            try:
                occ = parse_occ_symbol(symbol)
            except ValueError:
                occ = None
            option_quote = option_quotes.get(occ.canonical) if occ else None
            classification_symbol = occ.root if occ else symbol
            normalized.append(
                normalize_broker_position(
                    raw,
                    account_id=str(raw.get("account_id")),
                    quote=general_quotes.get(occ.root if occ else symbol),
                    option_quote=option_quote,
                    classification=classifications.get(classification_symbol),
                    instrument_terms=terms_by_symbol.get(symbol),
                    greek=greeks_by_symbol.get(occ.canonical) if occ else None,
                    context=self.context,
                    as_of=now,
                )
            )
        # Track missing balances explicitly so an unknown total never becomes zero.
        portfolio_value = 0.0
        portfolio_value_seen = False
        maintenance = 0.0
        maintenance_seen = False
        margin_balance = 0.0
        margin_balance_seen = False
        balances = []
        for account in accounts:
            current = account.current_balances
            initial = account.initial_balances
            value = current.equity if current and current.equity is not None else (
                initial.liquidation_value if initial else None
            )
            requirement = current.maintenance_requirement if current else None
            account_margin_balance = current.margin_balance if current else None
            if value is not None:
                portfolio_value += value
                portfolio_value_seen = True
            if requirement is not None:
                maintenance += requirement
                maintenance_seen = True
            if account_margin_balance is not None:
                margin_balance += account_margin_balance
                margin_balance_seen = True
            balances.append(
                {
                    "account_id": account.account_id,
                    "portfolio_value_usd": value,
                    "net_liquidating_value_usd": value,
                    "net_liquidating_value_source": (
                        "currentBalances.equity"
                        if current and current.equity is not None
                        else "initialBalances.liquidationValue"
                    ),
                    "maintenance_requirement_usd": requirement,
                    "maintenance_requirement_source": "currentBalances.maintenanceRequirement",
                    "margin_balance_usd": account_margin_balance,
                    "margin_balance_source": "currentBalances.marginBalance",
                    "available_funds_usd": current.available_funds if current else None,
                    "buying_power_usd": current.buying_power if current else None,
                }
            )
        kpis = portfolio_kpis(
            normalized,
            portfolio_value_usd=portfolio_value if portfolio_value_seen else None,
            maintenance_requirement_usd=maintenance if maintenance_seen else None,
            margin_loan_balance_usd=margin_balance if margin_balance_seen else None,
            context=self.context,
        )
        warnings.extend(kpis.warnings)
        warnings.extend(
            warning for position in normalized for warning in position.data_quality.warnings
        )
        modeled = sum(
            1
            for position in normalized
            if position.data_quality.status in {"usable", "usable_with_warnings"}
        )
        quality_status = "usable" if not warnings else "usable_with_warnings"
        if normalized and modeled == 0:
            quality_status = "insufficient_data"
        quality = DataQuality(
            status=quality_status,
            sources=tuple(sources),
            coverage=Coverage(
                positions_total=len(normalized),
                positions_modeled=modeled,
                market_value_percent=kpis.modeled_market_value_percent,
                gross_delta_percent=kpis.modeled_gross_delta_percent,
            ),
            risk_coverage=portfolio_risk_coverage(
                normalized,
                portfolio_value_usd=portfolio_value if portfolio_value_seen else None,
                context=self.context,
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        )
        # Derive snapshot identity from the normalized economic state.
        fingerprint = {
            "accounts": account_ids,
            "positions": [item.model_dump(mode="json") for item in normalized],
            "portfolio_value": portfolio_value if portfolio_value_seen else None,
            "configuration_version": configuration_version,
        }
        snapshot_id = "sha256:" + hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        return PortfolioSnapshotResponse(
            portfolio_snapshot_id=snapshot_id,
            configuration_version=configuration_version,
            as_of=now,
            display_timezone=str(self.context.get("risk.display_timezone", "America/Los_Angeles")),
            data_quality=quality,
            data=PortfolioSnapshotData(
                account_ids=account_ids,
                positions=normalized,
                kpis=kpis,
                balances=balances,
                capabilities=capabilities,
            ),
            cache=CacheMeta(state="miss", age_seconds=0.0, source_as_of=now),
        )


def _normalize_scope(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(dict.fromkeys(str(item).strip() for item in values if str(item).strip())))
    if not normalized or "all" in {item.lower() for item in normalized}:
        return ("all",)
    return normalized


def _configuration_version(context: Mapping[str, Any]) -> str:
    safe = {
        str(key): value
        for key, value in context.items()
        if str(key).startswith("risk.")
        and not any(secret in str(key).lower() for secret in ("password", "secret", "token", "api_key"))
        and isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(safe, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _cache_key(scope: tuple[str, ...], configuration_version: str) -> str:
    return "snapshot:" + hashlib.sha256(
        (",".join(scope) + "|" + configuration_version).encode("utf-8")
    ).hexdigest()


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _optional_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return _parse_datetime(str(value))
    except ValueError:
        return None


def _optional_positive_finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) and converted > 0 else None


def _option_subtype_from_classification(
    classification: Mapping[str, Any] | None,
) -> str | None:
    security_type = str((classification or {}).get("security_type") or "").upper()
    if security_type in {"ETF", "EXCHANGE_TRADED_FUND"}:
        return "etf_option"
    if security_type in {
        "EQUITY",
        "STOCK",
        "COMMON_STOCK",
        "ADR",
        "DEPOSITARY_RECEIPT",
    }:
        return "equity_option"
    if security_type in {"INDEX", "MARKET_INDEX"}:
        return "index_option"
    return None


def _future_symbol(value: str) -> bool:
    return _LISTED_FUTURE.fullmatch(value.strip().upper()) is not None


def _event_date(event: Mapping[str, Any]):
    raw = event.get("ex_dividend_date") or event.get("date")
    try:
        return datetime.fromisoformat(str(raw)[:10]).date()
    except ValueError:
        return None


def _ordinary_cash_dividend(event: Mapping[str, Any]) -> bool:
    classification = str(
        event.get("distribution_type") or event.get("type") or "cash"
    ).lower()
    return not any(
        marker in classification
        for marker in ("special", "return of capital", "stock dividend")
    )


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "NoSnapshotLoaded",
    "PortfolioApplicationService",
    "SnapshotMismatch",
]
