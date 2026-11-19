"""Bounded client for published Reference Data REST contracts."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
import logging
import re
from threading import RLock
from time import monotonic, sleep
from typing import Any, Mapping, Sequence
from urllib.parse import quote
import uuid

import httpx

from algotrading.refdata.stock_analysis.etf_universe import ETF_UNIVERSE


class ReferenceDataClientError(RuntimeError):
    """Non-secret structured RefData failure safe for Risk diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        server_error_code: str | None = None,
        server_message: str | None = None,
        server_details: Mapping[str, Any] | None = None,
        request_id: str | None = None,
        endpoint: str | None = None,
        requested_start: str | None = None,
        requested_end: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.server_error_code = server_error_code
        self.server_message = server_message
        self.server_details = dict(server_details) if server_details is not None else None
        self.request_id = request_id
        self.endpoint = endpoint
        self.requested_start = requested_start
        self.requested_end = requested_end
        self.retryable = retryable

    def diagnostic(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "http_status": self.http_status,
                "server_error_code": self.server_error_code,
                "server_message": self.server_message,
                "server_details": self.server_details,
                "request_id": self.request_id,
                "endpoint": self.endpoint,
                "requested_start": self.requested_start,
                "requested_end": self.requested_end,
                "retryable": self.retryable,
            }.items()
            if value is not None
        }


class ReferenceDataClient:
    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.context = context
        self.base_url = str(
            context.get("risk.refdata.base_url", "http://127.0.0.1:8787")
        ).rstrip("/")
        timeout = httpx.Timeout(
            connect=float(context.get("risk.refdata.connect_timeout_seconds", 2)),
            read=float(context.get("risk.refdata.read_timeout_seconds", 10)),
            write=float(context.get("risk.refdata.write_timeout_seconds", 10)),
            pool=float(context.get("risk.refdata.pool_timeout_seconds", 2)),
        )
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=int(context.get("risk.refdata.max_connections", 8)),
                max_keepalive_connections=int(context.get("risk.refdata.max_keepalive_connections", 4)),
            ),
            headers={"Accept": "application/json"},
        )
        self.logger = context.get("logger") or logging.getLogger("algotrading.risk.market_inputs")
        self._capabilities_lock = RLock()
        self._capabilities_cache: tuple[float, dict[str, Any]] | None = None
        self._risk_free_curve_lock = RLock()
        self._risk_free_curve_cache: dict[
            date, tuple[float, dict[str, Any]]
        ] = {}
        self._etf_exposure_lock = RLock()
        self._etf_exposure_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def capabilities(self) -> dict[str, Any]:
        ttl = max(
            1.0,
            float(self.context.get("risk.refdata.capabilities_cache_seconds", 300)),
        )
        now = monotonic()
        with self._capabilities_lock:
            cached = self._capabilities_cache
            if cached is not None and now - cached[0] <= ttl:
                return dict(cached[1])
            data = dict(self._request("GET", "/api/v1/capabilities")["data"])
            self._capabilities_cache = (now, data)
            return dict(data)

    def option_history_maximum_days(self) -> int | None:
        supported = self.capabilities().get("supported") or {}
        value = supported.get("option_market_history_max_days")
        return int(value) if value is not None else None

    def quotes(self, symbols: Sequence[str]) -> list[dict[str, Any]]:
        if not symbols:
            return []
        # Split provider-bound requests without changing the caller's symbol order.
        maximum = int(self.context.get("risk.refdata.maximum_quote_batch", 1000))
        rows: list[dict[str, Any]] = []
        for chunk in _chunks(tuple(dict.fromkeys(symbols)), maximum):
            payload = self._request(
                "GET",
                "/api/v1/quotes",
                params=[("symbols", ",".join(chunk))],
            )
            rows.extend(payload.get("data") or [])
        return rows

    def security_classifications(
        self, symbols: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Resolve held symbols through an exact, bounded universe query.

        Scanning the first pages of the full universe is not a valid lookup:
        holdings outside those pages silently lost their classification.  The
        universe API accepts repeated ``symbols`` parameters, so this remains
        database-backed and avoids profile-provider calls during snapshot
        construction.
        """
        wanted = tuple(
            sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
        )
        if not wanted:
            return []
        batch_size = min(
            1000,
            max(
                1,
                int(self.context.get("risk.refdata.maximum_classification_batch", 250)),
            ),
        )
        matched: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(wanted, batch_size):
            params = [("symbols", symbol) for symbol in chunk]
            params.extend(
                [
                    ("active_only", "false"),
                    ("include_stale_daily_bars", "true"),
                    ("limit", str(len(chunk))),
                    ("offset", "0"),
                ]
            )
            payload = self._request(
                "GET",
                "/api/v1/universe/symbols",
                params=params,
            )
            rows = payload.get("data") or []
            for row in rows:
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol in wanted:
                    matched[symbol] = dict(row)
            if len(matched) == len(wanted):
                break
        for symbol in wanted:
            definition = ETF_UNIVERSE.get(symbol)
            if (
                definition is None
                or definition.category != "GICS_SECTOR"
                or definition.exposure_method != "DECLARED_SECTOR"
            ):
                continue
            row = matched.setdefault(symbol, {
                "symbol": symbol,
                "security_type": "ETF",
            })
            row.update({
                "taxonomy": "GICS",
                "sector": definition.exposure_name,
                "company_name": row.get("company_name") or definition.fund_name,
            })
        return [matched[symbol] for symbol in sorted(matched)]

    def etf_exposure(self, symbol: str) -> dict[str, Any]:
        """Cache a fund-exposure lookup, including bounded negative responses."""
        normalized = str(symbol).strip().upper()
        if not re.fullmatch(r"[A-Z0-9.-]{1,32}", normalized):
            raise ValueError("ETF symbol contains unsupported characters")
        now = monotonic()
        with self._etf_exposure_lock:
            cached = self._etf_exposure_cache.get(normalized)
            if cached is not None and cached[0] > now:
                return dict(cached[1])
        try:
            payload = self._request("GET", f"/api/v1/etfs/{quote(normalized)}/exposure")
            data = dict(payload.get("data") or {})
            if data.get("symbol") != normalized:
                raise ReferenceDataClientError("ETF exposure symbol mismatch")
            # Cache unavailable ETF evidence briefly to avoid repeated provider failures.
            ttl = 86400.0 if data.get("status") == "available" else 3600.0
        except ReferenceDataClientError:
            data = {"symbol": normalized, "status": "unavailable", "reason": "provider_unavailable"}
            ttl = 3600.0
        with self._etf_exposure_lock:
            self._etf_exposure_cache[normalized] = (monotonic() + ttl, data)
        return dict(data)

    def universe_symbols(self, symbols: Sequence[str]) -> list[dict[str, Any]]:
        """Return exact, active, bounded universe rows for candidate discovery."""
        wanted = tuple(sorted({
            str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
        }))
        if not wanted:
            return []
        batch_size = min(
            250,
            max(1, int(self.context.get("risk.refdata.maximum_universe_batch", 100))),
        )
        matched: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(wanted, batch_size):
            params = [("symbols", symbol) for symbol in chunk]
            params.extend([
                ("active_only", "true"),
                ("include_stale_daily_bars", "false"),
                ("limit", str(len(chunk))),
                ("offset", "0"),
            ])
            payload = self._request(
                "GET", "/api/v1/universe/symbols", params=params
            )
            for row in payload.get("data") or []:
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol in wanted:
                    matched[symbol] = dict(row)
        return [matched[symbol] for symbol in sorted(matched)]

    def current_price(self, symbol: str) -> dict[str, Any]:
        """Return one published current-price object with its response metadata."""
        normalized = str(symbol).strip().upper()
        if not re.fullmatch(r"/?[A-Z0-9][A-Z0-9.\-]{0,30}", normalized):
            raise ValueError("symbol is not a supported equity or futures symbol")
        # A leading slash is provider notation (for example ``/ESU26``), not
        # part of RefData's path parameter.  Percent-encoding it does not help:
        # ASGI servers decode ``%2F`` before route matching and the request then
        # becomes ``.../symbols//ESU26/current-price``.  RefData accepts the
        # canonical contract and adds the provider slash at its own boundary.
        api_symbol = normalized.removeprefix("/")
        return self._request(
            "GET",
            f"/api/v1/symbols/{quote(api_symbol, safe='')}/current-price",
            params={"view": "price"},
        )

    def current_futures_price(self, contract: str) -> dict[str, Any]:
        """Return one price through RefData's explicit futures contract."""
        canonical = str(contract).strip().upper().removeprefix("/")
        if not re.fullmatch(
            r"[A-Z0-9]{1,6}[FGHJKMNQUVXZ][0-9]{1,2}", canonical
        ):
            raise ValueError("contract is not a supported futures symbol")
        return self._request(
            "GET",
            f"/api/v1/futures/{quote(canonical, safe='')}/current-price",
            params={"view": "price"},
        )

    def option_quotes(self, symbols: Sequence[str]) -> dict[str, Any]:
        if not symbols:
            return {"data": [], "missing_symbols": [], "meta": {}}
        maximum = int(self.context.get("risk.quotes.maximum_option_contracts_per_request", 500))
        rows, missing = [], []
        meta: dict[str, Any] = {}
        for chunk in _chunks(tuple(symbols), maximum):
            payload = self._request(
                "POST", "/api/v1/options/quotes", json={"symbols": list(chunk)}
            )
            rows.extend(payload.get("data") or [])
            missing.extend(payload.get("missing_symbols") or [])
            meta = payload.get("meta") or meta
        return {"data": rows, "missing_symbols": missing, "meta": meta}

    def option_contract_terms(self, symbols: Sequence[str]) -> dict[str, Any]:
        if not symbols:
            return {"data": [], "meta": {}}
        maximum = int(
            self.context.get(
                "risk.instrument_terms.maximum_option_contracts_per_request", 500
            )
        )
        rows: list[dict[str, Any]] = []
        meta: dict[str, Any] = {}
        for chunk in _chunks(tuple(dict.fromkeys(symbols)), maximum):
            payload = self._request(
                "POST",
                "/api/v1/options/contract-terms/query",
                json={"symbols": list(chunk)},
            )
            rows.extend(payload.get("data") or [])
            meta = payload.get("meta") or meta
        return {"data": rows, "meta": meta}

    def risk_free_curve(self, as_of: date) -> dict[str, Any]:
        ttl = max(
            0.0,
            float(
                self.context.get(
                    "risk.refdata.risk_free_curve_cache_seconds", 30 * 60
                )
            ),
        )
        with self._risk_free_curve_lock:
            now = monotonic()
            self._risk_free_curve_cache = {
                key: value
                for key, value in self._risk_free_curve_cache.items()
                if now - value[0] <= ttl
            }
            # Copy cached curve payloads so callers cannot mutate shared reference data.
            cached = self._risk_free_curve_cache.get(as_of)
            if cached is not None:
                return deepcopy(cached[1])

        # Do not hold the global cache lock across network I/O. This single-date
        # contract is also the rolling-upgrade fallback for older RefData
        # deployments, where distinct dates may still be fetched concurrently.
        payload = self._request(
            "GET",
            "/api/v1/rates/risk-free-curve",
            params={"as_of": as_of.isoformat()},
        )
        with self._risk_free_curve_lock:
            now = monotonic()
            cached = self._risk_free_curve_cache.get(as_of)
            if cached is not None and now - cached[0] <= ttl:
                return deepcopy(cached[1])
            self._risk_free_curve_cache[as_of] = (now, deepcopy(payload))
        return payload

    def risk_free_curves(
        self, as_of_dates: Sequence[date]
    ) -> dict[date, dict[str, Any]]:
        """Return point-in-time curves with one request for all local misses."""
        requested = tuple(sorted(set(as_of_dates)))
        if not requested:
            return {}
        ttl = max(
            0.0,
            float(
                self.context.get(
                    "risk.refdata.risk_free_curve_cache_seconds", 30 * 60
                )
            ),
        )
        with self._risk_free_curve_lock:
            now = monotonic()
            self._risk_free_curve_cache = {
                key: value
                for key, value in self._risk_free_curve_cache.items()
                if now - value[0] <= ttl
            }
            missing = [
                as_of for as_of in requested if as_of not in self._risk_free_curve_cache
            ]
        # Fetch only curve dates absent from the bounded local cache.
        if missing:
            payload = self._request(
                "POST",
                "/api/v1/rates/risk-free-curves/query",
                json={"as_of_dates": [item.isoformat() for item in missing]},
            )
            retrieved_at = monotonic()
            rows = payload.get("data") or []
            with self._risk_free_curve_lock:
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    try:
                        as_of = date.fromisoformat(str(row["requested_as_of"]))
                    except (KeyError, ValueError):
                        continue
                    if as_of in missing:
                        self._risk_free_curve_cache[as_of] = (
                            retrieved_at,
                            {"data": deepcopy(dict(row)), "meta": deepcopy(payload.get("meta") or {})},
                        )
        with self._risk_free_curve_lock:
            return {
                as_of: deepcopy(self._risk_free_curve_cache[as_of][1])
                for as_of in requested
                if as_of in self._risk_free_curve_cache
            }

    def option_chain(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/options/chains/query", json=dict(payload))

    def option_expirations(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST", "/api/v1/options/expirations/query", json=dict(payload)
        )

    def option_market_history(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Request exact historical option marks without automatic POST retries."""
        return self._request(
            "POST", "/api/v1/options/market-history/query", json=dict(payload)
        )

    def futures_specifications(self, contracts: Sequence[str]) -> dict[str, Any]:
        canonical = [
            str(contract).strip().upper().removeprefix("/")
            for contract in contracts
        ]
        return self._request(
            "POST",
            "/api/v1/futures/contract-specifications",
            json={"contracts": canonical},
        )

    def bars(
        self,
        symbols: Sequence[str],
        *,
        start: date,
        end: date,
        include_current_quote: bool = False,
        remote_query: bool = False,
    ) -> list[dict[str, Any]]:
        if not symbols:
            return []
        payload = self._request(
            "GET",
            "/api/v1/bars",
            params={
                "symbols": ",".join(dict.fromkeys(symbols)),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "include_current_quote": str(include_current_quote).lower(),
                "remote-query": str(remote_query).lower(),
            },
        )
        return list(payload.get("data") or [])

    def finra_short_volume(
        self,
        symbols: Sequence[str],
        *,
        as_of: date,
        session_limit: int,
    ) -> dict[str, Any]:
        """Return bounded consolidated FINRA observations from RefData."""
        wanted = tuple(dict.fromkeys(
            str(symbol).strip().upper()
            for symbol in symbols
            if str(symbol).strip()
        ))
        if not wanted:
            return {"sessions": [], "observations": []}
        maximum = int(
            self.context.get("risk.refdata.maximum_short_volume_batch", 500)
        )
        rows: list[dict[str, Any]] = []
        sessions: set[str] = set()
        for chunk in _chunks(wanted, max(1, min(maximum, 500))):
            payload = self._request(
                "GET",
                "/api/v1/finra/short-volume",
                params={
                    "symbols": ",".join(chunk),
                    "as_of": as_of.isoformat(),
                    "session_limit": str(session_limit),
                },
            )
            data = payload.get("data") or {}
            rows.extend(data.get("observations") or [])
            sessions.update(str(item) for item in data.get("sessions") or [])
        return {"sessions": sorted(sessions), "observations": rows}

    def historical_minute_bars(
        self,
        symbol: str,
        *,
        start_at: datetime,
        end_at: datetime,
    ) -> list[dict[str, Any]]:
        """Return bounded stored one-minute bars for one underlying."""
        payload = self._request(
            "GET",
            f"/api/v1/symbols/{str(symbol).strip().upper()}/bars",
            params={
                "start": start_at.astimezone(timezone.utc).isoformat(),
                "end": end_at.astimezone(timezone.utc).isoformat(),
                "interval": "1m",
                "market_hours_only": "false",
                "remote-query": "false",
            },
        )
        return list(payload.get("data") or [])

    def dividends(self, symbol: str, *, start: date, end: date) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            f"/api/v1/symbols/{symbol}/dividends",
            params={"start_date": start.isoformat(), "end_date": end.isoformat()},
        )
        return list(payload.get("data") or [])

    def dividend_calendar(self, *, start: date, end: date) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/api/v1/calendars/dividends",
            params={"start_date": start.isoformat(), "end_date": end.isoformat()},
        )
        return list(payload.get("data") or [])

    def earnings(self, symbol: str, *, start: date, end: date) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            f"/api/v1/symbols/{symbol}/earnings",
            params={"start_date": start.isoformat(), "end_date": end.isoformat()},
        )
        return list(payload.get("data") or [])

    def earnings_calendar(self, *, start: date, end: date) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/api/v1/calendars/earnings",
            params={"start_date": start.isoformat(), "end_date": end.isoformat()},
        )
        return list(payload.get("data") or [])

    def fomc_calendar(self, *, start: date, end: date) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/api/v1/calendars/fomc",
            params={"start_date": start.isoformat(), "end_date": end.isoformat()},
        )
        return list(payload.get("data") or [])

    def economic_calendar(self, *, start: date, end: date) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/api/v1/calendars/economic",
            params={"start_date": start.isoformat(), "end_date": end.isoformat()},
        )
        return list(payload.get("data") or [])

    def company_profile(self, symbol: str) -> dict[str, Any]:
        """Return the provider-refreshed, durable company profile contract."""
        payload = self._request(
            "GET",
            f"/api/v1/symbols/{str(symbol).strip().upper()}/company-information",
        )
        return dict(payload.get("data") or {})

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        attempts = max(1, int(self.context.get("risk.refdata.retry_count", 1)) + 1)
        # Keep one request ID across attempts for tracing a logical RefData call.
        request_id = uuid.uuid4().hex
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.client.request(
                    method,
                    path,
                    headers={"X-Request-ID": request_id},
                    **kwargs,
                )
                # Retry server failures only for safe GET requests.
                if response.status_code >= 500 and attempt + 1 < attempts and method == "GET":
                    sleep(min(0.2 * (2**attempt), 1.0))
                    continue
                if response.status_code >= 400:
                    error: Mapping[str, Any] = {}
                    try:
                        decoded = response.json()
                        if isinstance(decoded, Mapping) and isinstance(
                            decoded.get("error"), Mapping
                        ):
                            error = decoded["error"]
                    except Exception:
                        pass
                    code = str(error.get("code") or "unknown")
                    server_message = str(error.get("message") or "") or None
                    start, end = _requested_interval(kwargs)
                    raise ReferenceDataClientError(
                        f"Reference Data request failed status={response.status_code} code={code}",
                        http_status=response.status_code,
                        server_error_code=code,
                        server_message=server_message,
                        server_details=(
                            error.get("details")
                            if isinstance(error.get("details"), Mapping)
                            else None
                        ),
                        request_id=(
                            str(error.get("request_id") or "")
                            or getattr(response, "headers", {}).get("X-Request-ID")
                            or request_id
                        ),
                        endpoint=path,
                        requested_start=start,
                        requested_end=end,
                        retryable=response.status_code >= 500 or response.status_code == 429,
                    )
                payload = response.json()
                if not isinstance(payload, dict):
                    start, end = _requested_interval(kwargs)
                    raise ReferenceDataClientError(
                        "Reference Data response must be an object",
                        endpoint=path,
                        request_id=request_id,
                        requested_start=start,
                        requested_end=end,
                    )
                return payload
            except ReferenceDataClientError:
                raise
            except (httpx.TransportError, ValueError) as exc:
                last_error = exc
                if method != "GET" or attempt + 1 >= attempts:
                    break
                sleep(min(0.2 * (2**attempt), 1.0))
        start, end = _requested_interval(kwargs)
        raise ReferenceDataClientError(
            f"Reference Data request unavailable ({type(last_error).__name__})",
            server_error_code="TRANSPORT_UNAVAILABLE",
            server_message=str(last_error) if last_error is not None else None,
            request_id=request_id,
            endpoint=path,
            requested_start=start,
            requested_end=end,
            retryable=True,
        ) from last_error


def _chunks(values: Sequence[str], size: int) -> list[tuple[str, ...]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [tuple(values[index : index + size]) for index in range(0, len(values), size)]


def _requested_interval(kwargs: Mapping[str, Any]) -> tuple[str | None, str | None]:
    body = kwargs.get("json")
    params = kwargs.get("params")
    source = body if isinstance(body, Mapping) else params if isinstance(params, Mapping) else {}
    start = source.get("start_at", source.get("start"))
    end = source.get("end_at", source.get("end"))
    return (
        str(start) if start not in (None, "") else None,
        str(end) if end not in (None, "") else None,
    )


__all__ = ["ReferenceDataClient", "ReferenceDataClientError"]
