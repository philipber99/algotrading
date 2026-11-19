"""Direct MySQL persistence for nightly option-position price snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from algotrading.ContextFields import (
    PARAM_REFDATA_DB_DEFAULTDB,
    PARAM_REFDATA_DB_HOSTNAME,
    PARAM_REFDATA_DB_PASSWORD,
    PARAM_REFDATA_DB_USERNAME,
)
from algotrading.refdata.repositories.base import row_dict
from algotrading.refdata.repositories.connection import StockAnalysisConnectionFactory
from algotrading.refdata.api.option_symbols import parse_occ_symbol

from .models import (
    CapturedChainSnapshot,
    CapturedOptionPrice,
    CaptureSummary,
    MarkSource,
    OptionSurfaceObservation,
    OptionSurfaceSnapshot,
    OptionSurfaceSnapshotHeader,
)


class OptionPriceRepositoryUnavailable(RuntimeError):
    pass


class OptionSurfaceHistoryLimitExceeded(RuntimeError):
    pass


class OptionPriceRepository:
    def __init__(
        self,
        connection_factory: StockAnalysisConnectionFactory,
        *,
        maximum_surface_symbols: int = 500,
        maximum_surface_snapshots: int = 2_000,
        maximum_surface_points: int = 100_000,
        surface_point_batch_size: int = 500,
    ) -> None:
        self.connection_factory = connection_factory
        self.maximum_surface_symbols = max(1, int(maximum_surface_symbols))
        self.maximum_surface_snapshots = max(1, int(maximum_surface_snapshots))
        self.maximum_surface_points = max(1, int(maximum_surface_points))
        self.surface_point_batch_size = max(1, int(surface_point_batch_size))

    @classmethod
    def from_context(
        cls,
        context: Mapping[str, Any],
        *,
        connector: Any | None = None,
    ) -> "OptionPriceRepository":
        required = {
            "host": str(
                context.get("risk.option_prices.sql.host")
                or context.get("risk.transactions.sql.host")
                or context.get(PARAM_REFDATA_DB_HOSTNAME)
                or ""
            ).strip(),
            "database": str(
                context.get("risk.option_prices.sql.database")
                or context.get("risk.transactions.sql.database")
                or context.get(PARAM_REFDATA_DB_DEFAULTDB)
                or ""
            ).strip(),
            "user": str(
                context.get("risk.option_prices.sql.user")
                or context.get("risk.transactions.sql.user")
                or context.get(PARAM_REFDATA_DB_USERNAME)
                or ""
            ).strip(),
            "password": str(
                context.get("risk.option_prices.sql.password")
                or context.get("risk.transactions.sql.password")
                or context.get(PARAM_REFDATA_DB_PASSWORD)
                or ""
            ),
        }
        if not required["host"] or not required["database"] or not required["user"]:
            raise OptionPriceRepositoryUnavailable(
                "Risk option-price MySQL host, database, and user are not configured"
            )
        # Adapt Risk settings to the shared SQL connection factory without logging them.
        translated = dict(context)
        translated[PARAM_REFDATA_DB_HOSTNAME] = required["host"]
        translated[PARAM_REFDATA_DB_DEFAULTDB] = required["database"]
        translated[PARAM_REFDATA_DB_USERNAME] = required["user"]
        translated[PARAM_REFDATA_DB_PASSWORD] = required["password"]
        translated["refdata.stock_analysis.sql.port"] = int(
            context.get(
                "risk.option_prices.sql.port",
                context.get("risk.transactions.sql.port", 3306),
            )
        )
        translated["refdata.stock_analysis.sql.connection_timeout_seconds"] = int(
            context.get(
                "risk.option_prices.sql.connection_timeout_seconds",
                context.get("risk.transactions.sql.connection_timeout_seconds", 5),
            )
        )
        translated["refdata.stock_analysis.sql.connection_attempts"] = 1
        translated["refdata.stock_analysis.sql.pool.enabled"] = True
        translated["refdata.stock_analysis.sql.pool.size"] = int(
            context.get(
                "risk.option_prices.sql.pool_size",
                context.get("risk.transactions.sql.pool_size", 4),
            )
        )
        return cls(
            StockAnalysisConnectionFactory(translated, connector=connector),
            maximum_surface_symbols=int(
                context.get("risk.pnl.max_option_symbols", 500)
            ),
            maximum_surface_snapshots=int(
                context.get("risk.option_prices.max_surface_snapshots", 2_000)
            ),
            maximum_surface_points=int(
                context.get("risk.option_prices.max_surface_points", 100_000)
            ),
            surface_point_batch_size=int(
                context.get("risk.option_prices.surface_point_batch_size", 500)
            ),
        )

    def save_capture(
        self,
        summary: CaptureSummary,
        prices: Sequence[CapturedOptionPrice],
        *,
        chain_snapshots: Sequence[CapturedChainSnapshot] = (),
        preserve_existing: bool = False,
    ) -> None:
        run_sql = (
            "INSERT INTO risk_option_price_capture_runs "
            "(run_id, captured_at, status, account_count, position_count, "
            "quote_count, missing_count, provider_class, chain_count, "
            "surface_point_count, finished_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        chain_sql = (
            "INSERT INTO risk_option_chain_snapshots "
            "(run_id, underlying_symbol, expiration_date, spot, spot_source, "
            "underlying_quote_time, snapshot_status, snapshot_flags, error_code) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        surface_sql = (
            "INSERT INTO risk_option_surface_points "
            "(chain_snapshot_id, canonical_occ_symbol, quote_time, trade_time, "
            "bid, ask, mark, last, open_interest, volume, observation_source, "
            "security_status) VALUES ("
            + ",".join(["%s"] * 12)
            + ")"
        )
        price_sql = self._price_sql(preserve_existing)
        try:
            # Persist the run, chain snapshots, surface points, and marks atomically.
            with self.connection_factory.transaction(dictionary=True) as (_, cursor, _):
                cursor.execute(
                    run_sql,
                    (
                        summary.run_id,
                        _mysql_utc(summary.captured_at),
                        summary.status,
                        summary.account_count,
                        summary.position_count,
                        summary.quote_count,
                        summary.missing_count,
                        summary.provider_class[:255],
                        summary.chain_count,
                        summary.surface_point_count,
                        _mysql_utc(summary.captured_at),
                    ),
                )
                snapshot_ids: dict[tuple[str, date], int] = {}
                for snapshot in chain_snapshots:
                    cursor.execute(
                        chain_sql,
                        self._chain_parameters(summary.run_id, snapshot),
                    )
                    chain_snapshot_id = getattr(cursor, "lastrowid", None)
                    if not isinstance(chain_snapshot_id, int) or chain_snapshot_id < 1:
                        raise RuntimeError("MySQL did not return a chain snapshot ID")
                    snapshot_key = (
                        snapshot.underlying_symbol,
                        snapshot.expiration_date,
                    )
                    snapshot_ids[snapshot_key] = chain_snapshot_id
                    if snapshot.points:
                        cursor.executemany(
                            surface_sql,
                            [
                                self._surface_parameters(chain_snapshot_id, point)
                                for point in snapshot.points
                            ],
                        )
                if prices:
                    cursor.executemany(
                        price_sql,
                        [
                            self._price_parameters(
                                summary.run_id,
                                item,
                                snapshot_ids.get(
                                    (
                                        item.position.underlying_symbol,
                                        item.position.expiration_date,
                                    )
                                ),
                            )
                            for item in prices
                        ],
                    )
        except Exception as exc:
            raise OptionPriceRepositoryUnavailable(
                "Risk option-price capture write failed"
            ) from exc

    def save_backfill_batch(
        self,
        summary: CaptureSummary,
        prices: Sequence[CapturedOptionPrice],
        *,
        final: bool,
    ) -> None:
        """Persist one durable backfill batch and update its resumable run row."""
        run_sql = (
            "INSERT INTO risk_option_price_capture_runs "
            "(run_id, captured_at, status, account_count, position_count, "
            "quote_count, missing_count, provider_class, chain_count, "
            "surface_point_count, finished_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE captured_at=VALUES(captured_at), "
            "status=VALUES(status), account_count=VALUES(account_count), "
            "position_count=VALUES(position_count), quote_count=VALUES(quote_count), "
            "missing_count=VALUES(missing_count), provider_class=VALUES(provider_class), "
            "chain_count=VALUES(chain_count), "
            "surface_point_count=VALUES(surface_point_count), "
            "finished_at=VALUES(finished_at)"
        )
        price_sql = self._price_sql(preserve_existing=True)
        try:
            with self.connection_factory.transaction(dictionary=True) as (_, cursor, _):
                cursor.execute(
                    run_sql,
                    (
                        summary.run_id,
                        _mysql_utc(summary.captured_at),
                        summary.status if final else "running",
                        summary.account_count,
                        summary.position_count,
                        summary.quote_count,
                        summary.missing_count,
                        summary.provider_class[:255],
                        summary.chain_count,
                        summary.surface_point_count,
                        _mysql_utc(summary.captured_at) if final else None,
                    ),
                )
                if prices:
                    cursor.executemany(
                        price_sql,
                        [
                            self._price_parameters(summary.run_id, item, None)
                            for item in prices
                        ],
                    )
        except Exception as exc:
            raise OptionPriceRepositoryUnavailable(
                "Risk option-price backfill write failed"
            ) from exc

    def existing_session_dates(
        self,
        account_id: str,
        canonical_occ_symbol: str,
        *,
        start_date: date,
        end_date: date,
    ) -> set[date]:
        """Return already-persisted sessions for one account and contract."""
        sql = (
            "SELECT session_date FROM risk_option_position_prices "
            "WHERE account_id = %s AND canonical_occ_symbol = %s "
            "AND session_date >= %s AND session_date <= %s"
        )
        try:
            with self.connection_factory.cursor(dictionary=True) as (_, cursor, _):
                cursor.execute(
                    sql,
                    (account_id, canonical_occ_symbol, start_date, end_date),
                )
                rows = [row_dict(cursor, item) for item in cursor.fetchall()]
        except Exception as exc:
            raise OptionPriceRepositoryUnavailable(
                "Risk option-price existing-session query failed"
            ) from exc
        return {
            session
            for row in rows
            if (session := _date_value(row.get("session_date"))) is not None
        }

    def latest_successful_capture_at(
        self,
        *,
        provider_class: str,
        minimum_account_count: int,
    ) -> datetime | None:
        """Return the scheduled-capture watermark for transaction discovery."""
        sql = (
            "SELECT captured_at FROM risk_option_price_capture_runs "
            "WHERE status = 'success' AND provider_class = %s "
            "AND account_count >= %s ORDER BY captured_at DESC LIMIT 1"
        )
        try:
            with self.connection_factory.cursor(dictionary=True) as (_, cursor, _):
                cursor.execute(sql, (provider_class, minimum_account_count))
                row = cursor.fetchone()
        except Exception as exc:
            raise OptionPriceRepositoryUnavailable(
                "Risk option-price capture watermark query failed"
            ) from exc
        if row is None:
            return None
        value = row_dict(cursor, row).get("captured_at")
        if isinstance(value, datetime):
            return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
        return None

    @staticmethod
    def _price_sql(preserve_existing: bool) -> str:
        # Backfill preserves existing captures while live capture may update them.
        duplicate_clause = (
            "ON DUPLICATE KEY UPDATE id=id"
            if preserve_existing
            else (
                "ON DUPLICATE KEY UPDATE "
                "run_id=VALUES(run_id), provider_option_symbol=VALUES(provider_option_symbol), "
                "underlying_symbol=VALUES(underlying_symbol), expiration_date=VALUES(expiration_date), "
                "put_call=VALUES(put_call), strike=VALUES(strike), "
                "signed_quantity=VALUES(signed_quantity), captured_at=VALUES(captured_at), "
                "quote_time=VALUES(quote_time), trade_time=VALUES(trade_time), bid=VALUES(bid), "
                "ask=VALUES(ask), mark=VALUES(mark), last=VALUES(last), "
                "open_price=VALUES(open_price), high_price=VALUES(high_price), "
                "low_price=VALUES(low_price), close_price=VALUES(close_price), "
                "underlying_price=VALUES(underlying_price), provider_iv=VALUES(provider_iv), "
                "provider_delta=VALUES(provider_delta), provider_gamma=VALUES(provider_gamma), "
                "provider_theta_per_day=VALUES(provider_theta_per_day), "
                "provider_vega_per_1pct=VALUES(provider_vega_per_1pct), "
                "provider_rho_per_1pct=VALUES(provider_rho_per_1pct), "
                "open_interest=VALUES(open_interest), volume=VALUES(volume), "
                "source=VALUES(source), chain_snapshot_id=VALUES(chain_snapshot_id), "
                "mark_source=VALUES(mark_source)"
            )
        )
        return (
            "INSERT INTO risk_option_position_prices "
            "(run_id, account_id, canonical_occ_symbol, provider_option_symbol, "
            "underlying_symbol, expiration_date, put_call, strike, signed_quantity, "
            "captured_at, session_date, quote_time, trade_time, bid, ask, mark, last, "
            "open_price, high_price, low_price, close_price, underlying_price, "
            "provider_iv, provider_delta, provider_gamma, provider_theta_per_day, "
            "provider_vega_per_1pct, provider_rho_per_1pct, open_interest, volume, source, "
            "chain_snapshot_id, mark_source) "
            "VALUES (" + ",".join(["%s"] * 33) + ") " + duplicate_clause
        )

    def query_history(
        self,
        symbols: Sequence[str],
        *,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        normalized = tuple(dict.fromkeys(str(item).strip().upper() for item in symbols))
        if not normalized:
            return []
        sql = (
            "SELECT id, account_id, canonical_occ_symbol, session_date, captured_at, "
            "quote_time, trade_time, bid, ask, mark, last, open_price, high_price, "
            "low_price, close_price, underlying_price, provider_iv, provider_delta, "
            "provider_gamma, provider_theta_per_day, provider_vega_per_1pct, "
            "provider_rho_per_1pct, open_interest, volume, source, mark_source "
            "FROM risk_option_position_prices WHERE canonical_occ_symbol IN ("
            + ",".join(["%s"] * len(normalized))
            + ") AND session_date >= %s AND session_date <= %s "
            "ORDER BY canonical_occ_symbol, session_date, captured_at DESC, id DESC"
        )
        try:
            with self.connection_factory.cursor(dictionary=True) as (_, cursor, _):
                cursor.execute(sql, (*normalized, start_date, end_date))
                rows = [row_dict(cursor, item) for item in cursor.fetchall()]
        except Exception as exc:
            raise OptionPriceRepositoryUnavailable(
                "Risk option-price history query failed"
            ) from exc

        # Multiple accounts can hold the same contract. The quote is identical;
        # retain the latest captured observation for each contract/session.
        result: list[dict[str, Any]] = []
        # Keep one preferred observation per contract and session across account rows.
        seen: set[tuple[str, date]] = set()
        for row in rows:
            symbol = str(row.get("canonical_occ_symbol") or "").strip().upper()
            session = _date_value(row.get("session_date"))
            if not symbol or session is None or (symbol, session) in seen:
                continue
            seen.add((symbol, session))
            result.append(self._history_item(row, symbol, session))
        return result

    def query_surface_history(
        self,
        contract_symbols: Sequence[str],
        *,
        start_at: datetime,
        end_at: datetime,
    ) -> list[OptionSurfaceSnapshot]:
        """Load bounded daily chain slices for exact OCC contracts.

        Snapshot headers are selected by decoded root/expiration, while the
        second query deliberately returns every stored point in those slices so
        consumers can fit a local smile without one query per contract.
        """
        if start_at.tzinfo is None or start_at.utcoffset() is None:
            raise ValueError("start_at must be timezone aware")
        if end_at.tzinfo is None or end_at.utcoffset() is None:
            raise ValueError("end_at must be timezone aware")
        if end_at <= start_at:
            return []
        normalized = tuple(
            dict.fromkeys(
                parse_occ_symbol(str(item).strip().upper()).canonical
                for item in contract_symbols
                if str(item).strip()
            )
        )
        if not normalized:
            return []
        if len(normalized) > self.maximum_surface_symbols:
            raise OptionSurfaceHistoryLimitExceeded(
                "surface symbol count exceeds configured maximum"
            )
        # Load complete surface snapshots for the requested underlying-expiration pairs.
        pairs = tuple(
            sorted(
                {
                    (parsed.root, parsed.expiration)
                    for symbol in normalized
                    for parsed in (parse_occ_symbol(symbol),)
                },
                key=lambda item: (item[0], item[1]),
            )
        )
        pair_predicate = " OR ".join(
            "(snap.underlying_symbol = %s AND snap.expiration_date = %s)"
            for _ in pairs
        )
        header_sql = (
            "SELECT snap.chain_snapshot_id, run.run_id, run.captured_at, "
            "snap.underlying_symbol, snap.expiration_date, snap.spot, "
            "snap.spot_source, snap.underlying_quote_time, "
            "snap.snapshot_status, snap.snapshot_flags, snap.error_code "
            "FROM risk_option_chain_snapshots AS snap "
            "JOIN risk_option_price_capture_runs AS run ON run.run_id = snap.run_id "
            f"WHERE ({pair_predicate}) AND run.captured_at >= %s "
            "AND run.captured_at < %s "
            "ORDER BY run.captured_at, snap.underlying_symbol, "
            "snap.expiration_date, snap.chain_snapshot_id "
            "LIMIT %s"
        )
        pair_parameters = tuple(value for pair in pairs for value in pair)
        try:
            with self.connection_factory.cursor(dictionary=True) as (_, cursor, _):
                cursor.execute(
                    header_sql,
                    (
                        *pair_parameters,
                        _mysql_utc(start_at),
                        _mysql_utc(end_at),
                        self.maximum_surface_snapshots + 1,
                    ),
                )
                header_rows = [row_dict(cursor, item) for item in cursor.fetchall()]
                if len(header_rows) > self.maximum_surface_snapshots:
                    raise OptionSurfaceHistoryLimitExceeded(
                        "surface snapshot count exceeds configured maximum"
                    )
                snapshot_ids = [int(row["chain_snapshot_id"]) for row in header_rows]
                point_rows: list[dict[str, Any]] = []
                # Bound surface-point reads to avoid unbounded SQL parameter batches.
                for chunk in _chunks(snapshot_ids, self.surface_point_batch_size):
                    remaining = self.maximum_surface_points - len(point_rows)
                    point_sql = (
                        "SELECT chain_snapshot_id, canonical_occ_symbol, bid, ask, "
                        "mark, last, quote_time, trade_time, observation_source, "
                        "security_status FROM risk_option_surface_points "
                        "WHERE chain_snapshot_id IN ("
                        + ",".join(["%s"] * len(chunk))
                        + ") ORDER BY chain_snapshot_id, canonical_occ_symbol LIMIT %s"
                    )
                    cursor.execute(point_sql, (*chunk, remaining + 1))
                    batch = [row_dict(cursor, item) for item in cursor.fetchall()]
                    point_rows.extend(batch)
                    if len(point_rows) > self.maximum_surface_points:
                        raise OptionSurfaceHistoryLimitExceeded(
                            "surface point count exceeds configured maximum"
                        )
        except OptionSurfaceHistoryLimitExceeded:
            raise
        except Exception as exc:
            raise OptionPriceRepositoryUnavailable(
                "Risk option-price surface history query failed"
            ) from exc

        observations: dict[int, list[OptionSurfaceObservation]] = {
            snapshot_id: [] for snapshot_id in snapshot_ids
        }
        for row in point_rows:
            snapshot_id = _integer(row.get("chain_snapshot_id"))
            if snapshot_id is None:
                continue
            observations.setdefault(snapshot_id, []).append(
                OptionSurfaceObservation(
                    snapshot_id=snapshot_id,
                    canonical_occ_symbol=str(
                        row.get("canonical_occ_symbol") or ""
                    ).strip().upper(),
                    bid=_number(row.get("bid")),
                    ask=_number(row.get("ask")),
                    mark=_number(row.get("mark")),
                    last=_number(row.get("last")),
                    quote_time=_utc_datetime(row.get("quote_time")),
                    trade_time=_utc_datetime(row.get("trade_time")),
                    observation_source=_integer(row.get("observation_source")) or 0,
                    security_status=_integer(row.get("security_status")),
                )
            )
        result: list[OptionSurfaceSnapshot] = []
        for row in header_rows:
            snapshot_id = _integer(row.get("chain_snapshot_id"))
            if snapshot_id is None:
                continue
            expiration = _date_value(row.get("expiration_date"))
            captured_at = _utc_datetime(row.get("captured_at"))
            if expiration is None or captured_at is None:
                continue
            header = OptionSurfaceSnapshotHeader(
                snapshot_id=snapshot_id,
                run_id=str(row.get("run_id") or ""),
                captured_at=captured_at,
                underlying_symbol=str(row.get("underlying_symbol") or "").strip().upper(),
                expiration_date=expiration,
                spot=_number(row.get("spot")),
                spot_source=_integer(row.get("spot_source")),
                underlying_quote_time=_utc_datetime(
                    row.get("underlying_quote_time")
                ),
                snapshot_status=_integer(row.get("snapshot_status")) or 0,
                snapshot_flags=_integer(row.get("snapshot_flags")) or 0,
                error_code=(
                    str(row["error_code"]) if row.get("error_code") else None
                ),
            )
            result.append(
                OptionSurfaceSnapshot(
                    header=header,
                    observations=tuple(
                        sorted(
                            observations.get(snapshot_id, ()),
                            key=lambda item: item.canonical_occ_symbol,
                        )
                    ),
                )
            )
        return sorted(
            result,
            key=lambda item: (
                item.header.captured_at,
                item.header.underlying_symbol,
                item.header.expiration_date,
                item.header.snapshot_id,
            ),
        )

    @staticmethod
    def _price_parameters(
        run_id: str,
        item: CapturedOptionPrice,
        chain_snapshot_id: int | None,
    ) -> tuple[Any, ...]:
        position = item.position
        return (
            run_id,
            position.account_id,
            position.canonical_occ_symbol,
            position.provider_option_symbol,
            position.underlying_symbol,
            position.expiration_date,
            position.put_call,
            position.strike,
            position.signed_quantity,
            _mysql_utc(item.captured_at),
            item.session_date,
            _mysql_utc_optional(item.quote_time),
            _mysql_utc_optional(item.trade_time),
            item.bid,
            item.ask,
            item.mark,
            item.last,
            item.open,
            item.high,
            item.low,
            item.close,
            item.underlying_price,
            item.provider_iv,
            item.provider_delta,
            item.provider_gamma,
            item.provider_theta_per_day,
            item.provider_vega_per_1pct,
            item.provider_rho_per_1pct,
            item.open_interest,
            item.volume,
            item.source[:128],
            chain_snapshot_id,
            item.mark_source,
        )

    @staticmethod
    def _chain_parameters(
        run_id: str, item: CapturedChainSnapshot
    ) -> tuple[Any, ...]:
        return (
            run_id,
            item.underlying_symbol[:16],
            item.expiration_date,
            item.spot,
            item.spot_source,
            _mysql_utc_optional(item.underlying_quote_time),
            item.snapshot_status,
            item.snapshot_flags,
            item.error_code[:32] if item.error_code else None,
        )

    @staticmethod
    def _surface_parameters(
        chain_snapshot_id: int, item: Any
    ) -> tuple[Any, ...]:
        return (
            chain_snapshot_id,
            item.canonical_occ_symbol[:21],
            _mysql_utc_optional(item.quote_time),
            _mysql_utc_optional(item.trade_time),
            item.bid,
            item.ask,
            item.mark,
            item.last,
            item.open_interest,
            item.volume,
            item.observation_source,
            item.security_status,
        )

    @staticmethod
    def _history_item(
        row: Mapping[str, Any], symbol: str, session_date: date
    ) -> dict[str, Any]:
        # Match the RefData daily-bar chronology: a session close is observable
        # at the following UTC midnight, preventing intraday look-ahead.
        available_at = datetime.combine(
            session_date,
            time.min,
            tzinfo=timezone.utc,
        ) + timedelta(days=1)
        observed_at = (
            _utc_datetime(row.get("quote_time"))
            or _utc_datetime(row.get("trade_time"))
            or _utc_datetime(row.get("captured_at"))
        )
        return {
            "requested_symbol": symbol,
            "canonical_occ_symbol": symbol,
            "at": available_at,
            "session_date": session_date,
            "observed_at": observed_at,
            "status": "ok",
            "bid": _number(row.get("bid")),
            "ask": _number(row.get("ask")),
            "mark": _number(row.get("mark")),
            "last": _number(row.get("last")),
            "open": _number(row.get("open_price")),
            "high": _number(row.get("high_price")),
            "low": _number(row.get("low_price")),
            "close": _number(row.get("close_price")),
            "volume": _number(row.get("volume")),
            "underlying_price": _number(row.get("underlying_price")),
            "provider_iv": _number(row.get("provider_iv")),
            "provider_delta": _number(row.get("provider_delta")),
            "provider_gamma": _number(row.get("provider_gamma")),
            "provider_theta_per_day": _number(row.get("provider_theta_per_day")),
            "provider_vega_per_1pct": _number(row.get("provider_vega_per_1pct")),
            "provider_rho_per_1pct": _number(row.get("provider_rho_per_1pct")),
            "mark_basis": _mark_basis(row.get("mark_source")),
            "source": "risk_option_price_capture",
            "warnings": [],
            "error": None,
        }


def _mysql_utc(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("timestamp must be timezone aware")
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _mysql_utc_optional(value: datetime | None) -> datetime | None:
    return _mysql_utc(value) if value is not None else None


def _date_value(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted


def _utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _mark_basis(value: Any) -> str | None:
    try:
        source = MarkSource(int(value))
    except (TypeError, ValueError):
        return None
    return {
        MarkSource.PROVIDER_MARK: "provider_mark",
        MarkSource.BID_ASK_MIDPOINT: "bid_ask_midpoint",
        MarkSource.LAST_TRADE: "last_trade",
        MarkSource.CLOSE: "close",
    }[source]


def _chunks(values: Sequence[int], size: int) -> list[tuple[int, ...]]:
    bounded = max(1, int(size))
    return [tuple(values[index : index + bounded]) for index in range(0, len(values), bounded)]


__all__ = [
    "OptionPriceRepository",
    "OptionPriceRepositoryUnavailable",
    "OptionSurfaceHistoryLimitExceeded",
]
