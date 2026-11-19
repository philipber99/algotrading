"""Bounded, state-authorized ClickHouse repository for one screen request."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pyarrow as pa

from algotrading.refdata.options import Queries
from algotrading.refdata.options.Common import validate_database_identifier

from ..errors import (
    SnapshotNotFoundError,
    SnapshotNotTerminalError,
    SurfaceNotAvailableError,
)
from .models import (
    ContractRecord,
    ExpirationInput,
    SnapshotRecord,
    SurfaceFit,
    Universe,
    finite,
)


UTC = timezone.utc


def rows(result: Any) -> list[dict[str, Any]]:
    """Normalize Arrow and clickhouse-connect results without positional leaks."""

    if result is None:
        return []
    if isinstance(result, pa.Table):
        return result.to_pylist()
    if hasattr(result, "to_pylist"):
        return list(result.to_pylist())
    if hasattr(result, "named_results"):
        return [dict(item) for item in result.named_results()]
    if isinstance(result, Mapping):
        return [dict(result)]
    if isinstance(result, (list, tuple)):
        if not result:
            return []
        if isinstance(result[0], Mapping):
            return [dict(item) for item in result]
    names = list(getattr(result, "column_names", ()) or ())
    raw_rows = list(getattr(result, "result_rows", ()) or ())
    if names:
        return [dict(zip(names, item)) for item in raw_rows]
    raise TypeError(f"Unsupported ClickHouse result type: {type(result).__name__}")


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _integer(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


class ScreenerRepository:
    """Load a complete snapshot universe with no per-contract queries."""

    def __init__(self, client: Any, database: str) -> None:
        self.client = client
        self.database = validate_database_identifier(database)
        self.last_query_count = 0

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def load_universe(
        self,
        underlying_symbol: str,
        snapshot_id: str | None,
        *,
        price_basis: str,
        require_surface: bool,
        resolved_snapshot: SnapshotRecord | None = None,
    ) -> Universe:
        self.last_query_count = 0
        snapshot = resolved_snapshot or self.resolve_snapshot(
            underlying_symbol,
            snapshot_id,
            require_surface=require_surface,
        )

        expiration_rows = rows(
            Queries.load_expiration_inputs(
                self.client,
                self.database,
                snapshot.snapshot_id,
                arrow=False,
            )
        )
        self.last_query_count += 1
        expiration_inputs = {
            item.expiration_at: item
            for item in (self._expiration(row) for row in expiration_rows)
        }
        chain_rows = rows(
            Queries.load_effective_chain(
                self.client,
                self.database,
                snapshot.snapshot_id,
                arrow=False,
            )
        )
        self.last_query_count += 1
        failure_rows = rows(
            Queries.load_calculation_failures(
                self.client,
                self.database,
                snapshot_id=snapshot.snapshot_id,
                arrow=False,
            )
        )
        self.last_query_count += 1
        contracts = self._apply_failures(
            tuple(self._contract(row, snapshot) for row in chain_rows),
            failure_rows,
        )

        surface_fits: dict[tuple[datetime, str], SurfaceFit] = {}
        atm_volatility = self._fallback_atm_volatility(contracts)
        forward_variance: tuple[dict[str, Any], ...] = ()
        if require_surface:
            state_rows = rows(
                Queries.load_surface_state(
                    self.client,
                    self.database,
                    snapshot_ids=[snapshot.snapshot_id],
                    arrow=False,
                )
            )
            self.last_query_count += 1
            terminal = [
                row for row in state_rows
                if str(row.get("status", "")).lower()
                in {"complete", "complete_with_failures"}
            ]
            if not terminal:
                raise SurfaceNotAvailableError(
                    "The selected snapshot has no active terminal surface state",
                    details={"snapshot_id": snapshot.snapshot_id},
                )
            fit_rows = rows(
                Queries.load_surface_fits(
                    self.client,
                    self.database,
                    snapshot.snapshot_id,
                    price_basis,
                    arrow=False,
                )
            )
            self.last_query_count += 1
            atm_rows = rows(
                Queries.load_atm_term_structure(
                    self.client,
                    self.database,
                    snapshot.snapshot_id,
                    price_basis=price_basis,
                    arrow=False,
                )
            )
            self.last_query_count += 1
            for row in fit_rows:
                fit = self._surface_fit(row)
                if fit.expiration_at in expiration_inputs:
                    surface_fits[(fit.expiration_at, fit.price_basis)] = fit
            atm_candidates = [
                finite(row.get("fitted_iv_mid"))
                for row in atm_rows
            ]
            atm_candidates = [value for value in atm_candidates if value and value > 0]
            if atm_candidates:
                atm_volatility = float(atm_candidates[0])
            if not surface_fits or not math_is_positive(atm_volatility):
                raise SurfaceNotAvailableError(
                    "The selected snapshot has no usable fitted surface",
                    details={"snapshot_id": snapshot.snapshot_id, "price_basis": price_basis},
                )

        return Universe(
            snapshot=snapshot,
            contracts=contracts,
            expiration_inputs=expiration_inputs,
            surface_fits=surface_fits,
            atm_volatility=atm_volatility,
            forward_variance=forward_variance,
        )

    @staticmethod
    def _apply_failures(
        contracts: tuple[ContractRecord, ...],
        failure_rows: Iterable[Mapping[str, Any]],
    ) -> tuple[ContractRecord, ...]:
        snapshot_reason: str | None = None
        expiration_reasons: dict[datetime, str] = {}
        contract_reasons: dict[int, str] = {}
        for row in failure_rows:
            scope = str(row.get("failure_scope") or "").lower()
            reason = _optional_text(row.get("reason_code"))
            if reason is None:
                continue
            if scope == "contract":
                contract_id = _integer(row.get("contract_id"))
                if contract_id is not None:
                    contract_reasons[contract_id] = reason
            elif scope == "expiration" and row.get("expiration_at") is not None:
                expiration_reasons[_datetime(row["expiration_at"])] = reason
            elif scope == "snapshot":
                snapshot_reason = reason
        return tuple(
            replace(
                contract,
                active_failure_reason=(
                    contract_reasons.get(contract.contract_id)
                    or expiration_reasons.get(contract.expiration_at)
                    or snapshot_reason
                ),
            )
            for contract in contracts
        )

    def resolve_snapshot(
        self,
        underlying_symbol: str,
        snapshot_id: str | None,
        *,
        require_surface: bool = False,
    ) -> SnapshotRecord:
        snapshot_rows = rows(
            Queries.load_latest_completed_snapshot(
                self.client,
                self.database,
                underlying_symbol,
                snapshot_id,
                require_surface=require_surface,
                arrow=False,
            )
        )
        if not snapshot_rows:
            if snapshot_id is not None and self._snapshot_exists(snapshot_id):
                raise SnapshotNotTerminalError(
                    "The requested snapshot is not the active terminal calculation attempt",
                    details={"snapshot_id": snapshot_id},
                )
            raise SnapshotNotFoundError(
                "No active completed snapshot was found for the underlying symbol",
                details={"underlying_symbol": underlying_symbol, "snapshot_id": snapshot_id},
            )
        return self._snapshot(snapshot_rows[0])

    def _snapshot_exists(self, snapshot_id: str) -> bool:
        result = self.client.query(
            f"SELECT count() FROM {self.database}.options_chain_snapshots "
            "WHERE snapshot_id = {snapshot_id:UUID} LIMIT 1",
            parameters={"snapshot_id": snapshot_id},
        )
        raw = list(getattr(result, "result_rows", ()) or ())
        return bool(raw and int(raw[0][0]))

    @staticmethod
    def _snapshot(row: Mapping[str, Any]) -> SnapshotRecord:
        bid = finite(row.get("underlying_bid"))
        ask = finite(row.get("underlying_ask"))
        mark = finite(row.get("underlying_mark"))
        if mark is None and bid is not None and ask is not None:
            mark = (bid + ask) / 2.0
        if mark is None:
            mark = finite(row.get("chain_underlying_price")) or finite(
                row.get("underlying_last")
            )
        if mark is None or mark <= 0:
            raise SnapshotNotTerminalError(
                "The selected snapshot has no usable underlying mark"
            )
        return SnapshotRecord(
            snapshot_id=str(row["snapshot_id"]),
            source_collected_at=_datetime(row["source_collected_at"]),
            underlying_symbol=str(row["underlying_symbol"]).upper(),
            underlying_bid=bid,
            underlying_ask=ask,
            underlying_mark=mark,
            calculation_definition_id=int(row["calculation_definition_id"]),
            calculation_run_id=_optional_text(row.get("source_calculation_run_id")),
            calculation_attempt_id=_optional_text(row.get("active_attempt_id")),
        )

    @staticmethod
    def _expiration(row: Mapping[str, Any]) -> ExpirationInput:
        return ExpirationInput(
            expiration_at=_datetime(row["expiration_at"]),
            time_to_expiry_seconds=int(row["time_to_expiry_seconds"]),
            risk_free_rate=float(row["risk_free_rate"]),
            dividend_yield=float(row["dividend_yield"]),
            forward_price=float(row["forward_price"]),
            calculation_definition_id=int(row["calculation_definition_id"]),
            quality_flags=int(row.get("quality_flags") or 0),
        )

    @staticmethod
    def _contract(row: Mapping[str, Any], snapshot: SnapshotRecord) -> ContractRecord:
        expiration = _datetime(row["expiration_at"])
        quote = row.get("quote_time")
        return ContractRecord(
            contract_id=int(row["contract_id"]),
            option_symbol=str(row["option_symbol"]),
            put_call=str(row["put_call"]).upper(),
            strike=float(row["strike_price"]),
            expiration_at=expiration,
            dte=max(0.0, (expiration - snapshot.source_collected_at).total_seconds() / 86400.0),
            exercise_type=str(row.get("exercise_type") or "AMERICAN").upper(),
            multiplier=float(row.get("multiplier") or 0),
            is_mini=bool(row.get("is_mini")),
            is_nonstandard=bool(row.get("is_nonstandard")),
            symbol_map_mismatch=bool(row.get("symbol_map_mismatch")),
            bid=finite(row.get("bid")),
            ask=finite(row.get("ask")),
            bid_size=_integer(row.get("bid_size")),
            ask_size=_integer(row.get("ask_size")),
            volume=_integer(row.get("total_volume")),
            open_interest=_integer(row.get("open_interest")),
            quote_time=_datetime(quote) if quote is not None else None,
            effective_iv=finite(row.get("effective_iv")),
            delta=finite(row.get("effective_delta")),
            gamma=finite(row.get("effective_gamma")),
            theta=finite(row.get("effective_theta")),
            vega=finite(row.get("effective_vega")),
            rho=finite(row.get("effective_rho")),
            iv_value_source=str(row.get("iv_selection") or "source"),
            greeks_value_source=str(row.get("greeks_selection") or "source"),
            result_origin=str(row.get("result_origin") or "source_only"),
            quality_flags=int(row.get("result_quality_flags") or 0),
            active_failure_reason=_optional_text(row.get("active_failure_reason")),
        )

    @staticmethod
    def _surface_fit(row: Mapping[str, Any]) -> SurfaceFit:
        return SurfaceFit(
            expiration_at=_datetime(row["expiration_at"]),
            price_basis=str(row["price_basis"]).lower(),
            pricing_method=str(row.get("pricing_method") or "none"),
            fit_method=str(row["fit_method"]),
            svi_a=finite(row.get("svi_a")),
            svi_b=finite(row.get("svi_b")),
            svi_rho=finite(row.get("svi_rho")),
            svi_m=finite(row.get("svi_m")),
            svi_sigma=finite(row.get("svi_sigma")),
            pchip_k=tuple(float(value) for value in (row.get("pchip_k") or ())),
            pchip_total_variance=tuple(
                float(value) for value in (row.get("pchip_total_variance") or ())
            ),
            minimum_log_forward_moneyness=float(row["minimum_log_forward_moneyness"]),
            maximum_log_forward_moneyness=float(row["maximum_log_forward_moneyness"]),
            weighted_rmse=float(row.get("weighted_rmse") or 0),
            quality_flags=int(row.get("quality_flags") or 0),
            calculation_definition_id=int(
                row.get("source_calculation_definition_id") or 0
            ),
            surface_definition_id=int(row["surface_definition_id"]),
        )

    @staticmethod
    def _fallback_atm_volatility(contracts: Iterable[ContractRecord]) -> float:
        values = sorted(
            contract.effective_iv
            for contract in contracts
            if contract.effective_iv is not None and contract.effective_iv > 0
        )
        if not values:
            return float("nan")
        return float(values[len(values) // 2])


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text in {"", "None", "00000000-0000-0000-0000-000000000000"} else text


def math_is_positive(value: float) -> bool:
    try:
        return value > 0 and value < float("inf")
    except TypeError:
        return False


__all__ = ["ScreenerRepository", "rows"]
