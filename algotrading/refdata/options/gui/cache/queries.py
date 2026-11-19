"""Bounded, set-based ClickHouse extraction for cache generation."""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa

from .config import CacheConfig
from .errors import ValidationError
from .model import SURFACE_OBSERVATION_SCHEMA
from .sampling import SampledSnapshot, SurfaceStateSnapshot


UTC = timezone.utc


def _database_identifier(value: str) -> str:
    from algotrading.refdata.options.Common import validate_database_identifier

    return validate_database_identifier(value)


def _utc(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass
class ExtractionMetrics:
    queries: int = 0
    state_rows: int = 0
    sampled_snapshots: int = 0
    surface_rows: int = 0
    approximate_bytes: int = 0

    def as_dict(self) -> dict[str, int | float]:
        rows_per_snapshot = (
            self.surface_rows / self.sampled_snapshots
            if self.sampled_snapshots
            else 0.0
        )
        return {
            "queries": self.queries,
            "state_rows": self.state_rows,
            "sampled_snapshots": self.sampled_snapshots,
            "surface_rows": self.surface_rows,
            "surface_rows_per_sampled_snapshot": rows_per_snapshot,
            "approximate_bytes": self.approximate_bytes,
        }


@dataclass(frozen=True)
class SymbolSourceWatermark:
    symbol: str
    latest_chain_at: datetime | None
    latest_surface_at: datetime | None

    def surface_is_current(self, *, tolerance_seconds: int) -> bool:
        if self.latest_chain_at is None or self.latest_surface_at is None:
            return False
        return (
            abs(
                (self.latest_chain_at - self.latest_surface_at).total_seconds()
            )
            <= tolerance_seconds
        )

    def as_dict(self, *, tolerance_seconds: int) -> dict[str, Any]:
        return {
            "latest_chain_at": (
                self.latest_chain_at.isoformat()
                if self.latest_chain_at is not None
                else None
            ),
            "latest_surface_at": (
                self.latest_surface_at.isoformat()
                if self.latest_surface_at is not None
                else None
            ),
            "surface_is_current": self.surface_is_current(
                tolerance_seconds=tolerance_seconds
            ),
        }


class ClickHouseExtractor:
    """Read canonical state once, then join bounded facts to sampled metadata."""

    def __init__(
        self,
        client: Any,
        context: Mapping[str, Any],
        config: CacheConfig,
        *,
        logger: logging.Logger | None = None,
        build_id: str | None = None,
    ) -> None:
        self.client = client
        self.context = context
        self.config = config
        self.database = _database_identifier(
            str(context["options_chain.clickhouse.database"])
        )
        self.logger = logger or logging.getLogger(__name__)
        self.build_id = build_id or uuid.uuid4().hex
        self.metrics = ExtractionMetrics()

    @property
    def terminal_statuses(self) -> tuple[str, ...]:
        if not self.config.require_surface_completion:
            return (
                "complete",
                "complete_with_failures",
                "failed",
                "stale_running",
                "abandoned",
            )
        if self.config.allow_complete_with_failures:
            return ("complete", "complete_with_failures")
        return ("complete",)

    def latest_completed_timestamp(
        self,
        symbols: Sequence[str],
        *,
        end_utc: datetime | None = None,
    ) -> datetime | None:
        """Return only the latest completed source time; never scan for a minimum."""

        upper = ""
        parameters: dict[str, Any] = {
            "symbols": list(symbols),
            "statuses": list(self.terminal_statuses),
        }
        if end_utc is not None:
            upper = (
                "AND source_collected_at <= "
                "{end_utc:DateTime64(6, 'UTC')}"
            )
            parameters["end_utc"] = end_utc
        sql = f"""
SELECT max(source_collected_at) AS latest_source_collected_at
FROM {self.database}.options_chain_surface_state FINAL
PREWHERE underlying_symbol IN {{symbols:Array(String)}}
    {upper}
WHERE is_active
  AND toString(status) IN {{statuses:Array(String)}}
"""
        table = self._query_arrow(sql, parameters, phase="latest_source")
        if table.num_rows != 1 or table.num_columns != 1:
            return None
        value = table.column(0)[0].as_py()
        return _utc(value) if value is not None else None

    def source_watermarks(
        self,
        symbols: Sequence[str],
        *,
        end_utc: datetime | None = None,
    ) -> dict[str, SymbolSourceWatermark]:
        """Return the last raw chain and terminal surface time per symbol.

        A symbol whose two timestamps remain aligned may have intentionally
        stopped at that point.  A newer chain timestamp without a matching
        terminal surface remains a genuine surface-processing backlog.
        """

        upper = ""
        parameters: dict[str, Any] = {
            "symbols": list(symbols),
            "statuses": list(self.terminal_statuses),
        }
        if end_utc is not None:
            upper = "AND source_collected_at <= {end_utc:DateTime64(6, 'UTC')}"
            parameters["end_utc"] = end_utc
        sql = f"""
SELECT
    requested.underlying_symbol AS underlying_symbol,
    chains.latest_chain_at AS latest_chain_at,
    surfaces.latest_surface_at AS latest_surface_at
FROM
(
    SELECT arrayJoin({{symbols:Array(String)}}) AS underlying_symbol
) AS requested
LEFT JOIN
(
    SELECT underlying_symbol, max(source_collected_at) AS latest_chain_at
    FROM {self.database}.options_chain_snapshots
    PREWHERE underlying_symbol IN {{symbols:Array(String)}}
      {upper}
    GROUP BY underlying_symbol
) AS chains USING (underlying_symbol)
LEFT JOIN
(
    SELECT underlying_symbol, max(source_collected_at) AS latest_surface_at
    FROM {self.database}.options_chain_surface_state FINAL
    PREWHERE underlying_symbol IN {{symbols:Array(String)}}
      {upper}
    WHERE is_active
      AND toString(status) IN {{statuses:Array(String)}}
    GROUP BY underlying_symbol
) AS surfaces USING (underlying_symbol)
ORDER BY underlying_symbol
"""
        table = self._query_arrow(
            sql,
            parameters,
            phase="source_watermarks",
        )
        rows = {
            str(row["underlying_symbol"]): SymbolSourceWatermark(
                symbol=str(row["underlying_symbol"]),
                latest_chain_at=(
                    _utc(row["latest_chain_at"])
                    if row["latest_chain_at"] is not None
                    else None
                ),
                latest_surface_at=(
                    _utc(row["latest_surface_at"])
                    if row["latest_surface_at"] is not None
                    else None
                ),
            )
            for row in table.to_pylist()
        }
        return {
            symbol: rows.get(
                symbol,
                SymbolSourceWatermark(symbol, None, None),
            )
            for symbol in symbols
        }

    def load_surface_states(
        self,
        symbols: Sequence[str],
        start_utc: datetime,
        end_utc: datetime,
    ) -> tuple[SurfaceStateSnapshot, ...]:
        sql = f"""
SELECT
    toString(snapshot_id) AS snapshot_id,
    underlying_symbol,
    source_collected_at,
    surface_definition_id,
    source_calculation_definition_id AS calculation_definition_id,
    ifNull(toString(source_surface_run_id), '') AS source_surface_run_id,
    toString(status) AS status,
    successful_expirations
FROM {self.database}.options_chain_surface_state FINAL
PREWHERE underlying_symbol IN {{symbols:Array(String)}}
    AND source_collected_at >= {{start_utc:DateTime64(6, 'UTC')}}
    AND source_collected_at <= {{end_utc:DateTime64(6, 'UTC')}}
WHERE is_active
    AND toString(status) IN {{statuses:Array(String)}}
"""
        table = self._query_arrow(
            sql,
            {
                "symbols": list(symbols),
                "start_utc": start_utc,
                "end_utc": end_utc,
                "statuses": list(self.terminal_statuses),
            },
            phase="surface_states",
        )
        self.metrics.state_rows += table.num_rows
        if not table.num_rows:
            return ()
        columns = {name: table[name].to_pylist() for name in table.column_names}
        return tuple(
            SurfaceStateSnapshot(
                snapshot_id=str(columns["snapshot_id"][index]),
                underlying_symbol=str(columns["underlying_symbol"][index]),
                source_collected_at=_utc(columns["source_collected_at"][index]),
                surface_definition_id=int(columns["surface_definition_id"][index]),
                calculation_definition_id=int(
                    columns["calculation_definition_id"][index]
                ),
                source_surface_run_id=(
                    str(columns["source_surface_run_id"][index]) or None
                ),
                status=str(columns["status"][index]).lower(),
                successful_expirations=int(
                    columns["successful_expirations"][index]
                ),
            )
            for index in range(table.num_rows)
        )

    def iter_surface_tables(
        self,
        sampled: Sequence[SampledSnapshot],
    ) -> Iterator[pa.Table]:
        if not sampled:
            return
        symbols = sorted({item.state.underlying_symbol for item in sampled})
        start_utc = min(item.state.source_collected_at for item in sampled)
        end_utc = max(item.state.source_collected_at for item in sampled)
        external_data = self._sampled_external_data(sampled)
        sql = f"""
SELECT
    f.underlying_symbol AS underlying_symbol,
    st.session_date AS session_date,
    st.session_open_utc AS session_open_utc,
    st.session_close_utc AS session_close_utc,
    st.session_minute AS session_minute,
    st.sample_target_utc AS sample_target_utc,
    toString(f.snapshot_id) AS snapshot_id,
    f.source_collected_at AS source_collected_at,
    st.sample_age_seconds AS sample_age_seconds,
    f.expiration_at AS expiration_at,
    f.price_basis AS price_basis,
    f.surface_definition_id AS surface_definition_id,
    st.calculation_definition_id AS calculation_definition_id,
    f.pricing_method AS pricing_method,
    toString(f.fit_method) AS fit_method,
    f.svi_a AS svi_a,
    f.svi_b AS svi_b,
    f.svi_rho AS svi_rho,
    f.svi_m AS svi_m,
    f.svi_sigma AS svi_sigma,
    f.pchip_k AS pchip_k,
    f.pchip_total_variance AS pchip_total_variance,
    coalesce(
        nullIf(s.underlying_mark, 0),
        if(
            ifNull(s.underlying_bid, 0) > 0 AND ifNull(s.underlying_ask, 0) > 0,
            (s.underlying_bid + s.underlying_ask) / 2,
            NULL
        ),
        nullIf(s.underlying_last, 0),
        nullIf(s.chain_underlying_price, 0)
    ) AS spot_price,
    i.forward_price AS forward_price,
    i.time_to_expiry_seconds AS time_to_expiry_seconds,
    i.risk_free_rate AS risk_free_rate,
    i.dividend_yield AS dividend_yield,
    f.minimum_log_forward_moneyness AS minimum_log_forward_moneyness,
    f.maximum_log_forward_moneyness AS maximum_log_forward_moneyness,
    f.input_contract_count AS input_contract_count,
    f.accepted_contract_count AS accepted_contract_count,
    f.rejected_contract_count AS rejected_contract_count,
    f.weighted_rmse AS weighted_rmse,
    f.maximum_absolute_error AS maximum_absolute_error,
    f.butterfly_arbitrage_flags AS butterfly_arbitrage_flags,
    f.calendar_arbitrage_flags AS calendar_arbitrage_flags,
    f.quality_flags AS quality_flags,
    f.source_expiration_count AS source_expiration_count,
    countDistinct(f.expiration_at) OVER
    (
        PARTITION BY
            f.underlying_symbol,
            st.session_date,
            st.session_minute,
            f.snapshot_id,
            f.source_collected_at,
            f.surface_definition_id,
            f.source_surface_run_id
    ) AS extracted_expiration_count,
    st.successful_expirations AS state_successful_expirations
FROM
(
    SELECT
        snapshot_id, source_collected_at, underlying_symbol, expiration_at,
        price_basis, surface_definition_id, source_surface_run_id,
        pricing_method, fit_method, svi_a, svi_b, svi_rho, svi_m, svi_sigma,
        pchip_k, pchip_total_variance, minimum_log_forward_moneyness,
        maximum_log_forward_moneyness, input_contract_count,
        accepted_contract_count, rejected_contract_count, weighted_rmse,
        maximum_absolute_error, butterfly_arbitrage_flags,
        calendar_arbitrage_flags, quality_flags,
        countDistinct(expiration_at) OVER
        (
            PARTITION BY
                underlying_symbol,
                snapshot_id,
                source_collected_at,
                surface_definition_id,
                source_surface_run_id
        ) AS source_expiration_count
    FROM {self.database}.options_chain_surface_fits
    PREWHERE underlying_symbol IN {{symbols:Array(String)}}
      AND source_collected_at >= {{start_utc:DateTime64(6, 'UTC')}}
      AND source_collected_at <= {{end_utc:DateTime64(6, 'UTC')}}
    WHERE price_basis IN {{price_bases:Array(String)}}
) AS f
ALL INNER JOIN sampled_snapshots AS st
    ON st.underlying_symbol = f.underlying_symbol
   AND st.snapshot_id = f.snapshot_id
   AND st.source_collected_at = f.source_collected_at
   AND st.surface_definition_id = f.surface_definition_id
   AND isNotDistinctFrom(st.source_surface_run_id, f.source_surface_run_id)
ALL INNER JOIN
(
    SELECT
        snapshot_id, source_collected_at, underlying_symbol, expiration_at,
        calculation_definition_id, forward_price, time_to_expiry_seconds,
        risk_free_rate, dividend_yield
    FROM {self.database}.options_chain_expiration_inputs
    PREWHERE underlying_symbol IN {{symbols:Array(String)}}
      AND source_collected_at >= {{start_utc:DateTime64(6, 'UTC')}}
      AND source_collected_at <= {{end_utc:DateTime64(6, 'UTC')}}
) AS i
    ON i.underlying_symbol = f.underlying_symbol
   AND i.snapshot_id = f.snapshot_id
   AND i.source_collected_at = f.source_collected_at
   AND i.expiration_at = f.expiration_at
   AND i.calculation_definition_id = st.calculation_definition_id
ALL INNER JOIN
(
    SELECT
        snapshot_id, source_collected_at, underlying_symbol, underlying_mark,
        underlying_bid, underlying_ask, underlying_last, chain_underlying_price
    FROM {self.database}.options_chain_snapshots
    PREWHERE underlying_symbol IN {{symbols:Array(String)}}
      AND source_collected_at >= {{start_utc:DateTime64(6, 'UTC')}}
      AND source_collected_at <= {{end_utc:DateTime64(6, 'UTC')}}
) AS s
    ON s.underlying_symbol = f.underlying_symbol
   AND s.snapshot_id = f.snapshot_id
   AND s.source_collected_at = f.source_collected_at
"""
        parameters = {
            "symbols": symbols,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "price_bases": list(self.config.price_bases),
        }
        self.metrics.sampled_snapshots += len(sampled)
        local_surface_rows = 0
        for table in self._iter_query_arrow(
            sql,
            parameters,
            phase="surface_facts",
            external_data=external_data,
        ):
            if not table.num_rows:
                continue
            self._validate_expiration_cardinality(table)
            try:
                candidate = table.select(SURFACE_OBSERVATION_SCHEMA.names).cast(
                    SURFACE_OBSERVATION_SCHEMA,
                    safe=True,
                )
            except (KeyError, pa.ArrowInvalid, pa.ArrowTypeError) as exc:
                raise ValueError(f"Surface fact Arrow schema is invalid: {exc}") from exc
            self.metrics.surface_rows += candidate.num_rows
            local_surface_rows += candidate.num_rows
            yield candidate
        rows_per_snapshot = (
            local_surface_rows / len(sampled) if sampled else 0.0
        )
        self.logger.info(
            "event=cache_extract_cardinality sampled_snapshots=%s "
            "surface_rows=%s surface_rows_per_sampled_snapshot=%.3f",
            len(sampled),
            local_surface_rows,
            rows_per_snapshot,
        )

    def _validate_expiration_cardinality(self, table: pa.Table) -> None:
        """Fail before publication if a JOIN discarded fitted expirations."""

        required = (
            "source_expiration_count",
            "extracted_expiration_count",
            "state_successful_expirations",
        )
        missing = [name for name in required if name not in table.column_names]
        if missing:
            raise ValidationError(
                f"Surface extraction is missing cardinality columns: {missing}"
            )

        values = {
            name: np.asarray(
                table[name].combine_chunks().to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            for name in required
        }
        source = values["source_expiration_count"]
        extracted = values["extracted_expiration_count"]
        state = values["state_successful_expirations"]
        join_mismatch = source != extracted
        state_mismatch = (
            (state > 0)
            & (source != state)
            & ("mid" in self.config.price_bases)
        )
        invalid = join_mismatch | state_mismatch
        if not np.any(invalid):
            return

        examples: list[dict[str, Any]] = []
        indices = np.flatnonzero(invalid)[:5]
        for index in indices:
            examples.append(
                {
                    "underlying_symbol": table["underlying_symbol"][
                        int(index)
                    ].as_py(),
                    "snapshot_id": table["snapshot_id"][int(index)].as_py(),
                    "source_collected_at": str(
                        table["source_collected_at"][int(index)].as_py()
                    ),
                    "session_date": str(
                        table["session_date"][int(index)].as_py()
                    ),
                    "session_minute": int(
                        table["session_minute"][int(index)].as_py()
                    ),
                    "source_expirations": int(source[int(index)]),
                    "extracted_expirations": int(extracted[int(index)]),
                    "state_successful_expirations": int(state[int(index)]),
                }
            )
        raise ValidationError(
            "Surface extraction discarded or omitted fitted expirations; "
            f"join_mismatch_rows={int(np.count_nonzero(join_mismatch))}, "
            f"state_mismatch_rows={int(np.count_nonzero(state_mismatch))}, "
            f"examples={examples}"
        )

    @staticmethod
    def _sampled_external_data(sampled: Sequence[SampledSnapshot]) -> Any:
        from clickhouse_connect.driver.external import ExternalData

        def payload(item: SampledSnapshot) -> dict[str, Any]:
            return {
                "snapshot_id": item.state.snapshot_id,
                "underlying_symbol": item.state.underlying_symbol,
                "source_collected_at": item.state.source_collected_at.isoformat(),
                "surface_definition_id": item.state.surface_definition_id,
                "calculation_definition_id": item.state.calculation_definition_id,
                "source_surface_run_id": item.state.source_surface_run_id,
                "successful_expirations": item.state.successful_expirations,
                "session_date": item.session.session_date.isoformat(),
                "session_open_utc": item.session.open_utc.isoformat(),
                "session_close_utc": item.session.close_utc.isoformat(),
                "session_minute": item.session_minute,
                "sample_target_utc": item.target_utc.isoformat(),
                "sample_age_seconds": item.age_seconds,
            }

        data = b"".join(
            (
                json.dumps(payload(item), separators=(",", ":")) + "\n"
            ).encode("utf-8")
            for item in sampled
        )
        return ExternalData(
            data=data,
            file_name="sampled_snapshots.jsonl",
            fmt="JSONEachRow",
            structure=(
                "snapshot_id UUID, underlying_symbol String, "
                "source_collected_at DateTime64(6, 'UTC'), "
                "surface_definition_id UInt32, calculation_definition_id UInt32, "
                "source_surface_run_id Nullable(UUID), successful_expirations UInt32, "
                "session_date Date, "
                "session_open_utc DateTime64(6, 'UTC'), "
                "session_close_utc DateTime64(6, 'UTC'), session_minute UInt16, "
                "sample_target_utc DateTime64(6, 'UTC'), sample_age_seconds Float32"
            ),
        )

    def _query_arrow(
        self,
        sql: str,
        parameters: Mapping[str, Any],
        *,
        phase: str,
        external_data: Any | None = None,
    ) -> pa.Table:
        tables = list(
            self._iter_query_arrow(
                sql,
                parameters,
                phase=phase,
                external_data=external_data,
            )
        )
        if not tables:
            return pa.table({})
        return pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    def _iter_query_arrow(
        self,
        sql: str,
        parameters: Mapping[str, Any],
        *,
        phase: str,
        external_data: Any | None = None,
    ) -> Iterator[pa.Table]:
        query_id = f"time_of_day_cache:{self.build_id}:{phase}:{self.metrics.queries}"
        self.logger.info(
            "event=cache_extract_query phase=%s symbols=%s start=%s end=%s query_id=%s",
            phase,
            len(parameters.get("symbols", ())),
            parameters.get("start_utc"),
            parameters.get("end_utc"),
            query_id,
        )
        kwargs: dict[str, Any] = {"parameters": dict(parameters)}
        if external_data is not None:
            kwargs["external_data"] = external_data
        stream_query = getattr(self.client, "query_arrow_stream", None)
        if callable(stream_query):
            result = self._call_arrow(stream_query, sql, query_id, kwargs)
            manager = result if hasattr(result, "__enter__") else contextlib.nullcontext(result)
            with manager as stream:
                yield from self._record_stream(stream, phase)
        else:
            query = self.client.query_arrow
            result = self._call_arrow(query, sql, query_id, kwargs)
            if hasattr(result, "read_all"):
                result = result.read_all()
            yield from self._record_stream((result,), phase)
        self.metrics.queries += 1

    @staticmethod
    def _call_arrow(
        method: Any,
        sql: str,
        query_id: str,
        kwargs: Mapping[str, Any],
    ) -> Any:
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}
        call_kwargs = dict(kwargs)
        if "query_id" in parameters:
            # Unit-test fakes and older internal wrappers expose query_id
            # directly.
            call_kwargs["query_id"] = query_id
        elif "transport_settings" in parameters:
            # clickhouse-connect carries the server-visible query identifier
            # as the standard HTTP header.
            call_kwargs["transport_settings"] = {
                "X-ClickHouse-Query-Id": query_id
            }
        return method(sql, **call_kwargs)

    def _record_stream(self, stream: Iterable[Any], phase: str) -> Iterator[pa.Table]:
        rows = 0
        arrow_bytes = 0
        for block in stream:
            if isinstance(block, pa.RecordBatch):
                table = pa.Table.from_batches([block])
            elif isinstance(block, pa.Table):
                table = block
            elif hasattr(block, "arrow"):
                table = block.arrow()
            else:
                table = pa.Table.from_arrays(
                    [pa.array(column) for column in zip(*block.result_rows)],
                    names=block.column_names,
                )
            rows += table.num_rows
            arrow_bytes += int(table.nbytes)
            self.metrics.approximate_bytes += int(table.nbytes)
            yield table
        self.logger.info(
            "event=cache_extract_query phase=%s rows=%s arrow_bytes=%s status=completed",
            phase,
            rows,
            arrow_bytes,
        )
