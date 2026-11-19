"""Explicit, resumable five-minute compaction of closed option-chain months.

A normalized_month manifest records a destructive in-place compaction, not an
archive file. Its empty archive_path and null archive_file_hash are intentional.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from algotrading.refdata.options.Archival import (
    ArchiveLock,
    ArchiveManifest,
    ManifestState,
    RawWeekArchiver,
    _utc_now,
)
from algotrading.refdata.options.Common import validate_database_identifier
from algotrading.refdata.options.Schema import SCHEMA_GENERATION


UTC = timezone.utc
POLICY_VERSION = 1
MINIMUM_AGE_DAYS = 90
INTERVAL_SECONDS = 300
MAX_AGE_SECONDS = 120
MONTHLY_TABLES = (
    "options_chain_snapshots",
    "options_chain_observations",
    "options_chain_observation_details",
    "options_chain_expiration_inputs",
    "options_chain_calculation_results",
    "options_chain_calculation_failures",
    "options_chain_calculation_reuse_provenance",
    "options_chain_calculation_state",
    "options_chain_surface_fits",
    "options_chain_surface_nodes",
    "options_chain_surface_failures",
    "options_chain_surface_state",
)
REPLACING_TABLES = frozenset(
    {
        "options_chain_calculation_failures",
        "options_chain_calculation_state",
        "options_chain_surface_failures",
        "options_chain_surface_state",
    }
)


def month_interval(month: date) -> tuple[datetime, datetime]:
    if month.day != 1:
        raise ValueError("--resample-month must be the first day of a month")
    next_month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    return (
        datetime(month.year, month.month, 1, tzinfo=UTC),
        datetime(next_month.year, next_month.month, 1, tzinfo=UTC),
    )


def _policy() -> dict[str, int]:
    return {
        "version": POLICY_VERSION,
        "minimum_age_days": MINIMUM_AGE_DAYS,
        "interval_seconds": INTERVAL_SECONDS,
        "max_age_seconds": MAX_AGE_SECONDS,
    }


class MonthlyResampler:
    def __init__(
        self, raw_archiver: RawWeekArchiver, *, logger: Optional[logging.Logger] = None
    ):
        self.raw = raw_archiver
        self.client = raw_archiver.client
        self.database = raw_archiver.database
        self.repository = raw_archiver.repository
        self.logger = logger or raw_archiver.logger
        self.config = raw_archiver.config

    def _table(self, canonical: str) -> str:
        return validate_database_identifier(
            self.raw.table_names.get(canonical, canonical)
        )

    def _scalar(self, sql: str, parameters: Optional[Mapping[str, Any]] = None) -> int:
        rows = self.client.query(sql, parameters=dict(parameters or {})).result_rows
        if not rows:
            raise RuntimeError("Monthly resampling query returned no row")
        return int(rows[0][0] or 0)

    @staticmethod
    def _names(manifest: ArchiveManifest) -> tuple[str, dict[str, str], dict[str, str]]:
        prefix = f"options_rs_{manifest.archive_id.hex}"
        ids = f"{prefix}_ids"
        sparse = {
            name: f"{prefix}_{index:02d}_s" for index, name in enumerate(MONTHLY_TABLES)
        }
        backup = {
            name: f"{prefix}_{index:02d}_b" for index, name in enumerate(MONTHLY_TABLES)
        }
        return ids, sparse, backup

    def eligible_months(
        self, now: datetime, selected: Optional[date] = None
    ) -> list[date]:
        if selected is not None:
            _, end = month_interval(selected)
            if end > now - timedelta(days=MINIMUM_AGE_DAYS):
                self.logger.info(
                    "resample_month month=%s status=skipped reason=too_recent", selected
                )
                return []
            return [selected]
        rows = self.client.query(
            "SELECT partition FROM system.parts WHERE database = {database:String} "
            "AND table = {table:String} AND active GROUP BY partition ORDER BY partition",
            parameters={
                "database": self.database,
                "table": self._table("options_chain_snapshots"),
            },
        ).result_rows
        months: list[date] = []
        for row in rows:
            partition = str(row[0])
            if len(partition) != 6 or not partition.isdigit():
                raise RuntimeError(f"Unexpected snapshot month partition: {partition}")
            month = date(int(partition[:4]), int(partition[4:]), 1)
            if month_interval(month)[1] <= now - timedelta(days=MINIMUM_AGE_DAYS):
                months.append(month)
        return months

    def _guards(self, start: datetime, end: datetime) -> None:
        params = {"start": start, "end": end}
        raw = self._table("options_chain_raw")
        if self._scalar(
            f"SELECT count() FROM {self.database}.{raw} WHERE collected_at_utc >= "
            "{start:DateTime64(6, 'UTC')} AND collected_at_utc < {end:DateTime64(6, 'UTC')}",
            params,
        ):
            raise RuntimeError(
                "Monthly resampling requires no raw rows in the month"
            )
        manifest_table = self._table("options_chain_archive_manifest")
        if self._scalar(
            f"SELECT count() FROM {self.database}.{manifest_table} FINAL "
            "WHERE archive_kind = 'raw_week' AND range_start < {end:DateTime64(6, 'UTC')} "
            "AND range_end > {start:DateTime64(6, 'UTC')} AND state = 'writing'",
            params,
        ):
            raise RuntimeError(
                "Monthly resampling found a writing overlapping raw archive"
            )
        snapshots = self._table("options_chain_snapshots")
        calc = self._table("options_chain_calculation_state")
        surface = self._table("options_chain_surface_state")
        if self._scalar(
            f"SELECT count() FROM {self.database}.{snapshots} AS s LEFT JOIN "
            f"(SELECT * FROM {self.database}.{calc} FINAL) AS c "
            "ON c.snapshot_id = s.snapshot_id AND c.source_collected_at = s.source_collected_at "
            "WHERE s.source_collected_at >= {start:DateTime64(6, 'UTC')} "
            "AND s.source_collected_at < {end:DateTime64(6, 'UTC')} "
            "AND (isNull(c.snapshot_id) OR NOT c.is_active "
            "OR c.status NOT IN ('complete', 'complete_with_failures'))",
            params,
        ):
            raise RuntimeError("Monthly resampling found an incomplete calculation")
        if self._scalar(
            f"SELECT count() FROM {self.database}.{surface} FINAL "
            "WHERE source_collected_at >= {start:DateTime64(6, 'UTC')} "
            "AND source_collected_at < {end:DateTime64(6, 'UTC')} "
            "AND is_active AND status NOT IN ('complete', 'complete_with_failures')",
            params,
        ):
            raise RuntimeError("Monthly resampling found an incomplete surface attempt")
        attempts = self._table("options_chain_processing_attempts")
        if self._scalar(
            f"SELECT count() FROM (SELECT attempt_id, "
            "argMax(status, tuple(isNotNull(finished_at), "
            "ifNull(finished_at, started_at))) AS latest_status, "
            "argMax(finished_at, tuple(isNotNull(finished_at), "
            "ifNull(finished_at, started_at))) AS latest_finished_at "
            f"FROM {self.database}.{attempts} WHERE source_collected_at >= "
            "{start:DateTime64(6, 'UTC')} AND source_collected_at < "
            "{end:DateTime64(6, 'UTC')} GROUP BY attempt_id) "
            "WHERE latest_status = 'running' AND isNull(latest_finished_at)",
            params,
        ):
            raise RuntimeError("Monthly resampling found an active processing attempt")

    def _selection_sql(self) -> str:
        snapshots = self._table("options_chain_snapshots")
        calc = self._table("options_chain_calculation_state")
        surface = self._table("options_chain_surface_state")
        return f"""
SELECT snapshot_id, source_collected_at
FROM
(
    SELECT snapshot_id, source_collected_at,
           row_number() OVER
           (PARTITION BY source, underlying_symbol, sample_at
            ORDER BY source_collected_at DESC, snapshot_id DESC) AS position
    FROM
    (
        SELECT s.snapshot_id AS snapshot_id,
               s.source_collected_at AS source_collected_at,
               s.source AS source,
               s.underlying_symbol AS underlying_symbol,
               if(s.source_collected_at =
                    toStartOfInterval(s.source_collected_at, INTERVAL 5 MINUTE),
                  toStartOfInterval(s.source_collected_at, INTERVAL 5 MINUTE),
                  toStartOfInterval(s.source_collected_at, INTERVAL 5 MINUTE)
                    + INTERVAL 5 MINUTE) AS sample_at
        FROM {self.database}.{snapshots} AS s
        INNER JOIN (SELECT * FROM {self.database}.{calc} FINAL) AS c
          ON c.snapshot_id = s.snapshot_id
         AND c.source_collected_at = s.source_collected_at
        INNER JOIN (SELECT * FROM {self.database}.{surface} FINAL) AS u
          ON u.snapshot_id = s.snapshot_id
         AND u.source_collected_at = s.source_collected_at
        WHERE s.source_collected_at >= {{start:DateTime64(6, 'UTC')}}
          AND s.source_collected_at < {{end:DateTime64(6, 'UTC')}}
          AND c.is_active AND u.is_active
          AND c.status IN ('complete', 'complete_with_failures')
          AND u.status IN ('complete', 'complete_with_failures')
          AND (u.successful_expirations > 0 OR u.terminal_fit_failures > 0)
    ) AS candidates
    WHERE toUnixTimestamp64Micro(toDateTime64(sample_at, 6, 'UTC')) -
          toUnixTimestamp64Micro(source_collected_at) <= {MAX_AGE_SECONDS * 1000000}
) AS ranked
WHERE position = 1
"""

    def _check_staging_capacity(self, month_key: int, *, dry_run: bool) -> None:
        source_bytes = self._scalar(
            "SELECT sum(bytes_on_disk) FROM system.parts WHERE database = {database:String} "
            "AND active AND partition = {partition:String} "
            "AND table IN {tables:Array(String)}",
            {
                "database": self.database,
                "partition": str(month_key),
                "tables": [self._table(name) for name in MONTHLY_TABLES],
            },
        )
        free_bytes = self._scalar(
            "SELECT sum(free_space) FROM system.disks WHERE lower(type) = 'local'"
        )
        self.logger.info(
            "resample_capacity month=%s source_bytes=%s local_free_bytes=%s",
            month_key,
            source_bytes,
            free_bytes,
        )
        if not dry_run and source_bytes and free_bytes < source_bytes:
            raise RuntimeError(
                "Insufficient local free space for monthly staging and rollback copies"
            )

    def _measure_staging(self, manifest: ArchiveManifest, month_key: int) -> None:
        ids, sparse, backup = self._names(manifest)
        bytes_on_disk = self._scalar(
            "SELECT sum(bytes_on_disk) FROM system.parts WHERE database = {database:String} "
            "AND active AND partition = {partition:String} AND table IN {tables:Array(String)}",
            {
                "database": self.database,
                "partition": str(month_key),
                "tables": [ids, *sparse.values(), *backup.values()],
            },
        )
        self.logger.info(
            "resample_staging month=%s staged_partition_bytes=%s tables=%s",
            month_key,
            bytes_on_disk,
            1 + 2 * len(MONTHLY_TABLES),
        )

    def _logical_count(self, table: str, month_key: int, *, final: bool = False) -> int:
        return self._scalar(
            f"SELECT count() FROM {self.database}.{table}{' FINAL' if final else ''} "
            "WHERE toYYYYMM(source_collected_at) = {month:UInt32}",
            {"month": month_key},
        )

    def _snapshot_hash(self, table: str, month_key: int) -> bytes:
        digest = hashlib.sha256()
        result = self.client.query(
            f"SELECT toString(snapshot_id) FROM {self.database}.{table} "
            "WHERE toYYYYMM(source_collected_at) = {month:UInt32} ORDER BY snapshot_id",
            parameters={"month": month_key},
        )
        for row in result.result_rows:
            digest.update(str(row[0]).encode("ascii"))
            digest.update(b"\n")
        return digest.digest()

    def _new_manifest(self, start: datetime, end: datetime) -> ArchiveManifest:
        now = _utc_now()
        return ArchiveManifest(
            archive_id=uuid.uuid4(),
            archive_kind="normalized_month",
            range_start=start,
            range_end=end,
            schema_generation=SCHEMA_GENERATION,
            dataset_generation_id=self.raw._dataset_generation_id(),
            state=ManifestState.PLANNED,
            archive_path="",
            semantic_content_hash=None,
            archive_file_hash=None,
            row_count=0,
            distinct_snapshot_count=0,
            uncompressed_bytes=0,
            compressed_bytes=0,
            compression_settings_json=json.dumps(_policy(), sort_keys=True),
            owner_id=self.raw.owner_id,
            lease_expires_at=now + timedelta(seconds=self.config.lease_seconds),
            attempt_count=0,
            created_at=now,
            written_at=None,
            verified_at=None,
            purged_at=None,
            last_error="",
            updated_at=now,
        )

    def _settings(self, manifest: ArchiveManifest) -> dict[str, Any]:
        try:
            settings = json.loads(manifest.compression_settings_json)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Invalid monthly compaction manifest settings") from exc
        if any(settings.get(key) != value for key, value in _policy().items()):
            raise RuntimeError(
                "Monthly compaction policy changed for an existing month"
            )
        if (
            manifest.schema_generation != SCHEMA_GENERATION
            or manifest.dataset_generation_id != self.raw._dataset_generation_id()
        ):
            raise RuntimeError(
                "Monthly compaction dataset or schema generation changed"
            )
        return settings

    def _validate_completed(self, manifest: ArchiveManifest, month_key: int) -> None:
        settings = self._settings(manifest)
        counts = settings.get("retained_counts")
        if not isinstance(counts, dict) or set(counts) != set(MONTHLY_TABLES):
            raise RuntimeError("Completed monthly manifest lacks table counts")
        for name in MONTHLY_TABLES:
            actual = self._logical_count(
                self._table(name), month_key, final=name in REPLACING_TABLES
            )
            if actual != int(counts[name]):
                raise RuntimeError(f"Completed monthly partition changed: {name}")
        if (
            self._snapshot_hash(self._table("options_chain_snapshots"), month_key)
            != manifest.semantic_content_hash
        ):
            raise RuntimeError("Completed monthly snapshot set changed")

    def _staging_exists(self, manifest: ArchiveManifest) -> bool:
        ids, sparse, backup = self._names(manifest)
        return bool(
            self._scalar(
                "SELECT count() FROM system.tables WHERE database = {database:String} "
                "AND name IN {names:Array(String)}",
                {
                    "database": self.database,
                    "names": [ids, *sparse.values(), *backup.values()],
                },
            )
        )

    def _drop_staging(self, manifest: ArchiveManifest) -> None:
        ids, sparse, backup = self._names(manifest)
        for table in (ids, *sparse.values(), *backup.values()):
            self.client.command(f"DROP TABLE IF EXISTS {self.database}.{table} SYNC")

    def _stage(self, manifest: ArchiveManifest, month_key: int) -> dict[str, Any]:
        ids, sparse, backup = self._names(manifest)
        self._drop_staging(manifest)
        source_snapshot_hash = self._snapshot_hash(
            self._table("options_chain_snapshots"), month_key
        )
        self.client.command(
            f"CREATE TABLE {self.database}.{ids} "
            "(snapshot_id UUID, source_collected_at DateTime64(6, 'UTC')) "
            "ENGINE = MergeTree PARTITION BY toYYYYMM(source_collected_at) "
            "ORDER BY snapshot_id"
        )
        self.client.command(
            f"INSERT INTO {self.database}.{ids} (snapshot_id, source_collected_at) "
            + self._selection_sql(),
            parameters={"start": manifest.range_start, "end": manifest.range_end},
        )
        selected = self._logical_count(ids, month_key)
        input_counts: dict[str, int] = {}
        retained_counts: dict[str, int] = {}
        for name in MONTHLY_TABLES:
            source = self._table(name)
            self.client.command(
                f"CREATE TABLE {self.database}.{sparse[name]} AS {self.database}.{source}"
            )
            self.client.command(
                f"CREATE TABLE {self.database}.{backup[name]} AS {self.database}.{source}"
            )
            input_counts[name] = self._logical_count(
                source, month_key, final=name in REPLACING_TABLES
            )
            self.client.command(
                f"ALTER TABLE {self.database}.{backup[name]} "
                f"ATTACH PARTITION {{month:UInt32}} FROM {self.database}.{source}",
                parameters={"month": month_key},
            )
            self.client.command(
                f"INSERT INTO {self.database}.{sparse[name]} "
                f"SELECT * FROM {self.database}.{source}"
                f"{' FINAL' if name in REPLACING_TABLES else ''} "
                "WHERE toYYYYMM(source_collected_at) = {month:UInt32} "
                f"AND snapshot_id IN (SELECT snapshot_id FROM {self.database}.{ids})",
                parameters={"month": month_key},
            )
            retained_counts[name] = self._logical_count(sparse[name], month_key)
            expected_selected_rows = self._scalar(
                f"SELECT count() FROM {self.database}.{source}"
                f"{' FINAL' if name in REPLACING_TABLES else ''} "
                "WHERE toYYYYMM(source_collected_at) = {month:UInt32} "
                f"AND snapshot_id IN (SELECT snapshot_id FROM {self.database}.{ids})",
                {"month": month_key},
            )
            if retained_counts[name] != expected_selected_rows:
                raise RuntimeError(f"Staged monthly rows incomplete: {name}")
        if retained_counts["options_chain_snapshots"] != selected:
            raise RuntimeError("Staged snapshot count does not match selection")
        return {
            **_policy(),
            "input_counts": input_counts,
            "retained_counts": retained_counts,
            "selected_count": selected,
            "source_snapshot_hash": source_snapshot_hash.hex(),
        }

    def _validate_staging(
        self, manifest: ArchiveManifest, settings: Mapping[str, Any], month_key: int
    ) -> None:
        ids, sparse, backup = self._names(manifest)
        if self._logical_count(ids, month_key) != int(settings["selected_count"]):
            raise RuntimeError("Staged selection count changed")
        for name in MONTHLY_TABLES:
            if self._logical_count(sparse[name], month_key) != int(
                settings["retained_counts"][name]
            ):
                raise RuntimeError(f"Staged monthly row count changed: {name}")
            if self._logical_count(
                backup[name], month_key, final=name in REPLACING_TABLES
            ) != int(settings["input_counts"][name]):
                raise RuntimeError(f"Monthly rollback copy changed: {name}")
        if (
            self._snapshot_hash(sparse["options_chain_snapshots"], month_key)
            != manifest.semantic_content_hash
        ):
            raise RuntimeError("Staged snapshot set changed")
        bad_observations = self._scalar(
            f"SELECT count() FROM {self.database}.{sparse['options_chain_snapshots']} AS s "
            f"LEFT JOIN (SELECT snapshot_id, count() AS n, uniqExact(contract_id) AS distinct_n FROM {self.database}.{sparse['options_chain_observations']} GROUP BY snapshot_id) AS o USING snapshot_id "
            f"LEFT JOIN (SELECT snapshot_id, count() AS n FROM {self.database}.{sparse['options_chain_observation_details']} GROUP BY snapshot_id) AS d USING snapshot_id "
            "WHERE s.stored_observation_count != ifNull(o.n, 0) "
            "OR s.stored_distinct_contract_count != ifNull(o.distinct_n, 0) "
            "OR s.stored_detail_count != ifNull(d.n, 0)"
        )
        if bad_observations:
            raise RuntimeError(
                f"Staged snapshots have {bad_observations} observation count mismatches"
            )
        bad_states = self._scalar(
            f"SELECT count() FROM {self.database}.{sparse['options_chain_snapshots']} AS s "
            f"LEFT JOIN {self.database}.{sparse['options_chain_calculation_state']} AS c USING snapshot_id "
            f"LEFT JOIN {self.database}.{sparse['options_chain_surface_state']} AS u USING snapshot_id "
            "WHERE isNull(c.snapshot_id) OR NOT c.is_active "
            "OR c.status NOT IN ('complete', 'complete_with_failures') "
            "OR isNull(u.snapshot_id) OR NOT u.is_active "
            "OR u.status NOT IN ('complete', 'complete_with_failures') "
            "OR (u.successful_expirations = 0 AND u.terminal_fit_failures = 0)"
        )
        if bad_states:
            raise RuntimeError(
                f"Staged snapshots have {bad_states} invalid processing states"
            )
        calculation_state = sparse["options_chain_calculation_state"]
        results = sparse["options_chain_calculation_results"]
        calculation_failures = sparse["options_chain_calculation_failures"]
        bad_calculations = self._scalar(f"""
SELECT count()
FROM {self.database}.{calculation_state} AS st
LEFT JOIN
(
    SELECT snapshot_id, calculation_definition_id, source_calculation_run_id,
           count() AS successes
    FROM {self.database}.{results}
    GROUP BY snapshot_id, calculation_definition_id, source_calculation_run_id
) AS r ON r.snapshot_id = st.snapshot_id
      AND r.calculation_definition_id = st.calculation_definition_id
      AND isNotDistinctFrom(r.source_calculation_run_id, st.source_calculation_run_id)
LEFT JOIN
(
    SELECT snapshot_id, calculation_definition_id, source_calculation_run_id,
           countIf(failure_scope = 'contract') AS contracts,
           countIf(failure_scope = 'expiration') AS expirations,
           countIf(failure_scope = 'snapshot') AS snapshots
    FROM {self.database}.{calculation_failures}
    WHERE is_active
    GROUP BY snapshot_id, calculation_definition_id, source_calculation_run_id
) AS f ON f.snapshot_id = st.snapshot_id
      AND f.calculation_definition_id = st.calculation_definition_id
      AND isNotDistinctFrom(f.source_calculation_run_id, st.source_calculation_run_id)
WHERE st.is_active AND
    (st.successful_results != ifNull(r.successes, 0)
     OR st.terminal_contract_failures != ifNull(f.contracts, 0)
     OR st.terminal_expiration_failures != ifNull(f.expirations, 0)
     OR st.snapshot_failure_count != ifNull(f.snapshots, 0)
     OR st.eligible_contracts != st.expected_observations)
""")
        if bad_calculations:
            raise RuntimeError(
                f"Staged calculation totals disagree for {bad_calculations} snapshots"
            )
        surface_state = sparse["options_chain_surface_state"]
        fits = sparse["options_chain_surface_fits"]
        nodes = sparse["options_chain_surface_nodes"]
        surface_failures = sparse["options_chain_surface_failures"]
        required = "node_type = 'atm_forward' OR (node_type = 'delta' AND node_value IN (0.10, 0.25, 0.40))"
        bad_surfaces = self._scalar(f"""
SELECT count()
FROM {self.database}.{surface_state} AS st
LEFT JOIN
(
    SELECT snapshot_id, surface_definition_id, source_surface_run_id,
           countDistinctIf(expiration_at, price_basis IN ('mid', 'source')) AS expirations
    FROM {self.database}.{fits}
    GROUP BY snapshot_id, surface_definition_id, source_surface_run_id
) AS sf ON sf.snapshot_id = st.snapshot_id
       AND sf.surface_definition_id = st.surface_definition_id
       AND isNotDistinctFrom(sf.source_surface_run_id, st.source_surface_run_id)
LEFT JOIN
(
    SELECT snapshot_id, surface_definition_id, source_surface_run_id,
           countIf({required}) AS required_nodes,
           countIf(NOT ({required})) AS optional_nodes
    FROM {self.database}.{nodes}
    GROUP BY snapshot_id, surface_definition_id, source_surface_run_id
) AS sn ON sn.snapshot_id = st.snapshot_id
       AND sn.surface_definition_id = st.surface_definition_id
       AND isNotDistinctFrom(sn.source_surface_run_id, st.source_surface_run_id)
LEFT JOIN
(
    SELECT snapshot_id, surface_definition_id, source_surface_run_id,
           countIf(failure_scope = 'fit') AS fits,
           countIf(failure_scope = 'node' AND ({required})) AS required_nodes,
           countIf(failure_scope = 'node' AND NOT ({required})) AS optional_nodes
    FROM {self.database}.{surface_failures}
    WHERE is_active
    GROUP BY snapshot_id, surface_definition_id, source_surface_run_id
) AS ff ON ff.snapshot_id = st.snapshot_id
       AND ff.surface_definition_id = st.surface_definition_id
       AND isNotDistinctFrom(ff.source_surface_run_id, st.source_surface_run_id)
WHERE st.is_active AND
    (st.successful_expirations != ifNull(sf.expirations, 0)
     OR st.terminal_fit_failures != ifNull(ff.fits, 0)
     OR st.successful_required_nodes != ifNull(sn.required_nodes, 0)
     OR st.terminal_required_node_failures != ifNull(ff.required_nodes, 0)
     OR st.successful_optional_nodes != ifNull(sn.optional_nodes, 0)
     OR st.terminal_optional_node_failures != ifNull(ff.optional_nodes, 0))
""")
        if bad_surfaces:
            raise RuntimeError(
                f"Staged surface totals disagree for {bad_surfaces} snapshots"
            )

    def _verify_source_unchanged(
        self, settings: Mapping[str, Any], month_key: int
    ) -> None:
        if (
            self._snapshot_hash(self._table("options_chain_snapshots"), month_key).hex()
            != settings["source_snapshot_hash"]
        ):
            raise RuntimeError("Source snapshot set changed after staging")
        for name in MONTHLY_TABLES:
            actual = self._logical_count(
                self._table(name), month_key, final=name in REPLACING_TABLES
            )
            if actual != int(settings["input_counts"][name]):
                raise RuntimeError(f"Source month changed after staging: {name}")

    def _replace_from(
        self, tables: Mapping[str, str], counts: Mapping[str, int], month_key: int
    ) -> None:
        for name in MONTHLY_TABLES:
            source = self._table(name)
            if int(counts[name]) == 0:
                self.client.command(
                    f"ALTER TABLE {self.database}.{source} DROP PARTITION {{month:UInt32}}",
                    parameters={"month": month_key},
                )
            else:
                self.client.command(
                    f"ALTER TABLE {self.database}.{source} REPLACE PARTITION "
                    f"{{month:UInt32}} FROM {self.database}.{tables[name]}",
                    parameters={"month": month_key},
                )

    def _verify_target(
        self, settings: Mapping[str, Any], month_key: int, digest: bytes
    ) -> None:
        for name in MONTHLY_TABLES:
            if self._logical_count(
                self._table(name), month_key, final=name in REPLACING_TABLES
            ) != int(settings["retained_counts"][name]):
                raise RuntimeError(f"Replacement partition row count mismatch: {name}")
        if (
            self._snapshot_hash(self._table("options_chain_snapshots"), month_key)
            != digest
        ):
            raise RuntimeError("Replacement snapshot set mismatch")

    def _rollback(
        self, manifest: ArchiveManifest, settings: Mapping[str, Any], month_key: int
    ) -> None:
        _, _, backup = self._names(manifest)
        self._replace_from(backup, settings["input_counts"], month_key)
        self._verify_source_unchanged(settings, month_key)

    def process_month(self, month: date, *, dry_run: bool = False) -> str:
        start, end = month_interval(month)
        if end > _utc_now() - timedelta(days=MINIMUM_AGE_DAYS):
            self.logger.info(
                "resample_month month=%s status=skipped reason=too_recent", month
            )
            return "too_recent"
        month_key = month.year * 100 + month.month
        lock_path = self.config.output_directory / self.config.lock_filename
        with ArchiveLock(lock_path):
            manifest = self.repository.load(start, end, kind="normalized_month")
            if manifest is not None:
                self._settings(manifest)
                if manifest.state is ManifestState.PURGED:
                    self._validate_completed(manifest, month_key)
                    if not dry_run and self._staging_exists(manifest):
                        try:
                            self._drop_staging(manifest)
                        except Exception:
                            self.logger.exception(
                                "resample_staging_cleanup_failed month=%s", month
                            )
                    self.logger.info(
                        "resample_month month=%s status=skipped reason=already_resampled",
                        month,
                    )
                    return "already_resampled"
            if (
                manifest is None
                and self._logical_count(
                    self._table("options_chain_snapshots"), month_key
                )
                == 0
            ):
                self.logger.info(
                    "resample_month month=%s status=skipped reason=empty_month", month
                )
                return "empty_month"
            if manifest is None or manifest.state in {
                ManifestState.PLANNED,
                ManifestState.WRITING,
                ManifestState.FAILED,
            }:
                self._guards(start, end)
            if manifest is None or manifest.state in {
                ManifestState.PLANNED,
                ManifestState.WRITING,
                ManifestState.FAILED,
            }:
                self._check_staging_capacity(month_key, dry_run=dry_run)
            if dry_run:
                if manifest is not None and manifest.state in {
                    ManifestState.VERIFIED,
                    ManifestState.PURGING,
                }:
                    self.logger.info(
                        "resample_month month=%s status=dry_run_resume state=%s",
                        month,
                        manifest.state.value,
                    )
                else:
                    selected = self._scalar(
                        "SELECT count() FROM (" + self._selection_sql() + ")",
                        {"start": start, "end": end},
                    )
                    self.logger.info(
                        "resample_month month=%s status=dry_run_eligible selected_snapshots=%s",
                        month,
                        selected,
                    )
                return "dry_run"
            if manifest is None:
                manifest = self._new_manifest(start, end)
                self.repository.save(manifest)
            if manifest.state in {
                ManifestState.PLANNED,
                ManifestState.WRITING,
                ManifestState.FAILED,
            }:
                if manifest.state is not ManifestState.WRITING:
                    manifest = self.repository.transition(
                        manifest,
                        ManifestState.WRITING,
                        owner_id=self.raw.owner_id,
                        lease_expires_at=_utc_now()
                        + timedelta(seconds=self.config.lease_seconds),
                        attempt_count=manifest.attempt_count + 1,
                        last_error="",
                    )
                try:
                    settings = self._stage(manifest, month_key)
                    _, sparse, _ = self._names(manifest)
                    digest = self._snapshot_hash(
                        sparse["options_chain_snapshots"], month_key
                    )
                    manifest = self.repository.transition(
                        manifest,
                        ManifestState.WRITTEN,
                        row_count=settings["selected_count"],
                        distinct_snapshot_count=settings["selected_count"],
                        semantic_content_hash=digest,
                        compression_settings_json=json.dumps(settings, sort_keys=True),
                    )
                except Exception as exc:
                    self.repository.transition(
                        manifest, ManifestState.FAILED, last_error=str(exc)
                    )
                    raise
            settings = self._settings(manifest)
            if manifest.state is ManifestState.WRITTEN:
                manifest = self.repository.transition(manifest, ManifestState.VERIFYING)
            if manifest.state is ManifestState.VERIFYING:
                try:
                    self._validate_staging(manifest, settings, month_key)
                    manifest = self.repository.transition(
                        manifest, ManifestState.VERIFIED
                    )
                except Exception as exc:
                    self.repository.transition(
                        manifest, ManifestState.FAILED, last_error=str(exc)
                    )
                    raise
            if manifest.state is ManifestState.VERIFIED:
                self._validate_staging(manifest, settings, month_key)
                self._verify_source_unchanged(settings, month_key)
                self._measure_staging(manifest, month_key)
                manifest = self.repository.transition(manifest, ManifestState.PURGING)
            if manifest.state is ManifestState.PURGING:
                try:
                    _, sparse, _ = self._names(manifest)
                    self._validate_staging(manifest, settings, month_key)
                    self._replace_from(sparse, settings["retained_counts"], month_key)
                    self._verify_target(
                        settings, month_key, manifest.semantic_content_hash
                    )
                except Exception:
                    try:
                        self._rollback(manifest, settings, month_key)
                        self.repository.transition(
                            manifest,
                            ManifestState.FAILED,
                            last_error="Cutover failed; original partitions restored",
                        )
                    except Exception:
                        self.logger.exception(
                            "resample_rollback_failed month=%s", month
                        )
                    raise
                manifest = self.repository.transition(
                    manifest, ManifestState.PURGED, owner_id="", lease_expires_at=None
                )
            try:
                self._drop_staging(manifest)
            except Exception:
                self.logger.exception("resample_staging_cleanup_failed month=%s", month)
            self.logger.info(
                "resample_month month=%s status=resampled selected_snapshots=%s",
                month,
                manifest.row_count,
            )
            return "resampled"
