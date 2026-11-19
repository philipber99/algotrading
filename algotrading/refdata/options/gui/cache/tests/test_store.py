from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import CacheConfig
from ..model import SESSION_INDEX_SCHEMA
from ..store import CacheStore
from .. import store as store_module


UTC = timezone.utc


def _session_table(rows: int) -> pa.Table:
    source = {
        "underlying_symbol": "MSFT",
        "session_date": date(2026, 7, 29),
        "session_open_utc": datetime(2026, 7, 29, 13, 30, tzinfo=UTC),
        "session_close_utc": datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
        "regular_session_minutes": 390,
        "is_early_close": False,
        "day_of_week": 2,
        "first_sample_minute": 5,
        "last_sample_minute": 390,
        "expected_bin_count": 78,
        "available_bin_count": 78,
        "coverage_fraction": 1.0,
        "opening_spot": 100.0,
        "opening_atm_iv": 0.2,
        "quality_flags": 0,
        "rejected_snapshot_count": 0,
        "stale_sample_count": 0,
    }
    return pa.Table.from_pylist([source] * rows, schema=SESSION_INDEX_SCHEMA)


class ParquetWriterTests(unittest.TestCase):
    def _config(self, root: Path) -> CacheConfig:
        return CacheConfig.from_context(
            {
                "options_chain.time_of_day_cache.output_directory": str(root),
                "options_chain.time_of_day_cache.symbols": "MSFT",
            }
        )

    def test_small_arrow_batches_are_buffered_into_real_row_groups(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = CacheStore(self._config(root))
            staging = store.create_staging("build-row-groups")
            batch = _session_table(500)
            written = store.write_parquet_batches(
                staging,
                dataset="session_index",
                relative_path="data/test/session.parquet",
                tables=(batch for _ in range(130)),
            )
            metadata = pq.read_metadata(written.staged_path)
            self.assertEqual(metadata.num_rows, 65_000)
            self.assertEqual(metadata.num_row_groups, 2)
            self.assertGreaterEqual(metadata.row_group(0).num_rows, 32_000)
            self.assertLessEqual(metadata.row_group(0).num_rows, 65_536)

    def test_fresh_file_is_hashed_only_once_during_publication(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = CacheStore(self._config(root))
            staging = store.create_staging("build-hash")
            with mock.patch.object(
                store_module,
                "sha256_file",
                wraps=store_module.sha256_file,
            ) as digest:
                written = store.write_parquet(
                    staging,
                    dataset="session_index",
                    relative_path="data/test/session.parquet",
                    table=_session_table(1),
                )
                store.publish(
                    staging,
                    {
                        "generation_id": "generation-hash",
                        "files": [],
                    },
                    [written],
                )
                self.assertEqual(digest.call_count, 1)


if __name__ == "__main__":
    unittest.main()
