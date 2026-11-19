from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone

import pyarrow as pa

from ..config import CacheConfig
from ..errors import ValidationError
from ..model import SURFACE_OBSERVATION_SCHEMA
from ..queries import ClickHouseExtractor
from ..sampling import SampledSnapshot, SurfaceStateSnapshot, TradingSession


UTC = timezone.utc


class _ArrowClient:
    def __init__(self, table):
        self.table = table
        self.sql = ""
        self.external_data = None
        self.query_id = None
        self.parameters = None

    def query_arrow(self, sql, *, parameters, external_data=None, query_id=None):
        self.sql = sql
        self.external_data = external_data
        self.query_id = query_id
        self.parameters = parameters
        return self.table


class SurfaceRowQueryTests(unittest.TestCase):
    def test_source_watermarks_compare_raw_chains_with_terminal_surfaces(self):
        chain_at = datetime(2026, 8, 4, 20, 0, tzinfo=UTC)
        table = pa.table(
            {
                "underlying_symbol": ["ACTIVE", "RETIRED"],
                "latest_chain_at": [chain_at, chain_at],
                "latest_surface_at": [chain_at, None],
            }
        )
        client = _ArrowClient(table)
        context = {
            "options_chain.clickhouse.database": "options",
            "options_chain.time_of_day_cache.output_directory": "/tmp/cache-test",
            "options_chain.time_of_day_cache.symbols": "ACTIVE,RETIRED",
        }
        extractor = ClickHouseExtractor(
            client,
            context,
            CacheConfig.from_context(context),
        )

        result = extractor.source_watermarks(
            ("ACTIVE", "RETIRED"),
            end_utc=chain_at,
        )

        self.assertIn("options_chain_snapshots", client.sql)
        self.assertIn("options_chain_surface_state FINAL", client.sql)
        self.assertEqual(client.sql.count("source_collected_at <="), 2)
        self.assertEqual(
            client.parameters["statuses"],
            ["complete", "complete_with_failures"],
        )
        self.assertEqual(result["ACTIVE"].latest_chain_at, chain_at)
        self.assertTrue(result["ACTIVE"].surface_is_current(tolerance_seconds=120))
        self.assertIsNone(result["RETIRED"].latest_surface_at)
        self.assertFalse(
            result["RETIRED"].surface_is_current(tolerance_seconds=120)
        )

    @staticmethod
    def _with_cardinality(
        table,
        *,
        source_expirations,
        extracted_expirations,
        state_expirations,
    ):
        size = table.num_rows
        return (
            table.append_column(
                "source_expiration_count",
                pa.array([source_expirations] * size, type=pa.uint64()),
            )
            .append_column(
                "extracted_expiration_count",
                pa.array([extracted_expirations] * size, type=pa.uint64()),
            )
            .append_column(
                "state_successful_expirations",
                pa.array([state_expirations] * size, type=pa.uint32()),
            )
        )

    def test_sampled_external_relation_replaces_second_final_scan(self):
        snapshot_id = "11111111-1111-1111-1111-111111111111"
        collected_at = datetime(2026, 7, 29, 14, 35, tzinfo=UTC)
        expiration_at = collected_at + timedelta(days=30)
        session = TradingSession(
            date(2026, 7, 29),
            collected_at - timedelta(minutes=5),
            collected_at + timedelta(hours=6),
            False,
        )
        row = {
            "underlying_symbol": "MSFT",
            "session_date": session.session_date,
            "session_open_utc": session.open_utc,
            "session_close_utc": session.close_utc,
            "session_minute": 5,
            "sample_target_utc": collected_at,
            "snapshot_id": snapshot_id,
            "source_collected_at": collected_at,
            "sample_age_seconds": 0.0,
            "expiration_at": expiration_at,
            "price_basis": "mid",
            "surface_definition_id": 1,
            "calculation_definition_id": 1,
            "pricing_method": "tian",
            "fit_method": "SVI",
            "svi_a": 0.04,
            "svi_b": 0.1,
            "svi_rho": -0.2,
            "svi_m": 0.0,
            "svi_sigma": 0.1,
            "pchip_k": [],
            "pchip_total_variance": [],
            "spot_price": 100.0,
            "forward_price": 101.0,
            "time_to_expiry_seconds": 30 * 86400,
            "risk_free_rate": 0.04,
            "dividend_yield": 0.01,
            "minimum_log_forward_moneyness": -0.2,
            "maximum_log_forward_moneyness": 0.2,
            "input_contract_count": 20,
            "accepted_contract_count": 18,
            "rejected_contract_count": 2,
            "weighted_rmse": 0.01,
            "maximum_absolute_error": 0.02,
            "butterfly_arbitrage_flags": 0,
            "calendar_arbitrage_flags": 0,
            "quality_flags": 0,
        }
        table = pa.Table.from_pylist([row], schema=SURFACE_OBSERVATION_SCHEMA)
        client = _ArrowClient(
            self._with_cardinality(
                table,
                source_expirations=1,
                extracted_expirations=1,
                state_expirations=1,
            )
        )
        context = {
            "options_chain.clickhouse.database": "options",
            "options_chain.time_of_day_cache.output_directory": "/tmp/cache-test",
            "options_chain.time_of_day_cache.symbols": "MSFT",
        }
        extractor = ClickHouseExtractor(
            client,
            context,
            CacheConfig.from_context(context),
        )
        sampled = SampledSnapshot(
            SurfaceStateSnapshot(
                snapshot_id,
                "MSFT",
                collected_at,
                1,
                1,
                None,
                "complete",
                1,
            ),
            session,
            5,
            collected_at,
            0.0,
        )

        tables = list(extractor.iter_surface_tables([sampled]))

        self.assertEqual(tables[0].schema, SURFACE_OBSERVATION_SCHEMA)
        self.assertNotIn("authoritative_state", client.sql)
        self.assertNotIn(" FINAL", client.sql)
        self.assertEqual(client.sql.count("ALL INNER JOIN"), 3)
        self.assertNotIn("ANY INNER JOIN", client.sql)
        self.assertIn("ALL INNER JOIN sampled_snapshots AS st", client.sql)
        self.assertIn("isNotDistinctFrom", client.sql)
        self.assertIn("source_expiration_count", client.sql)
        self.assertIn("extracted_expiration_count", client.sql)
        self.assertIn("state_successful_expirations", client.sql)
        self.assertGreaterEqual(client.sql.count("PREWHERE underlying_symbol"), 3)
        self.assertIn("s.underlying_symbol = f.underlying_symbol", client.sql)
        self.assertIsNotNone(client.external_data)
        self.assertTrue(client.query_id.startswith("time_of_day_cache:"))
        payload = client.external_data.files[0].data
        self.assertIn(snapshot_id.encode(), payload)
        self.assertIn(b'"session_minute":5', payload)
        self.assertIn(b'"successful_expirations":1', payload)

    def test_multiple_expirations_survive_extraction(self):
        snapshot_id = "11111111-1111-1111-1111-111111111111"
        collected_at = datetime(2026, 7, 29, 14, 35, tzinfo=UTC)
        session = TradingSession(
            date(2026, 7, 29),
            collected_at - timedelta(minutes=5),
            collected_at + timedelta(hours=6),
            False,
        )
        rows = []
        for days in (7, 30, 60):
            rows.append(
                {
                    "underlying_symbol": "MSFT",
                    "session_date": session.session_date,
                    "session_open_utc": session.open_utc,
                    "session_close_utc": session.close_utc,
                    "session_minute": 5,
                    "sample_target_utc": collected_at,
                    "snapshot_id": snapshot_id,
                    "source_collected_at": collected_at,
                    "sample_age_seconds": 0.0,
                    "expiration_at": collected_at + timedelta(days=days),
                    "price_basis": "mid",
                    "surface_definition_id": 1,
                    "calculation_definition_id": 1,
                    "pricing_method": "tian",
                    "fit_method": "SVI",
                    "svi_a": 0.04,
                    "svi_b": 0.1,
                    "svi_rho": -0.2,
                    "svi_m": 0.0,
                    "svi_sigma": 0.1,
                    "pchip_k": [],
                    "pchip_total_variance": [],
                    "spot_price": 100.0,
                    "forward_price": 101.0,
                    "time_to_expiry_seconds": days * 86400,
                    "risk_free_rate": 0.04,
                    "dividend_yield": 0.01,
                    "minimum_log_forward_moneyness": -0.2,
                    "maximum_log_forward_moneyness": 0.2,
                    "input_contract_count": 20,
                    "accepted_contract_count": 18,
                    "rejected_contract_count": 2,
                    "weighted_rmse": 0.01,
                    "maximum_absolute_error": 0.02,
                    "butterfly_arbitrage_flags": 0,
                    "calendar_arbitrage_flags": 0,
                    "quality_flags": 0,
                }
            )
        table = pa.Table.from_pylist(rows, schema=SURFACE_OBSERVATION_SCHEMA)
        client = _ArrowClient(
            self._with_cardinality(
                table,
                source_expirations=3,
                extracted_expirations=3,
                state_expirations=3,
            )
        )
        context = {
            "options_chain.clickhouse.database": "options",
            "options_chain.time_of_day_cache.output_directory": "/tmp/cache-test",
            "options_chain.time_of_day_cache.symbols": "MSFT",
        }
        extractor = ClickHouseExtractor(
            client,
            context,
            CacheConfig.from_context(context),
        )
        sampled = SampledSnapshot(
            SurfaceStateSnapshot(
                snapshot_id,
                "MSFT",
                collected_at,
                1,
                1,
                None,
                "complete",
                3,
            ),
            session,
            5,
            collected_at,
            0.0,
        )

        tables = list(extractor.iter_surface_tables([sampled]))

        self.assertEqual(sum(table.num_rows for table in tables), 3)
        self.assertEqual(
            extractor.metrics.as_dict()["surface_rows_per_sampled_snapshot"],
            3.0,
        )

    def test_expiration_collapse_fails_validation(self):
        snapshot_id = "11111111-1111-1111-1111-111111111111"
        collected_at = datetime(2026, 7, 29, 14, 35, tzinfo=UTC)
        session = TradingSession(
            date(2026, 7, 29),
            collected_at - timedelta(minutes=5),
            collected_at + timedelta(hours=6),
            False,
        )
        row = {
            name: None for name in SURFACE_OBSERVATION_SCHEMA.names
        }
        row.update(
            underlying_symbol="MSFT",
            session_date=session.session_date,
            session_open_utc=session.open_utc,
            session_close_utc=session.close_utc,
            session_minute=5,
            sample_target_utc=collected_at,
            snapshot_id=snapshot_id,
            source_collected_at=collected_at,
            sample_age_seconds=0.0,
            expiration_at=collected_at + timedelta(days=7),
            price_basis="mid",
            surface_definition_id=1,
            calculation_definition_id=1,
            pricing_method="tian",
            fit_method="SVI",
            svi_a=0.04,
            svi_b=0.1,
            svi_rho=-0.2,
            svi_m=0.0,
            svi_sigma=0.1,
            pchip_k=[],
            pchip_total_variance=[],
            spot_price=100.0,
            forward_price=101.0,
            time_to_expiry_seconds=7 * 86400,
            risk_free_rate=0.04,
            dividend_yield=0.01,
            minimum_log_forward_moneyness=-0.2,
            maximum_log_forward_moneyness=0.2,
            input_contract_count=20,
            accepted_contract_count=18,
            rejected_contract_count=2,
            weighted_rmse=0.01,
            maximum_absolute_error=0.02,
            butterfly_arbitrage_flags=0,
            calendar_arbitrage_flags=0,
            quality_flags=0,
        )
        table = pa.Table.from_pylist([row], schema=SURFACE_OBSERVATION_SCHEMA)
        client = _ArrowClient(
            self._with_cardinality(
                table,
                source_expirations=3,
                extracted_expirations=1,
                state_expirations=3,
            )
        )
        context = {
            "options_chain.clickhouse.database": "options",
            "options_chain.time_of_day_cache.output_directory": "/tmp/cache-test",
            "options_chain.time_of_day_cache.symbols": "MSFT",
        }
        extractor = ClickHouseExtractor(
            client,
            context,
            CacheConfig.from_context(context),
        )
        sampled = SampledSnapshot(
            SurfaceStateSnapshot(
                snapshot_id,
                "MSFT",
                collected_at,
                1,
                1,
                None,
                "complete",
                3,
            ),
            session,
            5,
            collected_at,
            0.0,
        )

        with self.assertRaisesRegex(
            ValidationError,
            "discarded or omitted fitted expirations",
        ):
            list(extractor.iter_surface_tables([sampled]))


if __name__ == "__main__":
    unittest.main()
