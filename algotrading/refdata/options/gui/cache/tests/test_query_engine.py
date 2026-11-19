from __future__ import annotations

import math
import unittest
from concurrent.futures import Future
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pyarrow as pa

from ..model import ProfileRequest
from ..query_engine import (
    CanonicalLeg,
    TimeOfDayQueryEngine,
    _group_surface_rows,
    _surface_scalar_rows,
)


UTC = timezone.utc


class _Adapter:
    def __init__(self):
        self.requests = []
        self.volatilities = []

    def price_many(self, requests, volatilities, *, requested_dtype):
        self.requests.extend(requests)
        self.volatilities.extend(volatilities)
        return np.asarray([request.strike * 0.01 for request in requests])


class _Registry:
    def __init__(self):
        self.adapter = _Adapter()

    def get(self, method):
        return self.adapter


class _Executor:
    def __init__(self):
        self.batches = []

    def submit(self, function, jobs):
        self.batches.append(jobs)
        future = Future()
        future.set_result(np.asarray([job["strike"] for job in jobs]))
        return future


class EnterAndHoldTests(unittest.TestCase):
    def test_forward_change_changes_moneyness_but_never_the_held_strike(self):
        session_date = date(2026, 7, 29)
        collected = datetime(2026, 7, 29, 14, 35, tzinfo=UTC)
        expiration = collected + timedelta(days=30)
        table = pa.table(
            {
                "underlying_symbol": ["MSFT", "MSFT"],
                "session_date": [session_date, session_date],
                "session_minute": [5, 10],
                "snapshot_id": ["first", "later"],
                "source_collected_at": [collected, collected + timedelta(minutes=5)],
                "sample_age_seconds": [0.0, 0.0],
                "expiration_at": [expiration, expiration],
                "price_basis": ["mid", "mid"],
                "pricing_method": ["tian", "tian"],
                "fit_method": ["SVI", "SVI"],
                "svi_a": [0.04, 0.04],
                "svi_b": [0.10, 0.10],
                "svi_rho": [-0.2, -0.2],
                "svi_m": [0.0, 0.0],
                "svi_sigma": [0.10, 0.10],
                "pchip_k": [[], []],
                "pchip_total_variance": [[], []],
                "spot_price": [100.0, 110.0],
                "forward_price": [100.0, 110.0],
                "time_to_expiry_seconds": [30 * 86400, 30 * 86400 - 300],
                "risk_free_rate": [0.04, 0.04],
                "dividend_yield": [0.01, 0.01],
                "quality_flags": [0, 0],
                "butterfly_arbitrage_flags": [0, 0],
                "calendar_arbitrage_flags": [0, 0],
                "minimum_log_forward_moneyness": [-0.2, -0.2],
                "maximum_log_forward_moneyness": [0.2, 0.2],
                "__file_rank": [0, 0],
            }
        )
        grouped = _group_surface_rows(_surface_scalar_rows(table))
        moneyness = math.log(1.05)
        leg = CanonicalLeg(
            index=0,
            side="long",
            put_call="CALL",
            quantity=1.0,
            strike=105.0,
            expiration_at=expiration,
            target_dte_seconds=30 * 86400,
            moneyness=moneyness,
            expiration_group=expiration.isoformat(),
            strike_group=None,
            shared_strike_group=None,
        )
        request = ProfileRequest(
            underlying_symbol="MSFT",
            as_of_utc=collected,
            mode="enter_and_hold",
            value_component="total_value",
            entry_session_minute=5,
            legs=[
                {
                    "side": "long",
                    "put_call": "CALL",
                    "quantity": 1,
                    "strike": 105,
                    "expiration_at": expiration,
                    "anchor_forward": 100,
                }
            ],
        )
        registry = _Registry()
        context = {
            "options_chain.time_of_day_cache.output_directory": "/tmp/cache-test",
            "options_chain.time_of_day_cache.symbols": "MSFT",
        }
        engine = TimeOfDayQueryEngine(context, solver_registry=registry)

        observations = engine._enter_hold_observations(
            grouped,
            [leg],
            request,
            [],
            table,
            float("inf"),
        )

        self.assertEqual([item.session_minute for item in observations], [5, 10])
        self.assertEqual(
            [request.strike for request in registry.adapter.requests],
            [105.0, 105.0],
        )
        self.assertEqual(
            [request.spot for request in registry.adapter.requests],
            [100.0, 110.0],
        )
        self.assertNotEqual(
            registry.adapter.volatilities[0],
            registry.adapter.volatilities[1],
        )

        iv_only_registry = _Registry()
        iv_only_engine = TimeOfDayQueryEngine(
            context,
            solver_registry=iv_only_registry,
        )
        iv_only_request = request.model_copy(
            update={"value_component": "iv_only"}
        )
        iv_only_engine._enter_hold_observations(
            grouped,
            [leg],
            iv_only_request,
            [],
            table,
            float("inf"),
        )
        self.assertEqual(
            [request.strike for request in iv_only_registry.adapter.requests],
            [105.0, 105.0],
        )
        self.assertEqual(
            [request.spot for request in iv_only_registry.adapter.requests],
            [100.0, 100.0],
        )

        comparable_registry = _Registry()
        comparable_engine = TimeOfDayQueryEngine(
            context,
            solver_registry=comparable_registry,
        )
        comparable_request = request.model_copy(
            update={
                "mode": "comparable_entry",
                "entry_session_minute": None,
            }
        )
        comparable_engine._comparable_observations(
            grouped,
            [leg],
            comparable_request,
            [],
            table,
            float("inf"),
        )
        np.testing.assert_allclose(
            [request.strike for request in comparable_registry.adapter.requests],
            [105.0, 115.5],
        )

    def test_pricing_executor_uses_bounded_configured_batches(self):
        executor = _Executor()
        context = {
            "options_chain.time_of_day_cache.output_directory": "/tmp/cache-test",
            "options_chain.time_of_day_cache.symbols": "MSFT",
            "options_chain.time_of_day_api.pricing_workers": 2,
            "options_chain.time_of_day_api.pricing_batch_size": 2,
            "options_chain.time_of_day_api.max_pending_pricing_batches": 2,
        }
        engine = TimeOfDayQueryEngine(context, pricing_executor=executor)
        jobs = [
            {
                "strike": float(index + 1),
                "pricing_method": "tian",
            }
            for index in range(5)
        ]

        values = engine._price_iv_jobs(
            jobs,
            deadline=engine.clock() + 5.0,
        )

        np.testing.assert_array_equal(values, [1, 2, 3, 4, 5])
        self.assertEqual([len(batch) for batch in executor.batches], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
