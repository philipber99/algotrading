from __future__ import annotations

import math
import unittest

import numpy as np
import pyarrow as pa
from unittest.mock import patch

from ..pricing import CompiledSurfaceIvEvaluator, evaluate_surface_iv_targets


class SurfaceTargetTests(unittest.TestCase):
    def test_svi_rows_are_vectorized_at_exact_targets(self):
        table = pa.table(
            {
                "fit_method": ["SVI", "SVI"],
                "svi_a": [0.04, 0.03],
                "svi_b": [0.10, 0.20],
                "svi_rho": [-0.2, 0.1],
                "svi_m": [0.0, -0.01],
                "svi_sigma": [0.10, 0.12],
                "pchip_k": [[], []],
                "pchip_total_variance": [[], []],
                "minimum_log_forward_moneyness": [-0.10, -0.20],
                "maximum_log_forward_moneyness": [0.10, 0.20],
                "time_to_expiry_seconds": [365.25 * 86400] * 2,
            }
        )
        targets = np.asarray([0.025, -0.075])
        values = evaluate_surface_iv_targets(table, targets)
        shifted = targets - np.asarray([0.0, -0.01])
        expected_variance = np.asarray([0.04, 0.03]) + np.asarray(
            [0.10, 0.20]
        ) * (
            np.asarray([-0.2, 0.1]) * shifted
            + np.sqrt(shifted * shifted + np.asarray([0.10, 0.12]) ** 2)
        )
        np.testing.assert_allclose(values, np.sqrt(expected_variance))

    def test_pchip_is_exact_and_never_extrapolates(self):
        table = pa.table(
            {
                "fit_method": ["PCHIP", "PCHIP"],
                "svi_a": [None, None],
                "svi_b": [None, None],
                "svi_rho": [None, None],
                "svi_m": [None, None],
                "svi_sigma": [None, None],
                "pchip_k": [[-0.1, 0.0, 0.1], [-0.1, 0.0, 0.1]],
                "pchip_total_variance": [[0.04, 0.05, 0.07]] * 2,
                "minimum_log_forward_moneyness": [-0.1, -0.1],
                "maximum_log_forward_moneyness": [0.1, 0.1],
                "time_to_expiry_seconds": [365.25 * 86400] * 2,
            }
        )
        values = evaluate_surface_iv_targets(table, [0.025, 0.2])
        self.assertTrue(math.isfinite(values[0]))
        self.assertGreater(values[0], 0.0)
        self.assertTrue(math.isnan(values[1]))

    def test_nonpositive_or_invalid_surface_results_are_nan(self):
        row = {
            "fit_method": "SVI",
            "svi_a": -1.0,
            "svi_b": 0.0,
            "svi_rho": 0.0,
            "svi_m": 0.0,
            "svi_sigma": 0.1,
            "pchip_k": [],
            "pchip_total_variance": [],
            "minimum_log_forward_moneyness": -0.1,
            "maximum_log_forward_moneyness": 0.1,
            "time_to_expiry_seconds": 365.25 * 86400,
        }
        self.assertTrue(math.isnan(evaluate_surface_iv_targets(row, 0.0)[0]))

    def test_compiled_pchip_constructs_once_and_reuses_interpolator(self):
        from .. import pricing

        real = pricing.PchipInterpolator
        with patch.object(pricing, "PchipInterpolator", wraps=real) as constructor:
            evaluator = CompiledSurfaceIvEvaluator(
                fit_method="PCHIP",
                minimum_log_forward_moneyness=-0.1,
                maximum_log_forward_moneyness=0.1,
                time_to_expiry_seconds=365.25 * 86400,
                pchip_k=[-0.1, 0.0, 0.1],
                pchip_total_variance=[0.04, 0.05, 0.07],
            )
            first = evaluator.evaluate(np.asarray([-0.05, 0.05]))
            second = evaluator.evaluate(np.asarray([[0.0], [0.2]]))
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(first.shape, (2,))
        self.assertEqual(second.shape, (2, 1))
        self.assertTrue(math.isnan(second[1, 0]))


if __name__ == "__main__":
    unittest.main()
