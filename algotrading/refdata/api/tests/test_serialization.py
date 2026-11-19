from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.serialization import dataframe_records, json_value, scalar_price


class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.tz = ZoneInfo("America/Los_Angeles")

    def test_dataframe_index_and_nulls(self):
        dataframe = pd.DataFrame(
            {
                "close": [Decimal("123.45")],
                "optional": [np.nan],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2026-07-24 06:30:00")], name="date"),
        )
        records = dataframe_records(dataframe, self.tz, "timestamp")
        self.assertEqual(records[0]["close"], 123.45)
        self.assertIsNone(records[0]["optional"])
        self.assertTrue(records[0]["timestamp"].endswith("-07:00"))

    def test_nested_numpy_values(self):
        value = json_value({"count": np.int64(3), "missing": float("inf")}, self.tz)
        self.assertEqual(value["count"], 3)
        self.assertIsNone(value["missing"])

    def test_scalar_price_accepts_series(self):
        self.assertEqual(scalar_price(pd.Series([101.0, 102.0])), 102.0)


if __name__ == "__main__":
    unittest.main()
