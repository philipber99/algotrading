from __future__ import annotations

import logging
import unittest
from datetime import date, datetime, timedelta

from algotrading.tda.TDAReferenceData import TDAReferenceData


def _quote(mark: float) -> dict[str, float]:
    return {
        "bidPrice": mark - 0.1,
        "askPrice": mark + 0.1,
        "mark": mark,
        "openPrice": mark - 1.0,
        "quoteTime": 1786982400123,
        "tradeTime": 1786982399456,
    }


class TdaQuoteBatchTests(unittest.TestCase):
    def make_provider(self, response):
        provider = TDAReferenceData.__new__(TDAReferenceData)
        provider._quote_cache = {}
        provider._quote_cache_expire = 5
        provider._request = lambda *_args, **_kwargs: response
        return provider

    def test_get_quotes_returns_partial_batch_when_provider_omits_a_symbol(self):
        provider = self.make_provider({"SPY": {"quote": _quote(500.0)}})

        result = provider.get_quotes(["SPY", "BAD"])

        self.assertEqual(list(result), ["SPY"])
        self.assertEqual(result["SPY"]["mark"], 500.0)
        self.assertEqual(result["SPY"]["quote_time"], 1786982400123)
        self.assertEqual(result["SPY"]["trade_time"], 1786982399456)

    def test_option_quotes_accept_current_schwab_timestamp_field_names(self):
        provider = self.make_provider({
            "SPY   260918P00500000": {"quote": _quote(10.0)},
        })

        result = provider.get_options_quotes(["SPY   260918P00500000"])

        self.assertEqual(
            result["SPY   260918P00500000"]["quote_time"],
            1786982400123,
        )
        self.assertEqual(
            result["SPY   260918P00500000"]["trade_time"],
            1786982399456,
        )

    def test_expired_quote_cache_is_refreshed(self):
        provider = self.make_provider({"SPY": {"quote": _quote(501.0)}})
        provider._quote_cache["SPY"] = (datetime.now() - timedelta(seconds=10), _quote(400.0))

        result = provider.get_quotes(["SPY"])

        self.assertEqual(result["SPY"]["mark"], 501.0)

    def test_index_quote_uses_last_price_when_mark_is_zero(self):
        quote = {**_quote(0.0), "lastPrice": 15.42}
        provider = self.make_provider({"$VIX": {"quote": quote}})

        result = provider.get_quotes(["VIX"])

        self.assertEqual(result["$VIX"]["mark"], 15.42)

    def test_futures_quote_preserves_provider_slash_and_mark_without_last_price(self):
        provider = self.make_provider({
            "/ESU26": {"quote": _quote(6500.25)},
        })

        result = provider.get_quotes(["/ESU26"])

        self.assertEqual(result["/ESU26"]["mark"], 6500.25)

    def test_futures_quote_uses_midpoint_when_mark_and_last_are_unavailable(self):
        quote = {**_quote(6500.0), "mark": None, "bidPrice": 6499.75,
                 "askPrice": 6500.25}
        provider = self.make_provider({"/ESU26": {"quote": quote}})

        result = provider.get_quotes(["/ESU26"])

        self.assertEqual(result["/ESU26"]["mark"], 6500.0)

    def test_vix_option_chain_uses_schwab_index_symbol_and_normalizes_response(self):
        calls = []
        provider = self.make_provider({
            "symbol": "$VIX",
            "callExpDateMap": {},
            "putExpDateMap": {},
        })
        provider._request = lambda uri, parameters=None, **kwargs: (
            calls.append((uri, parameters, kwargs))
            or {"symbol": "$VIX", "callExpDateMap": {}, "putExpDateMap": {}}
        )

        result = provider.get_options_chain("VIX")

        self.assertEqual(result["symbol"], "VIX")
        self.assertEqual(calls[0][1]["symbol"], "$VIX")

    def test_invalid_option_chain_logs_request_context(self):
        provider = self.make_provider({
            "errors": [{"status": "400", "detail": "Check Param Values"}],
        })
        messages = []
        provider._log = lambda message, level=logging.INFO: messages.append(
            (level, message)
        )

        result = provider.get_options_chain(
            "VIX",
            strike_count=40,
            from_date=date(2026, 9, 18),
            to_date=date(2026, 9, 18),
        )

        self.assertIsNone(result)
        self.assertEqual(messages[0][0], logging.WARNING)
        message = messages[0][1]
        self.assertIn("requested_symbol='VIX'", message)
        self.assertIn("provider_symbol='$VIX'", message)
        self.assertIn("'strikeCount': 40", message)
        self.assertIn("'fromDate': '2026-09-18'", message)
        self.assertIn("'toDate': '2026-09-18'", message)
        self.assertIn("'status': '400'", message)


if __name__ == "__main__":
    unittest.main()
