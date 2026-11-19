from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import logging
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from algotrading.refdata.api.app import create_app
from algotrading.refdata.api.service import RefDataService
from algotrading.refdata.api.tests.fakes import FakeMarketData
from algotrading.refdata.api.tests.test_stock_analysis import FakeStockService
from algotrading.refdata.stock_analysis.analytics.service import AnalyticsService


NOW = datetime(2026, 8, 7, 18, tzinfo=timezone.utc)
SECTORS = ("Technology", "Financial Services", "Energy", "Industrials")


class GeneratedAnalyticsRepository:
    """Twenty-security, four-sector fixture with no provider capability."""

    def __init__(self):
        self.calls: list[str] = []
        self.short_interest_watermark = "2026-08-07T18:00:00"
        self.members = []
        for index in range(20):
            symbol = f"S{index:02d}"
            self.members.append({
                "index_id": "SP500", "security_id": index + 1,
                "company_id": index + 101, "symbol": symbol,
                "exchange": "NASDAQ", "security_name": f"Security {index}",
                "company_name": f"Company {index}", "security_type": "common_stock",
                "taxonomy": "TEST", "sector": SECTORS[index // 5],
                "industry_group": f"{SECTORS[index // 5]} Group",
                "industry": f"{SECTORS[index // 5]} Industry",
                "sub_industry": None, "market_cap": Decimal(str(1_000_000_000 + index * 10_000_000)),
                "market_cap_date": date(2026, 8, 7),
            })

    def data_watermark(self, universe=None):
        self.calls.append("data_watermark")
        return {"daily_price": "2026-08-07", "minute_price": "2026-08-07T18:00:00", "membership": "fixture-v1", "finra_short_interest": self.short_interest_watermark}

    def universe_snapshot(self, universe, *, as_of):
        self.calls.append("universe_snapshot")
        return [{**row, "index_id": universe} for row in self.members]

    def daily_bars(self, symbols, *, start, end):
        self.calls.append("daily_bars")
        rows = []
        first = max(start, date(2021, 7, 28))
        for symbol in symbols:
            symbol_index = int(symbol[1:]) if symbol.startswith("S") and symbol[1:].isdigit() else {"SPY": 2, "QQQ": 4, "VIX": 6}.get(symbol, 1)
            direction = -1 if symbol_index % 5 == 0 else 1
            current = first
            observation = 0
            while current <= end:
                if current.weekday() < 5:
                    close = 50.0 + symbol_index + direction * observation * 0.02 + (observation % 7) * 0.01
                    rows.append({"symbol": symbol, "timestamp": datetime.combine(current, time(16), timezone.utc), "open": close - .1, "high": close + .2, "low": close - .2, "close": max(1.0, close), "volume": 1_000_000 + symbol_index * 1000 + observation * 10})
                    observation += 1
                current += timedelta(days=1)
        return rows

    def minute_bars(self, symbols, *, start, end):
        self.calls.append("minute_bars")
        rows = []
        for symbol in symbols:
            if symbol == "S19":
                continue
            symbol_index = int(symbol[1:]) if symbol.startswith("S") and symbol[1:].isdigit() else 1
            for minute in range(30):
                timestamp = datetime(2026, 8, 7, 14, 30, tzinfo=timezone.utc) + timedelta(minutes=minute)
                close = 100 + symbol_index + minute * (0.01 if symbol_index % 2 else -0.005)
                rows.append({"symbol": symbol, "timestamp": timestamp, "open": close, "high": close + .1, "low": close - .1, "close": close, "volume": 1000 + minute})
        return rows

    def comparable_intraday_volumes(self, symbols, *, start, as_of):
        self.calls.append("comparable_intraday_volumes")
        rows = []
        for symbol in symbols:
            if symbol == "S19":
                continue
            current = start
            while current <= as_of.date():
                if current.weekday() < 5:
                    rows.append({"symbol": symbol, "session_date": current, "cumulative_volume": 30_000})
                current += timedelta(days=1)
        return rows

    def company_profile(self, symbol):
        self.calls.append("company_profile")
        if symbol != "S00":
            return None
        return {**self.members[0], "legal_name": "Company Zero Inc."}

    def company_financials(self, company_id, *, as_of, limit=400):
        self.calls.append("company_financials")
        rows = []
        quarter_ends = [date(2024, 9, 30), date(2024, 12, 31), date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31), date(2026, 3, 31), date(2026, 6, 30)]
        for index, period_end in enumerate(quarter_ends, 1):
            rows.append({"financial_id": index, "company_id": company_id, "filing_id": index, "period_type": "quarterly", "fiscal_year": period_end.year, "fiscal_quarter": (period_end.month - 1) // 3 + 1, "period_end": period_end, "accepted_at": datetime.combine(period_end + timedelta(days=28), time(12), timezone.utc), "accession": f"q{index}", "source_provider": "TEST", "diluted_eps": Decimal("1.00"), "operating_cash_flow": Decimal("40"), "capital_expenditure": Decimal("-10"), "total_debt": Decimal("300"), "cash_and_equivalents": Decimal("50"), "stockholders_equity": Decimal("100"), "current_assets": Decimal("200"), "current_liabilities": Decimal("100"), "operating_income": Decimal("20"), "interest_expense": Decimal("10")})
        rows.append({**rows[-2], "financial_id": 90, "filing_id": 90, "accepted_at": datetime(2026, 5, 15, tzinfo=timezone.utc), "accession": "q-amended", "diluted_eps": Decimal("1.25")})
        for year in range(2021, 2026):
            rows.append({"financial_id": 100 + year, "company_id": company_id, "filing_id": 100 + year, "period_type": "annual", "fiscal_year": year, "fiscal_quarter": None, "period_end": date(year, 12, 31), "accepted_at": datetime(year + 1, 2, 15, tzinfo=timezone.utc), "accession": f"a{year}", "source_provider": "TEST", "diluted_eps": Decimal("4"), "operating_cash_flow": Decimal("160"), "capital_expenditure": Decimal("-40"), "total_debt": Decimal("300"), "cash_and_equivalents": Decimal("50"), "stockholders_equity": Decimal("100"), "current_assets": Decimal("200"), "current_liabilities": Decimal("100"), "operating_income": Decimal("80"), "interest_expense": Decimal("20")})
        return rows[:limit]

    def company_earnings(self, symbol, *, start, end):
        self.calls.append("company_earnings")
        return []

    def universe_financials(self, company_ids, *, as_of):
        self.calls.append("universe_financials")
        rows = []
        for company_id in company_ids:
            rows.extend(self.company_financials(company_id, as_of=as_of))
        return rows

    def company_estimates(self, company_id):
        self.calls.append("company_estimates")
        return [
            {"period_type": "annual", "target_fiscal_year": 2026, "target_period_end": date(2026, 6, 30), "eps_mean": Decimal("5"), "source_provider": "TEST", "provider_as_of": NOW - timedelta(hours=1)},
            {"period_type": "annual", "target_fiscal_year": 2027, "target_period_end": date(2027, 6, 30), "eps_mean": Decimal("6"), "eps_analyst_count": 10, "source_provider": "TEST", "provider_as_of": NOW - timedelta(hours=1)},
        ]

    def company_market_cap(self, security_id, *, as_of):
        self.calls.append("company_market_cap")
        return {"market_cap": Decimal("1000000000"), "market_date": as_of, "source_provider": "TEST"}

    def company_events(self, symbol, *, as_of):
        self.calls.append("company_events")
        return {"last_paid_dividend_amount": Decimal("0.25"), "last_paid_dividend_payment_date": date(2026, 7, 15), "indicated_annual_dividend": Decimal("1.00"), "next_ex_dividend_date": date(2026, 8, 12), "next_earnings_date": date(2026, 8, 13), "next_earnings_session": "amc"}

    def company_short_volume(self, symbol, *, as_of, limit=21):
        self.calls.append("company_short_volume")
        return [
            {
                "symbol": symbol,
                "timestamp": as_of - timedelta(days=index),
                "short_volume": Decimal("50") if index else Decimal("60"),
                "finra_total_volume": Decimal("100"),
                "split_multiplier": Decimal("1"),
            }
            for index in range(limit)
        ]

    def company_short_interest(self, symbol, *, as_of, limit=130):
        self.calls.append("company_short_interest")
        if symbol != "S00":
            return []
        return [{
            "symbol": symbol,
            "settlement_date": as_of - timedelta(days=index * 15),
            "current_short_position": 1_000_000 + index * 50_000,
            "previous_short_position": 950_000 + index * 50_000,
            "change_percent": Decimal("5.26"),
            "change_previous_number": 50_000,
            "average_daily_volume": 250_000,
            "days_to_cover": Decimal("4.2"),
            "stock_split": False,
            "revised": index == 0,
        } for index in range(limit)]

    def finra_short_volume_sessions(self, *, as_of, session_limit=21):
        return [{"timestamp": as_of - timedelta(days=index)} for index in range(session_limit)]

    def market_short_volume(self, symbols, *, as_of, session_limit=63):
        self.calls.append("market_short_volume")
        rows = []
        for symbol in symbols:
            if not (symbol.startswith("S") and symbol[1:].isdigit()):
                continue
            symbol_index = int(symbol[1:])
            for index in range(session_limit):
                rows.append({
                    "symbol": symbol,
                    "timestamp": as_of - timedelta(days=index),
                    "short_volume": Decimal(str(30 + symbol_index + index % 3)),
                    "finra_total_volume": Decimal("100"),
                    "split_multiplier": Decimal("1"),
                })
        return rows


class EconomicRepository:
    def rows_in_window(self, start, end):
        return [{"timestamp": datetime(2026, 8, 10, 8, 30, tzinfo=timezone.utc), "internal_name": "manufacturing_pmi", "report": "ISM Manufacturing PMI (renamed)", "period": "Jul", "previous": 49.5, "estimate": 50.0, "actual": None, "impact": "High"}]


class CalendarService:
    def trading_calendar(self, start, end):
        rows = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                rows.append({"date": current})
            current += timedelta(days=1)
        return SimpleNamespace(data=rows)


class CachedQuoteSource:
    def __init__(self):
        self.calls = 0

    def day_quotes(self, symbols):
        self.calls += 1
        rows = []
        for index, symbol in enumerate(symbols):
            rows.append({
                "symbol": symbol,
                "open": 100.0,
                "mark": 100.0 + (index + 1) / 100,
            })
        return SimpleNamespace(data=rows)


class StreamingHistoryProbe:
    def __init__(self):
        self.history_calls = []

    def get_realtime_price_history(self):
        raise AssertionError("Analytics must not download raw quote history")

    def get_intraday_history(self, symbol, interval):
        self.history_calls.append((symbol, interval))
        return None


class AnalyticsContractTests(unittest.TestCase):
    def setUp(self):
        self.origin = "https://workbench.example.test"
        self.context = {"logger": logging.getLogger("analytics-contract"), "local_timezone": "UTC", "refdata.api.timezone": "UTC", "refdata.api.cors_origins": [self.origin], "refdata.api.intraday_data_configured": True, "refdata.economic_calendar.auto_update.enabled": False, "refdata.earnings_calendar.auto_update.enabled": False}
        self.repository = GeneratedAnalyticsRepository()
        analytics = AnalyticsService(self.context, self.repository, economic_repository=EconomicRepository(), calendar_service=CalendarService(), now_fn=lambda: NOW)
        refdata = RefDataService(self.context, market_data_module=FakeMarketData(), now_fn=lambda: NOW)
        self.client_context = TestClient(create_app(context=self.context, service=refdata, stock_service=FakeStockService(), analytics_service=analytics))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_capabilities_and_cors_are_explicit(self):
        response = self.client.get("/api/v1/capabilities")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["data"]["available"]["stock_screening"])
        self.assertTrue(response.json()["data"]["available"]["finra_short_interest"])
        self.assertEqual(
            response.json()["data"]["supported"]["finra_short_interest_max_reports"], 130)
        self.assertTrue(response.json()["data"]["intraday_data_configured"])
        cors_get = self.client.get("/api/v1/capabilities", headers={"Origin": self.origin})
        self.assertEqual(cors_get.headers.get("access-control-allow-origin"), self.origin)
        denied_get = self.client.get("/api/v1/capabilities", headers={"Origin": "https://unconfigured.example"})
        self.assertNotIn("access-control-allow-origin", denied_get.headers)
        preflight = self.client.options("/api/v1/capabilities", headers={"Origin": self.origin, "Access-Control-Request-Method": "GET"})
        self.assertEqual(preflight.headers.get("access-control-allow-origin"), self.origin)
        denied = self.client.options("/api/v1/capabilities", headers={"Origin": "https://unconfigured.example", "Access-Control-Request-Method": "GET"})
        self.assertNotIn("access-control-allow-origin", denied.headers)

    def test_wildcard_cors_origin_is_rejected(self):
        with self.assertRaises(ValueError):
            create_app(context={"refdata.api.cors_origins": ["*"]}, service=object(), stock_service=object(), analytics_service=object())

    def test_market_and_sector_endpoint_contracts_and_cache(self):
        market = self.client.get("/api/v1/analytics/market-overview")
        self.assertEqual(market.status_code, 200, market.text)
        payload = market.json()
        self.assertEqual(len(payload["data"]["coverage"]["requested_symbols"]), 20)
        self.assertIn("S19", payload["data"]["coverage"]["intraday_unavailable_symbols"])
        self.assertLessEqual(len(payload["data"]["leaders"]), 10)
        self.assertEqual(
            [row["symbol"] for row in payload["data"]["indexes"]],
            ["SPY", "QQQ", "RSP", "VIX"],
        )
        self.assertEqual(
            set(payload["data"]["indexes"][0]["returns"]),
            {"1w", "1m", "6m"},
        )
        self.assertEqual(payload["data"]["market_short_volume"]["universe"], "SP500")
        self.assertEqual(payload["data"]["market_short_volume"]["equal_weighted"]["status"], "available")
        self.assertEqual(payload["data"]["market_short_volume"]["cap_weighted"]["status"], "available")
        self.assertGreater(
            payload["data"]["market_short_volume"]["cap_weighted"]["value"],
            payload["data"]["market_short_volume"]["equal_weighted"]["value"],
        )
        most_shorted = payload["data"]["most_shorted"]
        least_shorted = payload["data"]["least_shorted"]
        self.assertEqual(len(most_shorted), 10)
        self.assertEqual(len(least_shorted), 10)
        self.assertEqual(most_shorted[0]["symbol"], "S19")
        self.assertEqual(least_shorted[0]["symbol"], "S00")
        self.assertEqual(most_shorted[0]["observation_count"], 1)
        one_week_short = self.client.get(
            "/api/v1/analytics/market-overview",
            params={"mover_period": "1w"},
        )
        self.assertEqual(one_week_short.status_code, 200, one_week_short.text)
        self.assertEqual(
            one_week_short.json()["data"]["most_shorted"][0]["observation_count"],
            5,
        )
        leader_values = [row["return"]["value"] for row in payload["data"]["leaders"]]
        self.assertEqual(leader_values, sorted(leader_values, reverse=True))
        if payload["data"]["leaders"]:
            self.assertEqual(payload["data"]["leaders"][0]["relative_volume"]["basis"], "intraday_vs_completed_daily")
        self.assertEqual(payload["meta"]["sql_query_count"], 5)
        self.assertNotIn("data_watermark", self.repository.calls)
        self.assertNotIn("comparable_intraday_volumes", self.repository.calls)
        cached = self.client.get("/api/v1/analytics/market-overview")
        self.assertEqual(cached.headers["X-Cache"], "HIT")
        daily_reads = self.repository.calls.count("daily_bars")
        membership_reads = self.repository.calls.count("universe_snapshot")
        minute_reads = self.repository.calls.count("minute_bars")

        summary = self.client.get("/api/v1/analytics/sectors", params={"period": "3m"})
        self.assertEqual(summary.status_code, 200, summary.text)
        sectors = summary.json()["data"]["sectors"]
        self.assertEqual(len(sectors), 4)
        identifier = sectors[0]["classification_id"]
        constituents = self.client.get(f"/api/v1/analytics/sectors/{identifier}/constituents", params={"page_size": 2})
        self.assertEqual(constituents.status_code, 200, constituents.text)
        self.assertEqual(len(constituents.json()["data"]["constituents"]), 2)
        constituent = constituents.json()["data"]["constituents"][0]
        self.assertGreater(int(constituent["market_cap"]), 0)
        self.assertEqual(set(constituent["returns"]), {"1d", "3m", "1y"})
        self.assertEqual(set(constituent["channel_position"]), {"3m", "1y"})
        self.assertEqual(set(constituent["market_beta"]), {"3m", "1y"})
        self.assertEqual(set(constituent["sector_beta"]), {"3m", "1y"})
        sorted_one_year = self.client.get(
            f"/api/v1/analytics/sectors/{identifier}/constituents",
            params={"page_size": 2, "sort": "return_1y", "direction": "desc"},
        )
        self.assertEqual(sorted_one_year.status_code, 200, sorted_one_year.text)
        performance = self.client.get("/api/v1/analytics/sector-performance", params=[("classification_id", identifier), ("range", "3m")])
        self.assertEqual(performance.status_code, 200, performance.text)
        self.assertEqual(performance.json()["data"]["alignment"], "intersection_no_fill")
        pair_ids = [item["classification_id"] for item in sectors[:2]]
        relationships = self.client.get("/api/v1/analytics/sector-relationships", params={"left_id": pair_ids[0], "right_id": pair_ids[1]})
        self.assertEqual(relationships.status_code, 200, relationships.text)
        self.assertEqual(relationships.json()["data"]["pair"]["right_id"], pair_ids[1])
        replacement = self.client.get("/api/v1/analytics/sector-relationships", params={"left_id": pair_ids[1], "right_id": pair_ids[0]})
        self.assertEqual(replacement.json()["data"]["pair"]["left_id"], pair_ids[1])
        self.assertEqual(replacement.json()["data"]["pair"]["right_id"], pair_ids[0])
        one_day = self.client.get("/api/v1/analytics/sectors", params={"period": "1d"})
        self.assertEqual(one_day.status_code, 200, one_day.text)
        self.assertEqual(one_day.json()["data"]["period"], "1d")
        # The market request loaded the whole universe once. Different daily
        # periods and constituent widgets reuse those rows and membership.
        self.assertEqual(self.repository.calls.count("daily_bars"), daily_reads)
        self.assertEqual(
            self.repository.calls.count("universe_snapshot"), membership_reads
        )
        self.assertEqual(self.repository.calls.count("minute_bars"), minute_reads)
        scoreboard = self.client.get(
            "/api/v1/analytics/sectors",
            params={"period": "3m", "include_position_scoreboard": "true"},
        )
        self.assertEqual(scoreboard.status_code, 200, scoreboard.text)
        scoreboard_row = scoreboard.json()["data"]["sectors"][0]
        self.assertEqual(set(scoreboard_row["returns"]), {"1d", "1w", "1m", "3m"})
        self.assertEqual(set(scoreboard_row["channel_position"]), {"1w", "1m", "3m"})
        self.assertEqual(scoreboard_row["trailing_pe"]["status"], "available")
        self.assertEqual(scoreboard_row["five_year_mean_trailing_pe"]["status"], "available")
        three_year_performance = self.client.get("/api/v1/analytics/sector-performance", params=[("classification_id", identifier), ("range", "3y")])
        self.assertEqual(three_year_performance.status_code, 200, three_year_performance.text)
        self.assertEqual(three_year_performance.json()["data"]["range"], "3y")
        self.assertEqual(self.client.get("/api/v1/analytics/market-overview", params={"mover_limit": 51}).status_code, 422)
        too_many = [("classification_id", item["classification_id"]) for item in sectors for _ in range(2)][:6]
        self.assertEqual(self.client.get("/api/v1/analytics/sector-performance", params=too_many).status_code, 422)

    def test_bulk_market_overview_uses_quotes_without_streaming_history_fanout(self):
        repository = GeneratedAnalyticsRepository()
        quote_source = CachedQuoteSource()
        streaming = StreamingHistoryProbe()
        analytics = AnalyticsService(
            self.context,
            repository,
            live_intraday_source=streaming,
            quote_source=quote_source,
            now_fn=lambda: NOW,
        )

        result = analytics.market_overview(
            universe="SP100",
            benchmark="SPY",
            anchor="open",
            mover_period="today",
            mover_limit=10,
            weighting="cap_weighted",
        )

        self.assertEqual(quote_source.calls, 1)
        self.assertEqual(streaming.history_calls, [])
        self.assertTrue(result.data["leaders"])
        self.assertGreaterEqual(result.data["leaders"][0]["last_price"], 100.0)
        self.assertEqual(result.data["breadth"]["valid_count"], 20)
        self.assertEqual(result.data["coverage"]["intraday_unavailable_symbols"], [])

    def test_bulk_minute_snapshot_requests_only_missing_symbols(self):
        calls = []

        class MinuteSource(StreamingHistoryProbe):
            def get_intraday_bars(inner, symbols, start, end, *, timeout):
                calls.append((symbols, start, end, timeout))
                return [
                    {"symbol": "S19", "timestamp": "2026-08-07T06:30:00+00:00", "open": 99, "high": 102, "low": 98, "close": 101, "volume": 40},
                    {"symbol": "S19", "timestamp": "2026-08-07T06:31:00+00:00", "open": 101, "high": 102, "low": 100, "close": 102, "volume": 20},
                    {"symbol": "OTHER", "timestamp": "2026-08-07T06:31:00+00:00", "close": 999},
                    {"symbol": "S19", "timestamp": "2026-08-06T06:31:00+00:00", "close": 999},
                ]

        service = AnalyticsService(self.context, self.repository, live_intraday_source=MinuteSource(), now_fn=lambda: NOW)
        rows = service._today_minute_bars(["S00", "S19"], NOW.date(), NOW)
        self.assertEqual(calls, [(["S19"], datetime(2026, 8, 7, tzinfo=timezone.utc), NOW, 5)])
        live = [row for row in rows if row["symbol"] == "S19"]
        self.assertEqual([row["volume"] for row in live], [40, 20])
        self.assertEqual({row["symbol"] for row in rows}, {"S00", "S19"})
        service._today_minute_bars(["S00", "S19"], NOW.date(), NOW)
        self.assertEqual(len(calls), 1)

    def test_unavailable_minute_snapshot_keeps_market_quote_fallback(self):
        class WarmingSource(StreamingHistoryProbe):
            def get_intraday_bars(inner, *args, **kwargs):
                raise RuntimeError("snapshot warming")

        source = WarmingSource()
        service = AnalyticsService(self.context, self.repository, live_intraday_source=source,
                                   quote_source=CachedQuoteSource(), now_fn=lambda: NOW)
        result = service.market_overview(universe="SP100", benchmark="SPY", anchor="open",
                                         mover_period="today", mover_limit=10, weighting="cap_weighted")
        self.assertTrue(result.data["leaders"])
        self.assertEqual(result.data["coverage"]["intraday_unavailable_symbols"], [])
        self.assertEqual(source.history_calls, [])

    def test_minute_snapshot_return_uses_open_not_first_minute_close(self):
        service = AnalyticsService(self.context, self.repository, now_fn=lambda: NOW)
        day = date(2026, 8, 10)  # Monday uses the same session-open anchor.
        rows = {"SPY": [
            {"timestamp": datetime(2026, 8, 10, 6, 30, tzinfo=timezone.utc), "open": 100, "close": 102},
            {"timestamp": datetime(2026, 8, 10, 6, 31, tzinfo=timezone.utc), "open": 102, "close": 110},
        ]}
        for anchor in ("open", "monday"):
            result = service._selected_return("SPY", "today", anchor, day, {}, rows)
            self.assertAlmostEqual(result["value"], .1)
            self.assertEqual(result["anchor_value"], 100)

    def test_monday_anchor_uses_session_open_not_prior_close(self):
        analytics = AnalyticsService(
            self.context,
            self.repository,
            now_fn=lambda: datetime(2026, 8, 11, 18, tzinfo=timezone.utc),
        )
        day = date(2026, 8, 11)
        daily = {
            "SPY": [
                (datetime(2026, 8, 7, 16, tzinfo=timezone.utc), 100.0),
                (datetime(2026, 8, 10, 16, tzinfo=timezone.utc), 110.0),
            ]
        }
        daily_opens = {
            "SPY": [
                (datetime(2026, 8, 7, 9, 30, tzinfo=timezone.utc), 99.0),
                (datetime(2026, 8, 10, 9, 30, tzinfo=timezone.utc), 105.0),
            ]
        }
        intraday = {
            "SPY": [
                {"timestamp": datetime(2026, 8, 11, 9, 30, tzinfo=timezone.utc), "close": 112.0},
                {"timestamp": datetime(2026, 8, 11, 11, 0, tzinfo=timezone.utc), "close": 115.0},
            ]
        }

        monday = analytics._selected_return(
            "SPY", "today", "monday", day, daily, intraday,
            daily_opens=daily_opens,
        )
        prior_close = analytics._selected_return(
            "SPY", "today", "prior_close", day, daily, intraday,
            daily_opens=daily_opens,
        )

        self.assertAlmostEqual(monday["value"], 115.0 / 105.0 - 1.0)
        self.assertAlmostEqual(prior_close["value"], 115.0 / 110.0 - 1.0)
        self.assertEqual(monday["anchor_value"], 105.0)
        self.assertEqual(prior_close["anchor_value"], 110.0)

    def test_company_contract_has_point_in_time_valuation_and_catalyst(self):
        response = self.client.get("/api/v1/symbols/S00/analytics", params={"range": "1y"})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertEqual(data["valuation"]["current_forward_pe"]["target_period_end"], "2027-06-30")
        self.assertEqual(data["valuation"]["current_trailing_pe"]["eps_provenance"], "four_distinct_reported_quarters")
        self.assertTrue(data["valuation"]["five_year_mean_trailing_pe"]["point_in_time"])
        self.assertEqual(data["economic_catalysts"][0]["internal_name"], "manufacturing_pmi")
        self.assertEqual(data["short_volume_statistics"]["short_volume_percent"]["value"], 0.6)
        self.assertAlmostEqual(data["short_volume_statistics"]["one_day_one_week_ratio"]["value"], 0.6 / 0.52)
        self.assertAlmostEqual(data["short_volume_statistics"]["one_week_one_month_ratio"]["value"], 0.52 / (10.6 / 21))
        labels = {item["code"]: item for item in data["condition_labels"]}
        self.assertTrue(labels["incoming_economic_report"]["active"])
        self.assertEqual(self.repository.calls.count("company_estimates"), 1)

    def test_finra_short_interest_history_is_bounded_ordered_and_typed(self):
        response = self.client.get(
            "/api/v1/symbols/S00/short-interest",
            params={"report_limit": 2, "as_of": "2026-08-07"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertEqual(data["symbol"], "S00")
        self.assertEqual(data["report_limit"], 2)
        self.assertEqual(data["source"], "FINRA_CONSOLIDATED_SHORT_INTEREST")
        self.assertEqual(len(data["observations"]), 2)
        self.assertEqual(
            [row["settlement_date"] for row in data["observations"]],
            ["2026-07-23", "2026-08-07"],
        )
        self.assertEqual(data["latest"], data["observations"][-1])
        self.assertEqual(data["latest"]["short_interest"], 1_000_000)
        empty = self.client.get("/api/v1/symbols/S99/short-interest")
        self.assertEqual(empty.status_code, 200, empty.text)
        self.assertIsNone(empty.json()["data"]["latest"])
        invalid = self.client.get(
            "/api/v1/symbols/S00/short-interest", params={"report_limit": 131}
        )
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(
            self.client.get(
                "/api/v1/symbols/S00/short-interest",
                params={"report_limit": 0},
            ).status_code,
            422,
        )

    def test_short_interest_cache_invalidates_on_table_watermark(self):
        repository = GeneratedAnalyticsRepository()
        service = AnalyticsService(
            {**self.context, "refdata.analytics.cache.use_watermark": True},
            repository,
            now_fn=lambda: NOW,
        )
        first = service.company_short_interest(
            "S00", as_of=date(2026, 8, 7), report_limit=2
        )
        second = service.company_short_interest(
            "S00", as_of=date(2026, 8, 7), report_limit=2
        )
        self.assertEqual(first.data, second.data)
        self.assertEqual(repository.calls.count("company_short_interest"), 1)
        repository.short_interest_watermark = "2026-08-07T19:00:00"
        service.company_short_interest(
            "S00", as_of=date(2026, 8, 7), report_limit=2
        )
        self.assertEqual(repository.calls.count("company_short_interest"), 2)

    def test_finra_short_volume_history_is_bounded_and_scaled_zero_to_one(self):
        response = self.client.get(
            "/api/v1/finra/short-volume",
            params={"symbols": "S00,S01", "session_limit": 5, "as_of": "2026-08-07"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertEqual(data["symbols"], ["S00", "S01"])
        self.assertEqual(data["session_limit"], 5)
        self.assertEqual(len(data["sessions"]), 5)
        self.assertEqual(len(data["observations"]), 10)
        self.assertTrue(all(
            0 <= row["short_volume_percent"] <= 1
            for row in data["observations"]
        ))


if __name__ == "__main__":
    unittest.main()
