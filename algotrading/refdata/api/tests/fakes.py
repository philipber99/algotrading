from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

FIXED_NOW = datetime(2026, 7, 24, 10, 0, tzinfo=timezone.utc)


class FakeReferenceData:
    def __init__(self, owner) -> None:
        self.owner = owner
        self.last_bars_args = None
        self.last_quote_symbols = None

    def get_quotes(self, symbols):
        self.owner._called("provider_get_quotes")
        self.last_quote_symbols = list(symbols)
        return {
            symbol: {
                "open": 100.0 + index,
                "bid": 101.0 + index,
                "ask": 101.2 + index,
                "mark": 101.1 + index,
                "quote_time": int(FIXED_NOW.timestamp() * 1000),
                "trade_time": int(FIXED_NOW.timestamp() * 1000),
            }
            for index, symbol in enumerate(symbols)
        }

    def get_bars(self, symbol, start, end, granularity):
        self.owner._called("provider_get_bars")
        self.last_bars_args = (symbol, start, end, granularity)
        return pd.DataFrame(
            {
                "open": [100.0, 100.5],
                "close": [100.5, 101.0],
                "low": [99.8, 100.3],
                "high": [100.7, 101.2],
                "volume": [100, 120],
            },
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2026-07-24 09:30:00"),
                    pd.Timestamp("2026-07-24 09:31:00"),
                ],
                name="timestamp",
            ),
        )


class FakeMarketData:
    def __init__(self) -> None:
        self.trading_calendar_df = None
        self.cached_stock_splits = {}
        self.calls: dict[str, int] = {}
        self.last_prices_kwargs = {}
        self.last_prices_range = None
        self.last_prices_context = None
        self.trading_day = True
        self.trading_hours = True
        self.refdata_source = FakeReferenceData(self)

    def _called(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def get_refdata_source(self, context):
        self._called("get_refdata_source")
        return self.refdata_source

    def get_trading_calendar(self, context, start, end):
        self._called("get_trading_calendar")
        days = []
        current = start.date()
        while current <= end.date():
            if current.weekday() < 5:
                days.append(
                    {
                        "date": current,
                        "open": datetime.combine(current, time(9, 30)),
                        "close": datetime.combine(current, time(16, 0)),
                        "half_day": False,
                    }
                )
            current += timedelta(days=1)
        self.trading_calendar_df = pd.DataFrame(days)
        return self.trading_calendar_df

    def is_trading_day_today(self, context):
        self._called("is_trading_day_today")
        return self.trading_day

    def is_trading_hours(self, context):
        self._called("is_trading_hours")
        return self.trading_hours

    def prices_sql(self, context, symbol, start, end, **kwargs):
        self._called("prices_sql")
        self.last_prices_context = context
        self.last_prices_kwargs = kwargs
        self.last_prices_range = (start, end)
        return pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "close": [101.0, 102.0],
                "low": [99.5, 100.5],
                "high": [101.5, 102.5],
                "volume": [1000, 1200],
            },
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2026-07-23"),
                    pd.Timestamp("2026-07-24"),
                ],
                name="date",
            ),
        )

    def retrieve_current_price(self, context, symbol, bar=True):
        self._called("retrieve_current_price")
        self.last_current_price_symbol = symbol
        if bar:
            return pd.DataFrame(
                {
                    "open": [101.0],
                    "close": [102.0],
                    "low": [100.5],
                    "high": [102.5],
                    "volume": [1200],
                },
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-07-24 10:00:00")], name="timestamp"
                ),
            )
        return pd.Series(
            [102.0],
            index=pd.DatetimeIndex([pd.Timestamp("2026-07-24 10:00:00")]),
        )

    def get_stock_splits(self, context, symbol):
        self._called("get_stock_splits")
        assert symbol not in self.cached_stock_splits
        result = pd.DataFrame(
            {"symbol": [symbol], "type": ["SPLIT"], "ratio": [0.25]},
            index=pd.DatetimeIndex([pd.Timestamp("2020-08-31")], name="date"),
        )
        self.cached_stock_splits[symbol] = result
        return result

    def get_earnings_calendar(self, context, symbol, start, end):
        self._called("get_earnings_calendar")
        return pd.DataFrame(
            {
                "symbol": [symbol],
                "date": [date(2026, 7, 30)],
                "eps": [2.5],
                "eps_estimate": [2.4],
            }
        )

    def get_dividend_dates(self, context, symbol, start, end, search_criteria):
        self._called("get_dividend_dates")
        return pd.DataFrame(
            {
                "symbol": [symbol],
                "ex_dividend_date": [date(2026, 8, 15)],
                "amount": [0.75],
            }
        )

    def get_fomc_dates(self, context, start, end, last_day_only=True):
        self._called("get_fomc_dates")
        return [date(2026, 7, 29)]

    def get_economic_calendar(self, context, start, end):
        self._called("get_economic_calendar")
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-07-24 08:30:00")],
                "report": ["GDP"],
                "actual": [2.1],
                "estimate": [2.0],
                "previous": [1.9],
                "surprise": [0.1],
                "ind_move": [0.2],
            }
        )
