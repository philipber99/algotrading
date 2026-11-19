from __future__ import annotations

from datetime import datetime, timezone
from configparser import ConfigParser
from pathlib import Path
import logging

import pandas as pd

from ..live_models import OptionMarketHistoryRequest
from ..service import RefDataService
from .fakes import FakeMarketData


class ConfiguredPrimaryProvider:
    def __init__(self, context):
        self.context = context


class ConfiguredOptionHistoryProvider:
    instances = []

    def __init__(self, context):
        self.context = context
        self.calls = []
        self.__class__.instances.append(self)

    def get_bars(self, symbol, start, end, granularity):
        self.calls.append((symbol, start, end, granularity))
        return pd.DataFrame(
            {"open": [4.0], "high": [6.0], "low": [3.0], "close": [5.0]},
            index=pd.DatetimeIndex([start.date()], name="timestamp"),
        )


def test_primary_and_optional_option_history_provider_classes_come_from_context(
    tmp_path,
):
    ConfiguredOptionHistoryProvider.instances.clear()
    context = {
        "logger": logging.getLogger("provider-configuration"),
        "local_timezone": "UTC",
        "refdata.api.timezone": "UTC",
        "refdata.providers.prices": f"{__name__}.ConfiguredPrimaryProvider",
        "refdata.providers.option_history": (
            f"{__name__}.ConfiguredOptionHistoryProvider"
        ),
        "refdata.api.option_history_cache.path": str(tmp_path / "options.sqlite3"),
    }
    service = RefDataService(
        context,
        market_data_module=FakeMarketData(),
        now_fn=lambda: datetime(2026, 8, 20, tzinfo=timezone.utc),
    )

    assert isinstance(service._provider(), ConfiguredPrimaryProvider)
    result = service.option_market_history(
        OptionMarketHistoryRequest(
            symbols=["XYZ260918C00100000"],
            start_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
            end_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            interval="1d",
            include_greeks=False,
        )
    )
    history_provider = ConfiguredOptionHistoryProvider.instances[0]
    assert len(history_provider.calls) == 1
    assert result.meta.source == "ConfiguredOptionHistoryProvider"


def test_empty_option_history_override_reuses_primary_provider():
    context = {
        "logger": logging.getLogger("provider-default"),
        "local_timezone": "UTC",
    }
    market_data = FakeMarketData()
    service = RefDataService(context, market_data_module=market_data)
    assert service._option_history_provider() is market_data.refdata_source


def test_historical_bar_override_keeps_live_quotes_and_option_history_separate():
    context = {
        "logger": logging.getLogger("historical-provider-configuration"),
        "refdata.providers.prices": f"{__name__}.ConfiguredPrimaryProvider",
        "refdata.providers.historical_bars": f"{__name__}.ConfiguredOptionHistoryProvider",
    }
    service = RefDataService(context, market_data_module=FakeMarketData())
    assert isinstance(service._historical_bars_provider(), ConfiguredOptionHistoryProvider)
    assert isinstance(service._provider(), ConfiguredPrimaryProvider)
    assert service._option_history_provider() is service._provider()


def test_repository_configuration_uses_provider_namespace():
    config = ConfigParser(interpolation=None)
    config.optionxform = str
    config.read(Path(__file__).resolve().parents[4] / "config.ini")
    legacy = {
        "RefDataInterfaceClass", "BackupRefDataInterfaceClass",
        "StockSplitRefDataInterfaceClass", "DividendsRefDataInterfaceClass",
        "EarningsDateRefDataInterfaceClass", "FXRefDataInterfaceClass",
        "CryptoDataInterfaceClass",
    }
    assert all(not legacy.intersection(config[section]) for section in config.sections())
    assert config["Default"]["refdata.providers.prices"].endswith(".TDAReferenceData")
    assert config["refdata"]["refdata.providers.historical_bars"].endswith(".AlpacaReferenceData")
    assert config["refdata"]["refdata.providers.economic_calendar"] == (
        "algotrading.refdata.providers.fmp.FMPProvider"
    )
    assert config["refdata"]["refdata.providers.earnings_calendar"].endswith(".FMPProvider")
    assert config["refdata"]["refdata.providers.stock_splits"].endswith(".FMPProvider")
    assert "refdata.providers.earnings_calendar_fallback" in config["refdata"]
    assert "refdata.providers.dividends" in config["refdata"]
    assert "refdata.providers.fx" in config["refdata"]
