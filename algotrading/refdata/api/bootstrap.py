"""Configuration and runtime initialization for the REST service."""

from __future__ import annotations

import logging
import os
import sys
from datetime import time
from importlib import import_module
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_CONTEXT: dict[str, Any] = {
    "refdata.reload_start": "2021-01-01",
    "refdata.threadpool_size": 3,
    "refdata.commit_interval": 50,
    "refdata.retrieval_block_size_days_1day": 440,
    "refdata.retrieval_block_size_days_1min": 1,
    "refdata.api.host": "127.0.0.1",
    "refdata.api.port": 8787,
    "refdata.api.log_level": "INFO",
    "refdata.api.cache.enabled": True,
    "refdata.api.cache.max_entries": 1024,
    "refdata.api.cache.day_quotes_ttl_seconds": 2,
    "refdata.api.cache.detailed_intraday_bars_ttl_seconds": 2,
    "refdata.api.cache.option_quotes_ttl_seconds": 2,
    "refdata.api.cache.option_chain_ttl_seconds": 2,
    "refdata.api.option_history_cache.path": "",
    "refdata.api.option_history_cache.ttl_seconds": 604800,
    "refdata.api.option_market_history_max_days": 376,
    "refdata.api.option_market_history_max_rows": 250000,
    "refdata.providers.prices": "algotrading.tda.TDAReferenceData.TDAReferenceData",
    "refdata.providers.historical_bars": "",
    "refdata.bars.lookback_trading_days": 1,
    "refdata.bars.symbol_batch_size": 500,
    "refdata.bars.sql_batch_size": 1000,
    "alpaca.bars.requests_per_minute": 190,
    "alpaca.bars.historical_delay_minutes": 16,
    "refdata.providers.option_history": "",
    "refdata.api.cache.risk_free_curve_ttl_seconds": 86400,
    "refdata.api.cache.futures_specs_ttl_seconds": 86400,
    "refdata.api.cache.reference_path": "",
    "refdata.api.quote_precache.enabled": True,
    "refdata.api.quote_precache.poll_interval_seconds": 60,
    "refdata.api.quote_precache.indexes": "SP100,SP500,NASDAQ100",
    "refdata.api.quote_precache.additional_symbols": "SPY,QQQ,VIX",
    "refdata.api.quote_precache.batch_size": 500,
    "refdata.api.quote_precache.max_symbols": 1000,
    "refdata.api.quote_precache.max_entries": 2000,
    "refdata.api.quote_precache.shutdown_wait_seconds": 10,
    "refdata.api.max_quote_symbols": 1000,
    "refdata.api.max_scalar_quote_fallbacks": 25,
    "refdata.api.max_option_quote_symbols": 500,
    "refdata.api.cache.option_contract_metadata_ttl_seconds": 86400,
    "refdata.api.option_terms.allow_standard_occ_assumption": True,
    "refdata.api.option_terms.standard_exchange_timezone": "America/New_York",
    "refdata.api.option_terms.standard_am_settlement_local_time": "09:30",
    "refdata.api.option_terms.standard_expiration_local_time": "16:00",
    "refdata.api.options_chain.enabled": True,
    "refdata.api.options_chain.max_strike_count": 120,
    # Hedge Analysis keeps one bounded, three-month chain in memory so users
    # can move between expirations without repeatedly hitting the provider.
    "refdata.api.options_chain.max_date_span_days": 95,
    "refdata.api.options_chain.max_contracts_in_response": 2000,
    "refdata.api.options_chain.max_response_bytes": 5242880,
    "refdata.api.option_expirations.max_date_span_days": 550,
    "refdata.api.cache.option_expirations_ttl_seconds": 300,
    "refdata.api.max_bulk_bar_symbols": 1000,
    "refdata.api.max_bulk_bar_rows": 1000000,
    "refdata.api.daily_bars_cache.enabled": True,
    "refdata.api.daily_bars_cache.path": "tmp/refdata_api_daily_bars.sqlite3",
    "refdata.api.max_detailed_intraday_rows": 2000,
    "refdata.api.max_concurrent_upstream_calls": 8,
    "refdata.api.upstream_slot_wait_seconds": 5,
    "refdata.api.calendar_refresh_seconds": 21600,
    "refdata.api.gzip_minimum_size": 1000,
    # Browser access is deny-by-default. Configure explicit HTTPS origins only.
    "refdata.api.cors_origins": [],
    "refdata.etf.http.max_response_bytes": 33554432,
    "refdata.fred.enabled": True,
    "refdata.fred.api_key": "",
    "refdata.fred.base_url": "https://api.stlouisfed.org",
    "refdata.fred.timeout_seconds": 10,
    "refdata.fred.retry_count": 2,
    "refdata.fred.observation_limit": 10,
    "refdata.fred.series_tenors": "SOFR:0,DGS1MO:30,DGS3MO:90,DGS6MO:180,DGS1:365",
    "refdata.fred.curve_history_observation_lookback_days": 10,
    "refdata.fred.curve_history_workers": 5,
    "refdata.cme_reference.enabled": False,
    "refdata.cme_reference.api_id": "",
    "refdata.cme_reference.app_password": "",
    "refdata.cme_reference.base_url": "",
    "refdata.cme_reference.token_url": "https://auth.cmegroup.com/as/token.oauth2",
    "refdata.cme_reference.timeout_seconds": 10,
    "refdata.cme_reference.max_contracts_per_request": 100,
    "refdata.cme_reference.datamine_list_url": "https://datamine.new.cmegroup.com/api/list_entitlements_files",
    "refdata.cme_reference.datamine_download_url": "https://datamine.new.cmegroup.com/cme/api/v2/download",
    "refdata.cme_reference.datamine_retry_count": 2,
    "refdata.cme_reference.datamine_retry_backoff_seconds": 0.5,
    "refdata.cme_reference.datamine_max_response_bytes": 67108864,
    "refdata.cme_reference.datamine_entitlement_pending": False,
    # Entitled CME deployments may override this isolated adapter path after
    # confirming it against their current provider specification.
    "refdata.cme_reference.instrument_path_template": "/v3/instruments?globexSymbol={contract}",
    "refdata.futures.enabled": True,
    "refdata.futures.cme_feed": "EOD",
    "refdata.futures.history_years": 5,
    "refdata.futures.cftc.base_url": "https://publicreporting.cftc.gov/resource",
    "refdata.futures.cftc.app_token": "",
    "refdata.futures.cftc.timeout_seconds": 15,
    "refdata.futures.cftc.page_size": 5000,
    "refdata.futures.cftc.retry_count": 2,
    "refdata.futures.cftc.retry_backoff_seconds": 0.5,
    "refdata.futures.cboe.timeout_seconds": 15,
    "refdata.futures.cboe.retry_count": 2,
    "refdata.futures.cboe.retry_backoff_seconds": 0.5,
    "refdata.futures.cboe.max_response_bytes": 33554432,
    "refdata.futures.analytics.cache_ttl_seconds": 120,
    "refdata.futures.analytics.cache_max_entries": 256,
    "refdata.futures.analytics.position_change_noise_oi_bps": 25,
    "refdata.futures.analytics.meaningful_price_move": 0.0025,
    "refdata.futures.analytics.meaningful_oi_change": 0.005,
    "refdata.analytics.cache.max_entries": 256,
    "refdata.analytics.cache.intraday_ttl_seconds": 30,
    "refdata.analytics.cache.daily_ttl_seconds": 300,
    "refdata.analytics.slow_log_threshold_ms": 1000,
    "refdata.analytics.cache.use_watermark": False,
    "refdata.analytics.daily_cache.warmup_enabled": True,
    "refdata.analytics.daily_cache.warmup_universes": "SP100,SP500,NASDAQ100",
    "refdata.analytics.intraday_rows_ttl_seconds": 60,
    "refdata.analytics.relationships.max_dimension": 64,
    "refdata.stock_analysis.forward_pe.provider_preference": "",
    "refdata.api.initialize_streaming_prices": True,
    "refdata.api.require_streaming_prices": False,
    "sql.enable_cache": False,
    "refdata.providers.default": "algotrading.refdata.providers.fmp.FMPProvider",
    "refdata.providers.company_profile": "algotrading.refdata.providers.fmp.FMPProvider",
    "refdata.providers.company_classification": "algotrading.refdata.providers.fmp.FMPProvider",
    "refdata.providers.security_identifiers": "algotrading.refdata.providers.fmp.FMPProvider",
    "refdata.providers.company_filings": "algotrading.refdata.providers.sec.SECProvider",
    "refdata.providers.company_financials": "algotrading.refdata.providers.sec.SECProvider",
    "refdata.providers.market_cap": "algotrading.refdata.providers.fmp.FMPProvider",
    "refdata.providers.index_constituents": "algotrading.refdata.providers.index_constituents.IndexConstituentProvider",
    "refdata.index_constituents.providers.default": "algotrading.refdata.providers.fmp.FMPProvider",
    "refdata.index_constituents.providers.SP100": "algotrading.refdata.providers.index_constituents.IsharesOEFProvider",
    "refdata.index_constituents.providers.NASDAQ100": "algotrading.refdata.providers.index_constituents.IsharesCSNDXProvider",
    "refdata.index_constituents.providers.SP500": "algotrading.refdata.providers.index_constituents.StateStreetSPYProvider",
    "refdata.providers.analyst_estimates": "algotrading.refdata.providers.fmp.FMPProvider",
    "refdata.providers.company_guidance": "algotrading.refdata.providers.sec.SECProvider",
    "refdata.providers.company_news": "algotrading.refdata.providers.fmp.FMPProvider",
    "refdata.providers.economic_calendar": "algotrading.refdata.providers.fmp.FMPProvider",
    "refdata.providers.earnings_calendar": "algotrading.refdata.providers.fmp.FMPProvider",
    "refdata.providers.stock_splits": "algotrading.refdata.providers.fmp.FMPProvider",
    "refdata.economic_calendar.auto_update.enabled": True,
    "refdata.economic_calendar.auto_update.horizon_days": 60,
    "refdata.economic_calendar.auto_update.poll_interval_seconds": 1800,
    "refdata.economic_calendar.auto_update.coverage_tail_slack_days": 3,
    "refdata.economic_calendar.auto_update.daily_overlap_days": 1,
    "refdata.economic_calendar.auto_update.lock_wait_seconds": 0,
    "refdata.economic_calendar.weekly_update.lookback_days": 10,
    "refdata.earnings_calendar.auto_update.enabled": True,
    "refdata.earnings_calendar.auto_update.poll_interval_seconds": 1800,
    "refdata.earnings_calendar.auto_update.lock_wait_seconds": 0,
    "refdata.earnings_calendar.overlap_days": 14,
    "refdata.earnings_calendar.horizon_days": 90,
    "refdata.earnings_calendar.chunk_days": 30,
    "refdata.stock_analysis.sql.lock_wait_seconds": 5,
    "refdata.stock_analysis.sql.connection_timeout_seconds": 5,
    "refdata.stock_analysis.sql.connection_attempts": 3,
    "refdata.stock_analysis.sql.connection_retry_backoff_seconds": 0.5,
    "refdata.stock_analysis.sql.port": 3306,
    "refdata.stock_analysis.sql.pool.enabled": True,
    "refdata.stock_analysis.sql.pool.size": 8,
    "refdata.stock_analysis.news.cache_ttl_seconds": 1800,
    "refdata.stock_analysis.read_cache.max_entries": 256,
    "refdata.stock_analysis.read_cache.fundamentals_ttl_seconds": 1800,
    "refdata.stock_analysis.read_cache.overview_ttl_seconds": 1800,
    "refdata.stock_analysis.refresh.profile_seconds": 604800,
    "refdata.stock_analysis.refresh.classification_seconds": 604800,
    "refdata.stock_analysis.refresh.identifiers_seconds": 604800,
    "refdata.stock_analysis.refresh.filings_seconds": 86400,
    "refdata.stock_analysis.refresh.financials_seconds": 86400,
    "refdata.stock_analysis.refresh.market_cap_seconds": 86400,
    "refdata.stock_analysis.refresh.index_constituents_seconds": 86400,
    "refdata.stock_analysis.refresh.estimates_seconds": 86400,
    "refdata.stock_analysis.refresh.guidance_seconds": 86400,
    "refdata.stock_analysis.batch.max_workers": 4,
    "refdata.stock_analysis.batch.chunk_size": 100,
    "refdata.stock_analysis.batch.failure_threshold": 0,
    "financialmodelingprep.base_url": "https://financialmodelingprep.com/stable",
    "financialmodelingprep.timeout_sec": 10,
    "financialmodelingprep.rate_limit_per_minute": 300,
    "financialmodelingprep.rate_limit_retry_seconds": 60,
    "financialmodelingprep.daily_request_limit": 0,
    "financialmodelingprep.quota_timezone": "UTC",
    "financialmodelingprep.quota_boundary_hour": 0,
    "financialmodelingprep.quota_lock_wait_seconds": 5,
    "financialmodelingprep.profile_cache_ttl_seconds": 60,
    "financialmodelingprep.profile_cache_max_entries": 128,
    "financialmodelingprep.index_authoritative_snapshot.NASDAQ100": True,
    "financialmodelingprep.index_authoritative_snapshot.SP500": True,
    "refdata.index_constituents.ishares_oef.url": "https://www.ishares.com/us/products/239723/ishares-s-p-100-etf/latest-holdings.csv",
    "refdata.index_constituents.ishares_oef.connect_timeout_seconds": 5,
    "refdata.index_constituents.ishares_oef.read_timeout_seconds": 15,
    "refdata.index_constituents.ishares_oef.max_attempts": 3,
    "refdata.index_constituents.ishares_oef.retry_backoff_seconds": 0.5,
    "refdata.index_constituents.ishares_oef.max_response_bytes": 2097152,
    "refdata.index_constituents.ishares_oef.max_snapshot_age_days": 7,
    "refdata.index_constituents.ishares_oef.max_future_days": 1,
    "refdata.index_constituents.ishares_oef.min_equity_count": 100,
    "refdata.index_constituents.ishares_oef.max_equity_count": 102,
    "refdata.index_constituents.ishares_oef.min_equity_weight": "0.98",
    "refdata.index_constituents.ishares_oef.max_equity_weight": "1.01",
    "refdata.index_constituents.ishares_oef.symbol_aliases": "BRKB=BRK.B",
    "refdata.index_constituents.ishares_csndx.url": "https://www.ishares.com/ch/individual/en/products/253741/ishares-nasdaq-100-ucits-etf/1495092304805.ajax?dataType=fund&fileName=CSNDX_holdings&fileType=csv",
    "refdata.index_constituents.ishares_csndx.connect_timeout_seconds": 5,
    "refdata.index_constituents.ishares_csndx.read_timeout_seconds": 15,
    "refdata.index_constituents.ishares_csndx.max_attempts": 3,
    "refdata.index_constituents.ishares_csndx.retry_backoff_seconds": 0.5,
    "refdata.index_constituents.ishares_csndx.max_response_bytes": 2097152,
    "refdata.index_constituents.ishares_csndx.max_snapshot_age_days": 7,
    "refdata.index_constituents.ishares_csndx.max_future_days": 1,
    "refdata.index_constituents.ishares_csndx.min_equity_count": 100,
    "refdata.index_constituents.ishares_csndx.max_equity_count": 110,
    "refdata.index_constituents.ishares_csndx.min_equity_weight": "0.98",
    "refdata.index_constituents.ishares_csndx.max_equity_weight": "1.01",
    "refdata.index_constituents.ishares_csndx.symbol_aliases": "",
    "refdata.index_constituents.state_street_spy.url": "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx",
    "refdata.index_constituents.state_street_spy.connect_timeout_seconds": 5,
    "refdata.index_constituents.state_street_spy.read_timeout_seconds": 15,
    "refdata.index_constituents.state_street_spy.max_attempts": 3,
    "refdata.index_constituents.state_street_spy.retry_backoff_seconds": 0.5,
    "refdata.index_constituents.state_street_spy.max_response_bytes": 2097152,
    "refdata.index_constituents.state_street_spy.max_uncompressed_bytes": 8388608,
    "refdata.index_constituents.state_street_spy.max_snapshot_age_days": 7,
    "refdata.index_constituents.state_street_spy.max_future_days": 1,
    "refdata.index_constituents.state_street_spy.min_equity_count": 495,
    "refdata.index_constituents.state_street_spy.max_equity_count": 510,
    "refdata.index_constituents.state_street_spy.min_equity_weight": "0.90",
    "refdata.index_constituents.state_street_spy.max_equity_weight": "1.01",
    "refdata.index_constituents.state_street_spy.symbol_aliases": "",
    "sec.user_agent": "",
    "sec.base_url": "https://data.sec.gov",
    "sec.timeout_sec": 10,
    "sec.requests_per_second": 8,
}


def build_context(
    environment: str = "refdata", *, quiet_configuration: bool = False
) -> dict[str, Any]:
    algo_home = os.getenv("ALGO_HOME")
    if not algo_home:
        raise RuntimeError("ALGO_HOME environment variable is not set")

    common = import_module("algotrading.Common")
    fields = import_module("algotrading.ContextFields")

    # Layer the selected environment over the public service defaults.
    context = common.load_configuration(
        dict(DEFAULT_CONTEXT),
        str(Path(algo_home).expanduser().resolve()) + os.sep,
        environment,
        quiet=quiet_configuration,
    )
    # The Reference Data API never needs HTTP/WebSocket payload-level logging;
    # legacy provider debug modes can expose bearer tokens and credentials.
    context["tda.oauthlib_logging"] = False
    context["tda.websockets_logging"] = False
    _validate_option_terms_context(context)

    logger = configure_logger(context)
    context[fields.PARAM_LOGGER] = logger
    context["refdata.api.environment"] = environment
    initialize_streaming_prices(context, common, fields, logger)
    return context


def _validate_option_terms_context(context: dict[str, Any]) -> None:
    try:
        ZoneInfo(str(context.get("refdata.api.option_terms.standard_exchange_timezone")))
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            "refdata.api.option_terms.standard_exchange_timezone must name an installed IANA timezone"
        ) from exc
    for key in (
        "refdata.api.option_terms.standard_am_settlement_local_time",
        "refdata.api.option_terms.standard_expiration_local_time",
    ):
        raw = context.get(key)
        try:
            hour, minute = (int(item) for item in str(raw).split(":", 1))
            time(hour, minute)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an HH:MM local time") from exc


def configure_logger(context: dict[str, Any]) -> logging.Logger:
    logger = logging.getLogger("algotrading.refdata.api")
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s|%(levelname)s|%(name)s| %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = context.get("refdata.api.log_file")
    if log_file:
        path = Path(str(log_file)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    level_name = str(context.get("refdata.api.log_level", "INFO")).upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    return logger


def initialize_streaming_prices(
    context: dict[str, Any],
    common: Any,
    fields: Any,
    logger: logging.Logger,
) -> None:
    # Record streaming availability explicitly so callers can inspect degraded startup.
    context["refdata.api.streaming_prices_available"] = False

    if not _as_bool(context.get("refdata.api.initialize_streaming_prices", True)):
        logger.info("Streaming-prices initialization is disabled")
        return

    class_key = getattr(fields, "PARAM_STREAMING_PRICES_CLASS", None)
    instance_key = getattr(fields, "PARAM_STREAMING_PRICES", None)
    if not class_key or not instance_key or class_key not in context:
        logger.info("No streaming-prices client is configured")
        return

    # Restrict streaming subscriptions to the configured symbol universe.
    configured_symbols = context.get(
        "refdata.api.streaming_symbols",
        context.get("pricing.active_symbols", []),
    )
    if isinstance(configured_symbols, str):
        configured_symbols = [
            item.strip().upper()
            for item in configured_symbols.split(",")
            if item.strip()
        ]
    else:
        configured_symbols = list(configured_symbols or [])

    try:
        streaming_class = common.import_module_attr(context[class_key])
        context[instance_key] = streaming_class(context, configured_symbols)
        context["refdata.api.streaming_prices_available"] = True
        logger.info(
            "Streaming-prices client initialized with %d configured symbols",
            len(configured_symbols),
        )
    except Exception as exc:
        if _as_bool(context.get("refdata.api.require_streaming_prices", False)):
            logger.error(
                "Unable to initialize the streaming-prices client exception=%s",
                type(exc).__name__,
            )
            raise
        else:
            logger.info("Unable to initialize streaming-pricing client. Skipped.")

def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
