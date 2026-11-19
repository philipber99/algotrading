"""Repository-standard configuration and logging bootstrap."""

from __future__ import annotations

import logging
import math
import os
import sys
from datetime import time
from importlib import import_module
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import RiskApiConfig
from algotrading.risk.instruments.terms import (
    DEFAULT_FUTURES_SPECIFICATIONS,
    validate_futures_specification_table,
)
from algotrading.risk.market_inputs.freshness import (
    CALENDAR_VERSION,
    DEFAULT_MARKET_CALENDARS,
)


DEFAULT_CONTEXT: dict[str, Any] = {
    "risk.api.host": "127.0.0.1",
    "risk.api.port": 8789,
    "risk.api.log_level": "INFO",
    "risk.api.cors_origins": [],
    "risk.api.account_cache_ttl_seconds": 60,
    "risk.api.max_concurrent_broker_calls": 1,
    "risk.api.gzip_minimum_size": 1000,
    "risk.display_timezone": "America/Los_Angeles",
    "risk.refresh.remote_attempt_floor_seconds": 60,
    "risk.snapshot.ttl_seconds": 60,
    "risk.snapshot.hard_max_age_seconds": 300,
    "risk.cache.path": "",
    "risk.cache.max_entries": 5000,
    "risk.state.path": "",
    "risk.refdata.base_url": "http://127.0.0.1:8735",
    "risk.refdata.connect_timeout_seconds": 2,
    "risk.refdata.read_timeout_seconds": 10,
    "risk.refdata.retry_count": 1,
    "risk.refdata.risk_free_curve_cache_seconds": 1800,
    "risk.instrument_terms.cache_age_seconds": 86400,
    "risk.instrument_terms.maximum_option_contracts_per_request": 500,
    "risk.instrument_terms.allow_standard_occ_assumption": True,
    "risk.instrument_terms.standard_option_timezone": "America/New_York",
    "risk.instrument_terms.standard_option_expiration_local_time": "16:00",
    "risk.instrument_terms.static_futures_specifications": DEFAULT_FUTURES_SPECIFICATIONS,
    "risk.freshness.intraday_max_age_seconds": 900,
    "risk.freshness.low_delta_option_threshold": 0.10,
    "risk.freshness.low_delta_option_intraday_max_age_seconds": 14_400,
    "risk.freshness.completed_session_eod_allowed": True,
    "risk.freshness.mutual_fund_nav_max_session_lag": 1,
    "risk.freshness.regular_session_open_local_time": "09:30",
    "risk.freshness.regular_session_close_local_time": "16:00",
    "risk.freshness.calendar_version": CALENDAR_VERSION,
    "risk.freshness.calendars": DEFAULT_MARKET_CALENDARS,
    "risk.pricing.adaptive_base_steps": 256,
    "risk.pricing.adaptive_max_steps": 2048,
    "risk.pricing.price_absolute_tolerance": 0.01,
    "risk.pricing.price_relative_tolerance": 0.005,
    "risk.pricing.delta_absolute_tolerance": 0.01,
    "risk.pricing.greek_relative_tolerance": 0.05,
    "risk.pricing.greek_absolute_floor": 1e-6,
    "risk.option_history.minimum_contract_iv_observations": 126,
    "risk.option_history.adaptive_base_steps": 128,
    "risk.option_history.adaptive_max_steps": 256,
    "risk.option_history.greek_bridge_maximum_spot_return": 0.10,
    "risk.option_history.greek_bridge_maximum_iv_points": 15.0,
    "risk.option_history.greek_bridge_maximum_absolute_gamma": 0.20,
    "risk.option_history.greek_bridge_maximum_absolute_vega_per_1pct": 5.0,
    "risk.option_history.greek_bridge_maximum_dividend_yield": 0.08,
    "risk.option_history.greek_bridge_early_exercise_review_days": 30,
    "risk.option_history.greek_bridge_early_exercise_dividend_yield": 0.01,
    "risk.option_history.greek_bridge_minimum_dte": 7,
    "risk.option_history.greek_bridge_maximum_dte": 730,
    "risk.option_history.greek_bridge_maximum_log_moneyness": 0.25,
    "risk.option_history.greek_bridge_calibration": {},
    "risk.history.lookback_sessions": 252,
    "risk.history.minimum_observations": 126,
    "risk.history.daily_availability_convention": "next_utc_midnight",
    "risk.history.daily_availability_lag_seconds": 0,
    "risk.history.daily_bar_maximum_days": 376,
    "risk.history.option_volatility_factors": {},
    "risk.history.allow_fixed_current_iv_fallback": True,
    "risk.history.fixed_current_iv_greek_bridge_maximum_spot_return": 0.20,
    "risk.history.remote_daily_bar_symbols": ["SHY", "CIBR", "SOXL", "HEWJ"],
    "risk.history.risk_free_curve_prefetch_workers": 8,
    "risk.history.minimum_option_volatility_factor_overlap": 126,
    "risk.history.maximum_option_volatility_factor_error": 0.10,
    "risk.history.minimum_same_underlying_volatility_contracts": 2,
    "risk.history.maximum_same_underlying_volatility_dispersion_points": 3.0,
    "risk.history.material_rate_rho_usd_per_1pct": 10_000,
    "risk.history.rate_shocks_basis_points": {},
    "risk.history.allow_configured_economic_proxies": False,
    "risk.history.economic_proxies": {},
    "risk.scenarios.maximum_absolute_equity_shock": 0.50,
    "risk.scenarios.reprice_cache_max_entries": 20_000,
    "risk.scenarios.etf_response_minimum_observations": 126,
    "risk.scenarios.etf_response_ridge_alpha": 1e-3,
    "risk.scenarios.etf_response_minimum_r_squared": 0.20,
    "risk.calculations.maximum_excluded_marked_value_fraction": 0.03,
    "risk.scenarios.factor_mappings": {},
    "risk.scenarios.economic_instrument_mappings": {},
    "risk.factors.ridge_alpha": 1e-3,
    "risk.factors.minimum_observations": 126,
    "risk.tail_risk.minimum_gross_delta_coverage": 0.90,
    "risk.clusters.source_weights": {"history": 0.5, "factor": 0.2, "scenario": 0.3},
    "risk.clusters.minimum_pairwise_overlap": 60,
    "risk.hedge_lab.maximum_overhedge_fraction": 0.25,
    "risk.hedge_lab.maximum_candidate_quantity": 100,
    "risk.hedge_lab.maximum_option_symbols": 4,
    "risk.hedge_lab.maximum_preview_option_symbols": 2,
    "risk.hedge_lab.candidate_evaluation_workers": 3,
    "risk.hedge_lab.futures_roll_buffer_days": 7,
    "risk.hedge_lab.maximum_quote_age_seconds": 900,
    "risk.hedge_lab.allowed_futures_roots": ["ES", "MES", "NQ", "MNQ", "RTY", "M2K"],
    "risk.pnl.equity_history_lookback_days": 4,
    "risk.pnl.scenario_history_lookback_days": 4,
    "risk.pnl.remote_daily_bar_symbols": ["SCC", "TSCO", "VOLT"],
    "risk.quotes.maximum_option_contracts_per_request": 500,
    "risk.options.greek_source": "recalculated",
    "risk.options.fallback_to_provider": True,
    "risk.options.requested_dtype": "float64",
    "risk.options.maximum_spread_fraction_of_mid": 0.50,
    "risk.options.allow_wide_spread_inversion": True,
    "risk.options.maximum_last_trade_age_seconds": 900,
    "risk.options.cache_ttl_seconds": 60,
    "risk.options.tian.escrowed_dividend_mode_enabled": True,
    "risk.rates.max_age_days": 5,
    "risk.rates.allow_flat_extrapolation": True,
    "risk.rates.configured_flat_continuous_rate": None,
    "risk.dividends.default_continuous_yield": 0.0,
    "risk.alerts.assignment.expiry_window_trading_days": 2,
    "risk.alerts.assignment.maximum_extrinsic_usd": 0.10,
    "risk.alerts.margin.warning_utilization": 0.50,
    "risk.alerts.margin.critical_utilization": 0.70,
    "risk.alerts.position.conditions": [],
    "risk.alerts.position.rules": [],
    "risk.transactions.sql.host": "",
    "risk.transactions.sql.port": 3306,
    "risk.transactions.sql.database": "",
    "risk.transactions.sql.user": "",
    "risk.transactions.sql.password": "",
    "risk.transactions.sql.connection_timeout_seconds": 5,
    "risk.transactions.sql.pool_size": 4,
    "risk.transactions.sync.initial_start_at": None,
    "risk.transactions.sync.initial_lookback_days": 30,
    "risk.transactions.sync.overlap_minutes": 10,
    "risk.transactions.sync.chunk_days": 30,
    "risk.transactions.sync.minimum_split_seconds": 1,
    "risk.transactions.sync.advisory_lock_timeout_seconds": 0,
    "risk.option_prices.sql.host": "",
    "risk.option_prices.sql.port": 3306,
    "risk.option_prices.sql.database": "",
    "risk.option_prices.sql.user": "",
    "risk.option_prices.sql.password": "",
    "risk.option_prices.sql.connection_timeout_seconds": 5,
    "risk.option_prices.sql.pool_size": 4,
    "risk.option_prices.market_timezone": "America/New_York",
    "risk.option_prices.chain.strike_count": 20,
    "risk.option_prices.chain.max_strike_count": 80,
    "risk.option_prices.max_history_days": 376,
    "risk.option_prices.max_surface_snapshots": 2_000,
    "risk.option_prices.max_surface_points": 100_000,
    "risk.option_prices.surface_point_batch_size": 500,
    "risk.option_prices.backfill.timeout_seconds": 25,
    "risk.option_prices.backfill.max_attempts": 5,
    "risk.option_prices.backfill.max_requests_per_minute": 5,
}


def build_context(
    environment: str = "refdata", *, quiet_configuration: bool = False
) -> dict[str, Any]:
    """Load one config.ini environment using algotrading.Common."""

    algo_home = os.getenv("ALGO_HOME")
    if not algo_home:
        raise RuntimeError("ALGO_HOME environment variable is not set")

    root = str(Path(algo_home).expanduser().resolve())
    common = import_module("algotrading.Common")
    fields = import_module("algotrading.ContextFields")
    # Layer environment configuration over the non-secret service defaults.
    context = dict(DEFAULT_CONTEXT)
    context[fields.PARAM_ALGO_HOME] = root

    context = common.load_configuration(
        context,
        root + os.sep,
        environment,
        quiet=quiet_configuration,
    )
    # Provider payload-level logging can expose OAuth tokens and is never
    # required by the Risk API's sanitized account boundary.
    context["tda.oauthlib_logging"] = False
    context["tda.websockets_logging"] = False
    context[fields.PARAM_ALGO_HOME] = root
    context[fields.PARAM_APP_NAME] = context.get(fields.PARAM_APP_NAME) or "risk-api"
    context["risk.api.environment"] = environment
    _validate_option_price_chain_context(context)
    _validate_semantic_context(context)
    context[fields.PARAM_LOGGER] = configure_logger(context)
    RiskApiConfig.from_context(context)
    return context


def _validate_option_price_chain_context(context: dict[str, Any]) -> None:
    initial = _positive_integer(
        context.get("risk.option_prices.chain.strike_count"),
        "risk.option_prices.chain.strike_count",
    )
    maximum = _positive_integer(
        context.get("risk.option_prices.chain.max_strike_count"),
        "risk.option_prices.chain.max_strike_count",
    )
    if maximum < initial:
        raise ValueError(
            "risk.option_prices.chain.max_strike_count must not be smaller than "
            "risk.option_prices.chain.strike_count"
        )


def _validate_semantic_context(context: dict[str, Any]) -> None:
    table = context.get("risk.instrument_terms.static_futures_specifications")
    if not isinstance(table, dict):
        raise ValueError(
            "risk.instrument_terms.static_futures_specifications must be an object"
        )
    validate_futures_specification_table(table)
    base = _positive_integer(
        context.get("risk.pricing.adaptive_base_steps"),
        "risk.pricing.adaptive_base_steps",
    )
    maximum = _positive_integer(
        context.get("risk.pricing.adaptive_max_steps"),
        "risk.pricing.adaptive_max_steps",
    )
    # Adaptive pricing needs at least two tree depths to check convergence.
    if maximum < base * 2:
        raise ValueError(
            "risk.pricing.adaptive_max_steps must be at least twice adaptive_base_steps"
        )
    for key in (
        "risk.freshness.intraday_max_age_seconds",
        "risk.refdata.risk_free_curve_cache_seconds",
        "risk.freshness.low_delta_option_intraday_max_age_seconds",
        "risk.freshness.low_delta_option_threshold",
        "risk.calculations.maximum_excluded_marked_value_fraction",
        "risk.pricing.price_absolute_tolerance",
        "risk.pricing.price_relative_tolerance",
        "risk.pricing.delta_absolute_tolerance",
        "risk.pricing.greek_relative_tolerance",
        "risk.pricing.greek_absolute_floor",
    ):
        value = float(context.get(key))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    low_delta_threshold = float(context.get("risk.freshness.low_delta_option_threshold"))
    if low_delta_threshold > 1:
        raise ValueError("risk.freshness.low_delta_option_threshold must be at most 1")
    omission_limit = float(
        context.get("risk.calculations.maximum_excluded_marked_value_fraction")
    )
    if omission_limit >= 1:
        raise ValueError(
            "risk.calculations.maximum_excluded_marked_value_fraction must be below 1"
        )
    nav_session_lag = int(
        context.get("risk.freshness.mutual_fund_nav_max_session_lag")
    )
    if nav_session_lag < 0:
        raise ValueError(
            "risk.freshness.mutual_fund_nav_max_session_lag must be nonnegative"
        )
    # Validate evidence weights before they influence cluster aggregation.
    weights = context.get("risk.clusters.source_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("risk.clusters.source_weights must be a nonempty object")
    numeric = [float(value) for value in weights.values()]
    if any(not math.isfinite(value) or value < 0 for value in numeric) or sum(numeric) <= 0:
        raise ValueError("risk.clusters.source_weights must be finite and nonnegative")
    unsupported_sources = set(weights) - {"history", "historical", "factor", "scenario"}
    if unsupported_sources:
        raise ValueError("risk.clusters.source_weights contains an unsupported source")
    maximum_shock = float(context.get("risk.scenarios.maximum_absolute_equity_shock"))
    if not math.isfinite(maximum_shock) or not 0 < maximum_shock <= 1:
        raise ValueError(
            "risk.scenarios.maximum_absolute_equity_shock must be in (0, 1]"
        )
    etf_minimum = _positive_integer(
        context.get("risk.scenarios.etf_response_minimum_observations"),
        "risk.scenarios.etf_response_minimum_observations",
    )
    if etf_minimum < 2:
        raise ValueError(
            "risk.scenarios.etf_response_minimum_observations must be at least 2"
        )
    etf_ridge = float(context.get("risk.scenarios.etf_response_ridge_alpha"))
    if not math.isfinite(etf_ridge) or etf_ridge < 0:
        raise ValueError("risk.scenarios.etf_response_ridge_alpha must be nonnegative")
    etf_fit = float(context.get("risk.scenarios.etf_response_minimum_r_squared"))
    if not math.isfinite(etf_fit) or not 0 <= etf_fit <= 1:
        raise ValueError(
            "risk.scenarios.etf_response_minimum_r_squared must be in [0, 1]"
        )
    minimum_iv = _positive_integer(
        context.get("risk.option_history.minimum_contract_iv_observations"),
        "risk.option_history.minimum_contract_iv_observations",
    )
    if minimum_iv < 2:
        raise ValueError("risk.option_history.minimum_contract_iv_observations must be at least 2")
    maximum_quantity = _positive_integer(
        context.get("risk.hedge_lab.maximum_candidate_quantity"),
        "risk.hedge_lab.maximum_candidate_quantity",
    )
    if maximum_quantity < 1:
        raise ValueError("risk.hedge_lab.maximum_candidate_quantity must be positive")
    overhedge = float(context.get("risk.hedge_lab.maximum_overhedge_fraction"))
    if not math.isfinite(overhedge) or not 0 <= overhedge <= 1:
        raise ValueError("risk.hedge_lab.maximum_overhedge_fraction must be in [0, 1]")
    allowed_roots = context.get("risk.hedge_lab.allowed_futures_roots")
    if not isinstance(allowed_roots, list) or not allowed_roots:
        raise ValueError("risk.hedge_lab.allowed_futures_roots must be a nonempty list")
    unknown_roots = {
        str(root).upper().removeprefix("/") for root in allowed_roots
    } - set(table)
    if unknown_roots:
        raise ValueError("risk.hedge_lab.allowed_futures_roots contains unknown roots")
    # Fail startup when required session calendars or trading times are invalid.
    calendars = context.get("risk.freshness.calendars")
    if not isinstance(calendars, dict) or not {"XNYS", "CME_GLOBEX"} <= set(calendars):
        raise ValueError("risk.freshness.calendars must define XNYS and CME_GLOBEX")
    open_time = _configured_time(
        context.get("risk.freshness.regular_session_open_local_time"),
        "risk.freshness.regular_session_open_local_time",
    )
    close_time = _configured_time(
        context.get("risk.freshness.regular_session_close_local_time"),
        "risk.freshness.regular_session_close_local_time",
    )
    _configured_time(
        context.get("risk.instrument_terms.standard_option_expiration_local_time"),
        "risk.instrument_terms.standard_option_expiration_local_time",
    )
    if close_time <= open_time:
        raise ValueError("regular session close must be after regular session open")
    for key in (
        "risk.instrument_terms.standard_option_timezone",
        "risk.display_timezone",
    ):
        try:
            ZoneInfo(str(context.get(key)))
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"{key} must name an installed IANA timezone") from exc


def _configured_time(value: Any, name: str) -> time:
    try:
        hour, minute = (int(item) for item in str(value).split(":", 1))
        return time(hour, minute)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an HH:MM local time") from exc


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if converted < 1 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be a positive integer")
    return converted


def configure_logger(context: dict[str, Any]) -> logging.Logger:
    logger = logging.getLogger("algotrading.risk.api")
    logger.handlers.clear()
    logger.filters.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s|%(levelname)s|%(name)s| %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = context.get("risk.api.log_file")
    if log_file:
        path = Path(str(log_file)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    level_name = str(context.get("risk.api.log_level", "INFO")).upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    return logger
