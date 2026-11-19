"""Configuration and logging bootstrap for cache CLI/API processes."""

from __future__ import annotations

import logging
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from .cache.config import CACHE_CONFIG_PREFIX
from .config import API_CONFIG_PREFIX
from .screener.config import SCREENER_CONFIG_PREFIX


def default_context(algo_home: str) -> dict[str, Any]:
    fields = import_module("algotrading.ContextFields")
    return {
        fields.PARAM_ALGO_HOME: algo_home,
        "options_chain.clickhouse.host": "localhost",
        "options_chain.clickhouse.port": 8123,
        "options_chain.clickhouse.username": "default",
        "options_chain.clickhouse.password": "",
        "options_chain.clickhouse.database": "algotrading",
        "options_chain.clickhouse.secure": False,
        CACHE_CONFIG_PREFIX + "output_directory": os.path.join(
            algo_home, "refdata", "options_cache", "time_of_day"
        ),
        CACHE_CONFIG_PREFIX + "windows": "10d:10,3m:63",
        CACHE_CONFIG_PREFIX + "default_window": "10d",
        CACHE_CONFIG_PREFIX + "sample_interval_minutes": 5,
        CACHE_CONFIG_PREFIX + "max_snapshot_age_seconds": 120,
        CACHE_CONFIG_PREFIX + "price_bases": "mid",
        CACHE_CONFIG_PREFIX + "require_surface_completion": True,
        CACHE_CONFIG_PREFIX + "allow_complete_with_failures": True,
        CACHE_CONFIG_PREFIX + "parquet_compression": "zstd",
        CACHE_CONFIG_PREFIX + "parquet_compression_level": 6,
        CACHE_CONFIG_PREFIX + "chunk_symbol_count": 2,
        CACHE_CONFIG_PREFIX + "keep_previous_generations": 1,
        CACHE_CONFIG_PREFIX + "staging_grace_hours": 24,
        CACHE_CONFIG_PREFIX + "minimum_session_coverage": 0.90,
        CACHE_CONFIG_PREFIX + "max_rejected_row_fraction": 0.01,
        CACHE_CONFIG_PREFIX + "nagios_service_suffix": "TimeOfDayCache",
        API_CONFIG_PREFIX + "host": "0.0.0.0",
        API_CONFIG_PREFIX + "port": 8788,
        API_CONFIG_PREFIX + "manifest_poll_seconds": 5,
        API_CONFIG_PREFIX + "result_cache_mb": 512,
        API_CONFIG_PREFIX + "max_concurrent_queries": 4,
        API_CONFIG_PREFIX + "query_timeout_seconds": 5,
        API_CONFIG_PREFIX + "max_lookback_sessions": 1260,
        API_CONFIG_PREFIX + "gzip_minimum_size": 1000,
        API_CONFIG_PREFIX + "max_request_bytes": 1_048_576,
        API_CONFIG_PREFIX + "log_level": "INFO",
        API_CONFIG_PREFIX + "pricing_workers": 0,
        API_CONFIG_PREFIX + "pricing_batch_size": 256,
        API_CONFIG_PREFIX + "max_pending_pricing_batches": 0,
        SCREENER_CONFIG_PREFIX + "max_results_per_strategy": 50,
        SCREENER_CONFIG_PREFIX + "max_prescore_candidates_per_strategy": 500,
        SCREENER_CONFIG_PREFIX + "max_generated_candidates_per_strategy": 5000,
        SCREENER_CONFIG_PREFIX + "max_concurrent_screens": 2,
        SCREENER_CONFIG_PREFIX + "screen_timeout_seconds": 30,
        SCREENER_CONFIG_PREFIX + "result_cache_mb": 256,
        SCREENER_CONFIG_PREFIX + "simulation_paths": 5000,
        SCREENER_CONFIG_PREFIX + "maximum_simulation_paths": 25000,
        SCREENER_CONFIG_PREFIX + "spot_grid_points": 81,
        SCREENER_CONFIG_PREFIX + "maximum_holding_sessions": 30,
        SCREENER_CONFIG_PREFIX + "iv_residual_half_life_sessions": 5,
        SCREENER_CONFIG_PREFIX + "maximum_short_itm_fraction": 0.20,
    }


def load_context(environment: str, *, logger_name: str) -> dict[str, Any]:
    algo_home = os.getenv("ALGO_HOME")
    if not algo_home:
        raise RuntimeError("ALGO_HOME environment variable is not set")
    normalized_home = str(Path(algo_home).expanduser().resolve()) + os.sep
    common = import_module("algotrading.Common")
    fields = import_module("algotrading.ContextFields")
    sections = ["options_chain"]
    if environment != "options_chain":
        sections.append(environment)
    context = common.load_configuration(
        default_context(normalized_home),
        normalized_home,
        sections,
        quiet=True,
    )
    logger = configure_logging(context, logger_name)
    context[fields.PARAM_LOGGER] = logger
    context["time_of_day_cache.environment"] = environment
    return context


def configure_logging(
    context: dict[str, Any] | None,
    logger_name: str,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s|%(levelname)s|%(name)s| %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if context:
        log_file = context.get(API_CONFIG_PREFIX + "log_file")
        if log_file:
            path = Path(str(log_file)).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        level = str(context.get(API_CONFIG_PREFIX + "log_level", "INFO")).upper()
    else:
        level = "INFO"
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger
