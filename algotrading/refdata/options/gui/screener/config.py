"""Validated, bounded configuration for spread screening."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

from ..cache.errors import ConfigurationError


SCREENER_CONFIG_PREFIX = "options_chain.gui.screener."


def _value(context: Mapping[str, Any], name: str, default: Any) -> Any:
    return context.get(SCREENER_CONFIG_PREFIX + name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        raise ConfigurationError(f"Cannot parse boolean value {value!r}")
    return bool(value)


@dataclass(frozen=True)
class ScreenerConfig:
    max_results_per_strategy: int = 50
    max_prescore_candidates_per_strategy: int = 500
    max_generated_candidates_per_strategy: int = 5000
    max_full_scenario_candidates_per_strategy: int = 100
    max_concurrent_screens: int = 2
    screen_timeout_seconds: float = 30.0
    result_cache_mb: int = 256
    simulation_paths: int = 5000
    maximum_simulation_paths: int = 25000
    spot_grid_points: int = 81
    maximum_spot_grid_points: int = 161
    maximum_holding_sessions: int = 30
    maximum_dte: int = 365
    iv_residual_half_life_sessions: float = 5.0
    maximum_quote_age_seconds: int = 120
    minimum_open_interest: int = 10
    minimum_volume: int = 0
    maximum_relative_leg_width: float = 0.50
    maximum_relative_spread_width: float = 0.75
    maximum_short_itm_fraction: float = 0.20
    poor_execution_excluded: bool = True
    commission_per_contract: float = 0.0
    modeled_closing_slippage_fraction: float = 0.15
    minimum_iv: float = 1.0e-4
    maximum_iv: float = 10.0
    surface_boundary_probability_limit: float = 0.20
    minimum_target_hits_for_estimate: int = 25
    vertical_target_debit_fraction: float = 0.30
    vertical_target_credit_fraction: float = 0.50
    calendar_diagonal_target_debit_fraction: float = 0.20
    calendar_diagonal_target_credit_fraction: float = 0.50
    pricing_method: str = "tian"
    pricing_workers: int = 0
    pricing_batch_size: int = 4096
    max_pending_pricing_batches: int = 0
    pricing_worker_threads: int = 1

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "ScreenerConfig":
        max_results = int(_value(context, "max_results_per_strategy", 50))
        max_prescore = int(
            _value(context, "max_prescore_candidates_per_strategy", 500)
        )
        default_full_limit = max(max_results, min(100, max_prescore))
        config = cls(
            max_results_per_strategy=max_results,
            max_prescore_candidates_per_strategy=max_prescore,
            max_generated_candidates_per_strategy=int(
                _value(context, "max_generated_candidates_per_strategy", 5000)
            ),
            max_full_scenario_candidates_per_strategy=int(
                _value(
                    context,
                    "max_full_scenario_candidates_per_strategy",
                    default_full_limit,
                )
            ),
            max_concurrent_screens=int(_value(context, "max_concurrent_screens", 2)),
            screen_timeout_seconds=float(_value(context, "screen_timeout_seconds", 30)),
            result_cache_mb=int(_value(context, "result_cache_mb", 256)),
            simulation_paths=int(_value(context, "simulation_paths", 5000)),
            maximum_simulation_paths=int(
                _value(context, "maximum_simulation_paths", 25000)
            ),
            spot_grid_points=int(_value(context, "spot_grid_points", 81)),
            maximum_spot_grid_points=int(
                _value(context, "maximum_spot_grid_points", 161)
            ),
            maximum_holding_sessions=int(
                _value(context, "maximum_holding_sessions", 30)
            ),
            maximum_dte=int(_value(context, "maximum_dte", 365)),
            iv_residual_half_life_sessions=float(
                _value(context, "iv_residual_half_life_sessions", 5)
            ),
            maximum_quote_age_seconds=int(
                _value(context, "maximum_quote_age_seconds", 120)
            ),
            minimum_open_interest=int(_value(context, "minimum_open_interest", 10)),
            minimum_volume=int(_value(context, "minimum_volume", 0)),
            maximum_relative_leg_width=float(
                _value(context, "maximum_relative_leg_width", 0.50)
            ),
            maximum_relative_spread_width=float(
                _value(context, "maximum_relative_spread_width", 0.75)
            ),
            maximum_short_itm_fraction=float(
                _value(context, "maximum_short_itm_fraction", 0.20)
            ),
            poor_execution_excluded=_as_bool(
                _value(context, "poor_execution_excluded", True)
            ),
            commission_per_contract=float(
                _value(context, "commission_per_contract", 0.0)
            ),
            modeled_closing_slippage_fraction=float(
                _value(context, "modeled_closing_slippage_fraction", 0.15)
            ),
            minimum_iv=float(
                context.get("options_calcs.greeks.iv.minimum", 1.0e-4)
            ),
            maximum_iv=float(
                context.get("options_calcs.greeks.iv.maximum", 10.0)
            ),
            surface_boundary_probability_limit=float(
                _value(context, "surface_boundary_probability_limit", 0.20)
            ),
            minimum_target_hits_for_estimate=int(
                _value(context, "minimum_target_hits_for_estimate", 25)
            ),
            vertical_target_debit_fraction=float(
                _value(context, "vertical_target_debit_fraction", 0.30)
            ),
            vertical_target_credit_fraction=float(
                _value(context, "vertical_target_credit_fraction", 0.50)
            ),
            calendar_diagonal_target_debit_fraction=float(
                _value(context, "calendar_diagonal_target_debit_fraction", 0.20)
            ),
            calendar_diagonal_target_credit_fraction=float(
                _value(context, "calendar_diagonal_target_credit_fraction", 0.50)
            ),
            pricing_method=str(_value(context, "pricing_method", "tian"))
            .strip()
            .lower(),
            pricing_workers=int(_value(context, "pricing_workers", 0)),
            pricing_batch_size=int(_value(context, "pricing_batch_size", 4096)),
            max_pending_pricing_batches=int(
                _value(context, "max_pending_pricing_batches", 0)
            ),
            pricing_worker_threads=int(
                _value(context, "pricing_worker_threads", 1)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        integral_bounds = {
            "max_results_per_strategy": (self.max_results_per_strategy, 1, 100),
            "max_prescore_candidates_per_strategy": (
                self.max_prescore_candidates_per_strategy,
                self.max_results_per_strategy,
                5000,
            ),
            "max_generated_candidates_per_strategy": (
                self.max_generated_candidates_per_strategy,
                self.max_prescore_candidates_per_strategy,
                50000,
            ),
            "max_full_scenario_candidates_per_strategy": (
                self.max_full_scenario_candidates_per_strategy,
                self.max_results_per_strategy,
                self.max_prescore_candidates_per_strategy,
            ),
            "max_concurrent_screens": (self.max_concurrent_screens, 1, 32),
            "result_cache_mb": (self.result_cache_mb, 1, 4096),
            "simulation_paths": (self.simulation_paths, 100, 25000),
            "maximum_simulation_paths": (
                self.maximum_simulation_paths,
                self.simulation_paths,
                100000,
            ),
            "spot_grid_points": (self.spot_grid_points, 11, 161),
            "maximum_spot_grid_points": (
                self.maximum_spot_grid_points,
                self.spot_grid_points,
                401,
            ),
            "maximum_holding_sessions": (self.maximum_holding_sessions, 1, 252),
            "maximum_dte": (self.maximum_dte, 1, 730),
            "maximum_quote_age_seconds": (self.maximum_quote_age_seconds, 1, 86400),
            "minimum_open_interest": (self.minimum_open_interest, 0, 10_000_000),
            "minimum_volume": (self.minimum_volume, 0, 10_000_000),
            "minimum_target_hits_for_estimate": (
                self.minimum_target_hits_for_estimate,
                1,
                self.maximum_simulation_paths,
            ),
            "pricing_workers": (self.pricing_workers, 0, 64),
            "pricing_batch_size": (self.pricing_batch_size, 128, 50000),
            "max_pending_pricing_batches": (
                self.max_pending_pricing_batches,
                0,
                512,
            ),
            "pricing_worker_threads": (self.pricing_worker_threads, 1, 8),
        }
        for name, (value, minimum, maximum) in integral_bounds.items():
            if not minimum <= value <= maximum:
                raise ConfigurationError(
                    f"{SCREENER_CONFIG_PREFIX}{name} must be in [{minimum}, {maximum}]"
                )
        if self.spot_grid_points % 2 == 0:
            raise ConfigurationError("spot_grid_points must be odd")
        finite_positive = {
            "screen_timeout_seconds": self.screen_timeout_seconds,
            "iv_residual_half_life_sessions": self.iv_residual_half_life_sessions,
            "minimum_iv": self.minimum_iv,
            "maximum_iv": self.maximum_iv,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{SCREENER_CONFIG_PREFIX}{name} must be positive")
        if self.minimum_iv >= self.maximum_iv:
            raise ConfigurationError("minimum_iv must be below maximum_iv")
        fractions = {
            "maximum_relative_leg_width": self.maximum_relative_leg_width,
            "maximum_relative_spread_width": self.maximum_relative_spread_width,
            "maximum_short_itm_fraction": self.maximum_short_itm_fraction,
            "modeled_closing_slippage_fraction": self.modeled_closing_slippage_fraction,
            "surface_boundary_probability_limit": self.surface_boundary_probability_limit,
            "vertical_target_debit_fraction": self.vertical_target_debit_fraction,
            "vertical_target_credit_fraction": self.vertical_target_credit_fraction,
            "calendar_diagonal_target_debit_fraction": self.calendar_diagonal_target_debit_fraction,
            "calendar_diagonal_target_credit_fraction": self.calendar_diagonal_target_credit_fraction,
        }
        for name, value in fractions.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ConfigurationError(f"{SCREENER_CONFIG_PREFIX}{name} must be in [0, 1]")
        if not math.isfinite(self.commission_per_contract) or self.commission_per_contract < 0:
            raise ConfigurationError("commission_per_contract cannot be negative")
        if self.pricing_method not in {
            "tian",
            "crr",
            "crank_nicolson",
            "configured",
        }:
            raise ConfigurationError(
                f"{SCREENER_CONFIG_PREFIX}pricing_method must be one of "
                "tian, crr, crank_nicolson, configured"
            )

    @property
    def resolved_pricing_workers(self) -> int:
        if self.pricing_workers:
            return self.pricing_workers
        try:
            available = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available = os.cpu_count() or 1
        return min(8, max(1, available - 1 if available > 1 else 1))

    @property
    def resolved_max_pending_pricing_batches(self) -> int:
        if self.max_pending_pricing_batches:
            return self.max_pending_pricing_batches
        return max(1, self.resolved_pricing_workers * 2)


__all__ = ["SCREENER_CONFIG_PREFIX", "ScreenerConfig"]
