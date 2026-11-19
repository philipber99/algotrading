"""Tian third-moment-matching American binomial tree.

The public batch path is vectorized over the contract axis.  Python iteration
is limited to bounded chunks, tree levels, and fixed Greek bump scenarios; the
scalar implementation remains as a numerical reference and isolated fallback.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch

from algotrading.refdata.options.solvers.TreeCommon import aligned_arrays


ALGORITHM_VERSION = "tian_1993_vectorized_v2"
_YEAR_SECONDS = 365.25 * 86400.0


def tian_parameters(
    volatility: float,
    tenor_years: float,
    rate: float,
    dividend_yield: float,
    steps: int,
) -> tuple[float, float, float, float]:
    """Return ``(up, down, probability, discount)`` using stable primitives."""
    if steps <= 0 or tenor_years <= 0 or volatility <= 0:
        raise ValueError("invalid_tian_input")
    dt = tenor_years / steps
    variance_step = volatility * volatility * dt
    v_minus_one = math.expm1(variance_step)
    v = v_minus_one + 1.0
    sqrt_term = math.sqrt(max(0.0, v_minus_one * (v + 3.0)))
    growth = math.exp((rate - dividend_yield) * dt)
    common = 0.5 * growth * v
    up = common * (v + 1.0 + sqrt_term)
    down = common * (v + 1.0 - sqrt_term)
    denominator = up - down
    probability = (growth - down) / denominator if denominator else math.nan
    discount = math.exp(-rate * dt)
    if not (
        math.isfinite(up)
        and math.isfinite(down)
        and up > down > 0.0
        and math.isfinite(probability)
        and 0.0 < probability < 1.0
        and math.isfinite(discount)
    ):
        raise ValueError("invalid_tian_parameters")
    return up, down, probability, discount


def _price_one(
    spot: float,
    strike: float,
    tenor_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    *,
    steps: int,
    primary_greeks: bool = False,
) -> tuple[float, float, float]:
    """Retained scalar reference used only by parity and isolated rollback."""
    values = (spot, strike, tenor_years, rate, dividend_yield, volatility)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite_tian_input")
    if spot <= 0 or strike <= 0 or tenor_years <= 0 or volatility <= 0:
        raise ValueError("invalid_tian_input")
    up, down, probability, discount = tian_parameters(
        volatility, tenor_years, rate, dividend_yield, steps
    )
    j = np.arange(steps + 1, dtype=np.float64)
    stock = spot * np.power(up, j) * np.power(down, steps - j)
    option = (
        np.maximum(stock - strike, 0.0)
        if is_call
        else np.maximum(strike - stock, 0.0)
    )
    level_one: Optional[np.ndarray] = None
    level_two: Optional[np.ndarray] = None
    for level in range(steps - 1, -1, -1):
        option = discount * (
            probability * option[1:] + (1.0 - probability) * option[:-1]
        )
        stock = stock[:-1] / down
        intrinsic = (
            np.maximum(stock - strike, 0.0)
            if is_call
            else np.maximum(strike - stock, 0.0)
        )
        option = np.maximum(option, intrinsic)
        if primary_greeks and level == 2:
            level_two = option.copy()
        elif primary_greeks and level == 1:
            level_one = option.copy()
    price = float(option[0])
    if not primary_greeks:
        return price, math.nan, math.nan
    delta = gamma = math.nan
    if level_one is not None:
        denominator = spot * (up - down)
        if denominator:
            delta = float((level_one[1] - level_one[0]) / denominator)
    if level_two is not None:
        spot_dd = spot * down * down
        spot_ud = spot * up * down
        spot_uu = spot * up * up
        delta_down = (level_two[1] - level_two[0]) / (spot_ud - spot_dd)
        delta_up = (level_two[2] - level_two[1]) / (spot_uu - spot_ud)
        gamma = float((delta_up - delta_down) / (0.5 * (spot_uu - spot_dd)))
    return price, delta, gamma


def _torch_dtype(value: Any) -> torch.dtype:
    if value in (torch.float32, np.float32, np.dtype("float32"), "float32"):
        return torch.float32
    if value in (torch.float64, np.float64, np.dtype("float64"), "float64"):
        return torch.float64
    name = str(value).lower().replace("torch.", "").replace("numpy.", "")
    if name in {"float32", "<class 'numpy.float32'>"}:
        return torch.float32
    if name in {"float64", "<class 'numpy.float64'>"}:
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def _cpu_device(value: Any) -> torch.device:
    device = torch.device(value or "cpu")
    if device.type != "cpu":
        raise ValueError("Tian currently supports CPU execution only")
    return device


def _coerce_inputs(kwargs: dict[str, Any]) -> tuple[list[np.ndarray], np.ndarray]:
    arrays = aligned_arrays(
        kwargs["S"], kwargs["K"], kwargs["T"], kwargs["r"], kwargs["q"], kwargs["sigma"]
    )
    arrays = [np.ascontiguousarray(value) for value in arrays]
    calls = np.ascontiguousarray(np.asarray(kwargs["is_call"], dtype=bool).reshape(-1))
    if arrays and len(calls) != len(arrays[0]):
        raise ValueError("All Tian inputs must have the same length")
    return arrays, calls


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(numerator.dtype).eps * 16.0
    return torch.where(
        torch.abs(denominator) > epsilon,
        numerator / denominator,
        torch.full_like(numerator, float("nan")),
    )


@torch.jit.script
def _tian_backward_scripted(
    spot: torch.Tensor,
    strike: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    probability: torch.Tensor,
    discount: torch.Tensor,
    is_call: torch.Tensor,
    j: torch.Tensor,
    steps: int,
    return_primary_greeks: bool,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiled tree-level loop; the contract dimension remains aligned."""
    stock = (
        spot.unsqueeze(1)
        * torch.pow(up.unsqueeze(1), j)
        * torch.pow(down.unsqueeze(1), steps - j)
    )
    direction = torch.where(
        is_call, torch.ones_like(spot), -torch.ones_like(spot)
    ).unsqueeze(1)
    strike_layer = strike.unsqueeze(1)
    option = torch.clamp(direction * (stock - strike_layer), min=0.0)
    probability_layer = probability.unsqueeze(1)
    discount_layer = discount.unsqueeze(1)
    down_layer = down.unsqueeze(1)
    level_one = torch.full(
        (spot.shape[0], 2), float("nan"), dtype=spot.dtype, device=spot.device
    )
    level_two = torch.full(
        (spot.shape[0], 3), float("nan"), dtype=spot.dtype, device=spot.device
    )
    for level in range(steps - 1, -1, -1):
        option = torch.lerp(option[:, :-1], option[:, 1:], probability_layer)
        option.mul_(discount_layer)
        stock = stock[:, :-1]
        stock.div_(down_layer)
        intrinsic = torch.clamp(direction * (stock - strike_layer), min=0.0)
        option = torch.maximum(option, intrinsic)
        if return_primary_greeks and level == 2:
            level_two = option
        elif return_primary_greeks and level == 1:
            level_one = option
    nan = torch.full_like(spot, float("nan"))
    if not return_primary_greeks:
        return option[:, 0], nan, nan
    delta_denominator = spot * (up - down)
    delta = torch.where(
        torch.abs(delta_denominator) > epsilon,
        (level_one[:, 1] - level_one[:, 0]) / delta_denominator,
        nan,
    )
    spot_dd = spot * down.square()
    spot_ud = spot * up * down
    spot_uu = spot * up.square()
    down_denominator = spot_ud - spot_dd
    up_denominator = spot_uu - spot_ud
    delta_down = torch.where(
        torch.abs(down_denominator) > epsilon,
        (level_two[:, 1] - level_two[:, 0]) / down_denominator,
        nan,
    )
    delta_up = torch.where(
        torch.abs(up_denominator) > epsilon,
        (level_two[:, 2] - level_two[:, 1]) / up_denominator,
        nan,
    )
    gamma_denominator = 0.5 * (spot_uu - spot_dd)
    gamma = torch.where(
        torch.abs(gamma_denominator) > epsilon,
        (delta_up - delta_down) / gamma_denominator,
        nan,
    )
    return option[:, 0], delta, gamma


def _tian_parameters_tensor(
    tenor: torch.Tensor,
    rate: torch.Tensor,
    dividend: torch.Tensor,
    sigma: torch.Tensor,
    *,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vector Tian parameters with safe masked replacements."""
    finite_inputs = (
        torch.isfinite(tenor)
        & torch.isfinite(rate)
        & torch.isfinite(dividend)
        & torch.isfinite(sigma)
        & (tenor > 0.0)
        & (sigma > 0.0)
    )
    safe_tenor = torch.where(finite_inputs, tenor, torch.ones_like(tenor))
    safe_sigma = torch.where(finite_inputs, sigma, torch.full_like(sigma, 0.20))
    safe_rate = torch.where(finite_inputs, rate, torch.zeros_like(rate))
    safe_dividend = torch.where(finite_inputs, dividend, torch.zeros_like(dividend))
    dt = safe_tenor / int(steps)
    variance_step = safe_sigma.square() * dt
    v_minus_one = torch.expm1(variance_step)
    v = v_minus_one + 1.0
    sqrt_term = torch.sqrt(torch.clamp(v_minus_one * (v + 3.0), min=0.0))
    growth = torch.exp((safe_rate - safe_dividend) * dt)
    common = 0.5 * growth * v
    up = common * (v + 1.0 + sqrt_term)
    down = common * (v + 1.0 - sqrt_term)
    denominator = up - down
    probability = denominator.new_full(denominator.shape, 0.5)
    nonzero = torch.abs(denominator) > torch.finfo(denominator.dtype).eps
    probability = torch.where(nonzero, (growth - down) / denominator, probability)
    discount = torch.exp(-safe_rate * dt)
    valid = (
        finite_inputs
        & torch.isfinite(up)
        & torch.isfinite(down)
        & torch.isfinite(probability)
        & torch.isfinite(discount)
        & (up > down)
        & (down > 0.0)
        & (probability > 0.0)
        & (probability < 1.0)
    )
    safe_up = torch.where(valid, up, torch.full_like(up, 1.10))
    safe_down = torch.where(valid, down, torch.full_like(down, 0.90))
    safe_probability = torch.where(valid, probability, torch.full_like(probability, 0.50))
    safe_discount = torch.where(valid, discount, torch.ones_like(discount))
    return safe_up, safe_down, safe_probability, safe_discount, valid


def _tian_price_tensor(
    spot: torch.Tensor,
    strike: torch.Tensor,
    tenor: torch.Tensor,
    rate: torch.Tensor,
    dividend: torch.Tensor,
    sigma: torch.Tensor,
    is_call: torch.Tensor,
    *,
    steps: int,
    j_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor],
    return_primary_greeks: bool = False,
) -> Any:
    """American Tian tree vectorized across all rows in the tensor batch."""
    up, down, probability, discount, parameter_valid = _tian_parameters_tensor(
        tenor, rate, dividend, sigma, steps=steps
    )
    input_valid = (
        parameter_valid
        & torch.isfinite(spot)
        & torch.isfinite(strike)
        & (spot > 0.0)
        & (strike > 0.0)
    )
    safe_spot = torch.where(input_valid, spot, torch.ones_like(spot))
    safe_strike = torch.where(input_valid, strike, torch.ones_like(strike))
    key = (int(steps), safe_spot.device, safe_spot.dtype)
    j = j_cache.get(key)
    if j is None:
        j = torch.arange(steps + 1, device=safe_spot.device, dtype=safe_spot.dtype)
        j_cache[key] = j
    price, delta, gamma = _tian_backward_scripted(
        safe_spot,
        safe_strike,
        up,
        down,
        probability,
        discount,
        is_call,
        j,
        int(steps),
        return_primary_greeks,
        float(torch.finfo(safe_spot.dtype).eps * 16.0),
    )
    nan = torch.full_like(safe_spot, float("nan"))
    price = torch.where(input_valid, price, nan)
    if not return_primary_greeks:
        return price
    return (
        price,
        torch.where(input_valid, delta, nan),
        torch.where(input_valid, gamma, nan),
    )


def _tensor_chunk(
    arrays: list[np.ndarray], calls: np.ndarray, start: int, end: int,
    *, dtype: torch.dtype, device: torch.device,
) -> tuple[torch.Tensor, ...]:
    values = tuple(
        torch.as_tensor(value[start:end], dtype=dtype, device=device) for value in arrays
    )
    return (*values, torch.as_tensor(calls[start:end], dtype=torch.bool, device=device))


def _scalar_price_fallback(
    arrays: list[np.ndarray], calls: np.ndarray, *, steps: int
) -> np.ndarray:
    """Explicit kill-switch/reference route; never used by the vector path."""
    output = np.full(len(calls), np.nan, dtype=np.float64)
    for index, values in enumerate(zip(*arrays)):
        try:
            output[index] = _price_one(
                *(float(value) for value in values),
                bool(calls[index]),
                steps=steps,
            )[0]
        except (ArithmeticError, ValueError, OverflowError, FloatingPointError):
            continue
    return output


def _scalar_price_delta_fallback(
    arrays: list[np.ndarray], calls: np.ndarray, *, steps: int
) -> dict[str, np.ndarray]:
    prices = np.full(len(calls), np.nan, dtype=np.float64)
    deltas = np.full(len(calls), np.nan, dtype=np.float64)
    for index, values in enumerate(zip(*arrays)):
        try:
            price, delta, _gamma = _price_one(
                *(float(value) for value in values),
                bool(calls[index]),
                steps=steps,
                primary_greeks=True,
            )
            prices[index], deltas[index] = price, delta
        except (ArithmeticError, ValueError, OverflowError, FloatingPointError):
            continue
    return {"price": prices, "delta": deltas}


def _scalar_greeks_fallback(
    arrays: list[np.ndarray], calls: np.ndarray, *, steps: int
) -> dict[str, np.ndarray]:
    """The retained pre-vector Tian definitions for an explicit rollback."""
    names = ("model_price", "delta", "gamma", "theta_per_day", "vega", "rho")
    result = {
        name: np.full(len(calls), np.nan, dtype=np.float64) for name in names
    }
    for index, values in enumerate(zip(*arrays)):
        spot, strike, tenor, rate, dividend, volatility = map(float, values)
        try:
            price, delta, gamma = _price_one(
                spot,
                strike,
                tenor,
                rate,
                dividend,
                volatility,
                bool(calls[index]),
                steps=steps,
                primary_greeks=True,
            )
            tenor_bump = min(1.0 / 365.25, tenor * 0.5)
            shorter = _price_one(
                spot,
                strike,
                tenor - tenor_bump,
                rate,
                dividend,
                volatility,
                bool(calls[index]),
                steps=steps,
            )[0]
            sigma_bump = min(0.001, max(0.00025, volatility * 0.0025))
            sigma_down = max(1.0e-8, volatility - sigma_bump)
            price_vol_up = _price_one(
                spot,
                strike,
                tenor,
                rate,
                dividend,
                volatility + sigma_bump,
                bool(calls[index]),
                steps=steps,
            )[0]
            price_vol_down = _price_one(
                spot,
                strike,
                tenor,
                rate,
                dividend,
                sigma_down,
                bool(calls[index]),
                steps=steps,
            )[0]
            rate_bump = 1.0e-4
            price_rate_up = _price_one(
                spot,
                strike,
                tenor,
                rate + rate_bump,
                dividend,
                volatility,
                bool(calls[index]),
                steps=steps,
            )[0]
            price_rate_down = _price_one(
                spot,
                strike,
                tenor,
                rate - rate_bump,
                dividend,
                volatility,
                bool(calls[index]),
                steps=steps,
            )[0]
            result["model_price"][index] = price
            result["delta"][index] = delta
            result["gamma"][index] = gamma
            result["theta_per_day"][index] = (
                (shorter - price) / tenor_bump / 365.25
            )
            result["vega"][index] = (price_vol_up - price_vol_down) / (
                volatility + sigma_bump - sigma_down
            )
            result["rho"][index] = (
                (price_rate_up - price_rate_down) / (2.0 * rate_bump)
            )
        except (ArithmeticError, ValueError, OverflowError, FloatingPointError):
            continue
    return result


def _use_vector_path(kwargs: dict[str, Any]) -> bool:
    if bool(kwargs.get("vectorized_enabled", True)):
        return True
    if not bool(kwargs.get("scalar_fallback_enabled", True)):
        raise RuntimeError(
            "Tian vectorized execution is disabled and scalar fallback is disabled"
        )
    return False


@torch.no_grad()
def price_from_iv_batch(
    *,
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    q: Any,
    is_call: Any,
    sigma: Any,
    steps: int = 256,
    dtype: Any = torch.float64,
    device: Any = "cpu",
    chunk_size: int = 50_000,
    to_numpy: bool = True,
    vectorized_enabled: bool = True,
    scalar_fallback_enabled: bool = True,
    **_ignored: Any,
) -> Any:
    if int(steps) <= 0 or int(chunk_size) <= 0:
        raise ValueError("steps and chunk_size must be positive")
    actual_dtype = _torch_dtype(dtype)
    actual_device = _cpu_device(device)
    arrays, calls = _coerce_inputs(locals())
    if not _use_vector_path(locals()):
        output = _scalar_price_fallback(arrays, calls, steps=int(steps))
        return output if to_numpy else torch.from_numpy(output)
    output = np.full(
        len(calls), np.nan, dtype=np.float32 if actual_dtype is torch.float32 else np.float64
    )
    j_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
    for start in range(0, len(calls), int(chunk_size)):
        end = min(len(calls), start + int(chunk_size))
        spot, strike, tenor, rate, dividend, volatility, call = _tensor_chunk(
            arrays, calls, start, end, dtype=actual_dtype, device=actual_device
        )
        output[start:end] = _tian_price_tensor(
            spot, strike, tenor, rate, dividend, volatility, call,
            steps=int(steps), j_cache=j_cache,
        ).cpu().numpy()
    return output if to_numpy else torch.from_numpy(output)


@torch.no_grad()
def price_delta_from_iv_batch(**kwargs: Any) -> dict[str, Any]:
    steps = int(kwargs.get("steps", 256))
    chunk_size = int(kwargs.get("chunk_size", 50_000))
    if steps <= 0 or chunk_size <= 0:
        raise ValueError("steps and chunk_size must be positive")
    actual_dtype = _torch_dtype(kwargs.get("dtype", torch.float64))
    actual_device = _cpu_device(kwargs.get("device", "cpu"))
    arrays, calls = _coerce_inputs(kwargs)
    if not _use_vector_path(kwargs):
        result = _scalar_price_delta_fallback(arrays, calls, steps=steps)
        if not bool(kwargs.get("to_numpy", True)):
            return {name: torch.from_numpy(value) for name, value in result.items()}
        return result
    numpy_type = np.float32 if actual_dtype is torch.float32 else np.float64
    prices = np.full(len(calls), np.nan, dtype=numpy_type)
    deltas = np.full(len(calls), np.nan, dtype=numpy_type)
    j_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
    for start in range(0, len(calls), chunk_size):
        end = min(len(calls), start + chunk_size)
        spot, strike, tenor, rate, dividend, volatility, call = _tensor_chunk(
            arrays, calls, start, end, dtype=actual_dtype, device=actual_device
        )
        price, delta, _gamma = _tian_price_tensor(
            spot, strike, tenor, rate, dividend, volatility, call,
            steps=steps, j_cache=j_cache, return_primary_greeks=True,
        )
        prices[start:end] = price.cpu().numpy()
        deltas[start:end] = delta.cpu().numpy()
    result: dict[str, Any] = {"price": prices, "delta": deltas}
    if not bool(kwargs.get("to_numpy", True)):
        return {name: torch.from_numpy(value) for name, value in result.items()}
    return result


@torch.no_grad()
def greeks_from_iv_batch(**kwargs: Any) -> dict[str, Any]:
    steps = int(kwargs.get("steps", 256))
    chunk_size = int(kwargs.get("chunk_size", 50_000))
    if steps <= 0 or chunk_size <= 0:
        raise ValueError("steps and chunk_size must be positive")
    actual_dtype = _torch_dtype(kwargs.get("dtype", torch.float64))
    actual_device = _cpu_device(kwargs.get("device", "cpu"))
    arrays, calls = _coerce_inputs(kwargs)
    fusion_maximum = int(kwargs.get("fused_greeks_maximum_contracts", 0))
    fused_greeks_enabled = (
        bool(kwargs.get("fused_greeks_enabled", True))
        and chunk_size >= 6
        and (fusion_maximum <= 0 or len(calls) <= fusion_maximum)
    )
    if not _use_vector_path(kwargs):
        result = _scalar_greeks_fallback(arrays, calls, steps=steps)
        if not bool(kwargs.get("to_numpy", True)):
            return {name: torch.from_numpy(value) for name, value in result.items()}
        return result
    numpy_type = np.float32 if actual_dtype is torch.float32 else np.float64
    names = ("model_price", "delta", "gamma", "theta_per_day", "vega", "rho")
    result: dict[str, Any] = {
        name: np.full(len(calls), np.nan, dtype=numpy_type) for name in names
    }
    j_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
    # ``chunk_size`` remains a physical-row memory budget.  A fused Greek
    # tree contains six scenario rows per logical contract.
    logical_chunk_size = (
        max(1, chunk_size // 6) if fused_greeks_enabled else chunk_size
    )
    for start in range(0, len(calls), logical_chunk_size):
        end = min(len(calls), start + logical_chunk_size)
        spot, strike, tenor, rate, dividend, volatility, call = _tensor_chunk(
            arrays, calls, start, end, dtype=actual_dtype, device=actual_device
        )
        tenor_bump = torch.minimum(
            torch.full_like(tenor, 1.0 / 365.25), tenor * 0.5
        )
        sigma_bump = torch.minimum(
            torch.full_like(volatility, 0.001),
            torch.maximum(torch.full_like(volatility, 0.00025), volatility * 0.0025),
        )
        sigma_down = torch.clamp(volatility - sigma_bump, min=1.0e-8)
        rate_bump = torch.full_like(rate, 1.0e-4)
        if fused_greeks_enabled:
            expanded_price, expanded_delta, expanded_gamma = _tian_price_tensor(
                spot.repeat(6),
                strike.repeat(6),
                torch.cat(
                    (
                        tenor,
                        tenor - tenor_bump,
                        tenor,
                        tenor,
                        tenor,
                        tenor,
                    )
                ),
                torch.cat(
                    (
                        rate,
                        rate,
                        rate,
                        rate,
                        rate + rate_bump,
                        rate - rate_bump,
                    )
                ),
                dividend.repeat(6),
                torch.cat(
                    (
                        volatility,
                        volatility,
                        volatility + sigma_bump,
                        sigma_down,
                        volatility,
                        volatility,
                    )
                ),
                call.repeat(6),
                steps=steps,
                j_cache=j_cache,
                return_primary_greeks=True,
            )
            scenario_prices = expanded_price.reshape(6, -1)
            price = scenario_prices[0]
            shorter = scenario_prices[1]
            price_vol_up = scenario_prices[2]
            price_vol_down = scenario_prices[3]
            price_rate_up = scenario_prices[4]
            price_rate_down = scenario_prices[5]
            delta = expanded_delta[: end - start]
            gamma = expanded_gamma[: end - start]
        else:
            price, delta, gamma = _tian_price_tensor(
                spot, strike, tenor, rate, dividend, volatility, call,
                steps=steps, j_cache=j_cache, return_primary_greeks=True,
            )
            shorter = _tian_price_tensor(
                spot, strike, tenor - tenor_bump, rate, dividend, volatility, call,
                steps=steps, j_cache=j_cache,
            )
            price_vol_up = _tian_price_tensor(
                spot, strike, tenor, rate, dividend, volatility + sigma_bump, call,
                steps=steps, j_cache=j_cache,
            )
            price_vol_down = _tian_price_tensor(
                spot, strike, tenor, rate, dividend, sigma_down, call,
                steps=steps, j_cache=j_cache,
            )
            price_rate_up = _tian_price_tensor(
                spot, strike, tenor, rate + rate_bump, dividend, volatility, call,
                steps=steps, j_cache=j_cache,
            )
            price_rate_down = _tian_price_tensor(
                spot, strike, tenor, rate - rate_bump, dividend, volatility, call,
                steps=steps, j_cache=j_cache,
            )
        chunk_values = {
            "model_price": price,
            "delta": delta,
            "gamma": gamma,
            "theta_per_day": (shorter - price) / tenor_bump / 365.25,
            "vega": (price_vol_up - price_vol_down)
            / (volatility + sigma_bump - sigma_down),
            "rho": (price_rate_up - price_rate_down) / (2.0 * rate_bump),
        }
        for name, value in chunk_values.items():
            result[name][start:end] = value.cpu().numpy()
    if not bool(kwargs.get("to_numpy", True)):
        return {name: torch.from_numpy(value) for name, value in result.items()}
    return result
