"""Lazy QuantLib Crank–Nicolson adapter with Rannacher damping."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np


ALGORITHM_VERSION = "quantlib_fdm_crank_nicolson_v2_price_session"
MINIMUM_TENOR_SECONDS = 86400.0


def load_quantlib() -> Any:
    try:
        return importlib.import_module("QuantLib")
    except ImportError as exc:
        raise RuntimeError(
            "QuantLib is required for the configured crank_nicolson repricing method"
        ) from exc


def quantlib_version() -> str:
    return str(getattr(load_quantlib(), "__version__", "unknown"))


def _as_utc_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _ql_date(ql: Any, value: datetime) -> Any:
    return ql.Date(value.day, value.month, value.year)


@dataclass(frozen=True)
class FdmResult:
    model_price: float
    delta: float
    gamma: float
    theta_per_day: float
    vega_per_1pct: float
    rho_per_1pct: float
    vega_bump: float
    rho_bump: float


def _dates(
    tenor_seconds: float,
    source_collected_at: Any,
    expiration_at: Any,
) -> tuple[datetime, datetime]:
    if not math.isfinite(float(tenor_seconds)) or tenor_seconds < MINIMUM_TENOR_SECONDS:
        raise ValueError("crank_nicolson_requires_full_day_tenor")
    source = _as_utc_datetime(source_collected_at)
    expiration = _as_utc_datetime(expiration_at)
    if source is None:
        source = datetime(2020, 1, 2, tzinfo=timezone.utc)
    if expiration is None:
        days = max(1, int(round(float(tenor_seconds) / 86400.0)))
        expiration = source + timedelta(days=days)
    if expiration.date() <= source.date():
        raise ValueError("crank_nicolson_requires_distinct_maturity_date")
    return source, expiration


def tenor_diagnostics(
    tenor_seconds: float,
    source_collected_at: Any,
    expiration_at: Any,
) -> dict[str, float | str | bool]:
    """Expose the bounded date-only maturity distortion without changing inputs."""
    requested = float(tenor_seconds)
    source, expiration = _dates(
        requested, source_collected_at, expiration_at
    )
    calendar_seconds = float((expiration.date() - source.date()).days * 86400)
    absolute = abs(calendar_seconds - requested)
    relative = absolute / requested if requested > 0.0 else math.inf
    warning = absolute > 1.0
    return {
        "requested_tenor_seconds": requested,
        "calendar_day_tenor_seconds": calendar_seconds,
        "absolute_tenor_truncation_seconds": absolute,
        "relative_tenor_truncation_fraction": relative,
        "tenor_quality_warning": warning,
        "tenor_quality_reason": (
            "crank_nicolson_date_only_tenor_mismatch" if warning else "none"
        ),
    }


class CrankNicolsonPricingSession:
    """One contract's reusable QuantLib state for IV probes and final Greeks."""

    def __init__(
        self,
        *,
        spot: float,
        strike: float,
        tenor_seconds: float,
        rate: float,
        dividend_yield: float,
        is_call: bool,
        initial_volatility: float = 0.20,
        source_collected_at: Any = None,
        expiration_at: Any = None,
        time_grid: int = 100,
        space_grid: int = 200,
        damping_steps: int = 4,
        vega_relative_bump: float = 0.0025,
        vega_minimum_bump: float = 0.00025,
        vega_maximum_bump: float = 0.001,
        rho_bump: float = 0.0001,
    ) -> None:
        values = (
            spot,
            strike,
            tenor_seconds,
            rate,
            dividend_yield,
            initial_volatility,
            vega_relative_bump,
            vega_minimum_bump,
            vega_maximum_bump,
            rho_bump,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("nonfinite_crank_nicolson_input")
        if spot <= 0 or strike <= 0 or initial_volatility <= 0:
            raise ValueError("invalid_crank_nicolson_input")
        if int(time_grid) <= 0 or int(space_grid) <= 0 or int(damping_steps) <= 0:
            raise ValueError("Crank-Nicolson grids and damping steps must be positive")
        if (
            vega_relative_bump <= 0
            or vega_minimum_bump <= 0
            or vega_maximum_bump < vega_minimum_bump
            or rho_bump <= 0
        ):
            raise ValueError("Crank-Nicolson Greek bumps must be positive and ordered")
        source, expiration = _dates(
            float(tenor_seconds), source_collected_at, expiration_at
        )
        self.tenor_diagnostics = tenor_diagnostics(
            float(tenor_seconds), source_collected_at, expiration_at
        )
        ql = load_quantlib()
        evaluation_date = _ql_date(ql, source)
        maturity_date = _ql_date(ql, expiration)
        ql.Settings.instance().evaluationDate = evaluation_date
        day_count = ql.Actual365Fixed()
        calendar = ql.NullCalendar()
        spot_quote = ql.SimpleQuote(float(spot))
        self._base_rate = float(rate)
        self._rate_quote = ql.SimpleQuote(self._base_rate)
        dividend_quote = ql.SimpleQuote(float(dividend_yield))
        self._volatility_quote = ql.SimpleQuote(float(initial_volatility))
        risk_free_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(
                evaluation_date, ql.QuoteHandle(self._rate_quote), day_count
            )
        )
        dividend_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(
                evaluation_date, ql.QuoteHandle(dividend_quote), day_count
            )
        )
        volatility_curve = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                evaluation_date,
                calendar,
                ql.QuoteHandle(self._volatility_quote),
                day_count,
            )
        )
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(spot_quote),
            dividend_curve,
            risk_free_curve,
            volatility_curve,
        )
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if is_call else ql.Option.Put, float(strike)
        )
        exercise = ql.AmericanExercise(evaluation_date, maturity_date)
        self._option = ql.VanillaOption(payoff, exercise)
        engine = ql.FdBlackScholesVanillaEngine(
            process,
            int(time_grid),
            int(space_grid),
            int(damping_steps),
            ql.FdmSchemeDesc.CrankNicolson(),
        )
        self._option.setPricingEngine(engine)
        self._vega_relative_bump = float(vega_relative_bump)
        self._vega_minimum_bump = float(vega_minimum_bump)
        self._vega_maximum_bump = float(vega_maximum_bump)
        self._rho_bump = float(rho_bump)
        self._fdm_valuations = 0

    @property
    def fdm_valuations(self) -> int:
        """Number of QuantLib FDM valuations performed by this session."""
        return self._fdm_valuations

    def _npv(self) -> float:
        self._fdm_valuations += 1
        return float(self._option.NPV())

    @staticmethod
    def _validated_volatility(volatility: float) -> float:
        parsed = float(volatility)
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError("invalid_crank_nicolson_volatility")
        return parsed

    def price(self, volatility: float) -> float:
        """Run exactly one unbumped FDM valuation for an IV objective probe."""
        parsed = self._validated_volatility(volatility)
        self._volatility_quote.setValue(parsed)
        self._rate_quote.setValue(self._base_rate)
        price = self._npv()
        if not math.isfinite(price):
            raise FloatingPointError("nonfinite_crank_nicolson_price")
        return price

    def price_delta(self, volatility: float) -> tuple[float, float]:
        """Run one unbumped FDM valuation and return price plus direct delta."""
        parsed = self._validated_volatility(volatility)
        try:
            self._volatility_quote.setValue(parsed)
            self._rate_quote.setValue(self._base_rate)
            price = self._npv()
            delta = float(self._option.delta())
            if not math.isfinite(price) or not math.isfinite(delta):
                raise FloatingPointError("nonfinite_crank_nicolson_price_delta")
            return price, delta
        finally:
            # A failed QuantLib request must not poison a reused session.
            self._volatility_quote.setValue(parsed)
            self._rate_quote.setValue(self._base_rate)

    def delta(self, volatility: float) -> float:
        """Run one unbumped FDM valuation and return only direct delta."""
        return self.price_delta(volatility)[1]

    def greeks(self, volatility: float) -> FdmResult:
        """Calculate the final tuple once, reusing this contract's FDM engine."""
        parsed = self._validated_volatility(volatility)
        model_price = self.price(parsed)
        delta = float(self._option.delta())
        gamma = float(self._option.gamma())
        theta_per_day = float(self._option.theta()) / 365.0
        sigma_bump = min(
            self._vega_maximum_bump,
            max(self._vega_minimum_bump, abs(parsed) * self._vega_relative_bump),
        )
        sigma_down = max(1.0e-8, parsed - sigma_bump)
        try:
            self._volatility_quote.setValue(parsed + sigma_bump)
            price_vol_up = self._npv()
            self._volatility_quote.setValue(sigma_down)
            price_vol_down = self._npv()
            self._volatility_quote.setValue(parsed)
            vega_per_1pct = (
                (price_vol_up - price_vol_down)
                / (parsed + sigma_bump - sigma_down)
                * 0.01
            )

            self._rate_quote.setValue(self._base_rate + self._rho_bump)
            price_rate_up = self._npv()
            self._rate_quote.setValue(self._base_rate - self._rho_bump)
            price_rate_down = self._npv()
            rho_per_1pct = (
                (price_rate_up - price_rate_down) / (2.0 * self._rho_bump) * 0.01
            )
        finally:
            self._volatility_quote.setValue(parsed)
            self._rate_quote.setValue(self._base_rate)
        output = FdmResult(
            model_price,
            delta,
            gamma,
            theta_per_day,
            vega_per_1pct,
            rho_per_1pct,
            sigma_bump,
            self._rho_bump,
        )
        if not all(math.isfinite(value) for value in output.__dict__.values()):
            raise FloatingPointError("nonfinite_crank_nicolson_output")
        return output


def create_pricing_session(
    *,
    spot: float,
    strike: float,
    tenor_seconds: float,
    rate: float,
    dividend_yield: float,
    is_call: bool,
    initial_volatility: float = 0.20,
    source_collected_at: Any = None,
    expiration_at: Any = None,
    **kwargs: Any,
) -> CrankNicolsonPricingSession:
    return CrankNicolsonPricingSession(
        spot=spot,
        strike=strike,
        tenor_seconds=tenor_seconds,
        rate=rate,
        dividend_yield=dividend_yield,
        is_call=is_call,
        initial_volatility=initial_volatility,
        source_collected_at=source_collected_at,
        expiration_at=expiration_at,
        **kwargs,
    )


def greeks_from_iv(
    *,
    spot: float,
    strike: float,
    tenor_seconds: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    source_collected_at: Any = None,
    expiration_at: Any = None,
    **kwargs: Any,
) -> FdmResult:
    values = (spot, strike, tenor_seconds, rate, dividend_yield, volatility)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("nonfinite_crank_nicolson_input")
    session = create_pricing_session(
        spot=spot,
        strike=strike,
        tenor_seconds=tenor_seconds,
        rate=rate,
        dividend_yield=dividend_yield,
        is_call=is_call,
        initial_volatility=volatility,
        source_collected_at=source_collected_at,
        expiration_at=expiration_at,
        **kwargs,
    )
    return session.greeks(volatility)


def price_from_iv(**kwargs: Any) -> float:
    values = dict(kwargs)
    volatility = float(values.pop("volatility"))
    session = create_pricing_session(
        initial_volatility=volatility,
        **values,
    )
    return session.price(volatility)


def greeks_from_iv_batch(
    *,
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    q: Any,
    is_call: Any,
    sigma: Any,
    source_collected_at: Any = None,
    expiration_at: Any = None,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    arrays = [np.asarray(value).reshape(-1) for value in (S, K, T, r, q, is_call, sigma)]
    count = len(arrays[0])
    if any(len(value) != count for value in arrays[1:]):
        raise ValueError("All Crank-Nicolson inputs must have the same length")
    sources = [None] * count if source_collected_at is None else list(np.asarray(source_collected_at, dtype=object).reshape(-1))
    expirations = [None] * count if expiration_at is None else list(np.asarray(expiration_at, dtype=object).reshape(-1))
    result = {
        name: np.full(count, np.nan, dtype=np.float64)
        for name in (
            "model_price", "delta", "gamma", "theta_per_day",
            "vega_per_1pct", "rho_per_1pct"
        )
    }
    for index in range(count):
        try:
            solved = greeks_from_iv(
                spot=float(arrays[0][index]),
                strike=float(arrays[1][index]),
                tenor_seconds=float(arrays[2][index]) * 365.25 * 86400.0,
                rate=float(arrays[3][index]),
                dividend_yield=float(arrays[4][index]),
                is_call=bool(arrays[5][index]),
                volatility=float(arrays[6][index]),
                source_collected_at=sources[index],
                expiration_at=expirations[index],
                **kwargs,
            )
            for name in result:
                result[name][index] = getattr(solved, name)
        except (ArithmeticError, ValueError, RuntimeError, FloatingPointError):
            continue
    return result


def price_from_iv_batch(
    *,
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    q: Any,
    is_call: Any,
    sigma: Any,
    source_collected_at: Any = None,
    expiration_at: Any = None,
    **kwargs: Any,
) -> np.ndarray:
    arrays = [
        np.asarray(value).reshape(-1)
        for value in (S, K, T, r, q, is_call, sigma)
    ]
    count = len(arrays[0])
    if any(len(value) != count for value in arrays[1:]):
        raise ValueError("All Crank-Nicolson inputs must have the same length")
    sources = (
        [None] * count
        if source_collected_at is None
        else list(np.asarray(source_collected_at, dtype=object).reshape(-1))
    )
    expirations = (
        [None] * count
        if expiration_at is None
        else list(np.asarray(expiration_at, dtype=object).reshape(-1))
    )
    result = np.full(count, np.nan, dtype=np.float64)
    for index in range(count):
        try:
            result[index] = price_from_iv(
                spot=float(arrays[0][index]),
                strike=float(arrays[1][index]),
                tenor_seconds=float(arrays[2][index]) * 365.25 * 86400.0,
                rate=float(arrays[3][index]),
                dividend_yield=float(arrays[4][index]),
                is_call=bool(arrays[5][index]),
                volatility=float(arrays[6][index]),
                source_collected_at=sources[index],
                expiration_at=expirations[index],
                **kwargs,
            )
        except (ArithmeticError, ValueError, RuntimeError, FloatingPointError):
            continue
    return result


def price_delta_from_iv_batch(
    *,
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    q: Any,
    is_call: Any,
    sigma: Any,
    source_collected_at: Any = None,
    expiration_at: Any = None,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    arrays = [
        np.asarray(value).reshape(-1)
        for value in (S, K, T, r, q, is_call, sigma)
    ]
    count = len(arrays[0])
    if any(len(value) != count for value in arrays[1:]):
        raise ValueError("All Crank-Nicolson inputs must have the same length")
    sources = (
        [None] * count
        if source_collected_at is None
        else list(np.asarray(source_collected_at, dtype=object).reshape(-1))
    )
    expirations = (
        [None] * count
        if expiration_at is None
        else list(np.asarray(expiration_at, dtype=object).reshape(-1))
    )
    prices = np.full(count, np.nan, dtype=np.float64)
    deltas = np.full(count, np.nan, dtype=np.float64)
    for index in range(count):
        try:
            volatility = float(arrays[6][index])
            session = create_pricing_session(
                spot=float(arrays[0][index]),
                strike=float(arrays[1][index]),
                tenor_seconds=float(arrays[2][index]) * 365.25 * 86400.0,
                rate=float(arrays[3][index]),
                dividend_yield=float(arrays[4][index]),
                is_call=bool(arrays[5][index]),
                initial_volatility=volatility,
                source_collected_at=sources[index],
                expiration_at=expirations[index],
                **kwargs,
            )
            # One valuation populates both NPV and direct delta in QuantLib.
            prices[index], deltas[index] = session.price_delta(volatility)
        except (ArithmeticError, ValueError, RuntimeError, FloatingPointError):
            continue
    return {"price": prices, "delta": deltas}


def solve_iv_batch(*, price_obs: Any, **kwargs: Any) -> np.ndarray:
    from algotrading.refdata.options.solvers.IvSolver import solve_iv

    targets = np.asarray(price_obs, dtype=np.float64).reshape(-1)
    sigma_hints = np.asarray(kwargs.pop("sigma"), dtype=np.float64).reshape(-1)
    count = len(targets)
    source_values = kwargs.pop("source_collected_at", None)
    expiration_values = kwargs.pop("expiration_at", None)
    sources = (
        [None] * count
        if source_values is None
        else list(np.asarray(source_values, dtype=object).reshape(-1))
    )
    expirations = (
        [None] * count
        if expiration_values is None
        else list(np.asarray(expiration_values, dtype=object).reshape(-1))
    )
    arrays = {
        name: np.asarray(kwargs.pop(name)).reshape(-1)
        for name in ("S", "K", "T", "r", "q", "is_call")
    }
    output = np.full(count, np.nan, dtype=np.float64)
    for index in range(count):
        hint = float(sigma_hints[index])
        session = create_pricing_session(
            spot=float(arrays["S"][index]),
            strike=float(arrays["K"][index]),
            tenor_seconds=float(arrays["T"][index]) * 365.25 * 86400.0,
            rate=float(arrays["r"][index]),
            dividend_yield=float(arrays["q"][index]),
            is_call=bool(arrays["is_call"][index]),
            initial_volatility=(
                hint if math.isfinite(hint) and hint > 0.0 else 0.20
            ),
            source_collected_at=sources[index],
            expiration_at=expirations[index],
            **kwargs,
        )
        solved = solve_iv(
            target_price=targets[index],
            source_iv=hint,
            price_function=session.price,
        )
        if solved.status == "complete":
            output[index] = solved.iv
    return output


def solve_iv_and_greeks_batch(*, price_obs: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    solved = solve_iv_batch(price_obs=price_obs, **dict(kwargs))
    greek_kwargs = dict(kwargs)
    greek_kwargs["sigma"] = solved
    result = greeks_from_iv_batch(**greek_kwargs)
    result["iv"] = solved
    return result
