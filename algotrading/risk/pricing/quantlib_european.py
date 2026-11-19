"""QuantLib analytic Black-Scholes-Merton fallback for European options only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from threading import RLock


_QUANTLIB_LOCK = RLock()


class EuropeanPricingUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class EuropeanValuation:
    price: float
    implied_volatility: float
    delta: float
    gamma: float
    theta_per_day: float
    vega_per_1pct: float
    rho_per_1pct: float
    algorithm_version: str = "quantlib_analytic_european_engine_v1"


def solve_european_iv_and_greeks(
    *,
    put_call: str,
    spot: float,
    strike: float,
    valuation_at: datetime,
    expiration_at: datetime,
    rate: float,
    dividend_yield: float,
    target_price: float,
    minimum_iv: float = 0.0001,
    maximum_iv: float = 5.0,
    accuracy: float = 1e-8,
    maximum_evaluations: int = 200,
) -> EuropeanValuation:
    # Serialize QuantLib valuations because the evaluation date is shared state.
    with _QUANTLIB_LOCK:
        ql, option, process, volatility_quote = _build_option(
            put_call=put_call,
            spot=spot,
            strike=strike,
            valuation_at=valuation_at,
            expiration_at=expiration_at,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=0.20,
        )
        try:
            implied = float(
                option.impliedVolatility(
                    float(target_price),
                    process,
                    float(accuracy),
                    int(maximum_evaluations),
                    float(minimum_iv),
                    float(maximum_iv),
                )
            )
            volatility_quote.setValue(implied)
            return _valuation(option, implied)
        except Exception as exc:
            raise EuropeanPricingUnavailable(
                "QuantLib could not invert the European option mark"
            ) from exc


def price_european(
    *,
    put_call: str,
    spot: float,
    strike: float,
    valuation_at: datetime,
    expiration_at: datetime,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> EuropeanValuation:
    with _QUANTLIB_LOCK:
        _ql, option, _process, _quote = _build_option(
            put_call=put_call,
            spot=spot,
            strike=strike,
            valuation_at=valuation_at,
            expiration_at=expiration_at,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
        )
        try:
            return _valuation(option, float(volatility))
        except Exception as exc:
            raise EuropeanPricingUnavailable(
                "QuantLib could not price the European option"
            ) from exc


def _build_option(
    *,
    put_call: str,
    spot: float,
    strike: float,
    valuation_at: datetime,
    expiration_at: datetime,
    rate: float,
    dividend_yield: float,
    volatility: float,
):
    values = (spot, strike, rate, dividend_yield, volatility)
    if not all(math.isfinite(float(value)) for value in values):
        raise EuropeanPricingUnavailable("European pricing inputs must be finite")
    if spot <= 0 or strike <= 0 or volatility <= 0:
        raise EuropeanPricingUnavailable("European spot, strike, and volatility must be positive")
    valuation = _aware_utc(valuation_at)
    expiration = _aware_utc(expiration_at)
    if expiration <= valuation or expiration.date() <= valuation.date():
        raise EuropeanPricingUnavailable("European option tenor must be positive")
    try:
        import QuantLib as ql
    except ImportError as exc:
        raise EuropeanPricingUnavailable("QuantLib is not installed") from exc
    evaluation_date = ql.Date(valuation.day, valuation.month, valuation.year)
    maturity_date = ql.Date(expiration.day, expiration.month, expiration.year)
    with _QUANTLIB_LOCK:
        # Set a common valuation date for the process and flat yield curves.
        ql.Settings.instance().evaluationDate = evaluation_date
        day_count = ql.Actual365Fixed()
        calendar = ql.NullCalendar()
        spot_quote = ql.SimpleQuote(float(spot))
        volatility_quote = ql.SimpleQuote(float(volatility))
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(spot_quote),
            ql.YieldTermStructureHandle(
                ql.FlatForward(evaluation_date, float(dividend_yield), day_count)
            ),
            ql.YieldTermStructureHandle(
                ql.FlatForward(evaluation_date, float(rate), day_count)
            ),
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(
                    evaluation_date,
                    calendar,
                    ql.QuoteHandle(volatility_quote),
                    day_count,
                )
            ),
        )
        option_type = (
            ql.Option.Call if str(put_call).upper() == "CALL" else ql.Option.Put
        )
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(option_type, float(strike)),
            ql.EuropeanExercise(maturity_date),
        )
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return ql, option, process, volatility_quote


def _valuation(option, implied_volatility: float) -> EuropeanValuation:
    # Convert annual theta and unit-rate sensitivities to the public daily/1% units.
    values = {
        "price": float(option.NPV()),
        "delta": float(option.delta()),
        "gamma": float(option.gamma()),
        "theta_per_day": float(option.thetaPerDay()),
        "vega_per_1pct": float(option.vega()) / 100.0,
        "rho_per_1pct": float(option.rho()) / 100.0,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise EuropeanPricingUnavailable("QuantLib returned a nonfinite European valuation")
    return EuropeanValuation(
        implied_volatility=float(implied_volatility),
        **values,
    )


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise EuropeanPricingUnavailable("European valuation timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "EuropeanPricingUnavailable",
    "EuropeanValuation",
    "price_european",
    "solve_european_iv_and_greeks",
]
