"""Forward-parameterised Black-76 Greeks.

Every Greek here is an analytic derivative of
:func:`optlab.core.black.black76_price`. They are validated in the test suite
against central finite differences of the pricer itself, which is the only
check that reliably catches a wrong constant or a dropped discount factor.

Conventions
-----------
``vega`` and ``rho`` are per unit of volatility / rate (per ``1.00``, not per
percentage point) and ``theta`` is per year. The two helpers at the bottom
convert to the units traders actually quote.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from optlab.core.black import black76_price, d1_d2, option_sign

__all__ = [
    "delta",
    "dual_delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
    "greek_table",
    "per_percentage_point",
    "per_calendar_day",
]


def _prepare(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    return (
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(volatility, dtype=float),
        np.asarray(time_to_expiry, dtype=float),
        np.asarray(discount, dtype=float),
    )


def delta(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
) -> NDArray[np.float64]:
    """Sensitivity to the forward, ``dV/dF = w D N(w d1)``.

    This is *forward* delta. Spot delta differs by the factor ``dF/dS``;
    which one you want depends on whether you hedge with the underlying or
    with a future, and confusing the two is a classic desk error.
    """
    forward, strike, volatility, time_to_expiry, discount = _prepare(
        forward, strike, volatility, time_to_expiry, discount
    )
    sign = option_sign(option_type)
    d1, _ = d1_d2(forward, strike, volatility, time_to_expiry)
    return discount * sign * norm.cdf(sign * d1)


def dual_delta(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
) -> NDArray[np.float64]:
    """Sensitivity to the strike, ``dV/dK = -w D N(w d2)``.

    ``-(dC/dK)/D`` is the risk-neutral probability of expiring in the money,
    which is why this Greek appears whenever probabilities are read off a
    chain.
    """
    forward, strike, volatility, time_to_expiry, discount = _prepare(
        forward, strike, volatility, time_to_expiry, discount
    )
    sign = option_sign(option_type)
    _, d2 = d1_d2(forward, strike, volatility, time_to_expiry)
    return -discount * sign * norm.cdf(sign * d2)


def gamma(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
) -> NDArray[np.float64]:
    """``d2V/dF2``. Identical for calls and puts, since parity is linear in ``F``."""
    forward, strike, volatility, time_to_expiry, discount = _prepare(
        forward, strike, volatility, time_to_expiry, discount
    )
    option_sign(option_type)  # validate only
    d1, _ = d1_d2(forward, strike, volatility, time_to_expiry)
    total_vol = volatility * np.sqrt(np.maximum(time_to_expiry, 0.0))
    safe = np.where(total_vol > 0.0, total_vol, 1.0)
    out = discount * norm.pdf(d1) / (forward * safe)
    return np.where(total_vol > 0.0, out, 0.0)


def vega(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
) -> NDArray[np.float64]:
    """``dV/dsigma``. Identical for calls and puts, and never negative.

    Vega is the natural weight for any fit to market quotes: matching prices
    equally weights a $0.02 wing option against a $25 straddle, whereas
    dividing by vega expresses the error in volatility points, which is the
    unit a trader actually reasons in.
    """
    forward, strike, volatility, time_to_expiry, discount = _prepare(
        forward, strike, volatility, time_to_expiry, discount
    )
    option_sign(option_type)  # validate only
    d1, _ = d1_d2(forward, strike, volatility, time_to_expiry)
    return discount * forward * norm.pdf(d1) * np.sqrt(np.maximum(time_to_expiry, 0.0))


def theta(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
    rate: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """``dV/dt`` per year, holding the forward fixed.

    ``rate`` enters only through the discount factor rolling up. Pass the
    implied rate from :mod:`optlab.market.forward` for a carry-aware theta,
    or leave it at zero for pure time decay.
    """
    forward, strike, volatility, time_to_expiry, discount = _prepare(
        forward, strike, volatility, time_to_expiry, discount
    )
    rate = np.asarray(rate, dtype=float)
    sign = option_sign(option_type)
    d1, d2 = d1_d2(forward, strike, volatility, time_to_expiry)

    sqrt_t = np.sqrt(np.maximum(time_to_expiry, 0.0))
    safe_t = np.where(sqrt_t > 0.0, sqrt_t, 1.0)
    decay = np.where(
        sqrt_t > 0.0,
        -discount * forward * norm.pdf(d1) * volatility / (2.0 * safe_t),
        0.0,
    )
    carry = rate * discount * sign * (
        forward * norm.cdf(sign * d1) - strike * norm.cdf(sign * d2)
    )
    return decay + carry


def rho(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
) -> NDArray[np.float64]:
    """``dV/dr`` holding the *forward* fixed: pure discounting sensitivity.

    Under this parameterisation the rate no longer moves the forward, so rho
    collapses to ``-T V``. That is deliberate: it separates "the discount
    curve moved" from "the forward moved", two effects the spot
    parameterisation tangles together.
    """
    forward, strike, volatility, time_to_expiry, discount = _prepare(
        forward, strike, volatility, time_to_expiry, discount
    )
    price = black76_price(forward, strike, volatility, time_to_expiry, discount, option_type)
    return -time_to_expiry * price


def vanna(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
) -> NDArray[np.float64]:
    """``d2V/dF dsigma``: how delta moves when volatility moves.

    Non-zero vanna is why a skewed market's delta hedge differs from the
    flat-volatility hedge, making it the cheapest formal link between the
    shape of the smile and hedging P&L.
    """
    forward, strike, volatility, time_to_expiry, discount = _prepare(
        forward, strike, volatility, time_to_expiry, discount
    )
    option_sign(option_type)
    d1, d2 = d1_d2(forward, strike, volatility, time_to_expiry)
    safe_vol = np.where(volatility > 0.0, volatility, 1.0)
    out = -discount * norm.pdf(d1) * d2 / safe_vol
    return np.where(volatility > 0.0, out, 0.0)


def volga(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
) -> NDArray[np.float64]:
    """``d2V/dsigma2``: convexity of value in volatility.

    Positive volga is why wing options gain when volatility-of-volatility
    rises even if spot never moves.
    """
    forward, strike, volatility, time_to_expiry, discount = _prepare(
        forward, strike, volatility, time_to_expiry, discount
    )
    option_sign(option_type)
    d1, d2 = d1_d2(forward, strike, volatility, time_to_expiry)
    base = vega(forward, strike, volatility, time_to_expiry, discount, option_type)
    safe_vol = np.where(volatility > 0.0, volatility, 1.0)
    out = base * d1 * d2 / safe_vol
    return np.where(volatility > 0.0, out, 0.0)


def greek_table(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
    rate: ArrayLike = 0.0,
) -> dict[str, NDArray[np.float64]]:
    """Compute every Greek in one pass, keyed by name."""
    common = (forward, strike, volatility, time_to_expiry, discount, option_type)
    return {
        "delta": delta(*common),
        "dual_delta": dual_delta(*common),
        "gamma": gamma(*common),
        "vega": vega(*common),
        "theta": theta(*common, rate=rate),
        "rho": rho(*common),
        "vanna": vanna(*common),
        "volga": volga(*common),
    }


def per_percentage_point(value: ArrayLike) -> NDArray[np.float64]:
    """Convert a per-unit sensitivity (vega, rho) to a per-1% figure."""
    return np.asarray(value, dtype=float) / 100.0


def per_calendar_day(value: ArrayLike, days_per_year: float = 365.0) -> NDArray[np.float64]:
    """Convert an annualised theta to a per-calendar-day figure."""
    return np.asarray(value, dtype=float) / days_per_year
