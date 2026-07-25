"""Forward-parameterised Black-76 pricing.

Everything in this package prices off the *forward* ``F`` and a *discount
factor* ``D``, not off ``(spot, rate, dividend yield)``.

The reason is practical. A spot-based Black-Scholes formula forces you to
supply a risk-free rate and a dividend yield that you do not observe. The
forward and the discount factor, in contrast, can be read straight out of the
option quotes themselves (see :mod:`optlab.market.forward`), so the model
stops depending on two numbers the user has to guess.

The two parameterisations are related by

.. math::

    F = S e^{(r - q)T}, \\qquad D = e^{-rT},

and :func:`bs_price` is provided as a thin wrapper for the textbook case.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

OptionType = Literal["call", "put"]

__all__ = [
    "OptionType",
    "option_sign",
    "d1_d2",
    "black76_price",
    "bs_price",
    "forward_from_spot",
    "intrinsic_value",
    "price_bounds",
    "parity_gap",
]


def option_sign(option_type: ArrayLike) -> NDArray[np.float64]:
    """Return ``+1`` for calls and ``-1`` for puts.

    Accepts a scalar string or an array of strings, which lets every pricing
    function below handle a mixed call/put chain in a single vectorised call
    instead of looping row by row.
    """
    types = np.asarray(option_type, dtype=object)
    sign = np.where(types == "call", 1.0, np.where(types == "put", -1.0, np.nan))
    sign = np.asarray(sign, dtype=float)
    if np.isnan(sign).any():
        bad = np.unique(np.asarray(types)[np.isnan(sign)])
        raise ValueError(f"option_type must be 'call' or 'put', got {bad.tolist()}")
    return sign


def forward_from_spot(
    spot: ArrayLike,
    rate: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """Forward price implied by a spot, a rate and a continuous dividend yield."""
    spot = np.asarray(spot, dtype=float)
    rate = np.asarray(rate, dtype=float)
    time_to_expiry = np.asarray(time_to_expiry, dtype=float)
    dividend_yield = np.asarray(dividend_yield, dtype=float)
    return spot * np.exp((rate - dividend_yield) * time_to_expiry)


def d1_d2(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Black-76 ``d1`` and ``d2``.

    Written in terms of total volatility ``sigma * sqrt(T)`` so that the
    degenerate cases (zero time, zero volatility) collapse to ``+/-inf``
    rather than dividing by zero.
    """
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    volatility = np.asarray(volatility, dtype=float)
    time_to_expiry = np.asarray(time_to_expiry, dtype=float)

    total_vol = volatility * np.sqrt(np.maximum(time_to_expiry, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_moneyness = np.log(forward / strike)
        d1 = (log_moneyness + 0.5 * total_vol**2) / total_vol
        d2 = d1 - total_vol

    # sigma*sqrt(T) == 0: the payoff is deterministic, so push d1/d2 to the
    # correct infinity and let N(d) become the 0/1 indicator.
    degenerate = total_vol <= 0.0
    if np.any(degenerate):
        limit = np.where(log_moneyness > 0.0, np.inf, -np.inf)
        limit = np.where(log_moneyness == 0.0, 0.0, limit)
        d1 = np.where(degenerate, limit, d1)
        d2 = np.where(degenerate, limit, d2)
    return np.asarray(d1, dtype=float), np.asarray(d2, dtype=float)


def black76_price(
    forward: ArrayLike,
    strike: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
) -> NDArray[np.float64]:
    """Black-76 price of a European option.

    .. math::

        V = \\omega D \\left[ F N(\\omega d_1) - K N(\\omega d_2) \\right]

    with :math:`\\omega = +1` for a call and :math:`-1` for a put. Writing
    both payoffs with a single sign keeps calls and puts on one code path,
    which is what makes put-call parity hold *exactly* in floating point
    rather than approximately.
    """
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    discount = np.asarray(discount, dtype=float)
    sign = option_sign(option_type)

    d1, d2 = d1_d2(forward, strike, volatility, time_to_expiry)
    return discount * sign * (forward * norm.cdf(sign * d1) - strike * norm.cdf(sign * d2))


def bs_price(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    time_to_expiry: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
    option_type: ArrayLike = "call",
) -> NDArray[np.float64]:
    """Textbook spot-based Black-Scholes-Merton price.

    Provided for tests and for comparison against published values; the rest
    of the library uses :func:`black76_price` so that no rate or dividend
    assumption is baked in.
    """
    rate = np.asarray(rate, dtype=float)
    time_to_expiry = np.asarray(time_to_expiry, dtype=float)
    forward = forward_from_spot(spot, rate, time_to_expiry, dividend_yield)
    discount = np.exp(-rate * time_to_expiry)
    return black76_price(forward, strike, volatility, time_to_expiry, discount, option_type)


def intrinsic_value(
    forward: ArrayLike,
    strike: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
) -> NDArray[np.float64]:
    """Discounted intrinsic value ``D * max(w * (F - K), 0)``.

    This is the *forward* intrinsic value, which is the correct European
    lower bound. It is not the same as ``max(S - K, 0)``.
    """
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    discount = np.asarray(discount, dtype=float)
    sign = option_sign(option_type)
    return discount * np.maximum(sign * (forward - strike), 0.0)


def price_bounds(
    forward: ArrayLike,
    strike: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """No-arbitrage lower and upper bounds for a European option.

    ``D max(w(F-K), 0) <= V <= D F`` for a call and ``<= D K`` for a put.
    A quote outside this band is a genuine arbitrage rather than a modelling
    disagreement: exploiting it requires no view on volatility at all.
    """
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    discount = np.asarray(discount, dtype=float)
    sign = option_sign(option_type)

    lower = intrinsic_value(forward, strike, discount, option_type)
    upper = np.where(sign > 0, discount * forward, discount * strike)
    return lower, np.asarray(upper, dtype=float)


def parity_gap(
    call_price: ArrayLike,
    put_price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    discount: ArrayLike = 1.0,
) -> NDArray[np.float64]:
    """Put-call parity residual ``(C - P) - D (F - K)``.

    Zero for arbitrage-free European quotes. The sign says which leg of the
    conversion/reversal is cheap: a positive gap means the call is rich
    relative to the put, so the trade is sell call / buy put / buy forward.
    """
    call_price = np.asarray(call_price, dtype=float)
    put_price = np.asarray(put_price, dtype=float)
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    discount = np.asarray(discount, dtype=float)
    return (call_price - put_price) - discount * (forward - strike)
