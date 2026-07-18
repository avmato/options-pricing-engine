"""Analytical Greeks for the Black-Scholes model."""

import numpy as np
from scipy.stats import norm

from src.black_scholes import calculate_d1_d2


def call_delta(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
) -> float | np.ndarray:
    """Return the Black-Scholes delta of a European call option."""

    d1, _ = calculate_d1_d2(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
    )

    return norm.cdf(d1)


def put_delta(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
) -> float | np.ndarray:
    """Return the Black-Scholes delta of a European put option."""

    d1, _ = calculate_d1_d2(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
    )

    return norm.cdf(d1) - 1.0

def gamma(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
) -> float | np.ndarray:
    """Return the Black-Scholes gamma of a European option."""

    d1, _ = calculate_d1_d2(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
    )

    spot_array = np.asarray(spot, dtype=float)
    volatility_array = np.asarray(volatility, dtype=float)
    time_array = np.asarray(time_to_expiry, dtype=float)

    return norm.pdf(d1) / (
        spot_array
        * volatility_array
        * np.sqrt(time_array)
    )

def vega(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
) -> float | np.ndarray:
    """Return Black-Scholes vega per one unit of volatility."""

    d1, _ = calculate_d1_d2(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
    )

    spot_array = np.asarray(spot, dtype=float)
    time_array = np.asarray(time_to_expiry, dtype=float)

    return (
        spot_array
        * norm.pdf(d1)
        * np.sqrt(time_array)
    )

def call_theta(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
    per_day: bool = False,
) -> float | np.ndarray:
    """Return Black-Scholes call theta, annually or per calendar day."""

    d1, d2 = calculate_d1_d2(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
    )

    spot_array = np.asarray(spot, dtype=float)
    strike_array = np.asarray(strike, dtype=float)
    rate_array = np.asarray(rate, dtype=float)
    volatility_array = np.asarray(volatility, dtype=float)
    time_array = np.asarray(time_to_expiry, dtype=float)

    first_term = -(
        spot_array
        * norm.pdf(d1)
        * volatility_array
    ) / (2.0 * np.sqrt(time_array))

    second_term = -(
        rate_array
        * strike_array
        * np.exp(-rate_array * time_array)
        * norm.cdf(d2)
    )

    theta = first_term + second_term

    if per_day:
        return theta / 365.0

    return theta

def put_theta(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
    per_day: bool = False,
) -> float | np.ndarray:
    """Return Black-Scholes put theta, annually or per calendar day."""

    d1, d2 = calculate_d1_d2(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
    )

    spot_array = np.asarray(spot, dtype=float)
    strike_array = np.asarray(strike, dtype=float)
    rate_array = np.asarray(rate, dtype=float)
    volatility_array = np.asarray(volatility, dtype=float)
    time_array = np.asarray(time_to_expiry, dtype=float)

    first_term = -(
        spot_array
        * norm.pdf(d1)
        * volatility_array
    ) / (2.0 * np.sqrt(time_array))

    second_term = (
        rate_array
        * strike_array
        * np.exp(-rate_array * time_array)
        * norm.cdf(-d2)
    )

    theta = first_term + second_term

    if per_day:
        return theta / 365.0

    return theta

def call_rho(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
    per_rate_point: bool = False,
) -> float | np.ndarray:
    """Return Black-Scholes call rho per unit or percentage-point rate move."""

    _, d2 = calculate_d1_d2(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
    )

    strike_array = np.asarray(strike, dtype=float)
    rate_array = np.asarray(rate, dtype=float)
    time_array = np.asarray(time_to_expiry, dtype=float)

    rho = (
        strike_array
        * time_array
        * np.exp(-rate_array * time_array)
        * norm.cdf(d2)
    )

    if per_rate_point:
        return rho / 100.0

    return rho


def put_rho(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
    per_rate_point: bool = False,
) -> float | np.ndarray:
    """Return Black-Scholes put rho per unit or percentage-point rate move."""

    _, d2 = calculate_d1_d2(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
    )

    strike_array = np.asarray(strike, dtype=float)
    rate_array = np.asarray(rate, dtype=float)
    time_array = np.asarray(time_to_expiry, dtype=float)

    rho = -(
        strike_array
        * time_array
        * np.exp(-rate_array * time_array)
        * norm.cdf(-d2)
    )

    if per_rate_point:
        return rho / 100.0

    return rho