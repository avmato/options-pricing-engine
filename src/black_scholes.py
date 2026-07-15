"""Black-Scholes pricing utilities."""

import numpy as np
from scipy.stats import norm


def calculate_d1_d2(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Calculate the d1 and d2 terms used in Black-Scholes."""

    spot_array = np.asarray(spot, dtype=float)
    strike_array = np.asarray(strike, dtype=float)
    volatility_array = np.asarray(volatility, dtype=float)
    time_array = np.asarray(time_to_expiry, dtype=float)
    rate_array = np.asarray(rate, dtype=float)

    if np.any(spot_array <= 0):
        raise ValueError("Spot prices must be positive.")

    if np.any(strike_array <= 0):
        raise ValueError("Strike prices must be positive.")

    if np.any(volatility_array <= 0):
        raise ValueError("Volatility must be positive.")

    if np.any(time_array <= 0):
        raise ValueError("Time to expiry must be positive.")

    volatility_time = volatility_array * np.sqrt(time_array)

    d1 = (
        np.log(spot_array / strike_array)
        + (rate_array + 0.5 * volatility_array**2) * time_array
    ) / volatility_time

    d2 = d1 - volatility_time

    return d1, d2


def black_scholes_call(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
) -> float | np.ndarray:
    """Return the Black-Scholes price of a European call option."""

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
    time_array = np.asarray(time_to_expiry, dtype=float)

    discounted_strike = strike_array * np.exp(
        -rate_array * time_array
    )

    return (
        spot_array * norm.cdf(d1)
        - discounted_strike * norm.cdf(d2)
    )


def black_scholes_put(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
) -> float | np.ndarray:
    """Return the Black-Scholes price of a European put option."""

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
    time_array = np.asarray(time_to_expiry, dtype=float)

    discounted_strike = strike_array * np.exp(
        -rate_array * time_array
    )

    return (
        discounted_strike * norm.cdf(-d2)
        - spot_array * norm.cdf(-d1)
    )