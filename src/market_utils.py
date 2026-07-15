"""Basic utilities for working with European options."""

import numpy as np


def call_intrinsic_value(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
) -> float | np.ndarray:
    """Return the intrinsic value of a European call option."""
    return np.maximum(spot - strike, 0.0)


def put_intrinsic_value(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
) -> float | np.ndarray:
    """Return the intrinsic value of a European put option."""
    return np.maximum(strike - spot, 0.0)

def moneyness(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
) -> float | np.ndarray:
    """Return spot divided by strike."""
    strike_array = np.asarray(strike)

    if np.any(strike_array <= 0):
        raise ValueError("Strike prices must be positive.")

    return spot / strike_array


def log_moneyness(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
) -> float | np.ndarray:
    """Return the natural logarithm of spot divided by strike."""
    spot_array = np.asarray(spot)
    strike_array = np.asarray(strike)

    if np.any(spot_array <= 0):
        raise ValueError("Spot prices must be positive.")

    if np.any(strike_array <= 0):
        raise ValueError("Strike prices must be positive.")

    return np.log(spot_array / strike_array)


def years_to_expiry(
    days_to_expiry: float | np.ndarray,
) -> float | np.ndarray:
    """Convert calendar days to years using a 365-day convention."""
    days_array = np.asarray(days_to_expiry)

    if np.any(days_array < 0):
        raise ValueError("Days to expiry cannot be negative.")

    return days_array / 365.0