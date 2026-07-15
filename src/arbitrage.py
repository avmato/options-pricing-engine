"""No-arbitrage bounds for European options."""

import numpy as np


def discounted_strike(
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
) -> float | np.ndarray:
    """Return the present value of the strike."""
    return np.asarray(strike) * np.exp(
        -np.asarray(rate) * np.asarray(time_to_expiry)
    )


def call_price_bounds(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return the no-arbitrage lower and upper bounds for a European call."""
    present_value_strike = discounted_strike(
        strike,
        rate,
        time_to_expiry,
    )

    lower_bound = np.maximum(
        np.asarray(spot) - present_value_strike,
        0.0,
    )

    upper_bound = np.asarray(spot)

    return lower_bound, upper_bound


def put_price_bounds(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return the no-arbitrage lower and upper bounds for a European put."""
    present_value_strike = discounted_strike(
        strike,
        rate,
        time_to_expiry,
    )

    lower_bound = np.maximum(
        present_value_strike - np.asarray(spot),
        0.0,
    )

    upper_bound = present_value_strike
    return lower_bound, upper_bound
    
def put_call_parity_residual(
    call_price: float | np.ndarray,
    put_price: float | np.ndarray,
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
) -> float | np.ndarray:
    """Return the deviation from European put-call parity."""
    present_value_strike = discounted_strike(
        strike,
        rate,
        time_to_expiry,
    )

    return (
        np.asarray(call_price)
        - np.asarray(put_price)
        - np.asarray(spot)
        + present_value_strike
    )