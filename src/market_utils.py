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