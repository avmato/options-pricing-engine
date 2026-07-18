"""Implied volatility solvers."""

from collections.abc import Callable

from src.black_scholes import (
    black_scholes_call,
    black_scholes_put,
)

from src.arbitrage import call_price_bounds, put_price_bounds



def implied_volatility_bisection(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: str,
    lower_volatility: float = 1e-6,
    upper_volatility: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> float:
    """Return implied volatility using the bisection method."""

    if option_type == "call":
        pricing_function: Callable[..., float] = black_scholes_call
    elif option_type == "put":
        pricing_function = black_scholes_put
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

        if market_price < 0:
            raise ValueError("Market price cannot be negative.")

    if option_type == "call":
        lower_bound, upper_bound = call_price_bounds(
            spot,
            strike,
            rate,
            time_to_expiry,
        )
    else:
        lower_bound, upper_bound = put_price_bounds(
            spot,
            strike,
            rate,
            time_to_expiry,
        )

    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)

    if market_price < lower_bound or market_price > upper_bound:
        raise ValueError(
            f"Market price must lie between "
            f"{lower_bound:.8f} and {upper_bound:.8f}."
        )

    def pricing_error(volatility: float) -> float:
        model_price = pricing_function(
            spot,
            strike,
            rate,
            volatility,
            time_to_expiry,
        )

        return float(model_price - market_price)

    lower_error = pricing_error(lower_volatility)
    upper_error = pricing_error(upper_volatility)

    if lower_error * upper_error > 0:
        raise ValueError(
            "The implied volatility root is not bracketed "
            "inside the chosen interval."
        )

    for _ in range(max_iterations):
        midpoint = 0.5 * (
            lower_volatility + upper_volatility
        )

        midpoint_error = pricing_error(midpoint)

        if abs(midpoint_error) < tolerance:
            return midpoint

        if lower_error * midpoint_error <= 0:
            upper_volatility = midpoint
            upper_error = midpoint_error
        else:
            lower_volatility = midpoint
            lower_error = midpoint_error

    raise RuntimeError(
        "Bisection did not converge within max_iterations."
    )