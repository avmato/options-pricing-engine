"""Solver benchmark: safeguarded Newton against plain bisection.

The comparison is run on a synthetic chain where the true volatility is known
exactly, so "accuracy" means distance from the right answer rather than
agreement between two implementations.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from optlab.core.black import black76_price
from optlab.core.iv import implied_vol, implied_vol_bisection

__all__ = ["benchmark_solvers", "synthetic_benchmark_chain"]


def synthetic_benchmark_chain(
    n_strikes: int = 400,
    forward: float = 500.0,
    time_to_expiry: float = 0.25,
    discount: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strikes, option types, true volatilities and exact prices.

    The volatility is a skewed smile rather than a constant, so the benchmark
    exercises the wings where the two methods differ most.
    """
    strikes = np.linspace(0.55 * forward, 1.55 * forward, n_strikes)
    log_moneyness = np.log(strikes / forward)
    volatility = 0.18 - 0.25 * log_moneyness + 0.6 * log_moneyness**2
    option_type = np.where(strikes >= forward, "call", "put")
    prices = black76_price(forward, strikes, volatility, time_to_expiry, discount, option_type)
    return strikes, option_type, volatility, prices


def _time(callable_, repeats: int) -> tuple[object, float]:
    start = time.perf_counter()
    for _ in range(repeats):
        result = callable_()
    return result, (time.perf_counter() - start) / repeats


def benchmark_solvers(repeats: int = 20) -> pd.DataFrame:
    """Time and accuracy for each solver on identical inputs.

    Bisection appears twice, at two tolerances, because comparing a
    machine-precision method against a loosely converged one is not a
    comparison. The row that matters is bisection run to the same accuracy
    the Newton iteration reaches on its own.
    """
    forward, time_to_expiry, discount = 500.0, 0.25, 0.99
    strikes, option_type, true_vol, prices = synthetic_benchmark_chain(
        forward=forward, time_to_expiry=time_to_expiry, discount=discount
    )
    arguments = (prices, forward, strikes, time_to_expiry, discount, option_type)

    newton, newton_seconds = _time(lambda: implied_vol(*arguments), repeats)
    loose, loose_seconds = _time(
        lambda: implied_vol_bisection(*arguments, tolerance=1e-8), repeats
    )
    tight, tight_seconds = _time(
        lambda: implied_vol_bisection(*arguments, tolerance=1e-15), repeats
    )

    rows = [
        ("safeguarded Newton (this library)", newton.volatility, newton_seconds),
        ("bisection, tolerance 1e-8 (textbook)", loose, loose_seconds),
        ("bisection, tolerance 1e-15 (matched accuracy)", tight, tight_seconds),
    ]

    records: list[dict[str, object]] = []
    for label, recovered, seconds in rows:
        error = np.abs(recovered - true_vol)
        records.append(
            {
                "solver": label,
                "quotes": len(strikes),
                "max_abs_vol_error": float(np.nanmax(error)),
                "median_abs_vol_error": float(np.nanmedian(error)),
                "quotes_returned": int(np.sum(~np.isnan(recovered))),
                "microseconds_per_quote": seconds / len(strikes) * 1e6,
            }
        )

    frame = pd.DataFrame(records)
    matched = frame["microseconds_per_quote"].iloc[2]
    frame["speedup_at_matched_accuracy"] = matched / frame["microseconds_per_quote"]
    return frame
