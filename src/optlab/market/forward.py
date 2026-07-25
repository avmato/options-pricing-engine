"""Implied forward and discount factor from put-call parity.

A long call plus a short put at the same strike and expiry replicates the
underlying: at expiry the package is worth ``S_T - K`` whatever ``S_T`` does.
Its value today is therefore the value of that forward commitment,

.. math::

    C - P = D (F - K),

which, read as a function of the strike, is a straight line with slope
``-D`` and intercept ``D F``. Regressing the observed ``C - P`` on ``K``
recovers both quantities *from the quotes themselves*, so the pipeline never
has to be told a risk-free rate or a dividend yield.

Two practical warnings, both implemented here rather than left as caveats:

* **The forward is well identified; the rate usually is not.** ``F`` comes
  from the intercept and is pinned down to a few basis points. ``r`` comes
  from ``-log(-slope)/T``, and dividing a small slope error by a small ``T``
  amplifies it enormously. :class:`ForwardFit` therefore reports a confidence
  interval for the rate and a flag saying whether it is identified at all.
* **US equity options are American.** Early exercise inflates deep in-the-money
  puts, which tilts the regression line. The fit is restricted to a
  near-the-money band, where the early-exercise premium is negligible, and
  the band is a parameter so the sensitivity can be shown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

__all__ = [
    "ForwardFit",
    "fit_forward",
    "fit_forward_curve",
    "fit_forward_curve_fixed_discount",
    "forward_from_fixed_discount",
]

_MIN_STRIKES = 5


@dataclass(frozen=True)
class ForwardFit:
    """Result of one expiry's parity regression."""

    expiration: pd.Timestamp
    time_to_expiry: float
    spot: float
    forward: float
    discount: float
    implied_rate: float
    implied_dividend_yield: float
    forward_std_error: float
    discount_std_error: float
    rate_low: float
    rate_high: float
    rate_identified: bool
    r_squared: float
    max_abs_residual: float
    n_strikes: int
    band: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _weighted_least_squares(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``y = a + b x`` with weights; return coefficients, covariance, residuals."""
    design = np.column_stack([np.ones_like(x), x])
    weighted_design = design * weights[:, None]
    normal_matrix = design.T @ weighted_design
    coefficients = np.linalg.solve(normal_matrix, weighted_design.T @ y)

    residuals = y - design @ coefficients
    degrees_of_freedom = max(len(x) - 2, 1)
    variance = float(weights @ residuals**2) / degrees_of_freedom
    covariance = variance * np.linalg.inv(normal_matrix)
    return coefficients, covariance, residuals


def fit_forward(
    pairs: pd.DataFrame,
    *,
    band: float = 0.05,
    price_column: str = "mid",
    weight_by_spread: bool = True,
    rate_identification_tolerance: float = 0.02,
) -> ForwardFit:
    """Fit ``C - P = D(F - K)`` for a single expiry.

    Parameters
    ----------
    pairs:
        Output of :func:`optlab.market.quotes.pair_calls_and_puts` restricted
        to one expiration.
    band:
        Half-width in log-moneyness of the strike window used for the fit.
        Narrow enough to avoid American early-exercise contamination, wide
        enough to identify the slope.
    rate_identification_tolerance:
        A rate is called identified when its 95% confidence interval is
        narrower than this (in absolute rate units, so ``0.02`` is 200 bp).

    Raises
    ------
    ValueError
        If fewer than five paired strikes fall inside the band.
    """
    window = pairs[np.abs(pairs["log_moneyness"]) <= band].copy()
    if len(window) < _MIN_STRIKES:
        raise ValueError(
            f"need at least {_MIN_STRIKES} paired strikes within band={band}, got {len(window)}"
        )

    strikes = window["strike"].to_numpy(dtype=float)
    parity_spread = (
        window[f"{price_column}_call"].to_numpy(dtype=float)
        - window[f"{price_column}_put"].to_numpy(dtype=float)
    )

    if weight_by_spread:
        combined_spread = window["spread_call"].to_numpy(float) + window["spread_put"].to_numpy(float)
        weights = 1.0 / np.maximum(combined_spread, 0.01) ** 2
    else:
        weights = np.ones_like(strikes)
    weights = weights / weights.sum() * len(weights)

    coefficients, covariance, residuals = _weighted_least_squares(strikes, parity_spread, weights)
    intercept, slope = coefficients

    discount = float(-slope)
    forward = float(intercept / discount)
    time_to_expiry = float(window["time_to_expiry"].iloc[0])
    spot = float(window["spot"].iloc[0])

    discount_se = float(np.sqrt(covariance[1, 1]))
    # dF/d(intercept) = 1/D and dF/d(slope) = intercept/D^2, combined with the
    # full covariance rather than treating the two coefficients as independent.
    gradient = np.array([1.0 / discount, intercept / discount**2])
    forward_se = float(np.sqrt(gradient @ covariance @ gradient))

    implied_rate = float(-np.log(discount) / time_to_expiry)
    implied_dividend_yield = float(implied_rate - np.log(forward / spot) / time_to_expiry)

    # Propagate the slope uncertainty into the rate: this is where the
    # short-maturity amplification becomes visible.
    discount_low = max(discount - 1.96 * discount_se, 1e-6)
    discount_high = discount + 1.96 * discount_se
    rate_low = float(-np.log(discount_high) / time_to_expiry)
    rate_high = float(-np.log(discount_low) / time_to_expiry)

    weighted_mean = float(np.average(parity_spread, weights=weights))
    ss_res = float(weights @ residuals**2)
    ss_tot = float(weights @ (parity_spread - weighted_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return ForwardFit(
        expiration=window["expiration"].iloc[0],
        time_to_expiry=time_to_expiry,
        spot=spot,
        forward=forward,
        discount=discount,
        implied_rate=implied_rate,
        implied_dividend_yield=implied_dividend_yield,
        forward_std_error=forward_se,
        discount_std_error=discount_se,
        rate_low=rate_low,
        rate_high=rate_high,
        rate_identified=bool((rate_high - rate_low) < rate_identification_tolerance),
        r_squared=r_squared,
        max_abs_residual=float(np.max(np.abs(residuals))),
        n_strikes=len(window),
        band=band,
    )


def forward_from_fixed_discount(
    pairs: pd.DataFrame,
    discount: float,
    *,
    band: float = 0.05,
    price_column: str = "mid",
    weight_by_spread: bool = True,
) -> tuple[float, float]:
    """Imply the forward alone, taking the discount factor as given.

    This is the production-grade default. The discount factor belongs to the
    rate curve, where it is observed to a basis point; asking a noisy option
    chain to re-estimate it adds variance for no information. Rearranging
    parity strike by strike gives ``F = K + (C - P)/D``, and the spread of
    those estimates across strikes is a useful data-quality measure.

    Returns
    -------
    tuple
        ``(forward, standard_error_of_the_mean)``.
    """
    window = pairs[np.abs(pairs["log_moneyness"]) <= band]
    if len(window) < _MIN_STRIKES:
        raise ValueError(f"need at least {_MIN_STRIKES} paired strikes, got {len(window)}")

    parity_spread = (
        window[f"{price_column}_call"].to_numpy(float) - window[f"{price_column}_put"].to_numpy(float)
    )
    per_strike_forward = window["strike"].to_numpy(float) + parity_spread / discount

    if weight_by_spread:
        combined_spread = window["spread_call"].to_numpy(float) + window["spread_put"].to_numpy(float)
        weights = 1.0 / np.maximum(combined_spread, 0.01) ** 2
    else:
        weights = np.ones_like(per_strike_forward)

    forward = float(np.average(per_strike_forward, weights=weights))
    dispersion = float(np.sqrt(np.average((per_strike_forward - forward) ** 2, weights=weights)))
    return forward, dispersion / np.sqrt(len(window))


def fit_forward_curve(
    pairs: pd.DataFrame,
    *,
    band: float = 0.05,
    price_column: str = "mid",
    weight_by_spread: bool = True,
) -> pd.DataFrame:
    """Run :func:`fit_forward` for every expiration and collect the results.

    Expiries with too few paired strikes are skipped rather than raising, so
    that one thin expiry cannot take down a whole snapshot; the returned
    frame simply omits them.
    """
    fits: list[dict[str, object]] = []
    for _, chunk in pairs.groupby("expiration", sort=True):
        try:
            fit = fit_forward(
                chunk,
                band=band,
                price_column=price_column,
                weight_by_spread=weight_by_spread,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        fits.append(fit.to_dict())

    if not fits:
        raise ValueError("no expiration had enough paired strikes to fit a forward")

    frame = pd.DataFrame(fits)
    frame["days_to_expiry"] = frame["time_to_expiry"] * 365.0
    # A discount factor above one is a negative interest rate. On a snapshot
    # this never means the market is pricing negative rates; it means the
    # regression slope is dominated by quote noise.
    frame["discount_is_economic"] = frame["discount"] <= 1.0
    return frame.sort_values("time_to_expiry").reset_index(drop=True)


def fit_forward_curve_fixed_discount(
    pairs: pd.DataFrame,
    *,
    rate: float,
    band: float = 0.05,
    price_column: str = "mid",
    weight_by_spread: bool = True,
) -> pd.DataFrame:
    """Forward curve with the discount factor taken from an external rate.

    This is the default the audit runs on, and the reason is visible in the
    joint fit's own diagnostics. Writing parity as ``C - P = D(F - K)``, the
    intercept determines ``F`` while the slope determines ``D``, and the rate
    follows as ``r = -log(D)/T``. Differentiating,

    .. math::

        \\frac{\\partial r}{\\partial D} = -\\frac{1}{D T},

    so a slope error of one basis point becomes a rate error of ``1/T`` basis
    points: at one week, a 0.1% slope error is a 5% rate error. The forward,
    by contrast, is an intercept and inherits no such amplification.

    So the discount factor is taken where it is actually observable -- the
    money market -- and only the forward is implied from the options.
    """
    rows: list[dict[str, object]] = []
    for expiration, chunk in pairs.groupby("expiration", sort=True):
        time_to_expiry = float(chunk["time_to_expiry"].iloc[0])
        if time_to_expiry <= 0:
            continue
        discount = float(np.exp(-rate * time_to_expiry))
        try:
            forward, forward_se = forward_from_fixed_discount(
                chunk,
                discount,
                band=band,
                price_column=price_column,
                weight_by_spread=weight_by_spread,
            )
        except ValueError:
            continue

        spot = float(chunk["spot"].iloc[0])
        rows.append(
            {
                "expiration": expiration,
                "time_to_expiry": time_to_expiry,
                "days_to_expiry": time_to_expiry * 365.0,
                "spot": spot,
                "forward": forward,
                "discount": discount,
                "implied_rate": rate,
                "implied_dividend_yield": rate - float(np.log(forward / spot)) / time_to_expiry,
                "forward_std_error": forward_se,
                "n_strikes": int((np.abs(chunk["log_moneyness"]) <= band).sum()),
                "band": band,
                "discount_is_economic": discount <= 1.0,
            }
        )

    if not rows:
        raise ValueError("no expiration had enough paired strikes to imply a forward")
    return pd.DataFrame(rows).sort_values("time_to_expiry").reset_index(drop=True)
