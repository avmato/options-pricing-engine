"""Implied volatility inversion.

Three things separate this from the textbook bisection loop:

1. **Out-of-the-money transformation.** In-the-money quotes are dominated by
   intrinsic value, so the volatility-dependent part is a small difference of
   two large numbers. Put-call parity maps every in-the-money quote to its
   out-of-the-money twin *before* inverting, which removes that cancellation.
2. **Safeguarded Newton.** A Newton step using vega converges quadratically,
   but vega vanishes in the wings and the step can leave the bracket. Each
   iteration therefore keeps a valid bracket and falls back to bisection
   whenever the Newton step misbehaves -- the classic ``rtsafe`` scheme.
3. **Vectorisation.** The whole chain is inverted in one array pass, and the
   result carries a per-quote status code instead of silently returning NaN.

:func:`implied_vol_bisection` keeps the naive method available so the
accuracy/speed comparison in ``reports/`` can be reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optlab.core.black import black76_price, option_sign
from optlab.core.greeks import vega as black76_vega

__all__ = [
    "IVStatus",
    "IVResult",
    "implied_vol",
    "implied_vol_bisection",
    "initial_guess",
]

_MIN_VOL = 1e-9
_MAX_VOL = 5.0
# The volatility resolution the library promises: results are meaningful to
# about a hundredth of a basis point of volatility, not beyond.
_VOL_RESOLUTION = 1e-6
# Relative step in volatility below which the iteration is considered done.
_VOL_STEP_TOLERANCE = 1e-12


class IVStatus(IntEnum):
    """Why a given quote did or did not produce an implied volatility."""

    OK = 0
    BELOW_INTRINSIC = 1  # price under the no-arbitrage lower bound
    ABOVE_UPPER_BOUND = 2  # price above the no-arbitrage upper bound
    NOT_CONVERGED = 3  # bracket valid but tolerance not reached
    INVALID_INPUT = 4  # non-finite or non-positive time to expiry
    NOT_IDENTIFIED = 5  # vega too small for the price to pin down a volatility


@dataclass(frozen=True)
class IVResult:
    """Implied volatilities plus per-quote diagnostics.

    Attributes
    ----------
    volatility:
        Implied volatility, ``NaN`` where ``status != IVStatus.OK``.
    status:
        Per-quote :class:`IVStatus` code.
    iterations:
        Number of solver iterations actually executed (shared across the
        vectorised batch).
    price_error:
        ``model_price - target_price`` at the returned volatility, in the
        undiscounted forward measure. The honest accuracy measure.
    """

    volatility: NDArray[np.float64]
    status: NDArray[np.int8]
    iterations: int
    price_error: NDArray[np.float64]

    @property
    def ok(self) -> NDArray[np.bool_]:
        """Boolean mask of quotes that inverted successfully."""
        return self.status == int(IVStatus.OK)

    def summary(self) -> dict[str, int]:
        """Count quotes by status name, for reporting."""
        return {
            status.name: int(np.sum(self.status == int(status)))
            for status in IVStatus
            if np.any(self.status == int(status))
        }


def initial_guess(
    normalised_price: NDArray[np.float64],
    forward: NDArray[np.float64],
    strike: NDArray[np.float64],
    time_to_expiry: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Brenner-Subrahmanyam style starting point for the Newton iteration.

    At the money the Black-76 price is very nearly
    ``0.4 F sigma sqrt(T)``, which inverts to
    ``sigma ~ price / (0.4 F sqrt(T))``. Away from the money that
    underestimates, so the guess is inflated by the log-moneyness term. The
    guess only has to land in the right order of magnitude: the safeguarded
    iteration handles the rest.
    """
    sqrt_t = np.sqrt(np.maximum(time_to_expiry, _MIN_VOL))
    scale = np.maximum(0.5 * (forward + strike), _MIN_VOL)
    atm_guess = normalised_price / (0.3989422804014327 * scale * sqrt_t)
    log_moneyness = np.log(np.maximum(forward, _MIN_VOL) / np.maximum(strike, _MIN_VOL))
    wing_inflation = np.sqrt(2.0 * np.abs(log_moneyness)) / sqrt_t
    return np.clip(np.maximum(atm_guess, wing_inflation), 1e-3, 3.0)


def _to_otm(
    price: NDArray[np.float64],
    forward: NDArray[np.float64],
    strike: NDArray[np.float64],
    sign: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map every quote to its out-of-the-money twin via put-call parity.

    Returns the transformed undiscounted price and the transformed sign.
    ``C - P = F - K`` in the forward measure, so an in-the-money call at
    strike ``K`` carries the same information as the put at the same strike,
    priced at ``C - (F - K)``.
    """
    is_itm = sign * (forward - strike) > 0.0
    otm_price = np.where(is_itm, price - sign * (forward - strike), price)
    otm_sign = np.where(is_itm, -sign, sign)
    return otm_price, otm_sign


def implied_vol(
    price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> IVResult:
    """Invert Black-76 for volatility with a safeguarded Newton iteration.

    Parameters
    ----------
    price:
        Observed option price, in the same (discounted) units the pricer
        returns.
    tolerance:
        Absolute floor on the *undiscounted* price residual. The effective
        tolerance is ``max(tolerance, 1e-8 * price)``, so cheap wing quotes
        are held to a relative standard rather than an absolute one.

    Returns
    -------
    IVResult
        Volatilities plus a status code per quote. Quotes outside the
        no-arbitrage band are reported, not silently dropped: which quotes
        fail and why is itself a result.
    """
    price = np.asarray(price, dtype=float)
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    time_to_expiry = np.asarray(time_to_expiry, dtype=float)
    discount = np.asarray(discount, dtype=float)
    sign = option_sign(option_type)

    price, forward, strike, time_to_expiry, discount, sign = np.broadcast_arrays(
        price, forward, strike, time_to_expiry, discount, sign
    )
    shape = price.shape
    status = np.full(shape, int(IVStatus.OK), dtype=np.int8)

    invalid = ~np.isfinite(price) | ~np.isfinite(forward) | ~np.isfinite(strike)
    invalid |= ~(time_to_expiry > 0.0) | ~(discount > 0.0)
    status = np.where(invalid, int(IVStatus.INVALID_INPUT), status)

    # Work undiscounted and out-of-the-money: both improve conditioning.
    with np.errstate(divide="ignore", invalid="ignore"):
        undiscounted = price / discount
    target, otm_sign = _to_otm(undiscounted, forward, strike, sign)

    # An out-of-the-money option has zero intrinsic value, so the bounds are
    # simply 0 and (F for a call, K for a put).
    upper_bound = np.where(otm_sign > 0, forward, strike)
    below = (target < -tolerance) & ~invalid
    above = (target > upper_bound + tolerance) & ~invalid
    status = np.where(below, int(IVStatus.BELOW_INTRINSIC), status)
    status = np.where(above, int(IVStatus.ABOVE_UPPER_BOUND), status)

    solvable = status == int(IVStatus.OK)
    otm_type = np.where(otm_sign > 0, "call", "put")

    vol = initial_guess(np.where(solvable, target, 0.0), forward, strike, time_to_expiry)

    # Iterate on an *active set*. Most quotes converge in a handful of Newton
    # steps; a few deep wings fall back to bisection and need dozens. Running
    # the whole array until the slowest element is done makes every quote pay
    # the worst case, so converged quotes are dropped from the working arrays
    # at each step and only the stragglers are re-priced.
    active = np.flatnonzero(solvable.ravel())
    flat_vol = vol.ravel().copy()
    active_forward = forward.ravel()[active]
    active_strike = strike.ravel()[active]
    active_time = time_to_expiry.ravel()[active]
    active_target = target.ravel()[active]
    active_type = otm_type.ravel()[active]
    active_vol = flat_vol[active]
    low = np.full(active.shape, _MIN_VOL)
    high = np.full(active.shape, _MAX_VOL)

    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        if active.size == 0:
            break

        model = black76_price(active_forward, active_strike, active_vol, active_time, 1.0, active_type)
        residual = model - active_target
        slope = black76_vega(active_forward, active_strike, active_vol, active_time, 1.0, active_type)

        # The price is strictly increasing in volatility, so the residual's
        # sign tells us which side of the bracket to tighten.
        low = np.where(residual < 0.0, active_vol, low)
        high = np.where(residual > 0.0, active_vol, high)

        with np.errstate(divide="ignore", invalid="ignore"):
            newton = active_vol - residual / slope
        unusable = ~np.isfinite(newton) | (newton <= low) | (newton >= high) | (slope < 1e-12)
        candidate = np.where(unusable, 0.5 * (low + high), newton)

        # Converge on the volatility, not on the price. A deep wing option can
        # be worth 1e-9, and an absolute price tolerance of 1e-10 would then
        # stop the iteration while the price is still 10% wrong. The step in
        # volatility is scale-free and has no such failure mode.
        step = np.abs(candidate - active_vol)
        active_vol = candidate
        flat_vol[active] = active_vol

        still_running = step > _VOL_STEP_TOLERANCE * np.maximum(active_vol, 1e-3)
        if not still_running.all():
            active = active[still_running]
            active_forward = active_forward[still_running]
            active_strike = active_strike[still_running]
            active_time = active_time[still_running]
            active_target = active_target[still_running]
            active_type = active_type[still_running]
            active_vol = active_vol[still_running]
            low = low[still_running]
            high = high[still_running]

    vol = flat_vol.reshape(shape)

    model = black76_price(forward, strike, vol, time_to_expiry, 1.0, otm_type)
    residual = model - target
    # Judge convergence relative to the option's own price: an absolute
    # threshold is meaningless across quotes spanning ten orders of magnitude.
    residual_budget = np.maximum(tolerance, 1e-8 * np.abs(target))
    not_converged = solvable & (np.abs(residual) > residual_budget)
    status = np.where(not_converged, int(IVStatus.NOT_CONVERGED), status)

    # Identifiability. Deep in-the-money quotes are almost pure intrinsic
    # value, so mapping them to their out-of-the-money twin subtracts two
    # nearly equal numbers and the surviving extrinsic value is dominated by
    # rounding error. When vega is small enough that this rounding error
    # exceeds the volatility resolution we care about, the price simply does
    # not determine a volatility, and saying so is better than returning a
    # number the data cannot support.
    rounding_floor = np.finfo(float).eps * np.maximum(np.abs(forward - strike), 1.0)
    identifiability_floor = 100.0 * rounding_floor / _VOL_RESOLUTION
    final_vega = black76_vega(forward, strike, vol, time_to_expiry, 1.0, otm_type)
    unidentified = solvable & (final_vega < identifiability_floor)
    status = np.where(unidentified, int(IVStatus.NOT_IDENTIFIED), status)

    vol = np.where(status == int(IVStatus.OK), vol, np.nan)
    residual = np.where(status == int(IVStatus.OK), residual, np.nan)
    return IVResult(
        volatility=np.asarray(vol, dtype=float),
        status=np.asarray(status, dtype=np.int8),
        iterations=iterations,
        price_error=np.asarray(residual, dtype=float),
    )


def implied_vol_bisection(
    price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    discount: ArrayLike = 1.0,
    option_type: ArrayLike = "call",
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> NDArray[np.float64]:
    """Plain bisection on the raw (non-transformed) price.

    Kept as the baseline for the solver benchmark. It is robust but needs
    roughly one iteration per bit of accuracy, whereas the safeguarded Newton
    method roughly doubles the correct digits each step.
    """
    price = np.asarray(price, dtype=float)
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    time_to_expiry = np.asarray(time_to_expiry, dtype=float)
    discount = np.asarray(discount, dtype=float)
    option_sign(option_type)

    price, forward, strike, time_to_expiry, discount = np.broadcast_arrays(
        price, forward, strike, time_to_expiry, discount
    )
    low = np.full(price.shape, _MIN_VOL)
    high = np.full(price.shape, _MAX_VOL)

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        model = black76_price(forward, strike, mid, time_to_expiry, discount, option_type)
        too_cheap = model < price
        low = np.where(too_cheap, mid, low)
        high = np.where(too_cheap, high, mid)
        if np.all(high - low < tolerance):
            break

    vol = 0.5 * (low + high)
    at_edge = (vol < _MIN_VOL * 10) | (vol > _MAX_VOL * 0.999)
    return np.where(at_edge, np.nan, vol)
