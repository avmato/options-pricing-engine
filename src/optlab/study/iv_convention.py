"""How much of an observed smile is the market, and how much is the convention?

An implied volatility is only defined relative to a forward. Feed the inverter
a forward that is wrong by ``dF`` and the recovered volatility moves by
roughly ``dF * delta / vega`` -- and because a call and a put at the same
strike have deltas of opposite sign, the error pushes their implied
volatilities in *opposite* directions.

That produces a call/put implied volatility gap that looks like a market
feature (calls and puts "disagreeing") but is entirely an artefact of the
assumed rate and dividend yield. This module measures the size of that
artefact by inverting the same quotes twice: once with a textbook fixed rate
and zero dividends, once with the forward implied by put-call parity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optlab.core.black import forward_from_spot
from optlab.core.iv import implied_vol

__all__ = ["implied_vol_under_conventions", "call_put_gap_summary"]


def implied_vol_under_conventions(
    pairs: pd.DataFrame,
    forwards: pd.DataFrame,
    *,
    naive_rate: float = 0.04,
    naive_dividend_yield: float = 0.0,
    band: float | None = 0.15,
) -> pd.DataFrame:
    """Invert every paired quote under the naive and the parity-implied forward.

    Parameters
    ----------
    pairs:
        Output of :func:`optlab.market.quotes.pair_calls_and_puts`.
    forwards:
        Output of :func:`optlab.market.forward.fit_forward_curve`.
    naive_rate, naive_dividend_yield:
        The textbook assumption being tested.
    band:
        Optional log-moneyness restriction; the wings carry little vega and
        their implied volatilities are dominated by quote noise.
    """
    merged = pairs.merge(forwards[["expiration", "forward", "discount"]], on="expiration", how="inner")
    if band is not None:
        merged = merged[np.abs(merged["log_moneyness"]) <= band]
    merged = merged.reset_index(drop=True)
    if merged.empty:
        raise ValueError("no paired quotes left after merging forwards and applying the band")

    strike = merged["strike"].to_numpy(float)
    time_to_expiry = merged["time_to_expiry"].to_numpy(float)
    spot = merged["spot"].to_numpy(float)

    naive_forward = forward_from_spot(spot, naive_rate, time_to_expiry, naive_dividend_yield)
    naive_discount = np.exp(-naive_rate * time_to_expiry)
    fitted_forward = merged["forward"].to_numpy(float)
    fitted_discount = merged["discount"].to_numpy(float)

    result = merged[["expiration", "strike", "days_to_expiry", "log_moneyness"]].copy()
    result["forward_naive"] = naive_forward
    result["forward_parity"] = fitted_forward
    result["forward_error"] = naive_forward - fitted_forward

    for label, forward, discount in (
        ("naive", naive_forward, naive_discount),
        ("parity", fitted_forward, fitted_discount),
    ):
        for option_type in ("call", "put"):
            inverted = implied_vol(
                merged[f"mid_{option_type}"].to_numpy(float),
                forward,
                strike,
                time_to_expiry,
                discount,
                option_type,
            )
            result[f"iv_{option_type}_{label}"] = inverted.volatility
        result[f"gap_{label}"] = result[f"iv_call_{label}"] - result[f"iv_put_{label}"]

    return result


def call_put_gap_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the call/put implied volatility gap per expiry, both ways.

    The quantity that matters is the *mean* gap: a bias pushes calls and puts
    apart systematically, whereas quote noise only widens the dispersion. If
    switching to the parity forward collapses the mean gap towards zero, the
    apparent disagreement was the convention, not the market.
    """
    rows: list[dict[str, object]] = []
    for expiration, chunk in comparison.groupby("expiration", sort=True):
        row: dict[str, object] = {
            "expiration": expiration,
            "days_to_expiry": float(chunk["days_to_expiry"].iloc[0]),
            "n_pairs": int(len(chunk)),
            "forward_error": float(chunk["forward_error"].iloc[0]),
        }
        for label in ("naive", "parity"):
            gap = chunk[f"gap_{label}"].dropna()
            row[f"mean_gap_{label}"] = float(gap.mean()) if len(gap) else np.nan
            row[f"mean_abs_gap_{label}"] = float(gap.abs().mean()) if len(gap) else np.nan
            row[f"std_gap_{label}"] = float(gap.std()) if len(gap) else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["abs_gap_reduction"] = 1.0 - summary["mean_abs_gap_parity"] / summary["mean_abs_gap_naive"]
    return summary.sort_values("days_to_expiry").reset_index(drop=True)
