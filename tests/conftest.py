"""Shared fixtures.

The synthetic chain is the backbone of the test suite. It is generated from
Black-76 prices under a *flat* volatility, which makes it arbitrage-free by
construction: every bound, monotonicity, convexity, box and calendar relation
holds exactly. Any violation the audit reports on this chain is a bug in the
audit, and any violation it fails to report after one is injected is a
blind spot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optlab.core.black import black76_price

VALUATION = pd.Timestamp("2026-03-02 15:00:00", tz="UTC")


def make_chain(
    *,
    spot: float = 500.0,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    volatility: float = 0.20,
    strikes: np.ndarray | None = None,
    expiries: tuple[str, ...] = ("2026-03-20", "2026-04-17", "2026-06-19"),
    half_spread: float = 0.05,
    valuation: pd.Timestamp = VALUATION,
) -> pd.DataFrame:
    """Build an arbitrage-free option chain in the package's raw schema.

    With ``rate == dividend_yield == 0`` the discount factor is one and the
    forward equals the spot, so European and American prices coincide and
    every check in :mod:`optlab.audit` must pass exactly. Non-zero rates are
    used by the forward-recovery tests, where the European calendar relation
    no longer has to hold for deep in-the-money puts.
    """
    if strikes is None:
        strikes = np.arange(400.0, 601.0, 5.0)

    rows: list[pd.DataFrame] = []
    for expiry in expiries:
        expiry_stamp = pd.Timestamp(expiry, tz="America/New_York") + pd.Timedelta(hours=16)
        time_to_expiry = (expiry_stamp - valuation).total_seconds() / (365.0 * 86400.0)
        forward = spot * np.exp((rate - dividend_yield) * time_to_expiry)
        discount = np.exp(-rate * time_to_expiry)

        for option_type in ("call", "put"):
            mid = black76_price(forward, strikes, volatility, time_to_expiry, discount, option_type)
            # Narrow the spread rather than clamping the bid at zero: clamping
            # would move the midpoint and make the chain fail put-call parity
            # for cheap wing options, which is exactly what the tests check.
            leg_half_spread = np.minimum(half_spread, 0.9 * mid)
            rows.append(
                pd.DataFrame(
                    {
                        "strike": strikes,
                        "bid": mid - leg_half_spread,
                        "ask": mid + leg_half_spread,
                        "option_type": option_type,
                        "expiration": pd.Timestamp(expiry),
                        "spot": spot,
                        "download_timestamp": valuation,
                        "volume": 100.0,
                        "open_interest": 1000.0,
                        "last_trade_timestamp": valuation - pd.Timedelta(hours=1),
                    }
                )
            )

    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def clean_chain() -> pd.DataFrame:
    """Arbitrage-free chain with zero rates: every static relation holds exactly."""
    return make_chain()


@pytest.fixture
def rate_chain() -> pd.DataFrame:
    """Arbitrage-free chain with a non-zero rate and dividend yield."""
    return make_chain(rate=0.045, dividend_yield=0.013, half_spread=0.02)
