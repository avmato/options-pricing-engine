"""Yahoo Finance option-chain snapshots.

The download timestamp is recorded on every row. Option quotes go stale the
moment the market closes and the vendor's spot price is not necessarily
sampled at the same instant as the option quotes, so a snapshot without a
timestamp cannot be audited later -- you would have no way to tell a real
pricing anomaly from two prices taken minutes apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["download_chain", "select_expiries"]

COLUMN_MAP = {
    "strike": "strike",
    "bid": "bid",
    "ask": "ask",
    "lastPrice": "last_price",
    "volume": "volume",
    "openInterest": "open_interest",
    "impliedVolatility": "vendor_implied_volatility",
}


def select_expiries(available: list[str], target_days: list[int], reference: pd.Timestamp) -> list[str]:
    """Pick the listed expiry closest to each target maturity.

    Consecutive weeklies are nearly identical, so taking the five nearest
    expiries would give five views of the same maturity. Anchoring on target
    maturities spreads the sample across the term structure instead.
    """
    if not available:
        return []
    expiry_dates = pd.to_datetime(pd.Series(available))
    days = (expiry_dates - reference.normalize().tz_localize(None)).dt.days.to_numpy()

    chosen: list[str] = []
    for target in target_days:
        eligible = np.flatnonzero(days > 0)
        if eligible.size == 0:
            break
        best = eligible[np.argmin(np.abs(days[eligible] - target))]
        candidate = available[best]
        if candidate not in chosen:
            chosen.append(candidate)
    return chosen


def download_chain(
    ticker: str = "SPY",
    target_days: list[int] | None = None,
) -> pd.DataFrame:
    """Download option chains for the expiries nearest the target maturities.

    Returns the tidy schema the rest of the package expects: one row per
    contract, with ``option_type``, ``expiration``, ``spot`` and
    ``download_timestamp`` attached.
    """
    import yfinance as yf

    target_days = target_days or [7, 14, 30, 60, 90]
    handle = yf.Ticker(ticker)
    stamp = pd.Timestamp.utcnow()

    history = handle.history(period="1d", interval="1m")
    if history.empty:
        history = handle.history(period="5d")
    spot = float(history["Close"].iloc[-1])

    expiries = select_expiries(list(handle.options), target_days, stamp)
    if not expiries:
        raise RuntimeError(f"no expiries listed for {ticker}")

    frames: list[pd.DataFrame] = []
    for expiry in expiries:
        chain = handle.option_chain(expiry)
        for option_type, side in (("call", chain.calls), ("put", chain.puts)):
            frame = side.rename(columns=COLUMN_MAP)
            keep = [column for column in COLUMN_MAP.values() if column in frame.columns]
            frame = frame[keep].copy()
            frame["option_type"] = option_type
            frame["expiration"] = pd.Timestamp(expiry)
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["ticker"] = ticker
    combined["spot"] = spot
    combined["download_timestamp"] = stamp.isoformat()
    return combined
