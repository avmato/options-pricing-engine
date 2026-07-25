"""Time-to-expiry conventions.

US equity options stop trading at 16:00 New York time on the expiration date,
not at midnight UTC. For a one-week option the difference between the two
conventions is about 4% of the remaining life, which moves the implied
volatility by roughly half a volatility point -- larger than most of the
effects this project is trying to measure. The convention therefore gets its
own module and its own tests.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

__all__ = [
    "EXCHANGE_TIMEZONE",
    "MARKET_CLOSE_HOUR",
    "DAYS_PER_YEAR",
    "expiry_timestamp",
    "year_fraction",
]

EXCHANGE_TIMEZONE = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR = 16
DAYS_PER_YEAR = 365.0


def expiry_timestamp(expiration: pd.Series | pd.Timestamp | str) -> pd.Series | pd.Timestamp:
    """Attach the 16:00 New York close to a calendar expiration date.

    Handles the daylight-saving shift automatically by localising in the
    exchange timezone and then converting to UTC, rather than hard-coding a
    UTC offset.
    """
    parsed = pd.to_datetime(expiration)
    scalar_input = isinstance(parsed, pd.Timestamp)
    stamps: pd.Series = pd.Series([parsed]) if scalar_input else pd.Series(parsed)

    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_convert(EXCHANGE_TIMEZONE).dt.tz_localize(None)
    stamps = stamps.dt.normalize() + pd.Timedelta(hours=MARKET_CLOSE_HOUR)
    stamps = stamps.dt.tz_localize(EXCHANGE_TIMEZONE).dt.tz_convert("UTC")

    return stamps.iloc[0] if scalar_input else stamps


def year_fraction(
    expiration: pd.Series | pd.Timestamp | str,
    valuation: pd.Series | pd.Timestamp | str,
    days_per_year: float = DAYS_PER_YEAR,
) -> pd.Series | float:
    """ACT/365 year fraction between a valuation timestamp and the expiry close.

    Returns a negative number for an expiry already in the past; callers are
    expected to filter those rather than have them silently clamped.
    """
    expiry = expiry_timestamp(expiration)
    valuation_stamps = pd.to_datetime(valuation, utc=True)

    delta = expiry - valuation_stamps
    if isinstance(delta, pd.Timedelta):
        return float(delta.total_seconds()) / (days_per_year * 86400.0)
    return delta.dt.total_seconds() / (days_per_year * 86400.0)
