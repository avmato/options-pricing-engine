"""Utilities for option expiration times."""

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd


NEW_YORK_TIMEZONE = ZoneInfo(
    "America/New_York"
)


def calculate_time_to_expiry(
    expiration,
    valuation_timestamp,
) -> float:
    """Return time to expiry in years."""

    expiration_date = pd.to_datetime(
        expiration
    ).date()

    expiration_datetime = datetime.combine(
        expiration_date,
        time(hour=16),
        tzinfo=NEW_YORK_TIMEZONE,
    )

    valuation_datetime = pd.to_datetime(
        valuation_timestamp,
        utc=True,
    ).to_pydatetime()

    remaining_seconds = (
        expiration_datetime
        - valuation_datetime
    ).total_seconds()

    if remaining_seconds <= 0:
        raise ValueError(
            "Expiration must be after the valuation timestamp."
        )

    seconds_per_year = (
        365.0 * 24.0 * 60.0 * 60.0
    )

    return remaining_seconds / seconds_per_year