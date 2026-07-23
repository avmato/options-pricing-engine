"""Utilities for cleaning option-chain data."""

import numpy as np
import pandas as pd

def add_option_quality_columns(
    option_chain: pd.DataFrame,
) -> pd.DataFrame:
    """Add pricing and moneyness columns used for data cleaning."""

    cleaned = option_chain.copy()

    cleaned["mid_price"] = (
        cleaned["bid"] + cleaned["ask"]
    ) / 2.0

    cleaned["bid_ask_spread"] = (
        cleaned["ask"] - cleaned["bid"]
    )

    cleaned["relative_spread"] = np.where(
        cleaned["mid_price"] > 0,
        cleaned["bid_ask_spread"]
        / cleaned["mid_price"],
        np.nan,
    )

    cleaned["moneyness"] = (
        cleaned["strike"] / cleaned["spot"]
    )

    cleaned["log_moneyness"] = np.log(
        cleaned["moneyness"]
    )

    return cleaned

def filter_option_chain(
    option_chain: pd.DataFrame,
    maximum_relative_spread: float = 0.50,
    minimum_moneyness: float = 0.80,
    maximum_moneyness: float = 1.20,
) -> pd.DataFrame:
    """Filter invalid and low-quality option quotes."""

    filtered = option_chain.copy()

    valid_quotes = (
        (filtered["bid"] > 0)
        & (filtered["ask"] > 0)
        & (filtered["ask"] >= filtered["bid"])
        & (filtered["mid_price"] > 0)
    )

    acceptable_spread = (
        filtered["relative_spread"]
        <= maximum_relative_spread
    )

    acceptable_moneyness = (
        (filtered["moneyness"] >= minimum_moneyness)
        & (filtered["moneyness"] <= maximum_moneyness)
    )

    filtered = filtered[
        valid_quotes
        & acceptable_spread
        & acceptable_moneyness
    ].copy()

    filtered = filtered.reset_index(
        drop=True
    )

    return filtered

