from pathlib import Path

import numpy as np
import pandas as pd

from src.data_cleaning import (
    add_option_quality_columns,
    filter_option_chain,
)
from src.implied_volatility import (
    implied_volatility_bisection,
)
from src.time_utils import (
    calculate_time_to_expiry,
)


RISK_FREE_RATE = 0.04

input_path = Path(
    "data/raw/multiple_expiries/"
    "SPY_multiple_expiries_raw.csv"
)

output_directory = Path(
    "data/processed"
)

output_directory.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = (
    output_directory
    / "SPY_multiple_expiries_with_iv.csv"
)


raw_option_chain = pd.read_csv(
    input_path
)


quality_option_chain = (
    add_option_quality_columns(
        raw_option_chain
    )
)


cleaned_option_chain = filter_option_chain(
    quality_option_chain,
    maximum_relative_spread=0.50,
    minimum_moneyness=0.80,
    maximum_moneyness=1.20,
)


cleaned_option_chain[
    "time_to_expiry"
] = cleaned_option_chain.apply(
    lambda row: calculate_time_to_expiry(
        expiration=row["expiration"],
        valuation_timestamp=(
            row["download_timestamp"]
        ),
    ),
    axis=1,
)


cleaned_option_chain[
    "days_to_expiry"
] = (
    cleaned_option_chain[
        "time_to_expiry"
    ]
    * 365.0
)


def calculate_price_bounds(row):
    discounted_strike = (
        row["strike"]
        * np.exp(
            -RISK_FREE_RATE
            * row["time_to_expiry"]
        )
    )

    if row["option_type"] == "call":
        lower_bound = max(
            row["spot"]
            - discounted_strike,
            0.0,
        )

        upper_bound = row["spot"]

    elif row["option_type"] == "put":
        lower_bound = max(
            discounted_strike
            - row["spot"],
            0.0,
        )

        upper_bound = discounted_strike

    else:
        raise ValueError(
            "option_type must be "
            "'call' or 'put'."
        )

    return pd.Series(
        {
            "price_lower_bound": (
                lower_bound
            ),
            "price_upper_bound": (
                upper_bound
            ),
        }
    )


price_bounds = cleaned_option_chain.apply(
    calculate_price_bounds,
    axis=1,
)


cleaned_option_chain = pd.concat(
    [
        cleaned_option_chain,
        price_bounds,
    ],
    axis=1,
)


cleaned_option_chain[
    "passes_price_bounds"
] = (
    (
        cleaned_option_chain[
            "mid_price"
        ]
        >= cleaned_option_chain[
            "price_lower_bound"
        ]
    )
    & (
        cleaned_option_chain[
            "mid_price"
        ]
        <= cleaned_option_chain[
            "price_upper_bound"
        ]
    )
)


def calculate_row_iv(row):
    if not row["passes_price_bounds"]:
        return np.nan

    try:
        return implied_volatility_bisection(
            market_price=row["mid_price"],
            spot=row["spot"],
            strike=row["strike"],
            rate=RISK_FREE_RATE,
            time_to_expiry=(
                row["time_to_expiry"]
            ),
            option_type=row["option_type"],
        )

    except (ValueError, RuntimeError):
        return np.nan


cleaned_option_chain[
    "calculated_implied_volatility"
] = cleaned_option_chain.apply(
    calculate_row_iv,
    axis=1,
)


cleaned_option_chain.to_csv(
    output_path,
    index=False,
)


print("Raw shape:")
print(raw_option_chain.shape)

print()
print("Shape after quality columns:")
print(quality_option_chain.shape)

print()
print("Cleaned shape:")
print(cleaned_option_chain.shape)

print()
print("Rows by expiration:")
print(
    cleaned_option_chain.groupby(
        "expiration"
    ).size()
)

print()
print("Days to expiry by expiration:")
print(
    cleaned_option_chain.groupby(
        "expiration"
    )["days_to_expiry"].first()
)

print()
print("Successful IV calculations by expiration:")
print(
    cleaned_option_chain.groupby(
        "expiration"
    )[
        "calculated_implied_volatility"
    ].apply(
        lambda values: (
            values.notna().sum()
        )
    )
)

print()
print("Failed IV calculations by expiration:")
print(
    cleaned_option_chain.groupby(
        "expiration"
    )[
        "calculated_implied_volatility"
    ].apply(
        lambda values: (
            values.isna().sum()
        )
    )
)

print()
print("Rows failing price bounds:")
print(
    (
        ~cleaned_option_chain[
            "passes_price_bounds"
        ]
    ).sum()
)

print()
print("Calculated IV summary:")
print(
    cleaned_option_chain[
        "calculated_implied_volatility"
    ].describe()
)

print()
print(
    f"Saved processed data to: "
    f"{output_path}"
)