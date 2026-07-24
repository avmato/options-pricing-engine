from pathlib import Path

import numpy as np
import pandas as pd

from src.implied_volatility import implied_volatility_bisection
from src.time_utils import calculate_time_to_expiry


RISK_FREE_RATE = 0.04

input_path = Path(
    "data/clean/SPY_2026-08-21_clean_option_chain.csv"
)

option_chain = pd.read_csv(input_path)


option_chain["time_to_expiry"] = option_chain.apply(
    lambda row: calculate_time_to_expiry(
        expiration=row["expiration"],
        valuation_timestamp=row["download_timestamp"],
    ),
    axis=1,
)

option_chain["days_to_expiry"] = (
    option_chain["time_to_expiry"] * 365.0
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
            row["spot"] - discounted_strike,
            0.0,
        )

        upper_bound = row["spot"]

    elif row["option_type"] == "put":
        lower_bound = max(
            discounted_strike - row["spot"],
            0.0,
        )

        upper_bound = discounted_strike

    else:
        raise ValueError(
            "option_type must be 'call' or 'put'."
        )

    return pd.Series(
        {
            "price_lower_bound": lower_bound,
            "price_upper_bound": upper_bound,
        }
    )


price_bounds = option_chain.apply(
    calculate_price_bounds,
    axis=1,
)

option_chain = pd.concat(
    [
        option_chain,
        price_bounds,
    ],
    axis=1,
)


option_chain["passes_price_bounds"] = (
    (
        option_chain["mid_price"]
        >= option_chain["price_lower_bound"]
    )
    & (
        option_chain["mid_price"]
        <= option_chain["price_upper_bound"]
    )
)


def calculate_row_implied_volatility(row):
    if not row["passes_price_bounds"]:
        return np.nan

    try:
        return implied_volatility_bisection(
            market_price=row["mid_price"],
            spot=row["spot"],
            strike=row["strike"],
            rate=RISK_FREE_RATE,
            time_to_expiry=row["time_to_expiry"],
            option_type=row["option_type"],
        )

    except (ValueError, RuntimeError):
        return np.nan


option_chain["calculated_implied_volatility"] = (
    option_chain.apply(
        calculate_row_implied_volatility,
        axis=1,
    )
)


def add_monotonicity_flags(
    option_data,
    option_type,
):
    sorted_data = (
        option_data[
            option_data["option_type"] == option_type
        ]
        .sort_values("strike")
        .copy()
    )

    sorted_data["previous_mid_price"] = (
        sorted_data["mid_price"].shift(1)
    )

    if option_type == "call":
        sorted_data["passes_monotonicity"] = (
            sorted_data["previous_mid_price"].isna()
            | (
                sorted_data["mid_price"]
                <= sorted_data["previous_mid_price"]
            )
        )

    elif option_type == "put":
        sorted_data["passes_monotonicity"] = (
            sorted_data["previous_mid_price"].isna()
            | (
                sorted_data["mid_price"]
                >= sorted_data["previous_mid_price"]
            )
        )

    else:
        raise ValueError(
            "option_type must be 'call' or 'put'."
        )

    return sorted_data


call_data = add_monotonicity_flags(
    option_chain,
    option_type="call",
)

put_data = add_monotonicity_flags(
    option_chain,
    option_type="put",
)

option_chain = pd.concat(
    [
        call_data,
        put_data,
    ],
    ignore_index=True,
)

option_chain = option_chain.sort_values(
    [
        "option_type",
        "strike",
    ]
).reset_index(
    drop=True
)


print("Shape:")
print(option_chain.shape)

print()
print(
    option_chain[
        [
            "option_type",
            "strike",
            "expiration",
            "download_timestamp",
            "days_to_expiry",
            "time_to_expiry",
        ]
    ].head()
)

print()
print("Days-to-expiry summary:")
print(
    option_chain[
        "days_to_expiry"
    ].describe()
)

print()
print("Successful IV calculations:")
print(
    option_chain[
        "calculated_implied_volatility"
    ].notna().sum()
)

print()
print("Failed IV calculations:")
print(
    option_chain[
        "calculated_implied_volatility"
    ].isna().sum()
)

print()
print("Calculated IV summary:")
print(
    option_chain[
        "calculated_implied_volatility"
    ].describe()
)

print()
print("Rows failing theoretical price bounds:")
print(
    option_chain[
        ~option_chain["passes_price_bounds"]
    ][
        [
            "option_type",
            "strike",
            "spot",
            "mid_price",
            "price_lower_bound",
            "price_upper_bound",
        ]
    ]
)

print()
print("Call monotonicity violations:")
print(
    option_chain[
        (option_chain["option_type"] == "call")
        & (~option_chain["passes_monotonicity"])
    ][
        [
            "strike",
            "previous_mid_price",
            "mid_price",
            "calculated_implied_volatility",
        ]
    ]
)

print()
print("Put monotonicity violations:")
print(
    option_chain[
        (option_chain["option_type"] == "put")
        & (~option_chain["passes_monotonicity"])
    ][
        [
            "strike",
            "previous_mid_price",
            "mid_price",
            "calculated_implied_volatility",
        ]
    ]
)

print()
print("Number of call monotonicity violations:")
print(
    (
        (option_chain["option_type"] == "call")
        & (~option_chain["passes_monotonicity"])
    ).sum()
)

print()
print("Number of put monotonicity violations:")
print(
    (
        (option_chain["option_type"] == "put")
        & (~option_chain["passes_monotonicity"])
    ).sum()
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
    / "SPY_2026-08-21_option_chain_with_iv.csv"
)

option_chain.to_csv(
    output_path,
    index=False,
)

print()
print(
    f"Saved IV option chain to: {output_path}"
)