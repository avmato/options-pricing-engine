from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/SPY_multi_expiry_otm_smiles.csv"
)

OUTPUT_DIRECTORY = Path(
    "data/processed"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

LONG_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "SPY_iv_surface_grid_long.csv"
)

WIDE_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "SPY_iv_surface_grid_wide.csv"
)


NUMBER_OF_GRID_POINTS = 60


otm_smile_data = pd.read_csv(
    INPUT_PATH
)


valid_surface_data = otm_smile_data[
    otm_smile_data[
        "calculated_implied_volatility"
    ].notna()
].copy()


expiry_ranges = (
    valid_surface_data
    .groupby("expiration")
    .agg(
        minimum_log_moneyness=(
            "log_moneyness",
            "min",
        ),
        maximum_log_moneyness=(
            "log_moneyness",
            "max",
        ),
    )
)


common_minimum_log_moneyness = (
    expiry_ranges[
        "minimum_log_moneyness"
    ].max()
)

common_maximum_log_moneyness = (
    expiry_ranges[
        "maximum_log_moneyness"
    ].min()
)


if (
    common_minimum_log_moneyness
    >= common_maximum_log_moneyness
):
    raise ValueError(
        "The expirations do not have a common "
        "log-moneyness range."
    )


log_moneyness_grid = np.linspace(
    common_minimum_log_moneyness,
    common_maximum_log_moneyness,
    NUMBER_OF_GRID_POINTS,
)


surface_rows = []


for expiration, expiry_data in (
    valid_surface_data.groupby(
        "expiration"
    )
):
    expiry_data = (
        expiry_data
        .sort_values(
            "log_moneyness"
        )
        .drop_duplicates(
            subset="log_moneyness"
        )
        .copy()
    )

    observed_log_moneyness = (
        expiry_data[
            "log_moneyness"
        ].to_numpy()
    )

    observed_iv = (
        expiry_data[
            "calculated_implied_volatility"
        ].to_numpy()
    )

    days_to_expiry = float(
        expiry_data[
            "days_to_expiry"
        ].iloc[0]
    )

    time_to_expiry = float(
        expiry_data[
            "time_to_expiry"
        ].iloc[0]
    )

    interpolated_iv = np.interp(
        log_moneyness_grid,
        observed_log_moneyness,
        observed_iv,
    )

    for log_moneyness, implied_volatility in zip(
        log_moneyness_grid,
        interpolated_iv,
    ):
        surface_rows.append(
            {
                "expiration": expiration,
                "days_to_expiry": days_to_expiry,
                "time_to_expiry": time_to_expiry,
                "log_moneyness": log_moneyness,
                "interpolated_implied_volatility": (
                    implied_volatility
                ),
            }
        )


surface_grid_long = pd.DataFrame(
    surface_rows
)


surface_grid_long = (
    surface_grid_long
    .sort_values(
        [
            "days_to_expiry",
            "log_moneyness",
        ]
    )
    .reset_index(
        drop=True
    )
)


surface_grid_wide = (
    surface_grid_long
    .pivot(
        index="days_to_expiry",
        columns="log_moneyness",
        values=(
            "interpolated_implied_volatility"
        ),
    )
    .sort_index()
)


surface_grid_long.to_csv(
    LONG_OUTPUT_PATH,
    index=False,
)


surface_grid_wide.to_csv(
    WIDE_OUTPUT_PATH
)


print("Input OTM rows:")
print(
    len(otm_smile_data)
)

print()
print("Expiry log-moneyness ranges:")
print(
    expiry_ranges
)

print()
print("Common log-moneyness range:")
print(
    (
        common_minimum_log_moneyness,
        common_maximum_log_moneyness,
    )
)

print()
print("Number of grid points:")
print(
    NUMBER_OF_GRID_POINTS
)

print()
print("Surface grid long shape:")
print(
    surface_grid_long.shape
)

print()
print("Surface grid wide shape:")
print(
    surface_grid_wide.shape
)

print()
print("Grid rows by expiration:")
print(
    surface_grid_long.groupby(
        "expiration"
    ).size()
)

print()
print("Missing IV values in long grid:")
print(
    surface_grid_long[
        "interpolated_implied_volatility"
    ].isna().sum()
)

print()
print("Missing IV values in wide grid:")
print(
    surface_grid_wide.isna().sum().sum()
)

print()
print(
    f"Saved long surface grid to: "
    f"{LONG_OUTPUT_PATH}"
)

print(
    f"Saved wide surface grid to: "
    f"{WIDE_OUTPUT_PATH}"
)