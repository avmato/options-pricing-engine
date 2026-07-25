from pathlib import Path

import pandas as pd

from src.option_pipeline import (
    build_surface_grid,
    process_option_chain,
    select_otm_options,
)


RISK_FREE_RATE = 0.04
NUMBER_OF_GRID_POINTS = 60

RAW_INPUT_PATH = Path(
    "data/raw/multiple_expiries/"
    "SPY_multiple_expiries_raw.csv"
)

OUTPUT_DIRECTORY = Path(
    "data/processed/pipeline"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_CHAIN_PATH = (
    OUTPUT_DIRECTORY
    / "SPY_processed_option_chain.csv"
)

OTM_CHAIN_PATH = (
    OUTPUT_DIRECTORY
    / "SPY_otm_option_chain.csv"
)

SURFACE_LONG_PATH = (
    OUTPUT_DIRECTORY
    / "SPY_surface_grid_long.csv"
)

SURFACE_WIDE_PATH = (
    OUTPUT_DIRECTORY
    / "SPY_surface_grid_wide.csv"
)


def main():
    print("Reading raw option-chain data...")

    raw_option_chain = pd.read_csv(
        RAW_INPUT_PATH
    )

    print(
        f"Raw shape: {raw_option_chain.shape}"
    )

    print()
    print("Processing option chain...")

    processed_option_chain = process_option_chain(
        raw_option_chain=raw_option_chain,
        risk_free_rate=RISK_FREE_RATE,
        maximum_relative_spread=0.50,
        minimum_moneyness=0.80,
        maximum_moneyness=1.20,
    )

    print(
        "Processed shape: "
        f"{processed_option_chain.shape}"
    )

    print()
    print("Selecting OTM options...")

    otm_option_chain = select_otm_options(
        processed_option_chain
    )

    print(
        f"OTM shape: {otm_option_chain.shape}"
    )

    print()
    print("Building IV surface grid...")

    surface_grid_long, surface_grid_wide = (
        build_surface_grid(
            otm_option_chain=otm_option_chain,
            number_of_grid_points=(
                NUMBER_OF_GRID_POINTS
            ),
        )
    )

    print(
        "Long surface shape: "
        f"{surface_grid_long.shape}"
    )

    print(
        "Wide surface shape: "
        f"{surface_grid_wide.shape}"
    )

    print()
    print("Saving outputs...")

    processed_option_chain.to_csv(
        PROCESSED_CHAIN_PATH,
        index=False,
    )

    otm_option_chain.to_csv(
        OTM_CHAIN_PATH,
        index=False,
    )

    surface_grid_long.to_csv(
        SURFACE_LONG_PATH,
        index=False,
    )

    surface_grid_wide.to_csv(
        SURFACE_WIDE_PATH,
    )

    successful_iv_count = (
        processed_option_chain[
            "calculated_implied_volatility"
        ]
        .notna()
        .sum()
    )

    failed_iv_count = (
        processed_option_chain[
            "calculated_implied_volatility"
        ]
        .isna()
        .sum()
    )

    price_bound_failures = (
        ~processed_option_chain[
            "passes_price_bounds"
        ]
    ).sum()

    print()
    print("Pipeline summary:")
    print(
        f"Raw rows: {len(raw_option_chain)}"
    )

    print(
        "Rows after cleaning: "
        f"{len(processed_option_chain)}"
    )

    print(
        "Successful IV calculations: "
        f"{successful_iv_count}"
    )

    print(
        "Failed IV calculations: "
        f"{failed_iv_count}"
    )

    print(
        "Rows failing price bounds: "
        f"{price_bound_failures}"
    )

    print(
        f"OTM rows: {len(otm_option_chain)}"
    )

    print(
        "Surface missing values: "
        f"{surface_grid_wide.isna().sum().sum()}"
    )

    print()
    print("Rows by expiration:")
    print(
        processed_option_chain.groupby(
            "expiration"
        ).size()
    )

    print()
    print("Successful IV calculations by expiration:")
    print(
        processed_option_chain.groupby(
            "expiration"
        )[
            "calculated_implied_volatility"
        ].apply(
            lambda values: values.notna().sum()
        )
    )

    print()
    print("Saved files:")
    print(PROCESSED_CHAIN_PATH)
    print(OTM_CHAIN_PATH)
    print(SURFACE_LONG_PATH)
    print(SURFACE_WIDE_PATH)

    print()
    print("End-to-end pipeline completed successfully.")


if __name__ == "__main__":
    main()