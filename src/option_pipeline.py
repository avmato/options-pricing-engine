"""Reusable option-chain processing pipeline."""

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


def add_time_to_expiry_columns(
    option_chain: pd.DataFrame,
) -> pd.DataFrame:
    """Add time-to-expiry values in years and days."""

    result = option_chain.copy()

    result["time_to_expiry"] = result.apply(
        lambda row: calculate_time_to_expiry(
            expiration=row["expiration"],
            valuation_timestamp=(
                row["download_timestamp"]
            ),
        ),
        axis=1,
    )

    result["days_to_expiry"] = (
        result["time_to_expiry"] * 365.0
    )

    return result


def add_theoretical_price_bounds(
    option_chain: pd.DataFrame,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Add European call and put price bounds."""

    result = option_chain.copy()

    discounted_strike = (
        result["strike"]
        * np.exp(
            -risk_free_rate
            * result["time_to_expiry"]
        )
    )

    call_rows = (
        result["option_type"] == "call"
    )

    put_rows = (
        result["option_type"] == "put"
    )

    invalid_rows = ~(
        call_rows | put_rows
    )

    if invalid_rows.any():
        raise ValueError(
            "option_type must contain only "
            "'call' or 'put'."
        )

    result["price_lower_bound"] = np.nan
    result["price_upper_bound"] = np.nan

    result.loc[
        call_rows,
        "price_lower_bound",
    ] = np.maximum(
        (
            result.loc[
                call_rows,
                "spot",
            ]
            - discounted_strike.loc[
                call_rows
            ]
        ),
        0.0,
    )

    result.loc[
        call_rows,
        "price_upper_bound",
    ] = result.loc[
        call_rows,
        "spot",
    ]

    result.loc[
        put_rows,
        "price_lower_bound",
    ] = np.maximum(
        (
            discounted_strike.loc[
                put_rows
            ]
            - result.loc[
                put_rows,
                "spot",
            ]
        ),
        0.0,
    )

    result.loc[
        put_rows,
        "price_upper_bound",
    ] = discounted_strike.loc[
        put_rows
    ]

    result["passes_price_bounds"] = (
        (
            result["mid_price"]
            >= result["price_lower_bound"]
        )
        & (
            result["mid_price"]
            <= result["price_upper_bound"]
        )
    )

    return result


def add_implied_volatility(
    option_chain: pd.DataFrame,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Calculate implied volatility for each valid row."""

    result = option_chain.copy()

    def calculate_row_iv(row):
        if not row["passes_price_bounds"]:
            return np.nan

        try:
            return implied_volatility_bisection(
                market_price=float(
                    row["mid_price"]
                ),
                spot=float(
                    row["spot"]
                ),
                strike=float(
                    row["strike"]
                ),
                rate=float(
                    risk_free_rate
                ),
                time_to_expiry=float(
                    row["time_to_expiry"]
                ),
                option_type=row["option_type"],
            )

        except (ValueError, RuntimeError):
            return np.nan

    result[
        "calculated_implied_volatility"
    ] = result.apply(
        calculate_row_iv,
        axis=1,
    )

    return result


def process_option_chain(
    raw_option_chain: pd.DataFrame,
    risk_free_rate: float = 0.04,
    maximum_relative_spread: float = 0.50,
    minimum_moneyness: float = 0.80,
    maximum_moneyness: float = 1.20,
) -> pd.DataFrame:
    """Run the complete cleaning and IV pipeline."""

    result = add_option_quality_columns(
        raw_option_chain
    )

    result = filter_option_chain(
        result,
        maximum_relative_spread=(
            maximum_relative_spread
        ),
        minimum_moneyness=(
            minimum_moneyness
        ),
        maximum_moneyness=(
            maximum_moneyness
        ),
    )

    result = add_time_to_expiry_columns(
        result
    )

    result = add_theoretical_price_bounds(
        result,
        risk_free_rate=risk_free_rate,
    )

    result = add_implied_volatility(
        result,
        risk_free_rate=risk_free_rate,
    )

    return result


def select_otm_options(
    option_chain: pd.DataFrame,
) -> pd.DataFrame:
    """Select OTM puts below spot and OTM calls above spot."""

    valid_data = option_chain[
        option_chain[
            "calculated_implied_volatility"
        ].notna()
        & option_chain[
            "passes_price_bounds"
        ]
    ].copy()

    otm_puts = valid_data[
        (
            valid_data["option_type"]
            == "put"
        )
        & (
            valid_data["strike"]
            < valid_data["spot"]
        )
    ].copy()

    otm_calls = valid_data[
        (
            valid_data["option_type"]
            == "call"
        )
        & (
            valid_data["strike"]
            >= valid_data["spot"]
        )
    ].copy()

    result = pd.concat(
        [
            otm_puts,
            otm_calls,
        ],
        ignore_index=True,
    )

    result = result.sort_values(
        [
            "days_to_expiry",
            "log_moneyness",
        ]
    ).reset_index(
        drop=True
    )

    return result


def build_surface_grid(
    otm_option_chain: pd.DataFrame,
    number_of_grid_points: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interpolate all expiries over a shared moneyness grid."""

    if number_of_grid_points < 2:
        raise ValueError(
            "number_of_grid_points must be at least 2."
        )

    if otm_option_chain.empty:
        raise ValueError(
            "otm_option_chain must not be empty."
        )

    expiry_ranges = (
        otm_option_chain
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

    common_minimum = expiry_ranges[
        "minimum_log_moneyness"
    ].max()

    common_maximum = expiry_ranges[
        "maximum_log_moneyness"
    ].min()

    if common_minimum >= common_maximum:
        raise ValueError(
            "Expirations do not share a common "
            "log-moneyness range."
        )

    log_moneyness_grid = np.linspace(
        common_minimum,
        common_maximum,
        number_of_grid_points,
    )

    surface_rows = []

    for expiration, expiry_data in (
        otm_option_chain.groupby(
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

        interpolated_iv = np.interp(
            log_moneyness_grid,
            observed_log_moneyness,
            observed_iv,
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

        for (
            log_moneyness,
            implied_volatility,
        ) in zip(
            log_moneyness_grid,
            interpolated_iv,
        ):
            surface_rows.append(
                {
                    "expiration": expiration,
                    "days_to_expiry": (
                        days_to_expiry
                    ),
                    "time_to_expiry": (
                        time_to_expiry
                    ),
                    "log_moneyness": (
                        log_moneyness
                    ),
                    (
                        "interpolated_"
                        "implied_volatility"
                    ): implied_volatility,
                }
            )

    long_grid = pd.DataFrame(
        surface_rows
    )

    long_grid = long_grid.sort_values(
        [
            "days_to_expiry",
            "log_moneyness",
        ]
    ).reset_index(
        drop=True
    )

    wide_grid = (
        long_grid
        .pivot(
            index="days_to_expiry",
            columns="log_moneyness",
            values=(
                "interpolated_"
                "implied_volatility"
            ),
        )
        .sort_index()
    )

    return long_grid, wide_grid