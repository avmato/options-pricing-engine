import numpy as np
import pandas as pd
import pytest

from src.option_pipeline import (
    add_implied_volatility,
    add_theoretical_price_bounds,
    build_surface_grid,
    select_otm_options,
)


def test_add_theoretical_price_bounds():
    option_chain = pd.DataFrame(
        {
            "option_type": [
                "call",
                "put",
            ],
            "spot": [
                100.0,
                100.0,
            ],
            "strike": [
                100.0,
                100.0,
            ],
            "time_to_expiry": [
                1.0,
                1.0,
            ],
            "mid_price": [
                10.0,
                8.0,
            ],
        }
    )

    result = add_theoretical_price_bounds(
        option_chain,
        risk_free_rate=0.05,
    )

    discounted_strike = (
        100.0 * np.exp(-0.05)
    )

    assert result.loc[
        0,
        "price_lower_bound",
    ] == pytest.approx(
        max(
            100.0 - discounted_strike,
            0.0,
        )
    )

    assert result.loc[
        0,
        "price_upper_bound",
    ] == pytest.approx(
        100.0
    )

    assert result.loc[
        1,
        "price_lower_bound",
    ] == pytest.approx(
        max(
            discounted_strike - 100.0,
            0.0,
        )
    )

    assert result.loc[
        1,
        "price_upper_bound",
    ] == pytest.approx(
        discounted_strike
    )


def test_add_theoretical_price_bounds_rejects_invalid_type():
    option_chain = pd.DataFrame(
        {
            "option_type": [
                "invalid",
            ],
            "spot": [
                100.0,
            ],
            "strike": [
                100.0,
            ],
            "time_to_expiry": [
                1.0,
            ],
            "mid_price": [
                10.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="option_type",
    ):
        add_theoretical_price_bounds(
            option_chain,
            risk_free_rate=0.05,
        )


def test_add_implied_volatility_skips_invalid_price():
    option_chain = pd.DataFrame(
        {
            "option_type": [
                "call",
            ],
            "spot": [
                100.0,
            ],
            "strike": [
                100.0,
            ],
            "time_to_expiry": [
                1.0,
            ],
            "mid_price": [
                200.0,
            ],
            "passes_price_bounds": [
                False,
            ],
        }
    )

    result = add_implied_volatility(
        option_chain,
        risk_free_rate=0.05,
    )

    assert np.isnan(
        result.loc[
            0,
            "calculated_implied_volatility",
        ]
    )


def test_select_otm_options():
    option_chain = pd.DataFrame(
        {
            "option_type": [
                "put",
                "put",
                "call",
                "call",
            ],
            "strike": [
                90.0,
                110.0,
                90.0,
                110.0,
            ],
            "spot": [
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            "days_to_expiry": [
                30.0,
                30.0,
                30.0,
                30.0,
            ],
            "log_moneyness": [
                np.log(0.9),
                np.log(1.1),
                np.log(0.9),
                np.log(1.1),
            ],
            "calculated_implied_volatility": [
                0.20,
                0.20,
                0.20,
                0.20,
            ],
            "passes_price_bounds": [
                True,
                True,
                True,
                True,
            ],
        }
    )

    result = select_otm_options(
        option_chain
    )

    assert len(result) == 2

    assert set(
        zip(
            result["option_type"],
            result["strike"],
        )
    ) == {
        (
            "put",
            90.0,
        ),
        (
            "call",
            110.0,
        ),
    }


def test_build_surface_grid():
    otm_option_chain = pd.DataFrame(
        {
            "expiration": [
                "2026-08-01",
                "2026-08-01",
                "2026-09-01",
                "2026-09-01",
            ],
            "days_to_expiry": [
                10.0,
                10.0,
                40.0,
                40.0,
            ],
            "time_to_expiry": [
                10.0 / 365.0,
                10.0 / 365.0,
                40.0 / 365.0,
                40.0 / 365.0,
            ],
            "log_moneyness": [
                -0.10,
                0.10,
                -0.05,
                0.05,
            ],
            "calculated_implied_volatility": [
                0.30,
                0.20,
                0.28,
                0.22,
            ],
        }
    )

    long_grid, wide_grid = (
        build_surface_grid(
            otm_option_chain,
            number_of_grid_points=5,
        )
    )

    assert long_grid.shape == (
        10,
        5,
    )

    assert wide_grid.shape == (
        2,
        5,
    )

    assert (
        wide_grid
        .isna()
        .sum()
        .sum()
        == 0
    )


def test_build_surface_grid_rejects_too_few_points():
    otm_option_chain = pd.DataFrame(
        {
            "expiration": [
                "2026-08-01",
            ],
            "days_to_expiry": [
                10.0,
            ],
            "time_to_expiry": [
                10.0 / 365.0,
            ],
            "log_moneyness": [
                0.0,
            ],
            "calculated_implied_volatility": [
                0.20,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        build_surface_grid(
            otm_option_chain,
            number_of_grid_points=1,
        )


def test_build_surface_grid_rejects_empty_data():
    empty_data = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_surface_grid(
            empty_data
        )