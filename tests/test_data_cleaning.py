import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    add_option_quality_columns,
    filter_option_chain,
)


def make_sample_chain():
    return pd.DataFrame(
        {
            "bid": [1.0, 0.0, 2.0, 1.0],
            "ask": [1.2, 0.0, 1.5, 3.0],
            "strike": [90.0, 100.0, 110.0, 100.0],
            "spot": [100.0, 100.0, 100.0, 100.0],
            "option_type": ["call", "call", "put", "put"],
        }
    )


def test_adds_option_quality_columns():
    sample = make_sample_chain()

    result = add_option_quality_columns(sample)

    assert "mid_price" in result.columns
    assert "bid_ask_spread" in result.columns
    assert "relative_spread" in result.columns
    assert "moneyness" in result.columns
    assert "log_moneyness" in result.columns

    assert result.loc[0, "mid_price"] == pytest.approx(1.1)
    assert result.loc[0, "bid_ask_spread"] == pytest.approx(0.2)
    assert result.loc[0, "moneyness"] == pytest.approx(0.9)


def test_sets_relative_spread_to_nan_when_mid_price_is_zero():
    sample = make_sample_chain()

    result = add_option_quality_columns(sample)

    assert np.isnan(
        result.loc[1, "relative_spread"]
    )


def test_filters_invalid_quotes():
    sample = make_sample_chain()

    quality = add_option_quality_columns(sample)

    filtered = filter_option_chain(
        quality,
        maximum_relative_spread=0.50,
        minimum_moneyness=0.80,
        maximum_moneyness=1.20,
    )

    assert len(filtered) == 1
    assert filtered.loc[0, "strike"] == 90.0


def test_filters_outside_moneyness_range():
    sample = pd.DataFrame(
        {
            "bid": [1.0, 1.0],
            "ask": [1.1, 1.1],
            "strike": [70.0, 100.0],
            "spot": [100.0, 100.0],
            "option_type": ["call", "call"],
        }
    )

    quality = add_option_quality_columns(sample)

    filtered = filter_option_chain(
        quality,
        minimum_moneyness=0.80,
        maximum_moneyness=1.20,
    )

    assert len(filtered) == 1
    assert filtered.loc[0, "strike"] == 100.0