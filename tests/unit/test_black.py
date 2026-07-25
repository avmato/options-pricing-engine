"""Black-76 pricing: identities, bounds, and degenerate cases."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from optlab.core.black import (
    black76_price,
    bs_price,
    intrinsic_value,
    option_sign,
    parity_gap,
    price_bounds,
)

forwards = st.floats(min_value=1.0, max_value=5_000.0, allow_nan=False)
strikes = st.floats(min_value=1.0, max_value=5_000.0, allow_nan=False)
vols = st.floats(min_value=1e-3, max_value=3.0, allow_nan=False)
times = st.floats(min_value=1e-4, max_value=5.0, allow_nan=False)
rates = st.floats(min_value=-0.02, max_value=0.15, allow_nan=False)


def test_option_sign_rejects_unknown_type():
    with pytest.raises(ValueError, match="call.*put"):
        option_sign("straddle")


@given(forwards, strikes, vols, times, rates)
@settings(max_examples=200, deadline=None)
def test_put_call_parity_holds_exactly(forward, strike, volatility, time_to_expiry, rate):
    """C - P = D(F - K) is an identity, not an approximation.

    Because both legs go through one signed code path, this holds to floating
    point precision rather than to some tolerance.
    """
    discount = np.exp(-rate * time_to_expiry)
    call = black76_price(forward, strike, volatility, time_to_expiry, discount, "call")
    put = black76_price(forward, strike, volatility, time_to_expiry, discount, "put")

    gap = parity_gap(call, put, forward, strike, discount)
    scale = max(abs(float(call)), abs(float(put)), 1.0)
    assert abs(float(gap)) < 1e-9 * scale


@given(forwards, strikes, vols, times, rates, st.sampled_from(["call", "put"]))
@settings(max_examples=200, deadline=None)
def test_price_inside_no_arbitrage_bounds(forward, strike, volatility, time_to_expiry, rate, option_type):
    discount = np.exp(-rate * time_to_expiry)
    price = black76_price(forward, strike, volatility, time_to_expiry, discount, option_type)
    lower, upper = price_bounds(forward, strike, discount, option_type)

    assert float(price) >= float(lower) - 1e-9
    assert float(price) <= float(upper) + 1e-9


@given(forwards, strikes, times, rates, st.sampled_from(["call", "put"]))
@settings(max_examples=100, deadline=None)
def test_zero_volatility_collapses_to_intrinsic(forward, strike, time_to_expiry, rate, option_type):
    discount = np.exp(-rate * time_to_expiry)
    price = black76_price(forward, strike, 0.0, time_to_expiry, discount, option_type)
    expected = intrinsic_value(forward, strike, discount, option_type)
    assert float(price) == pytest.approx(float(expected), abs=1e-10)


@given(forwards, strikes, vols, times, st.sampled_from(["call", "put"]))
@settings(max_examples=200, deadline=None)
def test_price_is_increasing_in_volatility(forward, strike, volatility, time_to_expiry, option_type):
    """Vega is non-negative, so raising volatility can never lower the price.

    This is the monotonicity the implied-volatility solver relies on; if it
    ever failed, bracketing would be meaningless.
    """
    lower = black76_price(forward, strike, volatility, time_to_expiry, 1.0, option_type)
    higher = black76_price(forward, strike, volatility * 1.05, time_to_expiry, 1.0, option_type)
    assert float(higher) >= float(lower) - 1e-12


@given(
    forwards,
    st.floats(min_value=1.0, max_value=4_000.0),
    st.floats(min_value=1.01, max_value=1.5),
    vols,
    times,
)
@settings(max_examples=150, deadline=None)
def test_call_price_falls_and_put_price_rises_with_strike(forward, strike, ratio, volatility, time_to_expiry):
    higher_strike = strike * ratio
    assume(higher_strike <= 5_000.0)

    call_low = black76_price(forward, strike, volatility, time_to_expiry, 1.0, "call")
    call_high = black76_price(forward, higher_strike, volatility, time_to_expiry, 1.0, "call")
    put_low = black76_price(forward, strike, volatility, time_to_expiry, 1.0, "put")
    put_high = black76_price(forward, higher_strike, volatility, time_to_expiry, 1.0, "put")

    assert float(call_high) <= float(call_low) + 1e-10
    assert float(put_high) >= float(put_low) - 1e-10


def test_convexity_in_strike():
    """Butterfly prices are non-negative, i.e. the implied density is a density."""
    strikes = np.arange(50.0, 151.0, 1.0)
    prices = black76_price(100.0, strikes, 0.25, 0.5, 0.98, "call")
    second_difference = prices[:-2] - 2.0 * prices[1:-1] + prices[2:]
    assert np.all(second_difference >= -1e-12)


def test_matches_published_black_scholes_value():
    """Hull, 8th edition, chapter 15: S=42, K=40, r=10%, sigma=20%, T=0.5."""
    call = bs_price(42.0, 40.0, 0.10, 0.20, 0.5, 0.0, "call")
    put = bs_price(42.0, 40.0, 0.10, 0.20, 0.5, 0.0, "put")
    assert float(call) == pytest.approx(4.76, abs=0.005)
    assert float(put) == pytest.approx(0.81, abs=0.005)


def test_vectorises_over_mixed_option_types():
    """A mixed call/put array must price in one pass, not row by row."""
    types = np.array(["call", "put", "call", "put"])
    strikes = np.array([90.0, 90.0, 110.0, 110.0])
    prices = black76_price(100.0, strikes, 0.2, 1.0, 0.95, types)

    expected = [
        float(black76_price(100.0, strike, 0.2, 1.0, 0.95, option_type))
        for strike, option_type in zip(strikes, types, strict=True)
    ]
    assert prices == pytest.approx(expected)
