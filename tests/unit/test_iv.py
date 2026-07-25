"""Implied volatility solver: round-trips, wings, and failure reporting."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optlab.core.black import black76_price
from optlab.core.iv import IVStatus, implied_vol, implied_vol_bisection


@given(
    st.floats(min_value=50.0, max_value=2_000.0),
    st.floats(min_value=0.5, max_value=2.0),
    st.floats(min_value=0.02, max_value=2.0),
    st.floats(min_value=0.01, max_value=3.0),
    st.sampled_from(["call", "put"]),
)
@settings(max_examples=300, deadline=None)
def test_round_trip_recovers_the_input_volatility(
    forward, moneyness, volatility, time_to_expiry, option_type
):
    """Price at sigma, invert, get sigma back.

    The tolerance is tight on purpose: a solver that only recovers three
    digits is not good enough to measure a call/put gap of a few tenths of a
    volatility point.
    """
    strike = forward * moneyness
    discount = 0.97
    price = black76_price(forward, strike, volatility, time_to_expiry, discount, option_type)

    result = implied_vol(price, forward, strike, time_to_expiry, discount, option_type)
    if result.status[()] != IVStatus.OK:
        # Deep wings carry no vega, so the price cannot pin down a
        # volatility. The solver must say so rather than invent a number.
        assert result.status[()] in (IVStatus.NOT_IDENTIFIED, IVStatus.BELOW_INTRINSIC)
        assert np.isnan(result.volatility)
        return
    assert float(result.volatility) == pytest.approx(volatility, rel=1e-6, abs=1e-8)


def test_in_the_money_quotes_invert_as_accurately_as_out_of_the_money():
    """The parity transformation is what makes this true.

    A deep in-the-money call is mostly intrinsic value, so inverting its raw
    price means resolving a small extrinsic component inside a large number.
    Mapping to the out-of-the-money twin first removes the cancellation.
    """
    forward, time_to_expiry, discount, volatility = 500.0, 0.25, 0.99, 0.18
    strikes = np.array([250.0, 300.0, 400.0, 500.0, 600.0, 700.0])
    prices = black76_price(forward, strikes, volatility, time_to_expiry, discount, "call")

    result = implied_vol(prices, forward, strikes, time_to_expiry, discount, "call")
    recovered = result.volatility[result.ok]
    assert len(recovered) >= 5
    assert np.allclose(recovered, volatility, rtol=1e-7)


def test_solver_beats_bisection_on_accuracy():
    forward, time_to_expiry, discount, volatility = 100.0, 0.5, 0.98, 0.3
    strikes = np.linspace(60.0, 140.0, 41)
    prices = black76_price(forward, strikes, volatility, time_to_expiry, discount, "call")

    newton = implied_vol(prices, forward, strikes, time_to_expiry, discount, "call")
    bisection = implied_vol_bisection(prices, forward, strikes, time_to_expiry, discount, "call")

    newton_error = np.nanmax(np.abs(newton.volatility - volatility))
    bisection_error = np.nanmax(np.abs(bisection - volatility))
    assert newton_error < bisection_error
    assert newton_error < 1e-9


def test_unidentifiable_deep_itm_quote_is_flagged_not_guessed():
    """A quote with no vega left must return NaN, not a plausible-looking number.

    This is the failure mode that matters most in practice: a silent wrong
    answer propagates into every downstream statistic, whereas a NaN gets
    counted and reported.
    """
    result = implied_vol(
        black76_price(500.0, 250.0, 0.18, 0.25, 0.99, "call"), 500.0, 250.0, 0.25, 0.99, "call"
    )
    assert result.status[()] == IVStatus.NOT_IDENTIFIED
    assert np.isnan(result.volatility)


def test_price_below_intrinsic_is_reported_not_silently_dropped():
    result = implied_vol(1.0, 500.0, 400.0, 0.5, 1.0, "call")
    assert result.status[()] == IVStatus.BELOW_INTRINSIC
    assert np.isnan(result.volatility)


def test_price_above_upper_bound_is_reported():
    result = implied_vol(600.0, 500.0, 400.0, 0.5, 1.0, "call")
    assert result.status[()] == IVStatus.ABOVE_UPPER_BOUND


def test_expired_option_is_invalid_input():
    result = implied_vol(10.0, 500.0, 500.0, 0.0, 1.0, "call")
    assert result.status[()] == IVStatus.INVALID_INPUT


def test_summary_counts_statuses():
    forward, strikes = 100.0, np.array([80.0, 100.0, 120.0])
    prices = black76_price(forward, strikes, 0.25, 0.5, 1.0, "call")
    prices = np.append(prices, 0.0)  # a quote below intrinsic for the last strike
    strikes = np.append(strikes, 50.0)

    result = implied_vol(prices, forward, strikes, 0.5, 1.0, "call")
    summary = result.summary()
    assert summary["OK"] == 3
    assert summary["BELOW_INTRINSIC"] == 1


def test_solver_is_vectorised_over_a_whole_chain():
    forward = 500.0
    strikes = np.linspace(350.0, 650.0, 61)
    types = np.where(strikes >= forward, "call", "put")
    volatility = 0.15 + 0.4 * (np.log(strikes / forward)) ** 2  # a smile
    prices = black76_price(forward, strikes, volatility, 0.2, 0.99, types)

    result = implied_vol(prices, forward, strikes, 0.2, 0.99, types)
    assert result.ok.all()
    assert np.allclose(result.volatility, volatility, rtol=1e-7)
    assert result.iterations < 40
