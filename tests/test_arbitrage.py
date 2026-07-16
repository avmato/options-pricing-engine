import numpy as np
import pytest

from src.arbitrage import (
    call_price_bounds,
    discounted_strike,
    put_call_parity_residual,
    put_price_bounds,
)


def test_discounted_strike_at_zero_rate() -> None:
    result = discounted_strike(
        strike=100.0,
        rate=0.0,
        time_to_expiry=1.0,
    )

    assert result == pytest.approx(100.0)


def test_discounted_strike_is_less_than_strike_for_positive_rate() -> None:
    result = discounted_strike(
        strike=100.0,
        rate=0.05,
        time_to_expiry=1.0,
    )

    assert result < 100.0


def test_call_price_bounds() -> None:
    lower, upper = call_price_bounds(
        spot=100.0,
        strike=105.0,
        rate=0.04,
        time_to_expiry=30.0 / 365.0,
    )

    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(100.0)


def test_put_price_bounds() -> None:
    lower, upper = put_price_bounds(
        spot=100.0,
        strike=105.0,
        rate=0.04,
        time_to_expiry=30.0 / 365.0,
    )

    assert lower == pytest.approx(4.65536136024572)
    assert upper == pytest.approx(104.65536136024572)


def test_put_call_parity_residual_is_zero() -> None:
    spot = 100.0
    strike = 105.0
    rate = 0.04
    time_to_expiry = 30.0 / 365.0
    call_price = 2.5

    pv_strike = discounted_strike(
        strike,
        rate,
        time_to_expiry,
    )

    put_price = call_price - spot + pv_strike

    residual = put_call_parity_residual(
        call_price,
        put_price,
        spot,
        strike,
        rate,
        time_to_expiry,
    )

    assert residual == pytest.approx(0.0)


def test_put_call_parity_vectorized() -> None:
    spot = 100.0
    strikes = np.array([80.0, 100.0, 120.0])
    rate = 0.04
    time_to_expiry = 30.0 / 365.0

    call_prices = np.array([20.3, 3.0, 0.02])

    pv_strikes = discounted_strike(
        strikes,
        rate,
        time_to_expiry,
    )

    put_prices = call_prices - spot + pv_strikes

    residuals = put_call_parity_residual(
        call_prices,
        put_prices,
        spot,
        strikes,
        rate,
        time_to_expiry,
    )

    np.testing.assert_allclose(
        residuals,
        np.zeros(3),
        atol=1e-12,
    )