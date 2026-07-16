import numpy as np
import pytest

from src.black_scholes import (
    black_scholes_call,
    black_scholes_put,
    calculate_d1_d2,
)


def test_d1_d2_known_values() -> None:
    d1, d2 = calculate_d1_d2(
        spot=100.0,
        strike=105.0,
        rate=0.04,
        volatility=0.25,
        time_to_expiry=30.0 / 365.0,
    )

    assert d1 == pytest.approx(-0.5990280319445445)
    assert d2 == pytest.approx(-0.6707008043296689)


def test_d1_minus_d2_equals_sigma_sqrt_t() -> None:
    volatility = 0.25
    time_to_expiry = 30.0 / 365.0

    d1, d2 = calculate_d1_d2(
        spot=100.0,
        strike=105.0,
        rate=0.04,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
    )

    expected_difference = volatility * np.sqrt(time_to_expiry)

    assert d1 - d2 == pytest.approx(expected_difference)


def test_black_scholes_known_prices() -> None:
    call_price = black_scholes_call(
        spot=100.0,
        strike=105.0,
        rate=0.04,
        volatility=0.25,
        time_to_expiry=30.0 / 365.0,
    )

    put_price = black_scholes_put(
        spot=100.0,
        strike=105.0,
        rate=0.04,
        volatility=0.25,
        time_to_expiry=30.0 / 365.0,
    )

    assert call_price == pytest.approx(1.1676993366674289)
    assert put_price == pytest.approx(5.82306069691316)


def test_call_prices_decrease_with_strike() -> None:
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    call_prices = black_scholes_call(
        spot=100.0,
        strike=strikes,
        rate=0.04,
        volatility=0.25,
        time_to_expiry=30.0 / 365.0,
    )

    assert np.all(np.diff(call_prices) < 0)


def test_put_prices_increase_with_strike() -> None:
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    put_prices = black_scholes_put(
        spot=100.0,
        strike=strikes,
        rate=0.04,
        volatility=0.25,
        time_to_expiry=30.0 / 365.0,
    )

    assert np.all(np.diff(put_prices) > 0)


def test_put_call_parity_for_black_scholes_prices() -> None:
    spot = 100.0
    strikes = np.array([80.0, 100.0, 120.0])
    rate = 0.04
    volatility = 0.25
    time_to_expiry = 30.0 / 365.0

    call_prices = black_scholes_call(
        spot,
        strikes,
        rate,
        volatility,
        time_to_expiry,
    )

    put_prices = black_scholes_put(
        spot,
        strikes,
        rate,
        volatility,
        time_to_expiry,
    )

    parity_left = call_prices - put_prices
    parity_right = spot - strikes * np.exp(-rate * time_to_expiry)

    np.testing.assert_allclose(
        parity_left,
        parity_right,
        atol=1e-12,
    )

def test_rejects_nonpositive_spot() -> None:
    with pytest.raises(ValueError):
        calculate_d1_d2(
            spot=0.0,
            strike=100.0,
            rate=0.04,
            volatility=0.25,
            time_to_expiry=1.0,
        )


def test_rejects_nonpositive_strike() -> None:
    with pytest.raises(ValueError):
        calculate_d1_d2(
            spot=100.0,
            strike=0.0,
            rate=0.04,
            volatility=0.25,
            time_to_expiry=1.0,
        )


def test_rejects_nonpositive_volatility() -> None:
    with pytest.raises(ValueError):
        calculate_d1_d2(
            spot=100.0,
            strike=100.0,
            rate=0.04,
            volatility=0.0,
            time_to_expiry=1.0,
        )


def test_rejects_nonpositive_time_to_expiry() -> None:
    with pytest.raises(ValueError):
        calculate_d1_d2(
            spot=100.0,
            strike=100.0,
            rate=0.04,
            volatility=0.25,
            time_to_expiry=0.0,
        )

def test_rejects_invalid_value_inside_array() -> None:
    strikes = np.array([80.0, 100.0, -120.0])

    with pytest.raises(ValueError):
        calculate_d1_d2(
            spot=100.0,
            strike=strikes,
            rate=0.04,
            volatility=0.25,
            time_to_expiry=1.0,
        )