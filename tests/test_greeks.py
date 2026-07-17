import numpy as np
import pytest

from src.black_scholes import black_scholes_call
from src.greeks import call_delta, gamma, put_delta, vega


SPOT = 100.0
STRIKE = 100.0
RATE = 0.04
VOLATILITY = 0.25
TIME_TO_EXPIRY = 30.0 / 365.0


def test_call_delta_known_value() -> None:
    result = call_delta(
        SPOT,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    assert result == pytest.approx(0.5325601284118957)


def test_put_delta_known_value() -> None:
    result = put_delta(
        SPOT,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    assert result == pytest.approx(-0.4674398715881043)


def test_call_delta_minus_put_delta_equals_one() -> None:
    spots = np.array([80.0, 100.0, 120.0])

    call_deltas = call_delta(
        spots,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    put_deltas = put_delta(
        spots,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    np.testing.assert_allclose(
        call_deltas - put_deltas,
        np.ones(3),
    )


def test_gamma_known_value() -> None:
    result = gamma(
        SPOT,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    assert result == pytest.approx(0.05547613305250608)


def test_vega_known_value() -> None:
    result = vega(
        SPOT,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    assert result == pytest.approx(11.399205421747824)


def test_gamma_is_positive() -> None:
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    results = gamma(
        spots,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    assert np.all(results > 0.0)


def test_vega_is_positive() -> None:
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    results = vega(
        spots,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    assert np.all(results > 0.0)


def test_analytical_delta_matches_finite_difference() -> None:
    step = 0.01

    price_up = black_scholes_call(
        SPOT + step,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    price_down = black_scholes_call(
        SPOT - step,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    numerical_delta = (price_up - price_down) / (2.0 * step)

    analytical_delta = call_delta(
        SPOT,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    assert analytical_delta == pytest.approx(
        numerical_delta,
        abs=1e-6,
    )


def test_analytical_gamma_matches_finite_difference() -> None:
    step = 0.01

    price_up = black_scholes_call(
        SPOT + step,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    price = black_scholes_call(
        SPOT,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    price_down = black_scholes_call(
        SPOT - step,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    numerical_gamma = (
        price_up - 2.0 * price + price_down
    ) / step**2

    analytical_gamma = gamma(
        SPOT,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    assert analytical_gamma == pytest.approx(
        numerical_gamma,
        abs=1e-6,
    )


def test_analytical_vega_matches_finite_difference() -> None:
    step = 0.0001

    price_up = black_scholes_call(
        SPOT,
        STRIKE,
        RATE,
        VOLATILITY + step,
        TIME_TO_EXPIRY,
    )

    price_down = black_scholes_call(
        SPOT,
        STRIKE,
        RATE,
        VOLATILITY - step,
        TIME_TO_EXPIRY,
    )

    numerical_vega = (price_up - price_down) / (2.0 * step)

    analytical_vega = vega(
        SPOT,
        STRIKE,
        RATE,
        VOLATILITY,
        TIME_TO_EXPIRY,
    )

    assert analytical_vega == pytest.approx(
        numerical_vega,
        abs=1e-6,
    )