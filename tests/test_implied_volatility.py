import pytest

from src.black_scholes import (
    black_scholes_call,
    black_scholes_put,
)
from src.implied_volatility import implied_volatility_bisection


def test_recovers_call_implied_volatility():
    spot = 100.0
    strike = 100.0
    rate = 0.04
    volatility = 0.25
    time_to_expiry = 30.0 / 365.0

    market_price = black_scholes_call(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
    )

    recovered_iv = implied_volatility_bisection(
        market_price=float(market_price),
        spot=spot,
        strike=strike,
        rate=rate,
        time_to_expiry=time_to_expiry,
        option_type="call",
    )

    assert recovered_iv == pytest.approx(
        volatility,
        abs=1e-6,
    )


def test_recovers_put_implied_volatility():
    spot = 100.0
    strike = 100.0
    rate = 0.04
    volatility = 0.25
    time_to_expiry = 30.0 / 365.0

    market_price = black_scholes_put(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
    )

    recovered_iv = implied_volatility_bisection(
        market_price=float(market_price),
        spot=spot,
        strike=strike,
        rate=rate,
        time_to_expiry=time_to_expiry,
        option_type="put",
    )

    assert recovered_iv == pytest.approx(
        volatility,
        abs=1e-6,
    )


def test_rejects_call_price_above_upper_bound():
    with pytest.raises(ValueError):
        implied_volatility_bisection(
            market_price=120.0,
            spot=100.0,
            strike=100.0,
            rate=0.04,
            time_to_expiry=30.0 / 365.0,
            option_type="call",
        )


def test_rejects_invalid_option_type():
    with pytest.raises(ValueError):
        implied_volatility_bisection(
            market_price=3.0,
            spot=100.0,
            strike=100.0,
            rate=0.04,
            time_to_expiry=30.0 / 365.0,
            option_type="straddle",
        )


def test_fails_when_root_is_outside_volatility_interval():
    spot = 100.0
    strike = 100.0
    rate = 0.04
    true_volatility = 0.50
    time_to_expiry = 30.0 / 365.0

    market_price = black_scholes_call(
        spot,
        strike,
        rate,
        true_volatility,
        time_to_expiry,
    )

    with pytest.raises(ValueError):
        implied_volatility_bisection(
            market_price=float(market_price),
            spot=spot,
            strike=strike,
            rate=rate,
            time_to_expiry=time_to_expiry,
            option_type="call",
            lower_volatility=0.01,
            upper_volatility=0.20,
        )