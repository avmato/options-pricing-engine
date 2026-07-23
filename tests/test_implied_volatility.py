import pytest
import math
from src.black_scholes import (
    black_scholes_call,
    black_scholes_put,
)
from src.implied_volatility import implied_volatility_bisection
from src.implied_volatility import (
    implied_volatility_bisection,
    implied_volatility_for_chain,
)

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

def test_rejects_negative_market_price():
    with pytest.raises(
        ValueError,
        match="Market price cannot be negative",
    ):
        implied_volatility_bisection(
            market_price=-1.0,
            spot=100.0,
            strike=100.0,
            rate=0.04,
            time_to_expiry=30.0 / 365.0,
            option_type="call",
        )

@pytest.mark.parametrize(
    (
        "strike",
        "volatility",
        "time_to_expiry",
    ),
    [
        (80.0, 0.20, 30.0 / 365.0),
        (100.0, 0.25, 90.0 / 365.0),
        (120.0, 0.35, 365.0 / 365.0),
    ],
)
def test_recovers_call_iv_across_multiple_scenarios(
    strike,
    volatility,
    time_to_expiry,
):
    spot = 100.0
    rate = 0.04

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

def test_computes_implied_volatility_for_chain():
    spot = 100.0
    rate = 0.04

    strikes = [90.0, 100.0, 110.0]
    times_to_expiry = [
        30.0 / 365.0,
        30.0 / 365.0,
        30.0 / 365.0,
    ]
    true_volatilities = [0.20, 0.25, 0.30]

    market_prices = [
        float(
            black_scholes_call(
                spot,
                strike,
                rate,
                volatility,
                time_to_expiry,
            )
        )
        for strike, volatility, time_to_expiry in zip(
            strikes,
            true_volatilities,
            times_to_expiry,
        )
    ]

    recovered_volatilities = implied_volatility_for_chain(
        market_prices=market_prices,
        spot=spot,
        strikes=strikes,
        rate=rate,
        times_to_expiry=times_to_expiry,
        option_type="call",
    )

    assert recovered_volatilities == pytest.approx(
        true_volatilities,
        abs=1e-6,
    )

def test_chain_returns_nan_for_invalid_row():
    spot = 100.0
    rate = 0.04

    strikes = [90.0, 100.0, 110.0]
    times_to_expiry = [
        30.0 / 365.0,
        30.0 / 365.0,
        30.0 / 365.0,
    ]

    valid_price_1 = black_scholes_call(
        spot,
        strikes[0],
        rate,
        0.20,
        times_to_expiry[0],
    )

    valid_price_3 = black_scholes_call(
        spot,
        strikes[2],
        rate,
        0.30,
        times_to_expiry[2],
    )

    market_prices = [
        float(valid_price_1),
        -1.0,
        float(valid_price_3),
    ]

    recovered_volatilities = implied_volatility_for_chain(
        market_prices=market_prices,
        spot=spot,
        strikes=strikes,
        rate=rate,
        times_to_expiry=times_to_expiry,
        option_type="call",
    )

    assert recovered_volatilities[0] == pytest.approx(
        0.20,
        abs=1e-6,
    )

    assert math.isnan(
        recovered_volatilities[1]
    )

    assert recovered_volatilities[2] == pytest.approx(
        0.30,
        abs=1e-6,
    )

def test_chain_rejects_mismatched_input_lengths():
    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        implied_volatility_for_chain(
            market_prices=[10.0, 5.0, 2.0],
            spot=100.0,
            strikes=[90.0, 100.0],
            rate=0.04,
            times_to_expiry=[
                30.0 / 365.0,
                30.0 / 365.0,
                30.0 / 365.0,
            ],
            option_type="call",
        )