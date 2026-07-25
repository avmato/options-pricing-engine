"""Greeks validated against central finite differences of the pricer itself.

Comparing an analytic derivative to a hand-written expected value only checks
that two people made the same algebra mistake. Bumping the pricer does not:
if the closed form and the numerical derivative of the same function agree,
the closed form is right.
"""

from __future__ import annotations

import numpy as np
import pytest

from optlab.core.black import black76_price
from optlab.core.greeks import (
    delta,
    dual_delta,
    gamma,
    per_calendar_day,
    per_percentage_point,
    rho,
    theta,
    vanna,
    vega,
    volga,
)

BASE = {
    "forward": 500.0,
    "strike": 495.0,
    "volatility": 0.22,
    "time_to_expiry": 0.35,
    "discount": 0.985,
}

CASES = [
    pytest.param(option_type, strike, id=f"{option_type}-K{strike:.0f}")
    for option_type in ("call", "put")
    for strike in (400.0, 495.0, 500.0, 620.0)
]


def _price(option_type: str, **overrides) -> float:
    arguments = {**BASE, **overrides}
    return float(
        black76_price(
            arguments["forward"],
            arguments["strike"],
            arguments["volatility"],
            arguments["time_to_expiry"],
            arguments["discount"],
            option_type,
        )
    )


def _central_difference(option_type: str, field: str, step: float, **overrides) -> float:
    base_value = {**BASE, **overrides}[field]
    up = _price(option_type, **{**overrides, field: base_value + step})
    down = _price(option_type, **{**overrides, field: base_value - step})
    return (up - down) / (2.0 * step)


@pytest.mark.parametrize(("option_type", "strike"), [(case.values[0], case.values[1]) for case in CASES])
def test_delta_matches_finite_difference(option_type, strike):
    analytic = float(delta(**BASE | {"strike": strike}, option_type=option_type))
    numeric = _central_difference(option_type, "forward", 1e-4, strike=strike)
    assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize(("option_type", "strike"), [(case.values[0], case.values[1]) for case in CASES])
def test_dual_delta_matches_finite_difference(option_type, strike):
    analytic = float(dual_delta(**BASE | {"strike": strike}, option_type=option_type))
    numeric = _central_difference(option_type, "strike", 1e-4, strike=strike)
    assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize(("option_type", "strike"), [(case.values[0], case.values[1]) for case in CASES])
def test_vega_matches_finite_difference(option_type, strike):
    analytic = float(vega(**BASE | {"strike": strike}, option_type=option_type))
    numeric = _central_difference(option_type, "volatility", 1e-6, strike=strike)
    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize(("option_type", "strike"), [(case.values[0], case.values[1]) for case in CASES])
def test_gamma_matches_second_finite_difference(option_type, strike):
    analytic = float(gamma(**BASE | {"strike": strike}, option_type=option_type))
    step = 1e-2
    up = _price(option_type, strike=strike, forward=BASE["forward"] + step)
    mid = _price(option_type, strike=strike)
    down = _price(option_type, strike=strike, forward=BASE["forward"] - step)
    numeric = (up - 2.0 * mid + down) / step**2
    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-9)


@pytest.mark.parametrize(("option_type", "strike"), [(case.values[0], case.values[1]) for case in CASES])
def test_theta_matches_finite_difference(option_type, strike):
    """Theta is dV/dt, so it is minus the derivative with respect to maturity."""
    analytic = float(theta(**BASE | {"strike": strike}, option_type=option_type))
    numeric = -_central_difference(option_type, "time_to_expiry", 1e-6, strike=strike)
    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-6)


@pytest.mark.parametrize(("option_type", "strike"), [(case.values[0], case.values[1]) for case in CASES])
def test_vanna_and_volga_match_finite_differences(option_type, strike):
    step = 1e-5
    bumped_up = BASE | {"strike": strike, "volatility": BASE["volatility"] + step}
    bumped_down = BASE | {"strike": strike, "volatility": BASE["volatility"] - step}
    up = float(delta(**bumped_up, option_type=option_type))
    down = float(delta(**bumped_down, option_type=option_type))
    assert float(vanna(**BASE | {"strike": strike}, option_type=option_type)) == pytest.approx(
        (up - down) / (2.0 * step), rel=1e-4, abs=1e-7
    )

    vega_up = float(vega(**bumped_up, option_type=option_type))
    vega_down = float(vega(**bumped_down, option_type=option_type))
    assert float(volga(**BASE | {"strike": strike}, option_type=option_type)) == pytest.approx(
        (vega_up - vega_down) / (2.0 * step), rel=1e-4, abs=1e-6
    )


def test_gamma_and_vega_are_type_independent():
    """Parity is linear in the forward, so second-order Greeks cannot differ."""
    call_gamma = float(gamma(**BASE, option_type="call"))
    put_gamma = float(gamma(**BASE, option_type="put"))
    assert call_gamma == pytest.approx(put_gamma, rel=1e-12)
    assert float(vega(**BASE, option_type="call")) == pytest.approx(
        float(vega(**BASE, option_type="put")), rel=1e-12
    )


def test_rho_is_minus_time_times_price():
    price = _price("call")
    assert float(rho(**BASE, option_type="call")) == pytest.approx(-BASE["time_to_expiry"] * price)


def test_reporting_helpers_rescale():
    assert per_percentage_point(250.0) == pytest.approx(2.5)
    assert per_calendar_day(-36.5) == pytest.approx(-0.1)


def test_greeks_are_vectorised():
    strikes = np.array([450.0, 500.0, 550.0])
    values = vega(BASE["forward"], strikes, BASE["volatility"], BASE["time_to_expiry"], BASE["discount"])
    assert values.shape == (3,)
    assert np.all(values > 0.0)
