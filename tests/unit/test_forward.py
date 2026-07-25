"""Forward extraction: what parity can identify, and what it cannot."""

from __future__ import annotations

import numpy as np
import pytest

from optlab.market.forward import (
    fit_forward,
    fit_forward_curve,
    fit_forward_curve_fixed_discount,
    forward_from_fixed_discount,
)
from optlab.market.quotes import add_quote_metrics, pair_calls_and_puts
from tests.conftest import make_chain


def _pairs(**kwargs):
    return pair_calls_and_puts(add_quote_metrics(make_chain(**kwargs)))


def test_recovers_the_forward_and_discount_from_clean_quotes():
    """With exact prices the regression is exact, up to floating point."""
    rate, dividend_yield, spot = 0.045, 0.013, 500.0
    pairs = _pairs(rate=rate, dividend_yield=dividend_yield, half_spread=0.01)

    for _, chunk in pairs.groupby("expiration"):
        fit = fit_forward(chunk)
        time_to_expiry = fit.time_to_expiry
        expected_forward = spot * np.exp((rate - dividend_yield) * time_to_expiry)

        assert fit.forward == pytest.approx(expected_forward, rel=1e-9)
        assert fit.discount == pytest.approx(np.exp(-rate * time_to_expiry), rel=1e-9)
        assert fit.implied_rate == pytest.approx(rate, abs=1e-7)
        assert fit.implied_dividend_yield == pytest.approx(dividend_yield, abs=1e-7)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_fixed_discount_route_recovers_the_same_forward():
    rate, dividend_yield, spot = 0.045, 0.013, 500.0
    pairs = _pairs(rate=rate, dividend_yield=dividend_yield, half_spread=0.01)
    curve = fit_forward_curve_fixed_discount(pairs, rate=rate)

    expected = spot * np.exp((rate - dividend_yield) * curve["time_to_expiry"])
    assert np.allclose(curve["forward"], expected, rtol=1e-9)
    assert np.allclose(curve["implied_dividend_yield"], dividend_yield, atol=1e-7)


def test_forward_survives_quote_noise_that_destroys_the_rate():
    """The headline identifiability result, on data with a known answer.

    Adding a penny of independent noise to each quote barely moves the
    intercept -- averaging across strikes takes care of it -- but the slope
    is a difference of noisy numbers over a short strike range, and dividing
    its error by a small time to expiry blows the implied rate up.
    """
    rate, dividend_yield, spot = 0.045, 0.013, 500.0
    generator = np.random.default_rng(20260725)
    pairs = _pairs(rate=rate, dividend_yield=dividend_yield, half_spread=0.01)

    noisy = pairs.copy()
    for column in ("mid_call", "mid_put"):
        noisy[column] = noisy[column] + generator.normal(0.0, 0.01, size=len(noisy))

    shortest = noisy[noisy["expiration"] == noisy["expiration"].min()]
    fit = fit_forward(shortest)

    true_forward = spot * np.exp((rate - dividend_yield) * fit.time_to_expiry)
    forward_relative_error = abs(fit.forward - true_forward) / true_forward
    rate_relative_error = abs(fit.implied_rate - rate) / rate

    assert forward_relative_error < 1e-3
    assert rate_relative_error > 100 * forward_relative_error, (
        "the same noise must hit the rate orders of magnitude harder than the forward"
    )
    assert not fit.rate_identified


def test_rate_is_flagged_as_identified_when_quotes_are_clean_and_long_dated():
    pairs = _pairs(rate=0.045, dividend_yield=0.013, half_spread=0.001)
    curve = fit_forward_curve(pairs)
    assert curve["rate_identified"].all()
    assert curve["discount_is_economic"].all()


def test_fit_requires_enough_paired_strikes():
    pairs = _pairs()
    one_expiry = pairs[pairs["expiration"] == pairs["expiration"].min()]
    thin = one_expiry[one_expiry["strike"].isin([495.0, 500.0])]
    with pytest.raises(ValueError, match="at least"):
        fit_forward(thin)


def test_forward_from_fixed_discount_reports_cross_strike_dispersion():
    """Dispersion across strikes is a data-quality measure, so it must be real."""
    pairs = _pairs(rate=0.04, half_spread=0.01)
    chunk = pairs[pairs["expiration"] == pairs["expiration"].min()]
    time_to_expiry = float(chunk["time_to_expiry"].iloc[0])

    forward, std_error = forward_from_fixed_discount(chunk, float(np.exp(-0.04 * time_to_expiry)))
    assert forward > 0
    assert std_error == pytest.approx(0.0, abs=1e-6)

    noisy = chunk.copy()
    generator = np.random.default_rng(7)
    noisy["mid_call"] = noisy["mid_call"] + generator.normal(0, 0.05, len(noisy))
    _, noisy_std_error = forward_from_fixed_discount(noisy, float(np.exp(-0.04 * time_to_expiry)))
    assert noisy_std_error > std_error


def test_curve_is_sorted_and_carries_diagnostics():
    curve = fit_forward_curve(_pairs(rate=0.04))
    assert curve["time_to_expiry"].is_monotonic_increasing
    for column in ("forward_std_error", "rate_low", "rate_high", "r_squared", "n_strikes"):
        assert column in curve.columns
