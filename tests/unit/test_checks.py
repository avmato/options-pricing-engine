"""No-arbitrage checks: no false alarms on clean data, no blind spots on dirty data.

Both directions matter equally. A screen that never fires proves nothing, and
a screen that fires on arbitrage-free quotes would have made the whole
headline result meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optlab.audit.checks import (
    box_implied_rates,
    check_box_spread,
    check_butterfly_convexity,
    check_calendar_monotonicity,
    check_price_bounds,
    check_strike_monotonicity,
    check_vertical_spread_cap,
    parity_diagnostics,
)
from optlab.audit.runner import run_all_checks, survival_table
from optlab.market.forward import fit_forward_curve_fixed_discount
from optlab.market.quotes import add_quote_metrics, pair_calls_and_puts
from tests.conftest import make_chain


@pytest.fixture
def prepared(clean_chain):
    chain = add_quote_metrics(clean_chain)
    pairs = pair_calls_and_puts(chain)
    forwards = fit_forward_curve_fixed_discount(pairs, rate=0.0)
    return chain, pairs, forwards


def test_clean_chain_produces_no_violations_on_either_basis(prepared):
    chain, _, forwards = prepared
    violations = run_all_checks(chain, forwards)
    assert violations.empty, violations.to_string()


def test_survival_table_is_empty_when_nothing_is_violated(prepared):
    chain, _, forwards = prepared
    assert survival_table(run_all_checks(chain, forwards)).empty


def test_detects_a_call_that_is_too_cheap_for_its_strike(prepared):
    """Break monotonicity: make the 500 call cheaper than the 505 call."""
    chain, _, forwards = prepared
    broken = chain.copy()
    target = (broken["option_type"] == "call") & (broken["strike"] == 500.0) & (
        broken["expiration"] == pd.Timestamp("2026-03-20")
    )
    broken.loc[target, ["bid", "ask", "mid"]] = [0.10, 0.20, 0.15]

    violations = check_strike_monotonicity(broken, basis="executable")
    assert len(violations) >= 1
    assert (violations["check"] == "strike_monotonicity").all()
    assert violations["edge"].max() > 0


def test_detects_a_negative_butterfly(prepared):
    """Depress the middle strike so the implied density goes negative."""
    chain, _, forwards = prepared
    broken = chain.copy()
    target = (broken["option_type"] == "call") & (broken["strike"] == 500.0)
    broken.loc[target, ["bid", "ask", "mid"]] = [1.0, 1.2, 1.1]

    violations = check_butterfly_convexity(broken, basis="executable")
    assert len(violations) >= 1
    assert violations["edge"].max() > 0


def test_detects_a_quote_below_intrinsic(prepared):
    chain, _, forwards = prepared
    broken = chain.copy()
    target = (broken["option_type"] == "call") & (broken["strike"] == 420.0)
    broken.loc[target, ["bid", "ask", "mid"]] = [1.0, 2.0, 1.5]

    violations = check_price_bounds(broken, forwards, basis="executable")
    assert (violations["detail"] == "below intrinsic (buy)").any()
    # The edge is the distance from the ask up to the discounted intrinsic value.
    forward = float(forwards["forward"].iloc[0])
    assert violations["edge"].max() == pytest.approx(forward - 420.0 - 2.0, abs=1.0)


def test_detects_an_underpriced_box(prepared):
    """Make the 500/505 box cost less than its guaranteed payoff."""
    chain, _, forwards = prepared
    broken = chain.copy()
    target = (
        (broken["option_type"] == "put")
        & (broken["strike"] == 505.0)
        & (broken["expiration"] == pd.Timestamp("2026-03-20"))
    )
    broken.loc[target, ["bid", "ask", "mid"]] = [0.5, 0.6, 0.55]

    violations = check_box_spread(broken, forwards, basis="executable")
    assert len(violations) >= 1
    assert violations["edge"].max() > 0


def test_detects_a_calendar_inversion(prepared):
    chain, _, forwards = prepared
    broken = chain.copy()
    target = (
        (broken["option_type"] == "call")
        & (broken["strike"] == 500.0)
        & (broken["expiration"] == pd.Timestamp("2026-06-19"))
    )
    broken.loc[target, ["bid", "ask", "mid"]] = [0.05, 0.10, 0.075]

    violations = check_calendar_monotonicity(broken, basis="executable")
    assert len(violations) >= 1


def test_vertical_spread_cap_fires_when_a_spread_is_sold_above_its_payoff(prepared):
    chain, _, forwards = prepared
    broken = chain.copy()
    target = (
        (broken["option_type"] == "call")
        & (broken["strike"] == 500.0)
        & (broken["expiration"] == pd.Timestamp("2026-03-20"))
    )
    broken.loc[target, ["bid", "ask", "mid"]] = [400.0, 401.0, 400.5]

    violations = check_vertical_spread_cap(broken, forwards, basis="executable")
    assert len(violations) >= 1


def test_a_violation_at_the_mid_can_vanish_at_executable_prices():
    """The central claim of the project, on data where the answer is known.

    The chain is arbitrage-free at the mid by construction. Shifting one
    quote by less than the half-spread creates a mid-price violation that no
    taker could ever capture, so it must appear on the mid basis and be
    absent on the executable one.
    """
    chain = add_quote_metrics(make_chain(half_spread=0.50))

    nudged = chain.copy()
    target = (
        (nudged["option_type"] == "call")
        & (nudged["strike"] == 500.0)
        & (nudged["expiration"] == pd.Timestamp("2026-03-20"))
    )
    nudged.loc[target, "mid"] = nudged.loc[target, "mid"] - 0.45

    at_mid = check_butterfly_convexity(nudged, basis="mid")
    executable = check_butterfly_convexity(nudged, basis="executable")
    assert len(at_mid) >= 1
    assert executable.empty


def test_box_implied_rate_recovers_the_true_rate(rate_chain):
    """A box is a synthetic bond, so its implied rate must be the real one."""
    chain = add_quote_metrics(rate_chain)
    rates = box_implied_rates(chain, basis="mid")

    near_the_money = rates[
        (rates["strike_low"] > 480.0) & (rates["strike_high"] < 520.0) & (rates["strike_gap"] >= 5.0)
    ]
    assert len(near_the_money) > 0
    assert near_the_money["rate_mid"].median() == pytest.approx(0.045, abs=0.01)


def test_executable_box_rates_bracket_the_mid_rate(rate_chain):
    """Crossing the spread must make lending worse and borrowing dearer."""
    chain = add_quote_metrics(rate_chain)
    rates = box_implied_rates(chain, basis="executable")
    wide = rates[rates["strike_gap"] >= 20.0]
    assert (wide["rate_lend"] <= wide["rate_borrow"] + 1e-9).all()


def test_parity_residuals_are_zero_on_a_clean_chain(prepared):
    _, pairs, forwards = prepared
    residuals = parity_diagnostics(pairs, forwards)
    assert np.abs(residuals["parity_residual"]).max() < 1e-8
    assert (residuals["residual_beyond_spread"] == 0.0).all()
