"""End-to-end behaviour of the audit pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from optlab.pipeline import audit_chain, count_opportunities, violation_funnel
from tests.conftest import make_chain


def test_clean_chain_yields_an_empty_audit():
    result = audit_chain(make_chain(), discount_rate=0.0)
    assert result.violations.empty
    assert result.headline()["violations_at_mid"] == 0
    assert result.headline()["violations_executable"] == 0


def test_convention_error_shrinks_when_the_forward_is_implied():
    """A wrong rate creates a call/put gap that the parity forward removes.

    The chain is generated at a 4.5% rate and a 1.3% dividend yield. Inverting
    it under "4%, no dividends" must push call and put implied volatilities
    apart; inverting under the implied forward must not.
    """
    chain = make_chain(rate=0.045, dividend_yield=0.013, half_spread=0.005)
    result = audit_chain(chain, naive_rate=0.04, discount_rate=0.045)

    summary = result.iv_gap_summary
    assert (summary["mean_abs_gap_parity"] < summary["mean_abs_gap_naive"]).all()
    assert summary["mean_abs_gap_parity"].max() < 1e-6
    assert summary["mean_abs_gap_naive"].max() > 1e-4


def test_funnel_is_monotone_and_starts_from_the_mid_count():
    chain = make_chain(half_spread=0.40)
    broken = chain.copy()
    target = (broken["option_type"] == "call") & (broken["strike"] == 500.0)
    broken.loc[target, ["bid", "ask"]] = [0.05, 0.10]

    result = audit_chain(broken, discount_rate=0.0)
    funnel = result.funnel

    assert list(funnel["step"]) == [0, 1, 2, 3, 4]
    counts = funnel["violations"].to_numpy()
    assert np.all(np.diff(counts[1:]) <= 0), "each extra filter can only remove violations"


def test_stale_quotes_are_removed_by_the_funnel():
    """A violation carried by a months-old quote must not survive the funnel."""
    chain = make_chain(half_spread=0.02)
    stale = chain.copy()
    target = (
        (stale["option_type"] == "put")
        & (stale["strike"] == 560.0)
        & (stale["expiration"] == pd.Timestamp("2026-04-17"))
    )
    stale.loc[target, ["bid", "ask"]] = [1.0, 1.2]  # far below intrinsic
    stale.loc[target, "last_trade_timestamp"] = (
        stale.loc[target, "download_timestamp"] - pd.Timedelta(days=90)
    )
    stale.loc[target, "open_interest"] = 0.0

    result = audit_chain(stale, discount_rate=0.0)
    funnel = result.funnel

    assert funnel.loc[funnel["step"] == 1, "violations"].item() > 0
    assert funnel.loc[funnel["step"] == 4, "violations"].item() == 0


def test_quote_quality_flags_a_one_sided_snapshot():
    """Widen only the calls: the call/put spread ratio must pick it up."""
    chain = make_chain(half_spread=0.02)
    lopsided = chain.copy()
    calls = lopsided["option_type"] == "call"
    lopsided.loc[calls, "ask"] = lopsided.loc[calls, "ask"] + 0.5
    lopsided.loc[calls, "bid"] = np.maximum(lopsided.loc[calls, "bid"] - 0.5, 0.01)

    result = audit_chain(lopsided, discount_rate=0.0)
    assert (result.quote_quality["call_put_spread_ratio"] > 5).all()


def test_count_opportunities_matches_the_chain_shape():
    chain = make_chain(strikes=np.array([90.0, 95.0, 100.0, 105.0]), expiries=("2026-03-20",))
    counts = count_opportunities(chain.assign(days_to_expiry=1.0))

    assert counts["price_bounds"] == 8  # four strikes, two option types
    assert counts["strike_monotonicity"] == 6  # three adjacent pairs per type
    assert counts["butterfly_convexity"] == 4  # two adjacent triples per type
    assert counts["box_spread"] == 3  # three adjacent strike pairs
    assert counts["calendar_monotonicity"] == 0  # a single expiry has no calendar


def test_funnel_handles_an_empty_violation_set():
    assert violation_funnel(pd.DataFrame()).empty


def test_audit_reports_filter_attrition():
    chain = make_chain()
    dead = chain.copy()
    dead.loc[dead["strike"] == 400.0, ["bid", "ask"]] = [0.0, 0.0]

    result = audit_chain(dead, discount_rate=0.0)
    attrition = result.filter_report.to_frame()
    assert attrition["rows_removed"].sum() >= 2
    assert result.filter_report.final_rows < result.filter_report.initial_rows


def test_headline_is_json_serialisable_floats():
    result = audit_chain(make_chain(), discount_rate=0.0)
    for key, value in result.headline().items():
        assert isinstance(value, float), key
        assert not isinstance(value, np.generic), key
