"""End-to-end audit of one option-chain snapshot.

The pipeline is deliberately a straight line with every intermediate result
kept: load, measure, pair, fit forwards, screen for arbitrage, quantify the
convention error. Nothing is thrown away silently, because in this project
the rejects (quotes that fail a bound, expiries whose rate is unidentified)
are as much a result as the survivors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optlab.audit.checks import ExerciseStyle, box_implied_rates, parity_diagnostics
from optlab.audit.runner import run_all_checks, summarise_violations, survival_table
from optlab.market.forward import fit_forward_curve, fit_forward_curve_fixed_discount
from optlab.market.quotes import (
    FilterReport,
    add_quote_metrics,
    filter_quotes,
    load_chain,
    pair_calls_and_puts,
)
from optlab.study.iv_convention import call_put_gap_summary, implied_vol_under_conventions
from optlab.study.sensitivity import exercise_style_comparison, staleness_sensitivity

__all__ = ["AuditResult", "audit_chain", "audit_snapshot", "count_opportunities"]


@dataclass
class AuditResult:
    """Everything one snapshot produced."""

    chain: pd.DataFrame
    pairs: pd.DataFrame
    forwards: pd.DataFrame
    forwards_joint: pd.DataFrame
    liquidity: pd.DataFrame
    funnel: pd.DataFrame
    staleness: pd.DataFrame
    exercise_styles: pd.DataFrame
    violations: pd.DataFrame
    violation_summary: pd.DataFrame
    survival: pd.DataFrame
    parity: pd.DataFrame
    box_rates: pd.DataFrame
    iv_comparison: pd.DataFrame
    iv_gap_summary: pd.DataFrame
    filter_report: FilterReport
    quote_quality: pd.DataFrame

    def headline(self) -> dict[str, float]:
        """The numbers the report is built around."""
        mid = self.violations[self.violations["basis"] == "mid"]
        executable = self.violations[self.violations["basis"] == "executable"]
        survivors = self.funnel.iloc[-1] if len(self.funnel) else None
        return {
            "violations_after_all_filters": float(survivors["violations"]) if survivors is not None else 0.0,
            "edge_after_all_filters": float(survivors["total_edge"]) if survivors is not None else 0.0,
            "violations_at_mid": float(len(mid)),
            "violations_executable": float(len(executable)),
            "survival_rate": float(len(executable) / len(mid)) if len(mid) else float("nan"),
            "edge_at_mid": float(mid["edge"].sum()) if len(mid) else 0.0,
            "edge_executable": float(executable["edge"].sum()) if len(executable) else 0.0,
            "mean_abs_call_put_gap_naive": float(self.iv_gap_summary["mean_abs_gap_naive"].mean()),
            "mean_abs_call_put_gap_parity": float(self.iv_gap_summary["mean_abs_gap_parity"].mean()),
        }


def count_opportunities(chain: pd.DataFrame) -> dict[str, int]:
    """How many distinct portfolios each check could have flagged.

    Needed to turn violation counts into violation rates. Without this
    denominator a count says nothing about how unusual a breach is.
    """
    per_expiry_type = chain.groupby(["expiration", "option_type"], sort=False).size()
    per_expiry = chain.groupby("expiration", sort=False)["strike"].nunique()
    per_strike_type = chain.groupby(["strike", "option_type"], sort=False).size()

    return {
        "price_bounds": int(len(chain)),
        "strike_monotonicity": int(np.maximum(per_expiry_type - 1, 0).sum()),
        "vertical_spread_cap": int(np.maximum(per_expiry_type - 1, 0).sum()),
        "butterfly_convexity": int(np.maximum(per_expiry_type - 2, 0).sum()),
        "box_spread": int(np.maximum(per_expiry - 1, 0).sum()),
        "calendar_monotonicity": int(np.maximum(per_strike_type - 1, 0).sum()),
    }


def quote_quality_by_type(chain: pd.DataFrame, band: float = 0.05) -> pd.DataFrame:
    """Compare call-side and put-side quote quality near the money.

    A vendor snapshot is not guaranteed to be symmetric: one side can be
    refreshed while the other is stale, and every parity-based estimate then
    inherits that asymmetry. Measuring it is a prerequisite for trusting any
    of the results downstream.
    """
    near = chain[np.abs(chain["log_moneyness"]) <= band]
    grouped = (
        near.groupby(["expiration", "option_type"], sort=True)
        .agg(
            n=("mid", "size"),
            median_relative_spread=("relative_spread", "median"),
            median_spread=("spread", "median"),
        )
        .reset_index()
    )
    wide = grouped.pivot_table(
        index="expiration", columns="option_type", values="median_relative_spread", observed=False
    )
    wide.columns = [f"median_relative_spread_{column}" for column in wide.columns]
    wide["call_put_spread_ratio"] = (
        wide["median_relative_spread_call"] / wide["median_relative_spread_put"]
    )
    return wide.reset_index()


def liquidity_breakdown(
    violations: pd.DataFrame,
    *,
    moneyness_band: float = 0.05,
    max_relative_spread: float = 0.05,
) -> pd.DataFrame:
    """Split violations into the liquid core and everything else.

    A violation found in a contract quoted 40% wide, 20% out of the money and
    with no volume is a statement about the vendor's data, not about the
    market. Separating the two is what turns a raw count into a claim someone
    would defend on a desk.
    """
    if violations.empty:
        return pd.DataFrame(columns=["basis", "segment", "violations", "total_edge", "max_edge"])

    frame = violations.copy()
    in_core = (frame["log_moneyness"].abs() <= moneyness_band) & (
        frame["worst_relative_spread"] <= max_relative_spread
    )
    frame["segment"] = np.where(in_core, "liquid core", "wide or far from the money")

    return (
        frame.groupby(["basis", "segment"], sort=False)
        .agg(
            violations=("edge", "size"),
            total_edge=("edge", "sum"),
            max_edge=("edge", "max"),
            median_worst_spread=("worst_relative_spread", "median"),
        )
        .reset_index()
    )


def violation_funnel(
    violations: pd.DataFrame,
    *,
    max_last_trade_age_days: float = 2.0,
    max_relative_spread: float = 0.05,
) -> pd.DataFrame:
    """Apply data-quality filters one at a time and count what is left.

    This is the central table of the project. Each row removes one excuse:

    1. everything, priced at the mid;
    2. priced where a taker could actually trade;
    3. contracts anyone actually holds (non-zero open interest);
    4. contracts that have traded recently, so the quote is a live one;
    5. contracts whose spread is narrow enough to be a real market.

    Whatever survives all five is a candidate arbitrage. Anything that drops
    out tells you which assumption was doing the work.
    """
    if violations.empty:
        return pd.DataFrame(columns=["step", "filter", "violations", "total_edge", "max_edge"])

    at_mid = violations[violations["basis"] == "mid"]
    steps: list[tuple[str, pd.DataFrame]] = [("all quotes, priced at the mid", at_mid)]
    current = violations[violations["basis"] == "executable"]
    steps.append(("priced at executable (buy ask / sell bid)", current))

    current = current[current["min_open_interest"].fillna(0) > 0]
    steps.append(("open interest > 0 on every leg", current))

    current = current[current["max_last_trade_age_days"].fillna(np.inf) <= max_last_trade_age_days]
    steps.append((f"every leg traded within {max_last_trade_age_days:.0f} days", current))

    current = current[current["worst_relative_spread"].fillna(np.inf) <= max_relative_spread]
    steps.append((f"relative spread <= {max_relative_spread:.0%} on every leg", current))

    rows = [
        {
            "step": index,
            "filter": label,
            "violations": int(len(frame)),
            "total_edge": float(frame["edge"].sum()) if len(frame) else 0.0,
            "max_edge": float(frame["edge"].max()) if len(frame) else 0.0,
        }
        for index, (label, frame) in enumerate(steps)
    ]
    return pd.DataFrame(rows)


def audit_chain(
    chain: pd.DataFrame,
    *,
    forward_band: float = 0.05,
    iv_band: float | None = 0.15,
    naive_rate: float = 0.04,
    discount_rate: float = 0.043,
    exercise_style: ExerciseStyle = "american",
    max_relative_spread: float | None = None,
) -> AuditResult:
    """Run the full audit on an already-loaded chain.

    ``discount_rate`` is the externally observed money-market rate used to
    build the discount factor. Only the forward is implied from the options;
    see :func:`optlab.market.forward.fit_forward_curve_fixed_discount` for why
    the discount factor is not estimated from the quotes.

    ``exercise_style`` selects the no-arbitrage bounds. Listed US equity
    options are American, so that is the default.
    """
    measured = add_quote_metrics(chain)
    cleaned, report = filter_quotes(measured, max_relative_spread=max_relative_spread)

    pairs = pair_calls_and_puts(cleaned)
    forwards = fit_forward_curve_fixed_discount(pairs, rate=discount_rate, band=forward_band)

    # The joint fit is kept purely as a diagnostic: it is what you get if you
    # ask the option quotes to supply the discount factor as well.
    try:
        forwards_joint = fit_forward_curve(pairs, band=forward_band)
    except ValueError:
        forwards_joint = pd.DataFrame()

    violations = run_all_checks(cleaned, forwards, exercise_style=exercise_style)
    summary = summarise_violations(violations, count_opportunities(cleaned))

    comparison = implied_vol_under_conventions(pairs, forwards, naive_rate=naive_rate, band=iv_band)

    return AuditResult(
        chain=cleaned,
        pairs=pairs,
        forwards=forwards,
        forwards_joint=forwards_joint,
        liquidity=liquidity_breakdown(violations),
        funnel=violation_funnel(violations),
        staleness=staleness_sensitivity(violations),
        exercise_styles=exercise_style_comparison(cleaned, forwards),
        violations=violations,
        violation_summary=summary,
        survival=survival_table(violations),
        parity=parity_diagnostics(pairs, forwards),
        box_rates=box_implied_rates(cleaned, basis="executable"),
        iv_comparison=comparison,
        iv_gap_summary=call_put_gap_summary(comparison),
        filter_report=report,
        quote_quality=quote_quality_by_type(cleaned, band=forward_band),
    )


def audit_snapshot(path: str, **kwargs: object) -> AuditResult:
    """Load a raw chain CSV and audit it."""
    return audit_chain(load_chain(path), **kwargs)  # type: ignore[arg-type]
