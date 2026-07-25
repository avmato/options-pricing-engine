"""Run the full no-arbitrage suite and summarise what survives the spread."""

from __future__ import annotations

import pandas as pd

from optlab.audit.checks import (
    ExerciseStyle,
    PriceBasis,
    check_box_spread,
    check_butterfly_convexity,
    check_calendar_monotonicity,
    check_price_bounds,
    check_strike_monotonicity,
    check_vertical_spread_cap,
)

__all__ = ["run_all_checks", "summarise_violations", "survival_table"]

CHECK_ORDER = [
    "price_bounds",
    "strike_monotonicity",
    "vertical_spread_cap",
    "butterfly_convexity",
    "box_spread",
    "calendar_monotonicity",
]


def run_all_checks(
    chain: pd.DataFrame,
    forwards: pd.DataFrame,
    bases: tuple[PriceBasis, ...] = ("mid", "executable"),
    exercise_style: ExerciseStyle = "american",
) -> pd.DataFrame:
    """Apply every static check on every requested price basis.

    Returns one tidy frame of violations. An empty frame is a result, not a
    failure: it means the quotes are internally consistent at that basis.

    ``exercise_style`` defaults to American, matching listed US equity
    options. Passing ``"european"`` reproduces the textbook screen and is
    useful for measuring how much apparent arbitrage that assumption invents.
    """
    frames: list[pd.DataFrame] = []
    for basis in bases:
        frames.extend(
            [
                check_price_bounds(chain, forwards, basis, exercise_style=exercise_style),
                check_strike_monotonicity(chain, basis),
                check_vertical_spread_cap(chain, forwards, basis, exercise_style=exercise_style),
                check_butterfly_convexity(chain, basis),
                check_box_spread(chain, forwards, basis, exercise_style=exercise_style),
                check_calendar_monotonicity(chain, basis),
            ]
        )

    violations = pd.concat(frames, ignore_index=True)
    if violations.empty:
        return violations
    return violations.sort_values(["basis", "check", "edge"], ascending=[True, True, False]).reset_index(
        drop=True
    )


def summarise_violations(
    violations: pd.DataFrame,
    opportunities: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Count and size violations per check and price basis.

    ``opportunities`` optionally supplies the number of portfolios each check
    examined, so the violation *rate* can be reported instead of a raw count.
    A count alone is unreadable: 40 butterfly breaches out of 60 triples and
    40 out of 4,000 are entirely different claims.
    """
    if violations.empty:
        return pd.DataFrame(
            columns=["check", "basis", "violations", "total_edge", "median_edge", "max_edge"]
        )

    grouped = (
        violations.groupby(["check", "basis"], sort=False)
        .agg(
            violations=("edge", "size"),
            total_edge=("edge", "sum"),
            median_edge=("edge", "median"),
            max_edge=("edge", "max"),
        )
        .reset_index()
    )

    if opportunities:
        grouped["opportunities"] = grouped["check"].map(opportunities)
        grouped["violation_rate"] = grouped["violations"] / grouped["opportunities"]

    grouped["check"] = pd.Categorical(grouped["check"], categories=CHECK_ORDER, ordered=True)
    return grouped.sort_values(["check", "basis"]).reset_index(drop=True)


def survival_table(violations: pd.DataFrame) -> pd.DataFrame:
    """How many mid-price 'arbitrages' are still there once you cross the spread.

    This is the headline table of the project. The survival rate is the share
    of apparent violations that remain when every leg is executed at the
    price a taker would actually receive.
    """
    summary = summarise_violations(violations)
    if summary.empty:
        return summary

    wide = summary.pivot_table(
        index="check", columns="basis", values=["violations", "total_edge"], observed=False
    )
    wide.columns = ["_".join(str(level) for level in column) for column in wide.columns]
    wide = wide.fillna(0.0)

    for column in ("violations_mid", "violations_executable", "total_edge_mid", "total_edge_executable"):
        if column not in wide.columns:
            wide[column] = 0.0

    wide["survival_rate"] = wide["violations_executable"] / wide["violations_mid"].where(
        wide["violations_mid"] > 0
    )
    wide["edge_survival_rate"] = wide["total_edge_executable"] / wide["total_edge_mid"].where(
        wide["total_edge_mid"] > 0
    )
    return wide.reset_index()
