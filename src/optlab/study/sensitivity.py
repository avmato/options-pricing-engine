"""How much of the result depends on choices I made?

Two of the headline numbers rest on judgement calls: which exercise style the
bounds assume, and how old a quote may be before it stops counting as
evidence. Both are varied here so the reader can see the answer's shape
rather than take one setting on trust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optlab.audit.runner import run_all_checks

__all__ = ["exercise_style_comparison", "staleness_sensitivity"]


def exercise_style_comparison(chain: pd.DataFrame, forwards: pd.DataFrame) -> pd.DataFrame:
    """Count violations under American and European bounds, side by side.

    Listed US equity options are American, so the American row is the correct
    one. The European row is what the textbook screen reports, and the
    difference is arbitrage that the early-exercise right fully explains --
    a false positive rate that is worth knowing before trusting any published
    "options violate no-arbitrage" claim.
    """
    rows: list[dict[str, object]] = []
    for style in ("american", "european"):
        violations = run_all_checks(chain, forwards, exercise_style=style)
        for basis in ("mid", "executable"):
            subset = violations[violations["basis"] == basis]
            rows.append(
                {
                    "exercise_style": style,
                    "basis": basis,
                    "violations": int(len(subset)),
                    "total_edge": float(subset["edge"].sum()) if len(subset) else 0.0,
                }
            )

    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index="basis", columns="exercise_style", values="violations")
    pivot["false_positives_from_european"] = pivot["european"] - pivot["american"]
    pivot["share_invented"] = pivot["false_positives_from_european"] / pivot["european"]
    return pivot.reset_index()


def staleness_sensitivity(
    violations: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, np.inf),
    *,
    require_open_interest: bool = True,
) -> pd.DataFrame:
    """Survivors as a function of how stale a quote is allowed to be.

    The headline result drops to zero once quotes older than two days are
    excluded, and the obvious objection is that two days was chosen to make
    that happen. This table answers it: the count stays at zero over a wide
    range of thresholds, and only re-appears once quotes weeks old are
    admitted as evidence.
    """
    executable = violations[violations["basis"] == "executable"]
    if require_open_interest:
        executable = executable[executable["min_open_interest"].fillna(0) > 0]

    rows = []
    for threshold in thresholds:
        surviving = executable[executable["max_last_trade_age_days"].fillna(np.inf) <= threshold]
        rows.append(
            {
                "max_quote_age_days": threshold,
                "violations": int(len(surviving)),
                "total_edge": float(surviving["edge"].sum()) if len(surviving) else 0.0,
                "max_edge": float(surviving["edge"].max()) if len(surviving) else 0.0,
            }
        )
    return pd.DataFrame(rows)
