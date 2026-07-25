"""Figures for the report.

Each figure exists to make one claim visible. Nothing decorative.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = ["make_all_figures"]

PALETTE = {
    "mid": "#8a8f98",
    "executable": "#1f4e79",
    "accent": "#c1440e",
    "muted": "#b8bcc4",
}


def _style(axis: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=11, loc="left")
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.tick_params(labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def plot_funnel(funnel: pd.DataFrame, path: Path) -> None:
    """The headline: how many 'arbitrages' survive each honesty filter."""
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    labels = [text.replace(" (buy ask / sell bid)", "\n(buy ask / sell bid)") for text in funnel["filter"]]
    colours = [PALETTE["mid"]] + [PALETTE["executable"]] * (len(funnel) - 1)

    bars = axis.bar(range(len(funnel)), funnel["violations"], color=colours, width=0.62)
    for bar, count in zip(bars, funnel["violations"], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(funnel["violations"]) * 0.02,
            f"{int(count):,}",
            ha="center",
            fontsize=9,
        )

    axis.set_xticks(range(len(funnel)))
    axis.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
    _style(axis, "Apparent arbitrage, filter by filter", "", "violations")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_parity_residuals(parity: pd.DataFrame, path: Path) -> None:
    """Parity residuals against the spread they would have to beat."""
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    for _, chunk in parity.groupby("expiration"):
        days = float(chunk["days_to_expiry"].iloc[0])
        axis.plot(
            chunk["log_moneyness"],
            chunk["parity_residual"],
            linewidth=1.1,
            label=f"{days:.0f}d",
        )

    reference = parity.sort_values("log_moneyness")
    axis.fill_between(
        reference["log_moneyness"],
        -reference["combined_half_spread"],
        reference["combined_half_spread"],
        color=PALETTE["muted"],
        alpha=0.35,
        label="combined half-spread",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    _style(
        axis,
        "Put-call parity residuals sit inside the spread they would have to cross",
        "log-moneyness  log(K/F)",
        "residual  (C - P) - D(F - K)   [$]",
    )
    axis.legend(fontsize=7.5, ncols=3, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_convention_error(comparison: pd.DataFrame, path: Path) -> None:
    """The call/put implied volatility gap, before and after implying the forward."""
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for axis, label, title in (
        (axes[0], "naive", "assumed rate 4%, no dividends"),
        (axes[1], "parity", "forward implied from put-call parity"),
    ):
        for _, chunk in comparison.groupby("expiration"):
            days = float(chunk["days_to_expiry"].iloc[0])
            axis.plot(chunk["log_moneyness"], chunk[f"gap_{label}"], linewidth=1.0, label=f"{days:.0f}d")
        axis.axhline(0.0, color="black", linewidth=0.8)
        _style(axis, title, "log-moneyness", "call IV - put IV  [vol points]")

    axes[1].legend(fontsize=7.5, ncols=3, frameon=False)
    figure.suptitle(
        "Calls and puts only 'disagree' because the forward was assumed", fontsize=11, x=0.01, ha="left"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_staleness(chain: pd.DataFrame, path: Path) -> None:
    """Where the dead quotes live."""
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    ages = chain["last_trade_age_days"].to_numpy()
    moneyness = chain["log_moneyness"].to_numpy()
    finite = np.isfinite(ages) & np.isfinite(moneyness)

    axis.scatter(
        moneyness[finite],
        np.maximum(ages[finite], 0.01),
        s=9,
        alpha=0.45,
        color=PALETTE["executable"],
        edgecolors="none",
    )
    axis.axhline(2.0, color=PALETTE["accent"], linewidth=1.0, linestyle="--", label="2-day staleness cut")
    axis.set_yscale("log")
    _style(
        axis,
        "Quote staleness by moneyness: the wings have not traded in months",
        "log-moneyness  log(K/S)",
        "days since last trade (log scale)",
    )
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_box_rates(box_rates: pd.DataFrame, path: Path) -> None:
    """Financing rates implied by box spreads, as a bid/offer range."""
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    usable = box_rates[np.isfinite(box_rates["rate_lend"]) & np.isfinite(box_rates["rate_borrow"])]
    grouped = usable.groupby("days_to_expiry").agg(
        lend=("rate_lend", "median"), borrow=("rate_borrow", "median")
    )

    axis.fill_between(
        grouped.index,
        grouped["lend"] * 100,
        grouped["borrow"] * 100,
        color=PALETTE["executable"],
        alpha=0.25,
        label="executable range (lend / borrow)",
    )
    axis.plot(grouped.index, grouped["lend"] * 100, linewidth=1.2, color=PALETTE["executable"])
    axis.plot(grouped.index, grouped["borrow"] * 100, linewidth=1.2, color=PALETTE["executable"])
    _style(
        axis,
        "Financing rate implied by box spreads",
        "days to expiry",
        "implied rate [%]",
    )
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def make_all_figures(result, directory: Path) -> list[Path]:
    """Render every figure the report references; returns the paths written."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    jobs = [
        ("funnel.png", plot_funnel, result.funnel),
        ("parity_residuals.png", plot_parity_residuals, result.parity),
        ("convention_error.png", plot_convention_error, result.iv_comparison),
        ("quote_staleness.png", plot_staleness, result.chain),
        ("box_implied_rates.png", plot_box_rates, result.box_rates),
    ]
    for name, function, data in jobs:
        if data is None or len(data) == 0:
            continue
        path = directory / name
        function(data, path)
        written.append(path)
    return written
