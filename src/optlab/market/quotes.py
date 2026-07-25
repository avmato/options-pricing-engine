"""Option-chain quote handling: metrics, executable prices, and filtering.

The distinction this module insists on is between the *mid* price and the
*executable* price. Almost every published option study is run on midpoints,
which implicitly assumes you can trade at the middle of the spread. You
cannot: you buy at the ask and sell at the bid. Both price sets are carried
through the pipeline so that any result can be reported twice -- once as it
appears at the mid, once as it would actually have been traded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from optlab.core.timeutils import year_fraction

__all__ = [
    "REQUIRED_COLUMNS",
    "FilterReport",
    "load_chain",
    "add_quote_metrics",
    "filter_quotes",
    "pair_calls_and_puts",
]

REQUIRED_COLUMNS = (
    "strike",
    "bid",
    "ask",
    "option_type",
    "expiration",
    "spot",
    "download_timestamp",
)


@dataclass(frozen=True)
class FilterReport:
    """How many quotes each filter removed, and why.

    Reporting attrition per rule matters because the filters are not neutral:
    a maximum-spread rule removes wing options preferentially, which is
    exactly where the interesting structure lives.
    """

    initial_rows: int
    removed: dict[str, int] = field(default_factory=dict)
    final_rows: int = 0

    def to_frame(self) -> pd.DataFrame:
        """Tabular attrition summary, ordered by rows removed."""
        rows = [
            {"rule": rule, "rows_removed": count, "share_of_initial": count / max(self.initial_rows, 1)}
            for rule, count in self.removed.items()
        ]
        frame = pd.DataFrame(rows).sort_values("rows_removed", ascending=False)
        return frame.reset_index(drop=True)


VENDOR_COLUMN_ALIASES = {
    "lastTradeDate": "last_trade_timestamp",
    "lastPrice": "last_price",
    "openInterest": "open_interest",
    "impliedVolatility": "vendor_implied_volatility",
}


def load_chain(path: str) -> pd.DataFrame:
    """Read a raw option-chain CSV and normalise its columns and dtypes."""
    frame = pd.read_csv(path).rename(columns=VENDOR_COLUMN_ALIASES)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"option chain is missing required columns: {missing}")

    frame["expiration"] = pd.to_datetime(frame["expiration"])
    frame["download_timestamp"] = pd.to_datetime(frame["download_timestamp"], utc=True)
    frame["option_type"] = frame["option_type"].str.lower().str.strip()
    if "last_trade_timestamp" in frame.columns:
        frame["last_trade_timestamp"] = pd.to_datetime(frame["last_trade_timestamp"], utc=True)
    return frame


def add_quote_metrics(chain: pd.DataFrame) -> pd.DataFrame:
    """Add mid, spread, executable prices and time to expiry.

    ``buy_price``/``sell_price`` are the prices a taker actually gets. Every
    arbitrage test in :mod:`optlab.audit` can then be run twice by swapping
    which columns it reads.
    """
    result = chain.copy()

    result["mid"] = 0.5 * (result["bid"] + result["ask"])
    result["spread"] = result["ask"] - result["bid"]
    result["relative_spread"] = result["spread"] / result["mid"].where(result["mid"] > 0)

    # A taker lifts the offer to buy and hits the bid to sell.
    result["buy_price"] = result["ask"]
    result["sell_price"] = result["bid"]

    result["time_to_expiry"] = year_fraction(result["expiration"], result["download_timestamp"])
    result["days_to_expiry"] = result["time_to_expiry"] * 365.0
    result["log_moneyness"] = np.log(result["strike"] / result["spot"])

    result["is_two_sided"] = (result["bid"] > 0.0) & (result["ask"] > result["bid"])

    # Staleness. A quote is only evidence about the market if someone has
    # recently been willing to trade on it; a two-sided quote on a contract
    # that last printed months ago is a leftover, not a price.
    if "last_trade_timestamp" in result.columns:
        age = result["download_timestamp"] - result["last_trade_timestamp"]
        result["last_trade_age_days"] = age.dt.total_seconds() / 86400.0
    else:
        result["last_trade_age_days"] = np.nan
    if "open_interest" not in result.columns:
        result["open_interest"] = np.nan

    return result


def filter_quotes(
    chain: pd.DataFrame,
    *,
    max_relative_spread: float | None = None,
    min_log_moneyness: float | None = None,
    max_log_moneyness: float | None = None,
    min_time_to_expiry: float = 1.0 / 365.0,
    require_two_sided: bool = True,
) -> tuple[pd.DataFrame, FilterReport]:
    """Drop unusable quotes and report what each rule removed.

    Defaults are deliberately permissive. Arbitrage screening is supposed to
    look at the quotes as published; filtering aggressively first would
    quietly remove the very observations under test.
    """
    remaining = chain.copy()
    report_counts: dict[str, int] = {}
    initial = len(remaining)

    def apply(rule: str, keep: pd.Series) -> None:
        nonlocal remaining
        removed = int((~keep).sum())
        if removed:
            report_counts[rule] = removed
        remaining = remaining[keep]

    if require_two_sided:
        apply("not two-sided (bid<=0 or ask<=bid)", remaining["is_two_sided"])
    apply("expired or same-day", remaining["time_to_expiry"] >= min_time_to_expiry)

    if max_relative_spread is not None:
        keep = remaining["relative_spread"].le(max_relative_spread) | remaining["relative_spread"].isna()
        apply(f"relative spread > {max_relative_spread:.0%}", keep)
    if min_log_moneyness is not None:
        apply(f"log-moneyness < {min_log_moneyness}", remaining["log_moneyness"] >= min_log_moneyness)
    if max_log_moneyness is not None:
        apply(f"log-moneyness > {max_log_moneyness}", remaining["log_moneyness"] <= max_log_moneyness)

    report = FilterReport(initial_rows=initial, removed=report_counts, final_rows=len(remaining))
    return remaining.reset_index(drop=True), report


def pair_calls_and_puts(chain: pd.DataFrame) -> pd.DataFrame:
    """Join calls and puts on ``(expiration, strike)`` into one row each.

    Put-call parity is a statement about a *pair*, so every parity-based
    calculation starts here. Strikes quoted on only one side are dropped, and
    the count of such orphans is worth reporting: it is a liquidity signal.
    """
    calls = chain[chain["option_type"] == "call"]
    puts = chain[chain["option_type"] == "put"]

    value_columns = [
        "mid",
        "bid",
        "ask",
        "spread",
        "relative_spread",
        "buy_price",
        "sell_price",
    ]
    keys = ["expiration", "strike", "spot", "time_to_expiry", "days_to_expiry", "log_moneyness"]
    optional = [
        column
        for column in ("volume", "open_interest", "last_trade_age_days")
        if column in chain.columns
    ]

    merged = calls[keys + value_columns + optional].merge(
        puts[["expiration", "strike"] + value_columns + optional],
        on=["expiration", "strike"],
        suffixes=("_call", "_put"),
        how="inner",
    )
    return merged.sort_values(["expiration", "strike"]).reset_index(drop=True)
