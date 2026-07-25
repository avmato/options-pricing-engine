"""Static no-arbitrage checks on a raw option chain.

Every check below is a *static* arbitrage test: it constructs a portfolio
whose payoff at expiry is non-negative in every state of the world, and asks
whether that portfolio can be put on for a credit. No volatility model, no
distributional assumption, and no view on the underlying is involved -- which
is what makes a violation a genuine finding rather than a disagreement with
Black-Scholes.

Each check runs on two price bases:

``mid``
    Both legs priced at the midpoint. This is what almost every published
    study uses, and it silently assumes you can trade in the middle of the
    spread.
``executable``
    Legs you buy are priced at the ask, legs you sell at the bid. This is what
    a taker would actually pay.

The gap between the two is the entire point of the exercise: an "arbitrage"
that only exists at the mid is not an arbitrage, it is the spread.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "PriceBasis",
    "VIOLATION_COLUMNS",
    "check_price_bounds",
    "check_strike_monotonicity",
    "check_vertical_spread_cap",
    "check_butterfly_convexity",
    "check_box_spread",
    "check_calendar_monotonicity",
    "parity_diagnostics",
    "box_implied_rates",
]

PriceBasis = Literal["mid", "executable"]
ExerciseStyle = Literal["american", "european"]

# Listed US equity options are American. That is not a detail: early exercise
# widens the no-arbitrage band, and screening American quotes against European
# bounds manufactures violations that the exercise right fully explains. Every
# bound below is therefore parameterised by exercise style, defaulting to the
# one the instruments actually have.

# Edges below this are numerical dust, not trades. Listed option prices live
# on a penny grid, so nothing under a small fraction of a cent can be real;
# without a floor, exact arithmetic ties get reported as arbitrage.
DEFAULT_TOLERANCE = 1e-9

VIOLATION_COLUMNS = [
    "check",
    "basis",
    "expiration",
    "days_to_expiry",
    "option_type",
    "strikes",
    "edge",
    "edge_per_dollar_notional",
    "log_moneyness",
    "worst_relative_spread",
    "max_last_trade_age_days",
    "min_open_interest",
    "detail",
]


def _quality(legs: list[pd.Series]) -> dict[str, float]:
    """Worst-case data quality across the legs of one violating portfolio.

    A static arbitrage is only as trustworthy as its least trustworthy leg,
    so the aggregation is deliberately pessimistic: the widest spread, the
    stalest print, the thinnest open interest.
    """

    def _worst(field: str, reducer) -> float:
        values = [float(leg[field]) for leg in legs if field in leg and pd.notna(leg[field])]
        return float(reducer(values)) if values else float("nan")

    return {
        "worst_relative_spread": _worst("relative_spread", max),
        "max_last_trade_age_days": _worst("last_trade_age_days", max),
        "min_open_interest": _worst("open_interest", min),
    }


def _leg(frame: pd.DataFrame, side: Literal["buy", "sell"], basis: PriceBasis) -> pd.Series:
    """Price of one leg, given whether we are buying or selling it."""
    if basis == "mid":
        return frame["mid"]
    if basis == "executable":
        return frame["ask"] if side == "buy" else frame["bid"]
    raise ValueError(f"unknown price basis: {basis!r}")


def _empty_violations() -> pd.DataFrame:
    return pd.DataFrame(columns=VIOLATION_COLUMNS)


def _collect(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return _empty_violations()
    return pd.DataFrame(rows)[VIOLATION_COLUMNS]


def check_price_bounds(
    chain: pd.DataFrame,
    forwards: pd.DataFrame,
    basis: PriceBasis = "executable",
    tolerance: float = DEFAULT_TOLERANCE,
    exercise_style: ExerciseStyle = "american",
) -> pd.DataFrame:
    """Every option must trade between its intrinsic value and its cap.

    For a European option the band is ``D max(w(F-K), 0) <= V <= D F`` for a
    call and ``<= D K`` for a put. An American option can be exercised today,
    which raises the floor to the immediate exercise value ``max(w(S-K), 0)``
    and raises the cap to ``S`` for a call and ``K`` for a put.

    The put floor is where the two conventions differ most: for a deep
    in-the-money put the right to exercise now is worth strictly more than
    the discounted European payoff, so the European floor is too low and
    genuine violations slip through it.
    """
    merged = chain.merge(
        forwards[["expiration", "forward", "discount"]], on="expiration", how="inner"
    )
    if merged.empty:
        return _empty_violations()

    sign = np.where(merged["option_type"] == "call", 1.0, -1.0)
    lower = merged["discount"] * np.maximum(sign * (merged["forward"] - merged["strike"]), 0.0)
    upper = merged["discount"] * np.where(sign > 0, merged["forward"], merged["strike"])

    if exercise_style == "american":
        immediate_exercise = np.maximum(sign * (merged["spot"] - merged["strike"]), 0.0)
        lower = np.maximum(lower, immediate_exercise)
        upper = np.where(sign > 0, merged["spot"], merged["strike"])

    buy_price = _leg(merged, "buy", basis)
    sell_price = _leg(merged, "sell", basis)

    rows: list[dict[str, object]] = []
    for label, edge in (
        ("below intrinsic (buy)", lower - buy_price),
        ("above cap (sell)", sell_price - upper),
    ):
        hits = merged[edge > tolerance]
        for index, row in hits.iterrows():
            rows.append(
                {
                    "check": "price_bounds",
                    "basis": basis,
                    "expiration": row["expiration"],
                    "days_to_expiry": row["days_to_expiry"],
                    "option_type": row["option_type"],
                    "strikes": f"{row['strike']:.2f}",
                    "edge": float(edge.loc[index]),
                    "edge_per_dollar_notional": float(edge.loc[index] / row["strike"]),
                    "log_moneyness": float(row["log_moneyness"]),
                    **_quality([row]),
                    "detail": label,
                }
            )
    return _collect(rows)


def check_strike_monotonicity(
    chain: pd.DataFrame,
    basis: PriceBasis = "executable",
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """Calls must fall and puts must rise as the strike rises.

    If a lower-strike call is cheaper than a higher-strike one, buy the low
    strike and sell the high strike: the resulting call spread can never pay
    less than zero, so any credit received is free money.
    """
    rows: list[dict[str, object]] = []
    for (expiration, option_type), chunk in chain.groupby(["expiration", "option_type"], sort=True):
        ordered = chunk.sort_values("strike").reset_index(drop=True)
        if len(ordered) < 2:
            continue

        lower_leg = ordered.iloc[:-1].reset_index(drop=True)
        upper_leg = ordered.iloc[1:].reset_index(drop=True)

        # Long the spread that is guaranteed non-negative at expiry:
        # calls -> long low strike, short high strike; puts -> the reverse.
        if option_type == "call":
            cost = _leg(lower_leg, "buy", basis) - _leg(upper_leg, "sell", basis)
        else:
            cost = _leg(upper_leg, "buy", basis) - _leg(lower_leg, "sell", basis)

        breaches = cost < -tolerance
        for position in np.flatnonzero(breaches.to_numpy()):
            low_strike = float(lower_leg.loc[position, "strike"])
            high_strike = float(upper_leg.loc[position, "strike"])
            rows.append(
                {
                    "check": "strike_monotonicity",
                    "basis": basis,
                    "expiration": expiration,
                    "days_to_expiry": float(lower_leg.loc[position, "days_to_expiry"]),
                    "option_type": option_type,
                    "strikes": f"{low_strike:.2f}/{high_strike:.2f}",
                    "edge": float(-cost.iloc[position]),
                    "edge_per_dollar_notional": float(-cost.iloc[position] / (high_strike - low_strike)),
                    "log_moneyness": float(lower_leg.loc[position, "log_moneyness"]),
                    **_quality([lower_leg.loc[position], upper_leg.loc[position]]),
                    "detail": "non-negative spread available for a credit",
                }
            )
    return _collect(rows)


def check_vertical_spread_cap(
    chain: pd.DataFrame,
    forwards: pd.DataFrame,
    basis: PriceBasis = "executable",
    tolerance: float = DEFAULT_TOLERANCE,
    exercise_style: ExerciseStyle = "american",
) -> pd.DataFrame:
    """A vertical spread cannot be worth more than the strike gap it can pay.

    The payoff is capped at ``K2 - K1``. A European spread pays that only at
    expiry, so its value is capped at ``D (K2 - K1)``; an American spread can
    be assigned early, so the cap is the undiscounted ``K2 - K1``. Using the
    European cap on American quotes flags the difference -- a few cents per
    dollar of strike gap -- as arbitrage.
    """
    discounts = forwards.set_index("expiration")["discount"]
    rows: list[dict[str, object]] = []

    for (expiration, option_type), chunk in chain.groupby(["expiration", "option_type"], sort=True):
        if expiration not in discounts.index:
            continue
        discount = float(discounts.loc[expiration])
        ordered = chunk.sort_values("strike").reset_index(drop=True)
        if len(ordered) < 2:
            continue

        lower_leg = ordered.iloc[:-1].reset_index(drop=True)
        upper_leg = ordered.iloc[1:].reset_index(drop=True)
        gap = upper_leg["strike"] - lower_leg["strike"]

        if option_type == "call":
            credit = _leg(lower_leg, "sell", basis) - _leg(upper_leg, "buy", basis)
        else:
            credit = _leg(upper_leg, "sell", basis) - _leg(lower_leg, "buy", basis)

        cap = gap if exercise_style == "american" else discount * gap
        edge = credit - cap
        for position in np.flatnonzero((edge > tolerance).to_numpy()):
            low_strike = float(lower_leg.loc[position, "strike"])
            high_strike = float(upper_leg.loc[position, "strike"])
            rows.append(
                {
                    "check": "vertical_spread_cap",
                    "basis": basis,
                    "expiration": expiration,
                    "days_to_expiry": float(lower_leg.loc[position, "days_to_expiry"]),
                    "option_type": option_type,
                    "strikes": f"{low_strike:.2f}/{high_strike:.2f}",
                    "edge": float(edge.iloc[position]),
                    "edge_per_dollar_notional": float(edge.iloc[position] / gap.iloc[position]),
                    "log_moneyness": float(lower_leg.loc[position, "log_moneyness"]),
                    **_quality([lower_leg.loc[position], upper_leg.loc[position]]),
                    "detail": "spread sold above its maximum discounted payoff",
                }
            )
    return _collect(rows)


def check_butterfly_convexity(
    chain: pd.DataFrame,
    basis: PriceBasis = "executable",
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """Option prices must be convex in the strike.

    For ``K1 < K2 < K3`` the butterfly ``w1 V(K1) - V(K2) + w3 V(K3)`` with
    ``w1 = (K3-K2)/(K3-K1)`` and ``w3 = (K2-K1)/(K3-K1)`` has a non-negative
    payoff everywhere, so it must cost something. The weights handle unequal
    strike spacing, which matters because real chains are denser near the
    money.

    A negative butterfly price is also exactly a negative risk-neutral
    probability density around ``K2``: convexity in strike and a valid implied
    distribution are the same statement.
    """
    rows: list[dict[str, object]] = []
    for (expiration, option_type), chunk in chain.groupby(["expiration", "option_type"], sort=True):
        ordered = chunk.sort_values("strike").reset_index(drop=True)
        if len(ordered) < 3:
            continue

        left = ordered.iloc[:-2].reset_index(drop=True)
        middle = ordered.iloc[1:-1].reset_index(drop=True)
        right = ordered.iloc[2:].reset_index(drop=True)

        span = right["strike"] - left["strike"]
        left_weight = (right["strike"] - middle["strike"]) / span
        right_weight = (middle["strike"] - left["strike"]) / span

        cost = (
            left_weight * _leg(left, "buy", basis)
            + right_weight * _leg(right, "buy", basis)
            - _leg(middle, "sell", basis)
        )

        for position in np.flatnonzero((cost < -tolerance).to_numpy()):
            strikes = (
                float(left.loc[position, "strike"]),
                float(middle.loc[position, "strike"]),
                float(right.loc[position, "strike"]),
            )
            rows.append(
                {
                    "check": "butterfly_convexity",
                    "basis": basis,
                    "expiration": expiration,
                    "days_to_expiry": float(middle.loc[position, "days_to_expiry"]),
                    "option_type": option_type,
                    "strikes": "/".join(f"{value:.2f}" for value in strikes),
                    "edge": float(-cost.iloc[position]),
                    "edge_per_dollar_notional": float(-cost.iloc[position] / (strikes[2] - strikes[0])),
                    "log_moneyness": float(middle.loc[position, "log_moneyness"]),
                    **_quality([left.loc[position], middle.loc[position], right.loc[position]]),
                    "detail": "negative butterfly price implies negative implied density",
                }
            )
    return _collect(rows)


def _wide_by_type(chunk: pd.DataFrame, basis: PriceBasis) -> pd.DataFrame:
    """Reshape one expiry into per-strike call/put buy and sell prices."""
    frames = {}
    for option_type in ("call", "put"):
        side = chunk[chunk["option_type"] == option_type].set_index("strike")
        if side.empty:
            return pd.DataFrame()
        frames[f"{option_type}_buy"] = _leg(side, "buy", basis)
        frames[f"{option_type}_sell"] = _leg(side, "sell", basis)
        frames[f"{option_type}_relative_spread"] = side["relative_spread"]
        for field in ("last_trade_age_days", "open_interest"):
            if field in side.columns:
                frames[f"{option_type}_{field}"] = side[field]
        frames["log_moneyness"] = side["log_moneyness"]
    price_columns = [column for column in frames if column.endswith(("_buy", "_sell"))]
    wide = pd.DataFrame(frames).dropna(subset=price_columns)
    return wide.sort_index()


def _box_quality(lower: pd.Series, upper: pd.Series) -> dict[str, float]:
    """Worst-case quality across the four legs of a box spread."""
    legs = [
        pd.Series(
            {
                "relative_spread": row.get(f"{option_type}_relative_spread", np.nan),
                "last_trade_age_days": row.get(f"{option_type}_last_trade_age_days", np.nan),
                "open_interest": row.get(f"{option_type}_open_interest", np.nan),
            }
        )
        for row in (lower, upper)
        for option_type in ("call", "put")
    ]
    return _quality(legs)


def check_box_spread(
    chain: pd.DataFrame,
    forwards: pd.DataFrame,
    basis: PriceBasis = "executable",
    tolerance: float = DEFAULT_TOLERANCE,
    exercise_style: ExerciseStyle = "american",
) -> pd.DataFrame:
    """A box spread is a synthetic zero-coupon bond and must be priced like one.

    Long a ``K1/K2`` box -- long call and short put at ``K1``, short call and
    long put at ``K2`` -- pays exactly ``K2 - K1`` at expiry in every state.
    It is therefore worth ``D (K2 - K1)`` today, with no reference to the
    underlying at all. Paying less than that, or receiving more than that for
    the short box, is riskless profit.

    Exercise style breaks the symmetry between the two directions. Buying the
    box below ``D (K2 - K1)`` is arbitrage under either convention, since the
    American exercise right can only add value. Selling it, however, exposes
    you to early assignment, and an American box can be worth up to the
    undiscounted ``K2 - K1`` -- so the short-box test uses that wider cap.
    Testing the short box against the European value is the single largest
    source of false positives in this whole suite.

    The box is also the cleanest instrument in the chain for reading the
    market's financing rate, because it removes the underlying entirely;
    see :func:`box_implied_rates`.
    """
    discounts = forwards.set_index("expiration")["discount"]
    rows: list[dict[str, object]] = []

    for expiration, chunk in chain.groupby("expiration", sort=True):
        if expiration not in discounts.index:
            continue
        discount = float(discounts.loc[expiration])
        wide = _wide_by_type(chunk, basis)
        if len(wide) < 2:
            continue

        lower = wide.iloc[:-1]
        upper = wide.iloc[1:]
        gap = upper.index.to_numpy(dtype=float) - lower.index.to_numpy(dtype=float)
        days = float(chunk["days_to_expiry"].iloc[0])

        long_cost = (
            lower["call_buy"].to_numpy()
            - upper["call_sell"].to_numpy()
            + upper["put_buy"].to_numpy()
            - lower["put_sell"].to_numpy()
        )
        short_credit = (
            lower["call_sell"].to_numpy()
            - upper["call_buy"].to_numpy()
            + upper["put_sell"].to_numpy()
            - lower["put_buy"].to_numpy()
        )

        european_value = discount * gap
        # Long: pay less than the European value and the exercise right is free.
        # Short: you can be assigned early, so the binding cap is undiscounted.
        short_cap = gap if exercise_style == "american" else european_value
        for label, edge in (
            ("long box cheaper than its guaranteed payoff", european_value - long_cost),
            ("short box credit exceeds its maximum value", short_credit - short_cap),
        ):
            for position in np.flatnonzero(edge > tolerance):
                rows.append(
                    {
                        "check": "box_spread",
                        "basis": basis,
                        "expiration": expiration,
                        "days_to_expiry": days,
                        "option_type": "box",
                        "strikes": f"{lower.index[position]:.2f}/{upper.index[position]:.2f}",
                        "edge": float(edge[position]),
                        "edge_per_dollar_notional": float(edge[position] / gap[position]),
                        "log_moneyness": float(lower["log_moneyness"].iloc[position]),
                        **_box_quality(lower.iloc[position], upper.iloc[position]),
                        "detail": label,
                    }
                )
    return _collect(rows)


def check_calendar_monotonicity(
    chain: pd.DataFrame,
    basis: PriceBasis = "executable",
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """For American options, more time cannot be worth less.

    Holding the strike fixed, a longer-dated American option contains every
    right the shorter-dated one does, so it must cost at least as much. If it
    does not, buy the long-dated option and sell the short-dated one for a
    credit and exercise the resulting optionality for free.

    The same statement for European options requires equal forwards, which is
    why this check is only applied to American-style listed equity options.
    """
    rows: list[dict[str, object]] = []
    for (strike, option_type), chunk in chain.groupby(["strike", "option_type"], sort=True):
        ordered = chunk.sort_values("time_to_expiry").reset_index(drop=True)
        if len(ordered) < 2:
            continue

        near = ordered.iloc[:-1].reset_index(drop=True)
        far = ordered.iloc[1:].reset_index(drop=True)
        cost = _leg(far, "buy", basis) - _leg(near, "sell", basis)

        for position in np.flatnonzero((cost < -tolerance).to_numpy()):
            rows.append(
                {
                    "check": "calendar_monotonicity",
                    "basis": basis,
                    "expiration": far.loc[position, "expiration"],
                    "days_to_expiry": float(far.loc[position, "days_to_expiry"]),
                    "option_type": option_type,
                    "strikes": f"{float(strike):.2f}",
                    "edge": float(-cost.iloc[position]),
                    "edge_per_dollar_notional": float(-cost.iloc[position] / float(strike)),
                    "log_moneyness": float(far.loc[position, "log_moneyness"]),
                    **_quality([near.loc[position], far.loc[position]]),
                    "detail": (
                        f"{near.loc[position, 'days_to_expiry']:.0f}d vs "
                        f"{far.loc[position, 'days_to_expiry']:.0f}d"
                    ),
                }
            )
    return _collect(rows)


def parity_diagnostics(pairs: pd.DataFrame, forwards: pd.DataFrame) -> pd.DataFrame:
    """Put-call parity residuals at the mid, strike by strike.

    Reported as a diagnostic rather than as an arbitrage: closing the trade
    requires the underlying, so the executable version of this test is the
    box spread, which stays inside the options market.

    The residual is still informative -- it is the cleanest single measure of
    how internally consistent a vendor's snapshot is.
    """
    merged = pairs.merge(forwards[["expiration", "forward", "discount"]], on="expiration", how="inner")
    if merged.empty:
        return pd.DataFrame()

    result = merged[["expiration", "strike", "days_to_expiry", "log_moneyness"]].copy()
    result["parity_residual"] = (merged["mid_call"] - merged["mid_put"]) - merged["discount"] * (
        merged["forward"] - merged["strike"]
    )
    result["combined_half_spread"] = 0.5 * (merged["spread_call"] + merged["spread_put"])
    # A residual inside the combined half-spread is indistinguishable from
    # quote noise; only the excess is potentially real.
    result["residual_beyond_spread"] = np.maximum(
        np.abs(result["parity_residual"]) - result["combined_half_spread"], 0.0
    )
    return result


def box_implied_rates(chain: pd.DataFrame, basis: PriceBasis = "executable") -> pd.DataFrame:
    """Financing rate implied by every adjacent-strike box spread.

    A box costing ``B`` for a guaranteed ``K2 - K1`` at expiry implies a
    discount factor ``B / (K2 - K1)`` and hence a rate ``-log(D)/T``. Because
    the box has no exposure to the underlying, this is the market's own
    borrowing/lending rate -- and unlike the parity regression it does not
    require the slope of a noisy line, so it stays well conditioned at short
    maturities.

    At the executable basis the long and short boxes bracket the true rate,
    giving a bid/offer range for financing rather than a point estimate.
    """
    records: list[dict[str, object]] = []
    for expiration, chunk in chain.groupby("expiration", sort=True):
        wide = _wide_by_type(chunk, basis)
        if len(wide) < 2:
            continue
        time_to_expiry = float(chunk["time_to_expiry"].iloc[0])
        if time_to_expiry <= 0:
            continue

        lower = wide.iloc[:-1]
        upper = wide.iloc[1:]
        gap = upper.index.to_numpy(dtype=float) - lower.index.to_numpy(dtype=float)

        long_cost = (
            lower["call_buy"].to_numpy()
            - upper["call_sell"].to_numpy()
            + upper["put_buy"].to_numpy()
            - lower["put_sell"].to_numpy()
        )
        short_credit = (
            lower["call_sell"].to_numpy()
            - upper["call_buy"].to_numpy()
            + upper["put_sell"].to_numpy()
            - lower["put_buy"].to_numpy()
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            # Paying more for the box means lending at a lower rate, so the
            # long-box cost maps to the low end of the implied rate range.
            rate_from_long = -np.log(np.clip(long_cost / gap, 1e-12, None)) / time_to_expiry
            rate_from_short = -np.log(np.clip(short_credit / gap, 1e-12, None)) / time_to_expiry

        for position in range(len(gap)):
            records.append(
                {
                    "expiration": expiration,
                    "time_to_expiry": time_to_expiry,
                    "days_to_expiry": float(chunk["days_to_expiry"].iloc[0]),
                    "strike_low": float(lower.index[position]),
                    "strike_high": float(upper.index[position]),
                    "strike_gap": float(gap[position]),
                    "basis": basis,
                    "rate_lend": float(rate_from_long[position]),
                    "rate_borrow": float(rate_from_short[position]),
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["rate_mid"] = 0.5 * (frame["rate_lend"] + frame["rate_borrow"])
    return frame
