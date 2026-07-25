from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.greeks import (
    call_delta,
    call_rho,
    call_theta,
    gamma,
    vega,
)


RISK_FREE_RATE = 0.04

INPUT_PATH = Path(
    "data/processed/SPY_multiple_expiries_with_iv.csv"
)

OUTPUT_DIRECTORY = Path(
    "data/processed"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "SPY_real_call_chain_with_greeks.csv"
)

FIGURES_DIRECTORY = Path(
    "figures"
)

FIGURES_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


DELTA_FIGURE_PATH = (
    FIGURES_DIRECTORY
    / "SPY_real_call_delta.png"
)

GAMMA_FIGURE_PATH = (
    FIGURES_DIRECTORY
    / "SPY_real_call_gamma.png"
)

VEGA_FIGURE_PATH = (
    FIGURES_DIRECTORY
    / "SPY_real_call_vega.png"
)

THETA_FIGURE_PATH = (
    FIGURES_DIRECTORY
    / "SPY_real_call_theta.png"
)


option_chain = pd.read_csv(
    INPUT_PATH
)


call_data = option_chain[
    (
        option_chain["option_type"] == "call"
    )
    & (
        option_chain[
            "calculated_implied_volatility"
        ].notna()
    )
    & (
        option_chain["passes_price_bounds"]
    )
].copy()


call_data["call_delta"] = call_delta(
    spot=call_data["spot"],
    strike=call_data["strike"],
    rate=RISK_FREE_RATE,
    volatility=call_data[
        "calculated_implied_volatility"
    ],
    time_to_expiry=call_data[
        "time_to_expiry"
    ],
)


call_data["gamma"] = gamma(
    spot=call_data["spot"],
    strike=call_data["strike"],
    rate=RISK_FREE_RATE,
    volatility=call_data[
        "calculated_implied_volatility"
    ],
    time_to_expiry=call_data[
        "time_to_expiry"
    ],
)


call_data["vega_per_point"] = vega(
    spot=call_data["spot"],
    strike=call_data["strike"],
    rate=RISK_FREE_RATE,
    volatility=call_data[
        "calculated_implied_volatility"
    ],
    time_to_expiry=call_data[
        "time_to_expiry"
    ],
)


call_data["theta_per_day"] = call_theta(
    spot=call_data["spot"],
    strike=call_data["strike"],
    rate=RISK_FREE_RATE,
    volatility=call_data[
        "calculated_implied_volatility"
    ],
    time_to_expiry=call_data[
        "time_to_expiry"
    ],
    per_day=True,
)


call_data["rho_per_point"] = call_rho(
    spot=call_data["spot"],
    strike=call_data["strike"],
    rate=RISK_FREE_RATE,
    volatility=call_data[
        "calculated_implied_volatility"
    ],
    time_to_expiry=call_data[
        "time_to_expiry"
    ],
    per_rate_point=True,
)


call_data = call_data.sort_values(
    [
        "days_to_expiry",
        "strike",
    ]
).reset_index(
    drop=True
)


print("Number of valid call rows:")
print(
    len(call_data)
)

print()
print("Rows by expiration:")
print(
    call_data.groupby(
        "expiration"
    ).size()
)

print()
print("Greek sample:")
print(
    call_data[
        [
            "expiration",
            "days_to_expiry",
            "strike",
            "spot",
            "calculated_implied_volatility",
            "call_delta",
            "gamma",
            "vega_per_point",
            "theta_per_day",
            "rho_per_point",
        ]
    ].head(20)
)

print()
print("Greek summary:")
print(
    call_data[
        [
            "call_delta",
            "gamma",
            "vega_per_point",
            "theta_per_day",
            "rho_per_point",
        ]
    ].describe()
)


plt.figure(
    figsize=(11, 7)
)

for expiration, expiry_data in (
    call_data.groupby(
        "expiration"
    )
):
    expiry_data = expiry_data.sort_values(
        "log_moneyness"
    )

    days_to_expiry = expiry_data[
        "days_to_expiry"
    ].iloc[0]

    plt.plot(
        expiry_data["log_moneyness"],
        expiry_data["call_delta"],
        marker=".",
        markersize=3,
        linewidth=1,
        label=(
            f"{expiration} "
            f"({days_to_expiry:.1f} days)"
        ),
    )

plt.axvline(
    x=0.0,
    linestyle="--",
    label="ATM",
)

plt.xlabel(
    "Log-Moneyness: log(K / S)"
)

plt.ylabel(
    "Call Delta"
)

plt.title(
    "SPY Call Delta Across Expirations"
)

plt.legend(
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    DELTA_FIGURE_PATH,
    dpi=300,
)

plt.show()


plt.figure(
    figsize=(11, 7)
)

for expiration, expiry_data in (
    call_data.groupby(
        "expiration"
    )
):
    expiry_data = expiry_data.sort_values(
        "log_moneyness"
    )

    days_to_expiry = expiry_data[
        "days_to_expiry"
    ].iloc[0]

    plt.plot(
        expiry_data["log_moneyness"],
        expiry_data["gamma"],
        marker=".",
        markersize=3,
        linewidth=1,
        label=(
            f"{expiration} "
            f"({days_to_expiry:.1f} days)"
        ),
    )

plt.axvline(
    x=0.0,
    linestyle="--",
    label="ATM",
)

plt.xlabel(
    "Log-Moneyness: log(K / S)"
)

plt.ylabel(
    "Gamma"
)

plt.title(
    "SPY Call Gamma Across Expirations"
)

plt.legend(
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    GAMMA_FIGURE_PATH,
    dpi=300,
)

plt.show()


plt.figure(
    figsize=(11, 7)
)

for expiration, expiry_data in (
    call_data.groupby(
        "expiration"
    )
):
    expiry_data = expiry_data.sort_values(
        "log_moneyness"
    )

    days_to_expiry = expiry_data[
        "days_to_expiry"
    ].iloc[0]

    plt.plot(
        expiry_data["log_moneyness"],
        expiry_data["vega_per_point"],
        marker=".",
        markersize=3,
        linewidth=1,
        label=(
            f"{expiration} "
            f"({days_to_expiry:.1f} days)"
        ),
    )

plt.axvline(
    x=0.0,
    linestyle="--",
    label="ATM",
)

plt.xlabel(
    "Log-Moneyness: log(K / S)"
)

plt.ylabel(
    "Vega per 1 Volatility Point"
)

plt.title(
    "SPY Call Vega Across Expirations"
)

plt.legend(
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    VEGA_FIGURE_PATH,
    dpi=300,
)

plt.show()


plt.figure(
    figsize=(11, 7)
)

for expiration, expiry_data in (
    call_data.groupby(
        "expiration"
    )
):
    expiry_data = expiry_data.sort_values(
        "log_moneyness"
    )

    days_to_expiry = expiry_data[
        "days_to_expiry"
    ].iloc[0]

    plt.plot(
        expiry_data["log_moneyness"],
        expiry_data["theta_per_day"],
        marker=".",
        markersize=3,
        linewidth=1,
        label=(
            f"{expiration} "
            f"({days_to_expiry:.1f} days)"
        ),
    )

plt.axvline(
    x=0.0,
    linestyle="--",
    label="ATM",
)

plt.xlabel(
    "Log-Moneyness: log(K / S)"
)

plt.ylabel(
    "Call Theta per Day"
)

plt.title(
    "SPY Call Theta Across Expirations"
)

plt.legend(
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    THETA_FIGURE_PATH,
    dpi=300,
)

plt.show()


call_data.to_csv(
    OUTPUT_PATH,
    index=False,
)


print()
print(
    f"Saved call chain with Greeks to: "
    f"{OUTPUT_PATH}"
)

print(
    f"Saved delta figure to: "
    f"{DELTA_FIGURE_PATH}"
)

print(
    f"Saved gamma figure to: "
    f"{GAMMA_FIGURE_PATH}"
)

print(
    f"Saved vega figure to: "
    f"{VEGA_FIGURE_PATH}"
)

print(
    f"Saved theta figure to: "
    f"{THETA_FIGURE_PATH}"
)