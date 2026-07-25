from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


input_path = Path(
    "data/processed/SPY_multiple_expiries_with_iv.csv"
)

option_chain = pd.read_csv(
    input_path
)


valid_iv_data = option_chain[
    option_chain[
        "calculated_implied_volatility"
    ].notna()
].copy()


valid_iv_data = valid_iv_data[
    valid_iv_data["passes_price_bounds"]
].copy()


otm_put_data = valid_iv_data[
    (
        valid_iv_data["option_type"]
        == "put"
    )
    & (
        valid_iv_data["strike"]
        < valid_iv_data["spot"]
    )
].copy()


otm_call_data = valid_iv_data[
    (
        valid_iv_data["option_type"]
        == "call"
    )
    & (
        valid_iv_data["strike"]
        >= valid_iv_data["spot"]
    )
].copy()


otm_smile_data = pd.concat(
    [
        otm_put_data,
        otm_call_data,
    ],
    ignore_index=True,
)


otm_smile_data = (
    otm_smile_data
    .sort_values(
        [
            "days_to_expiry",
            "strike",
        ]
    )
    .reset_index(
        drop=True
    )
)


expiration_summary = (
    otm_smile_data
    .groupby("expiration")
    .agg(
        days_to_expiry=(
            "days_to_expiry",
            "first",
        ),
        number_of_rows=(
            "strike",
            "size",
        ),
        minimum_iv=(
            "calculated_implied_volatility",
            "min",
        ),
        median_iv=(
            "calculated_implied_volatility",
            "median",
        ),
        maximum_iv=(
            "calculated_implied_volatility",
            "max",
        ),
    )
    .sort_values(
        "days_to_expiry"
    )
)


atm_rows = []

for expiration, expiry_data in (
    valid_iv_data.groupby(
        "expiration"
    )
):
    expiry_data = expiry_data.copy()

    expiry_data[
        "absolute_log_moneyness"
    ] = (
        expiry_data[
            "log_moneyness"
        ].abs()
    )

    atm_row = expiry_data.loc[
        expiry_data[
            "absolute_log_moneyness"
        ].idxmin()
    ]

    atm_rows.append(
        atm_row
    )


atm_term_structure = pd.DataFrame(
    atm_rows
)

atm_term_structure = (
    atm_term_structure
    .sort_values(
        "days_to_expiry"
    )
    .reset_index(
        drop=True
    )
)


print("Valid IV rows:")
print(
    len(valid_iv_data)
)

print()
print("OTM put rows:")
print(
    len(otm_put_data)
)

print()
print("OTM call rows:")
print(
    len(otm_call_data)
)

print()
print("Total OTM rows:")
print(
    len(otm_smile_data)
)

print()
print("OTM rows by expiration:")
print(
    otm_smile_data.groupby(
        "expiration"
    ).size()
)

print()
print("Expiration summary:")
print(
    expiration_summary
)

print()
print("ATM term structure:")
print(
    atm_term_structure[
        [
            "expiration",
            "days_to_expiry",
            "option_type",
            "strike",
            "spot",
            "log_moneyness",
            "calculated_implied_volatility",
        ]
    ]
)


figures_directory = Path(
    "figures"
)

figures_directory.mkdir(
    parents=True,
    exist_ok=True,
)


processed_directory = Path(
    "data/processed"
)

processed_directory.mkdir(
    parents=True,
    exist_ok=True,
)


strike_figure_path = (
    figures_directory
    / "SPY_multi_expiry_otm_iv_by_strike.png"
)

log_moneyness_figure_path = (
    figures_directory
    / "SPY_multi_expiry_otm_iv_by_log_moneyness.png"
)

atm_term_structure_figure_path = (
    figures_directory
    / "SPY_atm_iv_term_structure.png"
)

otm_data_output_path = (
    processed_directory
    / "SPY_multi_expiry_otm_smiles.csv"
)

summary_output_path = (
    processed_directory
    / "SPY_multi_expiry_summary.csv"
)

atm_term_structure_output_path = (
    processed_directory
    / "SPY_atm_iv_term_structure.csv"
)


plt.figure(
    figsize=(11, 7)
)


for expiration, expiry_data in (
    otm_smile_data.groupby(
        "expiration"
    )
):
    expiry_data = expiry_data.sort_values(
        "strike"
    )

    days_to_expiry = (
        expiry_data[
            "days_to_expiry"
        ].iloc[0]
    )

    label = (
        f"{expiration} "
        f"({days_to_expiry:.1f} days)"
    )

    plt.plot(
        expiry_data["strike"],
        expiry_data[
            "calculated_implied_volatility"
        ],
        marker=".",
        markersize=4,
        linewidth=1,
        alpha=0.8,
        label=label,
    )


spot = otm_smile_data[
    "spot"
].iloc[0]


plt.axvline(
    x=spot,
    linestyle="--",
    label="Spot",
)

plt.xlabel(
    "Strike"
)

plt.ylabel(
    "Implied Volatility"
)

plt.title(
    "SPY OTM Implied Volatility Smiles "
    "Across Expirations"
)

plt.legend(
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    strike_figure_path,
    dpi=300,
)

plt.show()


plt.figure(
    figsize=(11, 7)
)


for expiration, expiry_data in (
    otm_smile_data.groupby(
        "expiration"
    )
):
    expiry_data = expiry_data.sort_values(
        "log_moneyness"
    )

    days_to_expiry = (
        expiry_data[
            "days_to_expiry"
        ].iloc[0]
    )

    label = (
        f"{expiration} "
        f"({days_to_expiry:.1f} days)"
    )

    plt.plot(
        expiry_data[
            "log_moneyness"
        ],
        expiry_data[
            "calculated_implied_volatility"
        ],
        marker=".",
        markersize=4,
        linewidth=1,
        alpha=0.8,
        label=label,
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
    "Implied Volatility"
)

plt.title(
    "SPY OTM Implied Volatility Smiles "
    "by Log-Moneyness"
)

plt.legend(
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    log_moneyness_figure_path,
    dpi=300,
)

plt.show()


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    atm_term_structure[
        "days_to_expiry"
    ],
    atm_term_structure[
        "calculated_implied_volatility"
    ],
    marker="o",
)

plt.xlabel(
    "Days to Expiry"
)

plt.ylabel(
    "ATM Implied Volatility"
)

plt.title(
    "SPY ATM Implied Volatility Term Structure"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    atm_term_structure_figure_path,
    dpi=300,
)

plt.show()


otm_smile_data.to_csv(
    otm_data_output_path,
    index=False,
)


expiration_summary.to_csv(
    summary_output_path
)


atm_term_structure.to_csv(
    atm_term_structure_output_path,
    index=False,
)


print()
print(
    f"Saved strike smile figure to: "
    f"{strike_figure_path}"
)

print(
    f"Saved log-moneyness smile figure to: "
    f"{log_moneyness_figure_path}"
)

print(
    f"Saved ATM term structure figure to: "
    f"{atm_term_structure_figure_path}"
)

print(
    f"Saved multi-expiry OTM data to: "
    f"{otm_data_output_path}"
)

print(
    f"Saved expiration summary to: "
    f"{summary_output_path}"
)

print(
    f"Saved ATM term structure data to: "
    f"{atm_term_structure_output_path}"
)