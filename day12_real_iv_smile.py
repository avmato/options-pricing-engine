from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


input_path = Path(
    "data/processed/SPY_2026-08-21_option_chain_with_iv.csv"
)

option_chain = pd.read_csv(input_path)


valid_iv_data = option_chain[
    option_chain[
        "calculated_implied_volatility"
    ].notna()
].copy()


call_data = (
    valid_iv_data[
        valid_iv_data["option_type"] == "call"
    ]
    .sort_values("strike")
    .copy()
)

put_data = (
    valid_iv_data[
        valid_iv_data["option_type"] == "put"
    ]
    .sort_values("strike")
    .copy()
)


spot = valid_iv_data["spot"].iloc[0]


otm_put_data = valid_iv_data[
    (valid_iv_data["option_type"] == "put")
    & (valid_iv_data["strike"] < spot)
].copy()

otm_call_data = valid_iv_data[
    (valid_iv_data["option_type"] == "call")
    & (valid_iv_data["strike"] >= spot)
].copy()


otm_smile_data = pd.concat(
    [
        otm_put_data,
        otm_call_data,
    ],
    ignore_index=True,
)

otm_smile_data = otm_smile_data.sort_values(
    "strike"
).reset_index(
    drop=True
)


print("Total rows with valid IV:")
print(len(valid_iv_data))

print()
print("Valid call IV rows:")
print(len(call_data))

print()
print("Valid put IV rows:")
print(len(put_data))

print()
print("OTM put rows:")
print(len(otm_put_data))

print()
print("OTM call rows:")
print(len(otm_call_data))

print()
print("Total OTM smile rows:")
print(len(otm_smile_data))

print()
print("OTM smile sample:")
print(
    otm_smile_data[
        [
            "option_type",
            "strike",
            "moneyness",
            "log_moneyness",
            "calculated_implied_volatility",
        ]
    ].head(15)
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


call_put_figure_path = (
    figures_directory
    / "SPY_2026-08-21_call_put_iv.png"
)

otm_strike_figure_path = (
    figures_directory
    / "SPY_2026-08-21_otm_iv_by_strike.png"
)

otm_log_moneyness_figure_path = (
    figures_directory
    / "SPY_2026-08-21_otm_iv_by_log_moneyness.png"
)

otm_data_output_path = (
    processed_directory
    / "SPY_2026-08-21_otm_iv_smile.csv"
)


plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    call_data["strike"],
    call_data[
        "calculated_implied_volatility"
    ],
    alpha=0.7,
    label="Calls",
)

plt.scatter(
    put_data["strike"],
    put_data[
        "calculated_implied_volatility"
    ],
    alpha=0.7,
    label="Puts",
)

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
    "SPY Call and Put Implied Volatility"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    call_put_figure_path,
    dpi=300,
)

plt.show()


plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    otm_put_data["strike"],
    otm_put_data[
        "calculated_implied_volatility"
    ],
    alpha=0.8,
    label="OTM Puts",
)

plt.scatter(
    otm_call_data["strike"],
    otm_call_data[
        "calculated_implied_volatility"
    ],
    alpha=0.8,
    label="OTM Calls",
)

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
    "SPY OTM Implied Volatility Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    otm_strike_figure_path,
    dpi=300,
)

plt.show()


plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    otm_put_data["log_moneyness"],
    otm_put_data[
        "calculated_implied_volatility"
    ],
    alpha=0.8,
    label="OTM Puts",
)

plt.scatter(
    otm_call_data["log_moneyness"],
    otm_call_data[
        "calculated_implied_volatility"
    ],
    alpha=0.8,
    label="OTM Calls",
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
    "SPY OTM Implied Volatility by Log-Moneyness"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    otm_log_moneyness_figure_path,
    dpi=300,
)

plt.show()


otm_smile_data.to_csv(
    otm_data_output_path,
    index=False,
)


print()
print(
    f"Saved call-put IV figure to: "
    f"{call_put_figure_path}"
)

print(
    f"Saved OTM strike figure to: "
    f"{otm_strike_figure_path}"
)

print(
    f"Saved OTM log-moneyness figure to: "
    f"{otm_log_moneyness_figure_path}"
)

print(
    f"Saved OTM smile data to: "
    f"{otm_data_output_path}"
)