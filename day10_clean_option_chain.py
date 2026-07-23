from pathlib import Path

import pandas as pd

from src.data_cleaning import (
    add_option_quality_columns,
    filter_option_chain,
)

input_path = Path(
    "data/raw/SPY_2026-08-21_option_chain.csv"
)

option_chain = pd.read_csv(input_path)

quality_analysis = add_option_quality_columns(
    option_chain
)

print("Raw shape:")
print(option_chain.shape)

print()
print("Shape after adding quality columns:")
print(quality_analysis.shape)

print()
print(
    quality_analysis[
        [
            "option_type",
            "strike",
            "spot",
            "bid",
            "ask",
            "mid_price",
            "bid_ask_spread",
            "relative_spread",
            "moneyness",
            "log_moneyness",
        ]
    ].head(15)
)

valid_quotes = (
    (quality_analysis["bid"] > 0)
    & (quality_analysis["ask"] > 0)
    & (quality_analysis["ask"] >= quality_analysis["bid"])
    & (quality_analysis["mid_price"] > 0)
)

acceptable_spread = (
    quality_analysis["relative_spread"] <= 0.50
)

acceptable_moneyness = (
    (quality_analysis["moneyness"] >= 0.80)
    & (quality_analysis["moneyness"] <= 1.20)
)

print()
print("Rows with valid bid-ask quotes:")
print(valid_quotes.sum())

print()
print("Rows inside moneyness range:")
print(acceptable_moneyness.sum())

print()
print("Rows with acceptable spread:")
print(acceptable_spread.sum())

print()
print("Rows passing valid quote and moneyness:")
print(
    (
        valid_quotes
        & acceptable_moneyness
    ).sum()
)

print()
print("Rows passing every filter:")
print(
    (
        valid_quotes
        & acceptable_spread
        & acceptable_moneyness
    ).sum()
)

cleaned_option_chain = filter_option_chain(
    quality_analysis,
    maximum_relative_spread=0.50,
    minimum_moneyness=0.80,
    maximum_moneyness=1.20,
)

print()
print("Cleaned shape:")
print(cleaned_option_chain.shape)

print()
print("Remaining option types:")
print(
    cleaned_option_chain[
        "option_type"
    ].value_counts()
)

print()
print("Cleaned quote sample:")
print(
    cleaned_option_chain[
        [
            "option_type",
            "strike",
            "spot",
            "bid",
            "ask",
            "mid_price",
            "relative_spread",
            "moneyness",
        ]
    ].head(20)
)

clean_data_directory = Path("data/clean")

clean_data_directory.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = (
    clean_data_directory
    / "SPY_2026-08-21_clean_option_chain.csv"
)

cleaned_option_chain.to_csv(
    output_path,
    index=False,
)

print()
print(f"Saved cleaned option chain to: {output_path}")