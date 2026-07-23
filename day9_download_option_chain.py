from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ticker_symbol = "SPY"

ticker = yf.Ticker(ticker_symbol)

expiration_dates = ticker.options

selected_expiration = "2026-08-21"

if selected_expiration not in expiration_dates:
    raise ValueError(
        f"{selected_expiration} is not available for {ticker_symbol}."
    )

print(f"Number of available expirations: {len(expiration_dates)}")
print(f"Selected expiration: {selected_expiration}")

option_chain = ticker.option_chain(
    selected_expiration
)

calls = option_chain.calls.copy()
puts = option_chain.puts.copy()

print()
print("Calls:")
print(calls.head())

print()
print("Puts:")
print(puts.head())

spot = float(ticker.fast_info["last_price"])
download_timestamp = datetime.now(
    timezone.utc
).isoformat()

calls["ticker"] = ticker_symbol
calls["option_type"] = "call"
calls["expiration"] = selected_expiration
calls["spot"] = spot
calls["download_timestamp"] = download_timestamp

puts["ticker"] = ticker_symbol
puts["option_type"] = "put"
puts["expiration"] = selected_expiration
puts["spot"] = spot
puts["download_timestamp"] = download_timestamp

print()
print(f"Spot price: {spot:.2f}")

print()
print("Call metadata:")
print(
    calls[
        [
            "ticker",
            "option_type",
            "expiration",
            "spot",
            "strike",
            "bid",
            "ask",
        ]
    ].head()
)

print()
print("Put metadata:")
print(
    puts[
        [
            "ticker",
            "option_type",
            "expiration",
            "spot",
            "strike",
            "bid",
            "ask",
        ]
    ].head()
)

option_chain_data = pd.concat(
    [calls, puts],
    ignore_index=True,
)

print()
print(f"Number of call rows: {len(calls)}")
print(f"Number of put rows: {len(puts)}")
print(f"Total option rows: {len(option_chain_data)}")

print()
print("Option types:")
print(option_chain_data["option_type"].value_counts())

raw_data_directory = Path("data/raw")

raw_data_directory.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = (
    raw_data_directory
    / f"{ticker_symbol}_{selected_expiration}_option_chain.csv"
)

option_chain_data.to_csv(
    output_path,
    index=False,
)

print()
print(f"Saved raw option chain to: {output_path}")

saved_option_chain = pd.read_csv(output_path)

print()
print("Saved CSV shape:")
print(saved_option_chain.shape)

print()
print("Saved CSV columns:")
print(saved_option_chain.columns.tolist())

print()
print("Missing values in key columns:")
print(
    saved_option_chain[
        [
            "strike",
            "bid",
            "ask",
            "option_type",
            "expiration",
            "spot",
        ]
    ].isna().sum()
)