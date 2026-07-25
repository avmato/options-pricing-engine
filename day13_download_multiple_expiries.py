from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


TICKER_SYMBOL = "SPY"

TARGET_DAYS_TO_EXPIRY = [
    7,
    14,
    30,
    60,
    90,
]

RAW_DATA_DIRECTORY = Path(
    "data/raw/multiple_expiries"
)

RAW_DATA_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


ticker = yf.Ticker(
    TICKER_SYMBOL
)

available_expirations = list(
    ticker.options
)

print("Number of available expirations:")
print(len(available_expirations))

print()
print("First available expirations:")
print(
    available_expirations[:10]
)


download_datetime = datetime.now(
    timezone.utc
)

download_timestamp = (
    download_datetime.isoformat()
)

download_date = pd.Timestamp(
    download_datetime
).normalize()


expiration_table = pd.DataFrame(
    {
        "expiration": available_expirations
    }
)

expiration_table[
    "expiration_datetime"
] = pd.to_datetime(
    expiration_table["expiration"],
    utc=True,
)

expiration_table[
    "approximate_days_to_expiry"
] = (
    expiration_table[
        "expiration_datetime"
    ]
    - download_date
).dt.total_seconds() / (
    24.0 * 60.0 * 60.0
)


selected_expirations = []

for target_days in TARGET_DAYS_TO_EXPIRY:
    available_choices = expiration_table[
        ~expiration_table[
            "expiration"
        ].isin(
            selected_expirations
        )
    ].copy()

    available_choices[
        "distance_from_target"
    ] = (
        available_choices[
            "approximate_days_to_expiry"
        ]
        - target_days
    ).abs()

    closest_row = available_choices.loc[
        available_choices[
            "distance_from_target"
        ].idxmin()
    ]

    selected_expirations.append(
        closest_row["expiration"]
    )


selected_expiration_table = (
    expiration_table[
        expiration_table[
            "expiration"
        ].isin(
            selected_expirations
        )
    ]
    .sort_values(
        "expiration_datetime"
    )
    .reset_index(
        drop=True
    )
)


print()
print("Target days to expiry:")
print(TARGET_DAYS_TO_EXPIRY)

print()
print("Selected expirations:")
print(
    selected_expiration_table[
        [
            "expiration",
            "approximate_days_to_expiry",
        ]
    ]
)


spot = float(
    ticker.fast_info["last_price"]
)


all_expiration_data = []


for expiration in selected_expiration_table[
    "expiration"
]:
    print()
    print(
        f"Downloading expiration: {expiration}"
    )

    option_chain = ticker.option_chain(
        expiration
    )

    calls = option_chain.calls.copy()
    puts = option_chain.puts.copy()

    calls["ticker"] = TICKER_SYMBOL
    calls["option_type"] = "call"
    calls["expiration"] = expiration
    calls["spot"] = spot
    calls["download_timestamp"] = (
        download_timestamp
    )

    puts["ticker"] = TICKER_SYMBOL
    puts["option_type"] = "put"
    puts["expiration"] = expiration
    puts["spot"] = spot
    puts["download_timestamp"] = (
        download_timestamp
    )

    combined_expiration_data = pd.concat(
        [
            calls,
            puts,
        ],
        ignore_index=True,
    )

    all_expiration_data.append(
        combined_expiration_data
    )

    print(
        f"Call rows: {len(calls)}"
    )

    print(
        f"Put rows: {len(puts)}"
    )

    print(
        "Total rows for expiration: "
        f"{len(combined_expiration_data)}"
    )


multiple_expiry_option_chain = pd.concat(
    all_expiration_data,
    ignore_index=True,
)


output_path = (
    RAW_DATA_DIRECTORY
    / "SPY_multiple_expiries_raw.csv"
)

multiple_expiry_option_chain.to_csv(
    output_path,
    index=False,
)


print()
print("Spot price:")
print(spot)

print()
print("Download timestamp:")
print(download_timestamp)

print()
print("Combined shape:")
print(
    multiple_expiry_option_chain.shape
)

print()
print("Rows by expiration and option type:")
print(
    multiple_expiry_option_chain.groupby(
        [
            "expiration",
            "option_type",
        ]
    ).size()
)

print()
print("Missing values in key columns:")
print(
    multiple_expiry_option_chain[
        [
            "strike",
            "bid",
            "ask",
            "option_type",
            "expiration",
            "spot",
            "download_timestamp",
        ]
    ].isna().sum()
)

print()
print(
    f"Saved raw multi-expiry data to: "
    f"{output_path}"
)