from src.implied_volatility import implied_volatility_bisection


spot = 100.0
strike = 100.0
rate = 0.04
time_to_expiry = 30.0 / 365.0


examples = [
    {
        "name": "Negative call price",
        "market_price": -1.0,
        "option_type": "call",
    },
    {
        "name": "Call price above spot",
        "market_price": 120.0,
        "option_type": "call",
    },
    {
        "name": "Negative put price",
        "market_price": -0.5,
        "option_type": "put",
    },
    {
        "name": "Invalid option type",
        "market_price": 3.0,
        "option_type": "straddle",
    },
]


for example in examples:
    print()
    print(example["name"])

    try:
        result = implied_volatility_bisection(
            market_price=example["market_price"],
            spot=spot,
            strike=strike,
            rate=rate,
            time_to_expiry=time_to_expiry,
            option_type=example["option_type"],
        )

        print("Recovered IV:", result)

    except ValueError as error:
        print("Rejected:", error)