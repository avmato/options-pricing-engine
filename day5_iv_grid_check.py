import numpy as np

from src.black_scholes import black_scholes_call
from src.implied_volatility import implied_volatility_bisection


spot = 100.0
rate = 0.04
time_to_expiry = 30.0 / 365.0

strikes = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
true_volatilities = np.array([0.15, 0.25, 0.40])

for true_volatility in true_volatilities:
    print()
    print("True volatility:", true_volatility)

    for strike in strikes:
        market_price = black_scholes_call(
            spot=spot,
            strike=strike,
            rate=rate,
            volatility=true_volatility,
            time_to_expiry=time_to_expiry,
        )

        try:
            recovered_iv = implied_volatility_bisection(
                market_price=float(market_price),
                spot=spot,
                strike=float(strike),
                rate=rate,
                time_to_expiry=time_to_expiry,
                option_type="call",
            )

            error = recovered_iv - true_volatility

            print(
                f"Strike={strike:6.1f} | "
                f"Price={market_price:10.6f} | "
                f"Recovered IV={recovered_iv:.8f} | "
                f"Error={error:.2e}"
            )

        except ValueError as error:
            print(
                f"Strike={strike:6.1f} | "
                f"Solver failed: {error}"
            )