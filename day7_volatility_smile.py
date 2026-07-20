import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.black_scholes import black_scholes_call
from src.implied_volatility import implied_volatility_bisection

spot = 100.0
rate = 0.04
time_to_expiry = 30.0 / 365.0

atm_volatility = 0.20
smile_curvature = 1.50


strike_values = np.linspace(
    70.0,
    130.0,
    25,
)

log_moneyness = np.log(
    strike_values / spot
)

true_volatility = (
    atm_volatility
    + smile_curvature * log_moneyness**2
)

smile_analysis = pd.DataFrame(
    {
        "strike": strike_values,
        "log_moneyness": log_moneyness,
        "true_volatility": true_volatility,
    }
)

print(smile_analysis)

smile_analysis["market_call_price"] = black_scholes_call(
    spot=spot,
    strike=smile_analysis["strike"].to_numpy(),
    rate=rate,
    volatility=smile_analysis["true_volatility"].to_numpy(),
    time_to_expiry=time_to_expiry,
)

print()
print(smile_analysis)

implied_volatilities = []

for _, row in smile_analysis.iterrows():
    implied_volatility = implied_volatility_bisection(
        market_price=row["market_call_price"],
        spot=spot,
        strike=row["strike"],
        rate=rate,
        time_to_expiry=time_to_expiry,
        option_type="call",
    )

    implied_volatilities.append(implied_volatility)

smile_analysis["implied_volatility"] = implied_volatilities

print()
print(
    smile_analysis[
        [
            "strike",
            "true_volatility",
            "market_call_price",
            "implied_volatility",
        ]
    ]
)

smile_analysis["volatility_error"] = (
    smile_analysis["implied_volatility"]
    - smile_analysis["true_volatility"]
)

maximum_absolute_error = (
    smile_analysis["volatility_error"].abs().max()
)

mean_absolute_error = (
    smile_analysis["volatility_error"].abs().mean()
)

print()
print(
    smile_analysis[
        [
            "strike",
            "true_volatility",
            "implied_volatility",
            "volatility_error",
        ]
    ]
)

print()
print(
    "Maximum absolute volatility error:",
    maximum_absolute_error,
)

print(
    "Mean absolute volatility error:",
    mean_absolute_error,
)

plt.figure(figsize=(8, 5))

plt.plot(
    smile_analysis["strike"],
    smile_analysis["true_volatility"],
    label="True Volatility",
)

plt.scatter(
    smile_analysis["strike"],
    smile_analysis["implied_volatility"],
    label="Recovered Implied Volatility",
)

plt.axvline(
    x=spot,
    linestyle="--",
    label="Spot",
)

plt.xlabel("Strike Price")
plt.ylabel("Volatility")
plt.title("Synthetic Volatility Smile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    smile_analysis["strike"],
    smile_analysis["market_call_price"],
)

plt.xlabel("Strike Price")
plt.ylabel("Call Price")
plt.title("Synthetic Call Prices Across Strikes")
plt.grid(True)
plt.tight_layout()
plt.show()