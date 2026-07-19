import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.greeks import (
    call_delta,
    gamma,
    vega,
    call_theta,
    call_rho,
)


spot_values = np.linspace(
    70.0,
    130.0,
    121,
)

strike = 100.0
rate = 0.04
volatility = 0.25
time_to_expiry = 30.0 / 365.0


analysis = pd.DataFrame(
    {
        "spot": spot_values,
        "call_delta": call_delta(
            spot=spot_values,
            strike=strike,
            rate=rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
        ),
        "gamma": gamma(
            spot=spot_values,
            strike=strike,
            rate=rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
        ),
        "vega_per_point": vega(
            spot=spot_values,
            strike=strike,
            rate=rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
        ) / 100.0,
        "theta_per_day": call_theta(
            spot=spot_values,
            strike=strike,
            rate=rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            per_day=True,
        ),
        "rho_per_point": call_rho(
            spot=spot_values,
            strike=strike,
            rate=rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            per_rate_point=True,
        ),
    }
)


print(analysis.head())
print()
print(analysis.tail())
print()

maximum_gamma_row = analysis.loc[
    analysis["gamma"].idxmax()
]

maximum_vega_row = analysis.loc[
    analysis["vega_per_point"].idxmax()
]

print("Maximum gamma:")
print(maximum_gamma_row)
print()

print("Maximum vega:")
print(maximum_vega_row)

plt.figure(figsize=(8, 5))
plt.plot(
    analysis["spot"],
    analysis["call_delta"],
)
plt.xlabel("Spot Price")
plt.ylabel("Call Delta")
plt.title("Call Delta vs Spot Price")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(
    analysis["spot"],
    analysis["gamma"],
)
plt.xlabel("Spot Price")
plt.ylabel("Gamma")
plt.title("Gamma vs Spot Price")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(
    analysis["spot"],
    analysis["vega_per_point"],
)
plt.xlabel("Spot Price")
plt.ylabel("Vega per 1 Volatility Point")
plt.title("Vega vs Spot Price")
plt.grid(True)
plt.tight_layout()
plt.show()

time_values = np.linspace(
    1.0 / 365.0,
    365.0 / 365.0,
    365,
)

theta_analysis = pd.DataFrame(
    {
        "days_to_expiry": time_values * 365.0,
        "theta_per_day": call_theta(
            spot=100.0,
            strike=100.0,
            rate=rate,
            volatility=volatility,
            time_to_expiry=time_values,
            per_day=True,
        ),
    }
)

plt.figure(figsize=(8, 5))
plt.plot(
    theta_analysis["days_to_expiry"],
    theta_analysis["theta_per_day"],
)
plt.xlabel("Days to Expiry")
plt.ylabel("Call Theta per Day")
plt.title("Call Theta vs Time to Expiry")
plt.grid(True)
plt.tight_layout()
plt.show()

rho_analysis = pd.DataFrame(
    {
        "days_to_expiry": time_values * 365.0,
        "rho_per_point": call_rho(
            spot=100.0,
            strike=100.0,
            rate=rate,
            volatility=volatility,
            time_to_expiry=time_values,
            per_rate_point=True,
        ),
    }
)

plt.figure(figsize=(8, 5))
plt.plot(
    rho_analysis["days_to_expiry"],
    rho_analysis["rho_per_point"],
)
plt.xlabel("Days to Expiry")
plt.ylabel("Call Rho per 1 Rate Point")
plt.title("Call Rho vs Time to Expiry")
plt.grid(True)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 8),
    sharex=True,
)

axes[0].plot(
    analysis["spot"],
    analysis["call_delta"],
)

axes[0].axvline(
    x=strike,
    linestyle="--",
    label="Strike",
)

axes[0].set_title("Call Delta vs Spot Price")
axes[0].set_ylabel("Call Delta")
axes[0].grid(True)
axes[0].legend()


axes[1].plot(
    analysis["spot"],
    analysis["gamma"],
)

axes[1].axvline(
    x=strike,
    linestyle="--",
    label="Strike",
)

axes[1].set_title("Gamma vs Spot Price")
axes[1].set_xlabel("Spot Price")
axes[1].set_ylabel("Gamma")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()

analysis["numerical_gamma"] = np.gradient(
    analysis["call_delta"],
    analysis["spot"],
)

analysis["gamma_error"] = (
    analysis["numerical_gamma"] - analysis["gamma"]
)

print()
print(
    analysis[
        [
            "spot",
            "call_delta",
            "gamma",
            "numerical_gamma",
            "gamma_error",
        ]
    ].head(10)
)

maximum_absolute_gamma_error = (
    analysis["gamma_error"].abs().max()
)

print()
print(
    "Maximum absolute gamma error:",
    maximum_absolute_gamma_error,
)

maximum_error_row = analysis.loc[
    analysis["gamma_error"].abs().idxmax()
]

print()
print("Row with maximum absolute gamma error:")
print(
    maximum_error_row[
        [
            "spot",
            "gamma",
            "numerical_gamma",
            "gamma_error",
        ]
    ]
)

mean_absolute_gamma_error = (
    analysis["gamma_error"].abs().mean()
)

print()
print(
    "Mean absolute gamma error:",
    mean_absolute_gamma_error,
)

plt.figure(figsize=(8, 5))

plt.plot(
    analysis["spot"],
    analysis["gamma"],
    label="Analytical Gamma",
)

plt.plot(
    analysis["spot"],
    analysis["numerical_gamma"],
    linestyle="--",
    label="Numerical Gamma",
)

plt.xlabel("Spot Price")
plt.ylabel("Gamma")
plt.title("Analytical vs Numerical Gamma")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

gamma_time_analysis = pd.DataFrame(
    {
        "days_to_expiry": time_values * 365.0,
        "gamma": gamma(
            spot=100.0,
            strike=strike,
            rate=rate,
            volatility=volatility,
            time_to_expiry=time_values,
        ),
    }
)

plt.figure(figsize=(8, 5))

plt.plot(
    gamma_time_analysis["days_to_expiry"],
    gamma_time_analysis["gamma"],
)

plt.xlabel("Days to Expiry")
plt.ylabel("Gamma")
plt.title("ATM Gamma vs Time to Expiry")
plt.grid(True)
plt.tight_layout()
plt.show()

vega_time_analysis = pd.DataFrame(
    {
        "days_to_expiry": time_values * 365.0,
        "vega_per_point": vega(
            spot=100.0,
            strike=strike,
            rate=rate,
            volatility=volatility,
            time_to_expiry=time_values,
        ) / 100.0,
    }
)

plt.figure(figsize=(8, 5))

plt.plot(
    vega_time_analysis["days_to_expiry"],
    vega_time_analysis["vega_per_point"],
)

plt.xlabel("Days to Expiry")
plt.ylabel("Vega per 1 Volatility Point")
plt.title("ATM Vega vs Time to Expiry")
plt.grid(True)
plt.tight_layout()
plt.show()

gamma_time_analysis["normalized_gamma"] = (
    gamma_time_analysis["gamma"]
    / gamma_time_analysis["gamma"].max()
)

vega_time_analysis["normalized_vega"] = (
    vega_time_analysis["vega_per_point"]
    / vega_time_analysis["vega_per_point"].max()
)

plt.figure(figsize=(8, 5))

plt.plot(
    gamma_time_analysis["days_to_expiry"],
    gamma_time_analysis["normalized_gamma"],
    label="Normalized Gamma",
)

plt.plot(
    vega_time_analysis["days_to_expiry"],
    vega_time_analysis["normalized_vega"],
    label="Normalized Vega",
)

plt.xlabel("Days to Expiry")
plt.ylabel("Normalized Sensitivity")
plt.title("ATM Gamma vs Vega Across Time to Expiry")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

expiry_days = [
    7,
    30,
    90,
    365,
]

plt.figure(figsize=(8, 5))

for days in expiry_days:
    expiry = days / 365.0

    gamma_values = gamma(
        spot=spot_values,
        strike=strike,
        rate=rate,
        volatility=volatility,
        time_to_expiry=expiry,
    )

    plt.plot(
        spot_values,
        gamma_values,
        label=f"{days} days",
    )

plt.axvline(
    x=strike,
    linestyle="--",
    label="Strike",
)

plt.xlabel("Spot Price")
plt.ylabel("Gamma")
plt.title("Gamma vs Spot Price for Different Expiries")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()