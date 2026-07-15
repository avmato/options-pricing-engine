import numpy as np

from src.market_utils import (
    log_moneyness,
    moneyness,
    years_to_expiry,
)


spot = 100.0
strikes = np.array([80, 90, 100, 110, 120], dtype=float)

money = moneyness(spot, strikes)
log_money = log_moneyness(spot, strikes)

days = np.array([7, 30, 90, 180, 365], dtype=float)
times = years_to_expiry(days)

print("Spot:", spot)
print("Strikes:", strikes)
print("Moneyness:", money)
print("Log-moneyness:", log_money)
print("Days to expiry:", days)
print("Years to expiry:", times)