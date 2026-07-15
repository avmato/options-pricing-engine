import numpy as np

from src.arbitrage import discounted_strike
from src.black_scholes import (
    black_scholes_call,
    black_scholes_put,
)


spot = 100.0
strikes = np.array([80, 90, 100, 110, 120], dtype=float)
rate = 0.04
volatility = 0.25
time_to_expiry = 30.0 / 365.0

call_prices = black_scholes_call(
    spot,
    strikes,
    rate,
    volatility,
    time_to_expiry,
)

put_prices = black_scholes_put(
    spot,
    strikes,
    rate,
    volatility,
    time_to_expiry,
)

pv_strikes = discounted_strike(
    strikes,
    rate,
    time_to_expiry,
)

parity_left = call_prices - put_prices
parity_right = spot - pv_strikes
parity_residuals = parity_left - parity_right

print("Strikes:", strikes)
print("Call prices:", call_prices)
print("Put prices:", put_prices)
print("C - P:", parity_left)
print("S - PV(K):", parity_right)
print("Parity residuals:", parity_residuals)
print(
    "Parity holds:",
    np.allclose(parity_left, parity_right),
)