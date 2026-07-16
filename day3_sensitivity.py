import numpy as np

from src.black_scholes import (
    black_scholes_call,
    black_scholes_put,
)


strike = 100.0
rate = 0.04
time_to_expiry = 30.0 / 365.0

spots = np.array(
    [80.0, 90.0, 100.0, 110.0, 120.0]
)

volatility = 0.25

call_prices_by_spot = black_scholes_call(
    spot=spots,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)

put_prices_by_spot = black_scholes_put(
    spot=spots,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)

print("Spot sensitivity")
print("Spots:", spots)
print("Call prices:", call_prices_by_spot)
print("Put prices:", put_prices_by_spot)

volatilities = np.array(
    [0.10, 0.20, 0.30, 0.40, 0.50]
)

spot = 100.0

call_prices_by_volatility = black_scholes_call(
    spot=spot,
    strike=strike,
    rate=rate,
    volatility=volatilities,
    time_to_expiry=time_to_expiry,
)

put_prices_by_volatility = black_scholes_put(
    spot=spot,
    strike=strike,
    rate=rate,
    volatility=volatilities,
    time_to_expiry=time_to_expiry,
)

print()
print("Volatility sensitivity")
print("Volatilities:", volatilities)
print("Call prices:", call_prices_by_volatility)
print("Put prices:", put_prices_by_volatility)