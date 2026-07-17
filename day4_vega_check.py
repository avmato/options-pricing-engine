import numpy as np

from src.greeks import vega


strike = 100.0
rate = 0.04
volatility = 0.25
time_to_expiry = 30.0 / 365.0

spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

raw_vegas = vega(
    spot=spots,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)

vegas_per_vol_point = raw_vegas / 100.0

print("Spots:", spots)
print("Raw vegas:", raw_vegas)
print("Vegas per 1 volatility point:", vegas_per_vol_point)

maximum_vega_index = np.argmax(raw_vegas)

print("Maximum vega spot:", spots[maximum_vega_index])
print("Maximum raw vega:", raw_vegas[maximum_vega_index])