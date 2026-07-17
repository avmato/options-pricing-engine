import numpy as np

from src.greeks import gamma


strike = 100.0
rate = 0.04
volatility = 0.25
time_to_expiry = 30.0 / 365.0

spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

gammas = gamma(
    spot=spots,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)

print("Spots:", spots)
print("Gammas:", gammas)

maximum_gamma_index = np.argmax(gammas)

print("Maximum gamma spot:", spots[maximum_gamma_index])
print("Maximum gamma:", gammas[maximum_gamma_index])