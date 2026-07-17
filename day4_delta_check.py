import numpy as np

from src.greeks import call_delta, put_delta


strike = 100.0
rate = 0.04
volatility = 0.25
time_to_expiry = 30.0 / 365.0

spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

call_deltas = call_delta(
    spot=spots,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)

put_deltas = put_delta(
    spot=spots,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)

print("Spots:", spots)
print("Call deltas:", call_deltas)
print("Put deltas:", put_deltas)
print("Call delta - put delta:", call_deltas - put_deltas)