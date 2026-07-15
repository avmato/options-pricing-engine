from src.arbitrage import (
    call_price_bounds,
    discounted_strike,
    put_price_bounds,
)


spot = 100.0
strike = 105.0
rate = 0.04
time_to_expiry = 30.0 / 365.0

pv_strike = discounted_strike(
    strike,
    rate,
    time_to_expiry,
)

call_lower, call_upper = call_price_bounds(
    spot,
    strike,
    rate,
    time_to_expiry,
)

put_lower, put_upper = put_price_bounds(
    spot,
    strike,
    rate,
    time_to_expiry,
)

print("Present value of strike:", pv_strike)
print("Call bounds:", call_lower, call_upper)
print("Put bounds:", put_lower, put_upper)