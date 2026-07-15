from src.arbitrage import (
    discounted_strike,
    put_call_parity_residual,
)


spot = 100.0
strike = 105.0
rate = 0.04
time_to_expiry = 30.0 / 365.0

call_price = 2.50

pv_strike = discounted_strike(
    strike,
    rate,
    time_to_expiry,
)

put_price = call_price - spot + pv_strike

residual = put_call_parity_residual(
    call_price,
    put_price,
    spot,
    strike,
    rate,
    time_to_expiry,
)

print("Call price:", call_price)
print("Put price implied by parity:", put_price)
print("Parity residual:", residual)