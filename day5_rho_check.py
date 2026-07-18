from src.greeks import call_rho, put_rho


spot = 100.0
strike = 100.0
rate = 0.04
volatility = 0.25
time_to_expiry = 30.0 / 365.0

call_rho_raw = call_rho(
    spot,
    strike,
    rate,
    volatility,
    time_to_expiry,
)

call_rho_per_point = call_rho(
    spot,
    strike,
    rate,
    volatility,
    time_to_expiry,
    per_rate_point=True,
)

put_rho_raw = put_rho(
    spot,
    strike,
    rate,
    volatility,
    time_to_expiry,
)

put_rho_per_point = put_rho(
    spot,
    strike,
    rate,
    volatility,
    time_to_expiry,
    per_rate_point=True,
)

print("Call rho raw:", call_rho_raw)
print("Call rho per 1 rate point:", call_rho_per_point)

print("Put rho raw:", put_rho_raw)
print("Put rho per 1 rate point:", put_rho_per_point)