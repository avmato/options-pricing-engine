from src.greeks import call_theta, put_theta


spot = 100.0
strike = 100.0
rate = 0.04
volatility = 0.25
time_to_expiry = 30.0 / 365.0

call_theta_annual = call_theta(
    spot,
    strike,
    rate,
    volatility,
    time_to_expiry,
)

call_theta_daily = call_theta(
    spot,
    strike,
    rate,
    volatility,
    time_to_expiry,
    per_day=True,
)

put_theta_annual = put_theta(
    spot,
    strike,
    rate,
    volatility,
    time_to_expiry,
)

put_theta_daily = put_theta(
    spot,
    strike,
    rate,
    volatility,
    time_to_expiry,
    per_day=True,
)

print("Call theta annual:", call_theta_annual)
print("Call theta daily:", call_theta_daily)
print("Put theta annual:", put_theta_annual)
print("Put theta daily:", put_theta_daily)

print(
    "Call annual / 365:",
    call_theta_annual / 365.0,
)

print(
    "Put annual / 365:",
    put_theta_annual / 365.0,
)