from src.black_scholes import calculate_d1_d2


spot = 100.0
strike = 105.0
rate = 0.04
volatility = 0.25
time_to_expiry = 30.0 / 365.0

d1, d2 = calculate_d1_d2(
    spot,
    strike,
    rate,
    volatility,
    time_to_expiry,
)

print("d1:", d1)
print("d2:", d2)
print("d1 - d2:", d1 - d2)
print(
    "sigma * sqrt(T):",
    volatility * time_to_expiry**0.5,
)