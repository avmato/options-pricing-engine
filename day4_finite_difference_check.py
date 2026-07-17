from src.black_scholes import black_scholes_call
from src.greeks import call_delta, gamma, vega


spot = 100.0
strike = 100.0
rate = 0.04
volatility = 0.25
time_to_expiry = 30.0 / 365.0

spot_step = 0.01
volatility_step = 0.0001


price = black_scholes_call(
    spot=spot,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)

price_spot_up = black_scholes_call(
    spot=spot + spot_step,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)

price_spot_down = black_scholes_call(
    spot=spot - spot_step,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)

numerical_delta = (
    price_spot_up - price_spot_down
) / (2.0 * spot_step)

numerical_gamma = (
    price_spot_up
    - 2.0 * price
    + price_spot_down
) / spot_step**2


price_volatility_up = black_scholes_call(
    spot=spot,
    strike=strike,
    rate=rate,
    volatility=volatility + volatility_step,
    time_to_expiry=time_to_expiry,
)

price_volatility_down = black_scholes_call(
    spot=spot,
    strike=strike,
    rate=rate,
    volatility=volatility - volatility_step,
    time_to_expiry=time_to_expiry,
)

numerical_vega = (
    price_volatility_up - price_volatility_down
) / (2.0 * volatility_step)


analytical_delta = call_delta(
    spot=spot,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)

analytical_gamma = gamma(
    spot=spot,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)

analytical_vega = vega(
    spot=spot,
    strike=strike,
    rate=rate,
    volatility=volatility,
    time_to_expiry=time_to_expiry,
)


print("Delta")
print("Analytical:", analytical_delta)
print("Numerical:", numerical_delta)
print("Difference:", analytical_delta - numerical_delta)

print()

print("Gamma")
print("Analytical:", analytical_gamma)
print("Numerical:", numerical_gamma)
print("Difference:", analytical_gamma - numerical_gamma)

print()

print("Vega")
print("Analytical:", analytical_vega)
print("Numerical:", numerical_vega)
print("Difference:", analytical_vega - numerical_vega)