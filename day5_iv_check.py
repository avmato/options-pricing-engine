from src.black_scholes import (
    black_scholes_call,
    black_scholes_put,
)
from src.implied_volatility import (
    implied_volatility_bisection,
)


spot = 100.0
strike = 100.0
rate = 0.04
true_volatility = 0.25
time_to_expiry = 30.0 / 365.0

call_market_price = black_scholes_call(
    spot,
    strike,
    rate,
    true_volatility,
    time_to_expiry,
)

put_market_price = black_scholes_put(
    spot,
    strike,
    rate,
    true_volatility,
    time_to_expiry,
)

call_implied_volatility = implied_volatility_bisection(
    market_price=float(call_market_price),
    spot=spot,
    strike=strike,
    rate=rate,
    time_to_expiry=time_to_expiry,
    option_type="call",
)

put_implied_volatility = implied_volatility_bisection(
    market_price=float(put_market_price),
    spot=spot,
    strike=strike,
    rate=rate,
    time_to_expiry=time_to_expiry,
    option_type="put",
)

print("True volatility:", true_volatility)
print("Call market price:", call_market_price)
print("Recovered call IV:", call_implied_volatility)
print("Put market price:", put_market_price)
print("Recovered put IV:", put_implied_volatility)