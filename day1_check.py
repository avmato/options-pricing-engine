import numpy as np

from src.market_utils import call_intrinsic_value, put_intrinsic_value


spot = 100.0
strikes = np.array([80, 90, 100, 110, 120], dtype=float)

call_values = call_intrinsic_value(spot, strikes)
put_values = put_intrinsic_value(spot, strikes)

print("Spot:", spot)
print("Strikes:", strikes)
print("Call intrinsic values:", call_values)
print("Put intrinsic values:", put_values)