from gpaw.spline import Spline
import numpy as np
a = np.array([1, 0.9, 0.1, 0.0])
s = Spline(0, 2.0, a)
dx = 0.0001
for x in [0.5, 1, 1.2, 3]:
    y, dydx = s.get_value_and_derivative(x)
    z = (s(x + dx) - s(x - dx)) / (2 * dx)
    print y, dydx - z
    assert abs(dydx - z) < 1e-7
