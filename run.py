from dynamics import f
from controllers import constant_controller
from scipy.integrate import odeint
from math import pi

import numpy as np
from matplotlib import pyplot as plt


def odeint_wrapper(x, t):
    # Calculate control
    u = constant_controller(x)
    # Return just xdot
    return f(x, u)


# Initial state
# 2 m/s speed, -pi/6 flight path angle, 100m in the air - everything else zero
x0 = [2, -pi/6, 0, 0, 0, 100, 0, 0]

# Create time points we want data at...
t = np.linspace(0, 100, 1000)

# Call odeint
solution = odeint(odeint_wrapper, x0, t)

print(solution)

plt.plot(t, solution[:,5])
plt.show()
