from dynamics import simulation_dynamics
from controllers import *
from scipy.integrate import odeint
from math import pi

import numpy as np
from matplotlib import pyplot as plt

from graphics import *

def odeint_wrapper(x, t):
    # Calculate control
    u = minimum_time_controller(x, 10.0, -10.0)
    # Return just xdot
    return simulation_dynamics(x, u)


# Initial state
# 2 m/s speed, -pi/6 flight path angle, 100m in the air - everything else zero
x0 = [2, -pi/6, 0, 0, 0, 100, 0, 0]

# Create time points we want data at...
t = np.linspace(0, 100, 1000)

# Call odeint
solution = odeint(odeint_wrapper, x0, t)

# Graphs solution
solution = np.transpose(solution)
visualizer = Visualizer(x_traj=np.transpose(solution))