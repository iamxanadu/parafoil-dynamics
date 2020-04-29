from dynamics import simulation_dynamics
from controllers import *
from scipy.integrate import solve_ivp
from math import pi

import numpy as np
from matplotlib import pyplot as plt

from graphics import *


def ivp_wrapper(t, x):
    # Calculate control
    u = constant_controller(x)
    # Return just xdot
    return simulation_dynamics(x, u)


# Initial state
# 2 m/s speed, -pi/6 flight path angle, 100m in the air - everything else zero

# Create time points we want data at...
t = np.linspace(0, 30, 100)

# Call odeint
solution = solve_ivp(ivp_wrapper, (0, 30), [
                     7, 0, 0, 30, 20, 100, 0, 0], t_eval=t)

# Graphs solution
visualizer = Visualizer(x_traj=solution.y, plot_heading=False)
