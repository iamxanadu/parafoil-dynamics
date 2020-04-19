from dynamics import simulation_dynamics
from controllers import constant_controller
from scipy.integrate import odeint
from math import pi

import numpy as np
from matplotlib import pyplot as plt

from graphics import *

def odeint_wrapper(x, t):
    # Calculate control
    u = constant_controller(x)
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
solution = np.insert(solution, 0, np.zeros((2, solution.shape[-1])), axis=0) # TODO: remove when V is handled as vector and not scalar
visualizer = Visualizer(x_traj=np.transpose(solution))