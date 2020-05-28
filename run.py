from dynamics import simulation_dynamics
from controllers import dubins_lyapunov_controller
from scipy.integrate import solve_ivp
import numpy as np
from graphics import Visualizer
from math import cos, tan

# NOTE sigma=0.5 seems to give a turn radius of about 10m while not effecting gamma much
# NOTE sigma=0.3 gives about a 15m radius

def integrand(t, x):
    #gam = x[1]
    #print(gam)
    #rmin = 10
    #umax = 1/(rmin*cos(gam))
    #umax = 0.1
    # print(-1/(tan(gam)))
    # Calculate control
    umax = 0.3
    a = 0.1
    eps = 2
    u = dubins_lyapunov_controller(x, a, eps, umax)
    
    # Return just xdot
    return simulation_dynamics(x, u)


# Initial state
# 2 m/s speed, -pi/6 flight path angle, 100m in the air - everything else zero

# Create time points we want data at...
tf = 200
t = np.linspace(0, tf, tf)

# Call odeint
solution = solve_ivp(integrand, (0, tf), [
                     7, -0.15, 0.01, -1000, 1000, 9000, 0, 0], t_eval=t)

# Graphs solution
visualizer = Visualizer(x_traj=solution.y, plot_heading=False)
