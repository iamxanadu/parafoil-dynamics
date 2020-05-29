from dynamics import simulation_dynamics
from controllers import dubins_lyapunov_controller
from scipy.integrate import solve_ivp
import numpy as np
from graphics import Visualizer

# NOTE sigma=0.5 seems to give a turn radius of about 10m while not effecting gamma much
# NOTE sigma=0.3 gives about a 15m radius


def integrand(t, x):
    umax = 0.3
    a = 0.1
    eps = 2
    u = dubins_lyapunov_controller(x, a, eps, umax)

    # Return just xdot
    return simulation_dynamics(t, x, u)


tf = 200
t = np.linspace(0, tf, tf)
x0 = [7, -0.15, 0.01, -1000, 1000, 9000, 0, 0]
# Call odeint
solution = solve_ivp(integrand, (0, tf), x0, t_eval=t)

# Graphs solution
visualizer = Visualizer(x_traj=solution.y, plot_heading=False)
