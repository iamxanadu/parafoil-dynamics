from dynamics import f

from scipy.integrate import odeint
import matplotlib.pyplot as pyplot
import numpy as np


def integrand(xs, t):

    u = [0.1, 0]

    return f(xs, u)


t = np.linspace(0, 60, 1001)
x0 = [1, 0, 0, 0, 0, 1000, 0, 0]

sol = odeint(integrand, x0, t)

