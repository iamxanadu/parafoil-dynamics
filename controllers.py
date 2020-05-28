from math import sin, cos, e, atan2, exp
from constants import g
import numpy as np


def constant_controller(x):
    """Dummy controller function. Returns zero commands for both inputs to the parafoil. Ordinarily you would do some calculations with x to produce the commands u.

    Arguments:
        x {iterable} -- the state of the parafoil

    Returns:
        list -- list of zeros for the control.
    """
    return [0.1, 0.1]


def dubins_lyapunov_controller(x, a, eps, umax):
    '''
    V = x[0]
    gam = x[1]
    '''
    psi = x[2]
    px = x[3]
    py = x[4]

    assert eps > 0, 'Epsilon must be a positive real number'
    assert ((-umax <= a) & (a < umax)
            ), 'a must satisfy the following: -umax <= a < umax'

    p = np.array([[px, py]]).T
    R = np.array([[cos(psi), sin(psi)], [-sin(psi), cos(psi)]])
    ptild = R@p
    xbar = ptild[0, 0]

    if xbar <= -eps:
        k = a
    elif xbar >= 0:
        k = umax
    else:
        n = (umax - a)
        expnt = 1/(xbar+eps) + 1/xbar
        d = 1 + exp(expnt)
        k = n/d + a

    # Convert dubins command to pseduo bank angle
    # r = 1/(k*cos(gam))
    # sig = atan2(V**2 * cos(gam), g*r)
    return [k, 0]
