from math import sin, cos, exp
import numpy as np


def constant_controller():
    """Dummy controller function. Returns zero commands for both inputs to the parafoil. Ordinarily you would do some calculations with x to produce the commands u.

    Arguments:
        x {iterable} -- the state of the parafoil

    Returns:
        list -- list of zeros for the control.
    """
    return [0.1, 0.1]


def dubins_lyapunov_controller(x, a, eps, umax):
    """A lyapunov-based controler which assumes that the parafoil is roughly Dubins.

    Arguments:
        x {list} -- The state of the parafoil.
        a {float} -- Lowest turning rate allowed for Dubins dynamics.
        eps {float} -- parameter determining how aggressive the controller is.
        umax {float} -- The max turning rate allowed for the Dubins dynamics.

    Returns:
        list -- The control for th parafoil (2x1).
    """

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
    # NOTE This doesn't seem to work well - not really sure why
    # TODO Figure out why this doesn't work - should allow setting of the final radius by max turn rate
    # r = 1/(k*cos(gam))
    # sig = atan2(V**2 * cos(gam), g*r)
    return [k, 0]
