import math
import sympy
import numpy as np
from constants import taue, taus
from constants import Cltrim, Cdtrim, delCl, delCd
from constants import rho, S, m, g


def simulation_dynamics(x, u, sympy_version=False, wx_func=None, wy_func=None):
    """Dynamics of a parafoil as a rigid body as described in Rademacher (2009)

    Args:
        x (iterable): state
        u (iterable): inputs
    """

    eps = 1e-8
    V, gamma, psi, x, y, h, sigma, epsilon = x
    comsigma, comepsilon = u

    # TODO Zero wind until I implement wind distribution
    wx = wy = 0
    if wx_func is not None and wy_func is not None:
        wx = wx_func
        wy = wy_func

    Cl = Cltrim + delCl * epsilon
    L = 1/2 * rho * V**2 * S * Cl
    Cd = Cdtrim + delCd * epsilon
    D = 1/2 * rho * V**2 * S * Cd

    W = m * g

    if sympy_version:
        dotV = -(D + W * sympy.sin(gamma)) / (m + eps)
        dotgamma = (L * sympy.cos(sigma) - W * sympy.cos(gamma)) / ((m * V) + eps)
        dotpsi = (L * sympy.sin(sigma)) / ((m * V * sympy.cos(gamma)) + eps)
        dotx = V * sympy.cos(gamma) * sympy.cos(psi) + wx
        doty = V * sympy.cos(gamma) * sympy.sin(psi) + wy
        doth = V * sympy.sin(gamma)
        dotsigma = (comsigma - sigma)*6.0/taus
        dotepsilon = (comepsilon - epsilon)*6.0/taue
    else:
        dotV = -(D + W * math.sin(gamma)) / (m + eps)
        dotgamma = (L * math.cos(sigma) - W * math.cos(gamma)) / ((m * V) + eps)
        dotpsi = (L * math.sin(sigma)) / ((m * V * math.cos(gamma)) + eps)
        dotx = V * math.cos(gamma) * math.cos(psi) + wx
        doty = V * math.cos(gamma) * math.sin(psi) + wy
        doth = V * math.sin(gamma)
        dotsigma = (comsigma - sigma)*6.0/taus
        dotepsilon = (comepsilon - epsilon)*6.0/taue
        return np.array([dotV, dotgamma, dotpsi, dotx, doty, doth, dotsigma, dotepsilon]).astype(float)

    return [dotV, dotgamma, dotpsi, dotx, doty, doth, dotsigma, dotepsilon]


def discrete_simulation_dynamics(x, u, dt, wx_func=None, wy_func=None):
    xdot = simulation_dynamics(x, u, sympy_version=False, wx_func=wx_func, wy_func=wy_func)
    return (dt * xdot) + x


def dynamics(x, u):
    """Dynamics of a parafoil as a rigid body as described in Rademacher (2009)

    Args:
        x (iterable): state
        u (iterable): inputs
    """
    V, gamma, psi, x, y, h = x
    sigma, epsilon = u

    # TODO Zero wind until I implement wind distribution
    wx = wy = 0

    Cl = Cltrim + delCl * epsilon
    L = 1/2 * rho * V**2 * S * Cl
    Cd = Cdtrim + delCd * epsilon
    D = 1/2 * rho * V**2 * S * Cd

    W = m * g

    dotV = -(D + W * math.sin(gamma)) / m
    dotgamma = (L * math.cos(sigma) - W * math.cos(gamma)) / (m * V)
    dotpsi = (L * math.sin(sigma)) / (m * V * math.cos(gamma))
    dotx = V * math.cos(gamma) * math.cos(psi) + wx
    doty = V * math.cos(gamma) * math.sin(psi) + wy
    doth = V * math.sin(gamma)

    return [dotV, dotgamma, dotpsi, dotx, doty, doth]


def discrete_dynamics(x, u, dt):
    return dt * dynamics(x, u) + x
