from math import sin, cos
from constants import taue, taus
from constants import Cltrim, Cdtrim, delCl, delCd
from constants import rho, S, m, g


def simulation_dynamics(x, u):
    """Dynamics of a parafoil as a rigid body as described in Rademacher (2009)

    Args:
        x (iterable): state
        u (iterable): inputs
    """
    V, gamma, psi, x, y, h, sigma, epsilon = x
    comsigma, comepsilon = u

    # TODO Zero wind until I implement wind distribution
    wx = wy = 0

    Cl = Cltrim + delCl * epsilon
    L = 1/2 * rho * V**2 * S * Cl
    Cd = Cdtrim + delCd * epsilon
    D = 1/2 * rho * V**2 * S * Cd

    W = m * g

    dotV = -(D + W * sin(gamma)) / m
    dotgamma = (L * cos(sigma) - W * cos(gamma)) / (m * V)
    dotpsi = (L * sin(sigma)) / (m * V * cos(gamma))
    dotx = V * cos(gamma) * cos(psi) + wx
    doty = V * cos(gamma) * sin(psi) + wy
    doth = V * sin(gamma)
    dotsigma = (comsigma - sigma)/taus
    dotepsilon = (comepsilon - epsilon)/taue

    return [dotV, dotgamma, dotpsi, dotx, doty, doth, dotsigma, dotepsilon]


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

    dotV = -(D + W * sin(gamma)) / m
    dotgamma = (L * cos(sigma) - W * cos(gamma)) / (m * V)
    dotpsi = (L * sin(sigma)) / (m * V * cos(gamma))
    dotx = V * cos(gamma) * cos(psi) + wx
    doty = V * cos(gamma) * sin(psi) + wy
    doth = V * sin(gamma)

    return [dotV, dotgamma, dotpsi, dotx, doty, doth]


def discrete_dynamics(x, u, dt):
    return dt * dynamics(x, u) + x
