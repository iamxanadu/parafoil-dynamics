import math
from constants import taue, taus
from constants import Cltrim, Cdtrim, delCl, delCd
from constants import rho, S, m, g

from sympy import Symbol, Function, Matrix
from sympy import sin, cos


def simulation_dynamics(t, x, u, wx_func=None, wy_func=None):
    """The parafoil dynamics given by Rademacher (2009). Also allows the input of a wind model.

    Arguments:
        t {float} -- Current integration time.
        x {list} -- The 8 members of the state.
        u {list} -- The 2 members of the control.

    Keyword Arguments:
        wx_func {[type]} -- [description] (default: {None})
        wy_func {[type]} -- [description] (default: {None})

    Raises:
        NotImplementedError: [description]

    Returns:
        list -- The state derivative.
    """

    V, gamma, psi, x, y, h, sigma, epsilon = x
    comsigma, comepsilon = u

    # TODO Zero wind until I implement wind distribution
    wx = wy = 0
    if wx_func is not None and wy_func is not None:
        raise NotImplementedError

    Cl = Cltrim + delCl * epsilon
    L = 1/2 * rho * V**2 * S * Cl
    Cd = Cdtrim + delCd * epsilon
    D = 1/2 * rho * V**2 * S * Cd

    W = m * g

    dotV = -(D + W * math.sin(gamma)) / (m)
    dotgamma = (L * math.cos(sigma) - W *
                math.cos(gamma)) / ((m * V))
    dotpsi = (L * math.sin(sigma)) / (m * V * math.cos(gamma))
    dotx = V * math.cos(gamma) * math.cos(psi) + wx
    doty = V * math.cos(gamma) * math.sin(psi) + wy
    doth = V * math.sin(gamma)
    dotsigma = (comsigma - sigma)/taus
    dotepsilon = (comepsilon - epsilon)/taue

    return [dotV, dotgamma, dotpsi, dotx, doty, doth, dotsigma, dotepsilon]


def planning_dynamics(x, u):
    """Dynamics of a parafoil as a rigid body as described in Rademacher (2009). Does not include lag on the controls.

    Args:
        x (iterable): state
        u (iterable): inputs
    """
    V, gamma, psi, x, y, h = x
    sigma, epsilon = u

    Cl = Cltrim + delCl * epsilon
    L = 1/2 * rho * V**2 * S * Cl
    Cd = Cdtrim + delCd * epsilon
    D = 1/2 * rho * V**2 * S * Cd

    W = m * g

    dotV = -(D + W * math.sin(gamma)) / m
    dotgamma = (L * math.cos(sigma) - W * math.cos(gamma)) / (m * V)
    dotpsi = (L * math.sin(sigma)) / (m * V * math.cos(gamma))
    dotx = V * math.cos(gamma) * math.cos(psi)
    doty = V * math.cos(gamma) * math.sin(psi)
    doth = V * math.sin(gamma)

    return [dotV, dotgamma, dotpsi, dotx, doty, doth]


def get_symbolic_dynamics():
    """Return a the symbolic dynamics for the parafoil (without the lag on the controls) in sympy form.

    Returns:
        tuple -- (x, u, dotx) where x is the state vector (6x1), u is the control vector (2x1), and dotx is the state derivative (6x1)
    """
    t = Symbol('t')
    m = Symbol('m')
    g = Symbol('g')
    rho = Symbol('rho')  # Air density
    S = Symbol('S')  # Reference area (m^2)
    Cltrim = Symbol('Clt')
    Cdtrim = Symbol('Cdt')
    dCl = Symbol('dCl')
    dCd = Symbol('dCd')

    v = Function('v')(t)
    psi = Function('psi')(t)
    gamma = Function('gamma')(t)
    x = Function('x')(t)
    y = Function('y')(t)
    h = Function('h')(t)

    sigma = Function('sigma')(t)
    epsilon = Function('epsilon')(t)

    W = m * g
    L = 1/2 * rho * S * v**2 * (Cltrim + dCl * epsilon)
    D = 1/2 * rho * S * v**2 * (Cdtrim + dCd * epsilon)

    vdot = -(D + W*sin(gamma)) / m
    gammadot = (L * cos(sigma) - W * cos(gamma)) / (m*v)
    psidot = L*sin(sigma) / (m*v*cos(gamma))
    xdot = v*cos(gamma)*cos(psi)
    ydot = v*cos(gamma)*sin(psi)
    hdot = v*sin(gamma)

    # Calculate the derivatives
    s = Matrix([v, gamma, psi, x, y, h])
    u = Matrix([sigma, epsilon])
    f = Matrix([vdot, gammadot, psidot, xdot, ydot, hdot])

    return (s, u, f)
