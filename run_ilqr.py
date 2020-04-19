from sympy import Symbol, Function, Matrix
from sympy import sin, cos
from sympy import pprint, init_printing
from sympy import simplify, lambdify

import constants

t = Symbol('t')

m = Symbol('m')
g = Symbol('g')
rho = Symbol('rho')  # Air density
S = Symbol('S')  # Reference area (m^2)
Cltrim = Symbol('Clt')
Cdtrim = Symbol('Cdt')
dCl = Symbol('dCl')
dCd = Symbol('dCd')

self.const_dict = {g: constants.g, m: constants.m, rho: constants.rho, Cltrim: constants.Cltrim,
            Cdtrim: constants.Cdtrim, dCl: constants.delCl, dCd: constants.delCd, S: constants.S}

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
self.xs = Matrix([v, gamma, psi, x, y, h])
self.u = Matrix([sigma, epsilon])
self.f = simplify(
    Matrix([vdot, gammadot, psidot, xdot, ydot, hdot]).subs(self.dc))