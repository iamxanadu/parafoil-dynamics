from sympy import Symbol, Function, Matrix, Derivative
from sympy import sin, cos
from sympy import pprint, init_printing
from sympy import simplify, lambdify
from sympy.solvers import solve

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
alpha = Symbol('alpha')

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

q = Matrix([x, y, h])
q_dest = Matrix([0, 0, 0])
q_dot = Matrix([xdot, ydot, hdot])
e = q_dest - q
edot = Derivative(e, t).doit()
r = edot + (alpha * e)
V_candidate = 0.5 * r.transpose() * r

V_candidate_dot = Derivative(V_candidate, t).doit()
x_final = solve(V_candidate_dot, x)
y_final = solve(V_candidate_dot, y)
h_final = solve(V_candidate_dot, h)
