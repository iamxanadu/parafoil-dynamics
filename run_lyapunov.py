from sympy import Symbol, Function, Matrix, Derivative
from sympy import sin, cos
from sympy import pprint, init_printing
from sympy import simplify, lambdify, solveset, S, re
from sympy.solvers import solve
from dynamics import *
from graphics import *

import constants
import numpy as np
import random

t = Symbol('t')
v = Function('v')(t)
psi = Function('psi')(t)
gamma = Function('gamma')(t)
x = Function('x')(t)
y = Function('y')(t)
h = Function('h')(t)
sigma = Function('sigma')(t)
epsilon = Function('epsilon')(t)
comsigma = Function('comsigma')(t)
comepsilon = Function('comepsilon')(t)

x_in = [v, gamma, psi, x, y, h, sigma, epsilon]
k_x_in = [comsigma, comepsilon]
dotv, dotgamma, dotpsi, dotx, doty, doth, dotsigma, dotepsilon = simulation_dynamics(x_in, k_x_in, sympy_version=True)

# W = m * g
# L = 1/2 * rho * S * v**2 * (Cltrim + dCl * epsilon)
# D = 1/2 * rho * S * v**2 * (Cdtrim + dCd * epsilon)

# vdot = -(D + W*sin(gamma)) / m
# gammadot = (L * cos(sigma) - W * cos(gamma)) / (m*v)
# psidot = L*sin(sigma) / (m*v*cos(gamma))
# xdot = v*cos(gamma)*cos(psi)
# ydot = v*cos(gamma)*sin(psi)
# hdot = v*sin(gamma)

q = Matrix(x_in)
q_dest = Matrix([0, 0, 0, 0, 0, 0, 0, 0])
e = q_dest - q
edot = Derivative(e, t).doit()
alpha = 0.15
r = edot + (alpha * e)
rdot = Derivative(r, t).doit()

V_candidate = 0.5 * r.dot(r.transpose())
V_candidate_dot = Derivative(V_candidate, t).doit()
V_candidate_dot = V_candidate_dot.subs(Derivative(x, t), dotx)
V_candidate_dot = V_candidate_dot.subs(Derivative(y, t), doty)
V_candidate_dot = V_candidate_dot.subs(Derivative(h, t), doth)
V_candidate_dot = V_candidate_dot.subs(Derivative(v, t), dotv)
V_candidate_dot = V_candidate_dot.subs(Derivative(gamma, t), dotgamma)
V_candidate_dot = V_candidate_dot.subs(Derivative(sigma, t), dotsigma)
V_candidate_dot = V_candidate_dot.subs(Derivative(epsilon, t), dotepsilon)
V_candidate_dot = V_candidate_dot.doit().doit()
V_candidate_dot = V_candidate_dot.subs(Derivative(v, t), dotv)
V_candidate_dot = V_candidate_dot.subs(Derivative(gamma, t), dotgamma)
V_candidate_dot = V_candidate_dot.subs(Derivative(psi, t), dotpsi)
V_candidate_dot = V_candidate_dot.subs(Derivative(epsilon, t), dotepsilon)
V_candidate_dot = V_candidate_dot.subs(Derivative(sigma, t), dotsigma)
V_candidate_dot = V_candidate_dot.doit().doit()
V_candidate_dot = V_candidate_dot.subs(Derivative(v, t), dotv)
V_candidate_dot = V_candidate_dot.subs(Derivative(gamma, t), dotgamma)
V_candidate_dot = V_candidate_dot.subs(Derivative(epsilon, t), dotepsilon)
V_candidate_dot = V_candidate_dot.subs(Derivative(sigma, t), dotsigma)



# x_in = [v, gamma, psi, x, y, h, sigma, epsilon]
x_current = np.array([0.5, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0])
u_last = np.array([0.0, 0.0])
u_current = np.array([0.0, 0.0])
t_current = 0.0
t_delta = 0.01
n = 30

test_x_traj = np.zeros((8, n+1))
test_x_traj[:, 0] = np.array(x_current)
for i in range(n):
    print(" * Iteration "+str(i)+" of "+str(n))

    V_candidate_dot_test = V_candidate_dot.subs({x: x_current[3], y: x_current[4], h: x_current[5], epsilon: x_current[7], gamma: x_current[1], sigma: x_current[6], v: x_current[0], psi: x_current[2]})
    if i == 0:
        V_candidate_dot_test = V_candidate_dot_test.subs({Derivative(comepsilon, t): 0.0, Derivative(comsigma, t): 0.0}).doit()

        # picks random comsigma
        comsigma_fin = (random.random()-0.5) * 2.0
        V_candidate_dot_test = V_candidate_dot_test.subs({comsigma: comsigma_fin})
        
        # picks random comepsilon within range
        comepsilon_range = solve(V_candidate_dot_test, comepsilon)
        if len(comepsilon_range) > 0:
            comepsilon_fin = comepsilon_range[0] + (random.random() * (comepsilon_range[-1] - comepsilon_range[0]))
        elif len(comepsilon_range) == 0:
            comepsilon_fin = 1e-6
        else:
            comepsilon_fin = comepsilon_range

        u_current[0] = re(comsigma_fin)
        u_current[1] = re(comepsilon_fin)
    elif i == 1:
        V_candidate_dot_test = V_candidate_dot_test.subs({Derivative(comepsilon, t): 0.0, Derivative(comsigma, t): 0.0}).doit()

        comsigma_cand = V_candidate_dot_test.subs({comepsilon: u_last[1]})
        comsigma_range = solve(comsigma_cand, comsigma)
        if len(comsigma_range) > 0:
            comsigma_fin = comsigma_range[0] + (random.random() * (comsigma_range[-1] - comsigma_range[0]))
        elif len(comsigma_range) == 0:
            comsigma_fin = 1e-6
        else:
            comsigma_fin = comsigma_range

        comeps_cand = V_candidate_dot_test.subs({comsigma: u_last[0]})
        comepsilon_range = solve(comeps_cand, comepsilon)
        if len(comepsilon_range) > 0:
            comepsilon_fin = comepsilon_range[0] + (random.random() * (comepsilon_range[-1] - comepsilon_range[0]))
        elif len(comepsilon_range) == 0:
            comepsilon_fin = 1e-6
        else:
            comepsilon_fin = comepsilon_range

        u_current[0] = re(comsigma_fin)
        u_current[1] = re(comepsilon_fin)
    else:
        u_delta = (u_current - u_last) * t_delta
        V_candidate_dot_test = V_candidate_dot_test.subs({Derivative(comepsilon, t): u_delta[1], Derivative(comsigma, t): u_delta[0]}).doit()

        comsigma_cand = V_candidate_dot_test.subs({comepsilon: u_last[1]})
        comsigma_range = solve(comsigma_cand, comsigma)
        if len(comsigma_range) > 0:
            comsigma_fin = comsigma_range[0] + (random.random() * (comsigma_range[-1] - comsigma_range[0]))
        else:
            comsigma_fin = comsigma_range

        comeps_cand = V_candidate_dot_test.subs({comsigma: u_last[0]})
        comepsilon_range = solve(comeps_cand, comepsilon)
        if len(comepsilon_range) > 0:
            comepsilon_fin = comepsilon_range[0] + (random.random() * (comepsilon_range[-1] - comepsilon_range[0]))
        else:
            comepsilon_fin = comepsilon_range

        u_current[0] = re(comsigma_fin)
        u_current[1] = re(comepsilon_fin)

    # bounds u
    u_current = np.clip(u_current, -10.0, 10.0)

    # executes dynamics and computes new state
    x_current = discrete_simulation_dynamics(x_current, u_current, t_delta)

    # saves state to trajectory
    test_x_traj[:, i+1] = np.array(x_current)

    # updates controls
    u_last = np.copy(u_current)

visualizer = Visualizer(x_traj=test_x_traj)



