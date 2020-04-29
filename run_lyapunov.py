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

### inits symbols & variables

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

### extracts derivatives from simulation dynamics

x_in = [v, gamma, psi, x, y, h, sigma, epsilon]
k_x_in = [comsigma, comepsilon]
dotv, dotgamma, dotpsi, dotx, doty, doth, dotsigma, dotepsilon = simulation_dynamics(x_in, k_x_in, sympy_version=True)

### prepares error and rate variables

# q = Matrix(x_in)
# q_dest = Matrix([0, 0, 0, 0, 0, 0, 0, 0])
q = Matrix([x, y, h])
q_dest = Matrix([0, 0, 0])
e = (q_dest - q)
edot = Derivative(e, t).doit()
alpha = Symbol('alpha')
r = edot + (alpha * e)
rdot = Derivative(r, t).doit()

### simplifies candidate Lyapunov function and its derivative

V_candidate = 0.5 * r.transpose().dot(r)
# V_candidate = (x ** 2.0) #0.5 * (h ** 2.0) * ((x ** 2.0) + (y ** 2.0))
print(V_candidate)
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
V_candidate = V_candidate.subs(Derivative(x, t), dotx)
V_candidate = V_candidate.subs(Derivative(y, t), doty)
V_candidate = V_candidate.subs(Derivative(h, t), doth)
V_candidate = V_candidate.subs(Derivative(v, t), dotv)
V_candidate = V_candidate.subs(Derivative(gamma, t), dotgamma)
V_candidate = V_candidate.subs(Derivative(sigma, t), dotsigma)
V_candidate = V_candidate.subs(Derivative(psi, t), dotpsi)
V_candidate = V_candidate.subs(Derivative(epsilon, t), dotepsilon)

# print("Vcand: "+str(V_candidate))
# print("Vcanddot: "+str(V_candidate_dot))

### iteratively plots state trajectory with Lyapunov control

#                    [v, gamma, psi,   x,   y, h, sigma, eps]
x_current = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 3.5, 0.0, 0.0]) # starts with x_0
u_last = np.array([0.0, 0.0]) # tracks previous controls state, used to calculate deriv(u)
u_current = np.array([0.0, 0.0])
t_current = 0.0
t_delta = 0.01
n = 200
K = 5.0

test_x_traj = np.zeros((8, n+1))
test_x_traj[:, 0] = np.array(x_current)
for i in range(n):
    print(" * Iteration "+str(i)+" of "+str(n))

    # substitutes state into Vdot, V
    alpha_r = 2.0
    V_candidate_dot_test = V_candidate_dot.subs({x: x_current[3], y: x_current[4], h: x_current[5], epsilon: x_current[7], gamma: x_current[1], sigma: x_current[6], v: x_current[0], psi: x_current[2], alpha: alpha_r})
    V_candidate_test = V_candidate.subs({x: x_current[3], y: x_current[4], h: x_current[5], epsilon: x_current[7], gamma: x_current[1], sigma: x_current[6], v: x_current[0], psi: x_current[2], alpha: alpha_r})

    if i == 0: # picks essentially random controls on first iteration
        # V_candidate_dot_test = V_candidate_dot_test.subs({Derivative(comepsilon, t): 0.0, Derivative(comsigma, t): 0.0}).doit()

        # # picks random comsigma
        # comsigma_fin = (random.random()-0.5) * 2.0
        # V_candidate_dot_test = V_candidate_dot_test.subs({comsigma: comsigma_fin})
        
        # # picks comepsilon
        # comepsilon_range = solve(V_candidate_dot_test, comepsilon)
        # if len(comepsilon_range) > 0:
        #     comepsilon_fin = comepsilon_range[0] + (random.random() * (comepsilon_range[-1] - comepsilon_range[0]))
        # elif len(comepsilon_range) == 0:
        #     comepsilon_fin = 1e-6
        # else:
        #     comepsilon_fin = comepsilon_range

        # # saves to current controls history
        # u_current[0] = re(comsigma_fin)
        # u_current[1] = re(comepsilon_fin)
        u_current[0] = 0.0
        u_current[1] = 0.0
    elif i == 1: # picks slightly less random controls on second iteration
        # V_candidate_dot_test = V_candidate_dot_test.subs({Derivative(comepsilon, t): 0.0, Derivative(comsigma, t): 0.0}).doit()

        # # picks comsigma
        # comsigma_cand = V_candidate_dot_test.subs({comepsilon: u_last[1]})
        # comsigma_range = solve(comsigma_cand, comsigma)
        # if len(comsigma_range) > 0:
        #     comsigma_fin = comsigma_range[0] + (random.random() * (comsigma_range[-1] - comsigma_range[0]))
        # elif len(comsigma_range) == 0:
        #     comsigma_fin = 1e-6
        # else:
        #     comsigma_fin = comsigma_range

        # # picks comepsilon
        # comeps_cand = V_candidate_dot_test.subs({comsigma: u_last[0]})
        # comepsilon_range = solve(comeps_cand, comepsilon)
        # if len(comepsilon_range) > 0:
        #     comepsilon_fin = comepsilon_range[0] + (random.random() * (comepsilon_range[-1] - comepsilon_range[0]))
        # elif len(comepsilon_range) == 0:
        #     comepsilon_fin = 1e-6
        # else:
        #     comepsilon_fin = comepsilon_range

        # # saves to current controls history
        # u_current[0] = re(comsigma_fin)
        # u_current[1] = re(comepsilon_fin)
        u_current[0] = 0.0
        u_current[1] = 0.0
    else:
        # u_delta = (u_current - u_last)
        # V_candidate_dot_test = V_candidate_dot_test.subs({Derivative(comepsilon, t): u_delta[1], Derivative(comsigma, t): u_delta[0]}).doit()

        # solves for sigma and epsilon instead
        V_candidate_dot_test = V_candidate_dot.subs({x: x_current[3], y: x_current[4], h: x_current[5], gamma: x_current[1], v: x_current[0], psi: x_current[2], alpha: alpha_r})
        V_candidate_test = V_candidate.subs({x: x_current[3], y: x_current[4], h: x_current[5], gamma: x_current[1], v: x_current[0], psi: x_current[2], alpha: alpha_r})

        # enforces exponential stability
        V_candidate_dot_test = (V_candidate_dot_test / V_candidate_test) + K

        # picks comsigma
        # comsigma_cand = V_candidate_dot_test.subs({comepsilon: u_last[1]})
        comsigma_cand = V_candidate_dot_test.subs({epsilon: u_last[1]})
        # comsigma_range = solve(comsigma_cand, comsigma)
        comsigma_range = solve(comsigma_cand, sigma)
        if len(comsigma_range) > 0:
            comsigma_fin = comsigma_range[0] + (random.random() * (comsigma_range[-1] - comsigma_range[0]))
        elif len(comsigma_range) == 0:
            comsigma_fin = 1e-6
        else:
            comsigma_fin = comsigma_range

        # picks comepsilon
        # comeps_cand = V_candidate_dot_test.subs({comsigma: u_last[0]})
        comeps_cand = V_candidate_dot_test.subs({sigma: u_last[0]})
        # comepsilon_range = solve(comeps_cand, comepsilon)
        comepsilon_range = solve(comeps_cand, epsilon)
        if len(comepsilon_range) > 0:
            comepsilon_fin = comepsilon_range[0] + (random.random() * (comepsilon_range[-1] - comepsilon_range[0]))
        elif len(comepsilon_range) == 0:
            comepsilon_fin = 1e-6
        else:
            comepsilon_fin = comepsilon_range

        # saves to current controls history
        u_current[0] = re(comsigma_fin)
        u_current[1] = re(comepsilon_fin)

        # calculates current V
        V_current = V_candidate_test.subs({comsigma: u_current[0], comepsilon: u_current[1]})
        print(" * Current V: "+str(V_current))

    # bounds u
    u_current = np.clip(u_current, -100.0, 100.0)

    # executes dynamics and computes new state
    x_current = discrete_simulation_dynamics(x_current, u_current, t_delta)

    # saves state to trajectory
    test_x_traj[:, i+1] = np.array(x_current)

    # updates controls
    u_last = np.copy(u_current)


### graphs final state trajectory
visualizer = Visualizer(x_traj=test_x_traj, plot_heading=False)



