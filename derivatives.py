from sympy import Symbol, Function, Matrix
from sympy import sin, cos
from sympy import pprint, init_printing
from sympy import simplify, lambdify

import numpy as np

import inspect

import constants

init_printing()


class derivatives():

    def __init__(self):
        t = Symbol('t')

        m = Symbol('m')
        g = Symbol('g')
        rho = Symbol('rho')  # Air density
        S = Symbol('S')  # Reference area (m^2)
        Cltrim = Symbol('Clt')
        Cdtrim = Symbol('Cdt')
        dCl = Symbol('dCl')
        dCd = Symbol('dCd')

        self.dc = {g: constants.g, m: constants.m, rho: constants.rho, Cltrim: constants.Cltrim,
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

        # First derivatives of dynamics
        self.fx = self.f.jacobian(self.xs)
        self.fu = self.f.jacobian(self.u)

        # Second derivative tensors for general cost
        self.fxx = self.fx.diff(self.xs)
        self.fus = self.fu.diff(self.xs)
        self.fuu = self.fu.diff(self.u)

        '''
        self.fx = lambdify((self.xs.tolist(), self.u.tolist()),
                           self.f.jacobian(self.xs), modules='numpy')
        self.fu = lambdify((self.xs.tolist(), self.u.tolist()),
                           self.f.jacobian(self.u), modules='numpy')
        '''

        self.func_fx = lambdify(
            (self.xs.tolist(), self.u.tolist()), self.fx, modules='numpy')

        self.func_fu = lambdify(
            (self.xs.tolist(), self.u.tolist()), self.fu, modules='numpy')

    def getQuadraticCostDerivatives(self, x, u, Q, Qf, R):
        return x.T@Q, u.T@R, Q, np.zeros((x.shape(0), u.shape(1))), R

    def getDynamicsJacobian(self, x, u):
        # NOTE x and u must be column vectors
        # TODO sub in the constants
        # TODO convert to a float type (is dtype=object right now)
        return self.func_fx(x, u), self.func_fu(x, u)

    def getDerivativesILQRQuadraticCost(self, x, u, Q, R, Qf):
        fx, fu = self.getDynamicsJacobian(x, u)
        lx, lu, lxx, lux, luu = self.getQuadraticCostDerivatives(x, u, Q, Qf, R)

        return lx, lu, lxx, lux, luu, fx, fu


d = derivatives()

'''
print(d.getDynamicsJacobian(
    np.array([[0, 1, 2, 3, 4, 5]]).T, np.array([[0.1, 0.1]]).T))
'''
