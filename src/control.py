from numpy import cos, radians, sin, sqrt, exp, clip, arcsin
from numpy import meshgrid, linspace, zeros_like
from math import atan2
import matplotlib.pyplot as plt


class GeometricController():
    def __init__(self, plant, r0=30, umax=0.5, ht=100, P_gamma=1.0):
        self.r0 = r0  # [m]
        self.umax = umax
        self.ht = ht
        self.P_gamma = P_gamma

        self.plant = plant

    def geometricYawControl(self, x: list, a=0, terminal_radius_mult=1.5):
        # NOTE u > 0 is ccw motion

        # Unpack state variables
        v = x[0]
        psi = x[2]
        px = x[3]
        py = x[4]

        # Controller derived parameters
        eps = terminal_radius_mult*self.r0
        u0 = v/self.r0

        # Make sure that the terminal guidance radius is positive
        assert eps > 0, 'Epsilon must be a positive real number'

        # Calculate position in shifted body frame
        xbar = px*cos(psi) + py*sin(psi)
        ybar = -px*sin(psi) + py*cos(psi) + self.r0
        rbar = sqrt(xbar**2 + ybar**2)

        # Calculate dubbins yaw output
        if rbar > eps:
            u = atan2(-ybar, -xbar)
        else:
            if xbar <= -eps:
                u = a
            elif xbar >= 0:
                u = u0
            else:
                n = (u0 - a)
                expnt = 1/(xbar+eps) + 1/xbar
                d = 1 + exp(expnt)
                u = n/d + a

        return u

    def calcSigmaFromPsiDot(self, x: list, u: float, umax=1.0):

        v = x[0]
        gamma = x[1]
        eps = x[7]

        psidot_cmd = clip(u, -self.umax, self.umax)

        Cl = self.plant.Cltrim + self.plant.delCl * eps
        L = 1/2 * self.plant.rho * v**2 * self.plant.S * Cl
        sigma = arcsin(psidot_cmd*self.plant.m*v*cos(gamma)/L)

        return sigma

    def calcDescentRate(self, x: list):

        # TODO Add a D term to deal with damping oscillations in the glide path while in terminal circle
        gamma = x[1]
        px = x[3]
        py = x[4]
        alt = x[5]

        r = sqrt(px**2 + py**2)

        if alt > self.ht:
            glide_c = atan2(self.ht-alt, r)
        else:
            glide_c = -radians(5)

        return clip(self.P_gamma * (gamma - glide_c), -self.umax, self.umax)

    def u(self, x: list):

        u = self.geometricYawControl(x)
        c_sigma = self.calcSigmaFromPsiDot(x, u)
        c_epsilon = self.calcDescentRate(x)
        return c_sigma, c_epsilon