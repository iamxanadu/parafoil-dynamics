from numpy import cos, radians, sin, sqrt, exp, clip, arcsin
from math import atan2

from .plant import RadmacherPlant


class GeometricController():
    def __init__(self, plant: RadmacherPlant, r0=30, umax=0.5, ht=100, P_gamma=1.0):
        """The geometric controller presented in this work. Utilizes a Lyapunov-based controller for steering adapted from [1].

        Args:
            plant (RadmacherPlant): Rademacher parafoil plant instance.
            r0 (int, optional): Radius of the terminal manifold for the controller in meters. Defaults to 30.
            umax (float, optional): The maximum control inputs allowed in radians. Defaults to 0.5.
            ht (int, optional): The altitude at which to transition to terminal descent rate. Defaults to 100.
            P_gamma (float, optional): Gain for the proportional control of the glide path angle. Defaults to 1.0.
        """
        self.r0 = r0  # [m]
        self.umax = umax
        self.ht = ht
        self.P_gamma = P_gamma

        self.plant = plant

    def geometricYawControl(self, x: list, a=0, terminal_radius_mult=1.5) -> float:
        """Generates the Lyapunov geometric yaw control based on state feedback.

        Args:
            x (list): The current state of the parafoil.
            a (int, optional): Parameter of the controller. See [1]. Defaults to 0.
            terminal_radius_mult (float, optional): Multiple of the terminal radius at which to switch to terminal Lyapunov guidance for the heading. Defaults to 1.5.

        Returns:
            float: Commanded turning rate in radians per second.
        """

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

    def calcSigmaFromPsiDot(self, x: list, u: float) -> float:
        """Calculates a pseudo-bank angle control from the turning rate.

        Args:
            x (list): The current parafoil state.
            u (float): The commanded turning rate.

        Returns:
            float: The commanded pseudo-bank angle.
        """

        v = x[0]
        gamma = x[1]
        eps = x[7]

        psidot_cmd = clip(u, -self.umax, self.umax)

        Cl = self.plant.Cltrim + self.plant.delCl * eps
        L = 1/2 * self.plant.rho * v**2 * self.plant.S * Cl
        sigma = arcsin(psidot_cmd*self.plant.m*v*cos(gamma)/L)

        return sigma

    def calcDescentRate(self, x: list) -> float:
        """Calculate descent rate for the parafoil.

        Args:
            x (list): The current parafoil state.

        Returns:
            float: The commanded pseudo-pitch angle.
        """

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

    def u(self, x: list) -> tuple:
        """Return the calculated controls for a given state.

        Args:
            x (list): The current state of the parafoil.

        Returns:
            tuple: The commanded pseudo-bank and pitch angles in that order.
        """

        u = self.geometricYawControl(x)
        c_sigma = self.calcSigmaFromPsiDot(x, u)
        c_epsilon = self.calcDescentRate(x)
        return c_sigma, c_epsilon
