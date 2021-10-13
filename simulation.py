from matplotlib import scale
from numpy import arange, arcsin, sqrt, interp, linspace
from math import pi
from numpy.random import normal
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq, irfft
from numpy import cos, sin, array, max, min, clip, exp
from math import atan2, asin
from scipy.integrate import solve_ivp
import time
# from numba import jit
from celluloid import Camera
from alive_progress import alive_bar

'''
Behavior of Von Karman wind model:
- Should be able to transparently call the class to get wind at any time
- Will need to update the model when height changes
-
'''
class Simulation():

    def __init__(self):
        self.S = 2  # m^2
        self.rho = 1.225  # kg/m^3
        self.m = 2.5  # kg
        self.g = 9.81  # m/s/s

        # Rademacher (2009)
        self.taus = 1.0
        self.taue = 0.8

        # Brown (1993)
        self.Cltrim = 0.5  # [0.4 1.0]
        self.Cdtrim = 0.09
        self.delCl = 0.467
        self.delCd = 0.19

        self.windOn = False

    def dispersion(self):
        pass

    def enableWind(self, windOn):
        self.windOn = windOn

    def geometricYawControl(self, x: list, r0=30):

        v = x[0]
        psi = x[2]
        px = x[3]
        py = x[4]

        eps = 1.5*r0
        a = 0
        u0 = v/r0

        assert eps > 0, 'Epsilon must be a positive real number'

        # p = np.array([[px, py]]).T
        # R = np.array([[cos(psi), sin(psi)], [-sin(psi), cos(psi)]])
        # ptild = R@p
        xbar = px*cos(psi) + py*sin(psi)
        ybar = -px*sin(psi) + py*cos(psi) + r0

        rbar = sqrt(xbar**2 + ybar**2)

        if rbar > 1.5*r0:
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

    def calcSigmaFromPsiDot(self, x: list, u: float, mu: float, umax=1.0):

        v = x[0]
        gamma = x[1]

        psidot_cmd = clip(u, -umax, umax)

        Cl = self.Cltrim + self.delCl * mu
        L = 1/2 * self.rho * v**2 * self.S * Cl
        sigma = arcsin(psidot_cmd*self.m*v*cos(gamma)/L)

        return sigma

    def control(self, x: list, mu: float, r0: float):
        # NOTE u > 0 is ccw motion
        u = self.geometricYawControl(x, r0)
        sigma = self.calcSigmaFromPsiDot(x, u, mu)
        return sigma, mu

    def run_to_time(self, t_step: float, t_f: float, x_0: list):
        state_hist = [array(x_0)]
        time_hist = [0]

        while time_hist[-1] + t_step < t_f:
            y_0 = state_hist[-1]
            t_next = time_hist[-1] + t_step
            u = self.control(y_0, -0.05, 30.0)
            print(f'Taking step at time {time_hist[-1]}')
            sol = solve_ivp(
                self._dynamics, (time_hist[-1], t_next), y_0,
                t_eval=[t_next], args=(u))
            if sol.status != 0:
                print('Integration failed!')
                break
            state_hist.append(sol.y[:, 0])
            time_hist.append(sol.t[0])

        t = time_hist
        y = array(state_hist).T
        self._plot_sim_results(t, y)
        return t, y

    def run(self, t_step: float, x_0: list):
        state_hist = [array(x_0)]
        time_hist = [0]
        ground_hit = False
        print('Running simulation')
        with alive_bar(100, manual=True) as bar:
            while ground_hit == False:
                y_0 = state_hist[-1]
                t_next = time_hist[-1] + t_step
                u = self.control(y_0, -0.05, 30.0)
                if y_0[5] < x_0[5]*0.1:
                    sol = solve_ivp(
                        self._dynamics, (time_hist[-1], t_next), y_0,
                        t_eval=[t_next], events=self._event_ground_hit, args=(u))
                    if sol.status != 0:
                        print('Integration failed!')
                        break
                    if sol.t_events[0].size == 0:
                        state_hist.append(sol.y[:, 0])
                        time_hist.append(sol.t[0])
                    else:
                        state_hist.append(sol.y_events[0][0])
                        time_hist.append(sol.t_events[0][0])
                        ground_hit = True
                else:
                    sol = solve_ivp(
                        self._dynamics, (time_hist[-1], t_next), y_0,
                        t_eval=[t_next], method='RK23', args=u)
                    if sol.status != 0:
                        print('Integration failed!')
                        break
                    state_hist.append(sol.y[:, 0])
                    time_hist.append(sol.t[0])
                percent = clip((x_0[5]-sol.y[5, 0])/x_0[5], 0.0, 1.0)
                bar(percent)
        t = time_hist
        y = array(state_hist).T
        self._plot_sim_results(t, y)
        return t, y

    def _event_ground_hit(self, t, state, comsigma, comepsilon):
        _, _, _, _, _, h, _, _ = state
        return h

    def _dynamics(self, t, state, comsigma, comepsilon):
        V, gamma, psi, x, y, h, sigma, epsilon = state

        # TODO Zero wind until I implement wind distribution
        wx = wy = 0
        # if wx_func is not None and wy_func is not None:
        #     raise NotImplementedError

        Cl = self.Cltrim + self.delCl * epsilon
        L = 1/2 * self.rho * V**2 * self.S * Cl
        Cd = self.Cdtrim + self.delCd * epsilon
        D = 1/2 * self.rho * V**2 * self.S * Cd

        W = self.m * self.g

        dotV = -(D + W * sin(gamma)) / (self.m)
        dotgamma = (L * cos(sigma) - W *
                    cos(gamma)) / ((self.m * V))
        dotpsi = (L * sin(sigma)) / (self.m * V * cos(gamma))
        dotx = V * cos(gamma) * cos(psi) + wx
        doty = V * cos(gamma) * sin(psi) + wy
        doth = V * sin(gamma)
        dotsigma = (comsigma - sigma)/self.taus
        dotepsilon = (comepsilon - epsilon)/self.taue

        return [dotV, dotgamma, dotpsi, dotx, doty, doth, dotsigma, dotepsilon]

    def _plot_sim_results(self, t: array, y: array, r0=30, heading=False):

        # TODO may want to try to add in speed via coloring or another graph; right now just visible in animation.
        fig, axs = plt.subplots(2, 2)
        camera = Camera(fig)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        n = y.shape[1]
        arrow_every = int(0.005*n)  # heading arrow on 0.5% of points
        theta = linspace(0, 2*pi, 100)
        # Ground track
        print('Generating GIF')
        ts = range(0, len(t), int(len(t)/100))
        with alive_bar(len(ts)) as bar:
            for i in ts:
                axs[0, 0].set_title("ground track + heading")
                axs[0, 0].set_xlabel('x (m)')
                axs[0, 0].set_ylabel('y (m)')
                axs[0, 0].plot(y[3, :i], y[4, :i], color='b')
                axs[0, 0].plot(r0*cos(theta), r0*sin(theta),
                               color='k', linestyle='dashed')
                axs[0, 0].set_aspect("equal", adjustable="datalim")
                axs[0, 0].set_box_aspect(3/4)
                if heading:
                    for (h, v, psi) in zip(y[3, :i:arrow_every], y[4, :i:arrow_every], y[2, :i:arrow_every]):
                        axs[0, 0].annotate('',
                                           xytext=(h, v),
                                           xy=(h + 0.01*cos(psi),
                                               v + 0.01*sin(psi)),
                                           arrowprops=dict(
                                               arrowstyle="fancy", color=None),
                                           size=20
                                           )

                # Altitude
                axs[0, 1].set_title("altitude")
                axs[0, 1].set_xlabel('t (s)')
                axs[0, 1].set_ylabel('alt (m)')
                axs[0, 1].plot(t[:i], y[5, :i], color='b')

                # Flight path angle
                axs[1, 0].set_title("flight path angle")
                axs[1, 0].set_xlabel('t (s)')
                axs[1, 0].set_ylabel('angle (deg)')
                axs[1, 0].plot(t[:i], 180/pi*y[1, :i], color='b')

                # Control
                axs[1, 1].set_title("control")
                axs[1, 1].set_xlabel('t (s)')
                axs[1, 1].set_ylabel('temp')
                axs[1, 1].legend(['sigma', 'epsilon'], loc='lower right')
                axs[1, 1].plot(t[:i], y[6, :i], color='b')
                axs[1, 1].plot(t[:i], y[7, :i], color='r')

                camera.snap()
                bar()

        animation = camera.animate(blit=True, interval=100)
        # Need to edit 'matplotlibrc' file here in order to tell it where imagemagick 'convert' binary is
        animation.save('out.gif', writer='pillow', fps=30)

    def phase_control(self, x, y):
        r0 = 30  # [m]
        u0 = 1/r0
        r = sqrt(x**2 + (y-r0)**2)
        rbar = sqrt(x**2 + y**2)

        if r <= r0:
            return 0
        elif rbar > 1:
            return rbar + x
        else:
            return u0

    def phase_plot(self):
        r0 = 30  # [m]
        bound = 10
        npts = 1000
        xbar, ybar = np.meshgrid(
            np.linspace(-bound, bound, npts), np.linspace(-bound, bound, npts))
        u, v = np.zeros_like(xbar), np.zeros_like(xbar)
        NI, NJ = xbar.shape
        for i in range(NI):
            for j in range(NJ):
                x, y = xbar[i, j], ybar[i, j]
                c = self.phase_control(x, y)
                u[i, j] = c*y + 1 - r0*c
                v[i, j] = -c*x
        fig = plt.figure()
        plt.streamplot(xbar, ybar, u, v)
        plt.axis('square')
        plt.show()


if __name__ == "__main__":
    s = Simulation()
    x0 = [7, 0, pi/2, -500, 100, 500, 0, 0]
    t, y = s.run(0.01, x0)
