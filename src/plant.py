'''
TODO
- Should handle the lowest level dynamics
- Should handle integrating the dynamics
- Should hold the current state
- Should have a current control too, which can be set before integration
'''
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import sin, cos, array, linspace, pi, degrees
import yaml


class RadmacherPlant():
    def __init__(self, config):
        self.x = self._makeInitialState()
        self.u = (0, 0)

        with open(config, 'r') as file:
            c = yaml.safe_load(file)

            self.S = c['S']
            self.rho = c['rho']
            self.m = c['m']
            self.g = c['g']

            self.taus = c['taus']
            self.taue = c['taue']

            self.Cltrim = c['Cltrim']
            self.Cdtrim = c['Cdtrim']
            self.delCl = c['delCl']
            self.delCd = c['delCd']

    def _dynamics(self, t, state, comsigma, comepsilon):
        V, gamma, psi, _, _, _, sigma, epsilon = state

        # TODO Zero wind until I implement wind distribution
        wx = wy = 0

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

    def _plotToIndex(self, axs, i: int, t: array, y: array, r0: float, theta: array, heading: bool, arrow_every: int):
        axs[0, 0].plot(y[3, :i], y[4, :i], color='b')
        axs[0, 0].plot(r0*cos(theta), r0*sin(theta),
                       color='k', linestyle='dashed')

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

        axs[0, 1].plot(t[:i], y[5, :i], color='b')

        axs[1, 0].plot(t[:i], degrees(y[1, :i]), color='b')

        axs[1, 1].plot(t[:i], y[6, :i], color='b')
        axs[1, 1].plot(t[:i], y[7, :i], color='r')

    def plotStateHistory(self, t: array, y: array, speed=1.0, r0=30, heading=False, produce_gif=False):

        n = len(t)
        arrow_every = int(0.005*n)  # heading arrow on 0.5% of points
        theta = linspace(0, 2*pi, 100)
        if int(n/100) == 0:
            step = 1
        else:
            step = 1+int(n/100)
        idx = list(range(0, n, step))

        fig, axs = plt.subplots(2, 2)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        # Ground track
        if heading:
            axs[0, 0].set_title("ground track + heading")
        else:
            axs[0, 0].set_title("ground track")
        axs[0, 0].set_xlabel('x (m)')
        axs[0, 0].set_ylabel('y (m)')
        axs[0, 0].set_aspect("equal", adjustable="datalim")
        axs[0, 0].set_box_aspect(3/4)

        # Altitude
        axs[0, 1].set_title("altitude")
        axs[0, 1].set_xlabel('t (s)')
        axs[0, 1].set_ylabel('alt (m)')

        # Flight path angle
        axs[1, 0].set_title("flight path angle")
        axs[1, 0].set_xlabel('t (s)')
        axs[1, 0].set_ylabel('angle (deg)')

        # Control
        axs[1, 1].set_title("control")
        axs[1, 1].set_xlabel('t (s)')
        axs[1, 1].set_ylabel('control magnitude (deg)')
        axs[1, 1].legend(['sigma', 'epsilon'], loc='lower right')

        if produce_gif:
            from celluloid import Camera
            from alive_progress import alive_bar

            camera = Camera(fig)

            print('Generating GIF')
            with alive_bar(len(idx)) as bar:
                for i in idx:
                    self._plotToIndex(axs, i, t, y, r0, theta,
                                      heading, arrow_every)
                    camera.snap()
                    bar()
            animation = camera.animate(blit=True, interval=10*speed)
            # NOTE May need to edit 'matplotlibrc' file here in order to tell matplotlib where imagemagick 'convert' binary is
            print('Please wait...')
            animation.save('out.gif', writer='pillow', fps=30)

        else:
            def animate(k):
                self._plotToIndex(axs, k, t, y, r0, theta,
                                  heading, arrow_every)
                return k

            ani = FuncAnimation(fig, animate, frames=idx,
                                interval=10, repeat=True)
            plt.show()

    def _makeInitialState(self):
        return [0.1, 0.05, 0, 0, 0, 0, 0, 0]

    def setState(self, x: list):
        # TODO enforce common sense state limits here?
        self.x = x

    def setControl(self, u: list):
        # TODO enforce some kind of limits here
        self.u = u

    def setPosition(self, px: float, py: float, h: float):
        self.x[3] = px
        self.x[4] = py
        self.x[5] = h

    def setYaw(self, psi: float):
        self.x[2] = psi

    def getState(self):
        return self.x

    def step(self, dt) -> list:
        sol = solve_ivp(self._dynamics, (0, dt), self.x,
                        t_eval=[dt], args=(self.u))
        if sol.status != 0:
            print('Integration failed!')
            return None
        self.x = sol.y[:, 0]
        return self.x
