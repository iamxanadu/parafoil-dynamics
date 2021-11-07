from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import ndarray, sin, cos, array, linspace, pi, degrees
import yaml


class RadmacherPlant():
    def __init__(self, config):
        """A plant for the parafoil dynamics given by Rademacher [1].

        [1]B. Rademacher, “In-flight trajectory planning and guidance for autonomous parafoils,” 2009, doi: 10.2514/1.44862.

        Args:
            config (string): The path to the YAML config with the parameters for the model.
        """
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

    def _dynamics(self, t: float, state: ndarray, comsigma: float, comepsilon: float) -> list:
        """Calculates the dynamics of the parafoil according to the model presented in Rademacher.

        Args:
            t (float): The current time (not used for calculation)
            state (ndarray): The current state. This is the velocity V in m/s, the pitch angle gamma in radians, the headins psi in radians, the x, y, and z positions in m, the pseudo-bank angle sigma in radians, and the pseudo-pitch comand epsilon in radians in that order. 
            comsigma (float): The commanded pseudo-bank angle in radians.
            comepsilon (float): The commanded pseudo-pitch angle in radians.

        Returns:
            list: The derivatives of all the state variables in the same order they appear in in the 'state' argument.
        """
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

    def _plotToIndex(self, lines: tuple, i: int, t: array, y: array, r0: float) -> tuple:
        """Updates the graph actors up to a given data index.

        Args:
            lines (tuple): List of animation actors to update.
            i (int): The index to update to.
            t (array): The array of time points.
            y (array): The arraay of state values.
            r0 (float): The radius of the terminal manifold for the parafoil.

        Returns:
            tuple: The list of actors to be updated.
        """

        lines[0].set_data(y[3, :i], y[4, :i])

        lines[1].set_data(t[:i], y[5, :i])

        lines[2].set_data(t[:i], degrees(y[1, :i]))

        lines[3].set_data(t[:i], degrees(y[6, :i]))
        lines[4].set_data(t[:i], degrees(y[7, :i]))

        return lines

    def plotStateHistory(self, t: array, y: array, speed=1.0, r0=30, produce_gif=False):
        """Plots the state history for the parafoil.

        Args:
            t (array): Array of time points.
            y (array): Array of parafoil states.
            speed (float, optional): Speed of the animation. Higher is faster. Defaults to 1.0.
            r0 (int, optional): Radius of the terminal manifold for the parafoil controller in meters. Defaults to 30.
            produce_gif (bool, optional): Whether to produce a gif output. Saves to 'traj.gif'. Defaults to False.
        """

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

        lines = (axs[0, 0].plot([], [], color='b')[0],
                 axs[0, 1].plot([], [], color='b')[0],
                 axs[1, 0].plot([], [], color='b')[0],
                 axs[1, 1].plot([], [], color='b')[0],
                 axs[1, 1].plot([], [], color='r')[0])

        # Ground track
        axs[0, 0].set_title("ground track")
        axs[0, 0].set_xlabel('x (m)')
        axs[0, 0].set_ylabel('y (m)')
        axs[0, 0].set_aspect("equal", adjustable="datalim")
        axs[0, 0].set_box_aspect(3/4)
        axs[0, 0].plot(r0*cos(theta), r0*sin(theta),
                       color='k', linestyle='dashed')

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

        axs[0, 0].update_datalim(list(zip(y[3, :], y[4, :])))
        axs[0, 1].update_datalim(list(zip(t, y[5, :])))
        axs[1, 0].update_datalim(list(zip(t, degrees(y[1, :]))))
        axs[1, 1].update_datalim(list(zip(t, degrees(y[7, :]))))
        axs[1, 1].update_datalim(list(zip(t, degrees(y[6, :]))))

        def animate(k):
            self._plotToIndex(lines, k, t, y, r0)
            return lines

        ani = FuncAnimation(fig, animate, frames=idx,
                            interval=10, repeat=True, blit=True)
        if produce_gif:
            print('Generating GIF...')
            ani.save('traj.gif', writer='pillow', fps=30)
            print('Done!')

        plt.show()

    def _makeInitialState(self):
        return [0.1, 0.05, 0, 0, 0, 0, 0, 0]

    def setState(self, x: list):
        """Set the state of the parafoil.

        Args:
            x (list): The state to be set. This is the velocity V in m/s, the pitch angle gamma in radians, the headins psi in radians, the x, y, and z positions in m, the pseudo-bank angle sigma in radians, and the pseudo-pitch comand epsilon in radians in that order.
        """
        # TODO enforce common sense state limits here?
        self.x = x

    def setControl(self, u: list):
        """Set the control of the parafoil

        Args:
            u (list): The control to be set. This is the commanded pseudo-bank angle and commanded pseudo-pitch angle in that order
        """
        # TODO enforce some kind of limits here
        self.u = u

    def setPosition(self, px: float, py: float, h: float):
        """Set the position of the parafoil.

        Args:
            px (float): x position in meters
            py (float): y position in meters
            h (float): altitude in meters
        """
        self.x[3] = px
        self.x[4] = py
        self.x[5] = h

    def setYaw(self, psi: float):
        """Set the yaw of the parafoil.

        Args:
            psi (float): Yaw angle in radians
        """
        self.x[2] = psi

    def getState(self) -> list:
        """Get the current parafoil state.

        Returns:
            list: The current parafoil state. This is the velocity V in m/s, the pitch angle gamma in radians, the headins psi in radians, the x, y, and z positions in m, the pseudo-bank angle sigma in radians, and the pseudo-pitch comand epsilon in radians in that order.
        """
        return self.x

    def step(self, dt: float) -> list:
        """Advance the state of the parafoil by a given time step.

        Args:
            dt (float): The time interval by which to advance the state

        Returns:
            list: The state of the parafoil after the step. This is the velocity V in m/s, the pitch angle gamma in radians, the headins psi in radians, the x, y, and z positions in m, the pseudo-bank angle sigma in radians, and the pseudo-pitch comand epsilon in radians in that order.
        """

        sol = solve_ivp(self._dynamics, (0, dt), self.x,
                        t_eval=[dt], args=(self.u))
        if sol.status != 0:
            print('Integration failed!')
            return None
        self.x = sol.y[:, 0]
        return self.x
