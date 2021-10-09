'''
TODO 
- Should handle the lowest level dynamics
- Should handle integrating the dynamics
- Should hold the current state 
- Should have a current control too, which can be set before integration
'''
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class RadmacherPlant():
    def __init__(self):
        self.x

    def _dynamics(self, t, state, comsigma, comepsilon):
        V, gamma, psi, _, _, _, sigma, epsilon = state

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

    def step(self, dt):
        sol = solve_ivp(self._dynamics, (0, dt), self.x,
                        t_eval=[dt], args=(self.u))
        self.x = sol.y[:, 0]
