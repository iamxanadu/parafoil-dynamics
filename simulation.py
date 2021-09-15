from matplotlib import scale
from numpy import arange, sqrt, interp, linspace
from math import pi
from numpy.random import normal
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq, irfft
from numpy import cos, sin, array, max, min
from scipy.integrate import solve_ivp
import time
# from numba import jit
from celluloid import Camera

'''
Behavior of Von Karman wind model:
- Should be able to transparently call the class to get wind at any time
- Will need to update the model when height changes
-
'''


class VonKarmanWind():
    def __init__(self, u20=15, h0=2500, t0=0, v0=5, N=1e3, fs=0.1):
        self.ft2m = 0.3048
        self.u20 = u20

        self.N = int(N)  # Number of samples to draw
        self.fs = fs  # Sampling spatial frequency (1/ft)

        self._update_turb_params(h0)
        self.update(v0, h0, 0)

    def _Psiu(self, w): return sigu**2*2*Lu/pi * 1/(1 + (a*Lu*w)**2)**(5/6)

    def _Psiv(self, w): return sigv**2*2*Lv/pi * \
        (1 + 8/3*(2*a*Lv*w)**2)/(1+(2*a*Lv*w)**2)**(11/6)

    def _Psiw(self, w): return sigw**2*2*Lw/pi * \
        (1 + 8/3*(2*a*Lw*w)**2)/(1+(2*a*Lw*w)**2)**(11/6)

    def _update_turb_params(self, h: float):
        if h < 2000:
            self.sigw = 0.1*self.u20
            self.sigu = self.sigw/(0.177 + 0.000823*h)**0.4
            self.sigv = self.sigw/(0.177 + 0.000823*h)**0.4
            self.Lu = h/(0.177 + 0.000823*h)**1.2
            self.Lv = self.Lu/2
            self.Lw = h/2
        else:
            self.sigu = 5
            self.sigv = 5
            self.sigw = 5
            self.Lu = 2500
            self.Lv = 2500/2
            self.Lw = 2500/2

    def update(self, v: float, h: float, t0: float):
        self.t0 = t0
        '''
        1) Generate three rows of unit variance white gaussian noise samples
        '''
        x = normal(size=(3, self.N))
        '''
        2) Go to the frequency domain. Use real fft because we only care about real arguments.
        '''
        n = rfft(x)
        wf = rfftfreq(self.N, 1/self.fs)
        '''
        3) Make shaping filters using PSDs
        '''
        Gu = sqrt(self.fs*self._Psiu(wf))
        Gv = sqrt(self.fs*self._Psiv(wf))
        Gw = sqrt(self.fs*self._Psiw(wf))
        '''
        4) Transform the noise
        '''
        Yu = Gu*n[0, :]
        Yv = Gv*n[1, :]
        Yw = Gw*n[2, :]
        '''
        5) Go back to the spatial domain
        '''
        yu = irfft(Yu)*self.ft2m
        yv = irfft(Yv)*self.ft2m
        yw = irfft(Yw)*self.ft2m

        self.y = (yu, yv, yw)
        self.t = arange(self.N) / self.fs / v

    def evaluate(self, t: float):
        yu = interp(t - self.t0, self.t, self.y[0])
        yv = interp(t - self.t0, self.t, self.y[1])
        yw = interp(t - self.t0, self.t, self.y[2])

        if t > self.t[-1] + self.t0 or t < self.t0:
            # time out of bounds; return zeros
            return 0, 0, 0
        else:
            return yu, yv, yw

    def plot_von_karman_psd(self):
        wfshift = linspace(-10, 10, 1000)
        plt.figure()
        plt.title("Von Karman Power Spectra")
        plt.plot(wfshift, self._Psiu(wfshift/1000))
        plt.plot(wfshift, self._Psiv(wfshift/1000))
        plt.legend(["Psi_u", "Psi_(v/w)"])
        plt.ylabel("power density (ft/s)^2/(rad/ft)")
        plt.xlabel("frequency (miliradian/ft)")
        plt.show()

    def plot_von_karman_realization(self):
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title("Von Karman Turbulance Realization")
        plt.plot(self.t, self.y[0])
        plt.ylabel("u (m/s)")
        plt.subplot(3, 1, 2)
        plt.plot(self.t, self.y[1])
        plt.ylabel("v (m/s)")
        plt.subplot(3, 1, 3)
        plt.plot(self.t, self.y[2])
        plt.ylabel("w (m/s)")
        plt.xlabel("time (s)")
        plt.show()


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

    def run_to_time(self, t_step: float, t_f: float, x_0: list):
        state_hist = [array(x_0)]
        time_hist = [0]

        while time_hist[-1] + t_step < t_f:
            y_0 = state_hist[-1]
            t_next = time_hist[-1] + t_step
            sol = solve_ivp(
                self._dynamics, (time_hist[-1], t_next), y_0,
                t_eval=[t_next])
            state_hist.append(sol.y[:, 0])
            time_hist.append(sol.t[0])

        return time_hist, state_hist

    def run(self, t_step: float, x_0: list):
        state_hist = [array(x_0)]
        time_hist = [0]
        ground_hit = False

        while ground_hit == False:
            y_0 = state_hist[-1]
            t_next = time_hist[-1] + t_step
            if y_0[5] < x_0[5]*0.1:
                sol = solve_ivp(
                    self._dynamics, (time_hist[-1], t_next), y_0,
                    t_eval=[t_next], events=self._event_ground_hit)
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
                    t_eval=[t_next], method='RK23')
                state_hist.append(sol.y[:, 0])
                time_hist.append(sol.t[0])

        t = time_hist
        y = array(state_hist).T
        self._plot_sim_results(t, y)
        return t, y

    def dispersion(self):
        pass

    def enableWind(self, windOn):
        self.windOn = windOn

    def _event_ground_hit(self, t, state):
        _, _, _, _, _, h, _, _ = state
        return h

    def _dynamics(self, t, state):
        V, gamma, psi, x, y, h, sigma, epsilon = state
        comsigma, comepsilon = 0, 0

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

    def _plot_sim_results(self, t: array, y: array, arrow_every=200):

        # TODO may want to try to add in speed via coloring or another graph; right now just visible in animation.
        fig, axs = plt.subplots(2, 2)
        camera = Camera(fig)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        # Ground track
        for i in range(0, len(t), int(len(t)/100)):
            axs[0, 0].set_title("ground track + heading and speed")
            axs[0, 0].set_xlabel('x (m)')
            axs[0, 0].set_ylabel('y (m)')
            axs[0, 0].plot(y[3, :i], y[4, :i], color='b')
            axs[0, 0].set_aspect("equal", adjustable="datalim")
            axs[0, 0].set_box_aspect(3/4)
            for (h, v, psi) in zip(y[3, :i:arrow_every], y[4, :i:arrow_every], y[2, :i:arrow_every]):
                axs[0, 0].annotate('',
                                   xytext=(h, v),
                                   xy=(h + 0.01*cos(psi), v + 0.01*sin(psi)),
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
            axs[1, 1].plot(t[:i], y[6, :i], color='b')
            axs[1, 1].plot(t[:i], y[7, :i], color='r')

            camera.snap()
            print(i)

        animation = camera.animate(blit=True, interval=10)
        # Need to edit 'matplotlibrc' file here in order to tell it where imagemagick 'convert' binary is
        animation.save('out.gif')  # , writer='imagemagick', fps=30)


if __name__ == "__main__":
    s = Simulation()
    x0 = [7, -0.15, 0.01, -100, 100, 10, 0, 0]
    tic = time.time()
    t, y = s.run(0.01, x0)
    elapsed = time.time() - tic
    print(elapsed)

    # vkwind = VonKarmanWind()
    # vkwind.plot_compare_von_karman_psd(40)

    # white_seq = normal(size=1000000)
    # f, Pxx = welch(white_seq, return_onesided=False)
    # plt.plot(f, Pxx)
    # plt.show()
