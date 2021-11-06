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

from alive_progress import alive_bar

from .control import GeometricController
from .plant import RadmacherPlant

'''
Behavior of Von Karman wind model:
- Should be able to transparently call the class to get wind at any time
- Will need to update the model when height changes
-
'''


class Simulation():

    def __init__(self, plant: RadmacherPlant, controller: GeometricController):
        self.plant = plant
        self.controller = controller

    def run_to_time(self, t_step: float, t_f: float, x_0: list, plot_result=False, generate_gif=False):

        self.plant.setState(x_0)

        state_hist = [array(x_0)]
        time_hist = [0]

        print('Running simulation...')
        with alive_bar(100, manual=True) as bar:
            while time_hist[-1] + t_step < t_f:
                y_0 = state_hist[-1]
                t_next = time_hist[-1] + t_step
                u = self.controller.u(y_0)
                self.plant.setControl(u)
                state_hist.append(self.plant.step(t_step))
                time_hist.append(t_next)

                percent = clip(t_next/t_f, 0.0, 1.0)
                bar(percent)
            bar(1.0)
        t = time_hist
        y = array(state_hist).T
        if plot_result:
            self.plant.plotStateHistory(t, y, produce_gif=generate_gif)
        return t, y

    def run(self, t_step: float, x_0: list, plot_result=False, generate_gif=False):

        self.plant.setState(x_0)

        state_hist = [array(x_0)]
        time_hist = [0]

        print('Running simulation...')
        with alive_bar(100, manual=True) as bar:
            while True:
                y_0 = state_hist[-1]

                if y_0[5] < 0:
                    bar(1.0)
                    break

                t_next = time_hist[-1] + t_step

                u = self.controller.u(y_0)
                self.plant.setControl(u)

                state_hist.append(self.plant.step(t_step))
                time_hist.append(t_next)

                # if y_0[5] < x_0[5]*0.1:
                #     sol = solve_ivp(
                #         self._dynamics, (time_hist[-1], t_next), y_0,
                #         t_eval=[t_next], events=self._event_ground_hit, args=(u))
                #     if sol.status != 0:
                #         print('Integration failed!')
                #         break
                #     if sol.t_events[0].size == 0:
                #         state_hist.append(sol.y[:, 0])
                #         time_hist.append(sol.t[0])
                #     else:
                #         state_hist.append(sol.y_events[0][0])
                #         time_hist.append(sol.t_events[0][0])
                #         ground_hit = True
                # else:
                #     sol = solve_ivp(
                #         self._dynamics, (time_hist[-1], t_next), y_0,
                #         t_eval=[t_next], method='RK23', args=u)
                #     if sol.status != 0:
                #         print('Integration failed!')
                #         break
                #     state_hist.append(sol.y[:, 0])
                #     time_hist.append(sol.t[0])

                y = state_hist[-1]
                percent = clip((x_0[5]-y[5])/x_0[5], 0.0, 1.0)
                bar(percent)

        t = time_hist
        y = array(state_hist).T
        if plot_result:
            self.plant.plotStateHistory(t, y, produce_gif=generate_gif)
        return t, y
