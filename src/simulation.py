from numpy import array, clip

from alive_progress import alive_bar

from .control import GeometricController
from .plant import RadmacherPlant


class Simulation():

    def __init__(self, plant: RadmacherPlant, controller: GeometricController):
        """A simulation of the Rademacher parafoil plant with the presented geometric controller.

        Args:
            plant (RadmacherPlant): Rademacher parafoil plant instance.
            controller (GeometricController): Geometric controller instance.
        """
        self.plant = plant
        self.controller = controller

    def run_to_time(self, t_step: float, t_f: float, x_0: list, plot_result=False, generate_gif=False) -> tuple:
        """Run the simulation to a given time.

        Args:
            t_step (float): Time step in seconds.
            t_f (float): Final time in seconds.
            x_0 (list): Initial state of the parafoil.
            plot_result (bool, optional): Whether to plot and animate the result. Defaults to False.
            generate_gif (bool, optional): Whether to produce and save a GIF of the animation. Defaults to False.

        Returns:
            tuple: The time and state history pair at each time step.
        """

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

    def run(self, t_step: float, x_0: list, plot_result=False, generate_gif=False) -> tuple:
        """Run the simulation until a ground hit.

        Args:
            t_step (float): Time step in seconds.
            x_0 (list): Initial state of the parafoil.
            plot_result (bool, optional): Whether to plot and animate the result. Defaults to False.
            generate_gif (bool, optional): Whether to produce and save a GIF of the animation. Defaults to False.

        Returns:
            tuple: The time and state history pair at each time step.
        """
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

                y = state_hist[-1]
                percent = clip((x_0[5]-y[5])/x_0[5], 0.0, 1.0)
                bar(percent)

        t = time_hist
        y = array(state_hist).T
        if plot_result:
            self.plant.plotStateHistory(t, y, produce_gif=generate_gif)
        return t, y
