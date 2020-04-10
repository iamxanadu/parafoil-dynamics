import matplotlib.pyplot as plt
import numpy as np


class Visualizer(object):
    def __init__(self):
        pass

    ### private methods ###

    # converts state trajectory into separate time-indexed state variables
    def process_state(self, x_traj):
        pass

    ### public methods ###

    # plots the state of the parafoil/rocket over time
    def plot_state_trajectory(self, x_traj):
        pass

    # samples lyapunov function to approximate range of landing zone
    def plot_landing_zone(self, lyapunov_function):
        pass


# unit tests when running graphics.py directly
if __name__ == "__main__":
    visualizer = Visualizer()