from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np


class Visualizer(object):
    def __init__(self, x_traj=None, lyapunov_function=None):
        if x_traj is not None:
            self.plot_state_trajectory(x_traj)
        elif lyapunov_function is not None:
            self.plot_landing_zone(lyapunov_function)

    ### private methods ###

    # converts state trajectory into separate time-indexed state variables
    def process_state(self, x_traj):
        time = np.arange(x_traj.shape[-1])
        x_pos = x_traj[0, :]
        y_pos = x_traj[1, :]
        z_pos = x_traj[2, :]
        return (time, x_pos, y_pos, z_pos)

    ### public methods ###

    # plots the state of the parafoil/rocket over time
    def plot_state_trajectory(self, x_traj):
        time, x_pos, y_pos, z_pos = self.process_state(x_traj)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.plot3D(x_pos, z_pos, y_pos, 'blue')
        plt.show()

    # samples lyapunov function to approximate range of landing zone
    def plot_landing_zone(self, lyapunov_function):
        pass


# unit tests when running graphics.py directly
if __name__ == "__main__":

    test_x_traj = np.zeros((3, 30))
    test_x_traj[0] = np.cos(np.arange(30) / 2.0)
    test_x_traj[1] = 1.0 - (np.arange(30) / 29.0)
    test_x_traj[2] = np.sin(np.arange(30) / 2.1)
    visualizer = Visualizer(x_traj=test_x_traj)
