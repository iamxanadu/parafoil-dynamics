from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


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
        V = x_traj[0:3, :]
        gamma = x_traj[3, :]
        psi = x_traj[4, :]
        x_pos = x_traj[5, :]
        y_pos = x_traj[6, :]
        h = x_traj[7, :]
        sigma = x_traj[8, :]
        eta = x_traj[9, :]
        return (time, V, gamma, psi, x_pos, y_pos, h, sigma, eta)

    ### public methods ###

    # plots the state of the parafoil/rocket over time
    def plot_state_trajectory(self, x_traj):
        time, V, gamma, psi, x_pos, y_pos, h, sigma, eta = self.process_state(x_traj)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.plot3D(x_pos, y_pos, h, 'blue')
        ax.scatter(x_pos, y_pos, h, marker='o')

        # plots vectors
        for i in range(V.shape[-1]):

            # plots v
            v = V[:, i]
            x = x_pos[i]
            y = y_pos[i]
            height = h[i]
            a = Arrow3D([x, x + v[0]], [y, y + v[1]], 
                [height, height + v[2]], mutation_scale=10, 
                lw=2, arrowstyle="-|>", color="r")
            ax.add_artist(a)

            # plots simple parafoil
            parafoil_width = 0.3 / 2
            parafoil_depth = 0.15 / 2

            # TODO: modify xs, ys, and hs depending on gamma, psi, and sigma(?)
            gm = gamma[i]
            ps = psi[i]
            xs = [x - parafoil_width, x + parafoil_width, x + parafoil_width, x - parafoil_width]
            ys = [y - parafoil_depth, y - parafoil_depth, y + parafoil_depth, y + parafoil_depth]
            hs = [height, height, height, height]

            verts = [list(zip(xs, ys, hs))]
            collection = Poly3DCollection(verts, linewidths=1, edgecolors='red', alpha=0.2, zsort='min')
            face_color = "salmon"
            collection.set_facecolor(face_color)
            ax.add_collection3d(collection)

        plt.show()

    # samples lyapunov function to approximate range of landing zone
    def plot_landing_zone(self, lyapunov_control_function):
        pass


# unit tests when running graphics.py directly
if __name__ == "__main__":

    test_x_traj = np.zeros((10, 30))
    test_x_traj[4] = -np.arange(30) * np.pi * 2.0 / 2.05
    test_x_traj[5] = np.cos(np.arange(30) / 2.0)
    test_x_traj[6] = np.sin(np.arange(30) / 2.1)
    test_x_traj[7] = 1.0 - (np.arange(30) / 29.0)
    test_x_traj[0] = -np.sin(np.arange(30) / 2.0) / 4.0
    test_x_traj[1] = np.cos(np.arange(30) / 2.1) / 4.0
    test_x_traj[2] = -1.0 / 10.0
    visualizer = Visualizer(x_traj=test_x_traj)
