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
        V = x_traj[0, :]
        gamma = x_traj[1, :]
        psi = x_traj[2, :]
        x_pos = x_traj[3, :]
        y_pos = x_traj[4, :]
        h = x_traj[5, :]
        sigma = x_traj[6, :]
        eta = x_traj[7, :]
        return (time, V, gamma, psi, x_pos, y_pos, h, sigma, eta)

    ### public methods ###

    # plots the state of the parafoil/rocket over time
    def plot_state_trajectory(self, x_traj):
        time, V, gamma, psi, x_pos, y_pos, h, sigma, eta = self.process_state(x_traj)

        fig = plt.figure(figsize=plt.figaspect(1.5))
        fig.suptitle('State Trajectory')

        ax0 = fig.add_subplot(2, 1, 1, projection='3d')
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Altitude')
        ax0.plot3D(x_pos, y_pos, h, 'blue')
        ax0.scatter(x_pos, y_pos, h, marker='o')

        # plots vectors
        parafoil_scale = ((np.amax(x_pos) - np.amin(x_pos)) + (np.amax(y_pos) - np.amin(y_pos))) / 4
        for i in range(V.shape[-1]):

            # plots simple parafoil
            parafoil_width = 0.3 * 0.5 * parafoil_scale
            parafoil_depth = 0.15 * 0.5 * parafoil_scale
            x = x_pos[i]
            y = y_pos[i]
            height = h[i]

            # modifies xs, ys, and hs depending on gamma, psi, and sigma(?)
            beta = gamma[i]
            alpha = psi[i] + (np.pi/2)
            points = np.array([
                [-parafoil_depth, -parafoil_width, 0],
                [-parafoil_depth, parafoil_width, 0],
                [parafoil_depth, parafoil_width, 0],
                [parafoil_depth, -parafoil_width, 0],
            ]).transpose()
            heading = np.array([
                [1.0, 0.0, 0.0],
            ]).transpose()
            transformation_matrix = np.array([
                [np.cos(alpha)*np.cos(beta), -np.sin(alpha), np.cos(alpha)*np.sin(beta)],
                [np.sin(alpha)*np.cos(beta), np.cos(alpha), np.sin(alpha)*np.sin(beta)],
                [-np.sin(beta), 0, np.cos(beta)],
            ])
            new_offsets = transformation_matrix.dot(points)
            xs = new_offsets[0, :] + x
            ys = new_offsets[1, :] + y
            hs = new_offsets[2, :] + height

            verts = [list(zip(xs, ys, hs))]
            collection = Poly3DCollection(verts, linewidths=1, edgecolors='red', alpha=0.2, zsort='min')
            face_color = "salmon"
            collection.set_facecolor(face_color)
            ax0.add_collection3d(collection)

            # plots v
            v = V[i]
            new_heading = transformation_matrix.dot(heading)
            a = Arrow3D([x, x + new_heading[0, 0] * v], [y, y + new_heading[1, 0] * v], 
                [height, height + new_heading[2, 0] * v], mutation_scale=10, 
                lw=2, arrowstyle="-|>", color="r")
            ax0.add_artist(a)

        # plots sigma and eta
        ax1 = fig.add_subplot(2, 1, 2)
        ax1.plot(time, sigma, 'tab:orange')
        ax1.plot(time, eta, 'tab:green')
        ax1.legend(['Sigma', 'Eta'])

        plt.show()

    # samples lyapunov function to approximate range of landing zone
    def plot_landing_zone(self, lyapunov_control_function):
        pass


# unit tests when running graphics.py directly
if __name__ == "__main__":

    test_x_traj = np.zeros((8, 30))
    test_x_traj[0] = (np.sin(np.arange(30) / 1.6) * 0.15) + 0.2
    test_x_traj[1] = (1.0 - np.arange(30) / 29.0) * 0.5
    test_x_traj[2] = np.arange(30) * np.pi * 4.5 / 29.0
    test_x_traj[3] = np.cos(np.arange(30) / 2.0)
    test_x_traj[4] = np.sin(np.arange(30) / 2.1)
    test_x_traj[5] = 1.0 - (np.arange(30) / 29.0)
    test_x_traj[6] = np.sin(np.arange(30) / 4.0)
    test_x_traj[7] = np.cos(np.arange(30) / 4.0)
    visualizer = Visualizer(x_traj=test_x_traj)
