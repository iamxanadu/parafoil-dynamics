import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from celluloid import Camera
import numpy as np
import logging as lg
from math import pi

lg.basicConfig(level=lg.INFO)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Visualizer(object):

    def __init__(self, t: np.ndarray, x_traj: np.ndarray, u_traj=None, max_size_no_subsample=500, subsample_divisor=100, lyapunov_function=None, plot_heading=True, goal_pos=None, target_trajectory=None, render_animation=False, anim_path="anim.gif", ):
        self.plot_heading = plot_heading
        self.goal_pos = goal_pos
        self.anim_path = anim_path

        n = x_traj.shape[-1]
        if n > max_size_no_subsample:
            every = n // subsample_divisor
            lg.info(
                f"Trajectory is too large to display effectively. Subsampling every {every}")
        else:
            every = 1

        self.time = t[::every]
        self.V = x_traj[0, ::every]
        self.gamma = x_traj[1, ::every]
        self.psi = x_traj[2, ::every]
        self.x_pos = x_traj[3, ::every]
        self.y_pos = x_traj[4, ::every]
        self.h = x_traj[5, ::every]
        self.sigma = x_traj[6, ::every]
        self.epsilon = x_traj[7, ::every]

        if u_traj is not None:
            self.com_sigma = u_traj[0, ::every]
            self.com_epsilon = u_traj[1, ::every]
        else:
            self.com_sigma = None
            self.com_epsilon = None

        if not render_animation:
            self.plot_state_trajectory()
            '''
            TODO Implement after wind model done
            elif lyapunov_function is not None:
                self.plot_landing_zone(lyapunov_function)
            '''
        else:

            fig = plt.figure()
            camera = Camera(fig)

            ax0 = fig.add_subplot(1, 1, 1, projection='3d')
            ax0.set_xlabel('X')
            ax0.set_ylabel('Y')
            ax0.set_zlabel('Altitude')

            # plots two points to enforce isometric grid
            x_min = np.amin(self.x_pos)
            y_min = np.amin(self.y_pos)
            h_min = np.amin(self.h)
            x_max = np.amax(self.x_pos)
            y_max = np.amin(self.y_pos)
            h_max = np.amin(self.h)

            center = np.array([np.sum(self.x_pos)/self.x_pos.shape[0],
                               np.sum(self.y_pos)/self.y_pos.shape[0], np.sum(self.h)/self.h.shape[0]])
            width = np.amax([x_max - x_min, y_max - y_min, h_max - h_min])
            bounds = np.stack([center - width/2, center + width/2], axis=0)
            ax0.scatter(bounds[:, 0], bounds[:, 1],
                        bounds[:, 2], marker='.', color='white')

            # renders animation frames
            parafoil_scale = ((np.amax(self.x_pos) - np.amin(self.x_pos)) +
                              (np.amax(self.y_pos) - np.amin(self.y_pos))) / 4
            for i in range(self.V.shape[-1]):
                self.animate(i, ax0, camera, parafoil_scale, self.V, self.gamma,
                             self.psi, self.x_pos, self.y_pos, self.h, self.sigma, self.epsilon, target_trajectory)

            anim = camera.animate()
            anim.save(self.anim_path, writer='imagemagick', fps=30)

    ### private methods ###

    # animation iteration
    def animate(self, i, ax, camera, parafoil_scale, V, gamma, psi, x_pos, y_pos, h, sigma, epsilon, target_trajectory=None):

        # adds parafoil model to render
        parafoil_width = 0.3 * 0.5 * parafoil_scale
        parafoil_depth = 0.15 * 0.5 * parafoil_scale
        x = x_pos[i]
        y = y_pos[i]
        height = h[i]

        # modifies xs, ys, and hs depending on gamma, psi, and sigma(?)
        beta = gamma[i]
        alpha = psi[i] + (np.pi)
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
            [np.cos(alpha)*np.cos(beta), -np.sin(alpha),
             np.cos(alpha)*np.sin(beta)],
            [np.sin(alpha)*np.cos(beta), np.cos(alpha),
             np.sin(alpha)*np.sin(beta)],
            [-np.sin(beta), 0, np.cos(beta)],
        ])
        new_offsets = transformation_matrix.dot(points)
        xs = new_offsets[0, :] + x
        ys = new_offsets[1, :] + y
        hs = new_offsets[2, :] + height

        verts = [list(zip(xs, ys, hs))]
        collection = Poly3DCollection(
            verts, linewidths=1, edgecolors='red', alpha=0.2, zsort='min')
        face_color = "salmon"
        collection.set_facecolor(face_color)
        ax.add_collection3d(collection)

        # saves frame
        camera.snap()
        lg.info(f" * Finished rendering frame {i}")

    ### public methods ###

    # plots the state of the parafoil/rocket over time
    def plot_state_trajectory(self, target_trajectory=None):

        fig = plt.figure(figsize=plt.figaspect(1))
        fig.suptitle('State Trajectory')

        ax0 = fig.add_subplot(2, 2, 1, projection='3d')
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Altitude')
        ax0.plot3D(self.x_pos, self.y_pos, self.h, 'blue')
        ax0.scatter(self.x_pos, self.y_pos, self.h, marker='o')

        # plots two points to enforce isometric grid
        x_min = np.amin(self.x_pos)
        y_min = np.amin(self.y_pos)
        h_min = np.amin(self.h)
        x_max = np.amax(self.x_pos)
        y_max = np.amin(self.y_pos)
        h_max = np.amin(self.h)

        center = np.array([np.sum(self.x_pos)/self.x_pos.shape[0],
                           np.sum(self.y_pos)/self.y_pos.shape[0], np.sum(self.h)/self.h.shape[0]])
        width = np.amax([x_max - x_min, y_max - y_min, h_max - h_min])
        bounds = np.stack([center - width/2, center + width/2], axis=0)
        ax0.scatter(bounds[:, 0], bounds[:, 1],
                    bounds[:, 2], marker='.', color='white')

        # TODO Add support for graphing a goal manifold

        # plots vectors
        parafoil_scale = ((np.amax(self.x_pos) - np.amin(self.x_pos)) +
                          (np.amax(self.y_pos) - np.amin(self.y_pos))) / 4

        for i in range(self.V.shape[-1]):

            # plots simple parafoil
            parafoil_width = 0.3 * 0.5 * parafoil_scale
            parafoil_depth = 0.15 * 0.5 * parafoil_scale
            x = self.x_pos[i]
            y = self.y_pos[i]
            height = self.h[i]

            # modifies xs, ys, and hs depending on gamma, psi, and sigma(?)
            beta = self.gamma[i]
            alpha = self.psi[i] + (np.pi)
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
                [np.cos(alpha)*np.cos(beta), -np.sin(alpha),
                 np.cos(alpha)*np.sin(beta)],
                [np.sin(alpha)*np.cos(beta), np.cos(alpha),
                 np.sin(alpha)*np.sin(beta)],
                [-np.sin(beta), 0, np.cos(beta)],
            ])
            new_offsets = transformation_matrix.dot(points)
            xs = new_offsets[0, :] + x
            ys = new_offsets[1, :] + y
            hs = new_offsets[2, :] + height

            verts = [list(zip(xs, ys, hs))]
            collection = Poly3DCollection(
                verts, linewidths=1, edgecolors='red', alpha=0.2, zsort='min')
            face_color = "salmon"
            collection.set_facecolor(face_color)
            ax0.add_collection3d(collection)

            # plots v
            if self.plot_heading:
                v = self.V[i]
                new_heading = transformation_matrix.dot(heading)
                a = Arrow3D([x, x + new_heading[0, 0] * v], [y, y + new_heading[1, 0] * v],
                            [height, height + new_heading[2, 0] * v], mutation_scale=10,
                            lw=2, arrowstyle="-|>", color="r")
                ax0.add_artist(a)

        # plots sigma and eta
        ax1 = fig.add_subplot(2, 2, 2)
        ax1.plot(self.time, self.sigma, 'tab:orange')
        ax1.plot(self.time, self.epsilon, 'tab:green')
        ax1.legend(['Sigma', 'Epsilon'])
        ax1.set_xlabel('Time (s)')

        if self.com_sigma is not None and self.com_epsilon is not None:
            ax1.plot(self.time, self.com_sigma, 'tab:cyan')
            ax1.plot(self.time, self.com_epsilon, 'tab:red')
            ax1.legend(
                ['Sigma', 'Epsilon', 'Commanded Sigma', 'Commanded Epsilon'])

        ax2 = fig.add_subplot(2, 2, 3)
        ax2.plot(self.time, self.V, 'tab:orange')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('m/s')
        ax2.set_title('Velocity')
        ax2.legend(['Velocity'])

        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(self.time, self.gamma, 'tab:cyan')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('rad')
        ax3.set_title('Flight Path Angle')
        ax3.legend(['Flight Path Angle'])

        plt.show()
