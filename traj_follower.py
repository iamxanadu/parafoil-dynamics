import numpy as np
import random
import csv
import os
import sys
from dynamics import *
from graphics import *


target_trajectory_path = "trajectories/LongWayOut.csv"
n_divisions = 50


def randomly_offset_state(state): # TODO: add randomization for other state variables
    new_state = np.copy(state)
    new_state[3] = new_state[3] + ((random.random() - 0.5) * 20.0)
    new_state[4] = new_state[4] + ((random.random() - 0.5) * 20.0)
    return new_state

def subdivide_trajectory(trajectory, divisions=50):
    new_traj_states = []

    for i in range(trajectory.shape[0]-1):
        prev_state = trajectory[i]
        curr_state = trajectory[i+1]

        # interpolates new states
        for j in range(divisions):
            alpha = j / divisions
            state = (prev_state * (1.0 - alpha)) + (curr_state * alpha)
            new_traj_states.append(state)

    return np.stack(new_traj_states, axis=0)

def parse_trajectory(path, divisions=None):

    # TODO: remove dummy trajectory
    # traj = np.zeros((1000, 8))
    # traj[:, 5] = (1.0-(np.arange(1000)/999.0))*35.0
    # traj[:, 3] = np.sin(np.arange(1000)/150.0)
    # traj[:, 4] = np.cos(np.arange(1000)/150.0)
    # return traj

    if os.path.exists(path):

        # loads trajectory from csv file
        if path[-3:] == "csv":
            traj_states = []
            time_delta_sum = 0.0
            last_t = None
            file = open(path, "r")
            for line in file:

                # builds trajectory
                components = line.strip('\n').split(',')
                t = float(components[0]) # isolates t
                state = np.zeros(len(components)-1)
                for i in range(len(components)-1):
                    state[i] = float(components[i+1])
                traj_states.append(state)

                # tracks average t_delta
                if last_t is not None:
                    time_delta_sum += t - last_t
                last_t = t

            t_delta = time_delta_sum / (len(traj_states) - 1)
            traj = np.stack(traj_states, axis=0)
            if divisions is not None:
                traj = subdivide_trajectory(traj, divisions)
                t_delta = t_delta / divisions
            return (traj, t_delta)

        # loads trajectory from npy file
        elif path[-3:] == "npy":
            traj = np.load(path)
            if divisions is not None:
                traj = subdivide_trajectory(traj, divisions)
            return (traj, None)

    return (np.zeros((1, 8)), None)

def simulate_actual_trajectory(target_trajectory, start_state, t_delta=None):
    k_d = 0.025
    k_pa = 0.035
    k_po = 0.01

    if t_delta is None:
        t_delta = 0.005

    current_x = start_state
    current_u = np.array([0.0, 0.0])
    trajectory_states = []
    n = target_trajectory.shape[0]
    for i in range(n):

        # implements PD controller from paper (p. 80)
        e_1 = current_x[3] - target_trajectory[i, 3]
        e_2 = current_x[4] - target_trajectory[i, 4]
        e_3 = current_x[2] - target_trajectory[i, 2]

        e_atrack = (e_1*np.cos(target_trajectory[i, 2])) + (e_2*np.sin(target_trajectory[i, 2]))
        e_xtrack = (e_1*np.sin(target_trajectory[i, 2])) - (e_2*np.cos(target_trajectory[i, 2]))

        V = current_x[0]
        gamma = current_x[1]
        psi_star_dot = target_trajectory[i, 2] - target_trajectory[max(0, i-1), 2]
        e_xtrack_dot = (-V * np.cos(gamma) * np.sin(e_3)) + ((e_1 * np.cos(target_trajectory[i, 2])) + (e_2 * np.sin(target_trajectory[i, 2])) * psi_star_dot)

        sig_base = psi_star_dot # TODO: may be incorrect
        current_u[0] = sig_base + (k_pa * e_xtrack) + (k_d * e_xtrack_dot)
        current_u[1] = k_po * e_atrack

        # simulates dynamics and saves state to trajectory
        current_x = np.copy(discrete_simulation_dynamics(current_x, current_u, t_delta))
        trajectory_states.append(np.copy(current_x))

        print(" * Finished simulating state "+str(i)+" of "+str(n))

    return np.stack(trajectory_states, axis=0)

if __name__ == "__main__":

    # parses arguments
    arguments = sys.argv
    render_animation = False
    if 'anim' in arguments[-1]:
        render_animation = True

    # loads & parses target trajectory
    target_trajectory, suggested_t_delta = parse_trajectory(target_trajectory_path, divisions=n_divisions)
    print(" !! Using suggested t_delta: "+str(suggested_t_delta))

    # initializes initial state
    # x_current = np.array([0.1, 0.0, 0.0, -1.0, 1.0, 4.5, 0.0, 0.0]) # starts with x_0
    x_current = randomly_offset_state(target_trajectory[0])

    # implements PID controller to follow target trajectory
    actual_trajectory = simulate_actual_trajectory(target_trajectory, x_current, suggested_t_delta)

    # plots actual trajectory or renders animation
    visualizer = Visualizer(x_traj=actual_trajectory.transpose(), plot_heading=False, target_trajectory=target_trajectory.transpose(), render_animation=render_animation)

