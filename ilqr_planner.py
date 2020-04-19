from dynamics import discrete_dynamics
import numpy as np
from sympy import lambdify

from constants import n_x, n_u


class ILQRPlanner():

    def __init__(self, f, lx, lu, lxx, lux, luu, fx, fu):
        self.xs = xs
        self.u = u
        self.f = f
        self.const_dict = const_dict

        self.fx = self.f.jacobian(self.xs)
        self.fu = self.f.jacobian(self.u)

        self.func_f = lambdify((self.xs.tolist(), self.u.tolist()), self.f.subs(
            self.const_dict), modules='numpy')

        self.func_fx = lambdify(
            (self.xs.tolist(), self.u.tolist()), self.fx.subs(self.const_dict), modules='numpy')

        self.func_fu = lambdify(
            (self.xs.tolist(), self.u.tolist()), self.fu.subs(self.const_dict), modules='numpy')

    def _discrete_dynamics(self, x, u, dt):
        return self.func_f(x, u) * dt + x

    def _ilqr_rollout(self, x0, u_trj):
        x_trj = np.zeros((u_trj.shape[0]+1, x0.shape[0]))
        x_trj[0] = x0
        for i in range(u_trj.shape[0]):
            x_trj[i+1] = discrete_dynamics(x_trj[i], u_trj[i])
        return x_trj

    def _ilqr_forward_pass(self, x_trj, u_trj, k_trj, K_trj):
        x_trj_new = np.zeros(x_trj.shape)
        x_trj_new[0, :] = x_trj[0, :]
        u_trj_new = np.zeros(u_trj.shape)
        for n in range(u_trj.shape[0]):
            u_trj_new[n, :] = u_trj[n, :] + k_trj[n] + \
                K_trj[n].dot(x_trj_new[n] - x_trj[n])
            x_trj_new[n+1, :] = discrete_dynamics(x_trj_new[n], u_trj_new[n])
        return x_trj_new, u_trj_new

    def _ilqr_backward_pass_quadratic_cost(self, x_trj, u_trj, Q, Qf, R, regu):
        k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
        K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
        expected_cost_redu = 0
        # TODO: Set terminal boundary condition here (V_x, V_xx)
        V_x = np.zeros((x_trj.shape[1],))
        V_xx = np.zeros((x_trj.shape[1], x_trj.shape[1]))
        for n in range(u_trj.shape[0]-1, -1, -1):
            # TODO: First compute derivatives, then the Q-terms
            l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = self._get_derivatives_ilqr_quadratic_cost(
                x_trj[n], u_trj[n], Q, Qf, R)
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._ilqr_Q_terms(
                l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
            # We add regularization to ensure that Q_uu is invertible and nicely conditioned
            Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
            k, K = self._ilqr_gains(Q_uu_regu, Q_u, Q_ux)
            k_trj[n, :] = k
            K_trj[n, :, :] = K
            V_x, V_xx = self._ilqr_V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
            expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)
        return k_trj, K_trj, expected_cost_redu

    def _ilqr_Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)
        Q_ux = l_ux + f_u.T.dot(V_xx).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx).dot(f_u)
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def _ilqr_V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
        V_x = Q_x - K.T.dot(Q_uu).dot(k)
        V_xx = Q_xx - K.T.dot(Q_uu).dot(K)
        return V_x, V_xx

    def _ilqr_gains(Q_uu, Q_u, Q_ux):
        Q_uu_inv = np.linalg.inv(Q_uu)
        k = -Q_uu_inv.dot(Q_u)
        K = -Q_uu_inv.dot(Q_ux)
        return k, K

    def _get_derivatives_ilqr_quadratic_cost(self, x, u, Q, Qf, R):
        fx = self.func_fx(x, u)
        fu = self.func_fu(x, u)
        lx = x.T@Q
        lu = u.T@R
        lxx = Q
        lux = np.zeros((x.shape(0), u.shape(1)))
        luu = R

        return lx, lu, lxx, lux, luu, fx, fu

    def plan_ilqr_trajectory_quadratic_cost(self, x0, Q, Qf, R, N, max_iter=50, regu_init=100):
        # First forward rollout
        # First forward rollout
        u_trj = np.random.randn(N-1, n_u)*0.0001
        x_trj = rollout(x0, u_trj)
        total_cost = cost_trj(x_trj, u_trj)
        regu = regu_init
        max_regu = 10000
        min_regu = 0.01

        # Setup traces
        cost_trace = [total_cost]
        expected_cost_redu_trace = []
        redu_ratio_trace = [1]
        redu_trace = []
        regu_trace = [regu]

        # Run main loop
        for it in range(max_iter):
            # Backward and forward pass
            k_trj, K_trj, expected_cost_redu = self._ilqr_backward_pass_quadratic_cost(
                x_trj, u_trj, Q, Qf, R, regu)
            x_trj_new, u_trj_new = self._ilqr_forward_pass(
                x_trj, u_trj, k_trj, K_trj)
            # Evaluate new trajectory
            total_cost = cost_trj(x_trj_new, u_trj_new)
            cost_redu = cost_trace[-1] - total_cost
            redu_ratio = cost_redu / abs(expected_cost_redu)
            # Accept or reject iteration
            if cost_redu > 0:
                # Improvement! Accept new trajectories and lower regularization
                redu_ratio_trace.append(redu_ratio)
                cost_trace.append(total_cost)
                x_trj = x_trj_new
                u_trj = u_trj_new
                regu *= 0.7
            else:
                # Reject new trajectories and increase regularization
                regu *= 2.0
                cost_trace.append(cost_trace[-1])
                redu_ratio_trace.append(0)
            regu = min(max(regu, min_regu), max_regu)
            regu_trace.append(regu)
            redu_trace.append(cost_redu)

            # Early termination if expected improvement is small
            if expected_cost_redu <= 1e-6:
                break

        return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace
