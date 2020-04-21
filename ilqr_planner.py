import numpy as np


class ILQRPlanner():

    def __init__(self, nx, nu, f, lf, lfx, lfxx, l, lx, lu, lxx, lux, luu, fx, fu, dt):
        self.f = f
        self.l = l

        self.lf = lf
        self.lfx = lfx
        self.lfxx = lfxx

        self.lx = lx
        self.lu = lu
        self.lxx = lxx
        self.lux = lux
        self.luu = luu
        self.fx = fx
        self.fu = fu

        self.dt = dt

        self.nx = nx
        self.nu = nu

    def _discrete_dynamics(self, x, u, dt):
        r = self.f(x, u)
        return r * dt + x

    def _ilqr_rollout(self, x0, u_trj):
        x_trj = np.zeros((u_trj.shape[0]+1, self.nx, 1))
        x_trj[0] = x0
        for i in range(u_trj.shape[0]):
            x_trj[i+1] = self._discrete_dynamics(x_trj[i], u_trj[i], self.dt)
        return x_trj

    def _ilqr_cost_trj(self, x_trj, u_trj):
        total = 0.0
        for i in range(u_trj.shape[0]):
            total += self.l(x_trj[i], u_trj[i])
        total += self.lf(x_trj[-1])
        return total

    def _ilqr_forward_pass(self, x_trj, u_trj, k_trj, K_trj):
        x_trj_new = np.zeros(x_trj.shape)
        x_trj_new[0, :] = x_trj[0, :]
        u_trj_new = np.zeros(u_trj.shape)
        for n in range(u_trj.shape[0]):
            u_trj_new[n, :] = u_trj[n, :] + k_trj[n] + \
                K_trj[n].dot(x_trj_new[n] - x_trj[n])
            x_trj_new[n+1, :] = self._discrete_dynamics(
                x_trj_new[n], u_trj_new[n], self.dt)
        return x_trj_new, u_trj_new

    def _ilqr_backward_pass(self, x_trj, u_trj, regu):
        k_trj = np.zeros(u_trj.shape)
        K_trj = np.zeros((u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]))
        expected_cost_redu = 0
        V_x = self.lfx(x_trj[-1])
        V_xx = self.lfxx(x_trj[-1])
        for n in range(u_trj.shape[0]-1, -1, -1):
            l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = self._get_derivatives_ilqr(
                x_trj[n], u_trj[n])
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._ilqr_Q_terms(
                l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
            # We add regularization to ensure that Q_uu is invertible and nicely conditioned
            Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
            k, K = self._ilqr_gains(Q_uu_regu, Q_u, Q_ux)
            k_trj[n, :] = k
            K_trj[n, :, :] = K
            V_x, V_xx = self._ilqr_V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
            expected_cost_redu += self._ilqr_expected_cost_reduction(
                Q_u, Q_uu, k)
        return k_trj, K_trj, expected_cost_redu

    def _ilqr_Q_terms(self, l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)
        Q_ux = l_ux + f_u.T.dot(V_xx).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx).dot(f_u)
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def _ilqr_V_terms(self, Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
        V_x = Q_x - K.T.dot(Q_uu).dot(k)
        V_xx = Q_xx - K.T.dot(Q_uu).dot(K)
        return V_x, V_xx

    def _ilqr_gains(self, Q_uu, Q_u, Q_ux):
        Q_uu_inv = np.linalg.inv(Q_uu)
        k = -Q_uu_inv.dot(Q_u)
        K = -Q_uu_inv.dot(Q_ux)
        return k, K

    def _get_derivatives_ilqr(self, x, u):
        fx = self.fx(x, u)
        fu = self.fu(x, u)
        lx = self.lx(x, u)
        lu = self.lu(x, u)
        lxx = self.lxx(x, u)
        lux = self.lux(x, u)
        luu = self.luu(x, u)

        return lx, lu, lxx, lux, luu, fx, fu

    def _ilqr_expected_cost_reduction(self, Q_u, Q_uu, k):
        return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))

    def plan_ilqr_trajectory(self, x0, N, max_iter=50, regu_init=100):
        # First forward rollout
        u_trj = np.random.randn(N-1, self.nu, 1)*0.0001
        x_trj = self._ilqr_rollout(x0, u_trj)
        total_cost = self._ilqr_cost_trj(x_trj, u_trj)
        regu = regu_init
        max_regu = 10000
        min_regu = 0.01

        # Setup traces
        cost_trace = [total_cost]
        redu_ratio_trace = [1]
        redu_trace = []
        regu_trace = [regu]

        # Run main loop
        for it in range(max_iter):
            # Backward and forward pass
            k_trj, K_trj, expected_cost_redu = self._ilqr_backward_pass(
                x_trj, u_trj, regu)
            x_trj_new, u_trj_new = self._ilqr_forward_pass(
                x_trj, u_trj, k_trj, K_trj)
            # Evaluate new trajectory
            total_cost = self._ilqr_cost_trj(x_trj_new, u_trj_new)
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


if __name__ == "__main__":
    from sympy import Symbol, Function, Matrix, MatrixSymbol
    from sympy import sin, cos
    from sympy import pprint, init_printing
    from sympy import simplify, lambdify

    from math import pi

    import constants

    from graphics import Visualizer

    init_printing()

    t = Symbol('t')

    m = Symbol('m')
    g = Symbol('g')
    rho = Symbol('rho')  # Air density
    S = Symbol('S')  # Reference area (m^2)
    Cltrim = Symbol('Clt')
    Cdtrim = Symbol('Cdt')
    dCl = Symbol('dCl')
    dCd = Symbol('dCd')

    const_dict = {g: constants.g, m: constants.m, rho: constants.rho, Cltrim: constants.Cltrim,
                  Cdtrim: constants.Cdtrim, dCl: constants.delCl, dCd: constants.delCd, S: constants.S}

    v = Function('v')(t)
    psi = Function('psi')(t)
    gamma = Function('gamma')(t)
    x = Function('x')(t)
    y = Function('y')(t)
    h = Function('h')(t)

    sigma = Function('sigma')(t)
    epsilon = Function('epsilon')(t)

    W = m * g
    L = 1/2 * rho * S * v**2 * (Cltrim + dCl * epsilon)
    D = 1/2 * rho * S * v**2 * (Cdtrim + dCd * epsilon)

    vdot = -(D + W*sin(gamma)) / m
    gammadot = (L * cos(sigma) - W * cos(gamma)) / (m*v)
    psidot = L*sin(sigma) / (m*v*cos(gamma))
    xdot = v*cos(gamma)*cos(psi)
    ydot = v*cos(gamma)*sin(psi)
    hdot = v*sin(gamma)

    # Calculate the derivatives
    xsl = [v, gamma, psi, x, y, h]
    ul = [sigma, epsilon]
    xs = Matrix(xsl)
    u = Matrix(ul)
    f = Matrix([vdot, gammadot, psidot, xdot, ydot, hdot])

    lamfd = lambdify([xsl, ul],
                     f.subs(const_dict), modules='numpy')
    lamfx = lambdify([xsl, ul], f.jacobian(
        xs).subs(const_dict), modules='numpy')
    lamfu = lambdify([xsl, ul], f.jacobian(
        u).subs(const_dict), modules='numpy')

    def fd(x, u):
        return lamfd(x.T[0], u.T[0])

    def fx(x, u):
        return lamfx(x.T[0], u.T[0])

    def fu(x, u):
        return lamfu(x.T[0], u.T[0])

    x0 = np.array([[2, -pi/6, 0, 300, 200, 100]]).T
    #u = np.array([0, 0])

    xr = np.array([[5, 0, -0.5, 0, 0, 0]]).T
    Q = np.diag([1, 1, 1, 10, 10, 10])
    R = np.diag([1, 2])
    Qf = np.diag([0.1, 0.1, 0.1, 10, 10, 10])

    def ls(x, u):
        return (x.T - xr.T).dot(Q).dot(x - xr) + u.T.dot(R).dot(u)

    def lf(x):
        return (x.T - xr.T).dot(Qf).dot(x - xr)

    def lfx(x):
        return Qf.dot(x - xr) + Qf.T.dot(x - xr)

    def lfxx(x):
        return Qf + Qf.T

    def lx(x, u):
        return Q.dot(x - xr) + Q.T.dot(x - xr)

    def lu(x, u):
        return R.dot(u) + R.T.dot(u)

    def lux(x, u):
        return np.zeros((u.shape[0], x.shape[0]))

    def lxx(x, u):
        return Q + Q.T

    def luu(x, u):
        return R + R.T

    ilqrp = ILQRPlanner(6, 2, fd, lf, lfx, lfxx, ls, lx,
                        lu, lxx, lux, luu, fx, fu, 0.1)

    x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = ilqrp.plan_ilqr_trajectory(x0, 100)
    u_trj = np.concatenate((u_trj, np.zeros((1, 2, 1))), axis=0)
    full_trj = np.concatenate((x_trj, u_trj), axis=1)

    viz = Visualizer(x_traj=np.transpose(np.squeeze(full_trj)))