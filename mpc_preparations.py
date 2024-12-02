import osqp
import numpy as np
import scipy as sp
from scipy import sparse


def cast_problem_to_qp(Ad, N, nx, nu, Bd, x0, xmin, xmax, umin, umax):


    # - linear dynamics
    Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) + sparse.kron(
        sparse.eye(N + 1, k=-1), Ad
    )
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
    Aeq = sparse.hstack([Ax, Bu])
    leq = np.hstack([-x0, np.zeros(N * nx)])
    ueq = leq
    # - input and state constraints
    Aineq = sparse.eye((N + 1) * nx + N * nu)
    lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])
    # - OSQP constraints
    A = sparse.vstack([Aeq, Aineq], format="csc")
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])
    return A, l, u


class LinearMpcController():
    def __init__(self, Ad, Bd, nx, nu, umin, umax, xmin, xmax,x0, P, q, N) -> None:
        self.Ad, self.Bd, self.nx, self.nu, self.umin, self.umax, self.xmin, self.xmax, self.N = Ad, Bd, nx, nu, umin, umax, xmin, xmax, N


        # Create an OSQP object
        self.prob = osqp.OSQP()
        self.A, self.l, self.u = cast_problem_to_qp(
            Ad, self.N, nx, nu, Bd, x0, xmin, xmax, umin, umax
        )
        # Setup workspace
        self.prob.setup(P, q, self.A, self.l, self.u)
    
    def calc_new_control(self, new_x0):
        self.l[:self.nx] = -new_x0
        self.u[:self.nx] = -new_x0
        self.prob.update(l=self.l, u=self.u)
        res = self.prob.solve()
        ctrl = res.x[-self.N * self.nu : -(self.N - 1) * self.nu]
        return ctrl