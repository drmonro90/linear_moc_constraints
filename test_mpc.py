import unittest
import osqp
import numpy as np
import scipy as sp
from scipy import sparse

from mpc_preparations import cast_problem_to_qp


def get_quadrocopter_system_variables():
    Ad = sparse.csc_matrix(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0488, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0016, 0.0, 0.0, 0.0992, 0.0, 0.0],
            [0.0, -0.0488, 0.0, 0.0, 1.0, 0.0, 0.0, -0.0016, 0.0, 0.0, 0.0992, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0992],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.9734, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0488, 0.0, 0.0, 0.9846, 0.0, 0.0],
            [0.0, -0.9734, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0488, 0.0, 0.0, 0.9846, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9846],
        ]
    )
    Bd = sparse.csc_matrix(
        [
            [0.0, -0.0726, 0.0, 0.0726],
            [-0.0726, 0.0, 0.0726, 0.0],
            [-0.0152, 0.0152, -0.0152, 0.0152],
            [-0.0, -0.0006, -0.0, 0.0006],
            [0.0006, 0.0, -0.0006, 0.0000],
            [0.0106, 0.0106, 0.0106, 0.0106],
            [0, -1.4512, 0.0, 1.4512],
            [-1.4512, 0.0, 1.4512, 0.0],
            [-0.3049, 0.3049, -0.3049, 0.3049],
            [-0.0, -0.0236, 0.0, 0.0236],
            [0.0236, 0.0, -0.0236, 0.0],
            [0.2107, 0.2107, 0.2107, 0.2107],
        ]
    )
    [nx, nu] = Bd.shape

    # Constraints
    u0 = 10.5916
    umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
    umax = np.array([13.0, 13.0, 13.0, 13.0]) - u0
    xmin = np.array(
        [
            -np.pi / 6,
            -np.pi / 6,
            -np.inf,
            -np.inf,
            -np.inf,
            -1.0,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
        ]
    )
    xmax = np.array(
        [
            np.pi / 6,
            np.pi / 6,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ]
    )
    return Ad, Bd, nx, nu, umin, umax, xmin, xmax


class TestUnityAgent(unittest.TestCase):
    def test_base_implementation(self):

        # Discrete time model of a quadcopter

        Ad, Bd, nx, nu, umin, umax, xmin, xmax = get_quadrocopter_system_variables()

        # Objective function
        Q = sparse.diags(
            [0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
        )
        QN = Q
        R = 0.1 * sparse.eye(4)

        # Initial and reference states
        x0 = np.zeros(12)
        xr = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Prediction horizon
        N = 10

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P, q, A, l, u = cast_problem_to_qp(
            Ad, Q, QN, N, R, xr, nx, nu, Bd, x0, xmin, xmax, umin, umax
        )

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u)

        # Simulate in closed loop
        nsim = 15
        expected_ctrls_ = [
            np.array([-0.99153499, 1.74841857, -0.99153499, 1.74841857]),
            np.array([-0.99168608, 0.58148059, -0.99168608, 0.58148059]),
            np.array([-0.42591469, 0.00833044, -0.42591469, 0.00833044]),
            np.array([0.7501203, -0.7766365, 0.7501203, -0.7766365]),
            np.array([0.82922446, -0.82112962, 0.82922446, -0.82112962]),
            np.array([0.56016765, -0.55009931, 0.56016765, -0.55009931]),
            np.array([0.27258902, -0.26342178, 0.27258902, -0.26342178]),
            np.array([0.08073136, -0.07260389, 0.08073136, -0.07260389]),
            np.array([-0.01117416, 0.0183611, -0.01117416, 0.0183611]),
            np.array([-0.03711131, 0.04346485, -0.03711131, 0.04346485]),
            np.array([-0.03181643, 0.03743305, -0.03181643, 0.03743305]),
            np.array([-0.01799676, 0.02296192, -0.01799676, 0.02296192]),
            np.array([-0.00628856, 0.01067782, -0.00628856, 0.01067782]),
            np.array([0.00048928, 0.00339087, 0.00048928, 0.00339087]),
            np.array([0.00314154, 0.00028857, 0.00314154, 0.00028857]),
        ]
        ctrls_ = []
        for i in range(nsim):
            # Solve
            res = prob.solve()

            # Check solver status
            if res.info.status != "solved":
                raise ValueError("OSQP did not solve the problem!")

            # Apply first control input to the plant
            ctrl = res.x[-N * nu : -(N - 1) * nu]
            x0 = Ad @ x0 + Bd @ ctrl

            # Update initial state
            l[:nx] = -x0
            u[:nx] = -x0
            prob.update(l=l, u=u)
            ctrls_.append(ctrl)

        for i, val in enumerate(ctrls_):
            assert np.allclose(
                ctrls_[i], expected_ctrls_[i]
            ), f"not close {ctrls_[i]} and {expected_ctrls_[i]}"


if __name__ == "__main__":
    unittest.main()
