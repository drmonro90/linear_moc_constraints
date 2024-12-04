import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt

from control import StateSpace
import control

from control.matlab import *

from mpc_preparations import LinearMpcController

def main():
    A = np.array([[0, 1], [0,0]])
    As = np.kron(np.eye(2,dtype=int),A)
    B = np.array([[0], [1]])
    Bs = np.block([
        [B,    np.zeros_like(B)],
        [np.zeros_like(B), B]]
    )
    C = np.array([[1, 0]])
    Cs =  np.block([
        [C, np.zeros_like(C)],
        [np.zeros_like(C), C]
    ])
    D = np.array([[0,]])
    Ds = np.zeros((2,2))
    model = StateSpace(As,Bs,Cs,Ds)
    discr = control.matlab.c2d(model,0.1)


    Ad, Bd = discr.A, discr.B
    [nx, nu] = Bd.shape
    umin, umax = 2*[-10], 2*[10]
    xmin, xmax = 4*[-2], 4*[2]  

    # Objective function
    Q = sparse.diags(
        [10.0, 0.0,10.0, 0.0]
    )
    QN = Q
    R = 0.1 * sparse.eye(2)

    # Initial and reference states
    x0 = np.zeros(4)
    xr = np.array([1.0, 0.0,1.0, 0.0])

    # Prediction horizon
    N = 10

    P = sparse.block_diag(
        [sparse.kron(sparse.eye(N), Q), QN, sparse.kron(sparse.eye(N), R)], format="csc"
    )
    # - linear objective

    q = np.hstack([np.kron(np.ones(N), -Q @ xr), -QN @ xr, np.zeros(N * nu)])

    nsim = 100

    controller = LinearMpcController(Ad, Bd, nx, nu, umin, umax, xmin, xmax, x0, P, q, N)
    ctrls_ = []
    states_ = []
    for i in range(nsim):
        ctrl = controller.calc_new_control(x0)
        x0 = Ad @ x0 + Bd @ ctrl
        ctrls_.append(ctrl)
        states_.append(x0)
    plt.plot(ctrls_)
    plt.plot(states_)
    plt.show()
    for i, val in enumerate(ctrls_):
        assert np.allclose(
            ctrls_[i], expected_ctrls_[i]
        ), f"not close {ctrls_[i]} and {expected_ctrls_[i]}"



if __name__ == "__main__":
    main()