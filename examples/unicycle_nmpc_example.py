import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pimpc import Model


def unicycle_dynamics(x, u):
    px, py, theta = x
    v, omega = u
    return np.array([v * np.cos(theta), v * np.sin(theta), omega])


def linearize_discrete_unicycle(x, u, dt):
    theta = x[2]
    v = u[0]

    A = np.eye(3)
    A[0, 2] = -dt * v * np.sin(theta)
    A[1, 2] = dt * v * np.cos(theta)

    B = np.zeros((3, 2))
    B[0, 0] = dt * np.cos(theta)
    B[1, 0] = dt * np.sin(theta)
    B[2, 1] = dt

    x_next = x + dt * unicycle_dynamics(x, u)
    c = x_next - A @ x - B @ u
    return A, B, c


def build_ltv_matrices(x0, u_nom, dt):
    nx = x0.shape[0]
    nu, Np = u_nom.shape
    x_nom = np.zeros((nx, Np + 1))
    x_nom[:, 0] = x0

    for k in range(Np):
        x_nom[:, k + 1] = x_nom[:, k] + dt * unicycle_dynamics(x_nom[:, k], u_nom[:, k])

    A_seq = np.zeros((Np, nx, nx))
    B_seq = np.zeros((Np, nx, nu))
    c_seq = np.zeros((nx, Np))
    for k in range(Np):
        A_k, B_k, c_k = linearize_discrete_unicycle(x_nom[:, k], u_nom[:, k], dt)
        A_seq[k] = A_k
        B_seq[k] = B_k
        c_seq[:, k] = c_k

    return A_seq, B_seq, c_seq


def main():
    dt = 0.1
    Np = 25
    nsim = 60

    nx, nu = 3, 2
    x0 = np.array([0.0, 0.0, 0.0])
    u_prev = np.zeros(nu)
    u_nom = np.zeros((nu, Np))

    yref = np.array([2.0, 2.0, 0.0])
    uref = np.zeros(nu)

    model = Model()

    x_hist = np.zeros((nx, nsim + 1))
    u_hist = np.zeros((nu, nsim))
    x_hist[:, 0] = x0
    x_current = x0.copy()

    print("=" * 50)
    print("  PiMPC - Unicycle NMPC (LTV Linearization)")
    print("=" * 50)

    for k in range(nsim):
        A_seq, B_seq, c_seq = build_ltv_matrices(x_current, u_nom, dt)

        model.setup(
            A=A_seq,
            B=B_seq,
            C=np.eye(nx),
            Np=Np,
            Wy=np.diag([20.0, 20.0, 0.0]),
            Wu=0.1 * np.eye(nu),
            Wdu=0.1 * np.eye(nu),
            umin=np.array([0.0, -1.0]),
            umax=np.array([1.5, 1.0]),
            rho=1.0,
            tol=1e-5,
            maxiter=200,
            precond=False,
            accel=True,
            device="cpu",
        )

        results = model.solve(x_current, u_prev, yref, uref, c_seq, verbose=False)
        u_apply = results.u[:, 0]

        x_next = x_current + dt * unicycle_dynamics(x_current, u_apply)
        x_hist[:, k + 1] = x_next
        u_hist[:, k] = u_apply

        x_current = x_next
        u_prev = u_apply

        u_nom = np.hstack([results.u[:, 1:], results.u[:, -1:]])

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot results.") from exc

    t = np.arange(nsim + 1) * dt
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=False)

    axes[0].plot(x_hist[0, :], x_hist[1, :], color="blue", label="Trajectory")
    axes[0].scatter([yref[0]], [yref[1]], color="red", label="Goal")
    axes[0].set_ylabel("y (m)")
    axes[0].set_xlabel("x (m)")
    axes[0].legend(loc="best")
    axes[0].set_title("Unicycle Path")

    axes[1].plot(t[:-1], u_hist[0, :], color="green")
    axes[1].set_ylabel("v (m/s)")

    axes[2].plot(t[:-1], u_hist[1, :], color="purple")
    axes[2].set_ylabel("omega (rad/s)")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
