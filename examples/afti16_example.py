import numpy as np

from pimpc import Model


def build_afti16(ts=0.05):
    As = np.array(
        [
            [-0.0151, -60.5651, 0.0, -32.174],
            [-0.0001, -1.3411, 0.9929, 0.0],
            [0.00018, 43.2541, -0.86939, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    Bs = np.array(
        [
            [-2.516, -13.136],
            [-0.1689, -0.2514],
            [-17.251, -1.5766],
            [0.0, 0.0],
        ]
    )

    nx, nu = 4, 2
    try:
        from scipy.linalg import expm
    except ImportError as exc:
        raise RuntimeError("SciPy is required for the AFTI-16 example.") from exc

    M = expm(np.block([[As, Bs], [np.zeros((nu, nx)), np.zeros((nu, nu))]]) * ts)
    A = M[:nx, :nx]
    B = M[:nx, nx:]

    C = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    return A, B, C


def simulate_closed_loop(model, A, B, C, x0, u0, yref_traj, uref, w, nsim):
    nx, nu = B.shape
    ny = C.shape[0]

    x_hist = np.zeros((nx, nsim + 1))
    y_hist = np.zeros((ny, nsim + 1))
    u_hist = np.zeros((nu, nsim))
    iter_hist = np.zeros(nsim, dtype=int)

    x_hist[:, 0] = x0
    y_hist[:, 0] = C @ x0
    x_current = x0.copy()
    u_prev = u0.copy()

    print("Running closed-loop simulation...")

    for k in range(nsim):
        yref = yref_traj[:, k]
        results = model.solve(x_current, u_prev, yref, uref, w, verbose=False)

        u_apply = results.u[:, 0]
        x_next = A @ x_current + B @ u_apply
        y_next = C @ x_next

        u_hist[:, k] = u_apply
        x_hist[:, k + 1] = x_next
        y_hist[:, k + 1] = y_next
        iter_hist[k] = results.info.iterations

        x_current = x_next
        u_prev = u_apply

    print("Simulation completed.")
    print("-" * 40)
    print(f"  Total steps: {nsim}")
    print(f"  Average iterations: {np.mean(iter_hist):.1f}")

    return x_hist, y_hist, u_hist, iter_hist


def main():
    print("=" * 50)
    print("  PiMPC - AFTI-16 Closed-Loop Simulation")
    print("=" * 50)

    ts = 0.05
    A, B, C = build_afti16(ts)
    nx, nu, ny = 4, 2, 2

    Np = 5
    model = Model()
    model.setup(
        A=A,
        B=B,
        C=C,
        Np=Np,
        Wy=100.0 * np.eye(ny),
        Wu=0.0 * np.eye(nu),
        Wdu=0.1 * np.eye(nu),
        umin=-25.0 * np.ones(nu),
        umax=25.0 * np.ones(nu),
        xmin=np.array([-np.inf, -0.5, -np.inf, -100.0]),
        xmax=np.array([np.inf, 0.5, np.inf, 100.0]),
        rho=1.0,
        tol=1e-6,
        maxiter=10000,
        precond=True,
        accel=True,
        device="cpu",
    )

    x0 = np.zeros(nx)
    u0 = np.zeros(nu)
    nsim = 200

    yref_traj = np.vstack([np.zeros(nsim), np.concatenate([10.0 * np.ones(100), np.zeros(100)])])
    uref = np.zeros(nu)
    w = np.zeros(nx)

    x_hist, y_hist, u_hist, iter_hist = simulate_closed_loop(
        model, A, B, C, x0, u0, yref_traj, uref, w, nsim
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot results.") from exc

    t_y = np.arange(nsim + 1) * ts
    t_u = np.arange(nsim) * ts

    fig, axes = plt.subplots(5, 1, figsize=(8, 8), sharex=False)

    axes[0].plot(t_y, np.concatenate([yref_traj[1, :], [0.0]]), label="Reference", linestyle="--", color="black")
    axes[0].plot(t_y, y_hist[1, :], label="Pitch Angle", color="red")
    axes[0].set_ylabel("y2 (deg)")
    axes[0].legend(loc="upper right")

    axes[1].plot(t_y, y_hist[0, :], color="red")
    axes[1].set_ylabel("y1 (deg)")

    axes[2].plot(t_u, u_hist[0, :], color="blue")
    axes[2].set_ylabel("u1 (deg)")

    axes[3].plot(t_u, u_hist[1, :], color="blue")
    axes[3].set_ylabel("u2 (deg)")

    axes[4].plot(np.arange(nsim) + 1, iter_hist, color="green")
    axes[4].set_xlabel("Step")
    axes[4].set_ylabel("Iterations")

    plt.tight_layout()
    plt.savefig("example_afti16.png")
    plt.show()


if __name__ == "__main__":
    main()
