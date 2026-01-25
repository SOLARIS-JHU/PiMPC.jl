import time
import warnings

import numpy as np

from .results import Results, SolveInfo
from .utils import ensure_cupy


def _prepare_w_bar(w, nx, nu, Np, dtype):
    w_arr = np.asarray(w, dtype=dtype)
    if w_arr.ndim == 1:
        if w_arr.shape[0] != nx:
            raise ValueError("w must have shape (nx,) or (nx, Np).")
        w_bar = np.concatenate([w_arr.reshape(-1), np.zeros(nu, dtype=dtype)])
    elif w_arr.ndim == 2:
        if w_arr.shape != (nx, Np):
            raise ValueError("w must have shape (nx,) or (nx, Np).")
        w_bar = np.vstack([w_arr, np.zeros((nu, Np), dtype=dtype)])
    else:
        raise ValueError("w must have shape (nx,) or (nx, Np).")
    return w_bar


def solve(model, x0, u0, yref, uref, w, *, verbose=False):
    if not model.is_setup:
        raise RuntimeError("Model not setup. Call setup() first.")

    use_gpu = model.device == "gpu"
    if use_gpu and model.is_time_varying:
        warnings.warn("Time-varying models are not supported on GPU, falling back to CPU.", RuntimeWarning, stacklevel=2)
        use_gpu = False

    if use_gpu:
        try:
            x, u, du, info, warm = _solve_gpu(
                model, x0, u0, yref, uref, w, warm_vars=model.warm_vars, verbose=verbose
            )
        except RuntimeError:
            warnings.warn("GPU solver unavailable, falling back to CPU.", RuntimeWarning, stacklevel=2)
            x, u, du, info, warm = _solve_cpu(
                model, x0, u0, yref, uref, w, warm_vars=model.warm_vars, verbose=verbose
            )
    else:
        x, u, du, info, warm = _solve_cpu(
            model, x0, u0, yref, uref, w, warm_vars=model.warm_vars, verbose=verbose
        )

    model.warm_vars = warm
    return Results(x, u, du, info)


def _solve_cpu(model, x0, u0, yref, uref, w, *, warm_vars=None, verbose=False):
    nx, nu, ny, Np = model.nx, model.nu, model.ny, model.Np
    nx_bar = nx + nu
    dtype = model.dtype
    time_varying = model.is_time_varying

    if time_varying:
        if model.precond:
            raise ValueError("Preconditioning is not supported for time-varying models.")
        A_bar = np.zeros((Np, nx_bar, nx_bar), dtype=dtype)
        B_bar = np.zeros((Np, nx_bar, nu), dtype=dtype)
        for k in range(Np):
            A_bar[k, :nx, :nx] = model.A[k]
            A_bar[k, :nx, nx:] = model.B[k]
            A_bar[k, nx:, nx:] = np.eye(nu, dtype=dtype)
            B_bar[k, :nx, :] = model.B[k]
            B_bar[k, nx:, :] = np.eye(nu, dtype=dtype)
    else:
        A_bar = np.block([[model.A, model.B], [np.zeros((nu, nx), dtype=dtype), np.eye(nu, dtype=dtype)]])
        B_bar = np.vstack([model.B, np.eye(nu, dtype=dtype)])
    C_bar = np.hstack([model.C, np.zeros((ny, nu), dtype=dtype)])
    e_bar = np.concatenate([model.e, np.zeros(nu, dtype=dtype)])
    w_bar = _prepare_w_bar(w, nx, nu, Np, dtype)
    xmin_bar = np.concatenate([model.xmin, model.umin])
    xmax_bar = np.concatenate([model.xmax, model.umax])

    rho, tol, max_iter, eta = model.rho, model.tol, model.maxiter, model.eta

    if model.precond:
        E = np.sqrt(np.diag(A_bar.T @ A_bar))
        E_inv = 1.0 / E
        A_bar = (E[:, None] * A_bar) * E_inv[None, :]
        B_bar = E[:, None] * B_bar
        C_bar = C_bar * E_inv[None, :]
        e_bar = E * e_bar
        if w_bar.ndim == 1:
            w_bar = E * w_bar
        else:
            w_bar = E[:, None] * w_bar
        xmin_bar = E * xmin_bar
        xmax_bar = E * xmax_bar
    else:
        E = np.ones(nx_bar, dtype=dtype)
        E_inv = np.ones(nx_bar, dtype=dtype)

    C_part = C_bar[:, :nx]
    Q_bar = np.block(
        [
            [C_part.T @ model.Wy @ C_part, np.zeros((nx, nu), dtype=dtype)],
            [np.zeros((nu, nx), dtype=dtype), model.Wu],
        ]
    )
    Q_bar_N = np.block(
        [
            [C_part.T @ model.Wf @ C_part, np.zeros((nx, nu), dtype=dtype)],
            [np.zeros((nu, nx), dtype=dtype), model.Wu],
        ]
    )
    q_bar = np.concatenate([C_part.T @ model.Wy @ np.asarray(yref, dtype=dtype), model.Wu @ np.asarray(uref, dtype=dtype)])
    q_bar_N = np.concatenate([C_part.T @ model.Wf @ np.asarray(yref, dtype=dtype), model.Wu @ np.asarray(uref, dtype=dtype)])
    R_bar = model.Wdu

    if time_varying:
        J_B = np.empty((Np, nu, nx_bar), dtype=dtype)
        H_A = np.empty((Np, nx_bar, nx_bar), dtype=dtype)
        for k in range(Np):
            J_B[k] = np.linalg.solve(R_bar + rho * (B_bar[k].T @ B_bar[k]), B_bar[k].T)
            H_A[k] = np.linalg.solve(
                Q_bar + rho * np.eye(nx_bar, dtype=dtype) + rho * (A_bar[k].T @ A_bar[k]),
                np.eye(nx_bar, dtype=dtype),
            )
    else:
        J_B = np.linalg.solve(R_bar + rho * (B_bar.T @ B_bar), B_bar.T)
        H_A = np.linalg.solve(
            Q_bar + rho * np.eye(nx_bar, dtype=dtype) + rho * (A_bar.T @ A_bar), np.eye(nx_bar, dtype=dtype)
        )
    H_AN = np.linalg.solve(Q_bar_N + rho * np.eye(nx_bar, dtype=dtype), np.eye(nx_bar, dtype=dtype))

    w_term = w_bar[:, None] if w_bar.ndim == 1 else w_bar
    w_term_mid = w_term if w_bar.ndim == 1 else w_term[:, 1:Np]

    x_bar = np.concatenate([np.asarray(x0, dtype=dtype).reshape(-1), np.asarray(u0, dtype=dtype).reshape(-1)])

    if warm_vars is None:
        DU = np.zeros((nu, Np), dtype=dtype)
        X = np.zeros((nx_bar, Np + 1), dtype=dtype)
        X[:, 0] = E * x_bar
        V = np.zeros((nx_bar, Np), dtype=dtype)
        Z = np.zeros((nx_bar, Np), dtype=dtype)
        Theta = np.zeros((nx_bar, Np), dtype=dtype)
        Beta = np.zeros((nx_bar, Np), dtype=dtype)
        Lambda = np.zeros((nx_bar, Np), dtype=dtype)
    else:
        DU0, X0, V0, Z0, Theta0, Beta0, Lambda0 = warm_vars
        DU = np.hstack([DU0[:, 1:], DU0[:, -1:]]).astype(dtype, copy=False)
        X = np.hstack([(E * x_bar)[:, None], X0[:, 2:], X0[:, -1:]]).astype(dtype, copy=False)
        V = np.hstack([V0[:, 1:], V0[:, -1:]]).astype(dtype, copy=False)
        Z = np.hstack([Z0[:, 1:], Z0[:, -1:]]).astype(dtype, copy=False)
        Theta = np.hstack([Theta0[:, 1:], Theta0[:, -1:]]).astype(dtype, copy=False)
        Beta = np.hstack([Beta0[:, 1:], Beta0[:, -1:]]).astype(dtype, copy=False)
        Lambda = np.hstack([Lambda0[:, 1:], Lambda0[:, -1:]]).astype(dtype, copy=False)
        X[:, 1:] = E[:, None] * X[:, 1:]

    if model.accel:
        V_hat = V.copy()
        Z_hat = Z.copy()
        Theta_hat = Theta.copy()
        Beta_hat = Beta.copy()
        Lambda_hat = Lambda.copy()

    V_prev = V.copy()
    Z_prev = Z.copy()
    Theta_prev = Theta.copy()
    Beta_prev = Beta.copy()
    Lambda_prev = Lambda.copy()

    residuals = []
    alpha_prev = 1.0
    res_prev = np.inf
    res = np.inf
    converged = False

    if verbose:
        print("PiMPC ADMM Solver (CPU)")
        print("-" * 40)
        print(f"{'Iter':>6}  {'Residual':>12}")
        print("-" * 40)

    t_start = time.perf_counter()
    for iter_idx in range(1, max_iter + 1):
        V_prev[:] = V
        Z_prev[:] = Z
        Theta_prev[:] = Theta
        Beta_prev[:] = Beta
        Lambda_prev[:] = Lambda

        if model.accel:
            if time_varying:
                for k in range(Np):
                    DU[:, k] = J_B[k] @ (V_hat[:, k] - Beta_hat[:, k])
                if Np > 1:
                    for k in range(1, Np):
                        w_k = w_bar if w_bar.ndim == 1 else w_bar[:, k]
                        term = Z_hat[:, k] - V_hat[:, k] + Lambda_hat[:, k] - e_bar - w_k
                        X[:, k] = H_A[k] @ (
                            q_bar + rho * (Z_hat[:, k - 1] - Theta_hat[:, k - 1] + A_bar[k].T @ term)
                        )
                X[:, Np] = H_AN @ (q_bar_N + rho * (Z_hat[:, Np - 1] - Theta_hat[:, Np - 1]))
                BU = np.empty((nx_bar, Np), dtype=dtype)
                AX = np.empty((nx_bar, Np), dtype=dtype)
                for k in range(Np):
                    BU[:, k] = B_bar[k] @ DU[:, k]
                    AX[:, k] = A_bar[k] @ X[:, k]
            else:
                DU = J_B @ (V_hat - Beta_hat)
                if Np > 1:
                    X[:, 1:Np] = H_A @ (
                        q_bar[:, None]
                        + rho
                        * (
                            Z_hat[:, 0 : Np - 1]
                            - Theta_hat[:, 0 : Np - 1]
                            + A_bar.T
                            @ (Z_hat[:, 1:Np] - V_hat[:, 1:Np] + Lambda_hat[:, 1:Np] - e_bar[:, None] - w_term_mid)
                        )
                    )
                X[:, Np] = H_AN @ (q_bar_N + rho * (Z_hat[:, Np - 1] - Theta_hat[:, Np - 1]))
                BU = B_bar @ DU
                AX = A_bar @ X[:, 0:Np]
            Z = (2.0 * (X[:, 1 : Np + 1] + Theta_hat) + BU + Beta_hat + AX + e_bar[:, None] + w_term - Lambda_hat) / 3.0
            Z = np.minimum(np.maximum(Z, xmin_bar[:, None]), xmax_bar[:, None])
            V = 0.5 * (Z + BU + Beta_hat - AX - e_bar[:, None] - w_term + Lambda_hat)
            Theta = Theta_hat + X[:, 1 : Np + 1] - Z
            Beta = Beta_hat + BU - V
            Lambda = Lambda_hat + Z - AX - V - e_bar[:, None] - w_term

            diff_theta = Theta - Theta_hat
            diff_beta = Beta - Beta_hat
            diff_lambda = Lambda - Lambda_hat
            diff_z = Z - Z_hat
            diff_v = V - V_hat
            diff_zv = (Z - V) - (Z_hat - V_hat)
        else:
            if time_varying:
                for k in range(Np):
                    DU[:, k] = J_B[k] @ (V[:, k] - Beta[:, k])
                if Np > 1:
                    for k in range(1, Np):
                        w_k = w_bar if w_bar.ndim == 1 else w_bar[:, k]
                        term = Z[:, k] - V[:, k] + Lambda[:, k] - e_bar - w_k
                        X[:, k] = H_A[k] @ (q_bar + rho * (Z[:, k - 1] - Theta[:, k - 1] + A_bar[k].T @ term))
                X[:, Np] = H_AN @ (q_bar_N + rho * (Z[:, Np - 1] - Theta[:, Np - 1]))
                BU = np.empty((nx_bar, Np), dtype=dtype)
                AX = np.empty((nx_bar, Np), dtype=dtype)
                for k in range(Np):
                    BU[:, k] = B_bar[k] @ DU[:, k]
                    AX[:, k] = A_bar[k] @ X[:, k]
            else:
                DU = J_B @ (V - Beta)
                if Np > 1:
                    X[:, 1:Np] = H_A @ (
                        q_bar[:, None]
                        + rho
                        * (
                            Z[:, 0 : Np - 1]
                            - Theta[:, 0 : Np - 1]
                            + A_bar.T @ (Z[:, 1:Np] - V[:, 1:Np] + Lambda[:, 1:Np] - e_bar[:, None] - w_term_mid)
                        )
                    )
                X[:, Np] = H_AN @ (q_bar_N + rho * (Z[:, Np - 1] - Theta[:, Np - 1]))
                BU = B_bar @ DU
                AX = A_bar @ X[:, 0:Np]
            Z = (2.0 * (X[:, 1 : Np + 1] + Theta) + BU + Beta + AX + e_bar[:, None] + w_term - Lambda) / 3.0
            Z = np.minimum(np.maximum(Z, xmin_bar[:, None]), xmax_bar[:, None])
            V = 0.5 * (Z + BU + Beta - AX - e_bar[:, None] - w_term + Lambda)
            Theta = Theta + X[:, 1 : Np + 1] - Z
            Beta = Beta + BU - V
            Lambda = Lambda + Z - AX - V - e_bar[:, None] - w_term

            diff_theta = Theta - Theta_prev
            diff_beta = Beta - Beta_prev
            diff_lambda = Lambda - Lambda_prev
            diff_z = Z - Z_prev
            diff_v = V - V_prev
            diff_zv = (Z - V) - (Z_prev - V_prev)

        res = rho * np.sum(
            np.linalg.norm(diff_theta, axis=0) ** 2
            + np.linalg.norm(diff_beta, axis=0) ** 2
            + np.linalg.norm(diff_lambda, axis=0) ** 2
            + np.linalg.norm(diff_z, axis=0) ** 2
            + np.linalg.norm(diff_v, axis=0) ** 2
            + np.linalg.norm(diff_zv, axis=0) ** 2
        )

        residuals.append(float(res))
        if verbose:
            print(f"{iter_idx:6d}  {res:12.4e}")

        if res < tol:
            converged = True
            break

        if model.accel:
            if res < eta * res_prev:
                alpha = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * alpha_prev**2))
                momentum = (alpha_prev - 1.0) / alpha
                V_hat = V + momentum * (V - V_prev)
                Z_hat = Z + momentum * (Z - Z_prev)
                Theta_hat = Theta + momentum * (Theta - Theta_prev)
                Beta_hat = Beta + momentum * (Beta - Beta_prev)
                Lambda_hat = Lambda + momentum * (Lambda - Lambda_prev)
                res_prev = res
            else:
                alpha = 1.0
                V_hat[:] = V
                Z_hat[:] = Z
                Theta_hat[:] = Theta
                Beta_hat[:] = Beta
                Lambda_hat[:] = Lambda
                res_prev = res_prev / eta
            alpha_prev = alpha

    solve_time = time.perf_counter() - t_start

    if verbose:
        print("-" * 40)
        print(f"  Status: {'Converged' if converged else 'Not converged'}")
        print(f"  Iterations: {len(residuals)}")
        print(f"  Time: {solve_time * 1000:.4f} ms")
        print()

    X = E_inv[:, None] * X
    x_traj = X[:nx, :]
    u_traj = X[nx:, 1:]

    info = SolveInfo(
        solve_time=solve_time,
        iterations=len(residuals),
        converged=converged,
        obj_val=float("inf") if not residuals else residuals[-1],
    )
    warm = (DU, X, V, Z, Theta, Beta, Lambda)

    return x_traj, u_traj, DU, info, warm


def _solve_gpu(model, x0, u0, yref, uref, w, *, warm_vars=None, verbose=False):
    cp = ensure_cupy()

    nx, nu, ny, Np = model.nx, model.nu, model.ny, model.Np
    nx_bar = nx + nu
    dtype = cp.float32

    if model.is_time_varying:
        raise RuntimeError("Time-varying models are not supported on GPU.")

    A_bar = cp.asarray(
        np.block([[model.A, model.B], [np.zeros((nu, nx), dtype=model.dtype), np.eye(nu, dtype=model.dtype)]]),
        dtype=dtype,
    )
    B_bar = cp.asarray(np.vstack([model.B, np.eye(nu, dtype=model.dtype)]), dtype=dtype)
    C_bar = cp.asarray(np.hstack([model.C, np.zeros((ny, nu), dtype=model.dtype)]), dtype=dtype)
    e_bar = cp.asarray(np.concatenate([model.e, np.zeros(nu, dtype=model.dtype)]), dtype=dtype)
    w_bar = cp.asarray(_prepare_w_bar(w, nx, nu, Np, model.dtype), dtype=dtype)
    xmin_bar = cp.asarray(np.concatenate([model.xmin, model.umin]), dtype=dtype)
    xmax_bar = cp.asarray(np.concatenate([model.xmax, model.umax]), dtype=dtype)

    rho = dtype(model.rho)
    tol = dtype(model.tol)
    max_iter = model.maxiter
    eta = dtype(model.eta)

    if model.precond:
        E = cp.sqrt(cp.diag(A_bar.T @ A_bar))
        E_inv = dtype(1.0) / E
        A_bar = (E[:, None] * A_bar) * E_inv[None, :]
        B_bar = E[:, None] * B_bar
        C_bar = C_bar * E_inv[None, :]
        e_bar = E * e_bar
        if w_bar.ndim == 1:
            w_bar = E * w_bar
        else:
            w_bar = E[:, None] * w_bar
        xmin_bar = E * xmin_bar
        xmax_bar = E * xmax_bar
    else:
        E = cp.ones(nx_bar, dtype=dtype)
        E_inv = cp.ones(nx_bar, dtype=dtype)

    C_part = C_bar[:, :nx]
    Q_bar = cp.block(
        [
            [C_part.T @ cp.asarray(model.Wy, dtype=dtype) @ C_part, cp.zeros((nx, nu), dtype=dtype)],
            [cp.zeros((nu, nx), dtype=dtype), cp.asarray(model.Wu, dtype=dtype)],
        ]
    )
    Q_bar_N = cp.block(
        [
            [C_part.T @ cp.asarray(model.Wf, dtype=dtype) @ C_part, cp.zeros((nx, nu), dtype=dtype)],
            [cp.zeros((nu, nx), dtype=dtype), cp.asarray(model.Wu, dtype=dtype)],
        ]
    )
    q_bar = cp.concatenate(
        [
            C_part.T @ cp.asarray(model.Wy, dtype=dtype) @ cp.asarray(yref, dtype=dtype),
            cp.asarray(model.Wu, dtype=dtype) @ cp.asarray(uref, dtype=dtype),
        ]
    )
    q_bar_N = cp.concatenate(
        [
            C_part.T @ cp.asarray(model.Wf, dtype=dtype) @ cp.asarray(yref, dtype=dtype),
            cp.asarray(model.Wu, dtype=dtype) @ cp.asarray(uref, dtype=dtype),
        ]
    )
    R_bar = cp.asarray(model.Wdu, dtype=dtype)

    J_B = cp.linalg.solve(R_bar + rho * (B_bar.T @ B_bar), B_bar.T)
    H_A = cp.linalg.solve(Q_bar + rho * cp.eye(nx_bar, dtype=dtype) + rho * (A_bar.T @ A_bar), cp.eye(nx_bar, dtype=dtype))
    H_AN = cp.linalg.solve(Q_bar_N + rho * cp.eye(nx_bar, dtype=dtype), cp.eye(nx_bar, dtype=dtype))

    w_term = w_bar[:, None] if w_bar.ndim == 1 else w_bar
    w_term_mid = w_term if w_bar.ndim == 1 else w_term[:, 1:Np]

    x_bar = cp.asarray(np.concatenate([np.asarray(x0, dtype=model.dtype).reshape(-1), np.asarray(u0, dtype=model.dtype).reshape(-1)]), dtype=dtype)

    if warm_vars is None:
        DU = cp.zeros((nu, Np), dtype=dtype)
        X = cp.zeros((nx_bar, Np + 1), dtype=dtype)
        X[:, 0] = E * x_bar
        V = cp.zeros((nx_bar, Np), dtype=dtype)
        Z = cp.zeros((nx_bar, Np), dtype=dtype)
        Theta = cp.zeros((nx_bar, Np), dtype=dtype)
        Beta = cp.zeros((nx_bar, Np), dtype=dtype)
        Lambda = cp.zeros((nx_bar, Np), dtype=dtype)
    else:
        DU0, X0, V0, Z0, Theta0, Beta0, Lambda0 = warm_vars
        DU0 = cp.asarray(DU0, dtype=dtype)
        X0 = cp.asarray(X0, dtype=dtype)
        V0 = cp.asarray(V0, dtype=dtype)
        Z0 = cp.asarray(Z0, dtype=dtype)
        Theta0 = cp.asarray(Theta0, dtype=dtype)
        Beta0 = cp.asarray(Beta0, dtype=dtype)
        Lambda0 = cp.asarray(Lambda0, dtype=dtype)
        DU = cp.hstack([DU0[:, 1:], DU0[:, -1:]])
        X = cp.hstack([(E * x_bar)[:, None], X0[:, 2:], X0[:, -1:]])
        V = cp.hstack([V0[:, 1:], V0[:, -1:]])
        Z = cp.hstack([Z0[:, 1:], Z0[:, -1:]])
        Theta = cp.hstack([Theta0[:, 1:], Theta0[:, -1:]])
        Beta = cp.hstack([Beta0[:, 1:], Beta0[:, -1:]])
        Lambda = cp.hstack([Lambda0[:, 1:], Lambda0[:, -1:]])
        X[:, 1:] = E[:, None] * X[:, 1:]

    if model.accel:
        V_hat = V.copy()
        Z_hat = Z.copy()
        Theta_hat = Theta.copy()
        Beta_hat = Beta.copy()
        Lambda_hat = Lambda.copy()

    V_prev = V.copy()
    Z_prev = Z.copy()
    Theta_prev = Theta.copy()
    Beta_prev = Beta.copy()
    Lambda_prev = Lambda.copy()

    residuals = []
    alpha_prev = dtype(1.0)
    res_prev = dtype(1.0)
    res = dtype(np.inf)
    converged = False

    if verbose:
        print("PiMPC ADMM Solver (GPU)")
        print("-" * 40)
        print(f"{'Iter':>6}  {'Residual':>12}")
        print("-" * 40)

    t_start = time.perf_counter()
    for iter_idx in range(1, max_iter + 1):
        V_prev[:] = V
        Z_prev[:] = Z
        Theta_prev[:] = Theta
        Beta_prev[:] = Beta
        Lambda_prev[:] = Lambda

        if model.accel:
            DU = J_B @ (V_hat - Beta_hat)
            if Np > 1:
                X[:, 1:Np] = H_A @ (
                    q_bar[:, None]
                    + rho
                    * (
                        Z_hat[:, 0 : Np - 1]
                        - Theta_hat[:, 0 : Np - 1]
                        + A_bar.T
                        @ (Z_hat[:, 1:Np] - V_hat[:, 1:Np] + Lambda_hat[:, 1:Np] - e_bar[:, None] - w_term_mid)
                    )
                )
            X[:, Np] = H_AN @ (q_bar_N + rho * (Z_hat[:, Np - 1] - Theta_hat[:, Np - 1]))
            BU = B_bar @ DU
            AX = A_bar @ X[:, 0:Np]
            Z = (dtype(2.0) * (X[:, 1 : Np + 1] + Theta_hat) + BU + Beta_hat + AX + e_bar[:, None] + w_term - Lambda_hat) / dtype(3.0)
            Z = cp.minimum(cp.maximum(Z, xmin_bar[:, None]), xmax_bar[:, None])
            V = dtype(0.5) * (Z + BU + Beta_hat - AX - e_bar[:, None] - w_term + Lambda_hat)
            Theta = Theta_hat + X[:, 1 : Np + 1] - Z
            Beta = Beta_hat + BU - V
            Lambda = Lambda_hat + Z - AX - V - e_bar[:, None] - w_term

            diff_theta = Theta - Theta_hat
            diff_beta = Beta - Beta_hat
            diff_lambda = Lambda - Lambda_hat
            diff_z = Z - Z_hat
            diff_v = V - V_hat
            diff_zv = (Z - V) - (Z_hat - V_hat)
        else:
            DU = J_B @ (V - Beta)
            if Np > 1:
                X[:, 1:Np] = H_A @ (
                    q_bar[:, None]
                    + rho
                    * (
                        Z[:, 0 : Np - 1]
                        - Theta[:, 0 : Np - 1]
                        + A_bar.T @ (Z[:, 1:Np] - V[:, 1:Np] + Lambda[:, 1:Np] - e_bar[:, None] - w_term_mid)
                    )
                )
            X[:, Np] = H_AN @ (q_bar_N + rho * (Z[:, Np - 1] - Theta[:, Np - 1]))
            BU = B_bar @ DU
            AX = A_bar @ X[:, 0:Np]
            Z = (dtype(2.0) * (X[:, 1 : Np + 1] + Theta) + BU + Beta + AX + e_bar[:, None] + w_term - Lambda) / dtype(3.0)
            Z = cp.minimum(cp.maximum(Z, xmin_bar[:, None]), xmax_bar[:, None])
            V = dtype(0.5) * (Z + BU + Beta - AX - e_bar[:, None] - w_term + Lambda)
            Theta = Theta + X[:, 1 : Np + 1] - Z
            Beta = Beta + BU - V
            Lambda = Lambda + Z - AX - V - e_bar[:, None] - w_term

            diff_theta = Theta - Theta_prev
            diff_beta = Beta - Beta_prev
            diff_lambda = Lambda - Lambda_prev
            diff_z = Z - Z_prev
            diff_v = V - V_prev
            diff_zv = (Z - V) - (Z_prev - V_prev)

        res = rho * cp.sum(
            cp.linalg.norm(diff_theta, axis=0) ** 2
            + cp.linalg.norm(diff_beta, axis=0) ** 2
            + cp.linalg.norm(diff_lambda, axis=0) ** 2
            + cp.linalg.norm(diff_z, axis=0) ** 2
            + cp.linalg.norm(diff_v, axis=0) ** 2
            + cp.linalg.norm(diff_zv, axis=0) ** 2
        )

        residuals.append(float(res))
        if verbose:
            print(f"{iter_idx:6d}  {float(res):12.4e}")

        if res < tol:
            converged = True
            break

        if model.accel:
            if res < eta * res_prev:
                alpha = dtype(0.5) * (dtype(1.0) + cp.sqrt(dtype(1.0) + dtype(4.0) * alpha_prev**2))
                momentum = (alpha_prev - dtype(1.0)) / alpha
                V_hat = V + momentum * (V - V_prev)
                Z_hat = Z + momentum * (Z - Z_prev)
                Theta_hat = Theta + momentum * (Theta - Theta_prev)
                Beta_hat = Beta + momentum * (Beta - Beta_prev)
                Lambda_hat = Lambda + momentum * (Lambda - Lambda_prev)
                res_prev = res
            else:
                alpha = dtype(1.0)
                V_hat[:] = V
                Z_hat[:] = Z
                Theta_hat[:] = Theta
                Beta_hat[:] = Beta
                Lambda_hat[:] = Lambda
                res_prev = res / eta
            alpha_prev = alpha

    solve_time = time.perf_counter() - t_start

    if verbose:
        print("-" * 40)
        print(f"  Status: {'Converged' if converged else 'Not converged'}")
        print(f"  Iterations: {len(residuals)}")
        print(f"  Time: {solve_time * 1000:.4f} ms")
        print()

    DU = cp.asnumpy(DU)
    X = cp.asnumpy(E_inv[:, None] * X)
    V = cp.asnumpy(V)
    Z = cp.asnumpy(Z)
    Theta = cp.asnumpy(Theta)
    Beta = cp.asnumpy(Beta)
    Lambda = cp.asnumpy(Lambda)

    x_traj = X[:nx, :]
    u_traj = X[nx:, 1:]

    info = SolveInfo(
        solve_time=solve_time,
        iterations=len(residuals),
        converged=converged,
        obj_val=float("inf") if not residuals else residuals[-1],
    )
    warm = (DU, X, V, Z, Theta, Beta, Lambda)

    return x_traj, u_traj, DU, info, warm
