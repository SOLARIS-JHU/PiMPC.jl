"""
PiMPC - Parallel-in-Horizon MPC Solver (PyTorch)

Supports CPU and GPU (CUDA) via PyTorch, with optional batched solving
across multiple initial conditions in parallel.

Algorithm: ADMM with optional Nesterov acceleration and diagonal preconditioning.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import time
import torch


@dataclass
class Results:
    """Solve results."""
    x: torch.Tensor   # State trajectory  (nx, Np+1)  or (batch, nx, Np+1)
    u: torch.Tensor   # Input trajectory  (nu, Np)    or (batch, nu, Np)
    du: torch.Tensor  # Input increments  (nu, Np)    or (batch, nu, Np)
    solve_time: float
    iterations: int
    converged: bool
    obj_val: float


class Model:
    """
    PiMPC model. Create, configure with setup(), solve with solve().

    Example
    -------
    >>> m = Model()
    >>> m.setup(A=A, B=B, Np=20, umin=[-1.0], umax=[1.0], accel=True)
    >>> res = m.solve(x0, u0, yref, uref, w)
    >>> print(res.u[:, 0])   # optimal first input
    """

    def __init__(self):
        self.is_setup = False
        self.warm_vars = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def setup(self,
              A, B, Np: int,
              C=None, e=None,
              Wy=None, Wu=None, Wdu=None, Wf=None,
              xmin=None, xmax=None,
              umin=None, umax=None,
              dumin=None, dumax=None,
              rho: float = 1.0,
              tol: float = 1e-4,
              eta: float = 0.999,
              maxiter: int = 100,
              precond: bool = False,
              accel: bool = False,
              device: str = "cpu",
              dtype=torch.float64):
        """
        Configure the MPC problem.

        System model:  x_{k+1} = A x_k + B u_k + e
                       y_k     = C x_k

        Parameters
        ----------
        A, B        : system matrices
        Np          : prediction horizon
        C           : output matrix  (default: identity nx×nx)
        e           : affine offset  (default: zeros(nx))
        Wy,Wu,Wdu,Wf: weight matrices (defaults: identity)
        xmin/xmax   : state bounds   (default: ±inf)
        umin/umax   : input bounds   (default: ±inf)
        dumin/dumax : rate bounds    (default: ±inf)
        rho         : ADMM penalty
        tol         : convergence tolerance
        eta         : Nesterov restart threshold
        maxiter     : maximum iterations
        precond     : diagonal preconditioning
        accel       : Nesterov acceleration
        device      : "cpu" or "cuda" (or "cuda:0", etc.)
        dtype       : torch.float32 or torch.float64
        """
        def _t(x, shape=None):
            """Convert array-like to tensor on device."""
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=dtype)
            return torch.tensor(x, device=device, dtype=dtype)

        A = _t(A); B = _t(B)
        nx, nu = B.shape

        self.A = A
        self.B = B
        self.C = _t(C) if C is not None else torch.eye(nx, device=device, dtype=dtype)
        self.e = _t(e) if e is not None else torch.zeros(nx, device=device, dtype=dtype)

        ny = self.C.shape[0]
        self.nx, self.nu, self.ny, self.Np = nx, nu, ny, Np

        self.Wy  = _t(Wy)  if Wy  is not None else torch.eye(ny, device=device, dtype=dtype)
        self.Wu  = _t(Wu)  if Wu  is not None else torch.eye(nu, device=device, dtype=dtype)
        self.Wdu = _t(Wdu) if Wdu is not None else torch.eye(nu, device=device, dtype=dtype)
        self.Wf  = _t(Wf)  if Wf  is not None else self.Wy.clone()

        inf = float('inf')
        self.xmin = _t(xmin) if xmin is not None else torch.full((nx,), -inf, device=device, dtype=dtype)
        self.xmax = _t(xmax) if xmax is not None else torch.full((nx,),  inf, device=device, dtype=dtype)
        self.umin = _t(umin) if umin is not None else torch.full((nu,), -inf, device=device, dtype=dtype)
        self.umax = _t(umax) if umax is not None else torch.full((nu,),  inf, device=device, dtype=dtype)
        self.dumin = _t(dumin) if dumin is not None else torch.full((nu,), -inf, device=device, dtype=dtype)
        self.dumax = _t(dumax) if dumax is not None else torch.full((nu,),  inf, device=device, dtype=dtype)

        self.rho = rho
        self.tol = tol
        self.eta = eta
        self.maxiter = maxiter
        self.precond = precond
        self.accel = accel
        self.device = device
        self.dtype = dtype

        self.is_setup = True
        self.warm_vars = None

    # ------------------------------------------------------------------
    # solve  (single initial condition)
    # ------------------------------------------------------------------
    def solve(self, x0, u0, yref, uref, w=None, verbose: bool = False) -> Results:
        """
        Solve MPC for a single initial condition.

        Parameters
        ----------
        x0   : current state (nx,)
        u0   : previous input (nu,)
        yref : output reference (ny,)
        uref : input reference (nu,)
        w    : known disturbance (nx,), default zeros
        """
        assert self.is_setup, "Call setup() first."

        def _t(v):
            if v is None:
                return torch.zeros(self.nx, device=self.device, dtype=self.dtype)
            if isinstance(v, torch.Tensor):
                return v.to(device=self.device, dtype=self.dtype)
            return torch.tensor(v, device=self.device, dtype=self.dtype)

        x0   = _t(x0)
        u0   = _t(u0)
        yref = _t(yref)
        uref = _t(uref)
        w    = _t(w)

        x, u, du, info, warm = _solve(self, x0, u0, yref, uref, w,
                                      warm_vars=self.warm_vars,
                                      verbose=verbose)
        self.warm_vars = warm
        return Results(x=x, u=u, du=du, **info)

    # ------------------------------------------------------------------
    # solve_batch  (multiple initial conditions in parallel)
    # ------------------------------------------------------------------
    def solve_batch(self, x0, u0, yref, uref, w=None,
                    umin_steps=None, umax_steps=None,
                    verbose: bool = False) -> Results:
        """
        Solve MPC for a batch of initial conditions simultaneously.

        Parameters
        ----------
        x0         : (batch, nx)
        u0         : (batch, nu)
        yref       : (batch, ny)  or  (batch, ny, Np)  — constant or per-step output ref
        uref       : (batch, nu)  or  (batch, nu, Np)  — constant or per-step input ref
        w          : (batch, nx)  or  (nx,)  [broadcast], default zeros
        umin_steps : (batch, nu, Np) per-step lower bounds on u, or None
        umax_steps : (batch, nu, Np) per-step upper bounds on u, or None
                     Overrides the constant umin/umax from setup() for the Z-projection.
        """
        assert self.is_setup, "Call setup() first."

        def _t(v, default_shape):
            if v is None:
                return torch.zeros(default_shape, device=self.device, dtype=self.dtype)
            if isinstance(v, torch.Tensor):
                return v.to(device=self.device, dtype=self.dtype)
            return torch.tensor(v, device=self.device, dtype=self.dtype)

        x0   = _t(x0,   (1, self.nx))
        u0   = _t(u0,   (1, self.nu))
        yref = _t(yref, (1, self.ny))
        uref = _t(uref, (1, self.nu))
        w    = _t(w,    (1, self.nx))

        if x0.dim() == 1: x0 = x0.unsqueeze(0)
        batch = x0.shape[0]
        if u0.dim()   == 1: u0   = u0.unsqueeze(0).expand(batch, -1)
        # yref/uref: keep 3-D (B, n, Np) if provided, else broadcast 1-D/2-D
        if yref.dim() == 1: yref = yref.unsqueeze(0).expand(batch, -1)
        if uref.dim() == 1: uref = uref.unsqueeze(0).expand(batch, -1)
        if w.dim()    == 1: w    = w.unsqueeze(0).expand(batch, -1)

        if umin_steps is not None:
            umin_steps = umin_steps.to(device=self.device, dtype=self.dtype)
        if umax_steps is not None:
            umax_steps = umax_steps.to(device=self.device, dtype=self.dtype)

        x, u, du, info, _ = _solve_batch(self, x0, u0, yref, uref, w,
                                          umin_steps=umin_steps,
                                          umax_steps=umax_steps,
                                          verbose=verbose)
        return Results(x=x, u=u, du=du, **info)


# ======================================================================
# Internal solver  (single)
# ======================================================================

def _solve(m: Model, x0, u0, yref, uref, w,
           warm_vars=None, verbose: bool = False):
    nx, nu, ny, Np = m.nx, m.nu, m.ny, m.Np
    nx_bar = nx + nu
    dev, dt = m.device, m.dtype
    rho, tol, max_iter, eta = m.rho, m.tol, m.maxiter, m.eta

    # ---- Augmented system ----
    A_bar = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
    A_bar[:nx, :nx] = m.A
    A_bar[:nx, nx:] = m.B
    A_bar[nx:, nx:] = torch.eye(nu, device=dev, dtype=dt)

    B_bar = torch.zeros(nx_bar, nu, device=dev, dtype=dt)
    B_bar[:nx, :] = m.B
    B_bar[nx:, :] = torch.eye(nu, device=dev, dtype=dt)

    C_bar = torch.zeros(ny, nx_bar, device=dev, dtype=dt)
    C_bar[:, :nx] = m.C

    e_bar = torch.cat([m.e, torch.zeros(nu, device=dev, dtype=dt)])
    w_bar = torch.cat([w,   torch.zeros(nu, device=dev, dtype=dt)])

    xmin_bar = torch.cat([m.xmin, m.umin])
    xmax_bar = torch.cat([m.xmax, m.umax])

    # ---- Preconditioning ----
    if m.precond:
        E = torch.sqrt((A_bar.T @ A_bar).diag())
        E_diag = torch.diag(E)
        E_inv  = torch.diag(1.0 / E)
        A_bar  = E_diag @ A_bar @ E_inv
        B_bar  = E_diag @ B_bar
        C_bar  = C_bar @ E_inv
        e_bar  = E_diag @ e_bar
        w_bar  = E_diag @ w_bar
        xmin_bar = E * xmin_bar
        xmax_bar = E * xmax_bar
    else:
        E_diag = torch.eye(nx_bar, device=dev, dtype=dt)
        E_inv  = torch.eye(nx_bar, device=dev, dtype=dt)

    # ---- Cost matrices ----
    C_part = C_bar[:, :nx]
    Q_bar   = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
    Q_bar[:nx, :nx] = C_part.T @ m.Wy @ C_part
    Q_bar[nx:, nx:] = m.Wu

    Q_bar_N = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
    Q_bar_N[:nx, :nx] = C_part.T @ m.Wf @ C_part
    Q_bar_N[nx:, nx:] = m.Wu

    q_bar   = torch.cat([C_part.T @ m.Wy @ yref, m.Wu @ uref])
    q_bar_N = torch.cat([C_part.T @ m.Wf @ yref, m.Wu @ uref])
    R_bar   = m.Wdu

    # ---- Precomputed inverses ----
    I_nb = torch.eye(nx_bar, device=dev, dtype=dt)
    J_B  = torch.linalg.solve(R_bar + rho * B_bar.T @ B_bar, B_bar.T)   # (nu, nx_bar)  → DU = J_B @ (V - Beta)
    H_A  = torch.linalg.inv(Q_bar   + rho * I_nb + rho * A_bar.T @ A_bar)
    H_AN = torch.linalg.inv(Q_bar_N + rho * I_nb)

    x_bar = torch.cat([x0, u0])

    # ---- Initialize ADMM variables ----
    if warm_vars is None:
        DU     = torch.zeros(nu,    Np,    device=dev, dtype=dt)
        X      = torch.zeros(nx_bar, Np+1, device=dev, dtype=dt)
        X[:, 0] = E_diag @ x_bar
        V      = torch.zeros(nx_bar, Np, device=dev, dtype=dt)
        Z      = torch.zeros(nx_bar, Np, device=dev, dtype=dt)
        Theta  = torch.zeros(nx_bar, Np, device=dev, dtype=dt)
        Beta   = torch.zeros(nx_bar, Np, device=dev, dtype=dt)
        Lambda = torch.zeros(nx_bar, Np, device=dev, dtype=dt)
    else:
        DU0, X0, V0, Z0, Theta0, Beta0, Lambda0 = warm_vars
        DU     = torch.cat([DU0[:, 1:], DU0[:, -1:]], dim=1)
        X      = torch.cat([( E_diag @ x_bar).unsqueeze(1), X0[:, 2:], X0[:, -1:]], dim=1)
        V      = torch.cat([V0[:, 1:],      V0[:, -1:]],      dim=1)
        Z      = torch.cat([Z0[:, 1:],      Z0[:, -1:]],      dim=1)
        Theta  = torch.cat([Theta0[:, 1:],  Theta0[:, -1:]],  dim=1)
        Beta   = torch.cat([Beta0[:, 1:],   Beta0[:, -1:]],   dim=1)
        Lambda = torch.cat([Lambda0[:, 1:], Lambda0[:, -1:]], dim=1)
        X[:, 1:] = E_diag @ X[:, 1:]

    # ---- Acceleration copies ----
    if m.accel:
        V_hat = V.clone();     Z_hat = Z.clone()
        Theta_hat = Theta.clone(); Beta_hat = Beta.clone(); Lambda_hat = Lambda.clone()

    Z_prev     = Z.clone();     V_prev     = V.clone()
    Theta_prev = Theta.clone(); Beta_prev  = Beta.clone(); Lambda_prev = Lambda.clone()

    residuals = []
    alpha_prev, res_prev, res = 1.0, float('inf'), float('inf')
    converged = False

    if verbose:
        print("PiMPC ADMM Solver")
        print("-" * 40)
        print(f"  {'Iter':>6}  {'Residual':>12}")
        print("-" * 40)

    t_start = time.perf_counter()
    for it in range(1, max_iter + 1):
        Z_prev.copy_(Z);       V_prev.copy_(V)
        Theta_prev.copy_(Theta); Beta_prev.copy_(Beta); Lambda_prev.copy_(Lambda)

        if m.accel:
            _V, _Z, _Theta, _Beta, _Lambda = V_hat, Z_hat, Theta_hat, Beta_hat, Lambda_hat
        else:
            _V, _Z, _Theta, _Beta, _Lambda = V, Z, Theta, Beta, Lambda

        # -- DU update --
        DU = J_B @ (_V - _Beta)

        # -- X update (interior steps 1..Np-1, 0-indexed columns 1..Np-1) --
        if Np > 1:
            rhs_mid = (q_bar.unsqueeze(1)
                       + rho * (_Z[:, :Np-1] - _Theta[:, :Np-1]
                                + A_bar.T @ (_Z[:, 1:Np] - _V[:, 1:Np]
                                             + _Lambda[:, 1:Np]
                                             - e_bar.unsqueeze(1)
                                             - w_bar.unsqueeze(1))))
            X[:, 1:Np] = H_A @ rhs_mid

        # -- X terminal (column Np) --
        X[:, Np] = H_AN @ (q_bar_N + rho * (_Z[:, Np-1] - _Theta[:, Np-1]))

        # -- BU and AX --
        BU = B_bar @ DU                     # (nx_bar, Np)
        AX = A_bar @ X[:, :Np]             # (nx_bar, Np)

        # -- Z update (clamp) --
        Z = (2.0 * (X[:, 1:Np+1] + _Theta) + BU + _Beta + AX + e_bar.unsqueeze(1) + w_bar.unsqueeze(1) - _Lambda) / 3.0
        Z = torch.clamp(Z, xmin_bar.unsqueeze(1), xmax_bar.unsqueeze(1))

        # -- V update --
        V = 0.5 * (Z + BU + _Beta - AX - e_bar.unsqueeze(1) - w_bar.unsqueeze(1) + _Lambda)

        # -- Dual updates --
        Theta  = _Theta  + X[:, 1:Np+1] - Z
        Beta   = _Beta   + BU - V
        Lambda = _Lambda + Z - AX - V - e_bar.unsqueeze(1) - w_bar.unsqueeze(1)

        # -- Residual --
        if m.accel:
            dT = Theta - _Theta;  dB = Beta - _Beta;  dL = Lambda - _Lambda
            dZ = Z - _Z;          dV = V - _V
        else:
            dT = Theta - Theta_prev;  dB = Beta - Beta_prev;  dL = Lambda - Lambda_prev
            dZ = Z - Z_prev;          dV = V - V_prev

        res = float(rho * (dT.norm()**2 + dB.norm()**2 + dL.norm()**2
                           + dZ.norm()**2 + dV.norm()**2
                           + ((dZ - dV)).norm()**2))
        residuals.append(res)

        if verbose:
            print(f"  {it:>6}  {res:>12.4e}")

        if res < tol:
            converged = True
            break

        # -- Nesterov update --
        if m.accel:
            if res < eta * res_prev:
                alpha = 0.5 * (1.0 + (1.0 + 4.0 * alpha_prev**2)**0.5)
                mom = (alpha_prev - 1.0) / alpha
                V_hat     = V     + mom * (V     - V_prev)
                Z_hat     = Z     + mom * (Z     - Z_prev)
                Theta_hat = Theta + mom * (Theta - Theta_prev)
                Beta_hat  = Beta  + mom * (Beta  - Beta_prev)
                Lambda_hat = Lambda + mom * (Lambda - Lambda_prev)
                res_prev = res
            else:
                alpha = 1.0
                V_hat.copy_(V); Z_hat.copy_(Z)
                Theta_hat.copy_(Theta); Beta_hat.copy_(Beta); Lambda_hat.copy_(Lambda)
                res_prev = res_prev / eta
            alpha_prev = alpha

    solve_time = time.perf_counter() - t_start

    if verbose:
        print("-" * 40)
        print(f"  Status:     {'Converged' if converged else 'Not converged'}")
        print(f"  Iterations: {len(residuals)}")
        print(f"  Time:       {solve_time*1000:.4f} ms")

    # ---- Extract results ----
    X = E_inv @ X
    x_traj = X[:nx, :]
    u_traj = X[nx:, 1:]

    info = dict(solve_time=solve_time,
                iterations=len(residuals),
                converged=converged,
                obj_val=residuals[-1] if residuals else float('inf'))
    warm = (DU, X, V, Z, Theta, Beta, Lambda)
    return x_traj, u_traj, DU, info, warm


# ======================================================================
# Internal solver  (batched)
# ======================================================================

def _solve_batch(m: Model, x0, u0, yref, uref, w,
                 umin_steps=None, umax_steps=None,
                 warm_vars=None, verbose: bool = False):
    """
    Batched ADMM.  All tensors have a leading batch dimension B.

    x0   : (B, nx)
    u0   : (B, nu)
    yref : (B, ny)
    uref : (B, nu)
    w    : (B, nx)

    Variables layout:
      X      : (B, nx_bar, Np+1)
      DU     : (B, nu,    Np)
      V,Z    : (B, nx_bar, Np)
      Theta, Beta, Lambda : (B, nx_bar, Np)
    """
    nx, nu, ny, Np = m.nx, m.nu, m.ny, m.Np
    nx_bar = nx + nu
    dev, dt = m.device, m.dtype
    rho, tol, max_iter, eta = m.rho, m.tol, m.maxiter, m.eta
    B = x0.shape[0]

    # ---- Augmented system (shared across batch) ----
    A_bar = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
    A_bar[:nx, :nx] = m.A;  A_bar[:nx, nx:] = m.B
    A_bar[nx:, nx:] = torch.eye(nu, device=dev, dtype=dt)

    B_bar = torch.zeros(nx_bar, nu, device=dev, dtype=dt)
    B_bar[:nx, :] = m.B
    B_bar[nx:, :] = torch.eye(nu, device=dev, dtype=dt)

    C_bar = torch.zeros(ny, nx_bar, device=dev, dtype=dt)
    C_bar[:, :nx] = m.C

    # e_bar: (nx_bar,), w_bar: (B, nx_bar)
    e_bar = torch.cat([m.e, torch.zeros(nu, device=dev, dtype=dt)])  # (nx_bar,)
    w_bar = torch.cat([w.expand(B, -1), torch.zeros(B, nu, device=dev, dtype=dt)], dim=1)  # (B, nx_bar)

    xmin_bar = torch.cat([m.xmin, m.umin])  # (nx_bar,)
    xmax_bar = torch.cat([m.xmax, m.umax])

    # Per-step bounds: (B, nx_bar, Np) — only used when umin_steps is provided.
    # The x-part of the bounds stays constant (±inf); only the u-part varies.
    if umin_steps is not None:
        # umin_steps: (B, nu, Np)
        xmin_bar_steps = torch.empty(B, nx_bar, Np, device=dev, dtype=dt)
        xmax_bar_steps = torch.empty(B, nx_bar, Np, device=dev, dtype=dt)
        xmin_bar_steps[:, :nx, :] = m.xmin.view(1, nx, 1).expand(B, nx, Np)
        xmax_bar_steps[:, :nx, :] = m.xmax.view(1, nx, 1).expand(B, nx, Np)
        xmin_bar_steps[:, nx:, :] = umin_steps
        xmax_bar_steps[:, nx:, :] = umax_steps
    else:
        xmin_bar_steps = None
        xmax_bar_steps = None

    # ---- Preconditioning ----
    if m.precond:
        E      = torch.sqrt((A_bar.T @ A_bar).diag())
        E_diag = torch.diag(E)
        E_inv  = torch.diag(1.0 / E)
        A_bar  = E_diag @ A_bar @ E_inv
        B_bar  = E_diag @ B_bar
        C_bar  = C_bar  @ E_inv
        e_bar  = E_diag @ e_bar
        w_bar  = (E_diag @ w_bar.T).T      # (B, nx_bar)
        xmin_bar = E * xmin_bar
        xmax_bar = E * xmax_bar
    else:
        E_diag = torch.eye(nx_bar, device=dev, dtype=dt)
        E_inv  = torch.eye(nx_bar, device=dev, dtype=dt)

    # ---- Cost matrices ----
    C_part  = C_bar[:, :nx]
    Q_bar   = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
    Q_bar[:nx, :nx] = C_part.T @ m.Wy @ C_part
    Q_bar[nx:, nx:] = m.Wu

    Q_bar_N = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
    Q_bar_N[:nx, :nx] = C_part.T @ m.Wf @ C_part
    Q_bar_N[nx:, nx:] = m.Wu

    # Per-step references:  yref (B, ny) or (B, ny, Np),  uref likewise.
    # q_bar  : (B, nx_bar, Np)  — per-step linear cost vectors
    # q_bar_N: (B, nx_bar, 1)   — terminal
    _CW  = C_part.T @ m.Wy   # (nx, nx)
    _CWf = C_part.T @ m.Wf   # (nx, nx)
    if yref.dim() == 3:
        # yref: (B, ny, Np),  uref: (B, nu, Np)
        q_y   = _CW @ yref            # (B, nx, Np)
        q_u   = m.Wu @ uref           # (B, nu, Np)
        q_bar = torch.cat([q_y, q_u], dim=1)  # (B, nx_bar, Np)
        # Terminal: last column of yref / uref
        q_bar_N = torch.cat([
            (_CWf @ yref[:, :, -1:]),   # (B, nx, 1)
            (m.Wu @ uref[:, :, -1:]),   # (B, nu, 1)
        ], dim=1)                       # (B, nx_bar, 1)
    else:
        # Constant reference (original path)  yref: (B, ny)
        q_bar_const = torch.cat([
            (_CW @ yref.T).T,           # (B, nx)
            (m.Wu @ uref.T).T           # (B, nu)
        ], dim=1)                       # (B, nx_bar)
        q_bar = q_bar_const.unsqueeze(2).expand(B, nx_bar, Np).contiguous()
        q_bar_N = torch.cat([
            (_CWf @ yref.T).T,
            (m.Wu @ uref.T).T
        ], dim=1).unsqueeze(2)          # (B, nx_bar, 1)
    R_bar = m.Wdu

    # ---- Precomputed inverses ----
    I_nb = torch.eye(nx_bar, device=dev, dtype=dt)
    # J_B: (nu, nx_bar), H_A, H_AN: (nx_bar, nx_bar)
    J_B  = torch.linalg.solve(R_bar + rho * B_bar.T @ B_bar, B_bar.T)
    H_A  = torch.linalg.inv(Q_bar   + rho * I_nb + rho * A_bar.T @ A_bar)
    H_AN = torch.linalg.inv(Q_bar_N + rho * I_nb)

    # x_bar: (B, nx_bar)
    x_bar = torch.cat([x0, u0], dim=1)

    # ---- Init ADMM variables ----
    # Layout: (B, nx_bar, Np+1) for X, (B, nx_bar, Np) for others
    Ex_bar = (E_diag @ x_bar.T).T.unsqueeze(2)       # (B, nx_bar, 1)
    X      = torch.zeros(B, nx_bar, Np+1, device=dev, dtype=dt)
    X[:, :, :1] = Ex_bar
    DU     = torch.zeros(B, nu,    Np,   device=dev, dtype=dt)
    V      = torch.zeros(B, nx_bar, Np,  device=dev, dtype=dt)
    Z      = torch.zeros(B, nx_bar, Np,  device=dev, dtype=dt)
    Theta  = torch.zeros(B, nx_bar, Np,  device=dev, dtype=dt)
    Beta   = torch.zeros(B, nx_bar, Np,  device=dev, dtype=dt)
    Lambda = torch.zeros(B, nx_bar, Np,  device=dev, dtype=dt)

    if m.accel:
        V_hat = V.clone(); Z_hat = Z.clone()
        Theta_hat = Theta.clone(); Beta_hat = Beta.clone(); Lambda_hat = Lambda.clone()

    Z_prev = Z.clone(); V_prev = V.clone()
    Theta_prev = Theta.clone(); Beta_prev = Beta.clone(); Lambda_prev = Lambda.clone()

    # e_bar, w_bar for broadcasting: (1, nx_bar, 1)
    e_b = e_bar.view(1, nx_bar, 1)
    w_b = w_bar.unsqueeze(2)         # (B, nx_bar, 1)

    residuals = []
    alpha_prev, res_prev = 1.0, float('inf')
    converged = False

    if verbose:
        print(f"PiMPC ADMM Solver (batched B={B})")
        print("-" * 40)

    t_start = time.perf_counter()
    for it in range(1, max_iter + 1):
        Z_prev.copy_(Z);        V_prev.copy_(V)
        Theta_prev.copy_(Theta); Beta_prev.copy_(Beta); Lambda_prev.copy_(Lambda)

        if m.accel:
            _V, _Z, _T, _Be, _La = V_hat, Z_hat, Theta_hat, Beta_hat, Lambda_hat
        else:
            _V, _Z, _T, _Be, _La = V, Z, Theta, Beta, Lambda

        # DU: (B, nu, Np)  ← J_B @ (V - Beta)  [J_B: (nu, nx_bar)]
        DU = J_B @ (_V - _Be)

        # X interior  (columns 1..Np-1)
        if Np > 1:
            # rhs_mid: (B, nx_bar, Np-1)
            rhs_mid = (q_bar[:, :, :Np-1]
                       + rho * (_Z[:, :, :Np-1] - _T[:, :, :Np-1]
                                + A_bar.T @ (_Z[:, :, 1:Np] - _V[:, :, 1:Np]
                                             + _La[:, :, 1:Np]
                                             - e_b.expand(B, -1, Np-1)
                                             - w_b.expand(B, -1, Np-1))))
            X[:, :, 1:Np] = H_A @ rhs_mid

        # X terminal
        rhs_N = q_bar_N + rho * (_Z[:, :, Np-1:Np] - _T[:, :, Np-1:Np])
        X[:, :, Np:Np+1] = H_AN @ rhs_N

        # BU: (B, nx_bar, Np), AX: (B, nx_bar, Np)
        BU = B_bar @ DU
        AX = A_bar @ X[:, :, :Np]

        ew = e_b.expand(B, -1, Np) + w_b.expand(B, -1, Np)

        Z  = (2.0 * (X[:, :, 1:Np+1] + _T) + BU + _Be + AX + ew - _La) / 3.0
        if xmin_bar_steps is not None:
            Z  = torch.clamp(Z, xmin_bar_steps, xmax_bar_steps)
        else:
            Z  = torch.clamp(Z, xmin_bar.view(1, -1, 1), xmax_bar.view(1, -1, 1))
        V  = 0.5 * (Z + BU + _Be - AX - ew + _La)

        Theta  = _T  + X[:, :, 1:Np+1] - Z
        Beta   = _Be + BU - V
        Lambda = _La + Z - AX - V - ew

        # Residual (sum over batch)
        if m.accel:
            dT = Theta - _T;  dB = Beta - _Be;  dL = Lambda - _La
            dZ = Z - _Z;      dV = V - _V
        else:
            dT = Theta - Theta_prev; dB = Beta - Beta_prev; dL = Lambda - Lambda_prev
            dZ = Z - Z_prev;         dV = V - V_prev

        res = float(rho * (dT.norm()**2 + dB.norm()**2 + dL.norm()**2
                           + dZ.norm()**2 + dV.norm()**2
                           + (dZ - dV).norm()**2))
        residuals.append(res)

        if res < tol * B:   # scale tolerance by batch size
            converged = True
            break

        if m.accel:
            if res < eta * res_prev:
                alpha = 0.5 * (1.0 + (1.0 + 4.0 * alpha_prev**2)**0.5)
                mom = (alpha_prev - 1.0) / alpha
                V_hat     = V     + mom * (V     - V_prev)
                Z_hat     = Z     + mom * (Z     - Z_prev)
                Theta_hat = Theta + mom * (Theta - Theta_prev)
                Beta_hat  = Beta  + mom * (Beta  - Beta_prev)
                Lambda_hat = Lambda + mom * (Lambda - Lambda_prev)
                res_prev = res
            else:
                alpha = 1.0
                V_hat.copy_(V); Z_hat.copy_(Z)
                Theta_hat.copy_(Theta); Beta_hat.copy_(Beta); Lambda_hat.copy_(Lambda)
                res_prev = res_prev / eta
            alpha_prev = alpha

    solve_time = time.perf_counter() - t_start

    if verbose:
        print(f"  Status: {'Converged' if converged else 'Not converged'}, "
              f"iters={len(residuals)}, time={solve_time*1e3:.2f} ms")

    # ---- Extract results ----
    # Un-precondition: X: (B, nx_bar, Np+1)
    X = E_inv @ X
    x_traj = X[:, :nx, :]           # (B, nx, Np+1)
    u_traj = X[:, nx:, 1:]          # (B, nu, Np)

    info = dict(solve_time=solve_time,
                iterations=len(residuals),
                converged=converged,
                obj_val=residuals[-1] if residuals else float('inf'))
    return x_traj, u_traj, DU, info, None
