"""
Batched MPC solve example with solve-time profiling
----------------------------------------------------
Solves B MPC problems in parallel using solve_batch(), then profiles
median solve time and per-sample throughput across a range of batch sizes.

System: double integrator  x_{k+1} = A x_k + B u_k
  state  x = [position, velocity]  (nx=2)
  input  u = [force]               (nu=1)
  augmented state dimension: nx_bar = nx + nu = 3

Usage:
    python pytorch/example_batched.py
    python pytorch/example_batched.py --device cuda
    python pytorch/example_batched.py --B 64 --n-runs 30
    python pytorch/example_batched.py --batch-sizes 1,4,16,64,256
"""

import argparse
import time

import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorch.pimpc import Model


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

def build_double_integrator(dt=0.1):
    A = np.array([[1.0, dt ],
                  [0.0, 1.0]])
    B = np.array([[0.5 * dt**2],
                  [dt]])
    return A, B


# ---------------------------------------------------------------------------
# Profile helper
# ---------------------------------------------------------------------------

def profile(model, x0, u0, yref, uref, n_warmup=3, n_runs=20, device="cpu"):
    """Warm up, then return (result, array-of-wall-times-ms)."""
    for _ in range(n_warmup):
        model.solve_batch(x0, u0, yref, uref)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = model.solve_batch(x0, u0, yref, uref)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    return res, np.array(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--B",           type=int, default=16,
                        help="Batch size for the illustrative solve")
    parser.add_argument("--n-runs",      type=int, default=20)
    parser.add_argument("--n-warmup",    type=int, default=3)
    parser.add_argument("--batch-sizes", default="1,4,16,64",
                        help="Comma-separated batch sizes for the profile sweep "
                             "(default: 1,4,16,64; GPU supports much larger)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    A, B_mat = build_double_integrator(dt=0.1)
    nx, nu = 2, 1

    print("=" * 58)
    print("  PiMPC Batched Solve Example — double integrator")
    print("=" * 58)
    print(f"  nx={nx}, nu={nu}, nx_bar={nx+nu}  device={device}")
    print()

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    model = Model()
    model.setup(
        A=A, B=B_mat, Np=20,
        Wy   = 10.0 * np.eye(nx),   # penalise position + velocity
        Wdu  =  1.0 * np.eye(nu),   # penalise input increments
        umin = [-5.0], umax = [5.0],
        xmin = [-10.0, -3.0], xmax = [10.0, 3.0],
        rho     = 1.0,
        tol     = 0.1,    # loose: typically converges in ~15–50 iters
        maxiter = 200,
        accel   = True,
        device  = device,
        dtype   = torch.float64,
    )

    rng = np.random.default_rng(0)

    # -----------------------------------------------------------------------
    # [1] Illustrative solve
    # -----------------------------------------------------------------------
    B_sz = args.B

    x0   = torch.tensor(rng.uniform(-2.0, 2.0, (B_sz, nx)),
                        dtype=torch.float64, device=device)
    u0   = torch.zeros(B_sz, nu, dtype=torch.float64, device=device)
    yref = torch.zeros(B_sz, nx, dtype=torch.float64, device=device)   # target: origin
    uref = torch.zeros(B_sz, nu, dtype=torch.float64, device=device)

    print(f"[1] Batched solve  (B={B_sz}, drive {B_sz} envs to origin)")

    t0  = time.perf_counter()
    res = model.solve_batch(x0, u0, yref, uref)
    t_ms = (time.perf_counter() - t0) * 1e3

    print(f"  converged   : {res.converged}")
    print(f"  iterations  : {res.iterations}")
    print(f"  total time  : {t_ms:.2f} ms")
    print(f"  time/env    : {t_ms/B_sz:.4f} ms")
    print()

    # show first few environments
    n_show = min(6, B_sz)
    print(f"  {'env':>4}  {'pos₀':>7}  {'vel₀':>7}  │  {'u*(0)':>7}  "
          f"→  {'pos₁':>7}  {'vel₁':>7}")
    print("  " + "-" * 52)
    for i in range(n_show):
        pos = x0[i, 0].item();  vel = x0[i, 1].item()
        u   = res.u[i, 0, 0].item()
        p1  = A[0, 0]*pos + A[0, 1]*vel + float(B_mat[0, 0])*u
        v1  = A[1, 0]*pos + A[1, 1]*vel + float(B_mat[1, 0])*u
        print(f"  {i:>4}  {pos:>7.3f}  {vel:>7.3f}  │  {u:>7.3f}  "
              f"   {p1:>7.3f}  {v1:>7.3f}")

    # -----------------------------------------------------------------------
    # [2] Solve-time profile sweep
    # -----------------------------------------------------------------------
    print()
    print(f"[2] Solve-time profile  "
          f"(n_warmup={args.n_warmup}, n_runs={args.n_runs})")
    print()
    print(f"  {'B':>6}  {'iters':>6}  {'conv':>5}  "
          f"{'median ms':>10}  {'std ms':>7}  "
          f"{'ms/env':>8}  {'env/s':>9}  {'vs B=1':>8}")
    print("  " + "-" * 64)

    base_per_env = None
    for b in batch_sizes:
        x0_b   = torch.tensor(rng.uniform(-2.0, 2.0, (b, nx)),
                               dtype=torch.float64, device=device)
        u0_b   = torch.zeros(b, nu, dtype=torch.float64, device=device)
        yref_b = torch.zeros(b, nx, dtype=torch.float64, device=device)
        uref_b = torch.zeros(b, nu, dtype=torch.float64, device=device)

        res_b, times = profile(
            model, x0_b, u0_b, yref_b, uref_b,
            n_warmup=args.n_warmup, n_runs=args.n_runs, device=device)

        med      = float(np.median(times))
        std_     = float(np.std(times))
        per_env  = med / b
        env_per_s = b / (med * 1e-3)

        if base_per_env is None:
            base_per_env = per_env
        speedup = base_per_env / per_env

        print(f"  {b:>6}  {res_b.iterations:>6}  {str(res_b.converged):>5}  "
              f"{med:>10.3f}  {std_:>7.3f}  "
              f"{per_env:>8.4f}  {env_per_s:>9.0f}  {speedup:>7.1f}x")

    if device == "cpu":
        print()
        print("  Note: GPU benefits are most visible at large B (≥256) and for")
        print("  larger systems (nx_bar ≥ 64). See pytorch/benchmark_gpu.py.")

    print()
    print("=" * 58)


if __name__ == "__main__":
    main()
