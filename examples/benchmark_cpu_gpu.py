import argparse
import time

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
        raise RuntimeError("SciPy is required for the AFTI-16 benchmark.") from exc

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


def build_random_system(nx=8, nu=3, ny=4, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((nx, nx)) * 0.1
    radius = np.max(np.abs(np.linalg.eigvals(A)))
    if radius > 0:
        A = A * min(0.95 / radius, 1.0)
    B = rng.standard_normal((nx, nu)) * 0.1
    C = rng.standard_normal((ny, nx))
    return A, B, C


def setup_model(A, B, C, device, Np):
    nx, nu = B.shape
    ny = C.shape[0]
    model = Model()
    model.setup(
        A=A,
        B=B,
        C=C,
        Np=Np,
        Wy=10.0 * np.eye(ny),
        Wu=0.1 * np.eye(nu),
        Wdu=0.1 * np.eye(nu),
        rho=1.0,
        tol=1e-6,
        maxiter=200,
        precond=True,
        accel=True,
        device=device,
    )
    return model


def run_benchmark(model, A, B, repeats, warmup):
    nx, nu = B.shape
    ny = model.ny
    x0 = np.zeros(nx)
    u0 = np.zeros(nu)
    yref = np.zeros(ny)
    uref = np.zeros(nu)
    w = np.zeros(nx)

    for _ in range(warmup):
        results = model.solve(x0, u0, yref, uref, w, verbose=False)
        u0 = results.u[:, 0]
        x0 = A @ x0 + B @ u0

    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        results = model.solve(x0, u0, yref, uref, w, verbose=False)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        u0 = results.u[:, 0]
        x0 = A @ x0 + B @ u0

    return np.array(times_ms, dtype=float)


def summarize(times_ms):
    return {
        "mean": float(np.mean(times_ms)),
        "median": float(np.median(times_ms)),
        "std": float(np.std(times_ms)),
        "min": float(np.min(times_ms)),
        "max": float(np.max(times_ms)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PiMPC CPU vs GPU solver.")
    parser.add_argument("--system", choices=["afti16", "random"], default="afti16")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--Np", type=int, default=5)
    args = parser.parse_args()

    if args.system == "afti16":
        A, B, C = build_afti16()
    else:
        A, B, C = build_random_system()

    cpu_model = setup_model(A, B, C, device="cpu", Np=args.Np)
    cpu_times = run_benchmark(cpu_model, A, B, repeats=args.repeats, warmup=args.warmup)
    cpu_stats = summarize(cpu_times)

    print("CPU benchmark")
    print(f"  mean:   {cpu_stats['mean']:.3f} ms")
    print(f"  median: {cpu_stats['median']:.3f} ms")
    print(f"  std:    {cpu_stats['std']:.3f} ms")
    print(f"  min:    {cpu_stats['min']:.3f} ms")
    print(f"  max:    {cpu_stats['max']:.3f} ms")

    gpu_model = setup_model(A, B, C, device="gpu", Np=args.Np)
    if gpu_model.device != "gpu":
        print("GPU benchmark skipped (GPU not available or CuPy missing).")
        return

    gpu_times = run_benchmark(gpu_model, A, B, repeats=args.repeats, warmup=args.warmup)
    gpu_stats = summarize(gpu_times)

    print("GPU benchmark")
    print(f"  mean:   {gpu_stats['mean']:.3f} ms")
    print(f"  median: {gpu_stats['median']:.3f} ms")
    print(f"  std:    {gpu_stats['std']:.3f} ms")
    print(f"  min:    {gpu_stats['min']:.3f} ms")
    print(f"  max:    {gpu_stats['max']:.3f} ms")

    speedup = cpu_stats["mean"] / gpu_stats["mean"]
    print(f"Speedup (mean CPU / mean GPU): {speedup:.2f}x")


if __name__ == "__main__":
    main()
