# πMPC — PyTorch Port

PyTorch implementation of the PiMPC solver.  
Runs on CPU or CUDA GPU and supports batched solving across many initial conditions in parallel.

---

## Installation

No separate install is required beyond PyTorch:

```bash
pip install torch numpy
```

Import directly from the package root:

```python
import sys
sys.path.insert(0, "/path/to/PiMPC")
from pytorch.pimpc import Model
```

---

## Quick Start

```python
import numpy as np
import torch
from pytorch.pimpc import Model

# Discrete-time double integrator: x_{k+1} = A x_k + B u_k
dt = 0.1
A = np.array([[1.0, dt], [0.0, 1.0]])
B = np.array([[0.5 * dt**2], [dt]])

model = Model()
model.setup(
    A=A, B=B, Np=20,
    Wy=10.0 * np.eye(2),   # state tracking weight
    Wdu=1.0 * np.eye(1),   # input increment weight
    umin=[-5.0], umax=[5.0],
    rho=1.0, tol=1e-4, maxiter=200,
    accel=True,
    device="cpu",
    dtype=torch.float64,
)

x0   = np.array([1.0, 0.0])   # current state
u0   = np.array([0.0])        # previous input
yref = np.zeros(2)            # target: origin
uref = np.zeros(1)

res = model.solve(x0, u0, yref, uref)
print("Optimal first input:", res.u[:, 0])
print("Converged:", res.converged, "in", res.iterations, "iterations")
```

---

## Problem Formulation

$$
\min_{x, u} \sum_{k=0}^{N-1} \left( \|Cx_k - r_y\|_{W_y}^2 + \|u_k - r_u\|_{W_u}^2 + \|\Delta u_k\|_{W_{\Delta u}}^2 \right) + \|Cx_N - r_y\|_{W_f}^2
$$

subject to:

$$
x_{k+1} = A x_k + B u_k + e + w, \quad y_k = C x_k
$$
$$
x_{\min} \le x_k \le x_{\max}, \quad u_{\min} \le u_k \le u_{\max}, \quad \Delta u_{\min} \le \Delta u_k \le \Delta u_{\max}
$$

where `e` is a constant affine term (set in `setup`) and `w` is a per-solve known disturbance.

---

## API Reference

### `Model`

The main solver class.

```python
model = Model()
```

---

### `Model.setup()`

Configure the MPC problem. Must be called before `solve()` or `solve_batch()`.

```python
model.setup(
    A, B, Np,
    C=None, e=None,
    Wy=None, Wu=None, Wdu=None, Wf=None,
    xmin=None, xmax=None,
    umin=None, umax=None,
    dumin=None, dumax=None,
    rho=1.0, tol=1e-4, eta=0.999, maxiter=100,
    precond=False, accel=False,
    device="cpu", dtype=torch.float64,
)
```

**Required**

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | array-like `(nx, nx)` | State transition matrix |
| `B` | array-like `(nx, nu)` | Input matrix |
| `Np` | `int` | Prediction horizon |

**Problem (optional)**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | `I(nx)` | Output matrix `(ny, nx)`, sets `y = C x` |
| `e` | `zeros(nx)` | Constant affine offset in dynamics |
| `Wy` | `I(ny)` | Output tracking weight `(ny, ny)` |
| `Wu` | `I(nu)` | Input weight `(nu, nu)` |
| `Wdu` | `I(nu)` | Input-increment weight `(nu, nu)` |
| `Wf` | `Wy` | Terminal cost weight `(ny, ny)` |
| `xmin/xmax` | `±inf` | State box constraints `(nx,)` |
| `umin/umax` | `±inf` | Input box constraints `(nu,)` |
| `dumin/dumax` | `±inf` | Input-rate box constraints `(nu,)` |

**Solver (optional)**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho` | `1.0` | ADMM penalty parameter |
| `tol` | `1e-4` | Convergence tolerance |
| `eta` | `0.999` | Nesterov acceleration restart threshold |
| `maxiter` | `100` | Maximum ADMM iterations |
| `precond` | `False` | Enable diagonal preconditioning |
| `accel` | `False` | Enable Nesterov acceleration (recommended) |
| `device` | `"cpu"` | Compute device: `"cpu"`, `"cuda"`, `"cuda:0"`, etc. |
| `dtype` | `torch.float64` | Floating-point precision (`float32` or `float64`) |

---

### `Model.solve()`

Solve for a **single** initial condition. Stores warm-start variables internally for the next call.

```python
res = model.solve(x0, u0, yref, uref, w=None, verbose=False)
```

**Arguments**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `x0` | `(nx,)` | Current state |
| `u0` | `(nu,)` | Previous applied input |
| `yref` | `(ny,)` | Output reference |
| `uref` | `(nu,)` | Input reference |
| `w` | `(nx,)` | Known disturbance (default: zeros) |
| `verbose` | `bool` | Print iteration log |

**Returns** → [`Results`](#results)

---

### `Model.solve_batch()`

Solve for a **batch** of initial conditions simultaneously. Exploits GPU parallelism when `device="cuda"`.

```python
res = model.solve_batch(
    x0, u0, yref, uref, w=None,
    umin_steps=None, umax_steps=None,
    verbose=False,
)
```

**Arguments**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `x0` | `(B, nx)` | Batch of current states |
| `u0` | `(B, nu)` | Batch of previous inputs |
| `yref` | `(B, ny)` or `(B, ny, Np)` | Constant or per-step output reference |
| `uref` | `(B, nu)` or `(B, nu, Np)` | Constant or per-step input reference |
| `w` | `(B, nx)` or `(nx,)` | Known disturbance, broadcast if 1-D (default: zeros) |
| `umin_steps` | `(B, nu, Np)` | Per-step lower bound on `u`, overrides `setup` value |
| `umax_steps` | `(B, nu, Np)` | Per-step upper bound on `u`, overrides `setup` value |
| `verbose` | `bool` | Print iteration log |

**Returns** → [`Results`](#results) with leading batch dimension `B`.

> **Note:** Warm-starting is not carried over between `solve_batch` calls.

---

### `Results`

Dataclass returned by both `solve` and `solve_batch`.

| Field | Shape (single) | Shape (batch) | Description |
|-------|---------------|--------------|-------------|
| `x` | `(nx, Np+1)` | `(B, nx, Np+1)` | Predicted state trajectory |
| `u` | `(nu, Np)` | `(B, nu, Np)` | Optimal input sequence |
| `du` | `(nu, Np)` | `(B, nu, Np)` | Input increments |
| `solve_time` | `float` | `float` | Wall-clock solve time (seconds) |
| `iterations` | `int` | `int` | Number of ADMM iterations taken |
| `converged` | `bool` | `bool` | Whether tolerance was reached |
| `obj_val` | `float` | `float` | Final residual / objective value |

---

## GPU Acceleration

```python
model = Model()
model.setup(A=A, B=B, Np=20, accel=True, device="cuda", dtype=torch.float32)

# All tensors must be on the same device (or convertible)
x0   = torch.randn(B, nx, device="cuda", dtype=torch.float32)
u0   = torch.zeros(B, nu, device="cuda", dtype=torch.float32)
yref = torch.zeros(B, ny, device="cuda", dtype=torch.float32)
uref = torch.zeros(B, nu, device="cuda", dtype=torch.float32)

res = model.solve_batch(x0, u0, yref, uref)
```

Tips:
- Use `dtype=torch.float32` for maximum GPU throughput.
- Use `accel=True` (Nesterov) for faster convergence.
- All array-like inputs are automatically converted to tensors on the target device.

---

## Warm Starting

`solve()` automatically stores and reuses warm-start variables across consecutive calls:

```python
res1 = model.solve(x0, u0, yref, uref, w)   # cold start
res2 = model.solve(x1, res1.u[:, 0], yref, uref, w)  # warm start (automatic)
```

To reset the warm start:

```python
model.warm_vars = None
```

`solve_batch()` does **not** carry warm-start state between calls.

---


### Running the examples

```bash
# From repo root

# Batched double-integrator demo (CPU or GPU)
python pytorch/example_batched.py --device cuda --batch-sizes 1,64,256,1024
```

---

## Solver Settings Guide

| Scenario | Recommended settings |
|----------|---------------------|
| Tight tolerance (offline) | `tol=1e-6`, `maxiter=1000`, `accel=True`, `precond=True` |
| Real-time control (loose) | `tol=1e-3`, `maxiter=100–200`, `accel=True` |
| GPU large-batch | `dtype=torch.float32`, `accel=True`, `precond=False` |
| Ill-conditioned system | `precond=True`, increase `rho` (e.g. `rho=10`) |
| Long horizon (Np ≥ 50) | `accel=True`, `rho=1–10`, `dtype=float32` for GPU |
