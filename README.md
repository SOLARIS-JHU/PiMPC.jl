<div align="center">

```
   ╔═══════════════════════════════════════════════════════════╗
   ║                                                           ║
   ║       ██████████    ███╗   ███╗ ██████╗   ██████╗         ║
   ║        ╚██╔╔██╝     ████╗ ████║ ██╔══██╗ ██╔════╝         ║
   ║         ██║║██      ██╔████╔██║ ██████╔╝ ██║              ║
   ║         ██║║██      ██║╚██╔╝██║ ██╔═══╝  ██║              ║
   ║         ██║║███     ██║ ╚═╝ ██║ ██║      ╚██████╗         ║
   ║         ╚═╝╚══╝     ╚═╝     ╚═╝ ╚═╝       ╚═════╝         ║
   ║                                                           ║
   ║                            πMPC                           ║
   ║            [P]arallel-[i]n-Horizon MPC Solver             ║
   ║                                                           ║
   ╚═══════════════════════════════════════════════════════════╝
```

# πMPC.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2601.14414-b31b1b.svg)](https://arxiv.org/abs/2601.14414)

</div>

A **P**arallel-**i**n-horizon and construction-free MPC solver based on ADMM with GPU acceleration.

## Key Features

- :zap: **Parallel Execution**: Horizon-wise parallelization on GPU
- :dart: **Construction-Free**: Operates directly on system matrices (no MPC-to-QP conversion)
- :bulb: **Simple Code**: Easy deployment on embedded platforms
- :straight_ruler: **Long-Horizon Efficient**: Ideal for large prediction horizons
- :rocket: **Nesterov Acceleration**: Adaptive restart for fast convergence
- :fire: **Warm Starting**: Efficient trajectory tracking

## Problem Formulation

πMPC solves the following MPC problem:

$$
\begin{aligned}
\min_{x, u} \quad & \sum_{k=0}^{N-1} \left( \|Cx_{k} - r_y\|_{W_y}^2 + \|u_k - r_u\|_{W_u}^2 + \|\Delta u_k\|_{W_{\Delta u}}^2 \right) + \|Cx_N - r_y\|_{W_f}^2 \\
\text{s.t.} \quad & x_{k+1} = A x_k + B u_k + e, \quad k = 0, \ldots, N-1 \\
& x_{\min} \leq x_k \leq x_{\max}, \quad k = 0, \ldots, N \\
& u_{\min} \leq u_k \leq u_{\max}, \quad k = 0, \ldots, N-1 \\
& \Delta u_{\min} \leq \Delta u_k \leq \Delta u_{\max}, \quad k = 0, \ldots, N-1
\end{aligned}
$$

where $A$, $B$, $e$ are the state matrix, input matrix, and affine term of the discrete-time linear system, and $W_y$, $W_u$, $W_{\Delta u}$, $W_f$ are the output, input, input increment, and terminal weight matrices, respectively.

## Installation

```bash
using Pkg
Pkg.add(url="https://github.com/SOLARIS-JHU/PiMPC.jl")
```

### Support and Bug reports
For support or usage-related questions, please contact liangwu2019@gmail.com and bobyang17@163.com.

## Quick Start

```julia
using PiMPC
using LinearAlgebra

# Define discrete linear system: x_{k+1} = A*x_k + B*u_k
A = [1.0 0.1; 0.0 1.0]  # Double integrator
B = reshape([0.005; 0.1], :, 1)
nx, nu = size(B)

# Create and configure model
model = Model()
setup!(model;
    A = A, B = B, Np = 20,
    Wy = 10.0 * I(nx),     # Output weight
    Wdu = 0.1 * I(nu),     # Input increment weight
    umin = [-1.0],         # Input lower bound
    umax = [1.0],          # Input upper bound
    rho = 10.0,            # ADMM penalty
    maxiter = 500,
    accel = true           # Nesterov acceleration
)

# Solve
x0, u0 = [5.0, 0.0], [0.0]           # Initial state and input
yref, uref = zeros(nx), zeros(nu)    # Output and input references
w = zeros(nx)                        # Known disturbance (zeros if none)

results = solve!(model, x0, u0, yref, uref, w; verbose=true)

# Access results
println("Optimal input: ", results.u[:, 1])
println("Iterations: ", results.info.iterations)

# Warm start (automatic for subsequent solves)
x_next = results.x[:, 2]
u_next = results.u[:, 1]
results2 = solve!(model, x_next, u_next, yref, uref, w)
```

## Example

Run the AFTI-16 aircraft closed-loop simulation:

```bash
julia --project=. examples/AFTI16_example.jl
```

| | |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/95e99d31-d283-4cf9-9054-221199b13889" width="400"> | <img src="https://github.com/user-attachments/assets/287fa398-ea28-41f5-bf30-8d2bed285778" width="400"> |
| *AFTI-16 closed-loop trajectory tracking* | *Convergence iterations at each MPC step* |
| <img src="https://github.com/user-attachments/assets/8edecde7-9773-482b-a577-62522c28a1e2" width="400"> | <img src="https://github.com/user-attachments/assets/13fa5f74-d78d-4da1-bd41-bcb06a5bacfd" width="400"> |
| *Per-iteration time vs. horizon and dimension* | *CSTR trajectory tracking with disturbance* |

## API

### Model

Create a model, configure with `setup!()`, and solve with `solve!()`.

```julia
model = Model()
setup!(model; A, B, Np, kwargs...)
results = solve!(model, x0, u0, yref, uref, w)
```

### setup!

```julia
setup!(model; A, B, Np, kwargs...)
```

**Required Arguments:**
- `A`: State matrix
- `B`: Input matrix
- `Np`: Prediction horizon

**Optional Problem Arguments:**
- `C`: Output matrix (default: I, meaning y = x)
- `e`: Affine term in dynamics (default: zeros)
- `Wy`: Output weight matrix (default: I)
- `Wu`: Input weight matrix (default: I)
- `Wdu`: Input increment weight matrix (default: I)
- `Wf`: Terminal cost weight matrix (default: Wy)
- `xmin, xmax`: State constraints (default: ±Inf)
- `umin, umax`: Input constraints (default: ±Inf)
- `dumin, dumax`: Input increment constraints (default: ±Inf)

**Solver Settings:**
- `rho`: ADMM penalty parameter (default: 1.0)
- `tol`: Convergence tolerance (default: 1e-4)
- `eta`: Acceleration restart factor (default: 0.999)
- `maxiter`: Maximum iterations (default: 100)
- `precond`: Use preconditioning (default: false)
- `accel`: Use Nesterov acceleration (default: false)
- `device`: Compute device, `:cpu` or `:gpu` (default: :cpu)

### solve!

```julia
results = solve!(model, x0, u0, yref, uref, w; verbose=false)
```

**System Model:**
```
x_{k+1} = A * x_k + B * u_k + e + w
y_k = C * x_k
```

where:
- `e`: Constant affine term (defined in `setup!`)
- `w`: Known/measured disturbance (provided at solve time)

**Arguments:**
- `x0`: Current state (nx)
- `u0`: Previous input (nu)
- `yref`: Output reference (ny)
- `uref`: Input reference (nu)
- `w`: Known disturbance (nx), use `zeros(nx)` if none

**Returns** `Results`:
```julia
results.x              # State trajectory (nx × Np+1)
results.u              # Input trajectory (nu × Np)
results.du             # Input increment (nu × Np)
results.info.solve_time # Solve time (seconds)
results.info.iterations # Number of iterations
results.info.converged  # Whether converged
```

## GPU Acceleration

```julia
model = Model()
setup!(model; A=A, B=B, Np=20, accel=true, device=:gpu)
results = solve!(model, x0, u0, yref, uref, w)
```

Falls back to CPU automatically if GPU is not available.

## Citation

If you use πMPC in your research, please cite:

```bibtex
@inproceedings{wu2026piMPC,
      title={piMPC: A Parallel-in-horizon and Construction-free NMPC Solver},
      author={Liang Wu, Bo Yang, Yilin Mo, Yang Shi, and Jan Drgona},
      year={2026},
      eprint={2601.14414},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2601.14414},
      volume={},
      number={},
}
```
