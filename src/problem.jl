"""
    Model

PiMPC model. Create with `Model()`, configure with `setup!()`, solve with `solve!()`.

## Example
```julia
m = PiMPC.Model()
PiMPC.setup!(m; A=A, B=B, Np=20, umin=[-1.0], umax=[1.0])
results = PiMPC.solve!(m, x0, u0, yref, uref, w)
```
"""
mutable struct Model{T<:Real}
    # Problem data
    A::Matrix{T}
    B::Matrix{T}
    C::Matrix{T}
    e::Vector{T}
    nx::Int
    nu::Int
    ny::Int
    Np::Int

    # Weights
    Wy::Matrix{T}
    Wu::Matrix{T}
    Wdu::Matrix{T}
    Wf::Matrix{T}

    # Constraints
    xmin::Vector{T}
    xmax::Vector{T}
    umin::Vector{T}
    umax::Vector{T}
    dumin::Vector{T}
    dumax::Vector{T}

    # Settings
    rho::T
    tol::T
    eta::T
    maxiter::Int
    precond::Bool
    accel::Bool
    device::Symbol

    # State
    is_setup::Bool
    warm_vars::Union{Nothing, Tuple}

    function Model{T}() where T<:Real
        new{T}(
            Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0), Vector{T}(undef, 0),
            0, 0, 0, 0,
            Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
            Vector{T}(undef, 0), Vector{T}(undef, 0),
            Vector{T}(undef, 0), Vector{T}(undef, 0),
            Vector{T}(undef, 0), Vector{T}(undef, 0),
            T(1.0), T(1e-4), T(0.999), 100, false, false, :cpu,
            false, nothing
        )
    end
end

Model() = Model{Float64}()

"""
    setup!(model; A, B, Np, kwargs...)

Setup the MPC problem.

## System Model
```
x_{k+1} = A * x_k + B * u_k + e
y_k = C * x_k
```

## Required Arguments
- `A`: State matrix
- `B`: Input matrix
- `Np`: Prediction horizon

## Optional Problem Arguments
- `C`: Output matrix (default: I)
- `e`: Affine term (default: zeros)
- `Wy`: Output weight (default: I)
- `Wu`: Input weight (default: I)
- `Wdu`: Input increment weight (default: I)
- `Wf`: Terminal weight (default: Wy)
- `xmin, xmax`: State bounds (default: ±Inf)
- `umin, umax`: Input bounds (default: ±Inf)
- `dumin, dumax`: Input increment bounds (default: ±Inf)

## Solver Settings
- `rho`: ADMM penalty (default: 1.0)
- `tol`: Convergence tolerance (default: 1e-4)
- `eta`: Acceleration restart factor (default: 0.999)
- `maxiter`: Maximum iterations (default: 100)
- `precond`: Use preconditioning (default: false)
- `accel`: Use Nesterov acceleration (default: false)
- `device`: `:cpu` or `:gpu` (default: :cpu)
"""
function setup!(m::Model{T};
    # Required
    A::Matrix, B::Matrix, Np::Int,
    # Optional problem
    C::Union{Matrix, Nothing}=nothing,
    e::Union{Vector, Nothing}=nothing,
    Wy=nothing, Wu=nothing, Wdu=nothing, Wf=nothing,
    xmin=nothing, xmax=nothing,
    umin=nothing, umax=nothing,
    dumin=nothing, dumax=nothing,
    # Settings
    rho::Real=1.0, tol::Real=1e-4, eta::Real=0.999,
    maxiter::Int=100, precond::Bool=false,
    accel::Bool=false, device::Symbol=:cpu
) where T<:Real

    nx, nu = size(B)

    # Set matrices
    m.A = Matrix{T}(A)
    m.B = Matrix{T}(B)
    m.C = C === nothing ? Matrix{T}(I, nx, nx) : Matrix{T}(C)
    m.e = e === nothing ? zeros(T, nx) : Vector{T}(e)

    ny = size(m.C, 1)
    m.nx, m.nu, m.ny, m.Np = nx, nu, ny, Np

    # Set weights
    m.Wy  = Wy  === nothing ? Matrix{T}(I, ny, ny) : Matrix{T}(Wy)
    m.Wu  = Wu  === nothing ? Matrix{T}(I, nu, nu) : Matrix{T}(Wu)
    m.Wdu = Wdu === nothing ? Matrix{T}(I, nu, nu) : Matrix{T}(Wdu)
    m.Wf  = Wf  === nothing ? Matrix{T}(m.Wy)      : Matrix{T}(Wf)

    # Set constraints
    m.xmin  = xmin  === nothing ? fill(T(-Inf), nx) : Vector{T}(xmin)
    m.xmax  = xmax  === nothing ? fill(T(Inf), nx)  : Vector{T}(xmax)
    m.umin  = umin  === nothing ? fill(T(-Inf), nu) : Vector{T}(umin)
    m.umax  = umax  === nothing ? fill(T(Inf), nu)  : Vector{T}(umax)
    m.dumin = dumin === nothing ? fill(T(-Inf), nu) : Vector{T}(dumin)
    m.dumax = dumax === nothing ? fill(T(Inf), nu)  : Vector{T}(dumax)

    # Set solver settings
    if device == :gpu && !CUDA.functional()
        @warn "GPU not available, falling back to CPU"
        device = :cpu
    end
    m.rho, m.tol, m.eta = T(rho), T(tol), T(eta)
    m.maxiter, m.precond, m.accel, m.device = maxiter, precond, accel, device

    m.is_setup = true
    m.warm_vars = nothing

    return nothing
end

"""
    Results{T<:Real}

Solve results.

## Fields
- `x`: State trajectory (nx × Np+1)
- `u`: Input trajectory (nu × Np)
- `du`: Input increment (nu × Np)
- `info`: Solve information (solve_time, iterations, converged, obj_val)
"""
struct Results{T<:Real, S<:Real}
    x::Matrix{T}
    u::Matrix{T}
    du::Matrix{T}
    info::NamedTuple{(:solve_time, :iterations, :converged, :obj_val), Tuple{S, Int, Bool, S}}
end
