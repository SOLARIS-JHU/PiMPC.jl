using PiMPC
using LinearAlgebra
using Printf
using Plots

# ============================================
#           AFTI-16 Aircraft System
# ============================================

function build_AFTI16(Ts=0.05)
    # Continuous-time system matrices
    As = [-0.0151  -60.5651   0.0      -32.174;
          -0.0001   -1.3411   0.9929    0.0;
           0.00018  43.2541  -0.86939   0.0;
           0.0       0.0      1.0       0.0]

    Bs = [-2.516   -13.136;
          -0.1689   -0.2514;
         -17.251    -1.5766;
           0.0       0.0]

    # Discretization (zero-order hold)
    nx, nu = 4, 2
    M = exp([As Bs; zeros(nu, nx) zeros(nu, nu)] * Ts)
    A = M[1:nx, 1:nx]
    B = M[1:nx, nx+1:end]

    # Output matrix: angle of attack and pitch angle
    C = [0.0 1.0 0.0 0.0;    # Angle of attack (y1)
         0.0 0.0 0.0 1.0]    # Pitch angle (y2)

    return A, B, C
end

# ============================================
#       Closed-Loop MPC Simulation
# ============================================

function simulate_closed_loop(model, A, B, C, x0, u0, yref_traj, uref, w, Nsim)
    nx, nu = size(B)
    ny = size(C, 1)

    # Storage for trajectories
    x_hist = zeros(nx, Nsim + 1)
    y_hist = zeros(ny, Nsim + 1)
    u_hist = zeros(nu, Nsim)
    iter_hist = zeros(Int, Nsim)

    x_hist[:, 1] = x0
    y_hist[:, 1] = C * x0
    x_current = copy(x0)
    u_prev = copy(u0)

    println("Running closed-loop simulation...")

    for k in 1:Nsim
        # Get current reference
        yref = yref_traj[:, k]

        # Solve MPC problem
        results = solve!(model, x_current, u_prev, yref, uref, w; verbose=false)

        # Extract first control input
        u_apply = results.u[:, 1]

        # Apply control to system (simulate one step)
        x_next = A * x_current + B * u_apply
        y_next = C * x_next

        # Store results
        u_hist[:, k] = u_apply
        x_hist[:, k + 1] = x_next
        y_hist[:, k + 1] = y_next
        iter_hist[k] = results.info.iterations

        # Update for next iteration
        x_current = x_next
        u_prev = u_apply
        @info "Step $k"
        @info "u = $(u_apply)"
        @info "x = $(x_next)"
    end

    println("Simulation completed.")
    println("-" ^ 40)
    @printf("  Total steps: %d\n", Nsim)
    @printf("  Average iterations: %.1f\n", sum(iter_hist) / Nsim)

    return x_hist, y_hist, u_hist, iter_hist
end

# ============================================
#           Main Test
# ============================================

println("=" ^ 50)
println("  PiMPC - AFTI-16 Closed-Loop Simulation")
println("=" ^ 50)

# Build system
Ts = 0.05
A, B, C = build_AFTI16(Ts)
nx, nu, ny = 4, 2, 2

# Create and setup MPC model
Np = 5
model = Model()
setup!(model;
    A = A, B = B, C = C, Np = Np,
    Wy = 100.0 * diagm(ones(ny)),
    Wu = 0.0 * diagm(ones(nu)),
    Wdu = 0.1 * diagm(ones(nu)),
    umin = -25.0 * ones(nu),
    umax = 25.0 * ones(nu),
    xmin = [-Inf, -0.5, -Inf, -100.0],
    xmax = [Inf, 0.5, Inf, 100.0],
    rho = 1.0,
    tol = 1e-6,
    maxiter = 10000,
    precond = true,
    accel = true,
    device = :cpu
)

# Initial conditions
x0 = zeros(nx)
u0 = zeros(nu)

# Simulation length
Nsim = 200

# Time-varying reference trajectory
# y1 (angle of attack): always 0
# y2 (pitch angle): 10 deg for first 100 steps, then 0
yref_traj = [zeros(1, Nsim);
             [10.0 * ones(1, 100) zeros(1, 100)]]

uref = zeros(nu)
w = zeros(nx)

# Run simulation
x_hist, y_hist, u_hist, iter_hist = simulate_closed_loop(
    model, A, B, C, x0, u0, yref_traj, uref, w, Nsim
)

# ============================================
#           Plot Results
# ============================================

println("\nGenerating plots...")

t_y = (0:Nsim) * Ts  # Time vector for states/outputs
t_u = (0:Nsim-1) * Ts  # Time vector for inputs

# Top: Trajectory (4x1 subplot)
p1 = plot(t_y, [yref_traj[2, :]; 0], label="Reference",
          ylabel="y2 (deg)", lw=2, ls=:dash, color=:black, legend=:topright)
plot!(p1, t_y, y_hist[2, :], label="Pitch Angle", lw=2, color=:red)

p2 = plot(t_y, y_hist[1, :], ylabel="y1 (deg)", lw=2, color=:red, legend=false)

p3 = plot(t_u, u_hist[1, :], ylabel="u1 (deg)", lw=2, color=:blue, legend=false)

p4 = plot(t_u, u_hist[2, :], ylabel="u2 (deg)", lw=2, color=:blue, legend=false)

# Bottom: Iterations per step
p5 = plot(1:Nsim, iter_hist, xlabel="Step", ylabel="Iterations",
          lw=2, color=:green, legend=false)

# Combine: trajectory on top (4 rows), iterations on bottom (1 row)
fig = plot(p1, p2, p3, p4, p5, layout=@layout([a; b; c; d; e]), size=(800, 800))
display(fig)

println("\nPress Enter to exit...")
readline()
