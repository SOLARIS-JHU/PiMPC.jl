"""
    solve!(model, x0, u0, yref, uref, w; verbose=false)

Solve the MPC problem.

## Arguments
- `model::Model`: Configured model (call `setup!` first)
- `x0::Vector`: Current state (nx)
- `u0::Vector`: Previous input (nu)
- `yref::Vector`: Output reference (ny)
- `uref::Vector`: Input reference (nu)
- `w::Vector`: Known disturbance (nx), use `zeros(nx)` if none

## Returns
`Results` struct containing:
- `results.x`: State trajectory (nx × Np+1)
- `results.u`: Input trajectory (nu × Np)
- `results.du`: Input increment (nu × Np)
- `results.info`: Named tuple with (solve_time, iterations, converged, obj_val)

## Example
```julia
m = piMPC.Model()
piMPC.setup!(m; A=A, B=B, Np=20, umin=[-1.0], umax=[1.0], accel=true)
results = piMPC.solve!(m, x0, u0, yref, uref, zeros(nx))
println(results.u[:, 1])  # Optimal input
```
"""
function solve!(m::Model, x0::Vector, u0::Vector, yref::Vector, uref::Vector,
                w::Vector; verbose::Bool=false)
    !m.is_setup && error("Model not setup. Call setup!() first.")

    # Call internal solver
    if m.device == :gpu && CUDA.functional()
        x, u, du, info, warm = _solve_gpu(m, x0, u0, yref, uref, w; warm_vars=m.warm_vars, verbose=verbose)
    else
        x, u, du, info, warm = _solve_cpu(m, x0, u0, yref, uref, w; warm_vars=m.warm_vars, verbose=verbose)
    end

    # Update warm start
    m.warm_vars = warm

    return Results(x, u, du, info)
end

function _solve_cpu(m::Model, x0::Vector, u0::Vector, yref::Vector, uref::Vector,
                    w::Vector; warm_vars=nothing, verbose::Bool=false)
    nx, nu, ny, Np = m.nx, m.nu, m.ny, m.Np
    nx_bar = nx + nu

    # Augmented system
    A_bar = [m.A m.B; zeros(nu, nx) I(nu)]
    B_bar = [m.B; I(nu)]
    C_bar = [m.C zeros(ny, nu)]
    e_bar = [m.e; zeros(nu)]
    w_bar = [w; zeros(nu)]
    xmin_bar = [m.xmin; m.umin]
    xmax_bar = [m.xmax; m.umax]

    rho, tol, max_iter, eta = m.rho, m.tol, m.maxiter, m.eta

    # Preconditioning
    if m.precond
        E = sqrt.(diag(A_bar'*A_bar))
        E_diag, E_inv = Diagonal(E), Diagonal(1.0 ./ E)
        A_bar = E_diag * A_bar * E_inv
        B_bar = E_diag * B_bar
        C_bar = C_bar * E_inv
        e_bar = E_diag * e_bar
        w_bar = E_diag * w_bar
        xmin_bar = E .* xmin_bar
        xmax_bar = E .* xmax_bar
    else
        E_diag, E_inv = I(nx_bar), I(nx_bar)
    end

    # Cost matrices
    C_part = C_bar[:, 1:nx]
    Q_bar = [C_part' * m.Wy * C_part zeros(nx, nu); zeros(nu, nx) m.Wu]
    Q_bar_N = [C_part' * m.Wf * C_part zeros(nx, nu); zeros(nu, nx) m.Wu]
    q_bar = [C_part' * m.Wy * yref; m.Wu * uref]
    q_bar_N = [C_part' * m.Wf * yref; m.Wu * uref]
    R_bar = m.Wdu

    # Precompute inverse matrices
    J_B = inv(R_bar + rho * B_bar'*B_bar) * B_bar'
    H_A = inv(Q_bar + rho * I(nx_bar) + rho * A_bar' * A_bar)
    H_AN = inv(Q_bar_N + rho * I(nx_bar))

    x_bar = [x0; u0]

    # Initialize variables
    if warm_vars === nothing
        DU = zeros(nu, Np)
        X = zeros(nx_bar, Np+1); X[:,1] = E_diag * x_bar
        V, Z = zeros(nx_bar, Np), zeros(nx_bar, Np)
        Theta, Beta, Lambda = zeros(nx_bar, Np), zeros(nx_bar, Np), zeros(nx_bar, Np)
    else
        DU0, X0, V0, Z0, Theta0, Beta0, Lambda0 = warm_vars
        DU = [DU0[:, 2:end] DU0[:, end]]
        X = [E_diag * x_bar X0[:, 3:end] X0[:, end]]
        V = [V0[:, 2:end] V0[:, end]]
        Z = [Z0[:, 2:end] Z0[:, end]]
        Theta = [Theta0[:, 2:end] Theta0[:, end]]
        Beta = [Beta0[:, 2:end] Beta0[:, end]]
        Lambda = [Lambda0[:, 2:end] Lambda0[:, end]]
        X[:, 2:Np+1] = E_diag * X[:, 2:Np+1]
    end

    # Acceleration variables
    if m.accel
        V_hat, Z_hat = copy(V), copy(Z)
        Theta_hat, Beta_hat, Lambda_hat = copy(Theta), copy(Beta), copy(Lambda)
    end

    Z_prev, V_prev = copy(Z), copy(V)
    Theta_prev, Beta_prev, Lambda_prev = copy(Theta), copy(Beta), copy(Lambda)

    residuals = Float64[]
    alpha_prev, res_prev, res = 1.0, Inf, Inf
    converged = false

    verbose && println("piMPC ADMM Solver (CPU)")
    verbose && println("-" ^ 40)
    verbose && @printf("  %6s  %12s\n", "Iter", "Residual")
    verbose && println("-" ^ 40)

    t_start = time()
    for iter in 1:max_iter
        Z_prev .= Z; V_prev .= V
        Theta_prev .= Theta; Beta_prev .= Beta; Lambda_prev .= Lambda

        if m.accel
            @views begin
                DU .= J_B * (V_hat - Beta_hat)
                X[:,2:Np] .= H_A * (q_bar .+ rho .* (Z_hat[:, 1:Np-1] .- Theta_hat[:, 1:Np-1] .+
                              A_bar' * (Z_hat[:, 2:Np] .- V_hat[:, 2:Np] .+ Lambda_hat[:, 2:Np] .- e_bar .- w_bar)))
                X[:,Np+1] .= H_AN * (q_bar_N .+ rho * (Z_hat[:, Np] .- Theta_hat[:, Np]))
                BU, AX = B_bar * DU, A_bar * X[:, 1:Np]
                @. Z = (2.0 * (X[:, 2:Np+1] + Theta_hat) + BU + Beta_hat + AX + e_bar + w_bar - Lambda_hat) / 3.0
                @. Z = max(xmin_bar, min(xmax_bar, Z))
                @. V = 0.5 * (Z + BU + Beta_hat - AX - e_bar - w_bar + Lambda_hat)
                @. Theta = Theta_hat + X[:, 2:Np+1] - Z
                @. Beta = Beta_hat + BU - V
                @. Lambda = Lambda_hat + Z - AX - V - e_bar - w_bar
                res = rho * sum(norm(Theta[:,k] - Theta_hat[:,k])^2 + norm(Beta[:,k] - Beta_hat[:,k])^2 +
                               norm(Lambda[:,k] - Lambda_hat[:,k])^2 + norm(Z[:,k] - Z_hat[:,k])^2 +
                               norm(V[:,k] - V_hat[:,k])^2 + norm((Z[:,k] - V[:,k]) - (Z_hat[:,k] - V_hat[:,k]))^2
                               for k in 1:Np)
            end
        else
            @views begin
                DU .= J_B * (V - Beta)
                X[:,2:Np] .= H_A * (q_bar .+ rho .* (Z[:, 1:Np-1] .- Theta[:, 1:Np-1] .+
                              A_bar' * (Z[:, 2:Np] .- V[:, 2:Np] .+ Lambda[:, 2:Np] .- e_bar .- w_bar)))
                X[:,Np+1] .= H_AN * (q_bar_N .+ rho * (Z[:, Np] .- Theta[:, Np]))
                BU, AX = B_bar * DU, A_bar * X[:, 1:Np]
                @. Z = (2.0 * (X[:, 2:Np+1] + Theta) + BU + Beta + AX + e_bar + w_bar - Lambda) / 3.0
                @. Z = max(xmin_bar, min(xmax_bar, Z))
                @. V = 0.5 * (Z + BU + Beta - AX - e_bar - w_bar + Lambda)
                @. Theta = Theta + X[:, 2:Np+1] - Z
                @. Beta = Beta + BU - V
                @. Lambda = Lambda + Z - AX - V - e_bar - w_bar
                res = rho * sum(norm(Theta[:,k] - Theta_prev[:,k])^2 + norm(Beta[:,k] - Beta_prev[:,k])^2 +
                               norm(Lambda[:,k] - Lambda_prev[:,k])^2 + norm(Z[:,k] - Z_prev[:,k])^2 +
                               norm(V[:,k] - V_prev[:,k])^2 + norm((Z[:,k] - V[:,k]) - (Z_prev[:,k] - V_prev[:,k]))^2
                               for k in 1:Np)
            end
        end

        push!(residuals, res)
        verbose && @printf("  %6d  %12.4e\n", iter, res)

        if res < tol
            converged = true
            break
        end

        if m.accel
            if res < eta * res_prev
                alpha = 0.5*(1 + sqrt(1+4*alpha_prev^2))
                @. V_hat = V + (alpha_prev-1)/alpha * (V - V_prev)
                @. Z_hat = Z + (alpha_prev-1)/alpha * (Z - Z_prev)
                @. Theta_hat = Theta + (alpha_prev-1)/alpha * (Theta - Theta_prev)
                @. Beta_hat = Beta + (alpha_prev-1)/alpha * (Beta - Beta_prev)
                @. Lambda_hat = Lambda + (alpha_prev-1)/alpha * (Lambda - Lambda_prev)
                res_prev = res
            else
                alpha = 1.0
                V_hat .= V; Z_hat .= Z; Theta_hat .= Theta; Beta_hat .= Beta; Lambda_hat .= Lambda
                res_prev = res_prev / eta
            end
            alpha_prev = alpha
        end
    end
    solve_time = time() - t_start

    verbose && println("-" ^ 40)
    verbose && @printf("  Status: %s\n", converged ? "Converged" : "Not converged")
    verbose && @printf("  Iterations: %d\n", length(residuals))
    verbose && @printf("  Time: %.4f ms\n", solve_time * 1000)
    verbose && println()

    # Extract results
    X = E_inv * X
    x_traj = X[1:nx, :]
    u_traj = X[nx+1:end, 2:end]

    info = (
        solve_time = solve_time,
        iterations = length(residuals),
        converged = converged,
        obj_val = isempty(residuals) ? Inf : residuals[end]
    )
    warm = (DU, X, V, Z, Theta, Beta, Lambda)

    return x_traj, u_traj, DU, info, warm
end

function _solve_gpu(m::Model, x0::Vector, u0::Vector, yref::Vector, uref::Vector,
                    w::Vector; warm_vars=nothing, verbose::Bool=false)
    nx, nu, ny, Np = m.nx, m.nu, m.ny, m.Np
    nx_bar = nx + nu
    T = Float32

    A_bar = T.([m.A m.B; zeros(nu, nx) I(nu)])
    B_bar = T.([m.B; I(nu)])
    C_bar = T.([m.C zeros(ny, nu)])
    e_bar = T.([m.e; zeros(nu)])
    w_bar = T.([w; zeros(nu)])
    xmin_bar = T.([m.xmin; m.umin])
    xmax_bar = T.([m.xmax; m.umax])

    rho, tol, max_iter, eta = T(m.rho), T(m.tol), m.maxiter, T(m.eta)

    # Precompute constants (CUDA 3.x compatible)
    const_half = T(0.5)
    const_one = T(1)
    const_two = T(2)
    const_three = T(3)
    const_four = T(4)

    if m.precond
        E = sqrt.(diag(A_bar'*A_bar))
        E_diag, E_inv = Diagonal(E), Diagonal(const_one ./ E)
        A_bar = E_diag * A_bar * E_inv
        B_bar = E_diag * B_bar
        C_bar = C_bar * E_inv
        e_bar = E_diag * e_bar
        w_bar = E_diag * w_bar
        xmin_bar = E .* xmin_bar
        xmax_bar = E .* xmax_bar
    else
        E_diag = Matrix{T}(I, nx_bar, nx_bar)
        E_inv = Matrix{T}(I, nx_bar, nx_bar)
    end

    C_part = C_bar[:, 1:nx]
    Q_bar = [C_part' * T.(m.Wy) * C_part zeros(T, nx, nu); zeros(T, nu, nx) T.(m.Wu)]
    Q_bar_N = [C_part' * T.(m.Wf) * C_part zeros(T, nx, nu); zeros(T, nu, nx) T.(m.Wu)]
    q_bar = [C_part' * T.(m.Wy) * T.(yref); T.(m.Wu) * T.(uref)]
    q_bar_N = [C_part' * T.(m.Wf) * T.(yref); T.(m.Wu) * T.(uref)]
    R_bar = T.(m.Wdu)

    J_B = inv(R_bar + rho * B_bar'*B_bar) * B_bar'
    H_A = inv(Q_bar + rho * Matrix{T}(I, nx_bar, nx_bar) + rho * A_bar' * A_bar)
    H_AN = inv(Q_bar_N + rho * Matrix{T}(I, nx_bar, nx_bar))

    x_bar = T.([x0; u0])

    d_A_bar, d_B_bar = cu(A_bar), cu(B_bar)
    d_xmin_bar, d_xmax_bar = cu(xmin_bar), cu(xmax_bar)
    d_q_bar, d_q_bar_N, d_J_B, d_H_A, d_H_AN = cu(q_bar), cu(q_bar_N), cu(J_B), cu(H_A), cu(H_AN)
    d_e_bar, d_w_bar = cu(e_bar), cu(w_bar)

    if warm_vars === nothing
        d_DU = CUDA.zeros(T, nu, Np)
        d_X = CUDA.zeros(T, nx_bar, Np+1); d_X[:,1] = cu(E_diag * x_bar)
        d_V, d_Z = CUDA.zeros(T, nx_bar, Np), CUDA.zeros(T, nx_bar, Np)
        d_Theta, d_Beta, d_Lambda = CUDA.zeros(T, nx_bar, Np), CUDA.zeros(T, nx_bar, Np), CUDA.zeros(T, nx_bar, Np)
    else
        DU0, X0, V0, Z0, Theta0, Beta0, Lambda0 = warm_vars
        d_DU = cu(T.([DU0[:, 2:end] DU0[:, end]]))
        d_X = cu(T.([E_diag * x_bar X0[:, 3:end] X0[:, end]]))
        d_V = cu(T.([V0[:, 2:end] V0[:, end]]))
        d_Z = cu(T.([Z0[:, 2:end] Z0[:, end]]))
        d_Theta = cu(T.([Theta0[:, 2:end] Theta0[:, end]]))
        d_Beta = cu(T.([Beta0[:, 2:end] Beta0[:, end]]))
        d_Lambda = cu(T.([Lambda0[:, 2:end] Lambda0[:, end]]))
        d_X[:,2:Np+1] = cu(E_diag) * d_X[:,2:Np+1]
    end

    if m.accel
        d_V_hat, d_Z_hat = copy(d_V), copy(d_Z)
        d_Theta_hat, d_Beta_hat, d_Lambda_hat = copy(d_Theta), copy(d_Beta), copy(d_Lambda)
    end

    d_V_prev, d_Z_prev = copy(d_V), copy(d_Z)
    d_Theta_prev, d_Beta_prev, d_Lambda_prev = copy(d_Theta), copy(d_Beta), copy(d_Lambda)

    residuals = Float64[]
    alpha_prev, res_prev, res = const_one, const_one, T(Inf)
    converged = false

    verbose && println("piMPC ADMM Solver (GPU)")
    verbose && println("-" ^ 40)
    verbose && @printf("  %6s  %12s\n", "Iter", "Residual")
    verbose && println("-" ^ 40)

    t_start = time()
    for iter in 1:max_iter
        d_V_prev .= d_V; d_Z_prev .= d_Z
        d_Theta_prev .= d_Theta; d_Beta_prev .= d_Beta; d_Lambda_prev .= d_Lambda

        if m.accel
            @views begin
                d_DU .= d_J_B * (d_V_hat - d_Beta_hat)
                d_X[:,2:Np] .= d_H_A * (d_q_bar .+ rho .* (d_Z_hat[:, 1:Np-1] .- d_Theta_hat[:, 1:Np-1]) .+
                               d_A_bar' * (d_Z_hat[:, 2:Np] .- d_V_hat[:, 2:Np] .+ d_Lambda_hat[:, 2:Np] .- d_e_bar .- d_w_bar))
                d_X[:,Np+1] .= d_H_AN * (d_q_bar_N .+ rho .* (d_Z_hat[:, Np] .- d_Theta_hat[:, Np]))
                d_BU, d_AX = d_B_bar * d_DU, d_A_bar * d_X[:, 1:Np]
                @. d_Z = (const_two * (d_X[:, 2:Np+1] + d_Theta_hat) + d_BU + d_Beta_hat + d_AX + d_e_bar + d_w_bar - d_Lambda_hat) / const_three
                @. d_Z = max(d_xmin_bar, min(d_xmax_bar, d_Z))
                @. d_V = const_half * (d_Z + d_BU + d_Beta_hat - d_AX - d_e_bar - d_w_bar + d_Lambda_hat)
                @. d_Theta = d_Theta_hat + d_X[:, 2:Np+1] - d_Z
                @. d_Beta = d_Beta_hat + d_BU - d_V
                @. d_Lambda = d_Lambda_hat + d_Z - d_AX - d_V - d_e_bar - d_w_bar
                res = rho * (CUDA.norm(d_Theta - d_Theta_hat)^2 + CUDA.norm(d_Beta - d_Beta_hat)^2 +
                            CUDA.norm(d_Lambda - d_Lambda_hat)^2 + CUDA.norm(d_Z - d_Z_hat)^2 +
                            CUDA.norm(d_V - d_V_hat)^2 + CUDA.norm((d_Z - d_V) - (d_Z_hat - d_V_hat))^2)
            end
        else
            @views begin
                d_DU .= d_J_B * (d_V - d_Beta)
                d_X[:,2:Np] .= d_H_A * (d_q_bar .+ rho .* (d_Z[:, 1:Np-1] .- d_Theta[:, 1:Np-1]) .+
                               d_A_bar' * (d_Z[:, 2:Np] .- d_V[:, 2:Np] .+ d_Lambda[:, 2:Np] .- d_e_bar .- d_w_bar))
                d_X[:,Np+1] .= d_H_AN * (d_q_bar_N .+ rho * (d_Z[:, Np] .- d_Theta[:, Np]))
                d_BU, d_AX = d_B_bar * d_DU, d_A_bar * d_X[:, 1:Np]
                @. d_Z = (const_two * (d_X[:, 2:Np+1] + d_Theta) + d_BU + d_Beta + d_AX + d_e_bar + d_w_bar - d_Lambda) / const_three
                @. d_Z = max(d_xmin_bar, min(d_xmax_bar, d_Z))
                @. d_V = const_half * (d_Z + d_BU + d_Beta - d_AX - d_e_bar - d_w_bar + d_Lambda)
                @. d_Theta = d_Theta + d_X[:, 2:Np+1] - d_Z
                @. d_Beta = d_Beta + d_BU - d_V
                @. d_Lambda = d_Lambda + d_Z - d_AX - d_V - d_e_bar - d_w_bar
                res = rho * (CUDA.norm(d_Theta - d_Theta_prev)^2 + CUDA.norm(d_Beta - d_Beta_prev)^2 +
                            CUDA.norm(d_Lambda - d_Lambda_prev)^2 + CUDA.norm(d_Z - d_Z_prev)^2 +
                            CUDA.norm(d_V - d_V_prev)^2 + CUDA.norm((d_Z - d_V) - (d_Z_prev - d_V_prev))^2)
            end
        end

        push!(residuals, Float64(res))
        verbose && @printf("  %6d  %12.4e\n", iter, res)

        if res < tol
            converged = true
            break
        end

        if m.accel
            if res < eta * res_prev
                alpha = const_half * (const_one + sqrt(const_one + const_four * alpha_prev^2))
                momentum = (alpha_prev - const_one) / alpha
                @. d_V_hat = d_V + momentum * (d_V - d_V_prev)
                @. d_Z_hat = d_Z + momentum * (d_Z - d_Z_prev)
                @. d_Theta_hat = d_Theta + momentum * (d_Theta - d_Theta_prev)
                @. d_Beta_hat = d_Beta + momentum * (d_Beta - d_Beta_prev)
                @. d_Lambda_hat = d_Lambda + momentum * (d_Lambda - d_Lambda_prev)
                res_prev = res
            else
                alpha = const_one
                d_V_hat .= d_V; d_Z_hat .= d_Z; d_Theta_hat .= d_Theta; d_Beta_hat .= d_Beta; d_Lambda_hat .= d_Lambda
                res_prev = res / eta
            end
            alpha_prev = alpha
        end
    end
    solve_time = time() - t_start

    verbose && println("-" ^ 40)
    verbose && @printf("  Status: %s\n", converged ? "Converged" : "Not converged")
    verbose && @printf("  Iterations: %d\n", length(residuals))
    verbose && @printf("  Time: %.4f ms\n", solve_time * 1000)
    verbose && println()

    DU = Array(d_DU)
    X = E_inv * Array(d_X)
    V, Z = Array(d_V), Array(d_Z)
    Theta, Beta, Lambda = Array(d_Theta), Array(d_Beta), Array(d_Lambda)

    x_traj = X[1:nx, :]
    u_traj = X[nx+1:end, 2:end]

    info = (
        solve_time = solve_time,
        iterations = length(residuals),
        converged = converged,
        obj_val = isempty(residuals) ? Float64(Inf) : residuals[end]
    )
    warm = (DU, X, V, Z, Theta, Beta, Lambda)

    return x_traj, u_traj, DU, info, warm
end
