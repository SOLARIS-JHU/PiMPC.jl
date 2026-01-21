using Test
using PiMPC
using LinearAlgebra

@testset "PiMPC" begin
    @testset "Model + setup!" begin
        A = [1.0 0.1; 0.0 1.0]
        B = reshape([0.005; 0.1], :, 1)
        nx, nu = size(B)

        model = Model()
        setup!(model;
            A = A, B = B, Np = 20,
            umin = [-1.0], umax = [1.0],
            rho = 1.0, tol = 1e-4, maxiter = 100
        )

        @test model.is_setup == true
        @test model.nx == 2
        @test model.nu == 1
        @test model.Np == 20
    end

    @testset "solve!" begin
        A = [1.0 0.1; 0.0 1.0]
        B = reshape([0.005; 0.1], :, 1)
        nx, nu = size(B)

        model = Model()
        setup!(model;
            A = A, B = B, Np = 20,
            umin = [-1.0], umax = [1.0],
            rho = 1.0, tol = 1e-4, maxiter = 100
        )

        x0, u0 = [1.0, 0.0], [0.0]
        yref, uref, w = zeros(nx), zeros(nu), zeros(nx)

        results = solve!(model, x0, u0, yref, uref, w)

        @test size(results.x) == (2, 21)
        @test size(results.u) == (1, 20)
        @test size(results.du) == (1, 20)
        @test results.info.iterations > 0
        @test results.info.solve_time > 0
    end

    @testset "warm start (automatic)" begin
        A = [1.0 0.1; 0.0 1.0]
        B = reshape([0.005; 0.1], :, 1)
        nx, nu = size(B)

        model = Model()
        setup!(model; A = A, B = B, Np = 20, rho = 1.0, tol = 1e-6)

        x0, u0 = [1.0, 0.0], [0.0]
        yref, uref, w = zeros(nx), zeros(nu), zeros(nx)

        results1 = solve!(model, x0, u0, yref, uref, w)
        @test model.warm_vars !== nothing

        # Second solve uses warm start automatically
        results2 = solve!(model, [0.9, 0.05], u0, yref, uref, w)
        @test results2.info.iterations > 0
    end

    @testset "acceleration" begin
        A = [1.0 0.1; 0.0 1.0]
        B = reshape([0.005; 0.1], :, 1)
        nx, nu = size(B)

        model = Model()
        setup!(model;
            A = A, B = B, Np = 20,
            umin = [-1.0], umax = [1.0],
            rho = 10.0, accel = true, maxiter = 500
        )

        x0, u0 = [5.0, 0.0], [0.0]
        yref, uref, w = zeros(nx), zeros(nu), zeros(nx)

        results = solve!(model, x0, u0, yref, uref, w)

        @test size(results.x) == (2, 21)
        @test results.info.iterations > 0
    end

    @testset "output matrix C" begin
        A = [1.0 0.1; 0.0 1.0]
        B = reshape([0.005; 0.1], :, 1)
        C = [1.0 0.0]  # Only observe first state
        nx, nu = size(B)
        ny = size(C, 1)

        model = Model()
        setup!(model;
            A = A, B = B, C = C, Np = 20,
            umin = [-1.0], umax = [1.0]
        )

        @test model.ny == 1

        x0, u0 = [1.0, 0.0], [0.0]
        yref, uref, w = zeros(ny), zeros(nu), zeros(nx)

        results = solve!(model, x0, u0, yref, uref, w)
        @test size(results.x) == (2, 21)
    end

    @testset "affine term e" begin
        A = [1.0 0.1; 0.0 1.0]
        B = reshape([0.005; 0.1], :, 1)
        e = [0.01, 0.0]  # Constant drift
        nx, nu = size(B)

        model = Model()
        setup!(model;
            A = A, B = B, e = e, Np = 20,
            umin = [-1.0], umax = [1.0]
        )

        x0, u0 = [1.0, 0.0], [0.0]
        yref, uref, w = zeros(nx), zeros(nu), zeros(nx)

        results = solve!(model, x0, u0, yref, uref, w)
        @test size(results.x) == (2, 21)
    end
end
