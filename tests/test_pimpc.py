import numpy as np

from pimpc import Model


def test_model_setup():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.005], [0.1]])

    model = Model()
    model.setup(A=A, B=B, Np=20, umin=[-1.0], umax=[1.0], rho=1.0, tol=1e-4, maxiter=100)

    assert model.is_setup is True
    assert model.nx == 2
    assert model.nu == 1
    assert model.Np == 20


def test_solve():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.005], [0.1]])

    model = Model()
    model.setup(A=A, B=B, Np=20, umin=[-1.0], umax=[1.0], rho=1.0, tol=1e-4, maxiter=100)

    x0 = np.array([1.0, 0.0])
    u0 = np.array([0.0])
    yref = np.zeros(2)
    uref = np.zeros(1)
    w = np.zeros(2)

    results = model.solve(x0, u0, yref, uref, w)

    assert results.x.shape == (2, 21)
    assert results.u.shape == (1, 20)
    assert results.du.shape == (1, 20)
    assert results.info.iterations > 0
    assert results.info.solve_time > 0


def test_warm_start():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.005], [0.1]])

    model = Model()
    model.setup(A=A, B=B, Np=20, rho=1.0, tol=1e-6)

    x0 = np.array([1.0, 0.0])
    u0 = np.array([0.0])
    yref = np.zeros(2)
    uref = np.zeros(1)
    w = np.zeros(2)

    _ = model.solve(x0, u0, yref, uref, w)
    assert model.warm_vars is not None

    results2 = model.solve(np.array([0.9, 0.05]), u0, yref, uref, w)
    assert results2.info.iterations > 0


def test_acceleration():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.005], [0.1]])

    model = Model()
    model.setup(A=A, B=B, Np=20, umin=[-1.0], umax=[1.0], rho=10.0, accel=True, maxiter=200)

    x0 = np.array([5.0, 0.0])
    u0 = np.array([0.0])
    yref = np.zeros(2)
    uref = np.zeros(1)
    w = np.zeros(2)

    results = model.solve(x0, u0, yref, uref, w)

    assert results.x.shape == (2, 21)
    assert results.info.iterations > 0


def test_output_matrix_c():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.005], [0.1]])
    C = np.array([[1.0, 0.0]])

    model = Model()
    model.setup(A=A, B=B, C=C, Np=20, umin=[-1.0], umax=[1.0])

    assert model.ny == 1

    x0 = np.array([1.0, 0.0])
    u0 = np.array([0.0])
    yref = np.zeros(1)
    uref = np.zeros(1)
    w = np.zeros(2)

    results = model.solve(x0, u0, yref, uref, w)
    assert results.x.shape == (2, 21)


def test_affine_term_e():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.005], [0.1]])
    e = np.array([0.01, 0.0])

    model = Model()
    model.setup(A=A, B=B, e=e, Np=20, umin=[-1.0], umax=[1.0])

    x0 = np.array([1.0, 0.0])
    u0 = np.array([0.0])
    yref = np.zeros(2)
    uref = np.zeros(1)
    w = np.zeros(2)

    results = model.solve(x0, u0, yref, uref, w)
    assert results.x.shape == (2, 21)
