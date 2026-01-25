import numpy as np

from .utils import cupy_available, normalize_device, warn_gpu_fallback


class Model:
    def __init__(self, dtype=np.float64):
        self.dtype = dtype

        # Problem data
        self.A = np.empty((0, 0), dtype=dtype)
        self.B = np.empty((0, 0), dtype=dtype)
        self.C = np.empty((0, 0), dtype=dtype)
        self.e = np.empty((0,), dtype=dtype)
        self.nx = 0
        self.nu = 0
        self.ny = 0
        self.Np = 0

        # Weights
        self.Wy = np.empty((0, 0), dtype=dtype)
        self.Wu = np.empty((0, 0), dtype=dtype)
        self.Wdu = np.empty((0, 0), dtype=dtype)
        self.Wf = np.empty((0, 0), dtype=dtype)

        # Constraints
        self.xmin = np.empty((0,), dtype=dtype)
        self.xmax = np.empty((0,), dtype=dtype)
        self.umin = np.empty((0,), dtype=dtype)
        self.umax = np.empty((0,), dtype=dtype)
        self.dumin = np.empty((0,), dtype=dtype)
        self.dumax = np.empty((0,), dtype=dtype)

        # Settings
        self.rho = 1.0
        self.tol = 1e-4
        self.eta = 0.999
        self.maxiter = 100
        self.precond = False
        self.accel = False
        self.device = "cpu"

        # State
        self.is_setup = False
        self.warm_vars = None

    def setup(
        self,
        *,
        A,
        B,
        Np,
        C=None,
        e=None,
        Wy=None,
        Wu=None,
        Wdu=None,
        Wf=None,
        xmin=None,
        xmax=None,
        umin=None,
        umax=None,
        dumin=None,
        dumax=None,
        rho=1.0,
        tol=1e-4,
        eta=0.999,
        maxiter=100,
        precond=False,
        accel=False,
        device="cpu",
    ):
        A = np.asarray(A, dtype=self.dtype)
        B = np.asarray(B, dtype=self.dtype)
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be 2D arrays.")
        nx, nu = B.shape
        if A.shape != (nx, nx):
            raise ValueError("A must be square and compatible with B.")

        if C is None:
            C = np.eye(nx, dtype=self.dtype)
        else:
            C = np.asarray(C, dtype=self.dtype)
        if C.ndim != 2 or C.shape[1] != nx:
            raise ValueError("C must be 2D with shape (ny, nx).")
        ny = C.shape[0]

        if e is None:
            e = np.zeros(nx, dtype=self.dtype)
        else:
            e = np.asarray(e, dtype=self.dtype).reshape(-1)
        if e.shape != (nx,):
            raise ValueError("e must have shape (nx,).")

        def _mat_or_eye(val, size, name):
            if val is None:
                return np.eye(size, dtype=self.dtype)
            arr = np.asarray(val, dtype=self.dtype)
            if arr.shape != (size, size):
                raise ValueError(f"{name} must have shape ({size}, {size}).")
            return arr

        def _vec_or_fill(val, size, fill, name):
            if val is None:
                return np.full(size, fill, dtype=self.dtype)
            arr = np.asarray(val, dtype=self.dtype).reshape(-1)
            if arr.shape != (size,):
                raise ValueError(f"{name} must have shape ({size},).")
            return arr

        Wy = _mat_or_eye(Wy, ny, "Wy")
        Wu = _mat_or_eye(Wu, nu, "Wu")
        Wdu = _mat_or_eye(Wdu, nu, "Wdu")
        Wf = Wy if Wf is None else _mat_or_eye(Wf, ny, "Wf")

        xmin = _vec_or_fill(xmin, nx, -np.inf, "xmin")
        xmax = _vec_or_fill(xmax, nx, np.inf, "xmax")
        umin = _vec_or_fill(umin, nu, -np.inf, "umin")
        umax = _vec_or_fill(umax, nu, np.inf, "umax")
        dumin = _vec_or_fill(dumin, nu, -np.inf, "dumin")
        dumax = _vec_or_fill(dumax, nu, np.inf, "dumax")

        device = normalize_device(device)
        if device == "gpu" and not cupy_available():
            warn_gpu_fallback()
            device = "cpu"

        self.A = A
        self.B = B
        self.C = C
        self.e = e
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.Np = int(Np)

        self.Wy = Wy
        self.Wu = Wu
        self.Wdu = Wdu
        self.Wf = Wf

        self.xmin = xmin
        self.xmax = xmax
        self.umin = umin
        self.umax = umax
        self.dumin = dumin
        self.dumax = dumax

        self.rho = float(rho)
        self.tol = float(tol)
        self.eta = float(eta)
        self.maxiter = int(maxiter)
        self.precond = bool(precond)
        self.accel = bool(accel)
        self.device = device

        self.is_setup = True
        self.warm_vars = None

        return None

    def solve(self, x0, u0, yref, uref, w, *, verbose=False):
        from .solver import solve

        return solve(self, x0, u0, yref, uref, w, verbose=verbose)


def setup(model, **kwargs):
    return model.setup(**kwargs)
