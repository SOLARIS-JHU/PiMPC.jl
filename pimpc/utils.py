import warnings


def normalize_device(device):
    if device is None:
        return "cpu"
    if isinstance(device, str):
        value = device.lower()
    else:
        value = str(device).lower()
    value = value.lstrip(":")
    if value not in ("cpu", "gpu"):
        raise ValueError(f"Unsupported device '{device}'. Use 'cpu' or 'gpu'.")
    return value


def cupy_available():
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return True


def ensure_cupy():
    try:
        import cupy as cp
    except Exception as exc:
        raise RuntimeError("CuPy is required for GPU execution.") from exc
    return cp


def warn_gpu_fallback():
    warnings.warn("GPU not available, falling back to CPU", RuntimeWarning, stacklevel=2)
