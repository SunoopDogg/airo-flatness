"""Plane fitting and Z-detrending for roughness analysis."""

import numpy as np

_CUPY_COMPUTE_OK = False
try:
    import cupy
    # Verify actual GPU compute works (not just import)
    _test = cupy.array([1.0, 2.0])
    _ = _test @ _test
    _CUPY_COMPUTE_OK = True
    del _test
except Exception:
    pass


def _get_xp(*arrays):
    """Return cupy if all arrays are CuPy and GPU compute works, else numpy."""
    if not _CUPY_COMPUTE_OK:
        return np
    return cupy.get_array_module(*arrays)


def fit_plane(points):
    """Fit a plane Z = aX + bY + c to (N,3) points using normal equations.

    Sums are computed on-device (GPU-accelerated for CuPy inputs), but the
    tiny 3x3 solve always runs on CPU to avoid cusolver dependency.
    Coefficients are returned as Python floats so they work with both
    NumPy and CuPy arrays in downstream arithmetic.
    """
    xp = _get_xp(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Compute sums on device (fast for large N)
    sx  = float(x.sum())
    sy  = float(y.sum())
    n   = float(len(x))
    sxx = float(x @ x)
    sxy = float(x @ y)
    syy = float(y @ y)
    xz  = float(x @ z)
    yz  = float(y @ z)
    sz  = float(z.sum())

    # Solve the 3x3 normal-equation system on CPU (zero benefit from GPU)
    ATA = np.array([
        [sxx, sxy, sx],
        [sxy, syy, sy],
        [sx,  sy,  n ],
    ], dtype=np.float64)
    ATz = np.array([xz, yz, sz], dtype=np.float64)

    try:
        coeffs = np.linalg.solve(ATA, ATz)
    except np.linalg.LinAlgError:
        A = np.column_stack([
            x.get() if xp is not np else x,
            y.get() if xp is not np else y,
            np.ones(int(n)),
        ])
        z_cpu = z.get() if xp is not np else z
        coeffs, *_ = np.linalg.lstsq(A, z_cpu, rcond=None)

    return coeffs[0], coeffs[1], coeffs[2]


def detrend_points(points):
    """Remove planar trend from points, returning Z residuals.

    Args:
        points: (N, 3) array (NumPy or CuPy).

    Returns:
        (N,) array of Z residuals (same type as input).
    """
    a, b, c = fit_plane(points)
    return points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c)
