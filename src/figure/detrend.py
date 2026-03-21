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

    Returns coefficients as array elements (not Python floats) to keep
    GPU arrays on device when input is CuPy.
    """
    xp = _get_xp(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    sx, sy, n = x.sum(), y.sum(), xp.float64(len(x))
    sxx, sxy, syy = x @ x, x @ y, y @ y

    ATA = xp.array([
        [sxx, sxy, sx],
        [sxy, syy, sy],
        [sx,  sy,  n ],
    ], dtype=xp.float64)
    ATz = xp.array([x @ z, y @ z, z.sum()], dtype=xp.float64)

    try:
        coeffs = xp.linalg.solve(ATA, ATz)
    except Exception:
        A = xp.column_stack([x, y, xp.ones(len(x), dtype=x.dtype)])
        coeffs, *_ = xp.linalg.lstsq(A, z, rcond=None)

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
