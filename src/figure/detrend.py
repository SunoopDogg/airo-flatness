"""Plane fitting and Z-detrending for roughness analysis."""

import numpy as np

try:
    import cupy
    _get_xp = cupy.get_array_module
except ImportError:
    _get_xp = lambda *_: np


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
