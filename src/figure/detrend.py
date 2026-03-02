"""Plane fitting and Z-detrending for roughness analysis."""

import numpy as np


def fit_plane(points: np.ndarray) -> tuple[float, float, float]:
    """Fit a plane Z = aX + bY + c to (N,3) points using least squares.

    Returns:
        (a, b, c) plane coefficients.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    A = np.column_stack([x, y, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def detrend_points(points: np.ndarray) -> np.ndarray:
    """Remove planar trend from points, returning Z residuals.

    Args:
        points: (N, 3) array of XYZ coordinates.

    Returns:
        (N,) array of Z residuals after subtracting fitted plane.
    """
    a, b, c = fit_plane(points)
    z_fitted = a * points[:, 0] + b * points[:, 1] + c
    return points[:, 2] - z_fitted
