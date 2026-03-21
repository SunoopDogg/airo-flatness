"""3D surface mesh via griddata interpolation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata
import pyvista as pv

from utils import subsample_points


def build_surface_mesh(
    points: np.ndarray,
    max_points: int = 500_000,
    grid_resolution: int = 200,
    z_exaggeration: float = 1.0,
    seed: int = 42,
) -> pv.StructuredGrid:
    """Build an interpolated surface mesh from (N,3) points.

    Args:
        points: (N, 3) XYZ array.
        max_points: downsample if exceeding this count.
        grid_resolution: number of grid cells along the longer axis.
        z_exaggeration: multiply Z values for visual emphasis.
        seed: random seed for downsampling.

    Returns:
        PyVista StructuredGrid mesh.
    """
    pts, = subsample_points(points, max_points, seed)
    pts = pts.copy()

    if z_exaggeration != 1.0:
        z_mean = pts[:, 2].mean()
        pts[:, 2] = z_mean + (pts[:, 2] - z_mean) * z_exaggeration

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    x_span = x_max - x_min
    y_span = y_max - y_min
    longer = max(x_span, y_span)
    if longer == 0:
        raise ValueError(
            "Point cloud has zero spatial extent — cannot build surface mesh."
        )
    cell = longer / grid_resolution
    nx = max(int(x_span / cell), 2)
    ny = max(int(y_span / cell), 2)

    grid_x = np.linspace(x_min, x_max, nx)
    grid_y = np.linspace(y_min, y_max, ny)
    gx, gy = np.meshgrid(grid_x, grid_y)

    gz = griddata(pts[:, :2], pts[:, 2], (gx, gy), method="cubic")

    # Fill NaN holes with nearest-neighbor interpolation
    nan_mask = np.isnan(gz)
    if nan_mask.any():
        gz_nearest = griddata(pts[:, :2], pts[:, 2], (gx, gy), method="nearest")
        gz[nan_mask] = gz_nearest[nan_mask]

    # Clamp to input Z range to prevent cubic extrapolation artifacts
    np.clip(gz, pts[:, 2].min(), pts[:, 2].max(), out=gz)

    mesh = pv.StructuredGrid(gx, gy, gz)
    mesh["Z"] = gz.ravel(order="F")

    return mesh
