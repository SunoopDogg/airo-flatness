"""3D surface mesh via griddata interpolation + PyVista interactive viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
import pyvista as pv


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
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), max_points, replace=False)
        pts = points[idx].copy()
    else:
        pts = points.copy()

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


def show_surface_viewer(
    mesh: pv.StructuredGrid,
    save_dir: Path,
    dpi: int = 300,
) -> None:
    """Launch interactive PyVista viewer with capture keys.

    Key bindings:
        S — capture current view (PNG)
        I — isometric view + capture (PNG)
        M — 5-direction multi-capture (PNG)
        Q — quit
    """
    plotter = pv.Plotter(title="Surface Roughness — 3D View")
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars="Z",
        cmap="coolwarm",
        show_edges=False,
    )
    plotter.add_axes()

    window_size = (1920, 1080)
    plotter.window_size = window_size

    def capture_current():
        path = save_dir / "surface_3d_capture.png"
        plotter.screenshot(str(path))
        print(f"  Saved: {path}")

    def capture_isometric():
        plotter.view_isometric()
        plotter.render()
        path = save_dir / "surface_3d_isometric.png"
        plotter.screenshot(str(path))
        print(f"  Saved: {path}")

    def capture_multi():
        views = {
            "top": lambda: plotter.view_xy(),
            "front": lambda: plotter.view_xz(),
            "back": lambda: plotter.view_vector((0, 1, 0), viewup=(0, 0, 1)),
            "right": lambda: plotter.view_yz(),
            "left": lambda: plotter.view_vector((1, 0, 0), viewup=(0, 0, 1)),
        }
        print("  Capturing 5 views...")
        for name, set_view in views.items():
            set_view()
            plotter.render()
            path = save_dir / f"surface_3d_{name}.png"
            plotter.screenshot(str(path))
            print(f"    [{name}] saved: {path}")
        plotter.view_isometric()
        plotter.render()
        print("  All 5 captures complete.")

    plotter.add_key_event("S", capture_current)
    plotter.add_key_event("I", capture_isometric)
    plotter.add_key_event("M", capture_multi)

    print("  Keys: [S] capture, [I] isometric, [M] multi-view, [Q] quit")
    plotter.show()
