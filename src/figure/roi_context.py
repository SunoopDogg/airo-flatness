"""ROI context view — full point cloud with ROI bounding box overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from figure.roi_selector import filter_points_by_roi


def _subsample_points(
    points: np.ndarray,
    max_points: int,
    seed: int,
    colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Subsample points if count exceeds max_points.

    Args:
        points: (N, 3) array.
        max_points: maximum number of points to keep.
        seed: random seed for reproducibility.
        colors: optional (N, 3) RGB array to subsample alongside points.

    Returns:
        Tuple of (points, colors) where points is (M, 3) with M <= max_points,
        and colors is (M, 3) or None.
    """
    if len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), max_points, replace=False)
    sub_colors = colors[idx] if colors is not None else None
    return points[idx], sub_colors


def _build_roi_rectangle_2d(
    roi: tuple[float, float, float, float],
    z: float,
) -> pv.PolyData:
    """Build a 2D rectangle outline on the XY plane at given Z.

    Args:
        roi: (x_min, x_max, y_min, y_max).
        z: Z coordinate for the rectangle.

    Returns:
        PyVista PolyData with line cells forming a rectangle.
    """
    x_min, x_max, y_min, y_max = roi
    corners = np.array([
        [x_min, y_min, z],
        [x_max, y_min, z],
        [x_max, y_max, z],
        [x_min, y_max, z],
    ])
    lines = np.array([
        [2, 0, 1],
        [2, 1, 2],
        [2, 2, 3],
        [2, 3, 0],
    ])
    return pv.PolyData(corners, lines=lines.ravel())


def _create_point_cloud_plotter(
    pts: np.ndarray,
    colors: np.ndarray | None = None,
) -> pv.Plotter:
    """Create an off-screen plotter with the point cloud added.

    Args:
        pts: (N, 3) subsampled point array.
        colors: optional (N, 3) RGB float array in [0, 1]. When provided,
            renders with RGBA scalars instead of viridis Z colormap.

    Returns:
        Configured PyVista plotter with point cloud mesh added.
    """
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")
    plotter.window_size = (1920, 1080)

    cloud = pv.PolyData(pts)

    if colors is not None:
        rgba = np.empty((len(colors), 4), dtype=np.uint8)
        rgba[:, :3] = (colors * 255).astype(np.uint8)
        rgba[:, 3] = 255
        cloud["RGBA"] = rgba
        plotter.add_mesh(cloud, scalars="RGBA", rgba=True, point_size=1.0, render_points_as_spheres=False)
    else:
        cloud["Z"] = pts[:, 2]
        plotter.add_mesh(
            cloud,
            scalars="Z",
            cmap="viridis",
            point_size=1.0,
            render_points_as_spheres=False,
        )

    return plotter


def render_roi_context_2d(
    points: np.ndarray,
    roi: tuple[float, float, float, float],
    save_dir: Path,
    max_points: int = 500_000,
    seed: int = 42,
    dpi: int = 300,
    colors: np.ndarray | None = None,
    filename: str | None = None,
) -> None:
    """Render full point cloud with 2D ROI rectangle overlay on XY plane.

    The rectangle is placed at the Z median of all points.
    Saved as roi_context_2d.png in isometric view.

    Args:
        points: (N, 3) full point cloud.
        roi: (x_min, x_max, y_min, y_max) selected ROI bounds.
        save_dir: directory to save the output image.
        max_points: subsample limit for rendering performance.
        seed: random seed for subsampling.
        dpi: unused (PyVista uses window_size), kept for API consistency.
        colors: optional (N, 3) RGB float array in [0, 1] for per-point color.
        filename: output filename; defaults to "roi_context_2d.png".
    """
    pts, sub_colors = _subsample_points(points, max_points, seed, colors=colors)

    plotter = _create_point_cloud_plotter(pts, colors=sub_colors)

    z_median = float(np.median(pts[:, 2]))
    rect = _build_roi_rectangle_2d(roi, z_median)
    plotter.add_mesh(rect, color="red", style="wireframe", line_width=3)

    plotter.add_axes()
    plotter.view_isometric()
    if filename is None:
        out_filename = "roi_context_2d_rgb.png" if colors is not None else "roi_context_2d.png"
    else:
        out_filename = filename
    plotter.screenshot(str(save_dir / out_filename))
    plotter.close()


def render_roi_context_3d(
    points: np.ndarray,
    roi: tuple[float, float, float, float],
    save_dir: Path,
    max_points: int = 500_000,
    seed: int = 42,
    dpi: int = 300,
    colors: np.ndarray | None = None,
    filename: str | None = None,
) -> None:
    """Render full point cloud with 3D ROI wireframe box overlay.

    The box uses ROI XY bounds and Z min/max of points within the ROI.
    Saved as roi_context_3d.png in isometric view.

    Args:
        points: (N, 3) full point cloud.
        roi: (x_min, x_max, y_min, y_max) selected ROI bounds.
        save_dir: directory to save the output image.
        max_points: subsample limit for rendering performance.
        seed: random seed for subsampling.
        dpi: unused (PyVista uses window_size), kept for API consistency.
        colors: optional (N, 3) RGB float array in [0, 1] for per-point color.
        filename: output filename; defaults to "roi_context_3d.png".
    """
    pts, sub_colors = _subsample_points(points, max_points, seed, colors=colors)

    plotter = _create_point_cloud_plotter(pts, colors=sub_colors)

    roi_points = filter_points_by_roi(pts, roi)
    if len(roi_points) > 0:
        z_min = float(roi_points[:, 2].min())
        z_max = float(roi_points[:, 2].max())
    else:
        z_min = float(pts[:, 2].min())
        z_max = float(pts[:, 2].max())

    x_min, x_max, y_min, y_max = roi
    box = pv.Box((x_min, x_max, y_min, y_max, z_min, z_max))
    plotter.add_mesh(box, color="red", style="wireframe", line_width=3)

    plotter.add_axes()
    plotter.view_isometric()
    if filename is None:
        out_filename = "roi_context_3d_rgb.png" if colors is not None else "roi_context_3d.png"
    else:
        out_filename = filename
    plotter.screenshot(str(save_dir / out_filename))
    plotter.close()
