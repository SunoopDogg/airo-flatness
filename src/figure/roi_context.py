"""ROI context view — full point cloud with ROI bounding box overlay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pyvista as pv

from figure.roi_selector import QuadROI, filter_points_by_roi
from utils import subsample_points, to_rgba

ROI = tuple[float, float, float, float]

SCALAR_BAR_ARGS = {
    "title": "Z (m)",
    "vertical": True,
    "position_x": 0.02,
    "position_y": 0.05,
    "width": 0.05,
    "height": 0.35,
}

AXES_WIDGET_VIEWPORT = (0.08, 0.0, 0.22, 0.2)  # (xmin, ymin, xmax, ymax) normalized


def _clip_mesh_to_roi(
    mesh: pv.StructuredGrid,
    roi: ROI,
    z_range: tuple[float, float] | None = None,
) -> pv.PolyData | None:
    """Clip a surface mesh to the ROI polygon and Z range.

    Args:
        mesh: surface mesh to clip.
        roi: (x_min, x_max, y_min, y_max) tuple.
        z_range: optional (z_min, z_max) for Z clipping.
            Falls back to mesh Z extent when None.

    Returns:
        Clipped PolyData, or None if no cells remain.
    """
    if z_range is not None:
        z_lo, z_hi = float(z_range[0]), float(z_range[1])
    else:
        z_lo = float(mesh.points[:, 2].min())
        z_hi = float(mesh.points[:, 2].max())

    x_min, x_max, y_min, y_max = roi

    surface = mesh.extract_surface()
    clipped = surface.clip_box(
        [x_min, x_max, y_min, y_max, z_lo, z_hi], invert=False,
    )

    if clipped.n_points == 0:
        return None
    return clipped


ROI_COLOR = (0.0, 0.6, 0.0)
ROI_LABEL = "Region of Interest (ROI)"
ROI_LINE_WIDTH = 3
ROI_DASH_COUNT = 30
ROI_DASH_RATIO = 0.6


def _build_dashed_segments(
    p0: np.ndarray,
    p1: np.ndarray,
    num_dashes: int = ROI_DASH_COUNT,
    dash_ratio: float = ROI_DASH_RATIO,
) -> tuple[np.ndarray, list[list[int]]]:
    """Build dashed line segments between two 3D points.

    Returns:
        Tuple of (points array, line connectivity list).
    """
    points = []
    lines = []
    for i in range(num_dashes):
        t_start = i / num_dashes
        t_end = t_start + dash_ratio / num_dashes
        idx = len(points)
        points.append(p0 + t_start * (p1 - p0))
        points.append(p0 + t_end * (p1 - p0))
        lines.append([2, idx, idx + 1])
    return np.array(points), lines


def _assemble_dashed_polydata(
    verts: np.ndarray,
    edges: list[tuple[int, int]],
) -> pv.PolyData:
    """Assemble dashed line segments along edges into a single PolyData.

    Args:
        verts: (M, 3) vertex array.
        edges: list of (i, j) index pairs into verts.

    Returns:
        PyVista PolyData with dashed line cells.
    """
    all_points: list[np.ndarray] = []
    all_lines: list[list[int]] = []
    for i0, i1 in edges:
        pts, lines = _build_dashed_segments(verts[i0], verts[i1])
        offset = len(all_points)
        all_points.extend(pts)
        for ln in lines:
            all_lines.append([2, ln[1] + offset, ln[2] + offset])
    return pv.PolyData(np.array(all_points), lines=np.array(all_lines).ravel())


def _build_dashed_rectangle_2d(
    roi: ROI,
    z: float,
) -> pv.PolyData:
    """Build a dashed 2D quadrilateral outline on the XY plane at given Z."""
    x_min, x_max, y_min, y_max = roi
    corners_2d = np.array([
        [x_min, y_min], [x_max, y_min],
        [x_max, y_max], [x_min, y_max],
    ])
    verts = np.column_stack([corners_2d, np.full(4, z)])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return _assemble_dashed_polydata(verts, edges)


def _build_dashed_box_3d(
    roi: ROI,
    z_min: float,
    z_max: float,
) -> pv.PolyData:
    """Build a dashed 3D wireframe box."""
    x_min, x_max, y_min, y_max = roi
    corners_2d = np.array([
        [x_min, y_min], [x_max, y_min],
        [x_max, y_max], [x_min, y_max],
    ])
    bottom = np.column_stack([corners_2d, np.full(4, z_min)])
    top = np.column_stack([corners_2d, np.full(4, z_max)])
    verts = np.vstack([bottom, top])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return _assemble_dashed_polydata(verts, edges)


ROI_ZOOM = 2.0
ROI_PAN_DOWN_RATIO = 0.6  # fraction of scene height to pan downward after zoom


def _add_roi_legend(plotter: pv.Plotter) -> None:
    """Add ROI legend text in upper-right corner with academic styling.

    Uses VTK text actor directly to support background box, since
    PyVista 0.47 does not expose background_color in add_text().
    """
    import vtkmodules.vtkRenderingCore as vtk_rc

    actor = vtk_rc.vtkTextActor()
    actor.SetInput("\n    - - -   " + ROI_LABEL + "    \n")

    prop = actor.GetTextProperty()
    prop.SetFontFamilyToTimes()
    prop.SetFontSize(20)
    prop.SetColor(*ROI_COLOR)
    prop.SetShadow(True)
    prop.SetBackgroundColor(1.0, 1.0, 1.0)
    prop.SetBackgroundOpacity(0.7)
    prop.SetFrameColor(0.6, 0.6, 0.6)
    prop.SetFrame(True)
    prop.SetFrameWidth(1)

    # Position in upper-right corner (normalized display coordinates)
    coord = actor.GetPositionCoordinate()
    coord.SetCoordinateSystemToNormalizedDisplay()
    coord.SetValue(0.98, 0.95)
    prop.SetJustificationToRight()
    prop.SetVerticalJustificationToTop()

    plotter.renderer.AddViewProp(actor)


def _create_plotter(
    pts: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    title: str = "ROI Context",
) -> pv.Plotter:
    """Create an interactive plotter with the point cloud added.

    Args:
        pts: (N, 3) subsampled point array.
        colors: optional (N, 3) RGB float array in [0, 1]. When provided,
            renders with RGBA scalars instead of viridis Z colormap.
        title: window title displayed in the title bar.

    Returns:
        Configured PyVista plotter with point cloud mesh added.
    """
    plotter = pv.Plotter(title=title)
    plotter.set_background("white")
    plotter.window_size = (1280, 720)

    cloud = pv.PolyData(pts)

    if colors is not None:
        cloud["RGBA"] = to_rgba(colors)
        plotter.add_mesh(cloud, scalars="RGBA", rgba=True, point_size=2.0, render_points_as_spheres=True)
    else:
        cloud["Z"] = pts[:, 2]
        plotter.add_mesh(
            cloud,
            scalars="Z",
            cmap="viridis",
            point_size=2.0,
            render_points_as_spheres=True,
            scalar_bar_args=SCALAR_BAR_ARGS,
        )

    return plotter


def _bind_screenshot_key(
    plotter: pv.Plotter,
    save_dir: Path,
    filename: str,
) -> None:
    """Bind S key to capture screenshot + camera viewpoint JSON.

    Args:
        plotter: active PyVista plotter.
        save_dir: directory to save files into.
        filename: PNG filename (e.g. "roi_context_2d_rgb.png").
    """
    def _on_s_key() -> None:
        png_path = save_dir / filename
        plotter.screenshot(str(png_path))

        cam = plotter.camera
        stem = Path(filename).stem
        json_path = save_dir / f"{stem}_camera.json"
        viewpoint = {
            "position": list(cam.position),
            "focal_point": list(cam.focal_point),
            "view_up": list(cam.up),
        }
        json_path.write_text(json.dumps(viewpoint, indent=2))
        print(f"  Saved: {png_path.name}, {json_path.name}")

    plotter.add_key_event("s", _on_s_key)


def _render_roi_context(
    points: np.ndarray,
    roi: QuadROI | ROI,
    save_dir: Path,
    *,
    mode: Literal["2d", "3d"],
    max_points: int,
    seed: int,
    colors: np.ndarray | None,
    filename: str | None,
    mesh: pv.StructuredGrid | None,
    z_range: tuple[float, float] | None,
) -> None:
    """Shared implementation for 2D and 3D ROI context interactive viewing."""
    roi_bounds: ROI = roi.to_axis_aligned() if isinstance(roi, QuadROI) else roi

    if colors is not None:
        pts, sub_colors = subsample_points(points, max_points, seed, colors)
    else:
        pts, = subsample_points(points, max_points, seed)
        sub_colors = None

    # Determine output filename
    if filename is None:
        suffix = "_rgb" if colors is not None else ""
        filename = f"roi_context_{mode}{suffix}.png"

    title = f"ROI {mode.upper()} | S: capture, Q: close"
    plotter = _create_plotter(pts, colors=sub_colors, title=title)

    # Mesh overlay
    if mesh is not None:
        clipped = _clip_mesh_to_roi(mesh, roi_bounds, z_range=z_range)
        if clipped is not None:
            plotter.add_mesh(
                clipped, scalars="Z", cmap="coolwarm",
                show_edges=False, scalar_bar_args=SCALAR_BAR_ARGS,
            )

    # ROI shape
    if mode == "2d":
        z_mid = float(pts[:, 2].min() + pts[:, 2].max()) * 0.5
        shape = _build_dashed_rectangle_2d(roi_bounds, z_mid)
    elif mode == "3d":
        if z_range is not None:
            z_min, z_max = float(z_range[0]), float(z_range[1])
        else:
            if isinstance(roi, QuadROI):
                roi_points = filter_points_by_roi(pts, roi)
            else:
                x_min, x_max, y_min, y_max = roi_bounds
                mask = (
                    (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
                    (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
                )
                roi_points = pts[mask]
            if len(roi_points) > 0:
                z_min = float(roi_points[:, 2].min())
                z_max = float(roi_points[:, 2].max())
            else:
                z_min = float(pts[:, 2].min())
                z_max = float(pts[:, 2].max())
        shape = _build_dashed_box_3d(roi_bounds, z_min, z_max)
    else:
        raise ValueError(f"Invalid mode: {mode!r}, expected '2d' or '3d'")

    plotter.add_mesh(shape, color=ROI_COLOR, line_width=ROI_LINE_WIDTH)

    _add_roi_legend(plotter)
    plotter.add_axes(viewport=AXES_WIDGET_VIEWPORT)

    # Initial camera: isometric + zoom + pan-down
    plotter.view_isometric()
    plotter.camera.zoom(ROI_ZOOM)
    try:
        cam = plotter.camera
        pos = list(cam.position)
        fp = list(cam.focal_point)
        bounds = plotter.renderer.ComputeVisiblePropBounds()
        scene_height = bounds[5] - bounds[4]
        shift = scene_height * ROI_PAN_DOWN_RATIO
        pos[2] -= shift
        fp[2] -= shift
        cam.position = pos
        cam.focal_point = fp
    except (IndexError, TypeError, AttributeError):
        pass

    # S-key capture + interactive show
    _bind_screenshot_key(plotter, save_dir, filename)
    plotter.show()
    plotter.close()


def render_roi_context_2d(
    points: np.ndarray, roi: ROI, save_dir: Path,
    max_points: int = 500_000, seed: int = 42,
    colors: np.ndarray | None = None, filename: str | None = None,
    mesh: pv.StructuredGrid | None = None,
    z_range: tuple[float, float] | None = None,
) -> None:
    """Show interactive point cloud with 2D ROI rectangle overlay on XY plane."""
    _render_roi_context(points, roi, save_dir, mode="2d", max_points=max_points,
                        seed=seed, colors=colors, filename=filename, mesh=mesh, z_range=z_range)


def render_roi_context_3d(
    points: np.ndarray, roi: ROI, save_dir: Path,
    max_points: int = 500_000, seed: int = 42,
    colors: np.ndarray | None = None, filename: str | None = None,
    mesh: pv.StructuredGrid | None = None,
    z_range: tuple[float, float] | None = None,
) -> None:
    """Show interactive point cloud with 3D ROI wireframe box overlay."""
    _render_roi_context(points, roi, save_dir, mode="3d", max_points=max_points,
                        seed=seed, colors=colors, filename=filename, mesh=mesh, z_range=z_range)
