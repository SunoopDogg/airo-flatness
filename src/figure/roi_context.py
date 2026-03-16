"""ROI context view — full point cloud with ROI bounding box overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from figure.roi_selector import filter_points_by_roi, QuadROI

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
    roi: QuadROI | tuple[float, float, float, float],
    z_range: tuple[float, float] | None = None,
) -> pv.PolyData | None:
    """Clip a surface mesh to the ROI polygon and Z range.

    Args:
        mesh: surface mesh to clip.
        roi: QuadROI or (x_min, x_max, y_min, y_max) tuple.
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

    if not isinstance(roi, tuple):
        x_min, x_max, y_min, y_max = roi.to_axis_aligned()
    else:
        x_min, x_max, y_min, y_max = roi

    surface = mesh.extract_surface()
    clipped = surface.clip_box(
        [x_min, x_max, y_min, y_max, z_lo, z_hi], invert=False,
    )

    # For QuadROI, further clip to the actual polygon (not just AABB)
    if not isinstance(roi, tuple) and clipped.n_cells > 0:
        centers = clipped.cell_centers().points[:, :2]
        v = roi.vertices
        mask = np.ones(len(centers), dtype=bool)
        for i in range(4):
            j = (i + 1) % 4
            ex = v[j, 0] - v[i, 0]
            ey = v[j, 1] - v[i, 1]
            cross = ex * (centers[:, 1] - v[i, 1]) - ey * (centers[:, 0] - v[i, 0])
            mask &= cross >= 0
        cell_ids = np.where(mask)[0]
        if len(cell_ids) == 0:
            return None
        clipped = clipped.extract_cells(cell_ids)

    if clipped.n_points == 0:
        return None
    return clipped


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


def _build_dashed_rectangle_2d(
    roi: QuadROI | tuple[float, float, float, float],
    z: float,
) -> pv.PolyData:
    """Build a dashed 2D rectangle outline on the XY plane at given Z.

    Args:
        roi: QuadROI or (x_min, x_max, y_min, y_max) tuple.
        z: Z coordinate for the rectangle.

    Returns:
        PyVista PolyData with dashed line cells forming a rectangle.
    """
    if isinstance(roi, tuple):
        x_min, x_max, y_min, y_max = roi
        corners_2d = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ])
    else:
        corners_2d = roi.vertices
    corners = np.column_stack([corners_2d, np.full(len(corners_2d), z)])
    all_points = []
    all_lines = []
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i0, i1 in edges:
        pts, lines = _build_dashed_segments(corners[i0], corners[i1])
        offset = len(all_points)
        all_points.extend(pts)
        for ln in lines:
            all_lines.append([2, ln[1] + offset, ln[2] + offset])
    all_points = np.array(all_points)
    return pv.PolyData(all_points, lines=np.array(all_lines).ravel())


def _build_dashed_box_3d(
    roi: QuadROI | tuple[float, float, float, float],
    z_min: float,
    z_max: float,
) -> pv.PolyData:
    """Build a dashed 3D wireframe box.

    Args:
        roi: QuadROI or (x_min, x_max, y_min, y_max) tuple.
        z_min: bottom Z coordinate.
        z_max: top Z coordinate.

    Returns:
        PyVista PolyData with dashed line cells forming a box.
    """
    if isinstance(roi, tuple):
        x_min, x_max, y_min, y_max = roi
        corners_2d = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ])
    else:
        corners_2d = roi.vertices
    bottom = np.column_stack([corners_2d, np.full(4, z_min)])
    top = np.column_stack([corners_2d, np.full(4, z_max)])
    verts = np.vstack([bottom, top])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    all_points = []
    all_lines = []
    for i0, i1 in edges:
        pts, lines = _build_dashed_segments(verts[i0], verts[i1])
        offset = len(all_points)
        all_points.extend(pts)
        for ln in lines:
            all_lines.append([2, ln[1] + offset, ln[2] + offset])
    all_points = np.array(all_points)
    return pv.PolyData(all_points, lines=np.array(all_lines).ravel())


ROI_ZOOM = 2.0
ROI_PAN_UP = 0.6  # fraction of scene height to pan downward after zoom


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
    plotter.window_size = (1280, 720)

    cloud = pv.PolyData(pts)

    if colors is not None:
        rgba = np.empty((len(colors), 4), dtype=np.uint8)
        rgba[:, :3] = (colors * 255).astype(np.uint8)
        rgba[:, 3] = 255
        cloud["RGBA"] = rgba
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


def render_roi_context_2d(
    points: np.ndarray,
    roi: QuadROI | tuple[float, float, float, float],
    save_dir: Path,
    max_points: int = 500_000,
    seed: int = 42,
    dpi: int = 300,
    colors: np.ndarray | None = None,
    filename: str | None = None,
    mesh: pv.StructuredGrid | None = None,
    z_range: tuple[float, float] | None = None,
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
        mesh: optional surface mesh to overlay on the point cloud.
        z_range: optional (z_min, z_max) for mesh Z clipping.
    """
    pts, sub_colors = _subsample_points(points, max_points, seed, colors=colors)

    plotter = _create_point_cloud_plotter(pts, colors=sub_colors)

    if mesh is not None:
        clipped = _clip_mesh_to_roi(mesh, roi, z_range=z_range)
        if clipped is not None:
            plotter.add_mesh(clipped, scalars="Z", cmap="coolwarm", show_edges=False,
                            scalar_bar_args=SCALAR_BAR_ARGS)

    z_median = float(np.median(pts[:, 2]))
    rect = _build_dashed_rectangle_2d(roi, z_median)
    plotter.add_mesh(rect, color=ROI_COLOR, line_width=ROI_LINE_WIDTH)

    _add_roi_legend(plotter)
    plotter.add_axes(viewport=AXES_WIDGET_VIEWPORT)
    plotter.view_isometric()
    plotter.camera.zoom(ROI_ZOOM)
    # Pan camera upward to prevent bottom clipping after zoom
    try:
        cam = plotter.camera
        pos = list(cam.position)
        fp = list(cam.focal_point)
        bounds = plotter.renderer.ComputeVisiblePropBounds()
        scene_height = bounds[5] - bounds[4]  # Z range
        shift = scene_height * ROI_PAN_UP
        pos[2] -= shift
        fp[2] -= shift
        cam.position = pos
        cam.focal_point = fp
    except (IndexError, TypeError, AttributeError):
        pass  # skip pan in mock/test environments
    if filename is None:
        out_filename = "roi_context_2d_rgb.png" if colors is not None else "roi_context_2d.png"
    else:
        out_filename = filename
    plotter.screenshot(str(save_dir / out_filename))
    plotter.close()


def render_roi_context_3d(
    points: np.ndarray,
    roi: QuadROI | tuple[float, float, float, float],
    save_dir: Path,
    max_points: int = 500_000,
    seed: int = 42,
    dpi: int = 300,
    colors: np.ndarray | None = None,
    filename: str | None = None,
    mesh: pv.StructuredGrid | None = None,
    z_range: tuple[float, float] | None = None,
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
        mesh: optional surface mesh to overlay on the point cloud.
        z_range: optional (z_min, z_max) for mesh Z clipping and box Z extent.
    """
    pts, sub_colors = _subsample_points(points, max_points, seed, colors=colors)

    plotter = _create_point_cloud_plotter(pts, colors=sub_colors)

    if mesh is not None:
        clipped = _clip_mesh_to_roi(mesh, roi, z_range=z_range)
        if clipped is not None:
            plotter.add_mesh(clipped, scalars="Z", cmap="coolwarm", show_edges=False,
                            scalar_bar_args=SCALAR_BAR_ARGS)

    if z_range is not None:
        z_min, z_max = float(z_range[0]), float(z_range[1])
    else:
        roi_points = filter_points_by_roi(pts, roi)
        if len(roi_points) > 0:
            z_min = float(roi_points[:, 2].min())
            z_max = float(roi_points[:, 2].max())
        else:
            z_min = float(pts[:, 2].min())
            z_max = float(pts[:, 2].max())

    box = _build_dashed_box_3d(roi, z_min, z_max)
    plotter.add_mesh(box, color=ROI_COLOR, line_width=ROI_LINE_WIDTH)

    _add_roi_legend(plotter)
    plotter.add_axes(viewport=AXES_WIDGET_VIEWPORT)
    plotter.view_isometric()
    plotter.camera.zoom(ROI_ZOOM)
    # Pan camera upward to prevent bottom clipping after zoom
    try:
        cam = plotter.camera
        pos = list(cam.position)
        fp = list(cam.focal_point)
        bounds = plotter.renderer.ComputeVisiblePropBounds()
        scene_height = bounds[5] - bounds[4]  # Z range
        shift = scene_height * ROI_PAN_UP
        pos[2] -= shift
        fp[2] -= shift
        cam.position = pos
        cam.focal_point = fp
    except (IndexError, TypeError, AttributeError):
        pass  # skip pan in mock/test environments
    if filename is None:
        out_filename = "roi_context_3d_rgb.png" if colors is not None else "roi_context_3d.png"
    else:
        out_filename = filename
    plotter.screenshot(str(save_dir / out_filename))
    plotter.close()
