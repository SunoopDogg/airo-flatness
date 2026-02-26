"""Interactive ROI selection via matplotlib top-view scatter plot."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, SpanSelector


def select_roi(
    points: np.ndarray,
    max_display: int = 500_000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Show top-view scatter and let user drag-select a rectangular ROI.

    Args:
        points: (N, 3) point cloud array.
        max_display: max points to display (subsampled for performance).
        seed: random seed for subsampling.

    Returns:
        (x_min, x_max, y_min, y_max) of selected region.

    Raises:
        ValueError: if user closes window without selecting.
    """
    if len(points) > max_display:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), max_display, replace=False)
        display_pts = points[idx]
    else:
        display_pts = points

    roi_coords = {}

    def on_select(eclick, erelease):
        roi_coords["x_min"] = min(eclick.xdata, erelease.xdata)
        roi_coords["x_max"] = max(eclick.xdata, erelease.xdata)
        roi_coords["y_min"] = min(eclick.ydata, erelease.ydata)
        roi_coords["y_max"] = max(eclick.ydata, erelease.ydata)

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        display_pts[:, 0],
        display_pts[:, 1],
        c=display_pts[:, 2],
        cmap="viridis",
        s=0.1,
        rasterized=True,
    )
    fig.colorbar(scatter, ax=ax, label="Z (m)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.set_title("Drag to select ROI — close window when done")

    rect_kwargs = dict(edgecolor="red", linestyle="--", linewidth=1.5, facecolor="none")
    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        interactive=True,
        props=rect_kwargs,
    )

    plt.tight_layout()
    plt.show()

    if not roi_coords:
        raise ValueError("No ROI selected.")

    return (
        roi_coords["x_min"],
        roi_coords["x_max"],
        roi_coords["y_min"],
        roi_coords["y_max"],
    )


def filter_points_by_roi(
    points: np.ndarray,
    roi: tuple[float, float, float, float],
) -> np.ndarray:
    """Filter points within the ROI bounding box (X-Y only).

    Args:
        points: (N, 3) array.
        roi: (x_min, x_max, y_min, y_max).

    Returns:
        (M, 3) array of points within ROI.
    """
    x_min, x_max, y_min, y_max = roi
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )
    return points[mask]


def filter_points_by_z(
    points: np.ndarray,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Filter points within a Z value range.

    Args:
        points: (N, 3) array.
        z_min: minimum Z value (inclusive).
        z_max: maximum Z value (inclusive).

    Returns:
        (M, 3) array of points within Z range.
    """
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask]


def select_z_roi(
    points: np.ndarray,
    bins: int = 100,
) -> tuple[float, float]:
    """Show Z histogram and let user drag-select a Z value range.

    Args:
        points: (N, 3) point cloud array (already X-Y filtered).
        bins: number of histogram bins.

    Returns:
        (z_min, z_max) of selected Z range.

    Raises:
        ValueError: if user closes window without selecting.
    """
    z_values = points[:, 2]
    z_range = {}

    def on_select(z_lo, z_hi):
        z_range["z_min"] = min(z_lo, z_hi)
        z_range["z_max"] = max(z_lo, z_hi)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(z_values, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Z (m)")
    ax.set_ylabel("Point count")
    ax.set_title("Select Z Range (drag to select, close to confirm)")

    _span = SpanSelector(
        ax,
        on_select,
        direction="horizontal",
        useblit=True,
        props=dict(alpha=0.3, facecolor="red"),
        interactive=True,
    )

    plt.tight_layout()
    plt.show()

    if not z_range:
        raise ValueError("No Z range selected.")

    return z_range["z_min"], z_range["z_max"]
