"""2D height deviation heatmap — grid-based Z-range after detrending."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from figure.detrend import detrend_points


def compute_height_grid(
    points,
    target_grid: int = 100,
    min_points: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute per-cell Z-range grid after plane detrending.

    Accepts NumPy or CuPy arrays. Returns NumPy arrays (for matplotlib).
    """
    from figure.detrend import _get_xp
    xp = _get_xp(points)

    residuals = detrend_points(points)

    xs, ys = points[:, 0], points[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    longest = max(x_max - x_min, y_max - y_min)
    cell_size = longest / target_grid if longest > 0 else 1.0

    # Edges on same device as data for digitize
    x_edges = xp.arange(x_min, x_max + cell_size, cell_size)
    y_edges = xp.arange(y_min, y_max + cell_size, cell_size)
    if len(x_edges) < 2:
        x_edges = xp.array([x_min, x_min + cell_size])
    if len(y_edges) < 2:
        y_edges = xp.array([y_min, y_min + cell_size])

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    xi = xp.clip(xp.digitize(xs, x_edges) - 1, 0, nx - 1)
    yi = xp.clip(xp.digitize(ys, y_edges) - 1, 0, ny - 1)

    cell_idx = xi * ny + yi
    n_cells = nx * ny

    cell_min = xp.full(n_cells, xp.inf)
    cell_max = xp.full(n_cells, -xp.inf)
    xp.minimum.at(cell_min, cell_idx, residuals)
    xp.maximum.at(cell_max, cell_idx, residuals)

    cell_counts = xp.bincount(cell_idx.astype(xp.int64), minlength=n_cells)

    # Build grid and convert to NumPy for matplotlib
    grid = np.full((nx, ny), np.nan)
    valid = cell_counts >= min_points
    if xp is not np:
        ranges = (cell_max - cell_min).get()
        valid_np = valid.get()
    else:
        ranges = cell_max - cell_min
        valid_np = valid
    grid.ravel()[valid_np] = ranges[valid_np]

    # Edges as NumPy for matplotlib
    x_edges_np = x_edges.get() if xp is not np else x_edges
    y_edges_np = y_edges.get() if xp is not np else y_edges

    return grid, x_edges_np, y_edges_np, cell_size


def plot_height_heatmap(
    grid: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cell_size: float,
    save_dir: Path,
    dpi: int = 300,
) -> None:
    """Render and save the height deviation heatmap as PNG + PDF.

    Args:
        grid: (nx, ny) height range grid.
        x_edges, y_edges: bin edges.
        cell_size: cell size in meters.
        save_dir: output directory.
        dpi: output DPI.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#E0E0E0")

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    im = ax.imshow(
        grid.T,
        cmap=cmap,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height Range (m)")

    valid = grid[~np.isnan(grid)]
    if len(valid) > 0:
        stats_text = (
            f"Mean: {np.mean(valid):.3f} m\n"
            f"Max:  {np.max(valid):.3f} m\n"
            f"Cell: {cell_size:.2f} m"
        )
        ax.text(
            0.02, 0.97, stats_text,
            transform=ax.transAxes, verticalalignment="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Height Deviation Heatmap (Detrended)")

    plt.tight_layout()
    for fmt in ("png", "pdf"):
        plt.savefig(save_dir / f"height_heatmap.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()
