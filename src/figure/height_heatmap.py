"""2D height deviation heatmap — grid-based Z-range after detrending."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from figure.detrend import detrend_points


def compute_height_grid(
    points: np.ndarray,
    target_grid: int = 100,
    min_points: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute per-cell Z-range grid after plane detrending.

    Args:
        points: (N, 3) XYZ array.
        target_grid: number of cells along longest axis.
        min_points: minimum points per cell.

    Returns:
        (grid, x_edges, y_edges, cell_size)
        grid: (nx, ny) array of Z-range values, NaN for sparse cells.
    """
    residuals = detrend_points(points)

    xs = points[:, 0]
    ys = points[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    longest = max(x_max - x_min, y_max - y_min)
    cell_size = longest / target_grid if longest > 0 else 1.0

    x_edges = np.arange(x_min, x_max + cell_size, cell_size)
    y_edges = np.arange(y_min, y_max + cell_size, cell_size)
    if len(x_edges) < 2:
        x_edges = np.array([x_min, x_min + cell_size])
    if len(y_edges) < 2:
        y_edges = np.array([y_min, y_min + cell_size])

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    xi = np.clip(np.digitize(xs, x_edges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(ys, y_edges) - 1, 0, ny - 1)

    grid = np.full((nx, ny), np.nan)
    cell_idx = xi * ny + yi
    for idx in range(nx * ny):
        mask = cell_idx == idx
        if mask.sum() < min_points:
            continue
        cell_res = residuals[mask]
        grid[idx // ny, idx % ny] = cell_res.max() - cell_res.min()

    return grid, x_edges, y_edges, cell_size


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
