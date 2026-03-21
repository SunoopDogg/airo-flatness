"""X-direction height profile — strip-averaged Z residuals along X axis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from figure.detrend import detrend_points


def compute_height_profile(
    points,
    cell_size: float,
    strip_ratio: float = 0.5,
    min_points: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute strip-averaged Z residual profile along X axis.

    Accepts NumPy or CuPy arrays. Returns NumPy arrays (for matplotlib).
    """
    from figure.detrend import _get_xp
    xp = _get_xp(points)

    residuals = detrend_points(points)

    # Filter to central Y strip
    y = points[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    y_center = (y_min + y_max) / 2
    y_half = (y_max - y_min) * strip_ratio / 2
    strip_mask = (y >= y_center - y_half) & (y <= y_center + y_half)

    x_strip = points[strip_mask, 0]
    r_strip = residuals[strip_mask]

    if len(x_strip) == 0:
        return np.array([]), np.array([])

    # Bin along X
    x_min, x_max = float(x_strip.min()), float(x_strip.max())
    edges = xp.arange(x_min, x_max + cell_size, cell_size)
    if len(edges) < 2:
        edges = xp.array([x_min, x_min + cell_size])

    n_bins = len(edges) - 1
    bin_idx = xp.clip(xp.digitize(x_strip, edges) - 1, 0, n_bins - 1).astype(xp.int64)

    # Vectorized mean per bin using bincount
    counts = xp.bincount(bin_idx, minlength=n_bins)
    sums = xp.bincount(bin_idx, weights=r_strip, minlength=n_bins)

    valid = counts >= min_points
    means = xp.zeros(n_bins, dtype=r_strip.dtype)
    means[valid] = sums[valid] / counts[valid]

    # Bin centers
    centers = (edges[:-1] + edges[1:]) / 2

    # Filter to valid bins and convert to NumPy
    if xp is not np:
        valid_np = valid.get()
        x_centers = centers.get()[valid_np]
        z_means = means.get()[valid_np]
    else:
        x_centers = centers[valid]
        z_means = means[valid]

    return np.asarray(x_centers), np.asarray(z_means)


def plot_height_profile(
    x_centers: np.ndarray,
    z_means: np.ndarray,
    save_dir: Path,
    dpi: int = 300,
) -> None:
    """Render and save the X-direction height profile as PNG + PDF.

    Args:
        x_centers: (M,) array of X bin centers in meters.
        z_means: (M,) array of mean Z residuals in meters.
        save_dir: output directory.
        dpi: output DPI.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    z_mm = z_means * 1000
    sigma = float(np.std(z_mm))
    peak_to_valley = float(np.max(z_mm) - np.min(z_mm))

    fig, ax = plt.subplots(figsize=(10, 4))

    # ±1σ band
    ax.fill_between(x_centers, -sigma, sigma, color="#E0E0E0", label="±1σ band")

    # Z=0 reference line
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", label="Z=0")

    # Profile line (cubic spline for smooth curve)
    if len(x_centers) >= 4:
        x_smooth = np.linspace(x_centers[0], x_centers[-1], len(x_centers) * 10)
        spline = make_interp_spline(x_centers, z_mm, k=3)
        z_smooth = spline(x_smooth)
        ax.plot(x_smooth, z_smooth, color="black", linewidth=1.0, label="Height profile")
    else:
        ax.plot(x_centers, z_mm, color="black", linewidth=1.0, label="Height profile")

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Height Residual (mm)")

    stats_text = f"σ = {sigma:.2f} mm\nPeak-to-valley = {peak_to_valley:.2f} mm"
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    ax.legend()
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        plt.savefig(save_dir / f"height_profile.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()
