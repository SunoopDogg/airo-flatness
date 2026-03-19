"""Shared utilities — subsampling, color conversion, CLI helpers."""

import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np


def subsample_points(
    points: np.ndarray,
    max_points: int,
    seed: int = 42,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Subsample points and companion arrays with the same random indices.

    Args:
        points: (N, 3) point array.
        max_points: maximum number of points to keep.
        seed: random seed for reproducibility.
        *arrays: additional arrays to subsample with the same indices.

    Returns:
        (subsampled_points, *subsampled_arrays) in input order.
        Returns inputs unchanged if len(points) <= max_points.
    """
    if len(points) <= max_points:
        return (points, *arrays)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), max_points, replace=False)
    return (points[idx], *(a[idx] for a in arrays))


def to_rgba(colors: np.ndarray) -> np.ndarray:
    """Convert (N, 3) float RGB [0, 1] to (N, 4) uint8 RGBA.

    Args:
        colors: (N, 3) float array with values in [0, 1].

    Returns:
        (N, 4) uint8 array with alpha=255.
    """
    rgba = np.empty((len(colors), 4), dtype=np.uint8)
    rgba[:, :3] = (colors * 255).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def create_progress_bar(
    bar_width: int = 40,
    label: str = "Loading",
) -> Callable[[int, int], None]:
    """Create a progress bar callback for streaming operations.

    Args:
        bar_width: character width of the progress bar.
        label: prefix label displayed before the bar.

    Returns:
        Callback function with signature (current, total) -> None.
    """
    start_time = time.time()

    def progress(current: int, total: int) -> None:
        pct = current / total
        filled = int(bar_width * pct)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        elapsed = time.time() - start_time
        eta = (elapsed / pct - elapsed) if pct > 0 else 0
        print(
            f"\r{label}: [{bar}] {pct:6.1%}  "
            f"({current:,} / {total:,})  "
            f"ETA: {eta:.0f}s  ",
            end="",
            flush=True,
        )

    return progress


def format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable units."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def select_file(data_dir: Path) -> Path:
    """Display PLY file list in data_dir and let user choose interactively.

    Args:
        data_dir: directory containing .ply files.

    Returns:
        Path to the selected PLY file.
    """
    ply_files = sorted(data_dir.glob("*.ply"))

    if not ply_files:
        print(f"Error: No .ply files found in {data_dir}")
        sys.exit(1)

    print("\nAvailable point cloud files:")
    print("-" * 50)
    for i, f in enumerate(ply_files, 1):
        size = format_size(f.stat().st_size)
        print(f"  [{i}] {f.name:<20s} ({size})")
    print("-" * 50)

    while True:
        try:
            choice = input(f"\nSelect file number (1-{len(ply_files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(ply_files):
                return ply_files[idx]
            print(f"  Please enter a number between 1 and {len(ply_files)}")
        except (ValueError, EOFError):
            print("  Invalid input. Please enter a number.")
