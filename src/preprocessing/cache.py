"""Downsampling cache — NPZ-based save/load with source file invalidation."""

from pathlib import Path

import numpy as np


def format_voxel_size(voxel_size: float) -> str:
    """Convert voxel size float to filename-safe string.

    Examples: 0.01 -> "0_01", 0.0005 -> "0_0005"
    """
    s = f"{voxel_size:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "_")


def downsample_cache_path(
    ply_path: str | Path,
    voxel_size: float,
    cache_dir: str | Path,
) -> Path:
    """Build the cache file path for a downsampled result."""
    stem = Path(ply_path).stem
    voxel_str = format_voxel_size(voxel_size)
    return Path(cache_dir) / f"{stem}-{voxel_str}.npz"


def save_cache(
    cache_path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None,
    intensity: np.ndarray | None,
    classification: np.ndarray | None,
    source_path: str | Path,
) -> None:
    """Save downsampled data to NPZ cache with source file metadata."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_path)
    source_stat = source_path.stat()

    data = {
        "points": points,
        "source_mtime": np.float64(source_stat.st_mtime),
        "source_size": np.int64(source_stat.st_size),
    }
    if colors is not None:
        data["colors"] = colors
    if intensity is not None:
        data["intensity"] = intensity
    if classification is not None:
        data["classification"] = classification

    np.savez(cache_path, **data)


def load_cache(
    cache_path: str | Path,
    source_path: str | Path,
) -> dict | None:
    """Load cached data if valid. Returns None if cache missing or stale."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        cached = np.load(cache_path, allow_pickle=False)
    except Exception:
        return None

    source_path = Path(source_path)
    if not source_path.exists():
        return None

    source_stat = source_path.stat()
    cached_mtime = float(cached["source_mtime"])
    cached_size = int(cached["source_size"])

    if source_stat.st_mtime != cached_mtime or source_stat.st_size != cached_size:
        return None

    return {
        "points": cached["points"],
        "colors": cached["colors"] if "colors" in cached else None,
        "intensity": cached["intensity"] if "intensity" in cached else None,
        "classification": cached["classification"] if "classification" in cached else None,
    }
