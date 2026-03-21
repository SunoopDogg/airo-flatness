"""Shared load + GPU downsample + cache pipeline."""

import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from config import Config
from loader.ply_loader import load_ply_full, load_ply_sampled
from preprocessing.cache import downsample_cache_path, load_cache, save_cache


def load_and_downsample(
    filepath: Path,
    config: Config,
    progress_callback: Callable | None = None,
    gpu: bool = False,
) -> dict:
    """Load PLY file and apply GPU voxel downsampling (or fall back to random sampling).

    Args:
        filepath: Path to PLY file
        config: Config with downsampling parameters
        progress_callback: Optional progress callback (current, total) -> None

    Returns:
        dict with keys: points, colors, intensity, classification,
                        total_vertices, sampled_vertices
    """
    from loader.ply_loader import read_ply_header
    header = read_ply_header(filepath)
    total = header["vertex_count"]

    if not config.downsampling_enabled:
        start_time = time.time()
        data = load_ply_sampled(
            filepath,
            max_points=config.max_points,
            progress_callback=progress_callback,
            seed=config.random_seed,
            chunk_size=config.chunk_size,
        )
        elapsed = time.time() - start_time
        print(f"\n\nLoaded {data['sampled_vertices']:,} points in {elapsed:.1f}s")
        if gpu:
            import cupy as cp
            data["points"] = cp.asarray(data["points"])
        return data

    # Try cache
    cache_path = downsample_cache_path(
        filepath, config.downsampling_voxel_size, config.downsample_cache_dir,
    )
    cached = load_cache(cache_path, filepath)
    if cached is not None:
        print(f"Loaded from cache: {cache_path}")
        points = cached["points"]
        if gpu:
            import cupy as cp
            points = cp.asarray(points)
        return {
            "points": points,
            "colors": cached["colors"],
            "intensity": cached["intensity"],
            "classification": cached["classification"],
            "total_vertices": total,
            "sampled_vertices": len(cached["points"]),
        }

    # Full load
    print(f"\nLoading all {total:,} points...")
    start_time = time.time()
    data = load_ply_full(filepath, progress_callback=progress_callback)
    elapsed = time.time() - start_time
    print(f"\n\nLoaded {data['sampled_vertices']:,} points in {elapsed:.1f}s")

    # GPU downsample
    print("\nGPU voxel downsampling...")
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy is required for GPU voxel downsampling but is not installed. "
            "Install it with: pip install cupy-cuda12x\n"
            "Or disable downsampling: Config(downsampling_enabled=False)"
        )

    from preprocessing.downsampling import downsample_gpu

    has_intensity = data["intensity"] is not None
    has_classification = data["classification"] is not None

    gpu_points = cp.asarray(data["points"])
    gpu_colors = (
        cp.asarray((data["colors"] * 255).clip(0, 255).astype(np.uint8))
        if data["colors"] is not None
        else cp.zeros((len(data["points"]), 3), dtype=cp.uint8)
    )
    gpu_intensity = (
        cp.asarray(data["intensity"]) if has_intensity
        else cp.zeros(len(data["points"]), dtype=cp.float32)
    )
    gpu_classification = (
        cp.asarray(data["classification"]) if has_classification
        else cp.zeros(len(data["points"]), dtype=cp.float32)
    )

    ds_pts, ds_cols, ds_int, ds_cls = downsample_gpu(
        gpu_points, gpu_colors, gpu_intensity, gpu_classification,
        voxel_size=config.downsampling_voxel_size,
        gpu_chunk_size=config.gpu_chunk_size,
    )

    # GPU -> CPU (or keep on GPU if gpu=True)
    result_points = ds_pts if gpu else ds_pts.get()
    result = {
        "points": result_points,
        "colors": ds_cols.get().astype(np.float32) / 255.0,
        "intensity": ds_int.get() if has_intensity else None,
        "classification": ds_cls.get() if has_classification else None,
        "total_vertices": total,
        "sampled_vertices": len(ds_pts),
    }

    # Save cache (always use CPU points)
    cache_points = ds_pts.get() if gpu else result["points"]
    try:
        print(f"Saving cache to: {cache_path}")
        save_cache(
            cache_path, cache_points, result["colors"],
            result["intensity"], result["classification"], filepath,
        )
        print("Cache saved successfully")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return result


def load_from_downsampled_cache(npz_path: Path, gpu: bool = False) -> dict:
    """Load downsampled data directly from an NPZ cache file.

    Skips the full PLY load + GPU downsample pipeline. Used when the user
    explicitly selects a cached downsampled file.

    Args:
        npz_path: Path to the .npz cache file.

    Returns:
        dict with keys: points, colors, intensity, classification,
                        total_vertices, sampled_vertices
    """
    print(f"Loading cached data... {npz_path.name}")
    try:
        cached = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        print(f"Error: Failed to load {npz_path.name}: {e}")
        sys.exit(1)

    points = cached["points"]
    if gpu:
        import cupy as cp
        points = cp.asarray(points)
    n_points = len(cached["points"])
    return {
        "points": points,
        "colors": cached["colors"] if "colors" in cached else None,
        "intensity": cached["intensity"] if "intensity" in cached else None,
        "classification": cached["classification"] if "classification" in cached else None,
        "total_vertices": n_points,
        "sampled_vertices": n_points,
    }
