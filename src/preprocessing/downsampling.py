"""
GPU-accelerated point cloud downsampling using CuPy.

Supports voxel grid downsampling on GPU with weighted merge for chunked
processing.  Handles points, colors, intensity, and classification.

Key differences from airo-plyreader's implementation:
- Carries (sums, counts) through merge instead of re-averaging
- Uses GPU-based mode (majority vote) for classification
- Uses absolute bit-packed voxel keys for cross-chunk consistency
"""

import time
from dataclasses import dataclass

import cupy as cp


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class VoxelAccumulation:
    """Intermediate accumulation state for voxel downsampling.

    Stores *sums* (not averages) so that chunks can be merged correctly
    via weighted combination.  Classification is stored as per-voxel
    histograms (O(V*C)) instead of raw values (O(N)) to keep GPU memory
    bounded during incremental merges.
    """
    coord_sums: cp.ndarray       # (V, 3) float32 — sum of coordinates per voxel
    color_sums: cp.ndarray       # (V, 3) float32 — sum of colors per voxel
    intensity_sums: cp.ndarray   # (V,)   float32 — sum of intensity per voxel
    counts: cp.ndarray           # (V,)   float32 — number of points per voxel
    voxel_keys: cp.ndarray       # (V,)   int64   — unique linear voxel key
    class_labels: cp.ndarray     # (C,)   float32 — unique classification values
    class_counts: cp.ndarray     # (V, C) float32 — per-voxel count for each class


# ---------------------------------------------------------------------------
# Voxel key computation
# ---------------------------------------------------------------------------

def _compute_voxel_keys_absolute(
    points: cp.ndarray,
    voxel_size: float,
) -> cp.ndarray:
    """Compute globally consistent voxel keys using absolute floor indices.

    Uses absolute ``floor(coord / voxel_size)`` so that the same physical
    location always produces the same key, regardless of which chunk the
    point belongs to.  Keys are packed into int64 using 21 bits per axis
    (supports ~2M voxels per axis in each direction).

    Returns:
        Voxel keys as int64 array of shape (N,).
    """
    # Use float64 for division to avoid precision loss with small voxel sizes
    voxel_size_d = cp.float64(voxel_size)
    vi = cp.floor(points[:, 0].astype(cp.float64) / voxel_size_d).astype(cp.int64)
    vj = cp.floor(points[:, 1].astype(cp.float64) / voxel_size_d).astype(cp.int64)
    vk = cp.floor(points[:, 2].astype(cp.float64) / voxel_size_d).astype(cp.int64)

    # Use bit-packing: 21 bits per axis (handles -1048576 to +1048575)
    # Shift to unsigned range first
    offset = cp.int64(1 << 20)  # 1048576
    vi = vi + offset
    vj = vj + offset
    vk = vk + offset

    return (vi << 42) | (vj << 21) | vk


# ---------------------------------------------------------------------------
# Single-chunk accumulation
# ---------------------------------------------------------------------------

def _accumulate_voxel_chunk(
    points: cp.ndarray,
    colors: cp.ndarray,
    intensity: cp.ndarray,
    classification: cp.ndarray,
    voxel_size: float,
) -> VoxelAccumulation:
    """Accumulate point attributes into voxel sums for one chunk.

    Uses absolute bit-packed voxel keys so that the same physical voxel
    always maps to the same key regardless of chunk boundaries.

    Returns a :class:`VoxelAccumulation` holding **sums** and counts.
    """
    if len(points) == 0:
        return VoxelAccumulation(
            coord_sums=cp.empty((0, 3), dtype=cp.float32),
            color_sums=cp.empty((0, 3), dtype=cp.float32),
            intensity_sums=cp.empty(0, dtype=cp.float32),
            counts=cp.empty(0, dtype=cp.float32),
            voxel_keys=cp.empty(0, dtype=cp.int64),
            class_labels=cp.empty(0, dtype=cp.float32),
            class_counts=cp.empty((0, 0), dtype=cp.float32),
        )

    # Compute per-point voxel keys using absolute coordinates
    keys = _compute_voxel_keys_absolute(points, voxel_size)

    # Unique voxels and inverse mapping
    unique_keys, inverse = cp.unique(keys, return_inverse=True)
    n_voxels = len(unique_keys)

    # --- Coordinate sums ---
    coord_sums = cp.zeros((n_voxels, 3), dtype=cp.float32)
    for ax in range(3):
        cp.add.at(coord_sums[:, ax], inverse, points[:, ax])

    # --- Color sums ---
    colors_f = colors.astype(cp.float32)
    color_sums = cp.zeros((n_voxels, 3), dtype=cp.float32)
    for ax in range(3):
        cp.add.at(color_sums[:, ax], inverse, colors_f[:, ax])

    # --- Intensity sums ---
    intensity_sums = cp.zeros(n_voxels, dtype=cp.float32)
    cp.add.at(intensity_sums, inverse, intensity)

    # --- Counts ---
    counts = cp.zeros(n_voxels, dtype=cp.float32)
    cp.add.at(counts, inverse, cp.ones(len(points), dtype=cp.float32))

    # --- Classification histogram (per-voxel counts per class) ---
    unique_cls = cp.unique(classification)
    n_cls = len(unique_cls)
    class_counts = cp.zeros((n_voxels, n_cls), dtype=cp.float32)
    for ci in range(n_cls):
        mask = (classification == unique_cls[ci]).astype(cp.float32)
        cp.add.at(class_counts[:, ci], inverse, mask)

    return VoxelAccumulation(
        coord_sums=coord_sums,
        color_sums=color_sums,
        intensity_sums=intensity_sums,
        counts=counts,
        voxel_keys=unique_keys,
        class_labels=unique_cls.astype(cp.float32),
        class_counts=class_counts,
    )


# ---------------------------------------------------------------------------
# Merge two accumulations
# ---------------------------------------------------------------------------

def _merge_two(a: VoxelAccumulation, b: VoxelAccumulation) -> VoxelAccumulation:
    """Merge exactly two :class:`VoxelAccumulation` objects."""
    all_keys = cp.concatenate([a.voxel_keys, b.voxel_keys])
    merged_keys, key_inverse = cp.unique(all_keys, return_inverse=True)
    n_merged = len(merged_keys)

    coord_sums = cp.zeros((n_merged, 3), dtype=cp.float32)
    color_sums = cp.zeros((n_merged, 3), dtype=cp.float32)
    intensity_sums = cp.zeros(n_merged, dtype=cp.float32)
    counts = cp.zeros(n_merged, dtype=cp.float32)

    # Merge class histograms — union of class labels from both sides
    all_labels = cp.unique(cp.concatenate([a.class_labels, b.class_labels]))
    n_cls_merged = len(all_labels)
    class_counts = cp.zeros((n_merged, n_cls_merged), dtype=cp.float32)

    offset = 0
    for acc in (a, b):
        n_v = len(acc.voxel_keys)
        local_to_merged = key_inverse[offset:offset + n_v]
        offset += n_v

        for ax in range(3):
            cp.add.at(coord_sums[:, ax], local_to_merged, acc.coord_sums[:, ax])
            cp.add.at(color_sums[:, ax], local_to_merged, acc.color_sums[:, ax])
        cp.add.at(intensity_sums, local_to_merged, acc.intensity_sums)
        cp.add.at(counts, local_to_merged, acc.counts)

        # Map each class column from acc to the merged columns
        for ci in range(len(acc.class_labels)):
            merged_ci = int(cp.searchsorted(all_labels, acc.class_labels[ci]))
            cp.add.at(class_counts[:, merged_ci], local_to_merged, acc.class_counts[:, ci])

    return VoxelAccumulation(
        coord_sums=coord_sums,
        color_sums=color_sums,
        intensity_sums=intensity_sums,
        counts=counts,
        voxel_keys=merged_keys,
        class_labels=all_labels,
        class_counts=class_counts,
    )


def _merge_accumulations(accums: list[VoxelAccumulation]) -> VoxelAccumulation:
    """Merge multiple :class:`VoxelAccumulation` objects via pairwise reduction.

    Uses a tree-reduction pattern (merge pairs, then merge results) to keep
    peak GPU memory lower than concatenating all keys at once.
    """
    if len(accums) == 1:
        return accums[0]

    while len(accums) > 1:
        next_level = []
        for i in range(0, len(accums), 2):
            if i + 1 < len(accums):
                merged = _merge_two(accums[i], accums[i + 1])
            else:
                merged = accums[i]
            next_level.append(merged)
        accums = next_level
        cp.get_default_memory_pool().free_all_blocks()

    return accums[0]


# ---------------------------------------------------------------------------
# Finalize: sums → averages
# ---------------------------------------------------------------------------

def _finalize_voxels(
    accum: VoxelAccumulation,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Convert accumulated sums to per-voxel averages.

    Classification uses argmax over the per-voxel class histogram
    (majority vote).

    Returns:
        (points, colors, intensity, classification) as CuPy arrays.
    """
    n_voxels = len(accum.counts)
    if n_voxels == 0:
        return (
            cp.empty((0, 3), dtype=cp.float32),
            cp.empty((0, 3), dtype=cp.uint8),
            cp.empty(0, dtype=cp.float32),
            cp.empty(0, dtype=cp.float32),
        )

    counts_col = accum.counts[:, cp.newaxis]  # (V, 1) for broadcasting

    avg_pts = accum.coord_sums / counts_col
    avg_cols = (accum.color_sums / counts_col).clip(0, 255).astype(cp.uint8)
    avg_int = accum.intensity_sums / accum.counts

    if len(accum.class_labels) > 0:
        best_idx = cp.argmax(accum.class_counts, axis=1)
        avg_cls = accum.class_labels[best_idx]
    else:
        avg_cls = cp.zeros(n_voxels, dtype=cp.float32)

    return avg_pts, avg_cols, avg_int, avg_cls


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def downsample_voxel_grid_gpu(
    points: cp.ndarray,
    colors: cp.ndarray,
    intensity: cp.ndarray,
    classification: cp.ndarray,
    voxel_size: float,
    gpu_chunk_size: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Downsample a point cloud via voxel grid averaging on GPU.

    For large point clouds, processes in chunks to stay within GPU memory:

    1. Split into chunks of *gpu_chunk_size* points.
    2. Accumulate sums+counts per chunk using absolute voxel keys.
    3. Merge all accumulations incrementally.
    4. Finalize (divide sums by counts, compute mode for classification).

    Returns:
        (points, colors, intensity, classification) as CuPy arrays.
    """
    if len(points) == 0:
        return (
            cp.empty((0, 3), dtype=cp.float32),
            cp.empty((0, 3), dtype=cp.uint8),
            cp.empty(0, dtype=cp.float32),
            cp.empty(0, dtype=cp.float32),
        )

    n_points = len(points)

    # Single pass for small clouds
    if n_points <= gpu_chunk_size:
        accum = _accumulate_voxel_chunk(
            points, colors, intensity, classification,
            voxel_size,
        )
        return _finalize_voxels(accum)

    # Chunked processing — move input to CPU to free GPU memory,
    # then stream chunks and merge incrementally.
    import numpy as np_cpu
    points_cpu = cp.asnumpy(points)
    colors_cpu = cp.asnumpy(colors)
    intensity_cpu = cp.asnumpy(intensity)
    classification_cpu = cp.asnumpy(classification)
    del points, colors, intensity, classification
    cp.get_default_memory_pool().free_all_blocks()

    n_chunks = (n_points + gpu_chunk_size - 1) // gpu_chunk_size
    merged = None

    for i in range(n_chunks):
        start = i * gpu_chunk_size
        end = min(start + gpu_chunk_size, n_points)

        accum = _accumulate_voxel_chunk(
            cp.asarray(points_cpu[start:end]),
            cp.asarray(colors_cpu[start:end]),
            cp.asarray(intensity_cpu[start:end]),
            cp.asarray(classification_cpu[start:end]),
            voxel_size,
        )

        if merged is None:
            merged = accum
        else:
            merged = _merge_two(merged, accum)
            del accum

        cp.get_default_memory_pool().free_all_blocks()

    del points_cpu, colors_cpu, intensity_cpu, classification_cpu
    return _finalize_voxels(merged)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def downsample_gpu(
    points: cp.ndarray,
    colors: cp.ndarray,
    intensity: cp.ndarray,
    classification: cp.ndarray,
    voxel_size: float,
    gpu_chunk_size: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Main downsampling entry point with stats printing and OOM handling.

    Args:
        points: (N, 3) float32 CuPy array.
        colors: (N, 3) uint8 CuPy array.
        intensity: (N,) float32 CuPy array.
        classification: (N,) float32 CuPy array.
        voxel_size: Voxel edge length in metres.
        gpu_chunk_size: Maximum number of points per GPU chunk.

    Returns:
        Tuple of (points, colors, intensity, classification) as CuPy arrays.
    """
    print(f"  Downsampling: ENABLED - Method: voxel (GPU)")
    print(f"    Original points: {len(points):,}")

    start_time = time.time()

    chunk = gpu_chunk_size
    max_retries = 4
    result = None
    for attempt in range(max_retries + 1):
        try:
            result = downsample_voxel_grid_gpu(
                points, colors, intensity, classification,
                voxel_size, chunk,
            )
            break
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            if attempt == max_retries:
                raise RuntimeError(
                    f"GPU out of memory during downsampling after {max_retries} retries. "
                    f"Final chunk size: {chunk:,}. "
                    f"Set Config.gpu_chunk_size to a smaller value."
                )
            chunk = chunk // 2
            print(f"    GPU OOM — reducing chunk size to {chunk:,} (attempt {attempt + 2}/{max_retries + 1})")

    r_pts, r_cols, r_int, r_cls = result

    # Statistics
    n_orig = len(points)
    n_down = len(r_pts)
    reduction_ratio = n_down / n_orig if n_orig > 0 else 0
    points_removed = n_orig - n_down
    processing_time = time.time() - start_time

    print(f"    Voxel size: {voxel_size}m")
    print(f"    Downsampled points: {n_down:,}")
    print(f"    Points removed: {points_removed:,}")
    print(f"    Reduction ratio: {reduction_ratio:.2%}")
    print(f"    Processing time: {processing_time:.2f} seconds")

    return r_pts, r_cols, r_int, r_cls
