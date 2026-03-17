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
    via weighted combination.
    """
    coord_sums: cp.ndarray       # (V, 3) float32 — sum of coordinates per voxel
    color_sums: cp.ndarray       # (V, 3) float32 — sum of colors per voxel
    intensity_sums: cp.ndarray   # (V,)   float32 — sum of intensity per voxel
    counts: cp.ndarray           # (V,)   float32 — number of points per voxel
    voxel_keys: cp.ndarray       # (V,)   int64   — unique linear voxel key
    classification_vals: cp.ndarray  # (N,)   float32 — raw classification values
    classification_inv: cp.ndarray   # (N,)   int64   — maps each raw value to its voxel index


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
            classification_vals=cp.empty(0, dtype=cp.float32),
            classification_inv=cp.empty(0, dtype=cp.int64),
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

    # --- Classification (raw values + inverse for later mode) ---
    return VoxelAccumulation(
        coord_sums=coord_sums,
        color_sums=color_sums,
        intensity_sums=intensity_sums,
        counts=counts,
        voxel_keys=unique_keys,
        classification_vals=classification.copy(),
        classification_inv=inverse.astype(cp.int64),
    )


# ---------------------------------------------------------------------------
# Merge multiple accumulations
# ---------------------------------------------------------------------------

def _merge_accumulations(accums: list[VoxelAccumulation]) -> VoxelAccumulation:
    """Merge multiple :class:`VoxelAccumulation` objects.

    Voxels that appear in more than one chunk are combined by adding their
    sums and counts.  Classification raw values are re-indexed so that the
    inverse mapping remains consistent with the merged voxel key array.
    """
    if len(accums) == 1:
        return accums[0]

    # Concatenate all voxel keys from every accumulation
    all_keys = cp.concatenate([a.voxel_keys for a in accums])
    merged_keys, key_inverse = cp.unique(all_keys, return_inverse=True)
    n_merged = len(merged_keys)

    # Build coordinate-sum, color-sum, intensity-sum, counts for merged set
    coord_sums = cp.zeros((n_merged, 3), dtype=cp.float32)
    color_sums = cp.zeros((n_merged, 3), dtype=cp.float32)
    intensity_sums = cp.zeros(n_merged, dtype=cp.float32)
    counts = cp.zeros(n_merged, dtype=cp.float32)

    offset = 0  # running offset into key_inverse
    cls_vals_list = []
    cls_inv_list = []

    for a in accums:
        n_v = len(a.voxel_keys)
        # Mapping from this accumulation's local voxel indices → merged indices
        local_to_merged = key_inverse[offset:offset + n_v]
        offset += n_v

        # Scatter-add sums
        for ax in range(3):
            cp.add.at(coord_sums[:, ax], local_to_merged, a.coord_sums[:, ax])
            cp.add.at(color_sums[:, ax], local_to_merged, a.color_sums[:, ax])
        cp.add.at(intensity_sums, local_to_merged, a.intensity_sums)
        cp.add.at(counts, local_to_merged, a.counts)

        # Remap classification inverse indices through the merge mapping
        if len(a.classification_inv) > 0:
            remapped_inv = local_to_merged[a.classification_inv]
            cls_vals_list.append(a.classification_vals)
            cls_inv_list.append(remapped_inv)

    # Concatenate classification data
    if cls_vals_list:
        classification_vals = cp.concatenate(cls_vals_list)
        classification_inv = cp.concatenate(cls_inv_list)
    else:
        classification_vals = cp.empty(0, dtype=cp.float32)
        classification_inv = cp.empty(0, dtype=cp.int64)

    return VoxelAccumulation(
        coord_sums=coord_sums,
        color_sums=color_sums,
        intensity_sums=intensity_sums,
        counts=counts,
        voxel_keys=merged_keys,
        classification_vals=classification_vals,
        classification_inv=classification_inv,
    )


# ---------------------------------------------------------------------------
# GPU mode (majority vote) for classification
# ---------------------------------------------------------------------------

def _compute_mode_gpu(
    values: cp.ndarray,
    inverse: cp.ndarray,
    n_voxels: int,
) -> cp.ndarray:
    """GPU-based mode computation for classification per voxel.

    For each unique classification value, count occurrences per voxel via
    ``scatter_add``, then keep the value with the highest count per voxel.

    Returns:
        Array of shape (n_voxels,) with the mode classification per voxel.
    """
    if len(values) == 0:
        return cp.empty(0, dtype=cp.float32)

    unique_cls = cp.unique(values)

    # best_count[v] = highest count seen so far for voxel v
    best_count = cp.zeros(n_voxels, dtype=cp.float32)
    # best_val[v] = classification value corresponding to best_count[v]
    best_val = cp.zeros(n_voxels, dtype=cp.float32)

    for cls_val in unique_cls:
        # Mask: 1 where classification matches cls_val, 0 otherwise
        mask = (values == cls_val).astype(cp.float32)
        # Count occurrences of this class per voxel
        cls_count = cp.zeros(n_voxels, dtype=cp.float32)
        cp.add.at(cls_count, inverse, mask)
        # Update best where this class has higher count
        better = cls_count > best_count
        best_count = cp.where(better, cls_count, best_count)
        best_val = cp.where(better, float(cls_val), best_val)

    return best_val


# ---------------------------------------------------------------------------
# Finalize: sums → averages
# ---------------------------------------------------------------------------

def _finalize_voxels(
    accum: VoxelAccumulation,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Convert accumulated sums to per-voxel averages.

    Classification uses GPU-based mode (majority vote) instead of averaging.

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

    avg_cls = _compute_mode_gpu(
        accum.classification_vals,
        accum.classification_inv,
        n_voxels,
    )

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
    3. Merge all accumulations.
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

    # Chunked processing
    n_chunks = (n_points + gpu_chunk_size - 1) // gpu_chunk_size
    accums = []

    for i in range(n_chunks):
        start = i * gpu_chunk_size
        end = min(start + gpu_chunk_size, n_points)
        accum = _accumulate_voxel_chunk(
            points[start:end],
            colors[start:end],
            intensity[start:end],
            classification[start:end],
            voxel_size,
        )
        accums.append(accum)

    # Free GPU memory
    cp.get_default_memory_pool().free_all_blocks()

    merged = _merge_accumulations(accums)
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

    try:
        result = downsample_voxel_grid_gpu(
            points, colors, intensity, classification,
            voxel_size, gpu_chunk_size,
        )
    except cp.cuda.memory.OutOfMemoryError:
        print("    GPU OOM during downsampling — halving chunk size and retrying")
        try:
            result = downsample_voxel_grid_gpu(
                points, colors, intensity, classification,
                voxel_size, gpu_chunk_size // 2,
            )
        except cp.cuda.memory.OutOfMemoryError:
            raise RuntimeError(
                f"GPU out of memory during downsampling even with reduced chunk size. "
                f"Try reducing gpu_chunk_size further (tried: {gpu_chunk_size // 2:,}). "
                f"Set Config.gpu_chunk_size to a smaller value."
            )

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
