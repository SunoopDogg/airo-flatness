"""Floor Roughness Figure Tool — interactive ROI selection + publication figures."""

from datetime import datetime
from pathlib import Path

import matplotlib
# Set interactive backend once here — this is a separate entry point from main.py,
# so no conflict with Agg backend used in chart modules.
matplotlib.use("TkAgg")

from config import Config
from utils import create_progress_bar, select_downsampled_file, select_file, select_source


def main() -> None:
    from loader import ply_loader
    from figure.roi_selector import select_roi, filter_points_by_roi, select_z_roi, filter_points_by_z
    from figure.height_profile import compute_height_profile, plot_height_profile
    from figure.surface_3d import build_surface_mesh
    import numpy as np

    cfg = Config()

    # [1] Source selection
    source = select_source(cfg.downsample_cache_dir)

    from preprocessing.pipeline import load_and_downsample, load_from_downsampled_cache

    if source == "original":
        filepath = select_file(cfg.data_dir)

        header = ply_loader.read_ply_header(filepath)
        total = header["vertex_count"]
        print(f"\nFile: {filepath.name}")
        print(f"Total vertices: {total:,}")

        progress = create_progress_bar(label="Loading")
        data = load_and_downsample(filepath, cfg, progress_callback=progress, gpu=True)
    else:
        npz_path = select_downsampled_file(cfg.downsample_cache_dir)
        data = load_from_downsampled_cache(npz_path, gpu=True)
        filepath = npz_path
        print(f"\nFile: {npz_path.name}")
        print(f"Downsampled points: {data['sampled_vertices']:,}")

    print(f"\nProcessing {data['sampled_vertices']:,} / {data['total_vertices']:,} points")

    gpu_points = data["points"]  # CuPy array (GPU) or NumPy if no GPU
    colors = data["colors"]       # NumPy array (CPU)

    # CPU copy for ROI selection (matplotlib) and downstream CPU ops
    from figure.detrend import _CUPY_COMPUTE_OK
    try:
        import cupy as cp
        _HAS_CUPY = _CUPY_COMPUTE_OK  # Only use GPU if compute actually works
        points_cpu = cp.asnumpy(gpu_points) if isinstance(gpu_points, cp.ndarray) else gpu_points
    except ImportError:
        _HAS_CUPY = False
        points_cpu = gpu_points

    # [3] ROI selection (with retry loop)
    while True:
        print("\nSelect ROI region on the top-view plot (click 4 points, Q to apply).")
        try:
            roi = select_roi(
                points_cpu,
                max_display=cfg.fig_roi_subsample,
                seed=cfg.random_seed,
            )
        except ValueError:
            print("  No region selected. Please try again.")
            continue

        roi_points_cpu = filter_points_by_roi(points_cpu, roi)
        n_roi = len(roi_points_cpu)

        if n_roi < cfg.fig_roi_min_points:
            print(f"  Selected region has only {n_roi} points "
                  f"(minimum: {cfg.fig_roi_min_points}). Please select a larger area.")
            continue

        v = roi.vertices
        print(f"  ROI vertices: ({v[0,0]:.2f},{v[0,1]:.2f}) ({v[1,0]:.2f},{v[1,1]:.2f}) "
              f"({v[2,0]:.2f},{v[2,1]:.2f}) ({v[3,0]:.2f},{v[3,1]:.2f})")
        print(f"  Points in ROI: {n_roi:,}")
        break

    # [3c] Create output directory (moved up from [4])
    stem = filepath.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = cfg.results_dir / f"{stem}_figure_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {save_dir}")

    # [3b] Z ROI selection (with retry loop)
    while True:
        print("\nSelect Z range on the histogram (drag to select, then close window).")
        try:
            z_min, z_max = select_z_roi(roi_points_cpu)
        except ValueError:
            print("  No Z range selected. Please try again.")
            continue

        z_filtered_cpu = filter_points_by_z(roi_points_cpu, z_min, z_max)
        n_z = len(z_filtered_cpu)

        if n_z < cfg.fig_roi_min_points:
            print(f"  Z range has only {n_z} points "
                  f"(minimum: {cfg.fig_roi_min_points}). Please select a wider range.")
            continue

        print(f"  Z range: [{z_min:.4f}, {z_max:.4f}]")
        print(f"  Points after Z filter: {n_z:,}")
        break

    roi_points_gpu = cp.asarray(z_filtered_cpu) if _HAS_CUPY else z_filtered_cpu
    roi_points_cpu = z_filtered_cpu

    # [5] Compute cell size for height profile binning
    longest = max(
        float(roi_points_cpu[:, 0].max() - roi_points_cpu[:, 0].min()),
        float(roi_points_cpu[:, 1].max() - roi_points_cpu[:, 1].min()),
    )
    cell_size = longest / cfg.fig_heatmap_target_grid if longest > 0 else 1.0

    # [5c] Compute and save height profile
    print("\nGenerating X-direction height profile...")
    x_centers, z_means = compute_height_profile(roi_points_gpu, cell_size)
    if len(x_centers) > 0:
        plot_height_profile(x_centers, z_means, save_dir, dpi=cfg.fig_dpi)
        print("  Saved: height_profile.png")
    else:
        print("  Warning: no valid bins for height profile.")

    # [6] 3D surface mesh (interactive — launched last)
    print("\nBuilding 3D surface mesh...")
    try:
        mesh = build_surface_mesh(
            roi_points_cpu,
            max_points=cfg.fig_delaunay_max_points,
            grid_resolution=cfg.fig_grid_resolution,
            z_exaggeration=cfg.fig_z_exaggeration,
            seed=cfg.random_seed,
        )
    except ValueError as e:
        print(f"  Warning: {e}")
        print("  Skipping 3D surface mesh.")
        print(f"\nFigures saved to: {save_dir}")
        return

    print(f"  Mesh: {mesh.n_points:,} vertices, {mesh.n_cells:,} cells")

    # [3d] ROI context views (interactive, RGB with mesh overlay)
    from figure.roi_context import render_roi_context_2d, render_roi_context_3d
    if colors is not None:
        print("\nOpening ROI context viewers (S: capture, Q: close)...")
        render_roi_context_2d(
            points_cpu, roi, save_dir,
            colors=colors,
            mesh=mesh,
            z_range=(z_min, z_max),
            point_size=cfg.fig_point_size,
        )
        render_roi_context_3d(
            points_cpu, roi, save_dir,
            colors=colors,
            mesh=mesh,
            z_range=(z_min, z_max),
            point_size=cfg.fig_point_size,
        )
    else:
        print("  Warning: no color data — skipping ROI context views.")

    print(f"\nAll figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
