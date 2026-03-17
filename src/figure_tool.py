"""Floor Roughness Figure Tool — interactive ROI selection + 3 publication figures."""

import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
# Set interactive backend once here — this is a separate entry point from main.py,
# so no conflict with Agg backend used in chart modules.
matplotlib.use("TkAgg")

from config import Config
from utils import create_progress_bar, select_file


def main() -> None:
    from loader import ply_loader
    from figure.roi_selector import select_roi, filter_points_by_roi, select_z_roi, filter_points_by_z
    from figure.height_heatmap import compute_height_grid, plot_height_heatmap
    from figure.height_profile import compute_height_profile, plot_height_profile
    from figure.surface_3d import build_surface_mesh, show_surface_viewer
    import numpy as np

    cfg = Config()

    # [1] File selection
    filepath = select_file(cfg.data_dir)

    from preprocessing.pipeline import load_and_downsample

    header = ply_loader.read_ply_header(filepath)
    total = header["vertex_count"]
    print(f"\nFile: {filepath.name}")
    print(f"Total vertices: {total:,}")

    # [2] Load + downsample
    progress = create_progress_bar(label="Loading")
    data = load_and_downsample(filepath, cfg, progress_callback=progress)
    print(f"\nProcessing {data['sampled_vertices']:,} / {data['total_vertices']:,} points")

    points = data["points"]
    colors = data["colors"]

    # [3] ROI selection (with retry loop)
    while True:
        print("\nSelect ROI region on the top-view plot (click 4 points, Q to apply).")
        try:
            roi = select_roi(
                points,
                max_display=cfg.fig_roi_subsample,
                seed=cfg.random_seed,
            )
        except ValueError:
            print("  No region selected. Please try again.")
            continue

        roi_points = filter_points_by_roi(points, roi)
        n_roi = len(roi_points)

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
            z_min, z_max = select_z_roi(roi_points)
        except ValueError:
            print("  No Z range selected. Please try again.")
            continue

        z_filtered = filter_points_by_z(roi_points, z_min, z_max)
        n_z = len(z_filtered)

        if n_z < cfg.fig_roi_min_points:
            print(f"  Z range has only {n_z} points "
                  f"(minimum: {cfg.fig_roi_min_points}). Please select a wider range.")
            continue

        print(f"  Z range: [{z_min:.4f}, {z_max:.4f}]")
        print(f"  Points after Z filter: {n_z:,}")
        break

    roi_points = z_filtered

    # [5] Compute heatmap grid
    print("\nGenerating height deviation heatmap...")
    grid, x_edges, y_edges, cell_size = compute_height_grid(
        roi_points,
        target_grid=cfg.fig_heatmap_target_grid,
        min_points=cfg.fig_heatmap_min_points,
    )

    # [5b] Save heatmap
    plot_height_heatmap(grid, x_edges, y_edges, cell_size, save_dir, dpi=cfg.fig_dpi)
    print("  Saved: height_heatmap.png, height_heatmap.pdf")

    # [5c] Compute and save height profile
    print("\nGenerating X-direction height profile...")
    x_centers, z_means = compute_height_profile(roi_points, cell_size)
    if len(x_centers) > 0:
        plot_height_profile(x_centers, z_means, save_dir, dpi=cfg.fig_dpi)
        print("  Saved: height_profile.png, height_profile.pdf")
    else:
        print("  Warning: no valid bins for height profile.")

    # [6] 3D surface mesh (interactive — launched last)
    print("\nBuilding 3D surface mesh...")
    try:
        mesh = build_surface_mesh(
            roi_points,
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

    # [3d] ROI context views (rendered after mesh is built so RGB variants get mesh overlay)
    print("\nGenerating ROI context views...")
    from figure.roi_context import render_roi_context_2d, render_roi_context_3d
    render_roi_context_2d(
        points, roi, save_dir,
        max_points=cfg.fig_roi_subsample,
        seed=cfg.random_seed,
        dpi=cfg.fig_dpi,
    )
    print("  Saved: roi_context_2d.png")
    render_roi_context_3d(
        points, roi, save_dir,
        max_points=cfg.fig_roi_subsample,
        seed=cfg.random_seed,
        dpi=cfg.fig_dpi,
    )
    print("  Saved: roi_context_3d.png")

    if colors is not None:
        render_roi_context_2d(
            points, roi, save_dir,
            max_points=cfg.fig_roi_subsample,
            seed=cfg.random_seed,
            dpi=cfg.fig_dpi,
            colors=colors,
            mesh=mesh,
            z_range=(z_min, z_max),
        )
        print("  Saved: roi_context_2d_rgb.png")
        render_roi_context_3d(
            points, roi, save_dir,
            max_points=cfg.fig_roi_subsample,
            seed=cfg.random_seed,
            dpi=cfg.fig_dpi,
            colors=colors,
            mesh=mesh,
            z_range=(z_min, z_max),
        )
        print("  Saved: roi_context_3d_rgb.png")

    print("\nLaunching 3D viewer...")
    print("  Keys: [S] capture, [I] isometric, [M] multi-view, [Q] quit")
    show_surface_viewer(mesh, save_dir, dpi=cfg.fig_dpi)

    print(f"\nAll figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
