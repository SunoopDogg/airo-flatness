"""Point Cloud Viewer CLI — large PLY file memory-efficient visualization."""

import time

from config import Config
from utils import create_progress_bar, select_downsampled_file, select_file, select_source


def main() -> None:
    from loader import ply_loader
    from viewer import visualizer
    from extractor import extract_floor
    from chart import generate_all_charts
    from preprocessing.pipeline import load_and_downsample, load_from_downsampled_cache

    cfg = Config()

    # Source selection
    source = select_source(cfg.downsample_cache_dir)

    if source == "original":
        filepath = select_file(cfg.data_dir)

        # Header
        header = ply_loader.read_ply_header(filepath)
        total = header["vertex_count"]
        print(f"\nFile: {filepath.name}")
        print(f"Total vertices: {total:,}")

        # Load + downsample
        progress = create_progress_bar(label="Loading")
        data = load_and_downsample(filepath, cfg, progress_callback=progress)
    else:
        npz_path = select_downsampled_file(cfg.downsample_cache_dir)
        data = load_from_downsampled_cache(npz_path)
        filepath = npz_path
        print(f"\nFile: {npz_path.name}")
        print(f"Downsampled points: {data['sampled_vertices']:,}")

    print(f"\nProcessing {data['sampled_vertices']:,} / {data['total_vertices']:,} points")

    # Floor extraction
    print("\nExtracting floor (3-stage: peak + Z-filter + intensity/color)...")
    t0 = time.time()
    result = extract_floor(
        data["points"],
        colors=data["colors"],
        intensity=data["intensity"],
        config=cfg,
    )
    extraction_elapsed = time.time() - t0
    print(f"  Floor detected: Z = {result.peak_info.peak_z:.2f}m "
          f"[{result.peak_info.z_min:.2f}, {result.peak_info.z_max:.2f}]  "
          f"FWHM = {result.peak_info.fwhm:.2f}m")
    print(f"  Floor points: {result.floor_points:,} / {result.total_points:,} "
          f"({result.floor_ratio:.1%}) in {extraction_elapsed:.2f}s")

    # Charts
    result_path = generate_all_charts(
        points=data["points"],
        colors=data["colors"],
        intensity=data["intensity"],
        floor_result=result,
        filepath=filepath,
        config=cfg,
        elapsed_time=extraction_elapsed,
    )
    print(f"  Analysis charts saved to: {result_path}")

    # Visualize
    title = (f"{filepath.name} — "
             f"{data['sampled_vertices']:,} / {data['total_vertices']:,} points")
    print(f"\nLaunching viewer (GPU accelerated)...")
    print("  Press 1-4 to switch view mode, S for top-view screenshot")
    visualizer.visualize_point_cloud(
        points=data["points"],
        colors=data["colors"],
        floor_mask=result.floor_mask,
        floor_highlight_color=cfg.floor_highlight_color,
        non_floor_fallback_gray=cfg.non_floor_fallback_gray,
        title=title,
        point_size=cfg.point_size,
        results_dir=result_path,
    )


if __name__ == "__main__":
    main()
