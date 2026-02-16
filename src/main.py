"""Point Cloud Viewer CLI — 대용량 PLY 파일을 메모리 효율적으로 시각화."""

import sys
import time
from pathlib import Path

from config import Config


def format_size(size_bytes: int) -> str:
    """바이트를 사람이 읽기 쉬운 단위로 변환한다."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def select_file(data_dir: Path) -> Path:
    """data/ 디렉토리에서 PLY 파일 목록을 표시하고 사용자가 선택하게 한다."""
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


def main() -> None:
    from loader import ply_loader
    from viewer import visualizer
    from extractor import extract_floor
    from chart import generate_all_charts

    cfg = Config()
    filepath = select_file(cfg.data_dir)

    # 헤더 읽기
    header = ply_loader.read_ply_header(filepath)
    total = header["vertex_count"]
    print(f"\nFile: {filepath.name}")
    print(f"Total vertices: {total:,}")
    print(f"Sampling: {cfg.max_points:,} points")
    print()

    # 프로그레스 바
    start_time = time.time()
    bar_width = 40

    def progress(current: int, total: int) -> None:
        pct = current / total
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        elapsed = time.time() - start_time
        eta = (elapsed / pct - elapsed) if pct > 0 else 0
        print(
            f"\rSampling: [{bar}] {pct:6.1%}  "
            f"({current:,} / {total:,})  "
            f"ETA: {eta:.0f}s  ",
            end="",
            flush=True,
        )

    # 로딩
    data = ply_loader.load_ply_sampled(
        filepath,
        max_points=cfg.max_points,
        progress_callback=progress,
        seed=cfg.random_seed,
        chunk_size=cfg.chunk_size,
    )
    elapsed = time.time() - start_time
    print(f"\n\nLoaded {data['sampled_vertices']:,} points in {elapsed:.1f}s")

    # 바닥 추출 (항상 실행)
    print("\nExtracting floor (3-stage: peak + Z-filter + intensity/color)...")
    t0 = time.time()
    result = extract_floor(
        data["points"],
        colors=data["colors"],
        intensity=data["intensity"],
        num_bins=cfg.num_bins,
        width_multiplier=cfg.width_multiplier,
        intensity_percentile=cfg.intensity_percentile,
        color_tolerance=cfg.color_tolerance,
        color_std_threshold=cfg.color_std_threshold,
        prominence_ratio=cfg.prominence_ratio,
        min_peak_width=cfg.min_peak_width,
        fallback_z_ratio=cfg.fallback_z_ratio,
        tilt_fwhm_threshold=cfg.tilt_fwhm_threshold,
        tilt_width_multiplier=cfg.tilt_width_multiplier,
    )
    dt = time.time() - t0
    print(f"  Floor detected: Z = {result.peak_info.peak_z:.2f}m "
          f"[{result.peak_info.z_min:.2f}, {result.peak_info.z_max:.2f}]  "
          f"FWHM = {result.peak_info.fwhm:.2f}m")
    print(f"  Floor points: {result.floor_points:,} / {result.total_points:,} "
          f"({result.floor_ratio:.1%}) in {dt:.2f}s")

    # 차트 생성 (항상 실행)
    result_path = generate_all_charts(
        points=data["points"],
        colors=data["colors"],
        intensity=data["intensity"],
        floor_result=result,
        filepath=filepath,
        view_mode=4,
        elapsed_time=dt,
        width_multiplier=cfg.width_multiplier,
        intensity_percentile=cfg.intensity_percentile,
        color_tolerance=cfg.color_tolerance,
        num_bins=cfg.num_bins,
        max_points_loaded=cfg.max_points,
        results_dir=cfg.results_dir,
        dpi=cfg.chart_dpi,
        width_multiplier_sweep=cfg.width_multiplier_sweep,
        color_tolerance_sweep=cfg.color_tolerance_sweep,
        intensity_percentile_sweep=cfg.intensity_percentile_sweep,
        max_subsample=cfg.sensitivity_max_subsample,
        flatness_target_grid=cfg.flatness_target_grid,
        flatness_min_points=cfg.flatness_min_points,
    )
    print(f"  Analysis charts saved to: {result_path}")

    # 시각화
    title = (f"{filepath.name} — "
             f"{data['sampled_vertices']:,} / {data['total_vertices']:,} points")
    print(f"\nLaunching viewer (GPU accelerated)...")
    print("  Press 1-4 to switch view mode, S for top-view screenshot")
    visualizer.visualize_point_cloud(
        points=data["points"],
        colors=data["colors"],
        floor_mask=result.floor_mask,
        floor_highlight_color=cfg.floor_highlight_color,
        title=title,
        point_size=cfg.point_size,
        results_dir=result_path,
    )


if __name__ == "__main__":
    main()
