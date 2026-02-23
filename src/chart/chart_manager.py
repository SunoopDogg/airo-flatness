"""차트 오케스트레이터 — 6개 차트 + JSON 리포트를 생성한다."""

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from extractor.floor_extractor import FloorResult, StageFilterCounts
from chart.z_histogram import create_z_histogram_chart
from chart.filtering_funnel import create_filtering_funnel_chart
from chart.intensity_chart import create_intensity_chart
from chart.color_distance import create_color_distance_chart
from chart.floor_ratio import create_floor_ratio_chart
from chart.parameter_sensitivity import create_parameter_sensitivity_chart
from chart.report_writer import write_report
from chart.flatness_heatmap import create_flatness_heatmap_chart
from extractor.flatness_analyzer import analyze_flatness


def generate_all_charts(
    points: np.ndarray,
    colors: np.ndarray | None,
    intensity: np.ndarray | None,
    floor_result: FloorResult,
    filepath: Path,
    view_mode: int,
    elapsed_time: float = 0.0,
    width_multiplier: float = 5.0,
    intensity_percentile: float = 25.0,
    color_tolerance: float = 0.4,
    num_bins: int = 200,
    max_points_loaded: int = 5_000_000,
    results_dir: Path = Path("results"),
    dpi: int = 150,
    width_multiplier_sweep: list[float] | None = None,
    color_tolerance_sweep: list[float] | None = None,
    intensity_percentile_sweep: list[float] | None = None,
    max_subsample: int = 500_000,
    flatness_target_grid: int = 150,
    flatness_min_points: int = 3,
) -> Path:
    """6개 차트 + JSON 리포트를 생성하여 results/{timestamp}/ 에 저장한다.

    Args:
        points: (N, 3) 포인트 좌표
        colors: (N, 3) RGB 색상, None 가능
        intensity: (N,) intensity, None 가능
        floor_result: extract_floor() 반환값
        filepath: 원본 PLY 파일 경로
        view_mode: 사용된 뷰 모드 번호
        elapsed_time: 바닥 추출 소요 시간 (초)
        width_multiplier: 사용된 width_multiplier 파라미터
        intensity_percentile: 사용된 intensity_percentile 파라미터
        color_tolerance: 사용된 color_tolerance 파라미터
        num_bins: 사용된 num_bins 파라미터
        max_points_loaded: 최대 로드 설정값
        results_dir: 결과 저장 루트 폴더

    Returns:
        생성된 결과 폴더 경로
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(results_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Generating analysis charts -> {out_dir}/")

    # Chart 1: Z-히스토그램 + 피크 오버레이
    create_z_histogram_chart(
        peak_info=floor_result.peak_info,
        save_path=out_dir / "01_z_histogram_peak.png",
        dpi=dpi,
    )

    # Chart 2: 필터링 퍼널
    create_filtering_funnel_chart(
        stage_counts=floor_result.stage_counts,
        save_path=out_dir / "02_filtering_funnel.png",
        dpi=dpi,
    )

    # Chart 3: Intensity 히스토그램
    create_intensity_chart(
        intensity=intensity,
        floor_mask=floor_result.floor_mask,
        intensity_percentile=intensity_percentile,
        save_path=out_dir / "03_intensity_histogram.png",
        dpi=dpi,
    )

    # Chart 4: 색상 거리 히스토그램
    create_color_distance_chart(
        colors=colors,
        floor_mask=floor_result.floor_mask,
        color_tolerance=color_tolerance,
        save_path=out_dir / "04_color_distance.png",
        dpi=dpi,
    )

    # Chart 5: 바닥/비바닥 비율 도넛 차트
    create_floor_ratio_chart(
        floor_result=floor_result,
        filename=filepath.name,
        elapsed_time=elapsed_time,
        save_path=out_dir / "05_floor_ratio.png",
        dpi=dpi,
    )

    # Chart 6: 파라미터 민감도 (반복 실행)
    sensitivity_data = create_parameter_sensitivity_chart(
        points=points,
        colors=colors,
        intensity=intensity,
        width_multiplier=width_multiplier,
        color_tolerance=color_tolerance,
        intensity_percentile=intensity_percentile,
        save_path=out_dir / "06_parameter_sensitivity.png",
        width_multiplier_sweep=width_multiplier_sweep,
        color_tolerance_sweep=color_tolerance_sweep,
        intensity_percentile_sweep=intensity_percentile_sweep,
        max_subsample=max_subsample,
        dpi=dpi,
    )

    # Chart 7: 바닥 평탄도 히트맵
    floor_points = points[floor_result.floor_mask]
    flatness_result = analyze_flatness(
        floor_points,
        target_grid_size=flatness_target_grid,
        min_points_per_cell=flatness_min_points,
    )
    create_flatness_heatmap_chart(
        flatness_result=flatness_result,
        save_path=out_dir / "07_flatness_heatmap.png",
        dpi=dpi,
    )

    # JSON 리포트
    write_report(
        floor_result=floor_result,
        sensitivity_data=sensitivity_data,
        filepath=filepath,
        view_mode=view_mode,
        max_points_loaded=max_points_loaded,
        actual_points_loaded=len(points),
        elapsed_time=elapsed_time,
        width_multiplier=width_multiplier,
        intensity_percentile=intensity_percentile,
        color_tolerance=color_tolerance,
        num_bins=num_bins,
        output_path=out_dir / "report.json",
        flatness_data={
            "mean_tilt_degrees": round(float(flatness_result.mean_tilt), 4),
            "max_tilt_degrees": round(float(flatness_result.max_tilt), 4),
            "cell_size_meters": round(float(flatness_result.cell_size), 4),
            "valid_cells": flatness_result.valid_cell_count,
            "total_cells": flatness_result.total_cell_count,
        },
    )

    print(f"  Charts saved: {out_dir}/")
    return out_dir
