"""차트 오케스트레이터 — 6개 차트 + JSON 리포트를 생성한다."""

from datetime import datetime
from pathlib import Path

import numpy as np

from extractor.floor_extractor import FloorResult, StageFilterCounts
from chart.histogram_charts import create_z_histogram_chart, create_intensity_chart, create_color_distance_chart
from chart.summary_charts import create_filtering_funnel_chart, create_floor_ratio_chart
from chart.parameter_sensitivity import create_parameter_sensitivity_chart
from chart.report_writer import write_report
from chart.flatness_heatmap import create_flatness_heatmap_chart
from extractor.flatness_analyzer import analyze_flatness
from config import Config


def generate_all_charts(
    points: np.ndarray,
    colors: np.ndarray | None,
    intensity: np.ndarray | None,
    floor_result: FloorResult,
    filepath: Path,
    config: Config,
    elapsed_time: float = 0.0,
) -> Path:
    """6개 차트 + JSON 리포트를 생성하여 results/{timestamp}/ 에 저장한다.

    Args:
        points: (N, 3) 포인트 좌표
        colors: (N, 3) RGB 색상, None 가능
        intensity: (N,) intensity, None 가능
        floor_result: extract_floor() 반환값
        filepath: 원본 PLY 파일 경로
        config: Config 인스턴스
        elapsed_time: 바닥 추출 소요 시간 (초)

    Returns:
        생성된 결과 폴더 경로
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(config.results_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Generating analysis charts -> {output_dir}/")

    # Chart 1: Z-히스토그램 + 피크 오버레이
    create_z_histogram_chart(
        peak_info=floor_result.peak_info,
        save_path=output_dir / "01_z_histogram_peak.png",
        dpi=config.chart_dpi,
    )

    # Chart 2: 필터링 퍼널
    create_filtering_funnel_chart(
        stage_counts=floor_result.stage_counts,
        save_path=output_dir / "02_filtering_funnel.png",
        dpi=config.chart_dpi,
    )

    # Chart 3: Intensity 히스토그램
    create_intensity_chart(
        intensity=intensity,
        floor_mask=floor_result.floor_mask,
        intensity_percentile=config.intensity_percentile,
        save_path=output_dir / "03_intensity_histogram.png",
        dpi=config.chart_dpi,
    )

    # Chart 4: 색상 거리 히스토그램
    create_color_distance_chart(
        colors=colors,
        floor_mask=floor_result.floor_mask,
        color_tolerance=config.color_tolerance,
        color_std_threshold=config.color_std_threshold,
        save_path=output_dir / "04_color_distance.png",
        dpi=config.chart_dpi,
    )

    # Chart 5: 바닥/비바닥 비율 도넛 차트
    create_floor_ratio_chart(
        floor_result=floor_result,
        filename=filepath.name,
        elapsed_time=elapsed_time,
        save_path=output_dir / "05_floor_ratio.png",
        dpi=config.chart_dpi,
    )

    # Chart 6: 파라미터 민감도 (반복 실행)
    sensitivity_data = create_parameter_sensitivity_chart(
        points=points,
        colors=colors,
        intensity=intensity,
        config=config,
        save_path=output_dir / "06_parameter_sensitivity.png",
        dpi=config.chart_dpi,
    )

    # Chart 7: 바닥 평탄도 히트맵
    floor_points = points[floor_result.floor_mask]
    flatness_result = analyze_flatness(
        floor_points,
        target_grid_size=config.flatness_target_grid,
        min_points_per_cell=config.flatness_min_points,
    )
    create_flatness_heatmap_chart(
        flatness_result=flatness_result,
        save_path=output_dir / "07_flatness_heatmap.png",
        dpi=config.chart_dpi,
    )

    # JSON 리포트
    write_report(
        floor_result=floor_result,
        sensitivity_data=sensitivity_data,
        filepath=filepath,
        config=config,
        actual_points_loaded=len(points),
        elapsed_time=elapsed_time,
        output_path=output_dir / "report.json",
        flatness_data={
            "mean_tilt_degrees": round(float(flatness_result.mean_tilt), 4),
            "max_tilt_degrees": round(float(flatness_result.max_tilt), 4),
            "cell_size_meters": round(float(flatness_result.cell_size), 4),
            "valid_cells": flatness_result.valid_cell_count,
            "total_cells": flatness_result.total_cell_count,
        },
    )

    print(f"  Charts saved: {output_dir}/")
    return output_dir
