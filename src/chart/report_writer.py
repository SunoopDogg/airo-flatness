"""JSON 요약 리포트 생성기."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from extractor.floor_extractor import FloorResult


class _NumpyEncoder(json.JSONEncoder):
    """numpy 타입을 Python 기본 타입으로 변환하는 JSON 인코더."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_report(
    floor_result: FloorResult,
    sensitivity_data: dict,
    filepath: Path,
    view_mode: int,
    max_points_loaded: int,
    actual_points_loaded: int,
    elapsed_time: float,
    width_multiplier: float,
    intensity_percentile: float,
    color_tolerance: float,
    num_bins: int,
    output_path: Path,
    flatness_data: dict | None = None,
) -> None:
    """분석 결과를 JSON 리포트로 저장한다.

    Args:
        floor_result: extract_floor() 반환값
        sensitivity_data: create_parameter_sensitivity_chart() 반환값
        filepath: 원본 PLY 파일 경로
        view_mode: 사용된 뷰 모드 번호
        max_points_loaded: 최대 로드 가능 포인트 수 설정값
        actual_points_loaded: 실제 로드된 포인트 수
        elapsed_time: 바닥 추출 소요 시간 (초)
        width_multiplier: 사용된 width_multiplier 파라미터
        intensity_percentile: 사용된 intensity_percentile 파라미터
        color_tolerance: 사용된 color_tolerance 파라미터
        num_bins: 사용된 num_bins 파라미터
        output_path: 저장할 report.json 경로
    """
    peak = floor_result.peak_info
    sc = floor_result.stage_counts

    report = {
        "metadata": {
            "file": filepath.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "view_mode": view_mode,
            "max_points_loaded": max_points_loaded,
            "actual_points_loaded": actual_points_loaded,
            "elapsed_seconds": round(elapsed_time, 3),
        },
        "peak_detection": {
            "peak_z": round(float(peak.peak_z), 6),
            "z_min": round(float(peak.z_min), 6),
            "z_max": round(float(peak.z_max), 6),
            "fwhm": round(float(peak.fwhm), 6),
            "num_bins": num_bins,
        },
        "filtering": {
            "total_points": sc.total,
            "after_z_filter": sc.after_z_filter,
            "after_refinement": sc.after_refinement,
            "floor_ratio": round(floor_result.floor_ratio, 6),
        },
        "parameters_used": {
            "width_multiplier": width_multiplier,
            "intensity_percentile": intensity_percentile,
            "color_tolerance": color_tolerance,
            "num_bins": num_bins,
        },
        "sensitivity": sensitivity_data,
    }
    if flatness_data is not None:
        report["flatness"] = flatness_data

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, cls=_NumpyEncoder))
