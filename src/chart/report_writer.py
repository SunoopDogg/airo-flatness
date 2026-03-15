"""JSON 요약 리포트 생성기."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from extractor.floor_extractor import FloorResult
from config import Config


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
    config: Config,
    actual_points_loaded: int,
    elapsed_time: float,
    output_path: Path,
    flatness_data: dict | None = None,
) -> None:
    """분석 결과를 JSON 리포트로 저장한다.

    Args:
        floor_result: extract_floor() 반환값
        sensitivity_data: create_parameter_sensitivity_chart() 반환값
        filepath: 원본 PLY 파일 경로
        config: Config 인스턴스
        actual_points_loaded: 실제 로드된 포인트 수
        elapsed_time: 바닥 추출 소요 시간 (초)
        output_path: 저장할 report.json 경로
        flatness_data: 평탄도 분석 결과 dict (optional)
    """
    peak = floor_result.peak_info
    sc = floor_result.stage_counts

    report = {
        "metadata": {
            "file": filepath.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "view_mode": 4,
            "max_points_loaded": config.max_points,
            "actual_points_loaded": actual_points_loaded,
            "elapsed_seconds": round(elapsed_time, 3),
        },
        "peak_detection": {
            "peak_z": round(float(peak.peak_z), 6),
            "z_min": round(float(peak.z_min), 6),
            "z_max": round(float(peak.z_max), 6),
            "fwhm": round(float(peak.fwhm), 6),
            "num_bins": config.num_bins,
        },
        "filtering": {
            "total_points": sc.total,
            "after_z_filter": sc.after_z_filter,
            "after_refinement": sc.after_refinement,
            "floor_ratio": round(floor_result.floor_ratio, 6),
        },
        "parameters_used": {
            "width_multiplier": config.width_multiplier,
            "intensity_percentile": config.intensity_percentile,
            "color_tolerance": config.color_tolerance,
            "num_bins": config.num_bins,
        },
        "sensitivity": sensitivity_data,
    }
    if flatness_data is not None:
        report["flatness"] = flatness_data

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, cls=_NumpyEncoder))
