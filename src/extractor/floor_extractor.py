"""바닥 추출 파이프라인 — 3단계 하이브리드 오케스트레이터."""

from dataclasses import dataclass

import numpy as np

from config import Config
from .peak_detector import PeakInfo, detect_floor_peak


@dataclass
class StageFilterCounts:
    """각 필터링 단계 후의 포인트 수."""

    total: int
    after_z_filter: int
    after_refinement: int


@dataclass
class FloorResult:
    """바닥 추출 결과."""

    floor_mask: np.ndarray
    peak_info: PeakInfo
    total_points: int
    floor_points: int
    floor_ratio: float
    stage_counts: StageFilterCounts


def _create_z_mask(points: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Z-임계값으로 바닥 후보 마스크를 생성한다."""
    z = points[:, 2]
    return (z >= z_min) & (z <= z_max)


def _refine_by_intensity_color(
    z_mask: np.ndarray,
    colors: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    intensity_percentile: float = 25.0,
    color_tolerance: float = 0.3,
    color_std_threshold: float = 0.01,
) -> np.ndarray:
    """Intensity/Color로 바닥 마스크를 정제한다.

    z_mask AND (intensity_ok OR color_ok) 논리 적용.
    intensity와 colors 모두 None이면 z_mask를 그대로 반환.
    """
    has_filter = False
    refinement = np.zeros(len(z_mask), dtype=bool)

    if intensity is not None and z_mask.any():
        floor_intensity = intensity[z_mask]
        threshold = np.percentile(floor_intensity, intensity_percentile)
        refinement |= intensity >= threshold
        has_filter = True

    if colors is not None and z_mask.any():
        floor_colors = colors[z_mask]
        if floor_colors.std(axis=0).mean() > color_std_threshold:
            floor_mean = floor_colors.mean(axis=0)
            dist = np.linalg.norm(colors - floor_mean, axis=1)
            refinement |= dist <= color_tolerance
            has_filter = True

    if has_filter:
        return z_mask & refinement
    return z_mask


def extract_floor(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    config: Config | None = None,
) -> FloorResult:
    """3-stage hybrid pipeline for floor extraction.

    Args:
        points: (N, 3) point coordinates.
        colors: (N, 3) RGB colors (0~1), None to skip color filter.
        intensity: (N,) intensity values, None to skip intensity filter.
        config: Config object with extraction parameters. Uses defaults if None.
    """
    if config is None:
        config = Config()

    if len(points) == 0:
        return FloorResult(
            floor_mask=np.array([], dtype=bool),
            peak_info=detect_floor_peak(np.array([])),
            total_points=0,
            floor_points=0,
            floor_ratio=0.0,
            stage_counts=StageFilterCounts(total=0, after_z_filter=0, after_refinement=0),
        )

    # Stage 1: peak detection
    peak_info = detect_floor_peak(
        points[:, 2],
        num_bins=config.num_bins,
        width_multiplier=config.width_multiplier,
        prominence_ratio=config.prominence_ratio,
        min_peak_width=config.min_peak_width,
        fallback_z_ratio=config.fallback_z_ratio,
        tilt_fwhm_threshold=config.tilt_fwhm_threshold,
        tilt_width_multiplier=config.tilt_width_multiplier,
    )

    # Stage 2: Z-threshold filtering
    floor_mask = _create_z_mask(points, peak_info.z_min, peak_info.z_max)
    after_z_filter_count = int(floor_mask.sum())

    # Stage 3: Intensity/Color refinement
    floor_mask = _refine_by_intensity_color(
        floor_mask,
        colors=colors,
        intensity=intensity,
        intensity_percentile=config.intensity_percentile,
        color_tolerance=config.color_tolerance,
        color_std_threshold=config.color_std_threshold,
    )

    floor_count = int(floor_mask.sum())
    total_count = len(points)

    return FloorResult(
        floor_mask=floor_mask,
        peak_info=peak_info,
        total_points=total_count,
        floor_points=floor_count,
        floor_ratio=floor_count / total_count if total_count > 0 else 0.0,
        stage_counts=StageFilterCounts(
            total=total_count,
            after_z_filter=after_z_filter_count,
            after_refinement=floor_count,
        ),
    )
