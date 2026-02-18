"""바닥 추출 파이프라인 — 3단계 하이브리드 오케스트레이터."""

from dataclasses import dataclass

import numpy as np

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


def _apply_z_filter(points: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Z-임계값으로 바닥 후보 마스크를 생성한다."""
    z = points[:, 2]
    return (z >= z_min) & (z <= z_max)


def _refine_floor_mask(
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
    num_bins: int = 200,
    band_width: float | None = None,
    width_multiplier: float = 5.0,
    intensity_percentile: float = 25.0,
    color_tolerance: float = 0.4,
    color_std_threshold: float = 0.01,
    prominence_ratio: float = 0.1,
    min_peak_width: int = 3,
    fallback_z_ratio: float = 0.2,
    tilt_fwhm_threshold: float = 2.0,
    tilt_width_multiplier: float = 1.0,
) -> FloorResult:
    """3단계 하이브리드 파이프라인으로 바닥을 추출한다.

    Stage 1: Z-히스토그램 피크 자동 검출
    Stage 2: Z-임계값 필터링
    Stage 3: Intensity/Color 정제

    Args:
        points: (N, 3) 포인트 좌표 배열
        colors: (N, 3) RGB 색상 (0~1), None이면 색상 필터 스킵
        intensity: (N,) Intensity, None이면 Intensity 필터 스킵
        num_bins: 히스토그램 빈 수
        band_width: 고정 반폭(m). None이면 FWHM 자동.
        width_multiplier: FWHM 배율
        intensity_percentile: Stage 3 intensity 하한 백분위수
        color_tolerance: Stage 3 색상 유클리드 거리 허용 범위

    Returns:
        FloorResult with floor_mask, peak_info, statistics
    """
    if len(points) == 0:
        return FloorResult(
            floor_mask=np.array([], dtype=bool),
            peak_info=detect_floor_peak(np.array([])),
            total_points=0,
            floor_points=0,
            floor_ratio=0.0,
            stage_counts=StageFilterCounts(total=0, after_z_filter=0, after_refinement=0),
        )

    # Stage 1: 피크 검출
    peak_info = detect_floor_peak(
        points[:, 2],
        num_bins=num_bins,
        band_width=band_width,
        width_multiplier=width_multiplier,
        prominence_ratio=prominence_ratio,
        min_peak_width=min_peak_width,
        fallback_z_ratio=fallback_z_ratio,
        tilt_fwhm_threshold=tilt_fwhm_threshold,
        tilt_width_multiplier=tilt_width_multiplier,
    )

    # Stage 2: Z-임계값 필터링
    floor_mask = _apply_z_filter(points, peak_info.z_min, peak_info.z_max)
    after_z_filter_count = int(floor_mask.sum())

    # Stage 3: Intensity/Color 정제
    floor_mask = _refine_floor_mask(
        floor_mask,
        colors=colors,
        intensity=intensity,
        intensity_percentile=intensity_percentile,
        color_tolerance=color_tolerance,
        color_std_threshold=color_std_threshold,
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
