"""Z-히스토그램 피크 검출 — scipy.signal.find_peaks 기반 바닥 평면 자동 탐지."""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class PeakInfo:
    """검출된 바닥 피크 정보."""

    peak_z: float
    z_min: float
    z_max: float
    fwhm: float
    bin_edges: np.ndarray
    counts: np.ndarray


def detect_floor_peak(
    z_values: np.ndarray,
    num_bins: int = 200,
    prominence_ratio: float = 0.1,
    band_width: float | None = None,
    width_multiplier: float = 5.0,
    min_peak_width: int = 3,
    fallback_z_ratio: float = 0.2,
    tilt_fwhm_threshold: float = 2.0,
    tilt_width_multiplier: float = 1.0,
) -> PeakInfo:
    """Z축 히스토그램에서 바닥 피크를 자동 검출하여 Z 범위를 반환한다.

    Args:
        z_values: (N,) Z좌표 배열
        num_bins: 히스토그램 빈 수
        prominence_ratio: 최대 빈 카운트 대비 최소 prominence 비율
        band_width: 고정 반폭(m). None이면 FWHM 기반 자동 계산.
        width_multiplier: FWHM 대비 바닥 범위 배율 (band_width=None일 때 사용, 기본 2.5)

    Returns:
        PeakInfo with peak_z, z_min, z_max, fwhm, histogram data
    """
    if len(z_values) == 0:
        empty = np.array([], dtype=np.float64)
        return PeakInfo(
            peak_z=0.0, z_min=0.0, z_max=0.0, fwhm=0.0,
            bin_edges=empty, counts=np.array([], dtype=np.intp),
        )

    counts, bin_edges = np.histogram(z_values, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # 피크 검출
    peaks, props = find_peaks(
        counts,
        prominence=counts.max() * prominence_ratio,
        width=min_peak_width,
    )

    if len(peaks) == 0:
        # Fallback: 하위 fallback_z_ratio Z 범위를 바닥으로 간주
        z_range = z_values.max() - z_values.min()
        fallback_min = z_values.min()
        fallback_max = z_values.min() + z_range * fallback_z_ratio
        fallback_center = (fallback_min + fallback_max) / 2
        warnings.warn(
            f"No peaks found in Z histogram. Using fallback: "
            f"Z = [{fallback_min:.2f}, {fallback_max:.2f}]",
            stacklevel=2,
        )
        return PeakInfo(
            peak_z=fallback_center,
            z_min=fallback_min,
            z_max=fallback_max,
            fwhm=fallback_max - fallback_min,
            bin_edges=bin_edges,
            counts=counts,
        )

    # 가장 큰 피크 선택
    main_idx = np.argmax(counts[peaks])
    peak_z = bin_centers[peaks[main_idx]]
    fwhm = props["widths"][main_idx] * bin_width

    # 기울어진 바닥 경고
    if fwhm > tilt_fwhm_threshold:
        warnings.warn(
            f"Floor FWHM is unusually wide ({fwhm:.2f}m). "
            f"Floor may be tilted. Reducing width_multiplier to {tilt_width_multiplier}.",
            stacklevel=2,
        )
        width_multiplier = tilt_width_multiplier

    # Z 범위 결정
    if band_width is not None:
        half_w = band_width
    else:
        half_w = fwhm * width_multiplier / 2

    return PeakInfo(
        peak_z=peak_z,
        z_min=peak_z - half_w,
        z_max=peak_z + half_w,
        fwhm=fwhm,
        bin_edges=bin_edges,
        counts=counts,
    )
