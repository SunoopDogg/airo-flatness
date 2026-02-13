"""Z-히스토그램 + 피크 오버레이 차트 생성."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from extractor.peak_detector import PeakInfo


def create_z_histogram_chart(
    peak_info: PeakInfo,
    save_path: Path,
    dpi: int = 150,
    figsize: tuple = (12, 6),
) -> None:
    """Z축 히스토그램과 바닥 피크 검출 결과를 시각화한다.

    Args:
        peak_info: detect_floor_peak()가 반환한 피크 정보
        save_path: 저장할 이미지 파일 경로
        dpi: 출력 이미지 DPI
        figsize: 그림 크기 (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # bin_edges로부터 bin_centers 계산
    bin_centers = (peak_info.bin_edges[:-1] + peak_info.bin_edges[1:]) / 2
    bin_width = peak_info.bin_edges[1] - peak_info.bin_edges[0]

    # 막대 그래프: bin_centers vs counts
    ax.bar(
        bin_centers,
        peak_info.counts,
        width=bin_width * 0.9,
        color="steelblue",
        alpha=0.7,
        label="Point Count",
    )

    # 파란 음영: z_min ~ z_max 범위
    ax.axvspan(
        peak_info.z_min,
        peak_info.z_max,
        alpha=0.15,
        color="blue",
        label=f"Floor Band [{peak_info.z_min:.3f}, {peak_info.z_max:.3f}] m",
    )

    # 빨간 수직 점선: peak_z
    ax.axvline(
        peak_info.peak_z,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label=f"Peak Z = {peak_info.peak_z:.3f} m",
    )

    # 초록 라인: FWHM 표시 (피크 높이의 절반 수준에서 수평선)
    # peak_z에 가장 가까운 bin의 카운트 찾기
    if len(bin_centers) > 0:
        peak_bin_idx = np.argmin(np.abs(bin_centers - peak_info.peak_z))
        peak_count = peak_info.counts[peak_bin_idx]
        half_max = peak_count / 2.0
        fwhm_left = peak_info.peak_z - peak_info.fwhm / 2
        fwhm_right = peak_info.peak_z + peak_info.fwhm / 2
        ax.hlines(
            half_max,
            fwhm_left,
            fwhm_right,
            colors="green",
            linewidths=2.5,
            label=f"FWHM = {peak_info.fwhm:.3f} m",
        )

    # 텍스트 주석: 수치 정보
    y_max = peak_info.counts.max() if len(peak_info.counts) > 0 else 1
    annotation_text = (
        f"Peak Z:  {peak_info.peak_z:.4f} m\n"
        f"FWHM:    {peak_info.fwhm:.4f} m\n"
        f"Z min:   {peak_info.z_min:.4f} m\n"
        f"Z max:   {peak_info.z_max:.4f} m"
    )
    ax.text(
        0.02,
        0.97,
        annotation_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    # 축 레이블 및 제목
    ax.set_xlabel("Z (m)", fontsize=12)
    ax.set_ylabel("Point Count", fontsize=12)
    ax.set_title("Z-Axis Histogram with Floor Peak Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
