"""Histogram-based analysis charts — Z distribution, intensity, color distance."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
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


def create_intensity_chart(
    intensity: np.ndarray | None,
    floor_mask: np.ndarray,
    intensity_percentile: float,
    save_path: Path,
    dpi: int = 150,
    figsize: tuple = (10, 6),
) -> None:
    """바닥 영역의 intensity 분포를 히스토그램으로 시각화한다.

    intensity가 None이거나 바닥 포인트가 없으면 빈 차트(안내 메시지)를 저장한다.

    Args:
        intensity: (N,) intensity 배열, None이면 데이터 없음
        floor_mask: (N,) 바닥 포인트 불리언 마스크
        intensity_percentile: 임계값으로 사용된 백분위수 (예: 25.0)
        save_path: 저장할 이미지 파일 경로
        dpi: 출력 이미지 DPI
        figsize: 그림 크기 (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # intensity 데이터 없음 또는 바닥 포인트 없음
    if intensity is None or not floor_mask.any():
        ax.text(
            0.5,
            0.5,
            "Intensity 데이터 없음",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="gray",
        )
        ax.set_title("Floor Region Intensity Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Intensity", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return

    # 바닥 영역 intensity 추출
    floor_intensity = intensity[floor_mask]

    # 임계값 계산
    threshold = np.percentile(floor_intensity, intensity_percentile)

    # 통과/제거 마스크
    pass_mask = floor_intensity >= threshold
    fail_mask = ~pass_mask
    pass_count = int(pass_mask.sum())
    fail_count = int(fail_mask.sum())
    total = len(floor_intensity)
    pass_ratio = pass_count / total * 100 if total > 0 else 0.0
    fail_ratio = fail_count / total * 100 if total > 0 else 0.0

    # 히스토그램 공통 빈 설정
    bins = min(100, max(20, total // 50))
    hist_range = (floor_intensity.min(), floor_intensity.max())

    # 제거 영역 (임계값 미만) — 회색
    ax.hist(
        floor_intensity[fail_mask],
        bins=bins,
        range=hist_range,
        color="gray",
        alpha=0.6,
        label=f"Removed (< threshold): {fail_count:,} ({fail_ratio:.1f}%)",
    )

    # 통과 영역 (임계값 이상) — 녹색
    ax.hist(
        floor_intensity[pass_mask],
        bins=bins,
        range=hist_range,
        color="green",
        alpha=0.6,
        label=f"Passed (>= threshold): {pass_count:,} ({pass_ratio:.1f}%)",
    )

    # 빨간 수직선: percentile 임계값
    ax.axvline(
        threshold,
        color="red",
        linestyle="-",
        linewidth=2.0,
        label=f"Threshold (P{intensity_percentile:.0f}) = {threshold:.4f}",
    )

    # 텍스트: 수치 정보
    annotation_text = (
        f"Threshold (P{intensity_percentile:.0f}): {threshold:.4f}\n"
        f"Passed:  {pass_count:,} ({pass_ratio:.1f}%)\n"
        f"Removed: {fail_count:,} ({fail_ratio:.1f}%)"
    )
    ax.text(
        0.97,
        0.97,
        annotation_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    ax.set_xlabel("Intensity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Floor Region Intensity Distribution", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def create_color_distance_chart(
    colors: np.ndarray | None,
    floor_mask: np.ndarray,
    color_tolerance: float,
    color_std_threshold: float,
    save_path: Path,
    dpi: int = 150,
) -> None:
    """바닥 포인트의 평균색으로부터 유클리드 거리 분포를 히스토그램으로 시각화한다.

    Args:
        colors: (N, 3) RGB 색상 배열 (0~1). None이면 빈 차트 출력.
        floor_mask: (N,) 바닥 마스크 불리언 배열.
        color_tolerance: 색상 필터 허용 거리 임계값.
        color_std_threshold: 색상 필터 적용 최소 표준편차.
        save_path: 저장할 이미지 경로.
        dpi: 출력 이미지 DPI.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    # 색상 없거나 바닥 영역 분산이 너무 작은 경우 빈 차트
    use_color_filter = (
        colors is not None
        and floor_mask.any()
        and colors[floor_mask].std(axis=0).mean() > color_std_threshold
    )

    if not use_color_filter:
        ax.text(
            0.5,
            0.5,
            "색상 필터 미적용\n(colors=None 또는 바닥 영역 std < 0.01)",
            ha="center",
            va="center",
            fontsize=14,
            color="#cdd6f4",
            transform=ax.transAxes,
        )
        ax.set_title("Floor Color Distance Distribution", fontsize=14, color="#cdd6f4", pad=12)
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        ax.tick_params(colors="#cdd6f4")
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    # 바닥 평균색 계산
    floor_colors = colors[floor_mask]
    floor_mean = floor_colors.mean(axis=0)

    # 유클리드 거리 계산
    dist = np.linalg.norm(floor_colors - floor_mean, axis=1)

    # 히스토그램 구간 설정
    bins = np.linspace(0, dist.max() * 1.05, 60)
    counts, edges = np.histogram(dist, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # 녹색(통과) / 회색(제거) 음영 분리
    pass_mask_bins = centers <= color_tolerance
    ax.bar(
        centers[pass_mask_bins],
        counts[pass_mask_bins],
        width=edges[1] - edges[0],
        color="#a6e3a1",
        alpha=0.75,
        label="통과 (Pass)",
    )
    ax.bar(
        centers[~pass_mask_bins],
        counts[~pass_mask_bins],
        width=edges[1] - edges[0],
        color="#6c7086",
        alpha=0.75,
        label="제거 (Remove)",
    )

    # 음영 배경 영역
    ax.axvspan(0, color_tolerance, alpha=0.08, color="#a6e3a1")
    ax.axvspan(color_tolerance, dist.max() * 1.05, alpha=0.08, color="#6c7086")

    # 기준선 (빨간 수직선)
    ax.axvline(
        color_tolerance,
        color="#f38ba8",
        linewidth=2.0,
        linestyle="--",
        label=f"Tolerance = {color_tolerance:.2f}",
        zorder=5,
    )

    # 좌측 상단 바닥 평균색 스와치
    swatch = mpatches.Rectangle(
        (0.02, 0.80),
        0.06,
        0.12,
        transform=ax.transAxes,
        facecolor=tuple(float(c) for c in floor_mean),
        edgecolor="#cdd6f4",
        linewidth=1.2,
        zorder=10,
    )
    ax.add_patch(swatch)

    # 통과/제거 비율 계산
    n_pass = int((dist <= color_tolerance).sum())
    n_remove = len(dist) - n_pass
    pass_ratio = n_pass / len(dist) * 100 if len(dist) > 0 else 0.0
    remove_ratio = 100.0 - pass_ratio

    # 텍스트 정보
    rgb_text = f"RGB: ({floor_mean[0]:.3f}, {floor_mean[1]:.3f}, {floor_mean[2]:.3f})"
    info_text = (
        f"{rgb_text}\n"
        f"Tolerance: {color_tolerance:.2f}\n"
        f"통과: {n_pass:,} ({pass_ratio:.1f}%)  제거: {n_remove:,} ({remove_ratio:.1f}%)"
    )
    ax.text(
        0.10,
        0.85,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        color="#cdd6f4",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#313244", edgecolor="#45475a", alpha=0.85),
    )

    # 축 / 제목 스타일
    ax.set_title("Floor Color Distance Distribution", fontsize=14, color="#cdd6f4", pad=12)
    ax.set_xlabel("Euclidean Distance from Floor Mean Color", fontsize=11, color="#cdd6f4")
    ax.set_ylabel("Point Count", fontsize=11, color="#cdd6f4")
    ax.tick_params(colors="#cdd6f4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#45475a")
    ax.legend(fontsize=10, facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
