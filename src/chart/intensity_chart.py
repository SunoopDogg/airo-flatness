"""Intensity 히스토그램 차트 생성 — 바닥 영역 intensity 분포 시각화."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
