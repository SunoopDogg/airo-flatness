"""Chart 4: 색상 거리 히스토그램."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def create_color_distance_chart(
    colors: np.ndarray | None,
    floor_mask: np.ndarray,
    color_tolerance: float,
    save_path: Path,
    dpi: int = 150,
) -> None:
    """바닥 포인트의 평균색으로부터 유클리드 거리 분포를 히스토그램으로 시각화한다.

    Args:
        colors: (N, 3) RGB 색상 배열 (0~1). None이면 빈 차트 출력.
        floor_mask: (N,) 바닥 마스크 불리언 배열.
        color_tolerance: 색상 필터 허용 거리 임계값.
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
        and colors[floor_mask].std(axis=0).mean() > 0.01
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
