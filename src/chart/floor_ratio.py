"""Chart 5: 바닥/비바닥 비율 도넛 차트."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from extractor.floor_extractor import FloorResult


def create_floor_ratio_chart(
    floor_result: FloorResult,
    filename: str,
    elapsed_time: float,
    save_path: Path,
    dpi: int = 150,
    figsize: tuple = (8, 8),
) -> None:
    """바닥/비바닥 포인트 비율을 도넛 차트로 시각화한다.

    Args:
        floor_result: 바닥 추출 결과 (FloorResult).
        filename: 처리한 원본 파일명.
        elapsed_time: 처리 소요 시간 (초).
        save_path: 저장할 이미지 경로.
        dpi: 출력 이미지 DPI.
        figsize: 그림 크기 (width, height).
    """
    floor_pts = floor_result.floor_points
    non_floor_pts = floor_result.total_points - floor_pts
    floor_ratio_pct = floor_result.floor_ratio * 100.0

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    sizes = [floor_pts, non_floor_pts]
    colors = ["#f38ba8", "#6c7086"]  # 빨간 계열 / 회색 계열
    labels = [f"Floor: {floor_pts:,} pts", f"Non-Floor: {non_floor_pts:,} pts"]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="#1e1e2e", linewidth=2),
        pctdistance=0.75,
    )

    for autotext in autotexts:
        autotext.set_color("#cdd6f4")
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

    # 중앙 텍스트: floor_ratio %
    ax.text(
        0,
        0,
        f"{floor_ratio_pct:.1f}%\nFloor",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#f38ba8",
    )

    # 범례
    legend_patches = [
        plt.matplotlib.patches.Patch(color=colors[0], label=labels[0]),
        plt.matplotlib.patches.Patch(color=colors[1], label=labels[1]),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=11,
        facecolor="#313244",
        edgecolor="#45475a",
        labelcolor="#cdd6f4",
    )

    # 하단 정보 텍스트
    info_text = (
        f"File: {filename}\n"
        f"Total Points: {floor_result.total_points:,}\n"
        f"Elapsed: {elapsed_time:.2f}s"
    )
    fig.text(
        0.5,
        0.02,
        info_text,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#a6adc8",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#313244", edgecolor="#45475a", alpha=0.85),
    )

    ax.set_title("Floor / Non-Floor Point Ratio", fontsize=14, color="#cdd6f4", pad=16)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
