"""Summary charts — filtering funnel and floor ratio."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extractor.floor_extractor import FloorResult, StageFilterCounts


def create_filtering_funnel_chart(
    stage_counts: StageFilterCounts,
    save_path: Path,
    dpi: int = 150,
    figsize: tuple = (10, 5),
) -> None:
    """바닥 추출 파이프라인의 단계별 포인트 감소를 수평 막대 차트로 시각화한다.

    Args:
        stage_counts: 각 필터링 단계 후 포인트 수
        save_path: 저장할 이미지 파일 경로
        dpi: 출력 이미지 DPI
        figsize: 그림 크기 (width, height)
    """
    # 3단계 데이터 정의
    stages = ["Total", "After Z-Filter", "After Refinement"]
    counts = [
        stage_counts.total,
        stage_counts.after_z_filter,
        stage_counts.after_refinement,
    ]

    # 감소율 계산 (총 포인트 대비 %)
    total = stage_counts.total if stage_counts.total > 0 else 1
    ratios = [cnt / total * 100 for cnt in counts]

    # 블루 그라데이션 색상
    colors = ["#1565C0", "#1E88E5", "#64B5F6"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 수평 막대 그래프
    y_pos = np.arange(len(stages))
    bars = ax.barh(y_pos, counts, color=colors, alpha=0.85, height=0.5)

    # 각 막대에 포인트 수 + 감소율 텍스트 표시
    for i, (bar, count, ratio) in enumerate(zip(bars, counts, ratios)):
        # 막대 오른쪽 끝에 텍스트
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({ratio:.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    # Y축 레이블
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontsize=11)

    # X축 설정
    ax.set_xlabel("Point Count", fontsize=12)
    ax.set_xlim(0, total * 1.2)

    # 제목
    ax.set_title(
        "Floor Extraction Pipeline - Filtering Funnel",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # 위에서 아래로 Total → Refinement 순서

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


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
