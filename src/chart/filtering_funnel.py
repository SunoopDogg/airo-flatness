"""필터링 퍼널 차트 생성 — 3단계 포인트 감소 시각화."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from extractor.floor_extractor import StageFilterCounts


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
