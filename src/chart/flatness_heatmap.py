"""Floor Flatness Top-View Heatmap chart."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extractor.flatness_analyzer import FlatnessResult


def create_flatness_heatmap_chart(
    flatness_result: FlatnessResult,
    save_path: Path,
    dpi: int = 150,
) -> None:
    """바닥 평탄도 결과를 Top-View 히트맵으로 시각화한다.

    Args:
        flatness_result: analyze_flatness()가 반환한 평탄도 분석 결과
        save_path: 저장할 이미지 파일 경로
        dpi: 출력 이미지 DPI
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    # Colormap 설정: 녹색(평탄, 0°) -> 빨간색(기울어짐)
    cmap = plt.get_cmap("RdYlGn_r")
    cmap.set_bad(color="#E0E0E0")  # NaN 셀은 밝은 회색

    extent = [
        flatness_result.x_edges[0],
        flatness_result.x_edges[-1],
        flatness_result.y_edges[0],
        flatness_result.y_edges[-1],
    ]

    im = ax.imshow(
        flatness_result.tilt_grid.T,
        cmap=cmap,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    # 컬러바
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tilt Angle (°)", fontsize=11)

    # 주석 텍스트: 주요 수치 정보
    annotation_text = (
        f"Mean Tilt:   {flatness_result.mean_tilt:.2f}°\n"
        f"Max Tilt:    {flatness_result.max_tilt:.2f}°\n"
        f"Cell Size:   {flatness_result.cell_size:.2f} m\n"
        f"Valid Cells: {flatness_result.valid_cell_count} / {flatness_result.total_cell_count}"
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
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Floor Flatness — Top View Heatmap", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
