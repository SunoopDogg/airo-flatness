"""Chart 6: 파라미터 민감도 차트 — extract_floor 반복 실행 결과 시각화."""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from extractor.floor_extractor import extract_floor
from config import Config
from utils import subsample_points


def create_parameter_sensitivity_chart(
    points: np.ndarray,
    colors: np.ndarray | None,
    intensity: np.ndarray | None,
    config: Config,
    save_path: Path,
    dpi: int = 150,
) -> dict:
    """3개 파라미터 민감도를 라인 차트로 시각화하고 PNG를 저장한다.

    포인트가 많을 경우 최대 config.sensitivity_max_subsample로 서브샘플링하여 실행 시간을 줄인다.

    Args:
        points: (N, 3) 포인트 좌표
        colors: (N, 3) RGB 색상, None 가능
        intensity: (N,) intensity, None 가능
        config: Config 인스턴스
        save_path: 저장할 이미지 경로
        dpi: 출력 이미지 DPI

    Returns:
        민감도 데이터 dict (report.json 용)
    """
    # 서브샘플링
    n = len(points)
    if n > config.sensitivity_max_subsample:
        if colors is not None and intensity is not None:
            sub_points, sub_colors, sub_intensity = subsample_points(
                points, config.sensitivity_max_subsample, config.random_seed, colors, intensity)
        elif colors is not None:
            sub_points, sub_colors = subsample_points(
                points, config.sensitivity_max_subsample, config.random_seed, colors)
            sub_intensity = None
        elif intensity is not None:
            sub_points, sub_intensity = subsample_points(
                points, config.sensitivity_max_subsample, config.random_seed, intensity)
            sub_colors = None
        else:
            sub_points, = subsample_points(points, config.sensitivity_max_subsample, config.random_seed)
            sub_colors = None
            sub_intensity = None
    else:
        sub_points = points
        sub_colors = colors
        sub_intensity = intensity

    def sweep(param_name: str, values: list) -> list[float]:
        ratios = []
        for v in values:
            swept_config = replace(config, **{param_name: v})
            try:
                result = extract_floor(sub_points, colors=sub_colors, intensity=sub_intensity, config=swept_config)
                ratios.append(result.floor_ratio)
            except Exception:
                ratios.append(float("nan"))
        return ratios

    wm_ratios = sweep("width_multiplier", config.width_multiplier_sweep)
    ct_ratios = sweep("color_tolerance", config.color_tolerance_sweep)
    ip_ratios = sweep("intensity_percentile", config.intensity_percentile_sweep)

    # 차트 그리기
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)
    fig.suptitle("Parameter Sensitivity Analysis", fontsize=15, fontweight="bold")

    _plot_sensitivity_line(
        axes[0],
        config.width_multiplier_sweep,
        wm_ratios,
        config.width_multiplier,
        "Width Multiplier",
        "floor_ratio",
    )
    _plot_sensitivity_line(
        axes[1],
        config.color_tolerance_sweep,
        ct_ratios,
        config.color_tolerance,
        "Color Tolerance",
        "floor_ratio",
    )
    _plot_sensitivity_line(
        axes[2],
        config.intensity_percentile_sweep,
        ip_ratios,
        config.intensity_percentile,
        "Intensity Percentile",
        "floor_ratio",
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "width_multiplier": {
            "values": config.width_multiplier_sweep,
            "floor_ratios": wm_ratios,
        },
        "color_tolerance": {
            "values": config.color_tolerance_sweep,
            "floor_ratios": ct_ratios,
        },
        "intensity_percentile": {
            "values": config.intensity_percentile_sweep,
            "floor_ratios": ip_ratios,
        },
    }


def _plot_sensitivity_line(
    ax: plt.Axes,
    x_values: list,
    y_values: list[float],
    current_value: float,
    x_label: str,
    y_label: str,
) -> None:
    """단일 파라미터 민감도 라인 차트를 ax에 그린다."""
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)

    ax.plot(x, y, color="steelblue", linewidth=2.0, marker="o", markersize=5)

    # 현재 사용된 값에 빨간 마커
    try:
        cur_idx = x_values.index(current_value)
        ax.scatter(
            [current_value],
            [y_values[cur_idx]],
            color="red",
            s=80,
            zorder=5,
            label=f"Current: {current_value}",
        )
        ax.legend(fontsize=8)
    except ValueError:
        # 현재 값이 스윕 목록에 없으면 수직선으로 표시
        ax.axvline(current_value, color="red", linestyle="--", linewidth=1.5,
                   label=f"Current: {current_value}")
        ax.legend(fontsize=8)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(f"{x_label} Sensitivity", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
