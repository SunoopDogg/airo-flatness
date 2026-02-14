"""Chart 6: 파라미터 민감도 차트 — extract_floor 반복 실행 결과 시각화."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from extractor.floor_extractor import extract_floor
from config import Config


def create_parameter_sensitivity_chart(
    points: np.ndarray,
    colors: np.ndarray | None,
    intensity: np.ndarray | None,
    width_multiplier: float,
    color_tolerance: float,
    intensity_percentile: float,
    save_path: Path,
    width_multiplier_sweep: list[float] | None = None,
    color_tolerance_sweep: list[float] | None = None,
    intensity_percentile_sweep: list[float] | None = None,
    max_subsample: int = 500_000,
    dpi: int = 150,
) -> dict:
    """3개 파라미터 민감도를 라인 차트로 시각화하고 PNG를 저장한다.

    포인트가 많을 경우 최대 max_subsample로 서브샘플링하여 실행 시간을 줄인다.

    Args:
        points: (N, 3) 포인트 좌표
        colors: (N, 3) RGB 색상, None 가능
        intensity: (N,) intensity, None 가능
        width_multiplier: 현재 사용된 width_multiplier 값 (빨간 마커 표시)
        color_tolerance: 현재 사용된 color_tolerance 값
        intensity_percentile: 현재 사용된 intensity_percentile 값
        save_path: 저장할 이미지 경로
        width_multiplier_sweep: 스윕할 width_multiplier 값 목록 (None이면 Config 기본값)
        color_tolerance_sweep: 스윕할 color_tolerance 값 목록 (None이면 Config 기본값)
        intensity_percentile_sweep: 스윕할 intensity_percentile 값 목록 (None이면 Config 기본값)
        max_subsample: 최대 서브샘플 포인트 수

    Returns:
        민감도 데이터 dict (report.json 용)
    """
    cfg = Config()
    if width_multiplier_sweep is None:
        width_multiplier_sweep = cfg.width_multiplier_sweep
    if color_tolerance_sweep is None:
        color_tolerance_sweep = cfg.color_tolerance_sweep
    if intensity_percentile_sweep is None:
        intensity_percentile_sweep = cfg.intensity_percentile_sweep

    # 서브샘플링
    n = len(points)
    if n > max_subsample:
        idx = np.random.choice(n, max_subsample, replace=False)
        pts = points[idx]
        col = colors[idx] if colors is not None else None
        itn = intensity[idx] if intensity is not None else None
    else:
        pts = points
        col = colors
        itn = intensity

    def sweep(param_name: str, values: list, base_kwargs: dict) -> list[float]:
        ratios = []
        for v in values:
            kwargs = dict(base_kwargs)
            kwargs[param_name] = v
            try:
                result = extract_floor(pts, colors=col, intensity=itn, **kwargs)
                ratios.append(result.floor_ratio)
            except Exception:
                ratios.append(float("nan"))
        return ratios

    base = dict(
        width_multiplier=width_multiplier,
        color_tolerance=color_tolerance,
        intensity_percentile=intensity_percentile,
    )

    wm_ratios = sweep("width_multiplier", width_multiplier_sweep, base)
    ct_ratios = sweep("color_tolerance", color_tolerance_sweep, base)
    ip_ratios = sweep("intensity_percentile", intensity_percentile_sweep, base)

    # 차트 그리기
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)
    fig.suptitle("Parameter Sensitivity Analysis", fontsize=15, fontweight="bold")

    _plot_sensitivity_line(
        axes[0],
        width_multiplier_sweep,
        wm_ratios,
        width_multiplier,
        "Width Multiplier",
        "floor_ratio",
    )
    _plot_sensitivity_line(
        axes[1],
        color_tolerance_sweep,
        ct_ratios,
        color_tolerance,
        "Color Tolerance",
        "floor_ratio",
    )
    _plot_sensitivity_line(
        axes[2],
        intensity_percentile_sweep,
        ip_ratios,
        intensity_percentile,
        "Intensity Percentile",
        "floor_ratio",
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "width_multiplier": {
            "values": width_multiplier_sweep,
            "floor_ratios": wm_ratios,
        },
        "color_tolerance": {
            "values": color_tolerance_sweep,
            "floor_ratios": ct_ratios,
        },
        "intensity_percentile": {
            "values": intensity_percentile_sweep,
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
