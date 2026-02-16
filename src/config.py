"""Central configuration — all hyperparameters in one place."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Loader
    max_points: int = 5_000_000
    random_seed: int = 42
    chunk_size: int = 1_000_000

    # Floor extraction
    num_bins: int = 200
    width_multiplier: float = 2.5
    intensity_percentile: float = 25.0
    color_tolerance: float = 0.6
    color_std_threshold: float = 0.01

    # Flatness analysis
    flatness_target_grid: int = 150
    flatness_min_points: int = 3

    # Peak detector
    prominence_ratio: float = 0.1
    min_peak_width: int = 2
    fallback_z_ratio: float = 0.2
    tilt_fwhm_threshold: float = 2.0
    tilt_width_multiplier: float = 1.0

    # Viewer
    point_size: float = 1.0
    floor_highlight_color: tuple[float, float, float] = (1.0, 0.2, 0.2)
    non_floor_fallback_gray: float = 0.7

    # Sensitivity sweep
    sensitivity_max_subsample: int = 500_000
    width_multiplier_sweep: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    )
    color_tolerance_sweep: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    )
    intensity_percentile_sweep: list[float] = field(
        default_factory=lambda: [10, 20, 25, 30, 40, 50, 75]
    )

    # Chart
    chart_dpi: int = 150

    # Paths
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


# Alias matching task spec
load_config = get_config
