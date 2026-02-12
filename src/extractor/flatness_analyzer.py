from dataclasses import dataclass
import numpy as np


@dataclass
class FlatnessResult:
    tilt_grid: np.ndarray      # 2D array of tilt angles in degrees, NaN for insufficient data
    x_edges: np.ndarray        # grid X bin edges
    y_edges: np.ndarray        # grid Y bin edges
    mean_tilt: float
    max_tilt: float
    valid_cell_count: int
    total_cell_count: int
    cell_size: float           # meters


def analyze_flatness(
    floor_points: np.ndarray,
    target_grid_size: int = 150,
    min_points_per_cell: int = 3,
) -> FlatnessResult:
    """Analyze flatness of floor points by computing per-cell surface tilt angles.

    Args:
        floor_points: (N, 3) numpy array of XYZ points already filtered by floor mask.
        target_grid_size: number of grid cells along the longest axis.
        min_points_per_cell: minimum points required to compute tilt for a cell.

    Returns:
        FlatnessResult with tilt grid and summary statistics.
    """
    if floor_points is None or len(floor_points) == 0:
        empty_grid = np.full((1, 1), np.nan)
        return FlatnessResult(
            tilt_grid=empty_grid,
            x_edges=np.array([0.0, 1.0]),
            y_edges=np.array([0.0, 1.0]),
            mean_tilt=float("nan"),
            max_tilt=float("nan"),
            valid_cell_count=0,
            total_cell_count=1,
            cell_size=1.0,
        )

    xs = floor_points[:, 0]
    ys = floor_points[:, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    longest = max(x_range, y_range)

    if longest == 0:
        cell_size = 1.0
    else:
        cell_size = longest / target_grid_size

    # Build bin edges
    x_edges = np.arange(x_min, x_max + cell_size, cell_size)
    y_edges = np.arange(y_min, y_max + cell_size, cell_size)

    # Ensure at least 2 edges (1 cell)
    if len(x_edges) < 2:
        x_edges = np.array([x_min, x_min + cell_size])
    if len(y_edges) < 2:
        y_edges = np.array([y_min, y_min + cell_size])

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    # Bin each point (digitize returns 1-based indices)
    xi = np.digitize(xs, x_edges) - 1
    yi = np.digitize(ys, y_edges) - 1

    # Clip to valid range (handles points exactly on max edge)
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    tilt_grid = np.full((nx, ny), np.nan)

    for ix in range(nx):
        for iy in range(ny):
            mask = (xi == ix) & (yi == iy)
            cell_pts = floor_points[mask]
            if len(cell_pts) < min_points_per_cell:
                continue

            # Center the points
            centered = cell_pts - cell_pts.mean(axis=0)

            # Covariance matrix
            cov = centered.T @ centered  # 3x3

            # SVD: smallest singular value's vector = surface normal
            _, _, Vt = np.linalg.svd(cov)
            normal = Vt[-1]  # last row = smallest singular vector

            # Tilt angle from vertical (Z axis)
            normal_z = abs(normal[2])
            # Clamp for numerical safety before arccos
            normal_z = min(1.0, normal_z)
            tilt_angle = np.degrees(np.arccos(normal_z))
            tilt_grid[ix, iy] = tilt_angle

    valid_cell_count = int(np.sum(~np.isnan(tilt_grid)))
    total_cell_count = nx * ny

    mean_tilt = float(np.nanmean(tilt_grid)) if valid_cell_count > 0 else float("nan")
    max_tilt = float(np.nanmax(tilt_grid)) if valid_cell_count > 0 else float("nan")

    return FlatnessResult(
        tilt_grid=tilt_grid,
        x_edges=x_edges,
        y_edges=y_edges,
        mean_tilt=mean_tilt,
        max_tilt=max_tilt,
        valid_cell_count=valid_cell_count,
        total_cell_count=total_cell_count,
        cell_size=cell_size,
    )
