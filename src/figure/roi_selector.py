"""Interactive ROI selection via matplotlib top-view scatter plot."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.spatial import ConvexHull, QhullError


class QuadROI:
    """4개 꼭짓점으로 정의된 사각형 ROI (반시계방향 정렬)."""

    def __init__(self, points: np.ndarray) -> None:
        """ConvexHull로 4점을 반시계방향 정렬.

        Args:
            points: (4, 2) 꼭짓점 좌표 배열.

        Raises:
            ValueError: 볼록 사각형을 이루지 않는 경우.
        """
        points = np.asarray(points, dtype=float)
        if points.shape != (4, 2):
            raise ValueError("정확히 4개의 2D 좌표가 필요합니다.")
        try:
            hull = ConvexHull(points)
        except QhullError as e:
            raise ValueError("4점이 볼록 사각형을 이루지 않습니다.") from e
        if len(hull.vertices) != 4:
            raise ValueError("4점이 볼록 사각형을 이루지 않습니다.")
        self.vertices: np.ndarray = points[hull.vertices]

    def to_axis_aligned(self) -> tuple[float, float, float, float]:
        """AABB 바운딩 박스 반환.

        Returns:
            (x_min, x_max, y_min, y_max) 튜플.
        """
        return (
            float(self.vertices[:, 0].min()),
            float(self.vertices[:, 0].max()),
            float(self.vertices[:, 1].min()),
            float(self.vertices[:, 1].max()),
        )

    def contains(self, points_xy: np.ndarray) -> np.ndarray:
        """Cross product 기반 내부 판별.

        경계 위의 점은 내부로 판정 (>= 0).

        Args:
            points_xy: (N, 2) 좌표 배열.

        Returns:
            (N,) bool mask.
        """
        points_xy = np.asarray(points_xy, dtype=float)
        if len(points_xy) == 0:
            return np.array([], dtype=bool)

        # AABB 사전 필터
        x_min, x_max, y_min, y_max = self.to_axis_aligned()
        mask = (
            (points_xy[:, 0] >= x_min) & (points_xy[:, 0] <= x_max) &
            (points_xy[:, 1] >= y_min) & (points_xy[:, 1] <= y_max)
        )

        # AABB 내부 점만 cross product 판별
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            return mask

        v = self.vertices
        cand_pts = points_xy[candidates]
        for i in range(4):
            j = (i + 1) % 4
            ex = v[j, 0] - v[i, 0]
            ey = v[j, 1] - v[i, 1]
            cross = ex * (cand_pts[:, 1] - v[i, 1]) - ey * (cand_pts[:, 0] - v[i, 0])
            mask[candidates] &= cross >= 0

        return mask


def select_roi(
    points: np.ndarray,
    max_display: int = 500_000,
    seed: int = 42,
) -> QuadROI:
    """탑뷰 산점도에서 4개 점을 클릭하여 사각형 ROI 선택.

    Args:
        points: (N, 3) 포인트 클라우드 배열.
        max_display: 표시할 최대 점 수 (서브샘플링).
        seed: 서브샘플링 시드.

    Returns:
        선택된 4점으로 구성된 QuadROI.

    Raises:
        ValueError: 4점을 선택하지 않거나 볼록 사각형이 아닌 경우.
    """
    if len(points) > max_display:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), max_display, replace=False)
        display_pts = points[idx]
    else:
        display_pts = points

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        display_pts[:, 0],
        display_pts[:, 1],
        c=display_pts[:, 2],
        cmap="viridis",
        s=0.1,
        rasterized=True,
    )
    fig.colorbar(scatter, ax=ax, label="Z (m)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.set_title("Click 4 points to define ROI (right-click to undo, close to confirm)")

    plt.tight_layout()
    clicked = plt.ginput(4, timeout=-1)
    plt.close(fig)

    if len(clicked) < 4:
        raise ValueError("4개의 점을 선택해야 합니다.")

    return QuadROI(np.array(clicked))


def filter_points_by_roi(
    points: np.ndarray,
    roi: QuadROI,
) -> np.ndarray:
    """QuadROI 내부 점만 필터링 (X-Y 평면).

    Args:
        points: (N, 3) 배열.
        roi: QuadROI 객체.

    Returns:
        (M, 3) ROI 내부 점 배열.
    """
    mask = roi.contains(points[:, :2])
    return points[mask]


def filter_points_by_z(
    points: np.ndarray,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Filter points within a Z value range.

    Args:
        points: (N, 3) array.
        z_min: minimum Z value (inclusive).
        z_max: maximum Z value (inclusive).

    Returns:
        (M, 3) array of points within Z range.
    """
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask]


def select_z_roi(
    points: np.ndarray,
    bins: int = 100,
) -> tuple[float, float]:
    """Show Z histogram and let user drag-select a Z value range.

    Args:
        points: (N, 3) point cloud array (already X-Y filtered).
        bins: number of histogram bins.

    Returns:
        (z_min, z_max) of selected Z range.

    Raises:
        ValueError: if user closes window without selecting.
    """
    z_values = points[:, 2]
    z_range = {}

    def on_select(z_lo, z_hi):
        z_range["z_min"] = min(z_lo, z_hi)
        z_range["z_max"] = max(z_lo, z_hi)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(z_values, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Z (m)")
    ax.set_ylabel("Point count")
    ax.set_title("Select Z Range (drag to select, close to confirm)")

    _span = SpanSelector(
        ax,
        on_select,
        direction="horizontal",
        useblit=True,
        props=dict(alpha=0.3, facecolor="red"),
        interactive=True,
    )

    plt.tight_layout()
    plt.show()

    if not z_range:
        raise ValueError("No Z range selected.")

    return z_range["z_min"], z_range["z_max"]
