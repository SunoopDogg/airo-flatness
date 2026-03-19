"""Interactive ROI selection via matplotlib top-view scatter plot."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.spatial import ConvexHull, QhullError


def _minimum_bounding_rectangle(hull_points: np.ndarray) -> np.ndarray:
    """ConvexHull 점들의 최소 면적 외접 직사각형 꼭짓점 반환.

    Args:
        hull_points: (N, 2) ConvexHull 꼭짓점 (CCW 순서).

    Returns:
        (4, 2) 직사각형 꼭짓점 (CCW 순서).
    """
    best_area = np.inf
    best_rect = None

    n = len(hull_points)
    for i in range(n):
        # edge 방향 각도
        edge = hull_points[(i + 1) % n] - hull_points[i]
        angle = np.arctan2(edge[1], edge[0])
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # R(angle) — edge 방향 회전 행렬
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # -angle 회전으로 edge를 x축에 정렬: points @ rot.T = points @ R(-angle)
        rotated = hull_points @ rot.T

        # 축 정렬 바운딩 박스
        x_min, y_min = rotated.min(axis=0)
        x_max, y_max = rotated.max(axis=0)
        area = (x_max - x_min) * (y_max - y_min)

        if area < best_area:
            best_area = area
            # 바운딩 박스 꼭짓점 (회전된 좌표계)
            box_rotated = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ])
            # +angle 역회전으로 원래 좌표계 복원: box @ rot = box @ R(angle)
            best_rect = box_rotated @ rot

    # CCW 순서 보장 (signed area > 0)
    v = best_rect
    signed_area = 0.0
    for i in range(4):
        j = (i + 1) % 4
        signed_area += v[i, 0] * v[j, 1] - v[j, 0] * v[i, 1]
    if signed_area < 0:
        best_rect = best_rect[::-1]

    return best_rect


class QuadROI:
    """4개 꼭짓점으로 정의된 직사각형 ROI (최소 면적 외접 직사각형, 반시계방향 정렬)."""

    def __init__(self, points: np.ndarray) -> None:
        """ConvexHull + MABR로 4점의 최소 면적 외접 직사각형 생성.

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
        hull_pts = points[hull.vertices]
        self.vertices: np.ndarray = _minimum_bounding_rectangle(hull_pts)

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

    실시간으로 클릭한 점과 연결선을 표시하고, 4점 완료 시
    ConvexHull 정렬된 사각형을 미리보기로 표시한다.

    조작:
        좌클릭: 점 추가 (최대 4개)
        우클릭: 마지막 점 취소 (undo)
        R: 전체 초기화
        Q: ROI 확정 (4점 완료 상태에서만)

    Args:
        points: (N, 3) 포인트 클라우드 배열.
        max_display: 표시할 최대 점 수 (서브샘플링).
        seed: 서브샘플링 시드.

    Returns:
        선택된 4점으로 구성된 QuadROI.

    Raises:
        ValueError: 4점을 선택하지 않고 창을 닫은 경우.
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

    plt.tight_layout()

    # --- 내부 상태 ---
    state = {
        "clicked": [],           # list[tuple[float, float]]
        "confirmed_roi": None,   # QuadROI | None
        "markers": [],           # list[Line2D]
        "lines": [],             # list[Line2D]
        "polygon": None,         # Polygon | None
    }

    def _update_title():
        n = len(state["clicked"])
        if state["confirmed_roi"] is not None:
            ax.set_title("ROI confirmed — Q or close to apply, R to reset")
        else:
            ax.set_title(f"Click point {n + 1}/4 (R: reset, right-click: undo)")

    def _clear_artists():
        for m in state["markers"]:
            m.remove()
        state["markers"].clear()
        for ln in state["lines"]:
            ln.remove()
        state["lines"].clear()
        if state["polygon"] is not None:
            state["polygon"].remove()
            state["polygon"] = None

    def _draw_progress():
        """클릭된 점과 연결선을 그린다."""
        pts = state["clicked"]
        for x, y in pts:
            marker, = ax.plot(x, y, "ro", markersize=8)
            state["markers"].append(marker)
        for i in range(1, len(pts)):
            line, = ax.plot(
                [pts[i - 1][0], pts[i][0]],
                [pts[i - 1][1], pts[i][1]],
                "r--", linewidth=1.5,
            )
            state["lines"].append(line)

    def _show_polygon(roi: QuadROI):
        """ConvexHull 정렬된 사각형을 초록색으로 표시."""
        from matplotlib.patches import Polygon
        v = roi.vertices
        poly = Polygon(
            v, closed=True,
            edgecolor="green", facecolor="none",
            linewidth=2.5,
        )
        ax.add_patch(poly)
        state["polygon"] = poly

    def _try_complete():
        """4점으로 QuadROI 생성 시도. 성공 시 사각형 표시, 실패 시 리셋."""
        try:
            roi = QuadROI(np.array(state["clicked"]))
        except ValueError:
            _clear_artists()
            ax.set_title("Invalid shape — resetting. Click point 1/4")
            state["clicked"].clear()
            fig.canvas.draw_idle()
            return
        _clear_artists()
        _show_polygon(roi)
        state["confirmed_roi"] = roi
        _update_title()
        fig.canvas.draw_idle()

    def _reset():
        _clear_artists()
        state["clicked"].clear()
        state["confirmed_roi"] = None
        _update_title()
        fig.canvas.draw_idle()

    def on_click(event):
        # 안전 체크: axes 외부, toolbar 모드 활성 시 무시
        if event.inaxes is not ax:
            return
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode != "":
            return

        if event.button == 3:  # 우클릭 = undo
            if state["confirmed_roi"] is not None:
                # 사각형 상태 → 3점으로 복귀
                state["confirmed_roi"] = None
                _clear_artists()
                state["clicked"].pop()
                _draw_progress()
                _update_title()
                fig.canvas.draw_idle()
            elif state["clicked"]:
                state["clicked"].pop()
                _clear_artists()
                _draw_progress()
                _update_title()
                fig.canvas.draw_idle()
            return

        if event.button != 1:  # 좌클릭만 처리
            return

        # 4점 완료 상태에서 추가 클릭 무시
        if len(state["clicked"]) >= 4:
            return

        state["clicked"].append((event.xdata, event.ydata))

        if len(state["clicked"]) < 4:
            # 점/선 추가
            marker, = ax.plot(event.xdata, event.ydata, "ro", markersize=8)
            state["markers"].append(marker)
            if len(state["clicked"]) > 1:
                prev = state["clicked"][-2]
                line, = ax.plot(
                    [prev[0], event.xdata],
                    [prev[1], event.ydata],
                    "r--", linewidth=1.5,
                )
                state["lines"].append(line)
            _update_title()
            fig.canvas.draw_idle()
        else:
            # 4점 완료 → 사각형 생성 시도
            _try_complete()

    def on_key(event):
        if event.key == "q":
            if state["confirmed_roi"] is not None:
                plt.close(fig)
        elif event.key == "r":
            _reset()

    def on_close(_event):
        # Q에서 이미 confirmed_roi 설정된 경우 → 그대로 유지
        if state["confirmed_roi"] is not None:
            return
        # 4점 완료 상태에서 창 닫기 → 확정
        if len(state["clicked"]) == 4:
            try:
                state["confirmed_roi"] = QuadROI(np.array(state["clicked"]))
            except ValueError:
                pass

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    _update_title()
    plt.show()

    if state["confirmed_roi"] is None:
        raise ValueError("4개의 점을 선택해야 합니다.")

    return state["confirmed_roi"]


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

    span_selector = SpanSelector(
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
