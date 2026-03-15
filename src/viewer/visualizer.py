"""PyVista 기반 포인트 클라우드 3D 시각화 — GPU 가속 렌더링."""

from pathlib import Path

import numpy as np
import pyvista as pv

from utils import to_rgba

VIEW_MODES = {
    1: "Full Point Cloud",
    2: "Floor Only",
    3: "Non-Floor Only",
    4: "Highlighted Floor",
}

CAPTURE_VIEWS = ("topview", "front", "back", "right", "left")

VIEW_MODE_PREFIX = {
    1: "mode1_full",
    2: "mode2_floor",
    3: "mode3_nonfloor",
    4: "mode4_highlighted",
}


def _add_point_cloud(
    plotter: pv.Plotter,
    pts: np.ndarray,
    clr: np.ndarray | None,
    point_size: float,
) -> None:
    """Add a point cloud to the plotter with RGBA or viridis coloring."""
    cloud = pv.PolyData(pts)
    if clr is not None:
        cloud["RGBA"] = to_rgba(clr)
        plotter.add_mesh(
            cloud, scalars="RGBA", rgba=True,
            point_size=point_size, render_points_as_spheres=True,
        )
    else:
        plotter.add_mesh(
            cloud, scalars=pts[:, 2], cmap="viridis",
            point_size=point_size, render_points_as_spheres=True,
        )


def _select_points_for_mode(
    mode: int,
    points: np.ndarray,
    colors: np.ndarray | None,
    floor_mask: np.ndarray | None,
    floor_highlight_color: tuple[float, float, float],
    non_floor_gray: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Select points and colors for the given view mode.

    Returns:
        (pts, clr) tuple where clr may be None for viridis fallback.
    """
    if mode == 1 or floor_mask is None:
        return points, colors

    if mode == 2:
        mask = floor_mask
        return points[mask], colors[mask] if colors is not None else None

    if mode == 3:
        mask = ~floor_mask
        return points[mask], colors[mask] if colors is not None else None

    # mode == 4: highlighted floor
    highlight = np.array(floor_highlight_color, dtype=np.float32)
    if colors is not None:
        clr = colors.copy()
        clr[floor_mask] = highlight
    else:
        clr = np.full((len(points), 3), non_floor_gray, dtype=np.float32)
        clr[floor_mask] = highlight
    return points, clr


def _set_camera_view(plotter: pv.Plotter, view_name: str) -> None:
    """카메라를 지정된 방향으로 설정한다."""
    if view_name == "topview":
        plotter.view_xy()
    elif view_name == "front":
        plotter.view_xz()
    elif view_name == "back":
        plotter.view_vector((0, 1, 0), viewup=(0, 0, 1))
    elif view_name == "right":
        plotter.view_yz()
    elif view_name == "left":
        plotter.view_vector((1, 0, 0), viewup=(0, 0, 1))


def visualize_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    floor_mask: np.ndarray | None = None,
    floor_highlight_color: tuple[float, float, float] = (1.0, 0.2, 0.2),
    non_floor_fallback_gray: float = 0.7,
    title: str = "Point Cloud Viewer",
    point_size: float = 1.0,
    results_dir: Path | None = None,
) -> None:
    """포인트 클라우드를 3D 시각화한다.

    Args:
        points: (N, 3) 포인트 좌표 배열
        colors: (N, 3) RGB 색상 배열 (0.0~1.0), None이면 높이 기반 컬러맵 적용
        floor_mask: (N,) 바닥 포인트 마스크
        floor_highlight_color: 바닥 하이라이트 색상 (R, G, B)
        title: 시각화 창 제목
        point_size: 포인트 렌더링 크기
        results_dir: 결과 저장 디렉토리
    """
    plotter = pv.Plotter(title=title)

    def build_view(mode: int) -> None:
        """Clear and rebuild the scene for the given view mode."""
        plotter.clear()
        pts, clr = _select_points_for_mode(
            mode, points, colors, floor_mask,
            floor_highlight_color, non_floor_fallback_gray,
        )
        _add_point_cloud(plotter, pts, clr, point_size)
        plotter.enable_eye_dome_lighting()
        plotter.add_axes()
        current_mode[0] = mode
        plotter.render()
        mode_name = VIEW_MODES.get(mode, "Unknown")
        print(f"\n  View mode: [{mode}] {mode_name}")

    current_mode = [4]

    plotter.add_key_event("1", lambda: build_view(1))
    plotter.add_key_event("2", lambda: build_view(2))
    plotter.add_key_event("3", lambda: build_view(3))
    plotter.add_key_event("4", lambda: build_view(4))

    def on_s_key():
        save_dir = results_dir if results_dir is not None else Path(".")
        prefix = VIEW_MODE_PREFIX.get(current_mode[0], f"mode{current_mode[0]}")
        print(f"\n  Capturing 5 views (prefix: {prefix})...")
        for view_name in CAPTURE_VIEWS:
            _set_camera_view(plotter, view_name)
            plotter.render()
            filename = f"{prefix}_{view_name}.png"
            save_path = save_dir / filename
            plotter.screenshot(str(save_path))
            print(f"    [{view_name}] saved: {save_path}")
        plotter.view_isometric()
        plotter.render()
        print("  All 5 captures complete.")

    plotter.add_key_event("s", on_s_key)

    build_view(4)
    plotter.show()
