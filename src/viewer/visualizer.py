"""PyVista 기반 포인트 클라우드 3D 시각화 — GPU 가속 렌더링."""

from pathlib import Path

import numpy as np
import pyvista as pv

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

        if mode == 1 or floor_mask is None:
            cloud = pv.PolyData(points)
            if colors is not None:
                rgba = np.empty((len(colors), 4), dtype=np.uint8)
                rgba[:, :3] = (colors * 255).astype(np.uint8)
                rgba[:, 3] = 255
                cloud["RGBA"] = rgba
                plotter.add_mesh(cloud, scalars="RGBA", rgba=True,
                                 point_size=point_size, render_points_as_spheres=False)
            else:
                plotter.add_mesh(cloud, scalars=points[:, 2], cmap="viridis",
                                 point_size=point_size, render_points_as_spheres=False)

        elif mode == 2:
            floor_pts = points[floor_mask]
            cloud = pv.PolyData(floor_pts)
            if colors is not None:
                floor_clr = colors[floor_mask]
                rgba = np.empty((len(floor_clr), 4), dtype=np.uint8)
                rgba[:, :3] = (floor_clr * 255).astype(np.uint8)
                rgba[:, 3] = 255
                cloud["RGBA"] = rgba
                plotter.add_mesh(cloud, scalars="RGBA", rgba=True,
                                 point_size=point_size, render_points_as_spheres=False)
            else:
                plotter.add_mesh(cloud, scalars=floor_pts[:, 2], cmap="viridis",
                                 point_size=point_size, render_points_as_spheres=False)

        elif mode == 3:
            inv_mask = ~floor_mask
            nf_pts = points[inv_mask]
            cloud = pv.PolyData(nf_pts)
            if colors is not None:
                nf_clr = colors[inv_mask]
                rgba = np.empty((len(nf_clr), 4), dtype=np.uint8)
                rgba[:, :3] = (nf_clr * 255).astype(np.uint8)
                rgba[:, 3] = 255
                cloud["RGBA"] = rgba
                plotter.add_mesh(cloud, scalars="RGBA", rgba=True,
                                 point_size=point_size, render_points_as_spheres=False)
            else:
                plotter.add_mesh(cloud, scalars=nf_pts[:, 2], cmap="viridis",
                                 point_size=point_size, render_points_as_spheres=False)

        elif mode == 4:
            cloud = pv.PolyData(points)
            highlight = np.array(floor_highlight_color, dtype=np.float32)
            if colors is not None:
                clr = colors.copy()
                clr[floor_mask] = highlight
            else:
                clr = np.full((len(points), 3), 0.7, dtype=np.float32)
                clr[floor_mask] = highlight
            rgba = np.empty((len(clr), 4), dtype=np.uint8)
            rgba[:, :3] = (clr * 255).astype(np.uint8)
            rgba[:, 3] = 255
            cloud["RGBA"] = rgba
            plotter.add_mesh(cloud, scalars="RGBA", rgba=True,
                             point_size=point_size, render_points_as_spheres=False)

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
