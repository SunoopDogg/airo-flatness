"""Tests for ROI context view — interactive PyVista viewer."""

import json
import numpy as np
import pyvista as pv
import pytest
from unittest.mock import patch, MagicMock, ANY

from figure.roi_context import (
    _build_dashed_rectangle_2d,
    _build_dashed_box_3d,
    _create_plotter,
    _bind_screenshot_key,
    render_roi_context_2d,
    render_roi_context_3d,
)


class TestBuildDashedRectangle2d:
    def test_creates_polydata(self):
        """Should create a PolyData with dashed line segments."""
        rect = _build_dashed_rectangle_2d((1.0, 3.0, 2.0, 5.0), z=10.0)
        assert isinstance(rect, pv.PolyData)
        # 4 edges x 30 dashes x 2 endpoints = 240 points
        assert rect.n_points == 240
        # All points at z=10.0
        np.testing.assert_array_almost_equal(rect.points[:, 2], 10.0)

    def test_4_edges_of_dashes(self):
        """Should have 4 edges x 30 dashes = 120 line segments."""
        rect = _build_dashed_rectangle_2d((0.0, 1.0, 0.0, 1.0), z=0.0)
        assert rect.n_lines == 120


class TestBuildDashedBox3d:
    def test_creates_polydata(self):
        """Should create a PolyData with dashed line segments for 12 edges."""
        box = _build_dashed_box_3d((1.0, 3.0, 2.0, 5.0), z_min=0.0, z_max=10.0)
        assert isinstance(box, pv.PolyData)
        # 12 edges x 30 dashes x 2 endpoints = 720 points
        assert box.n_points == 720

    def test_12_edges_of_dashes(self):
        """Should have 12 edges x 30 dashes = 360 line segments."""
        box = _build_dashed_box_3d((0.0, 1.0, 0.0, 1.0), z_min=0.0, z_max=1.0)
        assert box.n_lines == 360


class TestCreatePlotter:
    def test_creates_interactive_plotter(self):
        """Should create an on-screen plotter with title (not off-screen)."""
        pts = np.random.default_rng(42).uniform(0, 10, (100, 3))
        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            _create_plotter(pts, title="Test Title")

            mock_pv.Plotter.assert_called_once_with(title="Test Title")
            mock_plotter.set_background.assert_called_once_with("white")

    def test_adds_point_cloud_with_z_scalars(self):
        """Should add point cloud with viridis Z colormap when no colors."""
        pts = np.random.default_rng(42).uniform(0, 10, (100, 3))
        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            _create_plotter(pts)

            _, kwargs = mock_plotter.add_mesh.call_args
            assert kwargs.get("cmap") == "viridis"


class TestBindScreenshotKey:
    def test_s_key_saves_png_and_json(self, tmp_path):
        """Pressing S should save PNG screenshot and camera JSON."""
        mock_plotter = MagicMock()
        mock_plotter.camera.position = (1.0, 2.0, 3.0)
        mock_plotter.camera.focal_point = (0.0, 0.0, 0.0)
        mock_plotter.camera.up = (0.0, 0.0, 1.0)

        _bind_screenshot_key(mock_plotter, tmp_path, "roi_context_2d_rgb.png")

        # Extract the handler function that was registered
        mock_plotter.add_key_event.assert_called_once_with("s", ANY)
        handler = mock_plotter.add_key_event.call_args[0][1]

        # Invoke the handler
        handler()

        # Verify screenshot was taken
        mock_plotter.screenshot.assert_called_once_with(
            str(tmp_path / "roi_context_2d_rgb.png")
        )

        # Verify camera JSON was saved
        json_path = tmp_path / "roi_context_2d_rgb_camera.json"
        assert json_path.exists()
        cam = json.loads(json_path.read_text())
        assert cam["position"] == [1.0, 2.0, 3.0]
        assert cam["focal_point"] == [0.0, 0.0, 0.0]
        assert cam["view_up"] == [0.0, 0.0, 1.0]

    def test_s_key_overwrites_on_repeat(self, tmp_path):
        """Pressing S twice should overwrite the same files."""
        mock_plotter = MagicMock()
        mock_plotter.camera.position = (1.0, 2.0, 3.0)
        mock_plotter.camera.focal_point = (0.0, 0.0, 0.0)
        mock_plotter.camera.up = (0.0, 0.0, 1.0)

        _bind_screenshot_key(mock_plotter, tmp_path, "test.png")
        handler = mock_plotter.add_key_event.call_args[0][1]

        handler()
        handler()

        # Screenshot called twice (overwrite)
        assert mock_plotter.screenshot.call_count == 2

        # JSON exists with latest values
        json_path = tmp_path / "test_camera.json"
        assert json_path.exists()


class TestRenderInteractive2d:
    def test_shows_interactive_window(self, tmp_path):
        """Should call plotter.show() for interactive display, not auto-screenshot."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            render_roi_context_2d(points, roi, tmp_path)

            # Should be interactive (title kwarg, not off_screen)
            call_kwargs = mock_pv.Plotter.call_args[1]
            assert "title" in call_kwargs
            assert call_kwargs.get("off_screen") is not True
            # Should show interactively and close after
            mock_plotter.show.assert_called_once()
            mock_plotter.close.assert_called_once()
            # Should NOT auto-screenshot
            mock_plotter.screenshot.assert_not_called()

    def test_title_contains_2d(self, tmp_path):
        """Window title should indicate 2D mode."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            render_roi_context_2d(points, roi, tmp_path)

            title = mock_pv.Plotter.call_args[1].get("title", "")
            assert "2D" in title

    def test_binds_s_key(self, tmp_path):
        """Should bind S key for screenshot capture."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            render_roi_context_2d(points, roi, tmp_path)

            mock_plotter.add_key_event.assert_called_once_with("s", ANY)


class TestRenderInteractive3d:
    def test_shows_interactive_window(self, tmp_path):
        """Should call plotter.show() for interactive 3D display."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            render_roi_context_3d(points, roi, tmp_path)

            mock_plotter.show.assert_called_once()
            mock_plotter.close.assert_called_once()
            mock_plotter.screenshot.assert_not_called()

    def test_title_contains_3d(self, tmp_path):
        """Window title should indicate 3D mode."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            render_roi_context_3d(points, roi, tmp_path)

            title = mock_pv.Plotter.call_args[1].get("title", "")
            assert "3D" in title


class TestRenderInteractiveRgb:
    def test_2d_rgb_uses_rgba_scalars(self, tmp_path):
        """When colors provided, add_mesh should use rgba=True."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        colors = rng.uniform(0, 1, (500, 3)).astype(np.float32)
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            render_roi_context_2d(points, roi, tmp_path, colors=colors)

            first_call = mock_plotter.add_mesh.call_args_list[0]
            _, kwargs = first_call
            assert kwargs.get("rgba") is True
            assert kwargs.get("scalars") == "RGBA"

    def test_2d_rgb_auto_filename_in_s_key(self, tmp_path):
        """When colors provided, S-key handler should use filename with _rgb suffix."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        colors = rng.uniform(0, 1, (500, 3)).astype(np.float32)
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            render_roi_context_2d(points, roi, tmp_path, colors=colors)

            # Extract and call the S-key handler
            handler = mock_plotter.add_key_event.call_args[0][1]
            handler()

            # Screenshot should use the auto-generated _rgb filename
            mock_plotter.screenshot.assert_called_once_with(
                str(tmp_path / "roi_context_2d_rgb.png")
            )


class TestPointSizeParam:
    def test_point_size_passed_to_plotter(self, tmp_path):
        """point_size parameter should be forwarded to add_mesh."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            render_roi_context_2d(points, roi, tmp_path, point_size=5.0)

            first_call = mock_plotter.add_mesh.call_args_list[0]
            _, kwargs = first_call
            assert kwargs.get("point_size") == 5.0


class TestZRangeFilter:
    def test_z_range_filters_roi_points(self, tmp_path):
        """Points inside ROI + Z range should be removed when z_range is provided."""
        rng = np.random.default_rng(42)
        # Create points: some inside ROI+Z, some outside
        inside = np.array([[3.0, 4.0, 5.0], [3.5, 4.5, 5.5]])  # inside ROI and Z
        outside = rng.uniform(0, 1, (100, 3))  # outside ROI bounds
        points = np.vstack([inside, outside])
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_polydata = MagicMock(side_effect=pv.PolyData)
            mock_pv.PolyData = mock_polydata

            render_roi_context_2d(points, roi, tmp_path, z_range=(4.0, 6.0))

            # The PolyData created for point cloud should have fewer points
            # (inside points removed)
            cloud_call = mock_pv.PolyData.call_args_list[0]
            rendered_pts = cloud_call[0][0]
            assert len(rendered_pts) == 100  # only outside points remain

    def test_no_filter_without_z_range(self, tmp_path):
        """Without z_range, all points should be rendered."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_polydata = MagicMock(side_effect=pv.PolyData)
            mock_pv.PolyData = mock_polydata

            render_roi_context_2d(points, roi, tmp_path)

            cloud_call = mock_pv.PolyData.call_args_list[0]
            rendered_pts = cloud_call[0][0]
            assert len(rendered_pts) == 500  # all points kept
