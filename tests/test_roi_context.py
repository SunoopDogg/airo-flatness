"""Tests for ROI context view rendering."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

from figure.roi_context import (
    _subsample_points,
    _build_roi_rectangle_2d,
    render_roi_context_2d,
    render_roi_context_3d,
)


class TestSubsamplePoints:
    def test_returns_all_when_under_limit(self):
        """Should return all points when count <= max_points."""
        points = np.random.default_rng(42).uniform(0, 10, (100, 3))
        result, _ = _subsample_points(points, max_points=200, seed=42)
        assert len(result) == 100

    def test_subsamples_when_over_limit(self):
        """Should downsample to max_points when count exceeds it."""
        points = np.random.default_rng(42).uniform(0, 10, (1000, 3))
        result, _ = _subsample_points(points, max_points=200, seed=42)
        assert len(result) == 200
        assert result.shape[1] == 3

    def test_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        points = np.random.default_rng(42).uniform(0, 10, (1000, 3))
        r1, _ = _subsample_points(points, max_points=200, seed=42)
        r2, _ = _subsample_points(points, max_points=200, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_subsamples_colors_with_same_indices(self):
        """Colors should be subsampled with the same indices as points."""
        rng = np.random.default_rng(0)
        points = rng.uniform(0, 10, (1000, 3))
        colors = rng.uniform(0, 1, (1000, 3))

        pts, sub_colors = _subsample_points(points, max_points=200, seed=42, colors=colors)

        assert sub_colors is not None
        assert len(sub_colors) == 200
        # Verify each subsampled color corresponds to the correct point
        for i in range(len(pts)):
            orig_idx = np.where((points == pts[i]).all(axis=1))[0][0]
            np.testing.assert_array_equal(sub_colors[i], colors[orig_idx])

    def test_colors_returned_as_is_when_under_limit(self):
        """Colors should be passed through unchanged when no subsampling occurs."""
        rng = np.random.default_rng(0)
        points = rng.uniform(0, 10, (100, 3))
        colors = rng.uniform(0, 1, (100, 3))

        pts, sub_colors = _subsample_points(points, max_points=200, seed=42, colors=colors)

        assert len(pts) == 100
        np.testing.assert_array_equal(sub_colors, colors)


class TestBuildRoiRectangle2d:
    def test_creates_rectangle_with_4_points(self):
        """Should create a PolyData with 4 corner points."""
        import pyvista as pv

        rect = _build_roi_rectangle_2d((1.0, 3.0, 2.0, 5.0), z=10.0)
        assert isinstance(rect, pv.PolyData)
        assert rect.n_points == 4
        # Verify corner coordinates
        expected = np.array([
            [1.0, 2.0, 10.0],
            [3.0, 2.0, 10.0],
            [3.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
        ])
        np.testing.assert_array_almost_equal(rect.points, expected)

    def test_creates_4_line_segments(self):
        """Should have 4 line cells forming a closed rectangle."""
        rect = _build_roi_rectangle_2d((0.0, 1.0, 0.0, 1.0), z=0.0)
        assert rect.n_lines == 4


class TestRenderRoiContext2d:
    def test_creates_output_file(self, tmp_path):
        """Should save roi_context_2d.png to save_dir."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter

            render_roi_context_2d(points, roi, tmp_path, max_points=500, seed=42)

            mock_pv.Plotter.assert_called_once_with(off_screen=True)
            mock_plotter.set_background.assert_called_once_with("white")
            mock_plotter.view_isometric.assert_called_once()
            mock_plotter.screenshot.assert_called_once_with(
                str(tmp_path / "roi_context_2d.png")
            )

    def test_adds_point_cloud_with_z_scalars(self, tmp_path):
        """Should add point cloud colored by Z values."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter

            render_roi_context_2d(points, roi, tmp_path, max_points=500, seed=42)

            add_mesh_calls = mock_plotter.add_mesh.call_args_list
            assert len(add_mesh_calls) >= 1
            # First add_mesh call is the point cloud
            _, kwargs = add_mesh_calls[0]
            assert kwargs.get("cmap") == "viridis"

    def test_rectangle_at_z_median(self, tmp_path):
        """2D bounding box should be placed at Z median of all points."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter

            render_roi_context_2d(points, roi, tmp_path, max_points=500, seed=42)

            add_mesh_calls = mock_plotter.add_mesh.call_args_list
            # Second add_mesh call is the bounding box
            assert len(add_mesh_calls) >= 2
            bb_args, bb_kwargs = add_mesh_calls[1]
            assert bb_kwargs.get("color") == "red"
            assert bb_kwargs.get("style") == "wireframe"


class TestRenderRoiContext2dRgb:
    def test_rgb_uses_rgba_scalars(self, tmp_path):
        """When colors provided, add_mesh should use rgba=True not cmap."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        colors = rng.uniform(0, 1, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter

            render_roi_context_2d(points, roi, tmp_path, max_points=500, seed=42, colors=colors)

            add_mesh_calls = mock_plotter.add_mesh.call_args_list
            assert len(add_mesh_calls) >= 1
            _, kwargs = add_mesh_calls[0]
            assert kwargs.get("rgba") is True
            assert kwargs.get("scalars") == "RGBA"
            assert "cmap" not in kwargs

    def test_rgb_saves_custom_filename(self, tmp_path):
        """When filename provided, should save to that filename."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        colors = rng.uniform(0, 1, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter

            render_roi_context_2d(points, roi, tmp_path, max_points=500, seed=42,
                                  colors=colors, filename="custom_2d.png")

            mock_plotter.screenshot.assert_called_once_with(
                str(tmp_path / "custom_2d.png")
            )

    def test_rgb_auto_filename(self, tmp_path):
        """When colors provided without filename, should auto-use _rgb suffix."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        colors = rng.uniform(0, 1, (500, 3)).astype(np.float32)
        roi = (2.0, 4.0, 3.0, 6.0)
        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            render_roi_context_2d(points, roi, tmp_path, max_points=500, seed=42, colors=colors)
            mock_plotter.screenshot.assert_called_once_with(str(tmp_path / "roi_context_2d_rgb.png"))


class TestRenderRoiContext3d:
    def test_creates_output_file(self, tmp_path):
        """Should save roi_context_3d.png to save_dir."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.Box.return_value = MagicMock()

            render_roi_context_3d(points, roi, tmp_path, max_points=500, seed=42)

            mock_pv.Plotter.assert_called_once_with(off_screen=True)
            mock_plotter.screenshot.assert_called_once_with(
                str(tmp_path / "roi_context_3d.png")
            )

    def test_box_uses_roi_z_range(self, tmp_path):
        """3D box should use Z min/max of points within ROI."""
        rng = np.random.default_rng(42)
        roi_pts = np.column_stack([
            rng.uniform(2, 4, 100),
            rng.uniform(3, 6, 100),
            rng.uniform(1, 5, 100),
        ])
        outside_pts = np.column_stack([
            rng.uniform(8, 10, 100),
            rng.uniform(8, 10, 100),
            rng.uniform(0, 10, 100),
        ])
        points = np.vstack([roi_pts, outside_pts])
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.Box.return_value = MagicMock()

            render_roi_context_3d(points, roi, tmp_path, max_points=500, seed=42)

            mock_pv.Box.assert_called_once()
            bounds = mock_pv.Box.call_args[0][0]
            assert bounds[0] == pytest.approx(2.0)
            assert bounds[1] == pytest.approx(4.0)
            assert bounds[2] == pytest.approx(3.0)
            assert bounds[3] == pytest.approx(6.0)
            assert bounds[4] >= 0.5
            assert bounds[5] <= 5.5

    def test_box_rendered_as_red_wireframe(self, tmp_path):
        """3D box should be red wireframe."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.Box.return_value = MagicMock()

            render_roi_context_3d(points, roi, tmp_path, max_points=500, seed=42)

            add_mesh_calls = mock_plotter.add_mesh.call_args_list
            assert len(add_mesh_calls) >= 2
            _, box_kwargs = add_mesh_calls[1]
            assert box_kwargs.get("color") == "red"
            assert box_kwargs.get("style") == "wireframe"

    def test_empty_roi_falls_back_to_full_z_range(self, tmp_path):
        """When no points fall within ROI, box Z should use full cloud range."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (100.0, 200.0, 100.0, 200.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.Box.return_value = MagicMock()

            render_roi_context_3d(points, roi, tmp_path, max_points=500, seed=42)

            mock_pv.Box.assert_called_once()
            bounds = mock_pv.Box.call_args[0][0]
            assert bounds[4] == pytest.approx(points[:, 2].min())
            assert bounds[5] == pytest.approx(points[:, 2].max())


class TestRenderRoiContext3dRgb:
    def test_rgb_uses_rgba_scalars(self, tmp_path):
        """When colors provided, add_mesh should use rgba=True not cmap."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        colors = rng.uniform(0, 1, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.Box.return_value = MagicMock()

            render_roi_context_3d(points, roi, tmp_path, max_points=500, seed=42, colors=colors)

            add_mesh_calls = mock_plotter.add_mesh.call_args_list
            assert len(add_mesh_calls) >= 1
            _, kwargs = add_mesh_calls[0]
            assert kwargs.get("rgba") is True
            assert kwargs.get("scalars") == "RGBA"
            assert "cmap" not in kwargs

    def test_rgb_saves_custom_filename(self, tmp_path):
        """When filename provided, should save to that filename."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        colors = rng.uniform(0, 1, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.Box.return_value = MagicMock()

            render_roi_context_3d(points, roi, tmp_path, max_points=500, seed=42,
                                  colors=colors, filename="custom_3d.png")

            mock_plotter.screenshot.assert_called_once_with(
                str(tmp_path / "custom_3d.png")
            )

    def test_rgb_auto_filename(self, tmp_path):
        """When colors provided without filename, should auto-use _rgb suffix."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        colors = rng.uniform(0, 1, (500, 3)).astype(np.float32)
        roi = (2.0, 4.0, 3.0, 6.0)
        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.Box.return_value = MagicMock()
            render_roi_context_3d(points, roi, tmp_path, max_points=500, seed=42, colors=colors)
            mock_plotter.screenshot.assert_called_once_with(str(tmp_path / "roi_context_3d_rgb.png"))
