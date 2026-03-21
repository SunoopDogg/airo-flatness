"""Tests for ROI context view rendering."""

import json
import numpy as np
import pyvista as pv
import pytest
from unittest.mock import patch, MagicMock, ANY

from figure.roi_context import (
    _build_dashed_rectangle_2d,
    _build_dashed_box_3d,
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


class TestRenderRoiContext2d:
    def test_calls_plotter_offscreen(self, tmp_path):
        """Should create off-screen plotter and take screenshot."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            render_roi_context_2d(points, roi, tmp_path, max_points=500, seed=42)

            mock_pv.Plotter.assert_called_once_with(off_screen=True)
            mock_plotter.set_background.assert_called_once_with("white")
            mock_plotter.view_isometric.assert_called_once()
            mock_plotter.screenshot.assert_called_once()
            mock_plotter.close.assert_called_once()


class TestRenderRoiContext3d:
    def test_calls_plotter_offscreen(self, tmp_path):
        """Should create off-screen plotter and take screenshot."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv, \
             patch("figure.roi_context.filter_points_by_roi", return_value=points[:10]):
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            render_roi_context_3d(points, roi, tmp_path, max_points=500, seed=42)

            mock_pv.Plotter.assert_called_once_with(off_screen=True)
            mock_plotter.screenshot.assert_called_once()
            mock_plotter.close.assert_called_once()


class TestRenderRoiContextRgb:
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

            render_roi_context_2d(points, roi, tmp_path, max_points=500, seed=42, colors=colors)

            first_call = mock_plotter.add_mesh.call_args_list[0]
            _, kwargs = first_call
            assert kwargs.get("rgba") is True
            assert kwargs.get("scalars") == "RGBA"

    def test_2d_rgb_auto_filename(self, tmp_path):
        """When colors provided, screenshot filename should include _rgb suffix."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (500, 3))
        colors = rng.uniform(0, 1, (500, 3)).astype(np.float32)
        roi = (2.0, 4.0, 3.0, 6.0)

        with patch("figure.roi_context.pv") as mock_pv:
            mock_plotter = MagicMock()
            mock_pv.Plotter.return_value = mock_plotter
            mock_pv.PolyData = pv.PolyData

            render_roi_context_2d(points, roi, tmp_path, max_points=500, seed=42, colors=colors)

            mock_plotter.screenshot.assert_called_once_with(
                str(tmp_path / "roi_context_2d_rgb.png")
            )
