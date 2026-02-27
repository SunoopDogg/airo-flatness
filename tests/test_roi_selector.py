"""Tests for ROI point filtering."""

import numpy as np
from unittest.mock import patch, MagicMock

from figure.roi_selector import filter_points_by_roi, filter_points_by_z
from figure.roi_selector import select_z_roi


class TestFilterPointsByRoi:
    def test_filters_within_bounds(self):
        """Points inside ROI should be returned."""
        points = np.array([
            [1.0, 1.0, 0.0],
            [5.0, 5.0, 0.0],
            [9.0, 9.0, 0.0],
        ])
        result = filter_points_by_roi(points, (0.0, 6.0, 0.0, 6.0))
        assert len(result) == 2
        assert np.allclose(result[0], [1.0, 1.0, 0.0])
        assert np.allclose(result[1], [5.0, 5.0, 0.0])

    def test_empty_roi_returns_empty(self):
        """ROI with no points inside should return empty array."""
        points = np.array([[10.0, 10.0, 0.0], [20.0, 20.0, 0.0]])
        result = filter_points_by_roi(points, (0.0, 1.0, 0.0, 1.0))
        assert len(result) == 0
        assert result.shape[1] == 3

    def test_boundary_points_included(self):
        """Points exactly on ROI boundary should be included."""
        points = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]])
        result = filter_points_by_roi(points, (0.0, 5.0, 0.0, 5.0))
        assert len(result) == 2


class TestFilterPointsByZ:
    def test_filters_within_z_range(self):
        """Points inside Z range should be returned."""
        points = np.array([
            [1.0, 1.0, 0.5],
            [2.0, 2.0, 1.5],
            [3.0, 3.0, 2.5],
            [4.0, 4.0, 3.5],
        ])
        result = filter_points_by_z(points, 1.0, 3.0)
        assert len(result) == 2
        assert np.allclose(result[0], [2.0, 2.0, 1.5])
        assert np.allclose(result[1], [3.0, 3.0, 2.5])

    def test_empty_result(self):
        """Z range with no matching points returns empty array."""
        points = np.array([[1.0, 1.0, 10.0], [2.0, 2.0, 20.0]])
        result = filter_points_by_z(points, 0.0, 1.0)
        assert len(result) == 0
        assert result.shape[1] == 3

    def test_boundary_values_included(self):
        """Points exactly on Z boundaries should be included."""
        points = np.array([
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 5.0],
            [3.0, 3.0, 10.0],
        ])
        result = filter_points_by_z(points, 0.0, 5.0)
        assert len(result) == 2
        assert np.allclose(result[0], [1.0, 1.0, 0.0])
        assert np.allclose(result[1], [2.0, 2.0, 5.0])


class TestSelectZRoi:
    @patch("figure.roi_selector.plt")
    def test_returns_selected_z_range(self, mock_plt):
        """Should return (z_min, z_max) from SpanSelector callback."""
        points = np.random.default_rng(42).uniform(0, 10, (1000, 3))

        captured_callback = {}

        def fake_span_selector(ax, onselect, direction, **kwargs):
            captured_callback["fn"] = onselect
            return MagicMock()

        with patch("figure.roi_selector.SpanSelector", side_effect=fake_span_selector):
            def fake_show():
                captured_callback["fn"](2.0, 8.0)

            mock_plt.show.side_effect = fake_show
            mock_plt.subplots.return_value = (MagicMock(), MagicMock())

            z_min, z_max = select_z_roi(points)

        assert z_min == 2.0
        assert z_max == 8.0

    @patch("figure.roi_selector.plt")
    def test_raises_if_no_selection(self, mock_plt):
        """Should raise ValueError if user closes without selecting."""
        points = np.random.default_rng(42).uniform(0, 10, (100, 3))

        with patch("figure.roi_selector.SpanSelector", return_value=MagicMock()):
            mock_plt.subplots.return_value = (MagicMock(), MagicMock())
            mock_plt.show.return_value = None

            try:
                select_z_roi(points)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "No Z range" in str(e)
