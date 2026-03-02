"""Tests for height deviation heatmap computation."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from figure.height_heatmap import compute_height_grid, plot_height_heatmap


class TestComputeHeightGrid:
    def test_flat_surface_zero_range(self):
        """A perfectly flat surface should have near-zero height range everywhere."""
        rng = np.random.default_rng(42)
        n = 10000
        xy = rng.uniform(0, 10, (n, 2))
        z = np.full(n, 3.0)
        points = np.column_stack([xy, z])
        grid, x_edges, y_edges, cell_size = compute_height_grid(
            points, target_grid=10, min_points=3
        )
        valid = grid[~np.isnan(grid)]
        assert np.all(valid < 1e-6)

    def test_rough_surface_nonzero_range(self):
        """Surface with noise should have positive height range."""
        rng = np.random.default_rng(42)
        n = 10000
        xy = rng.uniform(0, 10, (n, 2))
        z = rng.normal(0, 0.5, n)
        points = np.column_stack([xy, z])
        grid, x_edges, y_edges, cell_size = compute_height_grid(
            points, target_grid=10, min_points=3
        )
        valid = grid[~np.isnan(grid)]
        assert np.mean(valid) > 0.1

    def test_tilted_plane_detrended_zero(self):
        """A perfect tilted plane should yield near-zero range after detrending."""
        rng = np.random.default_rng(42)
        n = 10000
        xy = rng.uniform(0, 10, (n, 2))
        z = 2.0 * xy[:, 0] + 0.5 * xy[:, 1] + 1.0
        points = np.column_stack([xy, z])
        grid, x_edges, y_edges, cell_size = compute_height_grid(
            points, target_grid=10, min_points=3
        )
        valid = grid[~np.isnan(grid)]
        assert np.mean(valid) < 0.1

    def test_sparse_cells_nan(self):
        """Cells with fewer than min_points should be NaN."""
        points = np.array([[0.0, 0.0, 1.0], [5.0, 5.0, 2.0]])
        grid, x_edges, y_edges, cell_size = compute_height_grid(
            points, target_grid=10, min_points=3
        )
        assert np.sum(~np.isnan(grid)) <= 2


class TestPlotHeightHeatmap:
    def _make_grid(self):
        """Helper: generate a small heatmap grid for testing."""
        rng = np.random.default_rng(42)
        n = 10000
        xy = rng.uniform(0, 10, (n, 2))
        z = rng.normal(0, 0.5, n)
        points = np.column_stack([xy, z])
        return compute_height_grid(points, target_grid=10, min_points=3)

    def test_saves_heatmap_files(self, tmp_path):
        """Heatmap should be saved as PNG and PDF."""
        grid, x_edges, y_edges, cell_size = self._make_grid()
        plot_height_heatmap(grid, x_edges, y_edges, cell_size, tmp_path, dpi=72)
        assert (tmp_path / "height_heatmap.png").exists()
        assert (tmp_path / "height_heatmap.pdf").exists()
