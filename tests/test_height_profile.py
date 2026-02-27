"""Tests for X-direction height profile computation and plotting."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from figure.height_profile import compute_height_profile, plot_height_profile


class TestComputeHeightProfile:
    def test_flat_surface_zero_residuals(self):
        """A perfectly flat surface should have near-zero mean residuals."""
        rng = np.random.default_rng(42)
        n = 10000
        xy = rng.uniform(0, 10, (n, 2))
        z = np.full(n, 3.0)
        points = np.column_stack([xy, z])
        x_centers, z_means = compute_height_profile(points, cell_size=1.0)
        assert len(x_centers) > 0
        assert len(x_centers) == len(z_means)
        assert np.all(np.abs(z_means) < 1e-6)

    def test_rough_surface_nonzero_residuals(self):
        """Surface with noise should have non-zero residuals."""
        rng = np.random.default_rng(42)
        n = 10000
        xy = rng.uniform(0, 10, (n, 2))
        z = rng.normal(0, 0.5, n)
        points = np.column_stack([xy, z])
        x_centers, z_means = compute_height_profile(points, cell_size=1.0)
        assert np.std(z_means) > 0.01

    def test_tilted_plane_detrended_zero(self):
        """A perfect tilted plane should yield near-zero residuals after detrending."""
        rng = np.random.default_rng(42)
        n = 10000
        xy = rng.uniform(0, 10, (n, 2))
        z = 2.0 * xy[:, 0] + 0.5 * xy[:, 1] + 1.0
        points = np.column_stack([xy, z])
        x_centers, z_means = compute_height_profile(points, cell_size=1.0)
        assert np.all(np.abs(z_means) < 1e-4)

    def test_strip_ratio_filters_y(self):
        """Narrower strip_ratio should use fewer points (central Y band)."""
        rng = np.random.default_rng(42)
        n = 10000
        xy = rng.uniform(0, 10, (n, 2))
        z = rng.normal(0, 0.1, n)
        points = np.column_stack([xy, z])
        x_wide, z_wide = compute_height_profile(points, cell_size=1.0, strip_ratio=0.8)
        x_narrow, z_narrow = compute_height_profile(points, cell_size=1.0, strip_ratio=0.2)
        assert len(x_wide) >= len(x_narrow)

    def test_empty_bins_excluded(self):
        """Bins with no points should not appear in output."""
        rng = np.random.default_rng(42)
        n = 5000
        xy1 = rng.uniform([0, 0], [2, 10], (n, 2))
        xy2 = rng.uniform([8, 0], [10, 10], (n, 2))
        xy = np.vstack([xy1, xy2])
        z = rng.normal(0, 0.1, len(xy))
        points = np.column_stack([xy, z])
        x_centers, z_means = compute_height_profile(points, cell_size=1.0)
        assert not np.any((x_centers > 3) & (x_centers < 7))


class TestPlotHeightProfile:
    def _make_profile(self):
        """Helper: generate a rough surface profile for testing."""
        rng = np.random.default_rng(42)
        n = 10000
        xy = rng.uniform(0, 10, (n, 2))
        z = rng.normal(0, 0.5, n)
        points = np.column_stack([xy, z])
        return compute_height_profile(points, cell_size=1.0)

    def test_saves_profile_files(self, tmp_path):
        """Profile should be saved as PNG and PDF."""
        x_centers, z_means = self._make_profile()
        plot_height_profile(x_centers, z_means, tmp_path, dpi=72)
        assert (tmp_path / "height_profile.png").exists()
        assert (tmp_path / "height_profile.pdf").exists()
