"""Tests for plane fitting and detrending."""

import numpy as np
import pytest

from figure.detrend import fit_plane, detrend_points


class TestFitPlane:
    def test_flat_horizontal_plane(self):
        """Points on Z=5 plane should yield coeffs (0, 0, 5)."""
        rng = np.random.default_rng(42)
        xy = rng.uniform(0, 10, (100, 2))
        z = np.full(100, 5.0)
        points = np.column_stack([xy, z])
        a, b, c = fit_plane(points)
        assert abs(a) < 1e-6
        assert abs(b) < 1e-6
        assert abs(c - 5.0) < 1e-6

    def test_tilted_plane(self):
        """Points on Z = 0.5*X + 0.3*Y + 2 should recover those coefficients."""
        rng = np.random.default_rng(42)
        xy = rng.uniform(0, 10, (200, 2))
        z = 0.5 * xy[:, 0] + 0.3 * xy[:, 1] + 2.0
        points = np.column_stack([xy, z])
        a, b, c = fit_plane(points)
        assert abs(a - 0.5) < 1e-6
        assert abs(b - 0.3) < 1e-6
        assert abs(c - 2.0) < 1e-4


class TestDetrendPoints:
    def test_detrended_residuals_are_zero_for_perfect_plane(self):
        """A perfect plane should yield all-zero residuals."""
        rng = np.random.default_rng(42)
        xy = rng.uniform(0, 10, (100, 2))
        z = 0.5 * xy[:, 0] + 0.3 * xy[:, 1] + 2.0
        points = np.column_stack([xy, z])
        residuals = detrend_points(points)
        assert np.allclose(residuals, 0.0, atol=1e-6)

    def test_detrended_residuals_preserve_roughness(self):
        """Adding noise to a plane should produce non-zero residuals matching the noise."""
        rng = np.random.default_rng(42)
        xy = rng.uniform(0, 10, (500, 2))
        noise = rng.normal(0, 0.1, 500)
        z = 0.5 * xy[:, 0] + 2.0 + noise
        points = np.column_stack([xy, z])
        residuals = detrend_points(points)
        assert abs(np.std(residuals) - 0.1) < 0.02
