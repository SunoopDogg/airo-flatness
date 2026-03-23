"""Tests for GPU compute path — verify CuPy arrays produce same results as NumPy."""

import numpy as np
import pytest

try:
    import cupy as cp
    # Verify CuPy can actually run matmul (requires libcublasLt — not just importable)
    _a = cp.ones((2, 2), dtype=cp.float32)
    _a @ _a
    del _a
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


@pytest.fixture
def sample_points():
    """Tilted plane with noise — tests detrend + grid + profile together."""
    rng = np.random.default_rng(42)
    n = 10000
    xy = rng.uniform(0, 10, (n, 2))
    noise = rng.normal(0, 0.1, n)
    z = 0.5 * xy[:, 0] + 0.3 * xy[:, 1] + 2.0 + noise
    return np.column_stack([xy, z]).astype(np.float32)


class TestDetrendGpu:
    def test_gpu_matches_cpu(self, sample_points):
        from figure.detrend import detrend_points
        cpu_result = detrend_points(sample_points)
        gpu_result = detrend_points(cp.asarray(sample_points))
        np.testing.assert_allclose(cpu_result, gpu_result.get(), atol=1e-5)


class TestHeightGridGpu:
    def test_gpu_matches_cpu(self, sample_points):
        from figure.height_heatmap import compute_height_grid
        cpu_grid, _, _, _ = compute_height_grid(sample_points, target_grid=10)
        gpu_grid, _, _, _ = compute_height_grid(cp.asarray(sample_points), target_grid=10)
        # Both return NumPy
        np.testing.assert_allclose(
            np.nan_to_num(cpu_grid), np.nan_to_num(gpu_grid), atol=1e-5,
        )


class TestHeightProfileGpu:
    def test_gpu_matches_cpu(self, sample_points):
        from figure.height_profile import compute_height_profile
        cpu_x, cpu_z = compute_height_profile(sample_points, cell_size=1.0)
        gpu_x, gpu_z = compute_height_profile(cp.asarray(sample_points), cell_size=1.0)
        np.testing.assert_allclose(cpu_x, gpu_x, atol=1e-5)
        np.testing.assert_allclose(cpu_z, gpu_z, atol=1e-5)
