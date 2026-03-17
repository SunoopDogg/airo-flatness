"""Tests for GPU voxel grid downsampling."""

import numpy as np
import pytest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


@pytest.fixture
def simple_cloud():
    """Two clusters of points that should merge into 2 voxels at voxel_size=1.0."""
    points = cp.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [1.1, 1.1, 1.1],
        [1.2, 1.2, 1.2],
    ], dtype=cp.float32)
    colors = cp.array([
        [255, 0, 0],
        [255, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 255, 0],
    ], dtype=cp.uint8)
    intensity = cp.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=cp.float32)
    classification = cp.array([2, 2, 3, 5, 5], dtype=cp.float32)
    return points, colors, intensity, classification


class TestAccumulateVoxelChunk:
    def test_empty_input(self):
        from preprocessing.downsampling import _accumulate_voxel_chunk
        pts = cp.empty((0, 3), dtype=cp.float32)
        cols = cp.empty((0, 3), dtype=cp.uint8)
        inten = cp.empty(0, dtype=cp.float32)
        classif = cp.empty(0, dtype=cp.float32)
        result = _accumulate_voxel_chunk(pts, cols, inten, classif, voxel_size=1.0)
        assert len(result.coord_sums) == 0

    def test_reduces_point_count(self, simple_cloud):
        from preprocessing.downsampling import _accumulate_voxel_chunk
        points, colors, intensity, classification = simple_cloud
        result = _accumulate_voxel_chunk(points, colors, intensity, classification, voxel_size=1.0)
        assert len(result.counts) == 2

    def test_counts_are_correct(self, simple_cloud):
        from preprocessing.downsampling import _accumulate_voxel_chunk
        points, colors, intensity, classification = simple_cloud
        result = _accumulate_voxel_chunk(points, colors, intensity, classification, voxel_size=1.0)
        counts = cp.sort(result.counts)
        np.testing.assert_array_equal(counts.get(), [2, 3])


class TestMergeAccumulations:
    def test_non_overlapping_voxels(self):
        """Two chunks with completely different voxels should concatenate."""
        from preprocessing.downsampling import _accumulate_voxel_chunk, _merge_accumulations
        pts1 = cp.array([[0.1, 0.1, 0.1]], dtype=cp.float32)
        pts2 = cp.array([[5.1, 5.1, 5.1]], dtype=cp.float32)
        cols = cp.array([[255, 0, 0]], dtype=cp.uint8)
        inten = cp.array([1.0], dtype=cp.float32)
        classif = cp.array([2.0], dtype=cp.float32)

        a1 = _accumulate_voxel_chunk(pts1, cols, inten, classif, voxel_size=1.0)
        a2 = _accumulate_voxel_chunk(pts2, cols, inten, classif, voxel_size=1.0)
        merged = _merge_accumulations([a1, a2])
        assert len(merged.counts) == 2

    def test_overlapping_voxels_combine(self):
        """Two chunks with same voxel should merge counts."""
        from preprocessing.downsampling import _accumulate_voxel_chunk, _merge_accumulations
        pts1 = cp.array([[0.1, 0.1, 0.1]], dtype=cp.float32)
        pts2 = cp.array([[0.2, 0.2, 0.2]], dtype=cp.float32)
        cols = cp.array([[255, 0, 0]], dtype=cp.uint8)
        inten = cp.array([1.0], dtype=cp.float32)
        classif = cp.array([2.0], dtype=cp.float32)

        a1 = _accumulate_voxel_chunk(pts1, cols, inten, classif, voxel_size=1.0)
        a2 = _accumulate_voxel_chunk(pts2, cols, inten, classif, voxel_size=1.0)
        merged = _merge_accumulations([a1, a2])
        assert len(merged.counts) == 1
        assert float(merged.counts[0]) == 2.0


class TestFinalizeVoxels:
    def test_averages_coordinates(self, simple_cloud):
        from preprocessing.downsampling import _accumulate_voxel_chunk, _finalize_voxels
        points, colors, intensity, classification = simple_cloud
        accum = _accumulate_voxel_chunk(points, colors, intensity, classification, voxel_size=1.0)
        avg_pts, avg_cols, avg_int, avg_cls = _finalize_voxels(accum)
        assert avg_pts.shape == (2, 3)
        sorted_x = cp.sort(avg_pts[:, 0])
        np.testing.assert_allclose(sorted_x.get(), [0.2, 1.15], atol=1e-5)

    def test_classification_uses_mode(self, simple_cloud):
        from preprocessing.downsampling import _accumulate_voxel_chunk, _finalize_voxels
        points, colors, intensity, classification = simple_cloud
        accum = _accumulate_voxel_chunk(points, colors, intensity, classification, voxel_size=1.0)
        _, _, _, avg_cls = _finalize_voxels(accum)
        sorted_cls = cp.sort(avg_cls)
        np.testing.assert_array_equal(sorted_cls.get(), [2.0, 5.0])

    def test_intensity_is_averaged(self, simple_cloud):
        from preprocessing.downsampling import _accumulate_voxel_chunk, _finalize_voxels
        points, colors, intensity, classification = simple_cloud
        accum = _accumulate_voxel_chunk(points, colors, intensity, classification, voxel_size=1.0)
        _, _, avg_int, _ = _finalize_voxels(accum)
        sorted_int = cp.sort(avg_int)
        np.testing.assert_allclose(sorted_int.get(), [0.2, 0.45], atol=1e-5)


class TestDownsampleVoxelGridGpu:
    def test_small_cloud_single_pass(self, simple_cloud):
        from preprocessing.downsampling import downsample_voxel_grid_gpu
        points, colors, intensity, classification = simple_cloud
        r_pts, r_cols, r_int, r_cls = downsample_voxel_grid_gpu(
            points, colors, intensity, classification,
            voxel_size=1.0, gpu_chunk_size=100,
        )
        assert len(r_pts) == 2

    def test_chunked_processing_same_result(self, simple_cloud):
        from preprocessing.downsampling import downsample_voxel_grid_gpu
        points, colors, intensity, classification = simple_cloud
        r1 = downsample_voxel_grid_gpu(points, colors, intensity, classification,
                                        voxel_size=1.0, gpu_chunk_size=100)
        r2 = downsample_voxel_grid_gpu(points, colors, intensity, classification,
                                        voxel_size=1.0, gpu_chunk_size=2)
        assert len(r1[0]) == len(r2[0])
        r1_sorted = r1[0][cp.argsort(r1[0][:, 0])].get()
        r2_sorted = r2[0][cp.argsort(r2[0][:, 0])].get()
        np.testing.assert_allclose(r1_sorted, r2_sorted, atol=1e-5)

    def test_empty_input(self):
        from preprocessing.downsampling import downsample_voxel_grid_gpu
        pts = cp.empty((0, 3), dtype=cp.float32)
        cols = cp.empty((0, 3), dtype=cp.uint8)
        inten = cp.empty(0, dtype=cp.float32)
        classif = cp.empty(0, dtype=cp.float32)
        r_pts, r_cols, r_int, r_cls = downsample_voxel_grid_gpu(
            pts, cols, inten, classif, voxel_size=1.0, gpu_chunk_size=100,
        )
        assert len(r_pts) == 0

    def test_large_coordinate_range(self):
        """Points spanning >10m should not cause voxel key overflow."""
        from preprocessing.downsampling import downsample_voxel_grid_gpu
        pts = cp.array([
            [0.0, 0.0, 0.0],
            [0.0001, 0.0001, 0.0001],
            [50.0, 50.0, 50.0],
            [50.0001, 50.0001, 50.0001],
        ], dtype=cp.float32)
        cols = cp.zeros((4, 3), dtype=cp.uint8)
        inten = cp.zeros(4, dtype=cp.float32)
        classif = cp.zeros(4, dtype=cp.float32)
        r_pts, _, _, _ = downsample_voxel_grid_gpu(
            pts, cols, inten, classif, voxel_size=0.0005, gpu_chunk_size=100,
        )
        assert len(r_pts) == 2  # 2 voxels, not 4


class TestDownsampleGpu:
    def test_entry_point_returns_4_tuple(self, simple_cloud):
        from preprocessing.downsampling import downsample_gpu
        points, colors, intensity, classification = simple_cloud
        result = downsample_gpu(points, colors, intensity, classification,
                                voxel_size=0.5, gpu_chunk_size=100)
        assert len(result) == 4
        r_pts, r_cols, r_int, r_cls = result
        assert r_pts.shape[1] == 3
        assert r_cols.shape[1] == 3
        assert r_int.ndim == 1
        assert r_cls.ndim == 1
