"""Tests for load_from_downsampled_cache()."""

import numpy as np
import pytest

from preprocessing.cache import save_cache
from preprocessing.pipeline import load_from_downsampled_cache


class TestLoadFromDownsampledCache:
    def _create_npz(self, tmp_path, points, colors=None, intensity=None, classification=None):
        """Helper: create an NPZ file via save_cache and return its path."""
        source = tmp_path / "source.ply"
        source.write_bytes(b"dummy")
        npz_path = tmp_path / "test.npz"
        save_cache(npz_path, points, colors, intensity, classification, source)
        return npz_path

    def test_loads_all_fields(self, tmp_path):
        pts = np.random.rand(50, 3).astype(np.float32)
        cols = np.random.rand(50, 3).astype(np.float32)
        ints = np.random.rand(50).astype(np.float32)
        cls = np.ones(50, dtype=np.float32)

        npz_path = self._create_npz(tmp_path, pts, cols, ints, cls)
        data = load_from_downsampled_cache(npz_path)

        np.testing.assert_array_equal(data["points"], pts)
        np.testing.assert_array_equal(data["colors"], cols)
        np.testing.assert_array_equal(data["intensity"], ints)
        np.testing.assert_array_equal(data["classification"], cls)
        assert data["total_vertices"] == 50
        assert data["sampled_vertices"] == 50

    def test_none_optional_fields(self, tmp_path):
        pts = np.random.rand(20, 3).astype(np.float32)
        npz_path = self._create_npz(tmp_path, pts)
        data = load_from_downsampled_cache(npz_path)

        np.testing.assert_array_equal(data["points"], pts)
        assert data["colors"] is None
        assert data["intensity"] is None
        assert data["classification"] is None

    def test_partial_optional_fields(self, tmp_path):
        pts = np.random.rand(30, 3).astype(np.float32)
        cols = np.random.rand(30, 3).astype(np.float32)
        npz_path = self._create_npz(tmp_path, pts, colors=cols)
        data = load_from_downsampled_cache(npz_path)

        np.testing.assert_array_equal(data["colors"], cols)
        assert data["intensity"] is None
        assert data["classification"] is None

    def test_total_equals_sampled(self, tmp_path):
        pts = np.random.rand(100, 3).astype(np.float32)
        npz_path = self._create_npz(tmp_path, pts)
        data = load_from_downsampled_cache(npz_path)

        assert data["total_vertices"] == data["sampled_vertices"]
        assert data["total_vertices"] == 100

    def test_corrupt_npz_exits(self, tmp_path):
        corrupt = tmp_path / "corrupt.npz"
        corrupt.write_bytes(b"not a valid npz file")
        with pytest.raises(SystemExit):
            load_from_downsampled_cache(corrupt)


class TestRoundTripWithSaveCache:
    def test_save_cache_then_load_from_downsampled(self, tmp_path):
        """Verify NPZ saved by save_cache() is loadable by load_from_downsampled_cache()."""
        source = tmp_path / "scan.ply"
        source.write_bytes(b"dummy ply")
        cache_path = tmp_path / "scan-0_0005.npz"

        pts = np.random.rand(200, 3).astype(np.float32)
        cols = np.random.rand(200, 3).astype(np.float32)
        ints = np.random.rand(200).astype(np.float32)
        cls = np.array([2.0] * 200, dtype=np.float32)

        save_cache(cache_path, pts, cols, ints, cls, source)
        data = load_from_downsampled_cache(cache_path)

        np.testing.assert_array_equal(data["points"], pts)
        np.testing.assert_array_equal(data["colors"], cols)
        np.testing.assert_array_equal(data["intensity"], ints)
        np.testing.assert_array_equal(data["classification"], cls)
        assert data["total_vertices"] == 200
        assert data["sampled_vertices"] == 200
