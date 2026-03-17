"""Tests for downsampling cache (NPZ-based)."""

import time

import numpy as np
import pytest

from preprocessing.cache import (
    downsample_cache_path,
    format_voxel_size,
    load_cache,
    save_cache,
)


class TestFormatVoxelSize:
    def test_standard(self):
        assert format_voxel_size(0.01) == "0_01"

    def test_trailing_zeros(self):
        assert format_voxel_size(0.010) == "0_01"

    def test_small_value(self):
        assert format_voxel_size(0.0005) == "0_0005"


class TestCachePath:
    def test_generates_npz_path(self, tmp_path):
        path = downsample_cache_path("/data/scan.ply", 0.0005, cache_dir=tmp_path)
        assert path.suffix == ".npz"
        assert "scan" in path.name
        assert "0_0005" in path.name

    def test_different_voxel_sizes_different_paths(self, tmp_path):
        p1 = downsample_cache_path("/data/scan.ply", 0.0005, cache_dir=tmp_path)
        p2 = downsample_cache_path("/data/scan.ply", 0.001, cache_dir=tmp_path)
        assert p1 != p2


class TestSaveLoadCache:
    def test_round_trip(self, tmp_path):
        cache_path = tmp_path / "test.npz"
        source_path = tmp_path / "source.ply"
        source_path.write_bytes(b"dummy ply content")

        points = np.random.rand(100, 3).astype(np.float32)
        colors = np.random.rand(100, 3).astype(np.float32)
        intensity = np.random.rand(100).astype(np.float32)
        classification = np.array([1, 2, 3] * 33 + [1], dtype=np.float32)

        save_cache(cache_path, points, colors, intensity, classification, source_path)
        loaded = load_cache(cache_path, source_path)

        assert loaded is not None
        np.testing.assert_array_equal(loaded["points"], points)
        np.testing.assert_array_equal(loaded["colors"], colors)
        np.testing.assert_array_equal(loaded["intensity"], intensity)
        np.testing.assert_array_equal(loaded["classification"], classification)

    def test_invalidation_on_source_change(self, tmp_path):
        cache_path = tmp_path / "test.npz"
        source_path = tmp_path / "source.ply"
        source_path.write_bytes(b"original content")

        points = np.random.rand(10, 3).astype(np.float32)
        colors = np.random.rand(10, 3).astype(np.float32)
        intensity = np.random.rand(10).astype(np.float32)
        classification = np.zeros(10, dtype=np.float32)

        save_cache(cache_path, points, colors, intensity, classification, source_path)

        time.sleep(0.01)
        source_path.write_bytes(b"modified content")

        loaded = load_cache(cache_path, source_path)
        assert loaded is None

    def test_missing_cache_returns_none(self, tmp_path):
        loaded = load_cache(tmp_path / "nonexistent.npz", tmp_path / "source.ply")
        assert loaded is None

    def test_none_intensity_classification(self, tmp_path):
        cache_path = tmp_path / "test.npz"
        source_path = tmp_path / "source.ply"
        source_path.write_bytes(b"dummy")

        points = np.random.rand(10, 3).astype(np.float32)
        colors = np.random.rand(10, 3).astype(np.float32)

        save_cache(cache_path, points, colors, None, None, source_path)
        loaded = load_cache(cache_path, source_path)

        assert loaded is not None
        assert loaded["intensity"] is None
        assert loaded["classification"] is None
