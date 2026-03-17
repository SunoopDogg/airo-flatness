"""Tests for load_ply_full() — full PLY loading without sampling."""

import struct
from pathlib import Path

import numpy as np
import pytest

from loader.ply_loader import load_ply_full


def _make_test_ply(n_points: int, path: Path, has_intensity: bool = True) -> None:
    """Create a minimal binary PLY file for testing."""
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_points}",
        "property double x",
        "property double y",
        "property double z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if has_intensity:
        header_lines.append("property float scalar_Intensity")
        header_lines.append("property float scalar_Classification")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n_points):
            f.write(struct.pack("<ddd", float(i), float(i * 2), float(i * 3)))
            f.write(struct.pack("<BBB", i % 256, (i * 2) % 256, (i * 3) % 256))
            if has_intensity:
                f.write(struct.pack("<ff", float(i * 0.1), float(i % 5)))


class TestLoadPlyFull:
    def test_loads_all_points(self, tmp_path):
        ply_path = tmp_path / "test.ply"
        _make_test_ply(100, ply_path)
        data = load_ply_full(ply_path)
        assert data["points"].shape == (100, 3)
        assert data["sampled_vertices"] == 100
        assert data["total_vertices"] == 100

    def test_preserves_coordinates(self, tmp_path):
        ply_path = tmp_path / "test.ply"
        _make_test_ply(10, ply_path)
        data = load_ply_full(ply_path)
        np.testing.assert_allclose(data["points"][0], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(data["points"][1], [1.0, 2.0, 3.0], atol=1e-6)

    def test_colors_normalized_to_0_1(self, tmp_path):
        ply_path = tmp_path / "test.ply"
        _make_test_ply(10, ply_path)
        data = load_ply_full(ply_path)
        assert data["colors"].dtype == np.float32
        assert data["colors"].max() <= 1.0
        assert data["colors"].min() >= 0.0

    def test_loads_intensity_and_classification(self, tmp_path):
        ply_path = tmp_path / "test.ply"
        _make_test_ply(10, ply_path)
        data = load_ply_full(ply_path)
        assert data["intensity"] is not None
        assert data["classification"] is not None
        assert data["intensity"].shape == (10,)

    def test_no_intensity_fields(self, tmp_path):
        ply_path = tmp_path / "test.ply"
        _make_test_ply(10, ply_path, has_intensity=False)
        data = load_ply_full(ply_path)
        assert data["intensity"] is None
        assert data["classification"] is None

    def test_no_color_fields(self, tmp_path):
        """PLY file without color properties."""
        ply_path = tmp_path / "test.ply"
        header = "ply\nformat binary_little_endian 1.0\nelement vertex 5\nproperty double x\nproperty double y\nproperty double z\nend_header\n"
        with open(ply_path, "wb") as f:
            f.write(header.encode("ascii"))
            for i in range(5):
                f.write(struct.pack("<ddd", float(i), float(i), float(i)))
        data = load_ply_full(ply_path)
        assert data["colors"] is None
        assert data["points"].shape == (5, 3)

    def test_progress_callback(self, tmp_path):
        ply_path = tmp_path / "test.ply"
        _make_test_ply(100, ply_path)
        calls = []
        data = load_ply_full(ply_path, progress_callback=lambda c, t: calls.append((c, t)))
        assert len(calls) > 0
        assert calls[-1][0] == calls[-1][1]
