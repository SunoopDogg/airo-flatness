"""Tests for source selection UI functions."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from utils import select_downsampled_file, select_source


class TestSelectSource:
    def test_returns_original_when_cache_dir_empty(self, tmp_path):
        result = select_source(tmp_path)
        assert result == "original"

    def test_returns_original_when_cache_dir_missing(self, tmp_path):
        result = select_source(tmp_path / "nonexistent")
        assert result == "original"

    def test_returns_original_when_no_npz_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not an npz")
        result = select_source(tmp_path)
        assert result == "original"

    @patch("builtins.input", return_value="1")
    def test_returns_original_when_user_selects_1(self, mock_input, tmp_path):
        np.savez(tmp_path / "test.npz", points=np.zeros((1, 3)))
        result = select_source(tmp_path)
        assert result == "original"

    @patch("builtins.input", return_value="2")
    def test_returns_downsampled_when_user_selects_2(self, mock_input, tmp_path):
        np.savez(tmp_path / "test.npz", points=np.zeros((1, 3)))
        result = select_source(tmp_path)
        assert result == "downsampled"


class TestSelectDownsampledFile:
    def test_exits_when_no_npz_files(self, tmp_path):
        with pytest.raises(SystemExit):
            select_downsampled_file(tmp_path)

    @patch("builtins.input", return_value="1")
    def test_returns_selected_npz_path(self, mock_input, tmp_path):
        npz_path = tmp_path / "scan-0_0005.npz"
        np.savez(npz_path, points=np.zeros((10, 3)))
        result = select_downsampled_file(tmp_path)
        assert result == npz_path

    @patch("builtins.input", return_value="2")
    def test_selects_second_file(self, mock_input, tmp_path):
        np.savez(tmp_path / "a-0_0005.npz", points=np.zeros((5, 3)))
        npz_b = tmp_path / "b-0_01.npz"
        np.savez(npz_b, points=np.zeros((10, 3)))
        result = select_downsampled_file(tmp_path)
        assert result == npz_b

    @patch("builtins.input", return_value="1")
    def test_auto_selects_single_file(self, mock_input, tmp_path):
        npz_path = tmp_path / "only.npz"
        np.savez(npz_path, points=np.zeros((10, 3)))
        result = select_downsampled_file(tmp_path)
        # Should auto-select without prompting
        mock_input.assert_not_called()
        assert result == npz_path
