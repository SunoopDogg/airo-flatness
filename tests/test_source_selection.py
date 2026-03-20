"""Tests for source selection UI functions."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from utils import select_source


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
