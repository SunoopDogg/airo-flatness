"""Tests for 3D surface mesh building."""

import numpy as np
import pytest
import pyvista as pv

from figure.surface_3d import build_surface_mesh


class TestBuildSurfaceMesh:
    def test_basic_mesh_creation(self):
        """A grid of points should produce a valid StructuredGrid mesh."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        xx, yy = np.meshgrid(x, y)
        zz = rng.normal(0, 0.1, xx.shape)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        mesh = build_surface_mesh(points, max_points=5000, grid_resolution=50)
        assert isinstance(mesh, pv.StructuredGrid)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0
        assert "Z" in mesh.point_data

    def test_grid_resolution_controls_output_size(self):
        """Higher grid_resolution should produce more points."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (5000, 3))
        mesh_low = build_surface_mesh(points, max_points=5000, grid_resolution=20)
        mesh_high = build_surface_mesh(points, max_points=5000, grid_resolution=80)
        assert mesh_high.n_points > mesh_low.n_points

    def test_z_exaggeration(self):
        """Z exaggeration should amplify Z range."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 30)
        y = np.linspace(0, 10, 30)
        xx, yy = np.meshgrid(x, y)
        zz = rng.normal(0, 1.0, xx.shape)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        mesh_1x = build_surface_mesh(points, max_points=5000, grid_resolution=30, z_exaggeration=1.0)
        mesh_3x = build_surface_mesh(points, max_points=5000, grid_resolution=30, z_exaggeration=3.0)
        z_range_1x = mesh_1x.points[:, 2].max() - mesh_1x.points[:, 2].min()
        z_range_3x = mesh_3x.points[:, 2].max() - mesh_3x.points[:, 2].min()
        assert z_range_3x > z_range_1x * 2.5
