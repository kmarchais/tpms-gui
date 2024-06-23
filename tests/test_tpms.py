"""Tests for the TPMS class."""

import numpy as np

from tpms_gui.tpms import Tpms

# ruff: noqa: S101 (use of assert)


def test_default_tpms_generates_mesh() -> None:
    """Test that the default TPMS generates a mesh."""
    tpms = Tpms()
    assert tpms.grid.n_points > 0
    assert tpms.grid.n_cells > 0


def test_tpms_surface_function() -> None:
    """Test that the TPMS surface function can be set."""
    tpms = Tpms()
    tpms.surface_function = tpms.tpms_types["tri_gyroid"]
    assert tpms.surface_function == tpms.tpms_types["tri_gyroid"]


def test_tpms_resolution() -> None:
    """Test that the TPMS resolution can be set."""
    tpms = Tpms()
    res = 10
    tpms.resolution = res
    assert tpms.resolution == res


def test_tpms_offset() -> None:
    """Test that the TPMS offset can be set."""
    tpms = Tpms()
    offset = 0.1
    tpms.offset = offset
    assert tpms.offset == offset


def test_tpms_phase_shift() -> None:
    """Test that the TPMS phase shift can be set."""
    tpms = Tpms()
    phi = np.array((0.1, 0.2, 0.3))
    tpms.phase_shift = phi
    assert all(tpms.phase_shift == phi)


def test_tpms_cell_size() -> None:
    """Test that the TPMS cell size can be set."""
    tpms = Tpms()
    cell_size = np.array((0.1, 0.2, 0.3))
    tpms.cell_size = cell_size
    assert all(tpms.cell_size == cell_size)


def test_tpms_cell_repeat() -> None:
    """Test that the TPMS cell repeat can be set."""
    tpms = Tpms()
    cell_repeat = np.array((2, 3, 4))
    tpms.cell_repeat = cell_repeat
    assert all(tpms.cell_repeat == cell_repeat)


def test_tpms_generate_grid() -> None:
    """Test that the TPMS grid can be generated."""
    tpms = Tpms()
    tpms.generate_grid()
    assert tpms.grid.n_points > 0
    assert tpms.grid.n_cells > 0


def test_tpms_compute_surface_field() -> None:
    """Test that the TPMS surface field can be computed."""
    tpms = Tpms()
    tpms.compute_surface_field()


def test_tpms_compute_surfaces() -> None:
    """Test that the TPMS surfaces can be computed."""
    tpms = Tpms()
    tpms.compute_surfaces()


def test_tpms_sheet() -> None:
    """Test that the TPMS sheet can be computed."""
    tpms = Tpms()
    tpms.sheet()
