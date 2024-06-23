"""Triple Periodic Minimal Surfaces (TPMS) module."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
import pyvista as pv

from .surfaces import gyroid, neovius, schwarz_d, schwarz_p, tri_gyroid


class Tpms:
    """Triple Periodic Minimal Surfaces (TPMS) class."""

    def __init__(self: Tpms) -> None:
        """Initialize the TPMS object."""
        self.tpms_types = {
            "tri_gyroid": tri_gyroid,
            "gyroid": gyroid,
            "schwarz p": schwarz_p,
            "schwarz d": schwarz_d,
            "neovius": neovius,
        }

        self._surface_function = gyroid
        self._resolution = 20
        self._offset = 0.3
        self._phase_shift = np.array((0.0, 0.0, 0.0))
        self._cell_size = np.array((1.0, 1.0, 1.0))
        self._cell_repeat = np.array((1, 1, 1))

        self.grid = pv.StructuredGrid()

        self.generate_grid()
        self.compute_surface_field()
        self.compute_surfaces()
        self.sheet()

    def compute_density(self: Tpms) -> float:
        """Compute the density of the TPMS mesh."""
        grid_volume = np.prod(self.cell_size * self.cell_repeat)
        return -self.mesh.volume / grid_volume

    @property
    def surface_function(self: Tpms) -> Callable:
        """Return the surface function.

        Returns
        -------
        Callable: The surface function.

        """
        return self._surface_function

    @surface_function.setter
    def surface_function(self: Tpms, surface_function: Callable) -> None:
        self._surface_function = surface_function
        self.compute_surface_field()
        self.compute_surfaces()

    @property
    def resolution(self: Tpms) -> int:
        """Return the resolution.

        Returns
        -------
        int: The resolution.

        """
        return self._resolution

    @resolution.setter
    def resolution(self: Tpms, resolution: int) -> None:
        self._resolution = resolution
        self.generate_grid()
        self.compute_surface_field()
        self.compute_surfaces()

    @property
    def offset(self: Tpms) -> float:
        """Return the offset.

        Returns
        -------
        float: The offset.

        """
        return self._offset

    @offset.setter
    def offset(self: Tpms, offset: float) -> None:
        self._offset = offset
        self.compute_surfaces()

    @property
    def phase_shift(self: Tpms) -> npt.NDArray[np.floating]:
        """Return the phase shift.

        Returns
        -------
        npt.NDArray: The phase shift.

        """
        return self._phase_shift

    @phase_shift.setter
    def phase_shift(self: Tpms, phase_shift: tuple[float, float, float]) -> None:
        self._phase_shift = np.array(phase_shift)
        self.compute_surface_field()
        self.compute_surfaces()

    @property
    def cell_size(self: Tpms) -> npt.NDArray[np.floating]:
        """Return the cell size.

        Returns
        -------
        npt.NDArray: The cell size.

        """
        return self._cell_size

    @cell_size.setter
    def cell_size(self: Tpms, cell_size: tuple[float, float, float]) -> None:
        self._cell_size = np.array(cell_size)
        self.generate_grid()
        self.compute_surface_field()
        self.compute_surfaces()

    @property
    def cell_repeat(self: Tpms) -> npt.NDArray[np.uint8]:
        """Return the cell repeat.

        Returns
        -------
        npt.NDArray: The cell repeat.

        """
        return self._cell_repeat

    @cell_repeat.setter
    def cell_repeat(
        self: Tpms,
        cell_repeat: tuple[np.uint8, np.uint8, np.uint8],
    ) -> None:
        self._cell_repeat = np.array(cell_repeat)

        self.generate_grid()
        self.compute_surface_field()
        self.compute_surfaces()

    def generate_grid(self: Tpms) -> None:
        """Generate the grid on which to compute the TPMS surface field."""
        dim = self.cell_size * self.cell_repeat
        x = np.linspace(-0.5 * dim[0], 0.5 * dim[0], self.resolution)
        y = np.linspace(-0.5 * dim[1], 0.5 * dim[1], self.resolution)
        z = np.linspace(-0.5 * dim[2], 0.5 * dim[2], self.resolution)

        x, y, z = np.meshgrid(x, y, z)

        self.grid = pv.StructuredGrid(x, y, z)

    def compute_surface_field(self: Tpms) -> None:
        """Compute the surface field."""
        k = 2.0 * np.pi / self._cell_size

        coords = self.grid.points.T
        phi = self._phase_shift

        x, y, z = k[:, np.newaxis] * (coords + phi[:, np.newaxis])

        self.grid["surface"] = self.surface_function(x, y, z).flatten()

        self.min_field = self.grid["surface"].min()
        self.max_field = self.grid["surface"].max()
        self.min_offset = 0.0
        self.max_offset = 2.0 * max(-self.min_field, self.max_field)

    def compute_surfaces(self: Tpms) -> None:
        """Compute the lower and upper surfaces."""
        self.grid["lower_surface"] = self.grid["surface"] + 0.5 * self.offset
        self.grid["upper_surface"] = self.grid["surface"] - 0.5 * self.offset

    def sheet(self: Tpms) -> pv.UnstructuredGrid:
        """Generate the TPMS mesh from the grid and surfaces.

        Returns
        -------
        pv.UnstructuredGrid: The TPMS mesh.

        """
        self.mesh = self.grid.clip_scalar(
            scalars="lower_surface",
            invert=False,
        ).clip_scalar(scalars="upper_surface")
        return self.mesh
