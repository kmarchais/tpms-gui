"""Graphical User Interface for the TPMS app."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

import pyvista as pv
from matplotlib import colormaps
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3

from tpms_gui.tpms import Tpms


class Gui:
    """TPMS GUI app."""

    def __init__(self: Gui) -> None:
        """Initialize the app."""
        server = get_server()
        if server is None:
            err_msg = "No server found"
            raise RuntimeError(err_msg)
        self.server = server

        self.tpms = Tpms()

        self.cmap = "coolwarm"
        self.scalars = "surface"
        self.show_grid = False

        self.plotter = pv.Plotter(off_screen=True)
        self.mesh = self.tpms.mesh
        self.actor = self.plotter.add_mesh(
            self.mesh,
            name="TPMS",
            cmap=self.cmap,
            scalars=self.scalars,
        )
        self.plotter.add_axes()

        self.define_callbacks()
        self.draw()

    def run(self: Gui, **kwargs) -> None:
        """Run the app."""
        self.server.start(**kwargs)

    def draw(self: Gui) -> None:
        """Draw the app layout."""
        with SinglePageWithDrawerLayout(self.server) as layout:
            layout.title.text = "TPMS"
            with layout.toolbar:
                vuetify3.VSpacer()

                with vuetify3.VBtnToggle(
                    mandatory="force",
                    v_model=("show_grid", True),
                    color="primary",
                    elevation=0,
                    dense=True,
                    classes="ml-2",
                ) as self.toggle_group:
                    btn_kwargs = {
                        "color": "primary",
                        "fab": True,
                        "dark": True,
                        "elevation": 0,
                    }
                    vuetify3.VBtn(text="Mesh", **btn_kwargs)
                    vuetify3.VBtn(text="Grid", **btn_kwargs)

                vuetify3.VSelect(
                    label="Scalars",
                    v_model=("scalars", self.mesh.active_scalars_name),
                    items=("array_list", list(self.mesh.point_data.keys())),
                    hide_details=True,
                    density="compact",
                    outlined=True,
                    classes="pt-1 ml-2",
                    style="max-width: 250px",
                )
                vuetify3.VSelect(
                    label="Color map",
                    v_model=("cmap", "coolwarm"),
                    items=(
                        "cmap_list",
                        [cmap for cmap in colormaps if not cmap.endswith("_r")],
                    ),
                    hide_details=True,
                    density="compact",
                    outlined=True,
                    classes="pt-1 ml-2",
                    style="max-width: 250px",
                )
                vuetify3.VCheckbox(
                    label="Flip colormap",
                    v_model=("flip_cmap", False),
                    hide_details=True,
                    dense=True,
                    classes="pt-1 ml-2",
                )

            with layout.drawer:
                slider_kwargs = {
                    "hide_details": True,
                    "dense": True,
                    "thumb_label": True,
                    "classes": "pt-1 ml-2",
                    "style": "max-width: 250px",
                }
                vuetify3.VSelect(
                    label="Type",
                    v_model=("tpms_type", "gyroid"),
                    items=("tpms_list", list(self.tpms.tpms_types.keys())),
                    hide_details=True,
                    density="compact",
                    outlined=True,
                    classes="pt-1 ml-2",
                    style="max-width: 250px",
                )

                vuetify3.VDivider(**slider_kwargs, thickness=5)

                vuetify3.VSlider(
                    label="Resolution",
                    v_model=("resolution", 20),
                    min=10,
                    max=100,
                    step=10,
                    **slider_kwargs,
                )

                vuetify3.VDivider(**slider_kwargs, thickness=5)

                vuetify3.VSlider(
                    label="Offset",
                    v_model=("percent_offset", 0.5),
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    **slider_kwargs,
                )

                vuetify3.VDivider(**slider_kwargs, thickness=5)

                params: dict[str, dict[str, float]] = {
                    "Cell repeat": {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                    },
                    "Cell size": {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                    },
                    "Phase shift": {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                    },
                }
                for param in params:
                    var = param.lower().replace(" ", "_")
                    for axis in ("X", "Y", "Z"):
                        vuetify3.VSlider(
                            label=f"{param} {axis}",
                            v_model=(
                                f"{var}_{axis.lower()}",
                                params[param]["default"],
                            ),
                            min=params[param]["min"],
                            max=params[param]["max"],
                            step=params[param]["step"],
                            **slider_kwargs,
                        )
                    vuetify3.VDivider(**slider_kwargs, thickness=5)

            with layout.content, vuetify3.VContainer(
                fluid=True,
                classes="pa-0 fill-height",
            ):
                view = plotter_ui(self.plotter)
                self.server.controller.view_update = view.update

    def update_mesh(self: Gui) -> None:
        """Update the plotter."""
        self.tpms.sheet()

        self.mesh = self.tpms.grid if self.show_grid else self.tpms.mesh

    def update_actor(self: Gui) -> None:
        """Update the actor."""
        self.actor = self.plotter.add_mesh(
            self.mesh,
            name="TPMS",
            cmap=self.cmap,
            scalars=self.scalars,
        )

    def define_callbacks(self: Gui) -> None:
        """Define the callbacks for the app."""

        @self.server.state.change("show_grid")
        def set_display_type(
            *,
            show_grid: bool,
            **_: list | int | str | bool | None | dict,
        ) -> None:
            """Set the active display (Mesh or Grid) type and update the view."""
            self.show_grid = show_grid
            self.mesh = self.tpms.grid if show_grid else self.tpms.mesh
            self.update_actor()
            self.server.controller.view_update()

        @self.server.state.change("scalars")
        def set_scalars(
            scalars: str,
            **_: list | int | str | bool | None | dict,
        ) -> None:
            """Set the active scalars on the mesh and update the view."""
            self.scalars = scalars
            self.plotter.scalar_bar.SetTitle(scalars)
            self.actor.mapper.set_scalars(
                scalars=self.mesh.point_data[self.scalars],
                scalars_name=self.scalars,
                cmap=self.cmap,
            )
            self.server.controller.view_update()

        @self.server.state.change("cmap", "flip_cmap")
        def reset_cmap(
            cmap: str,
            *,
            flip_cmap: bool,
            **_: list | int | str | bool | None | dict,
        ) -> None:
            """Set the new colormap."""
            self.cmap = cmap
            self.actor.mapper.lookup_table.apply_cmap(self.cmap, flip=flip_cmap)
            self.server.controller.view_update()

        @self.server.state.change(
            "resolution",
            "cell_repeat_x",
            "cell_repeat_y",
            "cell_repeat_z",
            "cell_size_x",
            "cell_size_y",
            "cell_size_z",
        )
        def reset_grid(
            resolution: int,
            cell_repeat_x: np.uint8,
            cell_repeat_y: np.uint8,
            cell_repeat_z: np.uint8,
            cell_size_x: float,
            cell_size_y: float,
            cell_size_z: float,
            **_: list | int | str | bool | None | dict,
        ) -> None:
            """Recompute the grid and update the view."""
            self.tpms.resolution = resolution
            self.tpms.cell_repeat = (cell_repeat_x, cell_repeat_y, cell_repeat_z)
            self.tpms.cell_size = (cell_size_x, cell_size_y, cell_size_z)

            self.update_mesh()
            self.update_actor()
            self.server.controller.view_update()

        @self.server.state.change(
            "tpms_type",
            "phase_shift_x",
            "phase_shift_y",
            "phase_shift_z",
        )
        def reset_surface(
            tpms_type: str,
            phase_shift_x: float,
            phase_shift_y: float,
            phase_shift_z: float,
            **_: list | int | str | bool | None | dict,
        ) -> None:
            """Recompute the surface field and update the view."""
            self.tpms.surface_function = self.tpms.tpms_types[tpms_type]
            self.tpms.phase_shift = (phase_shift_x, phase_shift_y, phase_shift_z)

            self.update_mesh()
            self.update_actor()
            self.server.controller.view_update()

        @self.server.state.change("percent_offset")
        def reset_surfaces(
            percent_offset: float,
            **_: list | int | str | bool | None | dict,
        ) -> None:
            """Recompute the lower and upper surfaces and update the view.

            Args:
            ----
                percent_offset (float): The percentage offset.
                    0% is the minimum offset, 100% is the maximum offset.

            """
            self.tpms.offset = (
                self.tpms.min_offset
                + (self.tpms.max_offset - self.tpms.min_offset) * percent_offset
            )

            self.update_mesh()
            self.update_actor()
            self.server.controller.view_update()
