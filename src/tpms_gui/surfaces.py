"""Surfaces functions for TPMS generation."""

from typing import Any

import numpy as np
import numpy.typing as npt


def tri_cos(x: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    """Triangular cosine function."""
    period_x = (x + 0.5) % 1.0 - 0.5
    return 1.0 - 4.0 * np.abs(period_x)


def tri_sin(x: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    """Triangular sine function."""
    return tri_cos(x - 0.25)


def tri_gyroid(
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
    z: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Triangular gyroid surface function."""
    tau = 2 * np.pi
    return (
        tri_sin(x / tau) * tri_cos(y / tau)
        + tri_sin(y / tau) * tri_cos(z / tau)
        + tri_sin(z / tau) * tri_cos(x / tau)
    )


def gyroid(
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
    z: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Gyroid surface function.

    Args:
    ----
    x (npt.NDArray): The x-coordinate.
    y (npt.NDArray): The y-coordinate.
    z (npt.NDArray): The z-coordinate.

    Returns:
    -------
    npt.NDArray: The gyroid surface.

    """
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


def schwarz_p(
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
    z: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Schwarz P surface function.

    Args:
    ----
    x (npt.NDArray): The x-coordinate.
    y (npt.NDArray): The y-coordinate.
    z (npt.NDArray): The z-coordinate.

    Returns:
    -------
    npt.NDArray: The Schwarz P surface.

    """
    return np.cos(x) + np.cos(y) + np.cos(z)


def schwarz_d(
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
    z: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Schwarz D surface function.

    Args:
    ----
    x (npt.NDArray): The x-coordinate.
    y (npt.NDArray): The y-coordinate.
    z (npt.NDArray): The z-coordinate.

    Returns:
    -------
    npt.NDArray: The Schwarz D surface.

    """
    return (
        np.sin(x) * np.sin(y) * np.sin(z)
        + np.sin(x) * np.cos(y) * np.cos(z)
        + np.cos(x) * np.sin(y) * np.cos(z)
        + np.cos(x) * np.cos(y) * np.sin(z)
    )


def neovius(
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
    z: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Neovius surface function.

    Args:
    ----
    x (npt.NDArray): The x-coordinate.
    y (npt.NDArray): The y-coordinate.
    z (npt.NDArray): The z-coordinate.

    Returns:
    -------
    npt.NDArray: The Neovius surface.

    """
    return 3 * np.cos(x) + np.cos(y) + np.cos(z) + 4 * np.cos(x) * np.cos(y) * np.cos(z)
