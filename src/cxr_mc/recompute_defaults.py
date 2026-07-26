"""Profile-aware defaults shared by line and bremsstrahlung recomputes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np

PROFILE_NAMES = ("full", "survey")


def _validate_profile(profile: str) -> str:
    if profile not in PROFILE_NAMES:
        raise ValueError(f"profile must be one of {', '.join(PROFILE_NAMES)}")
    return profile


def settings(profile: str):
    """Return settings for ``profile``, including compatibility before profiles land."""
    from .config import default_settings

    profile = _validate_profile(profile)
    try:
        return cast(Any, default_settings)(profile=profile)
    except TypeError:
        current = default_settings()
        if profile == "full":
            return current
        return replace(current, n_electrons=60, n_electrons_brem=30)


def sweep(material: str, profile: str):
    """Return material sweep for ``profile``, with a standalone survey fallback."""
    from .config import material_sweep

    profile = _validate_profile(profile)
    try:
        return cast(Any, material_sweep)(material, profile=profile)
    except TypeError:
        current = material_sweep(material)
        if profile == "full":
            return current

        def reduced(values, limit):
            array = np.atleast_1d(np.asarray(values, dtype=float))
            if array.size <= limit:
                return array
            indices = np.linspace(0, array.size - 1, limit, dtype=int)
            return array[indices]

        line_by_energy = current.E_grid_line_by_energy
        if line_by_energy is not None:
            line_by_energy = {
                energy: np.asarray(grid, dtype=float)[::5]
                for energy, grid in line_by_energy.items()
            }
        brem = current.E_grid_brem
        if brem is not None:
            brem = np.asarray(brem, dtype=float)[::5]
        return replace(
            current,
            energy_keV=reduced(current.energy_keV, 2),
            thickness_ang=reduced(current.thickness_ang, 3),
            tilt_deg=reduced(current.tilt_deg, 5),
            tilt_azim_deg=reduced(current.tilt_azim_deg, 2),
            E_grid_line_by_energy=line_by_energy,
            E_grid_brem=brem,
            n_families=2,
        )


def uniform_bounds(grid) -> tuple[float, float, float]:
    """Return NumPy-arange ``start, stop, step`` for a regular profile grid."""
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("profile grid must contain at least two values")
    steps = np.diff(values)
    step = float(steps[0])
    if step <= 0 or not np.allclose(steps, step):
        raise ValueError("profile grid must be strictly increasing and uniform")
    return float(values[0]), float(values[-1] + step), step
