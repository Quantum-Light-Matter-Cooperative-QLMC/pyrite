"""Opt-in supported-band checks for bounded component-specific diagnostics.

This module never changes the shared X-ray continuum floor or clips a grid.
Each requested medium/component pair must supply its own supported bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._grid_semantics import node_bin_edges_and_widths


@dataclass(frozen=True)
class ComponentSupport:
    """Closed photon-energy bands [eV] for one explicit medium/component pair.

    Declaring support is an assertion by the owning provider, not permission
    to extrapolate another component's response into these bands.
    """

    medium: str
    component: str
    bands_eV: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.medium or not self.component or not self.bands_eV:
            raise ValueError("component support requires medium, component and bands")
        previous = 0.0
        for lower, upper in self.bands_eV:
            if not np.isfinite(lower + upper) or lower <= previous or upper <= lower:
                raise ValueError("component support bands must be positive, ordered and disjoint")
            previous = upper
        object.__setattr__(self, "bands_eV", tuple((float(a), float(b)) for a, b in self.bands_eV))


def component_grid(
    energy_eV: object,
    *,
    required: tuple[tuple[str, str], ...],
    support: tuple[ComponentSupport, ...],
    representation: Literal["nodes", "bins"] = "nodes",
) -> np.ndarray:
    """Validate an explicit diagnostic grid without moving its coordinates.

    ``required`` contains (medium, component) pairs. Missing capability or any
    unsupported interval fails with that pair named. A node grid evaluates only
    at its nodes; ``bins`` additionally requires the full midpoint-bin extent
    to lie in one supported interval per bin. Bin edges/widths use the shared
    canonical conversion. No detector QE, attenuation or atomic-table fallback
    is installed. Default production grid resolution remains independent.
    """
    energy = np.asarray(energy_eV, dtype=float)
    if energy.ndim != 1 or energy.size < 2 or not np.all(np.isfinite(energy)):
        raise ValueError("component grid needs at least two finite 1-D nodes")
    if energy[0] <= 0 or np.any(np.diff(energy) <= 0):
        raise ValueError("component grid nodes must be positive and strictly increasing")
    if representation not in ("nodes", "bins"):
        raise ValueError("component grid representation must be nodes or bins")
    if not required:
        raise ValueError("component grid must name its required medium/component pairs")
    inventory = {(item.medium, item.component): item for item in support}
    if len(inventory) != len(support):
        raise ValueError("duplicate medium/component capability")
    left = right = energy
    if representation == "bins":
        edges, _ = node_bin_edges_and_widths(energy)
        left, right = edges[:-1], edges[1:]
    for medium, component in required:
        item = inventory.get((medium, component))
        if item is None:
            raise ValueError(f"{medium}/{component}: no supported-band capability supplied")
        covered = np.zeros(left.shape, dtype=bool)
        for lower, upper in item.bands_eV:
            covered |= (left >= lower) & (right <= upper)
        if not np.all(covered):
            raise ValueError(
                f"{medium}/{component}: unsupported energy or gap; bands {item.bands_eV} eV"
            )
    return energy.copy()
