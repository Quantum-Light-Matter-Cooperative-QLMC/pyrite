"""Per-case photon-continuum grids of a sweep: declared bands and the automatic default.

Split from :mod:`pyrite.campaign.sweep`, which resolves each case's continuum
grid through :func:`case_brem_resolver`. A declared uniform band is floored at
its medium's start; an undeclared grid resolves the automatic refined geometric
grid (#278) over the film and, in a layered target, every layer beneath it
(#1164).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any

import numpy as np

from .._continuum_grid import automatic_continuum_grid
from .._energy_grid_encoding import encode_energy_grid
from .._photon_continuum_floor import continuum_medium_key, floored_lattice_start_eV
from ..materials import MediumSpec

if TYPE_CHECKING:
    from .sweep import Sweep

__all__ = [
    "automatic_continuum_medium",
    "automatic_continuum_stack_media",
    "case_brem_resolver",
    "floored_brem_case_grid",
]


def floored_brem_case_grid(sweep: Sweep, brem_grid: np.ndarray) -> Any:
    """Encode the continuum grid and floor a uniform band at the medium's start.

    A uniform band declares a `start`, but the energy below which the
    continuum and photon-escape models carry no validity is a property of
    the *medium*, not of the profile that declared the band. Resolving it
    here -- the first place the band and the material meet -- is what lets
    a profile-level default stay at 0.0 and still never produce a node
    outside the modelled band, and what keeps two profiles sharing the
    same material agreeing on the grid, which a per-profile stored start
    could not do. A declared start ABOVE the floor is an ordinary
    bandwidth choice and is kept, matching
    ``energy_grid.floor.geometric_continuum_grid``. Nonuniform grids are
    passed through: their builders resolve their own floor, and a sweep
    with no catalog material names no medium to take a floor from.
    """
    brem_case_grid = encode_energy_grid(brem_grid)
    if isinstance(brem_case_grid, tuple) and sweep.material is not None:
        start, stop, step = brem_case_grid
        brem_case_grid = (max(start, floored_lattice_start_eV(sweep.material, step)), stop, step)
    return brem_case_grid


def automatic_continuum_medium(sweep: Sweep) -> str | MediumSpec | None:
    """The medium whose automatic continuum grid every case uses, or ``None``.

    Cases resolve :func:`~pyrite._continuum_grid.automatic_continuum_grid` when
    no continuum grid is declared (``energy_bins.brem is None``: a profile with
    no ``E_grid_brem``, or a sweep built without one) and the sweep names a
    catalog material whose floor and absorption edges the grid is built from.
    Declared bands and explicit grids keep their own nodes.
    """
    if sweep.energy_bins.brem is not None or sweep.material is None:
        return None
    return continuum_medium_key(sweep.material)


def automatic_continuum_stack_media(sweep: Sweep) -> tuple[str, ...]:
    """Media beneath the film that the automatic continuum grid also spans (#1164).

    Every layer of a stack radiates and absorbs the continuum, so its floor and
    absorption edges bound the grid like the film's. Empty for a bare slab and
    for a sweep with a declared continuum grid; a layer of the film's own medium
    adds nothing and is dropped. Order follows the stack, without repeats.

    Validation: continuum-node-refinement
    """
    film = automatic_continuum_medium(sweep)
    if film is None:
        return ()
    assert sweep.target is not None  # Sweep.__post_init__ always resolves one
    below = sweep.target._below_film() or ()
    return tuple(dict.fromkeys(layer.material for layer in below if layer.material != film))


@lru_cache(maxsize=256)
def _automatic_grid_nodes(
    medium: str | MediumSpec, endpoint_eV: float, stack_media: tuple[str, ...] = ()
) -> np.ndarray:
    nodes = automatic_continuum_grid(medium, endpoint_eV, stack_media=stack_media)
    nodes.setflags(write=False)
    return nodes


def _automatic_case_brem_grid(
    medium: str | MediumSpec, energy_keV: float, stack_media: tuple[str, ...] = ()
) -> Any:
    """Exact refined geometric nodes from the stack floor to just past ``E0`` (#278, #1164).

    Validation: continuum-node-refinement
    """
    return encode_energy_grid(_automatic_grid_nodes(medium, energy_keV * 1e3, stack_media))


def case_brem_resolver(sweep: Sweep, brem_case_grid: Any) -> Callable[[float], Any]:
    """Per-``E0`` (keV) case continuum grid: automatic for an undeclared one (#278)."""
    medium = automatic_continuum_medium(sweep)
    if medium is not None:
        stack_media = automatic_continuum_stack_media(sweep)
        return partial(_automatic_case_brem_grid, medium, stack_media=stack_media)
    band = sweep.brem_band or sweep.energy_bins.brem is None
    return partial(_case_brem_grid, brem_case_grid, band=band)


def _case_brem_grid(brem_case_grid: Any, energy_keV: float, *, band: bool) -> Any:
    """A uniform band stops one step past this case's beam energy; others stay exact.

    ``band`` is False for an explicit ``EnergyBins.brem``: a uniform explicit
    grid is the requested nodes, not a step to extend to E0 (issue #815).
    """
    if not band or not isinstance(brem_case_grid, tuple):
        return brem_case_grid
    start, _, step = brem_case_grid
    return (start, energy_keV * 1e3 + step, step)
