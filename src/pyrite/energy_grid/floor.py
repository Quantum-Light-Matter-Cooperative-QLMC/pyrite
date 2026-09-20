"""Compatibility facade for the package-root photon-continuum-floor leaf.

The floor itself lives in :mod:`pyrite._photon_continuum_floor` so that
``campaign`` can resolve a declared band against a medium's floor without
importing this driver package, which would put the two in a static import
cycle (see ``docs/repo_map.md`` and the note in ``energy_grid/__init__.py``).
Only :func:`geometric_continuum_grid`, which nothing outside this package
needs, is defined here.
"""

from __future__ import annotations

import numpy as np

from .._photon_continuum_floor import (
    DATA_SUPPORT_LIMITS_EV,
    continuum_medium_key,
    data_support_floor_eV,
    floored_lattice_start_eV,
    photon_continuum_floor_eV,
)
from ..materials import MediumSpec

__all__ = [
    "DATA_SUPPORT_LIMITS_EV",
    "continuum_medium_key",
    "data_support_floor_eV",
    "floored_lattice_start_eV",
    "geometric_continuum_grid",
    "photon_continuum_floor_eV",
]


def geometric_continuum_grid(
    material: str | MediumSpec,
    stop_eV: float,
    num: int,
    *,
    floor_eV: float | None = None,
) -> np.ndarray:
    """Geometric continuum nodes from the derived floor to ``stop_eV``.

    The geometric baseline only -- no refinement near absorption edges or
    kinematic endpoints, which is a separate concern.

    Parameters
    ----------
    material
        Catalog crystal/media key or explicit homogeneous medium.
    stop_eV
        Highest node energy, strictly above the floor.
    num
        Endpoint-inclusive node count, at least 2.
    floor_eV
        Override for the lowest node. Refused below
        :func:`photon_continuum_floor_eV`, because a lower bound would place
        nodes where the emission and escape models carry no validity; passing a
        *higher* value is an ordinary bandwidth choice and is allowed.

    Returns
    -------
    numpy.ndarray
        Strictly increasing nodes in eV, ``E[0]`` exactly the resolved floor.

    Validation: photon-continuum-floor
    """
    derived = photon_continuum_floor_eV(material)
    if floor_eV is None:
        floor = derived
    else:
        floor = float(floor_eV)
        if not np.isfinite(floor):
            raise ValueError("floor_eV must be finite")
        if floor < derived:
            raise ValueError(
                f"floor_eV {floor:.6g} eV is below the derived continuum floor "
                f"{derived:.6g} eV for this medium; the continuum and photon-escape "
                "models carry no validity there, so widening the band downward needs a "
                "physics decision, not a smaller number"
            )
    stop = float(stop_eV)
    if not np.isfinite(stop) or stop <= floor:
        raise ValueError(f"stop_eV must be finite and above the floor {floor:.6g} eV")
    if int(num) != num or int(num) < 2:
        raise ValueError("num must be an integer of at least 2")
    return np.geomspace(floor, stop, int(num))
