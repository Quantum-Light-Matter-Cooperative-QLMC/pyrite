"""Compatibility facade for the package-root photon-continuum-floor leaf.

The floor itself lives in :mod:`pyrite._photon_continuum_floor` so that
``campaign`` can resolve a declared band against a medium's floor without
importing this driver package, which would put the two in a static import
cycle (see ``docs/repo_map.md`` and the note in ``energy_grid/__init__.py``).
:func:`geometric_continuum_grid` lives in :mod:`pyrite._continuum_grid` with the
rest of continuum node placement and is re-exported here.
"""

from .._continuum_grid import geometric_continuum_grid
from .._photon_continuum_floor import (
    DATA_SUPPORT_LIMITS_EV,
    continuum_medium_key,
    data_support_floor_eV,
    floored_lattice_start_eV,
    photon_continuum_floor_eV,
)

__all__ = [
    "DATA_SUPPORT_LIMITS_EV",
    "continuum_medium_key",
    "data_support_floor_eV",
    "floored_lattice_start_eV",
    "geometric_continuum_grid",
    "photon_continuum_floor_eV",
]
