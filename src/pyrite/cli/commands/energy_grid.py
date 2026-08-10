"""CLI adapter for photon-energy-grid commands.

Implementation remains in :mod:`pyrite.energy_grid`; this module owns its root
CLI registration path while grid derivation code stays beside its domain code.
"""

from pyrite.energy_grid import command

__all__ = ["command"]
