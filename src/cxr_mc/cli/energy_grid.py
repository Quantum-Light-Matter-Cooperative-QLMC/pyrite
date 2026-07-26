"""CLI adapter for photon-energy-grid commands.

Implementation remains in :mod:`cxr_mc.line_grid`; this module owns its root
CLI registration path while grid derivation code stays beside its domain code.
"""

from cxr_mc.line_grid import command

__all__ = ["command"]
