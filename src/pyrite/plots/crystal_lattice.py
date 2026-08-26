"""Compatibility re-export for :mod:`pyrite.plots.plotly.crystal_lattice`."""

from .._module_deprecations import warn_module_deprecation
from .plotly import crystal_lattice as _impl
from .plotly.crystal_lattice import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


warn_module_deprecation(__name__)
