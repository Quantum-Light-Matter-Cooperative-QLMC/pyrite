"""Compatibility re-export for :mod:`pyrite.plots.mpl.sweeps`."""

from .._module_deprecations import warn_module_deprecation
from .mpl import sweeps as _impl
from .mpl.sweeps import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


warn_module_deprecation(__name__)
