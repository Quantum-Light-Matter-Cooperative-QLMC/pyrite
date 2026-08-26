"""Compatibility re-export for :mod:`pyrite.plots.altair.sweeps`."""

from .._module_deprecations import warn_module_deprecation
from .altair import sweeps as _impl
from .altair.sweeps import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


warn_module_deprecation(__name__)
