"""Compatibility re-export for :mod:`pyrite.plots.altair.trajectories`."""

from .._module_deprecations import warn_module_deprecation
from .altair import trajectories as _impl
from .altair.trajectories import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


warn_module_deprecation(__name__)
