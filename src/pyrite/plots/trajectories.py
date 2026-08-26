"""Compatibility re-export for :mod:`pyrite.plots.mpl.trajectories`."""

from .._module_deprecations import warn_module_deprecation
from .mpl import trajectories as _impl
from .mpl.trajectories import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


warn_module_deprecation(__name__)
