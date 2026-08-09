"""Compatibility re-export for :mod:`cxr_mc.plots.mpl.trajectories`."""

from .mpl import trajectories as _impl
from .mpl.trajectories import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
