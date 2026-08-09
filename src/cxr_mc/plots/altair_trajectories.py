"""Compatibility re-export for :mod:`cxr_mc.plots.altair.trajectories`."""

from .altair import trajectories as _impl
from .altair.trajectories import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
