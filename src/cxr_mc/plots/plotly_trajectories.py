"""Compatibility re-export for :mod:`cxr_mc.plots.plotly.trajectories`."""

from .plotly import trajectories as _impl
from .plotly.trajectories import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
