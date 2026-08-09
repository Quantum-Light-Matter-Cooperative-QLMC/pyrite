"""Compatibility re-export for :mod:`cxr_mc.plots.mpl.sweeps`."""

from .mpl import sweeps as _impl
from .mpl.sweeps import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
