"""Compatibility re-export for :mod:`cxr_mc.plots.altair.sweeps`."""

from .altair import sweeps as _impl
from .altair.sweeps import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
