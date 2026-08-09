"""Compatibility re-export for :mod:`cxr_mc.plots.mpl.interactive`."""

from .mpl import interactive as _impl
from .mpl.interactive import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
