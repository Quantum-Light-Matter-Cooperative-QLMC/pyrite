"""Compatibility re-export for :mod:`cxr_mc.plots.mpl.spectra`."""

from .mpl import spectra as _impl
from .mpl.spectra import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
