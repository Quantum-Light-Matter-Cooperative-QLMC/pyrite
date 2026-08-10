"""Compatibility re-export for :mod:`pyrite.plots.altair.spectra`."""

from .altair import spectra as _impl
from .altair.spectra import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
