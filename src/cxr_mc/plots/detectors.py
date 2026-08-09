"""Compatibility re-export for :mod:`cxr_mc.plots.mpl.detectors`."""

from .mpl import detectors as _impl
from .mpl.detectors import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
