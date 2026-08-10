"""Compatibility re-export for :mod:`pyrite.plots.altair.detectors`."""

from .altair import detectors as _impl
from .altair.detectors import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
