"""Compatibility re-export for :mod:`cxr_mc.plots.plotly.render`."""

from .plotly import render as _impl
from .plotly.render import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
