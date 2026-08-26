"""Compatibility re-export for :mod:`pyrite.plots.plotly.render`."""

from .._module_deprecations import warn_module_deprecation
from .plotly import render as _impl
from .plotly.render import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)


warn_module_deprecation(__name__)
