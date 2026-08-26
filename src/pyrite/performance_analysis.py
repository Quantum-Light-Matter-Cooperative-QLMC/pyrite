"""Compatibility re-export for :mod:`pyrite.perf.performance_analysis`."""

from ._module_deprecations import warn_module_deprecation
from .perf.performance_analysis import *  # noqa: F401,F403

warn_module_deprecation(__name__)
