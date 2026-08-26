"""Compatibility re-export for :mod:`pyrite.perf.performance_profile`."""

from ._module_deprecations import warn_module_deprecation
from .perf.performance_profile import *  # noqa: F401,F403

warn_module_deprecation(__name__)
