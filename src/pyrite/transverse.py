"""Compatibility re-export for :mod:`pyrite.montecarlo.transverse`."""

from ._module_deprecations import warn_module_deprecation
from .montecarlo.transverse import *  # noqa: F401,F403

warn_module_deprecation(__name__)
