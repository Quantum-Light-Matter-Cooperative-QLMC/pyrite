"""Compatibility re-export for :mod:`pyrite.runs.blaze`."""

from ._module_deprecations import warn_module_deprecation
from .runs.blaze import *  # noqa: F401,F403

warn_module_deprecation(__name__)
