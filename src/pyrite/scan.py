"""Compatibility re-export for :mod:`pyrite.runs.scan`."""

from ._module_deprecations import warn_module_deprecation
from .runs.scan import *  # noqa: F401,F403

warn_module_deprecation(__name__)
