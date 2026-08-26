"""Compatibility re-export for :mod:`pyrite.runs.run`."""

from ._module_deprecations import warn_module_deprecation
from .runs.run import *  # noqa: F401,F403

warn_module_deprecation(__name__)
