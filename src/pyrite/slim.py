"""Compatibility re-export for :mod:`pyrite.checkpoints.slim`."""

from ._module_deprecations import warn_module_deprecation
from .checkpoints.slim import *  # noqa: F401,F403

warn_module_deprecation(__name__)
