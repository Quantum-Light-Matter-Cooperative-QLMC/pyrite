"""Compatibility re-export for :mod:`pyrite.checkpoints.archive`."""

from ._module_deprecations import warn_module_deprecation
from .checkpoints.archive import *  # noqa: F401,F403

warn_module_deprecation(__name__)
