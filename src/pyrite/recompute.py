"""Compatibility re-export for :mod:`pyrite.checkpoints.recompute`."""

from ._module_deprecations import warn_module_deprecation
from .checkpoints.recompute import *  # noqa: F401,F403

warn_module_deprecation(__name__)
