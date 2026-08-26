"""Compatibility re-export for :mod:`pyrite.checkpoints.recompute_defaults`."""

from ._module_deprecations import warn_module_deprecation
from .checkpoints.recompute_defaults import *  # noqa: F401,F403

warn_module_deprecation(__name__)
