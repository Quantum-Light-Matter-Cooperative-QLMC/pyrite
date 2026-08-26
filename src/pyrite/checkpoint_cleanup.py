"""Compatibility re-export for :mod:`pyrite.checkpoints.checkpoint_cleanup`."""

from ._module_deprecations import warn_module_deprecation
from .checkpoints.checkpoint_cleanup import *  # noqa: F401,F403

warn_module_deprecation(__name__)
