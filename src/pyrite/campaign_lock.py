"""Compatibility re-export for :mod:`pyrite.checkpoints.campaign_lock`."""

from ._module_deprecations import warn_module_deprecation
from .checkpoints.campaign_lock import *  # noqa: F401,F403

warn_module_deprecation(__name__)
