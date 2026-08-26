"""Compatibility re-export for :mod:`pyrite.campaign.sweep`."""

from ._module_deprecations import warn_module_deprecation
from .campaign.sweep import *  # noqa: F401,F403

warn_module_deprecation(__name__)
