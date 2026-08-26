"""Compatibility re-export for :mod:`pyrite.campaign.longitudinal`."""

from ._module_deprecations import warn_module_deprecation
from .campaign.longitudinal import *  # noqa: F401,F403

warn_module_deprecation(__name__)
