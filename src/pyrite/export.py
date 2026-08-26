"""Compatibility re-export for :mod:`pyrite.apps.export`."""

from ._module_deprecations import warn_module_deprecation
from .apps.export import *  # noqa: F401,F403

warn_module_deprecation(__name__)
