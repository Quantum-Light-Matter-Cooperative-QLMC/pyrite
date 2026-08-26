"""Compatibility re-export for :mod:`pyrite.apps.viewer`."""

from ._module_deprecations import warn_module_deprecation
from .apps.viewer import *  # noqa: F401,F403

warn_module_deprecation(__name__)
