"""Compatibility re-export for :mod:`pyrite.validation.validation_oracles`."""

from ._module_deprecations import warn_module_deprecation
from .validation.validation_oracles import *  # noqa: F401,F403

warn_module_deprecation(__name__)
