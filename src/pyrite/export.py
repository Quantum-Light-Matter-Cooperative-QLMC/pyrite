"""Compatibility re-export for :mod:`pyrite.cli.commands.export`."""

from ._module_deprecations import warn_module_deprecation
from .cli.commands.export import *  # noqa: F401,F403

warn_module_deprecation(__name__)
