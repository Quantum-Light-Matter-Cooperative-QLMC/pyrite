"""Compatibility re-export for :mod:`pyrite.cli.commands.check_config`."""

from ._module_deprecations import warn_module_deprecation
from .cli.commands.check_config import *  # noqa: F401,F403

warn_module_deprecation(__name__)
