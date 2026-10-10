"""Compatibility re-export for :mod:`pyrite.cli.commands._profile_cxr_grid`."""

import sys
from importlib import import_module

from pyrite._module_deprecations import warn_module_deprecation

warn_module_deprecation(__name__)
sys.modules[__name__] = import_module("pyrite.cli.commands._profile_cxr_grid")
