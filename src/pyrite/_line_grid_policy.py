"""Compatibility re-export for :mod:`pyrite._cxr_grid_policy`."""

import sys
from importlib import import_module

from pyrite._module_deprecations import warn_module_deprecation

warn_module_deprecation(__name__)
sys.modules[__name__] = import_module("pyrite._cxr_grid_policy")
