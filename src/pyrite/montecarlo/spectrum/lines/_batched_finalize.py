"""Compatibility re-export for :mod:`pyrite.montecarlo.spectrum.cxr._batched_finalize`."""

import sys
from importlib import import_module

from pyrite._module_deprecations import warn_module_deprecation

warn_module_deprecation(__name__)
sys.modules[__name__] = import_module("pyrite.montecarlo.spectrum.cxr._batched_finalize")
