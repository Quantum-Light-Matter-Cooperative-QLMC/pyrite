"""Compatibility re-export for :mod:`pyrite.montecarlo.spectrum.cxr`."""

import sys
from importlib import import_module

from pyrite._module_deprecations import warn_module_deprecation

warn_module_deprecation(__name__)
_canonical = import_module("pyrite.montecarlo.spectrum.cxr")
# A forwarding package's __path__ points at the canonical directory. Register
# its children as aliases too, preventing duplicate modules and divergent
# monkeypatch/cache state when a caller imports an old child path.
for _child in (
    "_attribution",
    "_batched",
    "_batched_finalize",
    "_bin_quadrature",
    "_bin_tree",
    "_formation",
    "_kernels",
    "_per_hkl",
    "_policy",
    "_setup",
    "_spectrum",
    "_temporal",
):
    sys.modules[f"{__name__}.{_child}"] = import_module(f"{_canonical.__name__}.{_child}")
sys.modules[__name__] = import_module("pyrite.montecarlo.spectrum.cxr")
