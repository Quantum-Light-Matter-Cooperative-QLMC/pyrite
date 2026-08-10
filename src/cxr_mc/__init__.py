"""Compatibility namespace for the former :mod:`cxr_mc` package.

The implementation moved to :mod:`pyrite`.  This bootstrap maps legacy root
and deep imports to the canonical module objects so old imports and pickle
module paths remain readable without executing implementation modules twice.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import CodeType, ModuleType

import pyrite as _canonical_package

_LEGACY_PREFIX = f"{__name__}."
_CANONICAL_PREFIX = f"{_canonical_package.__name__}."


class _LegacyModuleLoader(importlib.abc.Loader):
    def __init__(self, canonical_name: str) -> None:
        self.canonical_name = canonical_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        canonical = __import__(self.canonical_name, fromlist=["*"])
        sys.modules[module.__name__] = canonical

    def get_code(self, fullname: str) -> CodeType:
        """Return a trampoline for legacy ``python -m`` invocations."""
        source = (
            "import runpy\n"
            f"runpy.run_module({self.canonical_name!r}, run_name='__main__', alter_sys=True)\n"
        )
        return compile(source, f"<{fullname} compatibility trampoline>", "exec")


class _LegacyModuleFinder(importlib.abc.MetaPathFinder):
    """Map any import below ``cxr_mc`` to its ``pyrite`` owner."""

    _pyrite_legacy_namespace_finder = True

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_LEGACY_PREFIX):
            return None
        canonical_name = _CANONICAL_PREFIX + fullname.removeprefix(_LEGACY_PREFIX)
        canonical_spec = importlib.util.find_spec(canonical_name)
        if canonical_spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _LegacyModuleLoader(canonical_name),
            is_package=canonical_spec.submodule_search_locations is not None,
        )


if not any(getattr(finder, "_pyrite_legacy_namespace_finder", False) for finder in sys.meta_path):
    sys.meta_path.insert(0, _LegacyModuleFinder())

sys.modules[__name__] = _canonical_package
