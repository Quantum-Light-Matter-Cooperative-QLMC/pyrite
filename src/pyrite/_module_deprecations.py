"""Removal schedule for compatibility module import paths.

These modules remain behavior-preserving re-exports through the support window.
The registry gives each path an explicit canonical replacement and removal
target; ``tests/test_module_deprecations.py`` holds it to the live package tree
in both directions and to the shipping ``__version__``.

Each shim's module body calls `warn_module_deprecation`, so the removal target
below is a promise that was actually announced to importers rather than a
private note. The paths were carved out in 0.2.0 but stayed silent until 0.3.0,
so 0.3.0 is the release the two-minor window counts from -- an import path that
never warned has not spent its window (issue #68).
"""

import warnings
from dataclasses import dataclass

SUPPORT_WINDOW_MINORS = 2


@dataclass(frozen=True)
class ModuleDeprecation:
    """One compatibility module and its canonical import path."""

    module: str
    replacement: str
    deprecated_in: str
    remove_in: str


@dataclass(frozen=True)
class PublicExportDeprecation:
    """One deprecated package attribute and its canonical replacement."""

    module: str
    name: str
    replacement: str
    deprecated_in: str
    remove_in: str


def _window(deprecated_in: str, minors: int = SUPPORT_WINDOW_MINORS) -> str:
    major, minor, *_ = deprecated_in.split(".")
    return f"{major}.{int(minor) + minors}.0"


def _entry(
    module: str,
    replacement: str,
    *,
    since: str = "0.3.0",
) -> ModuleDeprecation:
    return ModuleDeprecation(module, replacement, since, _window(since))


def _public_export_entry(
    module: str,
    name: str,
    replacement: str,
    *,
    since: str = "0.3.0",
) -> PublicExportDeprecation:
    return PublicExportDeprecation(module, name, replacement, since, _window(since))


#: Keyed by the full compatibility module path used in an import.
#:
#: Empty since 0.5.0, when the flat re-export cohort was removed.
MODULE_DEPRECATIONS: dict[str, ModuleDeprecation] = {}


#: Keyed by the compatibility package and exported attribute.
#:
#: Empty since 0.5.0, when the relocated detector-response exports were removed.
PUBLIC_EXPORT_DEPRECATIONS: dict[tuple[str, str], PublicExportDeprecation] = {}


def warn_module_deprecation(module: str) -> None:
    """Announce *module* as a compatibility path and name its replacement.

    Called from the shim's own module body rather than from a ``__getattr__``,
    because ``import pyrite.slim`` binds a submodule through the import system
    and never reaches the parent package's attribute hook -- a lazy warning
    would stay silent for exactly the import the deprecation is about.

    Module bodies execute once per interpreter, so this fires exactly once per
    path with no separate guard; a caller that deliberately evicts the module
    from ``sys.modules`` and re-imports it is warned again, which is correct.
    """
    entry = MODULE_DEPRECATIONS.get(module)
    if entry is None:
        return

    warnings.warn(
        f"{entry.module} is deprecated and will be removed in {entry.remove_in}; "
        f"import {entry.replacement} instead",
        DeprecationWarning,
        stacklevel=2,
    )


def warn_public_export_deprecation(module: str, name: str) -> None:
    """Announce a relocated package attribute when its compatibility shim is used."""
    entry = PUBLIC_EXPORT_DEPRECATIONS.get((module, name))
    if entry is None:
        return
    warnings.warn(
        f"{entry.module}.{entry.name} is deprecated and will be removed in {entry.remove_in}; "
        f"import {entry.replacement} instead",
        DeprecationWarning,
        stacklevel=3,
    )
