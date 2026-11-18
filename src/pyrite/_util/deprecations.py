"""Removal schedules for deprecated module import paths and Python API spellings.

Module paths
------------

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
#: CXR vocabulary shims introduced in 0.11.0 were removed in 0.13.0.
#: The older flat re-export cohort was removed in 0.5.0.
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


# -- Python API spellings ---------------------------------------------------


@dataclass(frozen=True)
class ApiDeprecation:
    """One deprecated Python API spelling and its canonical replacement."""

    spelling: str
    replacement: str
    deprecated_in: str
    remove_in: str


def _api_entry(spelling: str, replacement: str, *, since: str) -> ApiDeprecation:
    return ApiDeprecation(spelling, replacement, since, _window(since))


#: Keyed by the deprecated spelling as a user writes it (issue #451). Each row is
#: held to the shipping version by ``tests/test_deprecation_schedule.py``.
# First eligible patch after tagged 0.20.4. The release PR must confirm the
# announcement version (and recalculate the window if bundled in a later minor).
ROOT_API_REPLACEMENTS = {}

API_DEPRECATIONS: dict[str, ApiDeprecation] = {
    # #1162: the typed transport product and stable two-step scoring replace the
    # untyped dict and the experimental #776 phase split.
    "simulate_trajectories()[...]": _api_entry(
        "simulate_trajectories()[...]", "Trajectories.segments[...]", since="0.35.0"
    ),
    "pyrite.phases.simulate_transport": _api_entry(
        "pyrite.phases.simulate_transport",
        "pyrite.transport.simulate_trajectories",
        since="0.35.0",
    ),
    "pyrite.phases.score_spectra": _api_entry(
        "pyrite.phases.score_spectra", "pyrite.score_spectra", since="0.35.0"
    ),
    "pyrite.phases.TransportedScene": _api_entry(
        "pyrite.phases.TransportedScene", "pyrite.transport.Trajectories", since="0.35.0"
    ),
    # #1242: "none" turns every optional non-boolean setting off (Physics
    # fields, profile/preset TOML and CLI values); pyrite._off_settings warns.
    '"off"': _api_entry('"off"', '"none"', since="0.36.0"),
    # #1253: adaptive decisions fall on geometric decision points; transport
    # chunks are sized automatically and never change the stop.
    "AdaptiveSampling(block_histories=...)": _api_entry(
        "AdaptiveSampling(block_histories=...)",
        "automatic transport chunks (omit block_histories)",
        since="0.36.0",
    ),
    "AdaptiveSampling(stability_blocks=...)": _api_entry(
        "AdaptiveSampling(stability_blocks=...)",
        "AdaptiveSampling(stability_checks=...)",
        since="0.36.0",
    ),
}


def warn_api(spelling: str, *, stacklevel: int = 3) -> None:
    """Warn that registered *spelling* is deprecated, naming its replacement."""
    entry = API_DEPRECATIONS[spelling]
    # An omitted argument becomes an error at the target; a renamed name is removed.
    outcome = "an error" if " without " in entry.spelling else "removed"
    warnings.warn(
        f"{entry.spelling} is deprecated and will be {outcome} in {entry.remove_in}; "
        f"use {entry.replacement} instead",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
