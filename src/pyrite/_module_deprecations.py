"""Removal schedule for compatibility module import paths.

These modules remain behavior-preserving re-exports through the support window.
The registry gives each path an explicit canonical replacement and removal
target; ``tests/test_module_deprecations.py`` holds it to the live package tree
in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass

SUPPORT_WINDOW_MINORS = 2


@dataclass(frozen=True)
class ModuleDeprecation:
    """One compatibility module and its canonical import path."""

    module: str
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
    since: str = "0.2.0",
) -> ModuleDeprecation:
    return ModuleDeprecation(module, replacement, since, _window(since))


#: Keyed by the full compatibility module path used in an import.
MODULE_DEPRECATIONS: dict[str, ModuleDeprecation] = {
    entry.module: entry
    for entry in (
        _entry("pyrite.analyze", "pyrite.apps.analyze"),
        _entry("pyrite.archive", "pyrite.checkpoints.archive"),
        _entry("pyrite.beam_metrics", "pyrite.campaign.beam_metrics"),
        _entry("pyrite.blaze", "pyrite.runs.blaze"),
        _entry("pyrite.campaign_lock", "pyrite.checkpoints.campaign_lock"),
        _entry("pyrite.check", "pyrite.apps.check"),
        _entry("pyrite.check_config", "pyrite.validation.check_config"),
        _entry(
            "pyrite.checkpoint_cleanup",
            "pyrite.checkpoints.checkpoint_cleanup",
        ),
        _entry("pyrite.config", "pyrite.campaign.config"),
        _entry("pyrite.export", "pyrite.apps.export"),
        _entry("pyrite.longitudinal", "pyrite.campaign.longitudinal"),
        _entry(
            "pyrite.performance_analysis",
            "pyrite.perf.performance_analysis",
        ),
        _entry(
            "pyrite.performance_profile",
            "pyrite.perf.performance_profile",
        ),
        _entry("pyrite.plots.altair_detectors", "pyrite.plots.altair.detectors"),
        _entry("pyrite.plots.altair_spectra", "pyrite.plots.altair.spectra"),
        _entry("pyrite.plots.altair_sweeps", "pyrite.plots.altair.sweeps"),
        _entry(
            "pyrite.plots.altair_trajectories",
            "pyrite.plots.altair.trajectories",
        ),
        _entry(
            "pyrite.plots.crystal_lattice",
            "pyrite.plots.plotly.crystal_lattice",
        ),
        _entry("pyrite.plots.detectors", "pyrite.plots.mpl.detectors"),
        _entry("pyrite.plots.interactive", "pyrite.plots.mpl.interactive"),
        _entry(
            "pyrite.plots.plotly_trajectories",
            "pyrite.plots.plotly.trajectories",
        ),
        _entry(
            "pyrite.plots.render_trajectories",
            "pyrite.plots.plotly.render",
        ),
        _entry("pyrite.plots.spectra", "pyrite.plots.mpl.spectra"),
        _entry("pyrite.plots.sweeps", "pyrite.plots.mpl.sweeps"),
        _entry("pyrite.plots.trajectories", "pyrite.plots.mpl.trajectories"),
        _entry("pyrite.profiles", "pyrite.campaign.profiles"),
        _entry("pyrite.recompute", "pyrite.checkpoints.recompute"),
        _entry(
            "pyrite.recompute_defaults",
            "pyrite.checkpoints.recompute_defaults",
        ),
        _entry("pyrite.run", "pyrite.runs.run"),
        _entry("pyrite.scan", "pyrite.runs.scan"),
        _entry("pyrite.slim", "pyrite.checkpoints.slim"),
        _entry("pyrite.sweep", "pyrite.campaign.sweep"),
        _entry("pyrite.transverse", "pyrite.campaign.transverse"),
        _entry(
            "pyrite.validation_background",
            "pyrite.validation.validation_background",
        ),
        _entry(
            "pyrite.validation_oracles",
            "pyrite.validation.validation_oracles",
        ),
        _entry("pyrite.viewer", "pyrite.apps.viewer"),
    )
}
