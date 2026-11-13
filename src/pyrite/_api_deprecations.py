"""Deprecation warnings and registries for renamed or now-required Python API spellings."""

import warnings
from dataclasses import dataclass

from ._module_deprecations import _window


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
