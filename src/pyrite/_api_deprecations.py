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

API_DEPRECATIONS: dict[str, ApiDeprecation] = {}
# #955: first announced in 0.27.0; never remove before replacements ship.
for _spelling in ("Result.spectral_flux", "Result.spectral_flux_units"):
    API_DEPRECATIONS[_spelling] = _api_entry(
        _spelling,
        "Result.accepted_spectra(assume_uniform=True) for scalar sources, "
        "or SpectralDensity.quantify(), followed by explicit rate(current=...), "
        "to_energy(), to_dataframe()/dataframe_units(); integrated_yield() for totals",
        since="0.27.0",
    )


for _spelling, _replacement in (
    ("Footprint.width", "Slab.width for slabs or Footprint.width_mm for stacks"),
    ("Footprint.height", "Slab.height for slabs or Footprint.height_mm for stacks"),
    ("Slab(footprint=None)", "Slab(width=None, height=None)"),
):
    API_DEPRECATIONS[_spelling] = _api_entry(_spelling, _replacement, since="0.27.0")


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
