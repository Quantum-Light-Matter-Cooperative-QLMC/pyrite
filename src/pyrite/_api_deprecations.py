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

#: Returned-dictionary keys cannot warn at runtime; these rows schedule their
#: removal and are documented in the owning docstring. They first shipped in
#: tagged 0.24.0 (no 0.23.1 was tagged).
_RESULT_KEY_DEPRECATED_IN = "0.24.0"

API_DEPRECATIONS = {
    spelling: _api_entry(spelling, replacement, since=_RESULT_KEY_DEPRECATED_IN)
    for spelling, replacement in (
        (
            "cxr_metrics()['peak_flux']",
            "cxr_metrics()['peak_spectral_flux_density']",
        ),
        (
            "cxr_metrics()['peak_flux_per_na']",
            "cxr_metrics()['peak_spectral_flux_density_per_na']",
        ),
    )
}
API_DEPRECATIONS["BunchLength"] = _api_entry(
    "BunchLength",
    "LongitudinalDistribution(kind='gaussian', ...) "
    "(shape='uniform' for a flat top; rms_duration= for a Pint time)",
    since="0.25.0",
)


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
