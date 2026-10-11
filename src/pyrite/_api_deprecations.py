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
API_DEPRECATIONS: dict[str, ApiDeprecation] = {
    entry.spelling: entry
    for entry in (
        _api_entry(
            "Beam() without transverse=",
            "Beam(transverse=Spot(fwhm_x_mm=1.0)) or Beam(transverse=PointSpot())",
            since="0.12.0",
        ),
        _api_entry(
            "Beam() without longitudinal=",
            "Beam(longitudinal=PointBunch()) or Beam(longitudinal=BunchLength(rms_fs=...))",
            since="0.12.0",
        ),
        _api_entry(
            "Slab() without footprint=",
            "Slab(footprint=Footprint(5.0, 5.0)) or Slab(footprint=None) for an infinite slab",
            since="0.12.0",
        ),
        _api_entry(
            "Stack() without footprint=",
            "Stack(footprint=Footprint(5.0, 5.0)) or Stack(footprint=None) for infinite extent",
            since="0.12.0",
        ),
        _api_entry("pyrite.Point", "pyrite.PointSpot", since="0.12.0"),
        _api_entry("simulate(pixel_scorer=)", "PlanarDetector(scorer=...)", since="0.12.0"),
        _api_entry("Scene(pixel_scorer=)", "PlanarDetector(scorer=...)", since="0.12.0"),
        _api_entry(
            "Numerics(backend=)",
            "PYRITE_MC_BACKEND set before importing pyrite (or `pyrite config setup`)",
            since="0.12.0",
        ),
        _api_entry("pyrite.campaign.beam.Point", "pyrite.campaign.beam.PointSpot", since="0.12.0"),
    )
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
