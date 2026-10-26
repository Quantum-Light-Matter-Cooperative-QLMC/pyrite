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
ROOT_API_DEPRECATED_IN = "0.20.5"
ROOT_API_REPLACEMENTS = {
    "pyrite.Acquisition": "pyrite.observations.Acquisition",
    "pyrite.AdaptiveSampling": "pyrite.config.AdaptiveSampling",
    "pyrite.AngularAcceptance": "pyrite.detectors.AngularAcceptance",
    "pyrite.BlazedGrooves": "pyrite.geometry.BlazedGrooves",
    "pyrite.BunchFrontTilt": "pyrite.beams.BunchFrontTilt",
    "pyrite.BunchLength": "pyrite.beams.BunchLength",
    "pyrite.CrystalCut": "pyrite.geometry.CrystalCut",
    "pyrite.CxrReflections": "pyrite.config.CxrReflections",
    "pyrite.EnergyBins": "pyrite.detectors.EnergyBins",
    "pyrite.Filter": "pyrite.geometry.Filter",
    "pyrite.FilterPlate": "pyrite.geometry.FilterPlate",
    "pyrite.FixedSampling": "pyrite.config.FixedSampling",
    "pyrite.Footprint": "pyrite.geometry.Footprint",
    "pyrite.GdfSource": "pyrite.beams.GdfSource",
    "pyrite.IdealPhotonCounter": "pyrite.detectors.IdealPhotonCounter",
    "pyrite.Layer": "pyrite.geometry.Layer",
    "pyrite.NativeAcquisition": "pyrite.observations.NativeAcquisition",
    "pyrite.Offsets": "pyrite.beams.Offsets",
    "pyrite.PixelGrid": "pyrite.detectors.PixelGrid",
    "pyrite.PixelScorer": "pyrite.config.PixelScorer",
    "pyrite.PlanarDetector": "pyrite.detectors.PlanarDetector",
    "pyrite.PlanarPose": "pyrite.detectors.PlanarPose",
    "pyrite.PointBunch": "pyrite.beams.PointBunch",
    "pyrite.PointSpot": "pyrite.beams.PointSpot",
    "pyrite.Precision": "pyrite.config.Precision",
    "pyrite.RadiationNumerics": "pyrite.config.RadiationNumerics",
    "pyrite.RectangularBody": "pyrite.geometry.RectangularBody",
    "pyrite.ResolvedObservation": "pyrite.observations.ResolvedObservation",
    "pyrite.SensorAcceptance": "pyrite.detectors.SensorAcceptance",
    "pyrite.Slab": "pyrite.geometry.Slab",
    "pyrite.Spot": "pyrite.beams.Spot",
    "pyrite.Stack": "pyrite.geometry.Stack",
    "pyrite.SubstanceRef": "pyrite.geometry.SubstanceRef",
    "pyrite.TransportNumerics": "pyrite.config.TransportNumerics",
    "pyrite.TransverseSpot": "pyrite.beams.TransverseSpot",
    "pyrite.directional_source": "pyrite.radiation.directional_source",
    "pyrite.directional_sources": "pyrite.radiation.directional_sources",
    "pyrite.load_numerics": "pyrite.config.load_numerics",
    "pyrite.load_physics": "pyrite.config.load_physics",
}

API_DEPRECATIONS: dict[str, ApiDeprecation] = {
    spelling: _api_entry(spelling, replacement, since=ROOT_API_DEPRECATED_IN)
    for spelling, replacement in ROOT_API_REPLACEMENTS.items()
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
