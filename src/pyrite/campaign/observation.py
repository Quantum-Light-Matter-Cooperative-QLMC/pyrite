"""Lower validated profile observation tables into frozen domain objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ..detectors import EnergyBins, IdealPhotonCounter, Timepix3
from ..instrument import (
    Acquisition,
    FilterPlate,
    PixelGrid,
    PixelScorer,
    PlanarDetector,
    PlanarPose,
    ResolvedObservation,
)
from ..materials import MaterialCatalog


def filter_from_config(row: Mapping[str, object]) -> FilterPlate:
    """Lower one validated ordered-filter row."""
    return FilterPlate(
        material=cast(str, row["material"]),
        thickness_mm=cast(float, row["thickness_mm"]),
        size_mm=cast(tuple[float, float], row["size_mm"]),
        pose=PlanarPose.from_observation(
            cast(float, row["distance_mm"]),
            cast(float, row.get("polar_deg", 90.0)),
            cast(float, row.get("azimuth_deg", 0.0)),
            cast(float, row.get("roll_deg", 0.0)),
            cast(tuple[float, float], row.get("offset_mm", (0.0, 0.0))),
        ),
        name=cast(str | None, row.get("name")),
    )


def _response_from_config(row: Mapping[str, object] | None):
    if row is None:
        return None
    if row.get("kind") == "ideal":
        return IdealPhotonCounter()
    if row.get("kind") != "timepix3":
        raise ValueError("observation response kind must be 'ideal' or 'timepix3'")
    values = {key: value for key, value in row.items() if key != "kind"}
    return Timepix3(**cast(dict[str, Any], values))


def physical_detector_from_config(
    row: Mapping[str, object],
    *,
    energy_bins: EnergyBins | None = None,
) -> PlanarDetector:
    """Lower validated physical geometry and its explicit response."""
    pose = PlanarPose.from_observation(
        cast(float, row["distance_mm"]),
        cast(float, row.get("polar_deg", 90.0)),
        cast(float, row.get("azimuth_deg", 0.0)),
        cast(float, row.get("roll_deg", 0.0)),
        cast(tuple[float, float], row.get("offset_mm", (0.0, 0.0))),
    )
    response = cast(Mapping[str, object] | None, row.get("response"))
    return PlanarDetector(
        pose=pose,
        pixels=PixelGrid(
            shape=cast(tuple[int, int], row.get("shape", (256, 256))),
            pitch_mm=cast(tuple[float, float], row.get("pitch_mm", (0.055, 0.055))),
        ),
        energy_bins=EnergyBins() if energy_bins is None else energy_bins,
        response=_response_from_config(response),
    )


def resolve_profile_observation(
    catalog: MaterialCatalog,
    profile_name: str,
    *,
    energy_bins: EnergyBins | None = None,
    detector_id: str | None = None,
) -> ResolvedObservation | None:
    """Resolve a complete counting observation, or ``None`` for legacy profiles.

    A geometry-only ``physical_detector`` remains the existing material-smoke
    configuration and does not silently acquire exposure/counting semantics.
    Physical geometry and ordered filters belong to the selected profile.
    """
    if profile_name not in catalog.profile_names:
        raise KeyError(f"unknown profile {profile_name!r}; have {list(catalog.profile_names)}")
    detectors = catalog.profile_detector_set(profile_name)
    if detector_id is None:
        detector_id = next(iter(detectors))
    try:
        row = detectors[detector_id]
    except KeyError:
        raise ValueError(
            f"unknown detector {detector_id!r} in profile {profile_name!r}; have {list(detectors)}"
        ) from None
    if row is None or "acquisition" not in row:
        return None
    detector = physical_detector_from_config(row, energy_bins=energy_bins)
    if detector.response is None:
        detector = PlanarDetector(
            pose=detector.pose,
            pixels=detector.pixels,
            energy_bins=detector.energy_bins,
            response=IdealPhotonCounter(),
        )
    scorer_row = cast(Mapping[str, object], row.get("scorer", {}))
    scorer = PixelScorer(
        angular_shape=cast(tuple[int, int], scorer_row.get("angular_shape", (1, 1))),
        reconstruction=cast(Any, scorer_row.get("reconstruction", "nearest_tile")),
    )
    acquisition_row = cast(Mapping[str, object], row["acquisition"])
    acquisition = Acquisition(**cast(dict[str, Any], dict(acquisition_row)))
    filters = tuple(
        filter_from_config(filter_row)
        for filter_row in catalog.profile_filters.get(profile_name, ())
    )
    return ResolvedObservation(
        detector=detector,
        scorer=scorer,
        filters=filters,
        acquisition=acquisition,
    )


__all__ = [
    "filter_from_config",
    "physical_detector_from_config",
    "resolve_profile_observation",
]
