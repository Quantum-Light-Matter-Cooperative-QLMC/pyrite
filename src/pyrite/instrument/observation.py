"""Frozen pixel-observation configuration and layered identities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ..detectors import IdealPhotonCounter, Timepix3
from ..materials import CATALOG, MediumSpec
from .model import (
    FilterPlate,
    PixelScorer,
    PlanarDetector,
    validate_downstream_scene,
)

AcquisitionMode = Literal["expected", "poisson"]
REALIZATION_RNG = "pyrite.coordinate-philox.v1"


def _finite_real(name: str, value: object, *, minimum: float, open_minimum: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    invalid_minimum = number <= minimum if open_minimum else number < minimum
    if not math.isfinite(number) or invalid_minimum:
        relation = "positive" if minimum == 0.0 and open_minimum else f">= {minimum:g}"
        raise ValueError(f"{name} must be finite and {relation}")
    return number


@dataclass(frozen=True)
class Acquisition:
    """Exposure, reporting-axis, threshold, and realization configuration.

    Reporting bins are half-open ``[edge_i, edge_{i+1})``. Underflow,
    overflow, and events below ``hit_threshold_eV`` are separate accounting in
    the acquisition core; this record does not conflate those concepts with a
    detector response's physical discriminator or energy resolution.
    """

    exposure_s: float
    measured_edges_eV: tuple[float, ...]
    hit_threshold_eV: float = 0.0
    mode: AcquisitionMode = "expected"
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exposure_s",
            _finite_real("Acquisition.exposure_s", self.exposure_s, minimum=0.0, open_minimum=True),
        )
        try:
            raw_edges = tuple(self.measured_edges_eV)
        except TypeError as exc:
            raise TypeError("Acquisition.measured_edges_eV must be an iterable") from exc
        if len(raw_edges) < 2:
            raise ValueError("Acquisition.measured_edges_eV requires at least two edges")
        edges = tuple(
            _finite_real(
                "Acquisition.measured_edges_eV",
                edge,
                minimum=0.0,
                open_minimum=False,
            )
            for edge in raw_edges
        )
        if any(right <= left for left, right in zip(edges, edges[1:], strict=False)):
            raise ValueError("Acquisition.measured_edges_eV must be strictly increasing")
        object.__setattr__(self, "measured_edges_eV", edges)
        object.__setattr__(
            self,
            "hit_threshold_eV",
            _finite_real(
                "Acquisition.hit_threshold_eV",
                self.hit_threshold_eV,
                minimum=0.0,
                open_minimum=False,
            ),
        )
        if self.mode not in ("expected", "poisson"):
            raise ValueError("Acquisition.mode must be 'expected' or 'poisson'")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0
        ):
            raise ValueError("Acquisition.seed must be a nonnegative integer or None")
        if self.mode == "poisson" and self.seed is None:
            raise ValueError("Acquisition poisson mode requires seed")
        if self.mode == "expected" and self.seed is not None:
            raise ValueError("Acquisition expected mode does not accept seed")
        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))

    @classmethod
    def uniform(
        cls,
        *,
        exposure_s: float,
        minimum_eV: float,
        maximum_eV: float,
        bin_width_eV: float,
        hit_threshold_eV: float = 0.0,
        mode: AcquisitionMode = "expected",
        seed: int | None = None,
    ) -> Acquisition:
        """Construct explicit uniform reporting edges from bounds and width."""
        minimum = _finite_real("minimum_eV", minimum_eV, minimum=0.0, open_minimum=False)
        maximum = _finite_real("maximum_eV", maximum_eV, minimum=0.0, open_minimum=True)
        width = _finite_real("bin_width_eV", bin_width_eV, minimum=0.0, open_minimum=True)
        if maximum <= minimum:
            raise ValueError("maximum_eV must be greater than minimum_eV")
        quotient = (maximum - minimum) / width
        bins = round(quotient)
        if bins <= 0 or not math.isclose(quotient, bins, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("measured energy range must be an integer multiple of bin_width_eV")
        edges = tuple(minimum + index * width for index in range(bins + 1))
        edges = (*edges[:-1], maximum)
        return cls(
            exposure_s=exposure_s,
            measured_edges_eV=edges,
            hit_threshold_eV=hit_threshold_eV,
            mode=mode,
            seed=seed,
        )


@dataclass(frozen=True)
class ResolvedObservation:
    """Fully lowered physical pixel observation, separate from source transport."""

    detector: PlanarDetector
    scorer: PixelScorer
    filters: tuple[FilterPlate, ...]
    acquisition: Acquisition

    def __post_init__(self) -> None:
        if not isinstance(self.detector, PlanarDetector):
            raise TypeError("ResolvedObservation.detector must be a PlanarDetector")
        if self.detector.pixels is None:
            raise ValueError("ResolvedObservation requires detector pixels")
        if self.detector.response is None:
            raise ValueError("ResolvedObservation requires an explicit detector response")
        if not isinstance(self.scorer, PixelScorer):
            raise TypeError("ResolvedObservation.scorer must be a PixelScorer")
        if any(
            requested > available
            for requested, available in zip(
                self.scorer.angular_shape,
                self.detector.pixels.shape,
                strict=True,
            )
        ):
            raise ValueError("observation angular shape cannot exceed detector pixel shape")
        try:
            resolved_filters = tuple(self.filters)
        except TypeError as exc:
            raise TypeError("ResolvedObservation.filters must be an iterable") from exc
        if any(not isinstance(plate, FilterPlate) for plate in resolved_filters):
            raise TypeError("ResolvedObservation.filters must contain FilterPlate objects")
        object.__setattr__(self, "filters", resolved_filters)
        if not isinstance(self.acquisition, Acquisition):
            raise TypeError("ResolvedObservation.acquisition must be an Acquisition")
        validate_downstream_scene(resolved_filters, self.detector)

    def scalar_detector(self):
        """Return the unchanged legacy source-case detector projection."""
        return self.detector.scalar_detector()


@dataclass(frozen=True)
class ObservationIdentity:
    """Layered SHA-256 identities for one fully produced observation."""

    true_spatial_digest: str
    response_digest: str
    acquisition_digest: str
    observation_digest: str
    payload: Mapping[str, Any]


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot canonicalize observation value {type(value).__name__}")


def _array_identity(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolved_composition(material: str | MediumSpec) -> tuple[tuple[str, float], ...]:
    if isinstance(material, MediumSpec):
        return material.composition
    if material in CATALOG.media:
        return CATALOG.media[material].composition
    return CATALOG.crystals[material].composition


def _response_semantics(response: object) -> str:
    if isinstance(response, Timepix3):
        return "clustered-photon-event-at-incident-ray-pixel"
    if isinstance(response, IdealPhotonCounter):
        return "ideal-energy-preserving-photon-event"
    return "response-defined"


def observation_identity(
    source_identity_digest: str,
    observation: ResolvedObservation,
    *,
    rep_rate_hz: float,
    bunch_charge_pc: float,
    representative_directions_lab: np.ndarray,
    attenuation_arrays: tuple[np.ndarray, ...],
) -> ObservationIdentity:
    """Build independent true-spatial, response, acquisition, and full digests."""
    if (
        not isinstance(source_identity_digest, str)
        or len(source_identity_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_identity_digest)
    ):
        raise ValueError("source_identity_digest must be a lowercase SHA-256 digest")
    detector = observation.detector
    true_payload = {
        "schema": "pyrite.true-spatial.v1",
        "source_identity_digest": source_identity_digest,
        "detector_geometry": {
            "pose": _canonical_value(detector.pose),
            "size_mm": list(detector.size_mm or ()),
            "pixels": _canonical_value(detector.pixels),
        },
        "filters": [
            {
                "composition": _canonical_value(_resolved_composition(plate.material)),
                "thickness_mm": plate.thickness_mm,
                "size_mm": list(plate.size_mm),
                "pose": _canonical_value(plate.pose),
            }
            for plate in observation.filters
        ],
        "scorer": _canonical_value(observation.scorer),
        "representative_directions_lab": _array_identity(representative_directions_lab),
        "attenuation_arrays": [_array_identity(array) for array in attenuation_arrays],
        "approximations": {
            "emission_source": "point-source-target-reference",
            "pixel_sampling": "centre-ray",
            "filter_interactions": "primary-attenuation-only",
        },
    }
    response = detector.response
    assert response is not None
    response_payload = {
        "schema": "pyrite.response.v1",
        "type": f"{type(response).__module__}.{type(response).__qualname__}",
        "config": _canonical_value(response),
        "event_semantics": _response_semantics(response),
        "spatial_calibration": (
            "uniform-uncalibrated" if isinstance(response, Timepix3) else "uniform-ideal"
        ),
    }
    acquisition_payload = {
        "schema": "pyrite.acquisition.v1",
        "config": _canonical_value(observation.acquisition),
        "bin_semantics": "half-open",
        "event_accounting": ["underflow", "overflow", "below-cut", "registered"],
        "normalization": {
            "rep_rate_hz": _finite_real(
                "rep_rate_hz", rep_rate_hz, minimum=0.0, open_minimum=False
            ),
            "bunch_charge_pc": _finite_real(
                "bunch_charge_pc", bunch_charge_pc, minimum=0.0, open_minimum=False
            ),
        },
        "realization_rng": (REALIZATION_RNG if observation.acquisition.mode == "poisson" else None),
    }
    true_digest = _digest(true_payload)
    response_digest = _digest(response_payload)
    acquisition_digest = _digest(acquisition_payload)
    links = {
        "schema": "pyrite.observation.v2",
        "true_spatial_digest": true_digest,
        "response_digest": response_digest,
        "acquisition_digest": acquisition_digest,
    }
    observation_digest = _digest(links)
    payload = MappingProxyType(
        {
            **links,
            "observation_identity_digest": observation_digest,
            "true_spatial": true_payload,
            "response": response_payload,
            "acquisition": acquisition_payload,
        }
    )
    return ObservationIdentity(
        true_spatial_digest=true_digest,
        response_digest=response_digest,
        acquisition_digest=acquisition_digest,
        observation_digest=observation_digest,
        payload=payload,
    )


__all__ = [
    "Acquisition",
    "AcquisitionMode",
    "ObservationIdentity",
    "REALIZATION_RNG",
    "ResolvedObservation",
    "observation_identity",
]
