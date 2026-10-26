"""User observation requests, independent of physical scene definitions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .._spec_defaults import UNSET, _Unset
from ..detectors.spec import Detector, DetectorResponse
from ..instrument.model import PixelScorer, PlanarDetector
from ..instrument.observation import Acquisition, NativeAcquisition, response_payload


@dataclass(frozen=True)
class Observation:
    """Select a detector, response, acquisition and numerical sampling policy.

    Parameters
    ----------
    detector
        ``"detector"`` references the scene's single detector. A planar
        detector value must match its registered physical geometry. A scalar
        :class:`~pyrite.Detector` requests a virtual angular observation.
    response
        Read-time response implementing ``score``. Omit to inherit a legacy
        detector response; ``None`` requests response-free output.
    acquisition
        Exposure, reporting bins and optional count realization. Omit to
        inherit legacy Scene acquisition; ``None`` requests spectra only.
    sampling
        Independent ``pyrite.config.AngularSampling`` numerical policy
        for a physical pixel detector. Omit to inherit legacy scoring;
        ``None`` selects its default policy.
    name
        Optional nonempty display/routing label, excluded from identity.

    Notes
    -----
    Differing explicit legacy and observation settings are rejected. Reuse a
    detector without embedded response/scoring settings for independent
    observation configurations. Scalar spectra remain response-free; response
    is applied explicitly when reading results. No transport runs at construction.
    """

    detector: str | Detector | PlanarDetector = "detector"
    response: DetectorResponse | None | _Unset = UNSET
    acquisition: Acquisition | NativeAcquisition | None | _Unset = UNSET
    sampling: PixelScorer | None | _Unset = UNSET
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.detector, (str, Detector, PlanarDetector)):
            raise TypeError("Observation.detector must be a detector reference or detector value")
        if isinstance(self.detector, str) and not self.detector.strip():
            raise ValueError("Observation.detector reference must be nonempty")
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError("Observation.name must be a nonempty string or None")
        if self.response is not UNSET and self.response is not None:
            if not callable(getattr(self.response, "score", None)):
                raise TypeError("Observation.response must implement score()")
        if self.acquisition is not UNSET and self.acquisition is not None:
            if not isinstance(self.acquisition, (Acquisition, NativeAcquisition)):
                raise TypeError("Observation.acquisition must be an Acquisition or native request")
        if self.sampling is not UNSET and self.sampling is not None:
            if not isinstance(self.sampling, PixelScorer):
                raise TypeError("Observation.sampling must be a PixelScorer or None")


def _same_response(first: object, second: object) -> bool:
    if first is second:
        return True
    if first is None or second is None:
        return False
    try:
        return response_payload(first) == response_payload(second)
    except TypeError:
        # Nonserializable custom responses can still be shared by object reference.
        return False


def _explicit_scorer(detector: PlanarDetector) -> bool:
    origin = getattr(detector, "_scorer_origin", None)
    return detector.scorer is not None if origin is None else origin[0]


def _check_direct_detector_settings(
    observation: Observation, selected: PlanarDetector, registered: PlanarDetector
) -> None:
    """Never silently drop legacy settings carried by a convenience geometry value."""
    response = registered.response if observation.response is UNSET else observation.response
    if selected.response is not None and not _same_response(selected.response, response):
        raise ValueError("Observation.detector.response conflicts with Observation.response")
    if _explicit_scorer(selected):
        sampling = registered.scorer if observation.sampling is UNSET else observation.sampling
        if selected.scorer != (sampling or PixelScorer()):
            raise ValueError("Observation.detector.scorer conflicts with Observation.sampling")


def _observation_detector(
    observation: Observation, registered: Detector | PlanarDetector | None
) -> Detector | PlanarDetector:
    selected = observation.detector
    if isinstance(selected, str):
        if selected != "detector":
            raise ValueError(f"unknown Observation.detector reference {selected!r}; use 'detector'")
        if registered is None:
            raise ValueError("Observation.detector 'detector' has no registered Scene.detector")
        return registered
    if isinstance(selected, Detector):
        return selected
    if not isinstance(registered, PlanarDetector) or (
        selected.pose != registered.pose
        or selected.size_mm != registered.size_mm
        or selected.pixels != registered.pixels
    ):
        raise ValueError("Observation.detector geometry conflicts with Scene.detector")
    _check_direct_detector_settings(observation, selected, registered)
    # Only the registered body is authoritative; object identity is irrelevant.
    return registered


def bind_observation(
    observation: Observation,
    detector: Detector | PlanarDetector | None,
    acquisition: Acquisition | NativeAcquisition | None,
) -> tuple[Detector | PlanarDetector, Acquisition | NativeAcquisition | None, Observation]:
    """Bind a request to existing geometry and build a legacy execution view."""
    selected = _observation_detector(observation, detector)
    response = selected.response if observation.response is UNSET else observation.response
    if selected.response is not None and not _same_response(selected.response, response):
        raise ValueError("Observation.response conflicts with legacy detector.response")
    requested_acquisition = (
        acquisition if observation.acquisition is UNSET else observation.acquisition
    )
    if acquisition is not None and acquisition != requested_acquisition:
        raise ValueError("Observation.acquisition conflicts with legacy Scene.acquisition")
    if isinstance(selected, PlanarDetector):
        if requested_acquisition is not None:
            if selected.pixels is None:
                raise ValueError("Observation.acquisition requires a physical pixel detector")
            if response is None or not callable(getattr(response, "native_score", None)):
                raise ValueError(
                    "Observation.acquisition requires a native_score detector response"
                )
        sampling = selected.scorer if observation.sampling is UNSET else observation.sampling
        bound = replace(selected, response=response, scorer=sampling)
        if _explicit_scorer(selected):
            if selected.scorer != bound.scorer:
                raise ValueError("Observation.sampling conflicts with legacy detector.scorer")
        sampling = bound.scorer
        reference: str | Detector = "detector"
    else:
        if observation.sampling is not UNSET and observation.sampling is not None:
            raise ValueError("Observation.sampling requires a physical pixel detector")
        if requested_acquisition is not None:
            raise ValueError("Observation.acquisition requires a physical pixel detector")
        sampling = None
        bound = replace(selected, response=response)
        reference = replace(selected, response=None)
    normalized = Observation(
        detector=reference,
        response=response,
        acquisition=requested_acquisition,
        sampling=sampling,
        name=observation.name,
    )
    return bound, requested_acquisition, normalized
