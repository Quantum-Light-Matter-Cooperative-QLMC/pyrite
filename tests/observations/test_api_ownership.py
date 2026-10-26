"""Contract and compatibility coverage for issue #703 ownership slice."""

from __future__ import annotations

import pickle
from dataclasses import asdict, fields

from pyrite.config import AngularSampling, PixelScorer
from pyrite.detectors import (
    AngularDetector,
    Detector,
    EagleXO,
    EagleXOResponse,
    IdealPhotonCounter,
    NativeBinnedResponse,
    PixelGrid,
    PlanarDetector,
    PlanarPose,
    Timepix3,
    Timepix3Response,
)
from pyrite.geometry import FilterPlate
from pyrite.instrument.acquisition import AcquisitionBatch as OriginalAcquisitionBatch
from pyrite.instrument.model import FilterPlate as OriginalFilterPlate
from pyrite.observations import AcquisitionBatch as LegacyAcquisitionBatch
from pyrite.observations import Observation
from pyrite.observations.spec import bind_observation
from pyrite.results import AcquisitionBatch


def test_semantic_names_are_original_classes_not_new_models() -> None:
    from pyrite.instrument import PixelScorer as OriginalPixelScorer
    from pyrite.observations import PixelScorer as LegacyPixelScorer
    from pyrite.detectors import FilterPlate as LegacyFilterPlate

    assert AngularDetector is Detector
    assert AngularSampling is PixelScorer is OriginalPixelScorer is LegacyPixelScorer
    assert Timepix3Response is Timepix3
    assert EagleXOResponse is EagleXO
    assert FilterPlate is LegacyFilterPlate is OriginalFilterPlate
    assert AcquisitionBatch is LegacyAcquisitionBatch is OriginalAcquisitionBatch
    assert pickle.loads(pickle.dumps(AngularSampling)) is OriginalPixelScorer
    assert pickle.loads(pickle.dumps(Timepix3Response)) is Timepix3


def test_native_binned_response_capability_is_optional() -> None:
    assert callable(getattr(IdealPhotonCounter(), "native_score", None))
    assert callable(getattr(Timepix3Response(), "native_score", None))
    assert not callable(getattr(EagleXOResponse(), "native_score", None))
    assert callable(NativeBinnedResponse.native_score)


def test_same_planar_geometry_supports_independent_sampling_and_response() -> None:
    pixels = PixelGrid(shape=(4, 4), pitch_mm=(0.055, 0.055))
    detector = PlanarDetector(PlanarPose((0, 0, 100.0)), pixels=pixels)
    assert "response" not in {field.name for field in fields(detector)}
    assert "scorer" not in {field.name for field in fields(detector)}
    first, _, first_request = bind_observation(
        Observation(sampling=AngularSampling((2, 2)), response=IdealPhotonCounter()),
        detector,
        None,
    )
    second, _, second_request = bind_observation(
        Observation(sampling=AngularSampling((1, 4)), response=None),
        detector,
        None,
    )
    assert asdict(first) == asdict(second) == asdict(detector)
    assert detector.scorer == AngularSampling()
    assert first.scorer == first_request.sampling == AngularSampling((2, 2))
    assert second.scorer == second_request.sampling == AngularSampling((1, 4))
    assert first.response == IdealPhotonCounter()
    assert second.response is None
    assert detector.response is None
