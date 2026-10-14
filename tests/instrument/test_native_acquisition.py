import numpy as np
import pytest
from scipy.constants import elementary_charge

from pyrite.detectors import Detector, IdealPhotonCounter, LegacyEDS
from pyrite.instrument import Acquisition, NativeAcquisition, score_acquisition
from pyrite.instrument.acquisition import resolve_acquisition
from pyrite.instrument.observation import acquisition_payload


def test_native_request_is_separate_from_concrete_acquisition():
    request = Acquisition.native(exposure_s=2, mode="poisson", seed=7)
    assert isinstance(request, NativeAcquisition)
    assert not isinstance(request, Acquisition)
    with pytest.raises(TypeError, match="resolve native"):
        acquisition_payload(request, rep_rate_hz=1, bunch_charge_pc=1)
    with pytest.raises(TypeError):
        Acquisition.native(exposure_s=1, measured_edges_eV=(0, 1))
    with pytest.raises(ValueError, match="requires seed"):
        Acquisition.native(exposure_s=1, mode="poisson")
    with pytest.raises(ValueError, match="positive"):
        Acquisition.native(exposure_s=0)


def test_ideal_union_uses_all_source_midpoint_cells_and_preserves_mass():
    detector = Detector(response=IdealPhotonCounter())
    axes = [np.array([10.0, 12.0]), np.array([2.0, 6.0, 10.0, 14.0])]
    request = Acquisition.native(exposure_s=1, hit_threshold_eV=3)
    acquisition, policy = resolve_acquisition(request, detector, axes)
    reverse, _ = resolve_acquisition(request, detector, axes[::-1])
    assert acquisition == reverse
    assert acquisition.measured_edges_eV == (0, 4, 8, 9, 11, 12, 13, 16)
    assert policy == {"policy": "native", "source": "source-midpoint-cells"}
    for axis in axes:
        batch = score_acquisition(
            detector,
            axis,
            np.ones((1, axis.size)),
            acquisition,
            rep_rate_hz=1,
            bunch_charge_pc=elementary_charge * 1e12,
            observation_digest="a" * 64,
            coordinates=np.array([[0, 0]]),
            component="line",
        )
        native = detector.native_score(axis, np.ones(axis.size))
        np.testing.assert_allclose(
            batch.expected.sum() + batch.below_cut_expected.sum(), native.events.sum()
        )
        np.testing.assert_array_equal(batch.underflow_expected, [0])
        np.testing.assert_array_equal(batch.overflow_expected, [0])


def test_native_union_limit_accepts_boundary_and_rejects_overflow():
    detector = Detector(response=IdealPhotonCounter())
    request = Acquisition.native(exposure_s=1)
    acquisition, _ = resolve_acquisition(request, detector, [np.arange(65_536) + 0.5])
    assert len(acquisition.measured_edges_eV) == 65_537
    with pytest.raises(ValueError, match="65,536.*explicit measured_edges_eV"):
        resolve_acquisition(request, detector, [np.arange(65_537) + 0.5])


def test_native_policy_rejects_unsupported_response():
    with pytest.raises(NotImplementedError, match="does not expose native measured bins"):
        resolve_acquisition(
            Acquisition.native(exposure_s=1),
            Detector(response=LegacyEDS()),
            [np.array([1.0, 2.0])],
        )
