"""Resident CUDA segments for a coupled-radiative MeV scene (#905).

``simulate_transport`` keeps a ``bremslib-soft-hard`` scene's segment rows on
the device; hard-photon completion downloads only the photon rows. The rows
and the scored ``Result`` must equal the host-segment path bit for bit. These
tests need a CUDA device and run through ``pyrite remote``.

Validation: gpu-transport-core
"""

import numpy as np
import pytest

from pyrite import Scene
from pyrite.beams import Beam, PointBunch, PointSpot
from pyrite.config import (
    CxrReflections,
    ExecutionConfig,
    FixedSampling,
    Numerics,
    TransportKernelTuning,
)
from pyrite.detectors import Detector, EnergyBins
from pyrite.geometry import Footprint, Slab
from pyrite.montecarlo import runner
from pyrite.phases import score_spectra, simulate_transport
from pyrite.units import ureg

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device"),
]


def _scene(electrons=512):
    """The #905 benchmark scene (W 1 MeV / 10 um) at a test electron count."""
    numerics = Numerics(
        sampling=FixedSampling(cxr_electrons=electrons, brem_electrons=electrons // 4),
        seed=7,
        energy_bins=EnergyBins().with_cxr_energies(np.linspace(750, 1_000, 51) * ureg.eV),
        cxr_reflections=CxrReflections(n_families=1),
    ).with_kernel_tuning(ExecutionConfig(transport=TransportKernelTuning(kernel_threads=32)))
    return Scene(
        Beam(kinetic_energy=1 * ureg.MeV, transverse=PointSpot(), longitudinal=PointBunch()),
        Slab(
            "tungsten",
            thickness=10 * ureg.um,
            tilt=45 * ureg.degree,
            footprint=Footprint(width=5 * ureg.mm, height=5 * ureg.mm),
        ),
        Detector(observation_angle=90 * ureg.degree),
        numerics=numerics,
    )


def _host(value):
    return cupy.asnumpy(value) if isinstance(value, cupy.ndarray) else value


def _assert_identical(reference, candidate, path=""):
    assert reference.keys() == candidate.keys(), path
    for key, expected in reference.items():
        name = f"{path}.{key}"
        expected, actual = _host(expected), _host(candidate[key])
        if isinstance(expected, np.ndarray):
            assert actual.dtype == expected.dtype, name
            assert actual.shape == expected.shape, name
            assert np.array_equal(
                np.ascontiguousarray(expected).view(np.uint8),
                np.ascontiguousarray(actual).view(np.uint8),
            ), name
        elif isinstance(expected, dict):
            _assert_identical(expected, actual, name)
        elif isinstance(expected, (list, tuple)) and any(
            isinstance(item, (np.ndarray, dict)) for item in expected
        ):
            assert len(actual) == len(expected), name
            _assert_identical(dict(enumerate(expected)), dict(enumerate(actual)), name)
        else:
            assert actual == expected, name


@pytest.fixture(scope="module")
def transports():
    scene = _scene()
    resident = simulate_transport(scene)
    original = runner._transport_case

    def host_transport(case, **kwargs):
        return original(case, **{**kwargs, "keep_segments_on_device": False})

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "_transport_case", host_transport)
        host = simulate_transport(scene)
    return resident, host


def test_radiative_segments_stay_resident(transports):
    resident, host = transports
    segs = resident._transport["segs"]
    assert segs["radiative"]["model"] == "bremslib-soft-hard"
    for key in ("r_mid", "E_end_keV", "hard_radiative_k_eV", "hard_radiative_direction"):
        assert isinstance(segs[key], cupy.ndarray), key
    assert isinstance(host._transport["segs"]["r_mid"], np.ndarray)
    assert np.count_nonzero(cupy.asnumpy(segs["hard_radiative_k_eV"]) > 0.0) > 0


def test_resident_rows_equal_host_rows(transports):
    resident, host = transports
    _assert_identical(host._transport["segs"], resident._transport["segs"], "segs")


def test_resident_result_equals_host_result(transports):
    resident, host = transports
    expected, actual = score_spectra(host), score_spectra(resident)
    assert actual.provenance["identity_digest"] == expected.provenance["identity_digest"]
    for family in ("brem", "characteristic"):
        reference, candidate = getattr(expected, family), getattr(actual, family)
        assert (reference is None) == (candidate is None), family
        if reference is not None:
            np.testing.assert_array_equal(candidate.density, reference.density, err_msg=family)
    reference, candidate = expected.cxr.incoherent, actual.cxr.incoherent
    np.testing.assert_array_equal(candidate.photon_energy_eV, reference.photon_energy_eV)
    np.testing.assert_array_equal(candidate.density, reference.density)
