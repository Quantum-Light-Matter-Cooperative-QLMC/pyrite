"""Public KernelTuning transport schedules are exact on the CUDA core (#776).

Choosing continuation or changing its row quantum through
``KernelTuning.transport`` must return bitwise the full-history transport and
spectra at fixed physics and seed, through normal ``simulate``. These tests
need a CUDA device and run through ``pyrite remote``.

Validation: gpu-transport-core
"""

import numpy as np
import pytest

import pyrite as pr
import pyrite.config as cfg
from pyrite.campaign.lowering import build_case
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.transport_options import kernel_schedule

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device"),
]

SCHEDULES = (
    cfg.TransportKernelTuning("full-history"),
    cfg.TransportKernelTuning("continuation", 64),
    cfg.TransportKernelTuning("continuation", 256),
)


def _numerics(transport):
    tuning = cfg.KernelTuning(transport_core="cuda", transport=transport)
    return pr.Numerics(cxr_trials=256, brem_trials=128, seed=7, execution=tuning)


def _scene(transport):
    return pr.Scene(
        pr.Beam(energy_keV=60.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "hopg", thickness_ang=20_000.0, tilt_deg=30.0, footprint=pr.geometry.Footprint(5.0, 5.0)
        ),
        pr.Detector(),
        numerics=_numerics(transport),
    )


def _bitwise(expected, actual, key):
    expected, actual = np.asarray(expected), np.asarray(actual)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, key
    assert np.array_equal(
        np.ascontiguousarray(expected).view(np.uint8), np.ascontiguousarray(actual).view(np.uint8)
    ), key


def _host(value):
    return cupy.asnumpy(value) if isinstance(value, cupy.ndarray) else value


def test_simulate_spectra_identical_across_schedules_and_quanta():
    results = []
    for transport in SCHEDULES:
        scene = _scene(transport)
        results.append(
            pr.simulate(scene.beam, scene.target, scene.detector, numerics=scene.numerics)
        )
    reference = results[0]
    for result in results[1:]:
        assert result.case.to_dict() == reference.case.to_dict()
        for name in ("cxr", "brem"):
            ref = reference.cxr.incoherent if name == "cxr" else reference.brem
            got = result.cxr.incoherent if name == "cxr" else result.brem
            _bitwise(ref.photon_energy_eV, got.photon_energy_eV, f"{name} grid")
            _bitwise(ref.density, got.density, f"{name} density")


def test_transport_arrays_identical_across_schedules_and_quanta():
    case = build_case(_scene(SCHEDULES[0]))
    digests = []
    for transport in SCHEDULES:
        with kernel_schedule(_numerics(transport)):
            segs = runner._transport_case(case, transport_core="cuda")["segs"]
        digests.append(
            {
                key: _host(value)
                for key, value in segs.items()
                if isinstance(_host(value), np.ndarray) and _host(value).dtype != object
            }
        )
    reference = digests[0]
    assert reference
    for candidate in digests[1:]:
        assert candidate.keys() == reference.keys()
        for key in reference:
            _bitwise(reference[key], candidate[key], key)
