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
from pyrite.campaign.profiles import case_content_key
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.transport_options import transport_tuning

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
        assert case_content_key(result.case) == case_content_key(reference.case)
        for name in ("cxr", "brem"):
            ref = reference.cxr.incoherent if name == "cxr" else reference.brem
            got = result.cxr.incoherent if name == "cxr" else result.brem
            _bitwise(ref.photon_energy_eV, got.photon_energy_eV, f"{name} grid")
            _bitwise(ref.density, got.density, f"{name} density")


def test_transport_arrays_identical_across_schedules_and_quanta():
    case = build_case(_scene(SCHEDULES[0]))
    digests = []
    for transport in SCHEDULES:
        with transport_tuning(_numerics(transport)):
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


@pytest.mark.slow
def test_tune_kernels_accepts_exact_quanta_on_a_real_device():
    """End to end: fresh-process workers, equivalence guard, typed result."""
    pytest.importorskip("optuna")
    from pyrite.optimization import KernelLattice, tune_kernels

    scene = _scene(cfg.TransportKernelTuning())
    scene = pr.Scene(
        scene.beam,
        scene.target,
        scene.detector,
        numerics=scene.numerics.with_kernel_tuning(cfg.KernelTuning()),
    )
    lattice = KernelLattice(schedule=("continuation",), continuation_rows=(64, 256))
    result = tune_kernels(scene, lattice, trials=2, warmups=1, repeats=1, rtol=0.0, atol=0.0)
    assert result.environment["backend"] == "cuda"
    assert all(t.state == "complete" for t in result.trials), result.trials
    assert result.best.transport_core == "cuda"


def test_phases_reproduce_simulate_on_the_device():
    """Resident CUDA segments rescored by score_spectra equal simulate bit for bit."""
    from pyrite.phases import score_spectra, simulate_transport

    scene = _scene(cfg.TransportKernelTuning())
    expected = pr.simulate(scene)
    transported = simulate_transport(scene)
    for result in (score_spectra(transported), score_spectra(transported)):
        _bitwise(expected.cxr.incoherent.density, result.cxr.incoherent.density, "cxr")
        _bitwise(expected.brem.density, result.brem.density, "brem")


@pytest.mark.slow
def test_phased_tune_kernels_runs_both_studies_on_a_real_device():
    """One worker serves both studies; spectrum candidates never re-transport."""
    pytest.importorskip("optuna")
    from pyrite.optimization import KernelLattice, tune_kernels

    scene = _scene(cfg.TransportKernelTuning())
    lattice = KernelLattice(schedule=("full-history", "auto"), cxr_chunk=(None, 4096))
    result = tune_kernels(scene, lattice, warmups=1, repeats=1)
    assert [p.phase for p in result.phases] == ["transport", "spectrum"]
    for study in result.phases:
        assert all(t.state == "complete" for t in study.trials), study.trials
    assert result.best.transport_core == "cuda"
