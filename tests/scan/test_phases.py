"""Separate transport and spectrum phases of a scalar simulation (#776)."""

from dataclasses import replace

import numpy as np
import pytest

import pyrite.config as cfg
from pyrite import Scene, api, simulate
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.config import Numerics
from pyrite.detectors import Detector, PlanarDetector, PlanarPose
from pyrite.geometry import Footprint, Slab
from pyrite.montecarlo import runner
from pyrite.phases import SimulationProgress, TransportedScene, score_spectra, simulate_transport


@pytest.fixture(scope="module")
def scene():
    return Scene(
        Beam(30.0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 1000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0)),
        Detector(),
        numerics=Numerics(cxr_trials=8, brem_trials=4, seed=3),
    )


@pytest.fixture(scope="module")
def transported(scene):
    return simulate_transport(scene)


def _assert_same(result, expected, rtol=0.0):
    for path in ("cxr.incoherent", "brem", "characteristic"):
        actual, reference = (_family(r, path) for r in (result, expected))
        assert (actual is None) == (reference is None)
        if actual is not None:
            np.testing.assert_array_equal(actual.photon_energy_eV, reference.photon_energy_eV)
            np.testing.assert_allclose(actual.density, reference.density, rtol=rtol, atol=0)


def _family(result, path):
    for part in path.split("."):
        result = getattr(result, part)
    return result


def test_phases_reproduce_simulate(scene, transported):
    assert isinstance(transported, TransportedScene)
    assert transported.segment_count > 0
    expected = simulate(scene)
    first = score_spectra(transported)
    _assert_same(first, expected)
    assert first.provenance["identity_digest"] == expected.provenance["identity_digest"]
    _assert_same(score_spectra(transported), first)  # scoring twice reuses the transport


def test_spectral_chunks_rescore_without_transport(transported, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("score_spectra transported again")

    monkeypatch.setattr(runner, "_transport_case", forbidden)
    tuning = cfg.KernelTuning(cxr_chunk=3, brem_chunk=5)
    result = score_spectra(transported, kernel_tuning=tuning)
    assert result.provenance["numerics"].kernel_tuning == tuning
    assert result.case == transported.case
    # Chunking changes only summation order: a few ulp of the backend precision.
    eps = np.finfo(result.cxr.incoherent.density.dtype).eps
    _assert_same(result, score_spectra(transported), rtol=100 * eps)


def test_transport_settings_are_fixed_by_the_transport(transported):
    changed = cfg.KernelTuning(transport=cfg.TransportKernelTuning(electron_batch=64))
    with pytest.raises(ValueError, match="only the spectral chunks and CXR launch"):
        score_spectra(transported, kernel_tuning=changed)
    with pytest.raises(TypeError, match="simulate_transport"):
        score_spectra(object())


def test_progress_reports_transport_batches_and_spectrum_stages(scene):
    events = []
    result = simulate(scene, progress=events.append)
    assert all(isinstance(event, SimulationProgress) for event in events)
    transport = [e for e in events if e.phase == "transport"]
    spectrum = [e for e in events if e.phase == "spectrum"]
    assert transport and transport[-1].done == transport[-1].total == 8
    assert [e.stage for e in spectrum] == list(runner.SPECTRUM_STAGES)
    assert spectrum[-1].fraction == 1.0
    _assert_same(result, simulate(scene))  # progress never changes results
    assert runner._TRANSPORT_PROGRESS.get() is None
    assert runner._SPECTRUM_PROGRESS.get() is None

    phased = []
    score_spectra(simulate_transport(scene, progress=phased.append), progress=phased.append)
    assert [e.phase for e in phased] == [e.phase for e in events]


def test_pixel_detectors_are_refused(scene, monkeypatch):
    monkeypatch.setattr(api, "_source_case", lambda *a: pytest.fail("lowered before refusing"))
    pixel = replace(
        scene, detector=PlanarDetector(PlanarPose.from_observation(100.0, 60.0), size_mm=(2.0, 2.0))
    )
    with pytest.raises(ValueError, match="scalar detectors"):
        simulate_transport(pixel)
