"""Two-step simulate-then-score workflow (#1162).

``score_spectra(simulate_trajectories(...), detector)`` must equal one-shot
``pyrite.simulate`` bit for bit on one backend, never transport, and refuse any
request that would have changed the transport before spectrum work.
"""

import warnings

import numpy as np
import pytest

import pyrite
from pyrite import Physics, simulate
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.config import FixedSampling, KernelTuning, Numerics, TransportKernelTuning
from pyrite.detectors import Detector, EnergyBins, PlanarDetector, PlanarPose
from pyrite.geometry import BlazedGrooves, Footprint, Layer, Slab, Stack
from pyrite.montecarlo import runner
from pyrite.scoring import ScoringRefusedError
from pyrite.transport import Trajectories, simulate_trajectories

NUMERICS = Numerics(sampling=FixedSampling(cxr_histories=8, brem_histories=4), seed=3)
BEAM = Beam(30.0, transverse=Spot(), longitudinal=PointBunch())
FLAT = Slab("hopg", 1000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0))
GROOVED = Slab(
    "hopg",
    2.0e4,
    tilt_deg=45.0,
    tilt_azim_deg=180.0,
    entrance_face=BlazedGrooves(2.0e4),
    footprint=Footprint(5.0, 5.0),
)
GROOVED_PHYSICS = Physics(radiative_model="uncoupled", inelastic_model="continuous")
STACK = Stack((Layer("hopg", 500.0), Layer("silicon", 500.0)), 30.0, footprint=None)
AUTO_GRID = Slab("mose2", 1000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0))
FAMILIES = ("cxr.incoherent", "brem", "characteristic")


def _family(result, path):
    for part in path.split("."):
        result = getattr(result, part)
    return result


def _assert_identical(result, expected):
    for path in FAMILIES:
        actual, reference = _family(result, path), _family(expected, path)
        assert (actual is None) == (reference is None)
        if actual is not None:
            np.testing.assert_array_equal(actual.photon_energy_eV, reference.photon_energy_eV)
            np.testing.assert_array_equal(actual.density, reference.density)
    assert result.provenance["identity_digest"] == expected.provenance["identity_digest"]


@pytest.fixture(scope="module")
def flat():
    return simulate_trajectories(BEAM, FLAT, numerics=NUMERICS, detector=Detector())


@pytest.mark.parametrize(
    ("target", "physics"),
    [
        pytest.param(FLAT, None, id="flat"),
        pytest.param(GROOVED, GROOVED_PHYSICS, id="grooved"),
        pytest.param(STACK, None, id="multilayer"),
        pytest.param(AUTO_GRID, None, id="auto-line-grid"),
    ],
)
def test_two_step_reproduces_simulate(target, physics):
    trajectories = simulate_trajectories(
        BEAM, target, physics=physics, numerics=NUMERICS, detector=Detector()
    )
    assert isinstance(trajectories, Trajectories) and trajectories.scorable
    expected = simulate(BEAM, target, Detector(), physics=physics, numerics=NUMERICS)
    _assert_identical(pyrite.score_spectra(trajectories), expected)
    if target is AUTO_GRID:
        assert trajectories.case.get("line_grid_policy") is not None


def test_detector_free_transport_scores_any_detector(monkeypatch):
    trajectories = simulate_trajectories(BEAM, FLAT, numerics=NUMERICS)
    with pytest.raises(ValueError, match="detector"):
        pyrite.score_spectra(trajectories)
    planar = PlanarDetector(PlanarPose.from_observation(100.0, 60.0), size_mm=(2.0, 2.0))
    expected = [simulate(BEAM, FLAT, d, numerics=NUMERICS) for d in (Detector(), planar)]
    monkeypatch.setattr(runner, "_transport_case", lambda *a, **k: pytest.fail("transported"))
    for detector, reference in zip((Detector(), planar), expected, strict=True):
        _assert_identical(pyrite.score_spectra(trajectories, detector), reference)


def test_one_transport_scores_independent_detectors_and_grids(flat, monkeypatch):
    fingerprint = flat.fingerprint
    side = Detector(observation_angle_deg=60.0)
    grid = Numerics(
        sampling=NUMERICS.sampling,
        seed=NUMERICS.seed,
        energy_bins=EnergyBins(cxr=np.linspace(1000.0, 4000.0, 64)),
    )
    expected_side = simulate(BEAM, FLAT, side, numerics=NUMERICS)
    expected_grid = simulate(BEAM, FLAT, Detector(), numerics=grid)
    monkeypatch.setattr(runner, "_transport_case", lambda *a, **k: pytest.fail("transported"))
    _assert_identical(pyrite.score_spectra(flat, side), expected_side)
    _assert_identical(pyrite.score_spectra(flat, numerics=grid), expected_grid)
    assert flat.segment_count > 0
    assert segment_fingerprint_now(flat) == fingerprint


def segment_fingerprint_now(trajectories):
    from pyrite.campaign.trajectory_product import segment_fingerprint

    return segment_fingerprint(trajectories.segments)


def _no_spectra(monkeypatch):
    monkeypatch.setattr(
        runner, "_spectrum_with_recovery", lambda *a, **k: pytest.fail("scored before refusing")
    )


@pytest.mark.parametrize(
    ("numerics", "physics", "field"),
    [
        pytest.param(Numerics(sampling=NUMERICS.sampling, seed=4), None, "seed", id="seed"),
        pytest.param(
            Numerics(sampling=FixedSampling(cxr_histories=9, brem_histories=4), seed=3),
            None,
            "Ne",
            id="population",
        ),
        pytest.param(
            NUMERICS.with_kernel_tuning(
                KernelTuning(transport=TransportKernelTuning(electron_batch=64))
            ),
            None,
            "numerics.kernel_tuning.transport",
            id="transport-tuning",
        ),
        pytest.param(NUMERICS, Physics(straggling=True), "straggling", id="transport-physics"),
    ],
)
def test_transport_inputs_are_refused(flat, numerics, physics, field, monkeypatch):
    _no_spectra(monkeypatch)
    with pytest.raises(ScoringRefusedError, match=field) as error:
        pyrite.score_spectra(flat, numerics=numerics, physics=physics)
    assert error.value.field == field


@pytest.mark.parametrize("override", [{"E_cut_keV": 5.0}, {"max_steps": 400_000}])
def test_explicit_engine_controls_are_not_scorable(override, monkeypatch):
    trajectories = simulate_trajectories(
        BEAM, FLAT, numerics=NUMERICS, detector=Detector(), **override
    )
    assert not trajectories.scorable
    _no_spectra(monkeypatch)
    (name,) = override
    with pytest.raises(ScoringRefusedError, match=f"{name}=None") as error:
        pyrite.score_spectra(trajectories)
    assert error.value.field == name


def test_default_cutoffs_are_the_production_cutoffs():
    thick = Slab("hopg", 1.0e5, tilt_deg=30.0, footprint=Footprint(5.0, 5.0))
    calls = [
        simulate_trajectories(BEAM, thick, numerics=NUMERICS, detector=Detector(), **kw)
        for kw in ({}, {"E_cut_keV": 5.0}, {"E_cut_keV": 1.0})
    ]
    default, lines_cut, brem_cut = calls
    # 4 of 8 histories are shared with the continuum and stop at its 1 keV
    # cutoff; the other 4 stop at the 5 keV line cutoff.
    assert lines_cut.segment_count < default.segment_count < brem_cut.segment_count


def test_coherent_scoring_requires_the_capture_placement(flat, monkeypatch):
    _no_spectra(monkeypatch)
    with pytest.raises(ScoringRefusedError, match="capture detector") as error:
        pyrite.score_spectra(
            flat, Detector(observation_angle_deg=60.0), physics=Physics(emission="coherent")
        )
    assert error.value.field == "detector"


def test_grooved_targets_refuse_another_observation_angle(monkeypatch):
    trajectories = simulate_trajectories(
        BEAM, GROOVED, physics=GROOVED_PHYSICS, numerics=NUMERICS, detector=Detector()
    )
    _no_spectra(monkeypatch)
    with pytest.raises((ScoringRefusedError, ValueError), match="grooves|observation"):
        pyrite.score_spectra(trajectories, Detector(observation_angle_deg=60.0))


def test_type_and_mapping_compatibility(flat):
    with pytest.raises(TypeError, match="Trajectories"):
        pyrite.score_spectra({"segs": {}})
    with pytest.warns(DeprecationWarning, match="Trajectories.segments"):
        rows = flat["L_ang"]
    assert rows is flat.segments["L_ang"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert flat.segments["L_ang"] is rows
