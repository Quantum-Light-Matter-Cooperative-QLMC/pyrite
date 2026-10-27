"""Real CUDA production selection, source components, persistence and replay.

Run with PYRITE_MC_BACKEND=cuda, PYRITE_TEST_BACKEND=cuda and --slow remotely.
Canonical physical tables are required. Public API tests never replace primary
transport or conditional samplers; one pair-feedback closure test controls only
the first-flight rate to exercise a rare branch. CPU processes skip before runs.
"""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.montecarlo import runner
from pyrite.results.aperture import aperture_from_payload

pytestmark = pytest.mark.slow


@pytest.fixture
def cuda(monkeypatch):
    from pyrite._backend import BACKEND

    if BACKEND.name != "cuda":
        pytest.skip("production test requires PYRITE_MC_BACKEND=cuda before import")
    # conftest isolates ordinary tests from user workspaces. This explicit
    # hardware-only run reads the synchronized canonical tables; all actual
    # checkpoint/capture writes still receive tmp_path destinations below.
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PYRITE_HOME", str(root))
    monkeypatch.setenv("PYRITE_CATALOG", str(root / "src/pyrite/data/catalog"))
    return BACKEND.xp


def _scene(component="brem", feedback=False):
    event = component != "brem"
    physics = pr.Physics(
        elastic_model="elsepa",
        inelastic_model="shell-soft-hard" if event or feedback else "continuous",
        inelastic_cutoff_eV=50.0 if event or feedback else None,
        atomic_electron_deflection="none",
        secondary_threshold_eV=1000 if event else None,
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        pair_production_model="penelope-2024" if component == "annihilation" else None,
        positron_transport=component == "annihilation",
        photon_transport=pr.PhotonTransport(
            (1000.0, 3000.0, 10000.0, 20000.0),
            source=component,
            n_samples=None if event else 32,
            source_ceiling_eV=20000,
            charged_feedback=feedback,
            seed=341,
        ),
    )
    return pr.Scene(
        pr.Beam(energy_keV=40, transverse=pr.PointSpot(), longitudinal=pr.PointBunch()),
        pr.Slab("silicon", thickness_ang=1000, tilt_deg=5, footprint=None),
        pr.PlanarDetector(pr.PlanarPose((0, 0, 1)), size_mm=(2, 2)),
        physics=physics,
        numerics=pr.Numerics(
            cxr_trials=4,
            brem_trials=4,
            transport_core="cuda",
            energy_bins=pr.EnergyBins(
                cxr=np.linspace(1500, 2000, 8), brem=np.arange(1000, 40001, 1000)
            ),
        ),
    )


@pytest.mark.parametrize("component", ["brem", "characteristic", "annihilation"])
@pytest.mark.parametrize("feedback", [False, True])
def test_public_cuda_selector_uses_one_primary_pass_and_returns_aperture(
    cuda, monkeypatch, component, feedback
):
    scene = _scene(component, feedback)
    calls = []
    original = runner._transport_case

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_transport_case", capture)
    result = api.simulate(
        scene.beam, scene.target, scene.detector, physics=scene.physics, numerics=scene.numerics
    )
    assert calls == [1]
    tally = result.aperture
    assert tally.n_primaries == 4
    assert tally.charged_feedback is feedback
    assert tally.provenance["execution"]["backend"] == "cuda"
    assert result.case["photon_transport"]["model"] == "ordinary-hit-photon-cuda.v1"
    assert (
        "observation.photon/cuda-reference"
        in result.provenance["resolved_physics"]["descriptor_ids"]
    )
    assert np.isfinite(tally.total_per_primary_eV).all()
    assert np.all(tally.total_per_primary_eV >= 0)
    assert result.aperture_dataframe().height == 3
    if component == "brem":
        assert tally.n_proposals == 32
        assert tally.detected_per_primary > 0
    elif component == "annihilation":
        # The primary 40 keV beam cannot create pairs: the physical zero-source
        # limit must remain an empty, correctly normalized aperture observation.
        assert tally.n_proposals == 0
        assert tally.detected_per_primary == 0


def test_cuda_aperture_replay_checkpoint_cas_and_cpu_portability(cuda, monkeypatch, tmp_path):
    from pyrite.checkpoints import _checkpoint_store as store
    from pyrite.checkpoints.persistence import _cas_payload_from_record
    from pyrite.montecarlo.runner.artifacts import spectrum_from_artifact
    from pyrite.montecarlo.trajectories import TrajectoryCapture, read_trajectory_artifact
    from pyrite.results.store import store_result

    scene = _scene()
    case = api.build_case(scene)
    capture = TrajectoryCapture(str(tmp_path))
    output = runner.run_case(case, transport_core="cuda", trajectory_capture=capture)
    tally = aperture_from_payload(output["photon_aperture"])

    def forbidden(*args, **kwargs):
        pytest.fail("artifact replay reran primary transport")

    monkeypatch.setattr(runner, "_transport_case", forbidden)
    replay = spectrum_from_artifact(read_trajectory_artifact(capture.path_for(case)))
    other = aperture_from_payload(replay["photon_aperture"])
    np.testing.assert_array_equal(other.total_per_primary_eV, tally.total_per_primary_eV)
    assert other.observation_identity_digest == tally.observation_identity_digest
    records = {}
    store_result(records, case, output)
    store.save("silicon", tmp_path, records)
    assert (tmp_path / "silicon" / "photon.h5").is_file()
    loaded = store.load("silicon", tmp_path)[case.name][case.E0_keV]
    payload = _cas_payload_from_record(loaded)["photon_aperture"]
    np.testing.assert_array_equal(
        aperture_from_payload(payload).total_per_primary_eV, tally.total_per_primary_eV
    )
    portable = tmp_path / "cuda-aperture.npz"
    np.savez(portable, metadata=np.array(json.dumps(payload["metadata"])), **payload["arrays"])
    script = """
import json, sys, numpy as np
from pyrite import _backend, datasets
assert _backend.BACKEND.name == 'cpu'
def forbidden(*args, **kwargs):
    raise AssertionError('finished observation reader requested hardware/scientific tables')
_backend.select_backend = datasets.require_dataset = forbidden
from pyrite.results.aperture import aperture_from_payload
with np.load(sys.argv[1]) as archive:
    value = aperture_from_payload(dict(metadata=json.loads(str(archive['metadata'])),
        arrays={k:archive[k] for k in archive.files if k != 'metadata'}))
assert value.provenance['execution']['backend'] == 'cuda'
print(value.observation_identity_digest)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(portable)],
        env={**os.environ, "PYRITE_MC_BACKEND": "cpu"},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == tally.observation_identity_digest


def test_cuda_mode_off_arrays_are_bitwise_stable_around_selected_run(cuda):
    scene = _scene()
    off = api.build_case(replace(scene, physics=replace(scene.physics, photon_transport=None)))
    before = runner.run_case(off, transport_core="cuda")
    expected = {}
    for key, value in before.items():
        if isinstance(value, (np.ndarray, cuda.ndarray)):
            host = cuda.asnumpy(value) if isinstance(value, cuda.ndarray) else value
            expected[key] = (host.dtype, host.shape, host.tobytes())
    selected = runner.run_case(api.build_case(scene), transport_core="cuda")
    assert selected["photon_aperture"]["metadata"]["provenance"]["execution"]["backend"] == "cuda"
    after = runner.run_case(off, transport_core="cuda")
    assert "photon_aperture" not in before and "photon_aperture" not in after
    for key, (dtype, shape, contents) in expected.items():
        host = cuda.asnumpy(after[key]) if isinstance(after[key], cuda.ndarray) else after[key]
        assert dtype == host.dtype and shape == host.shape
        assert contents == host.tobytes(), key


def test_cuda_pair_electron_positron_feedback_closes_rest_reservoir(cuda, monkeypatch):
    from pyrite.montecarlo.runner.photon_aperture import prepare_photon_case
    from pyrite.montecarlo.transport import photon_navigation as nav
    from pyrite.montecarlo.transport.pair_production import PAIR_THRESHOLD_EV
    from pyrite.montecarlo.transport.photon_coupling import (
        coupled_ledger_residual_eV,
        track_coupled_histories,
    )

    scene = _scene(feedback=True)
    scene = replace(scene, target=pr.Slab("silicon", thickness_ang=1e5, tilt_deg=5, footprint=None))
    context = prepare_photon_case(api.build_case(scene))
    assert context["stage"].base.transport["transport_core"] == "cuda"

    def pair_once(element, energy):
        energy = np.asarray(energy)
        return {
            name: np.where(energy > PAIR_THRESHOLD_EV, 1000.0, 0.0)
            if name == "pair_nuclear"
            else np.zeros_like(energy)
            for name in nav.PHOTON_CHANNEL_ORDER
        }

    # All pair final states, charged launches, hard photons, shell/Auger feedback
    # and annihilation remain real. Below-threshold descendants escape so the
    # parent rest-energy release can be checked independently of reabsorption.
    monkeypatch.setattr(nav, "photon_cross_sections_ang2", pair_once)
    n = 2
    tracked = track_coupled_histories(
        np.tile([0, 0, 5e4], (n, 1)),
        np.tile([0, 0, 1], (n, 1)),
        np.full(n, 1.03e6),
        np.ones(n),
        np.arange(n),
        np.zeros(n, np.int64),
        context["layers"],
        charged_stage=context["stage"],
        seed=341_907,
        photon_cutoff_eV=1000,
        backend="cuda",
    )
    assert tracked.rounds > 0
    np.testing.assert_allclose(tracked.ledger["positron_rest_eV"], 0, atol=1e-8)
    assert np.count_nonzero(tracked.exits["energy_eV"] > 4e5) >= 2 * n
    assert np.all(
        np.abs(coupled_ledger_residual_eV(tracked.ledger)) <= 1e-11 * tracked.ledger["incident_eV"]
    )
