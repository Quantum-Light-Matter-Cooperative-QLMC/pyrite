"""Real-device production and charged scene continuation; run remotely with --slow."""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.photon_aperture import prepare_photon_case
from pyrite.montecarlo.transport.photon_reentry import scene_feedback_stage
from tests.montecarlo.test_photon_coupling import silicon_stage as silicon_stage
from tests.montecarlo.test_photon_feedback import _closed, _rows
from tests.montecarlo.test_photon_production_cuda import _scene
from tests.montecarlo.test_photon_production_cuda import cuda as cuda

pytestmark = pytest.mark.slow


def _filtered_scene(component="brem", feedback=True, material="silicon"):
    scene = _scene(component, feedback)
    return replace(
        scene,
        filters=(pr.FilterPlate(material, 0.0005, (1.0, 1.0), pr.PlanarPose((0.0, 0.0, 0.2))),),
    )


@pytest.mark.parametrize("material", ["silicon", "tungsten"])
@pytest.mark.parametrize(
    "component,feedback",
    [("brem", True), ("characteristic", True), ("annihilation", True), ("characteristic", False)],
)
def test_public_cuda_filter_feedback_is_explicit_and_one_primary_pass(
    cuda, monkeypatch, component, feedback, material
):
    from pyrite.montecarlo.transport import photon_feedback as feedback_module
    from pyrite.montecarlo.transport import photon_reentry as module

    scene = _filtered_scene(component, feedback, material)
    calls, clips, launches = [], [], []
    original = runner._transport_case
    clip = module._layer_intervals
    launch = feedback_module._launch_cascade

    def transport(*args, **kw):
        calls.append(1)
        return original(*args, **kw)

    def device_clip(*args, **kw):
        assert kw["xp"] is cuda
        clips.append(1)
        return clip(*args, **kw)

    def charged(base, *args, **kw):
        assert base.transport["transport_core"] == "cuda"
        launches.append(1)
        return launch(base, *args, **kw)

    monkeypatch.setattr(runner, "_transport_case", transport)
    monkeypatch.setattr(module, "_layer_intervals", device_clip)
    monkeypatch.setattr(feedback_module, "_launch_cascade", charged)
    result = api.simulate(scene)
    assert calls == [1]
    tally = result.aperture
    info = tally.provenance["charged_feedback" if feedback else "relaxation_feedback"]
    assert info["model"] == "cuda-charged-scene-feedback-v1"
    assert info["box_materials"] == [material]
    assert info["tables"] and info["positron_tables"]
    assert info["navigation"] == "analytic-binary64-device"
    assert info["charged_transport_core"] == "cuda"
    assert np.isfinite(tally.total_per_primary_eV).all()
    assert (tally.total_per_primary_eV >= 0).all()
    assert tally.n_primaries == 4
    if feedback:
        assert clips  # includes an empty zero-source queue; no CPU geometry fallback


@pytest.mark.parametrize("species", [0, 1])
def test_real_cuda_charged_gap_filter_matches_cpu(cuda, silicon_stage, species):
    from pyrite.montecarlo.transport.photon_navigation import PhotonBox

    _, base = silicon_stage
    transport = {
        k: v
        for k, v in base.transport.items()
        if k not in ("layers", "thickness_ang", "crystal_width_mm", "crystal_height_mm")
    }
    for name in ("stopping_tables", "inelastic_materials"):
        transport[name] = transport[name] * 2
    comp = base.transport["layers"][0][2]
    layers = [(0.0, 1.0, comp)]
    box = PhotonBox(
        (0.0, 0.0, 3e7), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (1e8, 1e8, 1e7), comp
    )
    host = scene_feedback_stage(layers, boxes=[box], photon_cutoff_eV=1000.0, **transport)
    device = scene_feedback_stage(
        layers, boxes=[box], photon_cutoff_eV=1000.0, backend="cuda", **transport
    )
    rows = _rows([3e5], species, z=0.5)
    expected, actual = _closed(host, rows), _closed(device, rows)
    for name in ("escaped_eV", "deposited_eV", "binding_eV", "radiated_eV", "rest_energy_eV"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=2e-10, atol=1e-6
        )
    for table in ("photons", "escaped_particles"):
        for name, values in getattr(expected, table).items():
            other = getattr(actual, table)[name]
            if np.issubdtype(values.dtype, np.integer):
                np.testing.assert_array_equal(other, values)
            else:
                np.testing.assert_allclose(other, values, rtol=2e-10, atol=1e-5)
    assert actual.deposited_eV[0] > 100.0


def test_cuda_scene_boundaries_match_cpu_and_empty_batch_is_explicit(cuda):
    scene = _filtered_scene()
    context = prepare_photon_case(api.build_case(scene))
    stage = context["stage"]
    host = replace(stage, backend="cpu")
    rows = _rows(np.full(7, 3e5), z=0.0)
    rows["position_ang"][:, 2] = [-1.0, 0.0, 500.0, 1000.0, 2e5, 2e6, 1e8]
    for directions in ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]):
        rows["direction"][:] = directions
        for a, b in zip(stage._next(rows), host._next(rows), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-8)
    rows = {k: v[:0] for k, v in rows.items()}
    assert stage._next(rows)[0].size == 0
    assert stage.contains(rows).size == 0


def test_cuda_filter_capture_replay_checkpoint_cas_cpu_read(cuda, monkeypatch, tmp_path):
    import json
    import os
    import subprocess
    import sys

    from pyrite.checkpoints import _checkpoint_store as store
    from pyrite.checkpoints.persistence import _cas_payload_from_record
    from pyrite.montecarlo.runner.artifacts import spectrum_from_artifact
    from pyrite.montecarlo.trajectories import TrajectoryCapture, read_trajectory_artifact
    from pyrite.results.aperture import aperture_from_payload
    from pyrite.results.store import store_result

    case = api.build_case(_filtered_scene())
    capture = TrajectoryCapture(str(tmp_path))
    output = runner.run_case(case, transport_core="cuda", trajectory_capture=capture)
    tally = aperture_from_payload(output["photon_aperture"])

    def forbidden(*args, **kw):
        pytest.fail("replay requested primary transport")

    monkeypatch.setattr(runner, "_transport_case", forbidden)
    replay = spectrum_from_artifact(read_trajectory_artifact(capture.path_for(case)))
    other = aperture_from_payload(replay["photon_aperture"])
    np.testing.assert_array_equal(other.total_per_primary_eV, tally.total_per_primary_eV)
    assert other.observation_identity_digest == tally.observation_identity_digest
    records = {}
    store_result(records, case, output)
    store.save("silicon", tmp_path, records)
    loaded = store.load("silicon", tmp_path)[case.name][case.E0_keV]
    payload = _cas_payload_from_record(loaded)["photon_aperture"]
    assert (
        aperture_from_payload(payload).provenance["charged_feedback"]
        == tally.provenance["charged_feedback"]
    )
    portable = tmp_path / "reentry.npz"
    np.savez(portable, metadata=np.array(json.dumps(payload["metadata"])), **payload["arrays"])
    script = """
import json, sys, numpy as np
from pyrite import _backend, datasets
assert _backend.BACKEND.name == 'cpu'
def forbidden(*a, **kw):
    raise AssertionError('CPU reader requested accelerator or scientific tables')
_backend.select_backend = datasets.require_dataset = forbidden
from pyrite.results.aperture import aperture_from_payload
with np.load(sys.argv[1]) as z:
    tally=aperture_from_payload(dict(metadata=json.loads(str(z['metadata'])),
        arrays={k:z[k] for k in z.files if k != 'metadata'}))
assert tally.provenance['charged_feedback']['model']=='cuda-charged-scene-feedback-v1'
print(tally.observation_identity_digest)
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


def test_cuda_reentry_mode_off_arrays_are_bitwise_unchanged(cuda):
    scene = _filtered_scene()
    off = api.build_case(replace(scene, physics=replace(scene.physics, photon_transport=None)))
    before = runner.run_case(off, transport_core="cuda")
    expected = {
        k: (
            v.dtype,
            v.shape,
            cuda.asnumpy(v).tobytes() if isinstance(v, cuda.ndarray) else v.tobytes(),
        )
        for k, v in before.items()
        if isinstance(v, (np.ndarray, cuda.ndarray))
    }
    runner.run_case(api.build_case(scene), transport_core="cuda")
    after = runner.run_case(off, transport_core="cuda")
    for key, (dtype, shape, data) in expected.items():
        value = after[key]
        host = cuda.asnumpy(value) if isinstance(value, cuda.ndarray) else value
        assert dtype == host.dtype and shape == host.shape and data == host.tobytes(), key
