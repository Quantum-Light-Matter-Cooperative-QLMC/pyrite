"""The sweep coordinator validates remote references and owns all manifests."""

from hashlib import sha256

import numpy as np
import pytest

from pyrite.execution import LocalExecutor
from pyrite.execution.artifacts import ArtifactIntegrityError, CaseArtifactStore
from pyrite.materials import CATALOG
from pyrite.runs.run import run_sweep
from tests.helpers import fake_out


def _case(name="distributed"):
    return dict(
        name=name, crystal="hopg", composition=CATALOG.crystal("hopg").composition,
        hkl_list=[], B_ang2=0.8, E0_keV=30.0, thickness_ang=1e4,
        E_grid=(100.0, 200.0, 10.0), E_grid_line=(100.0, 200.0, 10.0),
        E_grid_brem=(0.0, 500.0, 50.0), theta_obs_rad=np.deg2rad(90.0),
        dtheta_obs_rad=np.deg2rad(2.0), tilt_deg=0.0, tilt_azim_deg=0.0,
        domega_sr=1e-4, beam_uvw=(0, 0, 1), mosaic_fwhm_rad=None,
        mosaic_mc_fwhm_rad=None, mosaic_mc_nodes=1, abs_layers=None,
        layer_radiators=None, brem_file=None, Ne=10, Ne_brem=5, seed=42,
        spec_chunk=None, brem_chunk=None,
    )


def _key(case):
    return sha256((case["name"] + str(case["E0_keV"])).encode()).hexdigest()


def _fail_runner(*_args, **_kwargs):
    raise AssertionError("completed worker artifacts must not be recomputed")


def test_executor_returning_references_is_persisted_by_coordinator(tmp_path):
    case = _case()
    store = CaseArtifactStore(tmp_path / "inbox")
    before_checkpoint = []

    def simulated_remote(cases, callback, **_options):
        for i, item in enumerate(cases):
            ref = store.publish("hopg", _key(item), fake_out(item))
            before_checkpoint.append(not (tmp_path / "hopg.pkl").exists())
            callback(i, item, ref)
        return [None] * len(cases)

    results = {}
    assert run_sweep(
        [case], results, checkpoint_path=str(tmp_path / "hopg.pkl"),
        progress=False, content_key_fn=_key, artifact_store=store,
        executor=LocalExecutor(simulated_remote),
    )
    assert before_checkpoint == [True]
    assert (tmp_path / "hopg.pkl").is_file()
    np.testing.assert_array_equal(results[case["name"]][case["E0_keV"]]["spec"],
                                  fake_out(case)["spec"])

    # Coordinator published standard checkpoints; replay from them requires no
    # worker execution, whether or not the shared artifact inbox still exists.
    assert run_sweep(
        [case], {}, checkpoint_path=str(tmp_path / "hopg.pkl"),
        progress=False, content_key_fn=_key, artifact_store=store,
        executor=LocalExecutor(_fail_runner),
    )


def test_restart_recovers_published_but_uncommitted_worker_output(tmp_path):
    case = _case()
    store = CaseArtifactStore(tmp_path / "inbox")
    store.publish("hopg", _key(case), fake_out(case))
    checkpoint = tmp_path / "hopg.pkl"
    assert not checkpoint.exists()  # worker has not written any authoritative metadata
    results = {}
    assert run_sweep(
        [case], results, checkpoint_path=str(checkpoint),
        progress=False, content_key_fn=_key, artifact_store=store,
        executor=LocalExecutor(_fail_runner),
    )
    assert checkpoint.is_file()
    assert "distributed" in results


def test_worker_cannot_substitute_wrong_physics_key(tmp_path):
    case = _case()
    store = CaseArtifactStore(tmp_path / "inbox")

    def bad_worker(cases, callback, **_kwargs):
        callback(0, cases[0], store.publish("hopg", "f" * 64, fake_out(cases[0])))
        return [None]

    with pytest.raises(ArtifactIntegrityError, match="content key"):
        run_sweep(
            [case], {}, checkpoint_path=str(tmp_path / "hopg.pkl"),
            progress=False, content_key_fn=_key, artifact_store=store,
            executor=LocalExecutor(bad_worker),
        )


def test_distributed_store_requires_real_case_identity_function(tmp_path):
    with pytest.raises(ValueError, match="requires content_key_fn"):
        run_sweep([_case()], {}, checkpoint_path=str(tmp_path / "hopg.pkl"),
                  progress=False, artifact_store=CaseArtifactStore(tmp_path / "inbox"))


def test_no_cache_disables_replay_of_worker_artifact(tmp_path):
    from tests.helpers import stub_run_cases

    case = _case()
    store = CaseArtifactStore(tmp_path / "inbox")
    store.publish("hopg", _key(case), fake_out(case))
    called = []

    def running_again(cases, **options):
        called.extend(cases)
        return stub_run_cases(cases, **options)

    assert run_sweep(
        [case], {}, checkpoint_path=str(tmp_path / "hopg.pkl"),
        progress=False, cache_read=False, cache_write=False, content_key_fn=_key,
        artifact_store=store, executor=LocalExecutor(running_again),
    )
    assert called == [case]


def test_recovered_artifact_supports_disk_retention_mode(tmp_path):
    case = _case()
    store = CaseArtifactStore(tmp_path / "inbox")
    store.publish("hopg", _key(case), fake_out(case))
    results = {}
    checkpoint = tmp_path / "hopg"
    assert run_sweep(
        [case], results, checkpoint_path=str(checkpoint),
        progress=False, retain_results=False,
        content_key_fn=_key, artifact_store=store,
        executor=LocalExecutor(_fail_runner),
    )
    assert (checkpoint / "line.h5").exists()
    record = results[case["name"]][case["E0_keV"]]
    assert record["case"]["name"] == case["name"]
    assert "spec" not in record  # disk retention owns arrays, not the mapping
