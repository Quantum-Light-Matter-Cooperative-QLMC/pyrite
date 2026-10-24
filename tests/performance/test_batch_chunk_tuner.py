"""Fast CPU-only tests for the optional two-variable CUDA performance tuner."""

import json
from pathlib import Path

import numpy as np
import pytest

from pyrite.perf import batch_chunk_tuner as tuner


def test_choices_are_discrete_and_unambiguous():
    assert tuner._choices("256,1024,4096") == (256, 1024, 4096)
    for invalid in ("", "100,0", "4,4", "0", "12.5", "hello"):
        with pytest.raises(Exception):
            tuner._choices(invalid)


def _save(path: Path, *, spectrum=1.0, segments=8):
    np.savez_compressed(
        path,
        E_grid=np.array([100., 200., 300.]),
        spec=np.array([spectrum, spectrum * 2, spectrum]),
        brem=np.array([0.1, 0.2, 0.3]),
        E_grid_brem=np.array([100., 300.]),
        brem_wide=np.array([0.1, 0.3]),
        n_segments=np.array(segments),
        eta=np.array(0.1),
        hit_frac=np.array(1.),
    )


def test_comparison_rejects_spectral_and_transport_changes(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _save(a)
    _save(b, spectrum=1.0 + 1e-6)
    assert tuner._equivalent(b, a, rtol=1e-4, atol=0)
    assert not tuner._equivalent(b, a, rtol=1e-8, atol=0)
    _save(b, segments=9)
    assert not tuner._equivalent(b, a, rtol=1e-4, atol=0)


def test_identity_includes_workload_and_device():
    kwargs = dict(batch_choices=(256, 512), chunk_choices=(1024, 2048),
                  warmups=1, repeats=3, rtol=1e-4, atol=1e-12,
                  baseline=(256, 1024), transport_core="cuda",
                  backend="cuda", precision="1", device="GPU A", version="0.20.0")
    a = tuner._study_identity(b"scene A", **kwargs)
    assert a == tuner._study_identity(b"scene A", **kwargs)
    assert a != tuner._study_identity(b"scene B", **kwargs)
    assert a != tuner._study_identity(b"scene A", **(kwargs | {"device": "GPU B"}))


def test_joint_study_resumes_and_rejects_changed_workload(tmp_path, monkeypatch):
    optuna = pytest.importorskip("optuna")
    scene = tmp_path / "scene.py"
    scene.write_text("def make_scene(): pass\n")
    storage = f"sqlite:///{tmp_path / 'results.db'}"
    runs = []

    def fake_worker(scene, *, batch, chunk, output_path, result_path, **kwargs):
        runs.append((batch, chunk))
        _save(output_path)
        return dict(median_s=0.01 + abs(batch - 512) / 10000 + abs(chunk - 2048) / 10000,
                    samples_s=[0.03, 0.04], admitted_cxr_chunk=chunk,
                    backend="cuda", precision="1", device="Fake GPU",
                    pyrite_version="0.20.0")

    monkeypatch.setattr(tuner, "_run_subprocess", fake_worker)
    first = tuner.tune(scene, batches=(256, 512), chunks=(1024, 2048), trials=4,
                       warmups=1, repeats=2, storage=storage, study_name="test")
    assert len(first.trials) == 4
    assert all("electron_batch" in t.params and "cxr_chunk" in t.params for t in first.trials)
    second = tuner.tune(scene, batches=(256, 512), chunks=(1024, 2048), trials=6,
                        warmups=1, repeats=2, storage=storage, study_name="test")
    assert len(second.trials) == 6
    assert len(runs) == 8  # reference on both calls; 4 + 2 candidate evaluations
    scene.write_text("# changed code\ndef make_scene(): pass\n")
    with pytest.raises(ValueError, match="differs in workload"):
        tuner.tune(scene, batches=(256, 512), chunks=(1024, 2048), trials=7,
                   warmups=1, repeats=2, storage=storage, study_name="test")


def test_failed_or_inconsistent_candidate_never_wins(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    scene = tmp_path / "scene.py"
    scene.write_text("def make_scene(): pass\n")
    def fake_worker(scene, *, batch, chunk, output_path, **kwargs):
        _save(output_path, spectrum=(2.0 if batch == 512 else 1.0))
        return dict(median_s=(0.01 if batch == 512 else 10.0), samples_s=[1.],
                    admitted_cxr_chunk=chunk, backend="cuda", precision="1",
                    device="Fake GPU", pyrite_version="0.20.0")
    monkeypatch.setattr(tuner, "_run_subprocess", fake_worker)
    st = tuner.tune(scene, batches=(256, 512), chunks=(1024,), trials=12,
                    warmups=1, repeats=1, baseline=(256, 1024),
                    storage=f"sqlite:///{tmp_path / 'a.db'}")
    valid = [t for t in st.trials if t.state.name == "COMPLETE"]
    assert valid and all(t.params["electron_batch"] == 256 for t in valid)
    assert any(t.state.name == "PRUNED" for t in st.trials)
