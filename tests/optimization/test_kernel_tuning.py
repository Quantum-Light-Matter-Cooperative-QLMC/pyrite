"""CPU tests of the public CUDA execution tuner with a fake worker (#776)."""

from pathlib import Path

import numpy as np
import pytest

import pyrite as pr
import pyrite.config as cfg
from pyrite.optimization import KernelLattice, KernelTuningResult, tune_kernels
from pyrite.optimization import kernel_tuning as kt

ENV = {"backend": "cuda", "device": "Fake GPU", "precision": "fp32", "pyrite_version": "x"}


def _scene(**execution):
    return pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "hopg", thickness_ang=10_000.0, tilt_deg=30.0, footprint=pr.geometry.Footprint(5.0, 5.0)
        ),
        pr.Detector(),
        numerics=pr.Numerics(execution=cfg.KernelTuning(**execution)),
    )


def _save(path: Path, scale=1.0, grid=(1.0, 2.0)):
    np.savez(
        path,
        **{
            "cxr.incoherent:grid": np.array(grid),
            "cxr.incoherent:density": np.array([1.0, 2.0]) * scale,
        },
    )


@pytest.fixture
def fake(monkeypatch):
    """Time by electron batch; continuation_rows=7 changes the spectrum."""
    runs = []

    def measure(scene, tuning, *, workdir, label, **_):
        runs.append(tuning)
        output = workdir / f"{label}.npz"
        _save(output, scale=1.5 if tuning.transport.continuation_rows == 7 else 1.0)
        batch = tuning.transport.electron_batch or 1024
        return output, {"samples_s": [1.0 / batch, 2.0 / batch, 3.0 / batch], "environment": ENV}

    monkeypatch.setattr(kt, "_measure", measure)
    return runs


def test_lattice_validation():
    with pytest.raises(ValueError, match="at least one"):
        KernelLattice()
    with pytest.raises(ValueError, match="distinct"):
        KernelLattice(cxr_chunk=(256, 256))
    with pytest.raises(ValueError, match="distinct"):
        KernelLattice(cxr_chunk=[256])


def test_candidate_changes_only_lattice_dimensions():
    base = cfg.KernelTuning(transport_core="cuda", brem_chunk=8)
    lattice = KernelLattice(electron_batch=(512,), schedule=("auto",), continuation_rows=(64,))
    got = lattice.candidate(
        base, {"electron_batch": 512, "schedule": "auto", "continuation_rows": 64}
    )
    assert got == cfg.KernelTuning(
        transport_core="cuda",
        brem_chunk=8,
        transport=cfg.TransportKernelTuning("auto", 64, 512),
    )
    with pytest.raises(ValueError, match="requires schedule"):
        lattice.candidate(base, {"schedule": "full-history", "continuation_rows": 64})


def test_equivalence_requires_exact_grids_and_close_densities(tmp_path):
    ref, cur = tmp_path / "ref.npz", tmp_path / "cur.npz"
    _save(ref)
    _save(cur, scale=1 + 1e-7)
    assert kt._equivalent(cur, ref, rtol=1e-4, atol=0) is None
    assert "beyond" in kt._equivalent(cur, ref, rtol=1e-9, atol=0)
    _save(cur, grid=(1.0, 2.5))
    assert "grid differs" in kt._equivalent(cur, ref, rtol=1, atol=1)


def test_tuning_picks_fastest_valid_candidate_and_records_rejections(fake):
    pytest.importorskip("optuna")
    lattice = KernelLattice(
        electron_batch=(2048, 4096),
        schedule=("full-history", "continuation"),
        continuation_rows=(None, 7),
    )
    result = tune_kernels(_scene(), lattice, trials=12, seed=3)
    assert isinstance(result, KernelTuningResult)
    assert result.environment == ENV
    assert result.baseline.tuning == cfg.KernelTuning(transport_core="cuda")
    reasons = {t.reason.split(":")[0] for t in result.trials if t.state == "rejected"}
    assert reasons <= {"invalid combination", "cxr.incoherent"}
    assert result.improved and result.best.transport.electron_batch == 4096
    assert result.best.transport.continuation_rows != 7
    assert result.speedup == pytest.approx(4.0)
    for trial in result.trials:
        if trial.tuning is not None and trial.tuning.transport.continuation_rows == 7:
            assert trial.state == "rejected"


def test_resampled_candidates_reuse_their_first_measurement(fake):
    pytest.importorskip("optuna")
    lattice = KernelLattice(
        electron_batch=(2048, 4096), continuation_rows=(None, 7), schedule=("continuation",)
    )
    result = tune_kernels(_scene(), lattice, trials=10, seed=1)
    repeats = [t for t in result.trials if t.repeat_of is not None]
    assert repeats  # four candidates, ten trials
    assert len(fake) == 1 + len(result.trials) - len(repeats)
    by_number = {t.number: t for t in result.trials}
    for trial in repeats:
        first = by_number[trial.repeat_of]
        assert first.repeat_of is None and first.tuning == trial.tuning
        assert (trial.state, trial.median_s, trial.reason) == (
            first.state,
            first.median_s,
            first.reason,
        )


def test_no_faster_candidate_keeps_baseline(fake):
    pytest.importorskip("optuna")
    result = tune_kernels(_scene(), KernelLattice(electron_batch=(256,)), trials=1)
    assert not result.improved
    assert result.best == result.baseline.tuning
    assert result.speedup == 1.0


def test_persistent_study_resumes_and_rejects_another_workload(fake, tmp_path):
    pytest.importorskip("optuna")
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    lattice = KernelLattice(electron_batch=(2048, 4096))
    tune_kernels(_scene(), lattice, trials=2, storage=storage)
    runs = len(fake)
    resumed = tune_kernels(_scene(), lattice, trials=2, storage=storage)
    assert len(fake) == runs + 1  # baseline re-measured, no new trial
    assert len(resumed.trials) == 2
    with pytest.raises(ValueError, match="stored study differs"):
        tune_kernels(_scene(), KernelLattice(electron_batch=(512,)), trials=3, storage=storage)


def test_rejects_non_cuda_baseline_and_bad_inputs():
    with pytest.raises(ValueError, match="CUDA transport core"):
        tune_kernels(_scene(transport_core="lockstep"), KernelLattice(cxr_chunk=(256,)))
    with pytest.raises(TypeError, match="Scene"):
        tune_kernels(object(), KernelLattice(cxr_chunk=(256,)))
    with pytest.raises(ValueError, match="rtol"):
        tune_kernels(_scene(), KernelLattice(cxr_chunk=(256,)), rtol=-1)
