"""CPU tests of the public CUDA execution tuner with a fake worker (#776)."""

import queue
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import pyrite as pr
import pyrite.config as cfg
from pyrite.optimization import KernelSearchSpace, KernelTuningResult, TuningProgress, tune_kernels
from pyrite.optimization import _kernel_worker as kw
from pyrite.optimization import kernel_tuning as kt
from pyrite.phases import SimulationProgress

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


class FakeWorker:
    """Transport time falls with electron batch, spectrum time with cxr chunk.

    ``continuation_rows=7`` changes the transport and ``cxr_chunk=3`` the
    spectrum, so both must be rejected.
    """

    runs: list = []

    def __init__(self, payload, workdir, timeout, on_event):
        self.workdir, self.on_event, self.environment = workdir, on_event, ENV

    def measure(self, request):
        tuning = kt.tuning_from_dict(request["tuning"])
        phase = request["phase"]
        FakeWorker.runs.append((phase, tuning))
        self.on_event(dict(phase="transport", stage="electrons", done=1, total=2, run=1, runs=2))
        repeats = request.get("repeats", 3)
        if phase == "transport":
            batch = tuning.transport.electron_batch or 1024
            digest = "bad" if tuning.transport.continuation_rows == 7 else "same"
            return [1.0 / batch] * repeats, {"digest": digest}
        output = self.workdir / f"{request['label']}.npz"
        _save(output, scale=1.5 if tuning.cxr_chunk == 3 else 1.0)
        chunk = tuning.cxr_chunk or 1024
        return [1.0 / chunk + 1.0 / (tuning.transport.electron_batch or 1024)] * repeats, {
            "spectra": str(output)
        }

    def close(self):
        pass


@pytest.fixture
def fake(monkeypatch):
    FakeWorker.runs = []
    monkeypatch.setattr(kt, "_Worker", FakeWorker)
    return FakeWorker.runs


def test_search_space_validation():
    with pytest.raises(ValueError, match="at least one"):
        KernelSearchSpace()
    with pytest.raises(ValueError, match="distinct"):
        KernelSearchSpace(cxr_chunk=(256, 256))
    with pytest.raises(ValueError, match="distinct"):
        KernelSearchSpace(cxr_chunk=[256])


def test_candidate_changes_only_search_space_dimensions():
    base = cfg.KernelTuning(transport_core="cuda", brem_chunk=8)
    space = KernelSearchSpace(electron_batch=(512,), schedule=("auto",), continuation_rows=(64,))
    got = space.candidate(
        base, {"electron_batch": 512, "schedule": "auto", "continuation_rows": 64}
    )
    assert got == cfg.KernelTuning(
        transport_core="cuda",
        brem_chunk=8,
        transport=cfg.TransportKernelTuning("auto", 64, 512),
    )
    with pytest.raises(ValueError, match="requires schedule"):
        space.candidate(base, {"schedule": "full-history", "continuation_rows": 64})
    launch = KernelSearchSpace(scratch_mib=(None, 4096), kernel_threads=(32, 128))
    assert launch.dimensions("transport") == ("scratch_mib", "kernel_threads")
    assert len(launch.choices(base, "transport")) == 4
    assert launch.candidate(base, {"scratch_mib": 4096, "kernel_threads": 32}).transport == (
        cfg.TransportKernelTuning(scratch_mib=4096, kernel_threads=32)
    )
    cxr = KernelSearchSpace(cxr_threads=(None, 256), cxr_energies_per_block=(3, 4))
    assert cxr.dimensions("spectrum") == ("cxr_threads", "cxr_energies_per_block")
    got = cxr.candidate(base, {"cxr_threads": 256, "cxr_energies_per_block": 4})
    assert (got.cxr_threads, got.cxr_energies_per_block, got.brem_chunk) == (256, 4, 8)


def test_spectrum_worker_applies_all_proposed_launch_controls(monkeypatch, tmp_path):
    """A phase-only trial must time its proposed launch, not the default one."""
    base = cfg.KernelTuning(transport_core="cuda", cxr_threads=512, cxr_energies_per_block=2)
    candidate = cfg.KernelTuning(
        transport_core="cuda",
        cxr_chunk=8192,
        brem_chunk=1024,
        cxr_threads=128,
        cxr_energies_per_block=4,
    )
    session = object.__new__(kw._Session)
    session.base = base
    session.scene = SimpleNamespace(
        numerics=SimpleNamespace(with_kernel_tuning=lambda value: value)
    )
    session.backend = SimpleNamespace(release_memory=Mock(), synchronize=Mock())
    session.workdir = tmp_path
    session._transport = SimpleNamespace(numerics=SimpleNamespace(kernel_tuning=base))
    session._warmed = set()
    session._forward = lambda *_args: lambda _event: None
    observed = []

    def fake_score(_transport, *, kernel_tuning, progress):
        assert callable(progress)
        observed.append(kernel_tuning)
        return SimpleNamespace(provenance={})

    monkeypatch.setattr("pyrite.phases.score_spectra", fake_score)
    monkeypatch.setattr(kw, "save_spectra", lambda _result, path: path.touch())
    samples, check = session.measure("spectrum", candidate, "spectrum-test", warmups=0, repeats=1)
    # One initial phase warmup and one timed score are both configured identically.
    assert len(observed) == 2
    assert all(setting == candidate for setting in observed)
    assert len(samples) == 1
    assert Path(check["spectra"]).exists()


def test_equivalence_requires_exact_grids_and_close_densities(tmp_path):
    ref, cur = tmp_path / "ref.npz", tmp_path / "cur.npz"
    _save(ref)
    _save(cur, scale=1 + 1e-7)
    assert kt._equivalent(cur, ref, rtol=1e-4, atol=0) is None
    assert "beyond" in kt._equivalent(cur, ref, rtol=1e-9, atol=0)
    _save(cur, grid=(1.0, 2.5))
    assert "grid differs" in kt._equivalent(cur, ref, rtol=1, atol=1)


def test_phases_tune_independently_and_join_winners(fake):
    pytest.importorskip("optuna")
    space = KernelSearchSpace(
        electron_batch=(2048, 4096),
        schedule=("full-history", "continuation"),
        continuation_rows=(None, 7),
        cxr_chunk=(None, 3, 4096),
    )
    result = tune_kernels(_scene(), space, seed=3)
    assert isinstance(result, KernelTuningResult)
    assert result.environment == ENV
    transport, spectrum = result.transport, result.spectrum
    # Exhaustive: 2 batches x (full-history + continuation x 2 quanta), and 3 chunks.
    assert len(transport.trials) == 6 and len(spectrum.trials) == 3
    assert {phase for phase, _ in fake} == {"transport", "spectrum"}
    # Transport candidates never vary chunks; spectrum candidates never vary transport.
    assert all(t.cxr_chunk is None for phase, t in fake if phase == "transport")
    assert all(
        t.transport == result.spectrum.baseline.tuning.transport for p, t in fake if p == "spectrum"
    )
    assert {t.reason for t in transport.trials if t.state == "rejected"} == {
        "transport arrays differ"
    }
    assert any("cxr.incoherent" in (t.reason or "") for t in spectrum.trials)
    assert transport.best.transport.electron_batch == 4096
    assert transport.best.transport.continuation_rows != 7
    assert spectrum.best.cxr_chunk == 4096
    assert result.best.transport == transport.best.transport
    assert result.best.cxr_chunk == 4096
    assert transport.speedup == pytest.approx(4.0)
    assert transport.confirmation_s is not None
    with pytest.raises(AttributeError, match="several phases"):
        result.baseline  # noqa: B018


def test_tpe_resamples_reuse_their_first_measurement(fake):
    pytest.importorskip("optuna")
    space = KernelSearchSpace(electron_batch=(2048, 4096, 8192, 16384))
    result = tune_kernels(_scene(), space, trials=3, seed=1)
    assert len(result.trials) == 3  # fewer trials than candidates: TPE, not exhaustive
    by_number = {t.number: t for t in result.trials}
    for trial in result.trials:
        if trial.repeat_of is not None:
            first = by_number[trial.repeat_of]
            assert first.tuning == trial.tuning and first.median_s == trial.median_s
    measured = [t for t in result.trials if t.repeat_of is None]
    assert len(fake) == 1 + len(measured) + 2 * 3  # baseline, candidates, confirmation


def test_total_objective_and_progress(fake):
    pytest.importorskip("optuna")
    observed = []
    result = tune_kernels(
        _scene(),
        KernelSearchSpace(electron_batch=(2048,), cxr_chunk=(4096,)),
        objective="total",
        on_progress=observed.append,
    )
    assert [p.phase for p in result.phases] == ["total"]
    assert {phase for phase, _ in fake} == {"total"}
    assert all(isinstance(p, TuningProgress) for p in observed)
    assert observed[0].state == "running" and observed[0].trial == 0
    assert any(p.state == "baseline" for p in observed)
    inner = [p for p in observed if p.simulation is not None]
    assert inner and inner[0].simulation == SimulationProgress("transport", "electrons", 1, 2)
    assert observed[-1].trial == observed[-1].trials == 1


def test_baseline_candidate_reuses_the_baseline_timing(fake):
    pytest.importorskip("optuna")
    result = tune_kernels(_scene(), KernelSearchSpace(cxr_chunk=(None, 4096)))
    same = next(t for t in result.trials if t.tuning.cxr_chunk is None)
    assert same.repeat_of == -1 and same.samples_s == result.baseline.samples_s
    # Baseline once plus its three confirmation halves; the equal candidate is never re-run.
    assert [t.cxr_chunk for _, t in fake].count(None) == 1 + 3
    assert result.best.cxr_chunk == 4096


def test_progress_counts_finished_trials(fake):
    pytest.importorskip("optuna")
    observed = []
    tune_kernels(
        _scene(), KernelSearchSpace(electron_batch=(2048, 4096, 8192)), on_progress=observed.append
    )
    running = [p.trial for p in observed if p.state == "running" and p.simulation is None]
    assert running[:4] == [0, 1, 2, 3]


def test_no_faster_candidate_keeps_baseline(fake):
    pytest.importorskip("optuna")
    result = tune_kernels(_scene(), KernelSearchSpace(electron_batch=(256,)))
    assert not result.improved
    assert result.best == result.baseline.tuning
    assert result.speedup == 1.0


def test_unconfirmed_speedup_keeps_baseline(fake, monkeypatch):
    pytest.importorskip("optuna")
    result = tune_kernels(_scene(), KernelSearchSpace(electron_batch=(1025,)), min_speedup=1.5)
    assert result.trials[0].median_s < result.baseline.median_s
    assert not result.improved and result.chosen is result.baseline
    assert result.transport.confirmation_s is not None


def test_persistent_study_resumes_and_rejects_another_workload(fake, tmp_path):
    pytest.importorskip("optuna")
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    space = KernelSearchSpace(electron_batch=(2048, 4096))
    tune_kernels(_scene(), space, storage=storage)
    runs = len(fake)
    resumed = tune_kernels(_scene(), space, storage=storage)
    assert len(resumed.trials) == 2
    # Only the baseline and the confirmation pairs run again; no candidate.
    assert {t.transport.electron_batch for _, t in fake[runs:]} == {None, 4096}
    assert len(fake) - runs == 1 + 2 * 3
    with pytest.raises(ValueError, match="stored study differs"):
        tune_kernels(_scene(), KernelSearchSpace(electron_batch=(512,)), storage=storage)


def test_rejects_non_cuda_baseline_and_bad_inputs():
    with pytest.raises(ValueError, match="CUDA transport core"):
        tune_kernels(_scene(transport_core="lockstep"), KernelSearchSpace(cxr_chunk=(256,)))
    with pytest.raises(TypeError, match="Scene"):
        tune_kernels(object(), KernelSearchSpace(cxr_chunk=(256,)))
    with pytest.raises(ValueError, match="rtol"):
        tune_kernels(_scene(), KernelSearchSpace(cxr_chunk=(256,)), rtol=-1)


def test_kernel_lattice_is_a_deprecated_alias():
    import pyrite.optimization as opt

    with pytest.warns(DeprecationWarning, match="KernelSearchSpace"):
        assert opt.KernelLattice is KernelSearchSpace


def _silent_worker(tmp_path, stderr="worker said this"):
    worker = kt._Worker(tmp_path / "payload", tmp_path, 0.001, lambda _message: None)
    worker._stderr = (tmp_path / "stderr").open("w+")
    worker._stderr.write(stderr)
    worker._lines = queue.Queue()
    return worker


def test_worker_timeout_reports_stderr(tmp_path):
    with pytest.raises(kt._WorkerError, match="no worker reply.*worker said this"):
        _silent_worker(tmp_path)._next()


def test_worker_interrupt_keeps_stderr_as_a_note(tmp_path):
    worker = _silent_worker(tmp_path)
    worker._lines = SimpleNamespace(get=Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt) as caught:
        worker._next()
    assert any("worker said this" in note for note in caught.value.__notes__)


@pytest.mark.parametrize(("reserved", "warns"), [(10.0, False), (4096.0, True)])
def test_caller_pool_is_released_and_live_memory_warned(monkeypatch, reserved, warns):
    backend = SimpleNamespace(
        name="cuda",
        release_memory=Mock(),
        allocator_stats=lambda: {"reserved_mib": reserved},
    )
    monkeypatch.setattr("pyrite._backend.BACKEND", backend)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kt._release_caller_device_memory()
    backend.release_memory.assert_called_once_with()
    assert bool(caught) is warns
