"""SLURM control-plane tests run without a SLURM installation or a GPU."""

import sys
from hashlib import sha256
from types import ModuleType

import pytest

from pyrite.execution import ResourceRequest
from pyrite.execution.artifacts import CaseArtifactRef, CaseArtifactStore
from pyrite.execution.slurm import SlurmArrayConfig, SlurmArrayExecutor, _script
from pyrite.execution.slurm_worker import execute_unit


def case(i):
    return {"name": f"config-{i}", "crystal": "hopg", "E0_keV": i, "seed": i}


def key(c):
    return sha256(str((c["E0_keV"], c["seed"])).encode()).hexdigest()


def test_slurm_script_cpu_gpu_and_validation(tmp_path):
    cfg = SlurmArrayConfig(partition="standard", max_concurrent=3)
    text = _script(tmp_path, 6, ResourceRequest(cpu_cores=4, memory_bytes=5 * 1024**3), cfg)
    assert "#SBATCH --array=0-5%3" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=5120M" in text
    assert "#SBATCH --gres=" not in text
    assert '--unit-index "${SLURM_ARRAY_TASK_ID' in text
    gpu = _script(
        tmp_path,
        2,
        ResourceRequest(cpu_cores=2, gpu_count=1, accelerator="rocm"),
        SlurmArrayConfig(gres="gpu:radeon8060s:1"),
    )
    assert "#SBATCH --gres=gpu:radeon8060s:1" in gpu
    assert "PYRITE_MC_BACKEND=rocm" in gpu
    with pytest.raises(ValueError, match="unsafe SLURM"):
        SlurmArrayConfig(partition="normal\n#SBATCH --account=intruder")
    with pytest.raises(ValueError, match="differs"):
        _script(
            tmp_path,
            2,
            ResourceRequest(gpu_count=1, accelerator="cuda"),
            SlurmArrayConfig(gres="gpu:2"),
        )


def test_cpu_job_array_end_to_end_without_real_slurm(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm

    store = CaseArtifactStore(tmp_path / "shared")
    cases = [case(i) for i in range(5)]
    seen = []
    counts = {"running": 0}

    def mock_submit(path):
        assert (path.parent / "plan.json").is_file()
        return "12345"

    def mock_active(_job):
        # Exercise the real worker reader, mocked numerical kernel, artifact
        # publication, and coordinator reference validation in one test.
        if counts["running"] == 0:
            counts["running"] += 1
            for i in range(3):
                execute_unit(executor.last_run_dir, i)
        return True

    fake_mc = ModuleType("pyrite.montecarlo")

    def run_cases(inputs, *, callback, **opts):
        assert opts["engine"] == "cpu"
        assert opts["max_workers"] == 1
        for idx, c in enumerate(inputs):
            callback(idx, c, {"spec": idx, "energy": c["E0_keV"]})
        return [None] * len(inputs)

    fake_mc.run_cases = run_cases
    monkeypatch.setitem(sys.modules, "pyrite.montecarlo", fake_mc)
    monkeypatch.setattr(slurm, "_submit", mock_submit)
    monkeypatch.setattr(slurm, "_job_active", mock_active)
    executor = SlurmArrayExecutor(
        store,
        key,
        resources=ResourceRequest(cpu_cores=2),
        config=SlurmArrayConfig(cases_per_task=2, poll_seconds=0.001),
    )
    result = executor.execute(
        cases, keep_results=False, callback=lambda i, c, ref: seen.append((i, c, ref))
    )
    assert result == [None] * 5
    assert [x[0] for x in seen] == list(range(5))
    assert all(isinstance(ref, CaseArtifactRef) for _, _, ref in seen)
    assert all(store.load(ref)["energy"] == c["E0_keV"] for _, c, ref in seen)
    assert executor.last_job_id == "12345"


def test_worker_partial_failure_publishes_survivors(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm

    store = CaseArtifactStore(tmp_path / "shared")
    cases = [case(0), case(1)]

    fake_mc = ModuleType("pyrite.montecarlo")

    def failing_run_cases(inputs, *, callback, **_opts):
        callback(0, inputs[0], {"spec": [1.0]})
        raise RuntimeError("simulated transport failure")

    fake_mc.run_cases = failing_run_cases
    monkeypatch.setitem(sys.modules, "pyrite.montecarlo", fake_mc)
    monkeypatch.setattr(slurm, "_submit", lambda path: "42")
    called = False

    def mock_active(_job):
        nonlocal called
        if not called:
            called = True
            with pytest.raises(RuntimeError, match="unit 0 failed"):
                execute_unit(executor.last_run_dir, 0)
        return True

    monkeypatch.setattr(slurm, "_job_active", mock_active)
    published = []
    executor = SlurmArrayExecutor(
        store, key, config=SlurmArrayConfig(poll_seconds=0.001, cases_per_task=2)
    )
    with pytest.raises(RuntimeError, match="task 0 failed"):
        executor.execute(
            cases, callback=lambda index, *_: published.append(index), keep_results=False
        )
    assert published == [0]
    assert store.reference("hopg", key(cases[1])) is None


def test_missing_status_recovers_survivors(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm

    store = CaseArtifactStore(tmp_path / "shared")
    cases = [case(0), case(1)]
    monkeypatch.setattr(slurm, "_submit", lambda path: "74")

    def mock_active(_job):
        store.publish("hopg", key(cases[0]), {"spec": 1})
        return False

    monkeypatch.setattr(slurm, "_job_active", mock_active)
    delivered = []
    executor = SlurmArrayExecutor(store, key, config=SlurmArrayConfig(cases_per_task=2))
    with pytest.raises(RuntimeError, match="incomplete tasks"):
        executor.execute(cases, callback=lambda i, *_: delivered.append(i), keep_results=False)
    assert delivered == [0]


def test_rejects_unsafe_incompatible_modes_before_submitting(tmp_path, monkeypatch):
    import pyrite.execution.slurm as slurm

    monkeypatch.setattr(slurm, "_submit", lambda *_: pytest.fail("must not submit"))
    store = CaseArtifactStore(tmp_path / "shared")
    ex = SlurmArrayExecutor(store, key)
    with pytest.raises(ValueError, match="bounded sweeps"):
        ex.execute([case(1)], should_stop=lambda: False)
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        ex.execute([case(1)], observation_directions=[[1, 0, 0]])
    with pytest.raises(ValueError, match="does not support"):
        ex.execute([case(1)], invented_option=True)
    ex = SlurmArrayExecutor(store, key, resources=ResourceRequest(cpu_cores=2))
    with pytest.raises(ValueError, match="exceeds"):
        ex.execute([case(1)], max_workers=3)
