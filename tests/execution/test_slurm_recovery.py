"""Native SLURM retries/cancellation: no scheduler, GPU or remote host required."""

import json
import sys
from types import ModuleType

import pytest

from pyrite.execution import ResourceRequest, RetryPolicy
from pyrite.execution.artifacts import CaseArtifactStore
from pyrite.execution.slurm import (
    ExecutionCancelledError,
    SlurmArrayConfig,
    SlurmArrayExecutor,
    _array_spec,
    _cancel_job,
    _submit,
)
from pyrite.execution.slurm_worker import execute_unit


def _case(index):
    return {"crystal": "hopg", "name": f"c-{index}", "E0_keV": index, "seed": index}


def _key(case):
    return f"{case['seed']:064x}"


def _executor(tmp_path, *, tasks=2, attempts=2):
    return SlurmArrayExecutor(
        CaseArtifactStore(tmp_path / "shared"), _key,
        resources=ResourceRequest(cpu_cores=1),
        config=SlurmArrayConfig(cases_per_task=tasks, poll_seconds=0.001),
        retry_policy=RetryPolicy(max_attempts=attempts),
    )


def test_array_spec_is_compact_and_validates():
    assert _array_spec([0, 1, 2, 4, 7, 8], 3) == "0-2,4,7-8%3"
    for bad in ([], [2, 1], [1, 1], [-1]):
        with pytest.raises(ValueError):
            _array_spec(bad, 2)


def test_sbatch_retry_overrides_array_and_tags_attempt(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm

    commands = []

    def run(argv, **kwargs):
        commands.append(argv)
        assert kwargs["check"] is True
        return type("Completed", (), {"stdout": "456;cluster\n"})()

    monkeypatch.setattr(slurm.subprocess, "run", run)
    assert _submit(tmp_path / "script.sbatch", array_spec="0,3-5%2", attempt=2) == "456"
    assert commands == [[
        "sbatch", "--parsable", "--array=0,3-5%2",
        "--export=ALL,PYRITE_SLURM_ATTEMPT=2", str(tmp_path / "script.sbatch")
    ]]
    _cancel_job("456")
    assert commands[-1] == ["scancel", "456"]
    with pytest.raises(ValueError):
        _submit(tmp_path / "script.sbatch", array_spec="0;rm -rf /%2")
    with pytest.raises(ValueError):
        _cancel_job("456;echo unsafe")


def test_partial_failure_retries_only_missing_units_no_recomputed_cases(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm

    ex = _executor(tmp_path, tasks=2)
    cases = [_case(i) for i in range(4)]
    submissions = []
    active_calls = 0
    run_names = []
    fake_mc = ModuleType("pyrite.montecarlo")

    def fake_runner(batch, *, callback, **kwargs):
        for i, case in enumerate(batch):
            run_names.append(case["name"])
            callback(i, case, {"spectrum": case["E0_keV"]})
            if case["E0_keV"] == 0 and len(submissions) == 1:
                raise RuntimeError("preemption after first published case")
        return [None] * len(batch)

    fake_mc.run_cases = fake_runner
    monkeypatch.setitem(sys.modules, "pyrite.montecarlo", fake_mc)

    def submit(_script, **kwargs):
        submissions.append(kwargs)
        return str(100 + len(submissions))

    monkeypatch.setattr(slurm, "_submit", submit)

    def active(_job):
        nonlocal active_calls
        active_calls += 1
        if active_calls == 1:
            with pytest.raises(RuntimeError, match="unit 0 failed"):
                execute_unit(ex.last_run_dir, 0)
            execute_unit(ex.last_run_dir, 1)
            return True
        if active_calls == 2:
            monkeypatch.setenv("PYRITE_SLURM_ATTEMPT", "2")
            execute_unit(ex.last_run_dir, 0)
            monkeypatch.delenv("PYRITE_SLURM_ATTEMPT")
            return True
        raise AssertionError("unexpected polling")

    monkeypatch.setattr(slurm, "_job_active", active)
    delivered = []
    out = ex.execute(cases, callback=lambda i, _c, ref: delivered.append((i, ref)),
                     keep_results=True)
    assert submissions == [{}, {"array_spec": "0%8", "attempt": 2}]
    assert sorted(i for i, _ in delivered) == [0, 1, 2, 3]
    assert sorted(run_names) == ["c-0", "c-1", "c-2", "c-3"]
    assert [ref.content_key for ref in out] == [_key(c) for c in cases]
    assert ex.submitted_job_ids == ["101", "102"]
    status = json.loads((ex.last_run_dir / "status" / "0.json").read_text())
    assert status["attempt"] == 2
    assert status["state"] == "succeeded"


def test_stale_status_must_not_complete_new_attempt(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm

    ex = _executor(tmp_path, attempts=2)
    case = _case(1)
    calls = 0

    def submit(_script, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            status = ex.last_run_dir / "status" / "0.json"
            status.parent.mkdir(exist_ok=True, parents=True)
            status.write_text(json.dumps({
                "format": "pyrite.slurm-task.v1", "unit_index": 0,
                "attempt": 1, "state": "failed", "error": "preempted",
            }))
        return str(calls)

    monkeypatch.setattr(slurm, "_submit", submit)
    monkeypatch.setattr(slurm, "_job_active", lambda _: False)
    with pytest.raises(RuntimeError, match="exhausted 2 attempt"):
        ex.execute([case])
    assert calls == 2


def test_retry_can_recover_missing_status_with_no_recompute(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm

    ex = _executor(tmp_path, attempts=2)
    case = _case(1)
    attempts = []

    def submit(_script, **options):
        attempts.append(options)
        if len(attempts) == 1:
            ex.store.publish(case["crystal"], _key(case), {"value": 9})
        return str(len(attempts))

    monkeypatch.setattr(slurm, "_submit", submit)
    monkeypatch.setattr(slurm, "_job_active", lambda job_id: False)
    delivered = []
    assert ex.execute([case], callback=lambda i, c, ref: delivered.append(i))
    assert delivered == [0]
    assert attempts == [{}]


def test_cancel_recovers_survivors_and_does_not_retry(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm

    ex = _executor(tmp_path, tasks=1, attempts=3)
    cases = [_case(0), _case(1)]
    jobs = []
    cancelled = []
    seen = []

    def submit(_script, **kwargs):
        jobs.append(kwargs)
        return "234"

    monkeypatch.setattr(slurm, "_submit", submit)
    monkeypatch.setattr(slurm, "_cancel_job", lambda job: cancelled.append(job))

    def active(_job):
        ex.store.publish("hopg", _key(cases[0]), {"value": 1})
        assert ex.cancel()
        return True

    monkeypatch.setattr(slurm, "_job_active", active)
    with pytest.raises(ExecutionCancelledError, match="1/2"):
        ex.execute(cases, callback=lambda i, *_: seen.append(i))
    assert seen == [0]
    assert jobs == [{}]
    assert cancelled == ["234"]
    assert ex.cancel() is False  # executor no longer running


def test_worker_skips_committed_cases_in_retry(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm

    ex = _executor(tmp_path)
    rows = [_case(0), _case(1)]
    ex.store.publish("hopg", _key(rows[0]), {"value": "persisted"})
    fake_mc = ModuleType("pyrite.montecarlo")
    run = []

    def fake_runner(cases, *, callback, **kwargs):
        run.extend(c["name"] for c in cases)
        for i, c in enumerate(cases):
            callback(i, c, {"value": "new"})

    fake_mc.run_cases = fake_runner
    monkeypatch.setitem(sys.modules, "pyrite.montecarlo", fake_mc)
    monkeypatch.setattr(slurm, "_submit", lambda *_args: "101")

    def active(_job):
        execute_unit(ex.last_run_dir, 0)
        return True

    monkeypatch.setattr(slurm, "_job_active", active)
    ex.execute(rows)
    assert run == ["c-1"]
    assert ex.store.load(ex.store.reference("hopg", _key(rows[0]))) == {"value": "persisted"}


def test_cancel_during_sbatch_still_cancels_new_job(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm

    ex = _executor(tmp_path)
    cancelled = []

    def submit(_script):
        # No job ID exists yet while sbatch is returning.
        assert ex.cancel() is True
        return "999"

    monkeypatch.setattr(slurm, "_submit", submit)
    monkeypatch.setattr(slurm, "_cancel_job", cancelled.append)
    with pytest.raises(ExecutionCancelledError):
        ex.execute([_case(1)])
    assert cancelled == ["999"]
    assert ex.cancel() is False


def test_retry_refuses_corrupted_preexisting_artifact(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm
    from pyrite.execution.artifacts import ArtifactIntegrityError

    ex = _executor(tmp_path)
    row = _case(1)
    ref = ex.store.publish("hopg", _key(row), {"value": 42})
    path = ex.store.path_for(ref.material, ref.content_key)
    with path.open("ab") as stream:
        stream.write(b"tampering")
    monkeypatch.setattr(slurm, "_submit", lambda _script: "1")
    monkeypatch.setattr(slurm, "_job_active", lambda _: False)
    with pytest.raises(ArtifactIntegrityError):
        ex.execute([row])
    assert ex.submitted_job_ids == ["1"]
