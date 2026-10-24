"""Native SLURM execution integrates with the real sweep checkpoint coordinator."""

from pyrite.execution import ResourceRequest
from pyrite.execution.artifacts import CaseArtifactStore
from pyrite.execution.slurm import SlurmArrayConfig, SlurmArrayExecutor
from pyrite.execution.slurm_worker import execute_unit
from pyrite.runs.run import run_sweep
from tests.execution.test_sweep_artifacts import _case, _key
from tests.helpers import fake_out


def test_job_array_publishes_through_standard_sweep_checkpoint(monkeypatch, tmp_path):
    import pyrite.execution.slurm as slurm
    import pyrite.montecarlo as mc

    store = CaseArtifactStore(tmp_path / "shared")
    cases = [_case("slurm-a"), _case("slurm-b")]
    checkpoint = tmp_path / "checkpoints" / "hopg"
    checkpoint.mkdir(parents=True)
    results = {}
    called = []

    def fake_runner(batch, *, callback, **kwargs):
        assert kwargs["engine"] == "cpu"
        for i, case in enumerate(batch):
            # The worker is not the checkpoint owner.
            assert not (checkpoint / "line.h5").exists()
            called.append(case["name"])
            callback(i, case, fake_out(case))
        return [None] * len(batch)

    monkeypatch.setattr(mc, "run_cases", fake_runner)
    monkeypatch.setattr(slurm, "_submit", lambda path: "1234")
    launched = False

    def active(_job):
        nonlocal launched
        if not launched:
            launched = True
            for index in range(2):
                execute_unit(executor.last_run_dir, index)
        return True

    monkeypatch.setattr(slurm, "_job_active", active)
    executor = SlurmArrayExecutor(
        store, _key, resources=ResourceRequest(cpu_cores=1),
        config=SlurmArrayConfig(cases_per_task=1, poll_seconds=0.001),
    )
    assert run_sweep(
        cases, results, checkpoint_path=str(checkpoint), progress=False,
        content_key_fn=_key, artifact_store=store, executor=executor,
    )
    assert called == ["slurm-a", "slurm-b"]
    assert (checkpoint / "line.h5").exists()
    assert set(results) == {"slurm-a", "slurm-b"}

    # Normal checkpoint resume remains the only owner of completion. Fully
    # completed cases never need a second SLURM allocation.
    monkeypatch.setattr(slurm, "_submit", lambda *_: (_ for _ in ()).throw(
        AssertionError("a fully cached campaign was submitted")))
    assert run_sweep(
        cases, {}, checkpoint_path=str(checkpoint), progress=False,
        content_key_fn=_key, artifact_store=store, executor=executor,
    )
